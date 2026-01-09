#!/usr/bin/env python3
"""
Backfill RT (VehiclePositions) data for all routes.

This script downloads GTFS-RT VehiclePositions archives from KoDa,
extracts ~30 evenly spaced pb files per hour, parses ALL vehicles,
and aggregates occupancy data by (date, hour, route_id, direction_id).

Output: Parquet file(s) with columns:
    date, hour, route_id, direction_id, n_obs, n_empty, n_many_seats,
    n_few_seats, n_standing, n_crushed, n_full, max_occupancy, avg_occupancy

Usage:
    # Process date range:
    python backfill_rt_data.py --start 2025-10-01 --end 2025-10-31
    
    # Process specific dates:
    python backfill_rt_data.py --dates 2025-10-01 2025-10-15 2025-10-30
    
    # Both (dates override any overlap):
    python backfill_rt_data.py --start 2025-10-01 --end 2025-10-07 --dates 2025-12-25
"""
from __future__ import annotations

import argparse
import os
import sys
import csv
import time
import json
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Set, Tuple, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from koda_processor import KoDaProcessor, KoDaConfig, _parse_pb_worker


# =============================================================================
# Configuration
# =============================================================================

OPERATOR = "skane"
FEED = "VehiclePositions"
SNAPSHOTS_PER_HOUR = 30

CACHE_DIR = Path(__file__).parent / "koda_cache"
OUT_DIR = Path(__file__).parent.parent / "out"

# Memory-conscious worker count (each worker uses ~300-500MB with trip_to_route mapping).
# With 6GB limit, use max 4 workers to leave room for main process.
MAX_WORKERS = 4


# =============================================================================
# Occupancy Status Mappings
# =============================================================================

# Occupancy -> numeric score for aggregation
OCC_SCORE = {
    "EMPTY": 0,
    "MANY_SEATS_AVAILABLE": 1,
    "FEW_SEATS_AVAILABLE": 2,
    "STANDING_ROOM_ONLY": 3,
    "CRUSHED_STANDING_ROOM_ONLY": 4,
    "FULL": 5,
}

# Occupancy status names for count columns
OCC_STATUS_NAMES = [
    "EMPTY",
    "MANY_SEATS_AVAILABLE",
    "FEW_SEATS_AVAILABLE",
    "STANDING_ROOM_ONLY",
    "CRUSHED_STANDING_ROOM_ONLY",
    "FULL",
]


# =============================================================================
# Utility Functions
# =============================================================================


def ensure_dirs() -> None:
    """Create cache and output directories if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_date(s: str) -> date:
    """Parse YYYY-MM-DD string to date."""
    return date.fromisoformat(s)


def daterange(d0: date, d1: date) -> List[date]:
    """Generate list of dates from d0 to d1 inclusive."""
    dates = []
    d = d0
    while d <= d1:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def month_tag(d: date) -> str:
    """Return YYYY-MM string for the given date."""
    return f"{d.year:04d}-{d.month:02d}"


def cached_koda_download(proc: KoDaProcessor, url: str, cache_path: Path,
                         max_wait_sec: int = 10*60, poll_interval_sec: int = 30) -> bytes:
    """
    Cache only real archives (zip/7z). If KoDa says 'being prepared', poll until ready.
    """
    def is_zip(data: bytes) -> bool:
        return data[:4] == b"PK\x03\x04"

    def is_7z(data: bytes) -> bool:
        return data[:6] == b"\x37\x7A\xBC\xAF\x27\x1C"

    def looks_like_json(data: bytes) -> bool:
        return len(data) > 0 and data[:1] in (b"{", b"[")

    # Validate existing cache
    if cache_path.exists() and cache_path.stat().st_size > 0:
        data = cache_path.read_bytes()
        if is_zip(data) or is_7z(data):
            return data
        cache_path.unlink(missing_ok=True)

    start = time.time()
    while True:
        data = proc.download_bytes(url)

        if is_zip(data) or is_7z(data):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)
            return data

        if looks_like_json(data):
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except Exception:
                payload = {}

            msg = (payload.get("message") or "").lower()
            if "being prepared" in msg:
                elapsed = time.time() - start
                if elapsed > max_wait_sec:
                    raise TimeoutError(f"KoDa file not ready after {int(elapsed)}s: {url}")
                print(f"  KoDa: being prepared -> retry in {poll_interval_sec}s")
                time.sleep(poll_interval_sec)
                continue

        raise ValueError(f"Unexpected response (not zip/7z). First bytes: {data[:200]!r}")


def process_day(
    ds: str,
    proc: KoDaProcessor,
    trip_to_route: Dict[str, Tuple[str, int]],
    rt_bytes: bytes,
) -> List[Dict]:
    """
    Process a single day's RT archive and return aggregated hourly rows for all routes.
    Uses multiprocessing for parallel pb file parsing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rt_dir = Path(tmp) / "rt"
        proc.extract_archive_bytes(rt_bytes, rt_dir)

        pb_files = sorted(proc.iter_pb_files(rt_dir))
        if not pb_files:
            raise RuntimeError("No .pb files in extracted RT archive")

        selected = proc.select_snapshots_per_hour(pb_files, SNAPSHOTS_PER_HOUR)
        print(f"  Found {len(pb_files):,} snapshots, selected {len(selected):,}")

        # Parallel parse using multiprocessing
        ignore_occ = proc.ignore_occupancy
        worker_args = [(pb, trip_to_route, ignore_occ) for pb in selected]
        
        num_workers = min(MAX_WORKERS, len(selected), cpu_count())
        
        all_observations: List[Tuple[int, str, int, str]] = []
        
        if num_workers > 1 and len(selected) > 10:
            print(f"  Parsing with {num_workers} workers...")
            with Pool(processes=num_workers) as pool:
                results = pool.map(_parse_pb_worker, worker_args)
            for r in results:
                all_observations.extend(r)
        else:
            # Sequential for small batches
            for i, pb in enumerate(selected, 1):
                if i % 50 == 0 or i == len(selected):
                    print(f"    Parsing: {i}/{len(selected)}")
                rows = proc.parse_vehiclepositions_snapshot(pb, trip_to_route)
                all_observations.extend(rows)

        # Aggregate by (hour, route_id, direction_id)
        # Structure: {(hour, route_id, direction_id): {"scores": [], "counts": {...}}}
        aggregation: Dict[Tuple[int, str, int], Dict] = defaultdict(
            lambda: {"scores": [], "counts": {s: 0 for s in OCC_STATUS_NAMES}}
        )
        
        for ts_utc, route_id, direction_id, occ in all_observations:
            if occ not in OCC_SCORE:
                continue
            dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(proc.tz_local)
            if dt_local.date().isoformat() != ds:
                continue
            
            key = (dt_local.hour, route_id, direction_id)
            aggregation[key]["scores"].append(OCC_SCORE[occ])
            if occ in aggregation[key]["counts"]:
                aggregation[key]["counts"][occ] += 1

        # Build output rows
        output_rows = []
        for (hour, route_id, direction_id), data in sorted(aggregation.items()):
            scores = data["scores"]
            counts = data["counts"]
            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            max_score = max(scores)

            output_rows.append({
                "date": ds,
                "hour": hour,
                "route_id": route_id,
                "direction_id": direction_id,
                "n_obs": len(scores),
                "n_empty": counts["EMPTY"],
                "n_many_seats": counts["MANY_SEATS_AVAILABLE"],
                "n_few_seats": counts["FEW_SEATS_AVAILABLE"],
                "n_standing": counts["STANDING_ROOM_ONLY"],
                "n_crushed": counts["CRUSHED_STANDING_ROOM_ONLY"],
                "n_full": counts["FULL"],
                "max_occupancy": max_score,
                "avg_occupancy": round(avg_score, 4),
            })

        return output_rows


def save_parquet(rows: List[Dict], out_path: Path):
    """Save rows as Parquet file."""
    if not rows:
        print(f"  No data to save")
        return
    
    table = pa.Table.from_pydict({
        "date": [r["date"] for r in rows],
        "hour": [r["hour"] for r in rows],
        "route_id": [r["route_id"] for r in rows],
        "direction_id": [r["direction_id"] for r in rows],
        "n_obs": [r["n_obs"] for r in rows],
        "n_empty": [r["n_empty"] for r in rows],
        "n_many_seats": [r["n_many_seats"] for r in rows],
        "n_few_seats": [r["n_few_seats"] for r in rows],
        "n_standing": [r["n_standing"] for r in rows],
        "n_crushed": [r["n_crushed"] for r in rows],
        "n_full": [r["n_full"] for r in rows],
        "max_occupancy": [r["max_occupancy"] for r in rows],
        "avg_occupancy": [r["avg_occupancy"] for r in rows],
    })
    
    pq.write_table(table, out_path, compression="snappy")
    print(f"  Saved {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill RT occupancy data for all routes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", type=parse_date, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=parse_date, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dates", nargs="+", type=parse_date, 
                        help="Specific dates to process (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: out/rt_<start>_<end>.parquet)")
    args = parser.parse_args()

    # Determine dates to process
    dates_to_process: Set[date] = set()
    
    if args.start and args.end:
        if args.start > args.end:
            parser.error("--start must be before or equal to --end")
        dates_to_process.update(daterange(args.start, args.end))
    
    if args.dates:
        dates_to_process.update(args.dates)
    
    if not dates_to_process:
        parser.error("Must specify either --start/--end or --dates")
    
    dates_sorted = sorted(dates_to_process)
    
    # Get API key
    api_key = os.environ.get("KODA_API_KEY")
    api_key = "JI7Axc5uiOA-xf6EV85BSKI-JelejQJRe8B59Bodmw8"
    if not api_key:
        raise RuntimeError("Missing KODA_API_KEY env var")

    ensure_dirs()

    cfg = KoDaConfig(
        operator=OPERATOR,
        feed=FEED,
        snapshots_per_hour=SNAPSHOTS_PER_HOUR,
        retries=10,
        retry_sleep_sec=3.0,
        http_timeout_sec=240,
    )
    proc = KoDaProcessor(api_key=api_key, config=cfg)

    # Output file
    if args.output:
        out_path = Path(args.output)
    else:
        start_str = dates_sorted[0].isoformat()
        end_str = dates_sorted[-1].isoformat()
        out_path = OUT_DIR / f"rt_all_routes_{start_str}_{end_str}.parquet"

    print(f"Processing {len(dates_sorted)} date(s)")
    print(f"Date range: {dates_sorted[0]} to {dates_sorted[-1]}")
    print(f"Output: {out_path}")
    print(f"Cache: {CACHE_DIR}")
    print()

    all_rows: List[Dict] = []
    failed_dates: List[str] = []
    
    trip_to_route: Optional[Dict[str, Tuple[str, int]]] = None
    current_month: Optional[str] = None

    for idx, d in enumerate(dates_sorted, start=1):
        ds = d.isoformat()
        print(f"[{idx}/{len(dates_sorted)}] Processing {ds}...")

        # Load static mapping for month if needed
        m = month_tag(d)
        if m != current_month or trip_to_route is None:
            print(f"  [Static] Loading mapping for month {m}...")
            static_cache = CACHE_DIR / "static" / f"{OPERATOR}_{m}.bin"
            static_url = proc.static_url(date(d.year, d.month, 1).isoformat())

            try:
                static_bytes = cached_koda_download(proc, static_url, static_cache)
                with tempfile.TemporaryDirectory() as tmp:
                    static_dir = Path(tmp) / "static"
                    proc.extract_archive_bytes(static_bytes, static_dir)
                    trip_to_route = proc.build_trip_to_route(static_dir)
                current_month = m
                print(f"  [Static] trip->route loaded: {len(trip_to_route):,} mappings")
            except Exception as e:
                print(f"  !! Static failed for month {m}: {e}")
                failed_dates.append(ds)
                continue

        # Download RT archive
        rt_cache = CACHE_DIR / "rt" / f"{OPERATOR}_{FEED}_{ds}.bin"
        rt_url = proc.rt_url(ds)

        try:
            rt_bytes = cached_koda_download(proc, rt_url, rt_cache)
        except Exception as e:
            print(f"  !! RT download failed for {ds}: {e}")
            failed_dates.append(ds)
            continue

        # Process day
        try:
            day_rows = process_day(ds, proc, trip_to_route, rt_bytes)
            all_rows.extend(day_rows)
            
            unique_routes = len(set(r["route_id"] for r in day_rows))
            unique_combos = len(set((r["route_id"], r["direction_id"]) for r in day_rows))
            print(f"  Collected {len(day_rows)} hourly rows for {unique_routes} routes ({unique_combos} route-direction combos)")
        except Exception as e:
            print(f"  !! Processing failed for {ds}: {e}")
            failed_dates.append(ds)
            continue

        # Small delay between days
        time.sleep(0.3)

    # Save all data
    print()
    save_parquet(all_rows, out_path)

    # Report failures
    if failed_dates:
        failed_path = out_path.with_suffix(".failed.txt")
        with open(failed_path, "w") as f:
            f.write("\n".join(failed_dates))
        print(f"Failed dates ({len(failed_dates)}): {failed_path}")

    print()
    print("DONE.")
    print(f"Output: {out_path}")
    print(f"Total rows: {len(all_rows):,}")
    if all_rows:
        print(f"Unique routes: {len(set(r['route_id'] for r in all_rows)):,}")
        print(f"Unique route-direction combos: {len(set((r['route_id'], r['direction_id']) for r in all_rows)):,}")


if __name__ == "__main__":
    main()
