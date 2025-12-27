import os
import io
import csv
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict, Counter

from koda_processor import KoDaProcessor, KoDaConfig


# ----------------- USER SETTINGS -----------------
OPERATOR = "skane"
FEED = "VehiclePositions"
ROUTE_ID = "9011012065200000"

START = date(2025, 9, 1)
END = date(2025, 9, 30)

SNAPSHOTS_PER_HOUR = 10

# Where we store downloads + outputs
CACHE_DIR = Path("cache")
RAW_DIR = CACHE_DIR / "raw"
OUT_DIR = Path("out")
OUT_CSV = OUT_DIR / f"{OPERATOR}_{START}_{END}_route_{ROUTE_ID}_hourly.csv"
# -------------------------------------------------


# Occupancy -> numeric score (for “average of snapshots”)
OCC_SCORE = {
    "EMPTY": 0,
    "MANY_SEATS_AVAILABLE": 1,
    "FEW_SEATS_AVAILABLE": 2,
    "STANDING_ROOM_ONLY": 3,
    "CRUSHED_STANDING_ROOM_ONLY": 4,
    "FULL": 5,
    "NOT_ACCEPTING_PASSENGERS": 6,
}

# For printing route health while running
def summarize_day(day_rows):
    """Compute simple day diagnostics for monitoring route quality."""
    if not day_rows:
        return {"hours_with_data": 0, "avg_conf": 0.0, "avg_obs": 0.0}
    hours_with_data = len(day_rows)
    avg_conf = sum(float(r["mode_confidence"]) for r in day_rows) / hours_with_data
    avg_obs = sum(int(r["n_obs"]) for r in day_rows) / hours_with_data
    return {"hours_with_data": hours_with_data, "avg_conf": avg_conf, "avg_obs": avg_obs}


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def load_completed_dates(out_csv: Path) -> set[str]:
    """
    Resume mechanism:
    If output CSV exists, read which dates are already present and skip them.
    """
    if not out_csv.exists():
        return set()

    done = set()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["date"])
    return done


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def cached_download(proc: KoDaProcessor, url: str, cache_path: Path) -> bytes:
    """
    Raw caching:
    - If cache_path exists, load bytes from disk (no API call).
    - Else download once and save.
    """
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    data = proc.download_bytes(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def main():
    api_key = os.environ.get("KODA_API_KEY")
    if not api_key:
        raise RuntimeError("Set KODA_API_KEY first. PowerShell: $env:KODA_API_KEY='...'")

    ensure_dirs()

    # Configure the processor:
    # - more retries helps against DNS hiccups
    cfg = KoDaConfig(
        operator=OPERATOR,
        feed=FEED,
        snapshots_per_hour=SNAPSHOTS_PER_HOUR,
        retries=10,
        retry_sleep_sec=3.0,
        http_timeout_sec=240,
    )
    proc = KoDaProcessor(api_key=api_key, config=cfg)

    # Resume: skip days already processed
    completed_dates = load_completed_dates(OUT_CSV)
    total_days = (END - START).days + 1
    print(f"Output: {OUT_CSV}")
    print(f"Resume: {len(completed_dates)} dates already in output, will skip them.")
    print(f"Planned days: {total_days}")

    # ----------------------------
    # 1) Download static ONCE (for the month)
    # ----------------------------
    month_tag = f"{START.year:04d}-{START.month:02d}"
    static_cache = RAW_DIR / "static" / f"{OPERATOR}_{month_tag}.bin"

    # Use START date to fetch static. (Good enough for a 1-month prototype.)
    static_url = proc.static_url(START.isoformat())
    print(f"\n[Static] Using one static download for month: {month_tag}")
    static_bytes = cached_download(proc, static_url, static_cache)

    # Extract static once and build trip->route mapping once
    # (We do this in-memory with a temp dir, but only once per run.)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        static_dir = tmp_path / "static"
        proc.extract_archive_bytes(static_bytes, static_dir)
        trip_to_route = proc.build_trip_to_route(static_dir)

    print(f"[Static] trip->route mapping loaded: {len(trip_to_route):,}")

    # ----------------------------
    # 2) Prepare output file (append mode)
    # ----------------------------
    file_exists = OUT_CSV.exists()
    out_f = OUT_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out_f,
        fieldnames=[
            "date",
            "hour",
            "route_id",
            "avg_occupancy_score",
            "label_mode",
            "mode_confidence",
            "n_obs",
            "n_snapshots_used",
        ],
    )
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    # ----------------------------
    # 3) Loop days (resumable)
    # ----------------------------
    processed = 0
    failed = 0

    for d in daterange(START, END):
        ds = d.isoformat()

        if ds in completed_dates:
            processed += 1
            print(f"[{processed}/{total_days}] {ds} already done, skipping.")
            continue

        print(f"\n[{processed+1}/{total_days}] Processing {ds} ...")

        # Download realtime archive with caching
        rt_cache = RAW_DIR / "rt" / f"{OPERATOR}_{FEED}_{ds}.bin"
        rt_url = proc.rt_url(ds)

        try:
            rt_bytes = cached_download(proc, rt_url, rt_cache)
        except Exception as e:
            failed += 1
            print(f"  !! Download failed for {ds}: {e}")
            continue

        # Extract RT and find pb files
        with tempfile.TemporaryDirectory() as tmp2:
            tmp2_path = Path(tmp2)
            rt_dir = tmp2_path / "rt"
            try:
                proc.extract_archive_bytes(rt_bytes, rt_dir)
            except Exception as e:
                failed += 1
                print(f"  !! Extract failed for {ds}: {e}")
                continue

            pb_files = sorted(proc.iter_pb_files(rt_dir))
            if not pb_files:
                failed += 1
                print(f"  !! No .pb files found for {ds}")
                continue

            # Select 10 snapshots/hour (evenly spaced)
            selected = proc.select_snapshots_per_hour(pb_files, k_per_hour=SNAPSHOTS_PER_HOUR)
            print(f"  Found {len(pb_files):,} snapshots, selected {len(selected):,} ({SNAPSHOTS_PER_HOUR}/hour)")

            # Aggregate per hour: we collect all occupancy scores for the target route
            scores_by_hour = defaultdict(list)
            labels_by_hour = defaultdict(list)

            # Progress inside a day
            for i, pb in enumerate(selected, start=1):
                if i % 50 == 0 or i == len(selected):
                    print(f"    Parsing snapshots: {i}/{len(selected)}")

                rows = proc.parse_vehiclepositions_snapshot(
                    pb_path=pb,
                    trip_to_route=trip_to_route,
                    target_route_id=ROUTE_ID,
                )

                # rows: (timestamp_utc, route_id, occupancy_status_name)
                for ts_utc, route_id, occ in rows:
                    # Keep only statuses we can score
                    if occ not in OCC_SCORE:
                        continue
                    # Convert timestamp to local hour using proc aggregate helper logic
                    # (Use the processor timezone)
                    from datetime import datetime, timezone
                    dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(proc.tz_local)

                    # Ensure we only write for the same local day we're processing
                    if dt_local.date().isoformat() != ds:
                        continue

                    scores_by_hour[dt_local.hour].append(OCC_SCORE[occ])
                    labels_by_hour[dt_local.hour].append(occ)

            # Build 24 hourly rows (or fewer if missing)
            day_rows = []
            for hour in range(24):
                obs_scores = scores_by_hour.get(hour, [])
                obs_labels = labels_by_hour.get(hour, [])
                if not obs_scores:
                    continue

                avg_score = sum(obs_scores) / len(obs_scores)

                # Mode label + confidence
                c = Counter(obs_labels)
                mode_label, mode_count = c.most_common(1)[0]
                conf = mode_count / sum(c.values())

                row = {
                    "date": ds,
                    "hour": hour,
                    "route_id": ROUTE_ID,
                    "avg_occupancy_score": f"{avg_score:.4f}",
                    "label_mode": mode_label,
                    "mode_confidence": f"{conf:.4f}",
                    "n_obs": len(obs_scores),
                    "n_snapshots_used": SNAPSHOTS_PER_HOUR,
                }
                writer.writerow(row)
                day_rows.append(row)

            out_f.flush()

            # Monitoring: how “good” is this route day-to-day?
            stats = summarize_day(day_rows)
            print(
                f"  Day summary: hours_with_data={stats['hours_with_data']}/24, "
                f"avg_conf={stats['avg_conf']:.3f}, avg_obs/hour={stats['avg_obs']:.1f}"
            )

        processed += 1

        # Polite throttle to reduce rate-limits (tiny sleep helps a lot)
        import time
        time.sleep(0.6)

    out_f.close()

    print("\n==== DONE ====")
    print(f"Output written to: {OUT_CSV.resolve()}")
    print(f"Processed days: {processed}/{total_days}")
    print(f"Failed days: {failed}")
    if failed > 0:
        print("If failures are rate-limit/network, rerun. It will resume and only do missing dates.")


if __name__ == "__main__":
    main()
