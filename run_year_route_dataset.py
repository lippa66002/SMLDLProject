import os
import csv
import time
import json
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict, Counter

from koda_processor import KoDaProcessor, KoDaConfig


# ---------------- SETTINGS ----------------
OPERATOR = "skane"
FEED = "VehiclePositions"
ROUTE_ID = "9011012065200000"

# Pick your year range
START = date(2024, 10, 1)
END   = date(2025, 8, 31)

SNAPSHOTS_PER_HOUR = 10  # set 4 if you're desperate for speed

# Put cache OUTSIDE OneDrive for speed
CACHE_DIR = Path(r"C:\koda_cache")
RAW_DIR = CACHE_DIR / "raw"

OUT_DIR = Path("out")
OUT_CSV = OUT_DIR / f"{OPERATOR}_{START}_{END}_route_{ROUTE_ID}_hourly.csv"
FAILED_DATES = OUT_DIR / f"{OPERATOR}_{START}_{END}_route_{ROUTE_ID}_failed_dates.txt"
# -----------------------------------------


# Occupancy -> numeric score for "hour average"
OCC_SCORE = {
    "EMPTY": 0,
    "MANY_SEATS_AVAILABLE": 1,
    "FEW_SEATS_AVAILABLE": 2,
    "STANDING_ROOM_ONLY": 3,
    "CRUSHED_STANDING_ROOM_ONLY": 4,
    "FULL": 5,
    "NOT_ACCEPTING_PASSENGERS": 6,
}


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def load_completed_dates(out_csv: Path) -> set[str]:
    if not out_csv.exists():
        return set()
    done = set()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["date"])
    return done


def month_tag(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def cached_koda_download(proc: KoDaProcessor, url: str, cache_path: Path,
                         max_wait_sec: int = 20*60, poll_interval_sec: int = 30) -> bytes:
    """
    Cache only real archives (zip/7z). If KoDa says 'being prepared', poll until ready.
    If cache contains JSON (bad cache from earlier runs), delete and refetch.
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
        # Bad cache -> delete
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


def main():
    api_key = os.environ.get("KODA_API_KEY")
    if not api_key:
        raise RuntimeError("Missing KODA_API_KEY env var. PowerShell: $env:KODA_API_KEY='...'")

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

    total_days = (END - START).days + 1
    completed = load_completed_dates(OUT_CSV)

    all_dates = [d.isoformat() for d in daterange(START, END)]
    missing = [ds for ds in all_dates if ds not in completed]

    print("Output:", OUT_CSV)
    print("Cache :", RAW_DIR)
    print(f"Days total: {total_days}, already done: {len(completed)}, missing: {len(missing)}")

    # Output CSV append
    file_exists = OUT_CSV.exists()
    out_f = OUT_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out_f,
        fieldnames=[
            "date", "hour", "route_id",
            "avg_occupancy_score",
            "label_mode", "mode_confidence",
            "n_obs", "n_snapshots_used",
        ],
    )
    if not file_exists:
        writer.writeheader()
        out_f.flush()

    trip_to_route = None
    current_month = None

    for idx, ds in enumerate(missing, start=1):
        d = date.fromisoformat(ds)
        print(f"\n[{idx}/{len(missing)}] Processing {ds} ...")

        # Monthly static mapping
        m = month_tag(d)
        if m != current_month or trip_to_route is None:
            print(f"[Static] Loading mapping for month {m} ...")
            static_cache = RAW_DIR / "static" / f"{OPERATOR}_{m}.bin"
            static_url = proc.static_url(date(d.year, d.month, 1).isoformat())

            try:
                static_bytes = cached_koda_download(proc, static_url, static_cache)
                with tempfile.TemporaryDirectory() as tmp:
                    static_dir = Path(tmp) / "static"
                    proc.extract_archive_bytes(static_bytes, static_dir)
                    trip_to_route = proc.build_trip_to_route(static_dir)
                current_month = m
                print(f"[Static] trip->route loaded: {len(trip_to_route):,}")
            except Exception as e:
                print(f"  !! Static failed month {m}: {e}")
                FAILED_DATES.parent.mkdir(parents=True, exist_ok=True)
                with FAILED_DATES.open("a", encoding="utf-8") as ff:
                    ff.write(ds + "\n")
                continue

        # Daily RT
        rt_cache = RAW_DIR / "rt" / f"{OPERATOR}_{FEED}_{ds}.bin"
        rt_url = proc.rt_url(ds)

        try:
            rt_bytes = cached_koda_download(proc, rt_url, rt_cache)
        except Exception as e:
            print(f"  !! RT download failed {ds}: {e}")
            with FAILED_DATES.open("a", encoding="utf-8") as ff:
                ff.write(ds + "\n")
            continue

        # Extract + pick snapshots + parse
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rt_dir = Path(tmp) / "rt"
                proc.extract_archive_bytes(rt_bytes, rt_dir)

                pb_files = sorted(proc.iter_pb_files(rt_dir))
                if not pb_files:
                    raise RuntimeError("No .pb files in extracted RT archive")

                selected = proc.select_snapshots_per_hour(pb_files, SNAPSHOTS_PER_HOUR)
                print(f"  Found {len(pb_files):,} snapshots, selected {len(selected):,}")

                scores_by_hour = defaultdict(list)
                labels_by_hour = defaultdict(list)

                for i, pb in enumerate(selected, start=1):
                    if i % 50 == 0 or i == len(selected):
                        print(f"    Parsing snapshots: {i}/{len(selected)}")

                    rows = proc.parse_vehiclepositions_snapshot(
                        pb_path=pb,
                        trip_to_route=trip_to_route,
                        target_route_id=ROUTE_ID,
                    )

                    for ts_utc, _, occ in rows:
                        if occ not in OCC_SCORE:
                            continue
                        dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(proc.tz_local)
                        if dt_local.date().isoformat() != ds:
                            continue
                        scores_by_hour[dt_local.hour].append(OCC_SCORE[occ])
                        labels_by_hour[dt_local.hour].append(occ)

                day_rows = 0
                for hour in range(24):
                    obs_scores = scores_by_hour.get(hour, [])
                    obs_labels = labels_by_hour.get(hour, [])
                    if not obs_scores:
                        continue

                    avg_score = sum(obs_scores) / len(obs_scores)
                    c = Counter(obs_labels)
                    mode_label, mode_count = c.most_common(1)[0]
                    conf = mode_count / sum(c.values())

                    writer.writerow({
                        "date": ds,
                        "hour": hour,
                        "route_id": ROUTE_ID,
                        "avg_occupancy_score": f"{avg_score:.4f}",
                        "label_mode": mode_label,
                        "mode_confidence": f"{conf:.4f}",
                        "n_obs": len(obs_scores),
                        "n_snapshots_used": SNAPSHOTS_PER_HOUR,
                    })
                    day_rows += 1

                out_f.flush()
                print(f"  Wrote {day_rows} hourly rows for {ds}")

        except Exception as e:
            print(f"  !! Failed processing {ds}: {e}")
            with FAILED_DATES.open("a", encoding="utf-8") as ff:
                ff.write(ds + "\n")
            continue

        time.sleep(0.6)

    out_f.close()
    print("\nDONE.")
    print("Output:", OUT_CSV)
    print("Failed dates log:", FAILED_DATES)


if __name__ == "__main__":
    main()
