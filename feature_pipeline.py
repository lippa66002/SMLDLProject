import os
import sys
import time
import json
import tempfile
import requests
import hopsworks
import pandas as pd
import holidays
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict, Counter

# Ensure local imports work
root_dir = str(Path().absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

from koda_processor import KoDaProcessor, KoDaConfig

# --- Configuration ---
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
KODA_API_KEY = os.environ.get("KODA_API_KEY")

OPERATOR = "skane"
FEED = "VehiclePositions"
ROUTE_ID = "9011012065200000"
SNAPSHOTS_PER_HOUR = 10

OCC_SCORE = {
    "EMPTY": 0,
    "MANY_SEATS_AVAILABLE": 1,
    "FEW_SEATS_AVAILABLE": 2,
    "STANDING_ROOM_ONLY": 3,
    "CRUSHED_STANDING_ROOM_ONLY": 4,
    "FULL": 5,
    "NOT_ACCEPTING_PASSENGERS": 6,
}


def month_tag(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def download_koda_temporary(proc: KoDaProcessor, url: str, max_wait_sec: int = 600, poll_interval_sec: int = 30) -> bytes:
    """Downloads KoDa data to memory, handling the 'being prepared' state."""
    
    def is_valid_archive(data: bytes) -> bool:
        return data[:4] == b"PK\x03\x04" or data[:6] == b"\x37\x7A\xBC\xAF\x27\x1C"

    def looks_like_json(data: bytes) -> bool:
        return len(data) > 0 and data[:1] in (b"{", b"[")

    start = time.time()
    while True:
        data = proc.download_bytes(url)

        if is_valid_archive(data):
            return data

        if looks_like_json(data):
            try:
                payload = json.loads(data.decode("utf-8", errors="replace"))
            except Exception:
                payload = {}

            if "being prepared" in (payload.get("message") or "").lower():
                elapsed = time.time() - start
                if elapsed > max_wait_sec:
                    raise TimeoutError(f"KoDa file not ready after {int(elapsed)}s: {url}")
                print(f"  KoDa: being prepared -> retry in {poll_interval_sec}s")
                time.sleep(poll_interval_sec)
                continue

        raise ValueError(f"Unexpected response. First bytes: {data[:200]!r}")


def run_traffic_pipeline(proc, traffic_fg):
    """Checks for missing dates, downloads KoDa data, parses, and uploads."""
    print("--- Starting Traffic Pipeline ---")
    
    # 1. Determine missing dates
    today = date.today()
    yesterday = today - timedelta(days=1)
    lookback_limit = today - timedelta(days=5)

    try:
        existing_df = traffic_fg.select(["date"]).read()
        existing_dates = set(existing_df["date"].astype(str).unique()) if not existing_df.empty else set()
    except Exception as e:
        print(f"Could not read FG dates, defaulting to empty. Error: {e}")
        existing_dates = set()

    missing = []
    curr = lookback_limit
    while curr <= yesterday:
        date_str = curr.isoformat()
        if date_str not in existing_dates:
            missing.append(date_str)
        curr += timedelta(days=1)

    print(f"Checking range: {lookback_limit} -> {yesterday}")
    print(f"Missing dates to fetch: {missing}")

    if not missing:
        print("No new traffic data needed.")
        return

    # 2. Process missing dates
    new_rows = []
    trip_to_route = None
    current_month = None

    for idx, ds in enumerate(missing, start=1):
        d = date.fromisoformat(ds)
        print(f"\n[{idx}/{len(missing)}] Processing {ds} ...")

        # Static Mapping (cache per month)
        m = month_tag(d)
        if m != current_month or trip_to_route is None:
            print(f"[Static] Loading mapping for month {m} ...")
            static_url = proc.static_url(date(d.year, d.month, 1).isoformat())
            try:
                static_bytes = download_koda_temporary(proc, static_url)
                with tempfile.TemporaryDirectory() as tmp:
                    static_dir = Path(tmp) / "static"
                    proc.extract_archive_bytes(static_bytes, static_dir)
                    trip_to_route = proc.build_trip_to_route(static_dir)
                current_month = m
                print(f"[Static] trip->route loaded: {len(trip_to_route):,}")
            except Exception as e:
                print(f"  !! Static failed month {m}: {e}")
                continue

        # RT Data
        try:
            rt_bytes = download_koda_temporary(proc, proc.rt_url(ds))
        except Exception as e:
            print(f"  !! RT download failed {ds}: {e}")
            continue

        # Extract & Parse
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
                        if occ not in OCC_SCORE: continue
                        dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(proc.tz_local)
                        if dt_local.date().isoformat() != ds: continue
                        
                        scores_by_hour[dt_local.hour].append(OCC_SCORE[occ])
                        labels_by_hour[dt_local.hour].append(occ)

                # Aggregate
                day_cnt = 0
                for hour in range(24):
                    obs_scores = scores_by_hour.get(hour, [])
                    if not obs_scores: continue

                    avg_score = sum(obs_scores) / len(obs_scores)
                    c = Counter(labels_by_hour[hour])
                    mode_label, mode_count = c.most_common(1)[0]
                    
                    new_rows.append({
                        "date": ds,
                        "hour": hour,
                        "route_id": ROUTE_ID,
                        "avg_occupancy_score": float(f"{avg_score:.4f}"),
                        "label_mode": mode_label,
                        "mode_confidence": float(f"{mode_count / sum(c.values()):.4f}"),
                        "n_obs": len(obs_scores),
                        "n_snapshots_used": SNAPSHOTS_PER_HOUR,
                    })
                    day_cnt += 1
                print(f"  Generated {day_cnt} hourly rows for {ds}")

        except Exception as e:
            print(f"  !! Failed processing {ds}: {e}")
            continue
        
        time.sleep(0.6)

    # 3. Upload
    if new_rows:
        df = pd.DataFrame(new_rows)
        
        # Helper for grouped label
        def group_label(lbl):
            if pd.isna(lbl): return np.nan
            return lbl if lbl in ["EMPTY", "MANY_SEATS_AVAILABLE"] else "CROWDED"

        df["label_grouped"] = df["label_mode"].apply(group_label)
        df["event_time"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
        df.drop(columns=["mode_confidence"], inplace=True)

        traffic_fg.insert(df)
        print(f"\nUploaded {len(new_rows)} new traffic rows.")
    else:
        print("\nNo new traffic data found/processed.")


def fetch_weather_data(start_date, end_date):
    if start_date > end_date:
        return pd.DataFrame()
    
    print(f"Fetching Weather: {start_date} -> {end_date}")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 55.605, "longitude": 13.003,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "precipitation", "windspeed_10m", "cloudcover"],
        "timezone": "Europe/Stockholm"
    }

    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        df = pd.DataFrame(data.get("hourly", {}))
        if df.empty: return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        df["hour"] = df["time"].dt.hour.astype("int64")
        df["event_time"] = df["time"]
        return df.drop(columns=["time"])
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return pd.DataFrame()


def run_weather_pipeline(weather_fg):
    """Updates weather forecast up to 5 days ahead."""
    print("\n--- Starting Weather Pipeline ---")
    
    forecast_limit = date.today() + timedelta(days=5)
    lookback_limit = date.today() - timedelta(days=5)

    # Find start date based on last FG entry
    try:
        w_max = weather_fg.select(["date"]).read()["date"].max()
        w_last = datetime.strptime(w_max, "%Y-%m-%d").date()
        w_start = max(w_last + timedelta(days=1), lookback_limit)
    except Exception:
        w_start = lookback_limit

    df = fetch_weather_data(w_start, forecast_limit)
    
    if not df.empty:
        weather_fg.insert(df)
        print(f"Uploaded {len(df)} new weather rows.")
    else:
        print("Weather up to date.")


def generate_calendar_data(start_date, end_date):
    if start_date > end_date:
        return pd.DataFrame()

    print(f"Generating Calendar: {start_date} -> {end_date}")
    date_range = pd.date_range(start_date, end_date)
    se_holidays = holidays.SE()

    data = []
    for d in date_range:
        is_weekend = d.weekday() >= 5
        is_holiday = d in se_holidays
        data.append({
            "date": d.strftime("%Y-%m-%d"),
            "event_time": d,
            "year": d.year,
            "month": d.month,
            "day": d.day,
            "weekday": d.day_name(),
            "is_weekend": is_weekend,
            "is_holiday_se": is_holiday,
            "is_workday_se": not is_weekend and not is_holiday
        })
    return pd.DataFrame(data)


def run_calendar_pipeline(calendar_fg):
    """Updates calendar features up to 5 days ahead."""
    print("\n--- Starting Calendar Pipeline ---")
    
    forecast_limit = date.today() + timedelta(days=5)
    lookback_limit = date.today() - timedelta(days=5)

    try:
        c_max = calendar_fg.select(["date"]).read()["date"].max()
        c_last = datetime.strptime(c_max, "%Y-%m-%d").date()
        c_start = max(c_last + timedelta(days=1), lookback_limit)
    except Exception:
        c_start = lookback_limit

    df = generate_calendar_data(c_start, forecast_limit)

    if not df.empty:
        calendar_fg.insert(df)
        print(f"Uploaded {len(df)} new calendar rows.")
    else:
        print("Calendar up to date.")


def main():
    # Setup Hopsworks
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project="occupancy",
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    # Get Feature Groups
    traffic_fg = fs.get_feature_group(name="skane_traffic", version=1)
    weather_fg = fs.get_feature_group(name="skane_weather", version=1)
    calendar_fg = fs.get_feature_group(name="sweden_calendar", version=1)

    # Setup KoDa Processor
    cfg = KoDaConfig(
        operator=OPERATOR,
        feed=FEED,
        snapshots_per_hour=SNAPSHOTS_PER_HOUR,
        retries=10,
        retry_sleep_sec=3.0,
        http_timeout_sec=240,
    )
    proc = KoDaProcessor(api_key=KODA_API_KEY, config=cfg)

    # Run Pipelines
    run_traffic_pipeline(proc, traffic_fg)
    run_weather_pipeline(weather_fg)
    run_calendar_pipeline(calendar_fg)

    print("\nAll pipelines finished.")


if __name__ == "__main__":
    main()