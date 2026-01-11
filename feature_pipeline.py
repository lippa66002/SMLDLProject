"""
Feature Pipeline for daily incremental updates.

This script processes daily traffic data from KoDa, fetches historical weather data,
and generates calendar features. Only historical data (up to yesterday) is saved to 
feature groups. Weather forecasts are fetched for inference but NOT stored.

Steps:
    1. Check for missing traffic dates (lookback window).
    2. Download and process KoDa RT data for missing dates.
    3. Fetch historical weather data (only for dates where traffic exists).
    4. Generate calendar features (only for dates where traffic exists).
    5. Upload to Hopsworks Feature Store (Version 2).
"""
from __future__ import annotations

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import dotenv
import hopsworks
import pandas as pd

dotenv.load_dotenv()

# Ensure local imports work
root_dir = str(Path().absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)

from koda_processor import KoDaProcessor, KoDaConfig
from transformations.weather_transforms import (
    fetch_weather_data,
    add_previous_day_weather,
    prepare_weather_types,
    add_event_time as add_weather_event_time,
)
from transformations.calendar_transforms import (
    generate_calendar_data,
    prepare_calendar_types,
    add_event_time as add_calendar_event_time,
)
from transformations.traffic_transforms import (
    prepare_traffic_types,
    add_event_time as add_traffic_event_time,
    validate_observation_counts,
)

# --- Configuration ---
HOPSWORKS_API_KEY = os.environ.get("HOPSWORKS_API_KEY")
KODA_API_KEY = os.environ.get("KODA_API_KEY")

OPERATOR = "skane"
FEED = "VehiclePositions"
SNAPSHOTS_PER_HOUR = 30
LOOKBACK_DAYS = 5

HOPSWORKS_PROJECT = "occupancy"
TRAFFIC_FG_NAME = "skane_traffic"
WEATHER_FG_NAME = "skane_weather"
CALENDAR_FG_NAME = "sweden_calendar"
FG_VERSION = 2

# Occupancy status mappings
OCC_SCORE = {
    "EMPTY": 0,
    "MANY_SEATS_AVAILABLE": 1,
    "FEW_SEATS_AVAILABLE": 2,
    "STANDING_ROOM_ONLY": 3,
    "CRUSHED_STANDING_ROOM_ONLY": 4,
    "FULL": 5,
}

OCC_STATUS_NAMES = [
    "EMPTY",
    "MANY_SEATS_AVAILABLE",
    "FEW_SEATS_AVAILABLE",
    "STANDING_ROOM_ONLY",
    "CRUSHED_STANDING_ROOM_ONLY",
    "FULL",
]


def month_tag(d: date) -> str:
    """Return YYYY-MM string for the given date."""
    return f"{d.year:04d}-{d.month:02d}"


def download_koda_temporary(
    proc: KoDaProcessor, url: str, max_wait_sec: int = 600, poll_interval_sec: int = 30
) -> bytes:
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
                    raise TimeoutError(
                        f"KoDa file not ready after {int(elapsed)}s: {url}"
                    )
                print(f"  KoDa: being prepared -> retry in {poll_interval_sec}s")
                time.sleep(poll_interval_sec)
                continue

        raise ValueError(f"Unexpected response. First bytes: {data[:200]!r}")


def process_day(
    ds: str,
    proc: KoDaProcessor,
    trip_to_route: Dict[str, Tuple[str, int]],
    rt_bytes: bytes,
) -> List[Dict]:
    """
    Process a single day's RT archive and return aggregated hourly rows for all routes.
    Matches the backfill approach with direction_id and per-status counts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rt_dir = Path(tmp) / "rt"
        proc.extract_archive_bytes(rt_bytes, rt_dir)

        pb_files = sorted(proc.iter_pb_files(rt_dir))
        if not pb_files:
            raise RuntimeError("No .pb files in extracted RT archive")

        selected = proc.select_snapshots_per_hour(pb_files, SNAPSHOTS_PER_HOUR)
        print(f"  Found {len(pb_files):,} snapshots, selected {len(selected):,}")

        # Sequential parsing (sufficient for daily incremental updates)
        all_observations: List[Tuple[int, str, int, str]] = []
        for i, pb in enumerate(selected, 1):
            if i % 50 == 0 or i == len(selected):
                print(f"    Parsing: {i}/{len(selected)}")
            rows = proc.parse_vehiclepositions_snapshot(pb, trip_to_route)
            all_observations.extend(rows)

        # Aggregate by (hour, route_id, direction_id)
        aggregation: Dict[Tuple[int, str, int], Dict] = defaultdict(
            lambda: {"scores": [], "counts": {s: 0 for s in OCC_STATUS_NAMES}}
        )

        for ts_utc, route_id, direction_id, occ in all_observations:
            if occ not in OCC_SCORE:
                continue
            dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(
                proc.tz_local
            )
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

            output_rows.append(
                {
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
                    "avg_occupancy": round(sum(scores) / len(scores), 4),
                    "max_occupancy": max(scores),
                    "mode_occupancy": max(set(scores), key=scores.count),
                }
            )

        return output_rows


def get_existing_dates_via_job(project, lookback_date: date) -> set:
    """
    Get existing dates from the traffic feature group using a Hopsworks Spark job.
    
    This approach runs a PySpark script server-side, avoiding Arrow Flight issues.
    The script queries the feature group and writes dates to a file that we download.
    
    Args:
        project: Hopsworks project object
        lookback_date: Oldest date to query
        
    Returns:
        Set of date strings (YYYY-MM-DD format)
    """
    print("Getting existing dates via Hopsworks Spark job...")
    
    jobs_api = project.get_jobs_api()
    dataset_api = project.get_dataset_api()
    
    # Path to the PySpark script (relative to project root)
    script_local_path = Path(__file__).parent / "hopsworks_jobs" / "get_existing_dates.py"
    script_remote_path = "Resources/get_existing_dates.py"
    output_remote_file = "Resources/existing_dates.txt"  # Single file output
    
    # Upload the script to Hopsworks (overwrite if exists)
    print(f"  Uploading {script_local_path.name} to Hopsworks...")
    try:
        if dataset_api.exists(script_remote_path):
            dataset_api.remove(script_remote_path)
        dataset_api.upload(str(script_local_path), "Resources", overwrite=True)
    except Exception as e:
        print(f"  Warning: Could not upload script: {e}")
        return set()
    
    # Create or get the job
    job_name = "get_existing_dates_job"
    job = None
    
    try:
        job = jobs_api.get_job(job_name)
        if job is not None:
            print(f"  Using existing job: {job_name}")
    except Exception:
        pass  # Job doesn't exist
    
    if job is None:
        # Job doesn't exist or retrieval failed, create it
        print(f"  Creating new job: {job_name}")
        try:
            spark_config = jobs_api.get_configuration("PYSPARK")
            spark_config["appPath"] = f"hdfs:///Projects/{project.name}/{script_remote_path}"
            job = jobs_api.create_job(job_name, spark_config)
        except Exception as e:
            print(f"  Could not create job: {e}")
            return set()
    
    if job is None:
        print("  Failed to get or create job")
        return set()
    
    # Run the job with the lookback date as argument
    lookback_str = lookback_date.isoformat()
    print(f"  Running job with lookback_date={lookback_str}...")
    
    try:
        execution = job.run(args=lookback_str, await_termination=True)
        
        if not execution.success:
            print(f"  Job failed! Check Hopsworks UI for logs.")
            return set()
            
        print(f"  Job completed successfully!")
    except Exception as e:
        print(f"  Job execution failed: {e}")
        return set()
    
    # Download the result file (single .txt file)
    try:
        if not dataset_api.exists(output_remote_file):
            print(f"  Output file not found: {output_remote_file}")
            return set()
        
        # Download the file to temp
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = dataset_api.download(
                output_remote_file, 
                local_path=tmp_dir,
                overwrite=True
            )
            
            # Read dates from the file
            dates = set()
            with open(local_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dates.add(line)
        
        print(f"  Found {len(dates)} existing dates")
        return dates
        
    except Exception as e:
        print(f"  Could not download results: {e}")
        return set()



def run_traffic_pipeline(proc: KoDaProcessor, traffic_fg, project) -> List[str]:
    """
    Check for missing dates, download KoDa data, parse, and upload.
    Returns list of dates that were processed (for weather/calendar sync).
    """
    print("--- Starting Traffic Pipeline ---")

    # 1. Determine date range to process (only historical: up to yesterday)
    today = date.today()
    yesterday = today - timedelta(days=1)
    lookback_limit = today - timedelta(days=LOOKBACK_DAYS)

    # Build list of all dates in lookback window
    dates_to_check = []
    curr = lookback_limit
    while curr <= yesterday:
        dates_to_check.append(curr.isoformat())
        curr += timedelta(days=1)

    print(f"Checking range: {lookback_limit} -> {yesterday}")
    print(f"Dates in window: {dates_to_check}")

    # 2. Get existing dates via Hopsworks Spark job (avoids Arrow Flight issues)
    existing_dates = get_existing_dates_via_job(project, lookback_limit)
    
    if not existing_dates:
        print("Could not determine existing dates. Will process all dates in window.")
        sys.exit(1)

    # 3. Determine which dates are missing
    missing = [d for d in dates_to_check if d not in existing_dates]
    print(f"Missing dates to fetch: {missing}")

    if not missing:
        print("No new traffic data needed.")
        return []

    # 2. Process missing dates
    new_rows = []
    trip_to_route: Optional[Dict[str, Tuple[str, int]]] = None
    current_month: Optional[str] = None
    processed_dates: List[str] = []

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
                with tempfile.TemporaryDirectory() as tmp_dir:
                    static_dir = Path(tmp_dir) / "static"
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

        # Process day
        try:
            day_rows = process_day(ds, proc, trip_to_route, rt_bytes)
            new_rows.extend(day_rows)

            if day_rows:
                unique_routes = len(set(r["route_id"] for r in day_rows))
                unique_combos = len(
                    set((r["route_id"], r["direction_id"]) for r in day_rows)
                )
                print(
                    f"  Collected {len(day_rows)} hourly rows for {unique_routes} routes ({unique_combos} route-direction combos)"
                )
                processed_dates.append(ds)
            else:
                print(f"  No data collected for {ds}")
        except Exception as e:
            print(f"  !! Failed processing {ds}: {e}")
            continue

        time.sleep(0.6)

    # 3. Upload
    if new_rows:
        df = pd.DataFrame(new_rows)

        # Validate observation counts
        if not validate_observation_counts(df):
            print(
                "WARNING: Observation count validation failed. Proceeding anyway."
            )

        df = prepare_traffic_types(df)
        df = add_traffic_event_time(df)

        traffic_fg.insert(
            df,
            write_options={"wait_for_job": True},
        )
        print(f"\nUploaded {len(new_rows)} new traffic rows.")
    else:
        print("\nNo new traffic data found/processed.")

    return processed_dates


def run_weather_pipeline(weather_fg, traffic_dates: List[str]) -> None:
    """
    Update weather data ONLY for historical dates where traffic data exists.
    Weather forecasts are NOT saved to the feature group.
    """
    print("\n--- Starting Weather Pipeline (Historical Only) ---")

    if not traffic_dates:
        # Check existing traffic dates if no new ones were processed
        print("No new traffic dates. Checking for missing weather in existing range...")
        try:
            traffic_df = weather_fg.select(["date"]).read()
            if not traffic_df.empty:
                print("Weather feature group already has data.")
                return
        except Exception:
            pass
        print("Weather up to date.")
        return

    # Determine date range from traffic dates
    traffic_date_objs = sorted([date.fromisoformat(d) for d in traffic_dates])
    min_date = traffic_date_objs[0]
    max_date = traffic_date_objs[-1]

    # Only fetch historical data (never future forecasts for feature group)
    today = date.today()
    if max_date >= today:
        max_date = today - timedelta(days=1)

    if min_date > max_date:
        print("No historical weather dates to fetch.")
        return

    print(f"Fetching weather for: {min_date} to {max_date}")

    # Fetch with previous day enabled for prev_* columns
    weather_df = fetch_weather_data(
        start_date=min_date, end_date=max_date, include_prev_day=True
    )

    if weather_df.empty:
        print("No weather data returned from API.")
        return

    # Add previous day columns
    weather_df = add_previous_day_weather(weather_df)

    # Filter to only keep the range we actually want (exclude the extra prev day)
    weather_df = weather_df[
        pd.to_datetime(weather_df["date"]).dt.date >= min_date
    ].copy()

    print(f"Prepared {len(weather_df)} weather rows.")

    weather_df = prepare_weather_types(weather_df)
    weather_df = add_weather_event_time(weather_df)

    # Use Hudi settings to avoid small file overhead for tiny datasets
    # Setting small.file.limit=0 disables bin-packing which causes slowdowns
    weather_fg.insert(
        weather_df,
        write_options={
            "wait_for_job": True,
            "hoodie.parquet.small.file.limit": "0",
        },
    )
    print(f"Uploaded {len(weather_df)} weather rows.")


def run_calendar_pipeline(calendar_fg, traffic_dates: List[str]) -> None:
    """
    Update calendar data ONLY for dates where traffic data exists.
    Future dates are NOT saved to the feature group.
    """
    print("\n--- Starting Calendar Pipeline (Historical Only) ---")

    if not traffic_dates:
        print("No new traffic dates. Calendar up to date.")
        return

    # Determine date range from traffic dates
    traffic_date_objs = sorted([date.fromisoformat(d) for d in traffic_dates])
    min_date = traffic_date_objs[0]
    max_date = traffic_date_objs[-1]

    # Only generate for historical data (never future dates for feature group)
    today = date.today()
    if max_date >= today:
        max_date = today - timedelta(days=1)

    if min_date > max_date:
        print("No historical calendar dates to generate.")
        return

    print(f"Generating calendar for: {min_date} to {max_date}")

    # Create date range DataFrame
    dates_pd = pd.date_range(start=min_date, end=max_date, freq="D")
    calendar_input_df = pd.DataFrame({"date": dates_pd})

    calendar_df = generate_calendar_data(calendar_input_df)
    print(f"Generated {len(calendar_df)} calendar rows.")

    calendar_df = prepare_calendar_types(calendar_df)
    calendar_df = add_calendar_event_time(calendar_df)

    # Use Hudi settings to avoid small file overhead for tiny datasets
    calendar_fg.insert(
        calendar_df,
        write_options={
            "wait_for_job": True,
            "hoodie.parquet.small.file.limit": "0",
        },
    )
    print(f"Uploaded {len(calendar_df)} calendar rows.")


def main():
    """Main entry point for the feature pipeline."""
    # Setup Hopsworks
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    # Get Feature Groups (Version 2)
    traffic_fg = fs.get_feature_group(name=TRAFFIC_FG_NAME, version=FG_VERSION)
    weather_fg = fs.get_feature_group(name=WEATHER_FG_NAME, version=FG_VERSION)
    calendar_fg = fs.get_feature_group(name=CALENDAR_FG_NAME, version=FG_VERSION)

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
    # Traffic pipeline returns the dates that were processed
    processed_traffic_dates = run_traffic_pipeline(proc, traffic_fg, project)

    # Calendar and Weather only update for dates where traffic was processed
    # Calendar runs first since it's faster and less prone to issues
    run_calendar_pipeline(calendar_fg, processed_traffic_dates)
    run_weather_pipeline(weather_fg, processed_traffic_dates)

    print("\nAll pipelines finished.")


if __name__ == "__main__":
    main()