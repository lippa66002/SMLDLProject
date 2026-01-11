"""
Batch Inference Pipeline for Dashboard Data Generation

This pipeline generates JSON data for an interactive dashboard showing:
- Past predictions (with actual values for comparison): last 3 days
- Future predictions: today + next 3 days

Output: docs/dashboard_data.json

Steps:
    1. Load trained models (avg_occupancy and max_occupancy)
    2. Get past traffic data from Hopsworks via Spark job (avoids Arrow timeouts)
    3. Get active routes for future dates from GTFS schedule
    4. Generate future predictions with weather forecasts
    5. Write dashboard JSON and update HTML
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for imports
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.utils.gtfs_schedule import get_active_routes_for_dates
from src.utils.transformations.calendar_transforms import generate_calendar_data
from src.utils.transformations.weather_transforms import (
    add_previous_day_weather,
    fetch_weather_data,
    fetch_weather_forecast,
)

# Import classes needed for pickle deserialization of trained models
# These classes are referenced in the pickled sklearn Pipelines
from src.pipelines.train_avg_occupancy import ClippedLGBMRegressor
from src.utils.transformations.type_utils import bool_to_int


# ============================================================================
# CONFIGURATION
# ============================================================================

HOPSWORKS_PROJECT = "occupancy"
FG_VERSION = 2

# Model configurations
MODEL_AVG_NAME = "occupancy_lgbm_clipped_avg_occupancy"
MODEL_MAX_NAME = "occupancy_lgbm_classifier_max_occupancy"  # Note: classifier, not clipped
MODEL_VERSION = 2

# Date range configuration: 7 days total (3 past + today + 3 future)
PAST_DAYS = 3
FUTURE_DAYS = 3  # Not counting today

# Output paths
DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
OUTPUT_JSON = DOCS_DIR / "dashboard_data.json"

# Feature definitions (must match training)
TRAFFIC_FEATURES = ["hour", "route_id", "direction_id"]
WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "cloudcover",
    "prev_temperature_2m",
    "prev_precipitation",
    "prev_windspeed_10m",
    "prev_cloudcover",
]
CALENDAR_FEATURES = [
    "month",
    "weekday",
    "is_weekend",
    "is_holiday_se",
    "is_workday_se",
]

ALL_FEATURES = TRAFFIC_FEATURES + WEATHER_FEATURES + CALENDAR_FEATURES


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def login_hopsworks():
    """Login to Hopsworks and return project."""
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY not set")

    return hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT,
        api_key_value=api_key,
    )


def load_models(mr) -> Tuple[Any, Any]:
    """
    Load both avg_occupancy and max_occupancy models from registry.

    Returns:
        Tuple of (avg_model, max_model) sklearn Pipeline objects
    """
    print("Loading models from Hopsworks...")

    try:
        avg_meta = mr.get_model(MODEL_AVG_NAME, version=MODEL_VERSION)
        avg_dir = avg_meta.download()
        avg_model = joblib.load(Path(avg_dir) / "model.pkl")
        print(f"  Loaded {MODEL_AVG_NAME} v{MODEL_VERSION}")
    except Exception as e:
        print(f"  Warning: Could not load avg model: {e}")
        avg_model = None

    try:
        max_meta = mr.get_model(MODEL_MAX_NAME, version=MODEL_VERSION)
        max_dir = max_meta.download()
        max_model = joblib.load(Path(max_dir) / "model.pkl")
        print(f"  Loaded {MODEL_MAX_NAME} v{MODEL_VERSION}")
    except Exception as e:
        print(f"  Warning: Could not load max model: {e}")
        max_model = None

    return avg_model, max_model


def make_predictions(
    df: pd.DataFrame  ,
    avg_model: Optional[Any],
    max_model: Optional[Any],
) -> pd.DataFrame:
    """
    Run predictions with both models.

    Input df must have all feature columns.
    Returns df with pred_avg and pred_max columns added.
    """
    df = df.copy()

    # Suppress scikit-learn warnings about feature names
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")

        if avg_model is not None:
            try:
                df["pred_avg"] = avg_model.predict(df[ALL_FEATURES])
                df["pred_avg"] = df["pred_avg"].clip(0, 5).round(2)
            except Exception as e:
                print(f"    Warning: avg prediction failed: {e}")
                df["pred_avg"] = np.nan

        if max_model is not None:
            try:
                df["pred_max"] = max_model.predict(df[ALL_FEATURES])
                # Max is typically int (0-5), so round
                df["pred_max"] = df["pred_max"].clip(0, 5).round(0).astype(int)
            except Exception as e:
                print(f"    Warning: max prediction failed: {e}")
                df["pred_max"] = np.nan

    return df


def prepare_calendar_df(dates: List[date]) -> pd.DataFrame:
    """Generate calendar features for a list of dates."""
    dates_pd = pd.DataFrame({"date": pd.to_datetime(dates)})
    calendar_df = generate_calendar_data(dates_pd)
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime("%Y-%m-%d")
    return calendar_df


# ============================================================================
# PAST PREDICTIONS (HINDCAST) - Using Hopsworks Spark Job
# ============================================================================

def fetch_past_data_via_job(
    project,
    min_date: date,
    max_date: date,
) -> Optional[pd.DataFrame]:
    """
    Fetch past traffic data via a Hopsworks Spark job.
    
    This approach runs a PySpark script server-side, avoiding Arrow Flight
    timeout issues that occur with large datasets.
    
    Args:
        project: Hopsworks project object
        min_date: Start date (inclusive)
        max_date: End date (inclusive)
        
    Returns:
        DataFrame with merged traffic/weather/calendar data, or None on failure
    """
    print("  Fetching past data via Hopsworks Spark job...")
    
    jobs_api = project.get_jobs_api()
    dataset_api = project.get_dataset_api()
    
    # Script paths
    script_local_path = Path(__file__).parent.parent / "hopsworks_jobs" / "export_past_traffic.py"
    script_remote_path = "Resources/export_past_traffic.py"
    output_remote_path = "Resources/past_traffic_data.parquet"
    
    # Upload the script
    print(f"    Uploading {script_local_path.name}...")
    try:
        if dataset_api.exists(script_remote_path):
            dataset_api.remove(script_remote_path)
        dataset_api.upload(str(script_local_path), "Resources", overwrite=True)
    except Exception as e:
        print(f"    Warning: Could not upload script: {e}")
        return None
    
    # Create or get the job
    job_name = "export_past_traffic_job"
    job = None
    
    try:
        job = jobs_api.get_job(job_name)
        if job is not None:
            print(f"    Using existing job: {job_name}")
    except Exception:
        pass  # Job doesn't exist
    
    if job is None:
        print(f"    Creating new job: {job_name}")
        try:
            spark_config = jobs_api.get_configuration("PYSPARK")
            spark_config["appPath"] = f"hdfs:///Projects/{project.name}/{script_remote_path}"
            job = jobs_api.create_job(job_name, spark_config)
        except Exception as e:
            print(f"    Could not create job: {e}")
            return None
    
    if job is None:
        print("    Failed to get or create job")
        return None
    
    # Run the job with date arguments
    args = f"{min_date.isoformat()} {max_date.isoformat()}"
    print(f"    Running job with args: {args}...")
    
    try:
        execution = job.run(args=args, await_termination=True)
        
        if not execution.success:
            print("    Job failed! Check Hopsworks UI for logs.")
            return None
            
        print("    Job completed successfully!")
    except Exception as e:
        print(f"    Job execution failed: {e}")
        return None
    
    # Download the parquet file
    print("    Downloading results...")
    try:
        if not dataset_api.exists(output_remote_path):
            print(f"    Output not found: {output_remote_path}")
            return None
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the parquet file (now a single file, not a directory)
            local_path = dataset_api.download(
                output_remote_path,
                local_path=tmp_dir,
                overwrite=True
            )
            
            # Read the parquet file
            df = pd.read_parquet(local_path)
            print(f"    Downloaded {len(df)} rows")
            return df
            
    except Exception as e:
        print(f"    Could not download results: {e}")
        return None


def generate_past_predictions(
    project,
    avg_model: Optional[Any],
    max_model: Optional[Any],
    past_dates: List[date],
) -> Dict[str, Dict]:
    """
    Generate predictions for past dates using actual traffic data.

    Uses a Hopsworks Spark job to fetch data, avoiding Arrow Flight timeouts.
    For each (date, route_id, direction_id, hour) we make predictions
    and compare with actual values.

    Returns nested dict: {route_id: {direction_id: {date: {hour: {...}}}}}
    """
    print(f"\n--- Generating Past Predictions ({len(past_dates)} days) ---")

    if not past_dates:
        return {}

    min_date = min(past_dates)
    max_date = max(past_dates)

    print(f"  Date range: {min_date} to {max_date}")

    # Fetch data via Hopsworks job
    merged_df = fetch_past_data_via_job(project, min_date, max_date)

    if merged_df is None or merged_df.empty:
        print("  No traffic data found for past dates.")
        return {}

    print(f"  Loaded {len(merged_df)} merged rows")

    # Ensure date column is string
    merged_df["date"] = merged_df["date"].astype(str)

    # Fill any NaN values in features
    for col in WEATHER_FEATURES:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    # Make predictions
    print("  Running inference...")
    merged_df = make_predictions(merged_df, avg_model, max_model)

    # Build result dict
    result: Dict[str, Dict] = {}

    for _, row in merged_df.iterrows():
        route_id = str(row["route_id"])
        direction_id = int(row["direction_id"])
        date_str = str(row["date"])
        hour = int(row["hour"])

        if route_id not in result:
            result[route_id] = {}
        if direction_id not in result[route_id]:
            result[route_id][direction_id] = {}
        if date_str not in result[route_id][direction_id]:
            result[route_id][direction_id][date_str] = {"type": "past", "hours": {}}

        hour_data = {
            "pred_avg": float(row["pred_avg"]) if pd.notna(row.get("pred_avg")) else None,
            "pred_max": int(row["pred_max"]) if pd.notna(row.get("pred_max")) else None,
            "actual_avg": float(row["avg_occupancy"]) if pd.notna(row.get("avg_occupancy")) else None,
            "actual_max": int(row["max_occupancy"]) if pd.notna(row.get("max_occupancy")) else None,
        }

        result[route_id][direction_id][date_str]["hours"][str(hour)] = hour_data

    print(f"  Generated predictions for {len(result)} routes")
    return result


# ============================================================================
# FUTURE PREDICTIONS (FORECAST)
# ============================================================================


def generate_future_predictions(
    avg_model: Optional[Any],
    max_model: Optional[Any],
    future_dates: List[date],
    route_names: Dict[str, Tuple[str, str]],
) -> Tuple[Dict[str, Dict], Dict[str, Tuple[str, str]]]:
    """
    Generate predictions for future dates.

    Uses GTFS schedule to determine active routes and weather forecasts.

    Returns:
        Tuple of:
        - Nested dict: {route_id: {direction_id: {date: {hour: {...}}}}}
        - Updated route_names dict with any new routes found
    """
    print(f"\n--- Generating Future Predictions ({len(future_dates)} days) ---")

    if not future_dates:
        return {}, route_names

    # Get active routes from GTFS schedule
    print("  Fetching GTFS schedule for active routes...")
    try:
        schedule_df = get_active_routes_for_dates(future_dates)
    except Exception as e:
        print(f"  Warning: Could not fetch GTFS schedule: {e}")
        return {}, route_names

    if schedule_df.empty:
        print("  No active routes found in GTFS schedule.")
        return {}, route_names

    print(f"  Found {len(schedule_df)} route-hour combinations")

    # Update route names from schedule
    for _, row in schedule_df.drop_duplicates("route_id").iterrows():
        route_id = str(row["route_id"])
        if route_id not in route_names:
            route_names[route_id] = (
                str(row["route_short_name"]),
                str(row["route_long_name"]),
            )

    # Get weather forecast
    print("  Fetching weather forecast...")
    min_date = min(future_dates)
    max_date = max(future_dates)

    # Get recent historical weather for prev_* columns
    hist_start = min_date - timedelta(days=1)
    try:
        hist_weather = fetch_weather_data(hist_start, min_date - timedelta(days=1), include_prev_day=False)
    except Exception:
        hist_weather = pd.DataFrame()

    # Get forecast
    try:
        forecast_weather = fetch_weather_forecast(min_date, max_date)
    except Exception as e:
        print(f"  Warning: Could not fetch weather forecast: {e}")
        forecast_weather = pd.DataFrame()

    # Combine and add prev_* columns
    if not hist_weather.empty and not forecast_weather.empty:
        weather_df = pd.concat([hist_weather, forecast_weather], ignore_index=True)
    elif not forecast_weather.empty:
        weather_df = forecast_weather
    else:
        weather_df = pd.DataFrame()

    if not weather_df.empty:
        weather_df = add_previous_day_weather(weather_df)
        weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.strftime("%Y-%m-%d")
        # Filter to only future dates
        future_date_strs = {d.isoformat() for d in future_dates}
        weather_df = weather_df[weather_df["date"].isin(future_date_strs)]
        print(f"  Weather data: {len(weather_df)} rows")

    # Generate calendar features
    print("  Generating calendar features...")
    calendar_df = prepare_calendar_df(future_dates)

    # Prepare schedule dataframe for prediction
    schedule_df["date"] = schedule_df["date"].astype(str)
    schedule_df["hour"] = schedule_df["hour"].astype(int)
    schedule_df["route_id"] = schedule_df["route_id"].astype(str)
    schedule_df["direction_id"] = schedule_df["direction_id"].astype(int)

    # Merge with calendar and weather
    merged_df = schedule_df.merge(calendar_df, on="date", how="left")

    if not weather_df.empty:
        merged_df = merged_df.merge(
            weather_df,
            on=["date", "hour"],
            how="left",
        )

    # Fill missing weather with defaults
    for col in WEATHER_FEATURES:
        if col not in merged_df.columns:
            merged_df[col] = 0
        else:
            merged_df[col] = merged_df[col].fillna(0)

    # Ensure all required columns exist
    for col in ALL_FEATURES:
        if col not in merged_df.columns:
            print(f"    Warning: Missing feature column {col}")
            merged_df[col] = 0

    # Make predictions
    print("  Running inference...")
    merged_df = make_predictions(merged_df, avg_model, max_model)

    # Build result dict
    result: Dict[str, Dict] = {}

    for _, row in merged_df.iterrows():
        route_id = str(row["route_id"])
        direction_id = int(row["direction_id"])
        date_str = str(row["date"])
        hour = int(row["hour"])

        if route_id not in result:
            result[route_id] = {}
        if direction_id not in result[route_id]:
            result[route_id][direction_id] = {}
        if date_str not in result[route_id][direction_id]:
            result[route_id][direction_id][date_str] = {"type": "future", "hours": {}}

        hour_data = {
            "pred_avg": float(row["pred_avg"]) if pd.notna(row.get("pred_avg")) else None,
            "pred_max": int(row["pred_max"]) if pd.notna(row.get("pred_max")) else None,
        }

        result[route_id][direction_id][date_str]["hours"][str(hour)] = hour_data

    print(f"  Generated predictions for {len(result)} routes")
    return result, route_names


# ============================================================================
# OUTPUT GENERATION
# ============================================================================


def merge_results(
    past: Dict[str, Dict],
    future: Dict[str, Dict],
    route_names: Dict[str, Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Merge past and future results into final dashboard structure.

    Returns:
        {
            "generated_at": "ISO timestamp",
            "routes": {
                "<route_id>": {
                    "short_name": "5",
                    "long_name": "Route description",
                    "directions": {
                        "0": {
                            "days": {
                                "2026-01-10": {"type": "past", "hours": {...}},
                                ...
                            }
                        }
                    }
                }
            }
        }
    """
    # Only include routes that have past data
    # This ensures all routes in the dashboard have both historical and forecast data
    all_route_ids = set(past.keys())

    routes_data = {}

    for route_id in sorted(all_route_ids):
        short_name, long_name = route_names.get(route_id, ("", ""))

        # Merge directions from past and future
        all_directions = set()
        if route_id in past:
            all_directions.update(past[route_id].keys())
        if route_id in future:
            all_directions.update(future[route_id].keys())

        directions_data = {}

        for direction_id in sorted(all_directions):
            days_data = {}

            # Add past days
            if route_id in past and direction_id in past[route_id]:
                for date_str, day_info in past[route_id][direction_id].items():
                    days_data[date_str] = day_info

            # Add future days
            if route_id in future and direction_id in future[route_id]:
                for date_str, day_info in future[route_id][direction_id].items():
                    days_data[date_str] = day_info

            directions_data[str(direction_id)] = {"days": days_data}

        routes_data[route_id] = {
            "short_name": short_name,
            "long_name": long_name,
            "directions": directions_data,
        }

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "routes": routes_data,
    }


def save_dashboard_json(data: Dict[str, Any], output_path: Path) -> None:
    """Save dashboard data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    file_size_kb = output_path.stat().st_size / 1024
    print(f"\nSaved dashboard data to {output_path} ({file_size_kb:.1f} KB)")


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main entry point for the batch inference pipeline."""
    print("=" * 70)
    print("BATCH INFERENCE PIPELINE - Dashboard Data Generation")
    print("=" * 70)

    # Calculate date ranges
    today = date.today()
    past_dates = [today - timedelta(days=i) for i in range(1, PAST_DAYS + 1)]
    past_dates.reverse()  # Oldest first
    future_dates = [today + timedelta(days=i) for i in range(FUTURE_DAYS + 1)]  # Includes today

    print(f"\nDate Configuration:")
    print(f"  Past dates:   {past_dates[0]} to {past_dates[-1]} ({len(past_dates)} days)")
    print(f"  Future dates: {future_dates[0]} to {future_dates[-1]} ({len(future_dates)} days)")

    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    project = login_hopsworks()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Load models
    avg_model, max_model = load_models(mr)

    if avg_model is None and max_model is None:
        print("ERROR: No models could be loaded. Exiting.")
        return

    # Track route names
    route_names: Dict[str, Tuple[str, str]] = {}

    # Generate past predictions (uses Spark job to avoid Arrow timeout)
    past_results = generate_past_predictions(project, avg_model, max_model, past_dates)

    # Extract route names from past traffic (if available)
    # The traffic_fg doesn't have route names, so we'll get them from GTFS

    # Generate future predictions (also updates route_names from GTFS)
    future_results, route_names = generate_future_predictions(
        avg_model, max_model, future_dates, route_names
    )

    # Merge and save
    print("\n--- Saving Dashboard Data ---")
    dashboard_data = merge_results(past_results, future_results, route_names)

    # Stats
    num_routes = len(dashboard_data["routes"])
    total_predictions = sum(
        sum(
            sum(len(day["hours"]) for day in dir_data["days"].values())
            for dir_data in route["directions"].values()
        )
        for route in dashboard_data["routes"].values()
    )

    print(f"  Routes: {num_routes}")
    print(f"  Total predictions: {total_predictions}")

    save_dashboard_json(dashboard_data, OUTPUT_JSON)

    print("\n" + "=" * 70)
    print("BATCH INFERENCE COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()