"""
Feature Backfill Pipeline

This script orchestrates the backfill of feature groups for:
- Traffic Data (aggregated, from parquet files)
- Weather Data (fetched from OpenMeteo, with previous day history)
- Calendar Data (generated based on date range)

Steps:
    1. Load and merge all parquet files from `out/`.
    2. Compute date range (inclusive of weather history requirement).
    3. Fetch/Generate Weather and Calendar data.
    4. Validate data using Great Expectations.
    5. Upload to Hopsworks Feature Store (Version 2).
"""
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List, Tuple

import dotenv
import great_expectations as ge
import hopsworks
import pandas as pd

dotenv.load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transformations.traffic_transforms import (
    load_traffic_parquets,
    prepare_traffic_types,
    add_event_time as add_traffic_event_time,
    validate_observation_counts
)
from transformations.weather_transforms import (
    fetch_weather_data,
    add_previous_day_weather,
    prepare_weather_types,
    add_event_time as add_weather_event_time
)
from transformations.calendar_transforms import (
    generate_calendar_data,
    prepare_calendar_types,
    add_event_time as add_calendar_event_time
)

# Configuration
OUT_DIR = Path(__file__).parent.parent / "out"
HOPSWORKS_PROJECT = "occupancy"
TRAFFIC_FG_NAME = "skane_traffic"
WEATHER_FG_NAME = "skane_weather"
CALENDAR_FG_NAME = "sweden_calendar"
FG_VERSION = 2

def get_parquet_files(directory: Path) -> List[Path]:
    """Find all parquet files in the directory."""
    files = list(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    print(f"Found {len(files)} parquet files.")
    return files

def get_date_range(df: pd.DataFrame, date_col: str = "date") -> Tuple[date, date]:
    """Extract min and max dates from dataframe."""
    # Ensure date column is datetime or comparable
    dates = pd.to_datetime(df[date_col]).dt.date
    return dates.min(), dates.max()

def main() -> None:
    """
    Execute the full backfill pipeline.

    Loads traffic data from parquet files, fetches weather data,
    generates calendar features, and uploads all to Hopsworks.
    """
    # =========================================================================
    # Step 1: Load Traffic Data
    # =========================================================================
    print("\n--- Processing Traffic Data ---")
    parquet_files = get_parquet_files(OUT_DIR)
    
    traffic_df = load_traffic_parquets(parquet_files)
    
    print(f"Loaded {len(traffic_df)} traffic rows.")
    
    # Validate observation counts
    if not validate_observation_counts(traffic_df):
        raise ValueError("Traffic data validation failed: individual occupancy counts do not sum to n_obs")
    print("Traffic observation counts validated successfully.")

    traffic_df = prepare_traffic_types(traffic_df)
    traffic_df = add_traffic_event_time(traffic_df)
    
    # =========================================================================
    # Step 2: Calculate Date Range
    # =========================================================================
    min_date, max_date = get_date_range(traffic_df)
    print(f"Date range: {min_date} to {max_date}")
    
    # =========================================================================
    # Step 3: Fetch Weather Data
    # =========================================================================
    print("\n--- Processing Weather Data ---")
    # Fetch with previous day enabled
    weather_df = fetch_weather_data(
        start_date=min_date, 
        end_date=max_date,
        include_prev_day=True
    )
    
    # Add prev day columns
    weather_df = add_previous_day_weather(weather_df)
    
    # Filter to only keep the range we actually want
    weather_df = weather_df[pd.to_datetime(weather_df["date"]).dt.date >= min_date]
    
    print(f"Loaded {len(weather_df)} weather rows.")
    weather_df = prepare_weather_types(weather_df)
    weather_df = add_weather_event_time(weather_df)
    
    # =========================================================================
    # Step 4: Generate Calendar Data
    # =========================================================================
    print("\n--- Processing Calendar Data ---")
    # Create a simple df with the date range
    dates_pd = pd.date_range(start=min_date, end=max_date, freq='D')
    calendar_input_df = pd.DataFrame({"date": dates_pd})
    
    calendar_df = generate_calendar_data(calendar_input_df)
    print(f"Generated {len(calendar_df)} calendar rows.")
    
    calendar_df = prepare_calendar_types(calendar_df)
    calendar_df = add_calendar_event_time(calendar_df)
    
    # =========================================================================
    # Step 5: Upload to Hopsworks Feature Store (Version 2)
    # =========================================================================
    print("\nConnecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT,
        api_key_value=os.environ.get("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    # --- Traffic Feature Group ---
    print(f"\nCreating Expectation Suite for {TRAFFIC_FG_NAME}...")
    traffic_suite = ge.core.ExpectationSuite(expectation_suite_name="traffic_suite_v2")
    
    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "date"}
        )
    )

    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "hour", "min_value": 0, "max_value": 23}
        )
    )

    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "route_id"}
        )
    )

    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "direction_id", "value_set": [0, 1]}
        )
    )
    
    # Count columns should be >= 0
    count_cols = [
        "n_obs", "n_empty", "n_many_seats", "n_few_seats", 
        "n_standing", "n_crushed", "n_full"
    ]
    for col in count_cols:
        traffic_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": 0}
            )
        )

    # Occupancy stats
    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "avg_occupancy", "min_value": 0, "max_value": 5}
        )
    )

    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "max_occupancy", "min_value": 0, "max_value": 5}
        )
    )

    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "mode_occupancy", "min_value": 0, "max_value": 5}
        )
    )

    traffic_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "event_time"}
        )
    )

    print(f"Uploading {TRAFFIC_FG_NAME} version {FG_VERSION}...")
    traffic_fg = fs.get_or_create_feature_group(
        name=TRAFFIC_FG_NAME,
        version=FG_VERSION,
        description="Aggregated hourly occupancy data for Skånetrafiken routes",
        primary_key=["route_id", "direction_id", "date", "hour"],
        partition_key=["date"],
        event_time="event_time",
        online_enabled=False,
        expectation_suite=traffic_suite,
    )

    traffic_fg.insert(traffic_df, write_options={"wait_for_job": True}, validation_options={"run_validation": True})
    
    # Add feature descriptions for traffic
    traffic_fg.update_feature_description("date", "Date of the traffic measurement (YYYY-MM-DD)")
    traffic_fg.update_feature_description("hour", "Hour of the day (0-23)")
    traffic_fg.update_feature_description("route_id", "Unique identifier for the bus route/line")
    traffic_fg.update_feature_description("direction_id", "Direction of travel (0 or 1)")
    traffic_fg.update_feature_description("n_obs", "Total number of bus observations recorded in this hour/route")
    traffic_fg.update_feature_description("n_empty", "Count of observations with EMPTY occupancy status")
    traffic_fg.update_feature_description("n_many_seats", "Count of observations with MANY_SEATS_AVAILABLE status")
    traffic_fg.update_feature_description("n_few_seats", "Count of observations with FEW_SEATS_AVAILABLE status")
    traffic_fg.update_feature_description("n_standing", "Count of observations with STANDING_ROOM_ONLY status")
    traffic_fg.update_feature_description("n_crushed", "Count of observations with CRUSHED_STANDING_ROOM_ONLY status")
    traffic_fg.update_feature_description("n_full", "Count of observations with FULL status")
    traffic_fg.update_feature_description("avg_occupancy", "Average occupancy score (0-5 scale) for this hour/route")
    traffic_fg.update_feature_description("max_occupancy", "Maximum occupancy score (0-5) observed in this hour/route")
    traffic_fg.update_feature_description("mode_occupancy", "Most frequent occupancy score (0-5) in this hour/route")
    traffic_fg.update_feature_description("event_time", "Timestamp for point-in-time correct joins")
    
    # --- Weather Feature Group ---
    print(f"\nCreating Expectation Suite for {WEATHER_FG_NAME}...")
    weather_suite = ge.core.ExpectationSuite(expectation_suite_name="weather_suite_v2")
    
    weather_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "date"}
        )
    )
    
    weather_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "hour", "min_value": 0, "max_value": 23}
        )
    )
    
    # Weather metrics (current and previous day)
    weather_cols = ["temperature_2m", "precipitation", "windspeed_10m", "cloudcover"]
    prev_weather_cols = [f"prev_{col}" for col in weather_cols]
    
    for col in weather_cols + prev_weather_cols:
        weather_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": col}
            )
        )
    
    for col in ["precipitation", "prev_precipitation"]:
        weather_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": 0, "max_value": 50}
            )
        )

    for col in ["windspeed_10m", "prev_windspeed_10m"]:
        weather_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": 0, "max_value": 100}
            )
        )

    for col in ["cloudcover", "prev_cloudcover"]:
        weather_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": 0, "max_value": 100}
            )
        )

    for col in ["temperature_2m", "prev_temperature_2m"]:
        weather_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_between",
                kwargs={"column": col, "min_value": -50, "max_value": 50}
            )
        )
    
    weather_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "event_time"}
        )
    )
    
    print(f"Uploading {WEATHER_FG_NAME} version {FG_VERSION}...")
    weather_fg = fs.get_or_create_feature_group(
        name=WEATHER_FG_NAME,
        version=FG_VERSION,
        description="Hourly weather data for Skåne (OpenMeteo) with previous day history",
        primary_key=["date", "hour"],
        partition_key=["date"],
        event_time="event_time",
        online_enabled=False,
        expectation_suite=weather_suite,
    )
    weather_fg.insert(weather_df, write_options={"wait_for_job": True}, validation_options={"run_validation": True})
    
    # Add feature descriptions for weather
    weather_fg.update_feature_description("date", "Date of the weather measurement (YYYY-MM-DD)")
    weather_fg.update_feature_description("hour", "Hour of the day (0-23)")
    weather_fg.update_feature_description("temperature_2m", "Temperature at 2 meters above ground (Celsius)")
    weather_fg.update_feature_description("precipitation", "Precipitation amount (mm)")
    weather_fg.update_feature_description("windspeed_10m", "Wind speed at 10 meters above ground (km/h)")
    weather_fg.update_feature_description("cloudcover", "Cloud cover percentage (0-100%)")
    weather_fg.update_feature_description("prev_temperature_2m", "Previous day's temperature at same hour (Celsius)")
    weather_fg.update_feature_description("prev_precipitation", "Previous day's precipitation at same hour (mm)")
    weather_fg.update_feature_description("prev_windspeed_10m", "Previous day's wind speed at same hour (km/h)")
    weather_fg.update_feature_description("prev_cloudcover", "Previous day's cloud cover at same hour (0-100%)")
    weather_fg.update_feature_description("event_time", "Timestamp for point-in-time correct joins")
    
    # --- Calendar Feature Group ---
    print(f"\nCreating Expectation Suite for {CALENDAR_FG_NAME}...")
    calendar_suite = ge.core.ExpectationSuite(expectation_suite_name="calendar_suite_v2")
    
    calendar_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "date"}
        )
    )
    calendar_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "month", "min_value": 1, "max_value": 12}
        )
    )
    # Weekday should be Mon-Sun
    calendar_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "weekday", "value_set": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
        )
    )
    
    # Boolean flags
    for col in ["is_weekend", "is_holiday_se", "is_workday_se"]:
        calendar_suite.add_expectation(
            ge.core.ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_in_set",
                kwargs={"column": col, "value_set": [True, False]}
            )
        )

    calendar_suite.add_expectation(
        ge.core.ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "event_time"}
        )
    )
    
    print(f"Uploading {CALENDAR_FG_NAME} version {FG_VERSION}...")
    calendar_fg = fs.get_or_create_feature_group(
        name=CALENDAR_FG_NAME,
        version=FG_VERSION,
        description="Calendar and holiday data for Sweden",
        primary_key=["date"],
        event_time="event_time",
        online_enabled=False,
        expectation_suite=calendar_suite,
    )
    calendar_fg.insert(calendar_df, write_options={"wait_for_job": True}, validation_options={"run_validation": True})
    
    # Add feature descriptions for calendar
    calendar_fg.update_feature_description("date", "Date in YYYY-MM-DD format")
    calendar_fg.update_feature_description("month", "Month of the year (1-12)")
    calendar_fg.update_feature_description("weekday", "Day of the week (Monday-Sunday)")
    calendar_fg.update_feature_description("is_weekend", "True if Saturday or Sunday")
    calendar_fg.update_feature_description("is_holiday_se", "True if Swedish public holiday")
    calendar_fg.update_feature_description("is_workday_se", "True if not weekend and not Swedish holiday")
    calendar_fg.update_feature_description("event_time", "Timestamp for point-in-time correct joins")
    
    print("\nSUCCESS: All feature groups uploaded and validated!")

if __name__ == "__main__":
    main()
