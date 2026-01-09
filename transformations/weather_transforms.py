"""
Weather data transformation functions.

These functions fetch weather data from OpenMeteo API and transform it
for use in Hopsworks feature groups, including previous-day weather features.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
import pandas as pd
import requests

from transformations.type_utils import (
    optimize_int_columns,
    optimize_float_columns,
)


# Weather configuration for Skåne
WEATHER_LAT = 55.605
WEATHER_LON = 13.003
WEATHER_TIMEZONE = "Europe/Stockholm"


def fetch_weather_data(
    start_date: date,
    end_date: date,
    lat: float = WEATHER_LAT,
    lon: float = WEATHER_LON,
    timezone: str = WEATHER_TIMEZONE,
    include_prev_day: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly weather data from OpenMeteo archive API.
    
    When include_prev_day is True, fetches one day before start_date
    to enable previous-day feature calculation without NaN values.
    
    Args:
        start_date: First date to fetch (or day after if include_prev_day)
        end_date: Last date to fetch
        lat: Latitude coordinate
        lon: Longitude coordinate
        timezone: Timezone string
        include_prev_day: If True, fetch extra day before start for prev calculations
        
    Returns:
        DataFrame with hourly weather data
    """
    # Adjust start date if we need previous day data
    actual_start = start_date - timedelta(days=1) if include_prev_day else start_date
    
    if actual_start > end_date:
        return pd.DataFrame()
    
    start_str = actual_start.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": ["temperature_2m", "precipitation", "windspeed_10m", "cloudcover"],
        "timezone": timezone,
    }
    
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    
    if df.empty:
        return df
    
    # Parse time column
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.normalize()
    df["hour"] = df["time"].dt.hour
    
    return df


def add_previous_day_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous day weather columns.
    
    For each hour, adds columns with the same hour's weather from the previous day:
    - prev_temperature_2m
    - prev_precipitation
    - prev_windspeed_10m
    - prev_cloudcover
    
    Args:
        df: DataFrame with weather data (must have date, hour columns)
        
    Returns:
        DataFrame with previous day weather columns added
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Sort by date and hour
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    
    weather_cols = ["temperature_2m", "precipitation", "windspeed_10m", "cloudcover"]
    
    # Create a lookup DataFrame indexed by (date, hour)
    df_indexed = df.set_index(["date", "hour"])
    
    prev_data = []
    for _, row in df.iterrows():
        current_date = row["date"]
        current_hour = row["hour"]
        prev_date = current_date - timedelta(days=1)
        
        try:
            prev_row = df_indexed.loc[(prev_date, current_hour)]
            prev_values = {f"prev_{col}": prev_row[col] for col in weather_cols}
        except KeyError:
            # Previous day not available
            prev_values = {f"prev_{col}": None for col in weather_cols}
        
        prev_data.append(prev_values)
    
    prev_df = pd.DataFrame(prev_data)
    df = pd.concat([df.reset_index(drop=True), prev_df], axis=1)
    
    return df


def prepare_weather_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize column types for weather data.
    
    Args:
        df: DataFrame with weather data
        
    Returns:
        DataFrame with optimized types
    """
    df = df.copy()
    
    # Drop the time column if present (we have date and hour)
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    
    # Ensure date is string format for Hopsworks primary key
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # Int8 columns
    int8_cols = ["hour", "cloudcover", "prev_cloudcover"]
    existing_int8 = [c for c in int8_cols if c in df.columns]
    df = optimize_int_columns(df, existing_int8, "int8")
    
    # Float32 columns
    float32_cols = [
        "temperature_2m", "precipitation", "windspeed_10m",
        "prev_temperature_2m", "prev_precipitation", "prev_windspeed_10m"
    ]
    existing_float32 = [c for c in float32_cols if c in df.columns]
    df = optimize_float_columns(df, existing_float32, "float32")
    
    return df


def add_event_time(df: pd.DataFrame, date_col: str = "date", hour_col: str = "hour") -> pd.DataFrame:
    """
    Add event_time column for Hopsworks.
    
    Args:
        df: DataFrame with date and hour columns
        date_col: Name of date column
        hour_col: Name of hour column
        
    Returns:
        DataFrame with event_time column added
    """
    df = df.copy()
    
    dt_series = pd.to_datetime(df[date_col])
    df["event_time"] = dt_series + pd.to_timedelta(df[hour_col], unit="h")
    
    return df
