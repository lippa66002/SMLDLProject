"""
Traffic data transformation functions.

These functions transform raw traffic occupancy data from parquet files
into a format suitable for Hopsworks feature groups.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union
import pandas as pd
import pyarrow.parquet as pq

from transformations.type_utils import (
    optimize_int_columns,
    optimize_float_columns,
    ensure_string_columns,
)


def load_traffic_parquets(paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Load and merge multiple parquet files into a single DataFrame.
    
    Args:
        paths: List of paths to parquet files
        
    Returns:
        Combined DataFrame with all traffic data
    """
    if not paths:
        raise ValueError("No parquet files provided")
    
    dfs = []
    for path in paths:
        df = pq.read_table(path).to_pandas()
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on primary key columns
    combined = combined.drop_duplicates(
        subset=["date", "hour", "route_id", "direction_id"],
        keep="last"
    )
    
    return combined


def prepare_traffic_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize column types for traffic data.
    
    Converts columns to appropriate smaller types:
    - route_id: string
    - direction_id, hour, max_occupancy: int8
    - n_obs, n_empty, n_many_seats, n_few_seats, n_standing, n_crushed, n_full: int16
    - avg_occupancy: float32
    
    Args:
        df: DataFrame with traffic data
        
    Returns:
        DataFrame with optimized types
    """
    df = df.copy()
    
    # String columns
    df = ensure_string_columns(df, ["route_id"])
    
    # Ensure date is string format for Hopsworks primary key
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # Small int columns (0-5 range or 0-1 range)
    int8_cols = ["direction_id", "hour", "max_occupancy"]
    df = optimize_int_columns(df, int8_cols, "int8")
    
    # Medium int columns (observation counts)
    int16_cols = [
        "n_obs", "n_empty", "n_many_seats", "n_few_seats",
        "n_standing", "n_crushed", "n_full"
    ]
    df = optimize_int_columns(df, int16_cols, "int16")
    
    # Float columns
    float32_cols = ["avg_occupancy"]
    df = optimize_float_columns(df, float32_cols, "float32")
    
    return df


def add_event_time(df: pd.DataFrame, date_col: str = "date", hour_col: str = "hour") -> pd.DataFrame:
    """
    Add event_time column for Hopsworks.
    
    Creates a timestamp by combining date and hour columns.
    
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


def validate_observation_counts(df: pd.DataFrame) -> bool:
    """
    Validate that individual occupancy counts sum to n_obs.
    
    Args:
        df: DataFrame with traffic data
        
    Returns:
        True if validation passes, False otherwise
    """
    count_cols = [
        "n_empty", "n_many_seats", "n_few_seats",
        "n_standing", "n_crushed", "n_full"
    ]
    
    # Check all columns exist
    missing = [col for col in count_cols if col not in df.columns]
    if missing or "n_obs" not in df.columns:
        return False
    
    # Sum the counts and compare to n_obs
    total_counts = df[count_cols].sum(axis=1)
    return (total_counts == df["n_obs"]).all()
