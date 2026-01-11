"""
Calendar data transformation functions.

These functions generate calendar features (holidays, weekends, workdays) from
date data for use in Hopsworks feature groups.
"""
from __future__ import annotations

import holidays
import pandas as pd

from .type_utils import optimize_int_columns

def generate_calendar_data(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Enriches a dataframe of dates with calendar features (holidays, weekends, etc).
    
    Args:
        df: DataFrame containing at least a 'date' column.
        date_col: Name of the date column.
        
    Returns:
        DataFrame with added calendar features.
    """
    df = df.copy()
    
    # Ensure datetime objects
    dt_series = pd.to_datetime(df[date_col])
    
    df["month"] = dt_series.dt.month
    df["weekday"] = dt_series.dt.day_name()
    df["is_weekend"] = dt_series.dt.dayofweek >= 5
    
    # Swedish Holidays
    se_holidays = holidays.SE()
    df["is_holiday_se"] = dt_series.apply(lambda x: x in se_holidays)
    
    # Workday: Not weekend AND not holiday
    df["is_workday_se"] = (~df["is_weekend"]) & (~df["is_holiday_se"])
    
    # Convert date back to string for Hopsworks
    df[date_col] = dt_series.dt.strftime("%Y-%m-%d")
    
    return df

def prepare_calendar_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize column types for calendar data.

    Converts month column to int16 for efficient storage.

    Args:
        df: DataFrame with calendar data.

    Returns:
        DataFrame with optimized column types.
    """
    df = df.copy()

    # Int columns
    int_cols = ["month"]
    df = optimize_int_columns(df, int_cols, "int16")

    return df


def add_event_time(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add event_time column for Hopsworks.

    For calendar data, event_time is the date timestamp (no hourly component).

    Args:
        df: DataFrame with date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with event_time column added.
    """
    df = df.copy()
    df["event_time"] = pd.to_datetime(df[date_col])
    return df
