"""
Type optimization utilities for feature data.

These utilities help reduce memory usage by converting columns to appropriate
smaller data types while preserving data integrity.
"""
from __future__ import annotations

from typing import List, Union
import pandas as pd
import numpy as np


def optimize_int_columns(
    df: pd.DataFrame,
    columns: List[str],
    target_type: str = "int16",
) -> pd.DataFrame:
    """
    Cast integer columns to a smaller integer type.
    
    Args:
        df: DataFrame to modify
        columns: List of column names to optimize  
        target_type: Target dtype (e.g., 'int8', 'int16', 'int32')
        
    Returns:
        DataFrame with optimized column types
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(target_type)
    return df


def optimize_float_columns(
    df: pd.DataFrame,
    columns: List[str],
    target_type: str = "float32",
) -> pd.DataFrame:
    """
    Cast float columns to a smaller float type.
    
    Args:
        df: DataFrame to modify
        columns: List of column names to optimize
        target_type: Target dtype (e.g., 'float16', 'float32')
        
    Returns:
        DataFrame with optimized column types
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(target_type)
    return df


def ensure_string_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Ensure specified columns are string type.
    
    Args:
        df: DataFrame to modify
        columns: List of column names to convert to string
        
    Returns:
        DataFrame with string columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def safe_downcast_int(
    series: pd.Series,
    min_val: int,
    max_val: int,
) -> pd.Series:
    """
    Safely downcast an integer series to the smallest type that can hold the range.
    
    Args:
        series: Pandas Series with integer values
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        Series with optimized dtype
    """
    # Check which integer type can hold the range
    if min_val >= 0 and max_val <= 255:
        return series.astype("uint8")
    elif min_val >= -128 and max_val <= 127:
        return series.astype("int8")
    elif min_val >= 0 and max_val <= 65535:
        return series.astype("uint16")
    elif min_val >= -32768 and max_val <= 32767:
        return series.astype("int16")
    elif min_val >= 0 and max_val <= 4294967295:
        return series.astype("uint32")
    elif min_val >= -2147483648 and max_val <= 2147483647:
        return series.astype("int32")
    else:
        return series.astype("int64")
