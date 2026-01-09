"""
Transformations module for feature pipeline data processing.

This module contains reusable transformation functions for:
- Traffic data (from parquet files)
- Weather data (from OpenMeteo API)
- Calendar data (holidays and date features)

These transformations are designed to be used by both the backfill pipeline
and the daily feature pipeline.
"""

from transformations.traffic_transforms import (
    load_traffic_parquets,
    prepare_traffic_types,
    add_event_time as add_traffic_event_time,
)

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

from transformations.type_utils import (
    optimize_int_columns,
    optimize_float_columns,
    ensure_string_columns,
)

__all__ = [
    # Traffic
    "load_traffic_parquets",
    "prepare_traffic_types",
    "add_traffic_event_time",
    # Weather
    "fetch_weather_data",
    "add_previous_day_weather",
    "prepare_weather_types",
    "add_weather_event_time",
    # Calendar
    "generate_calendar_data",
    "prepare_calendar_types",
    "add_calendar_event_time",
    # Utils
    "optimize_int_columns",
    "optimize_float_columns",
    "ensure_string_columns",
]
