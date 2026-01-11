"""
GTFS Schedule Module

This module provides utilities for fetching and parsing GTFS Regional Static data
to determine which routes are active on specific dates and their operating hours.

Uses:
    - GTFS Regional Static from opendata.samtrafiken.se (skane_extra.zip)
    - Standard GTFS static files (trips.txt, stop_times.txt, routes.txt)

Key Features:
    - Determines active trips for a given date range
    - Extracts first/last operating hours per trip  
    - Handles "traffic day" convention (hours > 24 mapped to next calendar day)
    - Aggregates to route-level operating windows
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import requests

# Configuration
SAMTRAFIKEN_BASE_URL = "https://opendata.samtrafiken.se/gtfs/skane"
EXTRA_ZIP_FILENAME = "skane_extra.zip"


@dataclass
class RouteInfo:
    """Information about a route from routes.txt."""

    route_id: str
    short_name: str
    long_name: str


@dataclass
class TripTimeBounds:
    """First and last arrival times for a trip."""

    first_arrival: str  # HH:MM:SS format (may be > 24:00:00)
    last_arrival: str


@dataclass
class ActiveRouteHour:
    """A single hour slot when a route is active."""

    date: date
    route_id: str
    direction_id: int
    hour: int
    route_short_name: str
    route_long_name: str


def fetch_gtfs_zip(url: str, api_key: str, timeout: int = 120) -> bytes:
    """
    Fetch a GTFS zip file from samtrafiken.se.

    Args:
        url: Full URL to the zip file
        api_key: API key for authentication (appended as ?key=)
        timeout: Request timeout in seconds

    Returns:
        Raw bytes of the zip file
    """
    # Use query parameter authentication (not header)
    full_url = f"{url}?key={api_key}"
    resp = requests.get(full_url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def fetch_skane_extra(api_key: Optional[str] = None) -> bytes:
    """
    Fetch the skane_extra.zip from samtrafiken.se.

    This zip contains trips_dated_vehicle_journey.txt with date-specific trip info.
    The file contains data for all upcoming operating days, so it only needs to be
    fetched once per pipeline run regardless of how many future dates are requested.

    Args:
        api_key: API key for GTFS Regional Static. If None, reads from env.

    Returns:
        Raw bytes of skane_extra.zip
    """
    if api_key is None:
        api_key = os.environ.get("GTFS_REGIONAL_API_KEY")
    if not api_key:
        raise RuntimeError("GTFS_REGIONAL_API_KEY not set")

    url = f"{SAMTRAFIKEN_BASE_URL}/{EXTRA_ZIP_FILENAME}"
    return fetch_gtfs_zip(url, api_key)


def fetch_skane_static(api_key: Optional[str] = None) -> bytes:
    """
    Fetch the main skane.zip GTFS static file from samtrafiken.se.

    Contains trips.txt, stop_times.txt, routes.txt, etc.

    Args:
        api_key: API key for GTFS Regional Static. If None, reads from env.

    Returns:
        Raw bytes of skane.zip
    """
    if api_key is None:
        api_key = os.environ.get("GTFS_REGIONAL_API_KEY")
    if not api_key:
        raise RuntimeError("GTFS_REGIONAL_API_KEY not set")

    url = f"{SAMTRAFIKEN_BASE_URL}/skane.zip"
    return fetch_gtfs_zip(url, api_key)


def parse_trips_dated_vehicle_journey(
    extra_zip_bytes: bytes, target_dates: Set[date]
) -> Dict[str, Set[date]]:
    """
    Parse trips_dated_vehicle_journey.txt to find trip_ids active on target dates.

    This file has format: trip_id,operating_day_date,dated_vehicle_journey_gid,journey_number
    The operating_day_date is in YYYYMMDD format (no dashes).

    Args:
        extra_zip_bytes: Raw bytes of skane_extra.zip
        target_dates: Set of dates to filter by

    Returns:
        Dict mapping trip_id -> set of dates it operates on (within target_dates)
    """
    # GTFS uses YYYYMMDD format (no dashes), not ISO format
    target_date_strs = {d.strftime("%Y%m%d") for d in target_dates}
    trip_dates: Dict[str, Set[date]] = defaultdict(set)

    with zipfile.ZipFile(io.BytesIO(extra_zip_bytes), "r") as zf:
        # Find the file (may be in root or subfolder)
        target_name = "trips_dated_vehicle_journey.txt"
        matching_files = [n for n in zf.namelist() if n.endswith(target_name)]

        if not matching_files:
            raise FileNotFoundError(
                f"{target_name} not found in skane_extra.zip"
            )

        with zf.open(matching_files[0], "r") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                op_date_str = row.get("operating_day_date", "")
                if op_date_str in target_date_strs:
                    trip_id = row.get("trip_id", "")
                    if trip_id:
                        # Parse YYYYMMDD format
                        parsed_date = datetime.strptime(op_date_str, "%Y%m%d").date()
                        trip_dates[trip_id].add(parsed_date)

    return dict(trip_dates)


def parse_stop_times_bounds(
    static_zip_bytes: bytes, trip_ids: Optional[Set[str]] = None
) -> Dict[str, TripTimeBounds]:
    """
    Parse stop_times.txt to extract first and last arrival times for each trip.

    Uses the fact that stop_times are ordered by stop_sequence within each trip,
    so the first occurrence is the first stop and we track until we see a new trip_id.

    Args:
        static_zip_bytes: Raw bytes of the main GTFS static zip
        trip_ids: Optional set of trip_ids to filter. If None, parse all.

    Returns:
        Dict mapping trip_id -> TripTimeBounds(first_arrival, last_arrival)
    """
    bounds: Dict[str, TripTimeBounds] = {}
    current_trip: Optional[str] = None
    current_first: Optional[str] = None
    current_last: Optional[str] = None

    with zipfile.ZipFile(io.BytesIO(static_zip_bytes), "r") as zf:
        target_name = "stop_times.txt"
        matching_files = [n for n in zf.namelist() if n.endswith(target_name)]

        if not matching_files:
            raise FileNotFoundError(f"{target_name} not found in static zip")

        with zf.open(matching_files[0], "r") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                trip_id = row.get("trip_id", "")

                # Filter if trip_ids provided
                if trip_ids is not None and trip_id not in trip_ids:
                    continue

                arrival_time = row.get("arrival_time", "")

                if trip_id != current_trip:
                    # Save previous trip if exists
                    if current_trip and current_first and current_last:
                        bounds[current_trip] = TripTimeBounds(
                            current_first, current_last
                        )

                    # Start new trip
                    current_trip = trip_id
                    current_first = arrival_time
                    current_last = arrival_time
                else:
                    # Same trip, update last arrival
                    current_last = arrival_time

            # Don't forget the last trip
            if current_trip and current_first and current_last:
                bounds[current_trip] = TripTimeBounds(current_first, current_last)

    return bounds


def parse_trips_route_direction(
    static_zip_bytes: bytes, trip_ids: Optional[Set[str]] = None
) -> Dict[str, Tuple[str, int]]:
    """
    Parse trips.txt to map trip_id -> (route_id, direction_id).

    Args:
        static_zip_bytes: Raw bytes of the main GTFS static zip
        trip_ids: Optional set of trip_ids to filter. If None, parse all.

    Returns:
        Dict mapping trip_id -> (route_id, direction_id)
    """
    mapping: Dict[str, Tuple[str, int]] = {}

    with zipfile.ZipFile(io.BytesIO(static_zip_bytes), "r") as zf:
        target_name = "trips.txt"
        matching_files = [n for n in zf.namelist() if n.endswith(target_name)]

        if not matching_files:
            raise FileNotFoundError(f"{target_name} not found in static zip")

        with zf.open(matching_files[0], "r") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                trip_id = row.get("trip_id", "")

                if trip_ids is not None and trip_id not in trip_ids:
                    continue

                route_id = row.get("route_id", "")
                direction_str = row.get("direction_id", "0")

                try:
                    direction_id = int(direction_str) if direction_str else 0
                except ValueError:
                    direction_id = 0

                if trip_id and route_id:
                    mapping[trip_id] = (route_id, direction_id)

    return mapping


def parse_routes_names(static_zip_bytes: bytes) -> Dict[str, RouteInfo]:
    """
    Parse routes.txt to get route_short_name and route_long_name.

    Args:
        static_zip_bytes: Raw bytes of the main GTFS static zip

    Returns:
        Dict mapping route_id -> RouteInfo
    """
    routes: Dict[str, RouteInfo] = {}

    with zipfile.ZipFile(io.BytesIO(static_zip_bytes), "r") as zf:
        target_name = "routes.txt"
        matching_files = [n for n in zf.namelist() if n.endswith(target_name)]

        if not matching_files:
            raise FileNotFoundError(f"{target_name} not found in static zip")

        with zf.open(matching_files[0], "r") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                route_id = row.get("route_id", "")
                short_name = row.get("route_short_name", "")
                long_name = row.get("route_long_name", "")

                if route_id:
                    routes[route_id] = RouteInfo(
                        route_id=route_id,
                        short_name=short_name,
                        long_name=long_name,
                    )

    return routes


def parse_time_to_hours(time_str: str) -> int:
    """
    Parse a GTFS time string (HH:MM:SS) to hours.

    GTFS uses "traffic day" convention where hours can exceed 24
    to indicate next calendar day.

    Args:
        time_str: Time in HH:MM:SS format (e.g., "25:03:00")

    Returns:
        Hour component as integer (may be >= 24)
    """
    if not time_str:
        return 0
    parts = time_str.split(":")
    if not parts:
        return 0
    try:
        return int(parts[0])
    except ValueError:
        return 0


def normalize_traffic_hour(
    traffic_hour: int, traffic_day: date
) -> Tuple[date, int]:
    """
    Convert a "traffic hour" to an actual calendar date and hour.

    Traffic hours >= 24 belong to the next calendar day.
    E.g., hour 25 on Jan 11 = hour 1 on Jan 12.

    Args:
        traffic_hour: Hour in traffic day convention (0-30+)
        traffic_day: The "traffic day" (operating day)

    Returns:
        Tuple of (actual_calendar_date, actual_hour_0_23)
    """
    if traffic_hour >= 24:
        actual_date = traffic_day + timedelta(days=1)
        actual_hour = traffic_hour - 24
    else:
        actual_date = traffic_day
        actual_hour = traffic_hour

    # Clamp hour to valid range
    actual_hour = min(23, max(0, actual_hour))

    return actual_date, actual_hour


def expand_hour_range(
    first_hour: int, last_hour: int, traffic_day: date
) -> List[Tuple[date, int]]:
    """
    Expand a range of traffic hours to a list of (date, hour) tuples.

    Handles hours > 24 by converting to next calendar day.

    Args:
        first_hour: First active hour (traffic day convention)
        last_hour: Last active hour (traffic day convention)
        traffic_day: The operating/traffic day

    Returns:
        List of (calendar_date, hour) tuples for all active hours
    """
    result = []
    for h in range(first_hour, last_hour + 1):
        actual_date, actual_hour = normalize_traffic_hour(h, traffic_day)
        result.append((actual_date, actual_hour))
    return result


def get_active_routes_for_dates(
    target_dates: List[date],
    api_key: Optional[str] = None,
    progress_callback: Optional[callable] = None,
    local_extra_zip: Optional[str] = None,
    local_static_zip: Optional[str] = None,
) -> pd.DataFrame:
    """
    Get all active routes and their operating hours for the given dates.

    This is the main entry point that orchestrates all parsing.

    Args:
        target_dates: List of dates to get active routes for
        api_key: GTFS API key. If None, reads from environment.
        progress_callback: Optional callback(step, total, message) for progress
        local_extra_zip: Optional path to local skane_extra.zip for testing
        local_static_zip: Optional path to local skane.zip for testing

    Returns:
        DataFrame with columns:
            - date: calendar date (actual, not traffic day)
            - hour: hour of day (0-23)
            - route_id: route identifier
            - direction_id: 0 or 1
            - route_short_name: line number (e.g., "5")
            - route_long_name: route description (e.g., "Västra Hamnen - Limhamn")
    """
    if not target_dates:
        return pd.DataFrame()

    target_set = set(target_dates)

    def log(step: int, total: int, msg: str):
        if progress_callback:
            progress_callback(step, total, msg)
        else:
            print(f"  [{step}/{total}] {msg}")

    total_steps = 6

    # Step 1: Fetch or load skane_extra.zip
    if local_extra_zip:
        log(1, total_steps, f"Loading local {local_extra_zip}...")
        with open(local_extra_zip, "rb") as f:
            extra_bytes = f.read()
    else:
        log(1, total_steps, "Fetching skane_extra.zip...")
        extra_bytes = fetch_skane_extra(api_key)

    # Step 2: Parse trips_dated_vehicle_journey for target dates
    log(2, total_steps, "Parsing trips_dated_vehicle_journey.txt...")
    trip_dates = parse_trips_dated_vehicle_journey(extra_bytes, target_set)
    active_trip_ids = set(trip_dates.keys())
    log(2, total_steps, f"Found {len(active_trip_ids)} active trips")

    if not active_trip_ids:
        return pd.DataFrame()

    # Step 3: Fetch or load main static GTFS
    if local_static_zip:
        log(3, total_steps, f"Loading local {local_static_zip}...")
        with open(local_static_zip, "rb") as f:
            static_bytes = f.read()
    else:
        log(3, total_steps, "Fetching skane.zip (static GTFS)...")
        static_bytes = fetch_skane_static(api_key)

    # Step 4: Parse stop_times for time bounds
    log(4, total_steps, "Parsing stop_times.txt for trip bounds...")
    trip_bounds = parse_stop_times_bounds(static_bytes, active_trip_ids)

    # Step 5: Parse trips.txt for route/direction mapping
    log(5, total_steps, "Parsing trips.txt for route/direction...")
    trip_route_dir = parse_trips_route_direction(static_bytes, active_trip_ids)

    # Step 6: Parse routes.txt for names
    log(6, total_steps, "Parsing routes.txt for route names...")
    route_info = parse_routes_names(static_bytes)

    # Now aggregate: for each route+direction+date, find the overall first/last hour
    # Key: (date, route_id, direction_id) -> set of all hours
    route_hours: Dict[Tuple[date, str, int], Set[int]] = defaultdict(set)

    for trip_id, operating_dates in trip_dates.items():
        # Get route and direction
        route_dir = trip_route_dir.get(trip_id)
        if not route_dir:
            continue
        route_id, direction_id = route_dir

        # Get time bounds
        bounds = trip_bounds.get(trip_id)
        if not bounds:
            continue

        first_hour = parse_time_to_hours(bounds.first_arrival)
        last_hour = parse_time_to_hours(bounds.last_arrival)

        # For each date this trip operates
        for traffic_day in operating_dates:
            # Expand to all hours and normalize to calendar dates
            for cal_date, cal_hour in expand_hour_range(
                first_hour, last_hour, traffic_day
            ):
                # Only include if the calendar date is in our target
                if cal_date in target_set:
                    route_hours[(cal_date, route_id, direction_id)].add(cal_hour)

    # Build output rows
    rows = []
    for (cal_date, route_id, direction_id), hours in sorted(route_hours.items()):
        info = route_info.get(route_id, RouteInfo(route_id, "", ""))

        for hour in sorted(hours):
            rows.append({
                "date": cal_date,
                "hour": hour,
                "route_id": route_id,
                "direction_id": direction_id,
                "route_short_name": info.short_name,
                "route_long_name": info.long_name,
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        # Convert date to string for consistency
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["direction_id"] = df["direction_id"].astype(int)
        df["hour"] = df["hour"].astype(int)

    return df


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def get_route_schedule_summary(
    target_dates: List[date],
    api_key: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Get a summary of route schedules grouped by route_id.

    Returns a dict like:
    {
        "route_123": {
            "short_name": "5",
            "long_name": "Västra Hamnen - Limhamn",
            "directions": {
                0: {
                    "2026-01-11": [6, 7, 8, 9, ...],  # active hours
                    ...
                },
                1: {...}
            }
        }
    }
    """
    df = get_active_routes_for_dates(target_dates, api_key)

    if df.empty:
        return {}

    summary: Dict[str, Dict] = {}

    for route_id in df["route_id"].unique():
        route_df = df[df["route_id"] == route_id]
        first_row = route_df.iloc[0]

        route_summary = {
            "short_name": first_row["route_short_name"],
            "long_name": first_row["route_long_name"],
            "directions": {},
        }

        for direction_id in route_df["direction_id"].unique():
            dir_df = route_df[route_df["direction_id"] == direction_id]
            dates_hours: Dict[str, List[int]] = {}

            for d in dir_df["date"].unique():
                hours = sorted(dir_df[dir_df["date"] == d]["hour"].tolist())
                dates_hours[d] = hours

            route_summary["directions"][int(direction_id)] = dates_hours

        summary[route_id] = route_summary

    return summary


if __name__ == "__main__":
    # Quick test
    from datetime import date, timedelta
    from dotenv import load_dotenv

    load_dotenv()

    today = date.today()
    dates = [today + timedelta(days=i) for i in range(4)]  # Today + 3 days

    print(f"Testing with dates: {dates}")
    df = get_active_routes_for_dates(dates, local_static_zip="data/skane.zip", local_extra_zip="data/skane_extra.zip")
    print(f"\nResult: {len(df)} rows")
    print(df.head(20) if not df.empty else "No data")
