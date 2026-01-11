"""
Unit tests for GTFS schedule parsing module.
"""
import pytest
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.gtfs_schedule import (
    normalize_traffic_hour,
    expand_hour_range,
    parse_time_to_hours,
    RouteInfo,
    TripTimeBounds,
)


class TestParseTimeToHours:
    """Tests for parse_time_to_hours function."""

    def test_normal_time(self):
        """Test parsing a normal time string."""
        assert parse_time_to_hours("08:30:00") == 8

    def test_zero_hour(self):
        """Test parsing midnight."""
        assert parse_time_to_hours("00:30:00") == 0

    def test_traffic_day_hour_25(self):
        """Test parsing traffic day hour 25 (1 AM next day)."""
        assert parse_time_to_hours("25:03:00") == 25

    def test_traffic_day_hour_26(self):
        """Test parsing traffic day hour 26 (2 AM next day)."""
        assert parse_time_to_hours("26:45:00") == 26

    def test_empty_string(self):
        """Test parsing empty string returns 0."""
        assert parse_time_to_hours("") == 0

    def test_malformed_string(self):
        """Test parsing malformed string returns 0."""
        assert parse_time_to_hours("invalid") == 0


class TestNormalizeTrafficHour:
    """Tests for normalize_traffic_hour function."""

    def test_normal_hour(self):
        """Test normalizing a normal hour (< 24)."""
        result = normalize_traffic_hour(10, date(2026, 1, 11))
        assert result == (date(2026, 1, 11), 10)

    def test_midnight(self):
        """Test normalizing midnight."""
        result = normalize_traffic_hour(0, date(2026, 1, 11))
        assert result == (date(2026, 1, 11), 0)

    def test_hour_23(self):
        """Test normalizing hour 23."""
        result = normalize_traffic_hour(23, date(2026, 1, 11))
        assert result == (date(2026, 1, 11), 23)

    def test_hour_24_rolls_over(self):
        """Test that hour 24 becomes 0 on the next day."""
        result = normalize_traffic_hour(24, date(2026, 1, 11))
        assert result == (date(2026, 1, 12), 0)

    def test_hour_25_rolls_over(self):
        """Test that hour 25 becomes 1 on the next day."""
        result = normalize_traffic_hour(25, date(2026, 1, 11))
        assert result == (date(2026, 1, 12), 1)

    def test_hour_26_rolls_over(self):
        """Test that hour 26 becomes 2 on the next day."""
        result = normalize_traffic_hour(26, date(2026, 1, 11))
        assert result == (date(2026, 1, 12), 2)

    def test_end_of_month_rollover(self):
        """Test rollover at end of month."""
        result = normalize_traffic_hour(25, date(2026, 1, 31))
        assert result == (date(2026, 2, 1), 1)

    def test_end_of_year_rollover(self):
        """Test rollover at end of year."""
        result = normalize_traffic_hour(25, date(2025, 12, 31))
        assert result == (date(2026, 1, 1), 1)


class TestExpandHourRange:
    """Tests for expand_hour_range function."""

    def test_single_hour(self):
        """Test expanding a single hour."""
        result = expand_hour_range(10, 10, date(2026, 1, 11))
        assert result == [(date(2026, 1, 11), 10)]

    def test_multiple_hours_same_day(self):
        """Test expanding multiple hours within the same day."""
        result = expand_hour_range(6, 9, date(2026, 1, 11))
        assert result == [
            (date(2026, 1, 11), 6),
            (date(2026, 1, 11), 7),
            (date(2026, 1, 11), 8),
            (date(2026, 1, 11), 9),
        ]

    def test_hours_crossing_midnight(self):
        """Test expanding hours that cross midnight to next day."""
        result = expand_hour_range(23, 26, date(2026, 1, 11))
        assert result == [
            (date(2026, 1, 11), 23),
            (date(2026, 1, 12), 0),
            (date(2026, 1, 12), 1),
            (date(2026, 1, 12), 2),
        ]

    def test_all_hours_after_midnight(self):
        """Test expanding hours that are all after midnight (traffic day)."""
        result = expand_hour_range(24, 26, date(2026, 1, 11))
        assert result == [
            (date(2026, 1, 12), 0),
            (date(2026, 1, 12), 1),
            (date(2026, 1, 12), 2),
        ]


class TestDataclasses:
    """Tests for dataclass structures."""

    def test_route_info_creation(self):
        """Test creating RouteInfo."""
        info = RouteInfo(route_id="123", short_name="5", long_name="Test Route")
        assert info.route_id == "123"
        assert info.short_name == "5"
        assert info.long_name == "Test Route"

    def test_trip_time_bounds_creation(self):
        """Test creating TripTimeBounds."""
        bounds = TripTimeBounds(first_arrival="06:30:00", last_arrival="25:15:00")
        assert bounds.first_arrival == "06:30:00"
        assert bounds.last_arrival == "25:15:00"


# Integration tests would require mocking HTTP requests
# These tests focus on the pure logic functions
