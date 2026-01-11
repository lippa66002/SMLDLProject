import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.transformations.traffic_transforms import (
    prepare_traffic_types,
    validate_observation_counts
)
from src.utils.transformations.weather_transforms import (
    add_previous_day_weather,
    prepare_weather_types
)
from src.utils.transformations.calendar_transforms import generate_calendar_data

class TestTransformations(unittest.TestCase):
    def test_prepare_traffic_types(self):
        data = {
            "date": ["2023-01-01", "2023-01-01"],
            "hour": [8, 9],
            "route_id": ["100", 200], # Mixed types to test string enforcement
            "direction_id": [0, 1],
            "n_obs": [10, 20],
            "n_empty": [5, 10],
            "n_many_seats": [5, 10],
            "n_few_seats": [0, 0],
            "n_standing": [0, 0],
            "n_crushed": [0, 0],
            "n_full": [0, 0],
            "max_occupancy": [1, 1],
            "avg_occupancy": [0.5, 0.5]
        }
        df = pd.DataFrame(data)
        
        transformed = prepare_traffic_types(df)
        
        # Check types
        self.assertTrue(pd.api.types.is_string_dtype(transformed["route_id"]))
        self.assertEqual(transformed["route_id"].iloc[1], "200")
        
        self.assertTrue(pd.api.types.is_integer_dtype(transformed["hour"]))
        self.assertEqual(transformed["hour"].iloc[0], 8)
        
        self.assertTrue(pd.api.types.is_float_dtype(transformed["avg_occupancy"]))
        
        # Check date string format
        self.assertEqual(transformed["date"].iloc[0], "2023-01-01")

    def test_validate_observation_counts(self):
        data = {
            "n_obs": [10, 20],
            "n_empty": [5, 10],
            "n_many_seats": [5, 5], # Error in 2nd row: 10+5 != 20
            "n_few_seats": [0, 0],
            "n_standing": [0, 0],
            "n_crushed": [0, 0],
            "n_full": [0, 0],
        }
        df = pd.DataFrame(data)
        
        # Should fail
        self.assertFalse(validate_observation_counts(df))
        
        # Fix it
        df.loc[1, "n_many_seats"] = 10
        self.assertTrue(validate_observation_counts(df))

    def test_add_previous_day_weather(self):
        # Create 2 days of data for 2 hours
        data = {
            "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01"), 
                     pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-02")],
            "hour": [10, 11, 10, 11],
            "temperature_2m": [10.0, 11.0, 5.0, 6.0],
            "precipitation": [0.0, 0.0, 1.0, 1.0],
            "windspeed_10m": [5.0, 5.0, 10.0, 10.0],
            "cloudcover": [50, 50, 100, 100]
        }
        df = pd.DataFrame(data)
        
        transformed = add_previous_day_weather(df)
        
        # Check shape
        self.assertIn("prev_temperature_2m", transformed.columns)
        
        # Check logic
        # Day 2 hour 10 should have prev values from Day 1 hour 10
        day2_h10 = transformed[(transformed["date"] == pd.Timestamp("2023-01-02")) & (transformed["hour"] == 10)].iloc[0]
        self.assertEqual(day2_h10["prev_temperature_2m"], 10.0)
        self.assertEqual(day2_h10["prev_precipitation"], 0.0)
        
        # Day 1 hour 10 should be NaN or None
        day1_h10 = transformed[(transformed["date"] == pd.Timestamp("2023-01-01")) & (transformed["hour"] == 10)].iloc[0]
        self.assertTrue(pd.isna(day1_h10["prev_temperature_2m"]))

    def test_prepare_calendar_data(self):
        data = {
            "date": ["2023-12-24", "2023-12-25", "2023-12-27"] 
        }
        df = pd.DataFrame(data)
        
        transformed = generate_calendar_data(df)
        
        # 2023-12-25 is Monday, Red day.
        row_25 = transformed[transformed["date"] == "2023-12-25"].iloc[0]
        self.assertTrue(row_25["is_holiday_se"])
        self.assertFalse(row_25["is_weekend"])
        self.assertFalse(row_25["is_workday_se"])
        
        row_27 = transformed[transformed["date"] == "2023-12-27"].iloc[0]
        self.assertFalse(row_27["is_holiday_se"])
        self.assertFalse(row_27["is_weekend"])
        self.assertTrue(row_27["is_workday_se"])

    def test_add_traffic_event_time(self):
        """Test that event_time is correctly created from date and hour."""
        from src.utils.transformations.traffic_transforms import add_event_time
        
        data = {
            "date": ["2023-01-01", "2023-01-02"],
            "hour": [8, 14],
        }
        df = pd.DataFrame(data)
        
        result = add_event_time(df)
        
        self.assertIn("event_time", result.columns)
        self.assertEqual(result["event_time"].iloc[0], pd.Timestamp("2023-01-01 08:00:00"))
        self.assertEqual(result["event_time"].iloc[1], pd.Timestamp("2023-01-02 14:00:00"))

    def test_add_weather_event_time(self):
        """Test weather event_time creation."""
        from src.utils.transformations.weather_transforms import add_event_time
        
        data = {
            "date": ["2023-06-15", "2023-06-15"],
            "hour": [0, 23],
        }
        df = pd.DataFrame(data)
        
        result = add_event_time(df)
        
        self.assertIn("event_time", result.columns)
        self.assertEqual(result["event_time"].iloc[0], pd.Timestamp("2023-06-15 00:00:00"))
        self.assertEqual(result["event_time"].iloc[1], pd.Timestamp("2023-06-15 23:00:00"))

    def test_add_calendar_event_time(self):
        """Test calendar event_time creation (date only, no hour)."""
        from src.utils.transformations.calendar_transforms import add_event_time
        
        data = {"date": ["2023-03-01", "2023-03-15"]}
        df = pd.DataFrame(data)
        
        result = add_event_time(df)
        
        self.assertIn("event_time", result.columns)
        self.assertEqual(result["event_time"].iloc[0], pd.Timestamp("2023-03-01"))
        self.assertEqual(result["event_time"].iloc[1], pd.Timestamp("2023-03-15"))


class TestTypeUtils(unittest.TestCase):
    """Tests for type optimization utility functions."""

    def test_optimize_int_columns_to_int8(self):
        """Test converting columns to int8."""
        from src.utils.transformations.type_utils import optimize_int_columns
        
        data = {"value": [1, 2, 3, 4, 5]}
        df = pd.DataFrame(data)
        
        result = optimize_int_columns(df, ["value"], "int8")
        
        self.assertEqual(result["value"].dtype, np.int8)

    def test_optimize_int_columns_to_int16(self):
        """Test converting columns to int16."""
        from src.utils.transformations.type_utils import optimize_int_columns
        
        data = {"count": [100, 200, 300]}
        df = pd.DataFrame(data)
        
        result = optimize_int_columns(df, ["count"], "int16")
        
        self.assertEqual(result["count"].dtype, np.int16)

    def test_optimize_int_columns_missing_column(self):
        """Test that missing columns are silently skipped."""
        from src.utils.transformations.type_utils import optimize_int_columns
        
        data = {"existing": [1, 2, 3]}
        df = pd.DataFrame(data)
        
        # Should not raise an error
        result = optimize_int_columns(df, ["existing", "nonexistent"], "int8")
        
        self.assertEqual(result["existing"].dtype, np.int8)

    def test_optimize_float_columns_to_float32(self):
        """Test converting columns to float32."""
        from src.utils.transformations.type_utils import optimize_float_columns
        
        data = {"temp": [1.5, 2.5, 3.5]}
        df = pd.DataFrame(data)
        
        result = optimize_float_columns(df, ["temp"], "float32")
        
        self.assertEqual(result["temp"].dtype, np.float32)

    def test_ensure_string_columns_from_int(self):
        """Test converting int columns to string."""
        from src.utils.transformations.type_utils import ensure_string_columns
        
        data = {"route_id": [100, 200, 300]}
        df = pd.DataFrame(data)
        
        result = ensure_string_columns(df, ["route_id"])
        
        self.assertTrue(pd.api.types.is_string_dtype(result["route_id"]))
        self.assertEqual(result["route_id"].iloc[0], "100")

    def test_ensure_string_columns_preserves_existing_strings(self):
        """Test that existing strings are preserved."""
        from src.utils.transformations.type_utils import ensure_string_columns
        
        data = {"name": ["Alice", "Bob", "Charlie"]}
        df = pd.DataFrame(data)
        
        result = ensure_string_columns(df, ["name"])
        
        self.assertEqual(result["name"].iloc[0], "Alice")

    def test_safe_downcast_int_uint8(self):
        """Test downcasting to uint8 for 0-255 range."""
        from src.utils.transformations.type_utils import safe_downcast_int
        
        series = pd.Series([0, 100, 255])
        result = safe_downcast_int(series, 0, 255)
        
        self.assertEqual(result.dtype, np.uint8)

    def test_safe_downcast_int_int8(self):
        """Test downcasting to int8 for -128 to 127 range."""
        from src.utils.transformations.type_utils import safe_downcast_int
        
        series = pd.Series([-100, 0, 100])
        result = safe_downcast_int(series, -128, 127)
        
        self.assertEqual(result.dtype, np.int8)


if __name__ == '__main__':
    unittest.main()
