"""
Unit tests for KoDaProcessor utility methods.

Tests the pure utility functions that don't require network access or
actual GTFS data files.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from koda_processor import KoDaProcessor, KoDaConfig


class TestKoDaProcessor(unittest.TestCase):
    """Tests for KoDaProcessor utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = KoDaProcessor(api_key="test_key")

    def test_select_evenly_spaced_empty_list(self):
        """Test selecting from empty list returns empty list."""
        result = self.processor.select_evenly_spaced([], 5)
        self.assertEqual(result, [])

    def test_select_evenly_spaced_fewer_than_k(self):
        """Test that all files are returned when len(files) < k."""
        files = [Path(f"/path/{i}.pb") for i in range(3)]
        result = self.processor.select_evenly_spaced(files, 10)
        self.assertEqual(result, files)

    def test_select_evenly_spaced_exact_k(self):
        """Test that all files are returned when len(files) == k."""
        files = [Path(f"/path/{i}.pb") for i in range(5)]
        result = self.processor.select_evenly_spaced(files, 5)
        self.assertEqual(result, files)

    def test_select_evenly_spaced_more_than_k(self):
        """Test that k evenly-spaced files are selected."""
        files = [Path(f"/path/{i:02d}.pb") for i in range(10)]
        result = self.processor.select_evenly_spaced(files, 3)

        # Should select first, middle, and last
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], Path("/path/00.pb"))
        self.assertEqual(result[-1], Path("/path/09.pb"))

    def test_select_evenly_spaced_no_duplicates(self):
        """Test that no duplicate files are selected."""
        files = [Path(f"/path/{i}.pb") for i in range(100)]
        result = self.processor.select_evenly_spaced(files, 30)

        # Check no duplicates
        self.assertEqual(len(result), len(set(result)))

    def test_hour_from_path(self):
        """Test extracting hour from protobuf file path."""
        pb_path = Path("/data/2025/01/15/08/file.pb")
        hour = self.processor.hour_from_path(pb_path)
        self.assertEqual(hour, 8)

    def test_hour_from_path_midnight(self):
        """Test extracting hour 0 (midnight)."""
        pb_path = Path("/data/2025/01/15/00/file.pb")
        hour = self.processor.hour_from_path(pb_path)
        self.assertEqual(hour, 0)

    def test_hour_from_path_evening(self):
        """Test extracting hour 23."""
        pb_path = Path("/data/2025/01/15/23/file.pb")
        hour = self.processor.hour_from_path(pb_path)
        self.assertEqual(hour, 23)

    def test_group_pb_by_hour(self):
        """Test grouping protobuf files by hour."""
        pb_files = [
            Path("/data/2025/01/15/08/a.pb"),
            Path("/data/2025/01/15/08/b.pb"),
            Path("/data/2025/01/15/09/c.pb"),
            Path("/data/2025/01/15/12/d.pb"),
        ]

        grouped = self.processor.group_pb_by_hour(pb_files)

        self.assertIn(8, grouped)
        self.assertIn(9, grouped)
        self.assertIn(12, grouped)
        self.assertEqual(len(grouped[8]), 2)
        self.assertEqual(len(grouped[9]), 1)
        self.assertEqual(len(grouped[12]), 1)

    def test_group_pb_by_hour_empty(self):
        """Test grouping empty list returns empty dict."""
        grouped = self.processor.group_pb_by_hour([])
        self.assertEqual(grouped, {})

    def test_static_url_format(self):
        """Test that static URL is properly formatted."""
        url = self.processor.static_url("2025-01-15")
        
        self.assertIn("gtfs-static", url)
        self.assertIn("skane", url)
        self.assertIn("2025-01-15", url)
        self.assertIn("test_key", url)

    def test_rt_url_format(self):
        """Test that RT URL is properly formatted."""
        url = self.processor.rt_url("2025-01-15")
        
        self.assertIn("gtfs-rt", url)
        self.assertIn("skane", url)
        self.assertIn("VehiclePositions", url)
        self.assertIn("2025-01-15", url)
        self.assertIn("test_key", url)


class TestKoDaConfig(unittest.TestCase):
    """Tests for KoDaConfig dataclass."""

    def test_default_values(self):
        """Test that default config values are set correctly."""
        config = KoDaConfig()
        
        self.assertEqual(config.operator, "skane")
        self.assertEqual(config.feed, "VehiclePositions")
        self.assertEqual(config.snapshots_per_hour, 30)
        self.assertEqual(config.timezone_local, "Europe/Stockholm")

    def test_custom_values(self):
        """Test that custom config values can be set."""
        config = KoDaConfig(
            operator="custom_operator",
            feed="TripUpdates",
            snapshots_per_hour=10,
        )
        
        self.assertEqual(config.operator, "custom_operator")
        self.assertEqual(config.feed, "TripUpdates")
        self.assertEqual(config.snapshots_per_hour, 10)


if __name__ == "__main__":
    unittest.main()
