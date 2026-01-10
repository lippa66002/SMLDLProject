"""
Unit tests for the LightGBM occupancy training pipeline.

Tests cover preprocessing, model pipeline construction, and evaluation utilities.
"""
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from train_avg_occupancy_lgbm import (
    build_preprocessor,
    get_hyperparameter_grid,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BOOLEAN_FEATURES,
)


class TestPreprocessor(unittest.TestCase):
    """Tests for the preprocessing pipeline."""

    def setUp(self):
        """Create sample data for testing."""
        self.sample_data = pd.DataFrame({
            # Numeric features
            "hour": [8, 12, 18, 23],
            "month": [1, 6, 12, 3],
            "temperature_2m": [5.0, 20.0, 10.0, 15.0],
            "precipitation": [0.0, 2.5, 0.0, 1.0],
            "windspeed_10m": [10.0, 5.0, 15.0, 8.0],
            "cloudcover": [50, 100, 0, 75],
            "prev_temperature_2m": [4.0, 19.0, 12.0, 14.0],
            "prev_precipitation": [0.0, 0.0, 1.0, 0.5],
            "prev_windspeed_10m": [12.0, 6.0, 10.0, 9.0],
            "prev_cloudcover": [60, 80, 20, 70],
            # Categorical features
            "route_id": ["100", "200", "100", "300"],
            "direction_id": ["0", "1", "0", "1"],
            "weekday": ["Monday", "Wednesday", "Friday", "Sunday"],
            # Boolean features
            "is_weekend": [False, False, False, True],
            "is_holiday_se": [False, False, True, False],
            "is_workday_se": [True, True, False, False],
        })

    def test_preprocessor_builds_successfully(self):
        """Test that the preprocessor can be built without errors."""
        preprocessor = build_preprocessor()
        self.assertIsNotNone(preprocessor)

    def test_preprocessor_transforms_data(self):
        """Test that the preprocessor can transform sample data."""
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(self.sample_data)
        
        # Should have transformed all features
        expected_n_features = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(BOOLEAN_FEATURES)
        self.assertEqual(transformed.shape[1], expected_n_features)
        self.assertEqual(transformed.shape[0], len(self.sample_data))

    def test_preprocessor_handles_missing_values(self):
        """Test that the preprocessor handles NaN values."""
        data_with_nan = self.sample_data.copy()
        data_with_nan.loc[0, "temperature_2m"] = np.nan
        data_with_nan.loc[1, "route_id"] = np.nan
        
        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(data_with_nan)
        
        # Should not contain any NaN after imputation
        self.assertFalse(np.isnan(transformed).any())

    def test_preprocessor_handles_unknown_categories(self):
        """Test that the preprocessor handles unseen categories at transform time."""
        preprocessor = build_preprocessor()
        preprocessor.fit(self.sample_data)
        
        # Create data with unseen category
        new_data = self.sample_data.copy()
        new_data.loc[0, "route_id"] = "UNSEEN_ROUTE"
        new_data.loc[0, "weekday"] = "UnknownDay"
        
        # Should not raise an error
        transformed = preprocessor.transform(new_data)
        self.assertEqual(transformed.shape[0], len(new_data))


class TestHyperparameterGrid(unittest.TestCase):
    """Tests for the hyperparameter search configuration."""

    def test_grid_contains_required_params(self):
        """Test that the grid contains all expected hyperparameters."""
        grid = get_hyperparameter_grid()
        
        required_params = [
            "model__n_estimators",
            "model__learning_rate",
            "model__max_depth",
            "model__num_leaves",
        ]
        
        for param in required_params:
            self.assertIn(param, grid)

    def test_grid_values_are_valid(self):
        """Test that hyperparameter values are within reasonable ranges."""
        grid = get_hyperparameter_grid()
        
        # n_estimators should be positive
        for n in grid["model__n_estimators"]:
            self.assertGreater(n, 0)
        
        # learning_rate should be between 0 and 1
        for lr in grid["model__learning_rate"]:
            self.assertGreater(lr, 0)
            self.assertLessEqual(lr, 1)
        
        # max_depth should be positive or -1 (unlimited)
        for md in grid["model__max_depth"]:
            self.assertTrue(md > 0 or md == -1)


class TestFeatureConfiguration(unittest.TestCase):
    """Tests for feature configuration constants."""

    def test_feature_lists_not_empty(self):
        """Test that feature lists are not empty."""
        self.assertGreater(len(CATEGORICAL_FEATURES), 0)
        self.assertGreater(len(NUMERIC_FEATURES), 0)
        self.assertGreater(len(BOOLEAN_FEATURES), 0)

    def test_no_duplicate_features(self):
        """Test that there are no duplicate features across lists."""
        all_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BOOLEAN_FEATURES
        self.assertEqual(len(all_features), len(set(all_features)))

    def test_expected_features_present(self):
        """Test that key expected features are present."""
        self.assertIn("hour", NUMERIC_FEATURES)
        self.assertIn("route_id", CATEGORICAL_FEATURES)
        self.assertIn("is_weekend", BOOLEAN_FEATURES)


if __name__ == "__main__":
    unittest.main()
