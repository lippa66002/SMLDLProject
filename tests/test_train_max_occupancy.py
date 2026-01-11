"""
Unit tests for the LightGBM max occupancy classification pipeline.

Tests cover preprocessing, model configuration, and custom metrics.
"""
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.train_max_occupancy import (
    build_preprocessor,
    get_hyperparameter_grid,
    get_class_weights,
    compute_sample_weights,
    ordinal_weighted_recall,
    ordinal_cost,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    BOOLEAN_FEATURES,
    NUM_CLASSES,
    CLASS_NAMES,
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
        
        expected_n_features = len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(BOOLEAN_FEATURES)
        self.assertEqual(transformed.shape[1], expected_n_features)
        self.assertEqual(transformed.shape[0], len(self.sample_data))


class TestClassWeights(unittest.TestCase):
    """Tests for class weight configuration."""

    def test_class_weights_all_present(self):
        """Test that all classes have weights defined."""
        weights = get_class_weights()
        for i in range(NUM_CLASSES):
            self.assertIn(i, weights)

    def test_class_weights_increase_with_crowdedness(self):
        """Test that weights increase for more crowded classes."""
        weights = get_class_weights()
        for i in range(1, NUM_CLASSES):
            self.assertGreaterEqual(weights[i], weights[i-1])

    def test_full_class_has_highest_weight(self):
        """Test that FULL (class 5) has the highest weight."""
        weights = get_class_weights()
        max_weight = max(weights.values())
        self.assertEqual(weights[5], max_weight)


class TestSampleWeights(unittest.TestCase):
    """Tests for sample weight computation."""

    def test_sample_weights_valid_range(self):
        """Test that sample weights are positive."""
        y = np.array([0, 1, 2, 3, 4, 5])
        weights = compute_sample_weights(y)
        self.assertTrue(np.all(weights > 0))

    def test_sample_weights_increase_with_class(self):
        """Test that sample weights increase with class value."""
        y = np.array([0, 1, 2, 3, 4, 5])
        weights = compute_sample_weights(y)
        for i in range(1, len(weights)):
            self.assertGreater(weights[i], weights[i-1])


class TestOrdinalMetrics(unittest.TestCase):
    """Tests for ordinal-aware metrics."""

    def test_weighted_recall_perfect_predictions(self):
        """Test that perfect predictions give high weighted recall."""
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        recall = ordinal_weighted_recall(y_true, y_pred)
        self.assertEqual(recall, 1.0)

    def test_ordinal_cost_zero_for_perfect(self):
        """Test that ordinal cost is zero for perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([0, 1, 2, 3, 4, 5])
        cost = ordinal_cost(y_true, y_pred)
        self.assertEqual(cost, 0.0)

    def test_ordinal_cost_asymmetric(self):
        """Test that underestimation is penalized more than overestimation."""
        # Underestimation: predicting 0 when actual is 5
        y_true_under = np.array([5])
        y_pred_under = np.array([0])
        cost_under = ordinal_cost(y_true_under, y_pred_under)

        # Overestimation: predicting 5 when actual is 0
        y_true_over = np.array([0])
        y_pred_over = np.array([5])
        cost_over = ordinal_cost(y_true_over, y_pred_over)

        # Underestimation should cost more
        self.assertGreater(cost_under, cost_over)


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


class TestClassNames(unittest.TestCase):
    """Tests for class configuration."""

    def test_num_classes_matches_names(self):
        """Test that NUM_CLASSES matches the length of CLASS_NAMES."""
        self.assertEqual(NUM_CLASSES, len(CLASS_NAMES))

    def test_class_names_are_valid(self):
        """Test that class names are non-empty strings."""
        for name in CLASS_NAMES:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)


if __name__ == "__main__":
    unittest.main()
