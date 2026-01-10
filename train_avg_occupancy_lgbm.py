"""
LightGBM Training Pipeline for Occupancy Prediction

Trains a LightGBM regression model to predict avg_occupancy (or max_occupancy)
using v2 feature groups from Hopsworks.

Features used:
    - Traffic: hour, route_id, direction_id
    - Weather: temperature_2m, precipitation, windspeed_10m, cloudcover, prev_* columns
    - Calendar: month, day, weekday, is_weekend, is_holiday_se, is_workday_se
"""
import os
import warnings
from pathlib import Path

import hopsworks
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from tqdm import tqdm

load_dotenv()

# ============================================================================
# CONFIGURATION - Change these to modify behavior
# ============================================================================

TARGET_COLUMN = "avg_occupancy"

# Hopsworks configuration
HOPSWORKS_PROJECT = "occupancy"
FG_VERSION = 2
FV_VERSION = 1
TRAIN_TEST_SPLIT_VERSION = 1

# Model configuration
MODEL_NAME = f"occupancy_lgbm_clipped_{TARGET_COLUMN}"
MODEL_DIR = Path(__file__).parent / "model_artifact"
PREDICTION_MIN = 0.0  # Minimum allowed prediction value
PREDICTION_MAX = 5.0  # Maximum allowed prediction value

# Data split configuration
TRAIN_START_DATE = "2023-10-01"
TRAIN_END_DATE = "2025-11-15"
TEST_START_DATE = "2025-11-16"
TEST_END_DATE = "2026-01-08"
CREATE_NEW_SPLIT = False  # Set to True to create a new train/test split, False to reuse existing

# Hyperparameter tuning configuration
TUNE_SAMPLE_FRACTION = 0.50  # Use 50% of data for tuning
TUNE_N_ITER = 50  # Number of random search iterations
RANDOM_STATE = 42


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Features to select from each feature group
# Note: event_time is included in the query for temporal splits but dropped before training
TRAFFIC_FEATURES = ["hour", "route_id", "direction_id"]
WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "cloudcover",
    "prev_temperature_2m",
    "prev_precipitation",
    "prev_windspeed_10m",
    "prev_cloudcover",
]
CALENDAR_FEATURES = [
    "month",
    "weekday",
    "is_weekend",
    "is_holiday_se",
    "is_workday_se",
]

# Feature types for preprocessing
CATEGORICAL_FEATURES = ["route_id", "direction_id", "weekday"]
NUMERIC_FEATURES = [
    "hour",
    "month",
    "temperature_2m",
    "precipitation",
    "windspeed_10m",
    "cloudcover",
    "prev_temperature_2m",
    "prev_precipitation",
    "prev_windspeed_10m",
    "prev_cloudcover",
]
BOOLEAN_FEATURES = ["is_weekend", "is_holiday_se", "is_workday_se"]


def login():
    """Login to Hopsworks and return project."""
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    if api_key is None:
        raise RuntimeError("HOPSWORKS_API_KEY environment variable is not set")

    return hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT,
        api_key_value=api_key,
    )


def create_feature_view(fs):
    """
    Create or get the feature view joining traffic, weather, and calendar data.

    Returns the feature view for training.
    """
    fg_traffic = fs.get_feature_group("skane_traffic", version=FG_VERSION)
    fg_weather = fs.get_feature_group("skane_weather", version=FG_VERSION)
    fg_calendar = fs.get_feature_group("sweden_calendar", version=FG_VERSION)

    # Build the query with joins
    query = (
        fg_traffic.select(TRAFFIC_FEATURES + [TARGET_COLUMN, "event_time"])
        .join(
            fg_calendar.select(CALENDAR_FEATURES),
            on=["date"],
        )
        .join(
            fg_weather.select(WEATHER_FEATURES),
            on=["date", "hour"],
        )
    )

    fv = fs.get_or_create_feature_view(
        name=f"occupancy_fv_{TARGET_COLUMN}",
        query=query,
        labels=[TARGET_COLUMN],
        description=f"Occupancy FV: traffic + calendar + weather for {TARGET_COLUMN} prediction",
        version=FV_VERSION,
    )

    return fv


def _bool_to_int(x):
    """Convert boolean columns to int for preprocessing (picklable)."""
    return x.astype(int)


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline for features.

    LightGBM handles categoricals natively, but we encode them
    for consistency in the sklearn pipeline.
    """
    # Numeric preprocessing: impute missing with median
    numeric_transformer = SimpleImputer(strategy="median")

    # Categorical preprocessing: impute + ordinal encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Boolean: convert to int first, then impute
    # SimpleImputer doesn't support bool dtype, so we convert to int
    boolean_transformer = Pipeline(
        steps=[
            ("to_int", FunctionTransformer(_bool_to_int, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bool", boolean_transformer, BOOLEAN_FEATURES),
        ],
        remainder="drop",  # Drop columns not specified
    )

    return preprocessor


def get_hyperparameter_grid() -> dict:
    """
    Define the hyperparameter search space for LightGBM.

    Expanded grid for more thorough search.
    """
    return {
        "model__n_estimators": [300, 500, 1000],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__max_depth": [5, 10, 20, -1],
        "model__num_leaves": [31, 63, 127, 255],
        "model__min_child_samples": [10, 20, 50, 100],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_alpha": [0, 0.01, 0.1],
        "model__reg_lambda": [0, 0.01, 0.1],
    }


class ClippedLGBMRegressor(lgb.LGBMRegressor):
    """
    LGBMRegressor wrapper that clips predictions to a valid range.
    
    Ensures all predictions fall within [PREDICTION_MIN, PREDICTION_MAX].
    """
    
    def predict(self, X, **kwargs):
        """Predict with clipping to valid range."""
        raw_predictions = super().predict(X, **kwargs)
        return np.clip(raw_predictions, PREDICTION_MIN, PREDICTION_MAX)


def tune_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """
    Perform hyperparameter tuning using a temporal split on a sample of the data.

    Uses a temporal train/validation split instead of random sampling to better
    reflect real-world prediction scenarios.

    Returns the best parameters found.
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING (using {TUNE_SAMPLE_FRACTION*100:.0f}% of data)")
    print(f"{'='*60}")

    # Take the first TUNE_SAMPLE_FRACTION of data (temporal split)
    sample_size = int(len(X_train) * TUNE_SAMPLE_FRACTION)
    X_sample = X_train.iloc[:sample_size]
    y_sample = y_train.iloc[:sample_size]

    # Split sample temporally: 80% train, 20% validation
    tune_split_idx = int(len(X_sample) * 0.8)
    X_tune_train = X_sample.iloc[:tune_split_idx]
    y_tune_train = y_sample.iloc[:tune_split_idx]
    X_tune_val = X_sample.iloc[tune_split_idx:]
    y_tune_val = y_sample.iloc[tune_split_idx:]

    print(f"Tuning train size: {len(X_tune_train):,} rows")
    print(f"Tuning val size:   {len(X_tune_val):,} rows")

    # Get parameter grid and generate random combinations
    param_grid = get_hyperparameter_grid()
    rng = np.random.RandomState(RANDOM_STATE)

    # Generate random parameter combinations
    param_combinations = []
    for _ in range(TUNE_N_ITER):
        params = {name: rng.choice(values) for name, values in param_grid.items()}
        param_combinations.append(params)

    print(f"Testing {TUNE_N_ITER} parameter combinations...\n")

    best_score = float("inf")
    best_params = None

    # Suppress the feature names warning during tuning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")

        for i, params in enumerate(tqdm(param_combinations, desc="Tuning progress", unit="config")):
            # Extract model parameters (remove 'model__' prefix)
            model_params = {k.replace("model__", ""): v for k, v in params.items()}

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "model",
                        ClippedLGBMRegressor(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbose=-1,
                            **model_params,
                        ),
                    ),
                ]
            )

            # Fit and evaluate
            pipeline.fit(X_tune_train, y_tune_train)
            predictions = pipeline.predict(X_tune_val)
            rmse = np.sqrt(mean_squared_error(y_tune_val, predictions))

            if rmse < best_score:
                best_score = rmse
                best_params = params

    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best RMSE: {best_score:.4f}")
    print(f"Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params


def train_final_model(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: dict
) -> Pipeline:
    """
    Train the final model on full training data with best parameters.
    """
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}")
    print(f"Training on {len(X_train):,} rows")

    # Extract model parameters (remove 'model__' prefix)
    model_params = {k.replace("model__", ""): v for k, v in best_params.items()}

    n_estimators = model_params.get("n_estimators", 100)
    print(f"Training {n_estimators} boosting rounds...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                ClippedLGBMRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                    **model_params,
                ),
            ),
        ]
    )

    # Suppress the feature names warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")
        pipeline.fit(X_train, y_train)

    print("Training complete!")

    return pipeline


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Evaluate the trained model on test data.

    Returns dictionary of metrics.
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Evaluating on {len(X_test):,} test rows")

    # Suppress the feature names warning during prediction
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")
        predictions = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    # Show prediction distribution
    print(f"\nPrediction statistics:")
    print(f"  Actual range:    [{y_test.min():.2f}, {y_test.max():.2f}]")
    print(f"  Predicted range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"  Actual mean:     {y_test.mean():.2f}")
    print(f"  Predicted mean:  {predictions.mean():.2f}")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def save_and_upload_model(
    model: Pipeline,
    metrics: dict,
    fv,
    mr,
    X_train: pd.DataFrame,
) -> None:
    """
    Save model locally and upload to Hopsworks Model Registry.
    """
    print("\n--- Saving Model ---")

    # Save locally
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # Upload to Hopsworks
    print("Uploading to Hopsworks Model Registry...")
    hs_model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        feature_view=fv,
        input_example=X_train.sample(1, random_state=RANDOM_STATE),
        description=f"LightGBM model for {TARGET_COLUMN} prediction",
    )
    hs_model.save(str(MODEL_DIR))
    print(f"Model uploaded: {MODEL_NAME}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print(f"TRAINING PIPELINE: Predicting {TARGET_COLUMN}")
    print("=" * 70)

    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    project = login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Create/get feature view
    print("\nSetting up feature view...")
    fv = create_feature_view(fs)

    # Materialize the split to disk/S3 (only if CREATE_NEW_SPLIT is True)
    if CREATE_NEW_SPLIT:
        print("\nCreating new train/test split...")
        fv.create_train_test_split(
            train_start=TRAIN_START_DATE,
            train_end=TRAIN_END_DATE,
            test_start=TEST_START_DATE,
            test_end=TEST_END_DATE,
            description=f"Temporal split: {TRAIN_START_DATE} - {TRAIN_END_DATE} train, {TEST_START_DATE} - {TEST_END_DATE} test",
            data_format="parquet",
            write_options={"wait_for_job": True},
        )
    else:
        print("\nUsing existing train/test split...")

    # Read the materialized files
    X_train, X_test, y_train, y_test = fv.get_train_test_split(
        training_dataset_version=TRAIN_TEST_SPLIT_VERSION
    )

    # Flatten labels if DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print(f"Train size: {len(X_train):,}")
    print(f"Test size:  {len(X_test):,}")
    print(f"Train columns: {list(X_train.columns)}")

    # Hyperparameter tuning
    best_params = tune_hyperparameters(X_train, y_train)

    # Train final model
    model = train_final_model(X_train, y_train, best_params)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save and upload
    save_and_upload_model(model, metrics, fv, mr, X_train)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
