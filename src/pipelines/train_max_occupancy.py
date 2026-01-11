"""
LightGBM Training Pipeline for Max Occupancy Classification

Trains a LightGBM classifier to predict max_occupancy as discrete classes (0-5).

Design considerations:
    - Ordinal-aware: Penalties scale with distance from true class
    - Asymmetric: Higher penalty for underestimating crowded situations
    - Minority-focused: Higher recall for crowded classes (2-5)

Classes:
    0 = EMPTY
    1 = MANY_SEATS_AVAILABLE  
    2 = FEW_SEATS_AVAILABLE
    3 = STANDING_ROOM_ONLY
    4 = CRUSHED_STANDING_ROOM_ONLY
    5 = FULL

Features used:
    - Traffic: hour, route_id, direction_id
    - Weather: temperature_2m, precipitation, windspeed_10m, cloudcover, prev_* columns
    - Calendar: month, weekday, is_weekend, is_holiday_se, is_workday_se
"""
import os
import tempfile
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from tqdm import tqdm

from src.utils.transformations.type_utils import bool_to_int

load_dotenv()

# ============================================================================
# CONFIGURATION - Change these to modify behavior
# ============================================================================

TARGET_COLUMN = "max_occupancy"

# Hopsworks configuration
HOPSWORKS_PROJECT = "occupancy"
FG_VERSION = 2
FV_VERSION = 1
TRAIN_TEST_SPLIT_VERSION = 1

# Model configuration
MODEL_NAME = f"occupancy_lgbm_classifier_{TARGET_COLUMN}"

# Class labels (ordinal)
CLASS_NAMES = [
    "EMPTY",
    "MANY_SEATS",
    "FEW_SEATS", 
    "STANDING",
    "CRUSHED",
    "FULL",
]
NUM_CLASSES = 6

# Data split configuration
TRAIN_START_DATE = "2022-10-01"
TRAIN_END_DATE = "2025-11-30"
TEST_START_DATE = "2025-12-01"
TEST_END_DATE = "2026-01-08"
CREATE_NEW_SPLIT = False  # Set to False to use existing split

# Hyperparameter tuning configuration
TUNE_SAMPLE_FRACTION = 0.30  # Use 30% of data for tuning
TUNE_N_ITER = 30  # Number of random search iterations
RANDOM_STATE = 42


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

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
    """
    fg_traffic = fs.get_feature_group("skane_traffic", version=FG_VERSION)
    fg_weather = fs.get_feature_group("skane_weather", version=FG_VERSION)
    fg_calendar = fs.get_feature_group("sweden_calendar", version=FG_VERSION)

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
        description=f"Occupancy FV for {TARGET_COLUMN} classification",
        version=FV_VERSION,
    )

    return fv


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing pipeline for features."""
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    boolean_transformer = Pipeline(
        steps=[
            ("to_int", FunctionTransformer(bool_to_int, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("bool", boolean_transformer, BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute asymmetric sample weights for ordinal classification.
    
    Weight strategy:
    - Base weight increases with class (crowded classes are more important)
    - This encourages higher recall for minority crowded classes
    - Underestimating crowded situations is penalized more during training
    
    Weight formula: weight = 1 + class_value * 0.5
    Class 0 -> 1.0, Class 5 -> 3.5
    """
    weights = 1.0 + y * 0.5
    return weights


def get_class_weights() -> dict:
    """
    Compute class weights to boost minority classes.
    
    Classes 2-5 are minority and safety-critical, so they get higher weight.
    """
    return {
        0: 1.0,   # EMPTY - majority class
        1: 1.5,   # MANY_SEATS - common
        2: 3.0,   # FEW_SEATS - minority
        3: 5.0,   # STANDING - rare
        4: 8.0,   # CRUSHED - very rare
        5: 10.0,  # FULL - extremely rare, safety-critical
    }


def get_hyperparameter_grid() -> dict:
    """Define the hyperparameter search space for LightGBM classifier."""
    return {
        "model__n_estimators": [300, 500, 1000],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [5, 10, 15, -1],
        "model__num_leaves": [31, 63, 127],
        "model__min_child_samples": [20, 50, 100],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__reg_alpha": [0, 0.01, 0.1],
        "model__reg_lambda": [0, 0.01, 0.1],
    }


def ordinal_weighted_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute weighted recall with emphasis on crowded classes.
    
    Higher recall for classes 2-5 is more important than classes 0-1.
    """
    class_weights = get_class_weights()
    weights = np.array([class_weights[i] for i in range(NUM_CLASSES)])
    
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES), zero_division=0)
    
    # Pad if some classes are missing
    if len(recall_per_class) < NUM_CLASSES:
        padded = np.zeros(NUM_CLASSES)
        padded[:len(recall_per_class)] = recall_per_class
        recall_per_class = padded
    
    weighted_recall = np.sum(recall_per_class * weights) / np.sum(weights)
    return weighted_recall


def ordinal_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute asymmetric ordinal cost.
    
    Underestimation (predicting lower than actual) is penalized more heavily,
    especially for crowded classes.
    
    Cost = sum of |y_true - y_pred| * asymmetric_factor
    Where asymmetric_factor = 2.0 if underestimating, 1.0 if overestimating
    """
    diff = y_true - y_pred
    
    # Underestimation penalty: 2x for underestimating crowded situations
    underestimate_mask = diff > 0  # true > pred means underestimation
    
    cost = np.abs(diff).astype(float)
    cost[underestimate_mask] *= 2.0  # Double penalty for underestimation
    
    # Additional penalty based on actual crowdedness
    cost *= (1 + y_true * 0.2)  # More penalty when actual is crowded
    
    return np.mean(cost)


def tune_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """
    Perform hyperparameter tuning using ordinal-aware metrics.
    """
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING (using {TUNE_SAMPLE_FRACTION*100:.0f}% of data)")
    print(f"{'='*60}")

    sample_size = int(len(X_train) * TUNE_SAMPLE_FRACTION)
    X_sample = X_train.iloc[:sample_size]
    y_sample = y_train.iloc[:sample_size].astype(int)

    tune_split_idx = int(len(X_sample) * 0.8)
    X_tune_train = X_sample.iloc[:tune_split_idx]
    y_tune_train = y_sample.iloc[:tune_split_idx]
    X_tune_val = X_sample.iloc[tune_split_idx:]
    y_tune_val = y_sample.iloc[tune_split_idx:]

    print(f"Tuning train size: {len(X_tune_train):,} rows")
    print(f"Tuning val size:   {len(X_tune_val):,} rows")
    
    # Show class distribution
    print(f"\nClass distribution in tuning data:")
    for i in range(NUM_CLASSES):
        count = (y_tune_train == i).sum()
        pct = count / len(y_tune_train) * 100
        print(f"  {i} ({CLASS_NAMES[i]}): {count:,} ({pct:.1f}%)")

    # Compute sample weights for training
    sample_weights = compute_sample_weights(y_tune_train.values)

    param_grid = get_hyperparameter_grid()
    rng = np.random.RandomState(RANDOM_STATE)

    param_combinations = []
    for _ in range(TUNE_N_ITER):
        params = {name: rng.choice(values) for name, values in param_grid.items()}
        param_combinations.append(params)

    print(f"\nTesting {TUNE_N_ITER} parameter combinations...")
    print("Optimizing for: weighted recall (crowded class focus)\n")

    best_score = -float("inf")
    best_params = None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")

        for params in tqdm(param_combinations, desc="Tuning progress", unit="config"):
            model_params = {k.replace("model__", ""): v for k, v in params.items()}

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    (
                        "model",
                        lgb.LGBMClassifier(
                            objective="multiclass",
                            num_class=NUM_CLASSES,
                            class_weight=get_class_weights(),
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                            verbose=-1,
                            **model_params,
                        ),
                    ),
                ]
            )

            # Fit with sample weights
            pipeline.fit(X_tune_train, y_tune_train, model__sample_weight=sample_weights)
            predictions = pipeline.predict(X_tune_val)
            
            # Use weighted recall as the optimization target
            score = ordinal_weighted_recall(y_tune_val.values, predictions)

            if score > best_score:
                best_score = score
                best_params = params

    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Weighted Recall: {best_score:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params


def train_final_model(
    X_train: pd.DataFrame, y_train: pd.Series, best_params: dict
) -> Pipeline:
    """Train the final model on full training data with best parameters."""
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL")
    print(f"{'='*60}")
    print(f"Training on {len(X_train):,} rows")

    model_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    y_train_int = y_train.astype(int)
    sample_weights = compute_sample_weights(y_train_int.values)

    n_estimators = model_params.get("n_estimators", 100)
    print(f"Training {n_estimators} boosting rounds...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="multiclass",
                    num_class=NUM_CLASSES,
                    class_weight=get_class_weights(),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                    **model_params,
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")
        pipeline.fit(X_train, y_train_int, model__sample_weight=sample_weights)

    print("Training complete!")

    return pipeline


def evaluate_model(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Evaluate the trained classifier with ordinal-aware metrics.
    """
    print(f"\n{'='*60}")
    print("MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Evaluating on {len(X_test):,} test rows")

    y_test_int = y_test.astype(int)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*feature names.*")
        predictions = model.predict(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test_int, predictions)
    weighted_recall = ordinal_weighted_recall(y_test_int.values, predictions)
    ordinal_cost_val = ordinal_cost(y_test_int.values, predictions)

    print(f"\nResults:")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Weighted Recall: {weighted_recall:.4f}")
    print(f"  Ordinal Cost:    {ordinal_cost_val:.4f}")

    # Per-class recall (important for minority classes)
    print(f"\nPer-class Recall:")
    recall_per_class = recall_score(y_test_int, predictions, average=None, labels=range(NUM_CLASSES), zero_division=0)
    for i, rec in enumerate(recall_per_class):
        count = (y_test_int == i).sum()
        print(f"  {i} ({CLASS_NAMES[i]:12}): {rec:.3f}  (n={count:,})")

    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test_int, predictions, labels=range(NUM_CLASSES))
    print("     Predicted ->")
    print("     " + "".join(f"{i:6}" for i in range(NUM_CLASSES)))
    for i, row in enumerate(cm):
        print(f"  {i}: " + "".join(f"{v:6}" for v in row))

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(
        y_test_int, predictions, 
        labels=range(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "weighted_recall": weighted_recall,
        "ordinal_cost": ordinal_cost_val,
    }


def save_and_upload_model(
    model: Pipeline,
    metrics: dict,
    fv,
    mr,
    X_train: pd.DataFrame,
) -> None:
    """Upload model to Hopsworks Model Registry."""
    print("\n--- Uploading Model to Hopsworks ---")

    # Use temp directory for upload (no local persistence)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)

        hs_model = mr.sklearn.create_model(
            name=MODEL_NAME,
            metrics=metrics,
            feature_view=fv,
            input_example=X_train.sample(1, random_state=RANDOM_STATE),
            description=f"LightGBM classifier for {TARGET_COLUMN} prediction (ordinal, asymmetric)",
        )
        hs_model.save(str(tmp_path))
        print(f"Model uploaded: {MODEL_NAME}")


def main():
    """Main training pipeline."""
    print("=" * 70)
    print(f"TRAINING PIPELINE: Classifying {TARGET_COLUMN}")
    print("=" * 70)

    print("\nConnecting to Hopsworks...")
    project = login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("\nSetting up feature view...")
    fv = create_feature_view(fs)

    if CREATE_NEW_SPLIT:
        print("\nCreating new train/test split...")
        fv.create_train_test_split(
            train_start=TRAIN_START_DATE,
            train_end=TRAIN_END_DATE,
            test_start=TEST_START_DATE,
            test_end=TEST_END_DATE,
            description=f"Temporal split for {TARGET_COLUMN}",
            data_format="parquet",
            write_options={"wait_for_job": True},
        )
    else:
        print("\nUsing existing train/test split...")

    X_train, X_test, y_train, y_test = fv.get_train_test_split(
        training_dataset_version=TRAIN_TEST_SPLIT_VERSION
    )

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print(f"Train size: {len(X_train):,}")
    print(f"Test size:  {len(X_test):,}")
    
    # Show overall class distribution
    print(f"\nClass distribution in training data:")
    y_train_int = y_train.astype(int)
    for i in range(NUM_CLASSES):
        count = (y_train_int == i).sum()
        pct = count / len(y_train_int) * 100
        print(f"  {i} ({CLASS_NAMES[i]}): {count:,} ({pct:.1f}%)")

    best_params = tune_hyperparameters(X_train, y_train)
    model = train_final_model(X_train, y_train, best_params)
    metrics = evaluate_model(model, X_test, y_test)
    save_and_upload_model(model, metrics, fv, mr, X_train)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
