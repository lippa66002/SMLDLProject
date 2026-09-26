from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

import joblib

# ---------------- SETTINGS ----------------
SPLIT_DIR = Path("out/splits")
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV   = SPLIT_DIR / "val.csv"
TEST_CSV  = SPLIT_DIR / "test.csv"

LABEL = "label_grouped"
LABEL_ORDER = ["EMPTY", "MANY_SEATS_AVAILABLE", "CROWDED"]

RANDOM_STATE = 42

# Where to save artifacts
MODEL_DIR = Path("out/model_artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model_pipeline.joblib"
CONFIG_PATH = MODEL_DIR / "best_model_config.json"
# -----------------------------------------


def load_splits():
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
    return pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Dynamically builds a pipeline that handles numeric and categorical features.
    """
    # IMPORTANT: include pandas nullable ints (Int64) by using np.number
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])


def run_experiment(name: str, features: list[str], trainval_df: pd.DataFrame, test_df: pd.DataFrame):
    # Clean labels
    trainval = trainval_df.dropna(subset=[LABEL]).copy()
    test = test_df.dropna(subset=[LABEL]).copy()

    # Only keep features that exist
    valid_features = [f for f in features if f in trainval.columns]
    if not valid_features:
        return {
            "run": name,
            "features_used": 0,
            "valid_features": [],
            "balanced_acc": None,
            "macro_f1": None,
            "weighted_f1": None,
            "note": "SKIPPED (no valid features)",
        }

    X_train, y_train = trainval[valid_features], trainval[LABEL]
    X_test, y_test = test[valid_features], test[LABEL]

    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    bal = balanced_accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    wf1 = f1_score(y_test, pred, average="weighted")

    print("\n" + "=" * 60)
    print(f"RUN: {name}")
    print("Features:", valid_features)
    print("=" * 60)

    existing_labels = [l for l in LABEL_ORDER if l in np.unique(y_test)]
    print(classification_report(y_test, pred, labels=existing_labels, digits=3))

    cm = confusion_matrix(y_test, pred, labels=existing_labels)
    cm_df = pd.DataFrame(cm, index=existing_labels, columns=existing_labels)
    print("\nConfusion Matrix (Rows=Actual, Cols=Pred):")
    print(cm_df)

    return {
        "run": name,
        "features_used": len(valid_features),
        "valid_features": valid_features,
        "balanced_acc": bal,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "note": "Success",
    }


def pick_winner(summary_df: pd.DataFrame) -> pd.Series:
    """
    Picks the best run by:
      1) highest macro_f1
      2) tiebreaker: highest balanced_acc
      3) final tiebreaker: most features (arbitrary, but deterministic)
    """
    df = summary_df.copy()

    # Keep only successful runs
    df = df[df["note"] == "Success"].copy()
    if df.empty:
        raise RuntimeError("No successful runs to choose a winner from.")

    df = df.sort_values(
        by=["balanced_acc", "macro_f1", "features_used"],
        ascending=[False, False, False],
        na_position="last",
    )
    return df.iloc[0]


def fit_and_save_best_model(best_row: pd.Series, trainval_df: pd.DataFrame):
    """
    Refits the winning pipeline on ALL train+val, then saves:
      - model pipeline (.joblib)
      - config json (features + metadata)
    """
    best_features = best_row["valid_features"]
    if not isinstance(best_features, list):
        # When loaded from DataFrame printing etc., it should still be list,
        # but this keeps things from exploding in weird cases.
        best_features = list(best_features)

    trainval = trainval_df.dropna(subset=[LABEL]).copy()
    X = trainval[best_features]
    y = trainval[LABEL]

    pipe = build_pipeline(X)
    pipe.fit(X, y)

    # Save model pipeline (preprocessing + classifier)
    joblib.dump(pipe, MODEL_PATH)

    # Save config + metadata
    config = {
        "label_column": LABEL,
        "label_order": LABEL_ORDER,
        "winner_run_name": best_row["run"],
        "winning_features": best_features,
        "metrics_on_test": {
            "balanced_acc": float(best_row["balanced_acc"]),
            "macro_f1": float(best_row["macro_f1"]),
            "weighted_f1": float(best_row["weighted_f1"]),
        },
        "model_artifact_path": str(MODEL_PATH),
        "random_state": RANDOM_STATE,
        "sklearn_pipeline": "preprocessor + RandomForestClassifier",
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "#" * 80)
    print("SAVED WINNING MODEL + CONFIG")
    print("#" * 80)
    print("Winner:", best_row["run"])
    print("Model saved to:", MODEL_PATH)
    print("Config saved to:", CONFIG_PATH)


def main():
    try:
        train_df, val_df, test_df = load_splits()
    except FileNotFoundError as e:
        print(e)
        return

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    # Feature sets aligned to your final dataset schema
    feature_sets = [
        ("A) Hour only", ["hour"]),
        ("B) Hour + Time Context", ["hour", "is_holiday_se", "is_workday_se", "is_weekend"]),
        ("C) Full Calendar", ["hour", "is_holiday_se", "is_workday_se", "is_weekend", "time_of_day_code", "weekday", "month"]),
        ("D) Calendar + Weather", [
            "hour", "is_holiday_se", "is_workday_se", "is_weekend",
            "time_of_day_code", "weekday", "month",
            "temperature_2m", "precipitation", "windspeed_10m", "cloudcover"
        ]),
    ]

    summaries = []
    for name, feats in feature_sets:
        summaries.append(run_experiment(name, feats, trainval_df, test_df))

    print("\n" + "#" * 80)
    print("FINAL SUMMARY COMPARISON")
    print("#" * 80)

    summary_df = pd.DataFrame(summaries)

    # Pretty print (without destroying numeric columns)
    pretty = summary_df.copy()
    for c in ["balanced_acc", "macro_f1", "weighted_f1"]:
        if c in pretty.columns:
            pretty[c] = pretty[c].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    print(pretty[["run", "features_used", "balanced_acc", "macro_f1", "weighted_f1", "note"]].to_string(index=False))

    # Pick winner + refit + save artifacts
    best_row = pick_winner(summary_df)
    fit_and_save_best_model(best_row, trainval_df)


if __name__ == "__main__":
    main()
