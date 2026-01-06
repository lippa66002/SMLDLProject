import os
from sys import api_version

import hopsworks
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
)

# ---------------- CONFIG ----------------
PROJECT_NAME = "occupancy"

FG_OCCUPANCY = "skane_traffic"
FG_CALENDAR  = "sweden_calendar"
FG_WEATHER   = "skane_weather"

FEATURE_VIEW_NAME = "occupancy_fv"
LABEL = "label_grouped"
LABEL_ORDER = ["EMPTY", "MANY_SEATS_AVAILABLE", "CROWDED"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

MODEL_NAME = "occupancy_rf"
MODEL_DIR = "../model_artifact"
# ---------------------------------------
api = os.environ.get("HOPSWORKS_API_KEY")
if api is None:
    raise RuntimeError("HOPSWORKS_API_KEY is not set")


def login():
    # Same pattern as the notebook
    return hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",                            # DNS of your Hopsworks instance
    project="occupancy",                      # Name of your Hopsworks project
    api_key_value= api    # Hopsworks API key value
)


def create_feature_view(fs):
    fg_occ = fs.get_feature_group("skane_traffic", version=1)
    fg_cal = fs.get_feature_group("sweden_calendar", version=1)
    fg_wx  = fs.get_feature_group("skane_weather", version=1)

    query = (
        fg_occ.select(['date', 'hour', 'route_id', 'label_grouped', 'event_time'])
        .join(fg_cal.select(["month", "day", "weekday", "is_weekend", "is_holiday_se", "is_workday_se"]), on=["date"])
        .join(fg_wx.select(["temperature_2m", "precipitation", "windspeed_10m", "cloudcover"]), on=["date", "hour"])
    )

    fv = fs.get_or_create_feature_view(
        name="occupancy_fv",
        query=query,
        labels=["label_grouped"],
        description="Occupancy FV: traffic + calendar + weather",
        version=1
    )

    return fv


def build_pipeline(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), cat_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]
    )


def evaluate(name, features, X_train, y_train, X_test, y_test):
    feats = [f for f in features if f in X_train.columns]
    if not feats:
        return None
    print(f"[DEBUG] Features actually used ({name}):", feats)

    pipe = build_pipeline(X_train[feats])
    pipe.fit(X_train[feats], y_train)
    preds = pipe.predict(X_test[feats])

    bal = balanced_accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro")
    wf1 = f1_score(y_test, preds, average="weighted")

    print("\n" + "=" * 60)
    print(f"RUN: {name}")
    print("=" * 60)

    labels_present = [l for l in LABEL_ORDER if l in np.unique(y_test)]
    print(classification_report(y_test, preds, labels=labels_present, digits=3))

    cm = confusion_matrix(y_test, preds, labels=labels_present)
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=labels_present, columns=labels_present))

    return {
        "name": name,
        "balanced_acc": bal,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "model": pipe,
        "features": feats,
    }


def main():
    project = login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fv = create_feature_view(fs)


    X_train, X_test, y_train, y_test = fv.train_test_split(
        description='Temporal split: Test data starts 2025-08-15',
        test_start="2025-08-15"
    )

    print("COLUMNS IN FEATURE VIEW:")
    for c in X_train.columns:
        print(c)

    # Flatten labels if needed
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    feature_sets = [
        ("Hour only", ["hour"]),
        ("Calendar", [
            "hour",
            "month",
            "weekday",
            "is_weekend",
            "is_holiday_se",
            "is_workday_se",
        ]),

        ("Calendar + Weather", [
            "hour",
            "month",
            "weekday",
            "is_weekend",
            "is_holiday_se",
            "is_workday_se",
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
            "cloudcover",
        ]),
    ]

    best = None
    for name, feats in feature_sets:
        res = evaluate(name, feats, X_train, y_train, X_test, y_test)
        if res and (best is None or res["macro_f1"] > best["macro_f1"]):
            best = res

    if best is None:
        print("No valid model trained.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], f"{MODEL_DIR}/model.pkl")

    model = mr.sklearn.create_model(
        name=MODEL_NAME,
        metrics={
            "balanced_acc": best["balanced_acc"],
            "macro_f1": best["macro_f1"],
            "weighted_f1": best["weighted_f1"],
        },
        feature_view=fv,
        input_example=X_train.sample(1),
        description=f"Best run: {best['name']} | Features: {best['features']}",
    )
    model.save(MODEL_DIR)
    print(f"[OK] Model saved: {MODEL_NAME}")


if __name__ == "__main__" :
    main()
