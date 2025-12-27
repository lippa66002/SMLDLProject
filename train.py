from pathlib import Path
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

# ---------------- SETTINGS ----------------
SPLIT_DIR = Path("out/splits")
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV   = SPLIT_DIR / "val.csv"
TEST_CSV  = SPLIT_DIR / "test.csv"

LABEL = "label_grouped"
# Ensure the confusion matrix follows a logical progression of "crowdedness"
LABEL_ORDER = ["EMPTY", "MANY_SEATS_AVAILABLE", "CROWDED"]

RANDOM_STATE = 42
# -----------------------------------------

def load_splits():
    for p in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing split file: {p}")
    return pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV)

def build_pipeline(X):
    """
    Dynamically builds a pipeline that handles numeric and categorical features.
    """
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Preprocessing for numerical data
    numeric_transformer = SimpleImputer(strategy="median")

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced", # Vital for small/imbalanced data
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

def run_experiment(name, features, trainval_df, test_df):
    # Clean labels
    trainval = trainval_df.dropna(subset=[LABEL]).copy()
    test = test_df.dropna(subset=[LABEL]).copy()

    # Only keep features that exist in the dataframe
    valid_features = [f for f in features if f in trainval.columns]

    if not valid_features:
        return {"run": name, "features_used": 0, "macro_f1": None, "note": "SKIPPED"}

    X_train, y_train = trainval[valid_features], trainval[LABEL]
    X_test, y_test = test[valid_features], test[LABEL]

    # Build and fit
    pipe = build_pipeline(X_train)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    # Metrics
    bal = balanced_accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro")
    wf1 = f1_score(y_test, pred, average="weighted")

    print("\n" + "=" * 60)
    print(f"RUN: {name}")
    print("=" * 60)

    # Classification Report with fixed label order
    existing_labels = [l for l in LABEL_ORDER if l in np.unique(y_test)]
    print(classification_report(y_test, pred, labels=existing_labels, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred, labels=existing_labels)
    cm_df = pd.DataFrame(cm, index=existing_labels, columns=existing_labels)
    print("\nConfusion Matrix (Rows=Actual, Cols=Pred):")
    print(cm_df)

    return {
        "run": name,
        "features_used": len(valid_features),
        "balanced_acc": bal,
        "macro_f1": mf1,
        "weighted_f1": wf1,
        "note": "Success"
    }

def main():
    try:
        train_df, val_df, test_df = load_splits()
    except FileNotFoundError as e:
        print(e)
        return

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    # Define Feature Sets
    feature_sets = [
        ("A) Hour only", ["hour"]),
        ("B) Hour + Time Context", ["hour", "is_holiday_se", "is_workday_se", "is_weekend"]),
        ("C) Full Calendar", ["hour", "is_holiday_se", "is_workday_se", "is_weekend", "time_of_day_code", "weekday", "month"]),
        ("D) Calendar + Weather", ["hour", "is_holiday_se", "is_workday_se", "is_weekend", "time_of_day_code", "weekday", "month", 
                                   "temperature_2m", "precipitation", "windspeed_10m", "cloudcover", "relativehumidity_2m", "pressure_msl"])
    ]

    summaries = []
    for name, feats in feature_sets:
        summaries.append(run_experiment(name, feats, trainval_df, test_df))

    print("\n" + "#" * 80)
    print("FINAL SUMMARY COMPARISON")
    print("#" * 80)
    
    summary_df = pd.DataFrame(summaries)
    # Formatting for readability
    cols_to_fix = ["balanced_acc", "macro_f1", "weighted_f1"]
    summary_df[cols_to_fix] = summary_df[cols_to_fix].applymap(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
    
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()