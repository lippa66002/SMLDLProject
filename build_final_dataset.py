from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import requests


# ---------------- USER SETTINGS ----------------
INPUT_CSV = Path("out/merged_4years_route_9011012065200000_hourly.csv")

OUT_DIR = Path("out")
FINAL_CSV = OUT_DIR / "final_dataset_features.csv"
SPLIT_DIR = OUT_DIR / "splits"

FILTER_ROUTE_ID = "9011012065200000"  # set to None to keep all routes

DATE_COL = "date"
HOUR_COL = "hour"
ROUTE_COL = "route_id"
LABEL_COL = "label_mode"

# Weather location (approx): Malmö area (Skåne)
WEATHER_LAT = 55.605
WEATHER_LON = 13.003
WEATHER_TIMEZONE = "Europe/Stockholm"

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
# ------------------------------------------------


def group_label(label: str) -> str:
    """
    Collapse sparse GTFS occupancy labels into 3 classes so stratification works.
    Keep original label_mode unchanged; this is for training + splitting.
    """
    if pd.isna(label):
        return np.nan
    if label == "EMPTY":
        return "EMPTY"
    if label == "MANY_SEATS_AVAILABLE":
        return "MANY_SEATS_AVAILABLE"
    # Everything else becomes "CROWDED"
    return "CROWDED"


def hour_to_period(hour: int) -> str:
    if 5 <= hour <= 9:
        return "morning"
    elif 10 <= hour <= 14:
        return "afternoon"
    elif 15 <= hour <= 18:
        return "evening"
    elif 19 <= hour <= 22:
        return "night"
    else:
        return "late_night_early_morning"


def add_swedish_holidays(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    try:
        import holidays
    except ImportError as e:
        raise ImportError("Missing dependency 'holidays'. Install with: pip install holidays") from e

    se_holidays = holidays.country_holidays("SE")
    d = df[date_col].dt.date
    df["is_holiday_se"] = d.apply(lambda x: 1 if x in se_holidays else 0).astype("Int64")
    df["is_workday_se"] = ((df["is_weekend"] == 0) & (df["is_holiday_se"] == 0)).astype("Int64")
    return df


def fetch_weather_hourly(start_date: str, end_date: str, lat: float, lon: float, tz: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "windspeed_10m",
            "cloudcover",
            "relativehumidity_2m",
            "pressure_msl",
        ]),
        "timezone": tz,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        raise RuntimeError(f"Weather API returned unexpected payload keys: {list(data.keys())}")

    w = pd.DataFrame(data["hourly"])
    w["datetime"] = pd.to_datetime(w["time"], errors="coerce")
    w = w.dropna(subset=["datetime"]).drop(columns=["time"])

    w["date"] = w["datetime"].dt.date.astype(str)
    w["hour"] = w["datetime"].dt.hour.astype(int)

    w = w[[
        "date", "hour",
        "temperature_2m", "precipitation", "windspeed_10m",
        "cloudcover", "relativehumidity_2m", "pressure_msl"
    ]]
    return w


def get_weather_with_cache(df: pd.DataFrame) -> pd.DataFrame:
    dmin = df[DATE_COL].dt.date.min().isoformat()
    dmax = df[DATE_COL].dt.date.max().isoformat()
    cache_path = OUT_DIR / f"weather_cache_{dmin}_to_{dmax}_{WEATHER_LAT}_{WEATHER_LON}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    w = fetch_weather_hourly(dmin, dmax, WEATHER_LAT, WEATHER_LON, WEATHER_TIMEZONE)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(cache_path, index=False)
    return w


def stratified_train_val_test_split(df: pd.DataFrame, stratify_col: str):
    from sklearn.model_selection import train_test_split

    y = df[stratify_col]

    train_df, temp_df = train_test_split(
        df,
        test_size=(TEST_SIZE + VAL_SIZE),
        random_state=RANDOM_STATE,
        stratify=y
    )

    y_temp = temp_df[stratify_col]
    val_fraction_of_temp = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_fraction_of_temp),
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    return train_df, val_df, test_df


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    for col in [DATE_COL, HOUR_COL, ROUTE_COL, LABEL_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {INPUT_CSV}")

    df[ROUTE_COL] = df[ROUTE_COL].astype(str)
    if FILTER_ROUTE_ID is not None:
        df = df[df[ROUTE_COL] == str(FILTER_ROUTE_ID)].copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[HOUR_COL] = pd.to_numeric(df[HOUR_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, HOUR_COL, LABEL_COL]).copy()
    df[HOUR_COL] = df[HOUR_COL].astype(int)

    # Grouped label for splitting/training
    df["label_grouped"] = df[LABEL_COL].apply(group_label)
    df = df.dropna(subset=["label_grouped"])

    # ---- Time features ----
    df["year"] = df[DATE_COL].dt.year.astype("Int64")
    df["month"] = df[DATE_COL].dt.month.astype("Int64")
    df["weekday"] = df[DATE_COL].dt.weekday.astype("Int64")
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype("Int64")

    df["time_of_day"] = df[HOUR_COL].apply(hour_to_period)
    time_order = ["late_night_early_morning", "morning", "afternoon", "evening", "night"]
    df["time_of_day_code"] = pd.Categorical(df["time_of_day"], categories=time_order, ordered=True).codes

    df = add_swedish_holidays(df, DATE_COL)

    # Optional numeric “quality” features (still usable for training)
    keep_optional = []
    for c in ["avg_occupancy_score", "n_obs", "n_snapshots_used"]:
        if c in df.columns:
            keep_optional.append(c)
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Weather merge ----
    weather = get_weather_with_cache(df)
    df["date_str"] = df[DATE_COL].dt.date.astype(str)

    df = df.merge(
        weather,
        left_on=["date_str", HOUR_COL],
        right_on=["date", "hour"],
        how="left"
    )

    df = df.drop(columns=["date_str", "date", "hour_y"], errors="ignore")
    # after merge: original hour is hour_x
    if "hour_x" in df.columns:
        df = df.rename(columns={"hour_x": "hour"})

    weather_cols = ["temperature_2m", "precipitation", "windspeed_10m",
                    "cloudcover", "relativehumidity_2m", "pressure_msl"]
    for c in weather_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- Final training-usable columns ----
    cols = [
        DATE_COL, HOUR_COL, ROUTE_COL,
        "label_grouped",  # used for stratification/training
        LABEL_COL,        # original label kept for analysis/reporting
        "year", "month", "weekday", "is_weekend",
        "is_holiday_se", "is_workday_se",
        "time_of_day_code",
        *weather_cols,
        *keep_optional,
    ]
    cols = [c for c in cols if c in df.columns]
    df_final = df[cols].copy()

    df_final.to_csv(FINAL_CSV, index=False)
    print("Wrote final dataset:", FINAL_CSV)
    print("Rows:", len(df_final))
    print("\nGrouped label distribution:")
    print(df_final["label_grouped"].value_counts())

    # ---- Splits (stratify on grouped labels) ----
    train_df, val_df, test_df = stratified_train_val_test_split(df_final, stratify_col="label_grouped")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print("\nSplit distributions (should be similar):")
    print("Train:\n", train_df["label_grouped"].value_counts(normalize=True).round(4))
    print("Val:\n", val_df["label_grouped"].value_counts(normalize=True).round(4))
    print("Test:\n", test_df["label_grouped"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()
