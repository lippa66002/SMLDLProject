import pandas as pd
from pathlib import Path

# ---- EDIT THESE ----
CSV_PATH = Path("out/merged_4years_route_9011012065200000_hourly.csv")
DATE_COL = "date"   # column like "2025-09-03"
# -------------------

def main():
    df = pd.read_csv(CSV_PATH)

    if DATE_COL not in df.columns:
        raise ValueError(f"Column '{DATE_COL}' not found in {CSV_PATH}")

    # Parse date
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # Month label like "2025-09"
    df["month"] = df[DATE_COL].dt.to_period("M").astype(str)

    # Weekday name (Monday, Tuesday, ...)
    df["weekday"] = df[DATE_COL].dt.day_name()

    # 1) Rows per month (hourly rows)
    rows_per_month = df["month"].value_counts().sort_index()

    # 2) Rows per weekday (hourly rows)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rows_per_weekday = (
        df["weekday"].value_counts()
        .reindex(weekday_order)
        .fillna(0)
        .astype(int)
    )

    # 3) Unique dates per weekday (count of distinct days)
    unique_dates = df[[DATE_COL, "weekday"]].drop_duplicates()
    unique_dates_per_weekday = (
        unique_dates["weekday"].value_counts()
        .reindex(weekday_order)
        .fillna(0)
        .astype(int)
    )

    print("\nRows per month (hourly rows):")
    print(rows_per_month.to_string())

    print("\nRows per weekday (hourly rows):")
    print(rows_per_weekday.to_string())

    print("\nUnique dates per weekday (number of distinct days):")
    print(unique_dates_per_weekday.to_string())

    print("\nTotal rows:", len(df))
    print("Unique dates:", df[DATE_COL].dt.date.nunique())

if __name__ == "__main__":
    main()
