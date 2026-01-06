from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import requests

# ---------------- USER SETTINGS ----------------
# Inputs
INPUT_CSV = Path("data/vehicle_data.csv")
OUT_DIR = Path("data/feature_groups")

# Config
DATE_COL = "date"
HOUR_COL = "hour"
ROUTE_COL = "route_id"

# Weather Config (Skåne)
WEATHER_LAT = 55.605
WEATHER_LON = 13.003
WEATHER_TIMEZONE = "Europe/Stockholm"

def prepare_date_range(df_traffic: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a continuous DataFrame of dates from the min to max date 
    found in traffic data to ensure no gaps in Weather/Calendar.
    """
    min_date = df_traffic[DATE_COL].min()
    max_date = df_traffic[DATE_COL].max()
    
    # Create a full range of dates
    full_range = pd.date_range(start=min_date, end=max_date, freq='D')
    df_dates = pd.DataFrame({DATE_COL: full_range})
    
    # Ensure date column is strictly standard datetime (normalized to midnight)
    df_dates[DATE_COL] = pd.to_datetime(df_dates[DATE_COL]).dt.normalize()
    
    print(f"Global Date Range: {min_date.date()} to {max_date.date()} ({len(df_dates)} days)")
    return df_dates

# ---------------- 1. TRAFFIC GROUP ----------------
def build_traffic_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts only traffic-specific observations. 
    Sparse data (only contains rows where buses actually ran).
    """
    print("Building Traffic Feature Group...")
    
    # Logic to group labels (from your original script)
    def group_label(label: str) -> str:
        if pd.isna(label): return np.nan
        if label in ["EMPTY", "MANY_SEATS_AVAILABLE"]: return label
        return "CROWDED"

    df = df.copy()
    
    # Generate the grouped label needed for training
    df["label_grouped"] = df["label_mode"].apply(group_label)
    
    # Select only traffic columns
    cols = [
        DATE_COL, HOUR_COL, ROUTE_COL,
        "label_grouped", "label_mode",
        "avg_occupancy_score", "n_obs", "n_snapshots_used",
        # timestamp is useful for point-in-time joins if available, 
        # otherwise Hopsworks uses (date, hour) as composite key
    ]
    
    # Filter columns that actually exist
    cols = [c for c in cols if c in df.columns]
    
    return df[cols]

# ---------------- 2. CALENDAR GROUP ----------------
def build_calendar_group(df_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Generates calendar features for EVERY day in the range.
    """
    print("Building Calendar Feature Group...")
    df = df_dates.copy()
    
    # Basic Calendar Features
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["day"] = df[DATE_COL].dt.day
    df["weekday"] = df[DATE_COL].dt.day_name()
    df["is_weekend"] = df[DATE_COL].dt.dayofweek >= 5
    
    # --- SWEDISH HOLIDAY LOGIC ---
    # (Simplified for portability. Ideally, use the 'workalendar' library or your custom list)
    # This is a placeholder for your is_holiday_se logic
    import holidays # pip install holidays
    se_holidays = holidays.SE()
    
    df["is_holiday_se"] = df[DATE_COL].apply(lambda x: x in se_holidays)
    
    # Logic for workday: Not weekend AND not holiday
    df["is_workday_se"] = (~df["is_weekend"]) & (~df["is_holiday_se"])
    
    return df

# ---------------- 3. WEATHER GROUP ----------------
def build_weather_group(df_dates: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches hourly weather for EVERY day in the range.
    Result will be approx len(df_dates) * 24 rows.
    """
    print("Building Weather Feature Group...")
    
    start_str = df_dates[DATE_COL].min().strftime("%Y-%m-%d")
    end_str = df_dates[DATE_COL].max().strftime("%Y-%m-%d")
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": ["temperature_2m", "precipitation", "windspeed_10m", "cloudcover"],
        "timezone": WEATHER_TIMEZONE
    }
    
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    
    # Construct DataFrame from hourly response
    hourly = data.get("hourly", {})
    df_weather = pd.DataFrame(hourly)
    
    # Rename 'time' to 'date' and 'hour' for joining
    # OpenMeteo returns ISO strings "2023-01-01T00:00"
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    df_weather[DATE_COL] = df_weather["time"].dt.normalize() # Date only
    df_weather[HOUR_COL] = df_weather["time"].dt.hour
    
    # Drop original full timestamp if not needed (or keep as event_time)
    df_weather = df_weather.drop(columns=["time"])
    
    return df_weather

# ---------------- MAIN EXECUTION ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Raw Traffic
    print(f"Loading {INPUT_CSV}...")
    df_raw = pd.read_csv(INPUT_CSV)
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])
    
    # 2. Create Continuous Date Backbone
    df_dates = prepare_date_range(df_raw)
    
    # 3. Build the 3 Groups
    traffic_df = build_traffic_group(df_raw)
    calendar_df = build_calendar_group(df_dates)
    weather_df = build_weather_group(df_dates)
    
    # 4. Save to CSV (Ready for Feature Backfill Notebooks)
    traffic_df.to_csv(OUT_DIR / "traffic_features.csv", index=False)
    calendar_df.to_csv(OUT_DIR / "calendar_features.csv", index=False)
    weather_df.to_csv(OUT_DIR / "weather_features.csv", index=False)
    
    print("\n--- Processing Complete ---")
    print(f"Traffic Shape:  {traffic_df.shape}")
    print(f"Calendar Shape: {calendar_df.shape} (One row per day)")
    print(f"Weather Shape:  {weather_df.shape} (24 rows per day)")
    print(f"Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()