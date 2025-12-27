import pandas as pd
from pathlib import Path

# ---- EDIT THESE PATHS ----
CSV_A = Path(r"out\skane_2025-09-01_2025-09-30_route_9011012065200000_hourly.csv")
CSV_B = Path(r"out\skane_2024-10-01_2025-08-31_route_9011012065200000_hourly.csv")
CSV_C = Path(r"out\skane_2023-10-01_2024-08-31_route_9011012065200000_hourly.csv")
CSV_D = Path(r"out\skane_2022-10-01_2023-09-30_route_9011012065200000_hourly.csv")
CSV_E = Path(r"out\skane_2021-10-01_2022-09-30_route_9011012065200000_hourly.csv")
OUT  = Path(r"out\merged_4years_route_9011012065200000_hourly.csv")
# -------------------------

KEY_COLS = ["date", "hour", "route_id"]

def load_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    # basic type cleanup
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce").astype("Int64")
    # keep route_id as string (these IDs can be big)
    df["route_id"] = df["route_id"].astype(str)
    return df

def main():
    a = load_csv(CSV_A)
    b = load_csv(CSV_B)
    c = load_csv(CSV_C)
    d = load_csv(CSV_D)
    e = load_csv(CSV_E)

    # Combine
    merged = pd.concat([a, b, c, d, e], ignore_index=True)

    # Drop rows with invalid keys
    merged = merged.dropna(subset=["date", "hour", "route_id"])

    # Deduplicate: keep the row with the largest n_obs (usually “better”)
    if "n_obs" in merged.columns:
        merged["n_obs_num"] = pd.to_numeric(merged["n_obs"], errors="coerce").fillna(0)
        merged = merged.sort_values(KEY_COLS + ["n_obs_num"], ascending=[True, True, True, False])
        merged = merged.drop_duplicates(subset=KEY_COLS, keep="first")
        merged = merged.drop(columns=["n_obs_num"])
    else:
        merged = merged.drop_duplicates(subset=KEY_COLS, keep="first")

    # Sort nicely
    merged = merged.sort_values(KEY_COLS).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)

    # Quick summary
    print("Wrote:", OUT)
    print("Rows:", len(merged))
    print("Date range:", merged["date"].min(), "->", merged["date"].max())
    print("Unique dates:", merged["date"].nunique())
    print("Unique route_id:", merged["route_id"].nunique())

if __name__ == "__main__":
    main()
