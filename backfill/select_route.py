from __future__ import annotations

import math
from pathlib import Path
import pandas as pd


# ---------- CONFIG ----------
INPUT_CSV = "skane_VehiclePositions_2025-09-04_hourly.csv"  # change to your file path
TOP_N = 30

# Filters (tune these depending on whether you're analyzing 1 day vs 1 month)
MIN_DAYS = 1            # for one day test keep 1; for one month try 20+
MIN_HOURS = 24          # for one day, maybe 5+; for one month, 200+
MIN_HOUR_COVERAGE = 6   # distinct hours (0-23)
MIN_LABELS = 2          # needs variation

OUTPUT_DIR = "route_selection_out"
# ---------------------------


def entropy_from_counts(counts: pd.Series) -> float:
    """Shannon entropy of a label distribution."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p, 2)
    return float(ent)


def main() -> None:
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        raise FileNotFoundError(f"Cannot find input file: {in_path.resolve()}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load
    df = pd.read_csv(in_path)

    required = {"date", "hour", "route_id", "occupancy_status", "count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Basic cleaning
    df = df.dropna(subset=["date", "hour", "route_id", "occupancy_status", "count"]).copy()
    df["hour"] = df["hour"].astype(int)
    df["count"] = df["count"].astype(int)
    df["route_id"] = df["route_id"].astype(str)
    df["occupancy_status"] = df["occupancy_status"].astype(str)

    # 2) Collapse to one label per (date, hour, route_id) using max count
    # If tie: pick the "worst" (more crowded) by custom ordering
    crowd_order = {
        "EMPTY": 0,
        "MANY_SEATS_AVAILABLE": 1,
        "FEW_SEATS_AVAILABLE": 2,
        "STANDING_ROOM_ONLY": 3,
        "CRUSHED_STANDING_ROOM_ONLY": 4,
        "FULL": 5,
        "NOT_ACCEPTING_PASSENGERS": 6,
        "NO_DATA_AVAILABLE": -1,
        "NOT_BOARDABLE": -1,
    }

    df["_crowd_rank"] = df["occupancy_status"].map(lambda x: crowd_order.get(x, 0))

    # Sort so that within each group:
    # - higher count wins
    # - if equal count, higher crowd_rank wins (more crowded)
    df_sorted = df.sort_values(
        by=["date", "hour", "route_id", "count", "_crowd_rank"],
        ascending=[True, True, True, False, False],
    )

    winners = (
        df_sorted.groupby(["date", "hour", "route_id"], as_index=False)
        .first()[["date", "hour", "route_id", "occupancy_status", "count"]]
        .rename(columns={"occupancy_status": "hourly_label", "count": "label_count"})
    )

    winners.to_csv(out_dir / "hourly_labels.csv", index=False)

    # 3) Route stats
    # coverage stats
    stats = winners.groupby("route_id").agg(
        n_hours=("hourly_label", "size"),
        n_days=("date", "nunique"),
        hour_coverage=("hour", "nunique"),
        n_labels=("hourly_label", "nunique"),
    ).reset_index()

    # label distribution + entropy per route
    dist = (
        winners.groupby(["route_id", "hourly_label"])
        .size()
        .rename("n")
        .reset_index()
    )

    entropies = []
    for rid, g in dist.groupby("route_id"):
        entropies.append((rid, entropy_from_counts(g.set_index("hourly_label")["n"])))
    ent_df = pd.DataFrame(entropies, columns=["route_id", "label_entropy"])

    stats = stats.merge(ent_df, on="route_id", how="left")
    stats["label_entropy"] = stats["label_entropy"].fillna(0.0)

    # 4) Filter
    filtered = stats[
        (stats["n_days"] >= MIN_DAYS) &
        (stats["n_hours"] >= MIN_HOURS) &
        (stats["hour_coverage"] >= MIN_HOUR_COVERAGE) &
        (stats["n_labels"] >= MIN_LABELS)
    ].copy()

    # 5) Rank: prioritize lots of data + temporal spread + variability
    filtered = filtered.sort_values(
        by=["n_hours", "hour_coverage", "label_entropy", "n_labels"],
        ascending=[False, False, False, False],
    )

    # Save outputs
    stats.sort_values(by="n_hours", ascending=False).to_csv(out_dir / "all_routes_stats.csv", index=False)
    filtered.to_csv(out_dir / "filtered_ranked_routes.csv", index=False)

    # 6) Print top candidates
    print("\nTop candidate routes (after filters):")
    if len(filtered) == 0:
        print("No routes passed filters. Relax thresholds (MIN_HOURS/MIN_DAYS/etc.) and rerun.")
        return

    print(filtered.head(TOP_N).to_string(index=False))

    best = filtered.iloc[0]
    print("\nRecommended route_id:")
    print(best["route_id"])
    print(
        f"(n_hours={best['n_hours']}, n_days={best['n_days']}, "
        f"hour_coverage={best['hour_coverage']}, n_labels={best['n_labels']}, "
        f"entropy={best['label_entropy']:.3f})"
    )


if __name__ == "__main__":
    main()
