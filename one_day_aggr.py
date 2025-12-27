import os
import tempfile
import csv
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from koda_processor import KoDaConfig, KoDaProcessor


# Assumes the KoDaConfig and KoDaProcessor classes are defined above or imported
# from your provided snippet.

def main():
    # 1. Setup Configuration
    api_key = os.environ.get("KODA_API_KEY")
    if not api_key:
        raise RuntimeError("Set KODA_API_KEY env var first.")

    date_str = "2025-09-04"  # Target date
    config = KoDaConfig(
        snapshots_per_hour=10,
        operator="skane"
    )
    processor = KoDaProcessor(api_key=api_key, config=config)

    out_csv = Path(f"{config.operator}_VehiclePositions_{date_str}_hourly.csv")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ---- 2) Download & Extract Static GTFS ----
        print(f"Downloading static GTFS for {date_str}...")
        static_dir = tmp_path / "static"
        static_bytes = processor.download_bytes(processor.static_url(date_str))
        processor.extract_archive_bytes(static_bytes, static_dir)

        trip_to_route = processor.build_trip_to_route(static_dir)
        print(f"Loaded {len(trip_to_route):,} trip mappings.")

        # ---- 3) Download & Extract Realtime Data ----
        print(f"Downloading Realtime PB files for {date_str}...")
        rt_dir = tmp_path / "rt"
        rt_bytes = processor.download_bytes(processor.rt_url(date_str))
        processor.extract_archive_bytes(rt_bytes, rt_dir)

        # ---- 4) Filter & Select Snapshots ----
        all_pb_files = sorted(list(processor.iter_pb_files(rt_dir)))
        selected_files = processor.select_snapshots_per_hour(
            all_pb_files,
            config.snapshots_per_hour
        )
        print(f"Selected {len(selected_files)} snapshots across 24 hours.")

        # ---- 5) Parse & Aggregate Counts ----
        # Key: (date_iso, hour, route_id, occupancy_status)
        counts = defaultdict(int)

        for pb in selected_files:
            rows = processor.parse_vehiclepositions_snapshot(pb, trip_to_route)
            for ts_utc, route_id, occ in rows:
                # Convert UTC timestamp to Local Time (e.g., Stockholm)
                dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(processor.tz_local)

                key = (dt_local.date().isoformat(), dt_local.hour, route_id, occ)
                counts[key] += 1

        # ---- 6) Write Final CSV ----
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "hour", "route_id", "occupancy_status", "count"])
            for (d, h, rid, occ), count in sorted(counts.items()):
                writer.writerow([d, h, rid, occ, count])

        print(f"Success! File saved to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()