import os
from datetime import date
from pathlib import Path

from koda_processor import KoDaProcessor, KoDaConfig


def main() -> None:
    api_key = os.environ.get("KODA_API_KEY")
    if not api_key:
        raise RuntimeError("Set KODA_API_KEY first. PowerShell: $env:KODA_API_KEY='...'")

    route_id = "9011012065200000"

    proc = KoDaProcessor(
        api_key=api_key,
        config=KoDaConfig(
            operator="skane",
            feed="VehiclePositions",
            snapshots_per_hour=10,
        ),
    )

    start = date(2025, 9, 1)
    end = date(2025, 9, 30)

    out_csv = Path(f"out/skane_{start.isoformat()}_{end.isoformat()}_{route_id}_hourly_labels.csv")

    proc.process_range_hourly_labels_for_route(
        start=start,
        end=end,
        target_route_id=route_id,
        out_csv=out_csv,
    )

    print(f"Done. Wrote: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
