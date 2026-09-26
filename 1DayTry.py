import os
import io
import csv
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, Tuple, List, DefaultDict
from collections import defaultdict

import requests
import py7zr
from google.transit import gtfs_realtime_pb2


# ------------ CONFIG ------------
KODA_BASE = "https://api.koda.trafiklab.se/KoDa/api/v2"
OPERATOR = "skane"
FEED = "VehiclePositions"
DATE_STR = "2025-09-04"          # change later
SNAPSHOTS_PER_HOUR = 10          # your request
LOCAL_TZ = ZoneInfo("Europe/Stockholm")

# Optional: ignore these occupancy states (often not useful)
IGNORE_OCCUPANCY = {"NO_DATA_AVAILABLE", "NOT_BOARDABLE"}
# --------------------------------


def download_bytes(url: str, timeout: int = 180) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def extract_archive_bytes(archive_bytes: bytes, out_dir: Path) -> None:
    # ZIP signature
    if archive_bytes[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
            zf.extractall(out_dir)
        return

    # 7z signature
    if archive_bytes[:6] == b"\x37\x7A\xBC\xAF\x27\x1C":
        with py7zr.SevenZipFile(io.BytesIO(archive_bytes), mode="r") as z:
            z.extractall(path=out_dir)
        return

    head = archive_bytes[:200]
    raise ValueError(f"Unknown archive format (got non-zip/non-7z). First bytes: {head!r}")


def find_first(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def build_trip_to_route(gtfs_static_root: Path) -> Dict[str, str]:
    trips_path = find_first(gtfs_static_root, "trips.txt")
    mapping: Dict[str, str] = {}
    with trips_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trip_id")
            rid = row.get("route_id")
            if tid and rid:
                mapping[tid] = rid
    return mapping


def iter_pb_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.pb")


def hour_from_path(pb_path: Path) -> int:
    """
    KoDa exports are usually .../YYYY/MM/DD/HH/<snapshot>.pb
    So the parent folder name is the hour: "00".."23"
    """
    try:
        return int(pb_path.parent.name)
    except ValueError:
        # Fallback: if layout differs, you can extend this later
        raise RuntimeError(f"Can't infer hour from path: {pb_path}")


def select_evenly_spaced(files: List[Path], k: int) -> List[Path]:
    """
    Pick k files evenly spaced across the sorted list.
    If fewer than k files, return all.
    """
    if not files:
        return []
    if len(files) <= k:
        return files

    # indices spaced across [0, len-1]
    idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)]
    # deduplicate while preserving order
    seen = set()
    selected = []
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            selected.append(files[idx])
    return selected


def parse_vehiclepositions_snapshot_counts(
    pb_path: Path,
    trip_to_route: Dict[str, str],
) -> List[Tuple[int, str, str]]:
    """
    Parse one snapshot and return a list of (timestamp_utc, route_id, occupancy_name).
    One entry per vehicle WITH occupancy.
    """
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(pb_path.read_bytes())

    header_ts = int(feed.header.timestamp) if feed.header.timestamp else None
    out = []

    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue

        v = ent.vehicle
        if not v.HasField("occupancy_status"):
            continue

        ts = int(v.timestamp) if v.HasField("timestamp") else header_ts
        if ts is None:
            continue

        trip_id = v.trip.trip_id if (v.HasField("trip") and v.trip.trip_id) else ""

        # route_id: prefer realtime; else map via trips.txt
        route_id = ""
        if v.HasField("trip") and v.trip.route_id:
            route_id = v.trip.route_id
        elif trip_id:
            route_id = trip_to_route.get(trip_id, "")

        if not route_id:
            continue

        occ_name = gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(v.occupancy_status)
        if occ_name in IGNORE_OCCUPANCY:
            continue

        out.append((ts, route_id, occ_name))

    return out


def main():
    api_key = os.environ.get("KODA_API_KEY")
    if not api_key:
        raise RuntimeError("Set KODA_API_KEY env var first (PowerShell: $env:KODA_API_KEY='...').")

    static_url = f"{KODA_BASE}/gtfs-static/{OPERATOR}?date={DATE_STR}&key={api_key}"
    rt_url = f"{KODA_BASE}/gtfs-rt/{OPERATOR}/{FEED}?date={DATE_STR}&key={api_key}"

    out_csv = Path(f"{OPERATOR}_{FEED}_{DATE_STR}_hourly.csv")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ---- 1) static: trips.txt for trip_id -> route_id ----
        static_dir = tmp_path / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        extract_archive_bytes(download_bytes(static_url), static_dir)

        trip_to_route = build_trip_to_route(static_dir)
        print(f"Loaded trip->route mapping: {len(trip_to_route):,}")

        # ---- 2) realtime: VehiclePositions snapshots ----
        rt_dir = tmp_path / "rt"
        rt_dir.mkdir(parents=True, exist_ok=True)
        extract_archive_bytes(download_bytes(rt_url), rt_dir)

        pb_files = sorted(iter_pb_files(rt_dir))
        if not pb_files:
            raise RuntimeError("No .pb files found after extraction (archive layout unexpected).")

        # ---- 3) group by hour (from folder name) ----
        by_hour: DefaultDict[int, List[Path]] = defaultdict(list)
        for pb in pb_files:
            h = hour_from_path(pb)
            by_hour[h].append(pb)

        # ---- 4) pick 10 evenly spaced snapshots per hour ----
        selected_files: List[Tuple[int, Path]] = []
        for h in sorted(by_hour.keys()):
            files = sorted(by_hour[h])
            chosen = select_evenly_spaced(files, SNAPSHOTS_PER_HOUR)
            selected_files.extend((h, p) for p in chosen)

        print(f"Total snapshots found: {len(pb_files):,}")
        print(f"Snapshots selected ({SNAPSHOTS_PER_HOUR}/hour): {len(selected_files):,}")

        # ---- 5) aggregate counts per (date, hour, route_id, occupancy_status) ----
        # counts[(date, hour, route_id, occ)] += 1
        counts: DefaultDict[Tuple[str, int, str, str], int] = defaultdict(int)

        for hour_folder, pb in selected_files:
            rows = parse_vehiclepositions_snapshot_counts(pb, trip_to_route)
            for ts_utc, route_id, occ in rows:
                dt_local = datetime.fromtimestamp(ts_utc, tz=timezone.utc).astimezone(LOCAL_TZ)
                # Use the LOCAL date/hour from timestamps (more correct than folder hour)
                key = (dt_local.date().isoformat(), dt_local.hour, route_id, occ)
                counts[key] += 1

        # ---- 6) write aggregated CSV ----
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "hour", "route_id", "occupancy_status", "count"])
            for (d, h, route_id, occ), c in sorted(counts.items()):
                w.writerow([d, h, route_id, occ, c])

        print(f"Wrote hourly aggregated counts to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
