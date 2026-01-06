from __future__ import annotations

import io
import os
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import requests
import py7zr
from zoneinfo import ZoneInfo
from google.transit import gtfs_realtime_pb2


@dataclass(frozen=True)
class KoDaConfig:
    base_url: str = "https://api.koda.trafiklab.se/KoDa/api/v2"
    operator: str = "skane"
    feed: str = "VehiclePositions"
    snapshots_per_hour: int = 10
    timezone_local: str = "Europe/Stockholm"
    http_timeout_sec: int = 240
    retries: int = 10
    retry_sleep_sec: float = 3.0


class KoDaProcessor:
    DEFAULT_IGNORE_OCCUPANCY = {"NO_DATA_AVAILABLE", "NOT_BOARDABLE"}

    def __init__(
        self,
        api_key: str,
        config: KoDaConfig = KoDaConfig(),
        ignore_occupancy: Optional[set[str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.cfg = config
        self.ignore_occupancy = ignore_occupancy or set(self.DEFAULT_IGNORE_OCCUPANCY)
        self.tz_local = ZoneInfo(self.cfg.timezone_local)
        self.session = session or requests.Session()

    # URLs
    def static_url(self, date_str: str) -> str:
        return f"{self.cfg.base_url}/gtfs-static/{self.cfg.operator}?date={date_str}&key={self.api_key}"

    def rt_url(self, date_str: str) -> str:
        return f"{self.cfg.base_url}/gtfs-rt/{self.cfg.operator}/{self.cfg.feed}?date={date_str}&key={self.api_key}"

    # Basic downloader (retry for network/DNS). Polling handled outside.
    def download_bytes(self, url: str) -> bytes:
        import time
        last_err = None
        for attempt in range(1, self.cfg.retries + 1):
            try:
                r = self.session.get(url, timeout=self.cfg.http_timeout_sec)
                r.raise_for_status()
                return r.content
            except Exception as e:
                last_err = e
                if attempt < self.cfg.retries:
                    time.sleep(self.cfg.retry_sleep_sec * attempt)
        raise RuntimeError(f"Failed to download after {self.cfg.retries} attempts: {url}\nLast error: {last_err}")

    # Extract ZIP or 7z bytes to a folder
    def extract_archive_bytes(self, archive_bytes: bytes, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # ZIP magic
        if archive_bytes[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                zf.extractall(out_dir)
            return

        # 7z magic
        if archive_bytes[:6] == b"\x37\x7A\xBC\xAF\x27\x1C":
            with py7zr.SevenZipFile(io.BytesIO(archive_bytes), mode="r") as z:
                z.extractall(path=out_dir)
            return

        head = archive_bytes[:200]
        raise ValueError(f"Unknown archive format (not zip/7z). First bytes: {head!r}")

    # Find a GTFS file inside extracted static archive
    def find_first(self, root: Path, name: str) -> Path:
        matches = list(root.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Could not find {name} under {root}")
        return matches[0]

    # trips.txt => trip_id -> route_id
    def build_trip_to_route(self, gtfs_static_root: Path) -> Dict[str, str]:
        trips_path = self.find_first(gtfs_static_root, "trips.txt")
        mapping: Dict[str, str] = {}
        with trips_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("trip_id")
                rid = row.get("route_id")
                if tid and rid:
                    mapping[tid] = rid
        return mapping

    # Locate protobuf files
    def iter_pb_files(self, root: Path) -> Iterable[Path]:
        yield from root.rglob("*.pb")

    # Hour inferred from folder name .../YYYY/MM/DD/HH/<file>.pb
    def hour_from_path(self, pb_path: Path) -> int:
        return int(pb_path.parent.name)

    # Select k evenly spaced from list
    def select_evenly_spaced(self, files: List[Path], k: int) -> List[Path]:
        if not files:
            return []
        if len(files) <= k:
            return files
        idxs = [round(i * (len(files) - 1) / (k - 1)) for i in range(k)]
        selected, seen = [], set()
        for idx in idxs:
            if idx not in seen:
                seen.add(idx)
                selected.append(files[idx])
        return selected

    # Group pb files per hour
    def group_pb_by_hour(self, pb_files: List[Path]) -> Dict[int, List[Path]]:
        by_hour: DefaultDict[int, List[Path]] = defaultdict(list)
        for pb in pb_files:
            by_hour[self.hour_from_path(pb)].append(pb)
        return {h: sorted(v) for h, v in by_hour.items()}

    # Select k per hour across 24 hours
    def select_snapshots_per_hour(self, pb_files: List[Path], k_per_hour: int) -> List[Path]:
        by_hour = self.group_pb_by_hour(pb_files)
        chosen: List[Path] = []
        for h in range(24):
            chosen.extend(self.select_evenly_spaced(by_hour.get(h, []), k_per_hour))
        return chosen

    # Parse one snapshot (VehiclePositions)
    def parse_vehiclepositions_snapshot(
        self,
        pb_path: Path,
        trip_to_route: Dict[str, str],
        target_route_id: Optional[str] = None,
    ) -> List[Tuple[int, str, str]]:
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(pb_path.read_bytes())

        header_ts = int(feed.header.timestamp) if feed.header.timestamp else None
        out: List[Tuple[int, str, str]] = []

        for ent in feed.entity:
            if not ent.HasField("vehicle"):
                continue
            v = ent.vehicle

            # We only care about occupancy for this dataset
            if not v.HasField("occupancy_status"):
                continue

            ts = int(v.timestamp) if v.HasField("timestamp") else header_ts
            if ts is None:
                continue

            # route_id resolution
            route_id = ""
            if v.HasField("trip") and v.trip.route_id:
                route_id = v.trip.route_id
            else:
                trip_id = v.trip.trip_id if (v.HasField("trip") and v.trip.trip_id) else ""
                if trip_id:
                    route_id = trip_to_route.get(trip_id, "")

            if not route_id:
                continue
            if target_route_id and route_id != target_route_id:
                continue

            occ = gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(v.occupancy_status)
            if occ in self.ignore_occupancy:
                continue

            out.append((ts, route_id, occ))

        return out
