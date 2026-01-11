"""
KoDa Processor Module

This module provides utilities for interacting with the KoDa (Kollektivtrafik Data)
API to download and process GTFS-RT VehiclePositions data.

Classes:
    KoDaConfig: Configuration dataclass for the processor.
    KoDaProcessor: Main class for downloading and parsing GTFS data.

Functions:
    _parse_pb_worker: Parallel worker function for parsing protobuf files.
"""
from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import py7zr
import requests
from google.transit import gtfs_realtime_pb2
from zoneinfo import ZoneInfo

@dataclass(frozen=True)
class KoDaConfig:
    """
    Configuration for the KoDa API processor.

    Attributes:
        base_url: Base URL for the KoDa API.
        operator: Transit operator identifier (e.g., 'skane').
        feed: GTFS-RT feed type (e.g., 'VehiclePositions').
        snapshots_per_hour: Number of evenly-spaced snapshots to select per hour.
        timezone_local: Local timezone for timestamp conversion.
        http_timeout_sec: HTTP request timeout in seconds.
        retries: Number of retry attempts for failed requests.
        retry_sleep_sec: Sleep duration between retry attempts.
    """

    base_url: str = "https://api.koda.trafiklab.se/KoDa/api/v2"
    operator: str = "skane"
    feed: str = "VehiclePositions"
    snapshots_per_hour: int = 30
    timezone_local: str = "Europe/Stockholm"
    http_timeout_sec: int = 240
    retries: int = 10
    retry_sleep_sec: float = 3.0


class KoDaProcessor:
    """
    Processor for downloading and parsing KoDa GTFS-RT data.

    This class provides methods to:
    - Download static GTFS and real-time VehiclePositions archives
    - Extract and parse protobuf files
    - Map trip IDs to route IDs and directions
    - Select evenly-spaced snapshots for sampling

    Attributes:
        api_key: KoDa API key for authentication.
        cfg: Configuration object with API settings.
        ignore_occupancy: Set of occupancy statuses to ignore.
        tz_local: Local timezone for timestamp conversion.
        session: HTTP session for requests.
    """

    DEFAULT_IGNORE_OCCUPANCY = {"NO_DATA_AVAILABLE", "NOT_BOARDABLE"}

    def __init__(
        self,
        api_key: str,
        config: KoDaConfig = KoDaConfig(),
        ignore_occupancy: Optional[set[str]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Initialize the KoDa processor.

        Args:
            api_key: API key for KoDa authentication.
            config: Configuration object with API settings.
            ignore_occupancy: Set of occupancy status strings to ignore.
            session: Optional requests.Session for HTTP calls.
        """
        self.api_key = api_key
        self.cfg = config
        self.ignore_occupancy = ignore_occupancy or set(self.DEFAULT_IGNORE_OCCUPANCY)
        self.tz_local = ZoneInfo(self.cfg.timezone_local)
        self.session = session or requests.Session()


    def static_url(self, date_str: str) -> str:
        """Generate URL for static GTFS archive for a given date."""
        return f"{self.cfg.base_url}/gtfs-static/{self.cfg.operator}?date={date_str}&key={self.api_key}"

    def rt_url(self, date_str: str) -> str:
        """Generate URL for real-time GTFS archive for a given date."""
        return f"{self.cfg.base_url}/gtfs-rt/{self.cfg.operator}/{self.cfg.feed}?date={date_str}&key={self.api_key}"


    def download_bytes(self, url: str) -> bytes:
        """
        Download content from URL with retry logic.

        Args:
            url: URL to download from.

        Returns:
            Downloaded content as bytes.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
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
        raise RuntimeError(
            f"Failed to download after {self.cfg.retries} attempts: {url}\nLast error: {last_err}"
        )


    def extract_archive_bytes(self, archive_bytes: bytes, out_dir: Path) -> None:
        """
        Extract a ZIP or 7z archive from bytes to a directory.

        Args:
            archive_bytes: Raw archive content as bytes.
            out_dir: Directory to extract contents to.

        Raises:
            ValueError: If the archive format is not recognized.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # ZIP magic bytes
        if archive_bytes[:4] == b"PK\x03\x04":
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
                zf.extractall(out_dir)
            return

        # 7z magic bytes
        if archive_bytes[:6] == b"\x37\x7A\xBC\xAF\x27\x1C":
            with py7zr.SevenZipFile(io.BytesIO(archive_bytes), mode="r") as z:
                z.extractall(path=out_dir)
            return

        head = archive_bytes[:200]
        raise ValueError(f"Unknown archive format (not zip/7z). First bytes: {head!r}")


    def find_first(self, root: Path, name: str) -> Path:
        """
        Find the first file matching a name in a directory tree.

        Args:
            root: Root directory to search from.
            name: Filename to search for.

        Returns:
            Path to the first matching file.

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        matches = list(root.rglob(name))
        if not matches:
            raise FileNotFoundError(f"Could not find {name} under {root}")
        return matches[0]

    def build_trip_to_route(self, gtfs_static_root: Path) -> Dict[str, Tuple[str, int]]:
        """
        Build a mapping from trip_id to (route_id, direction_id) from trips.txt.

        Args:
            gtfs_static_root: Root directory of extracted GTFS static files.

        Returns:
            Dictionary mapping trip_id to (route_id, direction_id) tuple.
        """
        trips_path = self.find_first(gtfs_static_root, "trips.txt")
        mapping: Dict[str, Tuple[str, int]] = {}
        with trips_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("trip_id")
                rid = row.get("route_id")
                dir_id_str = row.get("direction_id", "0")
                try:
                    dir_id = int(dir_id_str) if dir_id_str else 0
                except ValueError:
                    dir_id = 0
                if tid and rid:
                    mapping[tid] = (rid, dir_id)
        return mapping

    def iter_pb_files(self, root: Path) -> Iterable[Path]:
        """Yield all protobuf (.pb) files recursively from a directory."""
        yield from root.rglob("*.pb")

    def hour_from_path(self, pb_path: Path) -> int:
        """
        Extract hour from protobuf file path.

        Assumes path format: .../YYYY/MM/DD/HH/<file>.pb
        """
        return int(pb_path.parent.name)

    def select_evenly_spaced(self, files: List[Path], k: int) -> List[Path]:
        """
        Select k evenly-spaced files from a sorted list.

        Args:
            files: Sorted list of file paths.
            k: Number of files to select.

        Returns:
            List of k evenly-spaced files, or all files if len(files) <= k.
        """
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

    def group_pb_by_hour(self, pb_files: List[Path]) -> Dict[int, List[Path]]:
        """Group protobuf files by hour extracted from their paths."""
        by_hour: DefaultDict[int, List[Path]] = defaultdict(list)
        for pb in pb_files:
            by_hour[self.hour_from_path(pb)].append(pb)
        return {h: sorted(v) for h, v in by_hour.items()}

    def select_snapshots_per_hour(
        self, pb_files: List[Path], k_per_hour: int
    ) -> List[Path]:
        """
        Select k evenly-spaced snapshots for each hour (0-23).

        Args:
            pb_files: List of all protobuf files.
            k_per_hour: Number of snapshots to select per hour.

        Returns:
            List of selected protobuf file paths across all hours.
        """
        by_hour = self.group_pb_by_hour(pb_files)
        chosen: List[Path] = []
        for h in range(24):
            chosen.extend(self.select_evenly_spaced(by_hour.get(h, []), k_per_hour))
        return chosen

    # Parse one snapshot (VehiclePositions) - returns ALL routes with direction
    def parse_vehiclepositions_snapshot(
        self,
        pb_path: Path,
        trip_to_route: Dict[str, Tuple[str, int]],
    ) -> List[Tuple[int, str, int, str]]:
        """Parse a VehiclePositions pb file, returning (timestamp, route_id, direction_id, occupancy_status) for all vehicles."""
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(pb_path.read_bytes())

        header_ts = int(feed.header.timestamp) if feed.header.timestamp else None
        out: List[Tuple[int, str, int, str]] = []

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

            # route_id and direction_id resolution
            route_id = ""
            direction_id = 0
            if v.HasField("trip") and v.trip.route_id:
                route_id = v.trip.route_id
                # Try to get direction from trip_id lookup
                trip_id = v.trip.trip_id if v.trip.trip_id else ""
                if trip_id and trip_id in trip_to_route:
                    _, direction_id = trip_to_route[trip_id]
            else:
                trip_id = v.trip.trip_id if (v.HasField("trip") and v.trip.trip_id) else ""
                if trip_id and trip_id in trip_to_route:
                    route_id, direction_id = trip_to_route[trip_id]

            if not route_id:
                continue

            occ = gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(v.occupancy_status)
            if occ in self.ignore_occupancy:
                continue

            out.append((ts, route_id, direction_id, occ))

        return out


# Worker function for multiprocessing (must be at module level)
def _parse_pb_worker(args: Tuple[Path, Dict[str, Tuple[str, int]], set]) -> List[Tuple[int, str, int, str]]:
    """Worker function for parallel pb parsing."""
    pb_path, trip_to_route, ignore_occupancy = args
    from google.transit import gtfs_realtime_pb2
    
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(pb_path.read_bytes())
    
    header_ts = int(feed.header.timestamp) if feed.header.timestamp else None
    out: List[Tuple[int, str, int, str]] = []
    
    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        
        if not v.HasField("occupancy_status"):
            continue
        
        ts = int(v.timestamp) if v.HasField("timestamp") else header_ts
        if ts is None:
            continue
        
        route_id = ""
        direction_id = 0
        if v.HasField("trip") and v.trip.route_id:
            route_id = v.trip.route_id
            # Try to get direction from trip_id lookup
            trip_id = v.trip.trip_id if v.trip.trip_id else ""
            if trip_id and trip_id in trip_to_route:
                _, direction_id = trip_to_route[trip_id]
        else:
            trip_id = v.trip.trip_id if (v.HasField("trip") and v.trip.trip_id) else ""
            if trip_id and trip_id in trip_to_route:
                route_id, direction_id = trip_to_route[trip_id]
        
        if not route_id:
            continue
        
        occ = gtfs_realtime_pb2.VehiclePosition.OccupancyStatus.Name(v.occupancy_status)
        if occ in ignore_occupancy:
            continue
        
        out.append((ts, route_id, direction_id, occ))
    
    return out

