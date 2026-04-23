"""Utility functions for loading hazard data from GeoTIFF rasters.

This module provides memory-efficient hazard loading using lazy sampling,
which reads only the pixels at agent locations instead of loading entire
rasters into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Sequence, Iterable

import numpy as np
import rasterio

Coords = Tuple[int, int]
HazardEventSpec = Tuple[int, int, int, str, str | None]

_DEPTH_SCALE: float = 6.0
_DETERMINISTIC_FREQUENCY: float = 1e6


class LazyHazard:
    """Memory-efficient hazard that samples from GeoTIFF files on-demand.

    Instead of loading entire rasters into memory, this class stores file paths
    and uses rasterio.sample() to read only the pixels at agent locations.
    This reduces memory usage from ~4GB to <1MB for global hazard datasets.

    Parameters
    ----------
    events
        List of tuples ``(return_period, file_path)``.
    haz_type
        Hazard type tag (defaults to flood: ``"FL"``).
    nodata_threshold
        Values below this threshold are treated as nodata (default: -9000).
        Aqueduct uses -9999 as nodata, but some files have other negative values.
    """

    def __init__(
        self,
        events: List[Tuple[int, str]],
        haz_type: str = "FL",
        nodata_threshold: float = -9000.0,
    ) -> None:
        self.haz_type = haz_type
        self.nodata_threshold = nodata_threshold

        # Store event metadata
        self.event_files: List[Path] = []
        self.return_periods: List[int] = []
        self.frequency: np.ndarray

        for rp, fpath in events:
            path = Path(fpath).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Hazard file not found: {path}")
            self.event_files.append(path)
            self.return_periods.append(rp)

        # Frequencies = 1 / return period (events per year)
        self.frequency = 1.0 / np.array(self.return_periods, dtype=float)

        # Cache raster metadata from first file (all files should have same grid)
        with rasterio.open(self.event_files[0]) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.width = src.width
            self.height = src.height
            self.bounds = src.bounds
            self._nodata_value = src.nodata

    @property
    def n_events(self) -> int:
        """Number of hazard events (return periods)."""
        return len(self.event_files)

    def sample_at_coords(
        self,
        coords: List[Tuple[float, float]],
        event_idx: int,
    ) -> np.ndarray:
        """Sample hazard intensity at given (lon, lat) coordinates for one event.

        Parameters
        ----------
        coords
            List of (longitude, latitude) tuples to sample.
        event_idx
            Index of the event (0-based) to sample from.

        Returns
        -------
        depths
            Array of intensity values at each coordinate. NODATA values are
            replaced with 0.0.
        """
        if event_idx < 0 or event_idx >= self.n_events:
            raise IndexError(f"Event index {event_idx} out of range [0, {self.n_events})")

        with rasterio.open(self.event_files[event_idx]) as src:
            # rasterio.sample returns generator of arrays
            values = np.array([v[0] for v in src.sample(coords)])

        # Replace nodata with 0
        values = np.where(values < self.nodata_threshold, 0.0, values)
        # Also replace any remaining negative values (bad data)
        values = np.maximum(values, 0.0)

        return values

    def sample_all_events(
        self,
        coords: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Sample all events at given coordinates.

        Parameters
        ----------
        coords
            List of (longitude, latitude) tuples to sample.

        Returns
        -------
        intensities
            Array of shape (n_events, n_coords) with intensity values.
        """
        n_coords = len(coords)
        intensities = np.zeros((self.n_events, n_coords), dtype=np.float32)

        for i in range(self.n_events):
            intensities[i, :] = self.sample_at_coords(coords, i)

        return intensities


class SyntheticHazard:
    """Point-driven hazard matching the ``LazyHazard`` sampling interface.

    Synthetic node shocks use a normalized intensity in ``[0, 1]``. That value
    is mapped to a pseudo-depth via ``intensity * 6 m`` so the resulting damage
    is evaluated through the same flood damage curves used for raster hazards.
    """

    def __init__(
        self,
        affected_coords: Sequence[Tuple[float, float]],
        intensity: float,
        haz_type: str = "SYNTHETIC",
        radius_deg: float = 0.5,
        return_period: float | None = None,
    ) -> None:
        self.haz_type = haz_type
        self._affected = np.array(list(affected_coords), dtype=np.float64)
        self._pseudo_depth = float(intensity) * _DEPTH_SCALE
        self._radius_deg = float(radius_deg)

        if return_period is None or return_period <= 0:
            self.frequency = np.array([_DETERMINISTIC_FREQUENCY], dtype=np.float64)
        else:
            self.frequency = np.array([1.0 / float(return_period)], dtype=np.float64)

        self.return_periods: List[int] = [
            max(1, int(return_period)) if return_period else 1
        ]

    @property
    def n_events(self) -> int:
        return 1

    def sample_at_coords(
        self,
        coords: List[Tuple[float, float]],
        event_idx: int,
    ) -> np.ndarray:
        if event_idx != 0:
            raise IndexError(f"SyntheticHazard has only 1 event; got event_idx={event_idx}")

        result = np.zeros(len(coords), dtype=np.float64)
        if len(self._affected) == 0:
            return result

        query = np.array(coords, dtype=np.float64)
        for i in range(len(query)):
            diffs = self._affected - query[i]
            min_dist = float(np.min(np.hypot(diffs[:, 0], diffs[:, 1])))
            if min_dist <= self._radius_deg:
                result[i] = self._pseudo_depth
        return result

    def sample_all_events(
        self,
        coords: List[Tuple[float, float]],
    ) -> np.ndarray:
        return self.sample_at_coords(coords, 0).reshape(1, -1)


def parse_hazard_event_specs(rp_files: list[str] | str | None) -> list[HazardEventSpec]:
    """Parse CLI/JSON hazard specifications into event tuples.

    Supports warm-up windows by allowing the path segment to be ``None``.
    Entries with ``None`` still define a no-hazard interval in the scenario
    metadata, but no raster is loaded for that window.
    """
    if not rp_files:
        return []
    if isinstance(rp_files, str):
        rp_files = [rp_files]

    events: list[HazardEventSpec] = []
    for item in rp_files:
        try:
            rp_str, start_str, end_str, type_str, path_str = item.split(":", 4)
        except ValueError as exc:  # noqa: BLE001
            raise ValueError(
                f"Invalid hazard specification: {item}. Expected <RP>:<START>:<END>:<TYPE>:<path|None>."
            ) from exc
        path_value: str | None
        if path_str.strip().lower() in {"none", "null", ""}:
            path_value = None
        else:
            path_value = path_str
        events.append((int(rp_str), int(start_str), int(end_str), type_str, path_value))
    return events


def lazy_hazard_from_geotiffs(
    events: Iterable[Tuple[int, str]],
    haz_type: str = "FL",
) -> Tuple[LazyHazard, Sequence[float], Sequence[float]]:
    """Create a LazyHazard from GeoTIFF rasters without loading full data.

    This function reads only raster metadata, not the actual pixel data.
    Hazard intensities are sampled on-demand at agent locations.

    Parameters
    ----------
    events
        Iterable of tuples ``(return_period, file_path)``.
    haz_type
        Hazard type tag (defaults to flood: ``"FL"``).

    Returns
    -------
    hazard
        A LazyHazard instance that samples on-demand.
    lon_vals, lat_vals
        Sorted unique longitude and latitude arrays for grid construction.
        Note: These are for reference only; the model uses a coarser agent
        grid decoupled from the hazard resolution.
    """
    event_list = list(events)

    # Create lazy hazard
    haz = LazyHazard(event_list, haz_type=haz_type)

    # Build coordinate arrays from raster metadata (without loading data)
    rows = np.arange(haz.height)
    cols = np.arange(haz.width)

    # Get lon/lat for each row/col using the transform
    # transform.c = x origin, transform.a = x pixel size
    # transform.f = y origin, transform.e = y pixel size (negative for north-up)
    lon_vals = haz.transform.c + (cols + 0.5) * haz.transform.a
    lat_vals = haz.transform.f + (rows + 0.5) * haz.transform.e

    return haz, lon_vals, lat_vals
