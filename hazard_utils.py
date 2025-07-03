"""Utility functions to build CLIMADA hazards from external GeoTIFF rasters.

The typical use-case is to ingest Aqueduct flood rasters where each file
represents the inundation depth for a specific return period (e.g. 10, 50,
100-year floods). We convert those rasters into a CLIMADA ``Hazard`` with one
*event* per return period so the engine can later compute impacts.

There is **no error handling or fall-back logic** on purpose – if the files are
missing or malformed the program will crash immediately, making debugging
straightforward.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Sequence

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from climada.hazard import Hazard
from climada.hazard.centroids import Centroids  # type: ignore

Coords = Tuple[int, int]


def hazard_from_geotiffs(rp_files: Dict[int, str], haz_type: str = "FL") -> Tuple[Hazard, Sequence[float], Sequence[float]]:
    """Build a CLIMADA ``Hazard`` from GeoTIFF rasters keyed by return period.

    Parameters
    ----------
    rp_files
        Mapping *return period → path* of GeoTIFF files. All rasters must share
        the same spatial resolution, extent, and CRS.
    haz_type
        CLIMADA hazard type tag (defaults to flood: ``"FL"``).

    Returns
    -------
    hazard
        A CLIMADA ``Hazard`` instance with one event per return period.
    lon_vals, lat_vals
        Sorted unique longitude and latitude arrays. These are handy for
        translating grid indices <-> geographic coordinates in the ABM.
    """
    # Sort return periods to have a deterministic order / event_id sequence
    rps: List[int] = sorted(rp_files)

    data_arrays: List[np.ndarray] = []
    for rp in rps:
        path = Path(rp_files[rp]).expanduser()
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)  # depth or intensity values
            transform = src.transform if not data_arrays else transform  # noqa: PLW2901
            data_arrays.append(data)

    # Normalise all rasters to 0–1 so they plug directly into the agent
    # behaviour where 1.0 means "total loss".
    global_max = max(arr.max() for arr in data_arrays)
    if global_max == 0:
        raise ValueError("All rasters contain only zeros – cannot normalise.")

    # Build centroid coordinates from the *first* raster (they are all identical)
    height, width = data_arrays[0].shape
    rows, cols = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lon = np.array(xs).flatten()
    lat = np.array(ys).flatten()

    # Prepare intensity matrix [n_events, n_centroids]
    intensity = np.vstack([(arr / global_max).flatten() for arr in data_arrays])

    # Frequencies = 1 / return period (events per year)
    frequency = 1 / np.array(rps, dtype=float)

    # --- Assemble CLIMADA Hazard ------------------------------------------------------------------
    haz = Hazard()
    haz.tag.haz_type = haz_type
    haz.event_id = np.array(rps, dtype=int)
    haz.frequency = frequency
    haz.units = "m"  # Aqueduct provides inundation depth in metres

    cent = Centroids(pd.DataFrame({"lon": lon, "lat": lat}))
    haz.centroids = cent
    haz.intensity = intensity  # type: ignore[assignment]

    return haz, np.unique(lon), np.unique(lat) 