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
from typing import Dict, Tuple, List, Sequence, Iterable

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from climada.hazard import Hazard
from climada.hazard.centroids import Centroids  # type: ignore
from scipy.sparse import csr_matrix

Coords = Tuple[int, int]


def hazard_from_geotiffs(events: Iterable[Tuple[int, int, str]], haz_type: str = "FL") -> Tuple[Hazard, Sequence[float], Sequence[float]]:
    """Build a CLIMADA ``Hazard`` from GeoTIFF rasters.

    Each *event* is a tuple ``(return_period, year, file_path)``. Providing the
    year explicitly lets the caller mix multiple vintages of the same return
    period.

    Parameters
    ----------
    events
        Iterable of tuples ``(return_period, year, file_path)``.
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
    # Sort by year then RP for reproducible ordering
    sorted_events = sorted(events, key=lambda x: (x[1], x[0]))

    event_ids: List[int] = []  # we keep RP as id for simplicity
    event_names: List[str] = []
    event_dates: List[int] = []
    data_arrays: List[np.ndarray] = []

    for rp, year, fpath in sorted_events:
        path = Path(fpath).expanduser()
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)  # depth or intensity values
            transform = src.transform if not data_arrays else transform  # noqa: PLW2901
            data_arrays.append(data)

        event_ids.append(rp)
        event_names.append(f"RP{rp}_{year}")
        # Convert 1 Jan of year to ordinal date for Impact API
        import datetime as _dt
        event_dates.append(_dt.date(year, 1, 1).toordinal())

    # Normalise all rasters to 0–1 so they plug directly into the agent
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
    dense = np.vstack([(arr / global_max).flatten() for arr in data_arrays])
    intensity = csr_matrix(dense)

    # Frequencies = 1 / return period (events per year)
    frequency = 1 / np.array(event_ids, dtype=float)

    # --- Assemble CLIMADA Hazard ------------------------------------------------------------------
    haz = Hazard()
    # CLIMADA 4 uses attribute `haz_type` on the Hazard itself; older versions
    # expose it via `haz.tag.haz_type`. We set the direct attribute for
    # compatibility with recent releases.
    haz.haz_type = haz_type
    haz.event_id = np.array(event_ids, dtype=int)
    haz.event_name = np.array(event_names)
    haz.date = np.array(event_dates, dtype=int)
    haz.frequency = frequency
    haz.frequency_unit = "1/yr"
    haz.units = "m"  # Aqueduct provides inundation depth in metres

    cent = Centroids.from_lat_lon(lat, lon)
    haz.centroids = cent
    haz.intensity = intensity  # type: ignore[assignment]

    return haz, np.unique(lon), np.unique(lat) 