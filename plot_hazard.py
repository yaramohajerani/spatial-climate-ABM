#!/usr/bin/env python
"""Quick visualisation helper for CLIMADA hazard rasters.

Usage
-----
$ python plot_hazard.py --hazard-file demo_hist.hdf5               # interactive window
$ python plot_hazard.py --hazard-file demo_hist.hdf5 -o map.png    # save to PNG

If the file is compressed (``.hdf5.gz``) it will be automatically un-gzipped
once next to the original for faster subsequent calls.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from climada.hazard import Hazard
from scipy.sparse import issparse  # noqa: WPS433 – intentional dependency


def _load_hazard(path: Path, year: Optional[int] = None) -> Hazard:  # noqa: D401
    """Load a ``climada.hazard.Hazard`` object, optionally filtered by year.

    Also handles on-the-fly decompression of ``*.hdf5.gz`` archives so the user
    can pass either version. The decompressed copy is written next to the
    original to avoid repeating the cost every run.
    """
    if path.suffix == ".gz":
        dest = path.with_suffix("")
        if not dest.exists():
            print(f"[INFO] Decompressing {path.name} → {dest.name}")
            with gzip.open(path, "rb") as fin, open(dest, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        path = dest

    if not path.exists():
        raise FileNotFoundError(path)

    haz = Hazard.from_hdf5(str(path))

    if year is not None and hasattr(haz, "event_id"):
        years = np.array([int(str(eid)[:4]) for eid in haz.event_id])
        idx = np.where(years == year)[0]
        if idx.size == 0:
            raise ValueError(f"No events for year {year} found in {path}.")
        haz = haz.select(idx)

    return haz


def _max_intensity(haz: Hazard) -> np.ndarray:  # noqa: D401
    """Return max intensity across events as a 1-D dense array."""
    intensity = haz.intensity.max(axis=0)  # type: ignore [attr-defined]
    if issparse(intensity):
        intensity = intensity.toarray().ravel()
    else:
        intensity = np.asarray(intensity).ravel()
    return intensity


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot CLIMADA hazard intensity raster")
    p.add_argument("--hazard-file", required=True, help="Path to *.hdf5 or *.hdf5.gz file")
    p.add_argument("--hazard-year", type=int, help="Filter to events in this year (if available)")
    p.add_argument("-o", "--output", help="Write figure to this file instead of showing interactively")
    return p


def main() -> None:  # noqa: D401
    args = _parser().parse_args()

    path = Path(args.hazard_file).expanduser()
    haz = _load_hazard(path, args.hazard_year)

    lon = haz.centroids.lon
    lat = haz.centroids.lat
    intensity = _max_intensity(haz)

    # Normalise 0-1 for consistent colour scale
    if intensity.ptp() > 0:
        intensity = (intensity - intensity.min()) / intensity.ptp()

    # Try to detect regular grid so we can use pcolormesh (fast & pretty)
    unique_lon = np.unique(lon)
    unique_lat = np.unique(lat)
    is_regular = lon.size == unique_lon.size * unique_lat.size

    fig, ax = plt.subplots(figsize=(8, 4))
    if is_regular:
        # Build 2-D grid
        grid = np.full((unique_lat.size, unique_lon.size), np.nan)
        lon_to_x = {lo: i for i, lo in enumerate(unique_lon)}
        lat_to_y = {la: j for j, la in enumerate(unique_lat)}
        for lo, la, inten in zip(lon, lat, intensity):
            x = lon_to_x[lo]
            y = lat_to_y[la]
            grid[y, x] = inten  # pcolormesh expects [Y, X]

        mesh = ax.pcolormesh(unique_lon, unique_lat, grid, cmap="viridis", shading="auto")
        ax.set_title("Hazard intensity (max across events)")
        fig.colorbar(mesh, ax=ax, label="Normalised intensity")
    else:
        # Fallback: scatter plot – works for irregular meshes too
        sc = ax.scatter(lon, lat, c=intensity, s=15, cmap="viridis")
        ax.set_title("Hazard intensity (scatter)")
        fig.colorbar(sc, ax=ax, label="Normalised intensity")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if args.output:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure written to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main() 