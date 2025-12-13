#!/usr/bin/env python
"""Crop and/or down-sample large GeoTIFF flood rasters.

Typical Aqueduct tiles are ~100 MB – this helper can
1.  crop them to a smaller bounding box,
2.  resample them to a coarser grid, or
3.  resample to the exact 1° model grid with MAX aggregation.

All operations use MAX resampling to preserve peak flood depths.

Examples
--------
1) Resample to model grid (recommended for simulation):
   $ python preprocess_geotiff.py \
         --input  data/raw/*.tif \
         --model-grid \
         --output-dir data/model_grid/

   This creates 360×180 rasters aligned exactly to the model's 1° grid
   (origin at -180, 90). MAX aggregation ensures that if any part of a
   1° cell floods, the cell gets that maximum depth value.

2) Crop only:
   $ python preprocess_geotiff.py \
         --input  fl_10.tif fl_100.tif \
         --crop-bounds -74.5 40.5 -73.0 41.5 \
         --output-dir processed/

3) Resample to 0.05° (~5 km) pixels:
   $ python preprocess_geotiff.py \
         --input  fl_10.tif fl_100.tif \
         --target-resolution 0.05 \
         --output-dir processed/

4) Crop *and* scale by 0.25 (=> 4× fewer pixels per axis):
   $ python preprocess_geotiff.py \
         --input  *.tif \
         --crop-bounds -74 40 -73 42 \
         --scale-factor 0.25

Notes
-----
* Coordinates are interpreted in the raster CRS (normally EPSG:4326 for
  Aqueduct). There is no reprojection.
* The --model-grid option overrides --scale-factor and --target-resolution.
* The script will **crash** on invalid inputs – no fall-back logic is
  implemented to keep debugging simple.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.warp import reproject


BBox = Tuple[float, float, float, float]  # minx, miny, maxx, maxy


def _parse_args() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser(description="Crop and/or resample GeoTIFF rasters.")
    p.add_argument("--input", nargs="+", required=True, help="One or more input GeoTIFF files")

    crop = p.add_argument_group("Cropping")
    crop.add_argument(
        "--crop-bounds",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Crop bounds in raster CRS (e.g. lon/lat)",
    )

    res = p.add_argument_group("Resampling")
    res.add_argument(
        "--scale-factor",
        type=float,
        help="Multiply both width and height by this factor (<1 for downsampling)",
    )
    res.add_argument(
        "--target-resolution",
        type=float,
        help="Desired pixel size in raster units (e.g. degrees). Overrides scale-factor.",
    )
    res.add_argument(
        "--model-grid",
        action="store_true",
        help="Resample to exact 1° grid aligned with model (-180 to 180, -90 to 90). "
             "Uses MAX aggregation. Overrides scale-factor and target-resolution.",
    )

    p.add_argument(
        "--output-dir",
        default="./preprocessed",
        help="Directory to write processed files (created if missing)",
    )

    return p.parse_args()


def _determine_transform(scale_factor: float, transform: rasterio.Affine) -> rasterio.Affine:
    return rasterio.Affine(
        transform.a / scale_factor,
        transform.b,
        transform.c,
        transform.d,
        transform.e / scale_factor,
        transform.f,
    )


def _process_one(
    path: Path,
    out_dir: Path,
    crop_bounds: BBox | None,
    scale_factor: float | None,
    target_res: float | None,
    model_grid: bool = False,
) -> None:  # noqa: D401
    with rasterio.open(path) as src:
        window = None
        if crop_bounds is not None:
            window = from_bounds(*crop_bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            transform = src.window_transform(window)
        else:
            data = src.read(1)
            transform = src.transform

        # Model grid: exact 1° resolution aligned to (-180, 90) origin
        if model_grid:
            # Model grid: 360 x 180 cells, 1° resolution
            # Origin at (-180, 90), pixel size (1.0, -1.0)
            dst_width = 360
            dst_height = 180
            dst_transform = rasterio.Affine(1.0, 0.0, -180.0, 0.0, -1.0, 90.0)
            dest = np.empty((dst_height, dst_width), dtype=data.dtype)

            reproject(
                source=data,
                destination=dest,
                src_transform=transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.max,
            )
            data = dest
            transform = dst_transform
            suffix = "_model-grid-1deg"
        else:
            # Original resampling logic
            if target_res is not None:
                scale_factor = target_res / transform.a  # assuming square pixels & no rotation
            if scale_factor is not None and scale_factor != 1:
                new_height = int(data.shape[0] * scale_factor)
                new_width = int(data.shape[1] * scale_factor)
                dest = np.empty((new_height, new_width), dtype=data.dtype)
                new_transform = _determine_transform(scale_factor, transform)

                reproject(
                    source=data,
                    destination=dest,
                    src_transform=transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.max,
                )
                data = dest
                transform = new_transform
            suffix = f"_scaled-{scale_factor}"

        profile = src.profile.copy()
        profile.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": transform,
            "compress": "lzw",
        })

        out_dir.mkdir(parents=True, exist_ok=True)
        if crop_bounds is not None:
            out_path = out_dir / f"{path.stem}{suffix}_cropped-{crop_bounds}.tif"
        else:
            out_path = out_dir / f"{path.stem}{suffix}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"[OK] Wrote {out_path}  (shape: {data.shape})")


def main() -> None:  # noqa: D401
    args = _parse_args()

    crop_bounds = tuple(args.crop_bounds) if args.crop_bounds else None  # type: ignore[arg-type]
    scale_factor = args.scale_factor
    target_res = args.target_resolution
    model_grid = args.model_grid

    in_files: List[Path] = [Path(p).expanduser() for p in args.input]
    out_dir = Path(args.output_dir).expanduser()

    for f in in_files:
        if not f.exists():
            raise FileNotFoundError(f)
        _process_one(f, out_dir, crop_bounds, scale_factor, target_res, model_grid)


if __name__ == "__main__":
    main() 