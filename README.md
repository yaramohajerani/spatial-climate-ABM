# Spatial economic ABM for climate risk

This repository provides a **spatial agent-based model** (ABM) that couples basic economic behaviour with climate-hazard impacts. It is built with [Mesa](https://mesa.readthedocs.io) and is designed to plug into [CLIMADA](https://github.com/CLIMADA-project/climada_python) as the climate-impact engine.

## Features included

* Agents distributed on spatial grid corresponding to the input climate data file (`mesa.space.MultiGrid`).
* `HouseholdAgent` and `FirmAgent` instances randomly distributed (number provided by user).
* CLIMADA hazard data used for input climate shocks.
* Migration by households when local risk exceeds a threshold.
* Firm production reduced in proportion to local hazard intensity.
* Collection of yearly GDP and migrant counts.
* Results saved to `simulation_results.csv`.
* **live dashboard** built with Mesa's Solara API – launch with `--viz` to watch the grid and charts update in real time.

## Quick start

### 1. Create and activate a virtual environment (optional but recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model

The model now expects external GeoTIFF rasters instead of CLIMADA HDF5 files. Each
file is mapped to a *hazard event* via an explicit triple:

```
<RETURN-PERIOD>:<HAZARD_TYPE>:<path/to/geotiff.tif>
```

You can pass any number of `--rp-file` arguments – one per event – to mix
different return periods and years, for example:

```bash
# Mixed hazards: a 10-year flood and a 50-year wildfire raster
python run_simulation.py \
    --rp-file 10:FL:data/flood_rp10.tif \
    --rp-file 50:WF:data/wildfire_rp50.tif
```

Add `--viz` to launch the interactive Solara dashboard instead of the headless
batch run.

### Optional preprocessing

Aqueduct tiles are large (~100 MB). Use the helper under
`prepare_hazard/preprocess_geotiff.py` to crop or down-sample rasters before
feeding them into the ABM:

```bash
# Crop to bounding box and resample to half resolution
python prepare_hazard/preprocess_geotiff.py \
    --input raw/*.tif \
    --crop-bounds -74 40 -73 42 \
    --scale-factor 0.5 \
    --output-dir data/processed
```

Point the `--rp-file` paths to the generated `*_processed.tif` files. The model
will automatically select an appropriate vulnerability curve for each
`HAZARD_TYPE` (flood = JRC depth-damage, others default to linear until you
add custom curves). The Solara dashboard and `simulation_results.csv` remain
unchanged. A second CSV `applied_events.csv` logs which cells flooded each
year.