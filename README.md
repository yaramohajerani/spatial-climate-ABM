# Spatial economic ABM for climate risk

This repository provides a **spatial agent-based model** (ABM) that couples basic economic behaviour with climate-hazard impacts. It is built with [Mesa](https://mesa.readthedocs.io) and is designed to plug into [CLIMADA](https://github.com/CLIMADA-project/climada_python) as the climate-impact engine.

## Features included
* Spatial grid derived from user-supplied GeoTIFF hazard rasters (`mesa.space.MultiGrid`).
* Two agent classes:
  * `HouseholdAgent` (supplies labour, earns wages, consumes goods).
  * `FirmAgent` (multi-input Leontief production; supports agriculture, manufacturing, services … sectors).
* Supply-chain topology
  * Random, distance-weighted network **or** deterministic JSON file (`--topology`) with `lon/lat` coordinates & directed edges.
* Endogenous markets
  * Dynamic goods price (±5 % based on inventory).
  * Firm-specific wages that adjust endogenously: firms raise wages when they fail to hire and cut them when labour is abundant. Households choose employers by maximising *wage − distance_cost × distance* where the distance cost is heterogeneous across households.
  * Capital stock used in production, depreciates 2 % each step; firms reinvest when capital-constrained.
* Labour geography
  * A **1 ° geographic catchment** is converted to grid-cell units at start-up so the hiring radius is independent of raster resolution.
  * When a custom topology JSON is supplied, households are spawned near existing firms (within the same 1 ° radius) to guarantee local labour supply.
* Climate shocks via CLIMADA
  * Per-cell loss fraction applied to capital, inventories and productive capacity (`damage_factor` with 50 % annual recovery).
* Heterogenous adaptive risk behaviour
  * **Households:** each household samples the maximum normalised hazard within a random radius of **1–50 grid cells**. If that value exceeds **0.1** they instantly relocate to a safer land cell (and are counted in the `migrants_this_step` metric).
  * **Firms:** each firm monitors hazard in its own random radius (also 1–50 cells). When the local maximum hazard surpasses 0.1 the firm increases its **capital requirement** – operationalised as a 20 % rise in the Leontief capital coefficient. In the absence of new events the requirement relaxes back towards its baseline at a *firm-specific* decay rate between 20 % and 50 % per year.
* Full data capture every step:
  * Wealth, capital, production, consumption, labour sold, prices, wage, average risk.
  * Model-level CSV `simulation_results.csv`.
  * Agent panel CSV `simulation_agents.csv` (money, capital, etc.).
  * Composite figure `dashboard_timeseries.png` saved from dashboard.
* Reproducibility – pass `--seed` to freeze random placement, hazard draws and dashboard runs.
* **Interactive dashboard** (`--viz`)
  * Hazard map, network map (labour vs goods flows), live plots for all metrics, Save & Exit button.

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

### 4. Compare hazard vs baseline runs

To quantify the effect of climate hazards on the economy, run the helper
`compare_simulations.py`.  It launches two simulations back-to-back – **with**
and **without** hazard impacts – using the *exact same* random placement of
agents (controlled via `--seed`).  The script automatically overlays every
tracked metric in a multi-panel figure.

```bash
# 30-year run, same hazard raster as above
python compare_simulations.py \
    --rp-file 10:FL:data/flood_rp10.tif \
    --steps 30 \
    --seed 42 \
    --out comparison_plot.png
```

The resulting `comparison_plot.png` lets you visually inspect, for each metric
(total production, wealth, capital stock, wage, etc.), how trajectories differ
between the risk-free baseline and the hazard-exposed economy.