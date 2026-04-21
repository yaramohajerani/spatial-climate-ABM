# Spatial Climate-Economy Agent-Based Model

[![arXiv](https://img.shields.io/badge/arXiv-2509.18633-b31b1b.svg)](https://arxiv.org/abs/2509.18633)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![NeurIPS 2025 CCAI](https://img.shields.io/badge/NeurIPS%202025-Climate%20Change%20AI-green.svg)](https://www.climatechange.ai/papers/neurips2025/19)

Open-source Python framework for modelling cascading physical climate risk in
spatial supply-chain economies. The model combines geospatial flood hazards
with an agent-based economy of firms and households, so it can represent both
direct asset losses and indirect disruptions propagated through labor markets,
input linkages, prices, and firm finance.

The current codebase supports hazard-conditional firm adaptation through
multiple deployment channels, including:

- `capital_hardening` for direct-loss attenuation
- `backup_suppliers` for input-continuity support
- `stockpiling` and `reserved_capacity` as additional experimental strategies
  (`stockpiling` currently means a larger finished-goods buffer, not extra input inventories)

## What the Framework Does

- Samples flood hazard rasters directly from GeoTIFF files at firm locations
- Simulates firms and households on a spatial grid with supply-chain and labor
  interactions
- Records direct hazard exposure, supplier disruption, and never-hit cascade
  burden diagnostics
- Supports matched-seed ensembles for reproducible scenario comparison
- Writes self-describing summary and member CSVs with `Meta_*` fields so runs
  remain interpretable after the fact

## Installation

Tested with Python `3.11`.

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
# Required only for reading the JRC Excel damage-function workbook
pip install openpyxl

# Required only for the interactive dashboard
pip install solara
```

`CLIMADA` is no longer required. Flood damage is read directly from the JRC
depth-damage workbook in `data/`.

## Quick Start

Run a small headless test:

```bash
python run_simulation.py --param-file quick_test_parameters.json
```

Run the main paper-style 20-seed comparison:

```bash
python run_simulation.py --param-file aqueduct_riverine_parameters_rcp8p5.json --no-hazards --no-adaptation --n-seeds 20 --seed-start 41
python run_simulation.py --param-file aqueduct_riverine_parameters_rcp8p5.json --no-adaptation --n-seeds 20 --seed-start 41
python run_simulation.py --param-file aqueduct_riverine_parameters_rcp8p5.json --adaptation-strategy capital_hardening --adaptation-sensitivity-min 0.5 --adaptation-sensitivity-max 1.5 --n-seeds 20 --seed-start 41
python run_simulation.py --param-file aqueduct_riverine_parameters_rcp8p5.json --adaptation-strategy backup_suppliers --adaptation-sensitivity-min 0.8 --adaptation-sensitivity-max 1.4 --n-seeds 20 --seed-start 41
```

Plot the paper-style time-series comparison from saved CSVs:

```bash
python plot_from_csv.py \
  --csv-files simulation_baseline_noadaptation_...csv simulation_hazard_noadaptation_...csv simulation_hazard_capital_hardening_...csv simulation_hazard_backup_suppliers_...csv \
  --show-ensemble-band \
  --plot-start-year 2020
```

Plot the cascade-risk diagnostics:

```bash
python plot_cascade_risk.py \
  --csv-files simulation_hazard_noadaptation_...csv simulation_hazard_capital_hardening_...csv simulation_hazard_backup_suppliers_...csv \
  --show-ensemble-band \
  --out cascade_risk.png
```

Run the matched-seed sensitivity analysis:

```bash
python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --n-seeds 10 --seed-start 41
```

Launch the interactive dashboard:

```bash
python run_simulation.py --param-file aqueduct_riverine_parameters_rcp8p5.json --viz
```

## Core Scripts

- [`run_simulation.py`](run_simulation.py):
  main CLI for single-seed and multi-seed runs
- [`plot_from_csv.py`](plot_from_csv.py):
  maintained time-series plotter
- [`plot_cascade_risk.py`](plot_cascade_risk.py):
  figure for direct-vs-indirect cascade diagnostics
- [`sensitivity_analysis.py`](sensitivity_analysis.py):
  matched-seed continuity-sensitivity sweeps
- [`merge_ensemble_members.py`](merge_ensemble_members.py):
  merge multiple `*_members.csv` batches into one ensemble
- [`check_consistency.py`](check_consistency.py):
  scenario preflight checks
- [`visualization.py`](visualization.py):
  Solara dashboard

## Scenario Configuration

Scenarios are usually defined through JSON parameter files such as
[`quick_test_parameters.json`](quick_test_parameters.json)
and
[`aqueduct_riverine_parameters_rcp8p5.json`](aqueduct_riverine_parameters_rcp8p5.json).

Key configuration blocks:

- `rp_files`: hazard schedule encoded as
  `RP:START_STEP:END_STEP:HAZARD_TYPE:path`
- `steps`, `start_year`, `steps_per_year`: simulation horizon
- `topology`: firm locations and supply-chain edges
- `consumption_ratios`: final-demand allocation over household-purchased sectors
  (`retail`, `wholesale`, `services`; non-final sectors are ignored with a warning)
- `adaptation`: hazard-conditional firm adaptation settings

Using `None` as the hazard `path` encodes an explicit no-hazard warm-up window,
for example:

```json
"10:1:80:FL:None"
```

This is how the shared warm-up period is represented in the main flood
experiments.

## Adaptation System

Each firm carries a continuity-capacity state, implemented in the code as
`continuity_capital` for backward compatibility. Firms update continuity targets
from hazard-conditioned signals and finance maintenance or new continuity
spending only from residual post-operations cash.

Main settings in the `adaptation` block:

- `enabled`
- `decision_interval`
- `ewma_alpha`
- `observation_radius`
- `adaptation_sensitivity_min`
- `adaptation_sensitivity_max`
- `max_adaptation_increment`
- `continuity_decay`
- `maintenance_cost_rate`
- `adaptation_strategy`
- `min_money_survival`
- `replacement_frequency`

The current manuscript focuses on `capital_hardening` and `backup_suppliers`,
but the code also includes `stockpiling` and `reserved_capacity`. In the
current implementation, `stockpiling` increases the finished-goods inventory
buffer rather than building a separate input stock.

## Outputs and Reproducibility

Single-seed and multi-seed runs write time-series outputs in the repository
root by default.

Main output types:

- `simulation_*.csv`: model-level time series for single-seed runs, or ensemble
  summaries for multi-seed runs
- `simulation_*_members.csv`: member-level aggregate trajectories for each seed
- `simulation_*_agents.csv`: optional combined agent panel when
  `--save-agent-ensemble` is enabled
- `simulation_*_ensemble.png`: quick ensemble plot from the runner

Summary and member CSVs include `Meta_*` fields that record the effective
scenario label, parameter file, topology file, hazard schedule, seed range, and
adaptation settings. This makes saved outputs self-describing and easier to
merge, re-plot, or audit later.

The model also records cascade-risk diagnostics used in the paper, including:

- `Ever_Directly_Hit_Firm_Share`
- `Never_Hit_Currently_Disrupted_Firm_Share`
- `Never_Hit_Supplier_Disruption_Burden_Share`
- `Never_Hit_Production_Share`
- `Never_Hit_Capital_Share`

## Validation and Preflight Checks

Stock-flow regression test:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache python -m pytest tests/test_stock_flow_closure.py -q
```

Scenario consistency preflight:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache python check_consistency.py --param-file aqueduct_riverine_parameters_rcp8p5.json --no-hazards --no-adaptation --steps 80 --min-tail-production 1200 --min-tail-consumption 700 --min-tail-labor 600 --min-tail-active-firms 30
```

These checks are used to verify accounting closure, tail activity, and scenario
sanity before committing to longer ensemble runs.

## Topologies and Hazard Data

Firm locations and supply chains are provided through topology JSON files such
as:

- [`riverine_firm_topology_100.json`](riverine_firm_topology_100.json)
- [`sample_firm_topology.json`](sample_firm_topology.json)

Hazard rasters live under [`data/`](data/).
Optional preprocessing utilities are in
[`prepare_hazard/`](prepare_hazard/).

Example preprocessing command:

```bash
python prepare_hazard/preprocess_geotiff.py \
  --input data/raw/*.tif \
  --model-grid \
  --resolution 0.25 \
  --resampling mean \
  --output-dir data/model_grid/
```

## Repository Layout

```text
agents.py                  Agent behavior
model.py                   Main EconomyModel
run_simulation.py          CLI runner
plot_from_csv.py           Maintained time-series plotter
plot_cascade_risk.py       Cascade-risk figure
sensitivity_analysis.py    Matched-seed sensitivity sweeps
merge_ensemble_members.py  Ensemble merge utility
check_consistency.py       Scenario preflight checks
visualization.py           Solara dashboard
prepare_hazard/            Optional raster preprocessing utilities
data/                      Hazard rasters and JRC damage workbook
manuscript/                Current paper source and submission assets
presentation/              Slides
tests/                     Regression tests
```

## License

This project is released under the terms of the
[`LICENSE`](LICENSE).
