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

The codebase also supports first-class external shock APIs so callers do not 
need wrapper-owned monkeypatches to model transport disruption:

- `HazardRasterEvent` for GeoTIFF-backed hazard windows
- `NodeShock` for explicit direct-damage shocks by coordinates or firm ids
- `LaneShock` for explicit supplier -> buyer transport throttles
- `RouteShock` for disruptions keyed by topology `route_dependencies`
- `build_model(...)` and `run_model(...)` for direct Python use from the repo root
  with the same core controls exposed by the CLI, including input recipes, firm
  replacement, dynamic supplier search, and transport-shock toggling

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

Run directly from Python:

```python
from api import run_model
from shock_inputs import NodeShock, RouteShock

model, results_df, agents_df = run_model(
    steps=12,
    num_households=100,
    num_firms=30,
    node_shocks=[
        NodeShock(
            label="Port outage",
            hazard_type="PORT_OUTAGE",
            intensity=0.4,
            start_step=3,
            end_step=5,
            affected_coords=[(32.5, 29.9)],
        )
    ],
    route_shocks=[
        RouteShock(
            label="Red Sea bottleneck",
            route_tag="RED_SEA",
            intensity=0.5,
            start_step=3,
            end_step=8,
        )
    ],
)
```

`NodeShock.intensity` is normalized to `[0, 1]` and mapped to a synthetic flood
pseudo-depth of `intensity × 6 m` before the upstream damage curves are applied.

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
- `raster_hazard_events`: structured equivalent of `rp_files`
- `node_shocks`: explicit direct-damage events with timing, normalized intensity
  (`intensity × 6 m` pseudo-depth), and coordinates / firm ids
- `lane_shocks`: explicit supplier -> buyer transport throttles
- `route_shocks`: route-tag transport disruptions keyed by `route_dependencies`
- `steps`, `start_year`, `steps_per_year`: simulation horizon
- `topology`: firm locations and supply-chain edges
- `damage_functions_path`, `land_boundaries_path`: optional explicit resource paths for environments where the data lives outside `upstream/data`
- `consumption_ratios`: final-demand allocation over household-purchased sectors
- `final_consumption_sectors`: sectors eligible for household final demand in
  calibrated runs; omit to use the legacy default (`retail`, `wholesale`,
  `services`)
- `sector_coefficients`: optional override for per-sector Leontief technical
  coefficients (labor, input, capital shares); generated by
  `prepare_parameters/calibrate_from_io.py` — omit to use built-in defaults
- `input_recipe_ranges`: optional override for inter-sector intermediate-input
  fractions; generated by `prepare_parameters/calibrate_from_io.py`
- `adaptation`: hazard-conditional firm adaptation settings

The CLI `--no-hazards` flag disables direct raster/node hazard impacts and
transport shocks for a true no-shock baseline. In direct Python use, pass
`apply_hazard_impacts=False` for no direct hazard damage; `apply_transport_shocks`
defaults to the same value but can be set explicitly for logistics-only shock
experiments.

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

Reproducibility is controlled by the model seed. Hazard severity sampling uses a
model-owned NumPy generator rather than global NumPy RNG state, so multiple
models can run in the same Python process without coupling their hazard draws.

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

## Calibration from Input-Output Tables

The model's sector-level technical coefficients, inter-sector supply
fractions, and household consumption ratios can be calibrated from real
national accounts data using the scripts in [`prepare_parameters/`](prepare_parameters/).
Without calibration the model uses internally hand-tuned defaults.

### What gets calibrated

| Parameter | Source | Model target |
|---|---|---|
| Labor/input/capital share per sector | Labor compensation and GOS / gross output | `sector_coefficients` in param file |
| Inter-sector supply fractions | A matrix (column-normalized IO flows) | `input_recipe_ranges` in param file |
| Household consumption ratios | Final demand vector (household column) | `consumption_ratios` in param file |
| Firms per sector in topology | Gross output shares | `generate_topology.py` input |

### Data sources

**WIOD 2016 Nov release** — 44 countries × 56 ISIC Rev.4 sectors, years 2000–2014.
Free download, no registration required:

> <https://www.rug.nl/ggdc/valuechain/wiod/wiod-2016-release>

Two table types are available; choose based on your use case:

| Table type | File pattern | Best for | Format | Labor/capital data |
|---|---|---|---|---|
| **WIOT** (World IO Table) | `WIOT{YEAR}_Nov16_ROW.xlsb` | Generic or multi-country calibration | Binary Excel; requires `pyxlsb` | Derived from WIOT VA row (60/40 split) |
| **NIOT** (National IO Table) | `{COUNTRY}_NIOT_nov16.xlsx` | Country-specific calibration | Standard Excel | From `Socio_Economic_Accounts.xlsx` |

Place downloaded files in `data/io/`. The SEA file `Socio_Economic_Accounts.xlsx`
(all countries in one workbook) is needed for accurate labor/capital coefficients
when using NIOT.

**Important units note:** NIOT files are in millions of USD at current prices; the
SEA file is in millions of national currency. The script automatically converts SEA
values to USD using the NIOT value-added row as a scaling reference. For the WIOT
(global aggregate), the SEA cannot be used across countries and the VA row is used
directly instead.

Available NIOT country codes: AUS AUT BEL BGR BRA CAN CHE CHN CYP CZE DEU DNK
ESP EST FIN FRA GBR GRC HRV HUN IDN IND IRL ITA JPN KOR LTU LUX LVA MEX MLT
NLD NOR POL PRT ROU RUS SVK SVN SWE TUR TWN USA

### Sector concordance

[`prepare_parameters/niot_concordance_default.json`](prepare_parameters/niot_concordance_default.json)
maps the 56 ISIC Rev.4 sector codes (A01, B, C10-C12, …) used by both the
WIOT and NIOT to the model's 7 sectors (`commodity`, `agriculture`,
`components`, `manufacturing`, `retail`, `wholesale`, `services`). Edit this
file to adjust the mapping for your region or research focus. Each ISIC code
must appear in exactly one model sector.

The default concordance keeps producer sectors economically literal:
consumer-goods manufacturing codes such as C10-C12, C13-C15, and C31_C32 stay
in `manufacturing` rather than being relabeled as `retail`. Household final
demand is handled separately by the calibrated `consumption_ratios` and
`final_consumption_sectors` keys. This lets households buy directly from
manufacturing, agriculture, or other mapped sectors when the IO final-demand
vector records those purchases, without changing the sector's supply-chain role.

### Step-by-step calibration workflow

**1. Calibrate parameters from IO table**

Generic/global calibration using the WIOT (recommended default):

```bash
python prepare_parameters/calibrate_from_io.py \
    --wiot-file data/io/WIOT2014_Nov16_ROW.xlsb \
    --out prepare_parameters/calibrated_parameters.json
```

Country-specific calibration using a NIOT (e.g. India):

```bash
python prepare_parameters/calibrate_from_io.py \
    --niot-file data/io/IND_NIOT_nov16.xlsx \
    --sea-file  data/io/Socio_Economic_Accounts.xlsx \
    --year 2014 \
    --out prepare_parameters/calibrated_parameters_IND.json
```

**Calibration uses IO shares as production guidance, not as a zero-inventory startup state:**

1. **Self-supply is removed** (diagonal of A matrix zeroed). IO tables contain large
   intra-sector flows (energy firms buying energy, steel mills buying steel). These are
   real in national accounts, but at the model-sector level they mostly represent
   within-sector trade rather than a distinct cascade channel. Pass `--self-supply`
   to retain them for pure IO-table analysis.

2. **Recipes are sparsified by cumulative coverage.** For each buyer sector, the
   calibration keeps the largest supplier-sector shares until they explain 90% of
   intermediate input cost, then renormalizes. This avoids treating every tiny IO
   flow as a hard Leontief requirement while preserving the dominant cascade channels.

3. **Circular economies bootstrap from calibrated inventories.** The model initializes
   firms with demand-consistent output, working capital, capital, and input inventories,
   so IO cycles do not have to be broken by deleting downstream dependencies. The
   default `--primary-sectors commodity agriculture` is a narrow resource-sector
   closure: natural-resource sectors can produce from labor, capital, and land/resource
   endowments, while services and manufacturing remain connected through IO demand.

At runtime, recipe shares guide procurement by sector while aggregate intermediate
input availability binds production. This matches the model's reduced-form treatment
of intermediate inputs as substitutable within a calibrated input bundle.

Year is auto-detected from the WIOT filename (`WIOT2014` → 2014) and defaults to
2014 for NIOT. Country is auto-detected from the NIOT filename (`IND_NIOT` → IND).

This writes an intermediate JSON file containing `sector_coefficients`,
`input_recipe_ranges`, `consumption_ratios`, and `sector_output_shares`. The
checked-in calibration keeps only the final riverine RCP8.5 run artifacts; this
intermediate file can be regenerated when the topology or run parameters need to
be rebuilt.

Optional flags:
- `--self-supply` — retain intra-sector diagonal flows (not recommended for simulation; see note above)
- `--recipe-coverage FLOAT` — keep dominant supplier links until this share of intermediate input cost is represented (default 0.90)
- `--primary-sectors SECTOR [SECTOR ...]` — explicitly declare labor/capital-only primary producers (default: `commodity agriculture`)
- `--min-recipe-share FLOAT` — ignore very small supplier shares before cumulative coverage selection (default 0.02)
- `--year INT` — override year selection
- `--country ISO3` — override country detection for NIOT

**2. Generate a calibrated firm topology**

```bash
python prepare_parameters/generate_topology.py \
    --calibrated-params prepare_parameters/calibrated_parameters.json \
    --total-firms 100 \
    --bbox -180.0 -60.0 180.0 70.0 \
    --land-shapefile data/ne_110m_admin_0_countries/ \
    --out riverine_firm_topology_100_calibrated.json \
    --seed 42
```

`--bbox` is `LON_MIN LAT_MIN LON_MAX LAT_MAX`. The example covers South/Southeast
Asia; the checked-in riverine calibration uses the broader `-180 -60 180 70`
extent shown above. The Natural Earth shapefile
(`data/ne_110m_admin_0_countries/`) is already in the repository and is used
automatically as the land mask.

Firms are allocated across sectors in proportion to the gross output shares
from the IO table. Supply edges are drawn by inverse-distance-weighted
sampling, consistent with the calibrated `input_recipe_ranges`. A post-pass
ensures every buyer firm has at least one supplier per required sector.

**3. Build a run parameter file**

```bash
python prepare_parameters/build_run_parameters.py \
    --calibrated-params prepare_parameters/calibrated_parameters.json \
    --topology riverine_firm_topology_100_calibrated.json \
    --rp-from aqueduct_riverine_parameters_rcp8p5.json \
    --out calibrated_rcp8p5_parameters.json
```

This merges the calibrated economic parameters with the retained riverine
topology and RCP8.5 hazard schedule. Common overrides:

```bash
# Change horizon, household count, seed, or strategy:
python prepare_parameters/build_run_parameters.py \
    --calibrated-params prepare_parameters/calibrated_parameters.json \
    --topology riverine_firm_topology_100_calibrated.json \
    --steps 200 --num-households 1000 --seed 7 \
    --adaptation-strategy backup_suppliers \
    --out calibrated_rcp8p5_parameters.json

# Add a hazard schedule (no-hazard warmup + flood event):
python prepare_parameters/build_run_parameters.py \
    --calibrated-params prepare_parameters/calibrated_parameters.json \
    --topology riverine_firm_topology_100_calibrated.json \
    --rp-files "10:1:80:FL:None" "10:81:200:FL:data/processed/flood.tif" \
    --steps 200 \
    --out calibrated_hazard_parameters.json

# Inherit adaptation settings from an existing parameter file:
python prepare_parameters/build_run_parameters.py \
    --calibrated-params prepare_parameters/calibrated_parameters.json \
    --topology riverine_firm_topology_100_calibrated.json \
    --adaptation-from aqueduct_riverine_parameters_rcp8p5.json \
    --out calibrated_rcp8p5_parameters.json

# Inherit a full hazard schedule from an existing parameter file
# (also inherits steps, start_year, steps_per_year, grid_resolution):
python prepare_parameters/build_run_parameters.py \
    --calibrated-params prepare_parameters/calibrated_parameters.json \
    --topology riverine_firm_topology_100_calibrated.json \
    --rp-from   aqueduct_riverine_parameters_rcp8p5.json \
    --adaptation-from aqueduct_riverine_parameters_rcp8p5.json \
    --out calibrated_rcp8p5_parameters.json
```

Defaults: 80 steps (20 years at 4 steps/year), 5× households per firm,
`start_year` inferred from the calibration metadata, `capital_hardening`
adaptation. All defaults are documented in `build_run_parameters.py --help`.

Then run as usual:

```bash
python run_simulation.py --param-file calibrated_rcp8p5_parameters.json
```

The `sector_coefficients` key overrides the model's default hand-tuned values
for that run only; existing parameter files without this key are unaffected.

### Notes on interpretation

- The IO table gives value-share coefficients (dollars of input per dollar of
  output). The model's Leontief coefficients are in physical units. These are
  consistent when prices are normalised to 1.0 at initialisation, which holds
  for the model's startup price convention.
- If `LABOR_COEFF + INPUT_COEFF > 1.0` for any sector the calibration script
  emits a warning. This usually indicates an unusual concordance mapping or
  a sector with negative gross operating surplus in the source data.
- Calibrated parameter files include `final_consumption_sectors`, so household
  final demand is not limited to the legacy built-in set
  (`retail`, `wholesale`, `services`). Direct household purchases from mapped
  manufacturing, agriculture, commodity, or components sectors are preserved in
  `consumption_ratios`.
- Firm prices follow a slow-moving normal unit-cost anchor plus a scarcity
  markup. The markup rises when finished-goods inventories are below target or
  prior-period sell-through is high, and falls when inventory is abundant.
  Cost increases pass through faster than cost decreases; cost decreases pass
  through more slowly when scarcity is high. This prevents input-constrained
  downturns from mechanically turning into deflation just because wages and
  current input costs fall in the same period, while still allowing deflation
  when weak-demand conditions persist.

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
prepare_hazard/            Raster preprocessing utilities
prepare_parameters/        IO-table calibration and topology generation
data/                      Hazard rasters and JRC damage workbook
manuscript/                Current paper source and submission assets
presentation/              Slides
tests/                     Regression tests
```

## License

This project is released under the terms of the
[`LICENSE`](LICENSE).
