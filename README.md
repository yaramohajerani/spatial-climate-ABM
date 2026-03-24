# Spatial Climate-Economy Agent-Based Model

[![arXiv](https://img.shields.io/badge/arXiv-2509.18633-b31b1b.svg)](https://arxiv.org/abs/2509.18633)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![NeurIPS 2025 CCAI](https://img.shields.io/badge/NeurIPS%202025-Climate%20Change%20AI-green.svg)](https://www.climatechange.ai/papers/neurips2025/19)

This is a spatial agent-based model (ABM) for simulating climate-economy interactions. The model couples economic behavior with climate hazard impacts using Mesa for agent-based modeling. As an example, we assess the impact of acute flooding using JRC flood depth-damage functions for impact assessment.

## Citation

If you use this model in your research, please cite:

> Mohajerani, Y. (2025). *Adaptive Learning in Spatial Agent-Based Models for Climate Risk Assessment: A Geospatial Framework with Evolutionary Economic Agents*. arXiv preprint arXiv:2509.18633. https://doi.org/10.48550/arXiv.2509.18633

This work was accepted to the [NeurIPS 2025 Tackling Climate Change with Machine Learning](https://www.climatechange.ai/papers/neurips2025/19).

```bibtex
@article{mohajerani2025adaptive,
  title={Adaptive Learning in Spatial Agent-Based Models for Climate Risk Assessment: A Geospatial Framework with Evolutionary Economic Agents},
  author={Mohajerani, Yara},
  journal={arXiv preprint arXiv:2509.18633},
  year={2025},
  doi={10.48550/arXiv.2509.18633}
}
```

## Model Overview

The model simulates economic agents (households and firms) on a spatial grid while exposing them to climate hazards sampled from GeoTIFF rasters. Agents interact through labor markets, supply chains, and goods markets while adapting to climate risks through migration, wage adjustments, and capital investment decisions.

### Key Features

- **Spatial Environment**: Configurable resolution global grid (default 1°, supports 0.5°, 0.25°) with agents placed based on topology files
- **Agent Types**: Households (labor suppliers) and firms (producers with Leontief technology)
- **Economic Markets**: Endogenous wages, dynamic pricing, labor mobility, demand-driven production, and supply chain networks
- **Climate Integration**: Lazy hazard sampling from full-resolution GeoTIFF rasters with JRC region-specific damage curves
- **Adaptive Behaviors**: Optional household relocation plus hazard-conditional firm resilience investment with local observation and firm-level learning
- **Liquidity-Dependent Recovery**: Post-disaster recovery speed scales with firm liquidity (firms with more capital can afford faster repairs)
- **Minimum Wage Floor**: Wage offers bounded below at 40% of initial wage, a proxy consistent with ILO (2016) observations on minimum wages in high-income economies
- **Firm Adaptation System**: Hazard-conditional resilience capital with neighborhood hazard observation and a firm-level contextual bandit over discrete adaptation investments
- **Multiple Sectors**: Commodity, manufacturing, and retail sectors with sector-specific production coefficients and configurable household final-demand ratios over final-good sectors
- **Circular-Flow Closure**: Households receive wages plus firm payouts, and total money is tracked explicitly so the household-firm system remains stock-flow closed in no-entry/no-exit runs

### Model Highlights

- **Memory-efficient hazard loading**: Samples hazard values only at agent locations using `rasterio.sample()`, avoiding loading full rasters into memory
- **Region-specific damage functions**: JRC depth-damage curves for Europe, Asia, Africa, North America, South America, and Oceania
- **No preprocessing required**: Works directly with full-resolution Aqueduct flood rasters (~90MB each)
- **Sector-local labor market**: Households search nearby same-sector firms first, then fall back to the broader market with a distance penalty
- **Final-goods market discipline**: Households buy only from final-good sectors; upstream sectors sell to firms rather than directly to households
- **Adaptation System**: Hazard-conditional resilience learning with end-of-period adaptation funding, avoided-loss rewards, and stock-flow-consistent firm reorganization

## Core Architecture

### Spatial Grid and Environment

The model uses a `mesa.space.MultiGrid` with configurable resolution (default 1° = 360×180 cells, or 0.5° = 720×360, or 0.25° = 1440×720 cells). This is decoupled from the hazard raster resolution, allowing efficient use of full-resolution global hazard data. Agents are placed based on coordinates specified in topology files.

### Agent Classes

#### HouseholdAgent
- **Labor Supply**: Each household supplies 1 unit of labor per step
- **Employment Choice**: Uses staged search: nearby same-sector firms are tried first, then the wider market is searched with wage-distance utility and a remote-search penalty (cross-sector fallback remains allowed)
- **Consumption**: Uses a disposable-income rule based on current labor income, recently distributed firm payouts, recent adaptation-service income, and a small draw from money holdings above a target cash buffer; allocates that budget across final-good sectors with fractional purchases when budget < unit price
- **Relocation**: Optional and disabled in the main scenario parameter files. The current relocation logic can still be toggled on for experiments, but it is not used in the reported core scenarios.

#### FirmAgent
- **Production Technology**: Leontief production function with labor, material inputs, and capital
- **Sector-Specific Coefficients**: Each sector has different labor, input, and capital requirements per unit output:
  - Commodity: labor=0.6, input=0.0, capital=0.7 (capital-intensive extraction, no upstream inputs)
  - Manufacturing: labor=0.3, input=0.6, capital=0.6 (automated, capital & input intensive)
  - Retail: labor=0.5, input=0.4, capital=0.2 (moderate labor, low capital needs)
- **Adaptation System**: A resilience-capital stock, firm-level hazard beliefs, neighborhood loss observation, firm-level UCB policy learning, end-of-period adaptation funding from residual cash, and bankruptcy reorganization with parent-state inheritance
- **Wage Setting**: Revenue-based wage targeting — wages track revenue per worker × a fixed labor share of 0.5, with smooth adjustment (10% toward target per step); minimum wage floor at 40% of initial wage
- **Dynamic Pricing**: Markup pricing — price = unit cost × (1 + markup), where markup is set by sell-through rate; prices track costs bidirectionally with no cost-floor ratchet
- **Damage Recovery**: Liquidity-dependent recovery rate (20%–50% per step) so stressed firms recover more slowly
- **Input Procurement**: Inputs from connected suppliers are treated as substitutable units and are purchased from the cheapest available suppliers first
- **Production Planning**: Firms plan output from expected sales plus inventory buffers, hire only up to planned vacancies, seed startup inventories/capacity from labour-scaled final demand, expand capital only from residual cash above a working-capital buffer, and operate in phased within-step order (labour, production, then household consumption)
- **Profit Distribution**: Positive profits above the operating buffer are split between retained-earnings capital expansion and household payouts; because there is no explicit capital-goods sector, investment spending is recycled to households as reduced-form investment income to preserve monetary closure
- **Adaptation-State Reorganization**: Failed firms are reorganized in place under inherited same-sector adaptation state and, if needed, recapitalized by household equity contributions; replacements do not mint new firm cash, capital, or inventories

### Climate Hazard System

#### Lazy Hazard Loading
The model uses memory-efficient lazy loading for hazard data:
- **No full raster loading**: Only reads pixel values at agent locations (~100 agents vs 933M pixels)
- **Direct GeoTIFF access**: Uses `rasterio.sample()` for on-demand sampling
- **Memory usage**: <1MB vs ~4GB for loading full global rasters
- **Performance**: ~6ms sampling time vs ~4.5s for full load

#### Hazard Events
- **Format**: `"RP:START_STEP:END_STEP:HAZARD_TYPE:path/to/geotiff.tif"`
- **Return Periods**: RP2 (50% annual), RP10 (10% annual), RP100 (1% annual)
- **Sampling**: Independent per-cell draws each step based on return period frequencies
- **Intensity Combination**: Maximum depth used when multiple events affect same cell

#### JRC Damage Functions
Damage is calculated using JRC Global Flood Depth-Damage Functions:
- **Source**: `data/global_flood_depth_damage_functions.xlsx`
- **Region detection**: Automatic based on agent coordinates (Europe, Asia, Africa, etc.)
- **Sector mapping**:
  - `residential` → Residential buildings
  - `commodity`, `manufacturing` → Industrial buildings
  - `retail` → Commercial buildings
- **Interpolation**: Linear interpolation between depth-damage points

### Firm Adaptation System
- **Adaptation Stock**: Each firm carries a `resilience_capital` stock in `[0, 1]` that attenuates direct hazard losses and speeds recovery
- **Hazard Context**: Firms maintain EWMAs of expected direct loss, realized direct loss, nearby observed direct loss, supplier disruption, and current resilience stock
- **Neighborhood Observation**: Firms observe flood-related losses among nearby firms within a configurable radius, allowing anticipatory adaptation before own exposure
- **Action Set**: Every 4 steps, firms choose among `maintain`, `small`, and `large` resilience investments
- **Funding Timing**: Adaptation actions are chosen before the hazard is sampled, but maintenance and new resilience spending are funded only at period close from residual post-operations cash; newly installed resilience capital affects the next period rather than the current one
- **Firm-Level Policy Learning**: Each firm maintains its own tabular contextual-UCB rule over the discretized hazard state; nearby observations affect the state, but action values are updated from the firm's own informative hazard windows
- **Reward Signal**: Completed adaptation windows are updated with avoided direct loss minus adaptation cost, normalized by the firm's own direct loss over the window
- **Population Dynamics**: Bankrupt firms are reorganized in place, preserving stock-flow closure while inheriting adaptation state and firm-level bandit memory from successful same-sector parents

## Usage

### Running Simulations

Use a JSON parameter file:

```bash
# Run with full-resolution hazard files (no preprocessing needed!)
python run_simulation.py --param-file gfdl_rcp45_parameters.json

# Or with preprocessed files
python run_simulation.py --param-file aqueduct_riverine_parameters.json
```

### Parameter File Format

```json
{
  "num_households": 650,
  "grid_resolution": 0.25,
  "rp_files": [
    "2:1:80:FL:data/inunriver_rcp4p5_0000GFDL-ESM2M_2030_rp00002.tif",
    "10:1:80:FL:data/inunriver_rcp4p5_0000GFDL-ESM2M_2030_rp00010.tif",
    "100:1:80:FL:data/inunriver_rcp4p5_0000GFDL-ESM2M_2030_rp00100.tif"
  ],
  "steps": 320,
  "seed": 42,
  "topology": "small_firm_topology.json",
  "start_year": 2020,
  "steps_per_year": 4,
  "consumption_ratios": {
    "retail": 1.0
  },
  "adaptation": {
    "enabled": true,
    "decision_interval": 4,
    "reward_window": 4,
    "ewma_alpha": 0.2,
    "ucb_c": 1.0,
    "observation_radius": 4,
    "action_increments": [0.0, 0.05, 0.1],
    "resilience_decay": 0.01,
    "maintenance_cost_rate": 0.005,
    "loss_reduction_max": 0.6,
    "min_money_survival": 1.0,
    "replacement_frequency": 10
  }
}
```

### Sensitivity Analysis

Run a UCB exploration sweep to verify robustness of the hazard-conditional adaptation results:

```bash
# Full sensitivity analysis
python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json

# Quick test (50 steps)
python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --quick
```

This runs the hazard+adaptation scenario across four UCB exploration strengths (`c = 0.25, 0.5, 1.0, 2.0`) and produces a comparison plot and summary CSV table.

### Data Preprocessing (Optional)

For resampling rasters to model grid resolution (optional - model works with full-resolution files):

```bash
# Resample to 0.25° model grid with mean aggregation (represents cell-average flood depth)
python prepare_hazard/preprocess_geotiff.py \
    --input data/raw/*.tif \
    --model-grid \
    --resolution 0.25 \
    --resampling mean \
    --output-dir data/model_grid/

# Or crop and scale for testing
python prepare_hazard/preprocess_geotiff.py \
    --input raw/*.tif \
    --crop-bounds -74 40 -73 42 \
    --scale-factor 0.5 \
    --output-dir data/processed
```

### Custom Network Topology

Specify firm locations and supply chains via JSON:

```json
{
  "firms": [
    {"id": 1, "lon": 106.7, "lat": 10.8, "sector": "commodity", "capital": 5.0},
    {"id": 2, "lon": 100.5, "lat": 13.7, "sector": "manufacturing", "capital": 3.0},
    {"id": 3, "lon": 101.2, "lat": 14.1, "sector": "retail", "capital": 1.5}
  ],
  "edges": [
    {"src": 1, "dst": 2},
    {"src": 2, "dst": 3}
  ]
}
```

Include at least one final-good sector (`retail`, `wholesale`, or `services`) in a custom topology. Households only buy from those sectors; topologies with only upstream sectors will have no final demand. Optional topology `capital` values act as a floor on seeded installed capacity, but the startup operating state is then rescaled from labour-consistent expected sales, so these values are not the sole determinant of initial production scale.

## Output Files

- **simulation_*.csv**: Model-level time series (production, wealth, wages, prices, risk, bottleneck counts, and adaptation diagnostics)
- **Stock-flow diagnostics in `simulation_*.csv`**: `Total_Money`, `Money_Drift`, `Firm_Dividends_Paid`, `Firm_Investment_Spending`, `Household_Labor_Income`, `Household_Dividend_Income`, `Household_Capital_Income`, `Household_Adaptation_Income`, and adaptation state summaries such as `Average_Local_Observed_Loss` and `Adaptation_Updates`
- **simulation_*_agents.csv**: Agent-level panel data (money, capital, production, sector, type, seller-sector demand, and firm-level adaptation states including `resilience_capital`, `local_observed_loss`, `adaptation_action`, and `adaptation_reward`). Household `sector` values in this file are initialization/placement cohort tags, not purchased-good categories.
- **simulation_*_timeseries.png**: Multi-panel plots of key metrics. Household consumption panels use actual household purchases, and any sector breakdown of final demand is derived from seller sectors rather than household cohort labels.
- **simulation_*_sector_bottlenecks.png**: Sector-level bottleneck analysis

## Model Parameters

### Agent Configuration
- `num_households`: Number of household agents (default: 100, configurable via parameter file)
- `num_firms`: Number of firm agents (overridden by topology file)
- `grid_resolution`: Spatial resolution in degrees (default: 1.0, options: 0.5, 0.25)
- `seed`: Random seed for reproducibility

### Economic Parameters
- **Sector-Specific Technical Coefficients**:
  | Sector | Labor | Input | Capital | Notes |
  |--------|-------|-------|---------|-------|
  | Commodity | 0.6 | 0.0 | 0.7 | Capital-intensive extraction, no upstream inputs |
  | Manufacturing | 0.3 | 0.6 | 0.6 | Automated, capital & input intensive |
  | Retail | 0.5 | 0.4 | 0.2 | Moderate labor, low capital needs |
- **Depreciation Rate**: 0.2% per step
- **Consumption Ratios**: Configurable household spending by final-good sector only. For the current 100-firm riverine topology the relevant setting is `{"retail": 1.0}`.
- **Household Consumption Rule**: `0.9 × disposable_income + 0.02 × max(0, money - 50)`, where disposable income is current labor income plus the previous period's dividends, reduced-form investment income, and adaptation-service income
- **Wage Mechanism**: Revenue-based targeting — firms set wages at `revenue_per_worker × labor_share`, with a fixed labour-share parameter of `0.5` and 10% smooth adjustment per step. Wages are structurally bounded by revenue and self-correct during downturns.
- **Minimum Wage Floor**: 40% of initial mean wage, a proxy consistent with ILO (2016) observations (40–60% of median)
- **Production Planning**: Firms target expected sales plus an inventory buffer, translate that into vacancies and input demand, preserve a working-capital buffer, initialize inventories and capital from labour-consistent demand, clear household demand after the current period's production is complete, and close the period by splitting positive profits between dividends and internally financed capital expansion subject to liquidity constraints

## Validation

- **Stock-flow closure regression**: `PYTHONPYCACHEPREFIX=/tmp/codex-pycache /Users/yaramohajerani/mamba/envs/climada_env/bin/python -m pytest tests/test_stock_flow_closure.py -q`
- **What it checks**: Total money stays constant to floating-point tolerance in a closed no-hazard economy, household-side payout income matches firm-side payout spending, the fixed labor-share wage rule behaves as expected, and the small sample economy remains economically active over the last 10 steps rather than collapsing to zero output.
- **Scenario preflight check**: `PYTHONPYCACHEPREFIX=/tmp/codex-pycache /Users/yaramohajerani/mamba/envs/climada_env/bin/python check_consistency.py --param-file aqueduct_riverine_parameters_rcp8p5.json --no-hazards --no-adaptation --steps 80 --min-tail-production 1200 --min-tail-consumption 700 --min-tail-labor 600 --min-tail-active-firms 30`
- **What it checks**: Money drift, household-firm payout mismatches, replacements, flooding counts, tail activity levels, and active-versus-dormant firm counts on the full configured topology before you commit to a long scenario batch.

### Adaptation Parameters
- `adaptation.enabled`: Enable/disable hazard-conditional firm adaptation (default: true)
- `adaptation.decision_interval`: Steps between bandit decisions (default: 4)
- `adaptation.reward_window`: Steps used to evaluate each action (default: 4)
- `adaptation.ewma_alpha`: Smoothing parameter for hazard-state expectations (default: 0.2)
- `adaptation.ucb_c`: Firm-level contextual-bandit exploration coefficient (default: 1.0)
- `adaptation.observation_radius`: Manhattan-radius threshold for observing nearby firm losses (default: 4 grid cells)
- `adaptation.action_increments`: Resilience-capital increments for maintain/small/large actions (default: `[0.0, 0.05, 0.1]`)
- `adaptation.resilience_decay`: Per-step depreciation of resilience capital (default: 0.01)
- `adaptation.maintenance_cost_rate`: Per-step carrying cost on resilience capital (default: 0.005)
- `adaptation.loss_reduction_max`: Maximum fraction of direct loss that resilience can attenuate at full stock (default: 0.6)
- `adaptation.min_money_survival`: Minimum money before firm failure (default: 1.0)
- `adaptation.replacement_frequency`: Steps between replacement cycles (default: 10)

### Climate Parameters
- **Recovery Rate**: Liquidity-dependent; 20% (money≈0) to 50% (money≥200) of remaining damage recovered per step
- **Vulnerability**: JRC region-specific depth-damage curves
- **Sampling**: Independent per-cell Poisson process based on return periods
- **Household Relocation**: Optional but disabled by default in current parameter files and CLI defaults. Hazard transmission in the core scenarios runs through firms, prices, wages, and the supply chain rather than through household migration.

## File Organization

```
├── model.py              # Main EconomyModel class
├── agents.py             # HouseholdAgent and FirmAgent classes
├── run_simulation.py     # CLI runner with parameter file support
├── sensitivity_analysis.py # UCB exploration sensitivity analysis
├── hazard_utils.py       # LazyHazard class for memory-efficient sampling
├── damage_functions.py   # JRC damage functions from Excel
├── trophic_utils.py      # Network topology analysis utilities (not used in core runtime logic)
├── visualization.py      # Solara interactive dashboard
├── prepare_hazard/       # Data preprocessing utilities (optional)
├── data/                 # Hazard rasters and damage function Excel
│   └── global_flood_depth_damage_functions.xlsx
└── *_parameters.json     # Parameter files for different scenarios
```

## Dependencies

### Core Requirements
- Mesa ≥3.2.0 (agent-based modeling framework)
- NumPy, Pandas (data processing)
- Rasterio (GeoTIFF reading)
- GeoPandas (spatial data)
- openpyxl (Excel reading for damage functions)
- Matplotlib (visualization)

### Optional
- Solara (interactive dashboard)

### Not Required
- CLIMADA is no longer needed - damage functions are loaded directly from the JRC Excel file

## Performance

| Aspect | Value |
|--------|-------|
| Hazard loading | ~1ms (metadata only) |
| Hazard sampling | ~6ms per step (90 agents) |
| Memory for hazards | <1MB |
| Agent grid | Configurable: 64,800 (1°), 259,200 (0.5°), or 1,036,800 (0.25°) cells |
| Full raster support | 933M pixels (no preprocessing needed) |

## Development

### Model Extension Points
- **New Agent Types**: Inherit from `mesa.Agent`, implement `step()` method
- **Additional Hazards**: Add sector mappings in `damage_functions.py`
- **Economic Mechanisms**: Modify production functions in agent `step()` methods
- **New Regions**: Update `get_region_from_coords()` in `damage_functions.py`

### Testing and Validation
- Use consistent `seed` values for reproducible results
- Monitor key invariants: wealth conservation, market clearing
- Compare outcomes across parameter variations
