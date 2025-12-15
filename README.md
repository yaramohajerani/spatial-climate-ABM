# Spatial Climate-Economy Agent-Based Model

[![arXiv](https://img.shields.io/badge/arXiv-2509.18633-b31b1b.svg)](https://arxiv.org/abs/2509.18633)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![NeurIPS 2025 CCAI](https://img.shields.io/badge/NeurIPS%202025-Climate%20Change%20AI-green.svg)](https://www.climatechange.ai/papers/neurips2025/19)

This is a spatial agent-based model (ABM) for simulating climate-economy interactions. The model couples economic behavior with climate hazard impacts using Mesa for agent-based modeling. As an eaxmple, we assess the impact of acute flooding using JRC flood depth-damage functions for impact assessment.

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
- **Economic Markets**: Endogenous wages, dynamic pricing, labor mobility, and supply chain networks
- **Climate Integration**: Lazy hazard sampling from full-resolution GeoTIFF rasters with JRC region-specific damage curves
- **Adaptive Behaviors**: Risk-based household migration and firm capital adjustments
- **Firm Learning System**: Evolutionary strategy-based learning for budget allocation, pricing, wage setting, and risk adaptation
- **Multiple Sectors**: Commodity, manufacturing, and retail sectors with sector-specific production coefficients and configurable consumption ratios

### Model Highlights

- **Memory-efficient hazard loading**: Samples hazard values only at agent locations using `rasterio.sample()`, avoiding loading full rasters into memory
- **Region-specific damage functions**: JRC depth-damage curves for Europe, Asia, Africa, North America, South America, and Oceania
- **No preprocessing required**: Works directly with full-resolution Aqueduct flood rasters (~90MB each)
- **Distance-soft labor and goods markets**: Households consider all firms; distance is a disutility but there is no hard radius
- **Learning System**: Evolutionary strategy learning with fitness-based replacement of weak firms

## Core Architecture

### Spatial Grid and Environment

The model uses a `mesa.space.MultiGrid` with configurable resolution (default 1° = 360×180 cells, or 0.5° = 720×360, or 0.25° = 1440×720 cells). This is decoupled from the hazard raster resolution, allowing efficient use of full-resolution global hazard data. Agents are placed based on coordinates specified in topology files.

### Agent Classes

#### HouseholdAgent
- **Labor Supply**: Each household supplies 1 unit of labor per step
- **Employment Choice**: Maximizes `wage - distance_cost × distance` utility across all firms (cross-sector employment allowed)
- **Consumption**: Allocates budget across sectors based on configurable ratios; supports fractional purchases when budget < unit price
- **Risk Behavior**: Monitors hazard within random radius (1-50 cells), relocates when max hazard > 0.1

#### FirmAgent
- **Production Technology**: Leontief production function with labor, material inputs, and capital
- **Sector-Specific Coefficients**: Each sector has different labor, input, and capital requirements per unit output:
  - Commodity: labor=0.6, input=0.0, capital=0.7 (capital-intensive extraction, no upstream inputs)
  - Manufacturing: labor=0.3, input=0.6, capital=0.6 (automated, capital & input intensive)
  - Retail: labor=0.5, input=0.4, capital=0.2 (moderate labor, low capital needs)
- **Learning System**: Evolutionary strategy learning with 6 adaptive parameters and fitness-based selection
- **Wage Setting**: Raises wages after 4 consecutive cycles of labor shortage (persistent shortage threshold prevents wage spirals)
- **Dynamic Pricing**: Supply-demand driven adjustments with cost-floor mechanism
- **Input Procurement**: Inputs from connected suppliers are substitutable (sum-based)

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
  - `commodity`, `manufacturing`, `retail` → Industrial buildings
- **Interpolation**: Linear interpolation between depth-damage points

### Firm Learning System
- **Strategy Parameters**: Budget allocation weights, risk sensitivity, price aggressiveness, wage responsiveness
- **Performance Tracking**: 10-step memory of money, production, capital, and limiting factors
- **Fitness Evaluation**: Weighted combination of money growth (40%), production consistency (30%), survival bonus (20%), resource balance (10%)
- **Population Dynamics**: Failed firms replaced by mutated offspring of successful firms

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
    "commodity": 0.25,
    "manufacturing": 0.45,
    "retail": 0.30
  },
  "learning": {
    "enabled": true,
    "memory_length": 10,
    "mutation_rate": 0.05,
    "adaptation_frequency": 5,
    "min_money_survival": 1.0,
    "replacement_frequency": 10
  }
}
```

### Data Preprocessing (Optional)

For resampling rasters to model grid resolution (optional - model works with full-resolution files):

```bash
# Resample to 0.25° model grid with max aggregation (preserves peak flood depths)
python prepare_hazard/preprocess_geotiff.py \
    --input data/raw/*.tif \
    --model-grid \
    --resolution 0.25 \
    --resampling max \
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

## Output Files

- **simulation_*.csv**: Model-level time series (production, wealth, wages, prices, risk)
- **simulation_*_agents.csv**: Agent-level panel data (money, capital, production, sector, type)
- **simulation_*_timeseries.png**: Multi-panel plots of key metrics
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
- **Consumption Ratios**: Configurable household spending by sector (default: 25% commodity, 45% manufacturing, 30% retail)
- **Wage Adjustment Threshold**: 4 consecutive cycles of labor shortage required before wage increase

### Learning Parameters
- `learning.enabled`: Enable/disable firm learning (default: true)
- `learning.memory_length`: Steps of performance history (default: 10)
- `learning.mutation_rate`: Strategy mutation standard deviation (default: 0.05)
- `learning.adaptation_frequency`: Steps between strategy evaluations (default: 5)
- `learning.min_money_survival`: Minimum money before firm failure (default: 1.0)
- `learning.replacement_frequency`: Steps between replacement cycles (default: 10)

### Climate Parameters
- **Recovery Rate**: 50% of damage factor recovered per step
- **Vulnerability**: JRC region-specific depth-damage curves
- **Sampling**: Independent per-cell Poisson process based on return periods

## File Organization

```
├── model.py              # Main EconomyModel class
├── agents.py             # HouseholdAgent and FirmAgent classes
├── run_simulation.py     # CLI runner with parameter file support
├── hazard_utils.py       # LazyHazard class for memory-efficient sampling
├── damage_functions.py   # JRC damage functions from Excel
├── trophic_utils.py      # Network topology analysis
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
