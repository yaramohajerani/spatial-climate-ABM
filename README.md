# Spatial Climate-Economy Agent-Based Model

This is a spatial agent-based model (ABM) for simulating climate-economy interactions. The model couples economic behavior with climate hazard impacts using Mesa for agent-based modeling and CLIMADA for climate impact assessment.

## Model Overview

The model simulates economic agents (households and firms) on a spatial grid derived from climate hazard rasters. Agents interact through labor markets, supply chains, and goods markets while adapting to climate risks through migration, wage adjustments, and capital investment decisions.

### Key Features

- **Spatial Environment**: Grid derived from GeoTIFF hazard rasters with land mask using Natural Earth country boundaries
- **Agent Types**: Households (labor suppliers) and firms (producers with Leontief technology)
- **Economic Markets**: Endogenous wages, dynamic pricing, labor mobility, and supply chain networks
- **Climate Integration**: CLIMADA-based damage functions with independent hazard sampling per grid cell
- **Adaptive Behaviors**: Risk-based household migration and firm capital adjustments
- **Firm Learning System**: Evolutionary strategy-based learning for budget allocation, pricing, wage setting, and risk adaptation
- **Multiple Sectors**: Agriculture, manufacturing, wholesale, retail, services, and commodity sectors

### Recent Model Improvements

**Firm Learning System (08/25)**:
- **Evolutionary Strategy Learning**: Firms evolve strategies for budget allocation, pricing, wage setting, and risk adaptation
- **Performance-Based Selection**: Failed firms are replaced with mutated versions of successful firms
- **Adaptive Decision Making**: 6 key parameters control firm behavior: budget weights, risk sensitivity, pricing aggressiveness, wage responsiveness
- **Fitness Evaluation**: Multi-component scoring based on money growth, production consistency, survival time, and resource balance
- **Population Dynamics**: Bankruptcy prevention through learned optimal behaviors, with up to 25% of firms replaced per step

**Enhanced Economic Realism (07/25)**:
- **Responsive Wage Dynamics**: Wages now adjust quickly to financial constraints - firms cut wages aggressively when cash-limited, preventing artificial firm failures
- **Scarcity-Based Pricing**: Prices rise naturally when production stops or inventory is low, creating realistic scarcity premiums
- **Improved Budget Allocation**: Manufacturing firms prioritize labor budget when cash-constrained, ensuring they can afford workers
- **Relaxed Price Bounds**: Removed artificial price caps (1.5x → 1000x household wealth) to allow natural economic growth over decades
- **Market-Driven Inflation**: Prices can grow organically without artificial constraints, enabling realistic multi-decade simulations

**Key Economic Behaviors**:
- High unemployment → Rapid wage cuts → Firm survival
- Low production → Higher prices → Better firm cash flow
- Cash constraints → Labor budget prioritization → Continued employment
- Learning pressure → Strategy evolution → Improved adaptation
- Market forces → Natural price/wage equilibrium → Sustainable growth

## Core Architecture

### Spatial Grid and Environment

The model uses a `mesa.space.MultiGrid` derived from input GeoTIFF raster dimensions. A 1° geographic catchment is automatically converted to grid cell units to ensure consistent distance calculations across different raster resolutions. Agents are only placed on land cells identified using Natural Earth country boundaries (excluding Antarctica).

### Agent Classes

#### HouseholdAgent
- **Labor Supply**: Each household supplies 1 unit of labor per step
- **Employment Choice**: Maximizes `wage - distance_cost × distance` utility function within same-sector firms
- **Consumption**: Spends 50% of money on goods, targeting 2-3 different trophic level ranges for variety
- **Risk Behavior**: Monitors hazard within random radius (1-50 cells), relocates when max hazard > 0.1
- **Job-Driven Migration**: Relocates closer to same-sector firms after 3 consecutive steps without work
- **Sector Specialization**: Only works for firms in the same sector, updates nearby firm list each step

#### FirmAgent
- **Production Technology**: Leontief production function with labor, material inputs, and capital
- **Technical Coefficients**: 0.5 units each of labor, inputs, and capital per unit output
- **Learning System**: Evolutionary strategy learning with 6 adaptive parameters and fitness-based selection
- **Wage Setting**: Learned wage responsiveness - cuts wages aggressively when cash-constrained, raises when labor-limited
- **Dynamic Pricing**: Learned pricing aggressiveness with supply-demand driven adjustments and scarcity premiums
- **Input Procurement**: Independent input requirements from each connected supplier (all sectors)
- **Budget Allocation**: Learned budget weights across labor, inputs, and capital based on previous limiting factor and strategy evolution
- **Capital Investment**: Purchase additional capital when capital-constrained from cheapest available sellers
- **Risk Adaptation**: Learned risk sensitivity - increase capital requirements when local hazard > 0.1, with firm-specific relaxation rates

### Economic Mechanics

#### Labor Markets
- Households choose employers within work radius based on wage-distance utility (same sector only)
- **Responsive wage dynamics**: Firms cut wages aggressively when cash-constrained, raise when labor-limited
- **Financial constraint detection**: Firms that can't afford 2+ workers trigger rapid wage cuts (20% adjustment)
- **Market-driven adjustment**: Wage changes respond to unemployment rate and firm financial health

#### Supply Chain Networks
- **Random Networks**: Distance-weighted probabilistic connections between firms
- **Custom Topology**: JSON-specified firm locations and directed supply relationships
- **Trophic Levels**: Computed network hierarchy determines production sequencing and initial inventory allocation

#### Firm Learning System
- **Learning Architecture**: Evolutionary strategy with performance tracking and fitness-based selection
- **Strategy Parameters**: 
  - Budget allocation weights (labor, inputs, capital): 0.8-1.2× base allocation
  - Risk sensitivity: 0.5-1.5× hazard response aggressiveness  
  - Price aggressiveness: 0.5-1.5× pricing adjustment magnitude
  - Wage responsiveness: 0.5-1.5× wage adjustment speed
- **Performance Tracking**: 10-step memory of money, production, capital, and limiting factors
- **Fitness Evaluation**: Weighted combination of money growth (40%), production consistency (30%), survival bonus (20%), resource balance (10%)
- **Strategy Adaptation**: Hill-climbing with mutation every 5 steps - reinforce successful changes, randomize after failures
- **Population Dynamics**: Failed firms (money < 1.0 or 50% wealth decline) replaced by mutated offspring of successful firms
- **Evolutionary Pressure**: Up to 25% of firms replaced per step after step 5, with fitness-weighted parent selection

#### Production and Trade
- **Leontief Technology**: Output limited by minimum of labor/coeff, inputs/coeff, capital/coeff
- **Damage Factor**: Climate impacts reduce productive capacity with 50% recovery per step
- **Budget Allocation**: 
  - Allocates 90% of cash across labor, inputs (per supplier), and capital
  - Learned budget weights modify base allocation (0.8-1.2× multipliers)
  - Previous limiting factor gets bonus weight scaled by learned responsiveness
  - Each connected supplier gets independent input budget allocation
- **Inventory Management**: Finished goods inventory with independent input tracking per supplier
- **Dynamic Pricing**: 
  - Learned price aggressiveness scales all adjustments (0.5-1.5× multiplier)
  - Scarcity pricing: +5% when no recent production and low inventory
  - Supply-demand: +2% when inventory < 0.5× recent production, -2% when > 3×
  - Cash-flow pricing: +3% when money < 3× wage offer
  - Affordability bounds: Gentle correction at 1000× average household wealth

### Climate Hazard System

#### Hazard Events
- **Format**: `"RP:START_STEP:END_STEP:HAZARD_TYPE:path/to/geotiff.tif"`
- **Sampling**: Independent per-cell draws each step based on return period frequencies
- **Multiple Events**: Supports overlapping hazard types and time windows
- **Intensity Combination**: Maximum depth used when multiple events affect same cell

#### Impact Calculation
- **Vulnerability Curves**: JRC flood curves for FL hazard type, linear for others
- **Damage Application**: Multiplicative combination across hazard types
- **Asset Impact**: Reduces firm capital stock, household capital, and firm inventories
- **Production Impact**: Temporary capacity reduction via damage_factor

#### Time Resolution
- Default: 4 steps per year (quarterly resolution)
- Configurable via `steps_per_year` parameter
- Hazard probabilities adjusted accordingly (p_step = p_annual / steps_per_year)

## Usage

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Parameter Files (Recommended)
Create a JSON parameter file for complex simulations:

```json
{
  "rp_files": [
    "10:1:20:FL:data/processed/flood_2030_rp10.tif",
    "10:21:40:FL:data/processed/flood_2050_rp10.tif",
    "100:1:20:FL:data/processed/flood_2030_rp100.tif"
  ],
  "steps": 70,
  "seed": 42,
  "start_year": 2020,
  "steps_per_year": 4,
  "topology": "sample_firm_topology.json",
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

#### Running Simulations

```bash
# Headless simulation
python run_simulation.py --param-file parameters.json

# Interactive dashboard
python run_simulation.py --param-file parameters.json --viz

# Direct command line (single hazard)
python run_simulation.py --rp-file "10:1:20:FL:flood_rp10.tif" --steps 50 --seed 42

# Compare baseline vs hazard scenarios
python compare_simulations.py --param-file parameters.json --out comparison.png
```

### Data Preprocessing

For large rasters, use the preprocessing utility:

```bash
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
    {"id": 1, "lon": -73.5, "lat": 40.7, "sector": "agriculture", "capital": 100.0},
    {"id": 2, "lon": -73.6, "lat": 40.8, "sector": "manufacturing", "capital": 150.0}
  ],
  "edges": [
    {"src": 1, "dst": 2}
  ]
}
```

## Output Files

### Standard Outputs
- **simulation_results.csv**: Model-level time series (production, wealth, wages, prices, risk, average fitness, firm replacements)
- **simulation_agents.csv**: Agent-level panel data (money, capital, production, sector, type, fitness, survival_time)
- **applied_events.csv**: Log of hazard events (step, event_name, event_id)
- **simulation_timeseries.png**: Multi-panel plots of key metrics including learning system dynamics
- **dashboard_timeseries.png**: Composite plots from interactive dashboard

### Comparison Analysis
- **comparison_plot.png**: Side-by-side hazard vs baseline scenarios including firm fitness and replacement dynamics
- **comparison_plot_agents.csv**: Agent-level data for both scenarios with learning metrics

## Model Parameters

### Agent Configuration
- `num_households`: Number of household agents (default: 100)
- `num_firms`: Number of firm agents (default: 20, overridden by topology file)
- `seed`: Random seed for reproducibility

### Spatial Parameters
- `work_radius`: Employment/shopping radius in grid cells (auto-computed from 1° geographic)
- Grid dimensions automatically derived from hazard raster extent

### Economic Parameters
- **Technical Coefficients**: Labor (0.5), Input (0.5), Capital (0.5) per unit output
- **Depreciation Rate**: 0.2% per step (≈0.8% annually for quarterly steps)
- **Price Adjustment**: Supply-demand driven with scarcity premiums and upper bounds (1000× household wealth ceiling)
- **Wage Adjustment**: Responsive to financial constraints with rapid adjustment (20%) when cash-limited
- **Budget Allocation**: All firms allocate 90% of cash with previous limiting factor getting 30% bonus weight

### Risk Parameters
- **Household Migration**: Threshold 0.1, monitoring radius 1-50 cells
- **Firm Adaptation**: Capital requirement increase 20%, relaxation 20-50% per year (modulated by learned risk sensitivity)
- **Relocation Cost**: 10% of wealth and capital lost during migration

### Learning Parameters
- **Learning System**: Enabled/disabled via `learning.enabled` (default: true)
- **Memory Length**: Steps of performance history tracked (default: 10)
- **Mutation Rate**: Standard deviation for strategy mutations (default: 0.05)
- **Adaptation Frequency**: Steps between strategy evaluations (default: 5)  
- **Survival Threshold**: Minimum money level before firm failure (default: 1.0)
- **Replacement Frequency**: Steps between evolutionary replacement cycles (default: 10)

### Climate Parameters
- **Recovery Rate**: 50% of damage factor recovered per step
- **Vulnerability**: JRC depth-damage curves for floods, linear for other hazards
- **Sampling**: Independent per-cell Poisson process based on return periods

## File Organization

```
├── model.py              # Main EconomyModel class
├── agents.py             # HouseholdAgent and FirmAgent classes
├── run_simulation.py     # CLI runner with parameter file support
├── compare_simulations.py # Baseline vs hazard comparison utility
├── visualization.py      # Solara interactive dashboard
├── hazard_utils.py       # GeoTIFF processing and CLIMADA integration
├── trophic_utils.py      # Network topology analysis
├── prepare_hazard/       # Data preprocessing utilities
├── data/                 # Raw and processed hazard rasters
├── frames/               # Animation frames from dashboard
└── *_parameters.json     # Parameter files for different scenarios
```

## Development

### Key Dependencies
- Mesa ≥3.2.0 (agent-based modeling framework)
- CLIMADA ≥4.1.0 (climate impact assessment)
- Solara (interactive dashboard)
- Rasterio, GDAL (GeoTIFF processing)
- NumPy, Pandas, Matplotlib (data processing and visualization)

### Model Extension Points
- **New Agent Types**: Inherit from `mesa.Agent`, implement `step()` method
- **Additional Hazards**: Add vulnerability curves in `_build_vulnerability()`
- **Economic Mechanisms**: Modify production functions, market clearing in agent `step()` methods
- **Spatial Features**: Extend grid properties, distance calculations in `EconomyModel`

### Testing and Validation
- Use consistent `seed` values for reproducible results
- Monitor key invariants: wealth conservation, market clearing
- Compare outcomes across parameter variations
- Validate against empirical data where available

### Performance Considerations
- Grid size scales quadratically with memory usage
- Agent count affects per-step computation time
- Hazard raster resolution impacts both memory and disk I/O
- Use preprocessing for large datasets to optimize performance
