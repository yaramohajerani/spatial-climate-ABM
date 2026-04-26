# ODD Protocol for the Spatial Climate-Economy ABM

This document provides a detailed Overview, Design concepts, and Details (ODD) protocol for the agent-based model implemented in this repository. It is written against the current implementation in `model.py`, `agents.py`, `run_simulation.py`, and the associated configuration and manuscript files present in the repository on April 26, 2026.

The protocol distinguishes between:

- the general framework implemented in the codebase, and
- the main illustrative application used throughout the repository and manuscript (a 100-firm, 1000-household, quarterly, riverine-flood experiment on a 0.25 degree grid).

Where behavior is optional, scenario-dependent, or currently experimental, that is stated explicitly.

## 1. Overview

### 1.1 Purpose

The purpose of the model is to simulate how acute physical climate shocks propagate through a spatially explicit economy composed of firms and households. The model is designed to capture both:

- direct physical losses at exposed firm locations, and
- indirect cascading effects transmitted through supply chains, labor markets, firm finance, prices, and household demand.

The framework is especially intended for studying the difference between direct-loss reduction and continuity preservation. To do this, it implements a hazard-conditional firm adaptation system in which firms accumulate a dedicated continuity-capital stock and deploy it through one of several strategy channels. The main reported comparison in the repository contrasts:

- no adaptation,
- capital hardening, and
- backup supplier search.

The model is not an empirically calibrated representation of a specific national economy. It is a stylized, extensible, scenario-analysis framework whose main goals are:

- to integrate geospatial hazard data and economic interaction in one simulation,
- to keep stock-flow logic explicit and inspectable,
- to support matched-seed scenario comparison and ensemble analysis, and
- to expose systemic-risk diagnostics, especially disruption borne by firms that are never directly hit.

### 1.2 Entities, State Variables, and Scales

#### 1.2.1 Entity types

The model contains three main classes of entities:

| Entity | Description | Main role |
| --- | --- | --- |
| `EconomyModel` | The Mesa model object | Holds the grid, hazard schedules, network structure, transport shocks, and data collection |
| `HouseholdAgent` | Households distributed in space | Supply one unit of labor each step, receive pooled firm payouts, and consume final goods |
| `FirmAgent` | Spatially located firms linked by directed supplier -> buyer edges | Hire labor, purchase inputs, produce goods, set wages and prices, accumulate capital, experience hazard losses, and optionally adapt |

The model also uses non-agent structures:

- a spatial grid (`mesa.space.MultiGrid`, non-toroidal),
- a firm-firm supply network,
- household-firm neighborhood sets for labor search,
- hazard-event schedules,
- optional transport-disruption edge lists,
- optional reserved-capacity contracts,
- a land mask derived from Natural Earth polygons, and
- model- and agent-level output tables collected each step.

#### 1.2.2 Spatial scale

The model is spatially explicit. Agents occupy cells on a longitude-latitude grid. Grid resolution is configurable. In the main application it is 0.25 degrees, which yields a 1440 x 720 global grid.

Important spatial features are:

- firms are placed either from a topology file containing coordinates or by random placement on land cells,
- households are placed near firms in topology-driven scenarios and otherwise placed on land and connected afterward,
- hazard intensity is sampled at agent coordinates from GeoTIFF rasters or synthetic event sources,
- household labor search uses a local radius converted from an intended 1 degree geographic neighborhood,
- firm local observation of hazard stress uses a Manhattan-distance radius on the grid, and
- households can optionally relocate to safer or more employment-accessible locations.

The working radius used for labor search is not fixed in cells. It is computed from grid resolution so that the effective geographic radius remains roughly 1 degree across resolutions. The resulting cell radius is bounded between 3 and 30 cells.

#### 1.2.3 Temporal scale

Time is discrete. One simulation step corresponds to one generic planning-production-consumption-accounting period. In the main experiments:

- `steps_per_year = 4`, so one step corresponds to one quarter,
- the main long-run experiment runs for 400 steps (100 years),
- the shared warm-up window occupies the first 80 steps (2000 Q1 to 2019 Q4), and
- hazard windows activate after the warm-up.

Some submodels rely explicitly on quarterly interpretation, especially:

- annualized continuity-risk formation, and
- the default adaptation decision interval of 4 steps.

#### 1.2.4 Household state variables

Each household maintains:

- spatial position,
- money holdings,
- sector tag,
- list of nearby firms,
- heterogeneous distance-cost parameter for labor search,
- unemployment duration counter,
- labor sold in the current step,
- current-step labor income,
- last-step and current-step dividend income,
- last-step and current-step capital-income transfers,
- last-step and current-step adaptation-income transfers,
- current-step consumption.

Households do not carry explicit housing stock, debts, or firm-specific ownership claims.

#### 1.2.5 Firm state variables

Each firm maintains:

- spatial position,
- sector,
- money holdings,
- price,
- wage offer,
- capital stock,
- damage factor (effective productive condition),
- counterfactual damage factor (used for adaptation diagnostics),
- output inventory,
- input inventories by supplier,
- connected suppliers,
- current employees,
- expected sales,
- target output,
- target labor,
- target input requirements,
- target capital stock,
- base capital target,
- base inventory target,
- required working capital,
- working-capital credit limit and usage,
- current-step production and input consumption,
- current-step wage bill, input spending, depreciation, profits, dividends, capital spending, and adaptation spending,
- sales and revenue statistics for current and previous step,
- hazard-loss diagnostics,
- supplier-disruption diagnostics,
- hazard-operating-shortfall diagnostics,
- continuity-capital stock and related adaptation state,
- firm-specific adaptation sensitivity,
- exposure-state flags,
- survival time.

Firms do not maintain explicit balance-sheet debt instruments or equity shares. Short-term finance is represented by a bounded operating overdraft tied to sales capacity. Under the startup-reset failure policy, household equity recapitalization is used only to restore failed firms to their startup cash state.

#### 1.2.6 Environment and external data

The environment includes:

- a land mask for permissible placement,
- one or more hazard layers loaded lazily from raster files,
- a sparse step-specific hazard map at agent-occupied cells,
- JRC flood depth-damage functions loaded from an Excel workbook,
- a simplified region-mapping function from lon/lat to JRC region classes.

The framework supports hazard types as labels, but in the current implementation all direct-damage calculations are processed through flood depth-damage curves. Explicit node shocks therefore work by mapping normalized intensity into a pseudo-flood depth.

### 1.3 Process Overview and Scheduling

Each model step is phased. The implemented order is important because it determines which state variables affect current versus next-period behavior.

The step sequence is:

1. Increment the model time counter and reset transport-route metrics.
2. Reset firm adaptation accounting and decay existing continuity capital.
3. If the step is an adaptation decision step for a firm, refresh its continuity target and pending adaptation increment using the hazard information accumulated from previous steps.
4. Fund any capital repair deferred from the previous period, using available firm cash above the operating reserve.
5. Sample the current hazard realization at all occupied agent cells and apply direct firm losses.
6. Each firm plans operations: output target, labor demand, input demand, liquidity buffer, working-capital ceiling, and capital target.
7. If the reserved-capacity strategy is active, create reserved supplier contracts before procurement begins.
8. Households supply labor. They search for employers, optionally relocate if that feature is enabled, and sell labor to hiring firms.
9. Firms execute production in broad sector order. Commodity firms act before manufacturing firms, which act before retail/service firms. Ties within broad sector tiers are randomized each step.
10. During firm execution, wages and prices are updated first using prior-period information, then firms procure inputs, optionally rewire existing supplier links for unavailable required input sectors, optionally access continuity-enabled backup channels, produce output, clear employees, and depreciate capital.
11. Households consume final goods after current-period production is complete.
12. Firms close the accounting period: compute profits, install productive capital when no current direct loss blocks it, fund adaptation from residual cash, pay dividends, and update adaptive expectations and exposure diagnostics.
13. Firms partially recover damage factors after the current period's production and accounting have closed, so recovery affects the next period rather than smoothing the current shock.
14. The model updates the mean wage, collects model-level and agent-level observations, and then applies the configured firm-failure policy at the global replacement interval.

Optional transport shocks are applied only around the firm execution phase by temporarily patching supplier sale functions on affected edges.

## 2. Design Concepts

### 2.1 Basic Principles

The model combines several basic principles.

#### Spatial exposure and spatial interaction

Economic agents are geolocated. Hazard exposure depends on location. Labor search and local observational learning also depend on location. This makes indirect risk spatially mediated rather than purely network-based.

#### Demand-led production with technical bottlenecks

Firms do not produce for an exogenous output path. They plan production from expected sales plus an inventory buffer and then face labor, input, capital, and finance constraints. Actual production is the minimum of planned output and technical feasibility.

#### Supply-chain propagation

Firms purchase intermediate inputs from connected suppliers. When suppliers are disrupted, downstream output can fall even if downstream firms are never directly flooded. This is the core indirect-risk mechanism.

#### Rule-based behavioral closure

Household and firm behavior is intentionally simple and transparent rather than optimized by solving intertemporal problems. Wages, prices, investment, and adaptation all follow inspectable rules tied to observed operating conditions.

#### Stock-flow consistency at the level of money

The model tracks total household plus firm money as an accounting diagnostic. When firms spend on reduced-form capital installation or adaptation services, those payments are redistributed to households so that money remains inside the closed economy. Replacement-firm startup cash is financed by drawing cash from households rather than creating money.

#### Reduced-form adaptation stock

Climate adaptation is isolated in a continuity-capital stock separate from productive capital. This makes it possible to compare alternative deployment channels while holding constant the underlying risk-perception and funding rule.

### 2.2 Emergence

The model is designed to generate the following macro- and meso-level outcomes endogenously:

- aggregate production,
- aggregate capital,
- aggregate household consumption,
- wage and price dynamics,
- firm failure, exit, and startup-reset patterns,
- direct-loss incidence,
- supplier-disruption cascades,
- adaptation uptake,
- the share of disruption borne by firms that remain never directly hit.

These outcomes are not imposed directly. They emerge from repeated interaction among hazard realizations, labor allocation, supply-chain procurement, liquidity constraints, price and wage adjustment, and adaptation decisions.

### 2.3 Adaptation

Adaptation is explicit and endogenous for firms. The model implements:

- a continuity-capital stock `C` in `[0, 1]`,
- decay of existing continuity capital each step,
- periodic updates to a target continuity level based on perceived hazard risk,
- capped investment toward that target,
- post-operations funding from residual cash rather than pre-committed operating budgets.

The main deployment channels are:

- `capital_hardening`: continuity capital attenuates direct hazard loss fractions,
- `backup_suppliers`: continuity capital enables emergency purchases from non-primary suppliers without changing the standing topology,
- `reserved_capacity`: continuity capital secures priority access to reserved supplier inventory,
- `stockpiling`: continuity capital raises the target finished-goods buffer.

The first two are the main reported strategies in the repository. The latter two are present in the code but are more experimental.

### 2.4 Objectives

Households and firms do not solve formal dynamic optimization problems, but their rules correspond to simple behavioral objectives.

Households:

- seek employment that balances wage against travel distance,
- prefer same-sector employers,
- consume out of current income and accumulated wealth,
- optionally relocate away from hazardous or low-opportunity locations.

Firms:

- try to meet expected demand while maintaining inventory buffers,
- avoid exhausting liquidity by preserving a cash reserve,
- procure inputs at lowest available price subject to actual supply,
- set wages in line with prior revenue per worker,
- set prices as cost plus a modest markup tied to sell-through,
- preserve and rebuild productive capital before distributing profits,
- build continuity capacity only when hazard signals justify it.

### 2.5 Learning

The model contains one explicit learning-like mechanism: hazard-conditional adaptive expectations for firm continuity needs.

Each firm maintains exponentially weighted moving averages of:

- expected direct loss,
- realized direct loss,
- locally observed direct loss,
- supplier disruption,
- expected hazard-induced operating shortfall,
- locally observed hazard-induced operating shortfall.

Every `decision_interval` steps, the firm computes:

- perceived continuity risk = max(own expected operating shortfall, local observed operating shortfall),
- annualized risk = perceived risk * `steps_per_year`,
- continuity target = min(1, adaptation_sensitivity * annualized risk),
- pending continuity increment = min(max_adaptation_increment, target - current continuity capital).

There is no reinforcement learning, no belief updating about probability distributions, and no generic evolutionary strategy search in the current implementation.

### 2.6 Prediction

Agents form limited, adaptive predictions rather than rational expectations.

Households do not explicitly forecast future wages or prices.

Firms predict:

- expected sales using adaptive smoothing:

```text
expected_sales_(t+1) = 0.7 * expected_sales_t + 0.3 * realized_sales_t
```

- continuity needs using smoothed hazard-shortfall indicators from prior experience and local observations.

Firms do not anticipate future hazard realizations from return-period schedules. Their adaptation is reactive to experienced or observed hazard stress, not forward-looking on the basis of scenario metadata.

### 2.7 Sensing

Households sense:

- wage offers,
- firm distance,
- firm sector,
- whether firms are nearby,
- local hazard at their own cell for relocation decisions when relocation is enabled.

Firms sense:

- their own prior sales, revenue, inventories, and liquidity,
- supplier prices and inventory availability,
- current realized direct losses,
- current hazard-induced operating shortfalls,
- nearby firms' hazard-induced direct losses and operating shortfalls within a local radius,
- whether supplier shortages appear hazard-related,
- transport disruption on affected inbound links when optional transport shocks are active.

### 2.8 Interaction

Interactions occur through several channels.

#### Labor market

Households sell one unit of labor per step to firms that still have vacancies and can pay wages from operating cash capacity.

#### Goods markets

Firms buy intermediate goods from suppliers; households buy final goods from final-demand sectors. Transactions move both money and physical inventory.

#### Income distribution

Dividends, reduced-form capital-service payments, and reduced-form adaptation-service payments are distributed evenly to all households.

#### Equity recapitalization

Under the startup-reset failure policy, reset firms can be recapitalized by proportionally drawing money from households.

#### Hazard transmission

Firms share common hazard environments when they occupy exposed cells, and they share indirect risk through input linkages.

#### Information transmission

Firms observe hazard stress in nearby firms and internalize it into continuity decisions.

#### Transport constraints

Optional lane and route shocks reduce effective capacity on specific supplier -> buyer links.

#### Topology rewiring

When enabled, dynamic supplier search allows firms to replace existing supplier counterparties in response to input bottlenecks. This is modeled as rewiring of standing relationship slots rather than edge accumulation. It is therefore distinct from continuity-capital strategies such as `backup_suppliers`, which provide emergency procurement without changing the standing network.

### 2.9 Stochasticity

Several processes are stochastic:

- hazard occurrence per step for raster events, based on return-period frequencies converted to step probabilities,
- normalized adaptation sensitivity draws for firms,
- household-specific distance-cost draws,
- household ordering in labor supply and consumption each step,
- within-tier firm ordering in production,
- dynamic supplier candidate ordering before price and distance sort,
- backup-supplier candidate ordering before price sort,
- household relocation destination choice,
- optional route-shock firing when route shocks are given a return period.

The simulation supports explicit seeding, and the repository workflow relies heavily on matched-seed comparisons across scenarios.

### 2.10 Collectives

The model includes several meaningful collectives:

- economic sectors,
- supply-chain tiers (commodity, manufacturing, retail/other final sectors),
- nearby-firm sets for households,
- nearby-firm observation neighborhoods for adaptation,
- the set of never-directly-hit firms used for cascade diagnostics,
- optional reserved-capacity buyer-supplier contract groups.

These collectives are not independent agents, but they influence behavior and observation.

### 2.11 Observation

The model collects both model-level and agent-level outputs each step.

Key model-level observations include:

- aggregate firm production, consumption, wealth, capital, profits, inventory,
- aggregate household wealth, labor supplied, consumption, and incomes,
- mean wage and mean price,
- total money and money drift,
- counts of firms limited by labor, input, capital, demand, or finance,
- average and total adaptation spending,
- continuity-capital levels and adaptation targets,
- direct-loss and supplier-disruption diagnostics,
- working-capital credit usage,
- firm replacements,
- supplier-link rewiring counts when dynamic supplier search is enabled,
- counts and shares of ever-hit and never-hit firms,
- burden-share diagnostics for never-hit firms,
- route-exposure and transport-blockage metrics when relevant.

Key agent-level observations include:

- money,
- production and consumption,
- labor sold,
- capital,
- limiting factor,
- price and wage,
- inventories,
- sales and profits,
- incomes,
- adaptation and hazard diagnostics,
- route-exposure metrics,
- exposure-state flags.

Outputs are saved as:

- model-summary CSVs,
- agent-panel CSVs,
- ensemble summary and member CSVs for multi-seed runs,
- plotting outputs generated by separate scripts,
- network-evolution PNGs and companion `*_network_evolution.json` snapshot files, produced only when `dynamic_supplier_search` is enabled (the topology is otherwise static so there is nothing to track).

The network-evolution figure plots firms at their geographic grid positions across up to six evenly-spaced time panels. Gray lines show original-topology links still present; orange lines show rewired links (links that differ from the first snapshot). Red crosses mark failed firms. The saved JSON records firm metadata, positions, and the full snapshot list. It can be replayed with `run_simulation.py --network-evolution-json ... --out ...` to regenerate the figure without rerunning the simulation.

Scenario outputs include `Meta_*` metadata fields so that saved files are self-describing.

## 3. Details

### 3.1 Initialization

Initialization proceeds in the following order.

#### 3.1.1 Scenario parsing and model construction

The model can be instantiated directly through Python or via `run_simulation.py`. Configuration may come from:

- explicit function arguments,
- command-line arguments,
- JSON parameter files.

Important configuration groups are:

- hazard events,
- number of households and firms,
- grid resolution,
- topology path,
- timing (`steps`, `start_year`, `steps_per_year`),
- adaptation parameters,
- consumption ratios,
- hazard toggle and relocation toggle,
- dynamic supplier-search toggle,
- optional node, lane, and route shocks.

#### 3.1.2 Grid and land mask

The model constructs longitude and latitude arrays from `grid_resolution`. Grid width and height default to the lengths of these arrays unless explicitly overridden.

The model then computes or loads cached land coordinates by intersecting grid-cell centroids with country polygons from Natural Earth, excluding Antarctica. If the land dataset cannot be loaded, the model falls back to allowing placement on all valid grid cells and emits a warning.

#### 3.1.3 Hazard structures

Raster hazard events are normalized into structured hazard-event objects. GeoTIFFs are loaded lazily through `LazyHazard`, which stores file metadata and samples only agent-cell coordinates during runtime. Events with `path = None` are allowed and define no-hazard windows without loading any raster.

Optional node shocks are converted into `SyntheticHazard` objects and inserted into the hazard registry. Optional lane and route shocks are resolved to supply edges after agents and topology mappings are created.

#### 3.1.4 Firm creation

If a topology JSON is provided, firms are created from that file:

- positions come from explicit grid coordinates or nearest grid cells to lon/lat,
- sector labels are read from the topology,
- initial topology identifiers are preserved,
- directed edges are read as supplier -> buyer relationships.

The topology defines the initial number of supplier relationship slots. When `dynamic_supplier_search` is enabled, firms may rewire an existing supplier edge within a required supplier sector, but they do not create additional supplier slots. If a required supplier-sector link is missing from the topology, that is treated as a topology/input issue: the model warns and the missing input category can bind production rather than being filled automatically.

If no topology is provided, firms are placed randomly on land and a random distance-based trade network is generated.

In the main illustrative topology:

- 100 firms are used,
- 30 are commodity firms,
- 45 are manufacturing firms,
- 25 are retail firms,
- the topology includes both tiered supplier links and some within-sector links.

#### 3.1.5 Household creation

When a topology exists, households are placed near firms rather than uniformly at random. Household sector tags are allocated in proportion to firm-sector counts and households are placed within the work radius of firms in the corresponding sector.

In the main illustrative topology this yields:

- 1000 households total,
- 300 commodity-sector households,
- 450 manufacturing-sector households,
- 250 retail-sector households.

Without a topology, households are still placed on land and later connected to nearby firms.

#### 3.1.6 Household-firm neighborhood links

After agents are placed, each household is given a set of nearby firms in the same sector within the work radius. If no such firm exists, the nearest same-sector firm is assigned to guarantee at least one local candidate.

#### 3.1.7 Initial firm operating state

The model bootstraps firms from final demand rather than starting them with arbitrary random inventories.

Initialization solves for expected sales by:

1. allocating one unit of final household expenditure across eligible final-good sectors using configured consumption ratios,
2. converting sector expenditure into output units using current prices,
3. distributing that output across firms in those sectors,
4. iterating upstream through the supplier network using technical input coefficients,
5. scaling the resulting economy-wide demand so implied labor demand matches available household labor.

Given the bootstrapped expected sales, each firm is seeded with:

- expected sales,
- base inventory target,
- base capital target,
- target capital stock,
- enough output inventory to match the target,
- enough capital stock to match the demand-consistent floor,
- enough cash to cover a working-capital target.

If a topology-provided capital stock is below the demand-consistent seeding floor, it is raised and recorded as metadata.

#### 3.1.8 Initial household and firm finances

Defaults before demand-consistent seeding are:

- household money = 100,
- firm money = 100,
- base mean wage = 1.0.

After firm seeding, total model money is stored as the reference value for money-drift diagnostics.

#### 3.1.9 Initial adaptation and exposure state

At initialization, each firm draws a firm-specific adaptation sensitivity uniformly from the configured range and starts with:

- continuity capital = 0,
- all EWMAs = 0,
- no prior direct-hit status,
- no prior indirect-disruption status.

#### 3.1.10 Illustrative application defaults

The main flood application in `aqueduct_riverine_parameters_rcp8p5.json` uses:

- 1000 households,
- 100 firms from `riverine_firm_topology_100.json`,
- 0.25 degree resolution,
- 400 quarterly steps,
- `start_year = 2000`,
- an 80-step explicit no-hazard warm-up,
- riverine flood rasters for 2030, 2050, and 2080 periods under RCP8.5,
- retail-only final demand,
- adaptation enabled by default, with strategy chosen by scenario.

### 3.2 Input Data

The model is driven by several input-data classes.

#### 3.2.1 Hazard rasters

Hazard rasters are given as scheduled events:

```text
return_period : start_step : end_step : hazard_type : path
```

For each active raster event:

- annual frequency is computed as `1 / return_period`,
- step probability is computed from annual frequency and `steps_per_year`,
- a Bernoulli draw determines whether that return-period event fires at a given cell in a given step,
- depth is sampled at the occupied cell if the event fires.

If multiple return periods of the same hazard type fire in the same step, the maximum depth is kept. If multiple hazard types apply, damage fractions are combined multiplicatively:

```text
combined_loss = 1 - product_k (1 - loss_k)
```

#### 3.2.2 Damage functions

Flood damage uses JRC Global Flood Depth-Damage Functions read from `data/global_flood_depth_damage_functions.xlsx`. Sector names are mapped to JRC asset classes:

- commodity and manufacturing -> industrial buildings,
- retail, wholesale, and services -> commercial buildings,
- residential -> residential buildings.

Region selection is based on a simplified lon/lat mapping to JRC regions rather than country-specific calibration.

#### 3.2.3 Firm topology

Topology JSON files supply:

- firm identifiers,
- firm coordinates,
- firm sectors,
- optional initial capital,
- directed edges,
- optional route dependency tags on firms when transport shocks are used.

The directed edges define the initial supplier relationship slots used by dynamic supplier search. The current implementation does not use a separate `max_dynamic_suppliers_per_sector` setting: if a topology omits a required supplier-sector slot, the model does not create one dynamically.

#### 3.2.4 Household demand weights

Household consumption ratios are read from configuration. Only final-demand sectors are eligible:

- retail,
- wholesale,
- services.

Weights assigned to non-final sectors are ignored with a warning.

#### 3.2.5 Optional explicit shock inputs

The framework supports three explicit non-raster shock classes:

- `NodeShock`: direct damage applied to specified coordinates or topology firm IDs,
- `LaneShock`: reduced capacity on a specific supplier -> buyer edge,
- `RouteShock`: reduced capacity on inbound supply edges associated with firms exposed to a route tag.

These are optional framework extensions. The main riverine-flood experiments primarily use raster hazards.

### 3.3 Submodels

#### 3.3.1 Household labor-supply submodel

Each household supplies one unit of labor per step.

The search rule is staged:

1. search nearby same-sector firms first,
2. if unsuccessful, search the wider firm set.

Firms are ranked by utility:

```text
U_hf = wage_offer_f
       - distance_cost_h * ManhattanDistance(h, f)
       + sector_match_bonus
       - sector_mismatch_penalty
       - remote_search_penalty_if_not_nearby
```

Implemented constants are:

- sector match bonus = 0.15,
- sector mismatch penalty = 0.20,
- remote-search penalty = 0.10,
- household-specific distance cost ~ Uniform(0.01, 0.1).

The household applies in descending utility order. A labor sale succeeds only if the chosen firm still has vacancies and can pay the wage from operating cash capacity.

#### 3.3.2 Household relocation submodel

Household relocation exists but is disabled by default in the main experiments.

If enabled:

- a household relocates away from its current cell when local hazard exceeds 0.5,
- after 3 consecutive steps without work, a household relocates near a random firm to improve employment prospects.

Hazard relocation moves households to a random land cell with hazard at or below 0.5 if such a cell exists. Job-search relocation moves households to a land cell within the work radius of a target firm.

Relocation does not currently impose direct financial moving costs even though a relocation-cost constant is defined in the household class.

#### 3.3.3 Household consumption submodel

Household consumption occurs after current-period production and uses a budget rule:

```text
consumption_budget
  = 0.9 * (current labor income
           + last-step dividends
           + last-step capital income
           + last-step adaptation income)
  + 0.02 * max(0, money - 50)
```

The budget is capped by current money holdings.

Households then:

1. allocate the budget across final-good sectors using configured sector ratios,
2. within each sector, sort firms by price ascending,
3. buy goods until the sector budget is exhausted.

Purchases can be fractional. Only firms with positive output inventory can sell.

The model therefore closes the circular flow in a minimal way:

- firms pay wages and pooled payouts to households,
- households spend most of that income back on final goods,
- firm revenue depends on both household final demand and inter-firm demand.

#### 3.3.4 Firm planning submodel

At the start of each step, each firm resets current-period flow variables and plans operations.

The planning logic is:

1. set a liquidity buffer:

```text
liquidity_buffer = max(10, 0.15 * current money)
```

2. set an inventory target:

```text
inventory_target = max(1, expected_sales * effective_inventory_buffer_ratio)
```

3. compute demand-driven desired output:

```text
demand_driven_output = max(0, expected_sales + inventory_target - current_output_inventory)
```

4. cap desired output by capital, damage, and finance constraints,
5. derive target labor and target intermediate-input quantities from technical coefficients.

The effective inventory buffer ratio is usually 0.25, but under the experimental `stockpiling` strategy it increases with continuity capital and perceived hazard risk.

Working-capital finance is bounded:

- firms preserve a cash buffer,
- payroll and intermediate-input purchases may use a sales-backed overdraft,
- the overdraft is capped by both expected variable-cost needs and a revenue anchor.

This submodel determines whether a firm enters the production phase as demand-limited, finance-limited, or technically constrained.

#### 3.3.5 Wage-setting submodel

Wages are updated at the start of the firm's execution phase using prior-period information.

If the firm had workers and positive revenue last step:

```text
target_wage = (revenue_last_step / last_hired_labor) * 0.5
```

If it had no workers:

```text
target_wage = model_mean_wage * 1.02
```

If it had workers but no revenue, it keeps its current wage.

Wages then adjust gradually:

```text
wage_offer_(t+1) = wage_offer_t + 0.1 * (target_wage - wage_offer_t)
```

There is a wage floor equal to 40 percent of the initial mean wage.

#### 3.3.6 Price-setting submodel

Prices are updated before procurement and production.

Unit cost is:

```text
unit_cost = (labor_coeff * wage_offer + input_coeff * average_input_price) / damage_factor
```

Markup depends on prior-period sell-through:

```text
markup = 0.02 + 0.30 * sell_through
target_price = unit_cost * (1 + markup)
```

Sell-through is previous-period sales divided by previous-period goods available for sale.

Prices adjust gradually:

```text
price_(t+1) = price_t + 0.2 * (target_price - price_t)
```

There is an absolute floor of 0.5.

#### 3.3.7 Intermediate-input procurement submodel

After updating wages and prices, firms procure the intermediate inputs required for planned output. Input requirements are set by a firm-specific sector recipe drawn from predetermined sector-level ranges at initialization. The recipe defines the complementary supplier-sector categories required for production, while the supply network defines which firms can provide each category. Suppliers within the same required sector are substitutes, but inventories from one required sector do not satisfy requirements for another required sector.

Key features are:

- connected suppliers in the same supplier sector are treated as substitutes,
- supplier-sector requirements are derived from the firm's input recipe, not from raw supplier edge counts,
- primary suppliers are sorted by price,
- firms buy cheapest available input first,
- purchases require real supplier inventory and real buyer cash capacity,
- input inventories are stored by supplier ID,
- if dynamic supplier search is enabled, firms with unresolved required input demand can replace one existing supplier link in the affected supplier sector,
- rewiring preserves the number of supplier links in that buyer-sector relationship set; it changes the counterparty, not the degree implied by the topology,
- replacement candidates must be active same-sector firms with available inventory or current production and are ranked by price and then distance,
- unavailable current suppliers are replaced first; otherwise, the highest-priced current supplier is replaced only if the best candidate is cheaper,
- when a topology omits a supplier for a required recipe sector, the model emits a runtime warning and the unresolved recipe input can bind production,
- if required supplier-sector inventory is unavailable, production may become input-limited.

Supplier disruption is measured as the share of desired inputs that remain unavailable when the shortage is attributed to hazard-related causes.

#### 3.3.8 Production submodel

Production uses a Leontief structure with sector-specific coefficients:

| Sector | Labor coeff | Input coeff | Capital coeff |
| --- | ---: | ---: | ---: |
| Commodity / agriculture | 0.6 | 0.0 | 0.7 |
| Manufacturing | 0.3 | 0.6 | 0.6 |
| Retail / wholesale | 0.5 | 0.4 | 0.2 |
| Services | 0.9 | 0.1 | 0.1 |

Actual output is:

```text
possible_output = min(
    target_output,
    output_from_labor * damage_factor,
    output_from_inputs * damage_factor,
    output_from_capital * damage_factor
)
```

Inputs consumed to generate realized output are scaled by the pre-damage quantity needed to support the realized post-damage output. Produced goods are added to finished-goods inventory.

Employees are cleared after production, and their number is stored as `last_hired_labor` for next-step wage targeting.

Capital depreciates at 0.2 percent per step.

#### 3.3.9 Direct-hazard damage submodel

Hazard sampling is performed only at occupied agent cells.

For each occupied cell and active event:

- a step-level Bernoulli draw determines whether the event occurs at that cell,
- if it occurs, the raster depth at that cell is sampled,
- damage fraction is interpolated from the JRC curve for the agent's sector and region.

For firms, direct damage affects:

- capital stock,
- output inventory,
- input inventories,
- the damage factor that reduces effective productivity.

Households are not directly assigned flood asset losses in the current implementation.

The model also tracks:

- raw loss fraction,
- adapted loss fraction,
- counterfactual direct loss value,
- realized direct loss value,
- direct-loss expense for accounting purposes.

When a firm experiences any raw direct loss, its `ever_directly_hit` flag becomes true.

#### 3.3.10 Damage-recovery submodel

After production, market transactions, and accounting close for the current period, firms partially recover their damage factors. The recovered damage state is therefore available to the next period's planning and production, not to the period in which the shock occurred.

Recovery is liquidity-dependent. The per-step recovery rate is:

```text
recovery_rate = 0.2 + 0.3 * min(1, liquidity_proxy / liquidity_anchor)
```

where the liquidity anchor depends on recent or expected revenue, capital target, and working-capital needs.

The model updates both:

- the actual damage factor, and
- a counterfactual damage factor that removes the adaptation benefit.

This allows the model to compare realized and counterfactual direct-loss burdens.

#### 3.3.11 Adaptation-decision submodel

At the start of each step, before the new hazard is sampled:

1. current continuity capital decays:

```text
C_t <- max(0, C_t * (1 - continuity_decay))
```

2. if the firm is on a decision step, it computes perceived hazard risk:

```text
perceived_risk = max(expected_operating_shortfall_ewma,
                     local_observed_shortfall_ewma)
```

3. the perceived risk is annualized and mapped into a target:

```text
continuity_target = min(1, adaptation_sensitivity * perceived_risk * steps_per_year)
```

4. the planned increment is capped:

```text
pending_increment = min(max_adaptation_increment,
                        max(0, continuity_target - current continuity capital))
```

The model records whether the action is:

- `dormant`,
- `adjust`,
- `hold`,
- or `reset` under the startup-reset failure policy.

#### 3.3.12 Adaptation-deployment submodels

#### Capital hardening

Under `capital_hardening`, continuity capital directly attenuates hazard loss:

```text
adapted_loss_fraction = raw_loss_fraction * (1 - continuity_capital)
```

This reduces all three direct-loss channels represented in the code:

- capital damage,
- output inventory damage,
- input inventory damage,
- and the corresponding productivity hit via the damage factor.

#### Backup suppliers

Under `backup_suppliers`, a hazard-affected firm with positive continuity capital can make emergency purchases from non-primary suppliers in the same supplier sectors. This is separate from `dynamic_supplier_search`: backup purchases do not create or rewire standing supplier edges, and dynamic supplier rewiring does not require continuity capital.

The rule is:

- identify non-primary firms in the same sectors as the primary suppliers,
- keep only firms with positive inventory,
- shuffle candidates for fairness, then sort by price,
- buy from the cheapest candidates first,
- stop when the input gap is closed, suppliers are exhausted, or the buyer's operating cash capacity is exhausted.

This is a real market transaction. Money moves to the seller and inventory moves to the buyer.

#### Reserved capacity

Under `reserved_capacity`, buyers pre-arrange access to a bounded slice of backup supplier inventory before procurement starts. The reserved quantity scales with:

- planned input needs, and
- continuity capital.

Reserved contracts are capped by:

- each supplier's reservable inventory share (`reserved_capacity_share`),
- a buyer-side contract price cap derived from primary supplier prices.

During procurement, the buyer can draw on these reserved units at the contract price.

#### Stockpiling

Under `stockpiling`, continuity capital increases the target finished-goods inventory buffer. It does not create separate input stockpiles in the current implementation.

#### 3.3.13 Adaptation-funding submodel

Adaptation is funded only after production and market transactions close, and only when the current period has no direct-loss event blocking discretionary payouts.

In ordinary no-direct-loss periods, the funding sequence is:

1. use positive post-loss net profit to restore base productive capital,
2. use some remaining profit for discretionary capital expansion,
3. if adaptation is enabled, fund continuity maintenance and new continuity investment from residual cash,
4. distribute remaining positive profit as dividends.

Maintenance spending is:

```text
maintenance_cost_rate * continuity_capital * adaptation_scale
```

New continuity investment converts money into continuity capital using the same scale factor. Spending is redistributed evenly to households as adaptation-service income.

This timing means continuity spending competes with retained earnings and dividends, not with same-period hiring and procurement.

#### 3.3.14 Profit allocation and capital-accumulation submodel

At close of step, firms compute:

```text
accounting_profit
  = revenue
    - wage_bill
    - input_spending
    - depreciation
```

Reported net profit subtracts direct-loss expense. Payout and discretionary allocation decisions use positive post-loss net profit, and any current-period direct loss blocks same-period capital repair, adaptation funding, and dividends.

When there is no current direct loss and positive post-loss net profit is available, earnings are allocated in this order:

1. productive-capital maintenance up to the base capital target,
2. discretionary expansion toward the target capital stock,
3. adaptation spending,
4. dividends.

Because there is no explicit capital-goods sector, productive-capital installation spending is redistributed to households as reduced-form capital income.

If direct hazard losses occur in the current period, capital repair is deferred. At the start of the next period, the firm uses available cash above its operating reserve to rebuild productive capital toward the base capital target before new hazards and operating decisions are realized.

#### 3.3.15 Sales, expectation updating, and exposure-state updating

At close of step, the firm updates:

- expected sales,
- prior sales and revenue variables,
- EWMAs of direct loss, local loss, supplier disruption, and operating shortfall,
- whether it has ever been indirectly disrupted before being directly hit.

Supplier disruption borne by never-hit firms is therefore observable without reconstructing the classification from the panel afterward.

#### 3.3.16 Firm failure-policy submodel

Firm failure is defined operationally by low cash:

```text
firm.money < min_money_survival
```

Firms are not replaced immediately. Instead, at a global replacement interval the `firm_replacement` parameter determines whether failed firms are reset in place or leave the active economy.

Under the default `firm_replacement = "startup_reset"`:

- all failed firms are identified,
- at most one quarter of firms are reset in a given sweep,
- the failed firm remains in place with its shell and links intact,
- startup expected sales, inventory, productive capital, price, wage, and cash targets are restored from the firm's initialization state,
- stale sales, input inventory, working-capital, damage, and adaptation state are reset,
- any cash needed to restore the startup cash target is financed using household equity transfers.

This produces an endogenous replacement lag between failure and re-entry. With quarterly steps and the default replacement frequency of 10, the realized lag ranges from effectively immediate at the next sweep to at most 10 quarters.

Under `firm_replacement = "none"`, failed firms are deactivated at the replacement sweep. They remain in the agent panel for diagnostics but no longer produce, hire, sell goods, supply inputs, or form active procurement relationships. This mode allows scenario runs in which climate risk permanently removes firms rather than immediately replacing productive capacity.

#### 3.3.17 Transport-disruption submodel

This submodel is optional.

Lane and route shocks affect specific supplier -> buyer edges by blocking a fraction of attempted sales. The model implements this by temporarily patching each affected supplier's `sell_goods_to_firm` method during the firm execution phase.

Route shocks can apply:

- to all inbound links of firms tagged with a disrupted route, or
- to only those edges whose longitudes cross a specified waypoint.

When transport shocks are active, the model records:

- attempted route sales,
- blocked route sales,
- attempted route revenue,
- blocked route revenue,
- inbound route exposure.

#### 3.3.18 Observation and export submodel

The model records outputs through Mesa's `DataCollector` and exports them via:

- `results_to_dataframe()`,
- `save_results()`,
- `run_simulation.py`,
- ensemble utilities.

Single-seed runs produce:

- a model-summary CSV,
- an agent-panel CSV.

Multi-seed runs can produce:

- member panels,
- scenario summaries,
- ensemble plots,
- merged ensemble datasets.

The repository's plotting and analysis scripts consume these files directly.

## 4. Scope Notes and Main Assumptions

The current implementation makes several intentional simplifications.

- Households are not directly assigned physical asset losses; their exposure is mainly indirect through wages, prices, profits, and optional relocation.
- Firms use a reduced-form intermediate-input structure: requirements are differentiated by firm-specific supplier-sector recipes, not by detailed product class.
- There is no explicit banking sector, government sector, or international trade block.
- Productive capital is a reduced-form stock, not a traded good produced by a capital-goods sector.
- Household ownership is pooled rather than firm-specific.
- Route and lane shocks are optional extensions, not required by the base flood model.
- Hazard type labels are generic, but direct damage is currently mapped through flood damage functions.

These simplifications are consistent with the framework's current purpose: transparent experimentation on direct and indirect acute physical risk rather than high-detail calibration of every macroeconomic institution.

## 5. Summary

In ODD terms, this repository implements a spatial, stochastic, networked household-firm ABM in which:

- climate hazards strike firms according to scheduled raster or synthetic events,
- firms propagate disruption through production, finance, and supplier networks,
- households transmit shocks through labor and final demand,
- firms can endogenously build a continuity-capital stock and deploy it through distinct resilience strategies,
- the model records both direct exposure and indirect cascade burden, including impacts on firms that remain never directly hit.

The main contribution of the framework is therefore not a single stylized macro outcome but a reproducible simulation architecture for comparing how direct-loss mitigation and continuity-preservation strategies reshape systemic physical climate risk over time.
