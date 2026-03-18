# Methodology Changes Log

Tracking changes to the model methodology that need to be reflected in the manuscript.

## 1. Liquidity-dependent damage recovery
- **What**: Recovery rate now scales with firm liquidity: 20% (money≈0) to 50% (money≥200) per step, instead of a fixed 50%.
- **Why**: Prevents unrealistic instant recovery for cash-strapped firms. Firms with more capital can afford faster repairs.
- **Where**: `agents.py`, FirmAgent damage recovery section.
- **Manuscript impact**: Update recovery rate description in methodology; note the liquidity-dependent mechanism in model description.

## 2. Minimum wage floor
- **What**: Wage offers cannot fall below 40% of the initial mean wage, preventing unrealistic wage collapse.
- **Why**: Proxy consistent with ILO (2016) observations that minimum wages in high-income economies typically fall between 40–60% of the median wage.
- **Where**: `agents.py`, FirmAgent wage adjustment section.
- **Manuscript impact**: Add minimum wage floor to model description; cite ILO (2016) as motivation.

## 3. Production-based fitness function
- **What**: Replaced the 4-component weighted fitness function (money growth 35%, production 25%, peak maintenance 20%, survival 20%) with a single metric: time-averaged production over the memory window.
- **Why**: The weighted sum caused firms to over-optimize for capital accumulation and liquidity at the expense of production — learning+hazard scenarios showed *lower* production than non-learning+hazard. A single production metric implicitly captures all aspects of firm health (capital, liquidity, labor, inputs must all be adequate to sustain output) and eliminates arbitrary weight choices.
- **Where**: `agents.py`, `FirmAgent._evaluate_fitness()`.
- **Manuscript impact**: Rewrite fitness function description; remove weight sensitivity analysis discussion and replace with memory window sensitivity analysis.

## 4. Quarterly time resolution (steps_per_year: 2 → 4)
- **What**: Increased temporal resolution from semi-annual to quarterly steps. Total steps doubled (150 → 300) to cover the same 75-year period. All hazard event step bounds updated accordingly.
- **Why**: Provides smoother dynamics, more granular hazard sampling, and a 10-step memory window that now corresponds to 2.5 years (appropriate for learning adaptation) rather than 5 years.
- **Where**: All `*_parameters.json` files; depreciation comment in `agents.py` now correctly reflects quarterly steps.
- **Manuscript impact**: Update temporal resolution description; note quarterly steps in methodology.

## 5. Memory window sensitivity analysis (replaces weight sensitivity)
- **What**: Sensitivity analysis now varies the `memory_length` parameter (5, 8, 10, 15, 20 steps = 1.25–5.0 years) instead of fitness weights (which no longer exist).
- **Why**: With the production-based fitness, the averaging window is the key tunable parameter. Verifies that qualitative conclusions are robust to this choice.
- **Where**: `sensitivity_analysis.py`.
- **Manuscript impact**: Replace fitness weight sensitivity results with memory window sensitivity results.

## 6. Pure-coefficient budget allocation
- **What**: Budget allocation weights now use pure Leontief technical coefficients (`LABOR_COEFF`, `INPUT_COEFF`, `CAPITAL_COEFF`) instead of coefficients scaled by nominal prices/wages.
- **Why**: The previous formula (`LABOR_COEFF × wage`, `CAPITAL_COEFF × price`) created a death spiral: when hazard-driven supply scarcity caused prices to rise faster than wages, the capital weight dominated the allocation (99% of budget to capital at late-period price/wage ratios), starving labor budgets. This caused 97% of firms to be labor-limited despite having ample total funds, driving employment below 50% and prices above 500×. Pure coefficients ensure stable allocation regardless of nominal price levels; the evolutionary strategy weights still allow learned adjustments.
- **Where**: `agents.py`, `FirmAgent.prepare_budget()`.
- **Manuscript impact**: Update budget allocation description in methodology; note that allocation is based on production technology rather than market prices. The input budget is split equally across suppliers rather than weighted by supplier price.

## 7. Bankruptcy-only firm replacement
- **What**: Removed the "persistent decline" check (`money[-1] < money[0] * 0.5`) from evolutionary firm replacement. Firms are now only replaced when actually bankrupt (`money < min_money_survival`).
- **Why**: During systemic crises (e.g., widespread flooding), all firms experience declining wealth simultaneously. The persistent decline check treated solvent-but-declining firms as failures, replacing them with $100 startups and destroying their accumulated wealth. In one instance, 7 firms averaging $1,400 each were replaced, instantly destroying ~$10,000 of system wealth and triggering an unrecoverable economic collapse. Bankruptcy-only replacement preserves the economy's money supply during downturns while still removing truly insolvent firms.
- **Where**: `model.py`, `EconomyModel._apply_evolutionary_pressure()`.
- **Manuscript impact**: Update firm replacement criteria in methodology; note that replacement is triggered only by insolvency, not by wealth decline, to avoid procyclical wealth destruction.
