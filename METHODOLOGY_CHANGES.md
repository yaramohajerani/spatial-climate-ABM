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

## 8. Revenue-based wage targeting (replaces shortage-signal heuristics)
- **What**: Replaced the ad-hoc wage adjustment mechanism (labor shortage cycle counting, unemployment rate checks, ±5% cap, wage ceiling) with revenue-based wage targeting. Firms now set wages as a fraction of revenue per worker: `target_wage = revenue_per_worker × labor_share`, where `labor_share = 0.5 × wage_responsiveness`. Wages adjust smoothly (10% toward target per step).
- **Why**: The previous mechanism counted consecutive steps of unmet labor demand and raised wages after 4 such cycles. In a Leontief economy with scarce labor, nearly all firms are perpetually labor-limited (the Leontief bottleneck, not genuine market failure), so the shortage signal was universally triggered. This caused continuous upward wage ratcheting. The learning system's `wage_responsiveness` parameter (range 0.1–3.0) amplified this: firms offering higher wages attracted more workers → higher fitness → evolutionary selection for maximum wage growth. Learning-case wages reached 3× the no-learning level (6.32 vs 2.3), making firms fragile to production shocks and triggering catastrophic collapse. Revenue-based targeting is self-correcting: wages are structurally bounded by what workers produce, and automatically decrease when revenue falls (e.g., during hazard events).
- **What was removed**: `labor_shortage_cycles` tracking, 4-step shortage threshold, unemployment rate checks in wage logic, ±5% adjustment cap, hard wage ceiling of 10.0. The `labor_demand` computation (how many workers the firm could productively use) was also removed as it is no longer needed.
- **What was kept**: Minimum wage floor at 40% of initial wage (ILO 2016 justified). The `wage_responsiveness` learning parameter now controls an economically meaningful quantity (labor share of revenue) rather than amplifying an arbitrary adjustment rate.
- **Where**: `agents.py`, `FirmAgent.step()` wage adjustment section; removed `labor_shortage_cycles` and `labor_demand` from `FirmAgent.__init__()` and production section.
- **Manuscript impact**: Replace wage adjustment description with revenue-based targeting mechanism. Remove references to shortage thresholds and adjustment caps. Note that `wage_responsiveness` now controls labor share. The economic justification is standard labor economics: in competitive markets, wages converge to the marginal revenue product of labor.

## 9. Markup pricing (replaces inventory-threshold pricing with cost-floor ratchet)
- **What**: Replaced the ad-hoc pricing mechanism (5-band inventory-ratio thresholds, hard cost floor, cost-floor ratchet, no-sales penalty, max-price clamp) with markup pricing. Price = `unit_cost × (1 + markup)`, where markup is determined by the sell-through rate (fraction of available goods sold). Sell-through of 1.0 → 50% markup; 0.5 → break-even; 0.0 → −50% markup (below-cost clearance). Prices adjust smoothly (20% toward target per step) with an absolute floor of 0.5.
- **Why**: The previous mechanism used a hard cost floor (`price ≥ variable_cost × 1.2`) that created a one-way ratchet via the supply chain: when hazard events raised upstream prices, downstream cost floors rose permanently, even after supply recovered. Prices could increase freely but never fall below the (ever-rising) cost floor. This caused prices to diverge from wages by orders of magnitude (prices reached 1000+ while wages reached ~15), eroding household purchasing power to zero. The new mechanism allows below-cost pricing when demand is weak, breaking the ratchet. Prices track costs bidirectionally: when upstream costs fall, downstream prices follow. The liquidity feedback provides automatic production correction — selling below cost drains firm cash → smaller budget → fewer workers → lower production → inventory stabilises → markup recovers.
- **What was removed**: 5-band inventory-ratio thresholds (0.5/0.8/1.5/2.0), target inventory calculation, hard cost floor and 20% margin, no-sales penalty override, max reasonable price clamp using avg household money, `no_sales_streak` and `sales_prev_step` tracking variables.
- **What was kept**: Absolute price floor of 0.5 (prevents degenerate zero/negative prices). Unit cost computation (wage × LABOR_COEFF + input_price × INPUT_COEFF) serves as the cost anchor, not a floor.
- **Where**: `agents.py`, `FirmAgent.step()` dynamic pricing section. Removed `no_sales_streak` and `sales_prev_step` from `FirmAgent.__init__()` and end-of-step bookkeeping.
- **Manuscript impact**: Replace pricing description with markup pricing mechanism. Note that prices are anchored to unit costs with demand-responsive margins. The sell-through → markup mapping is the single pricing rule; no inventory thresholds or cost floors. Below-cost pricing during weak demand is realistic (clearance/loss-leader behaviour) and breaks the supply-chain price ratchet.
