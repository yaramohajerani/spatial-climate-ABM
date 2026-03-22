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

## 10. Mean hazard sampling (replaces max sampling)
- **What**: Hazard rasters are now resampled to the model grid using **mean** aggregation instead of **max**. The preprocessed GeoTIFF filenames reflect this (`*-mean.tif` instead of the previous convention).
- **Why**: Max sampling produced overly extreme flood depths at the model grid scale (0.25°), since it selected the single worst pixel within each ~25km cell. Mean sampling better represents the average hazard exposure across the cell, producing more realistic damage levels and more stable economic dynamics.
- **Where**: `prepare_hazard/preprocess_geotiff.py` resampling step; all `*_parameters.json` files updated to point to mean-sampled rasters.
- **Manuscript impact**: Update hazard preprocessing description to note mean (not max) aggregation. Justify as representative average exposure rather than worst-case pixel.

## 11. Expanded firm network (65 → 100 firms)
- **What**: The firm topology was expanded from 65 to 100 firms (`riverine_firm_topology_100.json`).
- **Why**: A larger network provides better statistical coverage of the supply chain and more robust emergent dynamics. 100 firms with 1000 households gives a 10:1 household-to-firm ratio.
- **Where**: `riverine_firm_topology_100.json`; `aqueduct_riverine_parameters_rcp8p5.json` updated to reference the new topology.
- **Manuscript impact**: Update network size in model description. Note 100 firms and 1000 households.

## 12. Real-unit presentation for monetary metrics
- **What**: Firm liquidity and wage time series are now shown in real units (deflated by the mean price at each time step) rather than nominal dollars. Household consumption replaces the employment panel in the default plot layout (employment is near 100% throughout and can be stated in text).
- **Why**: In a closed economy without a monetary authority, nominal prices drift upward over time. Plotting wages and liquidity in nominal terms overstates firm health and masks purchasing-power erosion. Dividing by mean price converts to constant-purchasing-power units, making cross-scenario and cross-period comparisons meaningful.
- **Where**: `plot_from_csv_paper.py` — price deflator built per scenario; applied to `Firm_Liquidity` and `Mean_Wage` lines (main + sector breakdowns). Default 3×2 layout changed to Production, Capital, Liquidity (real), Household Consumption, Wage (real), Price. Former employment panel moved to optional `--show-inventory` row.
- **Manuscript impact**: Update figure descriptions and axis labels to note real units (deflated by mean price). Replace employment panel discussion with a sentence noting near-full employment. Add household consumption panel description.

## 13. Final-goods-only household demand
- **What**: Households now purchase only from final-good sectors (`retail`, `wholesale`, `services`). Upstream sectors such as commodity and manufacturing sell to firms, not directly to households. Household consumption ratios are interpreted only over final-good sectors and are renormalized accordingly.
- **Why**: The previous implementation let households buy upstream intermediate goods directly, which broke the intended three-tier supply chain and made retail structurally optional. In the 100-firm riverine topology this caused retail to collapse and the economy to degenerate into a distorted commodity/manufacturing labor contest. Restricting household demand to final goods restores coherent sector roles.
- **Where**: `agents.py`, household consumption logic; `model.py`, `get_final_consumption_ratios()` and default consumption ratio handling; parameter files updated so the riverine scenarios use `{"retail": 1.0}`.
- **Manuscript impact**: Clarify that household demand applies only to final-consumption sectors. Any text implying direct household purchases from commodity or manufacturing firms should be removed. Figures/text describing the three-tier chain should now align with the implemented market structure.

## 14. Demand-driven production planning and hiring
- **What**: Replaced budget-first hiring and per-supplier input reservation with demand-driven firm planning. Each firm now sets target output from expected sales plus an inventory buffer, translates that target into vacancies and intermediate-input needs, preserves a working-capital buffer, and treats capital spending as residual investment from surplus cash. Connected suppliers are treated as substitute sources of one aggregate intermediate good and procurement is from the cheapest available suppliers first.
- **Why**: The previous mechanism let firms hire until arbitrary budget buckets were exhausted rather than until planned production needs were met. Combined with direct household purchases from upstream sectors, this turned the model into a wage-budget competition for labor and produced the hazard-exceeds-baseline anomaly. Demand-driven planning is the simpler economic primitive: sales expectations determine output, output determines required labor and inputs, and investment comes after operating needs.
- **Note**: This supersedes the runtime role of the earlier pure-coefficient budget-allocation change. The core simulation no longer uses budget buckets as the primary coordination mechanism.
- **Where**: `agents.py`, `FirmAgent.plan_operations()`, `hire_labor()`, `sell_goods_to_firm()`, and `FirmAgent.step()`.
- **Manuscript impact**: Replace the budget-allocation description with a demand-driven operating cycle description. Explain that vacancies are derived from planned output, intermediate purchases are driven by required production, suppliers are treated as substitute sources of intermediate goods, and capital expenditure is residual rather than co-equal with payroll/input budgeting.

## 15. Demand-consistent startup state and sector-tier activation
- **What**: Replaced trophic-level-based startup inventories and runtime firm ordering with demand-consistent initialization and sector-tier activation. Final-demand shares are first converted into unit demand at current sector prices, then propagated upstream through the supplier network, and finally scaled so the implied labour demand matches the available household workforce. Initial inventories, working capital, and installed capital are then seeded from those expected sales. Runtime firm ordering now uses a simple sector-tier ordering (upstream before downstream) with random tie breaks within a tier.
- **Why**: Trophic seeding was not a stable organizing principle for the active 100-firm topology because the topology contains manufacturing cycles. More importantly, the legacy topology capital values were far below the flow capacity required to sustain the modeled labour force, so treating them as literal installed capacity forced the economy into an artificial low-output crash. A labour-scaled demand bootstrap is both simpler and more economically meaningful: firms start at a scale consistent with the available workforce and network technology rather than with a graph-theoretic proxy or arbitrary placeholder capital values.
- **Where**: `model.py`, `_solve_initial_expected_sales()`, `_seed_firm_operating_state()`, `_initialize_firm_operating_state()`, and `_sector_priority()`. Replacement firms in evolutionary turnover now use the same operating-state seeding logic.
- **Manuscript impact**: Remove or revise any statement that the runtime simulation depends on trophic levels for activation order or initial inventory assignment. If trophic levels remain in the paper, they should be framed as a network-analysis diagnostic rather than a core simulation mechanism.

## 16. Retained-earnings capital formation
- **What**: Removed the pseudo capital-goods purchase mechanism in which firms bought arbitrary other firms' output and converted it into capital stock. Capital is now treated as a reduced-form productive-capacity stock that expands directly from retained earnings after payroll, intermediate purchases, and liquidity buffers are covered.
- **Why**: The old mechanism allowed nonsensical capital formation paths, such as upstream commodity firms buying downstream goods as "capital", which inflated root-sector capacity and polluted intermediate-input inventories. It also prevented demand-constrained downstream sectors from repairing obvious capital shortfalls because investment was triggered only by instantaneous bottleneck labels. Direct retained-earnings capital formation is the cleaner reduced-form representation for this model since there is no explicit capital-goods sector.
- **Where**: `agents.py`, `FirmAgent.plan_operations()` and the capital-formation block in `FirmAgent.step()`; `model.py`, `_seed_firm_operating_state()`, and evolutionary replacement initialization.
- **Manuscript impact**: Describe capital as an internally financed productive-capacity state variable rather than as output purchased from other firms. Note that startup capital is set from demand-consistent operating scale and subsequent capital accumulation comes from retained earnings.

## 17. Within-period sequencing: labour, production, then consumption
- **What**: Split the single household/firms step order into phased markets. Households now sell labour first, firms then procure inputs and produce in upstream-to-downstream order, households consume after current-period production is available, and firms only close their sales/accounting state after all transactions for the period are complete.
- **Why**: The previous order let households buy only from yesterday's retail inventory and forced firms to replenish after the consumption market had already closed. That created a sawtooth stockout/replenishment cycle in baseline runs: one period of high sales would be followed by a production-only replenishment period, then another stockout period, with oscillations damping toward collapse. A phased within-period sequence is the simpler economic representation for a quarterly step: labour is hired, goods are produced, then households buy from current-period supply.
- **Where**: `agents.py`, `HouseholdAgent.supply_labor()`, `HouseholdAgent.consume_goods()`, `FirmAgent.step()`, and `FirmAgent.close_step()`; `model.py`, `EconomyModel.step()`.
- **Manuscript impact**: Update the simulation-sequence description to note that each period is phased rather than fully simultaneous. Household demand should be described as occurring after current-period production, not before it.

## 18. Seller-sector household-demand reporting
- **What**: Added explicit tracking of household purchases at the firm level via `household_sales_last_step`, and updated plotting so household-demand sector lines are derived from seller sectors rather than from household cohort labels.
- **Why**: The previous household-sector plots grouped households by their initialization tag (`commodity`, `manufacturing`, `retail`) and could be misread as product-sector demand. That created a false appearance that retail demand had collapsed to zero even when households were still buying final goods from retail firms. The new reporting matches the actual market transactions.
- **Where**: `agents.py`, firm sales bookkeeping; `model.py`, agent data collection; `run_simulation.py`, `plot_from_csv.py`, and `plot_from_csv_paper.py`, household-demand plotting.
- **Manuscript impact**: Any figure or text that interprets household demand by sector must refer to seller-sector final demand only. Household cohort tags should not be interpreted as consumption categories.

## 19. Quantitative results must be refreshed after the structural refactor
- **What**: The final-goods-only demand rule, demand-driven planning, labour-consistent startup seeding, retained-earnings capital formation, and phased within-period sequencing materially changed the model's steady-state behaviour and removed the earlier retail-collapse and hazard-exceeds-baseline artefacts.
- **Why**: Quantitative scenario results and figure narratives from pre-refactor runs are no longer representative of the current implementation. Keeping those numbers in the manuscript would misdescribe both the methodology and the model's repaired behaviour.
- **Where**: `JASSS0/TemplateJASSS.tex`, figure captions, results text, discussion, and conclusions.
- **Manuscript impact**: Regenerate the four-scenario time-series figures, bottleneck plots, and numerical summaries from the current implementation before submission. Remove or neutralize legacy claims that relied on the pre-refactor anomaly, especially statements that hazard scenarios systematically outperformed baseline or that sectoral household demand could be read from household cohort labels.
