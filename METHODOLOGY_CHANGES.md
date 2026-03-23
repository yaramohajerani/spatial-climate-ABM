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
- **What**: Replaced the ad-hoc wage adjustment mechanism (labor shortage cycle counting, unemployment rate checks, ±5% cap, wage ceiling) with revenue-based wage targeting. Firms now set wages as a fraction of revenue per worker: `target_wage = revenue_per_worker × labor_share`, with wages adjusting smoothly (10% toward target per step). The labour-share coefficient is now fixed at `0.5`; see also Change 30.
- **Why**: The previous mechanism counted consecutive steps of unmet labor demand and raised wages after 4 such cycles. In a Leontief economy with scarce labor, nearly all firms are perpetually labor-limited (the Leontief bottleneck, not genuine market failure), so the shortage signal was universally triggered. This caused continuous upward wage ratcheting. Revenue-based targeting is self-correcting: wages are structurally bounded by what workers produce, and automatically decrease when revenue falls (e.g., during hazard events).
- **What was removed**: `labor_shortage_cycles` tracking, 4-step shortage threshold, unemployment rate checks in wage logic, ±5% adjustment cap, hard wage ceiling of 10.0. The `labor_demand` computation (how many workers the firm could productively use) was also removed as it is no longer needed.
- **What was kept**: Minimum wage floor at 40% of initial wage (ILO 2016 justified).
- **Where**: `agents.py`, `FirmAgent.step()` wage adjustment section; removed `labor_shortage_cycles` and `labor_demand` from `FirmAgent.__init__()` and production section.
- **Manuscript impact**: Replace wage adjustment description with revenue-based targeting mechanism. Remove references to shortage thresholds and adjustment caps. The economic justification is standard labor economics: in competitive markets, wages converge to the marginal revenue product of labor.

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

## 20. Circular-flow closure between households and firms
- **What**: Added an explicit macro closure for the reduced-form economy. Households now receive not only wages but also firm payouts. Household consumption is based on current labour income plus the previous period's dividends and reduced-form investment income, alongside a small draw from money holdings above a target cash buffer. On the firm side, positive profits above the operating reserve are split between dividends and retained-earnings capital expansion.
- **Why**: The previous model had endogenous wages and household consumption but no ownership or payout channel returning firm surpluses to households. That caused household purchasing power to collapse while firm cash accumulated, and hazard scenarios could outperform baseline by slowing this cash extraction. The circular-flow closure restores the minimal macro mechanism used in standard toy macro ABMs: households earn wages and ownership income, then spend part of that disposable income back into the goods market.
- **Where**: `agents.py`, `HouseholdAgent.consume_goods()`, household income bookkeeping, `FirmAgent.close_step()`, and firm payroll/profit accounting; `model.py`, household-income distribution helpers and additional data collection fields.
- **Manuscript impact**: Rewrite the household budget equation and related text to describe a disposable-income consumption rule rather than `labour_sold × mean_wage`. State explicitly that households are the residual owners of firms and receive firm payouts in addition to wages.

## 21. Reduced-form investment spending no longer destroys money
- **What**: Investment spending is now recirculated to households as reduced-form investment income while increasing firms' productive-capacity stock. This keeps the model stock-flow closed even though there is no explicit capital-goods-producing sector.
- **Why**: The previous retained-earnings capital update still subtracted cash from firms without paying any modeled counterparty, so investment acted as a pure money sink. In a closed economy that breaks the monetary circuit. Recycling the spending is the minimal reduced-form closure until an explicit capital-goods sector is introduced.
- **Where**: `agents.py`, investment block in `FirmAgent.close_step()`; `model.py`, `distribute_household_income()` and stock-flow reporters.
- **Manuscript impact**: Clarify that capital accumulation is reduced-form. Either describe the investment-income recycling explicitly, or more generally state that the reduced model preserves monetary closure by returning investment spending to the household sector in the absence of a modeled capital-goods producer.

## 22. Per-period accounting reset moved ahead of the labour market
- **What**: Firm flow-accounting variables (`wage_bill_this_step`, `input_spend_this_step`, profits, dividends, investment spending) are now reset at the planning stage before households sell labour, rather than inside the production step after wages have already been paid.
- **Why**: Resetting those counters after the labour market erased payroll from profit accounting, overstated profits, and caused excessive payouts. The issue became visible only after adding dividend-based circular-flow closure, because profit accounting now directly governs household income.
- **Where**: `agents.py`, `FirmAgent.plan_operations()` and `FirmAgent.step()`.
- **Manuscript impact**: No conceptual change to the published algorithm, but the implementation description should be explicit that payroll is part of the current-period accounting before profits are distributed.

## 23. Stock-flow closure diagnostics and regression tests
- **What**: Added explicit reporters for `Total_Money`, `Money_Drift`, household labour/dividend/investment income, and firm-side dividend/investment payouts. Added regression tests that verify monetary conservation and that a closed no-hazard economy remains economically active rather than converging to zero output.
- **Why**: Once the model was repaired at the systemic level, we needed an equally systemic validation check. The new diagnostics make stock-flow closure observable in every run, and the regression tests guard against future changes reintroducing money sinks or dead-economy artefacts.
- **Where**: `model.py`, `DataCollector`; `tests/test_stock_flow_closure.py`.
- **Manuscript impact**: Add a short validation note that aggregate money is tracked as an accounting diagnostic and remains constant to numerical tolerance in closed-economy test runs, while production, consumption, and employment remain positive.

## 24. Household relocation no longer destroys money
- **What**: Removed the direct monetary haircut on household relocation. Households may still relocate for hazard or job-search reasons, but relocation itself no longer deletes a fraction of household money.
- **Why**: In a closed economy with only households and firms, relocation costs had no modeled counterparty and therefore acted as a money sink. This became visible in baseline runs as a gradual fall in `Total_Money` even when hazards and learning were off. Until an explicit housing/transport service sector is modeled, the cleaner reduced-form choice is to treat relocation as a positional/frictional change rather than a cash destruction mechanism.
- **Where**: `agents.py`, `HouseholdAgent._relocate_for_job()` and `HouseholdAgent._relocate()`.
- **Manuscript impact**: If relocation costs are discussed, they should no longer be described as monetary losses. Relocation remains behavioral but not a source of nominal money destruction.

## 25. Learning-time replacement is now stock-flow-consistent reorganization
- **What**: Replaced ex nihilo firm entry with in-place firm reorganization. When a firm fails under the learning system, the productive unit remains in place, inherits a mutated strategy from a successful parent, and can receive recapitalization from households as equity finance. Existing cash, inventories, capital, links, and location remain inside the modeled economy.
- **Why**: The previous replacement logic removed a failed firm and instantiated a new one with fresh working capital, inventories, and capacity. That violated stock-flow consistency by creating new balance-sheet resources during learning runs. Reorganization preserves the closed-economy accounting while retaining the intended evolutionary-selection mechanism.
- **Where**: `model.py`, `_apply_evolutionary_pressure()` and the new household-to-firm equity-transfer helper; `tests/test_stock_flow_closure.py`, replacement regression.
- **Manuscript impact**: Describe firm replacement as bankruptcy reorganization or ownership/management turnover rather than literal entry of a fully reseeded new establishment. Note that recapitalization is funded by the household sector, preserving the monetary circuit.

## 26. Household relocation disabled in the core scenarios
- **What**: Set `household_relocation` to `false` in the main parameter files and changed the model/CLI default to relocation-off. The relocation code remains available as an optional exploratory feature.
- **Why**: Under the current model, households are not directly flood-damaged and already search across all firms with only a soft distance penalty. The existing relocation rule therefore adds behavioral noise more than a well-identified climate-migration channel. Keeping it on in the main scenarios would add complexity without a correspondingly clear mechanism or interpretation.
- **Where**: `model.py`, `run_simulation.py`, `aqueduct_riverine_parameters_rcp8p5.json`, `aqueduct_riverine_parameters_rcp4p5.json`, `quick_test_parameters.json`, and `sample_firm_topology_parameters.json`.
- **Manuscript impact**: Update the methodology and scenario description to state that household relocation is currently disabled in the core reported experiments. If migration is discussed as future work, frame it as a feature that would need a better-founded residential-risk and labor-access mechanism.

## 27. Active-versus-dormant firm diagnostics
- **What**: Added active/dormant firm diagnostics to the scenario consistency checker. Firms are classified as active when production exceeds a configurable threshold and dormant otherwise; the checker now reports final active/dormant counts and tail-window means, with an optional minimum-active-firms gate.
- **Why**: Monetary consistency alone is not enough. A run can conserve money while collapsing into a highly concentrated or dormant economy. Active-versus-dormant counts provide a compact structural diagnostic that complements production, consumption, and labour aggregates.
- **Where**: `check_consistency.py`.
- **Manuscript impact**: No direct manuscript change required, but this diagnostic is useful when validating the reported scenario runs and could be mentioned in supplementary validation notes if needed.

## 28. Learning is now peer imitation with bounded exploration
- **What**: Replaced the previous within-firm random mutation rule with a same-sector imitation rule. Every adaptation cycle, a firm evaluates its recent production-based fitness, looks for fitter peers in the same sector, partially moves its learned strategy parameters toward a selected role model, and applies only small bounded exploratory mutation around that anchor. If no fitter peer is available, the firm makes only a very small local exploratory move around its incumbent strategy.
- **Why**: The earlier rule was labelled hill-climbing but in practice always mutated randomly, with larger mutations exactly when fitness was weak or volatile. Under hazard this turned learning into a random walk that could depress wages, labour demand, and output relative to the no-learning case. Peer imitation is a simpler and more standard evolutionary-ABM mechanism: adaptation diffuses strategies that are currently performing better under the same sectoral environment instead of amplifying noise during stress.
- **Where**: `agents.py`, `FirmAgent._adapt_strategy()` and the new helper methods for role-model selection and bounded mutation; `model.py` data collection now records firm strategy parameters for diagnostics.
- **Manuscript impact**: Rewrite the learning subsection to describe imitation-based adaptation rather than adaptive mutation strength. Clarify that the learning outputs now report the learned strategy parameters directly for debugging and result interpretation.

## 29. Hazard preflight checks now parse parameter-file flood specifications correctly
- **What**: Added explicit parsing of `rp_files` strings in `check_consistency.py` before constructing `EconomyModel`.
- **Why**: The consistency checker previously passed raw parameter-file hazard strings straight into the model, which broke hazard scenario validation even though the main simulation runner parsed them correctly. The checker now supports the same hazard input format as `run_simulation.py`.
- **Where**: `check_consistency.py`.
- **Manuscript impact**: No manuscript text required, but this fixes a validation tool inconsistency that affected hazard scenario preflight checks.

## 30. Wage responsiveness removed from the learning system
- **What**: Removed `wage_responsiveness` as an evolvable firm parameter and fixed the labour-share coefficient in the wage rule at `0.5`.
- **Why**: Allowing firms to learn the labour share improved fitness partly by reducing wages even in baseline no-hazard conditions. That makes the parameter behave like a structural income-distribution assumption rather than a climate-adaptation lever. Holding it fixed preserves endogenous wages through revenue per worker while preventing the evolutionary system from treating wage suppression as an adaptation strategy.
- **Where**: `agents.py` wage-setting and strategy initialization; `model.py` data collection now records a fixed labour-share diagnostic instead of average wage responsiveness; `tests/test_stock_flow_closure.py` includes a fixed-labour-share wage regression.
- **Manuscript impact**: Update the learning subsection from five to four evolvable parameters, rewrite the wage equation so labour share is fixed rather than learned, and explain that labour-share calibration is held structural in this reduced-form model pending richer labour-market calibration in future work.
