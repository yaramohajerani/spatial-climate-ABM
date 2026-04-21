# Model Logic Review — Validated Issues

This document keeps only the issues from the earlier review that were supported
by the codebase after validation. Items marked with `[x]` have been addressed
in the current implementation.

Removed items were either intentional modeling choices, outdated diagnoses, or
claims that were not supported by the implementation. To preserve traceability,
the retained issues keep their original numbering.

---

## 1. Stock-flow / monetary consistency

### [x] 1.1 Money can leak if the household set is empty
`agents.py` — `_install_capital`, `_fund_adaptation_after_operations`, and
`_pay_dividends` all debit firm money before routing the corresponding payment
through `model.distribute_household_income()`. That helper returns immediately
when `self._households` is empty, so in the edge case of a zero-household model
the cash leaves the firm and is not credited anywhere.

This is a real accounting bug for that edge case. The earlier review's claim
that ordinary relocation or migration could create the problem was overstated;
the issue is specifically the empty-household case.

### [x] 1.3 Topology capital is only partially preserved at startup
`model.py` — topology files populate each firm's initial `capital` value during
agent creation, but `_seed_firm_operating_state()` then applies
`firm.capital_stock = max(firm.capital_stock, capital_target)`.

So the topology value is not universally "overwritten", but it is silently
lifted for firms whose configured capital falls below the seeded minimum implied
by expected sales. This is a real configuration-to-effective-state mismatch and
should either be documented or surfaced explicitly in metadata/warnings.

### [x] 1.4 Inventory destruction is recorded diagnostically but not in cash profit
`model.py` / `agents.py` — flood damage immediately writes down
`inventory_output` and input inventories. The model does record direct-loss
diagnostics via `record_direct_losses()`, but `profit_this_step` is later
computed as an operating cash-profit measure:

`revenue - wage_bill - input_spend - depreciation`

That means hazard inventory write-downs do not enter the period profit measure
even though the physical loss is recorded elsewhere. The issue is therefore not
"unaccounted damage", but rather a mismatch between the documented profit
concept and the direct-loss accounting shown in diagnostics.

---

## 2. Core mathematics / production & pricing

### [x] 2.1 `hazard_operating_shortfall` uses an incomplete no-hazard ceiling
`agents.py` — `no_hazard_output_ceiling = min(self.demand_driven_output,
max_output_from_labor)` omits capital and input constraints. The realized
output ceiling does include labor, inputs, and capital.

As a result, the hazard-shortfall diagnostic can attribute output gaps to
hazards even when non-hazard capital or input scarcity is actually binding.

### [x] 2.3 `limiting_factor = "demand"` also absorbs financing constraints
`agents.py` — if `target_output < technical_output_limit`, the code sets
`limiting_factor = "demand"`. But `target_output` is already reduced in
`plan_operations()` by working-capital and liquidity limits before production
begins.

So the label "demand" is broader than it sounds: it includes cases where the
firm would like to produce more but was cut back by finance during planning.
This is a diagnostic-classification issue, not a production bug.

### [x] 2.4 Sell-through is based on inventory after hazard write-downs
`agents.py` — pricing comments say sell-through should reflect the previous
completed period, but the denominator is computed from current-period
`inventory_output + sales_last_step`. Because current inventory may already
have been reduced by hazard damage before pricing runs, hazards mechanically
raise sell-through and therefore markup pressure.

This is a real timing/measurement bug in the price diagnostic.

### [x] 2.6 Per-step hazard probability uses a linear approximation
`model.py` — `_sample_pixelwise_hazard()` converts annual hazard probability to
per-step probability using `p_annual / steps_per_year`.

That is only a first-order approximation to the correct sub-period Bernoulli
probability `1 - (1 - p_annual) ** (1 / steps_per_year)`. The difference is
material for high-frequency events such as RP=2.

---

## 3. Paper vs. code / documentation mismatches

### [x] 3.1 `stockpiling` currently means larger finished-goods buffers
`agents.py` — the `stockpiling` strategy increases
`_effective_inventory_buffer_ratio()`, which raises finished-goods inventory
targets. It does not create extra input inventories.

This is a real naming/documentation issue. The implementation is coherent, but
the strategy name can be misread as input stockpiling unless the docs are made
explicit.

### [x] 3.2 Final-demand ratios silently discard non-final sectors
`model.py` — `get_final_consumption_ratios()` filters user-provided
`consumption_ratios` down to `{"retail", "wholesale", "services"}` and then
renormalizes.

That behavior is intentional given the household-demand design, but the silent
drop is still a real usability/documentation issue because a user can supply
weights for upstream sectors and get a different effective configuration with
no warning.

### [x] 3.7 Counterfactual damage recovery shares the actual firm's liquidity state
`agents.py` — `counterfactual_damage_factor` recovers using the same
`base_recovery_rate` that is computed from the actual firm's current money.

This does not break the model's main dynamics, but it means the reported
counterfactual direct-loss path is not a fully independent no-adaptation
counterfactual. It is a proxy diagnostic coupled to realized finances.

---

## 4. Timing and ordering issues

### [x] 4.1 Hazard relocation does not refresh `nearby_firms`
`agents.py` — `_relocate()` moves the household but does not call
`_update_nearby_firms()`, unlike `_relocate_for_job()`.

That means a hazard-relocated household can retain a stale nearby-firm list for
subsequent labor-supply decisions until something else refreshes it.

### [x] 4.2 Planning and realized production use different damage states
`agents.py` — `plan_operations()` uses the post-hazard, pre-recovery
`damage_factor`, but `FirmAgent.step()` applies partial recovery before
computing realized production.

This is not necessarily wrong as a modeling choice, but it is a real temporal
mismatch between the planning state and the execution state, and it affects how
shortfalls should be interpreted.

### [x] 4.5 Hazard attribution for supplier shortages is coarse
`agents.py` — `_primary_supplier_shortage_is_hazard_related()` returns `True`
whenever any connected supplier shows any hazard-disruption signal.

That makes the hazard attribution intentionally broad, but it can still
over-attribute a buyer's shortage to hazards when the supplier is also facing
other problems such as weak liquidity or pricing effects.

### [x] 4.7 Reserved-capacity contracts reserve current inventory, not future output
`model.py` / `agents.py` — reserved-capacity contracts are created from the
supplier's currently available inventory before this step's production occurs.

If the intended interpretation is literal future delivery capacity, the current
implementation is too narrow; it behaves more like reserved inventory or
standing priority access to a slice of already-produced stock. This is mainly a
design/interpretation mismatch that should be clarified.

---

## 5. Measurement / reporting

### [x] 5.2 `hazard_operating_shortfall` is a proxy metric, not a full counterfactual
`agents.py` — the shortfall metric compares realized output against a
simplified no-hazard ceiling rather than against a separately simulated
counterfactual firm state. It also inherits the missing-capital problem noted
in 2.1.

This matters because downstream diagnostics, including the never-hit cascade
burden measures, use that proxy signal.

### [x] 5.4 Recovery speed depends on absolute money units
`agents.py` — recovery scales from 0.2 to 0.5 using `self.money / 200.0`.

That creates scale dependence: a nominal rescaling of all balances would change
recovery behavior even if the real state of the economy were otherwise
unchanged. This is a modeling sensitivity issue rather than an implementation
bug.

---

## 7. Sensitivity-analysis / runner

### [x] 7.2 Metadata does not capture all effective post-processing
`run_simulation.py` — the runner records the effective adaptation config, which
is good, but it still does not record every effective state transformation that
happens later inside the model.

Examples include:

- consumption ratios after final-demand filtering and renormalization
- startup capital floors imposed by `_seed_firm_operating_state()`
- other derived in-model state that differs from the raw parameter file

So the original review overstated this point, but a narrower metadata gap is
real.

---

## Severity Summary

- **High:** 1.1, 2.1, 2.4, 2.6, 4.1, 5.2
- **Medium:** 1.3, 1.4, 2.3, 3.2, 3.7, 4.2, 4.5, 7.2
- **Low / documentation-facing:** 3.1, 4.7, 5.4

The largest correctness risks are the hazard-shortfall measurement path
(`2.1`, `5.2`), the pricing denominator timing issue (`2.4`), the hazard
probability approximation (`2.6`), and the stale relocation-neighborhood bug
(`4.1`).
