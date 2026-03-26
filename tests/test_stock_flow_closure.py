from pathlib import Path

import numpy as np

from model import EconomyModel


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_closed_economy_model(*, adaptation_enabled: bool) -> EconomyModel:
    """Return a small no-hazard economy for accounting and adaptation tests."""
    return EconomyModel(
        num_households=300,
        hazard_events=[],
        seed=42,
        firm_topology_path=str(REPO_ROOT / "sample_firm_topology.json"),
        apply_hazard_impacts=False,
        adaptation_params={
            "enabled": adaptation_enabled,
            "decision_interval": 4,
            "reward_window": 4,
            "ewma_alpha": 0.2,
            "ucb_c": 1.0,
            "observation_radius": 4,
            "action_increments": [0.0, 0.05, 0.10],
            "resilience_decay": 0.01,
            "maintenance_cost_rate": 0.005,
            "loss_reduction_max": 0.6,
            "min_money_survival": 1.0,
            "replacement_frequency": 10,
        },
        consumption_ratios={"services": 0.7, "retail": 0.3},
        household_relocation=False,
    )


def test_total_money_is_conserved_in_closed_economy_without_adaptation() -> None:
    """Money should stay inside the household-firm circuit when adaptation is disabled."""
    model = build_closed_economy_model(adaptation_enabled=False)
    initial_total_money = model.total_money()

    for _ in range(40):
        model.step()

    df = model.results_to_dataframe()

    assert np.allclose(df["Total_Money"].to_numpy(), initial_total_money, atol=1e-6)
    assert np.allclose(df["Money_Drift"].to_numpy(), 0.0, atol=1e-6)
    assert np.allclose(
        df["Household_Dividend_Income"].to_numpy(),
        df["Firm_Dividends_Paid"].to_numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        df["Household_Capital_Income"].to_numpy(),
        df["Firm_Investment_Spending"].to_numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        df["Household_Adaptation_Income"].to_numpy(),
        df["Firm_Adaptation_Spending"].to_numpy(),
        atol=1e-6,
    )


def test_baseline_adaptation_stays_dormant_without_hazards() -> None:
    """Hazard-conditional adaptation should not drift in a baseline no-hazard run."""
    model = build_closed_economy_model(adaptation_enabled=True)

    for _ in range(12):
        model.step()

    df = model.results_to_dataframe()

    assert np.allclose(df["Money_Drift"].to_numpy(), 0.0, atol=1e-6)
    assert np.allclose(df["Firm_Adaptation_Spending"].to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(df["Household_Adaptation_Income"].to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(df["Average_Resilience_Capital"].to_numpy(), 0.0, atol=1e-9)
    assert all(not firm.ucb_action_counts for firm in model._firms)
    assert all(firm.current_policy_context is None for firm in model._firms)
    assert all(firm.adaptation_update_count == 0 for firm in model._firms)


def test_adaptation_spending_is_deferred_and_returned_to_households() -> None:
    """Adaptation should be funded after operations and remain stock-flow consistent."""
    model = build_closed_economy_model(adaptation_enabled=True)
    firm = model._firms[0]

    firm.money = 200.0
    initial_total_money = model.total_money()
    firm.base_capital_target = 100.0
    firm.resilience_capital = 0.5
    firm.begin_period_adaptation()
    decayed_resilience = firm.resilience_capital
    firm._queue_adaptation_investment(2)

    assert np.isclose(firm.money, 200.0, atol=1e-9)
    assert firm.adaptation_spending_this_step == 0.0
    assert np.isclose(firm.resilience_capital, decayed_resilience, atol=1e-9)

    firm._fund_adaptation_after_operations(available_cash=50.0, investable_profit=50.0)
    household_adaptation_income = sum(h.adaptation_income_this_step for h in model._households)

    assert firm.adaptation_spending_this_step > 0.0
    assert firm.money < 200.0
    assert firm.resilience_capital > decayed_resilience
    assert np.isclose(household_adaptation_income, firm.adaptation_spending_this_step, atol=1e-9)
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-9)


def test_firms_replace_base_capital_before_paying_dividends() -> None:
    """Positive profits should first rebuild the base capital target before dividends."""
    model = build_closed_economy_model(adaptation_enabled=False)
    firm = model._firms[0]

    firm.money = 200.0
    firm.base_capital_target = 100.0
    firm.target_capital_stock = 120.0
    firm.capital_stock = 90.0
    firm.revenue_this_step = 20.0
    firm.wage_bill_this_step = 0.0
    firm.input_spend_this_step = 0.0
    firm.depreciation_this_step = 0.0
    firm._liquidity_buffer = 0.0

    initial_total_money = model.total_money()
    firm.close_step()

    assert np.isclose(firm.investment_spending_this_step, 15.0, atol=1e-9)
    assert np.isclose(firm.capital_stock, 105.0, atol=1e-9)
    assert np.isclose(firm.dividends_paid_this_step, 5.0, atol=1e-9)
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-9)


def test_households_recirculate_excess_wealth_into_consumption() -> None:
    """Households should spend a modest share of cash above the target buffer."""
    model = build_closed_economy_model(adaptation_enabled=False)
    household = model._households[0]

    household.money = 150.0
    household.labor_income_this_step = 0.0
    household.dividend_income_last_step = 0.0
    household.capital_income_last_step = 0.0
    household.adaptation_income_last_step = 0.0
    household.consumption = 0.0

    for seller in model._firms:
        seller.price = 1.0
        if seller.sector in model.get_final_consumption_ratios():
            seller.inventory_output = 100.0
        else:
            seller.inventory_output = 0.0
    initial_total_money = model.total_money()

    household.consume_goods()

    expected_budget = household.CONSUMPTION_PROPENSITY_WEALTH * (
        150.0 - household.TARGET_CASH_BUFFER
    )
    assert np.isclose(household.consumption, expected_budget, atol=1e-9)
    assert np.isclose(household.money, 150.0 - expected_budget, atol=1e-9)
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-9)


def test_bounded_working_capital_credit_finances_operations_without_money_leakage() -> None:
    """Payroll and input purchases may use bounded operating credit without breaking closure."""
    model = build_closed_economy_model(adaptation_enabled=False)
    buyer = next(f for f in model._firms if f.connected_firms)
    supplier = buyer.connected_firms[0]
    household = next(h for h in model._households if h.sector == buyer.sector)

    buyer.money = 5.0
    buyer._liquidity_buffer = 10.0
    buyer.working_capital_credit_limit = 20.0
    buyer.working_capital_credit_used_this_step = 0.0
    buyer.target_labor = 1
    buyer.employees.clear()

    supplier.price = 2.0
    supplier.inventory_output = 10.0
    supplier.money = 100.0

    wage = 8.0
    initial_total_money = model.total_money()

    assert buyer.hire_labor(household, wage) is True
    assert np.isclose(buyer.money, -3.0, atol=1e-9)
    assert np.isclose(buyer.wage_bill_this_step, wage, atol=1e-9)
    assert np.isclose(buyer.working_capital_credit_used_this_step, 13.0, atol=1e-9)

    bought = supplier.sell_goods_to_firm(buyer, quantity=3.0)
    assert np.isclose(bought, 3.0, atol=1e-9)
    assert np.isclose(buyer.money, -9.0, atol=1e-9)
    assert np.isclose(buyer.input_spend_this_step, 6.0, atol=1e-9)
    assert np.isclose(buyer.working_capital_credit_used_this_step, 19.0, atol=1e-9)

    extra_bought = supplier.sell_goods_to_firm(buyer, quantity=2.0)
    assert np.isclose(extra_bought, 0.5, atol=1e-9)
    assert np.isclose(buyer.money, -10.0, atol=1e-9)
    assert np.isclose(buyer.working_capital_credit_used_this_step, 20.0, atol=1e-9)
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-9)


def test_firm_reorganization_preserves_total_money_and_inherits_adaptation_state() -> None:
    """Reorganization should keep money closed and copy parent adaptation state."""
    model = build_closed_economy_model(adaptation_enabled=True)

    for _ in range(6):
        model.step()

    retail_firms = [f for f in model._firms if f.sector == "retail"]
    failed_firm, parent = retail_firms[:2]
    for candidate in retail_firms[2:]:
        candidate.money = 0.0
    parent.resilience_capital = 0.4
    parent.expected_direct_loss_ewma = 0.2
    parent.realized_direct_loss_ewma = 0.15
    parent.local_observed_loss_ewma = 0.1
    parent.supplier_disruption_ewma = 0.1
    parent.ucb_action_counts = {(1, 1, 0, 0): [2, 1, 0]}
    parent.ucb_action_values = {(1, 1, 0, 0): [0.3, 0.5, 0.0]}

    failed_firm.money = 0.0
    initial_total_money = model.total_money()

    model._apply_firm_reorganization()

    assert model.total_firm_replacements >= 1
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-6)
    assert np.isclose(failed_firm.resilience_capital, parent.resilience_capital, atol=1e-9)
    assert failed_firm.expected_direct_loss_ewma == parent.expected_direct_loss_ewma
    assert failed_firm.realized_direct_loss_ewma == parent.realized_direct_loss_ewma
    assert failed_firm.local_observed_loss_ewma == parent.local_observed_loss_ewma
    assert failed_firm.supplier_disruption_ewma == parent.supplier_disruption_ewma
    assert failed_firm.ucb_action_counts == parent.ucb_action_counts
    assert failed_firm.ucb_action_values == parent.ucb_action_values


def test_resilience_capital_reduces_direct_losses_and_improves_recovery() -> None:
    """Higher resilience should attenuate capital, inventory, and downtime damage."""
    model = build_closed_economy_model(adaptation_enabled=True)
    low_resilience, high_resilience = model._firms[:2]

    for firm in (low_resilience, high_resilience):
        firm.capital_stock = 100.0
        firm.inventory_output = 20.0
        firm.inventory_inputs = {999: 10.0}
        firm.damage_factor = 1.0
        firm.counterfactual_damage_factor = 1.0
        firm.expected_sales = 15.0
        firm.price = 1.0
        firm.money = 100.0
        firm.begin_period_adaptation()

    low_resilience.resilience_capital = 0.0
    high_resilience.resilience_capital = 0.5
    raw_loss = 0.4

    for firm in (low_resilience, high_resilience):
        adapted_loss = firm.get_adapted_loss_fraction(raw_loss)
        firm.record_direct_losses(raw_loss, adapted_loss)
        firm.capital_stock *= 1.0 - adapted_loss
        firm.damage_factor *= 1.0 - adapted_loss
        firm.counterfactual_damage_factor *= 1.0 - raw_loss
        firm.inventory_output *= 1.0 - adapted_loss
        for supplier_id in list(firm.inventory_inputs.keys()):
            firm.inventory_inputs[supplier_id] *= 1.0 - adapted_loss

    assert high_resilience.capital_stock > low_resilience.capital_stock
    assert high_resilience.inventory_output > low_resilience.inventory_output
    assert high_resilience.inventory_inputs[999] > low_resilience.inventory_inputs[999]
    assert high_resilience.damage_factor > low_resilience.damage_factor

    low_resilience.step()
    high_resilience.step()

    assert high_resilience.damage_factor > low_resilience.damage_factor
    assert high_resilience.realized_direct_loss_this_step < low_resilience.realized_direct_loss_this_step


def test_bandit_explores_unseen_actions_then_prefers_higher_reward_action() -> None:
    """The firm-level tabular UCB policy should explore untried actions and then exploit stronger rewards."""
    model = build_closed_economy_model(adaptation_enabled=True)
    firm = model._firms[0]
    context = (2, 2, 1, 0)

    assert firm._choose_ucb_action(context) == 0

    firm.ucb_action_counts[context] = [1, 0, 0]
    firm.ucb_action_values[context] = [0.2, 0.0, 0.0]
    assert firm._choose_ucb_action(context) == 1

    firm.ucb_action_counts[context] = [1, 1, 0]
    firm.ucb_action_values[context] = [0.2, 0.4, 0.0]
    assert firm._choose_ucb_action(context) == 2

    peer = next(
        candidate for candidate in model._firms
        if candidate is not firm and candidate.sector == firm.sector
    )
    firm.ucb_action_counts[context] = [0, 0, 0]
    firm.ucb_action_values[context] = [0.0, 0.0, 0.0]
    firm.current_policy_context = context
    firm.current_action_index = 1
    firm.steps_in_current_window = firm.reward_window
    firm.window_raw_direct_loss = 10.0
    firm.window_adapted_direct_loss = 6.0
    firm.window_adaptation_cost = 0.0
    firm.window_peak_raw_loss_signal = 0.2
    assert firm._update_adaptation_policy_from_window() is True

    assert firm.ucb_action_counts[context][1] == 1
    assert np.isclose(firm.ucb_action_values[context][1], 0.4, atol=1e-9)
    assert np.isclose(firm.last_adaptation_reward, 0.4, atol=1e-9)
    assert peer.ucb_action_counts.get(context) is None

    firm.ucb_action_counts[context] = [6, 6, 6]
    firm.ucb_action_values[context] = [0.15, 0.75, 0.25]
    assert firm._choose_ucb_action(context) == 1


def test_bandit_skips_uninformative_low_loss_windows() -> None:
    """Tiny-loss windows should not update the firm-level adaptation bandit."""
    model = build_closed_economy_model(adaptation_enabled=True)
    firm = model._firms[0]
    context = (1, 1, 0, 0)

    firm.current_policy_context = context
    firm.current_action_index = 1
    firm.steps_in_current_window = firm.reward_window
    firm.window_raw_direct_loss = 10.0
    firm.window_adapted_direct_loss = 6.0
    firm.window_adaptation_cost = 0.0
    firm.window_peak_raw_loss_signal = 0.01
    firm._adaptation_policy_initialized = True
    updated = firm._update_adaptation_policy_from_window()

    assert updated is False
    assert firm.last_adaptation_reward != firm.last_adaptation_reward
    assert firm.adaptation_update_count == 0
    assert firm.ucb_action_counts.get(context) is None


def test_nearby_hazard_losses_enter_local_observed_loss_state() -> None:
    """Firms should observe nearby hazard losses even before they are hit themselves."""
    model = build_closed_economy_model(adaptation_enabled=True)
    focal = next(f for f in model._firms if f.sector == "services")
    neighbor = next(
        f for f in model._firms
        if f is not focal and f.sector == focal.sector
    )

    focal.pos = (10, 10)
    neighbor.pos = (11, 10)
    neighbor.raw_direct_loss_fraction_this_step = 0.4

    observed = model.get_local_observed_loss_fraction(focal)

    assert np.isclose(observed, 0.4, atol=1e-9)


def test_cascade_reporters_track_never_hit_firm_burden() -> None:
    """Systemic-risk reporters should separate directly hit firms from never-hit disrupted firms."""
    model = build_closed_economy_model(adaptation_enabled=True)

    for firm in model._firms:
        firm.ever_directly_hit = False
        firm.ever_indirectly_disrupted_before_direct_hit = False
        firm.production = 0.0
        firm.capital_stock = 0.0
        firm.supplier_disruption_this_step = 0.0
        firm.supplier_disruption_ewma = 0.0

    directly_hit = model._firms[0]
    indirectly_disrupted = model._firms[1]
    unaffected_never_hit = model._firms[2]

    directly_hit.ever_directly_hit = True
    directly_hit.production = 2.0
    directly_hit.capital_stock = 5.0
    directly_hit.supplier_disruption_ewma = 0.1

    indirectly_disrupted.production = 8.0
    indirectly_disrupted.capital_stock = 12.0
    indirectly_disrupted.supplier_disruption_this_step = 0.3
    indirectly_disrupted.supplier_disruption_ewma = 0.6
    indirectly_disrupted.ever_indirectly_disrupted_before_direct_hit = True

    unaffected_never_hit.production = 10.0
    unaffected_never_hit.capital_stock = 8.0
    unaffected_never_hit.supplier_disruption_ewma = 0.1

    model.datacollector.collect(model)
    row = model.results_to_dataframe().iloc[-1]
    total_firms = len(model._firms)

    assert row["Ever_Directly_Hit_Firms"] == 1
    assert np.isclose(row["Ever_Directly_Hit_Firm_Share"], 1.0 / total_firms, atol=1e-9)
    assert row["Never_Hit_Firms"] == total_firms - 1
    assert row["Never_Hit_Currently_Disrupted_Firms"] == 1
    assert np.isclose(row["Never_Hit_Currently_Disrupted_Firm_Share"], 1.0 / total_firms, atol=1e-9)
    assert np.isclose(row["Never_Hit_Supplier_Disruption_Burden_Share"], 0.875, atol=1e-9)
    assert np.isclose(row["Never_Hit_Production_Share"], 0.9, atol=1e-9)
    assert np.isclose(row["Never_Hit_Capital_Share"], 0.8, atol=1e-9)
    assert row["Ever_Indirectly_Disrupted_Before_Direct_Hit_Firms"] == 1
    assert np.isclose(
        row["Ever_Indirectly_Disrupted_Before_Direct_Hit_Firm_Share"],
        1.0 / total_firms,
        atol=1e-9,
    )


def test_households_try_nearby_same_sector_firms_before_remote_market() -> None:
    """Sector-local staged search should fill nearby vacancies before remote fallback."""
    model = build_closed_economy_model(adaptation_enabled=False)
    household = model._households[0]
    local_firm = next(f for f in model._firms if f.sector == "services")
    remote_firm = next(f for f in model._firms if f.sector == "retail")

    household.sector = local_firm.sector
    household.pos = local_firm.pos
    household.nearby_firms = [local_firm]

    for firm in model._firms:
        firm.money = 0.0
        firm.target_labor = 0
        firm.employees.clear()

    local_firm.money = 100.0
    local_firm.target_labor = 1
    local_firm.wage_offer = 1.0
    remote_firm.money = 100.0
    remote_firm.target_labor = 1
    remote_firm.wage_offer = 5.0

    household.supply_labor()

    assert household.labor_sold == 1.0
    assert household in local_firm.employees
    assert household not in remote_firm.employees
