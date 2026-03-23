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


def test_adaptation_spending_is_returned_to_households() -> None:
    """Adaptation maintenance must be a household-side income flow, not a money sink."""
    model = build_closed_economy_model(adaptation_enabled=True)
    firm = model._firms[0]
    initial_total_money = model.total_money()

    firm.resilience_capital = 0.5
    firm.base_capital_target = 100.0
    firm._liquidity_buffer = 0.0
    firm.begin_period_adaptation()

    household_adaptation_income = sum(h.adaptation_income_this_step for h in model._households)

    assert firm.adaptation_spending_this_step > 0.0
    assert np.isclose(household_adaptation_income, firm.adaptation_spending_this_step, atol=1e-9)
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
    parent.supplier_disruption_ewma = 0.1
    parent.ucb_action_counts = {(2, 1, 1, 1): [3, 2, 1]}
    parent.ucb_action_values = {(2, 1, 1, 1): [0.1, 0.4, 0.2]}

    failed_firm.money = 0.0
    initial_total_money = model.total_money()

    model._apply_firm_reorganization()

    assert model.total_firm_replacements >= 1
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-6)
    assert np.isclose(failed_firm.resilience_capital, parent.resilience_capital, atol=1e-9)
    assert failed_firm.expected_direct_loss_ewma == parent.expected_direct_loss_ewma
    assert failed_firm.realized_direct_loss_ewma == parent.realized_direct_loss_ewma
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
    """The tabular UCB policy should explore untried actions and then exploit stronger rewards."""
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

    for action_index, reward in ((0, 0.15), (1, 0.75), (2, 0.25)):
        firm.current_policy_context = context
        firm.current_action_index = action_index
        firm.window_raw_direct_loss = 10.0
        firm.window_adapted_direct_loss = 10.0 * (1.0 - reward)
        firm.window_adaptation_cost = 0.0
        firm._finalize_adaptation_window()

    firm.ucb_action_counts[context] = [6, 6, 6]
    firm.ucb_action_values[context] = [0.15, 0.75, 0.25]
    assert firm._choose_ucb_action(context) == 1
