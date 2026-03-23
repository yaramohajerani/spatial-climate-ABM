from pathlib import Path

import numpy as np

from model import EconomyModel


REPO_ROOT = Path(__file__).resolve().parents[1]


def build_closed_economy_model() -> EconomyModel:
    """Return a small no-hazard economy for financial-flow regression tests."""
    return EconomyModel(
        num_households=300,
        hazard_events=[],
        seed=42,
        firm_topology_path=str(REPO_ROOT / "sample_firm_topology.json"),
        apply_hazard_impacts=False,
        learning_params={
            "enabled": False,
            "memory_length": 10,
            "mutation_rate": 0.05,
            "adaptation_frequency": 5,
            "min_money_survival": 1.0,
            "replacement_frequency": 10,
        },
        consumption_ratios={"services": 0.7, "retail": 0.3},
        household_relocation=False,
    )


def test_total_money_is_conserved_in_closed_economy() -> None:
    """Money should stay inside the household-firm circuit when no hazard or entry/exit operates."""
    model = build_closed_economy_model()
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


def test_closed_economy_remains_economically_active() -> None:
    """The closure should preserve a live economy, not a zero-activity steady state."""
    model = build_closed_economy_model()

    for _ in range(40):
        model.step()

    df = model.results_to_dataframe()
    tail = df.tail(10)

    assert tail["Firm_Production"].mean() > 200.0
    assert tail["Household_Consumption"].mean() > 150.0
    assert tail["Household_Labor_Sold"].mean() > 250.0
    assert np.allclose(tail["Money_Drift"].to_numpy(), 0.0, atol=1e-6)


def test_firm_reorganization_preserves_total_money() -> None:
    """Learning-time firm replacement should reorganize balance sheets, not mint money."""
    model = build_closed_economy_model()

    for _ in range(6):
        model.step()

    failed_firm = model._firms[0]
    failed_firm.money = 0.0
    initial_total_money = model.total_money()
    model.current_step = max(model.current_step, 5)
    model.firm_learning_enabled = True

    model._apply_evolutionary_pressure()

    assert model.total_firm_replacements >= 1
    assert len(model._firms) > 0
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-6)


def test_learning_moves_toward_fitter_same_sector_peer() -> None:
    """Adaptive learning should imitate fitter peers rather than random-walk away from them."""
    model = build_closed_economy_model()

    retail_firms = [f for f in model._firms if f.sector == "retail"]
    learner, peer = retail_firms[:2]
    learner.learning_enabled = True
    peer.learning_enabled = True
    learner.MUTATION_RATE = 0.0

    learner.strategy = {key: 0.5 for key in learner.strategy}
    peer.strategy = {key: 2.0 for key in peer.strategy}
    learner.performance_history = [{"production": 1.0}] * 3
    peer.performance_history = [{"production": 10.0}] * 3

    learner._adapt_strategy()

    expected_value = (1.0 - learner.IMITATION_RATE) * 0.5 + learner.IMITATION_RATE * 2.0
    assert np.isclose(learner.fitness_score, 1.0, atol=1e-9)
    assert "wage_responsiveness" not in learner.strategy
    for key, value in learner.strategy.items():
        assert np.isclose(value, expected_value, atol=1e-9), key


def test_wage_target_uses_fixed_labor_share() -> None:
    """Wage targeting should use the fixed labor share rather than a learned parameter."""
    model = build_closed_economy_model()
    firm = model._firms[0]
    firm.wage_offer = 1.0
    firm.last_hired_labor = 2
    firm.revenue_last_step = 20.0
    starting_wage = firm.wage_offer
    expected_workers = firm.last_hired_labor
    expected_revenue = firm.revenue_last_step

    firm.step()

    target_wage = (expected_revenue / expected_workers) * firm.LABOR_SHARE
    expected_wage = starting_wage + 0.1 * (target_wage - starting_wage)
    assert np.isclose(firm.wage_offer, expected_wage, atol=1e-9)
