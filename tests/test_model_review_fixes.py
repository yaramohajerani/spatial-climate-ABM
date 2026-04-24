import json
import warnings
from pathlib import Path

import numpy as np

from model import EconomyModel


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_model(
    *,
    num_households: int = 300,
    consumption_ratios: dict | None = None,
    household_relocation: bool = False,
    adaptation_params: dict | None = None,
) -> EconomyModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return EconomyModel(
            num_households=num_households,
            hazard_events=[],
            seed=42,
            firm_topology_path=str(REPO_ROOT / "sample_firm_topology.json"),
            apply_hazard_impacts=False,
            adaptation_params=adaptation_params or {"enabled": False},
            consumption_ratios=consumption_ratios or {"services": 0.7, "retail": 0.3},
            household_relocation=household_relocation,
        )


def test_zero_household_payout_paths_do_not_leak_money() -> None:
    model = _build_model(num_households=0)
    firm = model._firms[0]

    firm.money = 200.0
    initial_total_money = model.total_money()

    capital_spending = firm._install_capital(25.0)
    firm._pay_dividends(distributable_earnings=40.0, operating_cash_reserve=0.0)
    firm.continuity_capital = 0.5
    firm.pending_adaptation_increment = 0.2
    maintenance_spending, investment_spending = firm._fund_adaptation_after_operations(
        available_cash=40.0,
        investable_profit=40.0,
    )

    assert capital_spending == 0.0
    assert maintenance_spending == 0.0
    assert investment_spending == 0.0
    assert np.isclose(firm.money, 200.0, atol=1e-9)
    assert np.isclose(model.total_money(), initial_total_money, atol=1e-9)


def test_hazard_relocation_refreshes_nearby_firms() -> None:
    model = _build_model(household_relocation=True)
    household = model._households[0]
    target_firm = next(firm for firm in model._firms if firm.sector == household.sector)

    household.nearby_firms = []
    model.land_coordinates = [target_firm.pos]
    household._relocate()

    assert household.pos == target_firm.pos
    assert target_firm in household.nearby_firms


def test_effective_configuration_metadata_tracks_filtered_demand_and_capital_floors() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        filtered_model = EconomyModel(
            num_households=300,
            hazard_events=[],
            seed=42,
            firm_topology_path=str(REPO_ROOT / "sample_firm_topology.json"),
            apply_hazard_impacts=False,
            adaptation_params={"enabled": False},
            consumption_ratios={"manufacturing": 0.4, "services": 0.6},
            household_relocation=False,
        )
        filtered_metadata = filtered_model.effective_configuration_metadata()
        default_model = EconomyModel(
            num_households=300,
            hazard_events=[],
            seed=42,
            firm_topology_path=str(REPO_ROOT / "sample_firm_topology.json"),
            apply_hazard_impacts=False,
            adaptation_params={"enabled": False},
            consumption_ratios={"services": 0.7, "retail": 0.3},
            household_relocation=False,
        )
        default_metadata = default_model.effective_configuration_metadata()

    assert json.loads(filtered_metadata["EffectiveConsumptionRatios"]) == {"services": 1.0}
    assert default_metadata["StartupCapitalFloorCount"] > 0
    assert any(
        "Ignoring non-final household consumption sectors" in str(item.message)
        for item in caught
    )
    assert any(
        "Raised startup capital" in str(item.message)
        for item in caught
    )


def test_price_sell_through_uses_stored_previous_period_availability() -> None:
    model = _build_model()
    firm = model._firms[0]

    firm.last_hired_labor = 1
    firm.revenue_last_step = 2.0
    firm.wage_offer = 1.0
    firm.price = 1.0
    firm.inventory_available_last_step = 20.0
    firm.sales_last_step = 10.0
    firm.inventory_output = 1.0
    firm.target_output = 0.0
    firm.damage_factor = 1.0
    firm.counterfactual_damage_factor = 1.0
    firm.connected_firms = []
    firm.INPUT_COEFF = 0.0
    firm.LABOR_COEFF = 0.6
    firm.employees.clear()

    firm.step()

    assert np.isclose(firm.price, 0.9404, atol=1e-6)


def test_finance_constraints_get_their_own_limiting_factor() -> None:
    model = _build_model()
    firm = next(f for f in model._firms if f.connected_firms)

    firm.money = 0.0
    firm.expected_sales = 1000.0
    firm.inventory_output = 0.0
    firm.capital_stock = 1000.0
    firm.damage_factor = 1.0
    firm.counterfactual_damage_factor = 1.0
    firm.price = 1.0
    firm.wage_offer = 1.0
    firm.INPUT_COEFF = 0.0
    for supplier in firm.connected_firms:
        supplier.price = 1.0
        supplier.inventory_output = 1_000_000.0

    firm.plan_operations()
    firm.employees.clear()
    firm.employees.extend([object()] * max(1, firm.target_labor + 1000))
    firm.step()

    assert firm.finance_constrained_this_step is True
    assert firm.limiting_factor == "finance"


def test_recovery_is_applied_after_close_step_and_not_before_planning() -> None:
    model = _build_model()
    firm = model._firms[0]
    observed_damage_during_planning: list[float] = []
    original_plan_operations = firm.plan_operations

    firm.damage_factor = 0.5
    firm.counterfactual_damage_factor = 0.4

    def wrapped_plan_operations() -> None:
        observed_damage_during_planning.append(firm.damage_factor)
        original_plan_operations()

    firm.plan_operations = wrapped_plan_operations

    model.step()

    assert observed_damage_during_planning
    assert np.isclose(observed_damage_during_planning[0], 0.5, atol=1e-9)
    assert firm.damage_factor > observed_damage_during_planning[0]


def test_inputs_from_different_supplier_sectors_do_not_substitute() -> None:
    model = _build_model()
    buyer = next(f for f in model._firms if f.sector == "services")
    agriculture_supplier = next(f for f in model._firms if f.sector == "agriculture")
    manufacturing_supplier = next(
        f for f in model._firms if f.sector == "manufacturing" and f is not buyer
    )

    buyer.connected_firms = [agriculture_supplier, manufacturing_supplier]
    buyer.inventory_inputs.clear()
    buyer.inventory_inputs[agriculture_supplier.unique_id] = 10.0
    buyer.inventory_inputs[manufacturing_supplier.unique_id] = 0.0
    buyer.target_output = 10.0
    buyer.demand_driven_output = 10.0
    buyer.capital_stock = 1_000.0
    buyer.capital_coeff = 1.0
    buyer.LABOR_COEFF = 0.0
    buyer.INPUT_COEFF = 1.0
    buyer.damage_factor = 1.0
    buyer.counterfactual_damage_factor = 1.0
    buyer.price = 1.0
    buyer.wage_offer = 1.0
    agriculture_supplier.inventory_output = 0.0
    manufacturing_supplier.inventory_output = 0.0
    manufacturing_supplier.raw_direct_loss_fraction_this_step = 0.3

    buyer.step()

    assert np.isclose(buyer.production, 0.0, atol=1e-9)
    assert buyer.raw_supplier_disruption_this_step > 0.0


def test_inputs_within_same_supplier_sector_remain_substitutable() -> None:
    model = _build_model()
    buyer = next(f for f in model._firms if f.sector == "services")
    manufacturing_suppliers = [f for f in model._firms if f.sector == "manufacturing" and f is not buyer]
    supplier_a, supplier_b = manufacturing_suppliers[:2]

    buyer.connected_firms = [supplier_a, supplier_b]
    buyer.inventory_inputs.clear()
    buyer.inventory_inputs[supplier_a.unique_id] = 10.0
    buyer.inventory_inputs[supplier_b.unique_id] = 0.0
    buyer.target_output = 10.0
    buyer.demand_driven_output = 10.0
    buyer.capital_stock = 1_000.0
    buyer.capital_coeff = 1.0
    buyer.LABOR_COEFF = 0.0
    buyer.INPUT_COEFF = 1.0
    buyer.damage_factor = 1.0
    buyer.counterfactual_damage_factor = 1.0
    buyer.price = 1.0
    buyer.wage_offer = 1.0
    supplier_a.inventory_output = 0.0
    supplier_b.inventory_output = 0.0

    buyer.step()

    assert np.isclose(buyer.production, 10.0, atol=1e-9)
    assert np.isclose(buyer.raw_supplier_disruption_this_step, 0.0, atol=1e-9)


def test_reserved_capacity_contracts_only_reserve_on_hand_inventory() -> None:
    model = _build_model(
        adaptation_params={
            "enabled": True,
            "adaptation_strategy": "reserved_capacity",
            "reserved_capacity_share": 0.35,
            "max_backup_suppliers": 5,
        },
    )
    buyer = next(firm for firm in model._firms if firm.connected_firms)
    supplier_sector = buyer.connected_firms[0].sector
    primary_ids = {supplier.unique_id for supplier in buyer.connected_firms}
    backup_supplier = next(
        firm
        for firm in model._firms
        if firm.sector == supplier_sector and firm.unique_id not in primary_ids and firm is not buyer
    )

    model.firm_adaptation_enabled = True
    model.adaptation_strategy = "reserved_capacity"
    for firm in model._firms:
        firm.adaptation_enabled = False
        firm.continuity_capital = 0.0
        firm.target_input_units = 0.0
        if firm.sector == supplier_sector and firm.unique_id not in primary_ids and firm is not backup_supplier:
            firm.inventory_output = 0.0

    buyer.adaptation_enabled = True
    buyer.continuity_capital = 1.0
    buyer.target_input_units = 100.0
    backup_supplier.inventory_output = 10.0
    backup_supplier.target_output = 100.0

    model._prepare_reserved_capacity_contracts()

    reserved_units = sum(
        quantity
        for supplier, quantity, _ in model.get_reserved_capacity_contracts(buyer)
        if supplier is backup_supplier
    )

    assert np.isclose(reserved_units, 10.0 * model.reserved_capacity_share, atol=1e-9)
