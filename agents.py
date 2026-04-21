from __future__ import annotations

from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from mesa import Agent
from collections import defaultdict

Coords = Tuple[int, int]


# ------------------------------------------------------------------------- #
# Optional forward references to avoid circular imports during type checking
# ------------------------------------------------------------------------- #
if TYPE_CHECKING:  # pragma: no cover
    from model import EconomyModel  # noqa: F401 – only for typing


class HouseholdAgent(Agent):
    """A household supplies one unit of labour, earns wages and buys goods."""

    # Fraction of wealth & capital lost when migrating to a new location
    RELOCATION_COST: float = 0.10
    CONSUMPTION_PROPENSITY_INCOME: float = 0.9
    CONSUMPTION_PROPENSITY_WEALTH: float = 0.02
    TARGET_CASH_BUFFER: float = 50.0
    SECTOR_MATCH_BONUS: float = 0.15
    SECTOR_MISMATCH_PENALTY: float = 0.20
    REMOTE_SEARCH_PENALTY: float = 0.10

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        money: float = 100.0,
        sector: str = "manufacturing",
    ) -> None:
        super().__init__(model)

        # Note: pos is handled by Mesa's grid.place_agent(), not set here
        self.money: float = money

        # Sector specialisation – household will preferentially work for firms in this sector
        self.sector: str = sector

        # Counter of consecutive steps without finding work (used for job-driven relocation)
        self._no_work_steps: int = 0

        # Trade-off coefficient between wage and distance when choosing employer.
        # Higher value → distance is more costly, so worker prefers closer firms.
        # Randomised per household to create heterogeneous behaviour.
        # Low values (0.01-0.1) model remote work being widely available.
        self.distance_cost: float = self.random.uniform(0.01, 0.1)

        # Aggregate statistics tracked per agent
        self.consumption: float = 0.0  # goods consumed (units)
        self.production: float = 0.0  # households don't produce goods but keep attr for consistency
        self.labor_sold: float = 0.0  # labour units sold this step
        self.labor_income_this_step: float = 0.0
        self.dividend_income_last_step: float = 0.0
        self.dividend_income_this_step: float = 0.0
        self.capital_income_last_step: float = 0.0
        self.capital_income_this_step: float = 0.0
        self.adaptation_income_last_step: float = 0.0
        self.adaptation_income_this_step: float = 0.0

        # Filled by the model after all agents are created
        self.nearby_firms: List["FirmAgent"] = []

    # ---------------- Internal helpers ------------------- #
    def _update_nearby_firms(self) -> None:
        """Refresh the list of nearby firms in the same sector within the model's work_radius."""

        radius = getattr(self.model, "work_radius", 3)
        self.nearby_firms.clear()
        # Use cached sector list from model for efficiency
        sector_firms = self.model._firms_by_sector.get(self.sector, [])
        for firm in sector_firms:
            dx = abs(self.pos[0] - firm.pos[0])
            dy = abs(self.pos[1] - firm.pos[1])
            if dx + dy <= radius:
                self.nearby_firms.append(firm)

    def _relocate_for_job(self) -> None:
        """Move closer to a random firm to improve employment prospects."""
        # Allow relocation to any firm (cross-sector employment is permitted)
        all_firms = self.model._firms
        if not all_firms:
            return  # no firms exist

        target_firm = self.random.choice(all_firms)
        fx, fy = target_firm.pos

        # Candidate cells within work radius of the target firm
        candidates: list[Coords] = []
        radius = getattr(self.model, "work_radius", 3)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                coord = (fx + dx, fy + dy)
                if coord in self.model.land_coordinates:
                    candidates.append(coord)

        if not candidates:
            return  # no land nearby

        new_pos = self.random.choice(candidates)
        if new_pos == self.pos:
            return  # already there

        self.model.grid.move_agent(self, new_pos)

        # Refresh nearby firms after moving
        self._update_nearby_firms()

    def _score_firm(
        self,
        firm: "FirmAgent",
        *,
        nearby_ids: set[int],
        remote_penalty: float = 0.0,
    ) -> float:
        """Return the household's utility from applying to a firm."""
        dx = abs(self.pos[0] - firm.pos[0])
        dy = abs(self.pos[1] - firm.pos[1])
        dist = dx + dy
        utility = firm.wage_offer - self.distance_cost * dist
        if firm.sector == self.sector:
            utility += self.SECTOR_MATCH_BONUS
        else:
            utility -= self.SECTOR_MISMATCH_PENALTY
        if firm.unique_id not in nearby_ids:
            utility -= remote_penalty
        return utility

    def _try_hire_from_ranked_list(self, firms: List["FirmAgent"], *, remote_penalty: float = 0.0) -> bool:
        """Attempt to sell labour to the best-ranked firm in ``firms``."""
        if not firms:
            return False

        nearby_ids = {firm.unique_id for firm in self.nearby_firms}
        scored = [
            (self._score_firm(firm, nearby_ids=nearby_ids, remote_penalty=remote_penalty), firm)
            for firm in firms
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        for _, firm in scored:
            if firm.hire_labor(self, firm.wage_offer):
                self.labor_sold += 1
                return True
        return False

    # ---------------- Mesa API ---------------- #
    def supply_labor(self) -> None:
        """Reset household state, optionally relocate, and sell labour."""

        # Carry passive income distributed after last period's goods market into the
        # current consumption decision, then reset current-period counters.
        self.dividend_income_last_step = self.dividend_income_this_step
        self.capital_income_last_step = self.capital_income_this_step
        self.adaptation_income_last_step = self.adaptation_income_this_step
        self.dividend_income_this_step = 0.0
        self.capital_income_this_step = 0.0
        self.adaptation_income_this_step = 0.0
        self.labor_income_this_step = 0.0

        # Reset per-step statistics
        self.production = 0.0
        self.labor_sold = 0.0
        self.consumption = 0.0

        # ---------------- Heuristic relocation decision --------------- #
        # Only relocate when flood depth exceeds 0.5m (significant flooding)
        if self.model.household_relocation_enabled and self._get_local_hazard() > 0.5:
            self._relocate()

        # 1. Choose employer based on a staged search:
        # prefer nearby same-sector firms first, then broaden to the full market.
        # This keeps labor tied to the local production structure while preserving
        # cross-sector fallback when the preferred market is weak.
        all_firms = self.model._firms
        if all_firms:
            primary_firms = self.nearby_firms if self.nearby_firms else [
                firm for firm in all_firms if firm.sector == self.sector
            ]
            if not self._try_hire_from_ranked_list(primary_firms):
                secondary_firms = [
                    firm for firm in all_firms
                    if firm not in primary_firms
                ]
                self._try_hire_from_ranked_list(
                    secondary_firms,
                    remote_penalty=self.REMOTE_SEARCH_PENALTY,
                )

    def consume_goods(self) -> None:
        """Buy final goods after firms have completed the current production cycle."""

        # Closed-economy household demand is driven by current labour income,
        # recently distributed firm payouts, and a small propensity to spend out
        # of accumulated wealth. This is the minimal circular-flow closure:
        # households buy goods from firms, but they also receive wages,
        # dividends, and capital-service income from firms.
        disposable_income = (
            self.labor_income_this_step
            + self.dividend_income_last_step
            + self.capital_income_last_step
            + self.adaptation_income_last_step
        )
        wealth_draw = max(0.0, self.money - self.TARGET_CASH_BUFFER)
        consumption_budget = (
            self.CONSUMPTION_PROPENSITY_INCOME * disposable_income
            + self.CONSUMPTION_PROPENSITY_WEALTH * wealth_draw
        )

        if consumption_budget > 0 and self.money > 0:
            # Cap at what we can actually afford.
            consumption_budget = min(consumption_budget, self.money)

            # Only final-good sectors are eligible for household demand.
            consumption_ratios = self.model.get_final_consumption_ratios()
            if consumption_ratios:
                # Group firms by sector
                firms_by_sector: dict[str, list] = {}
                for f in self.model._firms:
                    if f.inventory_output > 0 and f.sector in consumption_ratios:
                        sector = f.sector
                        if sector not in firms_by_sector:
                            firms_by_sector[sector] = []
                        firms_by_sector[sector].append(f)

                total_ratio = sum(consumption_ratios.values())
                if total_ratio > 0:
                    # Allocate budget to each final-good sector and buy
                    for sector, ratio in consumption_ratios.items():
                        if sector not in firms_by_sector:
                            continue  # no firms in this sector with inventory

                        sector_budget = consumption_budget * (ratio / total_ratio)
                        if sector_budget <= 0:
                            continue

                        # Sort firms in this sector by price (cheapest first)
                        sector_firms = sorted(firms_by_sector[sector], key=lambda f: f.price)

                        remaining_sector_budget = sector_budget
                        for seller in sector_firms:
                            if remaining_sector_budget <= 0:
                                break
                            # Allow fractional purchases - buy whatever we can afford
                            max_quantity = remaining_sector_budget / seller.price
                            qty_bought = seller.sell_goods_to_household(self, quantity=max_quantity)
                            if qty_bought > 0:
                                self.consumption += qty_bought
                                remaining_sector_budget -= qty_bought * seller.price

        # ---------------- End-of-step unemployment tracking ------------- #
        if self.labor_sold == 0:
            self._no_work_steps += 1
        else:
            self._no_work_steps = 0

        if self._no_work_steps >= 3 and self.model.household_relocation_enabled:
            # Could not find work for 3 consecutive steps → move closer to another firm of same sector
            self._relocate_for_job()
            self._no_work_steps = 0  # reset after relocation

    # ---------------- Mesa API ---------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Compatibility wrapper for one-shot scheduling."""

        self.supply_labor()
        self.consume_goods()

    def _get_local_hazard(self) -> float:
        """Return the hazard value at the agent's current cell."""
        return self.model.hazard_map.get(self.pos, 0.0)

    def _relocate(self) -> None:
        """Move the household to a random land cell with low current hazard."""
        safe_cells = [c for c in self.model.land_coordinates if self.model.hazard_map.get(c, 0.0) <= 0.5]
        if not safe_cells:
            safe_cells = self.model.land_coordinates  # fall back to any land
        new_pos = self.random.choice(safe_cells)
        self.model.grid.move_agent(self, new_pos)
        self._update_nearby_firms()


    # ---------------- End of Household.step() bookkeeping -------- #


class FirmAgent(Agent):
    """A firm with a simple Leontief production function and local trade."""

    # Productive capacity is a reduced-form balance-sheet stock rather than a
    # separate modeled capital-goods market. One unit of retained earnings can
    # therefore finance one unit of installed capacity.
    CAPITAL_INSTALLATION_COST: float = 1.0

    # Sector-specific Leontief technical coefficients (units required per unit output)
    # Lower coefficient = higher productivity (less input needed per output)
    SECTOR_COEFFICIENTS: dict = {
        # Commodity (raw materials): labor-intensive extraction, VERY capital-intensive (heavy equipment)
        "commodity": {"labor": 0.6, "input": 0.0, "capital": 0.7},
        "agriculture": {"labor": 0.6, "input": 0.0, "capital": 0.7},
        # Manufacturing: automated (low labor), capital-intensive, high input needs
        "manufacturing": {"labor": 0.3, "input": 0.6, "capital": 0.6},
        # Retail: moderate labor (modern retail has automation), low capital, moderate inputs
        "retail": {"labor": 0.5, "input": 0.4, "capital": 0.2},
        # Legacy support for old topology files
        "wholesale": {"labor": 0.5, "input": 0.4, "capital": 0.2},
        "services": {"labor": 0.9, "input": 0.1, "capital": 0.1},
    }
    # Default coefficients for unknown sectors
    DEFAULT_COEFFICIENTS: dict = {"labor": 0.5, "input": 0.5, "capital": 0.5}

    INVENTORY_BUFFER_RATIO: float = 0.25
    LIQUIDITY_BUFFER_RATIO: float = 0.15
    MIN_LIQUIDITY_BUFFER: float = 10.0
    WORKING_CAPITAL_CREDIT_REVENUE_SHARE: float = 1.0
    LABOR_SHARE: float = 0.5  # fixed labour share of revenue in wage targeting
    NO_WORKER_WAGE_PREMIUM: float = 1.02
    ADAPTATION_EXPECTED_WEIGHT: float = 0.5
    ADAPTATION_LOCAL_WEIGHT: float = 0.3
    ADAPTATION_SUPPLIER_WEIGHT: float = 0.2

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        sector: str = "manufacturing",
        capital_stock: float = 100.0,
    ) -> None:
        super().__init__(model)

        # Note: pos is handled by Mesa's grid.place_agent(), not set here
        self.sector = sector
        self.capital_stock = capital_stock

        # Set sector-specific production coefficients
        coeffs = self.SECTOR_COEFFICIENTS.get(sector, self.DEFAULT_COEFFICIENTS)
        self.LABOR_COEFF: float = coeffs["labor"]
        self.INPUT_COEFF: float = coeffs["input"]
        self.CAPITAL_COEFF: float = coeffs["capital"]

        # Set starting price based on sector and coefficients
        # Initial price should cover costs plus margin
        # cost = LABOR_COEFF * wage + INPUT_COEFF * input_price
        # For commodity (no inputs): cost = 0.6 * 1 = 0.6, price ~0.8
        # For manufacturing: cost = 0.3 * 1 + 0.6 * 0.8 = 0.78, price ~1.0
        # For retail: cost = 0.8 * 1 + 0.4 * 1.0 = 1.2, price ~1.5
        base_price_by_sector = {
            "commodity": 0.8,
            "agriculture": 0.8,
            "manufacturing": 1.0,
            "retail": 1.5,
            "wholesale": 1.5,
            "services": 1.2,
        }
        self.price: float = float(base_price_by_sector.get(self.sector, 1.2))
        self.money: float = 100.0

        # Firm-specific wage offer (labour price) – starts at the model's base wage
        self.wage_offer: float = model.mean_wage if hasattr(model, "mean_wage") else 1.0

        self.last_hired_labor: int = 0  # employees hired in previous step

        # Input inventory keyed by supplier AgentID
        self.inventory_inputs: dict[int, float] = defaultdict(float)
        self.inventory_output: float = 0.0  # finished goods

        # Links to other agents (filled by model)
        self.connected_firms: List["FirmAgent"] = []
        self.employees: List[HouseholdAgent] = []

        # Statistics
        self.production: float = 0.0  # units produced this step
        self.consumption: float = 0.0  # units of inputs consumed this step

        # Cumulative damage to productive capacity (1 = undamaged)
        self.damage_factor: float = 1.0
        self.counterfactual_damage_factor: float = 1.0

        # Sales tracking for pricing and wage mechanisms
        self.sales_last_step: float = 0.0
        self.revenue_last_step: float = 0.0
        self.sales_this_step: float = 0.0
        self.revenue_this_step: float = 0.0
        self.household_sales_last_step: float = 0.0
        self.household_sales_this_step: float = 0.0
        self.inventory_available_last_step: float = 0.0
        self.wage_bill_this_step: float = 0.0
        self.input_spend_this_step: float = 0.0
        self.depreciation_this_step: float = 0.0
        self.operating_profit_this_step: float = 0.0
        self.profit_this_step: float = 0.0
        self.direct_loss_expense_this_step: float = 0.0
        self.dividends_paid_this_step: float = 0.0
        self.investment_spending_this_step: float = 0.0
        self.adaptation_spending_this_step: float = 0.0
        self.adaptation_investment_this_step: float = 0.0
        self.adaptation_maintenance_this_step: float = 0.0
        self.counterfactual_direct_loss_this_step: float = 0.0
        self.realized_direct_loss_this_step: float = 0.0
        self.raw_direct_loss_fraction_this_step: float = 0.0
        self.adapted_direct_loss_fraction_this_step: float = 0.0
        self.supplier_disruption_this_step: float = 0.0
        self.raw_supplier_disruption_this_step: float = 0.0
        self.hazard_operating_shortfall_this_step: float = 0.0
        self.continuity_gap_coverage_this_step: float = 0.0
        self.continuity_input_coverage_this_step: float = 0.0
        self.backup_purchases_this_step: float = 0.0
        self.reserved_capacity_purchases_this_step: float = 0.0
        self.reserved_capacity_price_savings_this_step: float = 0.0

        # Capital coefficient parameters
        self.original_capital_coeff: float = self.CAPITAL_COEFF
        self.capital_coeff: float = self.CAPITAL_COEFF

        # Demand-planning state set each step by plan_operations()
        self.target_output: float = 0.0
        self.target_labor: int = 0
        self.target_input_units: float = 0.0
        self.expected_sales: float = 0.0
        self.demand_driven_output: float = 0.0
        self.base_inventory_target: float = 1.0
        self.base_capital_target: float = self.capital_stock
        self.target_capital_stock: float = self.capital_stock
        self.no_hazard_target_output: float = 0.0
        self.no_hazard_demand_driven_output: float = 0.0
        self.no_hazard_damage_factor: float = 1.0
        self._liquidity_buffer: float = 0.0
        self.required_working_capital: float = 0.0
        self.working_capital_credit_limit: float = 0.0
        self.working_capital_credit_used_this_step: float = 0.0
        self.finance_constrained_this_step: bool = False
        self.pre_hazard_damage_factor: float = 1.0
        self.pre_hazard_capital_stock: float = self.capital_stock
        self.pre_hazard_inventory_output: float = self.inventory_output
        self.pre_hazard_inventory_inputs: dict[int, float] = {}

        adaptation_config = getattr(model, "adaptation_config", {})
        self.adaptation_enabled: bool = getattr(model, "firm_adaptation_enabled", True)
        self.adaptation_strategy: str = str(adaptation_config.get("adaptation_strategy", "backup_suppliers"))
        self.decision_interval: int = int(adaptation_config.get("decision_interval", 4))
        self.ewma_alpha: float = float(adaptation_config.get("ewma_alpha", 0.2))
        self.resilience_decay: float = float(
            adaptation_config.get(
                "continuity_decay",
                adaptation_config.get("resilience_decay", 0.01),
            )
        )
        self.maintenance_cost_rate: float = float(adaptation_config.get("maintenance_cost_rate", 0.005))
        sensitivity_min = float(adaptation_config.get("adaptation_sensitivity_min", 2.0))
        sensitivity_max = float(adaptation_config.get("adaptation_sensitivity_max", 4.0))
        if sensitivity_max < sensitivity_min:
            sensitivity_max = sensitivity_min
        self.adaptation_sensitivity: float = float(
            self.random.uniform(sensitivity_min, sensitivity_max)
        )
        self.max_adaptation_increment: float = float(
            adaptation_config.get("max_adaptation_increment", 0.25)
        )

        self.continuity_capital: float = 0.0
        self.expected_direct_loss_ewma: float = 0.0
        self.realized_direct_loss_ewma: float = 0.0
        self.local_observed_loss_ewma: float = 0.0
        self.supplier_disruption_ewma: float = 0.0
        self.expected_operating_shortfall_ewma: float = 0.0
        self.local_observed_shortfall_ewma: float = 0.0
        self.last_adaptation_action: str = "dormant"
        self.last_adaptation_target: float = 0.0
        self.last_perceived_hazard_risk: float = 0.0
        self.last_continuity_target: float = 0.0
        self.last_perceived_continuity_risk: float = 0.0
        self.pending_adaptation_increment: float = 0.0
        self.adaptation_update_count: int = 0

        # Exposure-state diagnostics for cascading-risk analysis.
        self.ever_directly_hit: bool = False
        self.ever_indirectly_disrupted_before_direct_hit: bool = False

        # Survival tracking for firm reorganization
        self.survival_time: int = 0

    # ---------------- Adaptation System Methods ------------------------- #
    def begin_period_adaptation(self) -> None:
        """Reset period adaptation accounting before hazards are sampled."""

        self.adaptation_spending_this_step = 0.0
        self.adaptation_investment_this_step = 0.0
        self.adaptation_maintenance_this_step = 0.0
        self.counterfactual_direct_loss_this_step = 0.0
        self.realized_direct_loss_this_step = 0.0
        self.direct_loss_expense_this_step = 0.0
        self.raw_direct_loss_fraction_this_step = 0.0
        self.adapted_direct_loss_fraction_this_step = 0.0
        self.supplier_disruption_this_step = 0.0
        self.raw_supplier_disruption_this_step = 0.0
        self.hazard_operating_shortfall_this_step = 0.0
        self.continuity_gap_coverage_this_step = 0.0
        self.continuity_input_coverage_this_step = 0.0
        self.backup_purchases_this_step = 0.0
        self.reserved_capacity_purchases_this_step = 0.0
        self.reserved_capacity_price_savings_this_step = 0.0

        if not self.adaptation_enabled:
            return

        self.pending_adaptation_increment = 0.0
        self.continuity_capital = max(0.0, self.continuity_capital * (1.0 - self.resilience_decay))

    @property
    def resilience_capital(self) -> float:
        """Backward-compatible alias for the continuity stock."""
        return self.continuity_capital

    @resilience_capital.setter
    def resilience_capital(self, value: float) -> None:
        self.continuity_capital = float(value)

    def _perceived_hazard_risk(self) -> float:
        """Perceived continuity risk from own and nearby hazard-induced output gaps."""
        risk = max(self.expected_operating_shortfall_ewma, self.local_observed_shortfall_ewma)
        return float(min(1.0, max(0.0, risk)))

    def _target_resilience_from_risk(self, perceived_risk: float) -> float:
        """Map quarterly continuity risk into a continuity-capital target.

        Hazard-induced operating shortfall is measured per simulation step
        (quarterly in the current experiments). Firms react to the recurring
        annual burden implied by those shortfalls rather than to one quarter in
        isolation, so we annualize the perceived signal before applying the
        firm-specific continuity sensitivity.
        """
        steps_per_year = max(1, int(getattr(self.model, "steps_per_year", 4)))
        annualized_risk = min(1.0, max(0.0, perceived_risk) * steps_per_year)
        return float(min(1.0, self.adaptation_sensitivity * annualized_risk))

    def _select_adaptation_action(self) -> None:
        """Choose a continuity-capital target from adaptive hazard expectations."""
        perceived_risk = self._perceived_hazard_risk()
        self.last_perceived_hazard_risk = perceived_risk
        self.last_perceived_continuity_risk = perceived_risk

        target = self._target_resilience_from_risk(perceived_risk)
        self.last_adaptation_target = target
        self.last_continuity_target = target
        gap = max(0.0, target - self.continuity_capital)
        increment = min(gap, self.max_adaptation_increment)
        self.pending_adaptation_increment = increment

        if perceived_risk <= 1e-9 and self.continuity_capital <= 1e-9:
            self.last_adaptation_action = "dormant"
        elif increment > 1e-9:
            self.last_adaptation_action = "adjust"
            self.adaptation_update_count += 1
        else:
            self.last_adaptation_action = "hold"

    def _adaptation_scale(self) -> float:
        return max(1.0, self.base_capital_target)

    def _uses_adaptation_strategy(self, strategy: str) -> bool:
        return self.adaptation_strategy == strategy

    def _effective_inventory_buffer_ratio(self) -> float:
        buffer_ratio = self.INVENTORY_BUFFER_RATIO
        if self._uses_adaptation_strategy("stockpiling") and self.continuity_capital > 0:
            buffer_ratio += self.continuity_capital * self.last_perceived_hazard_risk
        return buffer_ratio

    def _has_hazard_disruption_signal(self) -> bool:
        """Return whether this firm shows direct or indirect hazard stress.

        The continuity module should react not only to directly flooded suppliers
        but also to suppliers that are short because they were disrupted further
        upstream. Including both current-period and recent-period disruption
        indicators preserves the hazard-conditioned interpretation while making
        multi-hop cascades visible to buyers.
        """
        eps = 1e-9
        return bool(
            self.raw_direct_loss_fraction_this_step > eps
            or self.adapted_direct_loss_fraction_this_step > eps
            or self.damage_factor < 0.999
            or self.counterfactual_damage_factor < 0.999
            or self.raw_supplier_disruption_this_step > eps
            or self.supplier_disruption_this_step > eps
            or self.hazard_operating_shortfall_this_step > eps
            or self.expected_operating_shortfall_ewma > eps
            or self.supplier_disruption_ewma > eps
        )

    def _primary_supplier_shortage_is_hazard_related(self) -> bool:
        return any(supplier._has_hazard_disruption_signal() for supplier in self.connected_firms)

    def _backup_supplier_count(self) -> int:
        if self.continuity_capital <= 0:
            return 0
        max_backup_count = max(0, int(getattr(self.model, "max_backup_suppliers", 5)))
        if max_backup_count <= 0:
            return 0
        return max(1, int(self.continuity_capital * max_backup_count))

    def _reserved_capacity_supplier_count(self) -> int:
        """Return the number of standby suppliers to contract for reserved capacity.

        Reserved capacity is a standing redundancy mechanism, so moderate
        continuity capital should diversify across more than a single supplier
        even before the stock becomes large enough to round up under the spot
        backup rule.
        """
        if self.continuity_capital <= 0:
            return 0
        max_backup_count = max(0, int(getattr(self.model, "max_backup_suppliers", 5)))
        if max_backup_count <= 0:
            return 0
        return max(1, int(np.ceil(self.continuity_capital * max_backup_count)))

    def _purchase_from_backup_suppliers(self, remaining_inputs_needed: float) -> tuple[float, float]:
        if remaining_inputs_needed <= 1e-9 or self._operating_cash_capacity() <= 1e-9:
            return remaining_inputs_needed, 0.0

        backup_purchases = 0.0
        backup_suppliers = self.model.find_backup_suppliers(self, self._backup_supplier_count())
        for supplier in backup_suppliers:
            if remaining_inputs_needed <= 1e-9 or self._operating_cash_capacity() <= 1e-9:
                break
            bought = supplier.sell_goods_to_firm(self, remaining_inputs_needed)
            if bought > 0:
                remaining_inputs_needed -= bought
                backup_purchases += bought
        return remaining_inputs_needed, backup_purchases

    def _reserved_capacity_target_units(self) -> float:
        """Return desired reserved input coverage for the current period.

        Continuity capital already embeds the firm's accumulated hazard beliefs
        through the adaptation rule. Multiplying again by the current-period
        risk signal made contract sizes collapse toward zero in practice, so
        reserved capacity now scales with planned input needs and the standing
        continuity stock itself.
        """
        if self.target_input_units <= 0 or self.continuity_capital <= 0:
            return 0.0
        return max(0.0, self.target_input_units * self.continuity_capital)

    def _reserved_capacity_price_cap(self) -> float:
        if not self.connected_firms:
            return float("inf")
        anchor_price = float(np.mean([max(s.price, 0.5) for s in self.connected_firms]))
        markup_cap = float(getattr(self.model, "reserved_capacity_markup_cap", 0.1))
        return max(0.5, anchor_price * (1.0 + markup_cap))

    def _purchase_from_reserved_capacity(self, remaining_inputs_needed: float) -> tuple[float, float, float]:
        if remaining_inputs_needed <= 1e-9 or self._operating_cash_capacity() <= 1e-9:
            return remaining_inputs_needed, 0.0, 0.0

        reserved_purchases = 0.0
        reserved_price_savings = 0.0
        for supplier, reserved_units, contract_unit_price in self.model.get_reserved_capacity_contracts(self):
            if remaining_inputs_needed <= 1e-9 or self._operating_cash_capacity() <= 1e-9:
                break
            requested_quantity = min(remaining_inputs_needed, reserved_units)
            bought = supplier.sell_goods_to_firm(
                self,
                requested_quantity,
                unit_price=contract_unit_price,
                reservation_buyer_id=self.unique_id,
            )
            if bought > 0:
                remaining_inputs_needed -= bought
                reserved_purchases += bought
                reserved_price_savings += bought * max(0.0, supplier.price - contract_unit_price)
        return remaining_inputs_needed, reserved_purchases, reserved_price_savings

    def _fund_adaptation_after_operations(
        self,
        *,
        available_cash: float,
        investable_profit: float,
    ) -> tuple[float, float]:
        """Fund maintenance and planned resilience investment after operations close.

        Adaptation no longer draws from working capital before hiring and procurement.
        Instead, the chosen action is financed from residual post-operations cash and
        only affects resilience capital going into the next period.
        """
        if not self.model._households:
            self.pending_adaptation_increment = 0.0
            return 0.0, 0.0

        scale = self._adaptation_scale()
        desired_spending = self.maintenance_cost_rate * self.continuity_capital * self._adaptation_scale()
        maintenance_spending = min(desired_spending, available_cash, investable_profit)
        if maintenance_spending > 0:
            self.money -= maintenance_spending
            self.adaptation_spending_this_step += maintenance_spending
            self.adaptation_maintenance_this_step += maintenance_spending
            self.model.distribute_household_income(maintenance_spending, income_kind="adaptation")
            available_cash -= maintenance_spending
            investable_profit -= maintenance_spending

        investment_spending = 0.0
        desired_increment = self.pending_adaptation_increment
        if desired_increment > 0 and available_cash > 0 and investable_profit > 0:
            desired_spending = desired_increment * scale
            investment_spending = min(desired_spending, available_cash, investable_profit)
            if investment_spending > 0:
                realized_increment = investment_spending / scale
                self.money -= investment_spending
                self.continuity_capital = min(1.0, self.continuity_capital + realized_increment)
                self.adaptation_spending_this_step += investment_spending
                self.adaptation_investment_this_step += investment_spending
                self.model.distribute_household_income(investment_spending, income_kind="adaptation")

        self.pending_adaptation_increment = 0.0
        return maintenance_spending, investment_spending

    def _available_cash_after_reserve(self, operating_cash_reserve: float) -> float:
        return max(0.0, self.money - operating_cash_reserve)

    def _operating_credit_floor(self) -> float:
        """Minimum cash balance allowed for operating outlays this period."""
        return self._liquidity_buffer - self.working_capital_credit_limit

    def _operating_cash_capacity(self) -> float:
        """Remaining payroll/input capacity including bounded working-capital credit."""
        return max(0.0, self.money - self._operating_credit_floor())

    def _update_peak_working_capital_credit_use(self) -> None:
        """Track the peak draw on the firm's bounded operating overdraft."""
        credit_draw = max(0.0, self._liquidity_buffer - self.money)
        self.working_capital_credit_used_this_step = max(
            self.working_capital_credit_used_this_step,
            min(self.working_capital_credit_limit, credit_draw),
        )

    def _spend_operating_cash(self, amount: float) -> bool:
        """Fund payroll/input spending from cash above buffer plus bounded overdraft."""
        if amount <= 0:
            return True
        if amount > self._operating_cash_capacity() + 1e-9:
            return False
        self.money -= amount
        self._update_peak_working_capital_credit_use()
        return True

    def _install_capital(self, spending: float) -> float:
        if spending <= 0 or not self.model._households:
            return 0.0
        installable_capital = spending / self.CAPITAL_INSTALLATION_COST
        self.money -= spending
        self.capital_stock += installable_capital
        self.investment_spending_this_step += spending
        self.model.distribute_household_income(
            spending,
            income_kind="capital",
        )
        return spending

    def _allocate_positive_profit(self, *, positive_profit: float, operating_cash_reserve: float) -> None:
        """Allocate positive profits across capital maintenance, expansion, and adaptation."""
        investment_share = 0.5
        available_cash = self._available_cash_after_reserve(operating_cash_reserve)

        maintenance_capital_gap = max(0.0, self.base_capital_target - self.capital_stock)
        maintenance_spending = min(
            positive_profit,
            maintenance_capital_gap * self.CAPITAL_INSTALLATION_COST,
            available_cash,
        )
        self._install_capital(maintenance_spending)

        available_cash = self._available_cash_after_reserve(operating_cash_reserve)
        residual_profit = max(0.0, positive_profit - self.investment_spending_this_step)
        expansion_capital_gap = max(0.0, self.target_capital_stock - self.capital_stock)
        discretionary_investment_spending = min(
            residual_profit * investment_share,
            expansion_capital_gap * self.CAPITAL_INSTALLATION_COST,
            available_cash,
        )
        self._install_capital(discretionary_investment_spending)

        available_cash = self._available_cash_after_reserve(operating_cash_reserve)
        investable_profit = max(0.0, positive_profit - self.investment_spending_this_step)
        if self.adaptation_enabled and available_cash > 0 and investable_profit > 0:
            self._fund_adaptation_after_operations(
                available_cash=available_cash,
                investable_profit=investable_profit,
            )

    def _pay_dividends(self, *, positive_profit: float, operating_cash_reserve: float) -> None:
        if not self.model._households:
            return
        available_cash = self._available_cash_after_reserve(operating_cash_reserve)
        desired_dividends = min(
            max(0.0, positive_profit - self.investment_spending_this_step - self.adaptation_spending_this_step),
            available_cash,
        )
        if desired_dividends > 0:
            self.money -= desired_dividends
            self.dividends_paid_this_step = desired_dividends
            self.model.distribute_household_income(
                desired_dividends,
                income_kind="dividend",
            )

    def _recovery_liquidity_anchor(self) -> float:
        revenue_anchor = max(
            self.expected_sales,
            self.sales_last_step,
            self.household_sales_last_step,
            1.0,
        ) * max(self.price, 0.5)
        capital_anchor = max(self.base_capital_target, self.target_capital_stock, 1.0) * max(self.price, 0.5)
        return max(50.0, revenue_anchor, capital_anchor, self.required_working_capital)

    def _recovery_rate_from_liquidity(self, liquidity_proxy: float) -> float:
        liquidity_anchor = max(1e-6, self._recovery_liquidity_anchor())
        liquidity_ratio = min(1.0, max(0.0, liquidity_proxy) / liquidity_anchor)
        return 0.2 + 0.3 * liquidity_ratio

    def _counterfactual_liquidity_proxy(self) -> float:
        excess_direct_loss = max(
            0.0,
            self.counterfactual_direct_loss_this_step - self.realized_direct_loss_this_step,
        )
        return max(0.0, self.money - excess_direct_loss)

    def _apply_damage_recovery(self) -> None:
        actual_recovery_rate = self._recovery_rate_from_liquidity(self.money)
        self.damage_factor += (1.0 - self.damage_factor) * actual_recovery_rate
        self.damage_factor = min(1.0, max(0.0, self.damage_factor))

        counterfactual_recovery_rate = self._recovery_rate_from_liquidity(
            self._counterfactual_liquidity_proxy()
        )
        self.counterfactual_damage_factor += (
            1.0 - self.counterfactual_damage_factor
        ) * counterfactual_recovery_rate
        self.counterfactual_damage_factor = min(1.0, max(0.0, self.counterfactual_damage_factor))

    def _update_post_step_state(self) -> None:
        """Update direct-loss diagnostics, continuity beliefs, and demand expectations."""
        downtime_value = max(
            self.expected_sales,
            self.sales_last_step,
            self.production,
            1.0,
        ) * max(self.price, 0.5)
        raw_recovery_drag = max(0.0, 1.0 - self.counterfactual_damage_factor)
        adapted_recovery_drag = max(0.0, 1.0 - self.damage_factor)
        self.counterfactual_direct_loss_this_step += raw_recovery_drag * downtime_value
        self.realized_direct_loss_this_step += adapted_recovery_drag * downtime_value

        alpha = self.ewma_alpha
        observed_raw_loss = max(self.raw_direct_loss_fraction_this_step, raw_recovery_drag)
        observed_adapted_loss = max(self.adapted_direct_loss_fraction_this_step, adapted_recovery_drag)
        local_observed_loss = self.model.get_local_observed_loss_fraction(self)
        local_observed_shortfall = self.model.get_local_observed_shortfall_fraction(self)
        self.expected_direct_loss_ewma = (1.0 - alpha) * self.expected_direct_loss_ewma + alpha * observed_raw_loss
        self.realized_direct_loss_ewma = (1.0 - alpha) * self.realized_direct_loss_ewma + alpha * observed_adapted_loss
        self.local_observed_loss_ewma = (
            (1.0 - alpha) * self.local_observed_loss_ewma
            + alpha * local_observed_loss
        )
        self.expected_operating_shortfall_ewma = (
            (1.0 - alpha) * self.expected_operating_shortfall_ewma
            + alpha * self.hazard_operating_shortfall_this_step
        )
        self.local_observed_shortfall_ewma = (
            (1.0 - alpha) * self.local_observed_shortfall_ewma
            + alpha * local_observed_shortfall
        )
        self.supplier_disruption_ewma = (
            (1.0 - alpha) * self.supplier_disruption_ewma
            + alpha * self.supplier_disruption_this_step
        )
        if self.supplier_disruption_this_step > 0 and not self.ever_directly_hit:
            self.ever_indirectly_disrupted_before_direct_hit = True

        self.expected_sales = 0.7 * self.expected_sales + 0.3 * self.sales_this_step
        self.sales_last_step = self.sales_this_step
        self.household_sales_last_step = self.household_sales_this_step
        self.revenue_last_step = self.revenue_this_step
        self.sales_this_step = 0.0
        self.household_sales_this_step = 0.0
        self.revenue_this_step = 0.0

    def estimate_direct_value_at_risk(self) -> float:
        avg_input_price = float(np.mean([s.price for s in self.connected_firms])) if self.connected_firms else self.price
        input_units = sum(self.inventory_inputs.values())
        downtime_units = max(self.expected_sales, self.sales_last_step, self.target_output, 1.0)
        inventory_value = self.inventory_output * max(self.price, 0.5)
        input_value = input_units * max(avg_input_price, 0.5)
        downtime_value = downtime_units * max(self.price, 0.5)
        return max(1.0, self.capital_stock + inventory_value + input_value + downtime_value)

    def get_adapted_loss_fraction(self, raw_loss_fraction: float) -> float:
        """Return adapted loss fraction, strategy-dependent.

        For capital_hardening, continuity_capital directly attenuates physical
        damage.  For other strategies this is a pass-through.
        """
        if self._uses_adaptation_strategy("capital_hardening") and self.continuity_capital > 0:
            return raw_loss_fraction * (1.0 - self.continuity_capital)
        return raw_loss_fraction

    def record_direct_losses(
        self,
        raw_loss_fraction: float,
        adapted_loss_fraction: float,
    ) -> None:
        direct_value_at_risk = self.estimate_direct_value_at_risk()
        self.raw_direct_loss_fraction_this_step = max(self.raw_direct_loss_fraction_this_step, raw_loss_fraction)
        self.adapted_direct_loss_fraction_this_step = max(self.adapted_direct_loss_fraction_this_step, adapted_loss_fraction)
        self.counterfactual_direct_loss_this_step += raw_loss_fraction * direct_value_at_risk
        self.realized_direct_loss_this_step += adapted_loss_fraction * direct_value_at_risk
        if raw_loss_fraction > 0:
            self.ever_directly_hit = True

    def copy_adaptation_state_from(self, parent: "FirmAgent") -> None:
        self.adaptation_strategy = parent.adaptation_strategy
        self.continuity_capital = parent.continuity_capital
        self.expected_direct_loss_ewma = parent.expected_direct_loss_ewma
        self.realized_direct_loss_ewma = parent.realized_direct_loss_ewma
        self.local_observed_loss_ewma = parent.local_observed_loss_ewma
        self.supplier_disruption_ewma = parent.supplier_disruption_ewma
        self.expected_operating_shortfall_ewma = parent.expected_operating_shortfall_ewma
        self.local_observed_shortfall_ewma = parent.local_observed_shortfall_ewma
        self.adaptation_sensitivity = parent.adaptation_sensitivity
        self.last_adaptation_action = "inherited"
        self.last_adaptation_target = parent.last_adaptation_target
        self.last_perceived_hazard_risk = parent.last_perceived_hazard_risk
        self.last_continuity_target = parent.last_continuity_target
        self.last_perceived_continuity_risk = parent.last_perceived_continuity_risk
        self.pending_adaptation_increment = 0.0
        self.adaptation_update_count = 0

    def reset_adaptation_state(self) -> None:
        """Reset adaptation state to a fresh post-entry draw."""
        adaptation_config = getattr(self.model, "adaptation_config", {})
        sensitivity_min = float(adaptation_config.get("adaptation_sensitivity_min", 2.0))
        sensitivity_max = float(adaptation_config.get("adaptation_sensitivity_max", 4.0))
        if sensitivity_max < sensitivity_min:
            sensitivity_max = sensitivity_min

        self.continuity_capital = 0.0
        self.expected_direct_loss_ewma = 0.0
        self.realized_direct_loss_ewma = 0.0
        self.local_observed_loss_ewma = 0.0
        self.supplier_disruption_ewma = 0.0
        self.expected_operating_shortfall_ewma = 0.0
        self.local_observed_shortfall_ewma = 0.0
        self.adaptation_sensitivity = float(self.random.uniform(sensitivity_min, sensitivity_max))
        self.last_adaptation_action = "reset"
        self.last_adaptation_target = 0.0
        self.last_perceived_hazard_risk = 0.0
        self.last_continuity_target = 0.0
        self.last_perceived_continuity_risk = 0.0
        self.pending_adaptation_increment = 0.0
        self.adaptation_update_count = 0

    # ---------------- Interaction helpers ----------------------------- #
    def hire_labor(self, household: HouseholdAgent, wage: float) -> bool:
        """Attempt to hire one unit of labour from *household*.

        Returns True if the contract succeeded (wage transferred),
        False otherwise (insufficient funds).
        """

        # Firms hire up to the vacancy count implied by planned output, not until cash is exhausted.
        if len(self.employees) >= self.target_labor:
            return False

        if not self._spend_operating_cash(wage):
            return False

        # Transfer wage while allowing a bounded sales-backed overdraft.
        household.money += wage
        household.labor_income_this_step += wage
        self.wage_bill_this_step += wage

        # Register labour for this production cycle
        self.employees.append(household)
        return True

    def sell_goods_to_household(self, household: HouseholdAgent, quantity: float = 1.0) -> float:
        """Sell up to *quantity* units (can be fractional) to *household*.

        Returns the actual units sold (float). If buyer cannot afford full
        quantity, sells the largest affordable fraction. Always returns 0 if
        no transaction occurs.
        """

        available_inventory = self.model.available_inventory_for_spot_sales(self)
        if quantity <= 0 or available_inventory <= 0:
            return 0.0

        qty = min(quantity, available_inventory)
        total_cost = qty * self.price
        if household.money < total_cost:
            # Adjust quantity downward to what buyer can afford
            qty_affordable = household.money / self.price
            qty = min(qty, qty_affordable)
            total_cost = qty * self.price
            if qty <= 0:
                return 0.0

        # Execute transaction with possibly reduced qty
        household.money -= total_cost
        self.money += total_cost
        self.inventory_output -= qty
        # Track sales for demand-based pricing
        self.sales_this_step += qty
        self.household_sales_this_step += qty
        self.revenue_this_step += total_cost
        return qty

    def sell_goods_to_firm(
        self,
        buyer: "FirmAgent",
        quantity: float = 1.0,
        *,
        unit_price: float | None = None,
        reservation_buyer_id: int | None = None,
    ) -> float:
        """Sell intermediate goods to another firm."""
        if reservation_buyer_id is None:
            available_inventory = self.model.available_inventory_for_spot_sales(self)
        else:
            available_inventory = self.model.available_reserved_inventory_for_buyer(
                self,
                reservation_buyer_id,
            )
        if quantity <= 0 or available_inventory <= 0:
            return 0.0

        qty = min(quantity, available_inventory)
        price = self.price if unit_price is None else max(0.0, float(unit_price))
        available_cash = buyer._operating_cash_capacity()
        if available_cash <= 0 or price <= 0:
            return 0.0

        max_affordable = available_cash / price
        qty = min(qty, max_affordable)
        if qty <= 0:
            return 0.0

        cost = qty * price

        # Transfer money & inventory
        if not buyer._spend_operating_cash(cost):
            return 0.0
        self.money += cost
        self.inventory_output -= qty
        if reservation_buyer_id is not None:
            self.model.consume_reserved_capacity(self, reservation_buyer_id, qty)
        buyer.input_spend_this_step += cost

        # Register under this supplier's id inside buyer
        buyer.inventory_inputs[self.unique_id] = buyer.inventory_inputs.get(self.unique_id, 0.0) + qty

        # Track sales for demand-based pricing
        self.sales_this_step += qty
        self.revenue_this_step += cost
        return qty

    # ---------------- Demand-planning helper ---------------------------- #
    def plan_operations(self) -> None:
        """Set output, vacancy, and liquidity targets from expected demand."""

        # Reset per-period flow accounting before households enter the labour market.
        # Wages are paid during ``supply_labor()``, before ``FirmAgent.step()``
        # runs, so resetting here preserves payroll in profit calculations.
        self.production = 0.0
        self.consumption = 0.0
        self.wage_bill_this_step = 0.0
        self.input_spend_this_step = 0.0
        self.depreciation_this_step = 0.0
        self.profit_this_step = 0.0
        self.dividends_paid_this_step = 0.0
        self.investment_spending_this_step = 0.0
        self.operating_profit_this_step = 0.0
        self.required_working_capital = 0.0
        self.working_capital_credit_limit = 0.0
        self.working_capital_credit_used_this_step = 0.0
        self.demand_driven_output = 0.0
        self.no_hazard_target_output = 0.0
        self.no_hazard_demand_driven_output = 0.0
        self.no_hazard_damage_factor = max(float(getattr(self, "pre_hazard_damage_factor", self.damage_factor)), 1e-6)
        self.finance_constrained_this_step = False

        self.capital_coeff = self.original_capital_coeff

        self._liquidity_buffer = max(self.MIN_LIQUIDITY_BUFFER, self.money * self.LIQUIDITY_BUFFER_RATIO)

        effective_buffer_ratio = self._effective_inventory_buffer_ratio()
        inventory_target = max(1.0, self.expected_sales * effective_buffer_ratio)
        demand_driven_output = max(0.0, self.expected_sales + inventory_target - self.inventory_output)
        no_hazard_inventory_output = max(
            0.0,
            float(getattr(self, "pre_hazard_inventory_output", self.inventory_output)),
        )
        no_hazard_demand_driven_output = max(
            0.0,
            self.expected_sales + inventory_target - no_hazard_inventory_output,
        )
        self.demand_driven_output = demand_driven_output
        self.no_hazard_demand_driven_output = no_hazard_demand_driven_output
        desired_output = demand_driven_output
        no_hazard_desired_output = no_hazard_demand_driven_output

        avg_input_price = 0.0
        if self.connected_firms:
            avg_input_price = float(np.mean([s.price for s in self.connected_firms]))

        effective_damage = max(self.damage_factor, 1e-6)
        no_hazard_damage = max(
            float(getattr(self, "pre_hazard_damage_factor", self.damage_factor)),
            1e-6,
        )
        capital_limit = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")
        no_hazard_capital_stock = max(
            0.0,
            float(getattr(self, "pre_hazard_capital_stock", self.capital_stock)),
        )
        no_hazard_capital_limit = (
            no_hazard_capital_stock / self.capital_coeff
            if self.capital_coeff
            else float("inf")
        )
        self.target_capital_stock = max(
            self.base_capital_target,
            (demand_driven_output / effective_damage) * self.capital_coeff,
        )
        desired_output = min(desired_output, capital_limit * self.damage_factor)
        no_hazard_desired_output = min(no_hazard_desired_output, no_hazard_capital_limit * no_hazard_damage)
        unit_variable_cost = (
            self.wage_offer * self.LABOR_COEFF
            + avg_input_price * self.INPUT_COEFF
        ) / effective_damage
        no_hazard_unit_variable_cost = (
            self.wage_offer * self.LABOR_COEFF
            + avg_input_price * self.INPUT_COEFF
        ) / no_hazard_damage

        revenue_anchor = (
            max(
                self.expected_sales,
                self.sales_last_step,
                self.household_sales_last_step,
                1.0,
            )
            * max(self.price, 0.5)
        )
        provisional_working_capital = max(0.0, desired_output * unit_variable_cost)
        self.working_capital_credit_limit = min(
            provisional_working_capital,
            revenue_anchor * self.WORKING_CAPITAL_CREDIT_REVENUE_SHARE,
        )
        available_operating_cash = self._operating_cash_capacity()
        if unit_variable_cost > 0:
            finance_limited_output = available_operating_cash / unit_variable_cost
            if finance_limited_output + 1e-9 < desired_output:
                self.finance_constrained_this_step = True
            desired_output = min(desired_output, finance_limited_output)
        if no_hazard_unit_variable_cost > 0:
            no_hazard_desired_output = min(
                no_hazard_desired_output,
                available_operating_cash / no_hazard_unit_variable_cost,
            )

        self.target_output = max(0.0, desired_output)
        self.no_hazard_target_output = max(0.0, no_hazard_desired_output)
        self.required_working_capital = self.target_output * unit_variable_cost
        self.working_capital_credit_limit = min(
            self.required_working_capital,
            revenue_anchor * self.WORKING_CAPITAL_CREDIT_REVENUE_SHARE,
        )

        required_pre_damage_output = self.target_output / effective_damage
        self.target_labor = int(np.ceil(required_pre_damage_output * self.LABOR_COEFF - 1e-9))
        self.target_input_units = required_pre_damage_output * self.INPUT_COEFF

    # -------------------------------------------------------------------- #

    # ---------------- Mesa API (production) --------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Purchase inputs, transform them with labour into output, then sell surplus."""

        # ---------------- Wage adjustment ----------------------------- #
        # Revenue-based wage targeting: wages track marginal revenue product of labor.
        # This replaces ad-hoc shortage-signal heuristics with a single economic principle:
        # firms pay workers a fraction of what they produce, so wages are structurally
        # bounded by firm revenue and self-correct during downturns.
        labor_share = self.LABOR_SHARE
        if self.last_hired_labor > 0 and self.revenue_last_step > 0:
            revenue_per_worker = self.revenue_last_step / self.last_hired_labor
            target_wage = revenue_per_worker * labor_share
        elif self.last_hired_labor == 0:
            # No workers last round. Use only a modest premium over the market
            # mean so empty firms can attract an initial worker without creating
            # a wage ratchet during slack baseline conditions.
            target_wage = self.model.mean_wage * self.NO_WORKER_WAGE_PREMIUM
        else:
            # Had workers but no revenue — hold current wage
            target_wage = self.wage_offer

        # Smooth adjustment: 10% toward target each step
        self.wage_offer += 0.1 * (target_wage - self.wage_offer)

        # Minimum wage floor at 40% of initial wage, as a proxy consistent with ILO (2016) observations that
        # minimum wages in high-income economies typically fall between 40–60% of the median wage.
        wage_floor = getattr(self.model, 'initial_mean_wage', 1.0) * 0.4
        self.wage_offer = float(max(wage_floor, self.wage_offer))

        # ---------------- Dynamic pricing ----------------------------- #
        # Markup pricing: price = unit_cost × (1 + markup), where markup is set
        # by sell-through rate.  This replaces ad-hoc inventory-threshold bands,
        # cost-floor ratchets, and price ceilings with one economic principle:
        # prices track costs and adjust margins based on realised demand.

        # Unit cost from actual production inputs
        avg_input_price = 0.0
        if self.connected_firms:
            avg_input_price = float(np.mean([s.price for s in self.connected_firms]))
        unit_cost = (
            self.wage_offer * self.LABOR_COEFF
            + avg_input_price * self.INPUT_COEFF
        ) / max(self.damage_factor, 1e-6)

        # Sell-through is based on realised demand from the previous full period.
        available = self.inventory_available_last_step
        if available > 0 and self.sales_last_step > 0:
            sell_through = min(1.0, self.sales_last_step / available)
        else:
            sell_through = 0.0

        # Target markup stays positive but modest:
        #   sell_through = 1.0  →  markup = +0.32
        #   sell_through = 0.5  →  markup = +0.17
        #   sell_through = 0.0  →  markup = +0.02
        # This avoids below-cost pricing while limiting long-run markup compounding.
        target_markup = 0.02 + 0.30 * sell_through
        target_price = unit_cost * (1.0 + target_markup)

        # Smooth adjustment: 20% toward target each step
        self.price += 0.2 * (target_price - self.price)
        self.price = float(max(0.5, self.price))  # absolute floor to prevent zero/negative

        labour_units = self._labor_available()
        effective_damage = max(self.damage_factor, 1e-6)

        # ----------------------------------------------------------------
        # 1. Purchase the aggregate intermediate input needed for planned output
        # ----------------------------------------------------------------
        desired_pre_damage_output = self.target_output / effective_damage
        desired_pre_damage_output = min(
            desired_pre_damage_output,
            labour_units / self.LABOR_COEFF if self.LABOR_COEFF else float("inf"),
            self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf"),
        )
        desired_input_units = desired_pre_damage_output * self.INPUT_COEFF

        current_input_units = 0.0
        if self.connected_firms:
            current_input_units = sum(
                self.inventory_inputs.get(supplier.unique_id, 0.0)
                for supplier in self.connected_firms
            )
        primary_inventory_capacity = current_input_units + sum(
            self.model.available_inventory_for_spot_sales(supplier)
            for supplier in self.connected_firms
        )

        remaining_inputs_needed = max(0.0, desired_input_units - current_input_units)
        if remaining_inputs_needed > 0 and self.connected_firms and self.INPUT_COEFF > 0:
            suppliers = sorted(
                [supplier for supplier in self.connected_firms if supplier.inventory_output > 0],
                key=lambda supplier: supplier.price,
            )
            for supplier in suppliers:
                if remaining_inputs_needed <= 1e-9 or self._operating_cash_capacity() <= 1e-9:
                    break
                bought = supplier.sell_goods_to_firm(self, remaining_inputs_needed)
                if bought > 0:
                    remaining_inputs_needed -= bought

        if desired_input_units > 1e-9:
            physical_shortfall_units = max(0.0, desired_input_units - primary_inventory_capacity)
            physical_shortfall_ratio = min(1.0, max(0.0, physical_shortfall_units / desired_input_units))
        else:
            physical_shortfall_ratio = 0.0
        hazard_affected_suppliers = (
            physical_shortfall_ratio > 1e-9
            and self._primary_supplier_shortage_is_hazard_related()
        )
        raw_supplier_disruption = physical_shortfall_ratio if hazard_affected_suppliers else 0.0
        self.raw_supplier_disruption_this_step = raw_supplier_disruption
        self.supplier_disruption_this_step = raw_supplier_disruption

        # --- Backup supplier search (continuity-capital mechanism) ---
        # Instead of fabricating phantom inputs, firms with continuity capital
        # search for real backup suppliers with actual inventory and buy at
        # market prices using actual cash.  This preserves macro closure.
        backup_purchases = 0.0
        reserved_capacity_purchases = 0.0
        reserved_capacity_price_savings = 0.0
        if (
            self._uses_adaptation_strategy("backup_suppliers")
            and hazard_affected_suppliers
            and remaining_inputs_needed > 1e-9
            and self.continuity_capital > 0
            and self._operating_cash_capacity() > 1e-9
        ):
            remaining_inputs_needed, backup_purchases = self._purchase_from_backup_suppliers(
                remaining_inputs_needed
            )
        elif (
            self._uses_adaptation_strategy("reserved_capacity")
            and hazard_affected_suppliers
            and remaining_inputs_needed > 1e-9
            and self.continuity_capital > 0
            and self._operating_cash_capacity() > 1e-9
        ):
            (
                remaining_inputs_needed,
                reserved_capacity_purchases,
                reserved_capacity_price_savings,
            ) = self._purchase_from_reserved_capacity(remaining_inputs_needed)
        self.backup_purchases_this_step = backup_purchases
        self.reserved_capacity_purchases_this_step = reserved_capacity_purchases
        self.reserved_capacity_price_savings_this_step = reserved_capacity_price_savings
        continuity_purchase_coverage = backup_purchases + reserved_capacity_purchases
        self.continuity_input_coverage_this_step = continuity_purchase_coverage  # backward compat

        # ----------------------------------------------------------------
        # 2. Compute possible output: demand target capped by technical limits
        # ----------------------------------------------------------------
        if self.connected_firms and self.INPUT_COEFF > 0:
            # sum(values()) captures both primary and backup-purchased inputs
            total_input_units = sum(self.inventory_inputs.values())
            max_output_from_inputs = total_input_units / self.INPUT_COEFF
        else:
            total_input_units = 0.0
            max_output_from_inputs = float("inf")

        max_output_from_capital = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")
        max_output_from_labor = labour_units / self.LABOR_COEFF if self.LABOR_COEFF else float("inf")

        actual_limits = {
            "labor": max_output_from_labor * self.damage_factor,
            "input": max_output_from_inputs * self.damage_factor,
            "capital": max_output_from_capital * self.damage_factor,
        }
        technical_output_limit = min(actual_limits.values())
        possible_output = min(self.target_output, technical_output_limit)

        # Update supplier disruption to reflect actual residual shortfall
        # after backup search.
        if desired_input_units > 1e-9:
            residual_hazard_shortfall_units = max(
                0.0,
                max(0.0, desired_input_units - primary_inventory_capacity) - continuity_purchase_coverage,
            )
            self.supplier_disruption_this_step = (
                min(1.0, max(0.0, residual_hazard_shortfall_units / desired_input_units))
                if hazard_affected_suppliers
                else 0.0
            )
        else:
            self.supplier_disruption_this_step = 0.0
        self.continuity_gap_coverage_this_step = continuity_purchase_coverage

        no_hazard_damage = max(self.no_hazard_damage_factor, 1e-6)
        no_hazard_capital_limit = (
            float(getattr(self, "pre_hazard_capital_stock", self.capital_stock)) / self.capital_coeff
            if self.capital_coeff
            else float("inf")
        )
        pre_hazard_total_input_units = sum(
            float(units)
            for units in getattr(self, "pre_hazard_inventory_inputs", {}).values()
        )
        if self.connected_firms and self.INPUT_COEFF > 0:
            no_hazard_desired_input_units = (self.no_hazard_target_output / no_hazard_damage) * self.INPUT_COEFF
            no_hazard_input_units = max(total_input_units, pre_hazard_total_input_units)
            if hazard_affected_suppliers:
                no_hazard_input_units = max(no_hazard_input_units, no_hazard_desired_input_units)
            no_hazard_input_limit = (no_hazard_input_units / self.INPUT_COEFF) * no_hazard_damage
        else:
            no_hazard_input_limit = float("inf")
        no_hazard_output_ceiling = min(
            self.no_hazard_target_output,
            max_output_from_labor * no_hazard_damage,
            no_hazard_capital_limit * no_hazard_damage,
            no_hazard_input_limit,
        )
        hazard_related_gap = (
            self.raw_direct_loss_fraction_this_step > 1e-9
            or self.adapted_direct_loss_fraction_this_step > 1e-9
            or raw_supplier_disruption > 1e-9
            or self.damage_factor < 0.999999
        )

        if self.target_output + 1e-9 < technical_output_limit:
            self.limiting_factor = "finance" if self.finance_constrained_this_step else "demand"
        else:
            self.limiting_factor = min(actual_limits, key=actual_limits.get)

        self.production = possible_output
        if no_hazard_output_ceiling > 1e-9 and hazard_related_gap:
            self.hazard_operating_shortfall_this_step = min(
                1.0,
                max(0.0, no_hazard_output_ceiling - possible_output) / no_hazard_output_ceiling,
            )
        else:
            self.hazard_operating_shortfall_this_step = 0.0
        if possible_output > 0:
            # Damage lowers effective output per unit input, so input use scales with
            # the pre-damage quantity required to achieve the realised output.
            total_inputs_needed = (possible_output / effective_damage) * self.INPUT_COEFF
            remaining_to_consume = total_inputs_needed

            for supplier in self.connected_firms:
                supp_id = supplier.unique_id
                if supp_id in self.inventory_inputs and remaining_to_consume > 0:
                    available = self.inventory_inputs[supp_id]
                    use_qty = min(available, remaining_to_consume)
                    self.inventory_inputs[supp_id] -= use_qty
                    remaining_to_consume -= use_qty

            # Consume backup-sourced inputs (stored under non-primary supplier IDs)
            if remaining_to_consume > 1e-9:
                primary_ids = {s.unique_id for s in self.connected_firms}
                for supp_id in list(self.inventory_inputs.keys()):
                    if supp_id in primary_ids or remaining_to_consume <= 1e-9:
                        continue
                    available = self.inventory_inputs[supp_id]
                    if available > 0:
                        use_qty = min(available, remaining_to_consume)
                        self.inventory_inputs[supp_id] -= use_qty
                        remaining_to_consume -= use_qty

            # Add production to inventory
            self.inventory_output += possible_output

            self.consumption = total_inputs_needed - remaining_to_consume

        # ----------------------------------------------------------------
        # 3. Clear employee list for next step
        # ----------------------------------------------------------------
        # Record labour count for next step's wage adjustment
        self.last_hired_labor = len(self.employees)
        self.employees.clear()
        
        self.survival_time += 1

        # ---------------- Capital depreciation ------------------------ #
        DEPR = 0.002  # 0.2 % per step (quarterly), roughly 0.8 % annually - reduced to prevent wealth drain
        self.depreciation_this_step = self.capital_stock * DEPR
        self.capital_stock *= (1 - DEPR)

    # ---------------- Internal helpers -------------------------------- #
    def _labor_available(self) -> int:
        """Return integer labour units hired for this tick."""
        return len(self.employees) 

    def close_step(self) -> None:
        """Persist realised sales after all market transactions for the period."""

        accounting_profit = (
            self.revenue_this_step
            - self.wage_bill_this_step
            - self.input_spend_this_step
            - self.depreciation_this_step
        )
        self.operating_profit_this_step = accounting_profit
        self.profit_this_step = accounting_profit - self.direct_loss_expense_this_step

        # Positive profits are either paid out to household owners as dividends
        # or recycled into installed capital. Because the model has no explicit
        # capital-goods firm, investment spending is transferred to households
        # as capital-service income so money stays inside the closed economy.
        #
        # The baseline closure needs firms to preserve their installed productive
        # base before distributing residual profits. We therefore fund capital
        # replacement up to the firm's base-capital target before allowing
        # discretionary expansion and dividends.
        operating_cash_reserve = max(
            self._liquidity_buffer,
            self.wage_bill_this_step + self.input_spend_this_step,
        )
        positive_profit = max(0.0, accounting_profit)
        self._allocate_positive_profit(
            positive_profit=positive_profit,
            operating_cash_reserve=operating_cash_reserve,
        )
        self._pay_dividends(
            positive_profit=positive_profit,
            operating_cash_reserve=operating_cash_reserve,
        )
        self.inventory_available_last_step = self.inventory_output + self.sales_this_step
        self._apply_damage_recovery()
        self._update_post_step_state()

    # ------------------------------------------------------------------ #
    #                        INTERNAL HELPERS                           #
    # ------------------------------------------------------------------ #
    def _get_local_hazard(self) -> float:
        """Return the hazard value at the agent's current cell."""
        return self.model.hazard_map.get(self.pos, 0.0)
