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

    # ---------------- Mesa API ---------------- #
    def supply_labor(self) -> None:
        """Reset household state, optionally relocate, and sell labour."""

        # Carry passive income distributed after last period's goods market into the
        # current consumption decision, then reset current-period counters.
        self.dividend_income_last_step = self.dividend_income_this_step
        self.capital_income_last_step = self.capital_income_this_step
        self.dividend_income_this_step = 0.0
        self.capital_income_this_step = 0.0
        self.labor_income_this_step = 0.0

        # Reset per-step statistics
        self.production = 0.0
        self.labor_sold = 0.0
        self.consumption = 0.0

        # ---------------- Heuristic relocation decision --------------- #
        # Only relocate when flood depth exceeds 0.5m (significant flooding)
        if self.model.household_relocation_enabled and self._get_local_hazard() > 0.5:
            self._relocate()

        # 1. Choose employer based on wage–distance utility (remote work allowed) --------------- #
        # Allow cross-sector employment to prevent labor market segmentation death spirals.
        # When one sector struggles, its workers can find jobs in healthier sectors.
        # Use all firms, not just same-sector firms.
        all_firms = self.model._firms
        if all_firms:
            scored: list[tuple[float, "FirmAgent"]] = []
            x0, y0 = self.pos
            for firm in all_firms:
                dx = abs(x0 - firm.pos[0])
                dy = abs(y0 - firm.pos[1])
                dist = dx + dy
                utility = firm.wage_offer - self.distance_cost * dist
                scored.append((utility, firm))

            scored.sort(key=lambda t: t[0], reverse=True)

            for _, firm in scored:
                if firm.hire_labor(self, firm.wage_offer):
                    self.labor_sold += 1
                    break

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

    # Learning parameters
    LEARNING_ENABLED: bool = True
    MEMORY_LENGTH: int = 10  # steps to track for performance evaluation
    MUTATION_RATE: float = 0.05  # standard deviation for strategy mutations
    ADAPTATION_FREQUENCY: int = 5  # steps between strategy evaluations

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

        # Sales tracking for pricing and wage mechanisms
        self.sales_last_step: float = 0.0
        self.revenue_last_step: float = 0.0
        self.sales_this_step: float = 0.0
        self.revenue_this_step: float = 0.0
        self.household_sales_last_step: float = 0.0
        self.household_sales_this_step: float = 0.0
        self.wage_bill_this_step: float = 0.0
        self.input_spend_this_step: float = 0.0
        self.depreciation_this_step: float = 0.0
        self.profit_this_step: float = 0.0
        self.dividends_paid_this_step: float = 0.0
        self.investment_spending_this_step: float = 0.0

        # ---------------- Risk behaviour parameters ------------------- #
        # Capital coefficient parameters
        self.original_capital_coeff: float = self.CAPITAL_COEFF
        self.capital_coeff: float = self.CAPITAL_COEFF  # dynamic value
        # Firm-specific relaxation ratio (0.2 → 20 % decay each step, etc.)
        self.relaxation_ratio: float = self.random.uniform(0.2, 0.5)

        # Demand-planning state set each step by plan_operations()
        self.target_output: float = 0.0
        self.target_labor: int = 0
        self.target_input_units: float = 0.0
        self.expected_sales: float = 0.0
        self.base_inventory_target: float = 1.0
        self.base_capital_target: float = self.capital_stock
        self.target_capital_stock: float = self.capital_stock
        self._liquidity_buffer: float = 0.0
        
        # Learning system components
        learning_config = getattr(model, 'learning_config', {})
        self.learning_enabled: bool = getattr(model, 'firm_learning_enabled', self.LEARNING_ENABLED)
        self.memory_length: int = learning_config.get('memory_length', self.MEMORY_LENGTH)
        self.adaptation_frequency: int = learning_config.get('adaptation_frequency', self.ADAPTATION_FREQUENCY)
        self.performance_history: list[dict] = []  # Track recent performance metrics
        self.strategy: dict[str, float] = self._initialize_strategy()
        self.fitness_score: float = 0.0
        self.steps_since_adaptation: int = 0
        
        # Survival tracking for evolutionary pressure
        self.survival_time: int = 0

    # ---------------- Learning System Methods ----------------------------- #
    def _initialize_strategy(self) -> dict[str, float]:
        """Initialize evolvable strategy parameters with small random variations."""
        return {
            'budget_labor_weight': self.random.uniform(0.8, 1.2),      # liquidity buffer multiplier
            'budget_input_weight': self.random.uniform(0.8, 1.2),      # inventory buffer multiplier
            'budget_capital_weight': self.random.uniform(0.8, 1.2),    # reinvestment multiplier
            'risk_sensitivity': self.random.uniform(0.5, 1.5),         # hazard response aggressiveness
            'wage_responsiveness': self.random.uniform(0.5, 1.5),      # wage adjustment responsiveness
        }
    
    def _record_performance(self) -> None:
        """Track production for fitness evaluation."""
        self.performance_history.append({'production': self.production})

        # Keep only recent history
        if len(self.performance_history) > self.memory_length:
            self.performance_history = self.performance_history[-self.memory_length:]
    
    def _evaluate_fitness(self) -> float:
        """Fitness = time-averaged production over the memory window.

        A single metric that implicitly captures all aspects of firm health:
        sustaining high production requires adequate capital, liquidity,
        labor, and input procurement across varying conditions.
        """
        if len(self.performance_history) < 2:
            return 0.0
        productions = [r['production'] for r in self.performance_history]
        return float(np.mean(productions))
    
    def _adapt_strategy(self) -> None:
        """Adjust strategy based on recent performance."""
        if not self.learning_enabled or len(self.performance_history) < 3:
            return
        
        current_fitness = self._evaluate_fitness()
        
        # Simple hill-climbing: if fitness improved, reinforce recent changes
        # If fitness declined, try random mutations
        if hasattr(self, '_previous_fitness'):
            if current_fitness > self._previous_fitness:
                # Success: make smaller adjustments in same direction
                mutation_strength = self.MUTATION_RATE * 0.5
            else:
                # Failure: try bigger random changes
                mutation_strength = self.MUTATION_RATE * 2.0
        else:
            mutation_strength = self.MUTATION_RATE
        
        # Mutate strategy parameters using configured mutation rate
        for key in self.strategy:
            if self.random.random() < 0.3:  # 30% chance to mutate each parameter
                self.strategy[key] *= (1.0 + self.random.gauss(0, mutation_strength))
                # Keep within reasonable bounds
                self.strategy[key] = max(0.1, min(3.0, self.strategy[key]))
        
        self._previous_fitness = current_fitness
        self.fitness_score = current_fitness

    # ---------------- Interaction helpers ----------------------------- #
    def hire_labor(self, household: HouseholdAgent, wage: float) -> bool:
        """Attempt to hire one unit of labour from *household*.

        Returns True if the contract succeeded (wage transferred),
        False otherwise (insufficient funds).
        """

        # Firms hire up to the vacancy count implied by planned output, not until cash is exhausted.
        if len(self.employees) >= self.target_labor:
            return False

        if self.money - wage < self._liquidity_buffer:
            return False

        # Transfer wage and preserve the working-capital buffer.
        self.money -= wage
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

        if quantity <= 0 or self.inventory_output <= 0:
            return 0.0

        qty = min(quantity, self.inventory_output)
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

    def sell_goods_to_firm(self, buyer: "FirmAgent", quantity: float = 1.0) -> float:
        """Sell intermediate goods to another firm."""
        if quantity <= 0 or self.inventory_output <= 0:
            return 0.0

        qty = min(quantity, self.inventory_output)
        available_cash = max(0.0, buyer.money - getattr(buyer, "_liquidity_buffer", 0.0))
        if available_cash <= 0 or self.price <= 0:
            return 0.0

        max_affordable = available_cash / self.price
        qty = min(qty, max_affordable)
        if qty <= 0:
            return 0.0

        cost = qty * self.price

        # Transfer money & inventory
        buyer.money -= cost
        self.money += cost
        self.inventory_output -= qty
        buyer.input_spend_this_step += cost

        # Register under this supplier's id inside buyer
        buyer.inventory_inputs[self.unique_id] += qty

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

        # Apply the hazard-driven capital requirement before planning vacancies.
        local_hazard = self._get_local_hazard()
        risk_sensitivity = self.strategy.get("risk_sensitivity", 1.0)
        if local_hazard > 0.1:
            self.capital_coeff *= 1.0 + (0.2 * risk_sensitivity)
        elif self.capital_coeff > self.original_capital_coeff:
            adjusted_relaxation = self.relaxation_ratio * (2.0 - risk_sensitivity)
            decay = (self.capital_coeff - self.original_capital_coeff) * adjusted_relaxation
            self.capital_coeff = max(self.original_capital_coeff, self.capital_coeff - decay)

        # Firms produce to expected sales plus a modest inventory buffer.
        inventory_buffer_ratio = max(0.0, 0.25 * self.strategy.get("budget_input_weight", 1.0))
        liquidity_buffer_ratio = min(0.5, max(0.05, 0.15 * self.strategy.get("budget_labor_weight", 1.0)))
        self._liquidity_buffer = max(10.0, self.money * liquidity_buffer_ratio)

        inventory_target = max(1.0, self.expected_sales * inventory_buffer_ratio)
        demand_driven_output = max(0.0, self.expected_sales + inventory_target - self.inventory_output)
        desired_output = demand_driven_output

        avg_input_price = 0.0
        if self.connected_firms:
            avg_input_price = float(np.mean([s.price for s in self.connected_firms]))

        effective_damage = max(self.damage_factor, 1e-6)
        capital_limit = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")
        self.target_capital_stock = max(
            self.base_capital_target,
            (demand_driven_output / effective_damage) * self.capital_coeff,
        )
        desired_output = min(desired_output, capital_limit * self.damage_factor)
        unit_variable_cost = (
            self.wage_offer * self.LABOR_COEFF
            + avg_input_price * self.INPUT_COEFF
        ) / effective_damage

        available_operating_cash = max(0.0, self.money - self._liquidity_buffer)
        if unit_variable_cost > 0:
            desired_output = min(desired_output, available_operating_cash / unit_variable_cost)

        self.target_output = max(0.0, desired_output)

        required_pre_damage_output = self.target_output / effective_damage
        self.target_labor = int(np.ceil(required_pre_damage_output * self.LABOR_COEFF - 1e-9))
        self.target_input_units = required_pre_damage_output * self.INPUT_COEFF

    # -------------------------------------------------------------------- #

    # ---------------- Mesa API (production) --------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Purchase inputs, transform them with labour into output, then sell surplus."""

        # ---------------- Damage recovery ----------------------------- #
        # ---------------- Wage adjustment ----------------------------- #
        # Revenue-based wage targeting: wages track marginal revenue product of labor.
        # This replaces ad-hoc shortage-signal heuristics with a single economic principle:
        # firms pay workers a fraction of what they produce, so wages are structurally
        # bounded by firm revenue and self-correct during downturns.
        labor_share = 0.5 * self.strategy.get('wage_responsiveness', 1.0)
        if self.last_hired_labor > 0 and self.revenue_last_step > 0:
            revenue_per_worker = self.revenue_last_step / self.last_hired_labor
            target_wage = revenue_per_worker * labor_share
        elif self.last_hired_labor == 0:
            # No workers last round — offer above market mean to attract someone
            target_wage = self.model.mean_wage * 1.1
        else:
            # Had workers but no revenue — hold current wage
            target_wage = self.wage_offer

        # Smooth adjustment: 10% toward target each step
        self.wage_offer += 0.1 * (target_wage - self.wage_offer)

        # Minimum wage floor at 40% of initial wage, as a proxy consistent with ILO (2016) observations that
        # minimum wages in high-income economies typically fall between 40–60% of the median wage.
        wage_floor = getattr(self.model, 'initial_mean_wage', 1.0) * 0.4
        self.wage_offer = float(max(wage_floor, self.wage_offer))

        # Liquidity-dependent recovery: firms with more capital recover faster because they can afford repairs. 
        # Base rate 20%, scaling up to 50% for well-capitalised firms.
        liquidity_ratio = min(1.0, self.money / 200.0)
        recovery_rate = 0.2 + 0.3 * liquidity_ratio
        self.damage_factor += (1.0 - self.damage_factor) * recovery_rate
        self.damage_factor = min(1.0, max(0.0, self.damage_factor))

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
        )

        # Sell-through is based on realised demand from the previous full period.
        available = self.inventory_output + self.sales_last_step
        if available > 0 and self.sales_last_step > 0:
            sell_through = min(1.0, self.sales_last_step / available)
        else:
            sell_through = 0.0

        # Target markup scales linearly with sell-through:
        #   sell_through = 1.0  →  markup = +0.5  (strong demand, 50% margin)
        #   sell_through = 0.5  →  markup =  0.0  (break-even)
        #   sell_through = 0.0  →  markup = -0.5  (weak demand, sell below cost)
        target_markup = sell_through - 0.5
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

        remaining_inputs_needed = max(0.0, desired_input_units - current_input_units)
        if remaining_inputs_needed > 0 and self.connected_firms and self.INPUT_COEFF > 0:
            suppliers = sorted(
                [supplier for supplier in self.connected_firms if supplier.inventory_output > 0],
                key=lambda supplier: supplier.price,
            )
            for supplier in suppliers:
                if remaining_inputs_needed <= 1e-9 or self.money <= self._liquidity_buffer:
                    break
                bought = supplier.sell_goods_to_firm(self, remaining_inputs_needed)
                if bought > 0:
                    remaining_inputs_needed -= bought

        # ----------------------------------------------------------------
        # 2. Compute possible output: demand target capped by technical limits
        # ----------------------------------------------------------------
        if self.connected_firms and self.INPUT_COEFF > 0:
            total_input_units = sum(
                self.inventory_inputs.get(supplier.unique_id, 0.0)
                for supplier in self.connected_firms
            )
            max_output_from_inputs = total_input_units / self.INPUT_COEFF
        else:
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

        if self.target_output + 1e-9 < technical_output_limit:
            self.limiting_factor = "demand"
        else:
            self.limiting_factor = min(actual_limits, key=actual_limits.get)

        self.production = possible_output
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

            # Add production to inventory
            self.inventory_output += possible_output

            self.consumption = total_inputs_needed - remaining_to_consume

        # ----------------------------------------------------------------
        # 3. Clear employee list for next step
        # ----------------------------------------------------------------
        # Record labour count for next step's wage adjustment
        self.last_hired_labor = len(self.employees)
        self.employees.clear()
        
        # ---------------- Learning system updates ----------------------- #
        self.survival_time += 1
        self._record_performance()
        self.steps_since_adaptation += 1
        
        # Periodically evaluate and adapt strategy
        if self.steps_since_adaptation >= self.adaptation_frequency:
            self._adapt_strategy()
            self.steps_since_adaptation = 0

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
        self.profit_this_step = accounting_profit

        # Positive profits are either paid out to household owners as dividends
        # or recycled into installed capital. Because the model has no explicit
        # capital-goods firm, investment spending is transferred to households
        # as capital-service income so money stays inside the closed economy.
        operating_cash_reserve = max(
            self._liquidity_buffer,
            self.wage_bill_this_step + self.input_spend_this_step,
        )
        available_cash = max(0.0, self.money - operating_cash_reserve)
        positive_profit = max(0.0, accounting_profit)
        investment_share = min(1.0, 0.5 * self.strategy.get("budget_capital_weight", 1.0))
        capital_gap = max(0.0, self.target_capital_stock - self.capital_stock)
        desired_investment_spending = min(
            positive_profit * investment_share,
            capital_gap * self.CAPITAL_INSTALLATION_COST,
            available_cash,
        )

        if desired_investment_spending > 0:
            installable_capital = desired_investment_spending / self.CAPITAL_INSTALLATION_COST
            self.money -= desired_investment_spending
            self.capital_stock += installable_capital
            self.investment_spending_this_step = desired_investment_spending
            self.model.distribute_household_income(
                desired_investment_spending,
                income_kind="capital",
            )

        available_cash = max(0.0, self.money - operating_cash_reserve)
        desired_dividends = min(
            max(0.0, positive_profit - self.investment_spending_this_step),
            available_cash,
        )
        if desired_dividends > 0:
            self.money -= desired_dividends
            self.dividends_paid_this_step = desired_dividends
            self.model.distribute_household_income(
                desired_dividends,
                income_kind="dividend",
            )

        self.expected_sales = 0.7 * self.expected_sales + 0.3 * self.sales_this_step
        self.sales_last_step = self.sales_this_step
        self.household_sales_last_step = self.household_sales_this_step
        self.revenue_last_step = self.revenue_this_step
        self.sales_this_step = 0.0
        self.household_sales_this_step = 0.0
        self.revenue_this_step = 0.0

    # ------------------------------------------------------------------ #
    #                        INTERNAL HELPERS                           #
    # ------------------------------------------------------------------ #
    def _get_local_hazard(self) -> float:
        """Return the hazard value at the agent's current cell."""
        return self.model.hazard_map.get(self.pos, 0.0) 
