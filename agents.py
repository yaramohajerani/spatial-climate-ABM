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

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        money: float = 100.0,
        labor_supply: float = 1.0,
    ) -> None:
        super().__init__(model)

        self.money: float = money
        self.labor_supply: float = labor_supply
        self.capital: float = 1.0

        # Trade-off coefficient between wage and distance when choosing employer.
        # Higher value → distance is more costly, so worker prefers closer firms.
        # Randomised per household to create heterogeneous behaviour.
        self.distance_cost: float = self.random.uniform(0.1, 0.5)

        # ---------------- Risk behaviour parameters ------------------- #
        # Randomised radius (in grid cells) within which the household
        # monitors hazard intensity. If the maximum normalised intensity
        # exceeds 0.5 anywhere in that radius the household will relocate.
        self.risk_radius: int = self.random.randint(1, 50)

        # Aggregate statistics tracked per agent
        self.consumption: float = 0.0  # goods consumed (units)
        self.production: float = 0.0  # households don't produce goods but keep attr for consistency
        self.labor_sold: float = 0.0  # labour units sold this step

        # Filled by the model after all agents are created
        self.nearby_firms: List["FirmAgent"] = []

    # ---------------- Mesa API ---------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Provide labour and consume goods each tick."""

        # Reset per-step statistics
        self.production = 0.0
        self.labor_sold = 0.0
        self.consumption = 0.0

        # ---------------- Heuristic relocation decision --------------- #
        if self._max_hazard_within_radius(self.risk_radius) > 0.1:
            self._relocate()

        if not self.nearby_firms:
            return  # isolated household – nothing to do

        # 1. Choose employer based on wage–distance utility --------------- #
        if self.nearby_firms:
            # Compute utility = offered wage – distance_cost * manhattan_distance
            scored: list[tuple[float, "FirmAgent"]] = []
            x0, y0 = self.pos
            for firm in self.nearby_firms:
                dx = abs(x0 - firm.pos[0])
                dy = abs(y0 - firm.pos[1])
                dist = dx + dy
                utility = firm.wage_offer - self.distance_cost * dist
                scored.append((utility, firm))

            # Sort by utility descending so households try best option first
            scored.sort(key=lambda t: t[0], reverse=True)

            for _, firm in scored:
                if firm.hire_labor(self, firm.wage_offer):
                    self.labor_sold += 1
                    break

        # 2. Buy one unit of goods if affordable -------------------------- #
        # Households consume a single generic good for simplicity.
        affordable_firms = [f for f in self.nearby_firms if f.inventory_output > 0 and f.price <= self.money]
        if affordable_firms:
            seller = self.random.choice(affordable_firms)
            qty_bought = seller.sell_goods_to_household(self, quantity=1)
            if qty_bought:
                self.consumption += qty_bought

    def _max_hazard_within_radius(self, radius: int) -> float:
        """Return the maximum hazard value within *radius* cells of current position."""
        x0, y0 = self.pos
        max_hazard = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                coord = (x0 + dx, y0 + dy)
                if coord in self.model.hazard_map:
                    max_hazard = max(max_hazard, self.model.hazard_map[coord])
        return max_hazard

    def _relocate(self) -> None:
        """Move the household to a random land cell with low current hazard."""
        safe_cells = [c for c in self.model.land_coordinates if self.model.hazard_map.get(c, 0.0) <= 0.5]
        if not safe_cells:
            safe_cells = self.model.land_coordinates  # fall back to any land
        new_pos = self.random.choice(safe_cells)
        self.model.grid.move_agent(self, new_pos)

        # Apply relocation cost: lose a share of money and physical capital
        self.money *= (1 - self.RELOCATION_COST)
        self.capital *= (1 - self.RELOCATION_COST)

        # Track migration for statistics
        self.model.migrants_this_step += 1


class FirmAgent(Agent):
    """A firm with a simple Leontief production function and local trade."""

    # Leontief technical coefficients (units required per unit output)
    LABOR_COEFF: float = 0.5  # less than 1 → higher productivity
    INPUT_COEFF: float = 0.5
    CAPITAL_COEFF: float = 0.5  # capital units per output unit

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        sector: str = "manufacturing",
        capital_stock: float = 100.0,
    ) -> None:
        super().__init__(model)

        self.sector = sector
        self.capital_stock = capital_stock

        # Firm-specific wage offer (labour price) – starts at model base wage
        self.wage_offer: float = model.mean_wage if hasattr(model, "mean_wage") else 1.0
        # Track whether labour was the binding constraint in the *previous* step
        self.labor_limited_last_step: bool = False
        self.last_hired_labor: int = 0  # employees hired in previous step

        # ---------------- Economic state ------------------------------- #
        self.money: float = 100.0
        from collections import defaultdict

        # Input inventory keyed by supplier AgentID (or None for generic labour)
        self.inventory_inputs: dict[int, int] = defaultdict(int)  # per-supplier stock
        self.inventory_output: int = 0  # finished goods

        # Links to other agents (filled by model)
        self.connected_firms: List["FirmAgent"] = []
        self.employees: List[HouseholdAgent] = []

        # Statistics
        self.production: float = 0.0  # units produced this step
        self.consumption: float = 0.0  # units of inputs consumed this step

        # Pricing  – initial value
        self.price: float = 1.0

        # Cumulative damage to productive capacity (1 = undamaged)
        self.damage_factor: float = 1.0

        # ---------------- Risk behaviour parameters ------------------- #
        # Radius monitored for hazard events (random per firm)
        self.risk_radius: int = self.random.randint(1, 50)
        # Capital coefficient parameters
        self.original_capital_coeff: float = self.CAPITAL_COEFF
        self.capital_coeff: float = self.CAPITAL_COEFF  # dynamic value
        # Firm-specific relaxation ratio (0.2 → 20 % decay each step, etc.)
        self.relaxation_ratio: float = self.random.uniform(0.2, 0.5)

    # ---------------- Interaction helpers ----------------------------- #
    def hire_labor(self, household: HouseholdAgent, wage: float) -> bool:
        """Attempt to hire one unit of labour from *household*.

        Returns True if the contract succeeded (wage transferred),
        False otherwise (insufficient funds).
        """

        if self.money < wage:
            return False

        # Transfer wage
        self.money -= wage
        household.money += wage

        # Register labour for this production cycle
        self.employees.append(household)
        return True

    def sell_goods_to_household(self, household: HouseholdAgent, quantity: int = 1) -> int:
        """Sell up to *quantity* units to *household*, return actual units sold."""
        if quantity <= 0 or self.inventory_output <= 0:
            return 0

        qty = min(quantity, self.inventory_output)
        total_cost = qty * self.price
        if household.money < total_cost:
            return 0  # buyer cannot afford

        # Execute transaction
        household.money -= total_cost
        self.money += total_cost
        self.inventory_output -= qty
        return qty

    def sell_goods_to_firm(self, buyer: "FirmAgent", quantity: int = 1) -> int:
        """Inter-firm sale of intermediate goods (generic price=1)."""
        if quantity <= 0 or self.inventory_output <= 0:
            return 0

        qty = min(quantity, self.inventory_output)
        cost = qty * self.price
        if buyer.money < cost:
            return 0

        # Transfer money & inventory
        buyer.money -= cost
        self.money += cost
        self.inventory_output -= qty
        # Register under this supplier's id inside buyer
        buyer.inventory_inputs[self.unique_id] += qty
        return qty

    # ---------------- Mesa API (production) --------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Purchase inputs, transform them with labour into output, then sell surplus."""

        # ---------------- Damage recovery ----------------------------- #
        # ---------------- Wage adjustment ----------------------------- #
        # Continuous wage update inspired by matching/Phillips-curve logic
        tightness = 1.0 - getattr(self.model, "unemployment_rate_prev", 0.0)
        signal = 1.0 if self.labor_limited_last_step else -0.5  # asymmetric
        strength = 0.1  # responsiveness coefficient

        adjustment = 1 + strength * signal * tightness
        # Friction: wage cuts (adjustment < 1) dampened by 50 %
        if adjustment < 1:
            adjustment = 1 - (1 - adjustment) * 0.5

        self.wage_offer *= adjustment
        self.wage_offer = float(min(10.0, max(0.1, self.wage_offer)))

        # Recover 50% of remaining damage each step
        self.damage_factor += (1.0 - self.damage_factor) * 0.5
        self.damage_factor = min(1.0, max(0.0, self.damage_factor))

        # ---------------- Dynamic pricing ----------------------------- #
        # Simple rule-of-thumb: if we had no stock leftover raise price, if large
        # stock (>5) lower price. Bound between 0.1 and 10.

        if self.inventory_output == 0:
            self.price *= 1.05
        elif self.inventory_output > 5:
            self.price *= 0.95

        self.price = float(min(10.0, max(0.1, self.price)))

        # Reset per-step statistics
        self.production = 0.0
        self.consumption = 0.0

        # ---------------- Hazard-driven capital adjustment ------------ #
        local_hazard = self._max_hazard_within_radius(self.risk_radius)
        if local_hazard > 0.1:
            # Increase capital requirement by 20 % whenever a significant event occurs
            self.capital_coeff *= 1.2
        else:
            # Gradually relax back towards original coefficient
            if self.capital_coeff > self.original_capital_coeff:
                decay = (self.capital_coeff - self.original_capital_coeff) * self.relaxation_ratio
                self.capital_coeff = max(self.original_capital_coeff, self.capital_coeff - decay)

        labour_units = self._labor_available()

        # ----------------------------------------------------------------
        # 1. Ensure each required input type has enough stock to match labour
        #    Each unit of output needs 1 unit from every supplier in connected_firms.
        # ----------------------------------------------------------------

        for supplier in self.connected_firms:
            current_stock = self.inventory_inputs.get(supplier.unique_id, 0)
            target_output_from_labour = labour_units / self.LABOR_COEFF
            target_input_needed = int(np.ceil(target_output_from_labour * self.INPUT_COEFF))
            required_qty = max(0, target_input_needed - current_stock)
            if required_qty == 0:
                continue

            purch_qty_needed = required_qty
            # Attempt purchase from the designated supplier only
            bought = supplier.sell_goods_to_firm(self, purch_qty_needed)
            # If supplier cannot fulfil entire order, we leave shortage (production will be limited)

        # ----------------------------------------------------------------
        # 2. Compute possible output per Leontief: min(labour, each input)
        # ----------------------------------------------------------------

        if self.connected_firms:
            min_input_units = min(self.inventory_inputs.get(s.unique_id, 0) for s in self.connected_firms)
            max_output_from_inputs = min_input_units / self.INPUT_COEFF
        else:
            max_output_from_inputs = float("inf")  # no material input constraint

        max_output_from_capital = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")

        max_possible = min(labour_units / self.LABOR_COEFF, max_output_from_inputs, max_output_from_capital)
        possible_output = int(max_possible * self.damage_factor)

        capital_limited = possible_output < max_possible and (max_output_from_capital <= labour_units / self.LABOR_COEFF)

        # Determine primary limiting factor for diagnostic plotting
        # Compare theoretical maxima before damage factor applied
        limits = {
            "labor": labour_units / self.LABOR_COEFF,
            "input": max_output_from_inputs,
            "capital": max_output_from_capital,
        }
        min_limit_val = min(limits.values())
        # Pick first factor that equals the min within small tolerance
        self.limiting_factor: str = next(k for k, v in limits.items() if abs(v - min_limit_val) < 1e-6)

        self.production = possible_output
        if possible_output > 0:
            # Consume inputs from each supplier
            for supplier in self.connected_firms:
                use_qty = int(possible_output * self.INPUT_COEFF)
                self.inventory_inputs[supplier.unique_id] -= use_qty

            # Add production to inventory
            self.inventory_output += possible_output

            self.consumption = possible_output * len(self.connected_firms) * self.INPUT_COEFF

        # ----------------------------------------------------------------
        # 3. Clear employee list for next step
        # ----------------------------------------------------------------
        # Record labour count and limiting factor for next step's adjustments
        self.last_hired_labor = len(self.employees)
        self.labor_limited_last_step = (self.limiting_factor == "labor")
        self.employees.clear()

        # ---------------- Capital depreciation ------------------------ #
        DEPR = 0.02  # 2% per step
        self.capital_stock *= (1 - DEPR)

        # ---------------- Capital purchase stage ----------------------- #
        if capital_limited and self.money > self.price:
            needed_capital_units = int(max_possible - possible_output)
            budget_units = int(self.money / self.price)
            to_buy = min(needed_capital_units, budget_units)

            if to_buy > 0:
                # Candidate sellers: upstream suppliers *and* any firm with stock
                all_firms = [ag for ag in self.model.agents if isinstance(ag, FirmAgent) and ag is not self]
                candidates = list({*self.connected_firms, *all_firms})  # set union, preserve objects

                # Randomise search order to avoid bias
                for seller in self.random.sample(candidates, len(candidates)):
                    if to_buy == 0:
                        break
                    if seller.inventory_output <= 0:
                        continue
                    qty = min(seller.inventory_output, to_buy)
                    cost = qty * seller.price
                    if self.money < cost:
                        qty = int(self.money / seller.price)
                    if qty <= 0:
                        continue
                    bought = seller.sell_goods_to_firm(self, qty)
                    if bought:
                        self.capital_stock += bought
                        to_buy -= bought

    # ---------------- Internal helpers -------------------------------- #
    def _labor_available(self) -> int:
        """Return integer labour units hired for this tick."""
        return len(self.employees) 

    # ------------------------------------------------------------------ #
    #                        INTERNAL HELPERS                           #
    # ------------------------------------------------------------------ #
    def _max_hazard_within_radius(self, radius: int) -> float:
        """Return the maximum hazard value within *radius* cells of current position."""
        x0, y0 = self.pos
        max_hazard = 0.0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                coord = (x0 + dx, y0 + dy)
                if coord in self.model.hazard_map:
                    max_hazard = max(max_hazard, self.model.hazard_map[coord])
        return max_hazard 