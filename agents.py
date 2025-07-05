from __future__ import annotations

from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from mesa import Agent

Coords = Tuple[int, int]


# ------------------------------------------------------------------------- #
# Optional forward references to avoid circular imports during type checking
# ------------------------------------------------------------------------- #
if TYPE_CHECKING:  # pragma: no cover
    from model import EconomyModel  # noqa: F401 – only for typing


class HouseholdAgent(Agent):
    """A household supplies one unit of labour, earns wages and buys goods."""

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
        self.capital: float = 1.0  # kept for compatibility with impact module

        # Aggregate statistics tracked per agent
        self.consumption: float = 0.0  # goods consumed (units)

        # Filled by the model after all agents are created
        self.nearby_firms: List["FirmAgent"] = []

    # ---------------- Mesa API ---------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Provide labour and consume goods each tick."""

        if not self.nearby_firms:
            return  # isolated household – nothing to do

        # 1. Offer labour to a random connected firm ----------------------- #
        firm = self.random.choice(self.nearby_firms)
        wage = self.model.base_wage
        if firm.hire_labor(self, wage):
            # Wage transferred inside `hire_labor`
            pass

        # 2. Buy one unit of goods if affordable -------------------------- #
        # Households consume a single generic good for simplicity.
        affordable_firms = [f for f in self.nearby_firms if f.inventory_output > 0 and f.price <= self.money]
        if affordable_firms:
            seller = self.random.choice(affordable_firms)
            qty_bought = seller.sell_goods_to_household(self, quantity=1)
            if qty_bought:
                self.consumption += qty_bought


class FirmAgent(Agent):
    """A firm with a simple Leontief production function and local trade."""

    # Leontief technical coefficients: input requirement per unit output
    LABOR_COEFF: float = 1.0  # labour units required per output unit
    INPUT_COEFF: float = 1.0  # intermediate good units required per output unit

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        sector: str = "manufacturing",
        capital_stock: float = 1.0,
    ) -> None:
        super().__init__(model)

        self.sector = sector
        self.capital_stock = capital_stock

        # ---------------- Economic state ------------------------------- #
        self.money: float = 100.0
        self.inventory_input: int = 0  # units of intermediate goods
        self.inventory_output: int = 0  # finished goods

        # Links to other agents (filled by model)
        self.connected_firms: List["FirmAgent"] = []
        self.employees: List[HouseholdAgent] = []

        # Statistics
        self.production: float = 0.0  # units produced this step
        self.consumption: float = 0.0  # units of inputs consumed this step

        # Pricing (fixed for now)
        self.price: float = 1.0

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
        buyer.inventory_input += qty
        return qty

    # ---------------- Mesa API (production) --------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Purchase inputs, transform them with labour into output, then sell surplus."""

        # Reset per-step statistics
        self.production = 0.0
        self.consumption = 0.0

        labour_units = self._labor_available()

        # ----------------------------------------------------------------
        # 1. Ensure sufficient intermediate inputs based on hired labour
        #    Desired input quantity so that input/labour ratio meets coefficients.
        # ----------------------------------------------------------------
        target_input_needed = int(np.ceil(labour_units * self.INPUT_COEFF / self.LABOR_COEFF))
        required_inputs = max(0, target_input_needed - self.inventory_input)

        if required_inputs > 0 and self.connected_firms:
            firms_shuffled = self.random.sample(self.connected_firms, len(self.connected_firms))
            for supplier in firms_shuffled:
                if required_inputs == 0:
                    break
                purch_qty = min(required_inputs, supplier.inventory_output)
                if purch_qty > 0:
                    bought = supplier.sell_goods_to_firm(self, purch_qty)
                    required_inputs -= bought

        # Update labour_units after potential change? Labour unchanged.
        # ----------------------------------------------------------------
        # 2. Produce output according to Leontief production function
        #    q = min(labour / a_L, input / a_I)
        # ----------------------------------------------------------------
        max_output_from_labor = labour_units / self.LABOR_COEFF if self.LABOR_COEFF else 0
        max_output_from_input = self.inventory_input / self.INPUT_COEFF if self.INPUT_COEFF else 0

        possible_output = int(min(max_output_from_labor, max_output_from_input))

        self.production = possible_output
        if possible_output > 0:
            # Consume inputs and register consumption stat
            input_used = int(possible_output * self.INPUT_COEFF)
            self.inventory_input -= input_used
            self.consumption = input_used

            # Generate output inventory
            self.inventory_output += possible_output

        # ----------------------------------------------------------------
        # 3. Clear employee list for next step
        # ----------------------------------------------------------------
        self.employees.clear()

    # ---------------- Internal helpers -------------------------------- #
    def _labor_available(self) -> int:
        """Return integer labour units hired for this tick."""
        return len(self.employees) 