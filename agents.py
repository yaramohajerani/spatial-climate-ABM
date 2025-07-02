from __future__ import annotations

from typing import Tuple

import numpy as np
from mesa import Agent

Coords = Tuple[int, int]


class HouseholdAgent(Agent):
    """A household lives at a grid position, supplies labor, consumes income and can migrate if risk is high."""

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        income: float = 1.0,
        capital: float = 1.0,
        sector: str = "services",
        migration_threshold: float | None = None,
    ) -> None:
        # In Mesa ≥3.0, Agent IDs are assigned automatically by the framework.
        super().__init__(model)
        self.income = income
        self.capital = capital
        self.sector = sector
        # allow per-household heterogeneity in migration behaviour
        self.migration_threshold = (
            migration_threshold if migration_threshold is not None else model.migration_threshold
        )

    # ---------------- Mesa API ---------------- #
    def step(self) -> None:  # noqa: D401, N802: Mesa naming convention
        """Household logic executed every model step."""
        risk = self.model.get_cell_risk(self.pos)

        # Decide whether to migrate
        if risk >= self.migration_threshold:
            self._migrate()

        # Supply labor to firms in the same cell (simplified).
        # Update income with a simple wage drawn from the model parameter.
        self.income = self.model.base_wage

    # ---------------- Internal helpers ---------------- #
    def _migrate(self) -> None:
        """Move to the lowest-risk neighbouring cell (or random low-risk cell)."""
        self.model.migrants_this_step += 1

        # All candidate cells sorted by ascending risk
        candidate_cells = [
            (coord, self.model.get_cell_risk(coord))
            for coord in self.model.valid_coordinates
        ]
        candidate_cells.sort(key=lambda x: x[1])

        # Filter out only those strictly below threshold
        low_risk_cells = [c for c, r in candidate_cells if r < self.migration_threshold]
        if low_risk_cells:
            target = self.random.choice(low_risk_cells)
            self.model.grid.move_agent(self, target)


class FirmAgent(Agent):
    """A firm produces goods using local labor and capital, and suffers damages when hit by hazards."""

    def __init__(
        self,
        model: "EconomyModel",
        pos: Coords,
        sector: str = "manufacturing",
        base_production: float = 1.0,
        capital_stock: float = 1.0,
    ) -> None:
        super().__init__(model)
        self.sector = sector
        self.base_production = base_production
        self.capital_stock = capital_stock
        self.current_production: float = 0.0

    # ---------------- Mesa API ---------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Firm production step with damage from hazard."""
        risk = self.model.get_cell_risk(self.pos)

        # Damage factor: production reduced proportional to risk
        damage_factor = 1.0 - risk
        self.current_production = max(0.0, self.base_production * damage_factor)

        # Aggregate GDP contribution
        self.model.total_gdp_this_step += self.current_production

        # Example adaptation investment stub (not implemented)
        # if risk > 0.7:
        #     self.invest_in_adaptation()

    # Placeholder for future adaptation behaviour
    def invest_in_adaptation(self) -> None:  # noqa: D401, N802
        pass 