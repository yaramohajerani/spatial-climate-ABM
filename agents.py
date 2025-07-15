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
        sector: str = "manufacturing",
    ) -> None:
        super().__init__(model)

        self.money: float = money
        self.labor_supply: float = labor_supply
        self.capital: float = 1.0

        # Sector specialisation – household will preferentially work for firms in this sector
        self.sector: str = sector

        # Counter of consecutive steps without finding work (used for job-driven relocation)
        self._no_work_steps: int = 0

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

    # ---------------- Internal helpers ------------------- #
    def _update_nearby_firms(self) -> None:
        """Refresh the list of nearby firms in the same sector within the model's work_radius."""

        radius = getattr(self.model, "work_radius", 3)
        self.nearby_firms.clear()
        for ag in self.model.agents:
            if not isinstance(ag, FirmAgent):
                continue
            if ag.sector != self.sector:
                continue
            dx = abs(self.pos[0] - ag.pos[0])
            dy = abs(self.pos[1] - ag.pos[1])
            if dx + dy <= radius:
                self.nearby_firms.append(ag)

    def _relocate_for_job(self) -> None:
        """Move closer to a random firm in the same sector to improve employment prospects."""

        firms_same_sector = [ag for ag in self.model.agents if isinstance(ag, FirmAgent) and ag.sector == self.sector]
        if not firms_same_sector:
            return  # no suitable firms exist

        target_firm = self.random.choice(firms_same_sector)
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

        # Apply relocation cost similar to climate‐driven migration
        self.money *= (1 - self.RELOCATION_COST)
        self.capital *= (1 - self.RELOCATION_COST)

        # Refresh nearby firms after moving
        self._update_nearby_firms()

    # ---------------- Mesa API ---------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Provide labour and consume goods each tick."""

        # Refresh employer list (in case we moved or firms relocated)
        self._update_nearby_firms()

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
        # Households can purchase from any retail firm in the economy.
        retail_firms = [
            f for f in self.model.agents
            if isinstance(f, FirmAgent) and f.sector == "retail" and f.inventory_output > 0 and f.price <= self.money
        ]

        if retail_firms:
            # Select the cheapest retail seller (break ties randomly)
            min_price = min(f.price for f in retail_firms)
            cheapest = [f for f in retail_firms if abs(f.price - min_price) < 1e-6]
            seller = self.random.choice(cheapest)
            # Attempt to buy up to 1 unit; method will reduce if only fraction affordable
            qty_bought = seller.sell_goods_to_household(self, quantity=1.0)
            if qty_bought:
                self.consumption += qty_bought

        # ---------------- End-of-step unemployment tracking ------------- #
        if self.labor_sold == 0:
            self._no_work_steps += 1
        else:
            self._no_work_steps = 0

        if self._no_work_steps >= 3:
            # Could not find work for 3 consecutive steps → move closer to another firm of same sector
            self._relocate_for_job()
            self._no_work_steps = 0  # reset after relocation

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

    # ---------------- End of Household.step() bookkeeping -------- #


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

        # Set starting price based on sector (for simplicity assume all firms start with the same price)
        base_price_by_sector = {
            "commodity": 1.0,
            "agriculture": 1.0,
            "manufacturing": 1.0,
            "wholesale": 1.0,
            "retail": 1.0,
            "services": 1.0,
        }
        self.price: float = float(base_price_by_sector.get(self.sector, 1.0))
        self.money: float = 100.0

        # Firm-specific wage offer (labour price) – starts at the model's base wage
        self.wage_offer: float = model.mean_wage if hasattr(model, "mean_wage") else 1.0

        # Track whether labour was the binding constraint in the *previous* step
        self.labor_limited_last_step: bool = False
        self.last_hired_labor: int = 0  # employees hired in previous step

        # Input inventory keyed by supplier AgentID (or None for generic labour)
        self.inventory_inputs: dict[int, int] = defaultdict(int)  # per-supplier stock
        self.inventory_output: int = 0  # finished goods

        # Links to other agents (filled by model)
        self.connected_firms: List["FirmAgent"] = []
        self.employees: List[HouseholdAgent] = []

        # Statistics
        self.production: float = 0.0  # units produced this step
        self.consumption: float = 0.0  # units of inputs consumed this step

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

        # Budget reservation placeholders (set each step by prepare_budget)
        self._budget_input: float = 0.0
        self._budget_capital: float = 0.0
        self._budget_labor: float = 0.0
        self._budget_input_per_supplier: dict[int, float] = {} # New placeholder for independent input budgets

    # ---------------- Interaction helpers ----------------------------- #
    def hire_labor(self, household: HouseholdAgent, wage: float) -> bool:
        # Reject hire if paying the wage would dip into funds reserved for inputs/capital
        if self._budget_labor < wage:
            return False
        """Attempt to hire one unit of labour from *household*.

        Returns True if the contract succeeded (wage transferred),
        False otherwise (insufficient funds).
        """

        if self.money < wage:
            return False

        # Transfer wage and update reservation tracking
        self.money -= wage
        self._budget_labor -= wage
        household.money += wage

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
        return qty

    def sell_goods_to_firm(self, buyer: "FirmAgent", quantity: int = 1) -> int:
        """Inter-firm sale of intermediate goods (generic price=1)."""
        if quantity <= 0 or self.inventory_output <= 0:
            return 0

        qty = min(quantity, self.inventory_output)
        cost = qty * self.price
        
        # Handle budget checking based on buyer's sector
        if buyer.sector == "retail":
            # Retail: use original interchangeable input budget
            budget_in = getattr(buyer, "_budget_input", 0.0)
            budget_cap = getattr(buyer, "_budget_capital", 0.0)

            if cost > (budget_in + budget_cap):
                return 0  # Not enough dedicated funds
                
            if buyer.money < cost:
                return 0

            # Transfer money & inventory
            buyer.money -= cost

            # Deduct from budgets prioritising input budget, then capital
            use_in = min(cost, budget_in)
            buyer._budget_input -= use_in  # type: ignore[attr-defined]
            buyer._budget_capital -= (cost - use_in)  # type: ignore[attr-defined]
            
        else:
            # Non-retail: use independent input budget per supplier
            budget_per_supplier = getattr(buyer, "_budget_input_per_supplier", {})
            supplier_budget = budget_per_supplier.get(self.unique_id, 0.0)
            budget_cap = getattr(buyer, "_budget_capital", 0.0)

            if cost > (supplier_budget + budget_cap):
                return 0  # Not enough dedicated funds for this supplier
                
            if buyer.money < cost:
                return 0

            # Transfer money & inventory
            buyer.money -= cost

            # Deduct from supplier-specific budget first, then capital
            use_supplier = min(cost, supplier_budget)
            buyer._budget_input_per_supplier[self.unique_id] -= use_supplier
            buyer._budget_capital -= (cost - use_supplier)  # type: ignore[attr-defined]
        
        self.money += cost
        self.inventory_output -= qty
        # Register under this supplier's id inside buyer
        buyer.inventory_inputs[self.unique_id] += qty
        return qty

    # ---------------- Budgeting helper ---------------------------------- #
    def prepare_budget(self) -> None:
        """Allocate current cash across labour, inputs and capital reserves.

        The allocation is guided by the previous step's limiting factor so the
        firm directs more budget towards whichever input was scarce last time.
        For non-retail sectors, each input good from connected firms is treated
        as independent and gets separate budget allocation.
        """

        # Base weights: technical coefficients × price proxies (Leontief logic)
        avg_input_price = np.mean([s.price for s in self.connected_firms]) if self.connected_firms else 1.0

        lim = getattr(self, "limiting_factor", "")

        # Allocate capital budget only if capital was the previous bottleneck
        cap_weight = self.capital_coeff * self.price if lim == "capital" else 0.0

        # For retail, treat inputs as interchangeable (homogeneous)
        # For other sectors, treat each input type as independent
        if self.sector == "retail":
            # Original behavior: single input budget
            weights = {
                "labor": self.LABOR_COEFF * self.wage_offer,
                "input": self.INPUT_COEFF * avg_input_price,
                "capital": cap_weight,
            }
            
            # Slightly prioritise last limiting factor
            if lim in weights and weights[lim] > 0:
                weights[lim] *= 1.3  # modest amplification

            total_w = sum(weights.values())
            if total_w <= 0:
                total_w = 1.0

            # Cash to be allocated (keep small buffer unallocated for flexibility)
            liquid = self.money
            # Keep 5% buffer unallocated
            liquid_alloc = liquid * 0.95
            reserve_labor = liquid_alloc * weights["labor"] / total_w
            reserve_input = liquid_alloc * weights["input"] / total_w
            reserve_cap = liquid_alloc * weights["capital"] / total_w

            # Labour reserve is implicit – it will be spent via wage payments up to this amount
            self._budget_labor = reserve_labor

            # Explicitly track non-labour reservations to block overspending during hiring
            self._budget_input = reserve_input
            self._budget_capital = reserve_cap
            
        else:
            # Non-retail sectors: independent inputs
            # Each connected firm represents a different input type
            num_input_types = len(self.connected_firms) if self.connected_firms else 1
            
            # Base weights for each input type
            input_weights = {}
            for supplier in self.connected_firms:
                input_weights[supplier.unique_id] = self.INPUT_COEFF * supplier.price
            
            # If no connected firms, use average price as fallback
            if not input_weights:
                input_weights["generic"] = self.INPUT_COEFF * avg_input_price
            
            # Total input weight is sum of all individual input weights
            total_input_weight = sum(input_weights.values())
            
            weights = {
                "labor": self.LABOR_COEFF * self.wage_offer,
                "input_total": total_input_weight,
                "capital": cap_weight,
            }
            
            # Prioritise last limiting factor
            if lim in weights and weights[lim] > 0:
                weights[lim] *= 1.3
            
            total_w = sum(weights.values())
            if total_w <= 0:
                total_w = 1.0

            # Cash to be allocated
            liquid = self.money
            liquid_alloc = liquid * 0.95
            reserve_labor = liquid_alloc * weights["labor"] / total_w
            reserve_input_total = liquid_alloc * weights["input_total"] / total_w
            reserve_cap = liquid_alloc * weights["capital"] / total_w

            # Set budgets
            self._budget_labor = reserve_labor
            self._budget_capital = reserve_cap
            
            # Allocate input budget per supplier (independent inputs)
            self._budget_input_per_supplier = {}
            if self.connected_firms:
                for supplier in self.connected_firms:
                    # Proportional allocation based on supplier's price
                    supplier_share = input_weights[supplier.unique_id] / total_input_weight
                    self._budget_input_per_supplier[supplier.unique_id] = reserve_input_total * supplier_share
            else:
                # Fallback for no connected firms
                self._budget_input = reserve_input_total 

    # -------------------------------------------------------------------- #

    # ---------------- Mesa API (production) --------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Purchase inputs, transform them with labour into output, then sell surplus."""

        # ---------------- Damage recovery ----------------------------- #
        # ---------------- Wage adjustment ----------------------------- #
        # Continuous wage update inspired by matching/Phillips-curve logic
        tightness = 1.0 - getattr(self.model, "unemployment_rate_prev", 0.0)
        # Determine direction based on type of labour constraint ----------------
        if self.labor_limited_last_step:
            # If firm could not hire AND lacks cash for current wage → demand-limited
            if self.money < self.wage_offer:
                signal = -1.0  # wages too expensive relative to budget
            else:
                # Could afford but still lacked workers → supply-limited, raise wage
                signal = 1.0
        else:
            # Not labour-constrained → mild downward pressure
            signal = -0.5
        strength = 0.2  # responsiveness coefficient

        adjustment = 1 + strength * signal * tightness
        # Friction: wage cuts (adjustment < 1) dampened by 50 %
        if adjustment < 1:
            adjustment = 1 - (1 - adjustment) * 0.5

        self.wage_offer *= adjustment
        # Keep wage positive but allow unbounded growth
        self.wage_offer = float(max(0.01, self.wage_offer))

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

        # Ensure strictly positive price; no upper cap
        self.price = float(max(0.01, self.price))

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
        # ----------------------------------------------------------------
        
        target_output_from_labour = labour_units / self.LABOR_COEFF
        
        if self.sector == "retail":
            # RETAIL: Treat material inputs as homogeneous good
            # We just need INPUT_COEFF units in total per unit of output, 
            # regardless of which supplier they come from.
            target_material_needed = int(np.ceil(target_output_from_labour * self.INPUT_COEFF))
            current_material = int(sum(self.inventory_inputs.values()))
            required_qty = max(0, target_material_needed - current_material)

            if required_qty > 0 and self.connected_firms:
                # Randomise supplier search order
                for supplier in self.random.sample(self.connected_firms, len(self.connected_firms)):
                    if required_qty <= 0:
                        break
                    bought = supplier.sell_goods_to_firm(self, required_qty)
                    required_qty -= bought
                    
        else:
            # NON-RETAIL: Each input good from connected firms is independent
            # Need INPUT_COEFF units from EACH supplier for maximum production
            for supplier in self.connected_firms:
                target_per_supplier = int(np.ceil(target_output_from_labour * self.INPUT_COEFF))
                current_from_supplier = self.inventory_inputs.get(supplier.unique_id, 0)
                required_from_supplier = max(0, target_per_supplier - current_from_supplier)
                
                if required_from_supplier > 0:
                    bought = supplier.sell_goods_to_firm(self, required_from_supplier)

        # ----------------------------------------------------------------
        # 2. Compute possible output per Leontief: min(labour, material, capital)
        # ----------------------------------------------------------------

        if self.sector == "retail":
            # RETAIL: Use sum of all inputs (interchangeable)
            total_input_units = sum(self.inventory_inputs.values())
            max_output_from_inputs = total_input_units / self.INPUT_COEFF if self.connected_firms else float("inf")
        else:
            # NON-RETAIL: Each input is independent - limited by minimum available
            if self.connected_firms:
                min_input_units = min(
                    self.inventory_inputs.get(supplier.unique_id, 0) 
                    for supplier in self.connected_firms
                )
                max_output_from_inputs = min_input_units / self.INPUT_COEFF
            else:
                max_output_from_inputs = float("inf")

        max_output_from_capital = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")

        max_possible = min(labour_units / self.LABOR_COEFF, max_output_from_inputs, max_output_from_capital)
        possible_output = max_possible * self.damage_factor

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
            # Consume material inputs based on sector type
            if self.sector == "retail":
                # RETAIL: Consume inputs proportionally across suppliers
                remaining_use = possible_output * self.INPUT_COEFF
                for supp_id in list(self.inventory_inputs.keys()):
                    if remaining_use <= 0:
                        break
                    available = self.inventory_inputs[supp_id]
                    use_qty = min(available, remaining_use)
                    self.inventory_inputs[supp_id] -= use_qty
                    remaining_use -= use_qty
            else:
                # NON-RETAIL: Consume INPUT_COEFF units from each connected supplier
                input_per_supplier = possible_output * self.INPUT_COEFF
                for supplier in self.connected_firms:
                    supp_id = supplier.unique_id
                    if supp_id in self.inventory_inputs:
                        available = self.inventory_inputs[supp_id]
                        use_qty = min(available, input_per_supplier)
                        self.inventory_inputs[supp_id] -= use_qty

            # Add production to inventory
            self.inventory_output += possible_output

            self.consumption = possible_output * self.INPUT_COEFF

        # ----------------------------------------------------------------
        # 3. Clear employee list for next step
        # ----------------------------------------------------------------
        # Record labour count and limiting factor for next step's adjustments
        self.last_hired_labor = len(self.employees)
        self.labor_limited_last_step = (self.limiting_factor == "labor")
        self.employees.clear()

        # ---------------- Capital depreciation ------------------------ #
        DEPR = 0.005  # 0.5 % per step (quarterly), roughly 2 % annually
        self.capital_stock *= (1 - DEPR)

        # ---------------- Capital purchase stage ----------------------- #
        # Purchase additional capital whenever it is the current bottleneck
        if self.limiting_factor == "capital" and self._budget_capital > 0:
            max_affordable_units = int(self._budget_capital / self.price)
            if max_affordable_units <= 0:
                max_affordable_units = int(self.money / self.price)

            to_buy = max_affordable_units

            if to_buy > 0:
                # Candidate sellers: any firm with inventory (including upstream suppliers)
                sellers = [f for f in self.model.agents if isinstance(f, FirmAgent) and f is not self and f.inventory_output > 0]
                self.random.shuffle(sellers)
                for seller in sellers:
                    if to_buy <= 0:
                        break
                    qty = min(to_buy, seller.inventory_output)
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