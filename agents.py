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
        
        # Note: pos is handled by Mesa's grid.place_agent(), not set here
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

        # Reset per-step statistics
        self.production = 0.0
        self.labor_sold = 0.0
        self.consumption = 0.0

        # ---------------- Heuristic relocation decision --------------- #
        if self._max_hazard_within_radius(self.risk_radius) > 0.1:
            self._relocate()

        # 1. Choose employer based on wage–distance utility (remote work allowed) --------------- #
        # Consider all firms of the same sector; distance is still a disutility but no hard radius.
        all_firms = [ag for ag in self.model.agents if isinstance(ag, FirmAgent) and ag.sector == self.sector]
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

        # 2. Buy goods based on wealth and needs (no proximity restriction) -------------------------- #
        # Households need variety from different trophic levels
        consumption_budget = self.money * 0.5
        
        if consumption_budget > 0:
            # Calculate trophic levels for all firms and find max
            firm_trophic_levels = {}
            max_trophic = 1.0
            firms_all = [f for f in self.model.agents if isinstance(f, FirmAgent)]
            for f in firms_all:
                trophic_level = 1.0 + len(f.connected_firms) * 0.3
                firm_trophic_levels[f.unique_id] = trophic_level
                max_trophic = max(max_trophic, trophic_level)
            
            # Randomly select 2-3 trophic level ranges for needed goods
            num_ranges = self.random.randint(2, 3)
            needed_ranges = []
            for _ in range(num_ranges):
                range_start = self.random.uniform(1.0, max_trophic - 0.2)
                range_width = self.random.uniform(0.2, 0.5)
                range_end = min(max_trophic, range_start + range_width)
                needed_ranges.append((range_start, range_end))
            
            for min_trophic, max_trophic_range in needed_ranges:
                # Find firms with outputs in this trophic range that have inventory
                available_firms = []
                for f in firms_all:
                    if f.inventory_output > 0:
                        trophic_level = firm_trophic_levels[f.unique_id]
                        if min_trophic <= trophic_level <= max_trophic_range:
                            available_firms.append(f)
                
                if available_firms:
                    # Choose cheapest firms in this range
                    range_budget = consumption_budget / len(needed_ranges)
                    affordable_firms = [f for f in available_firms if f.price <= range_budget]
                    if affordable_firms:
                        min_price = min(f.price for f in affordable_firms)
                        cheapest = [f for f in affordable_firms if abs(f.price - min_price) < 1e-6]
                        seller = self.random.choice(cheapest)
                        
                        # Buy as much as range budget allows
                        max_quantity = range_budget / seller.price
                        qty_bought = seller.sell_goods_to_household(self, quantity=max_quantity)
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

        # Set starting price based on sector (for simplicity assume all firms start with the same price)
        base_price_by_sector = {
            "commodity": 1.0,
            "agriculture": 1.0,
            "manufacturing": 1.0,
            "wholesale": 1.0,
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
        self.sales_total: float = 0.0  # units sold (households + firms) this step

        # Cumulative damage to productive capacity (1 = undamaged)
        self.damage_factor: float = 1.0

        # Sales tracking for demand-based pricing
        self.sales_last_step: float = 0.0
        self.revenue_last_step: float = 0.0
        self.sales_this_step: float = 0.0
        self.revenue_this_step: float = 0.0
        self.no_sales_streak: int = 0

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
        
        # Learning system components
        learning_config = getattr(model, 'learning_config', {})
        self.learning_enabled: bool = getattr(model, 'firm_learning_enabled', self.LEARNING_ENABLED)
        self.memory_length: int = learning_config.get('memory_length', self.MEMORY_LENGTH)
        self.mutation_rate: float = learning_config.get('mutation_rate', self.MUTATION_RATE)  
        self.adaptation_frequency: int = learning_config.get('adaptation_frequency', self.ADAPTATION_FREQUENCY)
        self.performance_history: list[dict] = []  # Track recent performance metrics
        self.strategy: dict[str, float] = self._initialize_strategy()
        self.fitness_score: float = 0.0
        self.steps_since_adaptation: int = 0
        
        # Survival tracking for evolutionary pressure
        self.birth_step: int = getattr(model, 'current_step', 0)
        self.survival_time: int = 0

    # ---------------- Learning System Methods ----------------------------- #
    def _initialize_strategy(self) -> dict[str, float]:
        """Initialize evolvable strategy parameters with small random variations."""
        return {
            'budget_labor_weight': self.random.uniform(0.8, 1.2),      # multiplier for labor budget allocation
            'budget_input_weight': self.random.uniform(0.8, 1.2),      # multiplier for input budget allocation  
            'budget_capital_weight': self.random.uniform(0.8, 1.2),    # multiplier for capital budget allocation
            'risk_sensitivity': self.random.uniform(0.5, 1.5),         # hazard response aggressiveness
            'price_aggressiveness': self.random.uniform(0.5, 1.5),     # pricing adjustment magnitude
            'wage_responsiveness': self.random.uniform(0.5, 1.5),      # wage adjustment responsiveness
        }
    
    def _record_performance(self) -> None:
        """Track current performance metrics for learning evaluation."""
        current_perf = {
            'step': getattr(self.model, 'current_step', 0),
            'money': self.money,
            'capital_stock': self.capital_stock,
            'production': self.production,
            'inventory': self.inventory_output,
            'limiting_factor': getattr(self, 'limiting_factor', ''),
            'wage_offer': self.wage_offer,
            'price': self.price,
        }
        
        self.performance_history.append(current_perf)
        
        # Keep only recent history
        if len(self.performance_history) > self.memory_length:
            self.performance_history = self.performance_history[-self.memory_length:]
    
    def _evaluate_fitness(self) -> float:
        """Calculate fitness based on recent performance."""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent = self.performance_history[-min(5, len(self.performance_history)):]
        
        # Components of fitness
        money_growth = (recent[-1]['money'] - recent[0]['money']) / max(1.0, recent[0]['money'])
        production_consistency = 1.0 - np.std([r['production'] for r in recent]) / max(1.0, np.mean([r['production'] for r in recent]))
        survival_bonus = min(1.0, self.survival_time / 20.0)  # bonus for surviving longer
        
        # Penalize extreme limiting factors
        recent_factors = [r['limiting_factor'] for r in recent]
        factor_diversity = len(set(recent_factors)) / max(1, len(recent_factors))
        
        # Combined fitness score
        fitness = (
            0.4 * np.tanh(money_growth * 2) +      # growth but diminishing returns
            0.3 * production_consistency +          # stable production
            0.2 * survival_bonus +                  # longevity reward
            0.1 * factor_diversity                  # balanced resource utilization
        )
        
        return max(0.0, fitness)
    
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
        # Track sales for demand-based pricing
        self.sales_this_step += qty
        self.revenue_this_step += total_cost
        self.sales_total += qty
        return qty

    def sell_goods_to_firm(self, buyer: "FirmAgent", quantity: int = 1) -> int:
        """Inter-firm sale of intermediate goods (generic price=1)."""
        if quantity <= 0 or self.inventory_output <= 0:
            return 0

        qty = min(quantity, self.inventory_output)
        cost = qty * self.price
        
        # Use independent input budget per supplier for all firms
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
        # Safely decrement supplier‐specific budget; initialise key if missing
        current_alloc = buyer._budget_input_per_supplier.get(self.unique_id, 0.0)
        buyer._budget_input_per_supplier[self.unique_id] = current_alloc - use_supplier
        buyer._budget_capital -= (cost - use_supplier)  # type: ignore[attr-defined]
        
        self.money += cost
        self.inventory_output -= qty
        # Register under this supplier's id inside buyer
        buyer.inventory_inputs[self.unique_id] += qty
        # Track sales for demand-based pricing
        self.sales_this_step += qty
        self.revenue_this_step += cost
        self.sales_total += qty
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

        # Allocate capital budget if capital was limiting OR a shock just hit
        cap_weight = 0.0
        if lim == "capital":
            cap_weight = self.capital_coeff * self.price
        else:
            # Seed some capital budget after damage or depleted stock
            if (self.capital_stock < self.original_capital_coeff * 5) or (self.damage_factor < 0.99) or (self.capital_coeff > self.original_capital_coeff * 1.05):
                cap_weight = self.capital_coeff * self.price * 0.3

        # All sectors: treat each input type as independent
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
            "labor": self.LABOR_COEFF * self.wage_offer * self.strategy.get('budget_labor_weight', 1.0),
            "input_total": total_input_weight * self.strategy.get('budget_input_weight', 1.0),
            "capital": cap_weight * self.strategy.get('budget_capital_weight', 1.0),
        }
        
        # Prioritise last limiting factor (learned response)
        if lim in weights and weights[lim] > 0:
            priority_multiplier = 1.0 + 0.3 * self.strategy.get('budget_' + lim + '_weight', 1.0)
            weights[lim] *= priority_multiplier
        
        total_w = sum(weights.values())
        if total_w <= 0:
            total_w = 1.0

        # Cash to be allocated
        liquid = max(0, self.money)  # Ensure non-negative
        liquid_alloc = liquid * 0.9
        reserve_labor = liquid_alloc * weights["labor"] / total_w
        # Guarantee enough for multiple hires so labour is not starved by budgeting
        reserve_labor = max(reserve_labor, self.wage_offer * 3)
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
        # Improved wage adjustment based on firm's actual financial constraints
        tightness = 1.0 - getattr(self.model, "unemployment_rate_prev", 0.0)
        
        # Determine adjustment based on firm's ability to afford labor
        if self.labor_limited_last_step:
            # If firm was labor-constrained, check if it's due to budget or supply
            if self.money < self.wage_offer * 2:  # Can't afford even 2 workers
                # Demand-limited: cut wages aggressively
                signal = -3.0  # Strong downward pressure when financially constrained
            else:
                # Supply-limited: raise wages to attract workers
                signal = 1.0
        else:
            # Not labor-constrained: mild downward pressure
            signal = -0.3
        
        # Increased responsiveness for wage cuts when firms can't afford labor (learned behavior)
        base_strength = 0.05 * self.strategy.get('wage_responsiveness', 1.0)
        if signal < 0 and self.money < self.wage_offer:
            strength = base_strength * 4  # Faster adjustment when financially constrained
        else:
            strength = base_strength  # Normal adjustment otherwise

        adjustment = 1 + strength * signal * tightness
        
        # Remove dampening for wage cuts - let market forces work
        # Only bound the adjustment to prevent extreme jumps
        adjustment = max(0.5, min(adjustment, 1.5))

        self.wage_offer *= adjustment
        # Keep wage positive and bounded to prevent wage explosion
        self.wage_offer = float(max(0.01, min(self.wage_offer, 10.0)))

        # Recover 50% of remaining damage each step
        self.damage_factor += (1.0 - self.damage_factor) * 0.5
        self.damage_factor = min(1.0, max(0.0, self.damage_factor))

        # ---------------- Dynamic pricing ----------------------------- #
        # Improved pricing based on supply-demand dynamics
        
        # Get household purchasing power indicator
        total_household_money = sum(getattr(ag, "money", 0) for ag in self.model.agents if isinstance(ag, HouseholdAgent))
        avg_household_money = total_household_money / max(1, sum(1 for ag in self.model.agents if isinstance(ag, HouseholdAgent)))
        
        # Calculate demand pressure based on inventory vs typical demand
        # Use a rolling average of recent production to avoid zero-production trap
        recent_production = getattr(self, "production", 0)
        
        # Learned pricing aggressiveness
        price_mult = self.strategy.get('price_aggressiveness', 1.0)
        affordable_cap = max(0.01, avg_household_money * 0.25)
        
        # Treat any sales (household or firm) as valid demand; only cut if literally no sales
        if self.sales_last_step <= 0 and self.inventory_output > 0:
            # No sales at all → aggressively cut prices until demand appears
            cut_factor = 0.6 if self.inventory_output > 5 else 0.75
            if self.no_sales_streak >= 1:
                cut_factor *= 0.8  # escalate cuts after multiple dry steps
            self.price *= cut_factor
            self.price = min(self.price, affordable_cap)
            # Skip further upward adjustments this step
        elif recent_production == 0 and self.inventory_output <= 2:
            # No production + low inventory = scarcity → raise prices
            self.price *= (1.0 + 0.05 * price_mult)
        elif recent_production == 0 and self.inventory_output > 5:
            # No production but plenty of stock → ease prices to clear inventory
            self.price *= (1.0 - 0.02 * price_mult)
        elif recent_production > 0:
            # Normal supply-demand pricing with learned aggressiveness
            inventory_ratio = self.inventory_output / max(1, recent_production)
            if inventory_ratio < 0.5 and self.price < avg_household_money * 0.3:
                # Low inventory and affordable → raise price
                self.price *= (1.0 + 0.02 * price_mult)
            elif inventory_ratio > 3.0 or self.price > avg_household_money * 0.5:
                # High inventory or unaffordable → lower price
                self.price *= (1.0 - 0.02 * price_mult)
        
        # Special case: if we're cash-constrained, try to raise prices to improve margins
        if self.money < self.wage_offer * 3 and self.price < avg_household_money * 2.0 and self.sales_last_step > 0:
            self.price *= 1.03  # Increase prices to improve cash flow
        
        # Only prevent truly extreme price explosions that would break the model
        max_reasonable_price = avg_household_money * 1000.0  
        self.price = float(max(0.01, min(self.price, max_reasonable_price)))
        if self.sales_last_step <= 0 and self.inventory_output > 0:
            self.price = min(self.price, affordable_cap)
        
        # Additional safety: if price gets extremely high relative to affordability,
        # add some gentle downward pressure (but still allow substantial growth)
        if self.price > avg_household_money * 500.0:
            self.price *= 0.999  # Very gentle correction for extreme prices

        # Reset per-step statistics
        self.production = 0.0
        self.consumption = 0.0

        # ---------------- Hazard-driven capital adjustment (learned response) ------------ #
        local_hazard = self._max_hazard_within_radius(self.risk_radius)
        risk_sensitivity = self.strategy.get('risk_sensitivity', 1.0)
        
        if local_hazard > 0.1:
            # Increase capital requirement based on learned risk sensitivity
            capital_increase = 1.0 + (0.2 * risk_sensitivity)
            self.capital_coeff *= capital_increase
        else:
            # Gradually relax back towards original coefficient (faster if less risk-sensitive)
            if self.capital_coeff > self.original_capital_coeff:
                adjusted_relaxation = self.relaxation_ratio * (2.0 - risk_sensitivity)  # less sensitive = faster relaxation
                decay = (self.capital_coeff - self.original_capital_coeff) * adjusted_relaxation
                self.capital_coeff = max(self.original_capital_coeff, self.capital_coeff - decay)

        labour_units = self._labor_available()

        # ----------------------------------------------------------------
        # 1. Ensure each required input type has enough stock to match labour
        # ----------------------------------------------------------------
        
        # Estimate feasible output given capital and damage to avoid over-ordering inputs
        cap_limit = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")
        target_output = min(labour_units / self.LABOR_COEFF, cap_limit) * self.damage_factor
        target_output = max(0.0, target_output)
        
        # Each input good from connected firms is independent
        # Need INPUT_COEFF units from EACH supplier for maximum production
        for supplier in self.connected_firms:
            target_per_supplier = int(np.ceil(target_output * self.INPUT_COEFF))
            current_from_supplier = self.inventory_inputs.get(supplier.unique_id, 0)
            required_from_supplier = max(0, target_per_supplier - current_from_supplier)
            
            if required_from_supplier > 0:
                supplier.sell_goods_to_firm(self, required_from_supplier)

        # ----------------------------------------------------------------
        # 2. Compute possible output per Leontief: min(labour, material, capital)
        # ----------------------------------------------------------------

        # Each input is independent - limited by minimum available
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
            # Consume INPUT_COEFF units from each connected supplier
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
        self.capital_stock *= (1 - DEPR)

        # ---------------- Capital purchase stage ----------------------- #
        # Purchase additional capital whenever it is the current bottleneck
        if self.limiting_factor == "capital" and self._budget_capital > 0 and self.money > 0:
            # Only attempt capital purchase if we have actual money and dedicated budget
            available_funds = min(self._budget_capital, self.money * 0.5)  # Don't spend all money on capital
            
            if available_funds > 0:
                # Find cheapest available capital goods
                sellers = [f for f in self.model.agents if isinstance(f, FirmAgent) and f is not self and f.inventory_output > 0]
                if sellers:
                    # Sort by price to buy from cheapest first
                    sellers.sort(key=lambda f: f.price)
                    
                    remaining_budget = available_funds
                    for seller in sellers:
                        if remaining_budget <= 0:
                            break
                        
                        # Calculate how much we can afford from this seller
                        max_affordable = int(remaining_budget / seller.price)
                        if max_affordable <= 0:
                            continue
                            
                        qty = min(max_affordable, seller.inventory_output)
                        bought = seller.sell_goods_to_firm(self, qty)
                        if bought:
                            self.capital_stock += bought
                            remaining_budget -= bought * seller.price

        # Persist sales metrics for next step's pricing decisions
        if self.sales_this_step <= 0:
            self.no_sales_streak += 1
        else:
            self.no_sales_streak = 0
        self.sales_last_step = self.sales_this_step
        self.revenue_last_step = self.revenue_this_step
        self.sales_this_step = 0.0
        self.revenue_this_step = 0.0
        self.sales_total = 0.0

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
