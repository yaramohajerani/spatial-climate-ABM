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

        # Apply relocation cost similar to climate‐driven migration
        self.money *= (1 - self.RELOCATION_COST)

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

        # 2. Buy goods based on wealth and needs (no proximity restriction) -------------------------- #
        # Households need goods from different sectors in fixed ratios (configurable)
        # Base consumption on recent wages earned plus a small fraction of savings
        base_consumption = self.labor_sold * self.model.mean_wage  # spend what we earned
        savings_consumption = max(0, self.money - 50) * 0.1  # plus 10% of savings above $50
        consumption_budget = base_consumption + savings_consumption

        if consumption_budget > 0 and self.money > 0:
            # Cap at what we can actually afford
            consumption_budget = min(consumption_budget, self.money * 0.8)

            # Get consumption ratios from model (configurable via parameter file)
            # Default: 30% commodities, 70% manufacturing
            consumption_ratios = getattr(self.model, 'consumption_ratios', {
                'commodity': 0.3,
                'manufacturing': 0.7,
            })

            # Group firms by sector
            firms_by_sector: dict[str, list] = {}
            for f in self.model._firms:
                if f.inventory_output > 0:
                    sector = f.sector
                    if sector not in firms_by_sector:
                        firms_by_sector[sector] = []
                    firms_by_sector[sector].append(f)

            # Allocate budget to each sector and buy
            for sector, ratio in consumption_ratios.items():
                if sector not in firms_by_sector:
                    continue  # no firms in this sector with inventory

                sector_budget = consumption_budget * ratio
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

        # Apply relocation cost: lose a share of money
        self.money *= (1 - self.RELOCATION_COST)

        # Track migration for statistics
        self.model.migrants_this_step += 1

    # ---------------- End of Household.step() bookkeeping -------- #


class FirmAgent(Agent):
    """A firm with a simple Leontief production function and local trade."""

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

        # Track consecutive cycles of labor shortage for wage adjustment
        self.labor_shortage_cycles: int = 0  # consecutive steps where firm couldn't hire needed workers
        self.last_hired_labor: int = 0  # employees hired in previous step
        self.labor_demand: int = 0  # how many workers the firm wanted to hire

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
        self.sales_prev_step: float = 0.0
        self.revenue_last_step: float = 0.0
        self.sales_this_step: float = 0.0
        self.revenue_this_step: float = 0.0
        self.no_sales_streak: int = 0

        # ---------------- Risk behaviour parameters ------------------- #
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
        """Calculate fitness based on recent performance.

        Components:
        - Money growth (35%): Profitability, log-scaled to balance small vs large firms
        - Production level (25%): Absolute output matters
        - Peak maintenance (20%): Penalizes decline from recent peak, but not recovery
        - Survival (20%): Longevity bonus
        """
        if len(self.performance_history) < 2:
            return 0.0

        recent = self.performance_history[-min(5, len(self.performance_history)):]

        # 1. Money growth (35%) - log-scaled to balance small vs large firms
        money_start = max(1.0, recent[0]['money'])
        money_end = max(1.0, recent[-1]['money'])
        log_money_growth = np.log(money_end / money_start)
        money_score = np.tanh(log_money_growth)  # Bound to [-1, 1]

        # 2. Production level (25%) - absolute production matters
        productions = [r['production'] for r in recent]
        mean_production = np.mean(productions)
        # Normalize by a reasonable baseline (10 units is "good")
        production_score = np.tanh(mean_production / 10.0)

        # 3. Peak maintenance (20%) - penalize decline, don't penalize recovery
        # Measures how close current production is to recent peak
        # Recovery: current at/near peak → high score
        # Decline: current far below peak → low score
        # Stable: current equals peak → high score
        peak_production = max(productions)
        current_production = productions[-1]
        if peak_production > 0:
            peak_ratio = current_production / peak_production
        else:
            peak_ratio = 1.0  # No production history, neutral
        peak_score = max(0.0, min(1.0, peak_ratio))

        # 4. Survival bonus (20%) - longevity reward
        survival_score = min(1.0, self.survival_time / 20.0)

        # Combined fitness score
        fitness = (
            0.35 * (money_score + 1) / 2 +    # Shift from [-1,1] to [0,1]
            0.25 * production_score +          # Already [0,1] due to tanh of positive
            0.20 * peak_score +                # Already [0,1]
            0.20 * survival_score              # Already [0,1]
        )

        return max(0.0, min(1.0, fitness))
    
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
        # Supply-demand based wage adjustment
        unemployment_rate = getattr(self.model, "unemployment_rate_prev", 0.0)

        # Determine adjustment based on labor market conditions
        # Only raise wages after 4 consecutive cycles of labor shortage (persistent issue)
        LABOR_SHORTAGE_THRESHOLD = 4
        if self.labor_shortage_cycles >= LABOR_SHORTAGE_THRESHOLD:
            # Firm has persistently wanted more workers but couldn't get them
            if self.money < self.wage_offer * 2:
                # Can't afford workers - hold wages steady (don't cut, that kills the economy)
                signal = 0.0
            else:
                # Supply-limited: raise wages to attract workers
                signal = 1.0
        else:
            # Not labor-constrained - adjust based on market conditions
            if unemployment_rate > 0.2:
                # High unemployment: modest downward pressure
                signal = -0.5
            elif unemployment_rate < 0.05:
                # Very low unemployment: upward pressure
                signal = 0.5
            else:
                # Normal conditions: hold steady
                signal = 0.0

        # Moderate adjustment strength
        base_strength = 0.03 * self.strategy.get('wage_responsiveness', 1.0)
        adjustment = 1 + base_strength * signal

        # Bound adjustment to prevent extreme jumps
        adjustment = max(0.95, min(adjustment, 1.05))

        self.wage_offer *= adjustment
        # Keep wage positive with a meaningful floor
        self.wage_offer = float(max(0.1, min(self.wage_offer, 10.0)))

        # Recover 50% of remaining damage each step
        self.damage_factor += (1.0 - self.damage_factor) * 0.5
        self.damage_factor = min(1.0, max(0.0, self.damage_factor))

        # ---------------- Dynamic pricing ----------------------------- #
        # Supply-demand based pricing using inventory levels

        # Calculate cost floor based on firm's actual costs plus profit margin
        # Use firm's own wage_offer (actual cost) not model mean
        avg_input_price = 0.0
        if self.connected_firms:
            avg_input_price = float(np.mean([s.price for s in self.connected_firms]))
        # Variable cost per unit
        variable_cost = (
            self.wage_offer * self.LABOR_COEFF
            + avg_input_price * self.INPUT_COEFF
        )
        # Cost floor includes 20% profit margin to ensure firm solvency
        cost_floor = variable_cost * 1.2
        cost_floor = max(0.5, cost_floor)  # Minimum floor of 0.5

        # Supply-demand indicator: compare inventory to recent production
        # High inventory relative to sales = excess supply = lower price
        # Low inventory relative to sales = excess demand = raise price
        current_sales = self.sales_last_step

        # Target inventory level: roughly 2x recent sales
        target_inventory = max(2.0, current_sales * 2.0)
        inventory_ratio = self.inventory_output / target_inventory if target_inventory > 0 else 1.0

        if inventory_ratio > 2.0:
            # Too much inventory: modest price cut to clear stock
            price_adjustment = 0.97
        elif inventory_ratio > 1.5:
            # Slightly high inventory: small price cut
            price_adjustment = 0.99
        elif inventory_ratio < 0.5:
            # Low inventory, high demand: raise price
            price_adjustment = 1.03
        elif inventory_ratio < 0.8:
            # Slightly low inventory: small price increase
            price_adjustment = 1.01
        else:
            # Balanced: hold price
            price_adjustment = 1.0

        # Apply no-sales penalty more gently
        if current_sales <= 0 and self.inventory_output > 0:
            # Have inventory but no sales: modest price cut
            price_adjustment = min(price_adjustment, 0.95)

        self.price *= price_adjustment

        # Clamp prices to sensible bounds
        # Price should be above cost floor and below reasonable maximum
        # Use cached household money from model for efficiency
        avg_household_money = self.model.get_avg_household_money()
        max_reasonable_price = max(cost_floor * 3.0, avg_household_money * 0.5)

        self.price = float(max(cost_floor, min(self.price, max_reasonable_price)))

        # Reset per-step statistics
        self.production = 0.0
        self.consumption = 0.0

        # ---------------- Hazard-driven capital adjustment (learned response) ------------ #
        local_hazard = self._get_local_hazard()
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

        # Sum inputs from all suppliers (they are substitutable, not complementary)
        # This allows production even if one supplier is out of stock, as long as
        # total inputs from all suppliers meet the requirement.
        if self.connected_firms:
            total_input_units = sum(
                self.inventory_inputs.get(supplier.unique_id, 0)
                for supplier in self.connected_firms
            )
            max_output_from_inputs = total_input_units / self.INPUT_COEFF
        else:
            max_output_from_inputs = float("inf")

        max_output_from_capital = self.capital_stock / self.capital_coeff if self.capital_coeff else float("inf")

        max_possible = min(labour_units / self.LABOR_COEFF, max_output_from_inputs, max_output_from_capital)
        possible_output = max_possible * self.damage_factor

        capital_limited = possible_output < max_possible and (max_output_from_capital <= labour_units / self.LABOR_COEFF)

        # Determine primary limiting factor for diagnostic plotting
        # Compare theoretical maxima before damage factor applied
        labor_capacity = labour_units / self.LABOR_COEFF
        limits = {
            "labor": labor_capacity,
            "input": max_output_from_inputs,
            "capital": max_output_from_capital,
        }
        min_limit_val = min(limits.values())
        # Pick first factor that equals the min within small tolerance
        self.limiting_factor: str = next(k for k, v in limits.items() if abs(v - min_limit_val) < 1e-6)

        # Calculate how many workers the firm could productively use
        # This is based on input and capital constraints (what limits production besides labor)
        max_useful_output = min(max_output_from_inputs, max_output_from_capital)
        # Convert output capacity to labor demand (how many workers needed for that output)
        self.labor_demand = int(np.ceil(max_useful_output * self.LABOR_COEFF)) if max_useful_output < float("inf") else labour_units

        self.production = possible_output
        if possible_output > 0:
            # Total inputs needed = output * INPUT_COEFF (consumed proportionally from all suppliers)
            total_inputs_needed = possible_output * self.INPUT_COEFF
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
        # Record labour count and limiting factor for next step's adjustments
        self.last_hired_labor = len(self.employees)

        # Track consecutive cycles of labor shortage:
        # A firm has a labor shortage if:
        # 1. It hired fewer workers than it demanded (demand not met), AND
        # 2. It still had budget remaining to hire more workers (shortage due to unavailability)
        has_labor_shortage = (
            len(self.employees) < self.labor_demand and
            self._budget_labor >= self.wage_offer  # Could afford at least one more worker
        )
        if has_labor_shortage:
            self.labor_shortage_cycles += 1
        else:
            self.labor_shortage_cycles = 0  # Reset if shortage resolved
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
                # Use cached firm list from model for efficiency
                sellers = [f for f in self.model._firms if f is not self and f.inventory_output > 0]
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
        self.sales_prev_step = self.sales_last_step
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
    def _get_local_hazard(self) -> float:
        """Return the hazard value at the agent's current cell."""
        return self.model.hazard_map.get(self.pos, 0.0) 
