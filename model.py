from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from agents import FirmAgent, HouseholdAgent
from hazard_utils import lazy_hazard_from_geotiffs, LazyHazard
from damage_functions import get_damage_functions, get_region_from_coords

Coords = Tuple[int, int]


class EconomyModel(Model):
    """Spatial ABM of an economy subject to climate risk."""

    FINAL_CONSUMPTION_SECTORS = {"retail", "wholesale", "services"}
    SECTOR_ORDER = {
        "commodity": 0,
        "agriculture": 0,
        "manufacturing": 1,
        "retail": 2,
        "wholesale": 2,
        "services": 2,
    }

    def __init__(
        self,
        # Grid size is derived from the hazard raster; manual override is still
        # possible for testing but not recommended.
        width: int | None = None,
        height: int | None = None,
        num_households: int = 100,
        num_firms: int = 20,
        # Iterable of (return_period, start_step, end_step, hazard_type, path) tuples
        hazard_events: Iterable[Tuple[int, int, int, str, str]] | None = None,
        seed: int | None = None,
        start_year: int = 0,
        steps_per_year: int = 4,
        firm_topology_path: str | None = None,
        apply_hazard_impacts: bool = True,
        learning_params: dict | None = None,
        consumption_ratios: dict | None = None,
        grid_resolution: float = 1.0,
        household_relocation: bool = False,
    ) -> None:  # noqa: D401
        super().__init__(seed=seed)

        # Ensure NumPy uses the same seed so hazard sampling is reproducible
        if seed is not None:
            np.random.seed(seed)

        # Flag to toggle whether sampled hazards actually affect agents.
        # If False, hazards are still sampled to preserve random draws but
        # impacts (capital loss, damage, relocation triggers) are disabled.
        self.apply_hazard_impacts: bool = apply_hazard_impacts

        # Flag to toggle household relocation (both hazard-driven and job-driven).
        # If False, households stay in place regardless of hazards or employment.
        self.household_relocation_enabled: bool = household_relocation

        # Learning system parameters
        self.learning_config = learning_params or {}
        self.firm_learning_enabled: bool = self.learning_config.get('enabled', True)
        self.min_money_survival: float = self.learning_config.get('min_money_survival', 1.0)
        self.replacement_frequency: int = self.learning_config.get('replacement_frequency', 10)

        # Household consumption ratios across final-good sectors.
        # Upstream sectors sell to firms, not directly to households.
        self.consumption_ratios: dict = consumption_ratios or {
            'retail': 1.0,
        }
        self.final_consumption_sectors = set(self.FINAL_CONSUMPTION_SECTORS)

        # -------------------- Performance optimization caches -------------------- #
        # Cached agent lists by type (updated when agents added/removed)
        self._households: List[HouseholdAgent] = []
        self._firms: List[FirmAgent] = []
        # Cached agents by sector
        self._firms_by_sector: Dict[str, List[FirmAgent]] = defaultdict(list)

        # Calendar mapping -------------------------------------------------- #
        self.start_year: int = start_year
        # Calendar granularity (e.g. 4 → quarterly, 12 → monthly)
        self.steps_per_year: int = steps_per_year

        # --- Spatial environment & custom topology --- #
        self._firm_topology: dict | None = None
        if firm_topology_path is not None:
            import json, pathlib
            topo_path = Path(firm_topology_path)
            if not topo_path.exists():
                raise FileNotFoundError(f"Firm topology JSON not found: {topo_path}")
            self._firm_topology = json.loads(topo_path.read_text())

            # Override num_firms to match the topology file so downstream
            # components (e.g. dashboards, logging) see the correct value.
            num_firms = len(self._firm_topology.get("firms", []))

        if hazard_events is None:
            raise ValueError("hazard_events must be provided.")

        # Group by hazard type while preserving order to keep mapping consistent
        grouped_files: dict[str, list[Tuple[int, str]]] = defaultdict(list)
        grouped_ranges: dict[str, list[Tuple[int, int]]] = defaultdict(list)

        for rp, start, end, htype, path in hazard_events:
            grouped_files[htype].append((rp, path))
            grouped_ranges[htype].append((start, end))

        # Store mapping of event index -> (start, end) per hazard type
        self._hazard_event_ranges: dict[str, List[Tuple[int, int]]] = dict(grouped_ranges)

        # Use lazy hazard loading for memory efficiency (samples on-demand)
        # This reduces memory from ~4GB to <1MB for global hazard datasets
        self.hazards: dict[str, LazyHazard] = {}

        first_haz = None
        for htype, grp in grouped_files.items():
            haz, _, _ = lazy_hazard_from_geotiffs(grp, haz_type=htype)
            # Store lazy hazard that samples on-demand
            self.hazards[htype] = haz
            if first_haz is None:
                first_haz = haz

        # Agent grid resolution in degrees per cell (e.g., 1.0, 0.5, 0.25)
        # This is decoupled from the hazard raster resolution.
        # When sampling hazards, we convert grid positions to lon/lat
        # and sample at those coordinates from the full-resolution raster.
        self.grid_resolution: float = grid_resolution

        # Build lon/lat arrays for agent placement
        self.lon_vals = np.arange(-180 + grid_resolution/2, 180, grid_resolution)
        self.lat_vals = np.arange(-90 + grid_resolution/2, 90, grid_resolution)

        # ---------------- Derived spatial metrics ------------------- #
        # Translate a desired 1° geographic radius into grid cells so that
        # household ↔ firm distance is resolution‐independent.
        if len(self.lon_vals) > 1:
            lon_step = float(np.mean(np.diff(self.lon_vals)))
        else:
            lon_step = 1.0
        if len(self.lat_vals) > 1:
            lat_step = float(np.mean(np.diff(self.lat_vals)))
        else:
            lat_step = 1.0

        cell_deg = max(1e-6, min(abs(lon_step), abs(lat_step)))  # prevent div-by-zero
        self.work_radius: int = int(np.ceil(1.0 / cell_deg))
        # Avoid pathological radii on very fine grids
        self.work_radius = max(3, min(self.work_radius, 30))

        # Derive grid dimensions from the raster unless explicitly overridden
        if width is None:
            width = len(self.lon_vals)
        if height is None:
            height = len(self.lat_vals)

        self.grid = MultiGrid(width, height, torus=False)
        # Alias required by Mesa visualisation helpers (they expect .space)
        self.space = self.grid  # type: ignore[assignment]
        self.valid_coordinates: List[Coords] = [
            (x, y) for y in range(height) for x in range(width)
        ]

        # Initialize per-cell hazard depth map (populated on-demand for agent cells only)
        # This is a sparse dict, not pre-populated for all cells
        self.hazard_map: Dict[Coords, float] = {}

        # Base wage used by firms when hiring labour
        self.mean_wage = 1.0
        # Initial wage preserved as constant for minimum wage floor (ILO convention)
        self.initial_mean_wage: float = 1.0

        # DataCollector – track aggregate production each step for inspection
        # Note: Uses cached agent lists (m._firms, m._households) for efficiency
        self.datacollector = DataCollector(
            model_reporters={
                "Firm_Production": lambda m: sum(f.production for f in m._firms),
                "Firm_Consumption": lambda m: sum(f.consumption for f in m._firms),
                "Firm_Wealth": lambda m: sum(f.money for f in m._firms),
                "Firm_Capital": lambda m: sum(f.capital_stock for f in m._firms),
                "Firm_Profits": lambda m: sum(f.profit_this_step for f in m._firms),
                "Firm_Dividends_Paid": lambda m: sum(f.dividends_paid_this_step for f in m._firms),
                "Firm_Investment_Spending": lambda m: sum(f.investment_spending_this_step for f in m._firms),
                "Firm_Inventory": lambda m: sum(f.inventory_output for f in m._firms),
                "Household_Wealth": lambda m: sum(h.money for h in m._households),
                "Household_Labor_Sold": lambda m: sum(h.labor_sold for h in m._households),
                "Household_Consumption": lambda m: sum(h.consumption for h in m._households),
                "Household_Labor_Income": lambda m: sum(h.labor_income_this_step for h in m._households),
                "Household_Dividend_Income": lambda m: sum(h.dividend_income_this_step for h in m._households),
                "Household_Capital_Income": lambda m: sum(h.capital_income_this_step for h in m._households),
                "Average_Risk": lambda m: np.mean(list(m.hazard_map.values())),
                "Mean_Wage": lambda m: m.mean_wage,
                "Mean_Price": lambda m: np.mean([f.price for f in m._firms]) if m._firms else 0.0,
                "Total_Money": lambda m: m.total_money(),
                "Money_Drift": lambda m: m.total_money() - getattr(m, "initial_total_money", 0.0),
                "Labor_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "labor"),
                "Capital_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "capital"),
                "Input_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "input"),
                "Demand_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "demand"),
                "Average_Firm_Fitness": lambda m: np.mean([f.fitness_score for f in m._firms]) if m._firms else 0.0,
                "Firm_Replacements": lambda m: getattr(m, 'total_firm_replacements', 0),
                "Total_Sales": lambda m: sum(f.sales_last_step for f in m._firms),
                "Total_Firms": lambda m: len(m._firms),
                "Flooded_Firms": lambda m: sum(1 for f in m._firms if m.hazard_map.get(f.pos, 0) > 0),
                "Flooded_Households": lambda m: sum(1 for h in m._households if m.hazard_map.get(h.pos, 0) > 0),
            },
            agent_reporters={
                "money": lambda a: getattr(a, "money", np.nan),
                "production": lambda a: getattr(a, "production", 0.0),
                "consumption": lambda a: getattr(a, "consumption", 0.0),
                "labor_sold": lambda a: getattr(a, "labor_sold", 0.0),
                "capital": lambda a: getattr(a, "capital_stock", getattr(a, "capital", 0.0)),
                "limiting_factor": lambda a: getattr(a, "limiting_factor", ""),
                "price": lambda a: getattr(a, "price", np.nan),
                "inventory": lambda a: getattr(a, "inventory_output", 0.0),
                "input_inventory": lambda a: sum(getattr(a, "inventory_inputs", {}).values()) if hasattr(a, "inventory_inputs") else 0.0,
                "wage": lambda a: getattr(a, "wage_offer", np.nan),
                "sector": lambda a: getattr(a, "sector", ""),
                "type": lambda a: type(a).__name__,
                "fitness": lambda a: getattr(a, "fitness_score", np.nan),
                "survival_time": lambda a: getattr(a, "survival_time", 0),
                "sales_last_step": lambda a: getattr(a, "sales_last_step", np.nan),
                "household_sales_last_step": lambda a: getattr(a, "household_sales_last_step", np.nan),
                "labor_income": lambda a: getattr(a, "labor_income_this_step", np.nan),
                "dividend_income": lambda a: getattr(a, "dividend_income_this_step", np.nan),
                "capital_income": lambda a: getattr(a, "capital_income_this_step", np.nan),
                "profit": lambda a: getattr(a, "profit_this_step", np.nan),
                "dividends_paid": lambda a: getattr(a, "dividends_paid_this_step", np.nan),
                "investment_spending": lambda a: getattr(a, "investment_spending_this_step", np.nan),
            },
        )

        # ---------------- Land mask to avoid placing agents in the ocean ---------------- #
        self.land_coordinates: List[Coords] = self._compute_land_coordinates()
        if not self.land_coordinates:
            raise ValueError(
                "No land cells found within the hazard raster extent – cannot initialise agents."
            )

        # --- Create agents & build trade network --- #
        self._init_agents(num_households, num_firms)
        # Build trade network: if topology was provided it already specified
        # firm–firm edges, but we still need to connect households to nearby
        # employers/shops. If no topology → build both firm & household links.

        if self._firm_topology is None:
            # Randomly generate full network (firms + households)
            self._build_trade_network()
        else:
            # Only households remain to be wired up.
            self._connect_households_to_firms()

        # Populate agent caches after all agents created
        self._rebuild_agent_caches()

        # NOTE: With LazyHazard, we no longer need CLIMADA exposures/centroids.
        # Hazard sampling is done directly from GeoTIFF files at agent locations.

        # Demand-based initial inventories and working capital.
        self._initialize_firm_operating_state()
        self.initial_total_money: float = self.total_money()

        # --- Step counter --- #
        self.current_step: int = 0

        # Learning system tracking
        self.steps_since_replacement: int = 0
        self.total_firm_replacements: int = 0
        self._debug_recent_replacements: list = []  # Track replaced agent IDs for debugging

    # --------------------------------------------------------------------- #
    #                        CACHE MANAGEMENT                               #
    # --------------------------------------------------------------------- #
    def _rebuild_agent_caches(self) -> None:
        """Rebuild all agent caches from scratch. Call after bulk agent changes."""
        self._households.clear()
        self._firms.clear()
        self._firms_by_sector.clear()

        for ag in self.agents:
            if isinstance(ag, HouseholdAgent):
                self._households.append(ag)
            elif isinstance(ag, FirmAgent):
                self._firms.append(ag)
                self._firms_by_sector[ag.sector].append(ag)

    def _register_agent(self, agent) -> None:
        """Register a single agent in the caches."""
        if isinstance(agent, HouseholdAgent):
            self._households.append(agent)
        elif isinstance(agent, FirmAgent):
            self._firms.append(agent)
            self._firms_by_sector[agent.sector].append(agent)

    def _unregister_agent(self, agent) -> None:
        """Remove a single agent from the caches."""
        if isinstance(agent, HouseholdAgent):
            if agent in self._households:
                self._households.remove(agent)
        elif isinstance(agent, FirmAgent):
            if agent in self._firms:
                self._firms.remove(agent)
            if agent in self._firms_by_sector.get(agent.sector, []):
                self._firms_by_sector[agent.sector].remove(agent)

    def get_final_consumption_ratios(self) -> Dict[str, float]:
        """Return household demand shares over final-good sectors only."""

        ratios = {
            sector: float(weight)
            for sector, weight in self.consumption_ratios.items()
            if sector in self.final_consumption_sectors and float(weight) > 0
        }
        if not ratios:
            available_final = {
                firm.sector
                for firm in self._firms
                if firm.sector in self.final_consumption_sectors
            }
            if not available_final:
                return {}
            equal_share = 1.0 / len(available_final)
            return {sector: equal_share for sector in sorted(available_final)}

        total = sum(ratios.values())
        if total <= 0:
            return {}
        return {sector: weight / total for sector, weight in ratios.items()}

    def total_money(self) -> float:
        """Return the total financial money held by firms and households."""
        return sum(f.money for f in self._firms) + sum(h.money for h in self._households)

    def distribute_household_income(self, amount: float, *, income_kind: str) -> None:
        """Distribute firm payouts evenly across the household sector."""
        if amount <= 0 or not self._households:
            return

        per_household = amount / len(self._households)
        for household in self._households:
            household.money += per_household
            if income_kind == "dividend":
                household.dividend_income_this_step += per_household
            elif income_kind == "capital":
                household.capital_income_this_step += per_household
            else:
                raise ValueError(f"Unknown household income kind: {income_kind}")

    def transfer_household_equity_to_firm(self, firm: FirmAgent, amount: float) -> float:
        """Recapitalize a firm by drawing cash proportionally from households.

        This models households providing equity finance to a reorganized firm
        without creating or destroying money inside the closed economy.
        """
        if amount <= 0 or not self._households:
            return 0.0

        total_household_money = sum(h.money for h in self._households)
        if total_household_money <= 0:
            return 0.0

        funded_amount = min(amount, total_household_money)
        for household in self._households:
            contribution = funded_amount * (household.money / total_household_money)
            household.money -= contribution

        firm.money += funded_amount
        return funded_amount

    def _sector_priority(self, firm: FirmAgent) -> tuple[int, float]:
        """Sort firms by broad supply-chain tier with random tie breaks."""

        return (self.SECTOR_ORDER.get(firm.sector, 1), self.random.random())

    def _solve_initial_expected_sales(self) -> Dict[int, float]:
        """Bootstrap firm demand from final-demand shares scaled to labour supply."""

        expected_sales = {firm.unique_id: 0.0 for firm in self._firms}
        final_ratios = self.get_final_consumption_ratios()
        total_ratio = sum(final_ratios.values())
        if total_ratio <= 0:
            return expected_sales

        # Start from one unit of household expenditure distributed by the
        # configured final-demand shares, then scale the resulting economy-wide
        # flow so the implied labour demand matches the available workforce.
        for sector, ratio in final_ratios.items():
            sector_firms = [f for f in self._firms if f.sector == sector]
            if not sector_firms:
                continue
            mean_price = float(np.mean([max(f.price, 0.5) for f in sector_firms]))
            if mean_price <= 0:
                continue
            sector_units = (ratio / total_ratio) / mean_price
            per_firm_units = sector_units / len(sector_firms)
            for firm in sector_firms:
                expected_sales[firm.unique_id] += per_firm_units

        base_demand = expected_sales.copy()

        # Iterate q = d + A q, where A propagates intermediate-input demand upstream.
        for _ in range(200):
            updated = base_demand.copy()
            for buyer in self._firms:
                if buyer.INPUT_COEFF <= 0 or not buyer.connected_firms:
                    continue
                supplier_share = buyer.INPUT_COEFF / len(buyer.connected_firms)
                if supplier_share <= 0:
                    continue
                buyer_sales = expected_sales.get(buyer.unique_id, 0.0)
                if buyer_sales <= 0:
                    continue
                for supplier in buyer.connected_firms:
                    updated[supplier.unique_id] += supplier_share * buyer_sales

            max_delta = max(abs(updated[uid] - expected_sales[uid]) for uid in expected_sales) if expected_sales else 0.0
            expected_sales = updated
            if max_delta < 1e-6:
                break

        labour_per_unit_expenditure = sum(
            expected_sales[firm.unique_id] * firm.LABOR_COEFF
            for firm in self._firms
        )
        available_labour = float(len(self._households))
        if labour_per_unit_expenditure > 0 and available_labour > 0:
            scale = available_labour / labour_per_unit_expenditure
            expected_sales = {
                uid: sales * scale
                for uid, sales in expected_sales.items()
            }

        return expected_sales

    def _seed_firm_operating_state(
        self,
        firm: FirmAgent,
        *,
        expected_sales: float,
        inventory_coverage: float = 1.0,
        capital_coverage: float = 1.0,
    ) -> None:
        """Set initial demand expectations, stock buffers, and productive capacity."""

        expected_sales = max(1.0, float(expected_sales))
        inventory_target = max(1.0, expected_sales * inventory_coverage)
        capital_target = max(1.0, expected_sales * firm.capital_coeff * capital_coverage)
        avg_input_price = float(np.mean([s.price for s in firm.connected_firms])) if firm.connected_firms else 0.0
        unit_variable_cost = (
            firm.wage_offer * firm.LABOR_COEFF
            + avg_input_price * firm.INPUT_COEFF
        )
        working_capital = max(10.0, expected_sales * unit_variable_cost * 1.25)

        firm.expected_sales = expected_sales
        firm.base_inventory_target = inventory_target
        firm.base_capital_target = capital_target
        firm.target_capital_stock = capital_target
        firm.inventory_output = max(firm.inventory_output, inventory_target)
        firm.capital_stock = max(firm.capital_stock, capital_target)
        firm.money = max(firm.money, working_capital)

    def _initialize_firm_operating_state(self) -> None:
        """Initialize firms from demand-consistent inventories and working capital."""

        expected_sales = self._solve_initial_expected_sales()
        for firm in self._firms:
            self._seed_firm_operating_state(
                firm,
                expected_sales=expected_sales.get(firm.unique_id, 1.0),
            )

    # --------------------------------------------------------------------- #
    #                             INITIALISERS                               #
    # --------------------------------------------------------------------- #
    def _compute_land_coordinates(self) -> List[Coords]:
        """Return list of grid coordinates whose lon/lat fall within any country polygon.

        We use Natural Earth low-resolution country boundaries available via
        ``geopandas``. The geometry union is evaluated once at model
        initialisation; the resulting list is reused whenever we need to
        randomly sample land cells (e.g. for placing agents).
        """

        try:
            world = gpd.read_file("./data/ne_110m_admin_0_countries")
            # Exclude Antarctica so agents are not spawned on that continent
            world = world[world["CONTINENT"] != "Antarctica"]
        except Exception:  # pragma: no cover – dataset missing / offline env
            # If the Natural Earth dataset is unavailable fall back to using all
            # cells to avoid crashing, but log a warning so the user is aware.
            import warnings

            warnings.warn(
                "Natural Earth dataset could not be loaded – agents may be placed over water.",
                RuntimeWarning,
            )
            return self.valid_coordinates.copy()

        land_geom = world.geometry.unary_union  # shapely (multi-)polygon

        land_coords: List[Coords] = []
        for x, y in self.valid_coordinates:
            lon = float(self.lon_vals[x])
            lat = float(self.lat_vals[y])
            if land_geom.contains(Point(lon, lat)):
                land_coords.append((x, y))

        return land_coords

    def _init_agents(self, num_households: int, num_firms: int) -> None:
        """Place households and firms randomly on grid."""
        # ---------------- Households ---------------- #
        if self._firm_topology is not None and num_firms:
            # Place households close to existing firms to guarantee labour supply.
            firm_agents_list: list[FirmAgent] = []  # populated later for firms
        else:
            firm_agents_list = []  # placeholder

        # First create a temporary list to hold household positions
        hh_positions: list[Coords] = []

        if self._firm_topology is not None:
            # We will place each household within radius 3 of a random firm.
            # Fetch firm positions once we finish creating firms below.
            pass  # postpone until firms created
        else:
            filtered_land = [c for c in self.land_coordinates if float(self.lat_vals[c[1]]) <= 60.0]
            if not filtered_land:
                filtered_land = self.land_coordinates
            for _ in range(num_households):
                pos = self.random.choice(filtered_land)
                hh_positions.append(pos)

        # ---------------- Firms --------------------- #
        if self._firm_topology is not None:
            id_to_agent: dict[int, FirmAgent] = {}
            for firm in self._firm_topology.get("firms", []):
                # Determine position: either explicit grid x,y or lon/lat
                if "x" in firm and "y" in firm:
                    x, y = int(firm["x"]), int(firm["y"])
                elif "lon" in firm and "lat" in firm:
                    # Find nearest grid cell
                    lon, lat = float(firm["lon"]), float(firm["lat"])
                    x = int(np.argmin(np.abs(self.lon_vals - lon)))  # type: ignore[arg-type]
                    y = int(np.argmin(np.abs(self.lat_vals - lat)))  # type: ignore[arg-type]
                else:
                    raise ValueError("Firm entry must contain either (x,y) or (lon,lat)")

                if (x, y) not in self.valid_coordinates:
                    raise ValueError(f"Firm {firm['id']} location out of grid bounds: {(x, y)}")

                ag = FirmAgent(
                    model=self,
                    pos=(x, y),
                    sector=firm.get("sector", "manufacturing"),
                )
                ag.capital_stock = float(firm.get("capital", 1.0))
                self.grid.place_agent(ag, (x, y))
                id_to_agent[firm["id"]] = ag

            # Now we have populated firm_agents_list
            firm_agents_list = list(id_to_agent.values())

            # Build directed edges (supplier -> buyer)
            for edge in self._firm_topology.get("edges", []):
                src_id, dst_id = edge["src"], edge["dst"]
                try:
                    supplier = id_to_agent[src_id]
                    buyer = id_to_agent[dst_id]
                except KeyError as exc:
                    raise ValueError(f"Edge references unknown firm id: {exc}") from exc
                buyer.connected_firms.append(supplier)

        else:
            # Limit random placement to economically active latitudes (≤ 60°N)
            filtered_land = [c for c in self.land_coordinates if float(self.lat_vals[c[1]]) <= 60.0]
            if not filtered_land:
                filtered_land = self.land_coordinates  # fall back if raster truncated

            for _ in range(num_firms):
                pos = self.random.choice(filtered_land)
                agent = FirmAgent(model=self, pos=pos, sector="manufacturing")
                self.grid.place_agent(agent, pos)
                firm_agents_list.append(agent)

        # ---------------- Place households if not already done --------------- #
        # For randomly generated networks (no topology), we still need household positions
        if not hh_positions and self._firm_topology is None:
            for _ in range(num_households):
                firm = self.random.choice(firm_agents_list)
                fx, fy = firm.pos
                candidates = []
                for dx in range(-self.work_radius, self.work_radius + 1):
                    for dy in range(-self.work_radius, self.work_radius + 1):
                        coord = (fx + dx, fy + dy)
                        if coord in self.land_coordinates:
                            candidates.append(coord)
                pos = self.random.choice(candidates) if candidates else firm.pos
                hh_positions.append(pos)

        # ---------------- Allocate households across sectors ------------------- #
        if self._firm_topology is not None and firm_agents_list:
            # Determine sector distribution based on number of firms per sector
            from collections import Counter

            sector_counts = Counter(f.sector for f in firm_agents_list)
            total_firms = sum(sector_counts.values())

            # Initial allocation proportional to firm share
            sector_alloc: dict[str, int] = {
                sec: int(round(num_households * cnt / total_firms)) for sec, cnt in sector_counts.items()
            }

            # Adjust rounding so total matches num_households
            diff = num_households - sum(sector_alloc.values())
            if diff != 0:
                # Distribute remainder starting with largest sectors
                sorted_secs = sorted(sector_counts.items(), key=lambda t: -t[1])
                idx = 0
                while diff != 0 and sorted_secs:
                    sec = sorted_secs[idx % len(sorted_secs)][0]
                    sector_alloc[sec] += 1 if diff > 0 else -1
                    diff += -1 if diff > 0 else 1
                    idx += 1

            # Build list of (pos, sector) tuples to instantiate households
            hh_positions_sector: list[tuple[Coords, str]] = []

            for sector, n_hh in sector_alloc.items():
                if n_hh <= 0:
                    continue
                # Firms in this sector
                sector_firms = [f for f in firm_agents_list if f.sector == sector]
                if not sector_firms:
                    continue  # should not happen

                for _ in range(n_hh):
                    firm = self.random.choice(sector_firms)
                    fx, fy = firm.pos
                    candidates = []
                    for dx in range(-self.work_radius, self.work_radius + 1):
                        for dy in range(-self.work_radius, self.work_radius + 1):
                            coord = (fx + dx, fy + dy)
                            if coord in self.land_coordinates:
                                candidates.append(coord)
                    pos_choice = self.random.choice(candidates) if candidates else firm.pos
                    hh_positions_sector.append((pos_choice, sector))

            # Replace hh_positions with sector-tagged list
            hh_positions = [pos for pos, _ in hh_positions_sector]

            # Finally create households with sector attribute from list
            for (pos, sector) in hh_positions_sector:
                hh = HouseholdAgent(model=self, pos=pos, sector=sector)
                self.grid.place_agent(hh, pos)

        else:
            # No topology – households already have positions in hh_positions
            for pos in hh_positions:
                hh = HouseholdAgent(model=self, pos=pos)
                self.grid.place_agent(hh, pos)

    # --------------------------------------------------------------------- #
    #                               MESA STEP                               #
    # --------------------------------------------------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Advance model by one timestep (representing one year)."""
        self.current_step += 1

        # Each year: sample hazard independently for every cell based on RP
        self._sample_pixelwise_hazard()

        # ---------------- Demand planning phase --------------------- #
        for firm in self._firms:
            firm.plan_operations()

        # Agent actions are phased so labour is hired before production and
        # households consume after firms have produced in the same period.

        # 1. Households – labour supply only
        households = self._households.copy()  # copy to allow shuffle without affecting cache
        self.random.shuffle(households)
        for hh in households:
            hh.supply_labor()

        # 2. Firms – hire labour accumulated in phase 1, purchase inputs,
        #    produce goods, and adjust prices/wages.
        firms = self._firms.copy()
        # Sort by broad supply-chain tier so upstream sectors replenish before downstream buyers.
        firms.sort(key=self._sector_priority)
        for firm in firms:
            firm.step()

        # 3. Households consume the goods produced in the current period.
        households = self._households.copy()
        self.random.shuffle(households)
        for hh in households:
            hh.consume_goods()

        # 4. Close the accounting period after all firm-to-firm and
        #    household transactions have been recorded.
        for firm in firms:
            firm.close_step()

        # ---------------- Record average wage for data collection ----- #
        if self._firms:
            self.mean_wage = float(np.mean([f.wage_offer for f in self._firms]))

        # Collect data BEFORE evolutionary pressure so we capture firms that
        # have already produced (and thus have limiting_factor set). New
        # replacement firms haven't stepped yet and would skew bottleneck counts.
        self.datacollector.collect(self)

        # ---------------- Evolutionary pressure (learning system) ---- #
        self.steps_since_replacement += 1
        if self.firm_learning_enabled and self.steps_since_replacement >= self.replacement_frequency:
            self._apply_evolutionary_pressure()
            self.steps_since_replacement = 0

    # --------------------------------------------------------------------- #
    #                         LEARNING SYSTEM METHODS                        #
    # --------------------------------------------------------------------- #
    def _apply_evolutionary_pressure(self) -> None:
        """Replace failed firms with mutated versions of successful ones."""
        if len(self._firms) < 2:
            return  # need at least 2 firms for evolution

        # Skip early in simulation when firms don't have enough history
        if self.current_step < 5:
            return

        # Identify failed firms (bankrupt only — money below survival threshold).
        # No "persistent decline" check: during systemic crises all firms decline,
        # and replacing solvent-but-declining firms destroys accumulated wealth.
        failed_firms = []
        for firm in self._firms:
            if firm.money < self.min_money_survival:
                failed_firms.append(firm)
        
        if not failed_firms:
            return  # no firms to replace

        # Limit replacements per step to prevent excessive processing
        max_replacements = max(1, len(self._firms) // 4)  # replace at most 25% of firms per step
        failed_firms = failed_firms[:max_replacements]

        # Identify successful firms (any positive production)
        successful_firms = [f for f in self._firms if f not in failed_firms and f.fitness_score > 0]
        if not successful_firms:
            successful_firms = [f for f in self._firms if f not in failed_firms]  # fallback to non-failed
        
        if not successful_firms:
            return  # pathological case
        
        # Replace failed firms with reorganized versions of successful ones.
        # The establishment shell, assets, and money stay inside the model; only
        # strategy and working-capital finance are refreshed.
        for failed_firm in failed_firms:
            # Choose parent based on fitness (weighted selection)
            if len(successful_firms) == 1:
                parent = successful_firms[0]
            else:
                weights = [f.fitness_score + 0.1 for f in successful_firms]  # add small baseline
                parent = self.random.choices(successful_firms, weights=weights, k=1)[0]

            # Reorganization is an ownership/strategy reset, not ex nihilo
            # firm creation. Preserve the failed firm's location, inventories,
            # capital stock, and supplier links.
            failed_firm.strategy = parent.strategy.copy()
            for key in failed_firm.strategy:
                if self.random.random() < 0.5:  # 50% chance to mutate each parameter
                    mutation = self.random.gauss(0, 0.1)  # 10% std deviation
                    failed_firm.strategy[key] *= (1.0 + mutation)
                    failed_firm.strategy[key] = max(0.1, min(3.0, failed_firm.strategy[key]))

            target_expected_sales = max(failed_firm.expected_sales, parent.expected_sales * 0.5, 1.0)
            inventory_target = max(1.0, float(target_expected_sales))
            capital_target = max(1.0, float(target_expected_sales) * failed_firm.capital_coeff)
            avg_input_price = (
                float(np.mean([s.price for s in failed_firm.connected_firms]))
                if failed_firm.connected_firms else 0.0
            )
            unit_variable_cost = (
                failed_firm.wage_offer * failed_firm.LABOR_COEFF
                + avg_input_price * failed_firm.INPUT_COEFF
            )
            working_capital_target = max(10.0, target_expected_sales * unit_variable_cost * 1.25)

            failed_firm.expected_sales = target_expected_sales
            failed_firm.base_inventory_target = inventory_target
            failed_firm.base_capital_target = max(failed_firm.capital_stock, capital_target)
            failed_firm.target_capital_stock = failed_firm.base_capital_target
            failed_firm.performance_history.clear()
            failed_firm.fitness_score = 0.0
            failed_firm.steps_since_adaptation = 0
            failed_firm.survival_time = 0
            failed_firm.sales_last_step = 0.0
            failed_firm.revenue_last_step = 0.0

            equity_needed = max(0.0, working_capital_target - failed_firm.money)
            self.transfer_household_equity_to_firm(failed_firm, equity_needed)

            self._debug_recent_replacements.append(failed_firm.unique_id)
            if len(self._debug_recent_replacements) > 20:
                self._debug_recent_replacements = self._debug_recent_replacements[-20:]

            self.total_firm_replacements += 1
            print(
                f"[EVOLUTION] Step {self.current_step}: Reorganized failed firm "
                f"{failed_firm.unique_id} using parent {parent.unique_id} "
                f"(total: {self.total_firm_replacements})"
            )

    # --------------------------------------------------------------------- #
    #                            EXPORT HELPERS                              #
    # --------------------------------------------------------------------- #
    def results_to_dataframe(self) -> pd.DataFrame:
        """Return model-level DataFrame containing tracked variables plus Date column."""
        df = self.datacollector.get_model_vars_dataframe().copy()
        if self.start_year:
            years = [self.start_year + idx / self.steps_per_year for idx in df.index]
            df.insert(0, "Year", years)
        return df

    def save_results(self, out_path: str | Path = "simulation_results.csv") -> None:
        """Save collected data to CSV."""
        df = self.results_to_dataframe()
        Path(out_path).with_suffix(".csv").write_text(df.to_csv(index=False))

        # Save per-agent time series -------------------------------------- #
        agents_df = self.datacollector.get_agent_vars_dataframe().reset_index()
        # Rename columns for clarity; DataCollector returns 'AgentID' index.
        agents_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")
        if self.start_year and "Step" in agents_df.columns:
            agents_df["Year"] = self.start_year + agents_df["Step"].astype(int) / self.steps_per_year

        # Generate agent filename based on the main filename
        out_path_obj = Path(out_path).with_suffix("")  # Remove extension if present
        agents_path = out_path_obj.with_name(f"{out_path_obj.stem}_agents.csv")
        agents_path.write_text(agents_df.to_csv(index=False))

    # ------------------------------------------------------------------ #
    #                       EVENT APPLICATION LOGIC                      #
    # ------------------------------------------------------------------ #

    def _sample_pixelwise_hazard(self) -> None:
        """Sample hazard only at agent locations using lazy loading.

        Probability cell flooded with intensity from raster *i* is 1/RP_i each
        year, independent across cells and RPs. If multiple RPs trigger on the
        same cell in the same year we keep the maximum depth.

        Uses LazyHazard to sample directly from full-resolution GeoTIFF files
        without loading entire rasters into memory. This reduces memory usage
        from ~4GB to <1MB for global hazard datasets.
        """
        # Collect unique agent cell positions and their geographic coordinates
        agent_cells: dict[Coords, Tuple[float, float]] = {}  # grid coord -> (lon, lat)

        for ag in self._firms:
            x, y = ag.pos
            coord = (x, y)
            if coord not in agent_cells:
                lon = float(self.lon_vals[x])
                lat = float(self.lat_vals[y])
                agent_cells[coord] = (lon, lat)

        for ag in self._households:
            x, y = ag.pos
            coord = (x, y)
            if coord not in agent_cells:
                lon = float(self.lon_vals[x])
                lat = float(self.lat_vals[y])
                agent_cells[coord] = (lon, lat)

        n_agent_cells = len(agent_cells)
        if n_agent_cells == 0:
            return

        # Convert to lists for sampling
        cell_coords = list(agent_cells.keys())  # grid coordinates
        geo_coords = [agent_cells[c] for c in cell_coords]  # (lon, lat) tuples

        # Reset hazard_map for agent cells only
        for coord in cell_coords:
            self.hazard_map[coord] = 0.0

        # Store intensity per agent cell for each hazard type
        # Dict: htype -> dict of coord -> intensity
        hazard_intensities: dict[str, dict[Coords, float]] = {}

        for htype, haz in self.hazards.items():
            # Initialize intensity dict for this hazard type
            cell_intensities: dict[Coords, float] = {c: 0.0 for c in cell_coords}
            ranges = self._hazard_event_ranges.get(htype, [])

            # Find active events for current step
            active_events: list[tuple[int, float]] = []
            for i in range(haz.n_events):
                if i < len(ranges):
                    start, end = ranges[i]
                    if not (start <= self.current_step <= end):
                        continue
                p_annual = haz.frequency[i]
                p_hit = p_annual / self.steps_per_year  # per-step probability
                if p_hit > 0:
                    active_events.append((i, p_hit))

            if active_events:
                for idx_ev, p_hit in active_events:
                    # Use lazy sampling - only reads pixels at agent locations
                    agent_depths = haz.sample_at_coords(geo_coords, idx_ev)

                    # Bernoulli sampling: each agent cell has p_hit chance of flooding
                    hit_mask = np.random.random(n_agent_cells) < p_hit
                    if hit_mask.any():
                        for i, coord in enumerate(cell_coords):
                            if hit_mask[i]:
                                cell_intensities[coord] = max(cell_intensities[coord], agent_depths[i])

            # Update hazard_map with max depth for this hazard type
            for coord, intensity in cell_intensities.items():
                self.hazard_map[coord] = max(self.hazard_map[coord], intensity)

            hazard_intensities[htype] = cell_intensities

        # If impacts disabled, zero-out intensities
        if not self.apply_hazard_impacts:
            for coord in cell_coords:
                self.hazard_map[coord] = 0.0
            hazard_intensities = {htype: {c: 0.0 for c in cell_coords} for htype in hazard_intensities}

        # Count flooded agent cells for logging
        flooded_cells = sum(1 for c in cell_coords if self.hazard_map[c] > 0)

        # Apply agent-specific damage using JRC damage functions
        # Get the global damage functions instance (loaded once, cached)
        damage_funcs = get_damage_functions()

        def apply_damage(ag, is_firm: bool):
            if ag.pos is None:
                agent_info = f"Agent {ag.unique_id} (type: {type(ag).__name__})"
                recent_replacements = getattr(self, '_debug_recent_replacements', [])
                was_replaced = ag.unique_id in recent_replacements
                raise ValueError(f"{agent_info} has None position. "
                               f"Recently replaced: {was_replaced}. "
                               f"Recent replacements: {recent_replacements[-5:]}. "
                               f"This likely indicates a bug in agent creation or placement.")

            coord = ag.pos  # (x, y) tuple
            agent_sector = ag.sector if is_firm else "residential"

            # Get agent's geographic coordinates for region-specific damage curves
            x, y = coord
            lon = float(self.lon_vals[x])
            lat = float(self.lat_vals[y])
            region = get_region_from_coords(lon, lat)

            # Calculate combined loss across all hazard types for this agent
            combined_loss_agent = 0.0
            for htype, intens_dict in hazard_intensities.items():
                intensity = intens_dict.get(coord, 0.0)
                if intensity == 0:
                    continue

                # Get damage fraction from JRC damage functions
                mdr = damage_funcs.get_damage_fraction(intensity, agent_sector, region)

                # Combine multiplicatively: 1 - prod(1 - loss)
                combined_loss_agent = 1 - (1 - combined_loss_agent) * (1 - mdr)

            if combined_loss_agent == 0:
                return

            # Apply damage to firms (households not directly affected by flood damage)
            if is_firm:
                ag.capital_stock *= 1 - combined_loss_agent
                ag.damage_factor *= 1 - combined_loss_agent
                ag.inventory_output = int(ag.inventory_output * (1 - combined_loss_agent))
                for k in list(ag.inventory_inputs.keys()):
                    ag.inventory_inputs[k] = int(ag.inventory_inputs[k] * (1 - combined_loss_agent))

        # Process firms only (households affected indirectly via employment/prices)
        for firm in self._firms:
            apply_damage(firm, is_firm=True)

        print(f"[INFO] Step {self.current_step}: flooded agent cells = {flooded_cells}/{n_agent_cells}")

    # ------------------------------------------------------------------ #
    #                        TRADE NETWORK BUILDER                       #
    # ------------------------------------------------------------------ #
    def _build_trade_network(self) -> None:
        """Randomly connect firms <-> firms and households <-> firms based on distance."""

        # Separate lists for convenience
        firm_agents = [ag for ag in self.agents if isinstance(ag, FirmAgent)]
        household_agents = [ag for ag in self.agents if isinstance(ag, HouseholdAgent)]

        # 1. Firm – firm trade network (directed edges)
        dist_scale = 5.0  # characteristic decay distance
        for f1 in firm_agents:
            for f2 in firm_agents:
                if f1 is f2:
                    continue
                dx = f1.pos[0] - f2.pos[0]
                dy = f1.pos[1] - f2.pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                prob = np.exp(-dist / dist_scale)
                if self.random.random() < prob:
                    f1.connected_firms.append(f2)

        # Ensure **at least one** root firm with no upstream suppliers so the
        # production chain can bootstrap from labour alone. Keep roughly 20 % of
        # firms as roots; all other firms must have ≥1 supplier (choose nearest
        # if none were assigned by the distance‐probability rule).

        if len(firm_agents) > 0:
            # Determine current roots
            roots = [f for f in firm_agents if not f.connected_firms]

            target_root_count = max(1, int(0.2 * len(firm_agents)))

            # If no roots, pick one at random
            if not roots:
                root = self.random.choice(firm_agents)
                root.connected_firms.clear()
                roots.append(root)

            # If too many roots, pick extras to connect to nearest supplier
            if len(roots) > target_root_count:
                surplus = roots[target_root_count:]
                for f in surplus:
                    # Connect to nearest other firm (excluding itself)
                    nearest = min(
                        (g for g in firm_agents if g is not f),
                        key=lambda g: (g.pos[0] - f.pos[0]) ** 2 + (g.pos[1] - f.pos[1]) ** 2,
                    )
                    f.connected_firms.append(nearest)

            # Now ensure every non-root firm has at least one supplier
            for f in firm_agents:
                if f.connected_firms:
                    continue
                if f in roots:
                    continue  # keep as root
                nearest = min(
                    (g for g in firm_agents if g is not f),
                    key=lambda g: (g.pos[0] - f.pos[0]) ** 2 + (g.pos[1] - f.pos[1]) ** 2,
                )
                f.connected_firms.append(nearest)

        # 2. Household – firm employment/consumption links (within radius)
        self._connect_households_to_firms()

    # ------------------------------------------------------------------ #
    #                HOUSEHOLD – FIRM LINK HELPER (reusable)            #
    # ------------------------------------------------------------------ #
    def _connect_households_to_firms(self) -> None:
        """Attach each household to nearby firms within a Manhattan radius of 3."""
        # Use cached lists if available, otherwise fall back to filtering
        if self._firms:
            firm_agents = self._firms
        else:
            firm_agents = [ag for ag in self.agents if isinstance(ag, FirmAgent)]

        if not firm_agents:
            return  # edge-case: no firms

        if self._households:
            household_agents = self._households
        else:
            household_agents = [ag for ag in self.agents if isinstance(ag, HouseholdAgent)]

        for hh in household_agents:
            # Avoid duplicating if links already present
            if hh.nearby_firms:
                continue

            # Use sector cache if available
            sector_firms = self._firms_by_sector.get(hh.sector, []) if self._firms_by_sector else None
            if sector_firms is None:
                sector_firms = [f for f in firm_agents if f.sector == hh.sector]

            for firm in sector_firms:
                dx = abs(hh.pos[0] - firm.pos[0])
                dy = abs(hh.pos[1] - firm.pos[1])
                if dx + dy <= self.work_radius:
                    hh.nearby_firms.append(firm)

            # Guarantee at least one connection (pick nearest firm *in same sector*) ----
            if not hh.nearby_firms:
                same_sector_firms = sector_firms if sector_firms else firm_agents
                nearest = min(
                    same_sector_firms,
                    key=lambda f: (f.pos[0] - hh.pos[0]) ** 2 + (f.pos[1] - hh.pos[1]) ** 2,
                )
                hh.nearby_firms.append(nearest) 
