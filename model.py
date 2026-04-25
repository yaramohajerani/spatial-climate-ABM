from __future__ import annotations

from collections import defaultdict
import math
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

try:  # pragma: no cover - package import path
    from .agents import FirmAgent, HouseholdAgent
    from .hazard_utils import lazy_hazard_from_geotiffs, LazyHazard, SyntheticHazard
    from .damage_functions import get_damage_functions, get_region_from_coords
    from .shock_inputs import (
        HazardRasterEvent,
        LaneShock,
        NodeShock,
        RouteShock,
        normalize_lane_shocks,
        normalize_node_shocks,
        normalize_raster_hazard_events,
        normalize_route_shocks,
    )
    from .transport_runtime import (
        dedupe_pairs,
        inbound_route_exposure_ratio,
        make_transport_patch,
        route_exposure_ratio,
    )
except ImportError:  # pragma: no cover - flat script import path
    from agents import FirmAgent, HouseholdAgent
    from hazard_utils import lazy_hazard_from_geotiffs, LazyHazard, SyntheticHazard
    from damage_functions import get_damage_functions, get_region_from_coords
    from shock_inputs import (
        HazardRasterEvent,
        LaneShock,
        NodeShock,
        RouteShock,
        normalize_lane_shocks,
        normalize_node_shocks,
        normalize_raster_hazard_events,
        normalize_route_shocks,
    )
    from transport_runtime import (
        dedupe_pairs,
        inbound_route_exposure_ratio,
        make_transport_patch,
        route_exposure_ratio,
    )

Coords = Tuple[int, int]
TransportPairs = List[Tuple[FirmAgent, FirmAgent]]
ActiveTransportBlock = Tuple[float, TransportPairs]


def _lon_between(src_lon: float, dst_lon: float, wp_lon: float) -> bool:
    span = (dst_lon - src_lon + 540) % 360 - 180
    wp_offset = (wp_lon - src_lon + 540) % 360 - 180
    return (0 <= wp_offset <= span) if span >= 0 else (span <= wp_offset <= 0)


class EconomyModel(Model):
    """Spatial ABM of an economy subject to climate risk."""

    FINAL_CONSUMPTION_SECTORS = {"retail", "wholesale", "services"}
    _land_coordinate_cache: Dict[Tuple[int, int, float], List[Coords]] = {}
    SECTOR_ORDER = {
        "commodity": 0,
        "agriculture": 0,
        "components": 1,
        "manufacturing": 2,
        "retail": 3,
        "wholesale": 3,
        "services": 3,
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
        hazard_events: Iterable[Tuple[int, int, int, str, str | None]] | None = None,
        raster_hazard_events: Iterable[HazardRasterEvent | Tuple[int, int, int, str, str | None] | dict[str, Any]] | None = None,
        node_shocks: Iterable[NodeShock | dict[str, Any]] | None = None,
        lane_shocks: Iterable[LaneShock | dict[str, Any]] | None = None,
        route_shocks: Iterable[RouteShock | dict[str, Any]] | None = None,
        seed: int | None = None,
        start_year: int = 0,
        steps_per_year: int = 4,
        firm_topology_path: str | None = None,
        apply_hazard_impacts: bool = True,
        adaptation_params: dict | None = None,
        consumption_ratios: dict | None = None,
        input_recipe_ranges: dict | None = None,
        firm_replacement: str = "startup_reset",
        dynamic_supplier_search: bool = True,
        max_dynamic_suppliers_per_sector: int = 2,
        grid_resolution: float = 1.0,
        household_relocation: bool = False,
        damage_functions_path: str | None = None,
        land_boundaries_path: str | None = None,
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

        # Adaptation system parameters.
        self.adaptation_config = adaptation_params or {}
        self.firm_adaptation_enabled: bool = self.adaptation_config.get("enabled", True)
        self.min_money_survival: float = self.adaptation_config.get("min_money_survival", 1.0)
        self.replacement_frequency: int = self.adaptation_config.get("replacement_frequency", 10)
        self.firm_replacement: str = str(firm_replacement or "startup_reset")
        if self.firm_replacement not in {"startup_reset", "none"}:
            raise ValueError("firm_replacement must be 'startup_reset' or 'none'")
        self.dynamic_supplier_search_enabled: bool = bool(dynamic_supplier_search)
        self.max_dynamic_suppliers_per_sector: int = max(0, int(max_dynamic_suppliers_per_sector))
        self.total_firm_exits: int = 0
        self.total_dynamic_supplier_edges: int = 0
        self._dynamic_supplier_pairs: set[tuple[int, int]] = set()
        self.adaptation_observation_radius: int = int(self.adaptation_config.get("observation_radius", 4))
        self.adaptation_updates_this_step: int = 0
        self.max_backup_suppliers: int = int(self.adaptation_config.get("max_backup_suppliers", 5))
        self.reserved_capacity_share: float = float(self.adaptation_config.get("reserved_capacity_share", 0.35))
        self.reserved_capacity_markup_cap: float = float(
            self.adaptation_config.get("reserved_capacity_markup_cap", 0.10)
        )
        self.adaptation_strategy: str = str(self.adaptation_config.get("adaptation_strategy", "backup_suppliers"))
        self._reserved_capacity_contracts: Dict[int, List[dict[str, object]]] = {}
        self._supplier_reserved_inventory: Dict[int, float] = {}

        # Household consumption ratios across final-good sectors.
        # Upstream sectors sell to firms, not directly to households.
        self.consumption_ratios: dict = consumption_ratios or {
            'retail': 1.0,
        }
        self.input_recipe_ranges: dict = copy.deepcopy(
            input_recipe_ranges
            if input_recipe_ranges is not None
            else FirmAgent.DEFAULT_INPUT_RECIPE_RANGES
        )
        self.final_consumption_sectors = set(self.FINAL_CONSUMPTION_SECTORS)
        self._consumption_ratio_warning_emitted: bool = False
        self._startup_capital_floor_overrides: list[dict[str, float | int]] = []

        # -------------------- Performance optimization caches -------------------- #
        # Cached agent lists by type (updated when agents added/removed)
        self._households: List[HouseholdAgent] = []
        self._firms: List[FirmAgent] = []
        # Cached agents by sector
        self._firms_by_sector: Dict[str, List[FirmAgent]] = defaultdict(list)
        self._firms_by_id: Dict[int, FirmAgent] = {}

        # Calendar mapping -------------------------------------------------- #
        self.start_year: int = start_year
        # Calendar granularity (e.g. 4 → quarterly, 12 → monthly)
        self.steps_per_year: int = steps_per_year
        self.damage_functions_path = (
            str(Path(damage_functions_path).expanduser())
            if damage_functions_path is not None
            else None
        )
        self.land_boundaries_path = (
            str(Path(land_boundaries_path).expanduser())
            if land_boundaries_path is not None
            else None
        )

        # --- Spatial environment & custom topology --- #
        self._firm_topology: dict | None = None
        if firm_topology_path is not None:
            topo_path = Path(firm_topology_path)
            if not topo_path.exists():
                raise FileNotFoundError(f"Firm topology JSON not found: {topo_path}")
            self._firm_topology = json.loads(topo_path.read_text())

            # Override num_firms to match the topology file so downstream
            # components (e.g. dashboards, logging) see the correct value.
            num_firms = len(self._firm_topology.get("firms", []))
        self._raw_topology: dict = self._firm_topology or {}

        self._raster_hazard_events = normalize_raster_hazard_events(
            raster_hazard_events,
            legacy_hazard_events=hazard_events,
        )
        self._node_shocks = normalize_node_shocks(node_shocks)
        self._lane_shocks = normalize_lane_shocks(lane_shocks)
        self._route_shocks = normalize_route_shocks(route_shocks)

        # Group by hazard type while preserving order to keep mapping consistent
        grouped_files: dict[str, list[Tuple[int, str]]] = defaultdict(list)
        grouped_ranges: dict[str, list[Tuple[int, int]]] = defaultdict(list)

        for event in self._raster_hazard_events:
            if event.path is None:
                continue
            grouped_files[event.hazard_type].append((event.return_period, event.path))
            grouped_ranges[event.hazard_type].append((event.start_step, event.end_step))

        # Store mapping of event index -> (start, end) per hazard type
        self._hazard_event_ranges: dict[str, List[Tuple[int, int]]] = dict(grouped_ranges)

        # Use lazy hazard loading for memory efficiency (samples on-demand)
        # This reduces memory from ~4GB to <1MB for global hazard datasets
        self.hazards: dict[str, LazyHazard] = {}

        for htype, grp in grouped_files.items():
            haz, _, _ = lazy_hazard_from_geotiffs(grp, haz_type=htype)
            # Store lazy hazard that samples on-demand
            self.hazards[htype] = haz

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
                "Firm_Profits": lambda m: sum(f.net_profit_this_step for f in m._firms),
                "Firm_Operating_Profits": lambda m: sum(f.operating_surplus_this_step for f in m._firms),
                "Firm_Direct_Loss_Expenses": lambda m: sum(f.direct_loss_expense_this_step for f in m._firms),
                "Firm_Dividends_Paid": lambda m: sum(f.dividends_paid_this_step for f in m._firms),
                "Firm_Investment_Spending": lambda m: sum(f.investment_spending_this_step for f in m._firms),
                "Firm_Working_Capital_Credit_Used": lambda m: sum(f.working_capital_credit_used_this_step for f in m._firms),
                "Firm_Inventory": lambda m: sum(f.inventory_output for f in m._firms),
                "Household_Wealth": lambda m: sum(h.money for h in m._households),
                "Household_Labor_Sold": lambda m: sum(h.labor_sold for h in m._households),
                "Household_Consumption": lambda m: sum(h.consumption for h in m._households),
                "Household_Labor_Income": lambda m: sum(h.labor_income_this_step for h in m._households),
                "Household_Dividend_Income": lambda m: sum(h.dividend_income_received_this_step for h in m._households),
                "Household_Capital_Income": lambda m: sum(h.capital_income_received_this_step for h in m._households),
                "Household_Adaptation_Income": lambda m: sum(h.adaptation_income_received_this_step for h in m._households),
                "Average_Risk": lambda m: np.mean(list(m.hazard_map.values())),
                "Mean_Wage": lambda m: m.mean_wage,
                "Mean_Price": lambda m: (
                    float(np.dot([f.price for f in m._firms], [f.sales_last_step for f in m._firms]) / max(sum(f.sales_last_step for f in m._firms), 1e-9))
                    if m._firms else 0.0
                ),
                "Total_Money": lambda m: m.total_money(),
                "Money_Drift": lambda m: m.total_money() - getattr(m, "initial_total_money", 0.0),
                "Labor_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "labor"),
                "Capital_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "capital"),
                "Input_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "input"),
                "Demand_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "demand"),
                "Finance_Limited_Firms": lambda m: sum(1 for f in m._firms if getattr(f, "limiting_factor", "") == "finance"),
                "Firm_Adaptation_Spending": lambda m: sum(f.adaptation_spending_this_step for f in m._firms),
                "Average_Continuity_Capital": lambda m: np.mean([f.continuity_capital for f in m._firms]) if m._firms else 0.0,
                "Average_Resilience_Capital": lambda m: np.mean([f.resilience_capital for f in m._firms]) if m._firms else 0.0,
                "Average_Expected_Direct_Loss": lambda m: np.mean([f.expected_direct_loss_ewma for f in m._firms]) if m._firms else 0.0,
                "Average_Realized_Direct_Loss": lambda m: np.mean([f.realized_direct_loss_ewma for f in m._firms]) if m._firms else 0.0,
                "Average_Local_Observed_Loss": lambda m: np.mean([f.local_observed_loss_ewma for f in m._firms]) if m._firms else 0.0,
                "Average_Supplier_Disruption": lambda m: np.mean([f.supplier_disruption_ewma for f in m._firms]) if m._firms else 0.0,
                "Average_Expected_Operating_Shortfall": lambda m: np.mean([f.expected_operating_shortfall_ewma for f in m._firms]) if m._firms else 0.0,
                "Average_Local_Observed_Shortfall": lambda m: np.mean([f.local_observed_shortfall_ewma for f in m._firms]) if m._firms else 0.0,
                "Average_Continuity_Target": lambda m: np.mean([f.last_continuity_target for f in m._firms]) if m._firms else 0.0,
                "Average_Perceived_Continuity_Risk": lambda m: np.mean([f.last_perceived_continuity_risk for f in m._firms]) if m._firms else 0.0,
                "Average_Continuity_Gap_Coverage": lambda m: np.mean([f.continuity_gap_coverage_this_step for f in m._firms]) if m._firms else 0.0,
                "Average_Continuity_Input_Coverage": lambda m: np.mean([f.continuity_input_coverage_this_step for f in m._firms]) if m._firms else 0.0,
                "Average_Raw_Supplier_Disruption": lambda m: np.mean([f.raw_supplier_disruption_this_step for f in m._firms]) if m._firms else 0.0,
                "Average_Backup_Purchases": lambda m: np.mean([f.backup_purchases_this_step for f in m._firms]) if m._firms else 0.0,
                "Total_Backup_Purchases": lambda m: sum(f.backup_purchases_this_step for f in m._firms),
                "Average_Reserved_Capacity_Purchases": lambda m: np.mean([f.reserved_capacity_purchases_this_step for f in m._firms]) if m._firms else 0.0,
                "Total_Reserved_Capacity_Purchases": lambda m: sum(f.reserved_capacity_purchases_this_step for f in m._firms),
                "Average_Reserved_Capacity_Price_Savings": lambda m: np.mean([f.reserved_capacity_price_savings_this_step for f in m._firms]) if m._firms else 0.0,
                "Total_Reserved_Capacity_Price_Savings": lambda m: sum(f.reserved_capacity_price_savings_this_step for f in m._firms),
                "Reserved_Capacity_Contracts": lambda m: m._count_reserved_capacity_contracts(),
                "Adaptation_Strategy": lambda m: getattr(m, "adaptation_strategy", "backup_suppliers"),
                "Average_Adaptation_Target": lambda m: np.mean([f.last_adaptation_target for f in m._firms]) if m._firms else 0.0,
                "Average_Perceived_Hazard_Risk": lambda m: np.mean([f.last_perceived_hazard_risk for f in m._firms]) if m._firms else 0.0,
                "Average_Working_Capital_Credit_Used": lambda m: np.mean([f.working_capital_credit_used_this_step for f in m._firms]) if m._firms else 0.0,
                "Average_Working_Capital_Credit_Limit": lambda m: np.mean([f.working_capital_credit_limit for f in m._firms]) if m._firms else 0.0,
                "Adaptation_Updates": lambda m: m.adaptation_updates_this_step,
                "Fixed_Labor_Share": lambda m: np.mean([getattr(f, "LABOR_SHARE", np.nan) for f in m._firms]) if m._firms else 0.0,
                "Firm_Replacements": lambda m: getattr(m, 'total_firm_replacements', 0),
                "Firm_Exits": lambda m: getattr(m, 'total_firm_exits', 0),
                "Active_Firms": lambda m: sum(1 for f in m._firms if f.active),
                "Dynamic_Supplier_Edges": lambda m: getattr(m, 'total_dynamic_supplier_edges', 0),
                "Total_Sales": lambda m: sum(f.sales_last_step for f in m._firms),
                "Total_Firms": lambda m: len(m._firms),
                "Flooded_Firms": lambda m: sum(1 for f in m._firms if m.hazard_map.get(f.pos, 0) > 0),
                "Flooded_Households": lambda m: sum(1 for h in m._households if m.hazard_map.get(h.pos, 0) > 0),
                "Ever_Directly_Hit_Firms": lambda m: m._count_ever_directly_hit_firms(),
                "Ever_Directly_Hit_Firm_Share": lambda m: m._share_ever_directly_hit_firms(),
                "Never_Hit_Firms": lambda m: m._count_never_hit_firms(),
                "Never_Hit_Currently_Disrupted_Firms": lambda m: m._count_never_hit_currently_disrupted_firms(),
                "Never_Hit_Currently_Disrupted_Firm_Share": lambda m: m._share_never_hit_currently_disrupted_firms(),
                "Never_Hit_Supplier_Disruption_Burden_Share": lambda m: m._share_supplier_disruption_borne_by_never_hit_firms(),
                "Never_Hit_Production_Share": lambda m: m._share_production_from_never_hit_firms(),
                "Never_Hit_Capital_Share": lambda m: m._share_capital_held_by_never_hit_firms(),
                "Ever_Indirectly_Disrupted_Before_Direct_Hit_Firms": lambda m: m._count_ever_indirectly_disrupted_before_direct_hit_firms(),
                "Ever_Indirectly_Disrupted_Before_Direct_Hit_Firm_Share": lambda m: m._share_ever_indirectly_disrupted_before_direct_hit_firms(),
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
                "active": lambda a: a.active if isinstance(a, FirmAgent) else np.nan,
                "survival_time": lambda a: getattr(a, "survival_time", 0),
                "sales_last_step": lambda a: getattr(a, "sales_last_step", np.nan),
                "household_sales_last_step": lambda a: getattr(a, "household_sales_last_step", np.nan),
                "labor_income": lambda a: getattr(a, "labor_income_this_step", np.nan),
                "dividend_income": lambda a: getattr(a, "dividend_income_received_this_step", np.nan),
                "capital_income": lambda a: getattr(a, "capital_income_received_this_step", np.nan),
                "adaptation_income": lambda a: getattr(a, "adaptation_income_received_this_step", np.nan),
                "profit": lambda a: getattr(a, "net_profit_this_step", np.nan),
                "operating_profit": lambda a: getattr(a, "operating_surplus_this_step", np.nan),
                "direct_loss_expense": lambda a: getattr(a, "direct_loss_expense_this_step", np.nan),
                "dividends_paid": lambda a: getattr(a, "dividends_paid_this_step", np.nan),
                "investment_spending": lambda a: getattr(a, "investment_spending_this_step", np.nan),
                "adaptation_spending": lambda a: getattr(a, "adaptation_spending_this_step", np.nan),
                "working_capital_credit_used": lambda a: getattr(a, "working_capital_credit_used_this_step", np.nan),
                "working_capital_credit_limit": lambda a: getattr(a, "working_capital_credit_limit", np.nan),
                "continuity_capital": lambda a: getattr(a, "continuity_capital", np.nan),
                "resilience_capital": lambda a: getattr(a, "resilience_capital", np.nan),
                "expected_direct_loss": lambda a: getattr(a, "expected_direct_loss_ewma", np.nan),
                "realized_direct_loss": lambda a: getattr(a, "realized_direct_loss_ewma", np.nan),
                "local_observed_loss": lambda a: getattr(a, "local_observed_loss_ewma", np.nan),
                "supplier_disruption": lambda a: getattr(a, "supplier_disruption_ewma", np.nan),
                "raw_supplier_disruption": lambda a: getattr(a, "raw_supplier_disruption_this_step", np.nan),
                "expected_operating_shortfall": lambda a: getattr(a, "expected_operating_shortfall_ewma", np.nan),
                "local_observed_shortfall": lambda a: getattr(a, "local_observed_shortfall_ewma", np.nan),
                "continuity_target": lambda a: getattr(a, "last_continuity_target", np.nan),
                "perceived_continuity_risk": lambda a: getattr(a, "last_perceived_continuity_risk", np.nan),
                "adaptation_target": lambda a: getattr(a, "last_adaptation_target", np.nan),
                "perceived_hazard_risk": lambda a: getattr(a, "last_perceived_hazard_risk", np.nan),
                "adaptation_action": lambda a: getattr(a, "last_adaptation_action", ""),
                "continuity_gap_coverage": lambda a: getattr(a, "continuity_gap_coverage_this_step", np.nan),
                "continuity_input_coverage": lambda a: getattr(a, "continuity_input_coverage_this_step", np.nan),
                "backup_purchases": lambda a: getattr(a, "backup_purchases_this_step", np.nan),
                "reserved_capacity_purchases": lambda a: getattr(a, "reserved_capacity_purchases_this_step", np.nan),
                "reserved_capacity_price_savings": lambda a: getattr(a, "reserved_capacity_price_savings_this_step", np.nan),
                "counterfactual_direct_loss": lambda a: getattr(a, "counterfactual_direct_loss_this_step", np.nan),
                "realized_direct_loss_value": lambda a: getattr(a, "realized_direct_loss_this_step", np.nan),
                "adaptation_updates": lambda a: getattr(a, "adaptation_update_count", np.nan),
                "labor_share": lambda a: getattr(a, "LABOR_SHARE", np.nan),
                "ever_directly_hit": lambda a: getattr(a, "ever_directly_hit", np.nan),
                "ever_indirectly_disrupted_before_direct_hit": lambda a: getattr(a, "ever_indirectly_disrupted_before_direct_hit", np.nan),
            },
        )
        self._install_transport_reporters()

        # ---------------- Land mask to avoid placing agents in the ocean ---------------- #
        self.land_coordinates: List[Coords] = self._load_or_compute_land_coordinates()
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
        self._assign_firm_input_recipes()
        self._warn_missing_recipe_supplier_coverage()
        self._topology_id_to_agent: Dict[int, FirmAgent] = {
            int(getattr(agent, "topology_id", agent.unique_id)): agent for agent in self._firms
        }
        self._primary_supplier_pairs: set[tuple[int, int]] = {
            (
                int(getattr(supplier, "topology_id", supplier.unique_id)),
                int(getattr(buyer, "topology_id", buyer.unique_id)),
            )
            for buyer in self._firms
            for supplier in getattr(buyer, "connected_firms", [])
            if supplier is not None
        }
        self._initialize_transport_route_metrics()
        self._register_node_shocks()
        self._precomputed_route_transport_edges: List[Tuple[RouteShock, List[Tuple[FirmAgent, FirmAgent]]]] = []
        self._precomputed_lane_transport_edges: List[Tuple[LaneShock, List[Tuple[FirmAgent, FirmAgent]]]] = []
        self._build_route_transport_maps()
        self._build_lane_transport_maps()
        self._precomputed_transport_edges = self._precomputed_route_transport_edges
        self._precomputed_link_edges = self._precomputed_lane_transport_edges

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
        self._firms_by_id.clear()

        for ag in self.agents:
            if isinstance(ag, HouseholdAgent):
                self._households.append(ag)
            elif isinstance(ag, FirmAgent):
                self._firms.append(ag)
                self._firms_by_sector[ag.sector].append(ag)
                self._firms_by_id[int(ag.unique_id)] = ag

    def _register_agent(self, agent) -> None:
        """Register a single agent in the caches."""
        if isinstance(agent, HouseholdAgent):
            self._households.append(agent)
        elif isinstance(agent, FirmAgent):
            self._firms.append(agent)
            self._firms_by_sector[agent.sector].append(agent)
            self._firms_by_id[int(agent.unique_id)] = agent

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
            self._firms_by_id.pop(int(agent.unique_id), None)

    def get_final_consumption_ratios(self) -> Dict[str, float]:
        """Return household demand shares over final-good sectors only."""
        dropped = sorted(
            sector
            for sector, weight in self.consumption_ratios.items()
            if sector not in self.final_consumption_sectors and float(weight) > 0
        )
        if dropped and not self._consumption_ratio_warning_emitted:
            warnings.warn(
                "Ignoring non-final household consumption sectors "
                f"{dropped}; only {sorted(self.final_consumption_sectors)} are eligible.",
                RuntimeWarning,
            )
            self._consumption_ratio_warning_emitted = True

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

    def effective_configuration_metadata(self) -> Dict[str, object]:
        """Return model-derived configuration details for run metadata."""
        import json

        startup_added_capital = sum(
            float(item["seeded_capital"]) - float(item["configured_capital"])
            for item in self._startup_capital_floor_overrides
        )
        return {
            "EffectiveConsumptionRatios": json.dumps(
                self.get_final_consumption_ratios(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "InputRecipeRanges": json.dumps(
                self.input_recipe_ranges,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "FirmReplacement": self.firm_replacement,
            "DynamicSupplierSearch": bool(self.dynamic_supplier_search_enabled),
            "MaxDynamicSuppliersPerSector": int(self.max_dynamic_suppliers_per_sector),
            "StartupCapitalFloorCount": int(len(self._startup_capital_floor_overrides)),
            "StartupCapitalFloorTotal": float(startup_added_capital),
            "StartupCapitalFloorFirms": json.dumps(
                self._startup_capital_floor_overrides,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ),
            "DamageFunctionsPath": self.damage_functions_path or "",
            "LandBoundariesPath": self.land_boundaries_path or "",
            "RasterHazardEventCount": int(len(self._raster_hazard_events)),
            "NodeShockCount": int(len(self._node_shocks)),
            "LaneShockCount": int(len(self._lane_shocks)),
            "RouteShockCount": int(len(self._route_shocks)),
        }

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
                household.dividend_income_received_this_step += per_household
            elif income_kind == "capital":
                household.capital_income_this_step += per_household
                household.capital_income_received_this_step += per_household
            elif income_kind == "adaptation":
                household.adaptation_income_this_step += per_household
                household.adaptation_income_received_this_step += per_household
            else:
                raise ValueError(f"Unknown household income kind: {income_kind}")

    def transfer_household_equity_to_firm(self, firm: FirmAgent, amount: float) -> float:
        """Recapitalize a firm by drawing cash proportionally from households.

        This models households providing equity finance to a startup-reset firm
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

    def get_local_observed_loss_fraction(self, focal_firm: FirmAgent) -> float:
        """Return the weighted mean nearby raw-loss signal observed by ``focal_firm``."""

        radius = max(0, int(self.adaptation_observation_radius))
        if radius <= 0:
            return 0.0

        weighted_loss = 0.0
        total_weight = 0.0
        x0, y0 = focal_firm.pos
        for other in self._firms:
            if other is focal_firm:
                continue
            dist = abs(x0 - other.pos[0]) + abs(y0 - other.pos[1])
            if dist > radius:
                continue
            raw_loss = getattr(other, "raw_direct_loss_fraction_this_step", 0.0)
            if raw_loss <= 0:
                continue
            weight = 1.0 / (1.0 + dist)
            weighted_loss += weight * raw_loss
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return weighted_loss / total_weight

    def get_local_observed_shortfall_fraction(self, focal_firm: FirmAgent) -> float:
        """Return the weighted mean nearby hazard-induced operating shortfall."""

        radius = max(0, int(self.adaptation_observation_radius))
        if radius <= 0:
            return 0.0

        weighted_shortfall = 0.0
        total_weight = 0.0
        x0, y0 = focal_firm.pos
        for other in self._firms:
            if other is focal_firm:
                continue
            dist = abs(x0 - other.pos[0]) + abs(y0 - other.pos[1])
            if dist > radius:
                continue
            shortfall = getattr(other, "hazard_operating_shortfall_this_step", 0.0)
            if shortfall <= 0:
                continue
            weight = 1.0 / (1.0 + dist)
            weighted_shortfall += weight * shortfall
            total_weight += weight

        if total_weight <= 0:
            return 0.0
        return weighted_shortfall / total_weight

    def find_backup_suppliers(
        self,
        buyer: FirmAgent,
        max_count: int,
        sector: str | None = None,
    ) -> List[FirmAgent]:
        """Return non-primary firms in the relevant supplier sector(s) with inventory.

        Backup suppliers are firms in the same sector(s) as the buyer's primary
        suppliers, or in one explicitly requested supplier sector, excluding the
        buyer itself and its existing ``connected_firms``. Results are shuffled
        for fairness then sorted by price (cheapest first).
        """
        if max_count <= 0:
            return []

        if sector is not None:
            supplier_sectors: set[str] = {sector}
        else:
            supplier_sectors = {s.sector for s in buyer.connected_firms}
        if not supplier_sectors:
            return []

        primary_ids = {s.unique_id for s in buyer.connected_firms}
        candidates: List[FirmAgent] = []
        for sector in supplier_sectors:
            for firm in self._firms_by_sector.get(sector, []):
                if firm is buyer or firm.unique_id in primary_ids:
                    continue
                if not firm.active:
                    continue
                if firm.inventory_output > 0:
                    candidates.append(firm)

        if not candidates:
            return []

        self.random.shuffle(candidates)
        candidates.sort(key=lambda f: f.price)
        return candidates[:max_count]

    def add_dynamic_supplier_edges(self, buyer: FirmAgent, sector: str) -> List[FirmAgent]:
        """Add bounded new supplier links in a required recipe sector."""
        if not self.dynamic_supplier_search_enabled:
            return []
        max_count = self.max_dynamic_suppliers_per_sector
        if max_count <= 0 or not buyer.active:
            return []

        dynamic_sector_suppliers = [
            supplier for supplier in buyer.connected_firms
            if supplier.sector == sector
            and (supplier.unique_id, buyer.unique_id) in self._dynamic_supplier_pairs
        ]
        if len(dynamic_sector_suppliers) >= max_count:
            return []

        existing_ids = {supplier.unique_id for supplier in buyer.connected_firms}
        candidates: List[FirmAgent] = []
        for supplier in self._firms_by_sector.get(sector, []):
            if supplier is buyer or supplier.unique_id in existing_ids:
                continue
            if not supplier.active:
                continue
            if supplier.inventory_output <= 1e-9 and supplier.production <= 1e-9:
                continue
            candidates.append(supplier)

        if not candidates:
            return []

        bx, by = buyer.pos
        self.random.shuffle(candidates)
        candidates.sort(
            key=lambda supplier: (
                supplier.price,
                abs(supplier.pos[0] - bx) + abs(supplier.pos[1] - by),
            )
        )
        slots = max_count - len(dynamic_sector_suppliers)
        new_suppliers: List[FirmAgent] = []
        for supplier in candidates[:slots]:
            buyer.connected_firms.append(supplier)
            pair = (supplier.unique_id, buyer.unique_id)
            if pair not in self._dynamic_supplier_pairs:
                self._dynamic_supplier_pairs.add(pair)
                self.total_dynamic_supplier_edges += 1
            new_suppliers.append(supplier)
        return new_suppliers

    def _reset_reserved_capacity_contracts(self) -> None:
        self._reserved_capacity_contracts = {}
        self._supplier_reserved_inventory = {}

    def _prepare_reserved_capacity_contracts(self) -> None:
        """Reserve a bounded backup slice for the reserved-capacity strategy.

        Contracts reserve against a supplier's currently available inventory.
        Reserved capacity is modeled as priority access to a bounded slice of
        on-hand stock, not as a hard commitment of future production.
        """
        self._reset_reserved_capacity_contracts()
        if not self.firm_adaptation_enabled or self.adaptation_strategy != "reserved_capacity":
            return

        buyers = [
            firm
            for firm in self._firms
            if firm.adaptation_enabled
            and firm.active
            and getattr(firm, "continuity_capital", 0.0) > 0
            and getattr(firm, "target_input_units", 0.0) > 0
            and firm.connected_firms
        ]
        if not buyers:
            return

        self.random.shuffle(buyers)
        reservable_inventory: Dict[int, float] = {}
        for buyer in buyers:
            contract_price_cap = buyer._reserved_capacity_price_cap()
            for sector, remaining_reserved_units in buyer._reserved_capacity_target_units_by_sector().items():
                if remaining_reserved_units <= 1e-9:
                    continue

                backup_suppliers = self.find_backup_suppliers(
                    buyer,
                    buyer._reserved_capacity_supplier_count(),
                    sector=sector,
                )
                for supplier in backup_suppliers:
                    if remaining_reserved_units <= 1e-9:
                        break
                    expected_available_supply = max(0.0, supplier.inventory_output)
                    supplier_limit = reservable_inventory.setdefault(
                        supplier.unique_id,
                        max(0.0, expected_available_supply * self.reserved_capacity_share),
                    )
                    reserved_units = min(supplier_limit, remaining_reserved_units)
                    if reserved_units <= 1e-9:
                        continue

                    contract_unit_price = min(max(supplier.price, 0.5), contract_price_cap)
                    self._reserved_capacity_contracts.setdefault(buyer.unique_id, []).append(
                        {
                            "supplier": supplier,
                            "quantity": reserved_units,
                            "unit_price": contract_unit_price,
                        }
                    )
                    self._supplier_reserved_inventory[supplier.unique_id] = (
                        self._supplier_reserved_inventory.get(supplier.unique_id, 0.0) + reserved_units
                    )
                    reservable_inventory[supplier.unique_id] -= reserved_units
                    remaining_reserved_units -= reserved_units

    def get_reserved_capacity_contracts(
        self,
        buyer: FirmAgent,
        sector: str | None = None,
    ) -> List[tuple[FirmAgent, float, float]]:
        contracts = []
        for contract in self._reserved_capacity_contracts.get(buyer.unique_id, []):
            quantity = float(contract.get("quantity", 0.0))
            if quantity <= 1e-9:
                continue
            supplier = contract["supplier"]
            if sector is not None and supplier.sector != sector:
                continue
            contracts.append(
                (
                    supplier,
                    quantity,
                    float(contract.get("unit_price", 0.0)),
                )
            )
        return contracts

    def _reserved_inventory_for_buyer(self, supplier: FirmAgent, buyer_id: int) -> float:
        for contract in self._reserved_capacity_contracts.get(buyer_id, []):
            if contract.get("supplier") is supplier:
                return float(contract.get("quantity", 0.0))
        return 0.0

    def available_inventory_for_spot_sales(self, supplier: FirmAgent) -> float:
        if not supplier.active:
            return 0.0
        reserved_total = self._supplier_reserved_inventory.get(supplier.unique_id, 0.0)
        return max(0.0, supplier.inventory_output - reserved_total)

    def available_reserved_inventory_for_buyer(self, supplier: FirmAgent, buyer_id: int) -> float:
        reserved_for_buyer = self._reserved_inventory_for_buyer(supplier, buyer_id)
        if reserved_for_buyer <= 0:
            return 0.0
        reserved_for_others = max(
            0.0,
            self._supplier_reserved_inventory.get(supplier.unique_id, 0.0) - reserved_for_buyer,
        )
        return max(
            0.0,
            min(reserved_for_buyer, supplier.inventory_output - reserved_for_others),
        )

    def consume_reserved_capacity(self, supplier: FirmAgent, buyer_id: int, quantity: float) -> None:
        if quantity <= 0:
            return
        for contract in self._reserved_capacity_contracts.get(buyer_id, []):
            if contract.get("supplier") is not supplier:
                continue
            used_quantity = min(float(contract.get("quantity", 0.0)), quantity)
            if used_quantity <= 0:
                return
            contract["quantity"] = max(0.0, float(contract.get("quantity", 0.0)) - used_quantity)
            self._supplier_reserved_inventory[supplier.unique_id] = max(
                0.0,
                self._supplier_reserved_inventory.get(supplier.unique_id, 0.0) - used_quantity,
            )
            return

    def _register_node_shocks(self) -> None:
        """Resolve node shocks to coordinates and register them as synthetic hazards."""
        if not self._node_shocks:
            return

        for shock in self._node_shocks:
            affected_coords = list(shock.affected_coords)
            if shock.firm_ids:
                for firm_id in shock.firm_ids:
                    firm = self._topology_id_to_agent.get(int(firm_id))
                    if firm is None or firm.pos is None:
                        warnings.warn(
                            f"NodeShock '{shock.label}' references unknown firm id {firm_id} — skipping.",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    x, y = firm.pos
                    affected_coords.append((float(self.lon_vals[x]), float(self.lat_vals[y])))

            if not affected_coords:
                warnings.warn(
                    f"NodeShock '{shock.label}' resolved no coordinates — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            haz = SyntheticHazard(
                affected_coords=affected_coords,
                intensity=shock.intensity,
                haz_type=shock.hazard_type,
                radius_deg=shock.radius_deg,
                return_period=shock.return_period,
            )
            haz_key = shock.hazard_type
            suffix = 0
            while haz_key in self.hazards:
                suffix += 1
                haz_key = f"{shock.hazard_type}_{suffix}"
            self.hazards[haz_key] = haz
            self._hazard_event_ranges[haz_key] = [(shock.start_step, shock.end_step)]

    def _build_route_transport_maps(self) -> None:
        if not self._route_shocks:
            return
        topology_firms = self._raw_topology.get("firms", [])
        topology_edges = self._raw_topology.get("edges", [])
        for shock in self._route_shocks:
            pairs = self._resolve_route_shock_edges(shock, topology_firms, topology_edges)
            self._precomputed_route_transport_edges.append((shock, pairs))
            if not pairs:
                warnings.warn(
                    f"RouteShock '{shock.label}' (tag: {shock.route_tag}) matched no supply edges in the topology.",
                    UserWarning,
                    stacklevel=2,
                )

    def _build_lane_transport_maps(self) -> None:
        if not self._lane_shocks:
            return
        for shock in self._lane_shocks:
            pairs = self._resolve_lane_shock_edges(shock)
            self._precomputed_lane_transport_edges.append((shock, pairs))
            if not pairs:
                warnings.warn(
                    f"LaneShock '{shock.label}' matched no primary supplier edge in the instantiated topology.",
                    UserWarning,
                    stacklevel=2,
                )

    def _resolve_route_shock_edges(
        self,
        shock: RouteShock,
        topology_firms: list,
        topology_edges: list,
    ) -> TransportPairs:
        firm_lookup: Dict[int, dict] = {int(f["id"]): f for f in topology_firms}
        exposed_ids = {
            int(f["id"])
            for f in topology_firms
            if shock.route_tag in f.get("route_dependencies", [])
        }
        if not exposed_ids:
            return []

        pairs: TransportPairs = []

        if shock.waypoint_lon is not None and topology_edges:
            for edge in topology_edges:
                dst_id = int(edge["dst"])
                if dst_id not in exposed_ids:
                    continue
                src_id = int(edge["src"])
                src_firm = firm_lookup.get(src_id)
                dst_firm = firm_lookup.get(dst_id)
                if not src_firm or not dst_firm:
                    continue
                if _lon_between(
                    float(src_firm["lon"]),
                    float(dst_firm["lon"]),
                    float(shock.waypoint_lon),
                ):
                    supplier_agent = self._topology_id_to_agent.get(src_id)
                    buyer_agent = self._topology_id_to_agent.get(dst_id)
                    if supplier_agent is not None and buyer_agent is not None:
                        pairs.append((supplier_agent, buyer_agent))
            if not pairs:
                warnings.warn(
                    f"RouteShock '{shock.label}' (tag: {shock.route_tag}) "
                    f"has waypoint_lon={shock.waypoint_lon} but no topology edges route through it.",
                    UserWarning,
                    stacklevel=2,
                )
            return dedupe_pairs(pairs)

        if topology_edges:
            for edge in topology_edges:
                dst_id = int(edge["dst"])
                if dst_id not in exposed_ids:
                    continue
                src_id = int(edge["src"])
                supplier_agent = self._topology_id_to_agent.get(src_id)
                buyer_agent = self._topology_id_to_agent.get(dst_id)
                if supplier_agent is not None and buyer_agent is not None:
                    pairs.append((supplier_agent, buyer_agent))
            if pairs:
                return dedupe_pairs(pairs)

        for exposed_id in exposed_ids:
            buyer_agent = self._topology_id_to_agent.get(exposed_id)
            if buyer_agent is None:
                continue
            for supplier_agent in getattr(buyer_agent, "connected_firms", []):
                if supplier_agent is not None:
                    pairs.append((supplier_agent, buyer_agent))

        return dedupe_pairs(pairs)

    def _resolve_lane_shock_edges(self, shock: LaneShock) -> TransportPairs:
        supplier_agent = self._topology_id_to_agent.get(int(shock.supplier_id))
        buyer_agent = self._topology_id_to_agent.get(int(shock.buyer_id))
        if supplier_agent is None or buyer_agent is None:
            warnings.warn(
                f"LaneShock '{shock.label}' references unknown firm ids "
                f"(supplier_id={shock.supplier_id}, buyer_id={shock.buyer_id}).",
                UserWarning,
                stacklevel=2,
            )
            return []
        if (int(shock.supplier_id), int(shock.buyer_id)) not in self._primary_supplier_pairs:
            warnings.warn(
                f"LaneShock '{shock.label}' lane "
                f"(supplier_id={shock.supplier_id}, buyer_id={shock.buyer_id}) "
                "is not an active primary supplier relationship in this topology.",
                UserWarning,
                stacklevel=2,
            )
            return []
        return [(supplier_agent, buyer_agent)]

    def _initialize_transport_route_metrics(self) -> None:
        for firm in self._firms:
            firm.route_sales_attempted_this_step = 0.0
            firm.route_sales_blocked_this_step = 0.0
            firm.route_revenue_attempted_this_step = 0.0
            firm.route_revenue_blocked_this_step = 0.0
            firm.inbound_route_sales_attempted_this_step = 0.0
            firm.inbound_route_sales_blocked_this_step = 0.0
            firm.inbound_route_revenue_attempted_this_step = 0.0
            firm.inbound_route_revenue_blocked_this_step = 0.0

    def _reset_transport_route_metrics(self) -> None:
        for firm in self._firms:
            firm.route_sales_attempted_this_step = 0.0
            firm.route_sales_blocked_this_step = 0.0
            firm.route_revenue_attempted_this_step = 0.0
            firm.route_revenue_blocked_this_step = 0.0
            firm.inbound_route_sales_attempted_this_step = 0.0
            firm.inbound_route_sales_blocked_this_step = 0.0
            firm.inbound_route_revenue_attempted_this_step = 0.0
            firm.inbound_route_revenue_blocked_this_step = 0.0

    def _install_transport_reporters(self) -> None:
        self.datacollector._new_model_reporter(
            "Average_Direct_Route_Exposure",
            lambda m: float(np.mean([route_exposure_ratio(f) for f in m._firms])) if m._firms else 0.0,
        )
        self.datacollector._new_model_reporter(
            "Total_Route_Sales_Attempted",
            lambda m: sum(getattr(f, "route_sales_attempted_this_step", 0.0) for f in m._firms),
        )
        self.datacollector._new_model_reporter(
            "Total_Route_Sales_Blocked",
            lambda m: sum(getattr(f, "route_sales_blocked_this_step", 0.0) for f in m._firms),
        )
        self.datacollector._new_model_reporter(
            "Total_Route_Revenue_Attempted",
            lambda m: sum(getattr(f, "route_revenue_attempted_this_step", 0.0) for f in m._firms),
        )
        self.datacollector._new_model_reporter(
            "Total_Route_Revenue_Blocked",
            lambda m: sum(getattr(f, "route_revenue_blocked_this_step", 0.0) for f in m._firms),
        )
        self.datacollector._new_model_reporter(
            "Average_Inbound_Route_Exposure",
            lambda m: float(np.mean([inbound_route_exposure_ratio(f) for f in m._firms])) if m._firms else 0.0,
        )
        self.datacollector._new_model_reporter(
            "Total_Inbound_Route_Revenue_Attempted",
            lambda m: sum(getattr(f, "inbound_route_revenue_attempted_this_step", 0.0) for f in m._firms),
        )
        self.datacollector._new_model_reporter(
            "Total_Inbound_Route_Revenue_Blocked",
            lambda m: sum(getattr(f, "inbound_route_revenue_blocked_this_step", 0.0) for f in m._firms),
        )
        self.datacollector._new_agent_reporter(
            "direct_route_exposure",
            lambda a: route_exposure_ratio(a),
        )
        self.datacollector._new_agent_reporter(
            "inbound_route_exposure",
            lambda a: inbound_route_exposure_ratio(a),
        )
        self.datacollector._new_agent_reporter(
            "inbound_route_revenue_attempted",
            lambda a: getattr(a, "inbound_route_revenue_attempted_this_step", np.nan),
        )
        self.datacollector._new_agent_reporter(
            "inbound_route_revenue_blocked",
            lambda a: getattr(a, "inbound_route_revenue_blocked_this_step", np.nan),
        )
        self.datacollector._new_agent_reporter(
            "route_sales_attempted",
            lambda a: getattr(a, "route_sales_attempted_this_step", np.nan),
        )
        self.datacollector._new_agent_reporter(
            "route_sales_blocked",
            lambda a: getattr(a, "route_sales_blocked_this_step", np.nan),
        )
        self.datacollector._new_agent_reporter(
            "route_revenue_attempted",
            lambda a: getattr(a, "route_revenue_attempted_this_step", np.nan),
        )
        self.datacollector._new_agent_reporter(
            "route_revenue_blocked",
            lambda a: getattr(a, "route_revenue_blocked_this_step", np.nan),
        )

    def _active_transport_blocks(self, step_number: int) -> List[ActiveTransportBlock]:
        active: List[ActiveTransportBlock] = []
        for shock, pairs in self._precomputed_route_transport_edges:
            if not (shock.start_step <= step_number <= shock.end_step):
                continue
            if shock.return_period is not None:
                p_fire = 1.0 - math.exp(-1.0 / (float(shock.return_period) * max(self.steps_per_year, 1)))
                if self.random.random() >= p_fire:
                    continue
            active.append((float(shock.intensity), pairs))

        for shock, pairs in self._precomputed_lane_transport_edges:
            if not (shock.start_step <= step_number <= shock.end_step):
                continue
            if shock.blocked_fraction <= 1e-9:
                continue
            active.append((float(shock.blocked_fraction), pairs))
        return active

    def _apply_transport_patches(
        self,
        active: List[ActiveTransportBlock],
    ) -> Dict[int, Tuple[FirmAgent, object]]:
        if not active:
            return {}

        supplier_blocks: Dict[int, Tuple[FirmAgent, Dict[int, float]]] = {
            int(supplier.unique_id): (supplier, {}) for supplier in self._firms
        }
        for blocked_fraction, pairs in active:
            for supplier_agent, buyer_agent in pairs:
                supplier_id = int(supplier_agent.unique_id)
                buyer_id = int(buyer_agent.unique_id)
                current = supplier_blocks[supplier_id][1].get(buyer_id, 0.0)
                supplier_blocks[supplier_id][1][buyer_id] = max(current, blocked_fraction)

        patches: Dict[int, Tuple[FirmAgent, object]] = {}
        for supplier_id, (supplier, blocked_buyers) in supplier_blocks.items():
            original = supplier.sell_goods_to_firm
            patches[supplier_id] = (supplier, original)
            supplier.sell_goods_to_firm = make_transport_patch(supplier, original, blocked_buyers)
        return patches

    def _remove_transport_patches(self, patches: Dict[int, Tuple[FirmAgent, object]]) -> None:
        for supplier, original in patches.values():
            supplier.sell_goods_to_firm = original

    def _count_reserved_capacity_contracts(self) -> int:
        return sum(
            1
            for contracts in self._reserved_capacity_contracts.values()
            for contract in contracts
            if float(contract.get("quantity", 0.0)) > 1e-9
        )

    @staticmethod
    def _safe_share(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _never_hit_firms(self) -> List[FirmAgent]:
        return [firm for firm in self._firms if not getattr(firm, "ever_directly_hit", False)]

    def _count_ever_directly_hit_firms(self) -> int:
        return sum(1 for firm in self._firms if getattr(firm, "ever_directly_hit", False))

    def _share_ever_directly_hit_firms(self) -> float:
        return self._safe_share(self._count_ever_directly_hit_firms(), len(self._firms))

    def _count_never_hit_firms(self) -> int:
        return len(self._never_hit_firms())

    def _count_never_hit_currently_disrupted_firms(self) -> int:
        return sum(
            1
            for firm in self._never_hit_firms()
            if getattr(firm, "supplier_disruption_this_step", 0.0) > 0
        )

    def _share_never_hit_currently_disrupted_firms(self) -> float:
        return self._safe_share(
            self._count_never_hit_currently_disrupted_firms(),
            len(self._firms),
        )

    def _share_supplier_disruption_borne_by_never_hit_firms(self) -> float:
        total_disruption = sum(getattr(firm, "supplier_disruption_ewma", 0.0) for firm in self._firms)
        never_hit_disruption = sum(
            getattr(firm, "supplier_disruption_ewma", 0.0)
            for firm in self._never_hit_firms()
        )
        return self._safe_share(never_hit_disruption, total_disruption)

    def _share_production_from_never_hit_firms(self) -> float:
        total_production = sum(getattr(firm, "production", 0.0) for firm in self._firms)
        never_hit_production = sum(
            getattr(firm, "production", 0.0)
            for firm in self._never_hit_firms()
        )
        return self._safe_share(never_hit_production, total_production)

    def _share_capital_held_by_never_hit_firms(self) -> float:
        total_capital = sum(getattr(firm, "capital_stock", 0.0) for firm in self._firms)
        never_hit_capital = sum(
            getattr(firm, "capital_stock", 0.0)
            for firm in self._never_hit_firms()
        )
        return self._safe_share(never_hit_capital, total_capital)

    def _count_ever_indirectly_disrupted_before_direct_hit_firms(self) -> int:
        return sum(
            1
            for firm in self._firms
            if getattr(firm, "ever_indirectly_disrupted_before_direct_hit", False)
        )

    def _share_ever_indirectly_disrupted_before_direct_hit_firms(self) -> float:
        return self._safe_share(
            self._count_ever_indirectly_disrupted_before_direct_hit_firms(),
            len(self._firms),
        )

    def _advance_adaptation_expectations(self) -> None:
        """Reset adaptation accounting and periodically refresh firm resilience targets."""

        self.adaptation_updates_this_step = 0

        for firm in self._firms:
            firm.begin_period_adaptation()

        if not self.firm_adaptation_enabled:
            return

        for firm in self._firms:
            if not firm.active:
                continue
            if not firm.adaptation_enabled:
                continue
            interval = max(1, int(firm.decision_interval))
            if (self.current_step - 1) % interval != 0:
                continue
            firm._select_adaptation_action()
            if firm.pending_adaptation_increment > 0:
                self.adaptation_updates_this_step += 1

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
                suppliers_by_sector: dict[str, list[FirmAgent]] = defaultdict(list)
                for supplier in buyer._technical_input_suppliers():
                    suppliers_by_sector[supplier.sector].append(supplier)
                if buyer.INPUT_COEFF <= 0 or not suppliers_by_sector:
                    continue
                buyer_sales = expected_sales.get(buyer.unique_id, 0.0)
                if buyer_sales <= 0:
                    continue
                for sector, sector_coeff in buyer._input_coefficients_by_sector().items():
                    sector_suppliers = suppliers_by_sector.get(sector, [])
                    if not sector_suppliers or sector_coeff <= 0:
                        continue
                    supplier_share = sector_coeff / len(sector_suppliers)
                    for supplier in sector_suppliers:
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
        technical_suppliers = firm._technical_input_suppliers()
        avg_input_price = float(np.mean([s.price for s in technical_suppliers])) if technical_suppliers else 0.0
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
        configured_capital = float(firm.capital_stock)
        seeded_capital = max(configured_capital, capital_target)
        if self._firm_topology is not None and seeded_capital > configured_capital + 1e-9:
            self._startup_capital_floor_overrides.append(
                {
                    "firm_id": int(firm.unique_id),
                    "configured_capital": configured_capital,
                    "seeded_capital": float(seeded_capital),
                }
            )
        firm.capital_stock = seeded_capital
        firm.money = max(firm.money, working_capital)
        firm.startup_expected_sales = expected_sales
        firm.startup_inventory_target = inventory_target
        firm.startup_capital_stock = seeded_capital
        firm.startup_money = firm.money
        firm.startup_price = firm.price
        firm.startup_wage_offer = firm.wage_offer

    def _initialize_firm_operating_state(self) -> None:
        """Initialize firms from demand-consistent inventories and working capital."""

        expected_sales = self._solve_initial_expected_sales()
        for firm in self._firms:
            self._seed_firm_operating_state(
                firm,
                expected_sales=expected_sales.get(firm.unique_id, 1.0),
            )
        if self._startup_capital_floor_overrides:
            warnings.warn(
                "Raised startup capital to the demand-consistent seeding floor for "
                f"{len(self._startup_capital_floor_overrides)} topology-defined firms.",
                RuntimeWarning,
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
            if self.land_boundaries_path:
                path = Path(self.land_boundaries_path).expanduser()
            else:
                local_path = Path(__file__).resolve().parent / "data" / "ne_110m_admin_0_countries"
                repo_path = Path(__file__).resolve().parents[1] / "data" / "ne_110m_admin_0_countries"
                path = local_path if local_path.exists() else repo_path
            world = gpd.read_file(str(path))
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

    def _load_or_compute_land_coordinates(self) -> List[Coords]:
        """Load cached land coordinates when available, otherwise compute and persist them."""

        path_tag = self.land_boundaries_path or "__default__"
        cache_key = (len(self.lon_vals), len(self.lat_vals), float(self.grid_resolution), path_tag)
        cached = self._land_coordinate_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        cache_dir = Path(__file__).resolve().parent / ".cache"
        res_tag = str(self.grid_resolution).replace(".", "p")
        path_suffix = Path(path_tag).name.replace(".", "_").replace("-", "_")
        cache_path = cache_dir / f"land_coordinates_{len(self.lon_vals)}x{len(self.lat_vals)}_{res_tag}_{path_suffix}.npy"

        if cache_path.exists():
            try:
                cached_array = np.load(cache_path, allow_pickle=False)
                coords = [tuple(map(int, row)) for row in cached_array.tolist()]
                self._land_coordinate_cache[cache_key] = coords
                return coords.copy()
            except Exception:
                pass

        coords = self._compute_land_coordinates()
        self._land_coordinate_cache[cache_key] = coords
        try:
            cache_dir.mkdir(exist_ok=True)
            np.save(cache_path, np.asarray(coords, dtype=np.int32), allow_pickle=False)
        except Exception:
            pass
        return coords.copy()

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
                # Preserve the external topology identifier for shock lookups.
                ag.topology_id = int(firm["id"])
                ag.capital_stock = float(firm.get("capital", 1.0))
                self.grid.place_agent(ag, (x, y))
                id_to_agent[int(firm["id"])] = ag

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

    def _parse_recipe_range(self, value: object) -> tuple[float, float]:
        if isinstance(value, (int, float)):
            share = float(value)
            return share, share
        if isinstance(value, dict):
            low = float(value.get("min", value.get("low", 0.0)))
            high = float(value.get("max", value.get("high", low)))
            return low, high
        if isinstance(value, (list, tuple)) and value:
            low = float(value[0])
            high = float(value[1] if len(value) > 1 else value[0])
            return low, high
        return 0.0, 0.0

    def _resolve_recipe_sector(self, sector: str, available_sectors: set[str]) -> str | None:
        if sector in available_sectors:
            return sector
        if sector in {"commodity", "agriculture"}:
            for fallback in ("commodity", "agriculture"):
                if fallback in available_sectors:
                    return fallback
        return sector

    def _draw_input_recipe_shares(
        self,
        firm: FirmAgent,
        available_sectors: set[str],
    ) -> dict[str, float]:
        if firm.INPUT_COEFF <= 0:
            return {}

        configured_ranges = self.input_recipe_ranges.get(firm.sector, {})
        raw_shares: dict[str, float] = defaultdict(float)
        for configured_sector, range_spec in configured_ranges.items():
            resolved_sector = self._resolve_recipe_sector(str(configured_sector), available_sectors)
            if resolved_sector is None:
                continue
            low, high = self._parse_recipe_range(range_spec)
            low = max(0.0, low)
            high = max(low, high)
            raw_shares[resolved_sector] += self.random.uniform(low, high)

        if not raw_shares:
            sector_counts: dict[str, int] = defaultdict(int)
            for supplier in firm.connected_firms:
                sector_counts[supplier.sector] += 1
            raw_shares.update({sector: float(count) for sector, count in sector_counts.items()})

        total = sum(raw_shares.values())
        if total <= 1e-12:
            return {}
        return {
            sector: share / total
            for sector, share in raw_shares.items()
            if share > 1e-12
        }

    def _assign_firm_input_recipes(self) -> None:
        available_sectors = {firm.sector for firm in self._firms}
        for firm in self._firms:
            firm.input_recipe_shares = self._draw_input_recipe_shares(firm, available_sectors)

    def _warn_missing_recipe_supplier_coverage(self) -> None:
        """Warn when a firm's recipe requires a supplier sector absent from topology."""
        for buyer in self._firms:
            if buyer.INPUT_COEFF <= 0 or not buyer.input_recipe_shares:
                continue

            for required_sector in buyer.input_recipe_shares:
                if any(supplier.sector == required_sector for supplier in buyer.connected_firms):
                    continue
                warnings.warn(
                    "Topology firm "
                    f"{getattr(buyer, 'topology_id', buyer.unique_id)} requires "
                    f"input sector {required_sector!r} but has no linked supplier "
                    "in that sector; this recipe input can bind production.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    # --------------------------------------------------------------------- #
    #                               MESA STEP                               #
    # --------------------------------------------------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Advance model by one timestep."""
        self.current_step += 1
        self._reset_transport_route_metrics()

        for household in self._households:
            household.begin_step_income_accounting()
        for firm in self._firms:
            firm.begin_step_financial_accounting()

        # Firms update resilience decisions before the new hazard state is sampled.
        self._advance_adaptation_expectations()

        # Capital destroyed in the previous step can only be rebuilt now, before
        # the new period's hazards and operating decisions are realized.
        for firm in self._firms:
            firm._fund_deferred_capital_repair_before_planning()

        # Sample hazard independently for every cell based on per-step RP draws.
        self._sample_pixelwise_hazard()

        # ---------------- Demand planning phase --------------------- #
        for firm in self._firms:
            firm.plan_operations()
        self._prepare_reserved_capacity_contracts()

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
        transport_patches = self._apply_transport_patches(
            self._active_transport_blocks(self.current_step)
        )
        try:
            for firm in firms:
                firm.step()
        finally:
            self._remove_transport_patches(transport_patches)

        # 3. Households consume the goods produced in the current period.
        households = self._households.copy()
        self.random.shuffle(households)
        for hh in households:
            hh.consume_goods()

        # 4. Close the accounting period after all firm-to-firm and
        #    household transactions have been recorded.
        for firm in firms:
            firm.close_step()

        # Recovery happens only after the current period's shock, planning,
        # production, and accounting have all closed, so any productivity
        # rebound affects the next step rather than smoothing the current one.
        for firm in self._firms:
            firm._apply_damage_recovery()

        # ---------------- Record average wage for data collection ----- #
        if self._firms:
            self.mean_wage = float(np.mean([f.wage_offer for f in self._firms]))

        # Collect data before the failure policy runs so this period's
        # bottleneck counts reflect firms that actually operated this step.
        self.datacollector.collect(self)

        # ---------------- Firm failure policy ------------------------ #
        self.steps_since_replacement += 1
        if self.steps_since_replacement >= self.replacement_frequency:
            self._apply_firm_failure_policy()
            self.steps_since_replacement = 0

    # --------------------------------------------------------------------- #
    #                         FIRM FAILURE METHODS                           #
    # --------------------------------------------------------------------- #
    def _failed_firms_due_for_policy(self) -> List[FirmAgent]:
        """Return active firms below the survival cash threshold."""
        failed_firms = [
            firm
            for firm in self._firms
            if firm.active and firm.money < self.min_money_survival
        ]
        max_failures = max(1, len(self._firms) // 4)
        return failed_firms[:max_failures]

    def _deactivate_failed_firm(self, firm: FirmAgent) -> None:
        """Remove a failed firm from active markets while keeping panel history."""
        firm.active = False
        firm.production = 0.0
        firm.consumption = 0.0
        firm.target_output = 0.0
        firm.target_labor = 0
        firm.target_input_units = 0.0
        firm.inventory_output = 0.0
        firm.inventory_inputs.clear()
        firm.employees.clear()
        firm.limiting_factor = "inactive"
        self.total_firm_exits += 1
        print(
            f"[EXIT] Step {self.current_step}: Deactivated failed firm "
            f"{firm.unique_id} (total exits: {self.total_firm_exits})"
        )

    def _reset_failed_firm_from_startup_state(self, firm: FirmAgent) -> None:
        """Reset a failed firm to its original startup operating state."""
        startup_expected_sales = float(getattr(firm, "startup_expected_sales", 1.0))
        startup_inventory_target = float(getattr(firm, "startup_inventory_target", 1.0))
        startup_capital_stock = float(
            getattr(
                firm,
                "startup_capital_stock",
                max(1.0, startup_expected_sales * firm.original_capital_coeff),
            )
        )
        startup_money = float(getattr(firm, "startup_money", 100.0))

        firm.expected_sales = max(1.0, startup_expected_sales)
        firm.active = True
        firm.base_inventory_target = max(1.0, startup_inventory_target)
        firm.base_capital_target = max(1.0, startup_capital_stock)
        firm.target_capital_stock = firm.base_capital_target
        firm.capital_stock = firm.base_capital_target
        firm.inventory_output = firm.base_inventory_target
        firm.inventory_inputs.clear()
        firm.price = float(getattr(firm, "startup_price", firm.price))
        firm.wage_offer = float(getattr(firm, "startup_wage_offer", self.initial_mean_wage))
        firm.damage_factor = 1.0
        firm.counterfactual_damage_factor = 1.0
        firm.deferred_capital_repair = False
        firm.survival_time = 0
        firm.sales_last_step = 0.0
        firm.revenue_last_step = 0.0
        firm.sales_this_step = 0.0
        firm.revenue_this_step = 0.0
        firm.household_sales_last_step = 0.0
        firm.household_sales_this_step = 0.0
        firm.inventory_available_last_step = firm.inventory_output
        firm.production = 0.0
        firm.consumption = 0.0
        firm.wage_bill_this_step = 0.0
        firm.input_spend_this_step = 0.0
        firm.depreciation_this_step = 0.0
        firm.operating_surplus_this_step = 0.0
        firm.net_profit_this_step = 0.0
        firm.direct_loss_expense_this_step = 0.0
        firm.dividends_paid_this_step = 0.0
        firm.investment_spending_this_step = 0.0
        firm.working_capital_credit_used_this_step = 0.0
        firm.working_capital_credit_limit = 0.0
        firm.raw_direct_loss_fraction_this_step = 0.0
        firm.adapted_direct_loss_fraction_this_step = 0.0
        firm.supplier_disruption_this_step = 0.0
        firm.raw_supplier_disruption_this_step = 0.0
        firm.hazard_operating_shortfall_this_step = 0.0
        firm.reset_adaptation_state()

        equity_needed = max(0.0, startup_money - firm.money)
        self.transfer_household_equity_to_firm(firm, equity_needed)

        self._debug_recent_replacements.append(firm.unique_id)
        if len(self._debug_recent_replacements) > 20:
            self._debug_recent_replacements = self._debug_recent_replacements[-20:]

        self.total_firm_replacements += 1
        print(
            f"[STARTUP_RESET] Step {self.current_step}: Reset failed firm "
            f"{firm.unique_id} from startup state "
            f"(total: {self.total_firm_replacements})"
        )

    def _apply_firm_failure_policy(self) -> None:
        """Apply the configured exit or startup-reset policy to failed firms."""
        if self.current_step < 5:
            return

        failed_firms = self._failed_firms_due_for_policy()
        if not failed_firms:
            return

        for firm in failed_firms:
            if self.firm_replacement == "none":
                self._deactivate_failed_firm(firm)
            else:
                self._reset_failed_firm_from_startup_state(firm)

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

        Annual event probabilities from raster *i* are converted into per-step
        Bernoulli draws using ``1 - (1 - p_annual) ** (1 / steps_per_year)``,
        independent across cells and return periods. If multiple return periods
        trigger on the same cell within a step, we keep the maximum depth.

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

        for firm in self._firms:
            firm.pre_hazard_damage_factor = float(firm.damage_factor)
            firm.pre_hazard_capital_stock = float(firm.capital_stock)
            firm.pre_hazard_inventory_output = float(firm.inventory_output)
            firm.pre_hazard_inventory_inputs = dict(firm.inventory_inputs)

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
                if p_annual >= 1.0:
                    p_hit = 1.0
                else:
                    p_hit = 1.0 - (1.0 - p_annual) ** (1.0 / max(1, self.steps_per_year))
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
        damage_funcs = get_damage_functions(self.damage_functions_path)

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
                adapted_loss_fraction = ag.get_adapted_loss_fraction(combined_loss_agent)
                pre_capital_stock = float(ag.capital_stock)
                pre_output_inventory = float(ag.inventory_output)
                capital_replacement_value = (
                    pre_capital_stock * float(getattr(ag, "CAPITAL_INSTALLATION_COST", 1.0))
                )
                supplier_price_map = {
                    supplier.unique_id: max(float(supplier.price), 0.5)
                    for supplier in ag.connected_firms
                }
                fallback_input_price = (
                    float(np.mean(list(supplier_price_map.values())))
                    if supplier_price_map
                    else max(float(ag.price), 0.5)
                )
                input_inventory_value = sum(
                    float(units) * supplier_price_map.get(int(supplier_id), fallback_input_price)
                    for supplier_id, units in ag.inventory_inputs.items()
                )
                ag.record_direct_losses(
                    raw_loss_fraction=combined_loss_agent,
                    adapted_loss_fraction=adapted_loss_fraction,
                )
                ag.direct_loss_expense_this_step += (
                    capital_replacement_value * adapted_loss_fraction
                    + pre_output_inventory * max(float(ag.price), 0.5) * adapted_loss_fraction
                    + input_inventory_value * adapted_loss_fraction
                )
                ag.capital_stock *= 1 - adapted_loss_fraction
                ag.damage_factor *= 1 - adapted_loss_fraction
                ag.counterfactual_damage_factor *= 1 - combined_loss_agent
                ag.inventory_output *= 1 - adapted_loss_fraction
                for k in list(ag.inventory_inputs.keys()):
                    ag.inventory_inputs[k] *= 1 - adapted_loss_fraction

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
