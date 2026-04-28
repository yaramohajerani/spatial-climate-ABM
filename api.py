from __future__ import annotations

from typing import Iterable, Mapping, Any

import pandas as pd

try:  # pragma: no cover - package import path
    from .model import EconomyModel
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
except ImportError:  # pragma: no cover - flat script / test import path
    from model import EconomyModel  # type: ignore[no-redef]
    from shock_inputs import (  # type: ignore[no-redef]
        HazardRasterEvent,
        LaneShock,
        NodeShock,
        RouteShock,
        normalize_lane_shocks,
        normalize_node_shocks,
        normalize_raster_hazard_events,
        normalize_route_shocks,
    )


def build_model(
    *,
    width: int | None = None,
    height: int | None = None,
    num_households: int = 100,
    num_firms: int = 20,
    hazard_events=None,
    raster_hazard_events: Iterable[HazardRasterEvent | Mapping[str, Any] | tuple[int, int, int, str, str | None]] | None = None,
    node_shocks: Iterable[NodeShock | Mapping[str, Any]] | None = None,
    lane_shocks: Iterable[LaneShock | Mapping[str, Any]] | None = None,
    route_shocks: Iterable[RouteShock | Mapping[str, Any]] | None = None,
    seed: int | None = None,
    start_year: int = 0,
    steps_per_year: int = 4,
    firm_topology_path: str | None = None,
    apply_hazard_impacts: bool = True,
    apply_transport_shocks: bool | None = None,
    adaptation_params: dict | None = None,
    consumption_ratios: dict | None = None,
    input_recipe_ranges: dict | None = None,
    firm_replacement: str = "startup_reset",
    dynamic_supplier_search: bool = True,
    grid_resolution: float = 1.0,
    household_relocation: bool = False,
    damage_functions_path: str | None = None,
    land_boundaries_path: str | None = None,
) -> EconomyModel:
    return EconomyModel(
        width=width,
        height=height,
        num_households=num_households,
        num_firms=num_firms,
        hazard_events=hazard_events,
        raster_hazard_events=normalize_raster_hazard_events(raster_hazard_events),
        node_shocks=normalize_node_shocks(node_shocks),
        lane_shocks=normalize_lane_shocks(lane_shocks),
        route_shocks=normalize_route_shocks(route_shocks),
        seed=seed,
        start_year=start_year,
        steps_per_year=steps_per_year,
        firm_topology_path=firm_topology_path,
        apply_hazard_impacts=apply_hazard_impacts,
        apply_transport_shocks=apply_transport_shocks,
        adaptation_params=adaptation_params,
        consumption_ratios=consumption_ratios,
        input_recipe_ranges=input_recipe_ranges,
        firm_replacement=firm_replacement,
        dynamic_supplier_search=dynamic_supplier_search,
        grid_resolution=grid_resolution,
        household_relocation=household_relocation,
        damage_functions_path=damage_functions_path,
        land_boundaries_path=land_boundaries_path,
    )


def run_model(
    *,
    steps: int,
    **model_kwargs,
) -> tuple[EconomyModel, pd.DataFrame, pd.DataFrame]:
    model = build_model(**model_kwargs)
    for _ in range(int(steps)):
        model.step()
    results_df = model.results_to_dataframe()
    agents_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agents_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")
    return model, results_df, agents_df
