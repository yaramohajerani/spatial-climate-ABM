from __future__ import annotations

import json
from pathlib import Path

from api import build_model, run_model
from run_simulation import _coerce_shock_inputs
from shock_inputs import (
    HazardRasterEvent,
    LaneShock,
    NodeShock,
    RouteShock,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DAMAGE_FUNCTIONS_PATH = _REPO_ROOT / "data" / "global_flood_depth_damage_functions.xlsx"
_LAND_BOUNDARIES_PATH = _REPO_ROOT / "data" / "ne_110m_admin_0_countries"


def _write_topology(tmp_path: Path, *, route_dependencies: list[str] | None = None) -> str:
    buyer = {
        "id": 2,
        "sector": "manufacturing",
        "lon": 1.0,
        "lat": 1.0,
        "capital": 5.0,
    }
    if route_dependencies:
        buyer["route_dependencies"] = list(route_dependencies)

    topology = {
        "firms": [
            {"id": 1, "sector": "commodity", "lon": 0.0, "lat": 0.0, "capital": 5.0},
            buyer,
            {"id": 3, "sector": "retail", "lon": 2.0, "lat": 1.0, "capital": 5.0},
            {"id": 4, "sector": "commodity", "lon": -1.0, "lat": 0.0, "capital": 5.0},
        ],
        "edges": [
            {"src": 1, "dst": 2},
            {"src": 2, "dst": 3},
        ],
    }
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(topology))
    return str(path)


def test_build_model_records_resource_paths_and_shock_counts(tmp_path: Path) -> None:
    topology_path = _write_topology(tmp_path, route_dependencies=["TEST_ROUTE"])

    model = build_model(
        num_households=10,
        firm_topology_path=topology_path,
        apply_hazard_impacts=False,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        seed=7,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
        node_shocks=[
            NodeShock(
                label="Node",
                hazard_type="CUSTOM_NODE",
                intensity=0.4,
                start_step=1,
                end_step=1,
                affected_coords=((0.0, 0.0),),
            )
        ],
        lane_shocks=[
            LaneShock(
                label="Lane",
                supplier_id=1,
                buyer_id=2,
                capacity_fraction=0.3,
                start_step=1,
                end_step=1,
            )
        ],
        route_shocks=[
            RouteShock(
                label="Route",
                route_tag="TEST_ROUTE",
                intensity=0.5,
                start_step=1,
                end_step=1,
            )
        ],
    )

    metadata = model.effective_configuration_metadata()

    assert metadata["DamageFunctionsPath"] == str(_DAMAGE_FUNCTIONS_PATH)
    assert metadata["LandBoundariesPath"] == str(_LAND_BOUNDARIES_PATH)
    assert metadata["RasterHazardEventCount"] == 0
    assert metadata["NodeShockCount"] == 1
    assert metadata["LaneShockCount"] == 1
    assert metadata["RouteShockCount"] == 1


def test_run_model_returns_model_and_agent_frames(tmp_path: Path) -> None:
    topology_path = _write_topology(tmp_path)

    model, results_df, agents_df = run_model(
        steps=2,
        num_households=10,
        firm_topology_path=topology_path,
        apply_hazard_impacts=False,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        seed=9,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
    )

    assert model.current_step == 2
    assert len(results_df) == 2
    assert {"Step", "AgentID"}.issubset(set(agents_df.columns))


def test_build_model_forwards_commercial_model_controls(tmp_path: Path) -> None:
    topology_path = _write_topology(tmp_path)

    model = build_model(
        num_households=10,
        firm_topology_path=topology_path,
        apply_hazard_impacts=False,
        apply_transport_shocks=True,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        input_recipe_ranges={"manufacturing": {"commodity": [1.0, 1.0]}},
        firm_replacement="none",
        dynamic_supplier_search=False,
        seed=10,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
    )

    metadata = model.effective_configuration_metadata()

    assert metadata["FirmReplacement"] == "none"
    assert metadata["DynamicSupplierSearch"] is False
    assert metadata["ApplyTransportShocks"] is True
    assert model.input_recipe_ranges["manufacturing"] == {"commodity": [1.0, 1.0]}


def test_param_input_normalization_expands_lane_link_lists() -> None:
    raster_events, node_events, lane_events, route_events = _coerce_shock_inputs(
        legacy_rp_files=["5:1:2:FL:None"],
        raster_hazard_events=None,
        node_shocks=[
            {
                "label": "Node",
                "hazard_type": "CUSTOM_NODE",
                "intensity": 0.7,
                "start_step": 3,
                "end_step": 4,
                "firm_ids": [2],
            }
        ],
        lane_shocks=[
            {
                "label": "Lane",
                "links": [[1, 2], [4, 2]],
                "capacity_fraction": 0.25,
                "start_step": 5,
                "end_step": 6,
            }
        ],
        route_shocks=[
            {
                "label": "Route",
                "route_tag": "TEST_ROUTE",
                "intensity": 0.6,
                "start_step": 7,
                "end_step": 8,
            }
        ],
    )

    assert raster_events == [
        HazardRasterEvent(return_period=5, start_step=1, end_step=2, hazard_type="FL", path=None)
    ]
    assert node_events == [
        NodeShock(
            label="Node",
            hazard_type="CUSTOM_NODE",
            intensity=0.7,
            start_step=3,
            end_step=4,
            firm_ids=(2,),
        )
    ]
    assert lane_events == [
        LaneShock(
            label="Lane #1",
            supplier_id=1,
            buyer_id=2,
            capacity_fraction=0.25,
            start_step=5,
            end_step=6,
        ),
        LaneShock(
            label="Lane #2",
            supplier_id=4,
            buyer_id=2,
            capacity_fraction=0.25,
            start_step=5,
            end_step=6,
        ),
    ]
    assert route_events == [
        RouteShock(
            label="Route",
            route_tag="TEST_ROUTE",
            intensity=0.6,
            start_step=7,
            end_step=8,
        )
    ]


def test_route_shock_from_mapping_requires_route_tag_or_bottleneck_id() -> None:
    try:
        RouteShock.from_mapping(
            {
                "label": "Route",
                "intensity": 0.4,
                "start_step": 1,
                "end_step": 2,
            }
        )
    except ValueError as exc:
        assert "route_tag or bottleneck_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RouteShock.from_mapping() to reject missing route identifiers")
