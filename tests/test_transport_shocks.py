from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from upstream import HazardRasterEvent, LaneShock, NodeShock, RouteShock, build_model, run_model


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DAMAGE_FUNCTIONS_PATH = _REPO_ROOT / "data" / "global_flood_depth_damage_functions.xlsx"
_LAND_BOUNDARIES_PATH = _REPO_ROOT / "data" / "ne_110m_admin_0_countries"


def _write_transport_topology(tmp_path: Path, *, route_dependencies: list[str] | None = None) -> str:
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
    path = tmp_path / "transport_topology.json"
    path.write_text(json.dumps(topology))
    return str(path)


def _write_global_raster(path: Path, depth: float) -> str:
    data = np.full((180, 360), depth, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=360,
        height=180,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-180.0, 90.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    return str(path)


def _firm_by_id(model, unique_id: int):
    return next(firm for firm in model._firms if firm.unique_id == unique_id)


def test_lane_shock_throttles_deliveries_without_direct_damage(tmp_path: Path) -> None:
    topology_path = _write_transport_topology(tmp_path)
    model = build_model(
        num_households=20,
        firm_topology_path=topology_path,
        apply_hazard_impacts=False,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        seed=5,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
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
    )

    supplier = _firm_by_id(model, 1)
    buyer = _firm_by_id(model, 2)
    supplier.inventory_output = 20.0
    supplier.price = 2.0

    patches = model._apply_transport_patches(model._active_transport_blocks(1))
    try:
        delivered = supplier.sell_goods_to_firm(buyer, quantity=10.0)
    finally:
        model._remove_transport_patches(patches)

    assert delivered == pytest.approx(3.0)
    assert supplier.raw_direct_loss_fraction_this_step == 0.0
    assert buyer.raw_direct_loss_fraction_this_step == 0.0

    _, results_df, _ = run_model(
        steps=1,
        num_households=20,
        firm_topology_path=topology_path,
        apply_hazard_impacts=False,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        seed=5,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
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
    )

    assert results_df["Average_Realized_Direct_Loss"].iloc[0] == 0.0
    assert results_df["Flooded_Firms"].iloc[0] == 0


def test_route_shock_resolves_pairs_from_route_dependencies(tmp_path: Path) -> None:
    topology_path = _write_transport_topology(tmp_path, route_dependencies=["TEST_ROUTE"])
    model = build_model(
        num_households=20,
        firm_topology_path=topology_path,
        apply_hazard_impacts=False,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        seed=7,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
        route_shocks=[
            RouteShock(
                label="Route",
                route_tag="TEST_ROUTE",
                intensity=0.4,
                start_step=1,
                end_step=1,
            )
        ],
    )

    shock, pairs = model._precomputed_route_transport_edges[0]

    assert shock.route_tag == "TEST_ROUTE"
    assert [(supplier.unique_id, buyer.unique_id) for supplier, buyer in pairs] == [(1, 2)]


def test_mixed_raster_node_and_route_shocks_run_together(tmp_path: Path) -> None:
    topology_path = _write_transport_topology(tmp_path, route_dependencies=["TEST_ROUTE"])
    raster_path = _write_global_raster(tmp_path / "global_depth.tif", depth=2.0)

    model, results_df, _ = run_model(
        steps=2,
        num_households=20,
        firm_topology_path=topology_path,
        apply_hazard_impacts=True,
        adaptation_params={"enabled": False},
        consumption_ratios={"retail": 1.0},
        seed=11,
        damage_functions_path=str(_DAMAGE_FUNCTIONS_PATH),
        land_boundaries_path=str(_LAND_BOUNDARIES_PATH),
        raster_hazard_events=[
            HazardRasterEvent(
                return_period=1,
                start_step=1,
                end_step=1,
                hazard_type="FL",
                path=raster_path,
            )
        ],
        node_shocks=[
            NodeShock(
                label="Node",
                hazard_type="CUSTOM_NODE",
                intensity=0.5,
                start_step=1,
                end_step=1,
                firm_ids=(1,),
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

    assert metadata["RasterHazardEventCount"] == 1
    assert metadata["NodeShockCount"] == 1
    assert metadata["RouteShockCount"] == 1
    assert "Average_Direct_Route_Exposure" in results_df.columns
    assert "Average_Realized_Direct_Loss" in results_df.columns
    assert float(results_df["Flooded_Firms"].max()) > 0.0
    assert float(results_df["Average_Realized_Direct_Loss"].max()) > 0.0
