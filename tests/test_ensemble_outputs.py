import json
from types import SimpleNamespace

import pandas as pd
import pytest

from ensemble_utils import apply_metadata, build_ensemble_summary, ensemble_seed_metadata
from plot_from_csv import summarize_members_for_plot
from run_simulation import (
    _base_metadata,
    _coerce_shock_inputs,
    _merge_market_structure_settings,
    _plot_network_evolution_from_json,
    _resolve_seed_list,
)
from shock_inputs import (
    HazardRasterEvent,
    LaneShock,
    NodeShock,
    RouteShock,
)


def test_resolve_seed_list_deduplicates_explicit_seeds() -> None:
    args = SimpleNamespace(seeds=[7, 3, 7, 5], n_seeds=None, seed_start=None, seed=42)
    assert _resolve_seed_list(args) == [7, 3, 5]


def test_resolve_seed_list_builds_consecutive_range() -> None:
    args = SimpleNamespace(seeds=None, n_seeds=4, seed_start=10, seed=42)
    assert _resolve_seed_list(args) == [10, 11, 12, 13]


def test_market_structure_cli_settings_override_parameter_file() -> None:
    args = SimpleNamespace(
        firm_replacement="none",
        dynamic_supplier_search=True,
        max_dynamic_suppliers_per_sector=4,
    )

    _merge_market_structure_settings(
        args,
        {
            "firm_replacement": "startup_reset",
            "dynamic_supplier_search": {
                "enabled": False,
                "max_suppliers_per_sector": 1,
            },
        },
    )

    assert args.firm_replacement == "none"
    assert args.dynamic_supplier_search is True
    assert args.max_dynamic_suppliers_per_sector == 4


def test_market_structure_settings_fall_back_to_parameter_file() -> None:
    args = SimpleNamespace(
        firm_replacement=None,
        dynamic_supplier_search=None,
        max_dynamic_suppliers_per_sector=None,
    )

    _merge_market_structure_settings(
        args,
        {
            "firm_replacement": "none",
            "dynamic_supplier_search": {
                "enabled": False,
                "max_suppliers_per_sector": 1,
            },
        },
    )

    assert args.firm_replacement == "none"
    assert args.dynamic_supplier_search is False
    assert args.max_dynamic_suppliers_per_sector == 1


def test_network_evolution_replay_does_not_overwrite_input_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    payload = {
        "firm_meta": {
            "1": {"sector": "commodity", "pos": [0, 0]},
            "2": {"sector": "manufacturing", "pos": [1, 1]},
        },
        "start_year": 2000,
        "steps_per_year": 4,
        "snapshots": [
            {
                "step": 0,
                "edges": [[1, 2]],
                "active_ids": [1, 2],
                "inactive_ids": [],
            }
        ],
    }
    json_path = tmp_path / "foo_network_evolution.json"
    original = json.dumps(payload, indent=2)
    json_path.write_text(original)

    out_path = _plot_network_evolution_from_json(
        json_path,
        None,
        SimpleNamespace(start_year=0, steps_per_year=4),
    )

    assert out_path == str(tmp_path / "foo_network_evolution.png")
    assert (tmp_path / "foo_network_evolution.png").exists()
    assert json_path.read_text() == original


def test_network_evolution_replay_rejects_json_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    payload = {
        "firm_meta": {
            "1": {"sector": "commodity", "pos": [0, 0]},
            "2": {"sector": "manufacturing", "pos": [1, 1]},
        },
        "snapshots": [
            {
                "step": 0,
                "edges": [[1, 2]],
                "active_ids": [1, 2],
                "inactive_ids": [],
            }
        ],
    }
    json_path = tmp_path / "foo_network_evolution.json"
    original = json.dumps(payload, indent=2)
    json_path.write_text(original)

    with pytest.raises(ValueError, match="must not use a .json extension"):
        _plot_network_evolution_from_json(
            json_path,
            tmp_path / "bad_output.json",
            SimpleNamespace(start_year=0, steps_per_year=4),
        )

    assert json_path.read_text() == original
    assert not (tmp_path / "bad_output.json").exists()


def test_member_summary_helpers_match_and_track_ensemble_size() -> None:
    member_df = pd.DataFrame(
        {
            "Scenario": ["Hazard + Adaptation"] * 6,
            "Step": [0, 0, 0, 1, 1, 1],
            "Year": [2030.0, 2030.0, 2030.0, 2030.25, 2030.25, 2030.25],
            "Seed": [1, 2, 3, 1, 2, 3],
            "Meta_ApplyHazards": [True] * 6,
            "Meta_AdaptationSensitivityMin": [2.0] * 6,
            "Meta_ParamFile": ["params.json"] * 6,
            "Firm_Production": [10.0, 14.0, 16.0, 12.0, 18.0, 24.0],
            "Household_Consumption": [4.0, 5.0, 7.0, 6.0, 8.0, 10.0],
            "Mean_Price": [2.0, 2.5, 3.0, 2.2, 2.4, 2.8],
        }
    )

    runner_summary = build_ensemble_summary(member_df, group_cols=["Scenario", "Step", "Year"])
    plot_summary = summarize_members_for_plot(member_df)

    sort_cols = ["Scenario", "Step", "EnsembleStatistic"]
    runner_summary = runner_summary.sort_values(sort_cols).reset_index(drop=True)
    plot_summary = plot_summary.sort_values(sort_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(runner_summary, plot_summary)

    mean_step0 = runner_summary[
        (runner_summary["Step"] == 0) & (runner_summary["EnsembleStatistic"] == "mean")
    ].iloc[0]
    assert mean_step0["EnsembleSize"] == 3
    assert mean_step0["Firm_Production"] == 40.0 / 3.0
    assert mean_step0["Household_Consumption"] == 16.0 / 3.0
    assert "Meta_AdaptationSensitivityMin" not in runner_summary.columns


def test_ensemble_utils_metadata_helpers_track_seed_provenance() -> None:
    df = pd.DataFrame({"Step": [0], "Value": [1.0]})

    seed_metadata = ensemble_seed_metadata([3, 5])
    enriched = apply_metadata(df, seed_metadata)

    assert enriched["Meta_SeedCount"].iloc[0] == 2
    assert enriched["Meta_SeedList"].iloc[0] == "3,5"


def test_base_metadata_matches_model_adaptation_defaults() -> None:
    args = SimpleNamespace(
        param_file=None,
        topology=None,
        start_year=0,
        steps_per_year=4,
        steps=10,
        num_households=100,
        grid_resolution=1.0,
        household_relocation=False,
        no_hazards=False,
        no_adaptation=False,
        adaptation_strategy=None,
        adaptation_sensitivity_min=None,
        adaptation_sensitivity_max=None,
        consumption_ratios={"retail": 1.0},
        save_agent_ensemble=False,
        ensemble_plot_stat="mean",
    )

    metadata = _base_metadata(
        args=args,
        events=[],
        apply_hazards=True,
        adaptation_enabled=True,
        adaptation_config={},
        scenario_label="hazard_backup_suppliers",
        timestamp="20260326_000000",
        param_data={},
    )

    assert metadata["Meta_ObservationRadius"] == 4.0
    assert metadata["Meta_EWMAAlpha"] == 0.2
    assert metadata["Meta_MinMoneySurvival"] == 1.0
    assert metadata["Meta_ReservedCapacityShare"] == 0.35
    assert metadata["Meta_ReservedCapacityMarkupCap"] == 0.10
    assert metadata["Meta_AdaptationSensitivityMidpoint"] == 3.0
    assert json.loads(metadata["Meta_ConsumptionRatios"]) == {"retail": 1.0}
    assert json.loads(metadata["Meta_EffectiveAdaptationConfig"]) == {
        "adaptation_sensitivity_max": 4.0,
        "adaptation_sensitivity_min": 2.0,
        "adaptation_strategy": "backup_suppliers",
        "continuity_decay": 0.01,
        "decision_interval": 4,
        "enabled": True,
        "ewma_alpha": 0.2,
        "maintenance_cost_rate": 0.005,
        "max_adaptation_increment": 0.25,
        "max_backup_suppliers": 5,
        "min_money_survival": 1.0,
        "observation_radius": 4.0,
        "replacement_frequency": 10,
        "reserved_capacity_markup_cap": 0.1,
        "reserved_capacity_share": 0.35,
    }


def test_base_metadata_tracks_param_and_cli_provenance() -> None:
    args = SimpleNamespace(
        param_file="params.json",
        topology="topology.json",
        start_year=2000,
        steps_per_year=4,
        steps=400,
        num_households=1000,
        grid_resolution=0.25,
        household_relocation=False,
        no_hazards=True,
        no_adaptation=False,
        adaptation_strategy="capital_hardening",
        adaptation_sensitivity_min=1.0,
        adaptation_sensitivity_max=2.0,
        consumption_ratios={"retail": 1.0},
        save_agent_ensemble=False,
        ensemble_plot_stat="mean",
    )

    metadata = _base_metadata(
        args=args,
        events=[(10, 1, 80, "FL", None)],
        apply_hazards=False,
        adaptation_enabled=True,
        adaptation_config={
            "enabled": True,
            "adaptation_strategy": "capital_hardening",
            "adaptation_sensitivity_min": 1.0,
            "adaptation_sensitivity_max": 2.0,
        },
        scenario_label="baseline_capital_hardening",
        timestamp="20260327_000000",
        param_data={
            "adaptation": {
                "enabled": True,
                "adaptation_strategy": "backup_suppliers",
                "adaptation_sensitivity_min": 2.0,
                "adaptation_sensitivity_max": 4.0,
            },
            "consumption_ratios": {"retail": 1.0},
        },
    )

    assert metadata["Meta_NoHazardsFlag"] is True
    assert metadata["Meta_CLIAdaptationStrategy"] == "capital_hardening"
    assert metadata["Meta_CLIAdaptationSensitivityMin"] == 1.0
    assert metadata["Meta_CLIAdaptationSensitivityMax"] == 2.0
    assert json.loads(metadata["Meta_ParamAdaptationConfig"]) == {
        "adaptation_sensitivity_max": 4.0,
        "adaptation_sensitivity_min": 2.0,
        "adaptation_strategy": "backup_suppliers",
        "enabled": True,
    }
    assert json.loads(metadata["Meta_EffectiveAdaptationConfig"]) == {
        "adaptation_sensitivity_max": 2.0,
        "adaptation_sensitivity_min": 1.0,
        "adaptation_strategy": "capital_hardening",
        "continuity_decay": 0.01,
        "decision_interval": 4,
        "enabled": True,
        "ewma_alpha": 0.2,
        "maintenance_cost_rate": 0.005,
        "max_adaptation_increment": 0.25,
        "max_backup_suppliers": 5,
        "min_money_survival": 1.0,
        "observation_radius": 4.0,
        "replacement_frequency": 10,
        "reserved_capacity_markup_cap": 0.1,
        "reserved_capacity_share": 0.35,
    }
    assert metadata["Meta_RunCommand"]


def test_base_metadata_records_explicit_resource_paths() -> None:
    args = SimpleNamespace(
        param_file="params.json",
        topology="topology.json",
        start_year=2000,
        steps_per_year=4,
        steps=40,
        num_households=1000,
        grid_resolution=0.25,
        household_relocation=False,
        no_hazards=False,
        no_adaptation=False,
        adaptation_strategy=None,
        adaptation_sensitivity_min=None,
        adaptation_sensitivity_max=None,
        consumption_ratios={"retail": 1.0},
        damage_functions_path="/tmp/damage.xlsx",
        land_boundaries_path="/tmp/land",
        save_agent_ensemble=False,
        ensemble_plot_stat="mean",
    )

    metadata = _base_metadata(
        args=args,
        events=[(10, 1, 40, "FL", "hazard.tif")],
        apply_hazards=True,
        adaptation_enabled=False,
        adaptation_config={},
        scenario_label="hazard_only",
        timestamp="20260421_000000",
        param_data={},
    )

    assert metadata["Meta_DamageFunctionsPath"] == "/tmp/damage.xlsx"
    assert metadata["Meta_LandBoundariesPath"] == "/tmp/land"


def test_coerce_shock_inputs_normalizes_new_param_sections() -> None:
    raster_events, node_events, lane_events, route_events = _coerce_shock_inputs(
        legacy_rp_files=["10:1:2:FL:None"],
        raster_hazard_events=[
            {
                "return_period": 25,
                "start_step": 3,
                "end_step": 4,
                "hazard_type": "EQ",
                "path": None,
            }
        ],
        node_shocks=[
            {
                "label": "Node",
                "hazard_type": "CUSTOM_NODE",
                "intensity": 0.6,
                "start_step": 1,
                "end_step": 2,
                "affected_coords": [[10.0, 20.0]],
            }
        ],
        lane_shocks=[
            {
                "label": "Lane",
                "links": [[1, 2], [3, 4]],
                "capacity_fraction": 0.4,
                "start_step": 5,
                "end_step": 6,
            }
        ],
        route_shocks=[
            {
                "label": "Route",
                "route_tag": "TEST_ROUTE",
                "intensity": 0.5,
                "start_step": 7,
                "end_step": 8,
            }
        ],
    )

    assert raster_events == [
        HazardRasterEvent(return_period=10, start_step=1, end_step=2, hazard_type="FL", path=None),
        HazardRasterEvent(return_period=25, start_step=3, end_step=4, hazard_type="EQ", path=None),
    ]
    assert node_events == [
        NodeShock(
            label="Node",
            hazard_type="CUSTOM_NODE",
            intensity=0.6,
            start_step=1,
            end_step=2,
            affected_coords=((10.0, 20.0),),
        )
    ]
    assert lane_events == [
        LaneShock(
            label="Lane #1",
            supplier_id=1,
            buyer_id=2,
            capacity_fraction=0.4,
            start_step=5,
            end_step=6,
        ),
        LaneShock(
            label="Lane #2",
            supplier_id=3,
            buyer_id=4,
            capacity_fraction=0.4,
            start_step=5,
            end_step=6,
        ),
    ]
    assert route_events == [
        RouteShock(
            label="Route",
            route_tag="TEST_ROUTE",
            intensity=0.5,
            start_step=7,
            end_step=8,
        )
    ]
