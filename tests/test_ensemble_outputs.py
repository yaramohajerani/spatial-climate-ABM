import json
from types import SimpleNamespace

import pandas as pd

from plot_from_csv_paper import summarize_members_for_plot
from run_simulation import _base_metadata, _build_ensemble_summary, _resolve_seed_list


def test_resolve_seed_list_deduplicates_explicit_seeds() -> None:
    args = SimpleNamespace(seeds=[7, 3, 7, 5], n_seeds=None, seed_start=None, seed=42)
    assert _resolve_seed_list(args) == [7, 3, 5]


def test_resolve_seed_list_builds_consecutive_range() -> None:
    args = SimpleNamespace(seeds=None, n_seeds=4, seed_start=10, seed=42)
    assert _resolve_seed_list(args) == [10, 11, 12, 13]


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

    runner_summary = _build_ensemble_summary(member_df)
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
        no_learning=False,
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
        no_learning=False,
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
