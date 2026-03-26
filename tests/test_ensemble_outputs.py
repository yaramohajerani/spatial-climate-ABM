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
    )

    metadata = _base_metadata(
        args=args,
        events=[],
        apply_hazards=True,
        adaptation_enabled=True,
        adaptation_config={},
        scenario_label="hazard_backup_suppliers",
        timestamp="20260326_000000",
    )

    assert metadata["Meta_ObservationRadius"] == 4.0
    assert metadata["Meta_MinMoneySurvival"] == 1.0
