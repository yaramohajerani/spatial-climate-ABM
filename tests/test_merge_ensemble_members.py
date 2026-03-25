import pandas as pd
import pytest

from merge_ensemble_members import derive_output_prefix, merge_member_dataframes


def _member_frame(seed_values, scenario="Hazard + Adaptation", *, ucb_c=1.0):
    rows = []
    for seed in seed_values:
        rows.extend(
            [
                {
                    "Scenario": scenario,
                    "Step": 0,
                    "Year": 2030.0,
                    "Seed": seed,
                    "Meta_ApplyHazards": scenario.startswith("Hazard"),
                    "Meta_AdaptationEnabled": "Adaptation" in scenario and "No Adaptation" not in scenario,
                    "Meta_UCB_C": ucb_c,
                    "Meta_ParamFile": "params.json",
                    "Meta_StepsPerYear": 4,
                    "Firm_Production": 100.0 + seed,
                    "Household_Consumption": 50.0 + seed,
                    "Mean_Price": 1.0 + seed / 100.0,
                },
                {
                    "Scenario": scenario,
                    "Step": 1,
                    "Year": 2030.25,
                    "Seed": seed,
                    "Meta_ApplyHazards": scenario.startswith("Hazard"),
                    "Meta_AdaptationEnabled": "Adaptation" in scenario and "No Adaptation" not in scenario,
                    "Meta_UCB_C": ucb_c,
                    "Meta_ParamFile": "params.json",
                    "Meta_StepsPerYear": 4,
                    "Firm_Production": 110.0 + seed,
                    "Household_Consumption": 55.0 + seed,
                    "Mean_Price": 1.1 + seed / 100.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_merge_member_dataframes_combines_non_overlapping_seed_batches() -> None:
    first = _member_frame([41, 42, 43])
    second = _member_frame([44, 45])

    merged_df, summary_df = merge_member_dataframes([first, second], labels=["first", "second"])

    assert merged_df["Seed"].nunique() == 5
    assert list(sorted(merged_df["Seed"].unique())) == [41, 42, 43, 44, 45]

    mean_step0 = summary_df[
        (summary_df["Step"] == 0) & (summary_df["EnsembleStatistic"] == "mean")
    ].iloc[0]
    assert mean_step0["EnsembleSize"] == 5
    assert mean_step0["Firm_Production"] == pytest.approx((141 + 142 + 143 + 144 + 145) / 5.0)
    assert mean_step0["Meta_SeedCount"] == 5
    assert mean_step0["Meta_SeedList"] == "41,42,43,44,45"
    assert merged_df["Meta_UCB_C"].iloc[0] == 1.0


def test_merge_member_dataframes_rejects_duplicate_seed_batches() -> None:
    first = _member_frame([41, 42, 43])
    second = _member_frame([43, 44, 45])

    with pytest.raises(ValueError, match="duplicate seeds"):
        merge_member_dataframes([first, second], labels=["first", "second"])


def test_merge_member_dataframes_rejects_mixed_scenarios() -> None:
    first = _member_frame([41, 42], scenario="Hazard + Adaptation")
    second = _member_frame([43, 44], scenario="Hazard + No Adaptation")

    with pytest.raises(ValueError, match="Scenario"):
        merge_member_dataframes([first, second], labels=["first", "second"])


def test_merge_member_dataframes_rejects_mismatched_metadata() -> None:
    first = _member_frame([41, 42], ucb_c=1.0)
    second = _member_frame([43, 44], ucb_c=0.5)

    with pytest.raises(ValueError, match="metadata"):
        merge_member_dataframes([first, second], labels=["first", "second"])


def test_derive_output_prefix_replaces_old_ensemble_stamp(tmp_path) -> None:
    first_path = tmp_path / "simulation_hazard_adaptation_example_ensemble5_20260324_165042_members.csv"
    prefix = derive_output_prefix([first_path], total_seeds=10)
    assert prefix.name.startswith("simulation_hazard_adaptation_example_ensemble10_")
    assert not prefix.name.endswith("_members")
