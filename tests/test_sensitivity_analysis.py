import pandas as pd
import pytest

from sensitivity_analysis import build_ensemble_summary, make_summary_table


def _member_frame() -> pd.DataFrame:
    rows = []
    for label, midpoint, scale in [
        ("Sensitivity 3.0", 3.0, 1.0),
        ("Sensitivity 5.0", 5.0, 0.8),
    ]:
        for seed, offset in [(41, 0.0), (42, 2.0)]:
            for step, year in [(0, 2030.0), (1, 2030.25)]:
                rows.append(
                    {
                        "Scenario": "Hazard + Adaptation",
                        "Sensitivity_Label": label,
                        "Adaptation_Sensitivity_Min": midpoint - 1.0,
                        "Adaptation_Sensitivity_Max": midpoint + 1.0,
                        "Adaptation_Sensitivity_Midpoint": midpoint,
                        "Step": step,
                        "Year": year,
                        "Seed": seed,
                        "Total_Firms": 2.0,
                        "Firm_Production": (10.0 + step + offset) * scale,
                        "Firm_Capital": (20.0 + step + offset) * scale,
                        "Firm_Wealth": (40.0 + step + offset) * scale,
                        "Household_Consumption": (8.0 + step + offset) * scale,
                        "Mean_Wage": (2.0 + 0.1 * step) * scale,
                        "Mean_Price": 2.0,
                        "Average_Realized_Direct_Loss": 0.05 * scale,
                        "Meta_ParamFile": "params.json",
                    }
                )
    return pd.DataFrame(rows)


def test_build_ensemble_summary_groups_by_sensitivity_and_ignores_metadata() -> None:
    member_df = _member_frame()
    summary_df = build_ensemble_summary(member_df)

    mean_step0 = summary_df[
        (summary_df["Sensitivity_Label"] == "Sensitivity 3.0")
        & (summary_df["Step"] == 0)
        & (summary_df["EnsembleStatistic"] == "mean")
    ].iloc[0]

    assert mean_step0["EnsembleSize"] == 2
    assert mean_step0["Adaptation_Sensitivity_Midpoint"] == 3.0
    assert mean_step0["Firm_Production"] == pytest.approx((10.0 + 12.0) / 2.0)
    assert "Meta_ParamFile" not in summary_df.columns


def test_make_summary_table_uses_seed_level_final_decade_ensemble_statistics() -> None:
    member_df = _member_frame()
    summary_df = build_ensemble_summary(member_df)
    table = make_summary_table(
        summary_df,
        steps_per_year=1,
        num_households=10,
        ensemble_stat="mean",
        member_df=member_df,
    )

    row_300 = table[table["Sensitivity Label"] == "Sensitivity 3.0"].iloc[0]
    row_500 = table[table["Sensitivity Label"] == "Sensitivity 5.0"].iloc[0]

    assert row_300["EnsembleSize"] == 2
    assert row_300["Production_Mean"] > row_500["Production_Mean"]
    assert row_300["RealLiquidity_Mean"] == pytest.approx((10.125 + 10.625) / 2.0)
    assert row_300["DirectLoss_Mean"] == pytest.approx(0.05)
