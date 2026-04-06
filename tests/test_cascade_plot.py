import sys

import pandas as pd

from plot_cascade_risk import main


def _summary_frame(scenario: str) -> pd.DataFrame:
    rows = []
    for step, year in [(0, 2025.0), (1, 2025.25)]:
        base = 0.10 + 0.02 * step
        for stat, offset in [("mean", 0.0), ("p10", -0.02), ("p90", 0.02)]:
            rows.append(
                {
                    "Scenario": scenario,
                    "Step": step,
                    "Year": year,
                    "EnsembleStatistic": stat,
                    "EnsembleSize": 2,
                    "Ever_Directly_Hit_Firm_Share": base + offset,
                    "Never_Hit_Currently_Disrupted_Firm_Share": base + 0.05 + offset,
                    "Never_Hit_Supplier_Disruption_Burden_Share": base + 0.10 + offset,
                    "Never_Hit_Production_Share": 0.80 + offset,
                }
            )
    return pd.DataFrame(rows)


def _member_frame(scenario: str) -> pd.DataFrame:
    rows = []
    for seed, seed_offset in [(41, -0.01), (42, 0.01)]:
        for step, year in [(0, 2025.0), (1, 2025.25)]:
            base = 0.10 + 0.02 * step + seed_offset
            rows.append(
                {
                    "Scenario": scenario,
                    "Seed": seed,
                    "Step": step,
                    "Year": year,
                    "Ever_Directly_Hit_Firm_Share": base,
                    "Never_Hit_Currently_Disrupted_Firm_Share": base + 0.05,
                    "Never_Hit_Supplier_Disruption_Burden_Share": base + 0.10,
                    "Never_Hit_Production_Share": 0.80 + seed_offset,
                }
            )
    return pd.DataFrame(rows)


def test_plot_cascade_risk_smoke(tmp_path, monkeypatch) -> None:
    hazard_a = tmp_path / "simulation_hazard_adaptation_test.csv"
    hazard_na = tmp_path / "simulation_hazard_noadaptation_test.csv"
    _summary_frame("Hazard + Adaptation").to_csv(hazard_a, index=False)
    _summary_frame("Hazard + No Adaptation").to_csv(hazard_na, index=False)
    _member_frame("Hazard + Adaptation").to_csv(
        tmp_path / "simulation_hazard_adaptation_test_members.csv",
        index=False,
    )
    _member_frame("Hazard + No Adaptation").to_csv(
        tmp_path / "simulation_hazard_noadaptation_test_members.csv",
        index=False,
    )

    out_path = tmp_path / "cascade_risk.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_cascade_risk.py",
            "--csv-files",
            str(hazard_a),
            str(hazard_na),
            "--show-ensemble-band",
            "--show-ensemble-members",
            "--out",
            str(out_path),
        ],
    )

    main()

    assert out_path.exists()
