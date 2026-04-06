from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model import EconomyModel
from hazard_utils import parse_hazard_event_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short simulation and verify stock-flow consistency.",
    )
    parser.add_argument("--param-file", type=str, required=True, help="JSON parameter file")
    parser.add_argument("--steps", type=int, default=None, help="Override step count from parameter file")
    parser.add_argument("--no-hazards", action="store_true", help="Disable hazard impacts")
    parser.add_argument("--no-adaptation", action="store_true", help="Disable hazard-conditional adaptation/reorganization")
    parser.add_argument("--no-learning", action="store_true", help="Deprecated alias for --no-adaptation")
    parser.add_argument("--tail-window", type=int, default=20, help="Window for tail activity summaries")
    parser.add_argument(
        "--money-tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed absolute drift in total money",
    )
    parser.add_argument(
        "--income-tolerance",
        type=float,
        default=1e-6,
        help="Maximum allowed mismatch between firm-side and household-side income flows",
    )
    parser.add_argument("--min-tail-production", type=float, default=None)
    parser.add_argument("--min-tail-consumption", type=float, default=None)
    parser.add_argument("--min-tail-labor", type=float, default=None)
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=1e-9,
        help="Production threshold below which a firm is treated as dormant",
    )
    parser.add_argument(
        "--min-tail-active-firms",
        type=float,
        default=None,
        help="Optional minimum mean number of active firms over the tail window",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> tuple[EconomyModel, int]:
    params = json.loads(Path(args.param_file).read_text())
    adaptation_config = dict(params.get("adaptation", params.get("learning", {})))
    if args.no_adaptation or args.no_learning:
        adaptation_config["enabled"] = False

    steps = int(args.steps if args.steps is not None else params.get("steps", 50))
    model = EconomyModel(
        num_households=int(params.get("num_households", 100)),
        hazard_events=[] if args.no_hazards else parse_hazard_event_specs(params.get("rp_files", [])),
        seed=params.get("seed"),
        start_year=int(params.get("start_year", 0)),
        steps_per_year=int(params.get("steps_per_year", 4)),
        firm_topology_path=params.get("topology"),
        apply_hazard_impacts=not args.no_hazards,
        adaptation_params=adaptation_config,
        consumption_ratios=params.get("consumption_ratios"),
        grid_resolution=float(params.get("grid_resolution", 1.0)),
        household_relocation=bool(params.get("household_relocation", False)),
    )
    return model, steps


def main() -> int:
    args = parse_args()
    model, steps = build_model(args)

    for _ in range(steps):
        model.step()

    df = model.results_to_dataframe()
    tail = df.tail(min(args.tail_window, len(df)))
    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")
    firm_agent_df = agent_df[agent_df["type"] == "FirmAgent"].copy()
    firm_agent_df["is_active"] = firm_agent_df["production"] > args.activity_threshold

    active_by_step = (
        firm_agent_df.groupby("Step")["is_active"].sum().astype(float)
        if not firm_agent_df.empty else np.array([])
    )
    total_firms = len(model._firms)
    final_active_firms = int(active_by_step.iloc[-1]) if len(active_by_step) else 0
    final_dormant_firms = int(total_firms - final_active_firms)
    tail_active_firms = float(active_by_step.tail(min(args.tail_window, len(active_by_step))).mean()) if len(active_by_step) else 0.0
    tail_dormant_firms = float(total_firms - tail_active_firms)

    max_abs_money_drift = float(df["Money_Drift"].abs().max())
    max_dividend_mismatch = float(
        np.abs(df["Household_Dividend_Income"] - df["Firm_Dividends_Paid"]).max()
    )
    max_capital_income_mismatch = float(
        np.abs(df["Household_Capital_Income"] - df["Firm_Investment_Spending"]).max()
    )
    max_adaptation_income_mismatch = float(
        np.abs(df["Household_Adaptation_Income"] - df["Firm_Adaptation_Spending"]).max()
    )

    print("Consistency summary")
    print(f"  steps: {steps}")
    print(f"  hazards_enabled: {not args.no_hazards}")
    print(f"  adaptation_enabled: {bool(model.firm_adaptation_enabled)}")
    print(f"  household_relocation_enabled: {bool(model.household_relocation_enabled)}")
    print(f"  initial_total_money: {model.initial_total_money:.12f}")
    print(f"  final_total_money: {model.total_money():.12f}")
    print(f"  max_abs_money_drift: {max_abs_money_drift:.12g}")
    print(f"  max_dividend_mismatch: {max_dividend_mismatch:.12g}")
    print(f"  max_capital_income_mismatch: {max_capital_income_mismatch:.12g}")
    print(f"  max_adaptation_income_mismatch: {max_adaptation_income_mismatch:.12g}")
    if "Firm_Replacements" in df.columns:
        print(f"  final_replacements: {int(df['Firm_Replacements'].iloc[-1])}")
    if "Flooded_Firms" in df.columns:
        print(f"  max_flooded_firms: {int(df['Flooded_Firms'].max())}")
    if "Flooded_Households" in df.columns:
        print(f"  max_flooded_households: {int(df['Flooded_Households'].max())}")
    print(f"  total_firms: {total_firms}")
    print(f"  activity_threshold: {args.activity_threshold:.12g}")
    print(f"  final_active_firms: {final_active_firms}")
    print(f"  final_dormant_firms: {final_dormant_firms}")
    print(f"  tail_mean_active_firms: {tail_active_firms:.6f}")
    print(f"  tail_mean_dormant_firms: {tail_dormant_firms:.6f}")
    print(f"  tail_mean_production: {float(tail['Firm_Production'].mean()):.6f}")
    print(f"  tail_mean_consumption: {float(tail['Household_Consumption'].mean()):.6f}")
    print(f"  tail_mean_labor: {float(tail['Household_Labor_Sold'].mean()):.6f}")
    print(f"  tail_mean_profit: {float(tail['Firm_Profits'].mean()):.6f}")

    failures: list[str] = []
    if max_abs_money_drift > args.money_tolerance:
        failures.append(
            f"max_abs_money_drift {max_abs_money_drift:.12g} exceeds tolerance {args.money_tolerance:.12g}"
        )
    if max_dividend_mismatch > args.income_tolerance:
        failures.append(
            f"dividend mismatch {max_dividend_mismatch:.12g} exceeds tolerance {args.income_tolerance:.12g}"
        )
    if max_capital_income_mismatch > args.income_tolerance:
        failures.append(
            f"capital-income mismatch {max_capital_income_mismatch:.12g} exceeds tolerance {args.income_tolerance:.12g}"
        )
    if max_adaptation_income_mismatch > args.income_tolerance:
        failures.append(
            f"adaptation-income mismatch {max_adaptation_income_mismatch:.12g} exceeds tolerance {args.income_tolerance:.12g}"
        )
    if args.min_tail_production is not None and float(tail["Firm_Production"].mean()) < args.min_tail_production:
        failures.append(
            f"tail_mean_production {float(tail['Firm_Production'].mean()):.6f} is below {args.min_tail_production:.6f}"
        )
    if args.min_tail_consumption is not None and float(tail["Household_Consumption"].mean()) < args.min_tail_consumption:
        failures.append(
            f"tail_mean_consumption {float(tail['Household_Consumption'].mean()):.6f} is below {args.min_tail_consumption:.6f}"
        )
    if args.min_tail_labor is not None and float(tail["Household_Labor_Sold"].mean()) < args.min_tail_labor:
        failures.append(
            f"tail_mean_labor {float(tail['Household_Labor_Sold'].mean()):.6f} is below {args.min_tail_labor:.6f}"
        )
    if args.min_tail_active_firms is not None and tail_active_firms < args.min_tail_active_firms:
        failures.append(
            f"tail_mean_active_firms {tail_active_firms:.6f} is below {args.min_tail_active_firms:.6f}"
        )

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
