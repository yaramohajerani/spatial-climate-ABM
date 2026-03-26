"""Sensitivity analysis for hazard-conditional continuity-capital adaptation.

Runs the hazard+adaptation scenario across a range of continuity-sensitivity
settings using matched seeds across all tested values.
The script writes:

- an ensemble-aware time-series plot
- a combined per-step summary CSV
- a combined member-level CSV
- a final-decade summary table

Usage:
    # Full ensemble run (uses seed settings from the parameter file unless overridden)
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json

    # Explicit 10-seed sweep
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --n-seeds 10 --seed-start 41

    # Quick test (50 steps)
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --quick --n-seeds 3 --seed-start 41

    # Re-plot from an existing sensitivity summary CSV
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --from-csv sensitivity_analysis_timeseries.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents import FirmAgent, HouseholdAgent
from ensemble_utils import ENSEMBLE_STAT_ORDER, METADATA_PREFIX, apply_metadata, build_ensemble_summary as summarize_ensemble
from hazard_utils import parse_hazard_event_specs
from model import EconomyModel
from run_simulation import _resolve_seed_list


SENSITIVITY_CONFIGS = {
    "Sensitivity 1.5": (1.0, 2.0),
    "Sensitivity 3.0": (2.0, 4.0),
    "Sensitivity 5.0": (4.0, 6.0),
    "Sensitivity 8.0": (7.0, 9.0),
}

PLOT_METRICS = [
    ("Firm_Production", "Mean Firm Production", "Units of Goods", "firm_mean"),
    ("Firm_Capital", "Mean Firm Capital", "Units of Capital", "firm_mean"),
    ("Firm_Wealth", "Mean Firm Liquidity", "Real Units ($ / Mean Price)", "firm_mean_real"),
    ("Household_Consumption", "Household Consumption", "Units of Goods", "hh_mean"),
    ("Mean_Wage", "Mean Wage", "Real Units ($ / Mean Price)", "real"),
    ("Mean_Price", "Mean Price", "$ / Unit of Goods", "raw"),
]

SUMMARY_METRICS = [
    ("Production", "Firm_Production", "firm_mean"),
    ("Capital", "Firm_Capital", "firm_mean"),
    ("RealLiquidity", "Firm_Wealth", "firm_mean_real"),
    ("Consumption", "Household_Consumption", "hh_mean"),
    ("RealWage", "Mean_Wage", "real"),
    ("Price", "Mean_Price", "raw"),
    ("DirectLoss", "Average_Realized_Direct_Loss", "raw"),
]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for hazard-conditional continuity-capital adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--param-file",
        required=True,
        help="JSON parameter file (same format as run_simulation.py)",
    )
    parser.add_argument(
        "--out",
        default="sensitivity_analysis.png",
        help="Output plot filename",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fewer steps for quick testing (50 steps instead of full)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the base random seed from the parameter file",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Optional explicit list of seeds for an ensemble sensitivity run",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of consecutive seeds to run starting from --seed or --seed-start",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Starting seed for --n-seeds (defaults to --seed)",
    )
    parser.add_argument(
        "--ensemble-stat",
        choices=("mean", "median"),
        default="mean",
        help="Statistic to highlight in the ensemble time-series plot",
    )
    parser.add_argument(
        "--show-ensemble-members",
        action="store_true",
        help="Overlay faint seed-level member lines on the ensemble time-series plot",
    )
    parser.add_argument(
        "--from-csv",
        default=None,
        help="Path to an existing sensitivity summary or member CSV; skips simulation and just plots/summarizes",
    )
    parser.add_argument(
        "--adaptation-strategy",
        choices=("backup_suppliers", "capital_hardening", "stockpiling"),
        default=None,
        help="Override adaptation strategy from the parameter file",
    )
    return parser.parse_args()


def _load_params(param_file: str) -> dict:
    path = Path(param_file)
    if not path.exists():
        raise SystemExit(f"Parameter file not found: {path}")
    return json.loads(path.read_text())


def _parse_events(rp_files: list[str]):
    return parse_hazard_event_specs(rp_files)


def _merge_seed_args(args, params: dict) -> list[int]:
    if args.seed is None:
        args.seed = int(params.get("seed", 42))
    if not args.seeds and "seeds" in params:
        args.seeds = [int(seed) for seed in params["seeds"]]
    if args.n_seeds is None and "n_seeds" in params:
        args.n_seeds = int(params["n_seeds"])
    if args.seed_start is None and "seed_start" in params:
        args.seed_start = int(params["seed_start"])
    return _resolve_seed_list(args)


def _event_signature(events: list[tuple[int, int, int, str, str | None]]) -> str:
    return ";".join(
        f"{rp}:{start}:{end}:{haz_type}:{path if path is not None else 'None'}"
        for rp, start, end, haz_type, path in events
    )


def _sensitivity_metadata(
    *,
    param_file: str,
    params: dict,
    events,
    seed_list: list[int],
    n_steps: int,
) -> dict[str, object]:
    topology_path = str(params.get("topology") or "")
    adaptation_config = params.get("adaptation", params.get("learning", {}))
    return {
        f"{METADATA_PREFIX}SensitivityParameter": "adaptation_sensitivity",
        f"{METADATA_PREFIX}AdaptationStrategy": str(adaptation_config.get("adaptation_strategy", "backup_suppliers")),
        f"{METADATA_PREFIX}ParamFile": str(param_file),
        f"{METADATA_PREFIX}ParamFileStem": Path(param_file).stem,
        f"{METADATA_PREFIX}TopologyPath": topology_path,
        f"{METADATA_PREFIX}TopologyStem": Path(topology_path).stem if topology_path else "",
        f"{METADATA_PREFIX}HazardEventCount": len(events),
        f"{METADATA_PREFIX}HazardEvents": _event_signature(events),
        f"{METADATA_PREFIX}StartYear": int(params.get("start_year", 0)),
        f"{METADATA_PREFIX}StepsPerYear": int(params.get("steps_per_year", 4)),
        f"{METADATA_PREFIX}StepsRequested": int(n_steps),
        f"{METADATA_PREFIX}NumHouseholds": int(params.get("num_households", 100)),
        f"{METADATA_PREFIX}GridResolution": float(params.get("grid_resolution", 1.0)),
        f"{METADATA_PREFIX}HouseholdRelocation": bool(params.get("household_relocation", True)),
        f"{METADATA_PREFIX}HHConsumptionPropensityIncome": float(HouseholdAgent.CONSUMPTION_PROPENSITY_INCOME),
        f"{METADATA_PREFIX}HHConsumptionPropensityWealth": float(HouseholdAgent.CONSUMPTION_PROPENSITY_WEALTH),
        f"{METADATA_PREFIX}HHTargetCashBuffer": float(HouseholdAgent.TARGET_CASH_BUFFER),
        f"{METADATA_PREFIX}FirmInventoryBufferRatio": float(FirmAgent.INVENTORY_BUFFER_RATIO),
        f"{METADATA_PREFIX}FirmLiquidityBufferRatio": float(FirmAgent.LIQUIDITY_BUFFER_RATIO),
        f"{METADATA_PREFIX}FirmMinLiquidityBuffer": float(FirmAgent.MIN_LIQUIDITY_BUFFER),
        f"{METADATA_PREFIX}FirmLaborShare": float(FirmAgent.LABOR_SHARE),
        f"{METADATA_PREFIX}NoWorkerWagePremium": float(FirmAgent.NO_WORKER_WAGE_PREMIUM),
        f"{METADATA_PREFIX}DecisionInterval": int(adaptation_config.get("decision_interval", 4)),
        f"{METADATA_PREFIX}ObservationRadius": float(adaptation_config.get("observation_radius", 4)),
        f"{METADATA_PREFIX}ResilienceDecay": float(adaptation_config.get("resilience_decay", 0.01)),
        f"{METADATA_PREFIX}ContinuityDecay": float(adaptation_config.get("continuity_decay", adaptation_config.get("resilience_decay", 0.01))),
        f"{METADATA_PREFIX}MaintenanceCostRate": float(adaptation_config.get("maintenance_cost_rate", 0.005)),
        f"{METADATA_PREFIX}MaxBackupSuppliers": int(adaptation_config.get("max_backup_suppliers", 5)),
        f"{METADATA_PREFIX}MaxAdaptIncrement": float(adaptation_config.get("max_adaptation_increment", 0.25)),
        f"{METADATA_PREFIX}SeedCount": len(seed_list),
        f"{METADATA_PREFIX}SeedList": ",".join(str(seed) for seed in seed_list),
        f"{METADATA_PREFIX}SeedMin": min(seed_list) if seed_list else "",
        f"{METADATA_PREFIX}SeedMax": max(seed_list) if seed_list else "",
    }


def _finalize_results(
    df: pd.DataFrame,
    *,
    label: str,
    sensitivity_min: float,
    sensitivity_max: float,
    seed: int,
    start_year: int,
    steps_per_year: int,
) -> pd.DataFrame:
    df = df.copy()
    df["Scenario"] = "Hazard + Adaptation"
    df["Sensitivity_Label"] = label
    df["Adaptation_Sensitivity_Min"] = float(sensitivity_min)
    df["Adaptation_Sensitivity_Max"] = float(sensitivity_max)
    df["Adaptation_Sensitivity_Midpoint"] = 0.5 * (float(sensitivity_min) + float(sensitivity_max))
    if "Step" not in df.columns:
        df["Step"] = df.index
    if start_year and "Year" not in df.columns:
        df["Year"] = start_year + df["Step"].astype(float) / steps_per_year
    df["Seed"] = int(seed)
    return df


def run_scenario(
    params: dict,
    events: list,
    label: str,
    sensitivity_min: float,
    sensitivity_max: float,
    n_steps: int,
    seed: int,
    adaptation_strategy: str | None = None,
) -> pd.DataFrame:
    adaptation_config = params.get("adaptation", params.get("learning", {}))
    adaptation_config = {
        **adaptation_config,
        "enabled": True,
        "adaptation_sensitivity_min": float(sensitivity_min),
        "adaptation_sensitivity_max": float(sensitivity_max),
    }
    if adaptation_strategy:
        adaptation_config["adaptation_strategy"] = adaptation_strategy

    model = EconomyModel(
        num_households=int(params.get("num_households", 100)),
        num_firms=20,
        hazard_events=events,
        seed=seed,
        apply_hazard_impacts=True,
        firm_topology_path=params.get("topology"),
        start_year=int(params.get("start_year", 0)),
        steps_per_year=int(params.get("steps_per_year", 4)),
        adaptation_params=adaptation_config,
        consumption_ratios=params.get("consumption_ratios"),
        grid_resolution=float(params.get("grid_resolution", 1.0)),
        household_relocation=bool(params.get("household_relocation", True)),
    )

    for _ in range(n_steps):
        model.step()

    return _finalize_results(
        model.results_to_dataframe(),
        label=label,
        sensitivity_min=sensitivity_min,
        sensitivity_max=sensitivity_max,
        seed=seed,
        start_year=int(params.get("start_year", 0)),
        steps_per_year=int(params.get("steps_per_year", 4)),
    )


def build_ensemble_summary(member_df: pd.DataFrame) -> pd.DataFrame:
    return summarize_ensemble(
        member_df,
        group_cols=[
            "Scenario",
            "Sensitivity_Label",
            "Adaptation_Sensitivity_Min",
            "Adaptation_Sensitivity_Max",
            "Adaptation_Sensitivity_Midpoint",
            "Step",
            "Year",
        ],
    )


def _sensitivity_output_paths(out_path: str) -> tuple[Path, Path, Path]:
    base = Path(out_path)
    summary_timeseries = base.with_name(f"{base.stem}_timeseries.csv")
    member_timeseries = base.with_name(f"{base.stem}_timeseries_members.csv")
    summary_table = base.with_suffix(".csv")
    return summary_timeseries, member_timeseries, summary_table


def save_timeseries(summary_df: pd.DataFrame, member_df: pd.DataFrame, out_path: str) -> tuple[Path, Path]:
    summary_path, members_path, _ = _sensitivity_output_paths(out_path)
    summary_df.to_csv(summary_path, index=False)
    member_df.to_csv(members_path, index=False)
    print(f"Summary timeseries saved to {summary_path}")
    print(f"Member timeseries saved to {members_path}")
    return summary_path, members_path


def load_timeseries(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    if "EnsembleStatistic" in df.columns:
        summary_df = df
        member_df = None
        if not path.stem.endswith("_members"):
            members_path = path.parent / f"{path.stem}_members.csv"
            if members_path.exists():
                member_df = pd.read_csv(members_path)
        return summary_df, member_df

    member_df = df
    summary_df = build_ensemble_summary(member_df)
    return summary_df, member_df


def _ordered_labels(frame: pd.DataFrame) -> list[str]:
    labels = list(frame["Sensitivity_Label"].dropna().unique())
    configured = [label for label in SENSITIVITY_CONFIGS if label in labels]
    extras = [label for label in labels if label not in configured]
    return configured + extras


def _normalize_series(df: pd.DataFrame, column: str, mode: str, *, num_households: int) -> pd.Series:
    values = df[column].astype(float)
    n_firms = df["Total_Firms"].replace(0, np.nan).astype(float) if "Total_Firms" in df.columns else pd.Series(1.0, index=df.index)
    price = df["Mean_Price"].replace(0, np.nan).astype(float) if "Mean_Price" in df.columns else pd.Series(1.0, index=df.index)

    if mode == "firm_mean":
        return values / n_firms
    if mode == "firm_mean_real":
        return (values / n_firms) / price
    if mode == "hh_mean":
        return values / float(num_households)
    if mode == "real":
        return values / price
    return values


def plot_sensitivity(
    summary_df: pd.DataFrame,
    out_path: str,
    num_households: int,
    *,
    ensemble_stat: str,
    member_df: pd.DataFrame | None = None,
    show_ensemble_members: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    labels = _ordered_labels(summary_df)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    is_ensemble = "EnsembleStatistic" in summary_df.columns and summary_df["EnsembleSize"].max() > 1

    for idx, (column, title, ylabel, norm_mode) in enumerate(PLOT_METRICS):
        ax = axes[idx]
        for label, color in zip(labels, colors):
            stat_mask = summary_df["Sensitivity_Label"] == label
            stat_df = summary_df.loc[stat_mask & (summary_df["EnsembleStatistic"] == ensemble_stat)].sort_values("Step")
            if stat_df.empty or column not in stat_df.columns:
                continue

            x_col = "Year" if "Year" in stat_df.columns else "Step"
            x_vals = stat_df[x_col]

            if show_ensemble_members and member_df is not None:
                member_mask = member_df["Sensitivity_Label"] == label
                for _, seed_grp in member_df.loc[member_mask].groupby("Seed"):
                    seed_grp = seed_grp.sort_values("Step")
                    if column not in seed_grp.columns:
                        continue
                    ax.plot(
                        seed_grp["Year"] if "Year" in seed_grp.columns else seed_grp["Step"],
                        _normalize_series(seed_grp, column, norm_mode, num_households=num_households),
                        color=color,
                        alpha=0.10,
                        linewidth=0.8,
                    )

            p10_df = summary_df.loc[stat_mask & (summary_df["EnsembleStatistic"] == "p10")].sort_values("Step")
            p90_df = summary_df.loc[stat_mask & (summary_df["EnsembleStatistic"] == "p90")].sort_values("Step")
            if is_ensemble and not p10_df.empty and not p90_df.empty and column in p10_df.columns and column in p90_df.columns:
                ax.fill_between(
                    x_vals.to_numpy(dtype=float),
                    _normalize_series(p10_df, column, norm_mode, num_households=num_households).to_numpy(dtype=float),
                    _normalize_series(p90_df, column, norm_mode, num_households=num_households).to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                )

            ax.plot(
                x_vals,
                _normalize_series(stat_df, column, norm_mode, num_households=num_households),
                label=label,
                color=color,
                linewidth=1.8,
                alpha=0.95,
            )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Year" if "Year" in summary_df.columns else "Step", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    strategy_label = ""
    if "Meta_AdaptationStrategy" in summary_df.columns:
        strategy_val = summary_df["Meta_AdaptationStrategy"].dropna()
        if not strategy_val.empty:
            strategy_label = f" [{str(strategy_val.iloc[0])}]"

    if is_ensemble:
        ensemble_size = int(summary_df["EnsembleSize"].max())
        stat_label = ensemble_stat.title()
        fig.suptitle(
            f"Sensitivity of Outcomes to Adaptation Sensitivity{strategy_label}\n(Hazard + Adaptation scenario; {ensemble_size}-seed ensemble {stat_label})",
            fontsize=13,
            y=1.02,
        )
    else:
        fig.suptitle(
            f"Sensitivity of Outcomes to Adaptation Sensitivity{strategy_label}\n(Hazard + Adaptation scenario; single seed)",
            fontsize=13,
            y=1.02,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sensitivity plot saved to {out_path}")


def _make_summary_table_from_members(
    member_df: pd.DataFrame,
    *,
    steps_per_year: int,
    num_households: int,
) -> pd.DataFrame:
    rows = []
    for label in _ordered_labels(member_df):
        label_df = member_df[member_df["Sensitivity_Label"] == label]
        per_seed_rows = []
        for seed, seed_df in label_df.groupby("Seed"):
            last_decade = seed_df.sort_values("Step").tail(steps_per_year * 10)
            row = {"Seed": int(seed)}
            for metric_name, column, mode in SUMMARY_METRICS:
                if column not in last_decade.columns:
                    continue
                values = _normalize_series(last_decade, column, mode, num_households=num_households)
                row[metric_name] = float(values.mean())
            per_seed_rows.append(row)

        seed_summary = pd.DataFrame(per_seed_rows)
        if seed_summary.empty:
            continue
        row = {
            "Sensitivity Label": label,
            "Adaptation_Sensitivity_Min": float(label_df["Adaptation_Sensitivity_Min"].iloc[0]),
            "Adaptation_Sensitivity_Max": float(label_df["Adaptation_Sensitivity_Max"].iloc[0]),
            "Adaptation_Sensitivity_Midpoint": float(label_df["Adaptation_Sensitivity_Midpoint"].iloc[0]),
            "EnsembleSize": int(seed_summary["Seed"].nunique()),
        }
        for metric_name, _, _ in SUMMARY_METRICS:
            if metric_name not in seed_summary.columns:
                continue
            values = seed_summary[metric_name]
            row[f"{metric_name}_Mean"] = float(values.mean())
            row[f"{metric_name}_Median"] = float(values.median())
            row[f"{metric_name}_Std"] = float(values.std(ddof=0))
            row[f"{metric_name}_P10"] = float(values.quantile(0.10))
            row[f"{metric_name}_P90"] = float(values.quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def _make_summary_table_from_summary(
    summary_df: pd.DataFrame,
    *,
    steps_per_year: int,
    num_households: int,
    ensemble_stat: str,
) -> pd.DataFrame:
    rows = []
    for label in _ordered_labels(summary_df):
        label_rows = {"Sensitivity Label": label}
        label_slice = summary_df[summary_df["Sensitivity_Label"] == label]
        if "Adaptation_Sensitivity_Midpoint" in label_slice.columns:
            label_rows["Adaptation_Sensitivity_Midpoint"] = float(label_slice["Adaptation_Sensitivity_Midpoint"].iloc[0])
        if "Adaptation_Sensitivity_Min" in label_slice.columns:
            label_rows["Adaptation_Sensitivity_Min"] = float(label_slice["Adaptation_Sensitivity_Min"].iloc[0])
        if "Adaptation_Sensitivity_Max" in label_slice.columns:
            label_rows["Adaptation_Sensitivity_Max"] = float(label_slice["Adaptation_Sensitivity_Max"].iloc[0])
        if "EnsembleSize" in label_slice.columns:
            label_rows["EnsembleSize"] = int(label_slice["EnsembleSize"].max())

        for metric_name, column, mode in SUMMARY_METRICS:
            for stat_name in [ensemble_stat, "median", "std", "p10", "p90"]:
                stat_slice = label_slice[label_slice["EnsembleStatistic"] == stat_name].sort_values("Step").tail(steps_per_year * 10)
                if stat_slice.empty or column not in stat_slice.columns:
                    continue
                values = _normalize_series(stat_slice, column, mode, num_households=num_households)
                suffix = stat_name.title() if stat_name != ensemble_stat else "Mean"
                label_rows[f"{metric_name}_{suffix}"] = float(values.mean())
        rows.append(label_rows)
    return pd.DataFrame(rows)


def make_summary_table(
    summary_df: pd.DataFrame,
    *,
    steps_per_year: int,
    num_households: int,
    ensemble_stat: str,
    member_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if member_df is not None and not member_df.empty:
        return _make_summary_table_from_members(
            member_df,
            steps_per_year=steps_per_year,
            num_households=num_households,
        )
    return _make_summary_table_from_summary(
        summary_df,
        steps_per_year=steps_per_year,
        num_households=num_households,
        ensemble_stat=ensemble_stat,
    )


def _print_summary_table(summary: pd.DataFrame) -> None:
    if summary.empty:
        print("No summary rows available.")
        return
    display_cols = [
        "Sensitivity Label",
        "Adaptation_Sensitivity_Midpoint",
        "EnsembleSize",
    ]
    for metric_name, _, _ in SUMMARY_METRICS:
        mean_col = f"{metric_name}_Mean"
        p10_col = f"{metric_name}_P10"
        p90_col = f"{metric_name}_P90"
        if all(col in summary.columns for col in [mean_col, p10_col, p90_col]):
            display_cols.extend([mean_col, p10_col, p90_col])
    display_df = summary[display_cols].copy()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(display_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    args = _parse_args()
    params = _load_params(args.param_file)
    steps_per_year = int(params.get("steps_per_year", 4))
    num_households = int(params.get("num_households", 100))
    seed_list = _merge_seed_args(args, params)

    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            raise SystemExit(f"Timeseries CSV not found: {csv_path}")
        print(f"Loading sensitivity data from {csv_path}")
        summary_df, member_df = load_timeseries(str(csv_path))
    else:
        events = _parse_events(params.get("rp_files", []))
        n_steps = 50 if args.quick else int(params.get("steps", 300))
        metadata = _sensitivity_metadata(
            param_file=args.param_file,
            params=params,
            events=events,
            seed_list=seed_list,
            n_steps=n_steps,
        )

        member_frames = []
        for label, (sensitivity_min, sensitivity_max) in SENSITIVITY_CONFIGS.items():
            print(f"\n{'=' * 72}")
            print(
                "Running sensitivity sweep: "
                f"{label}  sensitivity=[{sensitivity_min}, {sensitivity_max}]  seeds={seed_list}"
            )
            print(f"{'=' * 72}")
            for seed in seed_list:
                seed_df = run_scenario(
                    params,
                    events,
                    label,
                    sensitivity_min,
                    sensitivity_max,
                    n_steps,
                    seed,
                    adaptation_strategy=args.adaptation_strategy,
                )
                seed_df = apply_metadata(
                    seed_df,
                    {
                        **metadata,
                        f"{METADATA_PREFIX}AdaptationSensitivityMin": float(sensitivity_min),
                        f"{METADATA_PREFIX}AdaptationSensitivityMax": float(sensitivity_max),
                    },
                )
                member_frames.append(seed_df)

        member_df = pd.concat(member_frames, ignore_index=True)
        summary_df = build_ensemble_summary(member_df)
        summary_df = apply_metadata(summary_df, metadata)
        save_timeseries(summary_df, member_df, args.out)

    plot_sensitivity(
        summary_df,
        args.out,
        num_households,
        ensemble_stat=args.ensemble_stat,
        member_df=member_df,
        show_ensemble_members=args.show_ensemble_members,
    )

    summary = make_summary_table(
        summary_df,
        steps_per_year=steps_per_year,
        num_households=num_households,
        ensemble_stat=args.ensemble_stat,
        member_df=member_df,
    )
    print(f"\n{'=' * 96}")
    print("SENSITIVITY ANALYSIS SUMMARY (final-decade seed-level ensemble statistics)")
    print(f"{'=' * 96}")
    _print_summary_table(summary)

    _, _, table_path = _sensitivity_output_paths(args.out)
    summary.to_csv(table_path, index=False)
    print(f"\nSummary table saved to {table_path}")


if __name__ == "__main__":
    main()
