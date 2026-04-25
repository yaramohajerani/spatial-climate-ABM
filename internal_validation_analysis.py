from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hazard_utils import parse_hazard_event_specs
from model import EconomyModel


ROOT = Path(__file__).resolve().parent
MANUSCRIPT_DIR = ROOT / "manuscript"
PARAM_FILE = ROOT / "aqueduct_riverine_parameters_rcp8p5.json"

WARMUP_MEMBERS_GLOB = "simulation_baseline_noadaptation_*_members.csv"
WARMUP_END_STEP = 79
WARMUP_TARGET_START = 60
FINAL_DECADE_START = 360
FINAL_DECADE_END = 399
ROBUSTNESS_SEEDS = [41, 42, 43]
MAIN_SCENARIO_MEMBER_FILES = {
    "hazard_noadaptation": ROOT / "simulation_hazard_noadaptation_aqueduct_riverine_parameters_rcp8p5_riverine_firm_topology_100_ensemble20_20260327_212755_members.csv",
    "backup_suppliers": ROOT / "simulation_hazard_backup_suppliers_aqueduct_riverine_parameters_rcp8p5_riverine_firm_topology_100_ensemble20_20260327_214403_members.csv",
    "capital_hardening": ROOT / "simulation_hazard_capital_hardening_aqueduct_riverine_parameters_rcp8p5_riverine_firm_topology_100_ensemble20_20260327_213346_members.csv",
}


def _load_param_file() -> dict:
    with PARAM_FILE.open() as f:
        return json.load(f)


def _base_setup() -> tuple[dict, list[tuple[int, int, int, str, str | None]]]:
    param_data = _load_param_file()
    events = parse_hazard_event_specs(param_data["rp_files"])
    return param_data, events


def _base_model_kwargs(param_data: dict, events) -> dict:
    return {
        "num_households": int(param_data.get("num_households", 100)),
        "num_firms": 20,
        "hazard_events": events,
        "firm_topology_path": str(ROOT / param_data["topology"]),
        "start_year": int(param_data.get("start_year", 0)),
        "steps_per_year": int(param_data.get("steps_per_year", 4)),
        "consumption_ratios": param_data.get("consumption_ratios"),
        "grid_resolution": float(param_data.get("grid_resolution", 1.0)),
        "household_relocation": bool(param_data.get("household_relocation", False)),
    }


def _run_model(seed: int, *, apply_hazards: bool, adaptation_config: dict, steps: int, base_kwargs: dict) -> pd.DataFrame:
    model = EconomyModel(
        seed=seed,
        apply_hazard_impacts=apply_hazards,
        adaptation_params=adaptation_config,
        **base_kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(steps):
            model.step()
    df = model.results_to_dataframe().copy()
    if "Step" not in df.columns:
        df["Step"] = df.index
    return df


def _find_latest_warmup_members() -> Path:
    files = sorted(ROOT.glob(WARMUP_MEMBERS_GLOB))
    if not files:
        raise FileNotFoundError(f"No files found matching {WARMUP_MEMBERS_GLOB}")
    return files[-1]


def _warmup_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    warmup = df[df["Step"] <= WARMUP_END_STEP].copy()
    metrics = ["Firm_Production", "Household_Consumption", "Mean_Price", "Money_Drift"]
    summary = warmup.groupby("Step")[metrics].agg(["mean", lambda s: np.quantile(s, 0.1), lambda s: np.quantile(s, 0.9)])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()]
    summary = summary.rename(
        columns={
            "Firm_Production_<lambda_0>": "Firm_Production_p10",
            "Firm_Production_<lambda_1>": "Firm_Production_p90",
            "Household_Consumption_<lambda_0>": "Household_Consumption_p10",
            "Household_Consumption_<lambda_1>": "Household_Consumption_p90",
            "Mean_Price_<lambda_0>": "Mean_Price_p10",
            "Mean_Price_<lambda_1>": "Mean_Price_p90",
            "Money_Drift_<lambda_0>": "Money_Drift_p10",
            "Money_Drift_<lambda_1>": "Money_Drift_p90",
        }
    )
    summary = summary.reset_index()
    if "Year" in warmup.columns:
        years = warmup.groupby("Step")["Year"].mean().reset_index(drop=True)
        summary["Year"] = years

    rows = []
    agg = warmup.groupby("Step")[["Firm_Production", "Household_Consumption", "Mean_Price"]].mean()
    target_window = agg.loc[WARMUP_TARGET_START:WARMUP_END_STEP]
    target_means = target_window.mean()
    for metric in target_means.index:
        target = float(target_means[metric])
        rel = ((agg[metric] - target).abs() / target)
        stable_step = None
        for step in agg.index:
            if step > WARMUP_END_STEP:
                break
            if bool((rel.loc[step:WARMUP_END_STEP] <= 0.05).all()):
                stable_step = int(step)
                break
        rows.append(
            {
                "metric": metric,
                "late_warmup_mean": target,
                "stable_within_5pct_from_step": stable_step,
                "stable_within_5pct_from_year": summary.loc[summary["Step"] == stable_step, "Year"].iloc[0]
                if stable_step is not None and "Year" in summary.columns
                else np.nan,
            }
        )
    money_drift_abs_max = float(warmup["Money_Drift"].abs().max())
    rows.append(
        {
            "metric": "Money_Drift",
            "late_warmup_mean": float(target_window.shape[0]),
            "stable_within_5pct_from_step": np.nan,
            "stable_within_5pct_from_year": money_drift_abs_max,
        }
    )
    return summary, pd.DataFrame(rows)


def _plot_warmup(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("Firm_Production", "Firm Production"),
        ("Household_Consumption", "Household Consumption"),
        ("Money_Drift", "Money Drift"),
    ]
    x_col = "Year" if "Year" in summary.columns else "Step"
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (metric, title) in zip(axes_flat, metrics):
        x = summary[x_col].to_numpy()
        mean = summary[f"{metric}_mean"].to_numpy()
        p10 = summary[f"{metric}_p10"].to_numpy()
        p90 = summary[f"{metric}_p90"].to_numpy()
        ax.fill_between(x, p10, p90, color="#b9c9d9", alpha=0.35)
        ax.plot(x, mean, color="#2c5c85", linewidth=2.1)
        ax.set_title(title)
        ax.set_xlabel("Year" if x_col == "Year" else "Step")
        if metric != "Money_Drift":
            target = float(summary.loc[summary["Step"].between(WARMUP_TARGET_START, WARMUP_END_STEP), f"{metric}_mean"].mean())
            ax.axhline(target, color="#aa6f39", linestyle="--", linewidth=1.0)
            ax.axhspan(target * 0.95, target * 1.05, color="#e2c9ac", alpha=0.22)
        else:
            ax.axhline(0.0, color="#aa6f39", linestyle="--", linewidth=1.0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle("No-hazard warm-up convergence on the baseline ensemble", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _strategy_config(base_adaptation: dict, *, strategy: str, replacement_frequency: int) -> dict:
    config = dict(base_adaptation)
    config["enabled"] = True
    config["adaptation_strategy"] = strategy
    config["replacement_frequency"] = replacement_frequency
    if strategy == "backup_suppliers":
        config["adaptation_sensitivity_min"] = 0.8
        config["adaptation_sensitivity_max"] = 1.4
    elif strategy == "capital_hardening":
        config["adaptation_sensitivity_min"] = 0.5
        config["adaptation_sensitivity_max"] = 1.5
    return config


def _noadapt_config(base_adaptation: dict, *, replacement_frequency: int) -> dict:
    config = dict(base_adaptation)
    config["enabled"] = False
    config["replacement_frequency"] = replacement_frequency
    return config


def _final_decade_metrics(df: pd.DataFrame) -> dict[str, float]:
    late = df[df["Step"].between(FINAL_DECADE_START, FINAL_DECADE_END)].copy()
    real_liquidity = late["Firm_Wealth"] / late["Mean_Price"].replace(0, np.nan)
    return {
        "final_production": float(late["Firm_Production"].mean()),
        "final_consumption": float(late["Household_Consumption"].mean()),
        "final_real_liquidity": float(real_liquidity.mean()),
        "final_direct_loss": float(late["Average_Realized_Direct_Loss"].mean()),
        "final_supplier_disruption": float(late["Average_Supplier_Disruption"].mean()),
        "final_replacements": float(late["Firm_Replacements"].max()),
    }


def _seed_metrics_from_members(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    late = df[df["Step"].between(FINAL_DECADE_START, FINAL_DECADE_END)].copy()
    results: dict[int, dict[str, float]] = {}
    for seed, seed_df in late.groupby("Seed"):
        real_liquidity = seed_df["Firm_Wealth"] / seed_df["Mean_Price"].replace(0, np.nan)
        results[int(seed)] = {
            "final_production": float(seed_df["Firm_Production"].mean()),
            "final_consumption": float(seed_df["Household_Consumption"].mean()),
            "final_real_liquidity": float(real_liquidity.mean()),
            "final_direct_loss": float(seed_df["Average_Realized_Direct_Loss"].mean()),
            "final_supplier_disruption": float(seed_df["Average_Supplier_Disruption"].mean()),
            "final_replacements": float(seed_df["Firm_Replacements"].max()),
        }
    return results


def _existing_main_scenario_metrics() -> tuple[dict[tuple[int, int], dict[str, float]], list[dict[str, object]]]:
    baseline_members = pd.read_csv(MAIN_SCENARIO_MEMBER_FILES["hazard_noadaptation"])
    baseline_metrics_all = _seed_metrics_from_members(baseline_members)
    baseline_by_seed = {
        (10, seed): metrics
        for seed, metrics in baseline_metrics_all.items()
        if seed in ROBUSTNESS_SEEDS
    }

    comparison_rows: list[dict[str, object]] = []
    for strategy_key, strategy_label in [
        ("backup_suppliers", "Backup suppliers"),
        ("capital_hardening", "Capital hardening"),
    ]:
        members = pd.read_csv(MAIN_SCENARIO_MEMBER_FILES[strategy_key])
        metrics_all = _seed_metrics_from_members(members)
        for seed in ROBUSTNESS_SEEDS:
            metrics = metrics_all[seed]
            baseline = baseline_by_seed[(10, seed)]
            comparison_rows.append(
                {
                    "config_key": "freq10_inherit",
                    "config_label": "10-step\ninherit",
                    "strategy_key": strategy_key,
                    "strategy_label": strategy_label,
                    "seed": seed,
                    "production_recovery_pct": 100.0 * (metrics["final_production"] - baseline["final_production"]) / baseline["final_production"],
                    "consumption_recovery_pct": 100.0 * (metrics["final_consumption"] - baseline["final_consumption"]) / baseline["final_consumption"],
                    "real_liquidity_change_pct": 100.0 * (metrics["final_real_liquidity"] - baseline["final_real_liquidity"]) / baseline["final_real_liquidity"],
                    "direct_loss_reduction_pct": 100.0 * (baseline["final_direct_loss"] - metrics["final_direct_loss"]) / baseline["final_direct_loss"],
                    "supplier_disruption_reduction_pct": 100.0 * (baseline["final_supplier_disruption"] - metrics["final_supplier_disruption"]) / baseline["final_supplier_disruption"],
                    "final_replacements": metrics["final_replacements"],
                }
            )
    return baseline_by_seed, comparison_rows


def _run_reorganization_robustness() -> tuple[pd.DataFrame, pd.DataFrame]:
    param_data, events = _base_setup()
    base_kwargs = _base_model_kwargs(param_data, events)
    steps = int(param_data.get("steps", 400))
    base_adaptation = dict(param_data.get("adaptation", {}))

    config_labels = [
        ("freq5", 5, "5-step"),
        ("freq20", 20, "20-step"),
    ]
    strategies = [
        ("backup_suppliers", "Backup suppliers"),
        ("capital_hardening", "Capital hardening"),
    ]

    baseline_by_seed, comparison_rows = _existing_main_scenario_metrics()

    baseline_frequencies = [5, 20]
    for freq in baseline_frequencies:
        for seed in ROBUSTNESS_SEEDS:
            df = _run_model(
                seed,
                apply_hazards=True,
                adaptation_config=_noadapt_config(base_adaptation, replacement_frequency=freq),
                steps=steps,
                base_kwargs=base_kwargs,
            )
            baseline_by_seed[(freq, seed)] = _final_decade_metrics(df)

    for config_key, freq, display_label in config_labels:
        for strategy_key, strategy_label in strategies:
            seed_rows = []
            for seed in ROBUSTNESS_SEEDS:
                df = _run_model(
                    seed,
                    apply_hazards=True,
                    adaptation_config=_strategy_config(
                        base_adaptation,
                        strategy=strategy_key,
                        replacement_frequency=freq,
                    ),
                    steps=steps,
                    base_kwargs=base_kwargs,
                )
                metrics = _final_decade_metrics(df)
                baseline = baseline_by_seed[(freq, seed)]
                seed_rows.append(
                    {
                        "config_key": config_key,
                        "config_label": display_label,
                        "strategy_key": strategy_key,
                        "strategy_label": strategy_label,
                        "seed": seed,
                        "production_recovery_pct": 100.0 * (metrics["final_production"] - baseline["final_production"]) / baseline["final_production"],
                        "consumption_recovery_pct": 100.0 * (metrics["final_consumption"] - baseline["final_consumption"]) / baseline["final_consumption"],
                        "real_liquidity_change_pct": 100.0 * (metrics["final_real_liquidity"] - baseline["final_real_liquidity"]) / baseline["final_real_liquidity"],
                        "direct_loss_reduction_pct": 100.0 * (baseline["final_direct_loss"] - metrics["final_direct_loss"]) / baseline["final_direct_loss"],
                        "supplier_disruption_reduction_pct": 100.0 * (baseline["final_supplier_disruption"] - metrics["final_supplier_disruption"]) / baseline["final_supplier_disruption"],
                        "final_replacements": metrics["final_replacements"],
                    }
                )
            comparison_rows.extend(seed_rows)

    seed_level = pd.DataFrame(comparison_rows)
    summary = (
        seed_level.groupby(["config_key", "config_label", "strategy_key", "strategy_label"])
        .agg(
            production_recovery_pct=("production_recovery_pct", "mean"),
            consumption_recovery_pct=("consumption_recovery_pct", "mean"),
            real_liquidity_change_pct=("real_liquidity_change_pct", "mean"),
            direct_loss_reduction_pct=("direct_loss_reduction_pct", "mean"),
            supplier_disruption_reduction_pct=("supplier_disruption_reduction_pct", "mean"),
            final_replacements=("final_replacements", "mean"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    return seed_level, summary


def _plot_reorganization_robustness(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        ("production_recovery_pct", "Production Recovery\nvs. no adaptation (%)"),
        ("direct_loss_reduction_pct", "Direct Loss Reduction\nvs. no adaptation (%)"),
        ("supplier_disruption_reduction_pct", "Supplier Disruption Reduction\nvs. no adaptation (%)"),
    ]
    strategy_colors = {
        "Backup suppliers": "#4f8a5b",
        "Capital hardening": "#c77d36",
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7))

    config_order = list(summary["config_label"].drop_duplicates())
    x = np.arange(len(config_order))
    for ax, (metric, title) in zip(axes, metrics):
        for strategy in ["Backup suppliers", "Capital hardening"]:
            subset = summary[summary["strategy_label"] == strategy].copy()
            subset = subset.set_index("config_label").loc[config_order].reset_index()
            ax.plot(
                x,
                subset[metric].to_numpy(),
                marker="o",
                linewidth=2.0,
                color=strategy_colors[strategy],
                label=strategy,
            )
        ax.set_xticks(x, config_order)
        ax.set_title(title)
        ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Targeted reorganization robustness across interval and inheritance variants",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    warmup_members = pd.read_csv(_find_latest_warmup_members())
    warmup_summary, warmup_stats = _warmup_summary(warmup_members)
    warmup_summary.to_csv(MANUSCRIPT_DIR / "warmup_convergence_summary.csv", index=False)
    warmup_stats.to_csv(MANUSCRIPT_DIR / "warmup_convergence_stats.csv", index=False)
    _plot_warmup(warmup_summary, MANUSCRIPT_DIR / "warmup_convergence.png")

    seed_level, robustness_summary = _run_reorganization_robustness()
    seed_level.to_csv(MANUSCRIPT_DIR / "reorganization_robustness_targeted_seed_level.csv", index=False)
    robustness_summary.to_csv(MANUSCRIPT_DIR / "reorganization_robustness_targeted_summary.csv", index=False)
    _plot_reorganization_robustness(robustness_summary, MANUSCRIPT_DIR / "reorganization_robustness_targeted.png")


if __name__ == "__main__":
    main()
