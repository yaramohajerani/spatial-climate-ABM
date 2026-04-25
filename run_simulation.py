"""Run prototype climate-economy ABM headlessly and persist results."""

from pathlib import Path
import argparse
from datetime import datetime
import json
import shlex
import subprocess
import sys
import numpy as np
import pandas as pd

try:  # pragma: no cover - package import path
    from .model import EconomyModel
    from .agents import FirmAgent, HouseholdAgent
    from .ensemble_utils import (
        ENSEMBLE_STAT_ORDER,
        METADATA_PREFIX,
        apply_metadata as apply_ensemble_metadata,
        build_ensemble_summary as summarize_ensemble,
        ensemble_seed_metadata as summarize_seed_metadata,
    )
    from .hazard_utils import event_signature, parse_hazard_event_specs
    from .shock_inputs import (
        legacy_hazard_event_tuples,
        normalize_lane_shocks,
        normalize_node_shocks,
        normalize_raster_hazard_events,
        normalize_route_shocks,
    )
except ImportError:  # pragma: no cover - flat script import path
    from model import EconomyModel
    from agents import FirmAgent, HouseholdAgent
    from ensemble_utils import (
        ENSEMBLE_STAT_ORDER,
        METADATA_PREFIX,
        apply_metadata as apply_ensemble_metadata,
        build_ensemble_summary as summarize_ensemble,
        ensemble_seed_metadata as summarize_seed_metadata,
    )
    from hazard_utils import event_signature, parse_hazard_event_specs
    from shock_inputs import (
        legacy_hazard_event_tuples,
        normalize_lane_shocks,
        normalize_node_shocks,
        normalize_raster_hazard_events,
        normalize_route_shocks,
    )
# Runner now expects one or more --rp-file arguments in the form
# "<RP>:<START_STEP>:<END_STEP>:<TYPE>:<path>"

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:START:END:TYPE:PATH",
        help=(
            "Add a GeoTIFF file. Format: <RP>:<START_STEP>:<END_STEP>:<HAZARD_TYPE>:<path|None>. "
            "Required unless provided via --param-file. "
            "Example: --rp-file 100:1:20:FL:rp100_2030.tif or --rp-file 10:1:80:FL:None"
        ),
    )
    p.add_argument("--viz", action="store_true", help="Launch interactive Solara dashboard instead of headless run")
    p.add_argument("--steps", type=int, default=10, help="Number of timesteps to simulate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--seeds", nargs="+", type=int, help="Optional explicit list of seeds for an ensemble run")
    p.add_argument("--n-seeds", type=int, default=None, help="Number of consecutive seeds to run starting from --seed or --seed-start")
    p.add_argument("--seed-start", type=int, default=None, help="Starting seed for --n-seeds (defaults to --seed)")
    p.add_argument("--start-year", type=int, default=0, help="Base calendar year for step 0 (optional; used for plotting)")
    p.add_argument("--topology", type=str, help="Optional JSON file describing firm supply-chain topology")
    p.add_argument(
        "--param-file",
        type=str,
        help=(
            "Path to a JSON file containing parameter overrides. Keys can include "
            "rp_files (list), viz (bool), seed/seeds, topology (str), and ensemble settings."
        ),
    )
    p.add_argument("--no-hazards", action="store_true", help="Run baseline scenario without hazard impacts")
    p.add_argument("--no-adaptation", action="store_true", help="Disable hazard-conditional adaptation in firms")
    p.add_argument(
        "--adaptation-strategy",
        type=str,
        choices=["backup_suppliers", "capital_hardening", "stockpiling", "reserved_capacity"],
        default=None,
        help="Adaptation strategy for firms (only used when adaptation is enabled)",
    )
    p.add_argument(
        "--adaptation-sensitivity-min",
        type=float,
        default=None,
        help="Override adaptation_sensitivity_min from the parameter file",
    )
    p.add_argument(
        "--adaptation-sensitivity-max",
        type=float,
        default=None,
        help="Override adaptation_sensitivity_max from the parameter file",
    )
    p.add_argument("--no-learning", action="store_true", help="Deprecated alias for --no-adaptation")
    p.add_argument("--save-agent-ensemble", action="store_true", help="When running multiple seeds, also save the combined agent panel")
    p.add_argument("--ensemble-plot-stat", choices=("mean", "median"), default="mean", help="Statistic to highlight in ensemble plots and summaries")
    return p.parse_args()


def _resolve_seed_list(args) -> list[int]:
    """Return the ordered list of seeds to run."""
    if getattr(args, "seeds", None):
        seen = set()
        ordered = []
        for seed in args.seeds:
            seed = int(seed)
            if seed not in seen:
                ordered.append(seed)
                seen.add(seed)
        return ordered

    if getattr(args, "n_seeds", None):
        if args.n_seeds <= 0:
            raise SystemExit("--n-seeds must be positive")
        start = args.seed_start if args.seed_start is not None else args.seed
        return list(range(int(start), int(start) + int(args.n_seeds)))

    return [int(args.seed)]


STRATEGY_DISPLAY_NAMES = {
    "backup_suppliers": "Backup Suppliers",
    "capital_hardening": "Capital Hardening",
    "stockpiling": "Stockpiling",
    "reserved_capacity": "Reserved Capacity",
}


def _scenario_display(apply_hazards: bool, adaptation_enabled: bool, adaptation_strategy: str = "") -> str:
    base = "Hazard" if apply_hazards else "Baseline"
    if not adaptation_enabled:
        suffix = "No Adaptation"
    elif adaptation_strategy:
        suffix = STRATEGY_DISPLAY_NAMES.get(adaptation_strategy, adaptation_strategy)
    else:
        suffix = "Adaptation"
    return f"{base} + {suffix}"


def _scenario_label(apply_hazards: bool, adaptation_enabled: bool, adaptation_strategy: str = "") -> str:
    parts = ["hazard" if apply_hazards else "baseline"]
    if not adaptation_enabled:
        parts.append("noadaptation")
    elif adaptation_strategy:
        parts.append(adaptation_strategy)
    else:
        parts.append("adaptation")
    return "_".join(parts)


def _safe_git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # noqa: BLE001
        return ""
    return proc.stdout.strip()


def _metadata_json(value: object | None) -> str:
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized_adaptation_config(adaptation_config: dict | None) -> dict[str, object]:
    config = adaptation_config or {}
    continuity_decay = float(config.get("continuity_decay", config.get("resilience_decay", 0.01)))
    sensitivity_min = float(config.get("adaptation_sensitivity_min", 2.0))
    sensitivity_max = float(config.get("adaptation_sensitivity_max", 4.0))
    return {
        "enabled": bool(config.get("enabled", True)),
        "decision_interval": int(config.get("decision_interval", 4)),
        "ewma_alpha": float(config.get("ewma_alpha", 0.2)),
        "observation_radius": float(config.get("observation_radius", 4.0)),
        "adaptation_sensitivity_min": sensitivity_min,
        "adaptation_sensitivity_max": sensitivity_max,
        "max_adaptation_increment": float(config.get("max_adaptation_increment", 0.25)),
        "continuity_decay": continuity_decay,
        "maintenance_cost_rate": float(config.get("maintenance_cost_rate", 0.005)),
        "adaptation_strategy": str(config.get("adaptation_strategy", "backup_suppliers")),
        "max_backup_suppliers": int(config.get("max_backup_suppliers", 5)),
        "reserved_capacity_share": float(config.get("reserved_capacity_share", 0.35)),
        "reserved_capacity_markup_cap": float(config.get("reserved_capacity_markup_cap", 0.10)),
        "min_money_survival": float(config.get("min_money_survival", 1.0)),
        "replacement_frequency": int(config.get("replacement_frequency", 10)),
    }


def _coerce_shock_inputs(
    *,
    legacy_rp_files: list[str] | None,
    raster_hazard_events,
    node_shocks,
    lane_shocks,
    route_shocks,
):
    legacy_events = parse_hazard_event_specs(legacy_rp_files) if legacy_rp_files else []
    raster_events = normalize_raster_hazard_events(
        raster_hazard_events,
        legacy_hazard_events=legacy_events,
    )
    node_events = normalize_node_shocks(node_shocks)
    lane_events = normalize_lane_shocks(lane_shocks)
    route_events = normalize_route_shocks(route_shocks)
    return raster_events, node_events, lane_events, route_events


def _base_metadata(
    *,
    args,
    events,
    apply_hazards: bool,
    adaptation_enabled: bool,
    adaptation_config: dict,
    scenario_label: str,
    timestamp: str,
    param_data: dict | None = None,
) -> dict[str, object]:
    param_path = str(args.param_file) if args.param_file else ""
    topology_path = str(args.topology) if args.topology else ""
    param_data = param_data or {}
    param_adaptation_config = param_data.get("adaptation", param_data.get("learning"))
    effective_adaptation_config = _normalized_adaptation_config(adaptation_config)
    sensitivity_min = float(effective_adaptation_config["adaptation_sensitivity_min"])
    sensitivity_max = float(effective_adaptation_config["adaptation_sensitivity_max"])
    return {
        f"{METADATA_PREFIX}ScenarioLabel": scenario_label,
        f"{METADATA_PREFIX}ApplyHazards": bool(apply_hazards),
        f"{METADATA_PREFIX}AdaptationEnabled": bool(adaptation_enabled),
        f"{METADATA_PREFIX}AdaptationStrategy": str(effective_adaptation_config["adaptation_strategy"]) if adaptation_enabled else "",
        f"{METADATA_PREFIX}ParamFile": param_path,
        f"{METADATA_PREFIX}ParamFileStem": Path(param_path).stem if param_path else "",
        f"{METADATA_PREFIX}TopologyPath": topology_path,
        f"{METADATA_PREFIX}TopologyStem": Path(topology_path).stem if topology_path else "",
        f"{METADATA_PREFIX}RunCommand": " ".join(shlex.quote(arg) for arg in sys.argv),
        f"{METADATA_PREFIX}NoHazardsFlag": bool(getattr(args, "no_hazards", False)),
        f"{METADATA_PREFIX}NoAdaptationFlag": bool(getattr(args, "no_adaptation", False)),
        f"{METADATA_PREFIX}NoLearningFlag": bool(getattr(args, "no_learning", False)),
        f"{METADATA_PREFIX}CLIAdaptationStrategy": str(getattr(args, "adaptation_strategy", "") or ""),
        f"{METADATA_PREFIX}CLIAdaptationSensitivityMin": (
            "" if getattr(args, "adaptation_sensitivity_min", None) is None else float(args.adaptation_sensitivity_min)
        ),
        f"{METADATA_PREFIX}CLIAdaptationSensitivityMax": (
            "" if getattr(args, "adaptation_sensitivity_max", None) is None else float(args.adaptation_sensitivity_max)
        ),
        f"{METADATA_PREFIX}HazardEventCount": len(events),
        f"{METADATA_PREFIX}HazardEvents": event_signature(events),
        f"{METADATA_PREFIX}StartYear": int(args.start_year),
        f"{METADATA_PREFIX}StepsPerYear": int(args.steps_per_year),
        f"{METADATA_PREFIX}StepsRequested": int(args.steps),
        f"{METADATA_PREFIX}NumHouseholds": int(args.num_households),
        f"{METADATA_PREFIX}GridResolution": float(args.grid_resolution),
        f"{METADATA_PREFIX}HouseholdRelocation": bool(args.household_relocation),
        f"{METADATA_PREFIX}DamageFunctionsPath": str(getattr(args, "damage_functions_path", "") or ""),
        f"{METADATA_PREFIX}LandBoundariesPath": str(getattr(args, "land_boundaries_path", "") or ""),
        f"{METADATA_PREFIX}ConsumptionRatios": _metadata_json(getattr(args, "consumption_ratios", None)),
        f"{METADATA_PREFIX}ParamConsumptionRatios": _metadata_json(param_data.get("consumption_ratios")),
        f"{METADATA_PREFIX}ParamInputRecipeRanges": _metadata_json(param_data.get("input_recipe_ranges")),
        f"{METADATA_PREFIX}HHConsumptionPropensityIncome": float(HouseholdAgent.CONSUMPTION_PROPENSITY_INCOME),
        f"{METADATA_PREFIX}HHConsumptionPropensityWealth": float(HouseholdAgent.CONSUMPTION_PROPENSITY_WEALTH),
        f"{METADATA_PREFIX}HHTargetCashBuffer": float(HouseholdAgent.TARGET_CASH_BUFFER),
        f"{METADATA_PREFIX}FirmInventoryBufferRatio": float(FirmAgent.INVENTORY_BUFFER_RATIO),
        f"{METADATA_PREFIX}FirmLiquidityBufferRatio": float(FirmAgent.LIQUIDITY_BUFFER_RATIO),
        f"{METADATA_PREFIX}FirmMinLiquidityBuffer": float(FirmAgent.MIN_LIQUIDITY_BUFFER),
        f"{METADATA_PREFIX}FirmWorkingCapitalCreditRevenueShare": float(FirmAgent.WORKING_CAPITAL_CREDIT_REVENUE_SHARE),
        f"{METADATA_PREFIX}FirmLaborShare": float(FirmAgent.LABOR_SHARE),
        f"{METADATA_PREFIX}NoWorkerWagePremium": float(FirmAgent.NO_WORKER_WAGE_PREMIUM),
        f"{METADATA_PREFIX}DecisionInterval": int(effective_adaptation_config["decision_interval"]),
        f"{METADATA_PREFIX}EWMAAlpha": float(effective_adaptation_config["ewma_alpha"]),
        f"{METADATA_PREFIX}ObservationRadius": float(effective_adaptation_config["observation_radius"]),
        f"{METADATA_PREFIX}AdaptationSensitivityMin": sensitivity_min,
        f"{METADATA_PREFIX}AdaptationSensitivityMax": sensitivity_max,
        f"{METADATA_PREFIX}AdaptationSensitivityMidpoint": 0.5 * (sensitivity_min + sensitivity_max),
        f"{METADATA_PREFIX}ContinuitySensitivityMin": sensitivity_min,
        f"{METADATA_PREFIX}ContinuitySensitivityMax": sensitivity_max,
        f"{METADATA_PREFIX}MaxAdaptIncrement": float(effective_adaptation_config["max_adaptation_increment"]),
        f"{METADATA_PREFIX}ResilienceDecay": float(effective_adaptation_config["continuity_decay"]),
        f"{METADATA_PREFIX}ContinuityDecay": float(effective_adaptation_config["continuity_decay"]),
        f"{METADATA_PREFIX}MaintenanceCostRate": float(effective_adaptation_config["maintenance_cost_rate"]),
        f"{METADATA_PREFIX}MaxBackupSuppliers": int(effective_adaptation_config["max_backup_suppliers"]),
        f"{METADATA_PREFIX}ReservedCapacityShare": float(effective_adaptation_config["reserved_capacity_share"]),
        f"{METADATA_PREFIX}ReservedCapacityMarkupCap": float(effective_adaptation_config["reserved_capacity_markup_cap"]),
        f"{METADATA_PREFIX}MinMoneySurvival": float(effective_adaptation_config["min_money_survival"]),
        f"{METADATA_PREFIX}ReplacementFrequency": int(effective_adaptation_config["replacement_frequency"]),
        f"{METADATA_PREFIX}EffectiveAdaptationConfig": _metadata_json(effective_adaptation_config),
        f"{METADATA_PREFIX}ParamAdaptationConfig": _metadata_json(param_adaptation_config),
        f"{METADATA_PREFIX}EnsemblePlotStat": str(getattr(args, "ensemble_plot_stat", "mean")),
        f"{METADATA_PREFIX}SaveAgentEnsemble": bool(getattr(args, "save_agent_ensemble", False)),
        f"{METADATA_PREFIX}RunTimestamp": timestamp,
        f"{METADATA_PREFIX}GitCommit": _safe_git_commit(),
        f"{METADATA_PREFIX}SourceMemberFiles": "",
    }


def _model_effective_metadata(model: EconomyModel) -> dict[str, object]:
    return {
        f"{METADATA_PREFIX}{key}": value
        for key, value in model.effective_configuration_metadata().items()
    }


def _finalize_main_results(df: pd.DataFrame, *, scenario_display: str, seed: int, start_year: int, steps_per_year: int) -> pd.DataFrame:
    df = df.copy()
    df["Scenario"] = scenario_display
    if "Step" not in df.columns:
        df["Step"] = df.index
    if start_year and "Year" not in df.columns:
        df["Year"] = start_year + df["Step"].astype(float) / steps_per_year
    df["Seed"] = seed
    return df


def _finalize_agent_results(agent_df: pd.DataFrame, *, scenario_display: str, seed: int, start_year: int, steps_per_year: int) -> pd.DataFrame:
    agent_df = agent_df.copy()
    agent_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")
    agent_df["Scenario"] = scenario_display
    agent_df["Seed"] = seed
    if start_year and "Year" not in agent_df.columns and "Step" in agent_df.columns:
        agent_df["Year"] = start_year + agent_df["Step"].astype(float) / steps_per_year
    return agent_df


def _run_single_simulation(
    *,
    args,
    raster_events,
    node_shocks,
    lane_shocks,
    route_shocks,
    apply_shocks: bool,
    adaptation_config: dict,
    seed: int,
    scenario_display: str,
):
    model = EconomyModel(
        num_households=args.num_households,
        num_firms=20,
        raster_hazard_events=raster_events,
        node_shocks=node_shocks,
        lane_shocks=lane_shocks,
        route_shocks=route_shocks,
        seed=seed,
        apply_hazard_impacts=apply_shocks,
        firm_topology_path=args.topology,
        start_year=args.start_year,
        steps_per_year=args.steps_per_year,
        adaptation_params=adaptation_config,
        consumption_ratios=args.consumption_ratios,
        input_recipe_ranges=getattr(args, "input_recipe_ranges", None),
        grid_resolution=args.grid_resolution,
        household_relocation=args.household_relocation,
        damage_functions_path=getattr(args, "damage_functions_path", None),
        land_boundaries_path=getattr(args, "land_boundaries_path", None),
    )

    for _ in range(args.steps):
        model.step()

    df = _finalize_main_results(
        model.results_to_dataframe(),
        scenario_display=scenario_display,
        seed=seed,
        start_year=args.start_year,
        steps_per_year=args.steps_per_year,
    )
    agent_df = _finalize_agent_results(
        model.datacollector.get_agent_vars_dataframe().reset_index(),
        scenario_display=scenario_display,
        seed=seed,
        start_year=args.start_year,
        steps_per_year=args.steps_per_year,
    )
    return model, df, agent_df, _model_effective_metadata(model)


def _save_ensemble_plot(summary_df: pd.DataFrame, member_df: pd.DataFrame, output_path: Path, *, highlight_stat: str) -> None:
    """Create a lightweight ensemble plot with faint member lines and a summary line."""
    import matplotlib.pyplot as plt

    stat_df = summary_df[summary_df["EnsembleStatistic"] == highlight_stat].copy()
    p10_df = summary_df[summary_df["EnsembleStatistic"] == "p10"].copy()
    p90_df = summary_df[summary_df["EnsembleStatistic"] == "p90"].copy()
    if stat_df.empty:
        return

    scenario = stat_df["Scenario"].iloc[0]
    x_col = "Year" if "Year" in stat_df.columns else "Step"
    metrics = [
        ("Firm_Production", "Aggregate Firm Production", "Units of Goods", None),
        ("Firm_Capital", "Aggregate Firm Capital", "Units of Capital", None),
        ("Firm_Wealth", "Aggregate Real Firm Liquidity", "Real Dollars ($ / Mean Price)", "Mean_Price"),
        ("Household_Consumption", "Aggregate Household Consumption", "Units of Goods", None),
        ("Mean_Wage", "Mean Firm Wage Offer", "Real Dollars ($ / Mean Price)", "Mean_Price"),
        ("Mean_Price", "Mean Firm Price", "$ / Unit of Goods", None),
        ("Firm_Inventory", "Aggregate Firm Inventory", "Units of Goods", None),
        ("Household_Labor_Sold", "Aggregate Household Labor Sold", "Units of Labor", None),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    axes_flat = axes.flatten()
    color = "tab:red" if "Hazard" in scenario else "tab:blue"

    def _deflate(df_subset: pd.DataFrame, metric_col: str, deflator_col: str | None) -> np.ndarray:
        values = df_subset[metric_col].to_numpy(dtype=float)
        if deflator_col and deflator_col in df_subset.columns:
            prices = df_subset[deflator_col].to_numpy(dtype=float)
            prices = np.where(prices == 0, np.nan, prices)
            return np.where(np.isfinite(prices), values / prices, values)
        return values

    for ax, (metric, title, ylabel, deflator) in zip(axes_flat, metrics):
        if metric not in stat_df.columns:
            ax.set_visible(False)
            continue
        for _, member_grp in member_df.groupby("Seed"):
            member_grp = member_grp.sort_values(x_col)
            x_vals = member_grp[x_col].to_numpy()
            y_vals = _deflate(member_grp, metric, deflator)
            ax.plot(x_vals, y_vals, color=color, alpha=0.15, linewidth=0.8)

        stat_grp = stat_df.sort_values(x_col)
        x_vals = stat_grp[x_col].to_numpy()
        lower = p10_df.sort_values(x_col)
        upper = p90_df.sort_values(x_col)
        if not lower.empty and not upper.empty and metric in lower.columns and metric in upper.columns:
            lower_vals = _deflate(lower, metric, deflator)
            upper_vals = _deflate(upper, metric, deflator)
            ax.fill_between(x_vals, lower_vals, upper_vals, color=color, alpha=0.12)
        ax.plot(x_vals, _deflate(stat_grp, metric, deflator), color=color, linewidth=2.5, label=highlight_stat.title())
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)
        ax.legend(fontsize=8)

    fig.suptitle(f"{scenario} Ensemble ({member_df['Seed'].nunique()} seeds)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> None:  # noqa: D401
    args = _parse()
    param_data: dict = {}
    args.raster_hazard_events = None
    args.node_shocks = None
    args.lane_shocks = None
    args.route_shocks = None
    args.damage_functions_path = None
    args.land_boundaries_path = None
    args.steps_per_year = 4
    args.adaptation_params = {}
    args.consumption_ratios = None
    args.num_households = 100
    args.grid_resolution = 1.0
    args.household_relocation = False

    # ---------------- Optional parameter file ---------------------------- #
    if args.param_file:
        param_path = Path(args.param_file)
        if not param_path.exists():
            raise SystemExit(f"Parameter file not found: {param_path}")

        try:
            with param_path.open() as f:
                param_data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Failed to parse JSON parameter file: {param_path}") from exc

        # Merge parameters – CLI flags take precedence over file settings
        # 1. rp_files (list[str]) ----------------------------------------
        file_rp = param_data.get("rp_files") or param_data.get("rp_file")
        if file_rp and not args.rp_file:
            args.rp_file = file_rp
        elif file_rp and args.rp_file:
            # Combine – keep CLI order last so they override duplicates
            args.rp_file = file_rp + args.rp_file

        # 2. Viz flag ------------------------------------------------------
        if getattr(args, "viz", False) is False and param_data.get("viz"):
            args.viz = bool(param_data.get("viz"))

        # 3. Seed -----------------------------------------------------------
        if args.seed == 42 and "seed" in param_data:
            args.seed = int(param_data["seed"])
        if not args.seeds and "seeds" in param_data:
            args.seeds = [int(seed) for seed in param_data["seeds"]]
        if args.n_seeds is None and "n_seeds" in param_data:
            args.n_seeds = int(param_data["n_seeds"])
        if args.seed_start is None and "seed_start" in param_data:
            args.seed_start = int(param_data["seed_start"])

        # 3a. Steps ---------------------------------------------------------
        if args.steps == 10 and "steps" in param_data:
            args.steps = int(param_data["steps"])

        # 3b. Start year ----------------------------------------------------
        if args.start_year == 0 and "start_year" in param_data:
            args.start_year = int(param_data["start_year"])

        # 3c. Steps per year -----------------------------------------------
        args.steps_per_year = int(param_data.get("steps_per_year", 4))

        # 4. Topology path --------------------------------------------------
        if not args.topology and param_data.get("topology"):
            args.topology = str(param_data["topology"])

        # 5. Explicit shock sections ----------------------------------------
        args.raster_hazard_events = param_data.get("raster_hazard_events", None)
        args.node_shocks = param_data.get("node_shocks", None)
        args.lane_shocks = param_data.get("lane_shocks", None)
        args.route_shocks = param_data.get("route_shocks", None)

        # 6. Resource paths -------------------------------------------------
        if param_data.get("damage_functions_path") is not None:
            args.damage_functions_path = str(param_data["damage_functions_path"])
        if param_data.get("land_boundaries_path") is not None:
            args.land_boundaries_path = str(param_data["land_boundaries_path"])

        # 7. Adaptation parameters ------------------------------------------
        args.adaptation_params = param_data.get("adaptation", param_data.get("learning", {}))

        # 8. Consumption ratios by sector -----------------------------------
        args.consumption_ratios = param_data.get("consumption_ratios", None)
        args.input_recipe_ranges = param_data.get("input_recipe_ranges", None)

        # 9. Number of households -------------------------------------------
        args.num_households = int(param_data.get("num_households", 100))

        # 10. Grid resolution (degrees per cell) ----------------------------
        args.grid_resolution = float(param_data.get("grid_resolution", 1.0))

        # 11. Household relocation toggle -----------------------------------
        args.household_relocation = bool(param_data.get("household_relocation", False))
        if not args.save_agent_ensemble and "save_agent_ensemble" in param_data:
            args.save_agent_ensemble = bool(param_data.get("save_agent_ensemble", False))
        if args.ensemble_plot_stat == "mean" and "ensemble_plot_stat" in param_data:
            args.ensemble_plot_stat = str(param_data.get("ensemble_plot_stat", "mean"))

    try:
        raster_events, node_shocks, lane_shocks, route_shocks = _coerce_shock_inputs(
            legacy_rp_files=args.rp_file,
            raster_hazard_events=args.raster_hazard_events,
            node_shocks=args.node_shocks,
            lane_shocks=args.lane_shocks,
            route_shocks=args.route_shocks,
        )
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc

    events = legacy_hazard_event_tuples(raster_events)

    seed_list = _resolve_seed_list(args)
    if args.viz and len(seed_list) > 1:
        raise SystemExit("Multi-seed ensemble mode is not supported with --viz.")

    # If visualization requested, delegate to Solara which hosts the dashboard
    if args.viz:
        import os

        if node_shocks or lane_shocks or route_shocks:
            raise SystemExit("Visualization currently supports raster_hazard_events only.")

        env = os.environ.copy()
        # Pass hazard events to the dashboard so it can build the same model
        env["ABM_HAZARD_EVENTS"] = ";".join(
            f"{rp}:{s}:{e}:{t}:{p}" for rp, s, e, t, p in events
        )
        env["ABM_SEED"] = str(args.seed)
        if args.topology:
            env["ABM_TOPOLOGY_PATH"] = args.topology

        if args.start_year:
            env["ABM_START_YEAR"] = str(args.start_year)

        cmd = [sys.executable, "-m", "solara", "run", "visualization.py"]
        subprocess.run(cmd, env=env, check=False)
        return

    # Configure scenario settings
    has_shock_inputs = bool(raster_events or node_shocks or lane_shocks or route_shocks)
    apply_hazards = bool(has_shock_inputs and not args.no_hazards)
    disable_adaptation = bool(args.no_adaptation or args.no_learning)
    if disable_adaptation:
        adaptation_config = {**args.adaptation_params, "enabled": False}
    else:
        adaptation_config = args.adaptation_params
    adaptation_enabled = bool(adaptation_config.get("enabled", True))

    # CLI override for adaptation strategy
    if getattr(args, "adaptation_strategy", None) and adaptation_enabled:
        adaptation_config = {**adaptation_config, "adaptation_strategy": args.adaptation_strategy}
    if adaptation_enabled and (
        getattr(args, "adaptation_sensitivity_min", None) is not None
        or getattr(args, "adaptation_sensitivity_max", None) is not None
    ):
        adaptation_config = {**adaptation_config}
        if args.adaptation_sensitivity_min is not None:
            adaptation_config["adaptation_sensitivity_min"] = float(args.adaptation_sensitivity_min)
        if args.adaptation_sensitivity_max is not None:
            adaptation_config["adaptation_sensitivity_max"] = float(args.adaptation_sensitivity_max)
        sensitivity_min = float(adaptation_config.get("adaptation_sensitivity_min", 2.0))
        sensitivity_max = float(adaptation_config.get("adaptation_sensitivity_max", 4.0))
        if sensitivity_max < sensitivity_min:
            raise SystemExit("--adaptation-sensitivity-max must be >= --adaptation-sensitivity-min")
    adaptation_strategy = str(adaptation_config.get("adaptation_strategy", "")) if adaptation_enabled else ""

    # Generate scenario label for output files
    scenario_label = _scenario_label(apply_hazards, adaptation_enabled, adaptation_strategy)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topo_tag = ""
    if args.topology:
        topo_tag = Path(args.topology).stem
    param_tag = ""
    if args.param_file:
        param_tag = Path(args.param_file).stem

    # Build output filename with all tags
    tags = [scenario_label]
    if param_tag:
        tags.append(param_tag)
    if topo_tag:
        tags.append(topo_tag)
    if len(seed_list) > 1:
        tags.append(f"ensemble{len(seed_list)}")
    tags.append(timestamp)
    scenario_label_ts = "_".join(tags)
    scenario_display = _scenario_display(apply_hazards, adaptation_enabled, adaptation_strategy)
    metadata = {
        **_base_metadata(
            args=args,
            events=events,
            apply_hazards=apply_hazards,
            adaptation_enabled=adaptation_enabled,
            adaptation_config=adaptation_config,
            scenario_label=scenario_label,
            timestamp=timestamp,
            param_data=param_data,
        ),
        **summarize_seed_metadata(seed_list),
    }

    # Save results with scenario label + timestamp
    output_filename = f"simulation_{scenario_label_ts}"
    main_csv_path = f"{output_filename}.csv"
    agent_csv_path = f"{output_filename}_agents.csv"

    if len(seed_list) > 1:
        member_frames = []
        agent_member_frames = []
        effective_model_metadata: dict[str, object] | None = None
        for seed in seed_list:
            _, seed_df, seed_agent_df, seed_effective_metadata = _run_single_simulation(
                args=args,
                raster_events=raster_events,
                node_shocks=node_shocks,
                lane_shocks=lane_shocks,
                route_shocks=route_shocks,
                apply_shocks=apply_hazards,
                adaptation_config=adaptation_config,
                seed=seed,
                scenario_display=scenario_display,
            )
            if effective_model_metadata is None:
                effective_model_metadata = seed_effective_metadata
            member_frames.append(seed_df)
            if args.save_agent_ensemble:
                agent_member_frames.append(seed_agent_df)

        if effective_model_metadata:
            metadata = {**metadata, **effective_model_metadata}
        member_df = pd.concat(member_frames, ignore_index=True)
        member_df = apply_ensemble_metadata(member_df, metadata)
        summary_df = summarize_ensemble(member_df, group_cols=["Scenario", "Step", "Year"])
        summary_df = apply_ensemble_metadata(summary_df, metadata)
        summary_df.to_csv(main_csv_path, index=False)

        members_csv_path = f"{output_filename}_members.csv"
        member_df.to_csv(members_csv_path, index=False)

        if args.save_agent_ensemble and agent_member_frames:
            agent_member_df = pd.concat(agent_member_frames, ignore_index=True)
            agent_member_df = apply_ensemble_metadata(agent_member_df, metadata)
            agent_member_df.to_csv(agent_csv_path, index=False)

        ensemble_plot_path = Path(f"{output_filename}_ensemble.png")
        _save_ensemble_plot(
            summary_df,
            member_df,
            ensemble_plot_path,
            highlight_stat=args.ensemble_plot_stat,
        )

        print(f"Ensemble simulation complete for scenario: {scenario_label}")
        print(f"Seeds run: {seed_list}")
        print(f"Summary results saved as {main_csv_path}")
        print(f"Member results saved as {members_csv_path}")
        if args.save_agent_ensemble and agent_member_frames:
            print(f"Combined agent panel saved as {agent_csv_path}")
        print(f"Ensemble plot saved as {ensemble_plot_path}")
        return

    model, df, agent_df, effective_model_metadata = _run_single_simulation(
        args=args,
        raster_events=raster_events,
        node_shocks=node_shocks,
        lane_shocks=lane_shocks,
        route_shocks=route_shocks,
        apply_shocks=apply_hazards,
        adaptation_config=adaptation_config,
        seed=seed_list[0],
        scenario_display=scenario_display,
    )
    metadata = {**metadata, **effective_model_metadata}
    df = apply_ensemble_metadata(df, metadata)
    agent_df = apply_ensemble_metadata(agent_df, metadata)

    df.to_csv(main_csv_path, index=False)
    agent_df.to_csv(agent_csv_path, index=False)

    print(f"Simulation complete for scenario: {scenario_label}")
    print(f"Results saved as {main_csv_path} and {agent_csv_path}")

    # ------------------------- Plotting ---------------------------- #
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from trophic_utils import compute_trophic_levels

    rename_map = {"Base_Wage": "Mean_Wage", "Household_LaborSold": "Household_Labor_Sold"}
    df = df.rename(columns=rename_map)

    units = {
        "Firm_Production": "Units of Goods",
        "Firm_Consumption": "Units of Goods",
        "Firm_Wealth": "$",
        "Firm_Capital": "Units of Capital",
        "Household_Wealth": "$",
        "Household_Labor_Sold": "Units of Labor",
        "Household_Consumption": "Units of Goods",
        "Average_Risk": "Flood Depth (m)",
        "Mean_Wage": "$ / Unit of Labor",
        "Mean_Price": "$ / Unit of Goods",
        "Labor_Limited_Firms": "count",
        "Capital_Limited_Firms": "count",
        "Input_Limited_Firms": "count",
    }

    # Metric lists kept for readability; plotting uses metrics_left / metrics_right

    # ------------------ Compute trophic levels --------------------------- #
    firm_adj = {
        f.unique_id: [s.unique_id for s in f.connected_firms]
        for f in model.agents if isinstance(f, FirmAgent)
    }
    lvl_map = compute_trophic_levels(firm_adj)

    # Split firm vs household for convenience
    firm_df = agent_df[agent_df["type"] == "FirmAgent"].copy()
    firm_df["Level"] = firm_df["AgentID"].map(lvl_map)

    household_df = agent_df[agent_df["type"] == "HouseholdAgent"].copy()
    household_counts_by_step = household_df.groupby("Step").size() if not household_df.empty else None

    # Sector palette
    unique_sectors = sorted(firm_df["sector"].dropna().unique())
    sec_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sectors)))

    # Single, shared colour map for both firm & household plots
    color_by_sector = {
        sector: sec_colors[idx % len(sec_colors)]
        for idx, sector in enumerate(unique_sectors)
    }

    final_demand_sectors = [
        sector
        for sector in model.get_final_consumption_ratios().keys()
        if sector in unique_sectors
    ]

    # ---------------- Metric selection --------------------------- #
    metrics_left = [
        "Firm_Production",
        "Firm_Consumption",
        "Firm_Wealth",
        "Firm_Capital",
        "Mean_Price",
    ]

    metrics_right = [
        "Household_Labor_Sold",
        "Household_Consumption",
        "Household_Wealth",
        "Mean_Wage", 
        "Average_Risk", 
    ]

    rows = len(metrics_left)  # expect 5
    fig = plt.figure(figsize=(14, rows * 3))
    gs = gridspec.GridSpec(rows, 2, height_ratios=[1] * rows)
    axes_matrix = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(rows)]

    x_col = "Year" if args.start_year else "Step"
    if args.start_year:
        df["Year"] = args.start_year + df.index.astype(int) / args.steps_per_year

    # Map firm metric names to agent DataFrame columns
    firm_metric_map = {
        "Firm_Production": "production",
        "Firm_Consumption": "consumption",
        "Firm_Wealth": "money",
        "Firm_Capital": "capital",
        "Firm_Inventory": "inventory",
    }

    def _plot_firm(col, ax):
        if col == "Mean_Price":
            # Aggregate mean price line
            ax.plot(df[x_col], df[col], color="black", linewidth=2, label="Mean Price")
            # Sector breakdown
            for idx_sec, sector in enumerate(unique_sectors):
                sec_data = firm_df[firm_df["sector"] == sector]
                if sec_data.empty:
                    continue
                grp = sec_data.groupby("Step")["price"].mean()
                x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int)/args.steps_per_year
                ax.plot(x_vals, grp.values, color=sec_colors[idx_sec], linestyle="--", alpha=0.8, label=sector)
        elif col == "Sector_Trophic_Level":
            for idx_sec, sector in enumerate(unique_sectors):
                mean_lvl = firm_df[firm_df["sector"] == sector].groupby("Step")["Level"].mean()
                x_vals = mean_lvl.index if not args.start_year else args.start_year + mean_lvl.index.astype(int)/args.steps_per_year
                ax.plot(x_vals, mean_lvl.values, color=sec_colors[idx_sec], label=sector)
            ax.set_ylabel("trophic level")
        else:
            agent_col = firm_metric_map.get(col, col.lower())
            # Add mean line for all firms
            mean_grp = firm_df.groupby("Step")[agent_col].mean()
            x_vals = mean_grp.index if not args.start_year else args.start_year + mean_grp.index.astype(int)/args.steps_per_year
            ax.plot(x_vals, mean_grp.values, color="black", linewidth=2, label="Mean")
            # Sector breakdown
            for idx_sec, sector in enumerate(unique_sectors):
                grp = firm_df[firm_df["sector"] == sector].groupby("Step")[agent_col].mean()
                if grp.empty:
                    continue
                x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int)/args.steps_per_year
                ax.plot(x_vals, grp.values, color=sec_colors[idx_sec], linestyle="--", alpha=0.8, label=sector)
        ax.set_title(col.replace("_", " "), fontsize=10)
        ylabel = units.get(col, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    household_metric_map = {
        "Household_Wealth": "money",
        "Household_Labor_Sold": "labor_sold",
        "Household_Consumption": "consumption",
    }
    household_title_map = {
        "Household_Wealth": "Mean Household Wealth",
        "Household_Labor_Sold": "Mean Household Labor Sold",
        "Household_Consumption": "Mean Household Consumption",
    }

    def _plot_final_demand_by_sector(ax):
        if not final_demand_sectors or household_counts_by_step is None:
            return
        if "household_sales_last_step" not in firm_df.columns:
            return

        for sector in final_demand_sectors:
            sector_sales = (
                firm_df[firm_df["sector"] == sector]
                .groupby("Step")["household_sales_last_step"]
                .sum()
            )
            if sector_sales.empty:
                continue

            aligned_households = household_counts_by_step.reindex(sector_sales.index)
            per_household_sales = (
                sector_sales / aligned_households.replace(0, np.nan)
            ).dropna()
            if per_household_sales.empty:
                continue

            x_vals = per_household_sales.index if not args.start_year else args.start_year + per_household_sales.index.astype(int) / args.steps_per_year
            ax.plot(
                x_vals,
                per_household_sales.values,
                color=color_by_sector.get(sector, "grey"),
                linestyle="--",
                alpha=0.8,
                label=f"final demand: {sector}",
            )

    def _plot_household(col, ax):
        hh_col = household_metric_map.get(col, None)
        if hh_col:
            # Add mean line for all households
            mean_grp = household_df.groupby("Step")[hh_col].mean()
            x_vals = mean_grp.index if not args.start_year else args.start_year + mean_grp.index.astype(int) / args.steps_per_year
            ax.plot(x_vals, mean_grp.values, color="black", linewidth=2, label="Mean household")
            if col == "Household_Consumption":
                _plot_final_demand_by_sector(ax)
        else:
            # Fallback: plot aggregate series from df (e.g., Average_Risk)
            ax.plot(df[x_col], df[col], color="black", linewidth=2, label=col.replace("_", " "))

        ax.set_title(household_title_map.get(col, col.replace("_", " ")), fontsize=10)
        ylabel = units.get(col, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    # ---------------- Plotting row by row ----------------------------- #
    def _plot_bottleneck(ax):
        # ---------- Combined bottleneck stacked-area -------------------- #
        bt_series = {}
        for bt in ["labor", "capital", "input"]:
            counts = (firm_df[firm_df["limiting_factor"] == bt]
                        .groupby("Step").size().reindex(df.index, fill_value=0))
            bt_series[bt] = counts

        totals = sum(bt_series.values())
        pct_arrays = [100 * bt_series[bt] / totals for bt in ["labor", "capital", "input"]]

        ax.stackplot(df[x_col],
                            *pct_arrays,
                            labels=["Labour", "Capital", "Input"],
                            colors=["#1f77b4", "#d62728", "#2ca02c"],
                            alpha=0.7)

        ax.set_title("Production Bottlenecks (%)")
        ax.set_ylabel("% of firms")
        ax.set_xlabel(x_col)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc="upper center", ncol=3)

    def _plot_wage(ax):
        # plot mean wage
        wage_by_step = firm_df.groupby("Step")["wage"].mean()
        x_vals = wage_by_step.index if not args.start_year else args.start_year + wage_by_step.index.astype(int)/args.steps_per_year
        ax.plot(x_vals, wage_by_step.values, color="black", linewidth=2, label="Mean")

        # plot sector-level wage
        for idx_sec, sector in enumerate(unique_sectors):
            wage_by_step = firm_df[firm_df["sector"] == sector].groupby("Step")["wage"].mean()
            if wage_by_step.empty:
                continue
            x_vals = wage_by_step.index if not args.start_year else args.start_year + wage_by_step.index.astype(int)/args.steps_per_year
            ax.plot(x_vals, wage_by_step.values, label=sector, color=color_by_sector[sector], linestyle="--", alpha=0.8)
        ax.set_title("Wages by Sector", fontsize=10)
        ax.set_ylabel("$ / Unit of Labor")
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    for r in range(rows):
        # Left column
        metric_left = metrics_left[r]
        ax_left = axes_matrix[r][0]
        if metric_left.endswith("_Bottleneck"):
            bt_type = metric_left.split("_")[0].lower()  # capital
            ax_left.set_title(f"{bt_type.capitalize()} Bottlenecks", fontsize=10)
            _plot_bottleneck(bt_type, ax_left)
        else:
            _plot_firm(metric_left, ax_left)

        # Right column
        metric_right = metrics_right[r]
        ax_right = axes_matrix[r][1]
        if metric_right == "Mean_Wage":
            _plot_wage(ax_right)
        elif metric_right.endswith("_Bottleneck"):
            _plot_bottleneck(ax_right)            
        else:
            _plot_household(metric_right, ax_right)

    fig.tight_layout()
    timeseries_filename = f"simulation_{scenario_label_ts}_timeseries.png"
    fig.savefig(timeseries_filename, dpi=150)

    # ------------------- Sector-level bottleneck plot ------------------- #
    n_sec = len(unique_sectors)
    if n_sec > 0:
        fig_bt, axes_bt = plt.subplots(n_sec, 1, figsize=(12, 3 * n_sec), sharex=True)
        if n_sec == 1:
            axes_bt = [axes_bt]

        bt_colors = {"labor": "#1f77b4", "capital": "#d62728", "input": "#2ca02c"}

        for ax_sec, sector in zip(axes_bt, unique_sectors):
            # Counts per bottleneck
            series_bt = {}
            for bt in ["labor", "capital", "input"]:
                counts = (
                    firm_df[(firm_df["sector"] == sector) & (firm_df["limiting_factor"] == bt)]
                    .groupby("Step").size().reindex(df.index, fill_value=0)
                )
                series_bt[bt] = counts

            totals = sum(series_bt.values())
            pct_arrays = [100 * series_bt[bt] / totals.replace(0, np.nan) for bt in ["labor", "capital", "input"]]

            ax_sec.stackplot(
                df[x_col],
                *pct_arrays,
                labels=["Labour", "Capital", "Input"],
                colors=[bt_colors["labor"], bt_colors["capital"], bt_colors["input"]],
                alpha=0.7,
            )

            ax_sec.set_title(f"{sector} – Production Bottlenecks (%)")
            ax_sec.set_ylabel("% of firms")
            ax_sec.set_ylim(0, 100)
            ax_sec.legend(fontsize=8, loc="upper center", ncol=3)

        axes_bt[-1].set_xlabel(x_col)
        fig_bt.tight_layout()
        bottleneck_filename = f"simulation_{scenario_label_ts}_sector_bottlenecks.png"
        fig_bt.savefig(bottleneck_filename, dpi=150)
        plt.close(fig_bt)

    plt.close(fig)

    print(f"Plots saved as {timeseries_filename} and {bottleneck_filename if n_sec > 0 else 'no sector plots'}")


if __name__ == "__main__":
    main() 
