"""Run prototype climate-economy ABM headlessly and persist results."""

from pathlib import Path
import argparse
from datetime import datetime
import subprocess
import numpy as np
import pandas as pd
# Import model and agent classes
from model import EconomyModel
from agents import FirmAgent, HouseholdAgent
from ensemble_utils import (
    ENSEMBLE_STAT_ORDER,
    METADATA_PREFIX,
    apply_metadata as apply_ensemble_metadata,
    build_ensemble_summary as summarize_ensemble,
    ensemble_seed_metadata as summarize_seed_metadata,
)
from hazard_utils import parse_hazard_event_specs
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


def _scenario_display(apply_hazards: bool, adaptation_enabled: bool) -> str:
    base = "Hazard" if apply_hazards else "Baseline"
    suffix = "Adaptation" if adaptation_enabled else "No Adaptation"
    return f"{base} + {suffix}"


def _scenario_label(apply_hazards: bool, adaptation_enabled: bool) -> str:
    parts = ["hazard" if apply_hazards else "baseline"]
    parts.append("adaptation" if adaptation_enabled else "noadaptation")
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


def _event_signature(events: list[tuple[int, int, int, str, str | None]]) -> str:
    return ";".join(
        f"{rp}:{start}:{end}:{haz_type}:{path if path is not None else 'None'}"
        for rp, start, end, haz_type, path in events
    )


def _base_metadata(
    *,
    args,
    events,
    apply_hazards: bool,
    adaptation_enabled: bool,
    adaptation_config: dict,
    scenario_label: str,
    timestamp: str,
) -> dict[str, object]:
    param_path = str(args.param_file) if args.param_file else ""
    topology_path = str(args.topology) if args.topology else ""
    return {
        f"{METADATA_PREFIX}ScenarioLabel": scenario_label,
        f"{METADATA_PREFIX}ApplyHazards": bool(apply_hazards),
        f"{METADATA_PREFIX}AdaptationEnabled": bool(adaptation_enabled),
        f"{METADATA_PREFIX}ParamFile": param_path,
        f"{METADATA_PREFIX}ParamFileStem": Path(param_path).stem if param_path else "",
        f"{METADATA_PREFIX}TopologyPath": topology_path,
        f"{METADATA_PREFIX}TopologyStem": Path(topology_path).stem if topology_path else "",
        f"{METADATA_PREFIX}HazardEventCount": len(events),
        f"{METADATA_PREFIX}HazardEvents": _event_signature(events),
        f"{METADATA_PREFIX}StartYear": int(args.start_year),
        f"{METADATA_PREFIX}StepsPerYear": int(args.steps_per_year),
        f"{METADATA_PREFIX}StepsRequested": int(args.steps),
        f"{METADATA_PREFIX}NumHouseholds": int(args.num_households),
        f"{METADATA_PREFIX}GridResolution": float(args.grid_resolution),
        f"{METADATA_PREFIX}HouseholdRelocation": bool(args.household_relocation),
        f"{METADATA_PREFIX}HHConsumptionPropensityIncome": float(HouseholdAgent.CONSUMPTION_PROPENSITY_INCOME),
        f"{METADATA_PREFIX}HHConsumptionPropensityWealth": float(HouseholdAgent.CONSUMPTION_PROPENSITY_WEALTH),
        f"{METADATA_PREFIX}HHTargetCashBuffer": float(HouseholdAgent.TARGET_CASH_BUFFER),
        f"{METADATA_PREFIX}FirmInventoryBufferRatio": float(FirmAgent.INVENTORY_BUFFER_RATIO),
        f"{METADATA_PREFIX}FirmLiquidityBufferRatio": float(FirmAgent.LIQUIDITY_BUFFER_RATIO),
        f"{METADATA_PREFIX}FirmMinLiquidityBuffer": float(FirmAgent.MIN_LIQUIDITY_BUFFER),
        f"{METADATA_PREFIX}FirmWorkingCapitalCreditRevenueShare": float(FirmAgent.WORKING_CAPITAL_CREDIT_REVENUE_SHARE),
        f"{METADATA_PREFIX}FirmLaborShare": float(FirmAgent.LABOR_SHARE),
        f"{METADATA_PREFIX}NoWorkerWagePremium": float(FirmAgent.NO_WORKER_WAGE_PREMIUM),
        f"{METADATA_PREFIX}UCB_C": float(adaptation_config.get("ucb_c", 1.0)),
        f"{METADATA_PREFIX}DecisionInterval": int(adaptation_config.get("decision_interval", 4)),
        f"{METADATA_PREFIX}RewardWindow": int(adaptation_config.get("reward_window", 4)),
        f"{METADATA_PREFIX}ObservationRadius": float(adaptation_config.get("observation_radius", 3.0)),
        f"{METADATA_PREFIX}ResilienceDecay": float(adaptation_config.get("resilience_decay", 0.01)),
        f"{METADATA_PREFIX}MaintenanceCostRate": float(adaptation_config.get("maintenance_cost_rate", 0.005)),
        f"{METADATA_PREFIX}LossReductionMax": float(adaptation_config.get("loss_reduction_max", 0.6)),
        f"{METADATA_PREFIX}MinMoneySurvival": float(adaptation_config.get("min_money_survival", 10.0)),
        f"{METADATA_PREFIX}ReplacementFrequency": int(adaptation_config.get("replacement_frequency", 10)),
        f"{METADATA_PREFIX}RunTimestamp": timestamp,
        f"{METADATA_PREFIX}GitCommit": _safe_git_commit(),
        f"{METADATA_PREFIX}SourceMemberFiles": "",
    }


def _ensemble_seed_metadata(seed_list: list[int]) -> dict[str, object]:
    return summarize_seed_metadata(seed_list)


def _apply_metadata(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    return apply_ensemble_metadata(df, metadata)


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
    events,
    apply_hazards: bool,
    adaptation_config: dict,
    seed: int,
    scenario_display: str,
):
    model = EconomyModel(
        num_households=args.num_households,
        num_firms=20,
        hazard_events=events,
        seed=seed,
        apply_hazard_impacts=apply_hazards,
        firm_topology_path=args.topology,
        start_year=args.start_year,
        steps_per_year=args.steps_per_year,
        adaptation_params=adaptation_config,
        consumption_ratios=args.consumption_ratios,
        grid_resolution=args.grid_resolution,
        household_relocation=args.household_relocation,
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
    return model, df, agent_df


def _build_ensemble_summary(member_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize member-level model outputs by step and scenario."""
    return summarize_ensemble(member_df, group_cols=["Scenario", "Step", "Year"])


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

    # ---------------- Optional parameter file ---------------------------- #
    if args.param_file:
        import json, pathlib

        param_path = pathlib.Path(args.param_file)
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
        if not hasattr(args, "steps_per_year"):
            args.steps_per_year = int(param_data.get("steps_per_year", 4))

        # 4. Topology path --------------------------------------------------
        if not args.topology and param_data.get("topology"):
            args.topology = str(param_data["topology"])
        
        # 5. Adaptation parameters ------------------------------------------
        args.adaptation_params = param_data.get("adaptation", param_data.get("learning", {}))

        # 6. Consumption ratios by sector -----------------------------------
        args.consumption_ratios = param_data.get("consumption_ratios", None)

        # 7. Number of households -------------------------------------------
        args.num_households = int(param_data.get("num_households", 100))

        # 8. Grid resolution (degrees per cell) -----------------------------
        args.grid_resolution = float(param_data.get("grid_resolution", 1.0))

        # 9. Household relocation toggle -------------------------------------
        args.household_relocation = bool(param_data.get("household_relocation", True))
        if not args.save_agent_ensemble and "save_agent_ensemble" in param_data:
            args.save_agent_ensemble = bool(param_data.get("save_agent_ensemble", False))
        if args.ensemble_plot_stat == "mean" and "ensemble_plot_stat" in param_data:
            args.ensemble_plot_stat = str(param_data.get("ensemble_plot_stat", "mean"))

    # Ensure we have at least one RP spec after merging param file -----------
    if not args.rp_file:
        raise SystemExit("No --rp-file entries provided and none found in parameter file.")

    # First, parse the RP files into a list irrespective of --viz so we can
    # pass them on to a potential Solara dashboard.
    # Parsed as (return_period, start_step, end_step, hazard_type, path|None)
    events: list[tuple[int, int, int, str, str | None]] = []
    if args.rp_file:
        try:
            events = parse_hazard_event_specs(args.rp_file)
        except ValueError as exc:  # noqa: BLE001
            raise SystemExit(str(exc)) from exc

    seed_list = _resolve_seed_list(args)
    if args.viz and len(seed_list) > 1:
        raise SystemExit("Multi-seed ensemble mode is not supported with --viz.")

    # If visualization requested, delegate to Solara which hosts the dashboard
    if args.viz:
        import subprocess, sys, os

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

    # Ensure steps_per_year attribute exists even if no param file
    if not hasattr(args, "steps_per_year"):
        args.steps_per_year = 4
    
    # Ensure adaptation_params exists even if no param file
    if not hasattr(args, "adaptation_params"):
        args.adaptation_params = {}

    # Ensure consumption_ratios exists even if no param file
    if not hasattr(args, "consumption_ratios"):
        args.consumption_ratios = None  # model will use default

    # Ensure num_households exists even if no param file
    if not hasattr(args, "num_households"):
        args.num_households = 100  # default

    # Ensure grid_resolution exists even if no param file
    if not hasattr(args, "grid_resolution"):
        args.grid_resolution = 1.0  # default 1 degree

    # Ensure household_relocation exists even if no param file
    if not hasattr(args, "household_relocation"):
        args.household_relocation = False  # default disabled

    # Configure scenario settings
    apply_hazards = not args.no_hazards
    disable_adaptation = bool(args.no_adaptation or args.no_learning)
    if disable_adaptation:
        adaptation_config = {**args.adaptation_params, "enabled": False}
    else:
        adaptation_config = args.adaptation_params
    adaptation_enabled = bool(adaptation_config.get("enabled", True))

    # Generate scenario label for output files
    scenario_label = _scenario_label(apply_hazards, adaptation_enabled)
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
    scenario_display = _scenario_display(apply_hazards, adaptation_enabled)
    metadata = {
        **_base_metadata(
            args=args,
            events=events,
            apply_hazards=apply_hazards,
            adaptation_enabled=adaptation_enabled,
            adaptation_config=adaptation_config,
            scenario_label=scenario_label,
            timestamp=timestamp,
        ),
        **_ensemble_seed_metadata(seed_list),
    }

    # Save results with scenario label + timestamp
    output_filename = f"simulation_{scenario_label_ts}"
    main_csv_path = f"{output_filename}.csv"
    agent_csv_path = f"{output_filename}_agents.csv"

    if len(seed_list) > 1:
        member_frames = []
        agent_member_frames = []
        for seed in seed_list:
            _, seed_df, seed_agent_df = _run_single_simulation(
                args=args,
                events=events,
                apply_hazards=apply_hazards,
                adaptation_config=adaptation_config,
                seed=seed,
                scenario_display=scenario_display,
            )
            member_frames.append(seed_df)
            if args.save_agent_ensemble:
                agent_member_frames.append(seed_agent_df)

        member_df = pd.concat(member_frames, ignore_index=True)
        member_df = _apply_metadata(member_df, metadata)
        summary_df = _build_ensemble_summary(member_df)
        summary_df = _apply_metadata(summary_df, metadata)
        summary_df.to_csv(main_csv_path, index=False)

        members_csv_path = f"{output_filename}_members.csv"
        member_df.to_csv(members_csv_path, index=False)

        if args.save_agent_ensemble and agent_member_frames:
            agent_member_df = pd.concat(agent_member_frames, ignore_index=True)
            agent_member_df = _apply_metadata(agent_member_df, metadata)
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

    model, df, agent_df = _run_single_simulation(
        args=args,
        events=events,
        apply_hazards=apply_hazards,
        adaptation_config=adaptation_config,
        seed=seed_list[0],
        scenario_display=scenario_display,
    )
    df = _apply_metadata(df, metadata)
    agent_df = _apply_metadata(agent_df, metadata)

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
