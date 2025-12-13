"""Run prototype climate-economy ABM for 10 timesteps and persist results."""

from pathlib import Path
import argparse
from datetime import datetime
# Import model and agent classes
from model import EconomyModel
from agents import FirmAgent
# Runner now expects one or more --rp-file arguments in the form
# "<RP>:<START_STEP>:<END_STEP>:<TYPE>:<path>"

def _parse():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:START:END:TYPE:PATH",
        help=(
            "Add a GeoTIFF file. Format: <RP>:<START_STEP>:<END_STEP>:<HAZARD_TYPE>:<path>. "
            "Required unless provided via --param-file. "
            "Example: --rp-file 100:1:20:FL:rp100_2030.tif"
        ),
    )
    p.add_argument("--viz", action="store_true", help="Launch interactive Solara dashboard instead of headless run")
    p.add_argument("--steps", type=int, default=10, help="Number of timesteps to simulate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--start-year", type=int, default=0, help="Base calendar year for step 0 (optional; used for plotting)")
    p.add_argument("--topology", type=str, help="Optional JSON file describing firm supply-chain topology")
    p.add_argument(
        "--param-file",
        type=str,
        help="Path to a JSON file containing parameter overrides. Keys can include rp_files (list), viz (bool), seed (int), topology (str).",
    )
    p.add_argument("--no-hazards", action="store_true", help="Run baseline scenario without hazard impacts")
    p.add_argument("--no-learning", action="store_true", help="Disable evolutionary learning in firms")
    return p.parse_args()

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
        
        # 5. Learning parameters --------------------------------------------
        args.learning_params = param_data.get("learning", {})

        # 6. Consumption ratios by sector -----------------------------------
        args.consumption_ratios = param_data.get("consumption_ratios", None)

        # 7. Number of households -------------------------------------------
        args.num_households = int(param_data.get("num_households", 100))

    # Ensure we have at least one RP spec after merging param file -----------
    if not args.rp_file:
        raise SystemExit("No --rp-file entries provided and none found in parameter file.")

    # First, parse the RP files into a list irrespective of --viz so we can
    # pass them on to a potential Solara dashboard.
    # Parsed as (return_period, start_step, end_step, hazard_type, path)
    events: list[tuple[int, int, int, str, str]] = []
    if args.rp_file:
        for item in args.rp_file:
            try:
                rp_str, start_str, end_str, type_str, path_str = item.split(":", 4)
                events.append((int(rp_str), int(start_str), int(end_str), type_str, path_str))
            except ValueError as exc:  # noqa: BLE001
                raise SystemExit(
                    (
                        f"Invalid --rp-file format: {item}. Expected "
                        "<RP>:<START>:<END>:<TYPE>:<path>."
                    )
                ) from exc

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
    
    # Ensure learning_params exists even if no param file
    if not hasattr(args, "learning_params"):
        args.learning_params = {}

    # Ensure consumption_ratios exists even if no param file
    if not hasattr(args, "consumption_ratios"):
        args.consumption_ratios = None  # model will use default

    # Ensure num_households exists even if no param file
    if not hasattr(args, "num_households"):
        args.num_households = 100  # default

    # Configure scenario settings
    apply_hazards = not args.no_hazards
    if args.no_learning:
        learning_config = {**args.learning_params, "enabled": False}
    else:
        learning_config = args.learning_params
    learning_enabled = bool(learning_config.get("enabled", True))

    # Generate scenario label for output files
    scenario_parts = []
    if apply_hazards:
        scenario_parts.append("hazard")
    else:
        scenario_parts.append("baseline")
    
    if learning_enabled:
        scenario_parts.append("learning")
    else:
        scenario_parts.append("nolearning")
    
    scenario_label = "_".join(scenario_parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topo_tag = ""
    if args.topology:
        topo_tag = Path(args.topology).stem
    scenario_label_ts = f"{scenario_label}_{topo_tag}_{timestamp}" if topo_tag else f"{scenario_label}_{timestamp}"

    # Headless mode: run the simulation directly
    model = EconomyModel(
        num_households=args.num_households,
        num_firms=20,
        hazard_events=events,
        seed=args.seed,
        apply_hazard_impacts=apply_hazards,
        firm_topology_path=args.topology,
        start_year=args.start_year,
        steps_per_year=args.steps_per_year,
        learning_params=learning_config,
        consumption_ratios=args.consumption_ratios,
    )

    for _ in range(args.steps):
        model.step()

    # Add scenario information to the dataframes before saving
    df = model.results_to_dataframe()
    
    # Create scenario label for the data
    if apply_hazards:
        scenario_display = "Hazard"
    else:
        scenario_display = "Baseline"
    
    scenario_display += " + No Learning" if not learning_enabled else " + Learning"
    
    df["Scenario"] = scenario_display
    df["Step"] = df.index
    
    # Also add scenario to agent data
    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")
    agent_df["Scenario"] = scenario_display
    
    # Save results with scenario label + timestamp
    output_filename = f"simulation_{scenario_label_ts}"
    
    # Save main results
    main_csv_path = f"{output_filename}.csv"
    df.to_csv(main_csv_path, index=False)
    
    # Save agent results
    agent_csv_path = f"{output_filename}_agents.csv"
    agent_df.to_csv(agent_csv_path, index=False)
    
    print(f"Simulation complete for scenario: {scenario_label}")
    print(f"Results saved as {main_csv_path} and {agent_csv_path}")

    # ------------------------- Plotting ---------------------------- #
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    from trophic_utils import compute_trophic_levels

    df = model.results_to_dataframe()

    rename_map = {"Base_Wage": "Mean_Wage", "Household_LaborSold": "Household_Labor_Sold"}
    df = df.rename(columns=rename_map)

    units = {
        "Firm_Production": "Units of Goods",
        "Firm_Consumption": "Units of Goods",
        "Firm_Wealth": "$",
        "Firm_Capital": "Units of Capital",
        "Household_Wealth": "$",
        "Household_Capital": "Units of Capital",
        "Household_Labor_Sold": "Units of Labor",
        "Household_Consumption": "Units of Goods",
        "Average_Risk": "Score (0–1)",
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

    agent_df = model.datacollector.get_agent_vars_dataframe().reset_index()
    agent_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")

    # Split firm vs household for convenience
    firm_df = agent_df[agent_df["type"] == "FirmAgent"].copy()
    firm_df["Level"] = firm_df["AgentID"].map(lvl_map)

    household_df = agent_df[agent_df["type"] == "HouseholdAgent"].copy()

    # Sector palette
    unique_sectors = sorted(firm_df["sector"].dropna().unique())
    sec_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sectors)))

    # Single, shared colour map for both firm & household plots
    color_by_sector = {
        sector: sec_colors[idx % len(sec_colors)]
        for idx, sector in enumerate(unique_sectors)
    }

    unique_hh_sectors = sorted(household_df["sector"].dropna().unique())

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
        "Household_Capital": "capital",
        "Household_Labor_Sold": "labor_sold",
        "Household_Consumption": "consumption",
    }

    def _plot_household(col, ax):
        hh_col = household_metric_map.get(col, None)
        if hh_col:
            # Add mean line for all households
            mean_grp = household_df.groupby("Step")[hh_col].mean()
            x_vals = mean_grp.index if not args.start_year else args.start_year + mean_grp.index.astype(int) / args.steps_per_year
            ax.plot(x_vals, mean_grp.values, color="black", linewidth=2, label="Mean")
            # Sector breakdown
            for idx_sec, sector in enumerate(unique_hh_sectors):
                grp = household_df[household_df["sector"] == sector].groupby("Step")[hh_col].mean()
                if grp.empty:
                    continue
                x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int) / args.steps_per_year
                ax.plot(x_vals, grp.values, color=color_by_sector.get(sector, "grey"), linestyle="--", alpha=0.8, label=sector)
        else:
            # Fallback: plot aggregate series from df (e.g., Average_Risk)
            ax.plot(df[x_col], df[col], color="black", linewidth=2, label=col.replace("_", " "))

        ax.set_title(col.replace("_", " "), fontsize=10)
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
