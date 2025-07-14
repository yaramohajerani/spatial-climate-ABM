"""Run prototype climate-economy ABM for 10 timesteps and persist results."""

import argparse
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

        # 4. Topology path --------------------------------------------------
        if not args.topology and param_data.get("topology"):
            args.topology = str(param_data["topology"])

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

    # Headless mode: run the simulation directly
    model = EconomyModel(
        num_households=100,
        num_firms=20,
        shock_step=5,
        hazard_events=events,
        seed=args.seed,
        firm_topology_path=args.topology,
        start_year=args.start_year,
    )

    for _ in range(args.steps):
        model.step()

    model.save_results("simulation_results")

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

    firm_metrics = [
        "Firm_Production",
        "Firm_Consumption",
        "Firm_Wealth",
        "Firm_Capital",
        "Mean_Price",
    ]

    household_metrics = [
        "Household_Labor_Sold",
        "Household_Consumption",
        "Household_Wealth",
        "Mean_Wage",
        "Average_Risk",
    ]

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

    levels_sorted = sorted(firm_df["Level"].dropna().unique())
    cmap_levels = plt.cm.viridis

    # --------------- Plotting configuration ----------------------------- #
    bottleneck_types = [
        ("Capital_Limited_Firms", "Capital"),  # for left column
        ("Labor_Limited_Firms", "Labor"),     # for right column
        ("Input_Limited_Firms", "Input"),     # for right column
    ]

    # Add sector-based pricing to firm metrics
    unique_sectors = sorted(firm_df["sector"].dropna().unique())
    unique_hh_sectors = sorted(household_df["sector"].dropna().unique())

    metrics_left = [
        "Firm_Production",
        "Firm_Consumption",
        "Firm_Wealth",
        "Firm_Capital",
        "Firm_Inventory",
        "Mean_Price",
        "Capital_Bottleneck",
    ]

    metrics_right = [
        "Household_Labor_Sold",
        "Household_Consumption",
        "Household_Wealth",
        "Average_Risk",
        "Wage_By_Level",
        "Labor_Bottleneck",
        "Input_Bottleneck",
    ]

    rows = len(metrics_left)
    fig = plt.figure(figsize=(15, rows * 3))
    gs = gridspec.GridSpec(rows, 2, height_ratios=[1]*(rows))
    axes_matrix = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(rows)]

    x_col = "Year" if args.start_year else "Step"
    if args.start_year:
        df["Year"] = args.start_year + df.index.astype(int) / 4

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
            # Plot aggregate mean price
            ax.plot(df[x_col], df[col], color="black", linewidth=2, label="Mean Price")
            
            # Plot price by sector
            sector_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_sectors)))
            for idx_sec, sector in enumerate(unique_sectors):
                sector_data = firm_df[firm_df["sector"] == sector]
                if not sector_data.empty:
                    price_by_step = sector_data.groupby("Step")["price"].mean()
                    x_vals = price_by_step.index if not args.start_year else args.start_year + price_by_step.index.astype(int) / 4
                    ax.plot(x_vals, price_by_step.values, color=sector_colors[idx_sec], 
                           linestyle="--", alpha=0.7, label=f"{sector}")
        else:
            agent_col = firm_metric_map.get(col, col.lower())
            for idx_lvl, lvl in enumerate(levels_sorted):
                grp = firm_df[firm_df["Level"] == lvl].groupby("Step")[agent_col].sum()
                x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int) / 4
                color = cmap_levels(idx_lvl / max(1, len(levels_sorted) - 1))
                ax.plot(x_vals, grp.values, label=f"Level {lvl}", color=color)
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
        # Aggregate overall line
        ax.plot(df[x_col], df[col], color="black", linewidth=2, label="All Households")

        hh_col = household_metric_map.get(col, None)
        if hh_col:
            sector_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_hh_sectors)))
            for idx_sec, sector in enumerate(unique_hh_sectors):
                grp = household_df[household_df["sector"] == sector].groupby("Step")[hh_col].sum()
                if grp.empty:
                    continue
                x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int) / 4
                ax.plot(x_vals, grp.values, color=sector_colors[idx_sec], linestyle="--", alpha=0.7, label=f"{sector}")

        ax.set_title(col.replace("_", " "), fontsize=10)
        ylabel = units.get(col, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    # ---------------- Plotting row by row ----------------------------- #
    def _plot_bottleneck(btype_str: str, ax):
        for idx_lvl, lvl in enumerate(levels_sorted):
            mask = (firm_df["Level"] == lvl) & (firm_df["limiting_factor"] == btype_str.lower())
            counts = firm_df[mask].groupby("Step").size()
            if counts.empty:
                continue
            x_vals = counts.index if not args.start_year else args.start_year + counts.index.astype(int) / 4
            color = cmap_levels(idx_lvl / max(1, len(levels_sorted) - 1))
            ax.plot(x_vals, counts.values, label=f"Level {lvl}", color=color)
        ax.set_ylabel("count")
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    def _plot_wage(ax):
        sector_colors = plt.cm.Set1(np.linspace(0, 1, len(levels_sorted)))
        for idx_lvl, lvl in enumerate(levels_sorted):
            wage_by_step = firm_df[firm_df["Level"] == lvl].groupby("Step")["wage"].mean()
            if wage_by_step.empty:
                continue
            x_vals = wage_by_step.index if not args.start_year else args.start_year + wage_by_step.index.astype(int) / 4
            color = cmap_levels(idx_lvl / max(1, len(levels_sorted) - 1))
            ax.plot(x_vals, wage_by_step.values, label=f"Level {lvl}", color=color)
        ax.set_title("Wages by Level", fontsize=10)
        ax.set_ylabel("$ / Unit of Labor")
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    for r in range(rows):
        # Left column
        metric_left = metrics_left[r]
        ax_left = axes_matrix[r][0]
        if metric_left.endswith("_Bottleneck"):
            if metric_left.startswith("Capital"):
                ax_left.set_title("Capital Bottlenecks", fontsize=10)
                _plot_bottleneck("capital", ax_left)
        else:
            _plot_firm(metric_left, ax_left)

        # Right column
        metric_right = metrics_right[r]
        ax_right = axes_matrix[r][1]
        if metric_right == "Wage_By_Level":
            _plot_wage(ax_right)
        elif metric_right.endswith("_Bottleneck"):
            if metric_right.startswith("Labor"):
                ax_right.set_title("Labor Bottlenecks", fontsize=10)
                _plot_bottleneck("labor", ax_right)
            elif metric_right.startswith("Input"):
                ax_right.set_title("Input Bottlenecks", fontsize=10)
                _plot_bottleneck("input", ax_right)
        else:
            _plot_household(metric_right, ax_right)

    fig.tight_layout()
    fig.savefig("simulation_timeseries.png", dpi=150)
    plt.close(fig)

    print("Simulation complete. Results stored in simulation_results.csv and simulation_timeseries.png")


if __name__ == "__main__":
    main() 