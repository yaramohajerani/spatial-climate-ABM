import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from trophic_utils import compute_trophic_levels

from model import EconomyModel
from agents import FirmAgent


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run multiple simulations comparing different combinations of hazards and learning scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:START:END:TYPE:PATH",
        help=(
            "Add a GeoTIFF file in the form <RP>:<START_STEP>:<END_STEP>:<HAZARD_TYPE>:<path>. "
            "Can be used multiple times. Required unless --simulation_output is given."
        ),
    )
    p.add_argument("--steps", type=int, default=10, help="Number of timesteps / years to simulate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--start-year", type=int, default=0, help="Base calendar year for step 0 (used for x-axis)")
    p.add_argument("--out", type=str, default="comparison_plot.png", help="Output plot file")
    p.add_argument("--topology", type=str, help="Optional JSON file describing firm supply-chain topology")
    p.add_argument(
        "--param-file",
        type=str,
        help="Path to a JSON file containing parameter overrides (rp_files, steps, seed, topology).",
    )
    p.add_argument("--simulation_output", type=str, help="Existing CSV produced by a prior run; if given we skip running the model and only regenerate the plot")
    
    # Scenario selection - modular approach
    p.add_argument("--scenarios", nargs="+", 
                   choices=["baseline_learning", "baseline_nolearning", "hazard_learning", "hazard_nolearning"],
                   help="Specify exact scenarios to compare. Choose from: baseline_learning, baseline_nolearning, hazard_learning, hazard_nolearning")
    
    # Legacy flags for backward compatibility (deprecated)
    p.add_argument("--baseline", action="store_true", help="[DEPRECATED] Use --scenarios instead")
    p.add_argument("--hazard", action="store_true", help="[DEPRECATED] Use --scenarios instead")
    p.add_argument("--learning", action="store_true", help="[DEPRECATED] Use --scenarios instead")
    p.add_argument("--no-learning", action="store_true", help="[DEPRECATED] Use --scenarios instead")
    
    # Plotting options
    p.add_argument("--sectors", action="store_true", help="Show sector-level breakdown in plots")
    
    return p.parse_args()


def _parse_events(rp_files: list[str]):
    """Parse a list of RP file strings into tuples.

    Expected each item: "<RP>:<START>:<END>:<TYPE>:<PATH>".
    """

    events: list[tuple[int, int, int, str, str]] = []
    for item in rp_files:
        try:
            rp_str, start_str, end_str, type_str, path_str = item.split(":", 4)
            events.append((int(rp_str), int(start_str), int(end_str), type_str, path_str))
        except ValueError as exc:
            raise SystemExit(
                f"Invalid --rp-file format: {item}. Expected <RP>:<START>:<END>:<TYPE>:<path>."
            ) from exc
    return events


# ------------------------------------------------------------------ #
#                      Parameter file loader                         #
# ------------------------------------------------------------------ #


def _merge_param_file(args):  # noqa: ANN001
    """If --param-file given, merge its settings into the argparse result."""

    if not getattr(args, "param_file", None):
        return

    import json, pathlib

    pth = pathlib.Path(args.param_file)
    if not pth.exists():
        raise SystemExit(f"Parameter file not found: {pth}")

    try:
        data = json.loads(pth.read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to parse JSON parameter file: {pth}") from exc

    # rp_files --------------------------------------------------------
    file_rp = data.get("rp_files") or data.get("rp_file")
    if file_rp and not args.rp_file:
        args.rp_file = file_rp
    elif file_rp and args.rp_file:
        args.rp_file = file_rp + args.rp_file

    # steps -----------------------------------------------------------
    if args.steps == 10 and "steps" in data:
        args.steps = int(data["steps"])

    # steps_per_year --------------------------------------------------
    if not hasattr(args, "steps_per_year") and data.get("steps_per_year"):
        args.steps_per_year = int(data["steps_per_year"])

    # seed ------------------------------------------------------------
    if args.seed == 42 and "seed" in data:
        args.seed = int(data["seed"])

    # start_year ------------------------------------------------------
    if args.start_year == 0 and data.get("start_year"):
        args.start_year = int(data["start_year"])

    # topology --------------------------------------------------------
    if not args.topology and data.get("topology"):
        args.topology = str(data["topology"])
    
    # learning parameters ---------------------------------------------
    args.learning_params = data.get("learning", {})


def run_simulation(model: EconomyModel, n_steps: int):
    for _ in range(n_steps):
        model.step()
    return model.results_to_dataframe(), model  # return model for agent access


def main():
    args = _parse_args()

    # Merge optional config file
    _merge_param_file(args)
    
    # Ensure learning_params exists even if no param file
    if not hasattr(args, "learning_params"):
        args.learning_params = {}

    if not args.simulation_output and not args.rp_file:
        raise SystemExit("Either --rp-file or --simulation_output must be provided.")
    
    # Determine which scenarios to run
    if args.scenarios:
        # Use the new modular approach
        scenarios_to_run = []
        for scenario_name in args.scenarios:
            hazards = scenario_name.startswith("hazard")
            learning = scenario_name.endswith("_learning")  # Must end with "_learning", not just "learning"
            scenarios_to_run.append({
                "name": scenario_name,
                "hazards": hazards,
                "learning": learning
            })
    elif any([args.baseline, args.hazard, args.learning, args.no_learning]):
        # Legacy support for old flags
        print("Warning: Using deprecated flags. Please use --scenarios instead for future compatibility.")
        scenarios_to_run = []
        
        # Build scenario combinations based on legacy flags
        hazard_options = []
        if args.baseline:
            hazard_options.append(False)
        if args.hazard:
            hazard_options.append(True)
        
        learning_options = []
        if args.no_learning:
            learning_options.append(False)
        if args.learning:
            learning_options.append(True)
        
        # If no hazard options specified, include both
        if not hazard_options:
            hazard_options = [False, True]
        
        # If no learning options specified, include both
        if not learning_options:
            learning_options = [False, True]
        
        # Create all combinations
        for hazards in hazard_options:
            for learning in learning_options:
                name_parts = []
                if hazards:
                    name_parts.append("hazard")
                else:
                    name_parts.append("baseline")
                    
                if learning:
                    name_parts.append("learning")
                else:
                    name_parts.append("nolearning")
                    
                scenarios_to_run.append({
                    "name": "_".join(name_parts),
                    "hazards": hazards,
                    "learning": learning
                })
    else:
        # Default to baseline and hazard with learning (backward compatibility)
        scenarios_to_run = [
            {"name": "baseline_learning", "hazards": False, "learning": True},
            {"name": "hazard_learning", "hazards": True, "learning": True}
        ]

    if args.simulation_output:
        df_combined = pd.read_csv(args.simulation_output)
        print(f"Loaded data from {args.simulation_output}")
        
        # Also load the corresponding agent data file
        sim_path = Path(args.simulation_output)
        agent_path = sim_path.parent / f"{sim_path.stem}_agents.csv"
        if agent_path.exists():
            agent_df_combined = pd.read_csv(agent_path)
            print(f"Loaded agent data from {agent_path}")
        else:
            print(f"Warning: Agent data file not found at {agent_path}")
            print("Creating empty agent dataframe - some plots may not work correctly")
            agent_df_combined = pd.DataFrame()
    else:
        events = _parse_events(args.rp_file)
        
        # Run all specified scenarios
        scenario_dataframes = []
        scenario_models = []
        
        for scenario in scenarios_to_run:
            print(f"Running scenario: {scenario['name']} (hazards={scenario['hazards']}, learning={scenario['learning']})")
            
            # Configure learning parameters based on scenario
            if scenario['learning']:
                learning_config = args.learning_params
            else:
                learning_config = {**args.learning_params, "enabled": False}
            
            # Create and run model
            model = EconomyModel(
                num_households=100,
                num_firms=20,
                hazard_events=events,
                seed=args.seed,
                apply_hazard_impacts=scenario['hazards'],
                firm_topology_path=args.topology,
                start_year=args.start_year,
                steps_per_year=getattr(args, 'steps_per_year', 4),
                learning_params=learning_config,
            )
            
            df_scenario, model_ref = run_simulation(model, args.steps)
            
            # Create scenario label for plotting
            scenario_label_parts = []
            if scenario['hazards']:
                scenario_label_parts.append("Hazard")
            else:
                scenario_label_parts.append("Baseline")
                
            if scenario['learning']:
                scenario_label_parts.append("Learning")
            else:
                scenario_label_parts.append("No Learning")
                
            scenario_label = " + ".join(scenario_label_parts)
            df_scenario["Scenario"] = scenario_label
            df_scenario["Step"] = df_scenario.index
            
            print(f"  -> Created scenario with label: '{scenario_label}'")
            
            scenario_dataframes.append(df_scenario)
            scenario_models.append((model_ref, scenario['name']))

        # ---------------- Agent-level aggregation ---------------------- #
        def _compute_levels(model_obj: EconomyModel):
            """Return mapping firm_id->trophic level using weighted definition."""
            firms = [ag for ag in model_obj.agents if isinstance(ag, FirmAgent)]
            adjacency = {f.unique_id: [s.unique_id for s in f.connected_firms] for f in firms}
            return compute_trophic_levels(adjacency)

        # Build agent DataFrame combined
        def _agent_df(model_obj, scenario_label):
            df_ag = model_obj.datacollector.get_agent_vars_dataframe().reset_index()
            df_ag.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")

            # Assign levels only to firms; households keep NaN
            lvl_map = _compute_levels(model_obj)
            df_ag["Level"] = df_ag["AgentID"].map(lvl_map)
            df_ag["Scenario"] = scenario_label
            return df_ag

        # Process agent data for all scenarios
        agent_dataframes = []
        for i, (model_ref, scenario_name) in enumerate(scenario_models):
            scenario_label = scenario_dataframes[i]["Scenario"].iloc[0]  # Get the display label
            agent_df = _agent_df(model_ref, scenario_label)
            agent_dataframes.append(agent_df)
        
        agent_df_combined = pd.concat(agent_dataframes, ignore_index=True)
        
        # Combine all scenario dataframes
        df_combined = pd.concat(scenario_dataframes, ignore_index=True)
        
        # Generate output filename suffix based on scenarios
        scenario_names = sorted([s['name'] for s in scenarios_to_run])
        filename_suffix = "_".join(scenario_names)
        
        # Update output path to include scenario information
        out_path = Path(args.out)
        if out_path.stem == "comparison_plot":  # Default name
            new_stem = f"comparison_{filename_suffix}"
        else:
            new_stem = f"{out_path.stem}_{filename_suffix}"
        
        # Update args.out for later use
        args.out = str(out_path.parent / f"{new_stem}{out_path.suffix}")

        # Persist rich agent-level results
        agent_csv_path = out_path.parent / f"{new_stem}_agents.csv"
        agent_df_combined.to_csv(agent_csv_path, index=False)
        print(f"Agent-level data saved to {agent_csv_path}")

        # Persist combined DataFrame
        csv_path = out_path.parent / f"{new_stem}.csv"
        df_combined.to_csv(csv_path, index=False)
        print(f"Combined results saved to {csv_path}")

    # renaming for readability and polish of plot
    rename_map = {
        "Base_Wage": "Mean_Wage",
        "Household_LaborSold": "Household_Labor_Sold",
    }
    df_combined = df_combined.rename(columns=rename_map)

    # Units for y-axis labels
    units = {
        "Firm_Production": "Units of Goods",
        "Firm_Consumption": "Units of Goods",
        "Firm_Wealth": "$",
        "Firm_Capital": "Units of Capital",
        "Household_Wealth": "$",
        "Household_Capital": "Units of Capital",
        "Household_Labor_Sold": "Units of Labor",
        "Household_Consumption": "Units of Goods",
        "Average_Risk": "Flood Depth (m)",
        "Mean_Wage": "$ / Unit of Labor",
        "Mean_Price": "$ / Unit of Goods",
        "Labor_Limited_Firms": "count",
        "Capital_Limited_Firms": "count",
        "Input_Limited_Firms": "count",
        "Average_Firm_Fitness": "Score (0–1)",
        "Firm_Replacements": "count",
    }

    # Get unique scenarios and count them first
    unique_scenarios = sorted(df_combined["Scenario"].unique())
    n_scenarios = len(unique_scenarios)
    
    # ---- Metric selection - adapt bottleneck plots based on number of scenarios ---- #
    metrics_left = [
        "Firm_Production",
        "Firm_Consumption",
        "Firm_Wealth",
        "Firm_Capital",
        "Mean_Price",
        "Average_Risk",
        "Average_Firm_Fitness",
    ]

    metrics_right = [
        "Household_Labor_Sold",
        "Household_Consumption",
        "Household_Wealth",
        "Mean_Wage",
        "Firm_Replacements",
        "Bottleneck_Hazard",
        "Bottleneck_Baseline",
    ]
    
    # If we have more than 2 scenarios, replace some plots with additional bottleneck plots
    if n_scenarios > 2:
        # Replace Average_Risk and Average_Firm_Fitness with bottleneck plots for each scenario
        hazard_scenarios = [s for s in unique_scenarios if "Hazard" in s]
        baseline_scenarios = [s for s in unique_scenarios if "Baseline" in s]
        
        # Create bottleneck plot names for each scenario
        bottleneck_plots = []
        for scenario in unique_scenarios:
            safe_name = scenario.replace(" + ", "_").replace(" ", "_")
            bottleneck_plots.append(f"Bottleneck_{safe_name}")
        
        # Replace the last plots in left column with bottleneck plots
        if len(bottleneck_plots) >= 2:
            metrics_left[-2:] = bottleneck_plots[:2]  # Replace Average_Risk and Average_Firm_Fitness
        if len(bottleneck_plots) >= 4:
            metrics_right[-2:] = bottleneck_plots[2:4]  # Replace the last two right plots if needed

    rows = len(metrics_left)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, rows * 3))
    gs = gridspec.GridSpec(rows, 2, height_ratios=[1]*rows)
    axes_matrix = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(rows)]

    # Ensure axes is 2-D even if rows == 1
    if rows == 1:
        axes_matrix = axes_matrix.reshape(1, 2)

    # Helper to plot one metric
    x_col = "Year" if args.start_year else "Step"

    # Ensure steps_per_year attribute
    if not hasattr(args, "steps_per_year"):
        args.steps_per_year = 4

    if args.start_year:
        df_combined["Year"] = args.start_year + df_combined["Step"].astype(int) / args.steps_per_year

    # Separate combined agent data for convenience
    firm_agents_df = agent_df_combined[agent_df_combined["type"] == "FirmAgent"].copy()
    household_agents_df = agent_df_combined[agent_df_combined["type"] == "HouseholdAgent"].copy()

    unique_sectors = sorted(firm_agents_df["sector"].dropna().unique())
    sec_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sectors)))

    firm_metric_map = {
        "Firm_Production": "production",
        "Firm_Consumption": "consumption",
        "Firm_Wealth": "money",
        "Firm_Capital": "capital",
    }
    
    # Mapping for agent-level learning metrics
    firm_learning_metric_map = {
        "Average_Firm_Fitness": "fitness",
    }

    household_metric_map = {
        "Household_Wealth": "money",
        "Household_Capital": "capital",
        "Household_Labor_Sold": "labor_sold",
        "Household_Consumption": "consumption",
    }

    # Create color/style mappings
    
    # Create consistent color and style mapping: hazard=red, baseline=blue, learning=dashed, no-learning=solid
    scenario_style_map = {}
    for scenario in unique_scenarios:
        # Determine color based on hazard vs baseline
        if "Hazard" in scenario:
            color = "red"
        else:  # Baseline
            color = "blue"
        
        # Determine line style based on learning
        if "No Learning" in scenario:
            linestyle = "-"  # solid
        else:  # Learning
            linestyle = ":"  # dashed
            
        scenario_style_map[scenario] = {"color": color, "linestyle": linestyle}
    
    # Create sector color palettes (shades of red for hazard, shades of blue for baseline)
    sector_colors_hazard = ["#8B0000", "#CD5C5C", "#DC143C", "#FF6347", "#FF4500"]  # dark to light red shades
    sector_colors_baseline = ["#000080", "#4169E1", "#1E90FF", "#87CEEB", "#ADD8E6"]  # dark to light blue shades

    def _get_sector_color(scenario: str, sector_index: int) -> str:
        """Get the appropriate color for a sector based on scenario type."""
        if "Hazard" in scenario:
            return sector_colors_hazard[sector_index % len(sector_colors_hazard)]
        else:
            return sector_colors_baseline[sector_index % len(sector_colors_baseline)]
    
    def _plot_sector_lines(ax, df_agents, metric_col, scenarios, sectors, x_col, args, agent_type="firm"):
        """Plot sector-level lines for a given metric."""
        if not args.sectors:
            return
            
        for idx_sec, sector in enumerate(sectors):
            for scenario in scenarios:
                df_sec = df_agents[(df_agents["sector"] == sector) & (df_agents["Scenario"] == scenario)]
                if df_sec.empty:
                    continue
                
                values_by_step = df_sec.groupby("Step")[metric_col].mean()
                
                if values_by_step.empty:
                    continue
                    
                x_vals = values_by_step.index if not args.start_year else args.start_year + values_by_step.index.astype(int) / args.steps_per_year
                style = scenario_style_map[scenario]
                sector_color = _get_sector_color(scenario, idx_sec)
                
                ax.plot(x_vals, values_by_step.values, 
                       linestyle=style["linestyle"], color=sector_color, alpha=0.7,
                       label=f"{sector} – {scenario}")

    def _plot_metric(metric_name: str, ax):
        if metric_name == "Mean_Price":
            # Aggregate line per scenario
            for scenario, grp in df_combined.groupby("Scenario"):
                style = scenario_style_map[scenario]
                ax.plot(grp[x_col], grp[metric_name], 
                       color=style["color"], linestyle=style["linestyle"], 
                       label=f"Mean – {scenario}", linewidth=2)
            
            # Sector breakdown (only if --sectors flag is used)
            sectors = sorted(firm_agents_df["sector"].dropna().unique())
            _plot_sector_lines(ax, firm_agents_df, "price", unique_scenarios, sectors, x_col, args)
        elif metric_name == "Mean_Wage":
            # Aggregate mean wage per scenario
            for scenario, grp in df_combined.groupby("Scenario"):
                style = scenario_style_map[scenario]
                ax.plot(grp[x_col], grp[metric_name], 
                       color=style["color"], linestyle=style["linestyle"], 
                       label=f"Mean – {scenario}", linewidth=2)

            # Sector‐level wage lines from firm data (only if --sectors flag is used)
            sectors = sorted(firm_agents_df["sector"].dropna().unique())
            _plot_sector_lines(ax, firm_agents_df, "wage", unique_scenarios, sectors, x_col, args)
        elif metric_name == "Sector_Trophic_Level":
            sectors = sorted(firm_agents_df["sector"].dropna().unique())
            _plot_sector_lines(ax, firm_agents_df, "Level", unique_scenarios, sectors, x_col, args)
            ax.set_ylabel("trophic level")
        elif metric_name in firm_metric_map:
            agent_col = firm_metric_map[metric_name]
            # Add mean lines for each scenario
            for scenario in unique_scenarios:
                df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                mean_grp = df_scen.groupby("Step")[agent_col].mean()
                if not mean_grp.empty:
                    x_vals = mean_grp.index if not args.start_year else args.start_year + mean_grp.index.astype(int)/args.steps_per_year
                    style = scenario_style_map[scenario]
                    ax.plot(x_vals, mean_grp.values, 
                           color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                           label=f"Mean – {scenario}")
            
            # Sector breakdown (only if --sectors flag is used)
            if args.sectors:
                for scenario in unique_scenarios:
                    df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    style = scenario_style_map[scenario]
                    for idx_sec, sector in enumerate(unique_sectors):
                        grp = df_scen[df_scen["sector"] == sector].groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                        x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int)/args.steps_per_year
                        sector_color = _get_sector_color(scenario, idx_sec)
                        
                        ax.plot(x_vals, grp.values, 
                               color=sector_color, linestyle=style["linestyle"], 
                               alpha=0.7, label=f"{sector} – {scenario}")
        elif metric_name in firm_learning_metric_map:
            # Learning metrics (fitness, etc.) - agent level
            agent_col = firm_learning_metric_map[metric_name]
            # Add mean lines for each scenario
            for scenario in unique_scenarios:
                df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                mean_grp = df_scen.groupby("Step")[agent_col].mean()
                if not mean_grp.empty:
                    x_vals = mean_grp.index if not args.start_year else args.start_year + mean_grp.index.astype(int)/args.steps_per_year
                    style = scenario_style_map[scenario]
                    ax.plot(x_vals, mean_grp.values, 
                           color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                           label=f"Mean – {scenario}")
            
            # Sector breakdown (only if --sectors flag is used)
            if args.sectors:
                for scenario in unique_scenarios:
                    df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    style = scenario_style_map[scenario]
                    for idx_sec, sector in enumerate(unique_sectors):
                        grp = df_scen[df_scen["sector"] == sector].groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                        x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int)/args.steps_per_year
                        sector_color = _get_sector_color(scenario, idx_sec)
                        
                        ax.plot(x_vals, grp.values, 
                               color=sector_color, linestyle=style["linestyle"], 
                               alpha=0.7, label=f"{sector} – {scenario}")
        else:
            # Household metrics
            if metric_name in household_metric_map:
                hh_col = household_metric_map[metric_name]
                sectors = sorted(household_agents_df["sector"].dropna().unique())
                # Add mean lines for each scenario
                for scenario in unique_scenarios:
                    df_scen_hh = household_agents_df[household_agents_df["Scenario"] == scenario]
                    mean_grp = df_scen_hh.groupby("Step")[hh_col].mean()
                    if not mean_grp.empty:
                        x_vals = mean_grp.index if not args.start_year else args.start_year + mean_grp.index.astype(int) / args.steps_per_year
                        style = scenario_style_map[scenario]
                        ax.plot(x_vals, mean_grp.values, 
                               color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                               label=f"Mean – {scenario}")
                
                # Sector breakdown (only if --sectors flag is used)
                if args.sectors:
                    for scenario in unique_scenarios:
                        df_scen_hh = household_agents_df[household_agents_df["Scenario"] == scenario]
                        style = scenario_style_map[scenario]
                        for idx_sec, sector in enumerate(sectors):
                            grp = df_scen_hh[df_scen_hh["sector"] == sector].groupby("Step")[hh_col].mean()
                            if grp.empty:
                                continue
                            x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int) / args.steps_per_year
                            sector_color = _get_sector_color(scenario, idx_sec)
                            
                            ax.plot(x_vals, grp.values, 
                                   color=sector_color, linestyle=style["linestyle"], 
                                   alpha=0.7, label=f"{sector} – {scenario}")
            else:
                # Other aggregated metrics (risk, etc.)
                for scenario, grp in df_combined.groupby("Scenario"):
                    style = scenario_style_map[scenario]
                    ax.plot(grp[x_col], grp[metric_name], 
                           color=style["color"], linestyle=style["linestyle"], 
                           label=f"Mean – {scenario}", linewidth=2)

        ax.set_title(metric_name.replace("_", " "), fontsize=10)
        ylabel = units.get(metric_name, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    # Fill grid ----------------------------------------------------------- #
    subplot_labels = [chr(ord('a') + i) for i in range(rows * 2)]  # Generate a, b, c, ...
    
    for r in range(rows):
        # Left metrics
        if not metrics_left[r].startswith("Bottleneck_"):
            _plot_metric(metrics_left[r], axes_matrix[r][0])
        else:
            # Handle bottleneck plots in left column too
            ax_bt = axes_matrix[r][0]
            steps = df_combined["Step"].unique()
            
            if args.start_year:
                x_vals = args.start_year + steps.astype(int) / args.steps_per_year
            else:
                x_vals = steps

            def _pct_arrays(df_sub):
                arrs = {}
                for bt in ["labor", "capital", "input"]:
                    cnt = df_sub[df_sub["limiting_factor"] == bt].groupby("Step").size()
                    cnt = cnt.reindex(steps, fill_value=0)
                    arrs[bt] = cnt
                tot = sum(arrs.values())
                tot[tot == 0] = 1
                return [100 * arrs[bt] / tot for bt in ["labor", "capital", "input"]]

            # Extract scenario name from metric name
            scenario_part = metrics_left[r].replace("Bottleneck_", "").replace("_", " + ")
            matching_scenarios = [s for s in unique_scenarios if s == scenario_part]
            if matching_scenarios:
                df_sub = firm_agents_df[firm_agents_df["Scenario"] == matching_scenarios[0]]
                label_suffix = f"({matching_scenarios[0]})"
            else:
                # If no exact match, try to map to the correct scenario by index
                # This ensures each bottleneck plot gets a unique scenario
                scenario_index = r - len([m for m in metrics_left[:r] if not m.startswith("Bottleneck_")])
                if scenario_index < len(unique_scenarios):
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == unique_scenarios[scenario_index]]
                    label_suffix = f"({unique_scenarios[scenario_index]})"
                else:
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == unique_scenarios[0]]
                    label_suffix = f"({unique_scenarios[0]})"

            pct_arrays = _pct_arrays(df_sub)
            ax_bt.stackplot(x_vals, *pct_arrays,
                            labels=[f"Labour {label_suffix}", f"Capital {label_suffix}", f"Input {label_suffix}"],
                            colors=["#1f77b4", "#d62728", "#2ca02c"], alpha=0.7)

            ax_bt.set_title(f"Production Bottlenecks % {label_suffix}", fontsize=10)
            ax_bt.set_ylabel("% of firms")
            ax_bt.set_xlabel(x_col)
            ax_bt.set_ylim(0, 100)
            ax_bt.legend(fontsize=6, ncol=3)
        # Add subplot label
        axes_matrix[r][0].text(-0.1, 1.02, f'({subplot_labels[r * 2]})', transform=axes_matrix[r][0].transAxes, 
                              fontsize=12, fontweight='bold', va='bottom', ha='right')

        # Right metrics
        if not metrics_right[r].startswith("Bottleneck_"):
            _plot_metric(metrics_right[r], axes_matrix[r][1])
        else:
            # Dedicated stacked bottleneck plots
            ax_bt = axes_matrix[r][1]

            steps = df_combined["Step"].unique()
            
            # Convert steps to years for x-axis if start_year is provided
            if args.start_year:
                x_vals = args.start_year + steps.astype(int) / args.steps_per_year
            else:
                x_vals = steps

            def _pct_arrays(df_sub):
                arrs = {}
                for bt in ["labor", "capital", "input"]:
                    cnt = df_sub[df_sub["limiting_factor"] == bt].groupby("Step").size()
                    cnt = cnt.reindex(steps, fill_value=0)
                    arrs[bt] = cnt
                tot = sum(arrs.values())
                tot[tot == 0] = 1
                return [100 * arrs[bt] / tot for bt in ["labor", "capital", "input"]]

            # Determine which scenario to plot based on the metric name
            if metrics_right[r] == "Bottleneck_Hazard":
                hazard_scenarios = [s for s in unique_scenarios if "Hazard" in s and "Baseline" not in s]
                if hazard_scenarios:
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == hazard_scenarios[0]]
                    label_suffix = f"({hazard_scenarios[0]})"
                else:
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == unique_scenarios[0]]
                    label_suffix = f"({unique_scenarios[0]})"
            elif metrics_right[r] == "Bottleneck_Baseline":
                baseline_scenarios = [s for s in unique_scenarios if "Baseline" in s]
                if baseline_scenarios:
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == baseline_scenarios[0]]
                    label_suffix = f"({baseline_scenarios[0]})"
                else:
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == unique_scenarios[0]]
                    label_suffix = f"({unique_scenarios[0]})"
            else:
                # Dynamic bottleneck plot - extract scenario name from metric name
                scenario_part = metrics_right[r].replace("Bottleneck_", "").replace("_", " + ")
                matching_scenarios = [s for s in unique_scenarios if s == scenario_part]
                if matching_scenarios:
                    df_sub = firm_agents_df[firm_agents_df["Scenario"] == matching_scenarios[0]]
                    label_suffix = f"({matching_scenarios[0]})"
                else:
                    # If no exact match, try to map to the correct scenario by index
                    # This ensures each bottleneck plot gets a unique scenario
                    scenario_index = r - len([m for m in metrics_right[:r] if not m.startswith("Bottleneck_")])
                    if scenario_index < len(unique_scenarios):
                        df_sub = firm_agents_df[firm_agents_df["Scenario"] == unique_scenarios[scenario_index]]
                        label_suffix = f"({unique_scenarios[scenario_index]})"
                    else:
                        df_sub = firm_agents_df[firm_agents_df["Scenario"] == unique_scenarios[0]]
                        label_suffix = f"({unique_scenarios[0]})"

            # Calculate percentage arrays for the selected scenario
            pct_arrays = _pct_arrays(df_sub)

            ax_bt.stackplot(x_vals, *pct_arrays,
                            labels=[f"Labour {label_suffix}", f"Capital {label_suffix}", f"Input {label_suffix}"],
                            colors=["#1f77b4", "#d62728", "#2ca02c"], alpha=0.7)

            ax_bt.set_title(f"Production Bottlenecks % {label_suffix}", fontsize=10)
            ax_bt.set_ylabel("% of firms")
            ax_bt.set_xlabel(x_col)
            ax_bt.set_ylim(0, 100)
            ax_bt.legend(fontsize=6, ncol=3)
            
        # Add subplot label for right column
        axes_matrix[r][1].text(-0.1, 1.02, f'({subplot_labels[r * 2 + 1]})', transform=axes_matrix[r][1].transAxes, 
                              fontsize=12, fontweight='bold', va='bottom', ha='right')

    fig.tight_layout()
    Path(args.out).with_suffix(Path(args.out).suffix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main() 