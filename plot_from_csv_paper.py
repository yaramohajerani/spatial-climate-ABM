#!/usr/bin/env python3
"""Simple plotting script that creates plots from existing CSV files."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create plots from existing simulation CSV files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-files", 
        nargs="+",
        required=True,
        help="Paths to one or more main results CSV files to compare"
    )
    parser.add_argument(
        "--agents-csv", 
        help="Path to the agents CSV file (optional, will auto-detect if not provided)"
    )
    parser.add_argument(
        "--no-sector",
        action="store_true",
        help="Hide sector-level time series from agent data",
    )
    parser.add_argument(
        "--out", 
        default="recreated_plot.png",
        help="Output plot filename"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show_sector_series = not args.no_sector
    
    # Load and combine multiple CSV files
    dataframes = []
    agent_dataframes = []
    
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path}")
        
        # Extract scenario from filename or add step column if needed
        if "Scenario" not in df.columns:
            # Try to infer scenario from filename
            stem = csv_path.stem
            if "baseline" in stem and "nolearning" in stem:
                scenario_name = "Baseline + No Learning"
            elif "baseline" in stem and "learning" in stem:
                scenario_name = "Baseline + Learning"
            elif "hazard" in stem and "nolearning" in stem:
                scenario_name = "Hazard + No Learning"
            elif "hazard" in stem and "learning" in stem:
                scenario_name = "Hazard + Learning"
            else:
                scenario_name = stem.replace("simulation_", "").replace("_", " ").title()
            
            df["Scenario"] = scenario_name
        
        # Add step column if not present
        if "Step" not in df.columns:
            df["Step"] = df.index
            
        dataframes.append(df)
        
        # Load corresponding agent data
        if args.agents_csv:
            # Use explicitly provided agent CSV for all scenarios
            agents_path = Path(args.agents_csv)
            if agents_path.exists():
                agent_df = pd.read_csv(args.agents_csv)
                if "Scenario" not in agent_df.columns:
                    agent_df["Scenario"] = scenario_name
                agent_dataframes.append(agent_df)
                print(f"Loaded agent data from {agents_path}")
            else:
                print(f"Warning: Specified agent data file not found: {agents_path}")
        else:
            # Auto-detect agent CSV
            agents_path = csv_path.parent / f"{csv_path.stem}_agents.csv"
            if agents_path.exists():
                agent_df = pd.read_csv(agents_path)
                if "Scenario" not in agent_df.columns:
                    agent_df["Scenario"] = scenario_name
                agent_dataframes.append(agent_df)
                print(f"Loaded agent data from {agents_path}")
            else:
                print(f"Warning: No agent data found for {csv_path}")
    
    # Combine all dataframes
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    if agent_dataframes:
        agent_df_combined = pd.concat(agent_dataframes, ignore_index=True)
        # Separate firm and household agents
        firm_agents_df = agent_df_combined[agent_df_combined["type"] == "FirmAgent"].copy()
        household_agents_df = agent_df_combined[agent_df_combined["type"] == "HouseholdAgent"].copy()
    else:
        print("Warning: No agent data found - bottleneck plots will not work")
        firm_agents_df = pd.DataFrame()
        household_agents_df = pd.DataFrame()
    
    # Determine x-axis column - prefer Year column if available
    if "Year" in df_combined.columns:
        x_col = "Year"
    elif "Step" in df_combined.columns:
        x_col = "Step"
    else:
        # Create step column from index
        df_combined["Step"] = df_combined.index
        x_col = "Step"
    
    # Get unique scenarios
    unique_scenarios = sorted(df_combined["Scenario"].unique())
    print(f"Found scenarios: {unique_scenarios}")

    # Map Step to Year per scenario so agent data can use actual years
    step_to_year_map = {}
    if "Year" in df_combined.columns and "Step" in df_combined.columns:
        for scenario in unique_scenarios:
            scen_df = df_combined[df_combined["Scenario"] == scenario]
            mapping = (
                scen_df.dropna(subset=["Step", "Year"])
                .drop_duplicates(subset="Step")
                .set_index("Step")["Year"]
                .to_dict()
            )
            if mapping:
                step_to_year_map[scenario] = mapping

        if not step_to_year_map:
            fallback_map = (
                df_combined.dropna(subset=["Step", "Year"])
                .drop_duplicates(subset="Step")
                .set_index("Step")["Year"]
                .to_dict()
            )
            if fallback_map:
                step_to_year_map["__all__"] = fallback_map

    def add_year_from_step(df, scenario):
        """Attach Year values to agent data using mapping from aggregate results."""
        if "Year" in df.columns or "Step" not in df.columns or df.empty:
            return df
        year_map = step_to_year_map.get(scenario) or step_to_year_map.get("__all__")
        if not year_map:
            return df
        df_copy = df.copy()
        df_copy["Year"] = df_copy["Step"].map(year_map)
        return df_copy
    
    # Create color/style mapping: colors for hazard vs baseline, line styles for learning vs no learning
    scenario_style_map = {}
    for scenario in unique_scenarios:
        color = "tab:orange" if "hazard" in scenario.lower() else "tab:blue"
        linestyle = ":" if "no learning" in scenario.lower() else "-"
            
        scenario_style_map[scenario] = {"color": color, "linestyle": linestyle}
    
    # Define sector color palettes keyed by baseline vs hazard for consistency with main lines
    sector_colors_baseline = ["#6baed6", "#3182bd"]
    sector_colors_hazard = ["#ffb347", "#ff7f0e"]
    
    def get_sector_style(scenario, sector_idx):
        """Get color and style for a sector line based on scenario and sector index."""
        # Choose color palette based on hazard vs baseline
        if "hazard" in scenario.lower():
            color = sector_colors_hazard[sector_idx % len(sector_colors_hazard)]
        else:
            color = sector_colors_baseline[sector_idx % len(sector_colors_baseline)]
        
        # Choose line style based on learning vs no learning
        linestyle = ":" if "no learning" in scenario.lower() else "-"
        
        return {
            "color": color,
            "linestyle": linestyle, 
            "alpha": 0.8,
            "linewidth": 0.7,
            "zorder": 1
        }
    
    # Define the specific metrics for 3x3 layout (excluding household consumption)
    metrics = [
        "Firm_Production", "Firm_Wealth", "Firm_Capital", 
        "Mean_Price", "Mean_Wage", "Household_Labor_Sold",
        "Household_Wealth", "Bottleneck_Baseline", "Bottleneck_Hazard"
    ]
    
    # Create 3x3 subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(13, 6.5))
    
    # Units for y-axis labels
    units = {
        "Firm_Production": "Units of Goods",
        "Firm_Wealth": "$",
        "Firm_Capital": "Units of Capital", 
        "Mean_Price": "$ / Unit of Goods",
        "Mean_Wage": "$ / Unit of Labor",
        "Household_Labor_Sold": "Units of Labor",
        "Household_Consumption": "Units of Goods",
        "Household_Wealth": "$",
    }
    
    def plot_metric(metric_name, ax):
        """Plot a single metric."""
        
        # Define metric mappings for agent-level data
        firm_metric_map = {
            "Firm_Production": "production",
            "Firm_Wealth": "money", 
            "Firm_Capital": "capital"
        }
        
        household_metric_map = {
            "Household_Labor_Sold": "labor_sold",
            "Household_Consumption": "consumption",
            "Household_Wealth": "money"
        }
        
        if metric_name in ["Mean_Price", "Mean_Wage"]:
            # Plot main scenario lines from aggregate data
            for scenario, grp in df_combined.groupby("Scenario"):
                style = scenario_style_map[scenario]
                if metric_name in grp.columns:
                    ax.plot(grp[x_col], grp[metric_name], 
                           color=style["color"], linestyle=style["linestyle"], 
                           label=f"Mean - {scenario}", linewidth=2, zorder=3)
            
            # Add sector lines from agent data for wages and prices
            if show_sector_series and not firm_agents_df.empty and metric_name in ["Mean_Price", "Mean_Wage"]:
                agent_col = "price" if metric_name == "Mean_Price" else "wage"
                sectors = sorted(firm_agents_df["sector"].dropna().unique())
                
                for scenario in unique_scenarios:
                    if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                        df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = firm_agents_df  # Use all data if no scenario column
                    
                    df_scen = add_year_from_step(df_scen, scenario)
                    style = scenario_style_map[scenario]
                    
                    for idx_sec, sector in enumerate(sectors):
                        sector_data = df_scen[df_scen["sector"] == sector]
                        if sector_data.empty:
                            continue
                        
                        # Use Year column if available, otherwise Step
                        if "Year" in sector_data.columns:
                            grp = sector_data.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                            if grp.empty and "Step" in sector_data.columns:
                                grp = sector_data.groupby("Step")[agent_col].mean()
                        else:
                            grp = sector_data.groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                            
                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec)
                        
                        ax.plot(x_vals, grp.values, 
                               label=f"{sector} - {scenario}", **sector_style)
        
        elif metric_name in firm_metric_map:
            # Plot firm metrics with main lines and sector breakdown
            agent_col = firm_metric_map[metric_name]
            
            # Plot main scenario lines (mean across all firms)
            for scenario in unique_scenarios:
                if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                    df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                else:
                    df_scen = firm_agents_df  # Use all data if no scenario column
                
                if df_scen.empty:
                    continue
                
                df_scen = add_year_from_step(df_scen, scenario)
                    
                # Use Year column if available, otherwise Step
                if "Year" in df_scen.columns:
                    mean_grp = df_scen.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                    if mean_grp.empty and "Step" in df_scen.columns:
                        mean_grp = df_scen.groupby("Step")[agent_col].mean()
                else:
                    mean_grp = df_scen.groupby("Step")[agent_col].mean()
                if mean_grp.empty:
                    continue
                    
                x_vals = mean_grp.index
                style = scenario_style_map[scenario]
                ax.plot(x_vals, mean_grp.values, 
                       color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                       label=f"Mean - {scenario}", zorder=3)
            
            # Add sector lines
            if show_sector_series and not firm_agents_df.empty:
                sectors = sorted(firm_agents_df["sector"].dropna().unique())
                
                for scenario in unique_scenarios:
                    if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                        df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = firm_agents_df  # Use all data if no scenario column
                    
                    df_scen = add_year_from_step(df_scen, scenario)
                    style = scenario_style_map[scenario]
                    
                    for idx_sec, sector in enumerate(sectors):
                        sector_data = df_scen[df_scen["sector"] == sector]
                        # Use Year column if available, otherwise Step
                        if "Year" in sector_data.columns:
                            grp = sector_data.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                            if grp.empty and "Step" in sector_data.columns:
                                grp = sector_data.groupby("Step")[agent_col].mean()
                        else:
                            grp = sector_data.groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                            
                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec)
                        
                        ax.plot(x_vals, grp.values, 
                               label=f"{sector} - {scenario}", **sector_style)
        
        elif metric_name in household_metric_map:
            # Plot household metrics with main lines and sector breakdown
            agent_col = household_metric_map[metric_name]
            
            # Plot main scenario lines (mean across all households)
            for scenario in unique_scenarios:
                if not household_agents_df.empty and "Scenario" in household_agents_df.columns:
                    df_scen = household_agents_df[household_agents_df["Scenario"] == scenario]
                else:
                    df_scen = household_agents_df  # Use all data if no scenario column
                
                if df_scen.empty:
                    continue
                
                df_scen = add_year_from_step(df_scen, scenario)
                    
                # Use Year column if available, otherwise Step
                if "Year" in df_scen.columns:
                    mean_grp = df_scen.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                    if mean_grp.empty and "Step" in df_scen.columns:
                        mean_grp = df_scen.groupby("Step")[agent_col].mean()
                else:
                    mean_grp = df_scen.groupby("Step")[agent_col].mean()
                if mean_grp.empty:
                    continue
                    
                x_vals = mean_grp.index
                style = scenario_style_map[scenario]
                ax.plot(x_vals, mean_grp.values, 
                       color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                       label=f"Mean - {scenario}", zorder=3)
            
            # Add sector lines if household data has sectors
            if show_sector_series and not household_agents_df.empty and "sector" in household_agents_df.columns:
                sectors = sorted(household_agents_df["sector"].dropna().unique())
                
                for scenario in unique_scenarios:
                    if not household_agents_df.empty and "Scenario" in household_agents_df.columns:
                        df_scen = household_agents_df[household_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = household_agents_df  # Use all data if no scenario column
                    
                    df_scen = add_year_from_step(df_scen, scenario)
                    style = scenario_style_map[scenario]
                    
                    for idx_sec, sector in enumerate(sectors):
                        sector_data = df_scen[df_scen["sector"] == sector]
                        # Use Year column if available, otherwise Step
                        if "Year" in sector_data.columns:
                            grp = sector_data.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                            if grp.empty and "Step" in sector_data.columns:
                                grp = sector_data.groupby("Step")[agent_col].mean()
                        else:
                            grp = sector_data.groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                            
                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec)
                        
                        ax.plot(x_vals, grp.values, 
                               label=f"{sector} - {scenario}", **sector_style)
        
        elif metric_name.startswith("Bottleneck_"):
            # Bottleneck plots from agent data
            if firm_agents_df.empty:
                ax.text(0.5, 0.5, "No agent data\navailable", 
                       ha="center", va="center", transform=ax.transAxes)
                return
            
            # Determine which scenario to plot
            if metric_name == "Bottleneck_Baseline":
                target_scenarios = [s for s in unique_scenarios if "Baseline" in s]
            else:  # Bottleneck_Hazard
                target_scenarios = [s for s in unique_scenarios if "Hazard" in s]
            
            if not target_scenarios:
                target_scenarios = [unique_scenarios[0]]  # fallback
            
            scenario = target_scenarios[0]
            if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                df_sub = firm_agents_df[firm_agents_df["Scenario"] == scenario]
            else:
                df_sub = firm_agents_df  # Use all data if no scenario column

            df_sub = add_year_from_step(df_sub, scenario)
            
            if df_sub.empty:
                ax.text(0.5, 0.5, f"No data for\n{scenario}", 
                       ha="center", va="center", transform=ax.transAxes)
                return
            
            # Calculate bottleneck percentages
            # Use Year column if available, otherwise Step
            if "Year" in df_sub.columns:
                time_col = "Year"
                time_vals = sorted(df_sub["Year"].dropna().unique())
            else:
                time_col = "Step"
                time_vals = sorted(df_sub["Step"].unique())
            x_vals = time_vals
            
            # Create percentage arrays
            arrs = {}
            for bt in ["labor", "capital", "input"]:
                cnt = df_sub[df_sub["limiting_factor"] == bt].groupby(time_col).size()
                cnt = cnt.reindex(time_vals, fill_value=0)
                arrs[bt] = cnt
            
            tot = sum(arrs.values())
            tot[tot == 0] = 1  # avoid division by zero
            pct_arrays = [100 * arrs[bt] / tot for bt in ["labor", "capital", "input"]]
            
            ax.stackplot(x_vals, *pct_arrays,
                        labels=["Labour", "Capital", "Input"],
                        colors=["#1f77b4", "#d62728", "#2ca02c"], alpha=0.7)
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of firms")
        
        # Set title and labels
        title = metric_name.replace("_", " ").replace("Bottleneck ", "")
        if metric_name.startswith("Bottleneck_"):
            title = f"Production Bottlenecks ({title})"
        ax.set_title(title, fontsize=10)
        
        ylabel = units.get(metric_name, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        elif metric_name.startswith("Bottleneck_"):
            ax.set_ylabel("% of firms")
            
        ax.set_xlabel(x_col)
        
        # Handle legends - only show legend for bottleneck plots
        if metric_name.startswith("Bottleneck_"):
            ax.legend(fontsize=6, ncol=3, loc='lower center', framealpha=0.8)
    
    # Plot metrics in 3x3 grid
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        plot_metric(metric, axes[row, col])
    
    # Add subplot labels (a, b, c, ...)
    subplot_labels = [chr(ord('a') + i) for i in range(9)]
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        axes[row, col].text(-0.1, 1.02, f'({subplot_labels[i]})', 
                           transform=axes[row, col].transAxes, 
                           fontsize=12, fontweight='bold', va='bottom', ha='right')
    
    # Create shared legend for non-bottleneck plots
    # Get handles and labels from the first non-bottleneck plot
    legend_ax = None
    for i, metric in enumerate(metrics):
        if not metric.startswith("Bottleneck_"):
            row = i // 3
            col = i % 3
            legend_ax = axes[row, col]
            break
    
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        
        # Create shorter labels for shared legend
        short_labels = []
        for label in labels:
            if "Mean -" in label:
                # Main scenario lines - keep scenario name but make shorter
                scenario = label.replace("Mean - ", "")
                if "Baseline" in scenario and "No Learning" in scenario:
                    short_labels.append("Baseline-NL")
                elif "Baseline" in scenario and "Learning" in scenario:
                    short_labels.append("Baseline")
                elif "Hazard" in scenario and "No Learning" in scenario:
                    short_labels.append("Hazard-NL")
                elif "Hazard" in scenario and "Learning" in scenario:
                    short_labels.append("Hazard")
                else:
                    short_labels.append(scenario)
            else:
                # Sector lines - use abbreviations
                parts = label.split(" - ")
                if len(parts) >= 2:
                    sector = parts[0]
                    scenario = parts[1]
                    
                    # Abbreviate scenario names
                    if "Baseline" in scenario and "No Learning" in scenario:
                        scenario_abbrev = "Baseline-NL"  # Baseline No Learning
                    elif "Baseline" in scenario and "Learning" in scenario:
                        scenario_abbrev = "Baseline"   # Baseline Learning
                    elif "Hazard" in scenario and "No Learning" in scenario:
                        scenario_abbrev = "Hazard-NL"  # Hazard No Learning
                    elif "Hazard" in scenario and "Learning" in scenario:
                        scenario_abbrev = "Hazard"   # Hazard Learning
                    else:
                        scenario_abbrev = scenario[:3]
                    
                    short_labels.append(f"{sector}-{scenario_abbrev}")
                else:
                    short_labels.append(label[:8])  # Truncate if format is unexpected
        
        if handles:
            # Calculate number of columns to fit all items in one row
            ncols = len(handles)
            legend = fig.legend(handles, short_labels, loc='lower center', ncol=ncols, 
                               fontsize=10, bbox_to_anchor=(0.5, -0.02))
            # Make legend lines thicker for better visibility
            for i, line in enumerate(legend.get_lines()):
                # Check if this is a sector line (contains sector abbreviation)
                label = short_labels[i] if i < len(short_labels) else ""
                if any(abbrev in label for abbrev in ["commodity-", "manufacturing-"]):
                    line.set_linewidth(2)  # Thinner for sector lines
                else:
                    line.set_linewidth(3)  # Thicker for main scenario lines
    
    plt.suptitle("Baseline vs. RCP8.5 Agent Trajectories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Adjust layout to make room for bottom legend
    plt.subplots_adjust(bottom=0.095)
    
    # Save plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
