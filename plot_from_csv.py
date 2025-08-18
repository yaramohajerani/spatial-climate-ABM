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
        "--out", 
        default="recreated_plot.png",
        help="Output plot filename"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        help="Base year for x-axis (if not provided, will use Step column)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    
    # Determine x-axis column
    if args.start_year and "Year" in df_combined.columns:
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
    
    # Create color/style mapping: colors for hazard vs baseline, line styles for learning vs no learning
    scenario_style_map = {}
    for scenario in unique_scenarios:
        # Colors: red for hazard, blue for baseline
        if "hazard" in scenario.lower():
            color = "red"
        else:
            color = "blue"
        
        # Line styles: solid for learning, dotted for no learning
        if "no learning" in scenario.lower():
            linestyle = ":"  # dotted for no learning
        else:
            linestyle = "-"  # solid for learning
            
        scenario_style_map[scenario] = {"color": color, "linestyle": linestyle}
    
    # Define sector color palettes with very distinct variations
    sector_colors_hazard = ["#B22222", "#FF4500", "#FFD700", "#FF69B4", "#8B0000"]  # Very distinct red/orange/yellow/pink for hazard
    sector_colors_baseline = ["#000080", "#00CED1", "#32CD32", "#9400D3", "#4682B4"]  # Very distinct blue/cyan/green/purple for baseline
    
    def get_sector_style(scenario, sector_idx, style_map):
        """Get color and style for a sector line based on scenario and sector index."""
        # Get base style from scenario
        base_style = style_map[scenario]
        
        # Choose color palette based on scenario type
        if "hazard" in scenario.lower():
            color = sector_colors_hazard[sector_idx % len(sector_colors_hazard)]
        else:
            color = sector_colors_baseline[sector_idx % len(sector_colors_baseline)]
        
        return {
            "color": color,
            "linestyle": base_style["linestyle"], 
            "alpha": 0.8,
            "linewidth": 0.5,
            "zorder": 1
        }
    
    # Define the specific metrics for your 2x4 layout
    # Top row: firm production, firm wealth, firm capital, mean price, mean wage
    # Bottom row: labour sold, household consumption, household wealth, bottleneck baseline, bottleneck hazard
    top_metrics = ["Firm_Production", "Firm_Wealth", "Firm_Capital", "Mean_Price", "Mean_Wage"]
    bottom_metrics = ["Household_Labor_Sold", "Household_Consumption", "Household_Wealth", "Bottleneck_Baseline", "Bottleneck_Hazard"]
    
    # Create 2x5 subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
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
            if not firm_agents_df.empty and metric_name in ["Mean_Price", "Mean_Wage"]:
                agent_col = "price" if metric_name == "Mean_Price" else "wage"
                sectors = sorted(firm_agents_df["sector"].dropna().unique())
                
                for scenario in unique_scenarios:
                    if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                        df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = firm_agents_df  # Use all data if no scenario column
                    
                    style = scenario_style_map[scenario]
                    
                    for idx_sec, sector in enumerate(sectors):
                        sector_data = df_scen[df_scen["sector"] == sector]
                        if sector_data.empty:
                            continue
                        
                        grp = sector_data.groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                            
                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec, scenario_style_map)
                        
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
                    
                mean_grp = df_scen.groupby("Step")[agent_col].mean()
                if mean_grp.empty:
                    continue
                    
                x_vals = mean_grp.index
                style = scenario_style_map[scenario]
                ax.plot(x_vals, mean_grp.values, 
                       color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                       label=f"Mean - {scenario}", zorder=3)
            
            # Add sector lines
            if not firm_agents_df.empty:
                sectors = sorted(firm_agents_df["sector"].dropna().unique())
                
                for scenario in unique_scenarios:
                    if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                        df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = firm_agents_df  # Use all data if no scenario column
                    
                    style = scenario_style_map[scenario]
                    
                    for idx_sec, sector in enumerate(sectors):
                        grp = df_scen[df_scen["sector"] == sector].groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                            
                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec, scenario_style_map)
                        
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
                    
                mean_grp = df_scen.groupby("Step")[agent_col].mean()
                if mean_grp.empty:
                    continue
                    
                x_vals = mean_grp.index
                style = scenario_style_map[scenario]
                ax.plot(x_vals, mean_grp.values, 
                       color=style["color"], linewidth=2, linestyle=style["linestyle"], 
                       label=f"Mean - {scenario}", zorder=3)
            
            # Add sector lines if household data has sectors
            if not household_agents_df.empty and "sector" in household_agents_df.columns:
                sectors = sorted(household_agents_df["sector"].dropna().unique())
                
                for scenario in unique_scenarios:
                    if not household_agents_df.empty and "Scenario" in household_agents_df.columns:
                        df_scen = household_agents_df[household_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = household_agents_df  # Use all data if no scenario column
                    
                    style = scenario_style_map[scenario]
                    
                    for idx_sec, sector in enumerate(sectors):
                        grp = df_scen[df_scen["sector"] == sector].groupby("Step")[agent_col].mean()
                        if grp.empty:
                            continue
                            
                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec, scenario_style_map)
                        
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
            
            if df_sub.empty:
                ax.text(0.5, 0.5, f"No data for\n{scenario}", 
                       ha="center", va="center", transform=ax.transAxes)
                return
            
            # Calculate bottleneck percentages
            steps = df_sub["Step"].unique()
            x_vals = steps
            
            # Create percentage arrays
            arrs = {}
            for bt in ["labor", "capital", "input"]:
                cnt = df_sub[df_sub["limiting_factor"] == bt].groupby("Step").size()
                cnt = cnt.reindex(steps, fill_value=0)
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
        
        # Handle legends - compact format inside plot area
        if not metric_name.startswith("Bottleneck_"):
            handles, labels = ax.get_legend_handles_labels()
            
            # Create shorter labels for compact legend
            short_labels = []
            for label in labels:
                if "Mean -" in label:
                    # Main scenario lines - keep scenario name but make shorter
                    scenario = label.replace("Mean - ", "")
                    if "Baseline" in scenario and "No Learning" in scenario:
                        short_labels.append("Baseline-NL")
                    elif "Baseline" in scenario and "Learning" in scenario:
                        short_labels.append("Baseline-L")
                    elif "Hazard" in scenario and "No Learning" in scenario:
                        short_labels.append("Hazard-NL")
                    elif "Hazard" in scenario and "Learning" in scenario:
                        short_labels.append("Hazard-L")
                    else:
                        short_labels.append(scenario)
                else:
                    # Sector lines - use abbreviations
                    parts = label.split(" - ")
                    if len(parts) >= 2:
                        sector = parts[0]
                        scenario = parts[1]
                        
                        # Abbreviate sector names
                        if sector == "commodity":
                            sector_abbrev = "Com"
                        elif sector == "manufacturing":
                            sector_abbrev = "Man"
                        else:
                            sector_abbrev = sector[:3]
                        
                        # Abbreviate scenario names
                        if "Baseline" in scenario and "No Learning" in scenario:
                            scenario_abbrev = "B-NL"  # Baseline No Learning
                        elif "Baseline" in scenario and "Learning" in scenario:
                            scenario_abbrev = "B-L"   # Baseline Learning
                        elif "Hazard" in scenario and "No Learning" in scenario:
                            scenario_abbrev = "H-NL"  # Hazard No Learning
                        elif "Hazard" in scenario and "Learning" in scenario:
                            scenario_abbrev = "H-L"   # Hazard Learning
                        else:
                            scenario_abbrev = scenario[:3]
                        
                        short_labels.append(f"{sector_abbrev}-{scenario_abbrev}")
                    else:
                        short_labels.append(label[:8])  # Truncate if format is unexpected
            
            if handles:
                # Place legend inside plot area with small font
                ax.legend(handles, short_labels, fontsize=7, ncol=2, 
                         loc='upper right', framealpha=0.8)
        else:
            ax.legend(fontsize=6, ncol=3, loc='lower center', framealpha=0.8)
    
    # Plot top row metrics
    for i, metric in enumerate(top_metrics):
        plot_metric(metric, axes[0, i])
    
    # Plot bottom row metrics  
    for i, metric in enumerate(bottom_metrics):
        plot_metric(metric, axes[1, i])
    
    # Add subplot labels (a, b, c, ...)
    subplot_labels = [chr(ord('a') + i) for i in range(10)]
    for i in range(5):
        # Top row
        axes[0, i].text(-0.1, 1.02, f'({subplot_labels[i]})', 
                       transform=axes[0, i].transAxes, 
                       fontsize=12, fontweight='bold', va='bottom', ha='right')
        # Bottom row
        axes[1, i].text(-0.1, 1.02, f'({subplot_labels[i + 5]})', 
                       transform=axes[1, i].transAxes, 
                       fontsize=12, fontweight='bold', va='bottom', ha='right')
    
    plt.suptitle("Baseline vs. RCP8.5 Agent Trajectories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()