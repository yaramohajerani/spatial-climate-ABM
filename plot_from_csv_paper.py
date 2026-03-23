#!/usr/bin/env python3
"""Simple plotting script that creates plots from existing CSV files."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def infer_scenario_name(csv_path: Path) -> str:
    """Infer a human-readable scenario name from a results filename."""
    stem = csv_path.stem.lower()

    if "baseline" in stem:
        base = "Baseline"
    elif "hazard" in stem:
        base = "Hazard"
    else:
        base = csv_path.stem.replace("simulation_", "").replace("_", " ").title()

    if "noadaptation" in stem or "no_adaptation" in stem:
        mode = "No Adaptation"
    elif "adaptation" in stem:
        mode = "Adaptation"
    elif "nolearning" in stem or "no_learning" in stem:
        mode = "No Learning"
    elif "learning" in stem:
        mode = "Learning"
    else:
        return base

    return f"{base} + {mode}"


def is_hazard_scenario(scenario: str) -> bool:
    return "hazard" in scenario.lower()


def is_no_adaptation_scenario(scenario: str) -> bool:
    scenario_lower = scenario.lower()
    return (
        "no adaptation" in scenario_lower
        or "noadaptation" in scenario_lower
        or "no learning" in scenario_lower
        or "nolearning" in scenario_lower
    )


def is_adaptation_scenario(scenario: str) -> bool:
    scenario_lower = scenario.lower()
    return (
        ("adaptation" in scenario_lower and not is_no_adaptation_scenario(scenario))
        or ("learning" in scenario_lower and not is_no_adaptation_scenario(scenario))
    )


def scenario_abbrev(scenario: str) -> str:
    scenario_lower = scenario.lower()
    if "baseline" in scenario_lower:
        base = "Baseline"
    elif "hazard" in scenario_lower:
        base = "Hazard"
    else:
        return scenario

    if is_no_adaptation_scenario(scenario):
        return f"{base}-NA"
    if is_adaptation_scenario(scenario):
        return f"{base}-A"
    return base


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
        default="timeseries.png",
        help="Output plot filename"
    )
    parser.add_argument(
        "--bottleneck-out",
        default="bottleneck_plot.png",
        help="Output filename for the bottleneck plots"
    )
    parser.add_argument(
        "--show-inventory",
        action="store_true",
        help="Add a 4th row showing firm inventory and household labor sold"
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
            scenario_name = infer_scenario_name(csv_path)
            df["Scenario"] = scenario_name
        else:
            non_null_scenarios = df["Scenario"].dropna()
            scenario_name = (
                str(non_null_scenarios.iloc[0])
                if not non_null_scenarios.empty
                else infer_scenario_name(csv_path)
            )
        
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

    household_demand_sectors = []
    if not firm_agents_df.empty and "household_sales_last_step" in firm_agents_df.columns:
        demand_totals = firm_agents_df.groupby("sector")["household_sales_last_step"].sum(min_count=1)
        household_demand_sectors = sorted(demand_totals[demand_totals > 0].index.tolist())
    
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
    
    # Colors distinguish baseline vs hazard; line styles distinguish adaptation on/off.
    scenario_style_map = {}
    for scenario in unique_scenarios:
        color = "tab:red" if is_hazard_scenario(scenario) else "tab:blue"
        linestyle = "--" if is_no_adaptation_scenario(scenario) else "-"
        linewidth = 1.8 if is_adaptation_scenario(scenario) else 1.6
        scenario_style_map[scenario] = {
            "color": color,
            "linestyle": linestyle,
            "linewidth": linewidth,
        }

    # Define sector color palettes keyed by baseline vs hazard for consistency with main lines
    sector_colors_baseline = ["#6baed6", "#3182bd"]
    sector_colors_hazard = ["#ffb347", "#ff7f0e"]

    def get_sector_style(scenario, sector_idx):
        """Get color and style for a sector line based on scenario and sector index."""
        # Choose color palette based on hazard vs baseline
        if is_hazard_scenario(scenario):
            color = sector_colors_hazard[sector_idx % len(sector_colors_hazard)]
        else:
            color = sector_colors_baseline[sector_idx % len(sector_colors_baseline)]

        # Keep sector traces aligned with adaptation vs no-adaptation styling.
        linestyle = "--" if is_no_adaptation_scenario(scenario) else "-"

        return {
            "color": color,
            "linestyle": linestyle,
            "alpha": 0.8,
            "linewidth": 0.7,
            "zorder": 1
        }
    
    # Build per-scenario price deflator for converting nominal → real values.
    # Maps (scenario, year_or_step) → mean_price.  Metrics tagged as "real"
    # will be divided by the deflator so they are expressed in base-period units.
    price_deflator: dict[str, dict] = {}
    for scenario, grp in df_combined.groupby("Scenario"):
        if "Mean_Price" in grp.columns:
            deflator = grp.set_index(x_col)["Mean_Price"].to_dict()
            price_deflator[scenario] = deflator

    # Define time-series metrics (separate from bottlenecks)
    # Order: Production, Capital, Liquidity (real), Consumption, Wage (real), Price
    # Optional 4th row: Firm Inventory + Household Labor Sold
    ts_metrics = [
        "Firm_Production", "Firm_Capital",
        "Firm_Liquidity", "Household_Consumption",
        "Mean_Wage", "Mean_Price",
    ]

    # Optionally add diagnostic row
    if args.show_inventory:
        ts_metrics.extend(["Firm_Inventory", "Household_Labor_Sold"])

    # Metrics that should be deflated (divided by mean price) to show real values
    real_metrics = {"Firm_Liquidity", "Mean_Wage"}

    # Define bottleneck metrics separately
    bottleneck_metrics = [
        "Bottleneck_Baseline_Adaptation", "Bottleneck_Hazard_Adaptation",
        "Bottleneck_Baseline_NoAdaptation", "Bottleneck_Hazard_NoAdaptation"
    ]

    # Create time-series figure (3x2 or 4x2 layout depending on --show-inventory)
    n_rows = 4 if args.show_inventory else 3
    fig_height = 13 if args.show_inventory else 10
    fig_ts, axes_ts = plt.subplots(n_rows, 2, figsize=(12, fig_height))
    
    # Units for y-axis labels
    units = {
        "Firm_Production": "Aggregate Units of Goods",
        "Firm_Liquidity": "Real Dollars ($ / Mean Price)",
        "Firm_Capital": "Aggregate Units of Capital",
        "Firm_Inventory": "Aggregate Units of Goods",
        "Mean_Price": "$ / Unit of Goods",
        "Mean_Wage": "Real Dollars ($ / Mean Price)",
        "Household_Labor_Sold": "Aggregate Units of Labor",
        "Household_Consumption": "Aggregate Units of Goods",
        "Household_Liquidity": "Aggregate Dollars ($)",
    }
    metric_title_map = {
        "Firm_Production": "Aggregate Firm Production",
        "Firm_Liquidity": "Aggregate Real Firm Liquidity",
        "Firm_Capital": "Aggregate Firm Capital",
        "Firm_Inventory": "Aggregate Firm Inventory",
        "Mean_Price": "Mean Firm Price",
        "Mean_Wage": "Mean Firm Wage Offer",
        "Household_Labor_Sold": "Aggregate Household Labor Sold",
        "Household_Consumption": "Aggregate Household Consumption",
        "Household_Liquidity": "Aggregate Household Liquidity",
    }
    aggregate_metric_map = {
        "Firm_Production": "Firm_Production",
        "Firm_Liquidity": "Firm_Wealth",
        "Firm_Capital": "Firm_Capital",
        "Firm_Inventory": "Firm_Inventory",
        "Household_Consumption": "Household_Consumption",
        "Household_Labor_Sold": "Household_Labor_Sold",
        "Household_Liquidity": "Household_Wealth",
        "Mean_Price": "Mean_Price",
        "Mean_Wage": "Mean_Wage",
    }
    sector_aggregation = {
        "Firm_Production": "sum",
        "Firm_Liquidity": "sum",
        "Firm_Capital": "sum",
        "Firm_Inventory": "sum",
        "Mean_Price": "mean",
        "Mean_Wage": "mean",
    }
    
    def deflate(x_vals, y_vals, scenario, metric_name):
        """Divide y-values by the mean price at each x-value if metric is real."""
        if metric_name not in real_metrics:
            return y_vals
        deflator = price_deflator.get(scenario, {})
        if not deflator:
            return y_vals
        prices = np.array([deflator.get(x, np.nan) for x in x_vals])
        prices[prices == 0] = np.nan
        return np.where(np.isfinite(prices), y_vals / prices, y_vals)

    def plot_metric(metric_name, ax):
        """Plot a single metric.

        Args:
            metric_name: Name of the metric to plot
            ax: Matplotlib axes object
        """

        # Define metric mappings for agent-level data
        firm_metric_map = {
            "Firm_Production": "production",
            "Firm_Liquidity": "money",
            "Firm_Capital": "capital",
            "Firm_Inventory": "inventory"
        }

        household_metric_map = {
            "Household_Labor_Sold": "labor_sold",
            "Household_Consumption": "consumption",
            "Household_Liquidity": "money"
        }
        if metric_name in ["Mean_Price", "Mean_Wage"]:
            # Plot main scenario lines from aggregate data
            for scenario, grp in df_combined.groupby("Scenario"):
                style = scenario_style_map[scenario]
                if metric_name in grp.columns:
                    x_data = grp[x_col].values
                    y_data = deflate(x_data, grp[metric_name].values, scenario, metric_name)
                    ax.plot(x_data, y_data,
                           color=style["color"], linestyle=style["linestyle"],
                           label=f"Mean - {scenario}", linewidth=style["linewidth"], alpha=0.7, zorder=3)

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
                        y_vals = deflate(np.array(x_vals), grp.values, scenario, metric_name)
                        sector_style = get_sector_style(scenario, idx_sec)

                        ax.plot(x_vals, y_vals,
                               label=f"{sector} - {scenario}", **sector_style)

        elif metric_name in firm_metric_map:
            # Plot firm metrics with aggregate main lines and optional sector breakdown.
            agent_col = firm_metric_map[metric_name]
            aggregate_col = aggregate_metric_map.get(metric_name)

            # Plot main scenario lines from aggregate model outputs when available.
            if aggregate_col and aggregate_col in df_combined.columns:
                for scenario, grp in df_combined.groupby("Scenario"):
                    style = scenario_style_map[scenario]
                    x_vals = grp[x_col].values
                    y_vals = deflate(x_vals, grp[aggregate_col].values, scenario, metric_name)
                    ax.plot(
                        x_vals,
                        y_vals,
                        color=style["color"],
                        linewidth=style["linewidth"],
                        alpha=0.7,
                        linestyle=style["linestyle"],
                        label=f"Mean - {scenario}",
                        zorder=3,
                    )
            else:
                for scenario in unique_scenarios:
                    if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                        df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = firm_agents_df

                    if df_scen.empty:
                        continue

                    df_scen = add_year_from_step(df_scen, scenario)
                    if "Year" in df_scen.columns:
                        main_grp = df_scen.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                        if main_grp.empty and "Step" in df_scen.columns:
                            main_grp = df_scen.groupby("Step")[agent_col].mean()
                    else:
                        main_grp = df_scen.groupby("Step")[agent_col].mean()
                    if main_grp.empty:
                        continue

                    x_vals = np.array(main_grp.index)
                    y_vals = deflate(x_vals, main_grp.values, scenario, metric_name)
                    style = scenario_style_map[scenario]
                    ax.plot(
                        x_vals,
                        y_vals,
                        color=style["color"],
                        linewidth=style["linewidth"],
                        alpha=0.7,
                        linestyle=style["linestyle"],
                        label=f"Mean - {scenario}",
                        zorder=3,
                    )

            # Add sector lines
            if show_sector_series and not firm_agents_df.empty:
                sectors = sorted(firm_agents_df["sector"].dropna().unique())
                agg_mode = sector_aggregation.get(metric_name, "mean")

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
                            groupby_obj = sector_data.dropna(subset=["Year"]).groupby("Year")[agent_col]
                            grp = groupby_obj.sum() if agg_mode == "sum" else groupby_obj.mean()
                            if grp.empty and "Step" in sector_data.columns:
                                groupby_obj = sector_data.groupby("Step")[agent_col]
                                grp = groupby_obj.sum() if agg_mode == "sum" else groupby_obj.mean()
                        else:
                            groupby_obj = sector_data.groupby("Step")[agent_col]
                            grp = groupby_obj.sum() if agg_mode == "sum" else groupby_obj.mean()
                        if grp.empty:
                            continue

                        x_vals = np.array(grp.index)
                        y_vals = deflate(x_vals, grp.values, scenario, metric_name)
                        sector_style = get_sector_style(scenario, idx_sec)

                        ax.plot(x_vals, y_vals,
                               label=f"{sector} - {scenario}", **sector_style)
        
        elif metric_name in household_metric_map:
            # Plot household metrics with aggregate main lines.
            agent_col = household_metric_map[metric_name]
            aggregate_col = aggregate_metric_map.get(metric_name)

            if aggregate_col and aggregate_col in df_combined.columns:
                for scenario, grp in df_combined.groupby("Scenario"):
                    style = scenario_style_map[scenario]
                    x_vals = grp[x_col].values
                    y_vals = grp[aggregate_col].values
                    ax.plot(
                        x_vals,
                        y_vals,
                        color=style["color"],
                        linewidth=style["linewidth"],
                        alpha=0.7,
                        linestyle=style["linestyle"],
                        label=f"Mean - {scenario}",
                        zorder=3,
                    )
            else:
                for scenario in unique_scenarios:
                    if not household_agents_df.empty and "Scenario" in household_agents_df.columns:
                        df_scen = household_agents_df[household_agents_df["Scenario"] == scenario]
                    else:
                        df_scen = household_agents_df

                    if df_scen.empty:
                        continue

                    df_scen = add_year_from_step(df_scen, scenario)
                    if "Year" in df_scen.columns:
                        main_grp = df_scen.dropna(subset=["Year"]).groupby("Year")[agent_col].mean()
                        if main_grp.empty and "Step" in df_scen.columns:
                            main_grp = df_scen.groupby("Step")[agent_col].mean()
                    else:
                        main_grp = df_scen.groupby("Step")[agent_col].mean()
                    if main_grp.empty:
                        continue

                    x_vals = np.array(main_grp.index)
                    y_vals = main_grp.values
                    style = scenario_style_map[scenario]
                    ax.plot(
                        x_vals,
                        y_vals,
                        color=style["color"],
                        linewidth=style["linewidth"],
                        alpha=0.7,
                        linestyle=style["linestyle"],
                        label=f"Mean - {scenario}",
                        zorder=3,
                    )

            # For consumption, add actual household purchases by seller sector
            # when the agent CSV contains the dedicated household-sales field.
            if (
                show_sector_series
                and metric_name == "Household_Consumption"
                and household_demand_sectors
                and not firm_agents_df.empty
            ):
                for scenario in unique_scenarios:
                    if not household_agents_df.empty and "Scenario" in household_agents_df.columns:
                        hh_scen = household_agents_df[household_agents_df["Scenario"] == scenario]
                    else:
                        hh_scen = household_agents_df

                    if not firm_agents_df.empty and "Scenario" in firm_agents_df.columns:
                        firm_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                    else:
                        firm_scen = firm_agents_df

                    hh_scen = add_year_from_step(hh_scen, scenario)
                    firm_scen = add_year_from_step(firm_scen, scenario)

                    for idx_sec, sector in enumerate(household_demand_sectors):
                        sector_data = firm_scen[firm_scen["sector"] == sector]
                        if sector_data.empty:
                            continue

                        value_axis = "Year" if "Year" in sector_data.columns else "Step"
                        grp = sector_data.groupby(value_axis)["household_sales_last_step"].sum()
                        if grp.empty:
                            continue

                        x_vals = grp.index
                        sector_style = get_sector_style(scenario, idx_sec)

                        ax.plot(
                            x_vals,
                            grp.values,
                            label=f"Final demand: {sector} - {scenario}",
                            **sector_style,
                        )
        
        elif metric_name.startswith("Bottleneck_"):
            # Bottleneck plots from agent data
            if firm_agents_df.empty:
                ax.text(0.5, 0.5, "No agent data\navailable",
                       ha="center", va="center", transform=ax.transAxes)
                return

            # Determine which scenario to plot based on metric name
            # Format: Bottleneck_{Baseline|Hazard}_{Adaptation|NoAdaptation}
            is_baseline = "Baseline" in metric_name
            is_no_adaptation = metric_name.endswith("_NoAdaptation")

            base_type = "Baseline" if is_baseline else "Hazard"
            adaptation_type = "No Adaptation" if is_no_adaptation else "Adaptation"

            # Find matching scenario
            target_scenarios = [
                s for s in unique_scenarios
                if base_type in s and (
                    (is_no_adaptation and is_no_adaptation_scenario(s)) or
                    (not is_no_adaptation and is_adaptation_scenario(s))
                )
            ]

            if not target_scenarios:
                # Fallback: just match base type
                target_scenarios = [s for s in unique_scenarios if base_type in s]

            if not target_scenarios:
                target_scenarios = [unique_scenarios[0]]  # ultimate fallback

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
            title = f"Bottlenecks: {base_type} ({adaptation_type})"
        else:
            title = metric_title_map.get(metric_name, title)
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
    
    # Plot time-series metrics in 3x2 grid
    for i, metric in enumerate(ts_metrics):
        row = i // 2
        col = i % 2
        plot_metric(metric, axes_ts[row, col])

    # Add subplot labels (a, b, c, ...) to time-series figure
    for i, metric in enumerate(ts_metrics):
        row = i // 2
        col = i % 2
        label_char = chr(ord('a') + i)
        axes_ts[row, col].text(-0.1, 1.02, f'({label_char})',
                              transform=axes_ts[row, col].transAxes,
                              fontsize=12, fontweight='bold', va='bottom', ha='right')

    # Create shared legend for time-series plots
    handles, labels = axes_ts[0, 0].get_legend_handles_labels()

    # Create shorter labels for shared legend
    short_labels = []
    for label in labels:
        if "Mean -" in label:
            scenario = label.replace("Mean - ", "")
            short_labels.append(scenario_abbrev(scenario))
        else:
            parts = label.split(" - ")
            if len(parts) >= 2:
                sector = parts[0].replace("Final demand: ", "")
                scenario = parts[1]
                short_labels.append(f"{sector}-{scenario_abbrev(scenario)}")
            else:
                short_labels.append(label[:8])

    if handles:
        ncols = min(len(handles), 6)  # Max 6 columns
        legend = fig_ts.legend(handles, short_labels, loc='lower center', ncol=ncols,
                              fontsize=9, bbox_to_anchor=(0.5, -0.02))
        for i, line in enumerate(legend.get_lines()):
            label = short_labels[i] if i < len(short_labels) else ""
            if any(abbrev in label for abbrev in ["commodity-", "manufacturing-"]):
                line.set_linewidth(2)
            else:
                line.set_linewidth(3)

    fig_ts.suptitle("Baseline vs. RCP8.5 Agent Trajectories", fontsize=14, fontweight='bold')
    fig_ts.tight_layout()
    fig_ts.subplots_adjust(bottom=0.06)

    # Save time-series plot
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_ts.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Time-series plot saved to {out_path}")
    plt.close(fig_ts)

    # Create separate bottleneck figure (2x2 layout)
    fig_bn, axes_bn = plt.subplots(2, 2, figsize=(10, 8))

    # Plot bottleneck metrics
    for i, metric in enumerate(bottleneck_metrics):
        row = i // 2
        col = i % 2
        plot_metric(metric, axes_bn[row, col])

    # Add subplot labels to bottleneck figure
    for i, metric in enumerate(bottleneck_metrics):
        row = i // 2
        col = i % 2
        label_char = chr(ord('a') + i)
        axes_bn[row, col].text(-0.1, 1.02, f'({label_char})',
                              transform=axes_bn[row, col].transAxes,
                              fontsize=12, fontweight='bold', va='bottom', ha='right')

    fig_bn.suptitle("Production Bottleneck Analysis", fontsize=14, fontweight='bold')
    fig_bn.tight_layout()

    # Save bottleneck plot
    bn_out_path = Path(args.bottleneck_out)
    bn_out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_bn.savefig(bn_out_path, dpi=150, bbox_inches='tight')
    print(f"Bottleneck plot saved to {bn_out_path}")
    plt.close(fig_bn)



if __name__ == "__main__":
    main()
