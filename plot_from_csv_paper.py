#!/usr/bin/env python3
"""Simple plotting script that creates plots from existing CSV files."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import statsmodels.api as sm


def fit_linear(x, y):
    """Fit linear regression y = a + b*x using statsmodels.

    Returns:
        dict with coefficients, standard errors, r_squared, bic, model_name
    """
    n = len(x)
    if n < 3:
        return None

    try:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        intercept = model.params[0]
        slope = model.params[1]
        intercept_se = model.bse[0]
        slope_se = model.bse[1]

        return {
            "model": "linear",
            "coeffs": {"intercept": intercept, "slope": slope},
            "std_errors": {"intercept": intercept_se, "slope": slope_se},
            "r_squared": model.rsquared,
            "bic": model.bic,
            "equation": f"y = {intercept:.2e} + {slope:.2e}x",
            "predict": model.predict
        }
    except Exception:
        return None


def fit_quadratic(x, y):
    """Fit quadratic regression y = a + b*x + c*x^2 using statsmodels.

    Returns:
        dict with coefficients, standard errors, r_squared, bic, model_name
    """
    n = len(x)
    if n < 4:
        return None

    try:
        X = np.column_stack([np.ones(n), x, x**2])
        model = sm.OLS(y, X).fit()

        a, b, c = model.params
        a_se, b_se, c_se = model.bse

        return {
            "model": "quadratic",
            "coeffs": {"a": a, "b": b, "c": c},
            "std_errors": {"a": a_se, "b": b_se, "c": c_se},
            "r_squared": model.rsquared,
            "bic": model.bic,
            "equation": f"y = {a:.2e} + {b:.2e}x + {c:.2e}x²",
            "predict": model.predict
        }
    except Exception:
        return None


def fit_exponential(x, y):
    """Fit exponential regression y = a * exp(b*x) via log-linear OLS.

    Returns:
        dict with coefficients, standard errors, r_squared, bic, model_name
    """
    n = len(x)
    if n < 3:
        return None

    # Filter out non-positive y values for log transform
    mask = y > 0
    if np.sum(mask) < 3:
        return None

    x_valid = np.asarray(x)[mask]
    y_valid = np.asarray(y)[mask]

    try:
        # Log-linear regression: log(y) = log(a) + b*x
        log_y = np.log(y_valid)
        X = sm.add_constant(x_valid)
        model = sm.OLS(log_y, X).fit()

        log_a = model.params[0]
        b = model.params[1]
        log_a_se = model.bse[0]
        b_se = model.bse[1]
        a = np.exp(log_a)
        # Approximate SE for a using delta method: SE(a) ≈ a * SE(log_a)
        a_se = a * log_a_se

        # Calculate R² in original space
        y_pred = a * np.exp(b * x_valid)
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            "model": "exponential",
            "coeffs": {"a": a, "b": b},
            "std_errors": {"a": a_se, "b": b_se},
            "r_squared": r_squared,
            "bic": model.bic,  # BIC from log-space model
            "equation": f"y = {a:.2e} × exp({b:.2e}x)",
            "predict": lambda x_new, a=a, b=b: a * np.exp(b * x_new)
        }
    except Exception:
        return None


def select_best_model(models):
    """Select the model with the lowest BIC from a list of fitted models.

    Args:
        models: List of model result dicts (from fit_* functions)

    Returns:
        Best model dict or None if all models failed
    """
    valid_models = [m for m in models if m is not None]
    if not valid_models:
        return None
    return min(valid_models, key=lambda m: m["bic"])


def format_regression_text(model_result, scenario_name):
    """Format regression results for display in a text box.

    Shows the key coefficient(s) with uncertainties:
    - Linear: slope ± SE
    - Quadratic: linear and quadratic coefficients ± SE
    - Exponential: growth rate b ± SE

    Args:
        model_result: Dict from fit_* function
        scenario_name: Name of the scenario for labeling

    Returns:
        Formatted string for text box
    """
    if model_result is None:
        return f"{scenario_name}: No fit"

    # Shorten scenario name
    if "Baseline" in scenario_name and "No Learning" in scenario_name:
        short_name = "Base-NL"
    elif "Baseline" in scenario_name:
        short_name = "Base"
    elif "Hazard" in scenario_name and "No Learning" in scenario_name:
        short_name = "Haz-NL"
    elif "Hazard" in scenario_name:
        short_name = "Haz"
    else:
        short_name = scenario_name[:6]

    model_type = model_result["model"]
    coeffs = model_result["coeffs"]
    std_errors = model_result.get("std_errors", {})

    if model_type == "linear":
        slope = coeffs["slope"]
        slope_se = std_errors.get("slope", 0)
        return f"{short_name}: β={slope:.2e}±{slope_se:.2e}"

    elif model_type == "quadratic":
        b = coeffs["b"]  # linear term
        c = coeffs["c"]  # quadratic term
        b_se = std_errors.get("b", 0)
        c_se = std_errors.get("c", 0)
        return f"{short_name}: β₁={b:.2e}±{b_se:.2e}, β₂={c:.2e}±{c_se:.2e}"

    elif model_type == "exponential":
        b = coeffs["b"]  # growth rate
        b_se = std_errors.get("b", 0)
        return f"{short_name}: r={b:.2e}±{b_se:.2e}"

    return f"{short_name}: {model_type}"


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
    parser.add_argument(
        "--show-nolearning-bottlenecks",
        action="store_true",
        help="Add a 5th row showing No Learning bottleneck plots for Baseline and Hazard"
    )
    parser.add_argument(
        "--bottleneck-out",
        default="bottleneck_plot.png",
        help="Output filename for the bottleneck plots"
    )
    parser.add_argument(
        "--no-regression",
        action="store_true",
        help="Disable regression analysis on time-series plots"
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
        color = "tab:red" if "hazard" in scenario.lower() else "tab:blue"
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
    
    # Define time-series metrics (separate from bottlenecks)
    ts_metrics = [
        "Firm_Production", "Firm_Wealth",
        "Firm_Capital", "Mean_Price",
        "Mean_Wage", "Household_Labor_Sold",
    ]

    # Define bottleneck metrics separately
    bottleneck_metrics = [
        "Bottleneck_Baseline_Learning", "Bottleneck_Hazard_Learning",
        "Bottleneck_Baseline_NoLearning", "Bottleneck_Hazard_NoLearning"
    ]

    # Determine which regression model to use for each metric
    # Production, Wealth, Capital, Wage: linear vs quadratic
    # Price: exponential vs quadratic
    # Household_Labor_Sold: no regression (values are constant/discrete)
    regression_models = {
        "Firm_Production": ["linear", "quadratic"],
        "Firm_Wealth": ["linear", "quadratic"],
        "Firm_Capital": ["linear", "quadratic"],
        "Mean_Price": ["exponential", "quadratic"],
        "Mean_Wage": ["linear", "quadratic"],
    }

    # Create time-series figure (3x2 layout)
    fig_ts, axes_ts = plt.subplots(3, 2, figsize=(12, 10))
    
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
    
    # Store regression results for each metric
    regression_results = {}

    def plot_metric(metric_name, ax, model_types=None, show_regression=True):
        """Plot a single metric with optional regression analysis.

        Args:
            metric_name: Name of the metric to plot
            ax: Matplotlib axes object
            model_types: List of model types to compare (e.g., ["linear", "quadratic"])
            show_regression: Whether to show regression results text box
        """

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

        # Dictionary to store (x, y) data for regression fitting per scenario
        scenario_data = {}
        
        if metric_name in ["Mean_Price", "Mean_Wage"]:
            # Plot main scenario lines from aggregate data
            for scenario, grp in df_combined.groupby("Scenario"):
                style = scenario_style_map[scenario]
                if metric_name in grp.columns:
                    x_data = grp[x_col].values
                    y_data = grp[metric_name].values
                    ax.plot(x_data, y_data,
                           color=style["color"], linestyle=style["linestyle"],
                           label=f"Mean - {scenario}", linewidth=1.5, alpha=0.7, zorder=3)
                    # Store data for regression
                    scenario_data[scenario] = (x_data, y_data)
            
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

                x_vals = np.array(mean_grp.index)
                y_vals = mean_grp.values
                style = scenario_style_map[scenario]
                ax.plot(x_vals, y_vals,
                       color=style["color"], linewidth=1.5, alpha=0.7, linestyle=style["linestyle"],
                       label=f"Mean - {scenario}", zorder=3)
                # Store data for regression
                scenario_data[scenario] = (x_vals, y_vals)
            
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

                x_vals = np.array(mean_grp.index)
                y_vals = mean_grp.values
                style = scenario_style_map[scenario]
                ax.plot(x_vals, y_vals,
                       color=style["color"], linewidth=1.5, alpha=0.7, linestyle=style["linestyle"],
                       label=f"Mean - {scenario}", zorder=3)
                # Store data for regression
                scenario_data[scenario] = (x_vals, y_vals)
            
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

            # Determine which scenario to plot based on metric name
            # Format: Bottleneck_{Baseline|Hazard}_{Learning|NoLearning}
            is_baseline = "Baseline" in metric_name
            is_learning = metric_name.endswith("_Learning")

            base_type = "Baseline" if is_baseline else "Hazard"
            learning_type = "Learning" if is_learning else "No Learning"

            # Find matching scenario
            target_scenarios = [
                s for s in unique_scenarios
                if base_type in s and (
                    (is_learning and "No Learning" not in s) or
                    (not is_learning and "No Learning" in s)
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

        # Perform regression analysis for non-bottleneck metrics
        if show_regression and model_types and scenario_data and not metric_name.startswith("Bottleneck_"):
            reg_texts = []
            metric_reg_results = {}

            for scenario, (x_data, y_data) in scenario_data.items():
                # Remove NaN values
                mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_clean = np.asarray(x_data)[mask]
                y_clean = np.asarray(y_data)[mask]

                if len(x_clean) < 4:
                    continue

                # Fit all specified model types
                fitted_models = []
                for model_type in model_types:
                    if model_type == "linear":
                        fitted_models.append(fit_linear(x_clean, y_clean))
                    elif model_type == "quadratic":
                        fitted_models.append(fit_quadratic(x_clean, y_clean))
                    elif model_type == "exponential":
                        fitted_models.append(fit_exponential(x_clean, y_clean))

                # Select best model by BIC
                best_model = select_best_model(fitted_models)
                if best_model:
                    metric_reg_results[scenario] = best_model
                    reg_texts.append(format_regression_text(best_model, scenario))

            # Store results for this metric
            regression_results[metric_name] = metric_reg_results

            # Add text box with regression results
            if reg_texts:
                text_str = "\n".join(reg_texts)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=7,
                       verticalalignment='top', bbox=props)

        # Set title and labels
        title = metric_name.replace("_", " ").replace("Bottleneck ", "")
        if metric_name.startswith("Bottleneck_"):
            # Include which variant (Learning/No Learning) in title
            title = f"Bottlenecks: {base_type} ({learning_type})"
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
    
    # Determine whether to show regression
    show_regression = not args.no_regression

    # Plot time-series metrics in 3x2 grid
    for i, metric in enumerate(ts_metrics):
        row = i // 2
        col = i % 2
        model_types = regression_models.get(metric)  # None if not in dict
        plot_metric(metric, axes_ts[row, col], model_types=model_types, show_regression=show_regression)

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
            parts = label.split(" - ")
            if len(parts) >= 2:
                sector = parts[0]
                scenario = parts[1]
                if "Baseline" in scenario and "No Learning" in scenario:
                    scenario_abbrev = "Baseline-NL"
                elif "Baseline" in scenario and "Learning" in scenario:
                    scenario_abbrev = "Baseline"
                elif "Hazard" in scenario and "No Learning" in scenario:
                    scenario_abbrev = "Hazard-NL"
                elif "Hazard" in scenario and "Learning" in scenario:
                    scenario_abbrev = "Hazard"
                else:
                    scenario_abbrev = scenario[:3]
                short_labels.append(f"{sector}-{scenario_abbrev}")
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
        plot_metric(metric, axes_bn[row, col], model_types=None, show_regression=False)

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

    # Print regression summary
    if show_regression and regression_results:
        print("\n--- Regression Summary ---")
        for metric, scenarios in regression_results.items():
            print(f"\n{metric}:")
            for scenario, result in scenarios.items():
                model_type = result['model']
                coeffs = result['coeffs']
                std_errors = result.get('std_errors', {})
                r2 = result['r_squared']
                bic = result['bic']

                if model_type == "linear":
                    slope = coeffs['slope']
                    slope_se = std_errors.get('slope', 0)
                    print(f"  {scenario}: linear, slope={slope:.4e}±{slope_se:.4e} (R²={r2:.4f}, BIC={bic:.1f})")
                elif model_type == "quadratic":
                    b, c = coeffs['b'], coeffs['c']
                    b_se, c_se = std_errors.get('b', 0), std_errors.get('c', 0)
                    print(f"  {scenario}: quadratic, β₁={b:.4e}±{b_se:.4e}, β₂={c:.4e}±{c_se:.4e} (R²={r2:.4f}, BIC={bic:.1f})")
                elif model_type == "exponential":
                    rate = coeffs['b']
                    rate_se = std_errors.get('b', 0)
                    print(f"  {scenario}: exponential, rate={rate:.4e}±{rate_se:.4e} (R²={r2:.4f}, BIC={bic:.1f})")


if __name__ == "__main__":
    main()
