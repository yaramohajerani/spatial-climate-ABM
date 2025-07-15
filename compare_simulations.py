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
        description="Run two simulations – one with climate hazards and one without – and plot both timeseries side-by-side.",
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


def run_simulation(model: EconomyModel, n_steps: int):
    for _ in range(n_steps):
        model.step()
    return model.results_to_dataframe(), model  # return model for agent access


def main():
    args = _parse_args()

    # Merge optional config file
    _merge_param_file(args)

    if not args.simulation_output and not args.rp_file:
        raise SystemExit("Either --rp-file or --simulation_output must be provided.")

    if args.simulation_output:
        df_combined = pd.read_csv(args.simulation_output)
        print(f"Loaded data from {args.simulation_output}")
    else:
        events = _parse_events(args.rp_file)

        # ---------------- Simulation with hazards ---------------- #
        model_hazard = EconomyModel(
            num_households=100,
            num_firms=20,
            hazard_events=events,
            seed=args.seed,
            apply_hazard_impacts=True,
            firm_topology_path=args.topology,
            start_year=args.start_year,
            steps_per_year=args.steps_per_year,
        )
        df_hazard, model_haz_ref = run_simulation(model_hazard, args.steps)
        df_hazard["Scenario"] = "With Hazard"
        df_hazard["Step"] = df_hazard.index

        # ---------------- Baseline simulation ---------------------------- #
        model_baseline = EconomyModel(
            num_households=100,
            num_firms=20,
            hazard_events=events,
            seed=args.seed,
            apply_hazard_impacts=False,
            firm_topology_path=args.topology,
            start_year=args.start_year,
            steps_per_year=args.steps_per_year,
        )
        df_base, model_base_ref = run_simulation(model_baseline, args.steps)
        df_base["Scenario"] = "No Hazard"
        df_base["Step"] = df_base.index

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

        agent_haz = _agent_df(model_haz_ref, "With Hazard")
        agent_base = _agent_df(model_base_ref, "No Hazard")
        agent_df_combined = pd.concat([agent_haz, agent_base], ignore_index=True)

        # Persist rich agent-level results
        # Build agent-level CSV path by appending "_agents" before the extension
        out_path = Path(args.out)
        agent_csv_path = out_path.parent / f"{out_path.stem}_agents.csv"
        agent_df_combined.to_csv(agent_csv_path, index=False)
        print(f"Agent-level data saved to {agent_csv_path}")

        df_combined = pd.concat([df_hazard, df_base], ignore_index=True)

        # Persist combined DataFrame ------------------------------------------------
        csv_path = Path(args.out).with_suffix('.csv')
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
        "Average_Risk": "Score (0–1)",
        "Mean_Wage": "$ / Unit of Labor",
        "Mean_Price": "$ / Unit of Goods",
        "Labor_Limited_Firms": "count",
        "Capital_Limited_Firms": "count",
        "Input_Limited_Firms": "count",
    }

    # ---- Metric selection (5 left, 5 right) --------------------------- #
    metrics_left = [
        "Firm_Production",
        "Firm_Consumption",
        "Firm_Wealth",
        "Firm_Capital",
        "Mean_Price",
        "Average_Risk",
    ]

    metrics_right = [
        "Household_Labor_Sold",
        "Household_Consumption",
        "Household_Wealth",
        "Mean_Wage",
        "Bottleneck_Hazard",
        "Bottleneck_Baseline",
    ]

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
    sec_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_sectors)))

    firm_metric_map = {
        "Firm_Production": "production",
        "Firm_Consumption": "consumption",
        "Firm_Wealth": "money",
        "Firm_Capital": "capital",
    }

    household_metric_map = {
        "Household_Wealth": "money",
        "Household_Capital": "capital",
        "Household_Labor_Sold": "labor_sold",
        "Household_Consumption": "consumption",
    }

    def _plot_metric(metric_name: str, ax):
        if metric_name == "Mean_Price":
            # Aggregate line per scenario
            for scenario, grp in df_combined.groupby("Scenario"):
                ax.plot(grp[x_col], grp[metric_name], label=f"Mean – {scenario}")
            # Sector breakdown
            sectors = sorted(firm_agents_df["sector"].dropna().unique())
            sector_colors = plt.cm.Set1(np.linspace(0, 1, len(sectors)))
            for idx_sec, sector in enumerate(sectors):
                for scenario in ["With Hazard", "No Hazard"]:
                    df_sec = firm_agents_df[(firm_agents_df["sector"] == sector) & (firm_agents_df["Scenario"] == scenario)]
                    if df_sec.empty:
                        continue
                    price_by_step = df_sec.groupby("Step")["price"].mean()
                    x_vals = price_by_step.index if not args.start_year else args.start_year + price_by_step.index.astype(int) / args.steps_per_year
                    ax.plot(x_vals, price_by_step.values, linestyle="--", color=sector_colors[idx_sec], alpha=0.6,
                            label=f"{sector} – {scenario}")
        elif metric_name == "Sector_Trophic_Level":
            for scenario in ["With Hazard", "No Hazard"]:
                df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                for idx_sec, sector in enumerate(unique_sectors):
                    grp = df_scen[df_scen["sector"] == sector].groupby("Step")["Level"].mean()
                    if grp.empty:
                        continue
                    x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int)/args.steps_per_year
                    ls = "-" if scenario=="With Hazard" else "--"
                    ax.plot(x_vals, grp.values, color=sec_colors[idx_sec], linestyle=ls, label=f"{sector} – {scenario}")
            ax.set_ylabel("trophic level")
        elif metric_name in firm_metric_map:
            agent_col = firm_metric_map[metric_name]
            for scenario in ["With Hazard", "No Hazard"]:
                df_scen = firm_agents_df[firm_agents_df["Scenario"] == scenario]
                for idx_sec, sector in enumerate(unique_sectors):
                    grp = df_scen[df_scen["sector"] == sector].groupby("Step")[agent_col].sum()
                    if grp.empty:
                        continue
                    x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int)/args.steps_per_year
                    ls = "-" if scenario=="With Hazard" else "--"
                    ax.plot(x_vals, grp.values, color=sec_colors[idx_sec], linestyle=ls, label=f"{sector} – {scenario}")
        else:
            # Household metrics
            if metric_name in household_metric_map:
                hh_col = household_metric_map[metric_name]
                sectors = sorted(household_agents_df["sector"].dropna().unique())
                for scenario in ["With Hazard", "No Hazard"]:
                    df_scen_hh = household_agents_df[household_agents_df["Scenario"] == scenario]
                    # Sector lines
                    for idx_sec, sector in enumerate(sectors):
                        grp = df_scen_hh[df_scen_hh["sector"] == sector].groupby("Step")[hh_col].sum()
                        if grp.empty:
                            continue
                        x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int) / args.steps_per_year
                        ax.plot(x_vals, grp.values, color=sec_colors[idx_sec], linestyle="-" if scenario=="With Hazard" else "--", alpha=0.6, label=f"{sector} – {scenario}")
            else:
                # Other aggregated metrics (risk, etc.)
                for scenario, grp in df_combined.groupby("Scenario"):
                    ax.plot(grp[x_col], grp[metric_name], label=scenario)

        ax.set_title(metric_name.replace("_", " "), fontsize=10)
        ylabel = units.get(metric_name, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(x_col)
        ax.legend(fontsize=7)

    # Fill grid ----------------------------------------------------------- #
    for r in range(rows):
        # Left metrics
        _plot_metric(metrics_left[r], axes_matrix[r][0])

        # Right metrics
        if metrics_right[r] not in ("Bottleneck_Hazard", "Bottleneck_Baseline"):
            _plot_metric(metrics_right[r], axes_matrix[r][1])
        else:
            # Dedicated stacked bottleneck plots
            ax_bt = axes_matrix[r][1]

            steps = df_combined["Step"].unique()

            def _pct_arrays(df_sub):
                arrs = {}
                for bt in ["labor", "capital", "input"]:
                    cnt = df_sub[df_sub["limiting_factor"] == bt].groupby("Step").size()
                    cnt = cnt.reindex(steps, fill_value=0)
                    arrs[bt] = cnt
                tot = sum(arrs.values())
                tot[tot == 0] = 1
                return [100 * arrs[bt] / tot for bt in ["labor", "capital", "input"]]

            if metrics_right[r] == "Bottleneck_Hazard":
                df_sub = firm_agents_df[firm_agents_df["Scenario"] == "With Hazard"]
                pct_arrays = _pct_arrays(df_sub)
                label_suffix = "(Haz)"
                alpha_val = 0.7
            else:
                df_sub = firm_agents_df[firm_agents_df["Scenario"] == "No Hazard"]
                pct_arrays = _pct_arrays(df_sub)
                label_suffix = "(Base)"
                alpha_val = 0.7

            ax_bt.stackplot(steps, *pct_arrays,
                            labels=[f"Labour {label_suffix}", f"Capital {label_suffix}", f"Input {label_suffix}"],
                            colors=["#1f77b4", "#d62728", "#2ca02c"], alpha=alpha_val)

            ax_bt.set_title(f"Production Bottlenecks % {label_suffix}", fontsize=10)
            ax_bt.set_ylabel("% of firms")
            ax_bt.set_xlabel(x_col)
            ax_bt.set_ylim(0, 100)
            ax_bt.legend(fontsize=6, ncol=3)

    fig.tight_layout()
    Path(args.out).with_suffix(Path(args.out).suffix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main() 