import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        )
        df_base, model_base_ref = run_simulation(model_baseline, args.steps)
        df_base["Scenario"] = "No Hazard"
        df_base["Step"] = df_base.index

        # ---------------- Agent-level aggregation ---------------------- #
        def _compute_levels(model_obj: EconomyModel):
            firms = [ag for ag in model_obj.agents if isinstance(ag, FirmAgent)]
            id_map = {f.unique_id: f for f in firms}
            memo: dict[int, int] = {}

            def _lvl(fid: int, visiting: set[int]):
                if fid in memo:
                    return memo[fid]
                if fid in visiting:
                    return 0
                visiting.add(fid)
                f = id_map[fid]
                if not f.connected_firms:
                    l = 0
                else:
                    l = 1 + min(_lvl(s.unique_id, visiting) for s in f.connected_firms)
                visiting.remove(fid)
                memo[fid] = l
                return l

            for fid in id_map:
                _lvl(fid, set())
            return memo

        # Build agent DataFrame combined
        def _agent_df(model_obj, scenario_label):
            df_ag = model_obj.datacollector.get_agent_vars_dataframe().reset_index()
            df_ag.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")
            lvl_map = _compute_levels(model_obj)
            df_ag = df_ag[df_ag["type"] == "FirmAgent"].copy()
            df_ag["Level"] = df_ag["AgentID"].map(lvl_map)
            df_ag["Scenario"] = scenario_label
            return df_ag

        agent_haz = _agent_df(model_haz_ref, "With Hazard")
        agent_base = _agent_df(model_base_ref, "No Hazard")
        agent_df_combined = pd.concat([agent_haz, agent_base], ignore_index=True)

        # Persist rich agent-level results
        agent_csv_path = Path(args.out).with_suffix("_agents.csv")
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

    # Organise metrics: firms (col 0) vs households (col 1)
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

    rows = max(len(firm_metrics), len(household_metrics))
    extra_bottleneck_row = True
    if extra_bottleneck_row:
        rows += 1
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, rows * 3))
    gs = gridspec.GridSpec(rows, 2, height_ratios=[1]*(rows-1)+[1.2])
    axes_matrix = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(rows)]

    # Ensure axes is 2-D even if rows == 1
    if rows == 1:
        axes_matrix = axes_matrix.reshape(1, 2)

    # Helper to plot one metric
    x_col = "Year" if args.start_year else "Step"

    if args.start_year:
        df_combined["Year"] = args.start_year + df_combined["Step"].astype(int) / 4

    levels_sorted = sorted(agent_df_combined["Level"].dropna().unique())
    cmap_levels = plt.cm.viridis

    firm_metric_map = {
        "Firm_Production": "production",
        "Firm_Consumption": "consumption",
        "Firm_Wealth": "money",
        "Firm_Capital": "capital",
    }

    def _plot_metric(metric_name: str, ax):
        if metric_name == "Mean_Price":
            # Aggregate line per scenario
            for scenario, grp in df_combined.groupby("Scenario"):
                ax.plot(grp[x_col], grp[metric_name], label=f"Mean – {scenario}")
            # Sector breakdown
            sectors = sorted(agent_df_combined["sector"].dropna().unique())
            sector_colors = plt.cm.Set1(np.linspace(0, 1, len(sectors)))
            for idx_sec, sector in enumerate(sectors):
                for scenario in ["With Hazard", "No Hazard"]:
                    df_sec = agent_df_combined[(agent_df_combined["sector"] == sector) & (agent_df_combined["Scenario"] == scenario)]
                    if df_sec.empty:
                        continue
                    price_by_step = df_sec.groupby("Step")["price"].mean()
                    x_vals = price_by_step.index if not args.start_year else args.start_year + price_by_step.index.astype(int) / 4
                    ax.plot(x_vals, price_by_step.values, linestyle="--", color=sector_colors[idx_sec], alpha=0.6,
                            label=f"{sector} – {scenario}")
        elif metric_name in firm_metric_map:
            agent_col = firm_metric_map[metric_name]
            for scenario in ["With Hazard", "No Hazard"]:
                df_scen = agent_df_combined[agent_df_combined["Scenario"] == scenario]
                for idx_lvl, lvl in enumerate(levels_sorted):
                    grp = df_scen[df_scen["Level"] == lvl].groupby("Step")[agent_col].sum()
                    if grp.empty:
                        continue
                    x_vals = grp.index if not args.start_year else args.start_year + grp.index.astype(int) / 4
                    color = cmap_levels(idx_lvl / max(1, len(levels_sorted)-1))
                    ls = "-" if scenario=="With Hazard" else "--"
                    ax.plot(x_vals, grp.values, label=f"Lvl {lvl} – {scenario}", color=color, linestyle=ls)
        else:
            # household metrics etc.
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
        # Column 0 – firm metrics
        if r < len(firm_metrics):
            _plot_metric(firm_metrics[r], axes_matrix[r][0])
        else:
            if r == len(firm_metrics):
                # Plot all three bottleneck series on same axis
                ax = axes_matrix[r][0]
                bottlenecks = [
                    ("Labor_Limited_Firms", "tab:blue"),
                    ("Capital_Limited_Firms", "tab:red"),
                    ("Input_Limited_Firms", "tab:green"),
                ]
                for metric_name, color in bottlenecks:
                    for scenario, grp in df_combined.groupby("Scenario"):
                        ax.plot(grp["Step"], grp[metric_name], label=f"{metric_name} – {scenario}", color=color, alpha=0.7 if scenario=="With Hazard" else 0.4, linestyle="-" if scenario=="With Hazard" else "--")
                ax.set_title("Production Bottlenecks", fontsize=10)
                ax.set_ylabel("count")
                ax.set_xlabel("Step")
                ax.legend(fontsize=7)
            else:
                axes_matrix[r][0].set_visible(False)

        # Column 1 – household metrics
        if r < len(household_metrics):
            _plot_metric(household_metrics[r], axes_matrix[r][1])
        else:
            axes_matrix[r][1].set_visible(False)

    # Expand bottleneck plot across both columns
    gs_bott = gs[-1, :]
    ax_bott = fig.add_subplot(gs_bott)
    bottlenecks = [
        ("Labor_Limited_Firms", "tab:blue"),
        ("Capital_Limited_Firms", "tab:red"),
        ("Input_Limited_Firms", "tab:green"),
    ]
    # bottlenecks row – now detailed per level & scenario
    for metric_name, _color in bottlenecks:
        bt_type = metric_name.split("_")[0].lower()  # labor, capital, input
        for scenario in ["With Hazard", "No Hazard"]:
            df_scen = agent_df_combined[agent_df_combined["Scenario"] == scenario]
            for idx_lvl, lvl in enumerate(levels_sorted):
                mask = (df_scen["Level"]==lvl) & (df_scen["limiting_factor"]==bt_type)
                counts = df_scen[mask].groupby("Step").size()
                if counts.empty:
                    continue
                x_vals = counts.index if not args.start_year else args.start_year+counts.index.astype(int)/4
                color = cmap_levels(idx_lvl / max(1, len(levels_sorted)-1))
                ls = "-" if scenario=="With Hazard" else "--"
                ax_bott.plot(x_vals, counts.values, color=color, linestyle=ls,
                             label=f"{bt_type.capitalize()} L{lvl} {scenario}")
    ax_bott.set_title("Production Bottlenecks", fontsize=10)
    ax_bott.set_ylabel("count")
    ax_bott.set_xlabel("Step")
    ax_bott.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    # Hide the original small axes allocated for bottleneck row
    axes_matrix[-1][0].set_visible(False)
    axes_matrix[-1][1].set_visible(False)

    fig.tight_layout()
    Path(args.out).with_suffix(Path(args.out).suffix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main() 