import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model import EconomyModel


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run two simulations – one with climate hazards and one without – and plot both timeseries side-by-side.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--rp-file",
        action="append",
        metavar="RP:TYPE:PATH",
        required=True,
        help="Add a GeoTIFF file. Format: <RP>:<HAZARD_TYPE>:<path>. Can be used multiple times.",
    )
    p.add_argument("--steps", type=int, default=10, help="Number of timesteps / years to simulate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--out", type=str, default="comparison_plot.png", help="Output plot file")
    p.add_argument("--topology", type=str, help="Optional JSON file describing firm supply-chain topology")
    return p.parse_args()


def _parse_events(rp_files: list[str]):
    events: list[tuple[int, str, str]] = []
    for item in rp_files:
        try:
            rp_str, type_str, path_str = item.split(":", 2)
            events.append((int(rp_str), type_str, path_str))
        except ValueError as exc:
            raise SystemExit(f"Invalid --rp-file format: {item}. Expected <RP>:<TYPE>:<path>.") from exc
    return events


def run_simulation(model: EconomyModel, n_steps: int):
    for _ in range(n_steps):
        model.step()
    return model.results_to_dataframe()


def main():
    args = _parse_args()
    events = _parse_events(args.rp_file)

    # ---------------- Simulation with hazards ---------------- #
    model_hazard = EconomyModel(
        num_households=100,
        num_firms=20,
        hazard_events=events,
        seed=args.seed,
        apply_hazard_impacts=True,
        firm_topology_path=args.topology,
    )
    df_hazard = run_simulation(model_hazard, args.steps)
    df_hazard["Scenario"] = "With Hazard"
    df_hazard["Step"] = df_hazard.index  # preserve timestep before concatenation

    # ---------------- Baseline simulation (no hazard impacts) -------------- #
    model_baseline = EconomyModel(
        num_households=100,
        num_firms=20,
        hazard_events=events,
        seed=args.seed,
        apply_hazard_impacts=False,
        firm_topology_path=args.topology,
    )
    df_base = run_simulation(model_baseline, args.steps)
    df_base["Scenario"] = "No Hazard"
    df_base["Step"] = df_base.index

    # Combine results ------------------------------------------------------ #
    df_combined = pd.concat([df_hazard, df_base], ignore_index=True)

    # Persist combined DataFrame so users can inspect numeric differences
    csv_path = Path(args.out).with_suffix('.csv')
    df_combined.to_csv(csv_path, index=False)
    print(f"Combined results saved to {csv_path}")

    # Plot each metric ------------------------------------------------------ #
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
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 3), sharex=True)

    # Ensure axes is 2-D even if rows == 1
    if rows == 1:
        axes = axes.reshape(1, 2)

    # Helper to plot one metric
    def _plot_metric(metric_name: str, ax):
        for scenario, grp in df_combined.groupby("Scenario"):
            ax.plot(grp["Step"], grp[metric_name], label=scenario)
        ax.set_title(metric_name.replace("_", " "), fontsize=10)
        ylabel = units.get(metric_name, "")
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.set_xlabel("Step")
        ax.legend()

    # Fill grid ----------------------------------------------------------- #
    for r in range(rows):
        # Column 0 – firm metrics
        if r < len(firm_metrics):
            _plot_metric(firm_metrics[r], axes[r, 0])
        else:
            axes[r, 0].set_visible(False)

        # Column 1 – household metrics
        if r < len(household_metrics):
            _plot_metric(household_metrics[r], axes[r, 1])
        else:
            axes[r, 1].set_visible(False)

    fig.tight_layout()
    Path(args.out).with_suffix(Path(args.out).suffix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main() 