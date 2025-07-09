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
    metrics = [col for col in df_combined.columns if col not in {"Scenario", "Step"}]
    n_metrics = len(metrics)
    cols = 2
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3), sharex=True)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for scenario, grp in df_combined.groupby("Scenario"):
            ax.plot(grp["Step"], grp[metric], label=scenario)
        ax.set_title(metric)
        ax.set_xlabel("Step")
        ax.legend()

    # Hide unused subplots
    for ax in axes[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    Path(args.out).with_suffix(Path(args.out).suffix).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Comparison plot saved to {args.out}")


if __name__ == "__main__":
    main() 