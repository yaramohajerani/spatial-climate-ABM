"""Sensitivity analysis for hazard-conditional adaptation exploration.

Runs the hazard+adaptation scenario across a range of UCB exploration
coefficients to verify that qualitative conclusions are robust to the
bandit's exploration intensity.

Usage:
    # Full run (saves timeseries + summary CSVs and plot)
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json

    # Quick test (50 steps)
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --quick

    # Re-plot from existing timeseries CSV (no simulation needed)
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --from-csv sensitivity_timeseries.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import EconomyModel


# ------------------------------------------------------------------ #
#                   UCB exploration configurations                    #
# ------------------------------------------------------------------ #

UCB_CONFIGS = {
    "UCB c=0.25": 0.25,
    "UCB c=0.50": 0.50,
    "UCB c=1.00 (default)": 1.00,
    "UCB c=2.00": 2.00,
}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Sensitivity analysis for contextual-bandit exploration strength.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--param-file", required=True,
        help="JSON parameter file (same format as run_simulation.py)",
    )
    p.add_argument(
        "--out", default="sensitivity_analysis.png",
        help="Output plot filename",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Run fewer steps for quick testing (50 steps instead of full)",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed from param file",
    )
    p.add_argument(
        "--from-csv", default=None,
        help="Path to existing sensitivity timeseries CSV; skips simulation and just plots",
    )
    return p.parse_args()


def _load_params(param_file: str) -> dict:
    """Load and return parameter dict from JSON file."""
    pth = Path(param_file)
    if not pth.exists():
        raise SystemExit(f"Parameter file not found: {pth}")
    return json.loads(pth.read_text())


def _parse_events(rp_files: list[str]):
    """Parse RP file strings into event tuples (same logic as run_simulation.py)."""
    events = []
    for item in rp_files:
        try:
            rp_str, start_str, end_str, type_str, path_str = item.split(":", 4)
            events.append((int(rp_str), int(start_str), int(end_str), type_str, path_str))
        except ValueError as exc:
            raise SystemExit(
                f"Invalid rp_file format: {item}. Expected <RP>:<START>:<END>:<TYPE>:<path>."
            ) from exc
    return events


def run_scenario(params: dict, events: list, ucb_c: float,
                 n_steps: int, seed: int) -> pd.DataFrame:
    """Run a single hazard+adaptation scenario with the given UCB coefficient."""
    adaptation_config = params.get("adaptation", params.get("learning", {}))
    adaptation_config = {**adaptation_config, "enabled": True, "ucb_c": ucb_c}

    model = EconomyModel(
        num_households=int(params.get("num_households", 100)),
        num_firms=20,
        hazard_events=events,
        seed=seed,
        apply_hazard_impacts=True,
        firm_topology_path=params.get("topology"),
        start_year=int(params.get("start_year", 0)),
        steps_per_year=int(params.get("steps_per_year", 4)),
        adaptation_params=adaptation_config,
        consumption_ratios=params.get("consumption_ratios"),
        grid_resolution=float(params.get("grid_resolution", 1.0)),
        household_relocation=bool(params.get("household_relocation", True)),
    )

    for _ in range(n_steps):
        model.step()

    return model.results_to_dataframe()


# Columns to save in the timeseries CSV
TIMESERIES_COLS = [
    "Year", "Step", "Firm_Production", "Firm_Capital", "Firm_Wealth",
    "Mean_Wage", "Mean_Price", "Household_Consumption",
    "Household_Labor_Sold", "Total_Firms",
]


def save_timeseries(results: dict[str, pd.DataFrame], out_path: str):
    """Save per-step timeseries data for all UCB configs to a single CSV."""
    frames = []
    for label, df in results.items():
        kept = df[[c for c in TIMESERIES_COLS if c in df.columns]].copy()
        kept.insert(0, "UCB_Exploration", label)
        frames.append(kept)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"Timeseries data saved to {out_path}")


def load_timeseries(csv_path: str) -> dict[str, pd.DataFrame]:
    """Load timeseries CSV back into a dict keyed by UCB exploration label."""
    df = pd.read_csv(csv_path)
    results = {}
    for label, grp in df.groupby("UCB_Exploration", sort=False):
        results[label] = grp.drop(columns=["UCB_Exploration"]).reset_index(drop=True)
    return results


def plot_sensitivity(results: dict[str, pd.DataFrame], out_path: str,
                     num_households: int):
    """Plot key metrics across all UCB exploration configurations.

    Panels match the main timeseries figure layout:
      Production, Capital, Liquidity (real), Consumption, Wage (real), Price
    """
    # (column, title, ylabel, normalisation mode)
    metrics = [
        ("Firm_Production", "Mean Firm Production", "Units of Goods", "firm_mean"),
        ("Firm_Capital", "Mean Firm Capital", "Units of Capital", "firm_mean"),
        ("Firm_Wealth", "Mean Firm Liquidity", "Real Units ($ / Mean Price)", "firm_mean_real"),
        ("Household_Consumption", "Household Consumption", "Units of Goods", "hh_mean"),
        ("Mean_Wage", "Mean Wage", "Real Units ($ / Mean Price)", "real"),
        ("Mean_Price", "Mean Price", "$ / Unit of Goods", "raw"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for idx, (col, title, ylabel, norm_mode) in enumerate(metrics):
        ax = axes[idx]
        for (label, df), color in zip(results.items(), colors):
            if col not in df.columns:
                continue
            x = df["Year"] if "Year" in df.columns else df.index
            n_firms = df["Total_Firms"]
            price = df["Mean_Price"].replace(0, np.nan)

            if norm_mode == "firm_mean":
                y = df[col] / n_firms
            elif norm_mode == "firm_mean_real":
                y = (df[col] / n_firms) / price
            elif norm_mode == "real":
                y = df[col] / price
            elif norm_mode == "hh_mean":
                y = df[col] / num_households
            else:
                y = df[col]

            ax.plot(x, y, label=label, color=color, linewidth=1.5, alpha=0.85)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Year" if "Year" in next(iter(results.values())).columns else "Step",
                       fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Sensitivity of Outcomes to UCB Exploration Strength\n"
                 "(Hazard + Adaptation scenario)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sensitivity plot saved to {out_path}")


def make_summary_table(results: dict[str, pd.DataFrame],
                       steps_per_year: int,
                       num_households: int) -> pd.DataFrame:
    """Summary table of key metrics in the final decade across configs."""
    rows = []
    for label, df in results.items():
        last_decade = df.tail(steps_per_year * 10)
        n_firms = last_decade["Total_Firms"].mean()
        price = last_decade["Mean_Price"]

        prod = last_decade["Firm_Production"] / n_firms
        cap = last_decade["Firm_Capital"] / n_firms
        wage_nominal = last_decade["Mean_Wage"]
        wage_real = wage_nominal / price
        consumption = last_decade["Household_Consumption"] / num_households

        row = {
            "UCB Exploration": label,
            "Production": f"{prod.mean():.1f} ± {prod.std():.1f}",
            "Capital": f"{cap.mean():.1f} ± {cap.std():.1f}",
            "Real Wage": f"{wage_real.mean():.2f} ± {wage_real.std():.2f}",
            "Consumption": f"{consumption.mean():.2f} ± {consumption.std():.2f}",
            "Price": f"{price.mean():.1f} ± {price.std():.1f}",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = _parse_args()
    params = _load_params(args.param_file)
    steps_per_year = int(params.get("steps_per_year", 4))
    num_households = int(params.get("num_households", 100))

    if args.from_csv:
        # Re-plot from existing timeseries data
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            raise SystemExit(f"Timeseries CSV not found: {csv_path}")
        print(f"Loading timeseries from {csv_path}")
        results = load_timeseries(str(csv_path))
    else:
        # Run simulations
        events = _parse_events(params.get("rp_files", []))
        seed = args.seed if args.seed is not None else int(params.get("seed", 42))
        n_steps = 50 if args.quick else int(params.get("steps", 300))

        results: dict[str, pd.DataFrame] = {}

        for label, ucb_c in UCB_CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"Running: {label}  ucb_c={ucb_c}")
            print(f"{'='*60}")
            df = run_scenario(params, events, ucb_c, n_steps, seed)
            results[label] = df

        # Save full timeseries for future re-plotting
        ts_path = Path(args.out).with_name("sensitivity_timeseries.csv")
        save_timeseries(results, str(ts_path))

    plot_sensitivity(results, args.out, num_households)

    summary = make_summary_table(results, steps_per_year, num_households)
    print(f"\n{'='*80}")
    print("SENSITIVITY ANALYSIS SUMMARY (last-decade averages)")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

    table_path = Path(args.out).with_suffix(".csv")
    summary.to_csv(table_path, index=False)
    print(f"\nSummary table saved to {table_path}")


if __name__ == "__main__":
    main()
