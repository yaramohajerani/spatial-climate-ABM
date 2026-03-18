"""Sensitivity analysis for evolutionary learning memory window.

Runs the hazard+learning scenario across a range of memory_length
values to verify that qualitative conclusions are robust to the
choice of production averaging window.

Usage:
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json
    python sensitivity_analysis.py --param-file aqueduct_riverine_parameters_rcp8p5.json --quick
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import EconomyModel


# ------------------------------------------------------------------ #
#                   Memory length configurations                      #
# ------------------------------------------------------------------ #

# At steps_per_year=4, memory_length in steps maps to years as:
#   5 → 1.25 yr, 8 → 2 yr, 10 → 2.5 yr, 15 → 3.75 yr, 20 → 5 yr
MEMORY_CONFIGS = {
    "5 steps (1.25 yr)": 5,
    "8 steps (2.0 yr)": 8,
    "10 steps (2.5 yr, default)": 10,
    "15 steps (3.75 yr)": 15,
    "20 steps (5.0 yr)": 20,
}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Memory window sensitivity analysis for evolutionary learning.",
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


def run_scenario(params: dict, events: list, memory_length: int,
                 n_steps: int, seed: int) -> pd.DataFrame:
    """Run a single hazard+learning scenario with the given memory window."""
    learning_config = params.get("learning", {})
    learning_config = {**learning_config, "enabled": True, "memory_length": memory_length}

    model = EconomyModel(
        num_households=int(params.get("num_households", 100)),
        num_firms=20,
        hazard_events=events,
        seed=seed,
        apply_hazard_impacts=True,
        firm_topology_path=params.get("topology"),
        start_year=int(params.get("start_year", 0)),
        steps_per_year=int(params.get("steps_per_year", 4)),
        learning_params=learning_config,
        consumption_ratios=params.get("consumption_ratios"),
        grid_resolution=float(params.get("grid_resolution", 1.0)),
        household_relocation=bool(params.get("household_relocation", True)),
    )

    for _ in range(n_steps):
        model.step()

    return model.results_to_dataframe()


def plot_sensitivity(results: dict[str, pd.DataFrame], out_path: str,
                     num_households: int):
    """Plot key metrics across all memory length configurations."""
    metrics = [
        ("Firm_Production", "Mean Firm Production", "Units", "firm_mean"),
        ("Household_Labor_Sold", "Employment Rate", "Fraction", "employment"),
        ("Firm_Capital", "Mean Firm Capital", "Units", "firm_mean"),
        ("Firm_Wealth", "Mean Firm Liquidity", "$", "firm_mean"),
        ("Mean_Wage", "Mean Wage", "$ / Unit of Labour", "raw"),
        ("Mean_Price", "Mean Price", "$ / Unit of Goods", "raw"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for idx, (col, title, ylabel, norm_mode) in enumerate(metrics):
        ax = axes[idx]
        for (label, df), color in zip(results.items(), colors):
            x = df["Year"] if "Year" in df.columns else df.index
            n_firms = df["Total_Firms"]

            if norm_mode == "firm_mean":
                y = df[col] / n_firms
            elif norm_mode == "employment":
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

    fig.suptitle("Sensitivity of Outcomes to Fitness Memory Window Length\n"
                 "(Hazard + Learning scenario)",
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

        prod = last_decade["Firm_Production"] / n_firms
        empl = last_decade["Household_Labor_Sold"] / num_households
        cap = last_decade["Firm_Capital"] / n_firms
        wage = last_decade["Mean_Wage"]
        price = last_decade["Mean_Price"]

        rows.append({
            "Memory Window": label,
            "Production": f"{prod.mean():.1f} ± {prod.std():.1f}",
            "Employment": f"{empl.mean():.2f} ± {empl.std():.2f}",
            "Capital": f"{cap.mean():.1f} ± {cap.std():.1f}",
            "Wage": f"{wage.mean():.2f} ± {wage.std():.2f}",
            "Price": f"{price.mean():.1f} ± {price.std():.1f}",
        })

    return pd.DataFrame(rows)


def main():
    args = _parse_args()
    params = _load_params(args.param_file)
    events = _parse_events(params.get("rp_files", []))

    seed = args.seed if args.seed is not None else int(params.get("seed", 42))
    n_steps = 50 if args.quick else int(params.get("steps", 300))
    steps_per_year = int(params.get("steps_per_year", 4))
    num_households = int(params.get("num_households", 100))

    results: dict[str, pd.DataFrame] = {}

    for label, mem_len in MEMORY_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running: {label}  memory_length={mem_len}")
        print(f"{'='*60}")
        df = run_scenario(params, events, mem_len, n_steps, seed)
        results[label] = df

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
