#!/usr/bin/env python3
"""Plot systemic cascade exposure diagnostics for never-directly-hit firms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from plot_from_csv_paper import (
    infer_scenario_name,
    is_hazard_scenario,
    scenario_abbrev,
    scenario_main_color,
    summarize_members_for_plot,
)


CASCADE_METRICS = [
    (
        "Ever_Directly_Hit_Firm_Share",
        "Firms Ever Directly Hit",
        "Percent of Firms",
    ),
    (
        "Never_Hit_Currently_Disrupted_Firm_Share",
        "Never-Hit Firms with Supplier Disruption",
        "Percent of Firms",
    ),
    (
        "Never_Hit_Supplier_Disruption_Burden_Share",
        "Supplier Disruption Borne by Never-Hit Firms",
        "Percent of Supplier Disruption",
    ),
    (
        "Never_Hit_Production_Share",
        "Output Produced by Never-Hit Firms",
        "Percent of Aggregate Production",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot systemic cascade diagnostics from hazard scenario summaries",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-files",
        nargs="+",
        required=True,
        help="Hazard scenario summary CSVs or member CSVs",
    )
    parser.add_argument(
        "--out",
        default="cascade_risk.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--ensemble-stat",
        choices=("mean", "median"),
        default="mean",
        help="Central ensemble statistic to plot",
    )
    parser.add_argument(
        "--show-ensemble-members",
        action="store_true",
        help="Overlay faint individual seed trajectories when *_members.csv is available",
    )
    parser.add_argument(
        "--show-ensemble-band",
        action="store_true",
        help="Show the scenario-specific p10-p90 band when available",
    )
    parser.add_argument(
        "--title",
        default="Systemic Cascade Exposure Under RCP8.5",
        help="Figure title",
    )
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    member_frames: list[pd.DataFrame] = []
    band_frames: list[pd.DataFrame] = []

    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path}")

        if "Scenario" not in df.columns:
            scenario_name = infer_scenario_name(csv_path)
            df["Scenario"] = scenario_name
        else:
            non_null_scenarios = df["Scenario"].dropna()
            scenario_name = (
                str(non_null_scenarios.iloc[0])
                if not non_null_scenarios.empty
                else infer_scenario_name(csv_path)
            )

        if not is_hazard_scenario(scenario_name):
            print(f"Skipping non-hazard scenario in cascade figure: {scenario_name}")
            continue

        if "Step" not in df.columns:
            df["Step"] = df.index

        if "EnsembleStatistic" in df.columns:
            if args.show_ensemble_band:
                band_subset = df[df["EnsembleStatistic"].isin(["p10", "p90"])].copy()
                if not band_subset.empty:
                    band_frames.append(band_subset)
            selected = df[df["EnsembleStatistic"] == args.ensemble_stat].copy()
            if selected.empty:
                raise ValueError(f"{csv_path} does not contain EnsembleStatistic={args.ensemble_stat}")
            df = selected
            if args.show_ensemble_members:
                members_path = csv_path.parent / f"{csv_path.stem}_members.csv"
                if members_path.exists():
                    member_df = pd.read_csv(members_path)
                    if "Scenario" not in member_df.columns:
                        member_df["Scenario"] = scenario_name
                    if "Step" not in member_df.columns:
                        member_df["Step"] = member_df.index
                    member_frames.append(member_df)
                    print(f"Loaded ensemble members from {members_path}")
                else:
                    print(f"Warning: No ensemble members sidecar found for {csv_path}")
        elif "Seed" in df.columns:
            if args.show_ensemble_members:
                member_frames.append(df.copy())
            summary_df = summarize_members_for_plot(df)
            if args.show_ensemble_band:
                band_subset = summary_df[summary_df["EnsembleStatistic"].isin(["p10", "p90"])].copy()
                if not band_subset.empty:
                    band_frames.append(band_subset)
            df = summary_df[summary_df["EnsembleStatistic"] == args.ensemble_stat].copy()

        summary_frames.append(df)

    if not summary_frames:
        raise ValueError("No hazard scenario CSVs were loaded for cascade plotting.")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    member_df = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame()
    band_df = pd.concat(band_frames, ignore_index=True) if band_frames else pd.DataFrame()
    return summary_df, member_df, band_df


def main() -> None:
    args = parse_args()
    df_combined, member_df_combined, band_df_combined = _load_inputs(args)

    required_cols = [metric for metric, _, _ in CASCADE_METRICS]
    missing = [col for col in required_cols if col not in df_combined.columns]
    if missing:
        raise ValueError(
            "Cascade reporters are missing from the provided summaries. "
            f"Re-run the simulation with the updated code to populate: {', '.join(missing)}"
        )

    x_col = "Year" if "Year" in df_combined.columns else "Step"
    scenarios = sorted(df_combined["Scenario"].dropna().unique())
    if not scenarios:
        raise ValueError("No hazard scenarios available after filtering.")

    scenario_style_map = {
        scenario: {
            "color": scenario_main_color(scenario),
            "linewidth": 2.0,
        }
        for scenario in scenarios
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    def plot_ensemble_context(metric_col: str, scenario: str, ax) -> None:
        style = scenario_style_map[scenario]

        if args.show_ensemble_members and not member_df_combined.empty and metric_col in member_df_combined.columns:
            member_subset = member_df_combined[member_df_combined["Scenario"] == scenario]
            if not member_subset.empty:
                for _, member_grp in member_subset.groupby("Seed"):
                    member_grp = member_grp.sort_values(x_col)
                    ax.plot(
                        member_grp[x_col].to_numpy(),
                        member_grp[metric_col].to_numpy(dtype=float) * 100.0,
                        color=style["color"],
                        linewidth=0.8,
                        alpha=0.12,
                        zorder=1,
                    )

        if args.show_ensemble_band and not band_df_combined.empty and metric_col in band_df_combined.columns:
            band_subset = band_df_combined[band_df_combined["Scenario"] == scenario]
            p10 = band_subset[band_subset["EnsembleStatistic"] == "p10"][["Step", metric_col] + (["Year"] if "Year" in band_subset.columns else [])].copy()
            p90 = band_subset[band_subset["EnsembleStatistic"] == "p90"][["Step", metric_col] + (["Year"] if "Year" in band_subset.columns else [])].copy()
            key_cols = [x_col]
            p10 = p10.sort_values(x_col).rename(columns={metric_col: f"{metric_col}_p10"})
            p90 = p90.sort_values(x_col).rename(columns={metric_col: f"{metric_col}_p90"})
            band = p10.merge(p90[[x_col, f"{metric_col}_p90"]], on=key_cols, how="inner")
            if not band.empty:
                ax.fill_between(
                    band[x_col].to_numpy(),
                    band[f"{metric_col}_p10"].to_numpy(dtype=float) * 100.0,
                    band[f"{metric_col}_p90"].to_numpy(dtype=float) * 100.0,
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0,
                    zorder=2,
                )

    for idx, (metric_col, title, ylabel) in enumerate(CASCADE_METRICS):
        ax = axes[idx]
        for scenario in scenarios:
            scenario_df = df_combined[df_combined["Scenario"] == scenario].sort_values(x_col)
            if scenario_df.empty:
                continue
            plot_ensemble_context(metric_col, scenario, ax)
            style = scenario_style_map[scenario]
            ax.plot(
                scenario_df[x_col].to_numpy(),
                scenario_df[metric_col].to_numpy(dtype=float) * 100.0,
                color=style["color"],
                linewidth=style["linewidth"],
                label=f"{scenario_abbrev(scenario)} ({args.ensemble_stat})",
                zorder=3,
            )

        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Year" if x_col == "Year" else "Step")
        ax.set_ylim(0.0, 100.0)
        ax.grid(alpha=0.15, linewidth=0.4)
        label_char = chr(ord("a") + idx)
        ax.text(
            -0.12,
            1.02,
            f"({label_char})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    legend_pairs = []
    for scenario in scenarios:
        style = scenario_style_map[scenario]
        legend_pairs.append(
            (
                Line2D([0], [0], color=style["color"], linewidth=2.5),
                f"{scenario_abbrev(scenario)} ({args.ensemble_stat})",
            )
        )
    if args.show_ensemble_members and not member_df_combined.empty:
        legend_pairs.append(
            (
                Line2D([0], [0], color="#666666", linewidth=1.0, alpha=0.3),
                "Individual seed",
            )
        )
    if args.show_ensemble_band and not band_df_combined.empty:
        legend_pairs.append(
            (
                Patch(facecolor="#999999", edgecolor="none", alpha=0.18),
                "Scenario p10-p90 band",
            )
        )

    fig.legend(
        [handle for handle, _ in legend_pairs],
        [label for _, label in legend_pairs],
        loc="lower center",
        ncol=min(len(legend_pairs), 4),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(args.title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Cascade-risk plot saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
