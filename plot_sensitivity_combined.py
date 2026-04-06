"""Plot both adaptation-strategy sensitivity sweeps in one 3x4 figure.

This script reads existing sensitivity summary or member CSVs and produces a
combined figure that compares backup-supplier search and capital hardening
side-by-side for the same set of macro indicators.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ensemble_utils import build_ensemble_summary


PLOT_METRICS = [
    ("Firm_Production", "Production", "Units of Goods", "firm_mean"),
    ("Firm_Capital", "Capital", "Units of Capital", "firm_mean"),
    ("Firm_Wealth", "Firm Liquidity", "Real Units ($ / Mean Price)", "firm_mean_real"),
    ("Household_Consumption", "Consumption", "Units of Goods", "hh_mean"),
    ("Mean_Wage", "Real Wage", "Real Units ($ / Mean Price)", "real"),
    ("Mean_Price", "Price", "$ / Unit of Goods", "raw"),
]

LABEL_ORDER = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot combined sensitivity results for backup suppliers and capital hardening.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backup-csv",
        required=True,
        help="Sensitivity summary or member CSV for backup-supplier search.",
    )
    parser.add_argument(
        "--hardening-csv",
        required=True,
        help="Sensitivity summary or member CSV for capital hardening.",
    )
    parser.add_argument(
        "--out",
        default="manuscript/sensitivity_combined.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--plot-start-year",
        type=float,
        default=None,
        help="If provided, discard data before this calendar year when plotting.",
    )
    parser.add_argument(
        "--ensemble-stat",
        choices=("mean", "median"),
        default="mean",
        help="Statistic to highlight from the ensemble summaries.",
    )
    parser.add_argument(
        "--show-ensemble-members",
        action="store_true",
        help="Overlay faint member trajectories when sidecar member CSVs are available.",
    )
    parser.add_argument(
        "--num-households",
        type=int,
        default=None,
        help="Optional override for the number of households used in normalization.",
    )
    return parser.parse_args()


def _load_timeseries(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(f"Timeseries CSV not found: {path}")

    df = pd.read_csv(path)
    if "EnsembleStatistic" in df.columns:
        summary_df = df
        member_df = None
        if not path.stem.endswith("_members"):
            members_path = path.parent / f"{path.stem}_members.csv"
            if members_path.exists():
                member_df = pd.read_csv(members_path)
        return summary_df, member_df

    member_df = df
    summary_df = build_ensemble_summary(
        member_df,
        group_cols=[
            "Scenario",
            "Sensitivity_Label",
            "Adaptation_Sensitivity_Min",
            "Adaptation_Sensitivity_Max",
            "Adaptation_Sensitivity_Midpoint",
            "Step",
            "Year",
        ],
    )
    return summary_df, member_df


def _ordered_labels(*frames: pd.DataFrame) -> list[str]:
    labels: set[str] = set()
    for frame in frames:
        if frame is not None and "Sensitivity_Label" in frame.columns:
            labels.update(str(label) for label in frame["Sensitivity_Label"].dropna().unique())
    return sorted(labels, key=lambda label: (LABEL_ORDER.get(label, 99), label))


def _filter_plot_window(df: pd.DataFrame | None, plot_start_year: float | None) -> pd.DataFrame | None:
    if df is None or plot_start_year is None or "Year" not in df.columns:
        return df
    filtered = df[df["Year"].astype(float) >= float(plot_start_year)].copy()
    if filtered.empty:
        raise SystemExit(f"No rows remain after applying --plot-start-year {plot_start_year}.")
    return filtered


def _normalize_series(df: pd.DataFrame, column: str, mode: str, *, num_households: int) -> pd.Series:
    values = df[column].astype(float)
    n_firms = (
        df["Total_Firms"].replace(0, np.nan).astype(float)
        if "Total_Firms" in df.columns
        else pd.Series(1.0, index=df.index)
    )
    price = (
        df["Mean_Price"].replace(0, np.nan).astype(float)
        if "Mean_Price" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    if mode == "firm_mean":
        return values / n_firms
    if mode == "firm_mean_real":
        return (values / n_firms) / price
    if mode == "hh_mean":
        return values / float(num_households)
    if mode == "real":
        return values / price
    return values


def _infer_num_households(
    override: int | None,
    *frames: pd.DataFrame | None,
) -> int:
    if override is not None:
        return int(override)
    for frame in frames:
        if frame is None or "Meta_NumHouseholds" not in frame.columns:
            continue
        values = frame["Meta_NumHouseholds"].dropna()
        if not values.empty:
            return int(float(values.iloc[0]))
    return 100


def _range_text(summary_df: pd.DataFrame, prefix: str) -> str:
    rows = []
    for label in _ordered_labels(summary_df):
        label_df = summary_df[summary_df["Sensitivity_Label"] == label]
        if label_df.empty:
            continue
        lo = float(label_df["Adaptation_Sensitivity_Min"].iloc[0])
        hi = float(label_df["Adaptation_Sensitivity_Max"].iloc[0])
        rows.append(f"{label} [{lo:.1f}, {hi:.1f}]")
    return f"{prefix}: " + ", ".join(rows)


def _plot_panel(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    member_df: pd.DataFrame | None,
    *,
    labels: list[str],
    colors: dict[str, tuple[float, float, float, float]],
    column: str,
    norm_mode: str,
    num_households: int,
    ensemble_stat: str,
    show_ensemble_members: bool,
) -> tuple[list[float], list[float]]:
    mins: list[float] = []
    maxs: list[float] = []
    is_ensemble = "EnsembleStatistic" in summary_df.columns and summary_df["EnsembleSize"].max() > 1

    for label in labels:
        color = colors[label]
        stat_mask = summary_df["Sensitivity_Label"] == label
        stat_df = summary_df.loc[
            stat_mask & (summary_df["EnsembleStatistic"] == ensemble_stat)
        ].sort_values("Step")
        if stat_df.empty or column not in stat_df.columns:
            continue

        x_vals = stat_df["Year"] if "Year" in stat_df.columns else stat_df["Step"]

        if show_ensemble_members and member_df is not None:
            member_mask = member_df["Sensitivity_Label"] == label
            for _, seed_grp in member_df.loc[member_mask].groupby("Seed"):
                seed_grp = seed_grp.sort_values("Step")
                if column not in seed_grp.columns:
                    continue
                ax.plot(
                    seed_grp["Year"] if "Year" in seed_grp.columns else seed_grp["Step"],
                    _normalize_series(seed_grp, column, norm_mode, num_households=num_households),
                    color=color,
                    alpha=0.08,
                    linewidth=0.7,
                )

        p10_df = summary_df.loc[
            stat_mask & (summary_df["EnsembleStatistic"] == "p10")
        ].sort_values("Step")
        p90_df = summary_df.loc[
            stat_mask & (summary_df["EnsembleStatistic"] == "p90")
        ].sort_values("Step")
        if (
            is_ensemble
            and not p10_df.empty
            and not p90_df.empty
            and column in p10_df.columns
            and column in p90_df.columns
        ):
            lower = _normalize_series(p10_df, column, norm_mode, num_households=num_households)
            upper = _normalize_series(p90_df, column, norm_mode, num_households=num_households)
            mins.append(float(np.nanmin(lower.to_numpy(dtype=float))))
            maxs.append(float(np.nanmax(upper.to_numpy(dtype=float))))
            ax.fill_between(
                x_vals.to_numpy(dtype=float),
                lower.to_numpy(dtype=float),
                upper.to_numpy(dtype=float),
                color=color,
                alpha=0.12,
            )

        y_vals = _normalize_series(stat_df, column, norm_mode, num_households=num_households)
        mins.append(float(np.nanmin(y_vals.to_numpy(dtype=float))))
        maxs.append(float(np.nanmax(y_vals.to_numpy(dtype=float))))
        ax.plot(
            x_vals,
            y_vals,
            label=label,
            color=color,
            linewidth=1.8,
            alpha=0.95,
        )

    return mins, maxs


def _set_pair_ylim(left_ax: plt.Axes, right_ax: plt.Axes, mins: list[float], maxs: list[float]) -> None:
    if not mins or not maxs:
        return
    lower = min(mins)
    upper = max(maxs)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return
    if np.isclose(lower, upper):
        padding = 0.05 * max(abs(lower), 1.0)
    else:
        padding = 0.06 * (upper - lower)
    left_ax.set_ylim(lower - padding, upper + padding)
    right_ax.set_ylim(lower - padding, upper + padding)


def plot_combined(
    backup_summary: pd.DataFrame,
    backup_members: pd.DataFrame | None,
    hardening_summary: pd.DataFrame,
    hardening_members: pd.DataFrame | None,
    *,
    out_path: str,
    plot_start_year: float | None,
    ensemble_stat: str,
    show_ensemble_members: bool,
    num_households: int,
) -> None:
    backup_summary = _filter_plot_window(backup_summary, plot_start_year)
    backup_members = _filter_plot_window(backup_members, plot_start_year)
    hardening_summary = _filter_plot_window(hardening_summary, plot_start_year)
    hardening_members = _filter_plot_window(hardening_members, plot_start_year)

    labels = _ordered_labels(backup_summary, hardening_summary)
    palette = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    colors = {label: palette[idx] for idx, label in enumerate(labels)}

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    pair_positions = [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)]

    for (column, short_title, ylabel, norm_mode), (row, col_base) in zip(PLOT_METRICS, pair_positions):
        backup_ax = axes[row, col_base]
        hardening_ax = axes[row, col_base + 1]

        backup_mins, backup_maxs = _plot_panel(
            backup_ax,
            backup_summary,
            backup_members,
            labels=labels,
            colors=colors,
            column=column,
            norm_mode=norm_mode,
            num_households=num_households,
            ensemble_stat=ensemble_stat,
            show_ensemble_members=show_ensemble_members,
        )
        hardening_mins, hardening_maxs = _plot_panel(
            hardening_ax,
            hardening_summary,
            hardening_members,
            labels=labels,
            colors=colors,
            column=column,
            norm_mode=norm_mode,
            num_households=num_households,
            ensemble_stat=ensemble_stat,
            show_ensemble_members=show_ensemble_members,
        )

        backup_ax.set_title(f"{short_title} (BS)", fontsize=11)
        hardening_ax.set_title(f"{short_title} (CH)", fontsize=11)
        backup_ax.set_ylabel(ylabel, fontsize=9)
        hardening_ax.tick_params(labelleft=False)
        _set_pair_ylim(
            backup_ax,
            hardening_ax,
            backup_mins + hardening_mins,
            backup_maxs + hardening_maxs,
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=9)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=max(3, len(legend_labels)),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )

    ensemble_size = int(
        max(
            backup_summary["EnsembleSize"].max(),
            hardening_summary["EnsembleSize"].max(),
        )
    )
    stat_label = ensemble_stat.title()
    fig.suptitle(
        "Sensitivity of Outcomes to Adaptation Intensity\n"
        f"Matched {ensemble_size}-seed {stat_label} trajectories from hazard onset onward",
        fontsize=15,
        y=0.98,
    )
    fig.text(0.5, 0.915, _range_text(backup_summary, "BS ranges"), ha="center", fontsize=10)
    fig.text(0.5, 0.895, _range_text(hardening_summary, "CH ranges"), ha="center", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined sensitivity plot saved to {out_path}")


def main() -> None:
    args = _parse_args()
    backup_summary, backup_members = _load_timeseries(args.backup_csv)
    hardening_summary, hardening_members = _load_timeseries(args.hardening_csv)
    num_households = _infer_num_households(
        args.num_households,
        backup_summary,
        backup_members,
        hardening_summary,
        hardening_members,
    )
    plot_combined(
        backup_summary,
        backup_members,
        hardening_summary,
        hardening_members,
        out_path=args.out,
        plot_start_year=args.plot_start_year,
        ensemble_stat=args.ensemble_stat,
        show_ensemble_members=args.show_ensemble_members,
        num_households=num_households,
    )


if __name__ == "__main__":
    main()
