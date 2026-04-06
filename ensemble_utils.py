"""Shared helpers for ensemble summaries and metadata application."""

from __future__ import annotations

import numpy as np
import pandas as pd

ENSEMBLE_STAT_ORDER = ["mean", "median", "std", "p10", "p90"]
METADATA_PREFIX = "Meta_"
MERGE_VARIABLE_METADATA = {
    f"{METADATA_PREFIX}RunTimestamp",
    f"{METADATA_PREFIX}SeedCount",
    f"{METADATA_PREFIX}SeedList",
    f"{METADATA_PREFIX}SeedMin",
    f"{METADATA_PREFIX}SeedMax",
    f"{METADATA_PREFIX}SourceMemberFiles",
}


def apply_metadata(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    """Return a copy of *df* with constant metadata columns appended."""
    df = df.copy()
    for key, value in metadata.items():
        df[key] = value
    return df


def ensemble_seed_metadata(seed_values: list[int]) -> dict[str, object]:
    """Return normalized seed metadata for a run or merged ensemble."""
    ordered = [int(seed) for seed in seed_values]
    if not ordered:
        return {
            f"{METADATA_PREFIX}SeedCount": 0,
            f"{METADATA_PREFIX}SeedList": "",
            f"{METADATA_PREFIX}SeedMin": "",
            f"{METADATA_PREFIX}SeedMax": "",
        }
    return {
        f"{METADATA_PREFIX}SeedCount": len(ordered),
        f"{METADATA_PREFIX}SeedList": ",".join(str(seed) for seed in ordered),
        f"{METADATA_PREFIX}SeedMin": min(ordered),
        f"{METADATA_PREFIX}SeedMax": max(ordered),
    }


def build_ensemble_summary(
    member_df: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    """Summarize member-level outputs by the given grouping columns."""
    if member_df.empty:
        return member_df.copy()

    active_group_cols = [col for col in group_cols if col in member_df.columns]
    numeric_cols = [
        col
        for col in member_df.select_dtypes(include=[np.number]).columns
        if col not in set(active_group_cols + ["Seed"])
        and not col.startswith(METADATA_PREFIX)
    ]
    grouped = member_df.groupby(active_group_cols, sort=True)
    ensemble_size = grouped["Seed"].nunique().rename("EnsembleSize").reset_index()
    frames: list[pd.DataFrame] = []
    aggregations = {
        "mean": grouped[numeric_cols].mean(),
        "median": grouped[numeric_cols].median(),
        "std": grouped[numeric_cols].std().fillna(0.0),
        "p10": grouped[numeric_cols].quantile(0.10),
        "p90": grouped[numeric_cols].quantile(0.90),
    }
    for stat in ENSEMBLE_STAT_ORDER:
        stat_df = aggregations[stat].reset_index()
        stat_df["EnsembleStatistic"] = stat
        stat_df = stat_df.merge(ensemble_size, on=active_group_cols, how="left")
        frames.append(stat_df)

    summary_df = pd.concat(frames, ignore_index=True)
    summary_df["EnsembleStatistic"] = pd.Categorical(
        summary_df["EnsembleStatistic"],
        categories=ENSEMBLE_STAT_ORDER,
        ordered=True,
    )
    return summary_df.sort_values(active_group_cols + ["EnsembleStatistic"]).reset_index(drop=True)
