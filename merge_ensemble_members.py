#!/usr/bin/env python3
"""Merge multiple ensemble member batches into one larger ensemble summary."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from ensemble_utils import (
    MERGE_VARIABLE_METADATA,
    METADATA_PREFIX,
    apply_metadata,
    build_ensemble_summary as summarize_ensemble,
    ensemble_seed_metadata,
)


REQUIRED_MEMBER_COLUMNS = {"Scenario", "Step", "Seed"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple ensemble member CSV batches and regenerate a combined summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--member-files",
        nargs="+",
        required=True,
        help="Paths to one or more ensemble member CSVs (the *_members.csv files).",
    )
    parser.add_argument(
        "--out-prefix",
        help=(
            "Output file stem. Writes <prefix>.csv and <prefix>_members.csv. "
            "If omitted, a merged stem is derived from the first input."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-seeds",
        action="store_true",
        help="Allow repeated seed IDs across batches. Disabled by default because overlaps usually indicate a mistake.",
    )
    return parser.parse_args()


def build_ensemble_summary(member_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize member-level model outputs by step and scenario."""
    return summarize_ensemble(member_df, group_cols=["Scenario", "Step", "Year"])


def _step_year_map(df: pd.DataFrame) -> dict | None:
    if "Year" not in df.columns:
        return None
    mapping = (
        df.dropna(subset=["Step", "Year"])
        .drop_duplicates(subset="Step")
        .sort_values("Step")
        .set_index("Step")["Year"]
        .to_dict()
    )
    return mapping or None


def _metadata_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith(METADATA_PREFIX)]


def _constant_metadata(df: pd.DataFrame, *, ignore: set[str] | None = None) -> dict[str, object]:
    ignore = ignore or set()
    metadata: dict[str, object] = {}
    for col in _metadata_columns(df):
        if col in ignore:
            continue
        values = df[col].drop_duplicates().tolist()
        if len(values) > 1:
            raise ValueError(f"{col} varies within a single member file and cannot be merged safely.")
        metadata[col] = values[0] if values else ""
    return metadata


def _apply_metadata(df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    return apply_metadata(df, metadata)


def _ensemble_seed_metadata(seed_values: list[int]) -> dict[str, object]:
    return ensemble_seed_metadata(sorted(int(seed) for seed in seed_values))


def merge_member_dataframes(
    frames: list[pd.DataFrame],
    *,
    labels: list[str] | None = None,
    allow_duplicate_seeds: bool = False,
):
    """Validate and merge multiple member-level dataframes."""
    if not frames:
        raise ValueError("No member dataframes were provided.")

    labels = labels or [f"frame_{idx}" for idx in range(len(frames))]
    if len(labels) != len(frames):
        raise ValueError("labels length must match frames length")

    reference_columns: list[str] | None = None
    reference_scenario: str | None = None
    reference_steps: tuple | None = None
    reference_year_map: dict | None = None
    reference_metadata: dict[str, object] | None = None
    seen_seeds: set[int] = set()
    normalized_frames: list[pd.DataFrame] = []

    for label, df in zip(labels, frames):
        missing = REQUIRED_MEMBER_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
        if "EnsembleStatistic" in df.columns:
            raise ValueError(f"{label} looks like a summary CSV, not a member CSV.")

        scenario_values = sorted(df["Scenario"].dropna().unique())
        if len(scenario_values) != 1:
            raise ValueError(f"{label} must contain exactly one Scenario value, found {scenario_values}")
        scenario = str(scenario_values[0])

        column_set = set(df.columns)
        if reference_columns is None:
            reference_columns = list(df.columns)
        elif column_set != set(reference_columns):
            raise ValueError(f"{label} has incompatible columns relative to the first member file.")

        if reference_scenario is None:
            reference_scenario = scenario
        elif scenario != reference_scenario:
            raise ValueError(
                f"{label} has Scenario={scenario!r}, expected {reference_scenario!r}."
            )

        step_signature = tuple(sorted(df["Step"].dropna().unique().tolist()))
        if reference_steps is None:
            reference_steps = step_signature
        elif step_signature != reference_steps:
            raise ValueError(f"{label} has a different Step grid from the first member file.")

        year_map = _step_year_map(df)
        if reference_year_map is None:
            reference_year_map = year_map
        elif year_map != reference_year_map:
            raise ValueError(f"{label} has a different Step→Year mapping from the first member file.")

        metadata = _constant_metadata(df, ignore=MERGE_VARIABLE_METADATA)
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise ValueError(
                f"{label} has metadata that does not match the first member file."
            )

        seed_values = {int(seed) for seed in df["Seed"].dropna().unique()}
        overlapping = sorted(seed_values & seen_seeds)
        if overlapping and not allow_duplicate_seeds:
            raise ValueError(
                f"{label} contains duplicate seeds already present in earlier files: {overlapping}"
            )
        seen_seeds |= seed_values
        normalized_frames.append(df.reindex(columns=reference_columns).copy())

    merged_df = pd.concat(normalized_frames, ignore_index=True)
    sort_cols = [col for col in ["Scenario", "Seed", "Step"] if col in merged_df.columns]
    merged_df = merged_df.sort_values(sort_cols).reset_index(drop=True)
    merged_metadata = {
        **(reference_metadata or {}),
        **_ensemble_seed_metadata(sorted(int(seed) for seed in merged_df["Seed"].dropna().unique())),
        f"{METADATA_PREFIX}RunTimestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        f"{METADATA_PREFIX}SourceMemberFiles": ";".join(labels),
    }
    merged_df = _apply_metadata(merged_df, merged_metadata)
    summary_df = build_ensemble_summary(merged_df)
    summary_df = _apply_metadata(summary_df, merged_metadata)
    return merged_df, summary_df


def derive_output_prefix(member_paths: list[Path], *, total_seeds: int) -> Path:
    """Create a sensible merged output stem from the first member file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_stem = member_paths[0].stem
    if first_stem.endswith("_members"):
        first_stem = first_stem[: -len("_members")]
    first_stem = re.sub(r"_ensemble\d+_\d{8}_\d{6}$", "", first_stem)
    return member_paths[0].with_name(f"{first_stem}_ensemble{total_seeds}_{timestamp}")


def resolve_output_paths(out_prefix: str | None, member_paths: list[Path], total_seeds: int):
    """Return summary and member output paths."""
    if out_prefix:
        prefix_path = Path(out_prefix)
    else:
        prefix_path = derive_output_prefix(member_paths, total_seeds=total_seeds)

    if prefix_path.suffix == ".csv":
        summary_path = prefix_path
        members_path = prefix_path.with_name(f"{prefix_path.stem}_members.csv")
    else:
        summary_path = prefix_path.with_name(f"{prefix_path.name}.csv")
        members_path = prefix_path.with_name(f"{prefix_path.name}_members.csv")
    return summary_path, members_path


def main() -> None:
    args = parse_args()
    member_paths = [Path(path) for path in args.member_files]
    for path in member_paths:
        if not path.exists():
            raise FileNotFoundError(f"Member file not found: {path}")

    frames = [pd.read_csv(path) for path in member_paths]
    merged_df, summary_df = merge_member_dataframes(
        frames,
        labels=[str(path) for path in member_paths],
        allow_duplicate_seeds=args.allow_duplicate_seeds,
    )

    total_seeds = int(merged_df["Seed"].nunique())
    summary_path, members_path = resolve_output_paths(
        args.out_prefix,
        member_paths,
        total_seeds,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    members_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_path, index=False)
    merged_df.to_csv(members_path, index=False)

    scenario = str(merged_df["Scenario"].iloc[0])
    seeds = sorted(int(seed) for seed in merged_df["Seed"].dropna().unique())
    print(f"Merged ensemble for scenario: {scenario}")
    print(f"Total unique seeds: {total_seeds}")
    print(f"Seeds: {seeds}")
    print(f"Summary saved to {summary_path}")
    print(f"Members saved to {members_path}")


if __name__ == "__main__":
    main()
