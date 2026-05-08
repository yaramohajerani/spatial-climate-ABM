"""Calibrate ABM sector parameters from a WIOD Input-Output Table.

Supports two table types from the WIOD 2016 Nov release:

  WIOT (World IO Table, recommended for generic/global calibration)
    Aggregates all 44 countries into a single global IO table.
    Files: WIOT{YEAR}_Nov16_ROW.xlsb  (one per year, in data/io/)
    Use when the model represents a generic or multi-country economy.

  NIOT (National IO Table, use for country-specific calibration)
    Single-country IO table; gives domestic production structure.
    Files: {COUNTRY}_NIOT_nov16.xlsx  (one per country, in data/io/)
    Use when modeling a specific national economy (e.g. IND, IDN, CHN).

Both formats use the same ISIC Rev.4 sector codes (A01, B, C10-C12, …)
and the same sector concordance file.

Outputs calibrated_parameters.json containing:
  sector_coefficients  — labor/input/capital shares per sector (Leontief)
  input_recipe_ranges  — inter-sector supply fractions per buyer sector
  consumption_ratios   — household final-demand shares per sector
  sector_output_shares — gross-output share per sector (for topology sizing)

Data download
-------------
WIOD 2016 release (free, no registration):
  https://www.rug.nl/ggdc/valuechain/wiod/wiod-2016-release

Files needed (place in data/io/):
  WIOT{YEAR}_Nov16_ROW.xlsb       — global WIOT for chosen year (requires pyxlsb)
  {COUNTRY}_NIOT_nov16.xlsx        — national NIOT for a specific country
  Socio_Economic_Accounts.xlsx     — labour/capital accounts (all countries)

Available NIOT countries: AUS AUT BEL BGR BRA CAN CHE CHN CYP CZE DEU DNK
  ESP EST FIN FRA GBR GRC HRV HUN IDN IND IRL ITA JPN KOR LTU LUX LVA MEX
  MLT NLD NOR POL PRT ROU RUS SVK SVN SWE TUR TWN USA

Usage
-----
  # Generic/global calibration from WIOT (recommended default):
  python prepare_parameters/calibrate_from_io.py \\
      --wiot-file data/io/WIOT2014_Nov16_ROW.xlsb \\
      --sea-file  data/io/Socio_Economic_Accounts.xlsx \\
      --year 2014 \\
      --out prepare_parameters/calibrated_parameters.json

  # Country-specific calibration from NIOT:
  python prepare_parameters/calibrate_from_io.py \\
      --niot-file data/io/IND_NIOT_nov16.xlsx \\
      --sea-file  data/io/Socio_Economic_Accounts.xlsx \\
      --year 2014 \\
      --out prepare_parameters/calibrated_parameters_IND.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SECTORS = ["commodity", "agriculture", "components", "manufacturing",
                 "retail", "wholesale", "services"]
FINAL_DEMAND_SECTORS = {"retail", "wholesale", "services"}

_AGGREGATE_CODES = frozenset({
    "II_fob", "TXSP", "EXP_adj", "PURR", "PURNR", "VA", "IntTTM", "GO",
    "TOT",
})
_FD_CODES = frozenset({"CONS_h", "CONS_np", "CONS_g", "GFCF", "INVEN", "EXP"})

COEFF_FLOOR = 0.02
RECIPE_SHARE_MIN_DEFAULT = 0.02


# ---------------------------------------------------------------------------
# Concordance helpers
# ---------------------------------------------------------------------------

def _load_concordance(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def _build_reverse_concordance(concordance: dict[str, list[str]]) -> dict[str, str]:
    rev: dict[str, str] = {}
    for model_sector, codes in concordance.items():
        for code in codes:
            if code in rev:
                raise ValueError(f"Code {code!r} appears in multiple model sectors")
            rev[code] = model_sector
    return rev


# ---------------------------------------------------------------------------
# WIOT parser (global World IO Table, .xlsb format)
# ---------------------------------------------------------------------------

def _parse_wiot(
    wiot_path: Path,
    sea_path: Path | None,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Parse a global WIOD WIOT workbook and return (Z_df, fd_df, labor, gos, X_series).

    WIOT format (WIOT{YEAR}_Nov16_ROW.xlsb, sheet named by year):
    - Rows 0-5: headers.  Row 2 = sector codes per column, row 4 = country per column.
    - Rows 6+: data.  Col 0 = sector code, col 2 = country.  Cols 4+ = Z values.
    - Industry rows: 44 countries × 56 sectors = 2464 rows.
    - Industry columns: 44 countries × 56 sectors (each identified by row 2 code).
    - Final demand columns: CONS_h, CONS_np, CONS_g, GFCF, INVEN per country.
    - Aggregate rows at bottom: II_fob, TXSP, EXP_adj, PURR, PURNR, VA, IntTTM, GO.

    For global calibration we aggregate all country repetitions: sum rows with the
    same sector code (across all source countries) and columns with the same sector
    code (across all destination countries).  This yields a single 56×56 global Z.
    """
    print(f"  Reading WIOT (global): {wiot_path.name}  year={year}")
    df = pd.read_excel(wiot_path, sheet_name=str(year), header=None, engine="pyxlsb")

    # Build lookup: column index → sector code (for industry cols) or fd code
    col_sector: dict[int, str] = {}
    col_fd: dict[int, str] = {}
    for c in range(4, df.shape[1]):
        code = df.iloc[2, c]
        if isinstance(code, str):
            if code not in _AGGREGATE_CODES and code not in _FD_CODES:
                col_sector[c] = code  # industry column
            elif code in _FD_CODES:
                col_fd[c] = code

    # All unique industry sector codes (preserving first-seen order)
    sector_codes: list[str] = list(dict.fromkeys(col_sector.values()))

    # Aggregate Z: sum rows by source sector, columns by destination sector
    Z_agg: dict[str, dict[str, float]] = {s: {t: 0.0 for t in sector_codes} for s in sector_codes}
    fd_agg: dict[str, float] = {s: 0.0 for s in sector_codes}
    va_agg: dict[str, float] = {s: 0.0 for s in sector_codes}
    go_agg: dict[str, float] = {s: 0.0 for s in sector_codes}

    sector_code_set = set(sector_codes)

    for r in range(6, df.shape[0]):
        row_code = df.iloc[r, 0]
        if not isinstance(row_code, str):
            continue

        is_industry = row_code in sector_code_set
        is_va = row_code == "VA"
        is_go = row_code == "GO"

        if not (is_industry or is_va or is_go):
            continue

        row_data = df.iloc[r].tolist()

        if is_industry:
            for c, dst in col_sector.items():
                val = row_data[c]
                if isinstance(val, (int, float)) and val == val:  # NaN check
                    Z_agg[row_code][dst] += float(val)
            for c, fd_name in col_fd.items():
                if fd_name == "CONS_h":
                    val = row_data[c]
                    if isinstance(val, (int, float)) and val == val:
                        fd_agg[row_code] += float(val)
        elif is_va:
            for c, dst in col_sector.items():
                val = row_data[c]
                if isinstance(val, (int, float)) and val == val:
                    va_agg[dst] += float(val)
        elif is_go:
            for c, dst in col_sector.items():
                val = row_data[c]
                if isinstance(val, (int, float)) and val == val:
                    go_agg[dst] += float(val)

    Z_df = pd.DataFrame(
        {dst: {src: Z_agg[src][dst] for src in sector_codes} for dst in sector_codes}
    )
    fd_df = pd.DataFrame({"CONS_h": fd_agg})
    X_series = pd.Series(go_agg) if any(go_agg.values()) else (
        pd.Series({s: sum(Z_agg[s].values()) + fd_agg[s] for s in sector_codes})
    )

    # Labour and GOS: derived from the WIOT's own VA row (already in USD).
    # The SEA file is in national currencies, making cross-country aggregation
    # incompatible with the USD-denominated WIOT.  We split VA as 60% labour /
    # 40% capital, consistent with the global factor income literature.
    # (Karabarbounis & Neiman 2014 estimate a global labour share ~55-65%.)
    if sea_path is not None:
        warnings.warn(
            "SEA file is in national currencies and cannot be aggregated across "
            "countries for WIOT calibration. Using WIOT VA row with a 60/40 "
            "labour/capital split instead. For country-specific labour shares use "
            "--niot-file with the same --sea-file."
        )
    va_series = pd.Series(va_agg)
    labor = va_series * 0.6
    gos = va_series * 0.4

    return Z_df, fd_df, labor, gos, X_series


# ---------------------------------------------------------------------------
# NIOT parser (country-specific National IO Table, .xlsx format)
# ---------------------------------------------------------------------------

def _parse_niot(
    niot_path: Path,
    sea_path: Path | None,
    country: str,
    year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Parse a WIOD NIOT workbook and return (Z_df, fd_df, labor, gos, X_series).

    NIOT format ({COUNTRY}_NIOT_nov16.xlsx, sheet 'National IO-tables'):
    - Row 0: column headers (Year, Code, Description, Origin, <sector codes>, CONS_h, ..., GO)
    - Subsequent rows: one per (Year, Code, Origin) combination.
      Origin 'TOT' = total including both domestic production and imports.
    - VA and GO are stored as special Code rows whose values in the industry
      columns give per-sector value added and gross output.
    """
    print(f"  Reading NIOT: {niot_path.name}  country={country}  year={year}")
    df = pd.read_excel(niot_path, sheet_name="National IO-tables", header=0)

    cols = list(df.columns)
    try:
        origin_idx = cols.index("Origin")
        cons_h_idx = cols.index("CONS_h")
    except ValueError as exc:
        raise ValueError(f"Expected 'Origin' and 'CONS_h' columns in NIOT sheet: {exc}") from exc
    industry_col_codes: list[str] = cols[origin_idx + 1 : cons_h_idx]

    year_df = df[df["Year"] == year].copy()
    if year_df.empty:
        available = sorted(df["Year"].dropna().unique())
        raise ValueError(
            f"Year {year} not found in {niot_path.name}. Available years: {available}"
        )

    # Industry rows (A01…U) have Origin='Domestic' and Origin='Imports', never 'TOT'.
    # Aggregate rows (VA, GO, II_fob, …) have Origin='TOT' only.
    # Sum Domestic + Imports per industry code to get total intermediate flows.
    industry_year = year_df[~year_df["Code"].isin(_AGGREGATE_CODES)]
    numeric_cols = industry_col_codes + ["CONS_h"]
    industry_rows = (
        industry_year.groupby("Code")[numeric_cols]
        .sum()
        .reindex(index=industry_col_codes)
        .fillna(0.0)
        .astype(float)
    )

    Z_df = industry_rows[industry_col_codes].copy()
    fd_series = industry_rows["CONS_h"]
    fd_df = pd.DataFrame({"CONS_h": fd_series})

    # VA and GO are stored as aggregate 'TOT' rows
    year_tot = year_df[year_df["Origin"] == "TOT"]
    go_row = year_tot[year_tot["Code"] == "GO"]
    if go_row.empty:
        warnings.warn("'GO' row not found; computing gross output from column sums.")
        X_series = Z_df.sum(axis=0) + fd_series
    else:
        X_series = (
            go_row[industry_col_codes].iloc[0]
            .reindex(industry_col_codes)
            .fillna(0.0)
            .astype(float)
        )

    # VA from the NIOT (in USD) used as reference for unit conversion and fallback
    va_row = year_tot[year_tot["Code"] == "VA"]
    va_niot = (
        va_row[industry_col_codes].iloc[0].reindex(industry_col_codes).fillna(0.0).astype(float)
        if not va_row.empty
        else (X_series - Z_df.sum(axis=0)).clip(lower=0.0)
    )

    labor = pd.Series(0.0, index=industry_col_codes)
    gos = pd.Series(0.0, index=industry_col_codes)

    if sea_path is not None and sea_path.exists():
        print(f"  Reading SEA: {sea_path.name}  country={country}  year={year}")
        try:
            sea = pd.read_excel(sea_path, sheet_name="DATA", header=0)
            country_sea = sea[sea["country"] == country]
            if country_sea.empty:
                raise ValueError(
                    f"Country '{country}' not found in SEA. "
                    f"Available: {sorted(sea['country'].dropna().unique())}"
                )
            if year not in sea.columns:
                raise ValueError(f"Year {year} not in SEA.")

            # NIOT is in millions of USD; SEA is in millions of national currency.
            # Derive a USD scale factor from the ratio of NIOT VA to SEA VA so that
            # COMP and CAP (in national currency) are converted to USD before use.
            sea_va = country_sea[country_sea["variable"] == "VA"].set_index("code")
            sea_va_series = pd.Series(
                {code: float(sea_va.at[code, year])
                 for code in industry_col_codes if code in sea_va.index},
                index=industry_col_codes,
            ).fillna(0.0)
            sea_va_total = sea_va_series.sum()
            niot_va_total = va_niot.sum()
            if sea_va_total > 0 and niot_va_total > 0:
                usd_scale = niot_va_total / sea_va_total
            else:
                usd_scale = 1.0
                warnings.warn("Cannot compute USD scale from VA; using SEA values as-is.")

            for var, target in [("COMP", labor), ("CAP", gos)]:
                rows = country_sea[country_sea["variable"] == var].set_index("code")
                for code in industry_col_codes:
                    if code in rows.index:
                        val = rows.at[code, year]
                        target[code] = float(val) * usd_scale if not pd.isna(val) else 0.0
        except Exception as exc:
            warnings.warn(
                f"Failed to read SEA file ({exc}). "
                "Estimating labor/GOS from NIOT value-added (60%/40%)."
            )
            labor = va_niot * 0.6
            gos = va_niot * 0.4
    else:
        labor = va_niot * 0.6
        gos = va_niot * 0.4
        warnings.warn(
            "No SEA file provided. Estimating labor/GOS as 60%/40% of NIOT value added. "
            "Pass --sea-file data/io/Socio_Economic_Accounts.xlsx for accurate coefficients."
        )

    return Z_df, fd_df, labor, gos, X_series


# ---------------------------------------------------------------------------
# Aggregation and coefficient computation (shared by both parsers)
# ---------------------------------------------------------------------------

def _aggregate_to_model_sectors(
    Z_df: pd.DataFrame,
    fd_df: pd.DataFrame,
    labor: pd.Series,
    gos: pd.Series,
    X_series: pd.Series,
    rev_concordance: dict[str, str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    Z_agg = pd.DataFrame(0.0, index=MODEL_SECTORS, columns=MODEL_SECTORS)
    fd_agg = pd.Series(0.0, index=MODEL_SECTORS)
    labor_agg = pd.Series(0.0, index=MODEL_SECTORS)
    gos_agg = pd.Series(0.0, index=MODEL_SECTORS)
    X_agg = pd.Series(0.0, index=MODEL_SECTORS)

    for src_code in Z_df.index:
        src_model = rev_concordance.get(src_code)
        if src_model is None:
            continue
        for dst_code in Z_df.columns:
            dst_model = rev_concordance.get(dst_code)
            if dst_model is None:
                continue
            Z_agg.at[src_model, dst_model] += float(Z_df.at[src_code, dst_code])

        fd_agg[src_model] += float(fd_df.at[src_code, "CONS_h"]) if src_code in fd_df.index else 0.0
        labor_agg[src_model] += float(labor.get(src_code, 0.0))
        gos_agg[src_model] += float(gos.get(src_code, 0.0))
        X_agg[src_model] += float(X_series.get(src_code, 0.0))

    return Z_agg, fd_agg, labor_agg, gos_agg, X_agg


def _compute_coefficients(
    Z_agg: pd.DataFrame,
    fd_agg: pd.Series,
    labor_agg: pd.Series,
    gos_agg: pd.Series,
    X_agg: pd.Series,
    min_recipe_share: float,
) -> dict:
    results: dict = {
        "sector_coefficients": {},
        "input_recipe_ranges": {},
        "consumption_ratios": {},
        "sector_output_shares": {},
    }

    A = pd.DataFrame(0.0, index=MODEL_SECTORS, columns=MODEL_SECTORS)
    for j in MODEL_SECTORS:
        if X_agg[j] > 0:
            A[j] = Z_agg[j] / X_agg[j]

    for j in MODEL_SECTORS:
        x_j = X_agg[j]
        input_coeff = float(A[j].sum()) if x_j > 0 else 0.0
        labor_coeff = float(labor_agg[j] / x_j) if x_j > 0 else 0.4
        capital_coeff = float(gos_agg[j] / x_j) if x_j > 0 else 0.2

        input_coeff = max(COEFF_FLOOR, min(0.95, input_coeff))
        labor_coeff = max(COEFF_FLOOR, min(0.95, labor_coeff))
        capital_coeff = max(COEFF_FLOOR, min(0.95, capital_coeff))

        if labor_coeff + input_coeff > 1.0:
            warnings.warn(
                f"Sector '{j}': LABOR_COEFF ({labor_coeff:.3f}) + INPUT_COEFF "
                f"({input_coeff:.3f}) > 1.0. Check concordance or SEA data."
            )

        results["sector_coefficients"][j] = {
            "labor": round(labor_coeff, 4),
            "input": round(input_coeff, 4),
            "capital": round(capital_coeff, 4),
        }

    for j in MODEL_SECTORS:
        input_total = float(A[j].sum())
        if input_total < COEFF_FLOOR:
            results["input_recipe_ranges"][j] = {}
            continue
        recipe: dict[str, list[float]] = {}
        for i in MODEL_SECTORS:
            share = float(A.at[i, j]) / input_total
            if share >= min_recipe_share:
                recipe[i] = [round(share, 4), round(share, 4)]
        total = sum(v[0] for v in recipe.values())
        if total > 0:
            recipe = {k: [round(v[0] / total, 4), round(v[0] / total, 4)] for k, v in recipe.items()}
        results["input_recipe_ranges"][j] = recipe

    fd_total = float(fd_agg.sum())
    non_final = [s for s in MODEL_SECTORS if s not in FINAL_DEMAND_SECTORS and fd_agg[s] > 0]
    if non_final and fd_total > 0:
        non_final_share = sum(fd_agg[s] for s in non_final) / fd_total
        if non_final_share > 0.05:
            warnings.warn(
                f"Sectors {non_final} account for {non_final_share:.1%} of household "
                "final demand but are not final-demand sectors in the model "
                "(retail/wholesale/services). These flows will be ignored."
            )
    final_fd = {s: float(fd_agg[s]) for s in FINAL_DEMAND_SECTORS if fd_agg[s] > 0}
    final_total = sum(final_fd.values())
    if final_total > 0:
        results["consumption_ratios"] = {
            s: round(v / final_total, 4) for s, v in final_fd.items()
        }
    else:
        results["consumption_ratios"] = {"retail": 1.0}
        warnings.warn("No household final demand in retail/wholesale/services; defaulting to {retail: 1.0}.")

    x_total = float(X_agg.sum())
    results["sector_output_shares"] = (
        {s: round(float(X_agg[s]) / x_total, 4) for s in MODEL_SECTORS}
        if x_total > 0
        else {s: round(1.0 / len(MODEL_SECTORS), 4) for s in MODEL_SECTORS}
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _extract_country_from_filename(path: Path) -> str | None:
    m = re.match(r"^([A-Z]{3})_NIOT", path.name)
    return m.group(1) if m else None


def _extract_year_from_wiot_filename(path: Path) -> int | None:
    m = re.search(r"WIOT(\d{4})", path.name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--wiot-file", type=Path,
        help="WIOT global table (.xlsb, e.g. data/io/WIOT2014_Nov16_ROW.xlsb). "
             "Recommended for generic/global calibration.",
    )
    src.add_argument(
        "--niot-file", type=Path,
        help="NIOT country table (.xlsx, e.g. data/io/IND_NIOT_nov16.xlsx). "
             "Use for country-specific calibration.",
    )
    p.add_argument(
        "--sea-file", type=Path,
        default=Path("data/io/Socio_Economic_Accounts.xlsx"),
        help="SEA workbook with COMP/CAP accounts (default: data/io/Socio_Economic_Accounts.xlsx)",
    )
    p.add_argument(
        "--country", type=str, default=None,
        help="ISO-3 country code for NIOT (auto-detected from filename if omitted)",
    )
    p.add_argument(
        "--year", type=int, default=None,
        help="Table year (auto-detected from WIOT filename; default 2014 for NIOT)",
    )
    p.add_argument(
        "--concordance", type=Path,
        default=Path(__file__).parent / "niot_concordance_default.json",
        help="Sector concordance JSON (default: niot_concordance_default.json)",
    )
    p.add_argument(
        "--min-recipe-share", type=float, default=RECIPE_SHARE_MIN_DEFAULT,
        help=f"Drop supply links below this share (default: {RECIPE_SHARE_MIN_DEFAULT})",
    )
    p.add_argument(
        "--out", type=Path, default=Path("prepare_parameters/calibrated_parameters.json"),
        help="Output path (default: prepare_parameters/calibrated_parameters.json)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not args.concordance.exists():
        sys.exit(f"Concordance file not found: {args.concordance}")
    concordance = _load_concordance(args.concordance)
    rev_concordance = _build_reverse_concordance(concordance)
    n_codes = sum(len(v) for v in concordance.values())
    print(f"Loaded concordance: {n_codes} ISIC codes → {len(concordance)} model sectors")

    sea_path = args.sea_file if args.sea_file.exists() else None
    if sea_path is None:
        warnings.warn(f"SEA file not found at {args.sea_file}; will estimate labor/GOS from VA.")

    if args.wiot_file is not None:
        if not args.wiot_file.exists():
            sys.exit(f"WIOT file not found: {args.wiot_file}")
        year = args.year or _extract_year_from_wiot_filename(args.wiot_file) or 2014
        Z_df, fd_df, labor, gos, X_series = _parse_wiot(args.wiot_file, sea_path, year)
        source = f"WIOD 2016 WIOT (global aggregate)"
        source_file = args.wiot_file.name
        country = "GLOBAL"
    else:
        if not args.niot_file.exists():
            sys.exit(f"NIOT file not found: {args.niot_file}")
        country = args.country or _extract_country_from_filename(args.niot_file)
        if country is None:
            sys.exit("Cannot detect country from filename. Pass --country <ISO3>.")
        year = args.year or 2014
        Z_df, fd_df, labor, gos, X_series = _parse_niot(
            args.niot_file, sea_path, country, year
        )
        source = f"WIOD 2016 NIOT ({country})"
        source_file = args.niot_file.name

    print("Aggregating to model sectors...")
    Z_agg, fd_agg, labor_agg, gos_agg, X_agg = _aggregate_to_model_sectors(
        Z_df, fd_df, labor, gos, X_series, rev_concordance
    )

    print("Computing calibrated parameters...")
    results = _compute_coefficients(
        Z_agg, fd_agg, labor_agg, gos_agg, X_agg, args.min_recipe_share
    )

    results["_metadata"] = {
        "source": source,
        "source_file": source_file,
        "sea_file": args.sea_file.name if sea_path else None,
        "country": country,
        "year": year,
        "concordance_file": str(args.concordance),
        "min_recipe_share": args.min_recipe_share,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=True))
    print(f"\nCalibrated parameters written to: {args.out}")

    print("\n--- sector_coefficients ---")
    for sec, c in results["sector_coefficients"].items():
        print(f"  {sec:15s}  labor={c['labor']:.3f}  input={c['input']:.3f}  capital={c['capital']:.3f}")
    print("\n--- input_recipe_ranges ---")
    for buyer, recipe in results["input_recipe_ranges"].items():
        if recipe:
            print(f"  {buyer:15s} ← {', '.join(f'{s}:{v[0]:.2f}' for s, v in recipe.items())}")
        else:
            print(f"  {buyer:15s} ← (no intermediate inputs)")
    print("\n--- consumption_ratios ---")
    for sec, share in results["consumption_ratios"].items():
        print(f"  {sec:15s}  {share:.3f}")
    print("\n--- sector_output_shares ---")
    for sec, share in results["sector_output_shares"].items():
        print(f"  {sec:15s}  {share:.3f}")


if __name__ == "__main__":
    main()
