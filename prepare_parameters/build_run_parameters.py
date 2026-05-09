"""Build a simulation run parameter file from calibration outputs (step 3 of 3).

Combines the economic parameters from `calibrate_from_io.py` (step 1) and the
topology path from `generate_topology.py` (step 2) with simulation settings to
produce a complete parameter JSON ready for `run_simulation.py`.

Usage
-----
  # Minimal — uses defaults for all simulation settings:
  python prepare_parameters/build_run_parameters.py \\
      --calibrated-params prepare_parameters/calibrated_parameters.json \\
      --topology calibrated_topology.json \\
      --out calibrated_run_parameters.json

  # Override common simulation settings:
  python prepare_parameters/build_run_parameters.py \\
      --calibrated-params prepare_parameters/calibrated_parameters.json \\
      --topology calibrated_topology.json \\
      --steps 200 --num-households 1000 --seed 7 \\
      --adaptation-strategy backup_suppliers \\
      --out calibrated_run_parameters.json

  # Include a hazard schedule (warmup + flood event):
  python prepare_parameters/build_run_parameters.py \\
      --calibrated-params prepare_parameters/calibrated_parameters.json \\
      --topology calibrated_topology.json \\
      --rp-files "10:1:80:FL:None" "10:81:200:FL:data/processed/flood.tif" \\
      --steps 200 \\
      --out calibrated_hazard_parameters.json

  # Inherit adaptation settings from an existing parameter file:
  python prepare_parameters/build_run_parameters.py \\
      --calibrated-params prepare_parameters/calibrated_parameters.json \\
      --topology calibrated_topology.json \\
      --adaptation-from aqueduct_riverine_parameters_rcp8p5.json \\
      --out calibrated_run_parameters.json

  # Inherit both hazard schedule AND adaptation from an existing parameter file:
  python prepare_parameters/build_run_parameters.py \\
      --calibrated-params prepare_parameters/calibrated_parameters.json \\
      --topology calibrated_topology.json \\
      --rp-from aqueduct_riverine_parameters_rcp8p5.json \\
      --adaptation-from aqueduct_riverine_parameters_rcp8p5.json \\
      --out calibrated_rcp8p5_parameters.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_ADAPTATION = {
    "enabled": True,
    "decision_interval": 4,
    "ewma_alpha": 0.2,
    "observation_radius": 4,
    "adaptation_sensitivity_min": 0.5,
    "adaptation_sensitivity_max": 1.5,
    "max_adaptation_increment": 0.25,
    "continuity_decay": 0.01,
    "maintenance_cost_rate": 0.005,
    "adaptation_strategy": "capital_hardening",
    "max_backup_suppliers": 5,
    "reserved_capacity_share": 0.35,
    "reserved_capacity_markup_cap": 0.10,
    "min_money_survival": 1.0,
    "replacement_frequency": 10,
}

_HOUSEHOLDS_PER_FIRM = 5  # default ratio used when --num-households is omitted
_HAZARD_KEYS = ("rp_files", "raster_hazard_events", "node_shocks", "lane_shocks", "route_shocks")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _households_from_topology(topology_path: Path) -> int:
    """Derive a default household count from the topology firm count."""
    try:
        topo = _read_json(topology_path)
        n_firms = len(topo.get("firms", []))
        return max(100, n_firms * _HOUSEHOLDS_PER_FIRM)
    except Exception:
        return 500


def _start_year_from_calibration(cal: dict) -> int:
    """Infer a sensible start year from the calibration metadata."""
    year = cal.get("_metadata", {}).get("year")
    return int(year) if year else 2014


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--calibrated-params", type=Path, required=True,
        help="calibrated_parameters.json from calibrate_from_io.py (step 1)",
    )
    p.add_argument(
        "--topology", type=Path, required=True,
        help="Topology JSON from generate_topology.py (step 2)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("calibrated_run_parameters.json"),
        help="Output parameter file path (default: calibrated_run_parameters.json)",
    )

    # --- simulation settings ---
    sim = p.add_argument_group("simulation settings")
    sim.add_argument("--steps", type=int, default=None,
                     help="Total simulation steps (default: 80 = 20 years at 4 steps/year)")
    sim.add_argument("--steps-per-year", type=int, default=None)
    sim.add_argument("--start-year", type=int, default=None,
                     help="Simulation start year (default: inferred from calibration metadata)")
    sim.add_argument("--num-households", type=int, default=None,
                     help=f"Household count (default: {_HOUSEHOLDS_PER_FIRM}× firm count)")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--grid-resolution", type=float, default=1.0)
    sim.add_argument("--firm-replacement",
                     choices=["startup_reset", "none"], default="startup_reset")
    sim.add_argument("--no-dynamic-supplier-search", action="store_true",
                     help="Disable dynamic supplier rewiring (enabled by default)")
    sim.add_argument("--household-relocation", action="store_true",
                     help="Enable household relocation on hazard (disabled by default)")

    # --- hazard ---
    haz = p.add_argument_group("hazard events (optional; --rp-files and --rp-from are mutually exclusive)")
    haz.add_argument(
        "--rp-files", nargs="+", default=None, metavar="RP_SPEC",
        help="Hazard event strings in RP:START:END:TYPE:PATH format. "
             "Use None as path for a no-hazard warmup window.",
    )
    haz.add_argument(
        "--rp-from", type=Path, default=None, metavar="PARAM_FILE",
        help="Inherit all hazard events (rp_files, raster_hazard_events, node_shocks, "
             "lane_shocks, route_shocks) from an existing parameter JSON. Also inherits "
             "steps, start_year, steps_per_year, and grid_resolution as defaults "
             "(all overridable with explicit flags).",
    )

    # --- adaptation ---
    ada = p.add_argument_group("adaptation settings")
    ada.add_argument(
        "--adaptation-from", type=Path, default=None,
        help="Inherit the full 'adaptation' block from an existing parameter JSON "
             "(overrides all other adaptation flags)",
    )
    ada.add_argument(
        "--adaptation-strategy",
        choices=["capital_hardening", "backup_suppliers", "stockpiling", "reserved_capacity"],
        default=None,
        help="Adaptation strategy (default: capital_hardening)",
    )
    ada.add_argument("--no-adaptation", action="store_true",
                     help="Disable adaptation entirely")
    ada.add_argument("--adaptation-sensitivity-min", type=float, default=None)
    ada.add_argument("--adaptation-sensitivity-max", type=float, default=None)

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # --- validate inputs ---
    if args.rp_files and args.rp_from:
        sys.exit("--rp-files and --rp-from are mutually exclusive.")
    if not args.calibrated_params.exists():
        sys.exit(f"Calibrated parameters not found: {args.calibrated_params}")
    if not args.topology.exists():
        sys.exit(f"Topology file not found: {args.topology}")

    cal = _read_json(args.calibrated_params)
    meta = cal.get("_metadata", {})

    # --- extract calibrated economic parameters ---
    for key in ("sector_coefficients", "input_recipe_ranges", "consumption_ratios"):
        if key not in cal:
            sys.exit(f"Key '{key}' missing from {args.calibrated_params}. "
                      "Re-run calibrate_from_io.py to regenerate.")

    # --- hazard events from --rp-from ---
    rp_src_defaults: dict = {}
    hazard_source: dict = {}
    if args.rp_from is not None:
        if not args.rp_from.exists():
            sys.exit(f"--rp-from file not found: {args.rp_from}")
        rp_src_defaults = _read_json(args.rp_from)
        for k in _HAZARD_KEYS:
            if k in rp_src_defaults:
                hazard_source[k] = rp_src_defaults[k]
        if not hazard_source:
            print(f"Warning: no hazard keys found in {args.rp_from}; output will have no hazard events.")
        else:
            print(f"Inherited hazard events from: {args.rp_from} "
                  f"({', '.join(f'{k}[{len(v)}]' for k, v in hazard_source.items())})")

    # --- simulation settings ---
    # When --rp-from is given, inherit steps/timing/resolution from that file as
    # defaults so the hazard schedule's timestamps and raster grid stay consistent.
    start_year = (args.start_year
                  or rp_src_defaults.get("start_year")
                  or _start_year_from_calibration(cal))
    steps_per_year = args.steps_per_year or rp_src_defaults.get("steps_per_year", 4)
    steps = args.steps or rp_src_defaults.get("steps") or (20 * steps_per_year)
    grid_resolution = args.grid_resolution
    if grid_resolution == 1.0 and "grid_resolution" in rp_src_defaults:
        grid_resolution = rp_src_defaults["grid_resolution"]
    num_households = (
        args.num_households
        or rp_src_defaults.get("num_households")
        or _households_from_topology(args.topology)
    )

    # --- adaptation block ---
    adaptation = dict(_DEFAULT_ADAPTATION)
    if args.adaptation_from is not None:
        if not args.adaptation_from.exists():
            sys.exit(f"--adaptation-from file not found: {args.adaptation_from}")
        src = _read_json(args.adaptation_from)
        if "adaptation" not in src:
            sys.exit(f"No 'adaptation' block in {args.adaptation_from}")
        # Merge onto defaults so the inherited block doesn't have to be
        # exhaustive — any keys it omits fall back to canonical values, which
        # protects both downstream printing and runtime consumers.
        adaptation.update(src["adaptation"])
        print(f"Inherited adaptation settings from: {args.adaptation_from}")
    else:
        if args.no_adaptation:
            adaptation["enabled"] = False
        if args.adaptation_strategy is not None:
            adaptation["adaptation_strategy"] = args.adaptation_strategy
        if args.adaptation_sensitivity_min is not None:
            adaptation["adaptation_sensitivity_min"] = args.adaptation_sensitivity_min
        if args.adaptation_sensitivity_max is not None:
            adaptation["adaptation_sensitivity_max"] = args.adaptation_sensitivity_max

    # --- assemble output ---
    source_desc = (
        f"{meta.get('source_file', meta.get('source', 'unknown'))}, "
        f"country={meta.get('country', '?')}, year={meta.get('year', '?')}, "
        f"concordance={Path(meta.get('concordance_file', '')).name or '?'}"
    )

    has_hazard = bool(hazard_source or args.rp_files)
    params: dict = {
        "_comment": (
            f"Auto-generated by build_run_parameters.py "
            f"({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}). "
            + ("" if has_hazard else "Add rp_files or raster_hazard_events for a hazard scenario.")
        ),
        "_calibration": source_desc,

        "topology": str(args.topology),
        "num_households": num_households,
        "grid_resolution": grid_resolution,
        "start_year": start_year,
        "steps_per_year": steps_per_year,
        "steps": steps,
        "seed": args.seed,

        "firm_replacement": args.firm_replacement,
        "dynamic_supplier_search": {"enabled": not args.no_dynamic_supplier_search},
        "household_relocation": args.household_relocation,

        "sector_coefficients": cal["sector_coefficients"],
        "input_recipe_ranges": cal["input_recipe_ranges"],
        "consumption_ratios": cal["consumption_ratios"],
        "final_consumption_sectors": cal.get("final_consumption_sectors"),
        "sector_output_shares": cal.get("sector_output_shares"),

        "adaptation": adaptation,
    }

    # Inject hazard events (inline --rp-files takes priority; --rp-from otherwise)
    if args.rp_files:
        params["rp_files"] = args.rp_files
    else:
        params.update(hazard_source)

    # --- write ---
    args.out.write_text(json.dumps(params, indent=2, ensure_ascii=True))
    print(f"Parameter file written to: {args.out}")
    print()
    print(f"  topology:           {args.topology}")
    print(f"  num_households:     {num_households}")
    print(f"  steps:              {steps} ({steps // steps_per_year} years at {steps_per_year}/year)")
    print(f"  start_year:         {start_year}")
    print(f"  grid_resolution:    {grid_resolution}")
    print(f"  seed:               {args.seed}")
    print(f"  firm_replacement:   {args.firm_replacement}")
    print(f"  adaptation:         {adaptation['adaptation_strategy']} "
          f"(enabled={adaptation['enabled']})")
    if args.rp_files:
        print(f"  hazard:             {len(args.rp_files)} rp_files event(s) (inline)")
    elif hazard_source:
        summary = ", ".join(f"{k}[{len(v)}]" for k, v in hazard_source.items())
        print(f"  hazard:             {summary} (from {args.rp_from.name})")
    else:
        print("  hazard:             none (baseline)")
    print()
    print(f"  calibration source: {source_desc}")
    print()
    print("Run with:")
    print(f"  python run_simulation.py --param-file {args.out}")


if __name__ == "__main__":
    main()
