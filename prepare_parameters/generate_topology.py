"""Generate a firm topology JSON calibrated from IO table output shares.

Reads the `calibrated_parameters.json` produced by `calibrate_from_io.py`
and generates a topology JSON (firms + edges) that:

  - Allocates firms per sector proportional to gross-output shares
  - Places firms geographically within a user-specified bounding box,
    using the Natural Earth country boundaries shapefile that already
    exists in `data/ne_110m_admin_0_countries/`
  - Wires supply edges using inverse-distance weighted sampling, consistent
    with the sector-to-sector links in `input_recipe_ranges`

Usage
-----
  python prepare_parameters/generate_topology.py \\
      --calibrated-params prepare_parameters/calibrated_parameters.json \\
      --total-firms 100 \\
      --bbox 80.0 5.0 110.0 28.0 \\
      --land-shapefile data/ne_110m_admin_0_countries/ \\
      --out calibrated_topology.json \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

MODEL_SECTORS = ["commodity", "agriculture", "components", "manufacturing",
                 "retail", "wholesale", "services"]

DEFAULT_LAND_SHAPEFILE = Path(__file__).parent.parent / "data" / "ne_110m_admin_0_countries"
DEFAULT_SUPPLIERS_PER_BUYER = 2
CAPITAL_BASE = 3.0


def _haversine_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in degrees (proxy; avoids trig overhead in sampling)."""
    return math.hypot(lat2 - lat1, lon2 - lon1)


def _load_land_points(shapefile_dir: Path, bbox: tuple[float, float, float, float],
                      grid_step: float = 0.5) -> list[tuple[float, float]]:
    """Return land (lon, lat) points within bbox on a grid_step degree grid.

    Uses geopandas + shapely to test points against the country polygons.
    Falls back to a pure bounding-box grid (no land mask) if geopandas unavailable.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.arange(lon_min, lon_max + grid_step, grid_step)
    lats = np.arange(lat_min, lat_max + grid_step, grid_step)
    candidate_points = [(float(lo), float(la)) for la in lats for lo in lons]

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import unary_union

        shp_files = list(shapefile_dir.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp file in {shapefile_dir}")

        world = gpd.read_file(shp_files[0])
        land_geom = unary_union(world.geometry.dropna().values)
        from shapely.geometry import box as shapely_box
        clip_box = shapely_box(lon_min, lat_min, lon_max, lat_max)
        land_in_bbox = land_geom.intersection(clip_box)

        land_points = [(lo, la) for (lo, la) in candidate_points
                       if land_in_bbox.contains(Point(lo, la))]

        if not land_points:
            # Too coarse — relax to centroid grid
            land_points = candidate_points
            warnings.warn("No land points found within bbox at current grid step; using all bbox points.")

        return land_points

    except ImportError:
        warnings.warn("geopandas not available; using full bounding-box grid without land masking.")
        return candidate_points
    except Exception as exc:
        warnings.warn(f"Land mask failed ({exc}); using full bounding-box grid.")
        return candidate_points


def _allocate_firms_per_sector(total_firms: int, output_shares: dict[str, float],
                                required_supplier_sectors: set[str]) -> dict[str, int]:
    """Proportionally allocate total_firms across sectors with floor constraints."""
    raw = {s: total_firms * output_shares.get(s, 0.0) for s in MODEL_SECTORS}
    counts = {s: max(1 if s in required_supplier_sectors else 0, int(raw[s])) for s in MODEL_SECTORS}

    # Adjust total to exactly equal total_firms
    current_total = sum(counts.values())
    diff = total_firms - current_total
    if diff != 0:
        # Distribute remainder by fractional part (largest-remainder method)
        fracs = [(s, raw[s] - int(raw[s])) for s in MODEL_SECTORS]
        fracs.sort(key=lambda x: -x[1])
        for i in range(abs(diff)):
            sec = fracs[i % len(fracs)][0]
            counts[sec] += 1 if diff > 0 else -1
            counts[sec] = max(0, counts[sec])

    return counts


def _generate_firm_positions(sector_counts: dict[str, int],
                              land_points: list[tuple[float, float]],
                              rng: np.random.Generator) -> list[dict]:
    """Assign geographic positions to firms, returning firms list."""
    firms = []
    firm_id = 1
    for sector in MODEL_SECTORS:
        n = sector_counts[sector]
        if n == 0:
            continue
        indices = rng.integers(0, len(land_points), size=n)
        for idx in indices:
            lon, lat = land_points[idx]
            # Add small jitter to avoid co-located firms
            lon_j = float(lon) + float(rng.uniform(-0.05, 0.05))
            lat_j = float(lat) + float(rng.uniform(-0.05, 0.05))
            firms.append({"id": firm_id, "sector": sector, "lon": round(lon_j, 4), "lat": round(lat_j, 4)})
            firm_id += 1
    return firms


def _assign_capital(firms: list[dict], output_shares: dict[str, float]) -> None:
    """Assign initial capital proportional to sector output share."""
    mean_share = sum(output_shares.values()) / max(len(output_shares), 1)
    for firm in firms:
        share = output_shares.get(firm["sector"], mean_share)
        capital = CAPITAL_BASE * (share / mean_share) if mean_share > 0 else CAPITAL_BASE
        firm["capital"] = round(min(10.0, max(0.5, capital)), 2)


def _generate_edges(firms: list[dict],
                    input_recipe_ranges: dict[str, dict],
                    suppliers_per_buyer: int,
                    rng: np.random.Generator) -> list[dict]:
    """Generate supply edges using distance-weighted sampling."""
    by_sector: dict[str, list[dict]] = {s: [] for s in MODEL_SECTORS}
    for f in firms:
        by_sector[f["sector"]].append(f)

    edges: list[dict] = []
    edge_set: set[tuple[int, int]] = set()

    for buyer_sector, recipe in input_recipe_ranges.items():
        if not recipe:
            continue
        buyers = by_sector.get(buyer_sector, [])
        if not buyers:
            continue

        for supplier_sector in recipe:
            suppliers = by_sector.get(supplier_sector, [])
            if not suppliers:
                warnings.warn(f"No firms in supplier sector '{supplier_sector}' for buyers in '{buyer_sector}'.")
                continue

            for buyer in buyers:
                candidate_suppliers = [supplier for supplier in suppliers if supplier["id"] != buyer["id"]]
                if not candidate_suppliers:
                    continue
                k = min(suppliers_per_buyer, len(candidate_suppliers))

                # Compute inverse-distance weights
                dists = [_haversine_deg(buyer["lon"], buyer["lat"], s["lon"], s["lat"]) + 1e-6
                         for s in candidate_suppliers]
                weights = np.array([1.0 / d for d in dists])
                weights /= weights.sum()

                # Sample k suppliers without replacement
                chosen_indices = rng.choice(len(candidate_suppliers), size=k, replace=False, p=weights)
                for idx in chosen_indices:
                    supplier = candidate_suppliers[idx]
                    key = (supplier["id"], buyer["id"])
                    if key not in edge_set:
                        edges.append({"src": supplier["id"], "dst": buyer["id"]})
                        edge_set.add(key)

    return edges


def _verify_coverage(firms: list[dict], edges: list[dict],
                     input_recipe_ranges: dict[str, dict],
                     sector_coefficients: dict[str, dict]) -> list[str]:
    """Return list of warning strings for uncovered buyer-sector requirements."""
    id_to_sector = {f["id"]: f["sector"] for f in firms}
    buyer_suppliers = _buyer_supplier_sectors(edges, id_to_sector)

    issues = []
    for f in firms:
        sector = f["sector"]
        if not _sector_requires_inputs(sector, input_recipe_ranges, sector_coefficients):
            continue
        missing = set(input_recipe_ranges.get(sector, {})) - buyer_suppliers.get(f["id"], set())
        if missing:
            issues.append(f"Firm {f['id']} (sector={sector}) missing suppliers for: {missing}")
    return issues


def _buyer_supplier_sectors(edges: list[dict], id_to_sector: dict[int, str]) -> dict[int, set[str]]:
    buyer_suppliers: dict[int, set[str]] = {}
    for edge in edges:
        buyer_suppliers.setdefault(edge["dst"], set()).add(id_to_sector.get(edge["src"], ""))
    return buyer_suppliers


def _sector_requires_inputs(
    sector: str,
    input_recipe_ranges: dict[str, dict],
    sector_coefficients: dict[str, dict],
) -> bool:
    return bool(input_recipe_ranges.get(sector)) and sector_coefficients.get(sector, {}).get("input", 0) >= 0.01


def _patch_missing_coverage(firms: list[dict], edges: list[dict],
                             input_recipe_ranges: dict[str, dict],
                             sector_coefficients: dict[str, dict]) -> None:
    """Greedily add nearest uncovered supplier for any uncovered buyer-sector pair."""
    id_to_sector = {f["id"]: f["sector"] for f in firms}
    by_sector: dict[str, list[dict]] = {s: [] for s in MODEL_SECTORS}
    for f in firms:
        by_sector[f["sector"]].append(f)

    edge_set = {(e["src"], e["dst"]) for e in edges}
    buyer_suppliers = _buyer_supplier_sectors(edges, id_to_sector)

    for f in firms:
        sector = f["sector"]
        if not _sector_requires_inputs(sector, input_recipe_ranges, sector_coefficients):
            continue
        covered = buyer_suppliers.get(f["id"], set())
        for sup_sector in set(input_recipe_ranges.get(sector, {})) - covered:
            candidates = [candidate for candidate in by_sector.get(sup_sector, []) if candidate["id"] != f["id"]]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda s: _haversine_deg(f["lon"], f["lat"], s["lon"], s["lat"]))
            key = (nearest["id"], f["id"])
            if key not in edge_set:
                edges.append({"src": nearest["id"], "dst": f["id"]})
                edge_set.add(key)
                buyer_suppliers.setdefault(f["id"], set()).add(sup_sector)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrated-params", type=Path,
                   default=Path("prepare_parameters/calibrated_parameters.json"),
                   help="Path to calibrated_parameters.json from calibrate_from_io.py")
    p.add_argument("--total-firms", type=int, default=100,
                   help="Total number of firms to generate (default: 100)")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=[80.0, 5.0, 110.0, 28.0],
                   help="Geographic bounding box (default: 80 5 110 28, covers South/SE Asia)")
    p.add_argument("--land-shapefile", type=Path, default=DEFAULT_LAND_SHAPEFILE,
                   help=f"Natural Earth land boundaries shapefile dir (default: {DEFAULT_LAND_SHAPEFILE})")
    p.add_argument("--grid-step", type=float, default=0.5,
                   help="Degree spacing for candidate land grid (default: 0.5)")
    p.add_argument("--suppliers-per-buyer", type=int, default=DEFAULT_SUPPLIERS_PER_BUYER,
                   help=f"Target supplier firms per buyer per required sector (default: {DEFAULT_SUPPLIERS_PER_BUYER})")
    p.add_argument("--out", type=Path, default=Path("calibrated_topology.json"),
                   help="Output topology JSON path (default: calibrated_topology.json)")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not args.calibrated_params.exists():
        sys.exit(f"Calibrated parameters file not found: {args.calibrated_params}")

    params = json.loads(args.calibrated_params.read_text())
    output_shares = params.get("sector_output_shares", {s: 1.0 / len(MODEL_SECTORS) for s in MODEL_SECTORS})
    input_recipe_ranges = params.get("input_recipe_ranges", {})
    sector_coefficients = params.get("sector_coefficients", {})

    # Sectors that must appear as suppliers
    required_supplier_sectors: set[str] = set()
    for recipe in input_recipe_ranges.values():
        required_supplier_sectors.update(recipe.keys())

    rng = np.random.default_rng(args.seed)
    bbox = tuple(args.bbox)

    print(f"Loading land points within bbox {bbox}...")
    land_points = _load_land_points(args.land_shapefile, bbox, args.grid_step)
    print(f"  {len(land_points)} candidate land points found.")

    if not land_points:
        sys.exit("No land points found in bounding box. Adjust --bbox or --grid-step.")

    print("Allocating firms per sector...")
    sector_counts = _allocate_firms_per_sector(args.total_firms, output_shares, required_supplier_sectors)
    for s, n in sector_counts.items():
        print(f"  {s:15s}: {n} firms  (output share: {output_shares.get(s, 0):.3f})")

    print("Placing firms geographically...")
    firms = _generate_firm_positions(sector_counts, land_points, rng)
    _assign_capital(firms, output_shares)

    print("Generating supply edges...")
    edges = _generate_edges(firms, input_recipe_ranges, args.suppliers_per_buyer, rng)
    print(f"  {len(edges)} edges generated.")

    print("Verifying coverage...")
    issues = _verify_coverage(firms, edges, input_recipe_ranges, sector_coefficients)
    if issues:
        print(f"  {len(issues)} coverage gaps — patching...")
        _patch_missing_coverage(firms, edges, input_recipe_ranges, sector_coefficients)
        remaining = _verify_coverage(firms, edges, input_recipe_ranges, sector_coefficients)
        if remaining:
            for msg in remaining[:5]:
                warnings.warn(msg)
    else:
        print("  All buyer firms covered.")

    # Build output
    output = {
        "_metadata": {
            "generated_from": str(args.calibrated_params),
            "total_firms": args.total_firms,
            "bbox": list(bbox),
            "seed": args.seed,
            "sectors": {s: sector_counts[s] for s in MODEL_SECTORS},
        },
        "firms": firms,
        "edges": edges,
    }

    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=True))
    print(f"\nTopology written to: {args.out}")
    print(f"  {len(firms)} firms, {len(edges)} edges")


if __name__ == "__main__":
    main()
