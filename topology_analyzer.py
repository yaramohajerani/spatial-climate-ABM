"""Topology Analyzer

Usage:
    python topology_analyzer.py path/to/topology.json [--plot] [--param-file params.json] [--num-households 100]

The script reads a supply-chain topology JSON (with "firms" and "edges" keys),
computes the trophic level of every firm, prints summary statistics, and—if the
``--plot`` flag is given—saves two PDF visualisations:
  - *_trophic.pdf: Horizontal network layout colored by sector
  - *_map.pdf: Geographic map with firm locations and optional household locations

Options:
  --plot              Generate PDF plots
  --param-file        Parameter JSON file to read num_households and seed
  --num-households    Number of households to display on map (overrides param-file)
  --no-households     Skip household plotting even if param-file specifies num_households

Trophic level definition
------------------------
The trophic level (TL) of firm i is defined as:

    TL_i = 1 + Σ_j w_ij × TL_j

where w_ij = I_ij / Σ_k I_ik is the weight based on input purchases.

For this static analysis, we assume each firm buys 1 unit from each supplier,
so w_ij = 1 / (number of suppliers), giving:

    TL_i = 1 + average(TL of suppliers)

Primary producers (no suppliers) have TL = 1. These are identified by:
1. Sector in ["agriculture", "mining", "extraction"]
2. Zero in-degree in the supply network
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from trophic_utils import compute_trophic_levels

RAW_MATERIAL_SECTORS: List[str] = ["agriculture", "mining", "extraction"]


def load_topology(path: Path) -> tuple[nx.DiGraph, Dict[int, Dict[str, Any]]]:
    """Return directed supply graph and node attribute dict keyed by firm id."""
    with path.open() as f:
        data = json.load(f)

    firms = {f["id"]: f for f in data["firms"]}
    g = nx.DiGraph()
    g.add_nodes_from(firms.keys())

    # Store sector/name attributes for plotting
    for fid, attrs in firms.items():
        g.nodes[fid].update(attrs)

    for e in data["edges"]:
        g.add_edge(e["src"], e["dst"])

    return g, firms


def find_sources(g: nx.DiGraph) -> List[int]:
    """Identify primary producers using sector heuristic + zero in-degree."""
    sources: List[int] = []
    for node, attrs in g.nodes(data=True):
        sector = attrs.get("sector", "").lower()
        if sector in RAW_MATERIAL_SECTORS or g.in_degree(node) == 0:
            sources.append(node)
    return sources


def _compute_levels_weighted(g: nx.DiGraph) -> Dict[int, float]:
    """Compute trophic levels using the weighted-average definition."""

    adjacency: Dict[int, List[int]] = {n: list(g.predecessors(n)) for n in g.nodes()}

    # Edge weights if provided in JSON (e.g., "value")
    weight_map: Dict[Tuple[int, int], float] = {}
    for u, v, data in g.edges(data=True):  # u -> v (supplier -> buyer)
        w = float(data.get("value", 1.0))
        weight_map[(v, u)] = w  # note: key is (buyer, supplier)

    return compute_trophic_levels(adjacency, weight_map if weight_map else None)


def print_stats(g: nx.DiGraph, levels: Dict[int, int]):
    counts = Counter(levels.values())
    print("Trophic-level distribution:")
    for lvl in sorted(counts):
        print(f"  Level {lvl}: {counts[lvl]} firms")
    unreachable = set(g.nodes()) - levels.keys()
    if unreachable:
        print(f"  Unreachable from any source (isolated downstream?): {len(unreachable)} firms")
    print("\nDetailed listing (id, sector, level):")
    for node in sorted(g.nodes()):
        lvl = levels.get(node, None)
        sector = g.nodes[node].get("sector", "?")
        print(f"  {node:3d}  {sector:<15}  {lvl if lvl is not None else 'N/A'}")


def _abbr(sector: str) -> str:
    return {
        "agriculture": "A",
        # "mining": "M",
        "extraction": "E",
        "manufacturing": "M",
        "services": "S",
        "retail": "R",
        "wholesale": "W", 
    }.get(sector.lower(), sector[:1].upper())


def _layout_by_level_horizontal(g: nx.DiGraph, levels: Dict[int, float]) -> Dict[int, tuple[float, float]]:
    """Deterministic, non-overlapping horizontal layout by trophic level (left-to-right flow).

    X-position reflects actual trophic level of each firm.
    Y-position distributes firms vertically within trophic bands (by integer level) to avoid overlap.
    """
    if not levels:
        return {n: (i, 0) for i, n in enumerate(g.nodes())}

    # Group nodes by integer trophic level (band) for vertical distribution
    level_bands: Dict[int, List[int]] = {}
    max_lvl = max(levels.values())
    for n in g.nodes():
        lvl = levels.get(n, max_lvl + 1)
        band = int(lvl)  # Group by integer part
        level_bands.setdefault(band, []).append(n)

    pos: Dict[int, tuple[float, float]] = {}
    horizontal_scale = 3.0  # scale factor for x-axis
    node_spacing = 1.2  # vertical spacing between nodes in same band

    for band, nodes in sorted(level_bands.items()):
        count = len(nodes)
        # Sort nodes by their actual trophic level within the band
        nodes_sorted = sorted(nodes, key=lambda n: levels.get(n, 0))
        # Center the nodes vertically
        ys = np.linspace(-(count - 1) * node_spacing / 2, (count - 1) * node_spacing / 2, count)
        for y, node in zip(ys, nodes_sorted):
            # X = actual trophic level (scaled)
            x = float(levels.get(node, band)) * horizontal_scale
            pos[node] = (x, float(y))
    return pos


def _draw_network(ax, g: nx.DiGraph, levels: Dict[int, int]):
    """Draw network with horizontal left-to-right trophic layout."""
    # Color by sector instead of trophic level for better clarity
    sector_colors = {
        "commodity": "#1f77b4",      # blue
        "agriculture": "#1f77b4",    # blue
        "manufacturing": "#d62728",  # red
        "retail": "#2ca02c",         # green
        "wholesale": "#2ca02c",      # green
        "services": "#9467bd",       # purple
    }

    colours = []
    for n in g.nodes():
        sector = g.nodes[n].get("sector", "").lower()
        colours.append(sector_colors.get(sector, "lightgrey"))

    pos = _layout_by_level_horizontal(g, levels)

    # Draw edges with curved arrows for better visibility
    nx.draw_networkx_edges(
        g, pos, ax=ax,
        alpha=0.4,
        arrows=True,
        width=1.0,
        arrowstyle="-|>",
        arrowsize=12,
        edge_color="gray",
        connectionstyle="arc3,rad=0.1"
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=colours,
        node_size=400,
        edgecolors="black",
        linewidths=0.8,
    )

    # Label nodes with sector initial + id
    if g.number_of_nodes() <= 140:
        for n in g.nodes():
            lbl = f"{_abbr(g.nodes[n].get('sector', ''))}{n}"
            x, y = pos[n]
            ax.text(x, y, lbl, fontsize=7, ha="center", va="center", color="white",
                   fontweight="bold", zorder=5)

    # Sector legend
    unique_sectors = sorted({g.nodes[n].get("sector", "") for n in g.nodes()})
    if unique_sectors:
        handles = []
        labels_sec = []
        for sec in unique_sectors:
            color = sector_colors.get(sec.lower(), "lightgrey")
            handles.append(plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=color, markersize=10, markeredgecolor="black"))
            labels_sec.append(sec.capitalize())
        ax.legend(handles, labels_sec, title="Sector", fontsize=9, loc="upper right",
                 framealpha=0.9)

    # Add trophic level labels at top (by integer band)
    if levels:
        level_bands = {}
        for n in g.nodes():
            lvl = levels.get(n, 0)
            band = int(lvl)
            level_bands.setdefault(band, []).append(n)

        for band, nodes in sorted(level_bands.items()):
            # Position label at center of band's x-range
            x_positions = [pos[n][0] for n in nodes]
            x_center = (min(x_positions) + max(x_positions)) / 2
            max_y = max(pos[n][1] for n in nodes) + 1.5
            ax.text(x_center, max_y, f"Level {band}", fontsize=10, ha="center",
                   fontweight="bold", color="dimgray")

    ax.set_title("Supply Chain Network: Trophic Layout", fontsize=12, fontweight="bold")

    # NetworkX turns off axis by default - turn it back on for x-axis
    ax.set_frame_on(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=True, labelbottom=True)

    # Set x-axis ticks at integer trophic levels
    if levels:
        min_lvl = int(min(levels.values()))
        max_lvl = int(max(levels.values())) + 1
        tick_positions = [lvl * 3.0 for lvl in range(min_lvl, max_lvl + 1)]
        tick_labels = [str(lvl) for lvl in range(min_lvl, max_lvl + 1)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)

    ax.set_xlabel("Trophic Level", fontsize=11)


# ------------------------------------------------------------------ #
#                            MAP PLOTTER                             #
# ------------------------------------------------------------------ #


def plot_world_map(ax, g: nx.DiGraph, household_locations: List[Tuple[float, float]] = None):
    """Scatter firm locations on a world map coloured by sector with supply chain connections.

    Args:
        ax: Matplotlib axis
        g: NetworkX DiGraph with firm nodes
        household_locations: Optional list of (lon, lat) tuples for household positions
    """
    import pandas as pd

    # Build DataFrame of nodes
    records = []
    node_positions = {}  # Store positions for drawing arrows
    for n, data in g.nodes(data=True):
        lon = float(data.get("lon", 0))
        lat = float(data.get("lat", 0))
        records.append({"id": n, "sector": data.get("sector", ""), "lon": lon, "lat": lat})
        node_positions[n] = (lon, lat)

    df_nodes = pd.DataFrame.from_records(records)

    # World coastlines - use geodatasets if available, fall back to deprecated method
    import warnings
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except (ImportError, ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.plot(ax=ax, color="#f0f0f0", edgecolor="white", linewidth=0.3)

    # Consistent sector colors
    sector_colors = {
        "commodity": "#1f77b4",      # blue
        "agriculture": "#1f77b4",    # blue
        "manufacturing": "#d62728",  # red
        "retail": "#2ca02c",         # green
        "wholesale": "#2ca02c",      # green
        "services": "#9467bd",       # purple
    }

    # Create node-to-color mapping for arrows
    node_color_map = {}
    for n, data in g.nodes(data=True):
        sector = data.get("sector", "").lower()
        node_color_map[n] = sector_colors.get(sector, "gray")

    # Draw supply chain connections with supplier colors
    for src, dst in g.edges():
        if src in node_positions and dst in node_positions:
            supplier_color = node_color_map.get(src, "gray")
            temp_graph = nx.DiGraph()
            temp_graph.add_nodes_from([src, dst])
            temp_graph.add_edge(src, dst)

            nx.draw_networkx_edges(
                temp_graph,
                pos=node_positions,
                ax=ax,
                arrows=True,
                arrowstyle="-|>",
                edge_color=supplier_color,
                width=1.5,
                arrowsize=15,
                connectionstyle="arc3,rad=0.15",
                alpha=0.6,
                node_size=0,
            )

    # Plot household locations first (smaller, background)
    if household_locations:
        hh_lons = [loc[0] for loc in household_locations]
        hh_lats = [loc[1] for loc in household_locations]
        ax.scatter(hh_lons, hh_lats, color="#ffcc00", label="Households",
                  edgecolors="orange", s=20, alpha=0.7, zorder=3, marker="s")

    # Plot firm locations colored by sector (larger, foreground)
    sectors = sorted(df_nodes["sector"].unique())
    for sec in sectors:
        subset = df_nodes[df_nodes["sector"] == sec]
        color = sector_colors.get(sec.lower(), "gray")
        ax.scatter(subset["lon"], subset["lat"], color=color, label=sec.capitalize(),
                  edgecolors="black", s=80, alpha=0.9, zorder=5)

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title("Supply Chain Network: Geographic Distribution", fontsize=12, fontweight="bold")
    ax.legend(title="Agent Type", fontsize=9, loc="lower left", framealpha=0.9)

    # Add arrow legend
    ax.text(0.02, 0.98, "Arrows show supply chain flow\n(supplier → buyer)",
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_aspect("equal", adjustable="box")


def _generate_household_locations(
    g: nx.DiGraph, num_households: int, seed: int = 42
) -> List[Tuple[float, float]]:
    """Generate household locations clustered around firm locations.

    Households are placed with Gaussian noise around randomly selected firms.
    """
    rng = np.random.default_rng(seed)

    # Get firm locations
    firm_coords = []
    for n, data in g.nodes(data=True):
        lon = float(data.get("lon", 0))
        lat = float(data.get("lat", 0))
        firm_coords.append((lon, lat))

    if not firm_coords:
        return []

    firm_coords = np.array(firm_coords)
    household_locations = []

    # Generate households clustered around firms with some spread
    spread = 2.0  # degrees of spread around firm locations
    for _ in range(num_households):
        # Pick a random firm as center
        idx = rng.integers(len(firm_coords))
        center_lon, center_lat = firm_coords[idx]
        # Add Gaussian noise
        lon = center_lon + rng.normal(0, spread)
        lat = center_lat + rng.normal(0, spread)
        # Clamp to valid ranges
        lon = np.clip(lon, -180, 180)
        lat = np.clip(lat, -90, 90)
        household_locations.append((lon, lat))

    return household_locations


def main():
    parser = argparse.ArgumentParser(description="Supply-chain topology analyzer")
    parser.add_argument("topology", type=Path, help="Path to topology JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate PDF plots")
    parser.add_argument(
        "--param-file", type=Path, default=None,
        help="Parameter JSON file (for num_households and seed)"
    )
    parser.add_argument(
        "--num-households", type=int, default=0,
        help="Number of households to show on map (default: 0, use param-file or skip)"
    )
    parser.add_argument(
        "--no-households", action="store_true",
        help="Skip household plotting even if param-file specifies num_households"
    )
    args = parser.parse_args()

    g, _ = load_topology(args.topology)
    sources = find_sources(g)
    print(f"Identified {len(sources)} primary producers (sources). IDs: {sources}")

    levels = _compute_levels_weighted(g)
    print_stats(g, levels)

    if args.plot:
        # Determine number of households and seed
        num_households = args.num_households
        seed = 42
        if args.param_file and args.param_file.exists():
            with args.param_file.open() as f:
                params = json.load(f)
            num_households = num_households or params.get("num_households", 0)
            seed = params.get("seed", 42)

        # Generate household locations if requested (unless --no-households)
        household_locations = None
        if num_households > 0 and not args.no_households:
            household_locations = _generate_household_locations(g, num_households, seed)
            print(f"Generated {len(household_locations)} household locations")

        # --- Trophic level network plot (separate file) ---
        trophic_file = args.topology.with_name(args.topology.stem + "_trophic.pdf")
        fig_trophic, ax_trophic = plt.subplots(figsize=(14, 8))
        _draw_network(ax_trophic, g, levels)
        plt.tight_layout()
        plt.savefig(trophic_file, dpi=180, bbox_inches="tight")
        plt.close(fig_trophic)
        print(f"Saved trophic network plot to {trophic_file}")

        # --- Geographic map plot (separate file) ---
        map_file = args.topology.with_name(args.topology.stem + "_map.pdf")
        fig_map, ax_map = plt.subplots(figsize=(14, 10))
        plot_world_map(ax_map, g, household_locations)
        plt.tight_layout()
        plt.savefig(map_file, dpi=180, bbox_inches="tight")
        plt.close(fig_map)
        print(f"Saved geographic map to {map_file}")


if __name__ == "__main__":
    main() 
