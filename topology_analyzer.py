"""Topology Analyzer

Usage:
    python topology_analyzer.py path/to/topology.json [--plot]

The script reads a supply-chain topology JSON (with "firms" and "edges" keys),
computes the trophic level of every firm, prints summary statistics, and—if the
``--plot`` flag is given—saves a PDF visualisation coloured by trophic level.

Trophic level definition
------------------------
The trophic level of a firm is the **shortest** number of production steps that
separate it from any *primary producer*.

Primary producers are identified by either of the following heuristics:
1. Firms whose sector equals one of the *raw_material_sectors* list (defaults
   to ["agriculture", "mining", "extraction"]).
2. Firms with zero in-degree in the directed supply network (no suppliers).

You can tweak the list by editing ``raw_material_sectors`` below.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple

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
        "mining": "M",
        "extraction": "E",
        "manufacturing": "F",  # factory
        "services": "S",
        "retail": "R",
        "wholesale": "W",  # legacy
    }.get(sector.lower(), sector[:1].upper())


def _layout_by_level(g: nx.DiGraph, levels: Dict[int, int]) -> Dict[int, tuple[float, float]]:
    """Deterministic, non-overlapping horizontal layout by trophic level."""
    if not levels:
        return {n: (i, 0) for i, n in enumerate(g.nodes())}

    level_groups: Dict[int, List[int]] = {}
    max_lvl = max(levels.values())
    for n in g.nodes():
        lvl = levels.get(n, max_lvl + 1)
        level_groups.setdefault(lvl, []).append(n)

    pos: Dict[int, tuple[float, float]] = {}
    vertical_spacing = 2.2  # generous vertical room
    min_hspan = 8.0

    for lvl, nodes in sorted(level_groups.items()):
        count = len(nodes)
        # even spacing across a fixed horizontal span prevents overlap
        span = max(min_hspan, count * 1.1)
        xs = np.linspace(-span / 2, span / 2, count)
        y = -float(lvl) * vertical_spacing
        for x, node in zip(xs, sorted(nodes)):
            pos[node] = (float(x), y)
    return pos


def _draw_network(ax, g: nx.DiGraph, levels: Dict[int, int]):
    # Colour map by trophic level (None => grey)
    if levels:
        min_lvl = min(levels.values())
        max_lvl = max(levels.values())
    else:
        min_lvl = 1.0
        max_lvl = 1.0

    from matplotlib import cm, colors as mcolors
    norm = mcolors.Normalize(vmin=min_lvl, vmax=max_lvl)

    colours = []
    for n in g.nodes():
        lvl = levels.get(n, None)
        if lvl is None:
            colours.append("lightgrey")
        else:
            colours.append(cm.viridis(norm(lvl)))

    pos = _layout_by_level(g, levels)

    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, arrows=True, width=0.6, arrowstyle="-|>", arrowsize=8)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color=colours,
        node_size=320,
        edgecolors="black",
        linewidths=0.5,
    )

    # Label nodes with sector initial + id; white text for dark nodes
    if g.number_of_nodes() <= 140:
        mid_lvl = (max_lvl + min_lvl) / 2
        for n in g.nodes():
            lbl = f"{_abbr(g.nodes[n].get('sector', ''))}{n}"
            x, y = pos[n]
            font_color = "white" if levels.get(n, 0) < mid_lvl else "black"
            ax.text(x, y, lbl, fontsize=6.5, ha="center", va="center", color=font_color, zorder=5)

    # Add legend/colour bar for trophic levels
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Trophic level")

    # Sector legend (using label initials)
    unique_sectors = sorted({g.nodes[n].get("sector", "") for n in g.nodes()})
    if unique_sectors:
        handles = []
        labels_sec = []
        for sec in unique_sectors:
            handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey", markersize=6))
            labels_sec.append(f"{_abbr(sec)} = {sec}")
        ax.legend(handles, labels_sec, title="Sector key", fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    ax.set_title("Supply chain (trophic layout)")
    ax.axis("off")


# ------------------------------------------------------------------ #
#                            MAP PLOTTER                             #
# ------------------------------------------------------------------ #


def plot_world_map(ax, g: nx.DiGraph):
    """Scatter firm locations on a world map coloured by sector with supply chain connections."""

    # Build DataFrame of nodes
    records = []
    node_positions = {}  # Store positions for drawing arrows
    for n, data in g.nodes(data=True):
        lon = float(data.get("lon", 0))
        lat = float(data.get("lat", 0))
        records.append({"id": n, "sector": data.get("sector", ""), "lon": lon, "lat": lat})
        node_positions[n] = (lon, lat)

    import pandas as pd

    df_nodes = pd.DataFrame.from_records(records)

    # World coastlines
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    world.plot(ax=ax, color="lightgrey", edgecolor="white", linewidth=0.3)

    # Create color mapping first so we can use it for both nodes and edges
    sectors = sorted(df_nodes["sector"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    color_map = {sec: colors[i] for i, sec in enumerate(sectors)}
    
    # Create node-to-color mapping for arrows
    node_color_map = {}
    for n, data in g.nodes(data=True):
        sector = data.get("sector", "")
        node_color_map[n] = color_map.get(sector, "gray")
    
    # Draw supply chain connections with supplier colors
    # Create a simple graph for drawing edges individually
    edge_graph = nx.DiGraph()
    edge_graph.add_nodes_from(node_positions.keys())
    
    # Draw each edge with the supplier's color
    for src, dst in g.edges():
        if src in node_positions and dst in node_positions:
            supplier_color = node_color_map.get(src, "gray")
            
            # Create temporary graph with just this edge
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
                width=1.2,
                arrowsize=20,
                connectionstyle="arc3,rad=0.1",
                alpha=0.7,
                node_size=0,
            )

    # Plot firm locations colored by sector

    for sec in sectors:
        subset = df_nodes[df_nodes["sector"] == sec]
        ax.scatter(subset["lon"], subset["lat"], color=color_map[sec], label=sec, 
                  edgecolors="black", s=60, alpha=0.9, zorder=5)  # Higher zorder to appear above arrows

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Supply chain network: firm locations and connections")
    ax.legend(title="Sector", fontsize=8, loc="lower left")
    
    # Add arrow legend
    ax.text(0.02, 0.98, "→ Supply chain flow\n(supplier → buyer)", 
           transform=ax.transAxes, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax.set_aspect("equal", adjustable="box")


def main():
    parser = argparse.ArgumentParser(description="Supply-chain topology analyzer")
    parser.add_argument("topology", type=Path, help="Path to topology JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate a PDF plot")
    args = parser.parse_args()

    g, _ = load_topology(args.topology)
    sources = find_sources(g)
    print(f"Identified {len(sources)} primary producers (sources). IDs: {sources}")

    levels = _compute_levels_weighted(g)
    print_stats(g, levels)

    if args.plot:
        outfile = args.topology.with_name(args.topology.stem + "_network.pdf")
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        _draw_network(axes[0], g, levels)
        plot_world_map(axes[1], g)
        fig.suptitle("Topology overview", fontsize=12)
        plt.tight_layout()
        plt.savefig(outfile, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved combined network+map plot to {outfile}")


if __name__ == "__main__":
    main() 
