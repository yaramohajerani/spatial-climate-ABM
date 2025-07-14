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


def plot_network(g: nx.DiGraph, levels: Dict[int, int], outfile: Path):
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

    # -------------------------------------------------------------- #
    # Layout strategy:
    #   • x-positions from spring_layout for aesthetic spacing.
    #   • y-positions fixed by (negative) trophic level so higher levels appear lower.
    #   • This avoids an overly stretched horizontal diagram when many levels exist,
    #     yet still groups firms of similar level together.
    # -------------------------------------------------------------- #

    # Initial spring layout (returns dict: node -> [x,y])
    pos = nx.spring_layout(g, seed=42)

    # Rescale and align Y coordinate by trophic level
    lvl_vals = list(levels.values()) or [0]
    max_lvl = max(lvl_vals)
    for n in pos:
        lvl = levels.get(n, max_lvl + 1)
        # Keep spring x, but stack y by level (higher level → lower y)
        pos[n][1] = -float(lvl)
        # Optional small jitter to reduce perfect overlaps when many nodes share level
        pos[n][0] += 0.2 * (np.random.random() - 0.5)

    nx.draw_networkx_edges(g, pos, alpha=0.3, arrows=True, width=0.5)
    nx.draw_networkx_nodes(g, pos, node_color=colours, node_size=120, edgecolors="black", linewidths=0.3)

    # Label nodes with sector initial + id; white text for dark nodes
    if g.number_of_nodes() <= 120:
        def _abbr(sector: str) -> str:
            return {
                "agriculture": "A",
                "mining": "M",
                "extraction": "E",
                "manufacturing": "F",  # factory
                "services": "S",
                "wholesale": "W",
            }.get(sector.lower(), sector[:1].upper())

        ax = plt.gca()
        for n in g.nodes():
            lbl = f"{_abbr(g.nodes[n].get('sector', ''))}{n}"
            x, y = pos[n]
            font_color = "white" if levels.get(n, 0) < (max_lvl + min_lvl) / 2 else "black"
            ax.text(x, y, lbl, fontsize=6, ha="center", va="center", color=font_color)

    # Add legend/colour bar for trophic levels
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    ax = plt.gca()
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

    plt.title("Supply-chain network coloured by trophic level")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    print(f"Saved network plot to {outfile}")


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
        plot_network(g, levels, outfile)


if __name__ == "__main__":
    main() 