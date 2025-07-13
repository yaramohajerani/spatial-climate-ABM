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
from typing import List, Dict, Any

import networkx as nx
import matplotlib.pyplot as plt

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


def compute_trophic_levels(g: nx.DiGraph, sources: List[int]) -> Dict[int, int]:
    """Shortest path length from any source; unreachable => None (omitted)."""
    levels: Dict[int, int] = {}
    # Multi-source BFS
    for source in sources:
        lengths = nx.single_source_shortest_path_length(g, source)
        for node, dist in lengths.items():
            levels[node] = min(dist, levels.get(node, float("inf")))
    return levels


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
    max_lvl = max(levels.values()) if levels else 1
    colours = []
    for n in g.nodes():
        lvl = levels.get(n, None)
        if lvl is None:
            colours.append("lightgrey")
        else:
            colours.append(plt.cm.viridis(lvl / max_lvl))

    pos = nx.spring_layout(g, seed=42)
    nx.draw_networkx_edges(g, pos, alpha=0.3, arrows=True, width=0.5)
    nx.draw_networkx_nodes(g, pos, node_color=colours, node_size=100)

    # Label a few nodes (or all if small)
    if g.number_of_nodes() <= 50:
        labels = {n: str(n) for n in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels, font_size=6)

    # Add legend/colour bar for trophic levels
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_lvl))
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Trophic level")

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

    levels = compute_trophic_levels(g, sources)
    print_stats(g, levels)

    if args.plot:
        outfile = args.topology.with_name(args.topology.stem + "_network.pdf")
        plot_network(g, levels, outfile)


if __name__ == "__main__":
    main() 