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
    #   • y-positions fixed by (negative) trophic level so higher levels appear lower.
    #   • x-positions arranged to prevent overlaps within each level using systematic spacing.
    #   • This maintains trophic level grouping while ensuring no symbol overlaps.
    # -------------------------------------------------------------- #

    # Group nodes by trophic level
    lvl_vals = list(levels.values()) or [0]
    max_lvl = max(lvl_vals)
    
    # Group nodes by their trophic level
    level_groups = {}
    for n in g.nodes():
        lvl = levels.get(n, max_lvl + 1)
        if lvl not in level_groups:
            level_groups[lvl] = []
        level_groups[lvl].append(n)
    
    pos = {}
    node_radius = 0.15  # Circle radius for overlap detection
    circle_diameter = node_radius * 2  # Full circle diameter for buffer
    vertical_spacing = 2.0  # Spacing between trophic levels to prevent vertical overlap
    
    # Determine horizontal extent based on maximum nodes in any level
    max_nodes_per_level = max(len(nodes) for nodes in level_groups.values()) if level_groups else 1
    horizontal_extent = max(8.0, max_nodes_per_level * circle_diameter * 3)  # Minimum 8 units wide
    
    # Process each trophic level
    for lvl, nodes in level_groups.items():
        y_pos = -float(lvl) * vertical_spacing  # Higher levels appear lower with more spacing
        num_nodes = len(nodes)
        
        # Track positions used in this row only
        row_positions = []
        
        # For each node, find a random non-overlapping position
        for node in sorted(nodes):  # Sort for consistency
            max_attempts = 100  # Prevent infinite loops
            attempts = 0
            
            while attempts < max_attempts:
                # Random x position within the full horizontal extent
                x_pos = np.random.uniform(-horizontal_extent/2, horizontal_extent/2)
                
                # Check if this position overlaps with any existing position in this row
                overlaps = any(abs(x_pos - existing_x) < circle_diameter for existing_x in row_positions)
                
                if not overlaps:
                    row_positions.append(x_pos)
                    pos[node] = [x_pos, y_pos]
                    break
                
                attempts += 1
            
            # Fallback if we couldn't find a random position (shouldn't happen with reasonable node counts)
            if attempts >= max_attempts:
                # Use systematic spacing as fallback
                fallback_x = -horizontal_extent/2 + len(row_positions) * circle_diameter * 1.5
                row_positions.append(fallback_x)
                pos[node] = [fallback_x, y_pos]

    nx.draw_networkx_edges(g, pos, alpha=0.3, arrows=True, width=0.5)
    nx.draw_networkx_nodes(g, pos, node_color=colours, node_size=300, edgecolors="black", linewidths=0.5)

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


# ------------------------------------------------------------------ #
#                            MAP PLOTTER                             #
# ------------------------------------------------------------------ #


def plot_world_map(g: nx.DiGraph, outfile: Path):
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

    fig, ax = plt.subplots(figsize=(12, 8))  # Larger figure for better visibility
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
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')  # Higher DPI for better quality
    plt.close()
    print(f"Saved map plot with connections to {outfile}")


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

        # Additional world map plot
        outfile_map = args.topology.with_name(args.topology.stem + "_map.pdf")
        plot_world_map(g, outfile_map)


if __name__ == "__main__":
    main() 