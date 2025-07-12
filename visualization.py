"""Interactive Mesa visualization for the Economy–Climate ABM (Mesa ≥3.1).

Run the dashboard with:

    solara run visualization.py

This launches a local Solara web-app where you can watch the hazard map,
agent distribution, and key economic indicators – firm wealth, production
and consumption – evolve step-by-step. Parameters can be tweaked live.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import numpy as np
import geopandas as gpd
import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter
import networkx as nx
import json, pathlib, datetime
import matplotlib.pyplot as plt

# Capture optional start year passed via environment
_START_YEAR = None
try:
    _START_YEAR = int(os.getenv("ABM_START_YEAR", "0"))
except ValueError:
    _START_YEAR = None

from mesa.visualization import SolaraViz, make_plot_component

from agents import FirmAgent, HouseholdAgent
from model import EconomyModel


# ------------------------------------------------------------------ #
# Helper to parse the hazard event list passed via environment       #
# ------------------------------------------------------------------ #


def _parse_hazard_events() -> List[Tuple[int, int, int, str, str]]:
    """Parse the semicolon-separated ABM_HAZARD_EVENTS string.

    Expected format:
        "<RP>:<START>:<END>:<TYPE>:<PATH>;..."
    """

    env_str = os.getenv("ABM_HAZARD_EVENTS", "")
    if not env_str:
        return []

    events: List[Tuple[int, int, int, str, str]] = []
    for item in env_str.split(";"):
        if not item:
            continue
        try:
            rp_str, start_str, end_str, type_str, path_str = item.split(":", 4)
            events.append((int(rp_str), int(start_str), int(end_str), type_str, path_str))
        except ValueError:
            # Ignore malformed entries but keep going with others
            continue
    return events

# ------------------------------------------------------------------ #
# Load country boundaries once to avoid repeated I/O & warnings        #
# ------------------------------------------------------------------ #

try:
    _WORLD = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore[attr-defined]
except Exception:  # pragma: no cover – dataset missing / offline env
    _WORLD = None


# Mesa 3.x Solara visualization API
# ------------------------ Solara components ----------------------------- #

PLOT_WEALTH = make_plot_component("Firm_Wealth")
PLOT_PROD = make_plot_component("Firm_Production")
PLOT_CONS = make_plot_component("Firm_Consumption")
PLOT_PRICE = make_plot_component("Mean_Price")
PLOT_WAGE = make_plot_component("Mean_Wage")
PLOT_FIRM_CAP = make_plot_component("Firm_Capital")
PLOT_HH_WEALTH = make_plot_component("Household_Wealth")
PLOT_HH_CAP = make_plot_component("Household_Capital")
PLOT_HH_LABOR = make_plot_component("Household_Labor_Sold")
PLOT_HH_CONS = make_plot_component("Household_Consumption")

# ------------------------------------------------------------------ #
# Custom combined plot for production bottlenecks
# ------------------------------------------------------------------ #


@solara.component
def CombinedBottleneckPlot(model):  # noqa: ANN001
    """Plot labour-, capital-, and input-limited firm counts on one axis."""

    update_counter.get()

    import matplotlib.pyplot as _plt

    fig = _plt.Figure(figsize=(4, 2.5))
    ax = fig.subplots()

    df = model.results_to_dataframe()
    steps = df.index
    if _START_YEAR:
        years = _START_YEAR + steps
        x_vals = years
        x_label = "Year"
    else:
        x_vals = steps
        x_label = "Step"

    for metric, color in [
        ("Labor_Limited_Firms", "tab:blue"),
        ("Capital_Limited_Firms", "tab:red"),
        ("Input_Limited_Firms", "tab:green"),
    ]:
        if metric in df.columns:
            ax.plot(x_vals, df[metric], label=metric.replace("_", " "), color=color)

    ax.set_title("Production bottlenecks")
    ax.set_ylabel("Firm count")
    ax.set_xlabel(x_label)
    ax.legend(fontsize=6)

    fig.tight_layout()
    solara.FigureMatplotlib(fig)

# ----------------------------- Parameters ------------------------------- #

# Most parameters remain interactive sliders; the hazard list is fixed.

_BASE_PARAMS: Dict[str, Any] = {
    "num_households": {
        "type": "SliderInt",
        "label": "Households",
        "value": 100,
        "min": 10,
        "max": 300,
        "step": 10,
    },
    "num_firms": {
        "type": "SliderInt",
        "label": "Firms",
        "value": 20,
        "min": 5,
        "max": 100,
        "step": 5,
    },
    "shock_step": {
        "type": "SliderInt",
        "label": "Year of shock",
        "value": 5,
        "min": 1,
        "max": 20,
        "step": 1,
    },
}

# ----------------- Combined hazard + agents map component ---------------- #


@solara.component
def MapView(model):  # noqa: ANN001
    """Blended view: hazard field (background) + agents (foreground)."""

    # Trigger re-render on each model step
    update_counter.get()

    # Build 2-D array from model.hazard_map dict
    width, height = model.space.width, model.space.height
    grid = np.zeros((height, width), dtype=float)
    # Flip Y so row 0 corresponds to southernmost latitude (lat_min)
    for (x, y), val in model.hazard_map.items():
        grid[height - 1 - y, x] = val

    # Geographic extent (standard orientation: north ↑)
    lon_min, lon_max = float(model.lon_vals[0]), float(model.lon_vals[-1])
    lat_min, lat_max = float(model.lat_vals[0]), float(model.lat_vals[-1])

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    # ---------------- Hazard raster (imshow with dilation) ------------------ #

    # Dilate hazard cells to their neighbourhoods for clearer visibility
    dilated = grid.copy()
    hazard_idx = np.argwhere(grid > 0)
    for y_idx, x_idx in hazard_idx:
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                ny, nx = y_idx + dy, x_idx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    dilated[ny, nx] = max(dilated[ny, nx], grid[y_idx, x_idx])

    im = ax.imshow(
        dilated,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="Blues",
        interpolation="nearest",
        vmin=0,
        vmax=0.2,
    )

    # Make the colorbar shorter so it doesn't exceed the figure height
    fig.colorbar(
        im,
        ax=ax,
        label="Hazard intensity (normalised)",
        shrink=0.7,
        pad=0.01,    # Reduce space between colorbar and plot
        aspect=20,   # Make colorbar thinner
        fraction=0.1,  # Reduce fraction of original axes for colorbar
    )

    # ---------------- Overlay agent positions ------------------- #

    hhs_lon, hhs_lat = [], []
    firms_lon, firms_lat = [], []
    for ag in model.agents:
        x, y = ag.pos
        lon = float(model.lon_vals[x])
        lat = float(model.lat_vals[y])
        if isinstance(ag, HouseholdAgent):
            hhs_lon.append(lon)
            hhs_lat.append(lat)
        elif isinstance(ag, FirmAgent):
            firms_lon.append(lon)
            firms_lat.append(lat)

    if hhs_lon:
        ax.scatter(hhs_lon, hhs_lat, s=5, c="tab:green", label="Households", alpha=0.5, zorder=3)
    if firms_lon:
        ax.scatter(firms_lon, firms_lat, s=5, c="tab:red", marker="s", label="Firms", alpha=0.5, zorder=3)

    # Country boundaries
    if _WORLD is not None:
        _WORLD.boundary.plot(ax=ax, linewidth=0.5, color="black")

    ax.set_title("Hazard & agents")

    # Remove axis ticks/labels to maximise map canvas
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    # Legend outside plot area to avoid overlap, with items in a row
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        borderaxespad=0.0,
        frameon=False,
        ncol=2,  # Show legend items in a row
    )

    solara.FigureMatplotlib(fig)


# ------------------------------------------------------------------ #
# Network topology view                                              #
# ------------------------------------------------------------------ #


@solara.component
def NetworkView(model):  # noqa: ANN001
    """Visualise static supplier & labour network on a world map."""

    # Trigger on every step to update colours/sizes if desired
    update_counter.get()

    fig = Figure(figsize=(8, 5))
    ax = fig.subplots()

    # Plot country boundaries first
    if _WORLD is not None:
        _WORLD.boundary.plot(ax=ax, linewidth=0.5, color="black")

    # Build graph ----------------------------------------------------- #
    G = nx.DiGraph()

    positions = {}
    node_colors = []

    for ag in model.agents:
        x, y = ag.pos
        lon = float(model.lon_vals[x])
        lat = float(model.lat_vals[y])
        positions[ag.unique_id] = (lon, lat)

        if isinstance(ag, HouseholdAgent):
            node_colors.append("tab:green")
        else:
            node_colors.append("tab:red")

        G.add_node(ag.unique_id, agent=ag)

    # Edges: firm→firm, household→firm (labour)
    labour_edges   = []
    supplier_edges = []
    for ag in model.agents:
        if isinstance(ag, FirmAgent):
            for supplier in ag.connected_firms:
                G.add_edge(supplier.unique_id, ag.unique_id, relation="supplier")
                supplier_edges.append((supplier.unique_id, ag.unique_id))
        elif isinstance(ag, HouseholdAgent):
            for firm in ag.nearby_firms:
                G.add_edge(ag.unique_id, firm.unique_id, relation="labour")
                labour_edges.append((ag.unique_id, firm.unique_id))

    # Draw nodes and edges ------------------------------------------- #
    # Labour edges (green)
    nx.draw_networkx_edges(
        G,
        pos=positions,
        edgelist=labour_edges,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        edge_color="green",
        width=0.6,
        arrowsize=8,
        connectionstyle="arc3,rad=0.15",
        node_size=8,
    )

    # Supplier edges (red)
    nx.draw_networkx_edges(
        G,
        pos=positions,
        edgelist=supplier_edges,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        edge_color="red",
        width=0.8,
        arrowsize=8,
        connectionstyle="arc3,rad=-0.15",
        node_size=8,
    )

    # Scatter nodes manually to get legend control
    lons, lats, c = zip(*[(pos[0], pos[1], "green" if isinstance(G.nodes[n]["agent"], HouseholdAgent) else "red") for n, pos in positions.items()])
    ax.scatter(lons, lats, s=5, c=c, alpha=0.8, zorder=4)

    ax.set_title("Agent supply & labour network")

    # Remove axis ticks to declutter
    ax.set_xticks([])
    ax.set_yticks([])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    # Custom legend: labels for nodes & edge types
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Household', markerfacecolor='green', markersize=6),
        Line2D([0], [0], marker='s', color='w', label='Firm', markerfacecolor='red', markersize=6),
        Line2D([0], [0], color='green', lw=1, label='Labour flow'),
        Line2D([0], [0], color='red', lw=1, label='Goods flow'),
    ]
    ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=2)

    # ----- write out once per session -----
    if not hasattr(model, "_topology_dumped"):
        topology = {
            "nodes": [
                {"id": n, "kind": G.nodes[n]["agent"].__class__.__name__}
                for n in G.nodes()
            ],
            # edge = (source, target)
            "edges": [
                {"src": u, "dst": v, "relation": d.get("relation", "")}
                for u, v, d in G.edges(data=True)
            ],
            # adjacency for quick lookup
            "adjacency": {n: list(G.adj[n]) for n in G.nodes()},
        }

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = pathlib.Path(f"network_topology_{ts}.json")
        out_path.write_text(json.dumps(topology, indent=2))
        print(f"[NetworkView] topology written to {out_path.absolute()}")
        model._topology_dumped = True

    solara.FigureMatplotlib(fig)


# --------------------------------------------------------------------- #
# Create initial model instance (defaults can be overridden via sliders)
# --------------------------------------------------------------------- #


def make_page_with_custom_components() -> Any:  # noqa: D401, ANN401
    hazard_events = _parse_hazard_events()
    if not hazard_events:
        raise RuntimeError(
            "No hazard data were provided via ABM_HAZARD_EVENTS. Launch the dashboard "
            "using run_simulation.py with --viz and at least one --rp-file argument."
        )

    model_params = {**_BASE_PARAMS, "hazard_events": hazard_events}

    init_kwargs = {k: v["value"] if isinstance(v, dict) else v for k, v in _BASE_PARAMS.items()}
    init_kwargs["hazard_events"] = hazard_events
    seed_env = os.getenv("ABM_SEED")
    if seed_env is not None:
        init_kwargs["seed"] = int(seed_env)

    topo_env = os.getenv("ABM_TOPOLOGY_PATH")
    if topo_env:
        init_kwargs["firm_topology_path"] = topo_env

    model = EconomyModel(**init_kwargs)

    return SolaraViz(model, components=[Dashboard], model_params=model_params)  # type: ignore


# -------------------------- Combined dashboard --------------------------- #


@solara.component
def DashboardRow(model):  # noqa: ANN001
    """Map view plus stacked firm and household metric charts."""

    update_counter.get()

    with solara.Row():
        with solara.Column(style={"flex": "2", "minWidth": "900px", "minHeight": "900px", "height": "100%"}):
            MapView(model)
            NetworkView(model)
        with solara.Column(style={"flex": "1", "minWidth": "300px", "overflowY": "auto"}):
            solara.Markdown("## Firm metrics")
            PLOT_WEALTH(model)
            PLOT_FIRM_CAP(model)
            PLOT_PROD(model)
            PLOT_CONS(model)
            PLOT_PRICE(model)
            CombinedBottleneckPlot(model)
        with solara.Column(style={"flex": "1", "minWidth": "300px", "overflowY": "auto"}):
            solara.Markdown("## Household metrics")
            PLOT_HH_WEALTH(model)
            PLOT_HH_CAP(model)
            PLOT_HH_LABOR(model)
            PLOT_HH_CONS(model)
            PLOT_WAGE(model)

# ---------------- Save & Exit button ------------------------------------ #


@solara.component
def SaveExitButton(model):  # noqa: ANN001
    """Button that saves CSVs and terminates the dashboard process."""

    def _on_click():  # noqa: ANN202 – inner callback
        # 1. Persist CSVs
        model.save_results("dashboard_results")

        # 2. Build composite time-series figure --------------------------------
        df = model.results_to_dataframe()

        variables = [
            ("Firm_Wealth", "Firm wealth"),
            ("Firm_Capital", "Firm capital"),
            ("Firm_Production", "Firm production"),
            ("Firm_Consumption", "Firm consumption"),
            ("Mean_Price", "Mean price"),
            ("Household_Wealth", "Household wealth"),
            ("Household_Capital", "Household capital"),
            ("Household_Labor_Sold", "Household labour sold"),
            ("Household_Consumption", "Household consumption"),
            ("Mean_Wage", "Mean wage"),
        ]

        # One extra combined subplot for bottlenecks
        bottleneck_metrics = [
            ("Labor_Limited_Firms", "Labor limited"),
            ("Capital_Limited_Firms", "Capital limited"),
            ("Input_Limited_Firms", "Input limited"),
        ]

        rows = int(np.ceil((len(variables)+1) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(12, 2.5 * rows), sharex=True)
        axes = axes.flatten()

        for ax, (col, title) in zip(axes, variables):
            if col in df.columns:
                ax.plot(df.index, df[col])
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=8)

        # Plot bottlenecks in the last available axis
        ax_bott = axes[-1]
        for col, title in bottleneck_metrics:
            if col in df.columns:
                ax_bott.plot(df.index, df[col], label=title)
        ax_bott.set_title("Production bottlenecks", fontsize=9)
        ax_bott.tick_params(labelsize=8)
        ax_bott.legend(fontsize=7)

        fig.tight_layout()
        img_path = pathlib.Path("dashboard_timeseries.png")
        fig.savefig(img_path, dpi=150)
        plt.close(fig)

        print(f"[SaveExitButton] time-series figure saved to {img_path.absolute()}")

        # 3. Terminate app
        import os
        os._exit(0)

    solara.Button(label="Save & Exit", color="danger", on_click=_on_click)


@solara.component
def Dashboard(model):  # noqa: ANN001
    """Combined view with save button and dashboard row."""

    SaveExitButton(model)
    DashboardRow(model)


# The Solara entry point. The variable name must be `page`.
page = make_page_with_custom_components() 