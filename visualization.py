"""Interactive Mesa visualization for the Economy–Climate ABM (Mesa ≥3.1).

Run the dashboard with:

    solara run visualization.py

This launches a local Solara web-app where you can watch the grid, GDP and
migrant count evolve step-by-step, and tweak model parameters live.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os
import numpy as np
import geopandas as gpd
import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

from mesa.visualization import SolaraViz, make_space_component, make_plot_component

from agents import FirmAgent, HouseholdAgent
from model import EconomyModel


# ------------------------------------------------------------------ #
# Helper to parse the hazard event list passed via environment        #
# ------------------------------------------------------------------ #


def _parse_hazard_events() -> List[Tuple[int, str, str]]:
    """Parse the semicolon-separated ABM_HAZARD_EVENTS string.

    Expected format:
        "<RP>:<TYPE>:<PATH>;<RP2>:<TYPE2>:<PATH2>"
    """

    env_str = os.getenv("ABM_HAZARD_EVENTS", "")
    if not env_str:
        return []

    events: List[Tuple[int, str, str]] = []
    for item in env_str.split(";"):
        if not item:
            continue
        try:
            rp_str, type_str, path_str = item.split(":", 2)
            events.append((int(rp_str), type_str, path_str))
        except ValueError:
            # Ignore malformed entries but keep going with others
            continue
    return events


def agent_portrayal(agent: Any) -> Dict[str, Any]:  # noqa: ANN401
    """Return a simple dict understood by ``make_space_component``.

    Valid keys: ``color``, ``size``, ``marker`` (matplotlib style).
    """
    if isinstance(agent, HouseholdAgent):
        return {"color": "tab:blue", "size": 25}
    if isinstance(agent, FirmAgent):
        return {"color": "tab:red", "size": 35, "marker": "s"}
    return {"color": "grey", "size": 20}


# ------------------------------------------------------------------ #
# Load country boundaries once to avoid repeated I/O & warnings        #
# ------------------------------------------------------------------ #

try:
    _WORLD = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))  # type: ignore[attr-defined]
except Exception:  # pragma: no cover – dataset missing / offline env
    _WORLD = None


# Mesa 3.x Solara visualization API
# ------------------------ Solara components ----------------------------- #

SPACE = make_space_component(agent_portrayal)
PLOT_GDP = make_plot_component("GDP")
PLOT_MIG = make_plot_component("Migrants")

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

# ---------------------- Agent map component ----------------------------- #


@solara.component
def AgentsMap(model):  # noqa: ANN001
    """Scatter plot of agents overlaid on country borders (lon/lat axes)."""

    update_counter.get()

    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()

    # Country boundaries
    if _WORLD is not None:
        _WORLD.boundary.plot(ax=ax, linewidth=0.5, color="black")

    # Gather agent coordinates
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

    # Plot agents
    if hhs_lon:
        ax.scatter(hhs_lon, hhs_lat, s=10, c="tab:blue", label="Households", alpha=0.7)
    if firms_lon:
        ax.scatter(firms_lon, firms_lat, s=20, c="tab:red", marker="s", label="Firms", alpha=0.8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Agent locations")
    ax.legend(loc="upper right", markerscale=1.2, frameon=False)

    solara.FigureMatplotlib(fig)


# ---------------------- Hazard map component ----------------------------- #


@solara.component
def HazardMap(model):  # noqa: ANN001
    """Matplotlib heat-map of per-cell hazard depth with country borders."""

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

    fig = Figure(figsize=(4, 4))
    ax = fig.subplots()

    # Render background intensity (use nearest to avoid blurring tiny cells)
    im = ax.imshow(
        grid,
        origin="lower",  # row 0 now southernmost after flip
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="Blues",
        interpolation="nearest",
        vmin=0,
        vmax=0.5,  # fixed scale across timesteps (rasters are normalised 0–1)
    )
    fig.colorbar(im, ax=ax, label="Hazard intensity (normalised)")

    # Overlay flooded cells explicitly for visibility
    flooded_mask = grid > 0
    if flooded_mask.any():
        ys, xs = np.where(flooded_mask)
        lon_pts = [float(model.lon_vals[x]) for x in xs]
        lat_pts = [float(model.lat_vals[ys_idx]) for ys_idx in ys]
        depths = grid[ys, xs]
        ax.scatter(
            lon_pts,
            lat_pts,
            s=1,
            c=depths,
            cmap="Blues",
            vmin=0,
            vmax=0.5,
            marker="s",
            alpha=0.8,
        )

    # Country boundaries
    if _WORLD is not None:
        _WORLD.boundary.plot(ax=ax, linewidth=0.5, color="black")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Current hazard map (max depth)")
    if flooded_mask.any():
        ax.legend(loc="upper right", frameon=False)

    solara.FigureMatplotlib(fig)


# --------------------------------------------------------------------- #
# Create initial model instance (defaults can be overridden via sliders)
# --------------------------------------------------------------------- #


def make_page() -> Any:  # noqa: D401, ANN401
    hazard_events = _parse_hazard_events()
    if not hazard_events:
        raise RuntimeError(
            "No hazard data were provided via ABM_HAZARD_EVENTS. Launch the dashboard "
            "using run_simulation.py with --viz and at least one --rp-file argument."
        )

    # Inject the fixed hazard list so every model created from the UI uses it.
    model_params = {
        **_BASE_PARAMS,
        "hazard_events": hazard_events,
    }

    # Start with default parameter values to create the first model instance
    init_kwargs = {k: v["value"] if isinstance(v, dict) else v for k, v in _BASE_PARAMS.items()}
    init_kwargs["hazard_events"] = hazard_events

    model = EconomyModel(**init_kwargs)

    # ``components`` expects Solara Component objects; we silence type checker here
    return SolaraViz(
        model,
        components=[AgentsMap, HazardMap, PLOT_GDP, PLOT_MIG],
        model_params=model_params,
    )  # type: ignore


# The Solara entry point. The variable name must be `page`.
page = make_page() 