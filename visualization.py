"""Interactive Mesa visualization for the Economy–Climate ABM (Mesa ≥3.1).

Run the dashboard with:

    solara run visualization.py

This launches a local Solara web-app where you can watch the grid, GDP and
migrant count evolve step-by-step, and tweak model parameters live.
"""
from __future__ import annotations

from typing import Any, Dict
import os

from mesa.visualization import SolaraViz, make_space_component, make_plot_component

from agents import FirmAgent, HouseholdAgent
from model import EconomyModel


def agent_portrayal(agent: Any) -> Dict[str, Any]:  # noqa: ANN401
    """Return a simple dict understood by ``make_space_component``.

    Valid keys: ``color``, ``size``, ``marker`` (matplotlib style).
    """
    if isinstance(agent, HouseholdAgent):
        return {"color": "tab:blue", "size": 25}
    if isinstance(agent, FirmAgent):
        return {"color": "tab:red", "size": 35, "marker": "s"}
    return {"color": "grey", "size": 20}


# Mesa 3.x Solara visualization API
from mesa.visualization import SolaraViz, make_space_component, make_plot_component

# ------------------------ Solara components ----------------------------- #

SPACE = make_space_component(agent_portrayal)
PLOT_GDP = make_plot_component("GDP")
PLOT_MIG = make_plot_component("Migrants")

# ----------------------------- Parameters ------------------------------- #

model_params = {
    "width": 10,
    "height": 10,
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

# --------------------------------------------------------------------- #
# Create initial model instance (defaults can be overridden via sliders)
# --------------------------------------------------------------------- #

def make_page(hazard_file, hazard_year):
    model = EconomyModel(hazard_file=hazard_file, hazard_year=hazard_year, scenario="demo")
    return SolaraViz(model, components=[SPACE, PLOT_GDP, PLOT_MIG], model_params=model_params)

page = make_page(os.getenv("ABM_HAZARD_FILE"), os.getenv("ABM_HAZARD_YEAR")) 