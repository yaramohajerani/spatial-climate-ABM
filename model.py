from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from agents import FirmAgent, HouseholdAgent
from hazard_utils import hazard_from_geotiffs

from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine import Impact

Coords = Tuple[int, int]


class EconomyModel(Model):
    """Spatial ABM of an economy subject to climate risk."""

    def __init__(
        self,
        # Grid size is derived from the hazard raster; manual override is still
        # possible for testing but not recommended.
        width: int | None = None,
        height: int | None = None,
        num_households: int = 100,
        num_firms: int = 20,
        shock_step: int = 5,
        # Mapping return period → GeoTIFF file path
        hazard_rp_files: Dict[int, str] | None = None,
        hazard_type: str = "FL",  # CLIMADA hazard tag for flood
        seed: int | None = None,
    ) -> None:  # noqa: D401
        super().__init__(seed=seed)

        # --- Spatial environment --- #
        if hazard_rp_files is None or not hazard_rp_files:
            raise ValueError("hazard_rp_files must be provided and non-empty – no synthetic fallback available.")

        # Load the GeoTIFF rasters and build a CLIMADA Hazard
        self.hazard, self.lon_vals, self.lat_vals = hazard_from_geotiffs(
            hazard_rp_files,
            haz_type=hazard_type,
        )

        # Derive grid dimensions from the raster unless explicitly overridden
        if width is None:
            width = len(self.lon_vals)
        if height is None:
            height = len(self.lat_vals)

        self.grid = MultiGrid(width, height, torus=False)
        # Alias required by Mesa visualisation helpers (they expect .space)
        self.space = self.grid  # type: ignore[assignment]
        self.valid_coordinates: List[Coords] = [
            (x, y) for y in range(height) for x in range(width)
        ]

        # --- Hazard configuration --- #
        self.shock_step = shock_step  # timestep to introduce hazard
        self.hazard_type = hazard_type

        # Migration threshold represents maximum risk tolerated by households
        self.migration_threshold = 0.5
        self.base_wage = 1.0

        # Statistics containers
        self.total_gdp_this_step: float = 0.0
        self.migrants_this_step: int = 0

        self.datacollector = DataCollector(
            model_reporters={
                "GDP": lambda m: m.total_gdp_this_step,
                "Migrants": lambda m: m.migrants_this_step,
                "Average_Risk": lambda m: np.mean(list(m.hazard_map.values())),
            }
        )

        # --- Create agents --- #
        self._init_agents(num_households, num_firms)

        # Pre-compute exposures & vulnerability for impact calculation
        self._exposures = self._build_exposures()
        self._vuln_funcs = self._build_linear_vulnerability()

        # --- Step counter --- #
        self.current_step: int = 0

    # --------------------------------------------------------------------- #
    #                             INITIALISERS                               #
    # --------------------------------------------------------------------- #
    def _init_agents(self, num_households: int, num_firms: int) -> None:
        """Place households and firms randomly on grid."""
        # Place households
        for _ in range(num_households):
            pos = self.random.choice(self.valid_coordinates)
            agent = HouseholdAgent(
                model=self,
                pos=pos,
            )
            self.grid.place_agent(agent, pos)

        # Place firms
        for _ in range(num_firms):
            pos = self.random.choice(self.valid_coordinates)
            agent = FirmAgent(
                model=self,
                pos=pos,
                sector="manufacturing",
            )
            self.grid.place_agent(agent, pos)

    # --------------------------------------------------------------------- #
    #                              UTILITIES                                #
    # --------------------------------------------------------------------- #
    def get_cell_risk(self, pos: Coords) -> float:
        """Return hazard risk (0-1) for a given cell."""
        return self.hazard_map.get(pos, 0.0)

    # ------------------------------------------------------------------ #
    #                     IMPACT / EXPOSURE HELPERS                     #
    # ------------------------------------------------------------------ #

    def _build_exposures(self) -> Exposures:
        """Construct a CLIMADA Exposures object from all agents."""
        records = []
        for ag in self.agents:
            x, y = ag.pos
            lon = float(self.lon_vals[x])
            lat = float(self.lat_vals[y])
            # Use capital stock for firms, capital for households as exposure value
            if isinstance(ag, FirmAgent):
                val = ag.capital_stock
            else:
                val = ag.capital
            records.append({
                "latitude": lat,
                "longitude": lon,
                "value": val,
                "impf": 1,  # all agents share the same simple impact function
            })

        exp = Exposures(pd.DataFrame.from_records(records))
        exp.set_geom_points()
        return exp

    @staticmethod
    def _build_linear_vulnerability() -> ImpactFuncSet:
        """Create a very simple linear vulnerability (0→0, 1→100% damage)."""
        impf = ImpactFunc()
        impf.id = 1
        impf.haz_type = "FL"
        impf.intensity = np.array([0.0, 1.0])
        impf.mdd = np.array([0.0, 1.0])  # mean damage fraction linear
        impf.paa = np.array([1.0, 1.0])  # fully applicable
        impf.save_reg_curve()

        impf_set = ImpactFuncSet()
        impf_set.append(impf)
        return impf_set

    # --------------------------------------------------------------------- #
    #                               MESA STEP                               #
    # --------------------------------------------------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Advance model by one timestep (representing one year)."""
        self.current_step += 1

        # Trigger hazard once at the specified step; afterwards cells reset to 0 risk
        if self.current_step == self.shock_step:
            self._apply_hazard_event()
        else:
            # Pre- or post-shock: zero risk
            self.hazard_map = {c: 0.0 for c in self.valid_coordinates}

        # Reset per-timestep aggregates
        self.total_gdp_this_step = 0.0
        self.migrants_this_step = 0

        # Agent actions – shuffle agents and call their step method
        # In Mesa ≥3.0, self.agents provides an AgentSet; shuffle_do is the
        # analogue of RandomActivation.
        self.agents.shuffle_do("step")

        # Collect outputs
        self.datacollector.collect(self)

    # --------------------------------------------------------------------- #
    #                            EXPORT HELPERS                              #
    # --------------------------------------------------------------------- #
    def results_to_dataframe(self) -> pd.DataFrame:
        """Return model-level DataFrame containing tracked variables."""
        return self.datacollector.get_model_vars_dataframe().copy()

    def save_results(self, out_path: str | Path = "simulation_results.csv") -> None:
        """Save collected data to CSV."""
        df = self.results_to_dataframe()
        Path(out_path).with_suffix(".csv").write_text(df.to_csv(index=False))

    # ------------------------------------------------------------------ #
    #                       EVENT APPLICATION LOGIC                      #
    # ------------------------------------------------------------------ #

    def _apply_hazard_event(self) -> None:
        """Randomly sample an event from the hazard set and apply its damages."""
        # Choose event index proportional to frequency (higher probability for low RP)
        probs = self.hazard.frequency / self.hazard.frequency.sum()
        event_idx = int(self.random.choices(range(len(probs)), weights=probs)[0])

        # Update per-cell hazard map with the chosen event's intensity
        event_intensity = self.hazard.intensity[event_idx]
        for idx, coord in enumerate(self.valid_coordinates):
            self.hazard_map[coord] = float(event_intensity[idx])

        # --- Compute impacts using CLIMADA ------------------------------------
        self._exposures.impact_funcs = self._vuln_funcs  # attach vuln
        impact = Impact()
        impact.calc(self._exposures, self.hazard, save_mat=False)

        # damages for this event per exposure (same order as self._exposures)
        damages = impact.at_event[event_idx]

        # Apply damages to agents
        for ag, dmg in zip(self.agents, damages):
            loss_frac = dmg / max(ag.capital_stock if isinstance(ag, FirmAgent) else ag.capital, 1e-9)
            loss_frac = max(0.0, min(1.0, loss_frac))
            if isinstance(ag, FirmAgent):
                ag.capital_stock *= (1 - loss_frac)
            else:
                ag.capital *= (1 - loss_frac)

        print(f"[INFO] Applied hazard event {self.hazard.event_id[event_idx]} (frequency={self.hazard.frequency[event_idx]:.3f}/yr)") 