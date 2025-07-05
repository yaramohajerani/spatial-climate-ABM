from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from agents import FirmAgent, HouseholdAgent
from hazard_utils import hazard_from_geotiffs

from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine import ImpactCalc

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
        # Iterable of (return_period, year, path) tuples
        hazard_events: Iterable[Tuple[int, int, str]] | None = None,
        hazard_type: str = "FL",  # CLIMADA hazard tag for flood
        seed: int | None = None,
    ) -> None:  # noqa: D401
        super().__init__(seed=seed)

        # --- Spatial environment --- #
        if hazard_events is None:
            raise ValueError("hazard_events must be provided.")

        # Load the GeoTIFF rasters and build a CLIMADA Hazard
        self.hazard, self.lon_vals, self.lat_vals = hazard_from_geotiffs(
            hazard_events,
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

        # Log of applied events (step, event_name, event_id)
        self.applied_events: List[Tuple[int, str, int]] = []

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
                "impf_": 1,  # generic impact function id column recognised by CLIMADA
            })

        exp = Exposures(pd.DataFrame.from_records(records))
        # Geometry is now set during init in CLIMADA ≥5, but for backward
        # compatibility ensure lat/lon columns are translated into geometry.
        if not hasattr(exp, "geometry") or exp.geometry.is_empty.any():  # pragma: no cover
            exp.set_lat_lon()

        # Map exposures to hazard centroids (required before impact calc)
        exp.assign_centroids(self.hazard)
        return exp

    @staticmethod
    def _build_linear_vulnerability() -> ImpactFuncSet:
        """Create a very simple linear vulnerability (0→0, 1→100% damage)."""
        impf = ImpactFunc()
        impf.id = 1
        impf.haz_type = "FL"
        impf.name = "Linear"
        impf.intensity = np.array([0.0, 1.0])
        impf.mdd = np.array([0.0, 1.0])  # mean damage fraction linear
        impf.paa = np.array([1.0, 1.0])  # fully applicable
        impf.unit = "m"

        return ImpactFuncSet([impf])

    # --------------------------------------------------------------------- #
    #                               MESA STEP                               #
    # --------------------------------------------------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Advance model by one timestep (representing one year)."""
        self.current_step += 1

        # Each year: sample hazard independently for every cell based on RP
        self._sample_pixelwise_hazard()

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

        # also save event log
        if self.applied_events:
            import csv
            ev_path = Path(out_path).with_name("applied_events.csv")
            with ev_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Step", "Event_Name", "Event_ID"])
                writer.writerows(self.applied_events)

    # ------------------------------------------------------------------ #
    #                       EVENT APPLICATION LOGIC                      #
    # ------------------------------------------------------------------ #

    def _sample_pixelwise_hazard(self) -> None:
        """For each cell, draw independent floods for each return-period raster.

        Probability cell flooded with intensity from raster *i* is 1/RP_i each
        year, independent across cells and RPs. If multiple RPs trigger on the
        same cell in the same year we keep the maximum depth.
        """
        n_cells = len(self.valid_coordinates)
        annual_intensity = np.zeros(n_cells, dtype=float)

        # Iterate over each RP raster (event row)
        for i, rp in enumerate(self.hazard.event_name):
            p_hit = self.hazard.frequency[i]  # = 1 / RP
            if p_hit <= 0:
                continue

            # Bernoulli draw per cell – using numpy for speed
            hit_mask = self.random.random(n_cells) < p_hit
            if not hit_mask.any():
                continue

            row = self.hazard.intensity[i]
            if hasattr(row, "toarray"):
                depths = row.toarray().ravel()
            else:
                depths = np.asarray(row).ravel()

            # Update with maximum depth if multiple triggers
            annual_intensity[hit_mask] = np.maximum(
                annual_intensity[hit_mask], depths[hit_mask]
            )

        # Convert intensity→damage fraction via CLIMADA impact function
        impf = self._vuln_funcs.get_func(1) if hasattr(self._vuln_funcs, "get_func") else self._vuln_funcs[0]
        mdr = impf.calc_mdr(annual_intensity)  # array same length as n_cells

        # Update hazard_map dictionary (for visualization) and apply damages
        for idx, coord in enumerate(self.valid_coordinates):
            self.hazard_map[coord] = float(annual_intensity[idx])

        for ag in self.agents:
            x, y = ag.pos
            idx = y * len(self.lon_vals) + x  # row-major index
            loss_frac = np.clip(mdr[idx], 0.0, 1.0)
            if loss_frac == 0.0:
                continue
            if isinstance(ag, FirmAgent):
                ag.capital_stock *= 1 - loss_frac
            else:
                ag.capital *= 1 - loss_frac

        flooded_cells = (annual_intensity > 0).sum()
        print(f"[INFO] Year {self.current_step}: flooded cells = {flooded_cells}/{n_cells}") 