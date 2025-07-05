from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from agents import FirmAgent, HouseholdAgent
from hazard_utils import hazard_from_geotiffs

from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet

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
        # Iterable of (return_period, hazard_type, path) tuples
        hazard_events: Iterable[Tuple[int, str, str]] | None = None,
        hazard_type: str = "FL",  # CLIMADA hazard tag for flood
        seed: int | None = None,
    ) -> None:  # noqa: D401
        super().__init__(seed=seed)

        # --- Spatial environment --- #
        if hazard_events is None:
            raise ValueError("hazard_events must be provided.")

        from collections import defaultdict
        grouped: dict[str, list[Tuple[int, str]]] = defaultdict(list)
        for rp, htype, path in hazard_events:
            grouped[htype].append((rp, path))

        self.hazards: dict[str, Tuple[Hazard, ImpactFunc]] = {}

        first_lon, first_lat = None, None
        for htype, grp in grouped.items():
            haz, lon_vals, lat_vals = hazard_from_geotiffs(grp, haz_type=htype)
            vul = self._build_vulnerability(htype)
            self.hazards[htype] = (haz, vul)

            if first_lon is None:
                first_lon, first_lat = lon_vals, lat_vals

        self.lon_vals = first_lon  # type: ignore[assignment]
        self.lat_vals = first_lat

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

        # Initialize per-cell hazard depth map with zero intensity values
        self.hazard_map: Dict[Coords, float] = {
            coord: 0.0 for coord in self.valid_coordinates
        }

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

        # ---------------- Land mask to avoid placing agents in the ocean ---------------- #
        self.land_coordinates: List[Coords] = self._compute_land_coordinates()
        if not self.land_coordinates:
            raise ValueError(
                "No land cells found within the hazard raster extent – cannot initialise agents."
            )

        # --- Create agents --- #
        self._init_agents(num_households, num_firms)

        # Build exposures and assign centroids for each hazard
        self._exposures = self._build_exposures()
        for haz, _ in self.hazards.values():
            self._exposures.assign_centroids(haz)

        # --- Step counter --- #
        self.current_step: int = 0

        # Log of applied events (step, event_name, event_id)
        self.applied_events: List[Tuple[int, str, int]] = []

    # --------------------------------------------------------------------- #
    #                             INITIALISERS                               #
    # --------------------------------------------------------------------- #
    def _compute_land_coordinates(self) -> List[Coords]:
        """Return list of grid coordinates whose lon/lat fall within any country polygon.

        We use Natural Earth low-resolution country boundaries available via
        ``geopandas``. The geometry union is evaluated once at model
        initialisation; the resulting list is reused whenever we need to
        randomly sample land cells (e.g. for placing agents).
        """

        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            # Exclude Antarctica so agents are not spawned on that continent
            if "name" in world.columns:
                world = world[world["name"] != "Antarctica"]
            elif "continent" in world.columns:
                world = world[world["continent"] != "Antarctica"]
        except Exception:  # pragma: no cover – dataset missing / offline env
            # If the Natural Earth dataset is unavailable fall back to using all
            # cells to avoid crashing, but log a warning so the user is aware.
            import warnings

            warnings.warn(
                "Natural Earth dataset could not be loaded – agents may be placed over water.",
                RuntimeWarning,
            )
            return self.valid_coordinates.copy()

        land_geom = world.geometry.unary_union  # shapely (multi-)polygon

        land_coords: List[Coords] = []
        for x, y in self.valid_coordinates:
            lon = float(self.lon_vals[x])
            lat = float(self.lat_vals[y])
            if land_geom.contains(Point(lon, lat)):
                land_coords.append((x, y))

        return land_coords

    def _init_agents(self, num_households: int, num_firms: int) -> None:
        """Place households and firms randomly on grid."""
        # Place households only on land cells
        for _ in range(num_households):
            pos = self.random.choice(self.land_coordinates)
            agent = HouseholdAgent(model=self, pos=pos)
            self.grid.place_agent(agent, pos)

        # Place firms only on land cells (could use different logic if needed)
        for _ in range(num_firms):
            pos = self.random.choice(self.land_coordinates)
            agent = FirmAgent(model=self, pos=pos, sector="manufacturing")
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

        return exp

    @staticmethod
    def _build_vulnerability(haz_type: str = "FL") -> ImpactFunc:
        """Return a single ImpactFunc appropriate for the hazard type.

        The caller can still wrap the result in an ``ImpactFuncSet`` if they
        need CLIMADA's higher-level interfaces, but for the pixel-wise damage
        calculation performed inside this model we only ever use one curve.

        Supported hazard types:
        • ``FL`` (flood): JRC global depth–damage curve via ``climada_petals``.
          If the Petals dependency is unavailable we fall back to a simple
          linear 0-1 relationship between intensity and mean damage ratio.
        """
        try:
            if haz_type == "FL":
                from climada_petals.entity.impact_funcs.river_flood import ImpfRiverFlood  # type: ignore

                impf = ImpfRiverFlood.from_jrc_region_sector("Global", "residential")
                impf.id = 1
                return impf
        except Exception:  # noqa: BLE001
            pass  # fall back to linear

        impf = ImpactFunc(haz_type=haz_type, id=1, name="Linear")
        impf.intensity = np.array([0.0, 1.0])
        impf.mdd = np.array([0.0, 1.0])
        impf.paa = np.array([1.0, 1.0])
        impf.unit = "m"
        return impf

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
        max_depth = np.zeros(n_cells, dtype=float)
        combined_loss = np.zeros(n_cells, dtype=float)  # 0=no loss

        for htype, (haz, impf) in self.hazards.items():
            # Build per-hazard intensity map this year
            intens = np.zeros(n_cells, dtype=float)
            for i in range(haz.intensity.shape[0]):
                p_hit = haz.frequency[i]
                if p_hit <= 0:
                    continue
                hit_mask = np.random.random(n_cells) < p_hit
                if not hit_mask.any():
                    continue
                row = haz.intensity[i]
                depths = row.toarray().ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
                intens[hit_mask] = np.maximum(intens[hit_mask], depths[hit_mask])

            # Update global max depth for viz
            max_depth = np.maximum(max_depth, intens)

            # Compute loss fraction via vulnerability curve
            mdr = np.clip(impf.calc_mdr(intens), 0.0, 1.0)

            # Combine multiplicatively: 1 - prod(1 - loss)
            combined_loss = 1 - (1 - combined_loss) * (1 - mdr)

        # Write hazard_map for visualisation (max depth across hazards)
        for idx, coord in enumerate(self.valid_coordinates):
            self.hazard_map[coord] = float(max_depth[idx])

        # Apply combined loss to agents
        for ag in self.agents:
            x, y = ag.pos
            idx = y * len(self.lon_vals) + x
            loss_frac = combined_loss[idx]
            if loss_frac == 0:
                continue
            if isinstance(ag, FirmAgent):
                ag.capital_stock *= 1 - loss_frac
            else:
                ag.capital *= 1 - loss_frac

        flooded_cells = (max_depth > 0).sum()
        print(f"[INFO] Year {self.current_step}: flooded cells = {flooded_cells}/{n_cells}") 