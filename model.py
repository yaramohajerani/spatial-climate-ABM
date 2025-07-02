from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

from agents import FirmAgent, HouseholdAgent

Coords = Tuple[int, int]


class EconomyModel(Model):
    """Spatial ABM of an economy subject to climate risk."""

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        num_households: int = 100,
        num_firms: int = 20,
        shock_step: int = 5,
        hazard_type: str = "flood",
        scenario: str = "synthetic",
        hazard_file: str | None = None,
        hazard_year: int | None = None,
        seed: int | None = None,
    ) -> None:  # noqa: D401
        super().__init__(seed=seed)

        # --- Spatial environment --- #
        # If the user has not provided explicit grid dimensions, try to read
        # them from the hazard file so the ABM aligns perfectly with the
        # underlying climate raster.
        if (width is None or height is None) and hazard_file and scenario != "synthetic":
            try:
                lon_vals, lat_vals = self._unique_lon_lat_from_hazard(hazard_file, hazard_year)
                width = len(lon_vals)
                height = len(lat_vals)
                print(f"[INFO] Grid size inferred from hazard file: {width}×{height}")
            except Exception as exc:  # noqa: BLE001 – broad on purpose, we fall back gracefully
                print(f"[WARN] Could not infer grid from hazard file, defaulting to 10×10 ({exc}).")
                width = width or 10
                height = height or 10
        else:
            # Fallback when either no hazard file is supplied or the user has
            # explicitly chosen a custom grid size.
            width = width or 10
            height = height or 10

        self.grid = MultiGrid(width, height, torus=False)
        # Alias required by Mesa visualisation helpers (they expect .space)
        self.space = self.grid  # type: ignore[assignment]
        self.valid_coordinates: List[Coords] = [
            (x, y) for x in range(width) for y in range(height)
        ]

        # --- Hazard configuration --- #
        self.shock_step = shock_step  # timestep to introduce hazard
        self.hazard_type = hazard_type
        self.scenario = scenario
        self.hazard_file: str | None = hazard_file
        self.hazard_year: int | None = hazard_year
        self.hazard_map: Dict[Coords, float] = {
            coord: 0.0 for coord in self.valid_coordinates
        }

        # Tracks the provenance of the current hazard map for transparency
        # ("none", "climada:<path>", or "synthetic:<reason>")
        self.hazard_source: str = "none"

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

    def _generate_synthetic_hazard(self) -> Dict[Coords, float]:
        """Create a spatially correlated random hazard intensity field in [0,1]."""
        # Base random field
        raw = np.random.rand(len(self.valid_coordinates))
        # Add spatial smoothing by averaging neighbours (simple)
        risk_map: Dict[Coords, float] = {}
        for idx, coord in enumerate(self.valid_coordinates):
            # risk proportional to random value plus distance to centre (simulate river flood)
            x, y = coord
            centre_dist = np.hypot(
                x - (self.grid.width - 1) / 2,
                y - (self.grid.height - 1) / 2,
            )
            norm_dist = centre_dist / (np.hypot(self.grid.width / 2, self.grid.height / 2))
            risk = 0.7 * (1 - norm_dist) + 0.3 * raw[idx]
            risk_map[coord] = np.clip(risk, 0.0, 1.0)

        self.hazard_source = "synthetic:generated"
        print("[INFO] Using synthetic hazard field (no CLIMADA file specified).")

        return risk_map

    def _load_climada_hazard(self) -> Dict[Coords, float]:
        """Load a CLIMADA Hazard from ``self.hazard_file``.

        This *minimal* version avoids the previous maze of fall-backs.
        Requirements:
        • ``climada`` (core) must be installed.
        • ``self.hazard_file`` must point to a valid .hdf5 or .hdf5.gz file.

        If anything goes wrong we *raise* an exception so that the error is
        obvious instead of silently switching to a synthetic field.
        """
        from climada.hazard import Hazard
        import numpy as _np
        import gzip, shutil

        if not self.hazard_file:
            raise RuntimeError(
                "hazard_file must be provided – automatic demo fallback has been removed."
            )

        path = Path(self.hazard_file).expanduser()

        # If the user passes the compressed file, decompress it once next to
        # the original so future runs are fast.
        if path.suffix == ".gz":
            dest = path.with_suffix("")  # drop .gz
            if not dest.exists():
                print(f"[INFO] Decompressing {path.name} → {dest.name}")
                with gzip.open(path, "rb") as fin, open(dest, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            path = dest

        if not path.exists():
            raise FileNotFoundError(path)

        haz = Hazard.from_hdf5(str(path))
        self.hazard_source = f"climada:{path}"
        print(f"[INFO] Loaded CLIMADA hazard from {path}.")

        # Optional year filter
        if self.hazard_year is not None and hasattr(haz, "event_id"):
            years = _np.array([int(str(eid)[:4]) for eid in haz.event_id])
            idx = _np.where(years == self.hazard_year)[0]
            if idx.size == 0:
                raise ValueError(
                    f"No events for year {self.hazard_year} found in {path}."
                )
            haz = haz.select(idx)

        # Derive max intensity per centroid (sparse-aware)
        centroid_intensity = haz.intensity.max(axis=0)  # type: ignore[attr-defined]

        # Convert to a 1-D dense NumPy array (requires SciPy).
        from scipy.sparse import issparse  # noqa: WPS433 – intentional hard dependency

        if issparse(centroid_intensity):
            centroid_intensity = centroid_intensity.toarray().ravel()
        else:
            centroid_intensity = _np.asarray(centroid_intensity).ravel()

        # Normalise 0-1
        _min, _max = centroid_intensity.min(), centroid_intensity.max()
        print(f"Min: {_min}, Max: {_max}")
        if _max > _min:
            centroid_intensity = (centroid_intensity - _min) / (_max - _min)
        else:
            centroid_intensity = _np.zeros_like(centroid_intensity)

        # Build hazard map so that each Mesa cell matches exactly one raster
        # centroid from the hazard file.
        lon = haz.centroids.lon
        lat = haz.centroids.lat

        unique_lon = _np.unique(lon)
        unique_lat = _np.unique(lat)
        lon_to_x = {lo: i for i, lo in enumerate(unique_lon)}
        lat_to_y = {la: j for j, la in enumerate(unique_lat)}

        hazard_map: Dict[Coords, float] = {}
        for lo, la, inten in zip(lon, lat, centroid_intensity):
            x = lon_to_x[lo]
            y = lat_to_y[la]
            hazard_map[(x, y)] = float(inten)

        return hazard_map

    # ------------------------------------------------------------------ #
    #                     HELPER: GRID FROM HAZARD FILE                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unique_lon_lat_from_hazard(hazard_file: str, year: int | None = None):
        """Return sorted unique longitude and latitude values from a CLIMADA file.

        The function opens the file *once* to determine the spatial grid so that
        the ABM can adopt the exact same resolution. It purposefully avoids any
        Mesa-specific state so it can be called early in ``__init__``.
        """
        from climada.hazard import Hazard
        import numpy as _np
        import gzip, shutil

        path = Path(hazard_file).expanduser()

        # Handle on-the-fly decompression of *.hdf5.gz files so the user can
        # pass either version.
        if path.suffix == ".gz":
            dest = path.with_suffix("")
            if not dest.exists():
                with gzip.open(path, "rb") as fin, open(dest, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
            path = dest

        haz = Hazard.from_hdf5(str(path))

        if year is not None and hasattr(haz, "event_id"):
            years = _np.array([int(str(eid)[:4]) for eid in haz.event_id])
            idx = _np.where(years == year)[0]
            if idx.size:
                haz = haz.select(idx)

        lon = haz.centroids.lon
        lat = haz.centroids.lat
        return sorted(set(lon)), sorted(set(lat))

    # --------------------------------------------------------------------- #
    #                               MESA STEP                               #
    # --------------------------------------------------------------------- #
    def step(self) -> None:  # noqa: D401, N802
        """Advance model by one timestep (representing one year)."""
        self.current_step += 1

        # Introduce or update hazard map
        if self.current_step == self.shock_step:
            if self.scenario == "synthetic":
                self.hazard_map = self._generate_synthetic_hazard()
            else:
                self.hazard_map = self._load_climada_hazard()
        elif self.current_step > self.shock_step:
            # Simple decay of risk over time representing recovery (50% reduction per year)
            self.hazard_map = {k: v * 0.5 for k, v in self.hazard_map.items()}
        else:
            # Pre-shock: zero risk
            self.hazard_map = {k: 0.0 for k in self.valid_coordinates}

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