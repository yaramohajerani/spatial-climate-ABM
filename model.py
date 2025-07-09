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
        firm_topology_path: str | None = None,
        apply_hazard_impacts: bool = True,
    ) -> None:  # noqa: D401
        super().__init__(seed=seed)

        # Ensure NumPy uses the same seed so hazard sampling is reproducible
        if seed is not None:
            np.random.seed(seed)

        # Flag to toggle whether sampled hazards actually affect agents.
        # If False, hazards are still sampled to preserve random draws but
        # impacts (capital loss, damage, relocation triggers) are disabled.
        self.apply_hazard_impacts: bool = apply_hazard_impacts

        # --- Spatial environment & custom topology --- #
        self._firm_topology: dict | None = None
        if firm_topology_path is not None:
            import json, pathlib
            topo_path = Path(firm_topology_path)
            if not topo_path.exists():
                raise FileNotFoundError(f"Firm topology JSON not found: {topo_path}")
            self._firm_topology = json.loads(topo_path.read_text())

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

        # Base wage used by firms when hiring labour
        self.base_wage = 1.0

        # Compatibility stub – we no longer model migration but keep the attribute
        self.migrants_this_step: int = 0

        # DataCollector – track aggregate production each step for inspection
        self.datacollector = DataCollector(
            model_reporters={
                "Firm_Production": lambda m: sum(
                    ag.production for ag in m.agents if isinstance(ag, FirmAgent)
                ),
                "Firm_Consumption": lambda m: sum(
                    ag.consumption for ag in m.agents if isinstance(ag, FirmAgent)
                ),
                "Firm_Wealth": lambda m: sum(
                    ag.money for ag in m.agents if isinstance(ag, FirmAgent)
                ),
                "Firm_Capital": lambda m: sum(
                    ag.capital_stock for ag in m.agents if isinstance(ag, FirmAgent)
                ),
                "Household_Wealth": lambda m: sum(
                    ag.money for ag in m.agents if isinstance(ag, HouseholdAgent)
                ),
                "Household_Capital": lambda m: sum(
                    ag.capital for ag in m.agents if isinstance(ag, HouseholdAgent)
                ),
                "Household_LaborSold": lambda m: sum(
                    ag.labor_sold for ag in m.agents if isinstance(ag, HouseholdAgent)
                ),
                "Household_Consumption": lambda m: sum(
                    ag.consumption for ag in m.agents if isinstance(ag, HouseholdAgent)
                ),
                "Average_Risk": lambda m: np.mean(list(m.hazard_map.values())),
                "Base_Wage": lambda m: m.base_wage,
                "Mean_Price": lambda m: np.mean([ag.price for ag in m.agents if isinstance(ag, FirmAgent)]),
            },
            agent_reporters={
                "money": lambda a: getattr(a, "money", np.nan),
                "production": lambda a: getattr(a, "production", 0.0),
                "consumption": lambda a: getattr(a, "consumption", 0.0),
                "labor_sold": lambda a: getattr(a, "labor_sold", 0.0),
                "capital": lambda a: getattr(a, "capital_stock", getattr(a, "capital", 0.0)),
                "type": lambda a: type(a).__name__,
            },
        )

        # ---------------- Land mask to avoid placing agents in the ocean ---------------- #
        self.land_coordinates: List[Coords] = self._compute_land_coordinates()
        if not self.land_coordinates:
            raise ValueError(
                "No land cells found within the hazard raster extent – cannot initialise agents."
            )

        # --- Create agents & build trade network --- #
        self._init_agents(num_households, num_firms)
        # If topology provided, connections are encoded there; otherwise random
        if self._firm_topology is None:
            self._build_trade_network()

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
        # ---------------- Households ---------------- #
        for _ in range(num_households):
            pos = self.random.choice(self.land_coordinates)
            hh = HouseholdAgent(model=self, pos=pos)
            self.grid.place_agent(hh, pos)

        # ---------------- Firms --------------------- #
        if self._firm_topology is not None:
            id_to_agent: dict[int, FirmAgent] = {}
            for firm in self._firm_topology.get("firms", []):
                # Determine position: either explicit grid x,y or lon/lat
                if "x" in firm and "y" in firm:
                    x, y = int(firm["x"]), int(firm["y"])
                elif "lon" in firm and "lat" in firm:
                    # Find nearest grid cell
                    lon, lat = float(firm["lon"]), float(firm["lat"])
                    x = int(np.argmin(np.abs(self.lon_vals - lon)))  # type: ignore[arg-type]
                    y = int(np.argmin(np.abs(self.lat_vals - lat)))  # type: ignore[arg-type]
                else:
                    raise ValueError("Firm entry must contain either (x,y) or (lon,lat)")

                if (x, y) not in self.valid_coordinates:
                    raise ValueError(f"Firm {firm['id']} location out of grid bounds: {(x, y)}")

                ag = FirmAgent(
                    model=self,
                    pos=(x, y),
                    sector=firm.get("sector", "manufacturing"),
                )
                ag.capital_stock = float(firm.get("capital", 1.0))
                self.grid.place_agent(ag, (x, y))
                id_to_agent[firm["id"]] = ag

            # Build directed edges (supplier -> buyer)
            for edge in self._firm_topology.get("edges", []):
                src_id, dst_id = edge["src"], edge["dst"]
                try:
                    supplier = id_to_agent[src_id]
                    buyer = id_to_agent[dst_id]
                except KeyError as exc:
                    raise ValueError(f"Edge references unknown firm id: {exc}") from exc
                buyer.connected_firms.append(supplier)

        else:
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

        # Reset placeholder counters
        self.migrants_this_step = 0

        # Agent actions – shuffle agents and call their step method
        # In Mesa ≥3.0, self.agents provides an AgentSet; shuffle_do is the
        # analogue of RandomActivation.
        self.agents.shuffle_do("step")

        # ---------------- Dynamic wage adjustment --------------------- #
        households = [ag for ag in self.agents if isinstance(ag, HouseholdAgent)]
        total_supply = len(households) * 1.0  # each supplies one unit
        total_demand = sum(hh.labor_sold for hh in households)

        if total_supply > 0:
            ratio = total_demand / total_supply
            if ratio > 0.9:
                self.base_wage *= 1.05  # tight labour market
            elif ratio < 0.5:
                self.base_wage *= 0.95  # slack labour market

            self.base_wage = float(min(10.0, max(0.1, self.base_wage)))

        # Collect outputs after wage update so next step sees new wage
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

        # Save per-agent time series -------------------------------------- #
        agents_df = self.datacollector.get_agent_vars_dataframe().reset_index()
        # Rename columns for clarity; DataCollector returns 'AgentID' index.
        agents_df.rename(columns={"level_0": "Step", "level_1": "AgentID"}, inplace=True, errors="ignore")

        agents_path = Path(out_path).with_name("simulation_agents.csv")
        agents_path.write_text(agents_df.to_csv(index=False))

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

        # If impacts disabled, zero‐out losses and intensities so agents see
        # a risk‐free environment while preserving RNG sequence.
        if not self.apply_hazard_impacts:
            combined_loss[:] = 0  # noqa: E203 – NumPy slicing
            max_depth[:] = 0

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
                # Reduce productive capacity this year
                ag.damage_factor *= 1 - loss_frac
                # Damage inventories
                ag.inventory_output = int(ag.inventory_output * (1 - loss_frac))
                for k in list(ag.inventory_inputs.keys()):
                    ag.inventory_inputs[k] = int(ag.inventory_inputs[k] * (1 - loss_frac))
            else:
                ag.capital *= 1 - loss_frac

        flooded_cells = (max_depth > 0).sum()
        print(f"[INFO] Year {self.current_step}: flooded cells = {flooded_cells}/{n_cells}")

    # ------------------------------------------------------------------ #
    #                        TRADE NETWORK BUILDER                       #
    # ------------------------------------------------------------------ #
    def _build_trade_network(self) -> None:
        """Randomly connect firms <-> firms and households <-> firms based on distance."""

        # Separate lists for convenience
        firm_agents = [ag for ag in self.agents if isinstance(ag, FirmAgent)]
        household_agents = [ag for ag in self.agents if isinstance(ag, HouseholdAgent)]

        # 1. Firm – firm trade network (directed edges)
        dist_scale = 5.0  # characteristic decay distance
        for f1 in firm_agents:
            for f2 in firm_agents:
                if f1 is f2:
                    continue
                dx = f1.pos[0] - f2.pos[0]
                dy = f1.pos[1] - f2.pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                prob = np.exp(-dist / dist_scale)
                if self.random.random() < prob:
                    f1.connected_firms.append(f2)

        # Ensure **at least one** root firm with no upstream suppliers so the
        # production chain can bootstrap from labour alone. Keep roughly 20 % of
        # firms as roots; all other firms must have ≥1 supplier (choose nearest
        # if none were assigned by the distance‐probability rule).

        if len(firm_agents) > 0:
            # Determine current roots
            roots = [f for f in firm_agents if not f.connected_firms]

            target_root_count = max(1, int(0.2 * len(firm_agents)))

            # If no roots, pick one at random
            if not roots:
                root = self.random.choice(firm_agents)
                root.connected_firms.clear()
                roots.append(root)

            # If too many roots, pick extras to connect to nearest supplier
            if len(roots) > target_root_count:
                surplus = roots[target_root_count:]
                for f in surplus:
                    # Connect to nearest other firm (excluding itself)
                    nearest = min(
                        (g for g in firm_agents if g is not f),
                        key=lambda g: (g.pos[0] - f.pos[0]) ** 2 + (g.pos[1] - f.pos[1]) ** 2,
                    )
                    f.connected_firms.append(nearest)

            # Now ensure every non-root firm has at least one supplier
            for f in firm_agents:
                if f.connected_firms:
                    continue
                if f in roots:
                    continue  # keep as root
                nearest = min(
                    (g for g in firm_agents if g is not f),
                    key=lambda g: (g.pos[0] - f.pos[0]) ** 2 + (g.pos[1] - f.pos[1]) ** 2,
                )
                f.connected_firms.append(nearest)

        # 2. Household – firm employment/consumption links (within radius)
        work_radius = 3
        for hh in household_agents:
            for firm in firm_agents:
                dx = abs(hh.pos[0] - firm.pos[0])
                dy = abs(hh.pos[1] - firm.pos[1])
                if dx + dy <= work_radius:
                    hh.nearby_firms.append(firm)

            # Guarantee at least one connection (pick nearest firm) ---------
            if not hh.nearby_firms:
                nearest = min(
                    firm_agents,
                    key=lambda f: (f.pos[0] - hh.pos[0]) ** 2 + (f.pos[1] - hh.pos[1]) ** 2,
                )
                hh.nearby_firms.append(nearest) 