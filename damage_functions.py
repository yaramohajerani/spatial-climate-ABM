"""JRC flood depth-damage functions loaded directly from Excel.

This module provides damage functions for different building types and regions
without requiring CLIMADA. The functions are from the JRC Global Flood
Depth-Damage Functions database.

Reference:
    Huizinga, J., De Moel, H., Szewczyk, W. (2017). Global flood depth-damage
    functions: Methodology and the database with guidelines. EUR 28552 EN.
    doi: 10.2760/16510
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd


# Default path to the JRC damage functions Excel file
DEFAULT_DAMAGE_FUNCTIONS_PATH = Path(__file__).parent / "data" / "global_flood_depth_damage_functions.xlsx"

# Mapping from model sectors to JRC damage classes
SECTOR_TO_JRC_CLASS = {
    "residential": "Residential buildings",
    "commodity": "Industrial buildings",  # Raw materials / extraction
    "manufacturing": "Industrial buildings",  # Factories
    "retail": "Commercial buildings",  # Stores, shops
    "wholesale": "Commercial buildings",  # Distribution centers
    "services": "Commercial buildings",  # Offices, service businesses
    "commercial": "Commercial buildings",
    "agriculture": "Agriculture",
    "transport": "Transport",
    "infrastructure": "Infrastructure - roads",
}

# Region column indices in the Excel file (0-indexed after damage class and depth columns)
REGION_COLUMNS = {
    "EUROPE": 2,
    "North AMERICA": 3,
    "Centr&South AMERICA": 4,
    "ASIA": 5,
    "AFRICA": 6,
    "OCEANIA": 7,
    "GLOBAL": 8,
}

# Mapping from longitude ranges to regions (simplified)
def get_region_from_coords(lon: float, lat: float) -> str:
    """Determine JRC region from longitude/latitude coordinates.

    This is a simplified mapping. For more accurate results, use
    country boundaries or the ISO table in the Excel file.
    """
    # Europe
    if -25 <= lon <= 45 and 35 <= lat <= 72:
        return "EUROPE"
    # North America
    if -170 <= lon <= -50 and 15 <= lat <= 85:
        return "North AMERICA"
    # Central & South America
    if -120 <= lon <= -30 and -60 <= lat <= 15:
        return "Centr&South AMERICA"
    # Asia (including Middle East)
    if 25 <= lon <= 180 and -10 <= lat <= 80:
        return "ASIA"
    if -180 <= lon <= -140 and 50 <= lat <= 72:  # Eastern Russia/Alaska overlap
        return "ASIA"
    # Africa
    if -20 <= lon <= 55 and -40 <= lat <= 40:
        return "AFRICA"
    # Oceania
    if 100 <= lon <= 180 and -50 <= lat <= 0:
        return "OCEANIA"
    if -180 <= lon <= -100 and -50 <= lat <= 0:
        return "OCEANIA"
    # Default to global
    return "GLOBAL"


class JRCDamageFunctions:
    """JRC flood depth-damage functions loaded from Excel.

    Provides interpolated damage fractions for given flood depths,
    building types (sectors), and regions.
    """

    def __init__(self, excel_path: Optional[Path] = None) -> None:
        """Load damage functions from Excel file.

        Parameters
        ----------
        excel_path
            Path to the JRC damage functions Excel file.
            Defaults to data/global_flood_depth_damage_functions.xlsx
        """
        if excel_path is None:
            excel_path = DEFAULT_DAMAGE_FUNCTIONS_PATH

        self.excel_path = Path(excel_path)
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Damage functions file not found: {self.excel_path}")

        # Parse the Excel file
        self._damage_curves: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        self._load_damage_functions()

    def _load_damage_functions(self) -> None:
        """Parse the 'Damage functions' sheet from the Excel file."""
        df = pd.read_excel(self.excel_path, sheet_name='Damage functions', header=None)

        # Find data rows (skip header rows)
        current_class = None

        for idx, row in df.iterrows():
            if idx < 2:  # Skip header rows
                continue

            # Check if this row starts a new damage class
            if pd.notna(row.iloc[0]):
                current_class = str(row.iloc[0]).strip()
                if current_class not in self._damage_curves:
                    self._damage_curves[current_class] = {}

            # Skip if no current class or no depth value
            if current_class is None or pd.isna(row.iloc[1]):
                continue

            depth = float(row.iloc[1])

            # Extract damage fractions for each region
            for region, col_idx in REGION_COLUMNS.items():
                if region not in self._damage_curves[current_class]:
                    self._damage_curves[current_class][region] = ([], [])

                value = row.iloc[col_idx]
                if pd.notna(value) and value != '-':
                    try:
                        damage_frac = float(value)
                        self._damage_curves[current_class][region][0].append(depth)
                        self._damage_curves[current_class][region][1].append(damage_frac)
                    except (ValueError, TypeError):
                        pass

        # Convert lists to numpy arrays for interpolation
        for damage_class in self._damage_curves:
            for region in self._damage_curves[damage_class]:
                depths, fracs = self._damage_curves[damage_class][region]
                self._damage_curves[damage_class][region] = (
                    np.array(depths, dtype=np.float64),
                    np.array(fracs, dtype=np.float64),
                )

    def get_damage_fraction(
        self,
        depth: float,
        sector: str = "residential",
        region: str = "GLOBAL",
    ) -> float:
        """Get interpolated damage fraction for a given flood depth.

        Parameters
        ----------
        depth
            Flood depth in meters.
        sector
            Model sector name (residential, commodity, manufacturing, etc.)
        region
            JRC region name or "GLOBAL" for global average.

        Returns
        -------
        damage_fraction
            Fraction of asset value damaged (0.0 to 1.0).
        """
        # Map sector to JRC damage class
        jrc_class = SECTOR_TO_JRC_CLASS.get(sector.lower(), "Residential buildings")

        # Get the damage curve
        if jrc_class not in self._damage_curves:
            jrc_class = "Residential buildings"  # Fallback

        curves = self._damage_curves[jrc_class]

        # Try requested region, fall back to GLOBAL
        if region not in curves or len(curves[region][0]) == 0:
            region = "GLOBAL"
        if region not in curves or len(curves[region][0]) == 0:
            # Ultimate fallback: linear 0-1 over 0-6m
            return float(np.clip(depth / 6.0, 0.0, 1.0))

        depths, fracs = curves[region]

        # Interpolate
        if depth <= 0:
            return 0.0
        if depth >= depths[-1]:
            return float(fracs[-1])

        return float(np.interp(depth, depths, fracs))

    def get_damage_fractions(
        self,
        depths: np.ndarray,
        sector: str = "residential",
        region: str = "GLOBAL",
    ) -> np.ndarray:
        """Get interpolated damage fractions for multiple flood depths.

        Parameters
        ----------
        depths
            Array of flood depths in meters.
        sector
            Model sector name.
        region
            JRC region name.

        Returns
        -------
        damage_fractions
            Array of damage fractions (0.0 to 1.0).
        """
        # Map sector to JRC damage class
        jrc_class = SECTOR_TO_JRC_CLASS.get(sector.lower(), "Residential buildings")

        if jrc_class not in self._damage_curves:
            jrc_class = "Residential buildings"

        curves = self._damage_curves[jrc_class]

        if region not in curves or len(curves[region][0]) == 0:
            region = "GLOBAL"
        if region not in curves or len(curves[region][0]) == 0:
            return np.clip(depths / 6.0, 0.0, 1.0)

        curve_depths, curve_fracs = curves[region]

        # Vectorized interpolation
        result = np.interp(depths, curve_depths, curve_fracs)
        return np.clip(result, 0.0, 1.0)

    @property
    def available_classes(self) -> list:
        """List of available JRC damage classes."""
        return list(self._damage_curves.keys())

    @property
    def available_regions(self) -> list:
        """List of available regions."""
        return list(REGION_COLUMNS.keys())


# Global instance for convenience (lazy loaded)
_global_damage_functions: Optional[JRCDamageFunctions] = None


def get_damage_functions() -> JRCDamageFunctions:
    """Get the global JRCDamageFunctions instance (lazy loaded)."""
    global _global_damage_functions
    if _global_damage_functions is None:
        _global_damage_functions = JRCDamageFunctions()
    return _global_damage_functions


def calc_damage_fraction(
    depth: float,
    sector: str = "residential",
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    region: Optional[str] = None,
) -> float:
    """Calculate damage fraction for a flood depth.

    Convenience function that uses the global damage functions instance.

    Parameters
    ----------
    depth
        Flood depth in meters.
    sector
        Model sector name.
    lon, lat
        Coordinates to determine region (optional).
    region
        Explicit region name (overrides lon/lat).

    Returns
    -------
    damage_fraction
        Fraction of asset value damaged (0.0 to 1.0).
    """
    if region is None:
        if lon is not None and lat is not None:
            region = get_region_from_coords(lon, lat)
        else:
            region = "GLOBAL"

    return get_damage_functions().get_damage_fraction(depth, sector, region)
