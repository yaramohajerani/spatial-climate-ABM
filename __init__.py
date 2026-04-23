from __future__ import annotations

import sys
from pathlib import Path

_UPSTREAM_ROOT = Path(__file__).resolve().parent
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

try:  # pragma: no cover - package import path
    from .api import build_model, run_model
    from .shock_inputs import HazardRasterEvent, LaneShock, NodeShock, RouteShock
except ImportError:  # pragma: no cover - flat import / test runner context
    pass

__all__ = [
    "build_model",
    "run_model",
    "HazardRasterEvent",
    "NodeShock",
    "LaneShock",
    "RouteShock",
]
