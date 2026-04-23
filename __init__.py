from __future__ import annotations

import sys
from pathlib import Path

_UPSTREAM_ROOT = Path(__file__).resolve().parent
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

try:  # pragma: no cover - package import path
    from .api import build_model, run_model
    from .shock_inputs import HazardRasterEvent, LaneShock, NodeShock, RouteShock
except ImportError:
    if __package__:  # pragma: no cover - genuine package import failure
        raise
    from api import build_model, run_model  # type: ignore[no-redef]
    from shock_inputs import HazardRasterEvent, LaneShock, NodeShock, RouteShock  # type: ignore[no-redef]

__all__ = [
    "build_model",
    "run_model",
    "HazardRasterEvent",
    "NodeShock",
    "LaneShock",
    "RouteShock",
]
