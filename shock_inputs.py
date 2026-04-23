from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LegacyHazardEvent = tuple[int, int, int, str, str | None]
Coords = tuple[float, float]


def _coerce_coords(values: Sequence[Sequence[float]] | None) -> tuple[Coords, ...]:
    if not values:
        return ()
    coords: list[Coords] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Each coordinate must be a (lon, lat) pair, got {value!r}")
        coords.append((float(value[0]), float(value[1])))
    return tuple(coords)


def _coerce_ids(values: Sequence[int] | None) -> tuple[int, ...]:
    if not values:
        return ()
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class HazardRasterEvent:
    return_period: int
    start_step: int
    end_step: int
    hazard_type: str
    path: str | None

    def __post_init__(self) -> None:
        if self.return_period <= 0:
            raise ValueError(f"return_period must be positive, got {self.return_period}")
        if self.start_step > self.end_step:
            raise ValueError(
                f"start_step ({self.start_step}) must be <= end_step ({self.end_step})"
            )
        if not str(self.hazard_type).strip():
            raise ValueError("hazard_type must not be empty")
        if self.path is not None and not str(self.path).strip():
            raise ValueError("path must be a non-empty string or None")

    @property
    def haz_type(self) -> str:
        return self.hazard_type

    def to_legacy_tuple(self) -> LegacyHazardEvent:
        return (
            int(self.return_period),
            int(self.start_step),
            int(self.end_step),
            str(self.hazard_type),
            None if self.path is None else str(self.path),
        )

    @classmethod
    def from_legacy_tuple(cls, value: LegacyHazardEvent) -> "HazardRasterEvent":
        rp, start, end, hazard_type, path = value
        return cls(
            return_period=int(rp),
            start_step=int(start),
            end_step=int(end),
            hazard_type=str(hazard_type),
            path=None if path is None else str(path),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HazardRasterEvent":
        return cls(
            return_period=int(value["return_period"]),
            start_step=int(value["start_step"]),
            end_step=int(value["end_step"]),
            hazard_type=str(value.get("hazard_type", value.get("haz_type", "FL"))),
            path=None if value.get("path") is None else str(value["path"]),
        )


@dataclass(frozen=True)
class NodeShock:
    """Explicit direct-damage event for coordinates and/or topology firm ids.

    ``intensity`` is normalized to ``[0, 1]`` and mapped to a synthetic flood
    pseudo-depth of ``intensity * 6 m`` before passing through the upstream
    damage curves. This keeps node shocks on the same damage-function scale as
    raster flood inputs.
    """

    label: str
    hazard_type: str
    intensity: float
    start_step: int
    end_step: int
    return_period: float | None = None
    radius_deg: float = 0.5
    affected_coords: tuple[Coords, ...] = ()
    firm_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.intensity) <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if self.start_step > self.end_step:
            raise ValueError(
                f"start_step ({self.start_step}) must be <= end_step ({self.end_step})"
            )
        if self.return_period is not None and float(self.return_period) <= 0:
            raise ValueError(f"return_period must be positive, got {self.return_period}")
        if float(self.radius_deg) < 0:
            raise ValueError(f"radius_deg must be non-negative, got {self.radius_deg}")
        if not str(self.hazard_type).strip():
            raise ValueError("hazard_type must not be empty")
        if not self.affected_coords and not self.firm_ids:
            raise ValueError("NodeShock requires affected_coords and/or firm_ids")

    @property
    def haz_type(self) -> str:
        return self.hazard_type

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NodeShock":
        return cls(
            label=str(value.get("label", "Node shock")),
            hazard_type=str(value.get("hazard_type", value.get("haz_type", "CUSTOM_SHOCK"))),
            intensity=float(value["intensity"]),
            start_step=int(value["start_step"]),
            end_step=int(value["end_step"]),
            return_period=None if value.get("return_period") is None else float(value["return_period"]),
            radius_deg=float(value.get("radius_deg", 0.5)),
            affected_coords=_coerce_coords(value.get("affected_coords")),
            firm_ids=_coerce_ids(value.get("firm_ids")),
        )


@dataclass(frozen=True)
class LaneShock:
    label: str
    supplier_id: int
    buyer_id: int
    capacity_fraction: float
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.capacity_fraction) <= 1.0:
            raise ValueError(
                f"capacity_fraction must be in [0, 1], got {self.capacity_fraction}"
            )
        if self.start_step > self.end_step:
            raise ValueError(
                f"start_step ({self.start_step}) must be <= end_step ({self.end_step})"
            )

    @property
    def blocked_fraction(self) -> float:
        return 1.0 - float(self.capacity_fraction)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> list["LaneShock"]:
        label = str(value.get("label", "Lane shock"))
        capacity_fraction = float(value["capacity_fraction"])
        start_step = int(value["start_step"])
        end_step = int(value["end_step"])

        if "supplier_id" in value and "buyer_id" in value:
            return [
                cls(
                    label=label,
                    supplier_id=int(value["supplier_id"]),
                    buyer_id=int(value["buyer_id"]),
                    capacity_fraction=capacity_fraction,
                    start_step=start_step,
                    end_step=end_step,
                )
            ]

        raw_links = value.get("links")
        if not raw_links:
            raise ValueError("Lane shock mapping requires supplier_id/buyer_id or links")

        shocks: list[LaneShock] = []
        for index, raw_link in enumerate(raw_links, start=1):
            if not isinstance(raw_link, (list, tuple)) or len(raw_link) != 2:
                raise ValueError(f"Each lane link must be [supplier_id, buyer_id], got {raw_link!r}")
            shocks.append(
                cls(
                    label=label if len(raw_links) == 1 else f"{label} #{index}",
                    supplier_id=int(raw_link[0]),
                    buyer_id=int(raw_link[1]),
                    capacity_fraction=capacity_fraction,
                    start_step=start_step,
                    end_step=end_step,
                )
            )
        return shocks


@dataclass(frozen=True)
class RouteShock:
    label: str
    route_tag: str
    intensity: float
    start_step: int
    end_step: int
    return_period: float | None = None
    waypoint_lon: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.intensity) <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if self.start_step > self.end_step:
            raise ValueError(
                f"start_step ({self.start_step}) must be <= end_step ({self.end_step})"
            )
        if self.return_period is not None and float(self.return_period) <= 0:
            raise ValueError(f"return_period must be positive, got {self.return_period}")
        if not str(self.route_tag).strip():
            raise ValueError("route_tag must not be empty")

    @property
    def bottleneck_id(self) -> str:
        return self.route_tag

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RouteShock":
        route_tag = value.get("route_tag", value.get("bottleneck_id"))
        if route_tag is None or not str(route_tag).strip():
            raise ValueError("RouteShock requires route_tag or bottleneck_id")
        return cls(
            label=str(value.get("label", "Route shock")),
            route_tag=str(route_tag),
            intensity=float(value["intensity"]),
            start_step=int(value["start_step"]),
            end_step=int(value["end_step"]),
            return_period=None if value.get("return_period") is None else float(value["return_period"]),
            waypoint_lon=None if value.get("waypoint_lon") is None else float(value["waypoint_lon"]),
        )


def normalize_raster_hazard_events(
    raster_hazard_events: Iterable[HazardRasterEvent | Mapping[str, Any] | LegacyHazardEvent] | None = None,
    *,
    legacy_hazard_events: Iterable[LegacyHazardEvent] | None = None,
) -> list[HazardRasterEvent]:
    normalized: list[HazardRasterEvent] = []
    if legacy_hazard_events:
        normalized.extend(HazardRasterEvent.from_legacy_tuple(value) for value in legacy_hazard_events)
    if not raster_hazard_events:
        return normalized
    for value in raster_hazard_events:
        if isinstance(value, HazardRasterEvent):
            normalized.append(value)
        elif isinstance(value, Mapping):
            normalized.append(HazardRasterEvent.from_mapping(value))
        else:
            normalized.append(HazardRasterEvent.from_legacy_tuple(value))
    return normalized


def normalize_node_shocks(
    node_shocks: Iterable[NodeShock | Mapping[str, Any]] | None = None,
) -> list[NodeShock]:
    if not node_shocks:
        return []
    normalized: list[NodeShock] = []
    for value in node_shocks:
        if isinstance(value, NodeShock):
            normalized.append(value)
        elif isinstance(value, Mapping):
            normalized.append(NodeShock.from_mapping(value))
        else:
            raise TypeError(f"Unsupported node shock value: {value!r}")
    return normalized


def normalize_lane_shocks(
    lane_shocks: Iterable[LaneShock | Mapping[str, Any]] | None = None,
) -> list[LaneShock]:
    if not lane_shocks:
        return []
    normalized: list[LaneShock] = []
    for value in lane_shocks:
        if isinstance(value, LaneShock):
            normalized.append(value)
        elif isinstance(value, Mapping):
            normalized.extend(LaneShock.from_mapping(value))
        else:
            raise TypeError(f"Unsupported lane shock value: {value!r}")
    return normalized


def normalize_route_shocks(
    route_shocks: Iterable[RouteShock | Mapping[str, Any]] | None = None,
) -> list[RouteShock]:
    if not route_shocks:
        return []
    normalized: list[RouteShock] = []
    for value in route_shocks:
        if isinstance(value, RouteShock):
            normalized.append(value)
        elif isinstance(value, Mapping):
            normalized.append(RouteShock.from_mapping(value))
        else:
            raise TypeError(f"Unsupported route shock value: {value!r}")
    return normalized


def legacy_hazard_event_tuples(events: Iterable[HazardRasterEvent]) -> list[LegacyHazardEvent]:
    return [event.to_legacy_tuple() for event in events]
