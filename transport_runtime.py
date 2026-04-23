from __future__ import annotations

from typing import Dict, List, Tuple


def dedupe_pairs(pairs: List[Tuple[object, object]]) -> List[Tuple[object, object]]:
    seen: set[tuple[int, int]] = set()
    deduped: List[Tuple[object, object]] = []
    for supplier, buyer in pairs:
        key = (
            getattr(supplier, "unique_id", id(supplier)),
            getattr(buyer, "unique_id", id(buyer)),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((supplier, buyer))
    return deduped


def make_transport_patch(supplier_firm, original_sell, blocked_buyers: dict):
    def patched_sell(buying_firm, quantity=1.0, *, unit_price=None, reservation_buyer_id=None):
        requested_qty = max(0.0, float(quantity))
        price = getattr(supplier_firm, "price", 0.0) if unit_price is None else max(0.0, float(unit_price))
        attempted_revenue = requested_qty * price

        supplier_firm.route_sales_attempted_this_step += requested_qty
        supplier_firm.route_revenue_attempted_this_step += attempted_revenue

        buying_firm.inbound_route_sales_attempted_this_step = (
            getattr(buying_firm, "inbound_route_sales_attempted_this_step", 0.0)
            + requested_qty
        )
        buying_firm.inbound_route_revenue_attempted_this_step = (
            getattr(buying_firm, "inbound_route_revenue_attempted_this_step", 0.0)
            + attempted_revenue
        )

        bid = buying_firm.unique_id
        if bid in blocked_buyers:
            blocked_fraction = max(0.0, min(1.0, float(blocked_buyers[bid])))
            blocked_qty = requested_qty * blocked_fraction
            blocked_revenue = blocked_qty * price
            supplier_firm.route_sales_blocked_this_step += blocked_qty
            supplier_firm.route_revenue_blocked_this_step += blocked_revenue
            buying_firm.inbound_route_sales_blocked_this_step = (
                getattr(buying_firm, "inbound_route_sales_blocked_this_step", 0.0)
                + blocked_qty
            )
            buying_firm.inbound_route_revenue_blocked_this_step = (
                getattr(buying_firm, "inbound_route_revenue_blocked_this_step", 0.0)
                + blocked_revenue
            )
            scaled_qty = requested_qty * (1.0 - blocked_fraction)
            result = original_sell(
                buying_firm,
                scaled_qty,
                unit_price=unit_price,
                reservation_buyer_id=reservation_buyer_id,
            )
            supplier_firm.raw_supplier_disruption_this_step = max(
                supplier_firm.raw_supplier_disruption_this_step,
                blocked_fraction,
            )
            return result

        return original_sell(
            buying_firm,
            requested_qty,
            unit_price=unit_price,
            reservation_buyer_id=reservation_buyer_id,
        )

    return patched_sell


def route_exposure_ratio(firm) -> float:
    attempted = max(0.0, float(getattr(firm, "route_revenue_attempted_this_step", 0.0)))
    blocked = max(0.0, float(getattr(firm, "route_revenue_blocked_this_step", 0.0)))
    if attempted <= 1e-9:
        return 0.0
    return min(1.0, blocked / attempted)


def inbound_route_exposure_ratio(firm) -> float:
    attempted = max(0.0, float(getattr(firm, "inbound_route_revenue_attempted_this_step", 0.0)))
    blocked = max(0.0, float(getattr(firm, "inbound_route_revenue_blocked_this_step", 0.0)))
    if attempted <= 1e-9:
        return 0.0
    return min(1.0, blocked / attempted)
