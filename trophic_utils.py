# trophic_utils.py
"""Utility functions to compute trophic levels consistently across the project.

Trophic Level Definition
------------------------
The trophic level (TL) of firm i is defined as:

    TL_i = 1 + Σ_j w_ij × TL_j

where the weight w_ij represents the fraction of firm i's input spending
that goes to supplier j:

    w_ij = I_ij / Σ_k I_ik

Here I_ij is the monetary value of inputs purchased from supplier j.
Firms with no suppliers (primary producers) have TL = 1.

For the static analysis (topology_analyzer.py), we assume each firm buys
1 unit from each supplier, so w_ij = 1 / (number of suppliers).

The algorithm uses iterative relaxation that converges even if cycles are
present. For acyclic networks, convergence occurs in one pass.

Usage example
-------------
>>> adj = {1: [], 2: [1], 3: [1, 2]}
>>> levels = compute_trophic_levels(adj)
>>> levels[1]  # No suppliers -> TL = 1
1.0
>>> levels[2]  # Buys from firm 1 -> TL = 1 + 1.0 = 2.0
2.0
>>> levels[3]  # Buys from firms 1,2 -> TL = 1 + (1.0 + 2.0)/2 = 2.5
2.5
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Mapping, Optional


def compute_trophic_levels(
    adjacency: Mapping[int, List[int]],
    edge_weights: Optional[Mapping[Tuple[int, int], float]] = None,
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """Return mapping *firm_id -> trophic level*.

    Parameters
    ----------
    adjacency : mapping
        Dictionary ``buyer_id -> list[supplier_id]``.
    edge_weights : mapping, optional
        Mapping ``(buyer_id, supplier_id) -> weight``.  If omitted, each
        supplier contributes equal weight.
    max_iter, tol : int, float
        Convergence parameters for the iterative solver used when the network
        contains cycles.  In a DAG the solution converges in a single pass.
    """

    # Initial levels: roots (no suppliers) = 1, others = 1 (will be updated).
    levels: Dict[int, float] = {
        node: 1.0 for node in adjacency.keys()
    }

    # Pre-compute supplier weight sums for efficiency
    weight_sum: Dict[int, float] = {}
    if edge_weights is None:
        weight_sum = {n: float(len(supp)) for n, supp in adjacency.items()}
    else:
        for n, suppliers in adjacency.items():
            weight_sum[n] = sum(float(edge_weights.get((n, s), 1.0)) for s in suppliers) or 1.0

    # Iterative relaxation --------------------------------------------------
    for _ in range(max_iter):
        max_delta = 0.0
        for node, suppliers in adjacency.items():
            if not suppliers:  # root – labour only
                continue  # level stays at 1

            # Compute weighted average of supplier levels
            if edge_weights is None:
                total = len(suppliers)
                avg = sum(levels[s] for s in suppliers) / max(1, total)
            else:
                w_levels = 0.0
                for s in suppliers:
                    w = float(edge_weights.get((node, s), 1.0))
                    w_levels += w * levels[s]
                avg = w_levels / weight_sum[node]

            new_lvl = 1.0 + avg
            max_delta = max(max_delta, abs(new_lvl - levels[node]))
            levels[node] = new_lvl

        if max_delta < tol:
            break

    return levels 