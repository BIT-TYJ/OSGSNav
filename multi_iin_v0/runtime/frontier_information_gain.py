"""Information-gain-first frontier ranking for the additive Multi-IIN runner."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def resolve_target_local_step(info: Optional[Dict[str, Any]], fallback: int) -> int:
    """Return the current target's clock, falling back to the runner clock."""

    try:
        value = (info or {}).get("time")
        if value is None:
            raise ValueError("missing target-local time")
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(fallback))


def _disk_window(
    shape: Tuple[int, int], row: int, col: int, radius_px: int
) -> Tuple[slice, slice, np.ndarray]:
    height, width = int(shape[0]), int(shape[1])
    radius = max(1, int(radius_px))
    r0, r1 = max(0, row - radius), min(height, row + radius + 1)
    c0, c1 = max(0, col - radius), min(width, col + radius + 1)
    rr, cc = np.ogrid[r0:r1, c0:c1]
    disk = (rr - int(row)) ** 2 + (cc - int(col)) ** 2 <= radius ** 2
    return slice(r0, r1), slice(c0, c1), disk


def rank_frontier_candidates(
    candidates: Iterable[Dict[str, Any]],
    explored_map: np.ndarray,
    *,
    visited_map: Optional[np.ndarray] = None,
    radius_px: int = 30,
    fmm_cost_weight: float = 0.20,
    visited_ratio_penalty: float = 0.25,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rank reachable frontiers by local unknown gain with a mild path penalty.

    FMM has already removed blocked, unreachable and excessive-detour points.
    This function only reorders the surviving candidates.
    """

    explored = np.asarray(explored_map) > 0.5
    if explored.ndim != 2:
        raise ValueError("explored_map must be a 2-D array")
    if visited_map is None:
        visited = np.zeros_like(explored, dtype=bool)
    else:
        visited = np.asarray(visited_map) > 0
        if visited.shape != explored.shape:
            visited = np.zeros_like(explored, dtype=bool)

    enriched: List[Dict[str, Any]] = []
    for source in candidates:
        candidate = dict(source)
        point = candidate.get("point")
        if point is None or len(point) < 2:
            continue
        row, col = int(point[0]), int(point[1])
        if not (0 <= row < explored.shape[0] and 0 <= col < explored.shape[1]):
            continue
        rows, cols, disk = _disk_window(explored.shape, row, col, radius_px)
        unknown_count = int(np.count_nonzero((~explored[rows, cols]) & disk))
        visited_count = int(np.count_nonzero(visited[rows, cols] & disk))
        disk_count = max(1, int(np.count_nonzero(disk)))
        candidate["unknown_gain_cells"] = unknown_count
        candidate["unknown_gain_ratio"] = float(unknown_count) / disk_count
        candidate["visited_ratio"] = float(visited_count) / disk_count
        enriched.append(candidate)

    if not enriched:
        return [], {
            "enabled": True,
            "candidate_count": 0,
            "radius_px": int(max(1, radius_px)),
        }

    gains = np.asarray(
        [float(item["unknown_gain_cells"]) for item in enriched], dtype=np.float32
    )
    costs = np.asarray(
        [float(item.get("fmm_cost", np.inf)) for item in enriched], dtype=np.float32
    )
    finite_costs = costs[np.isfinite(costs)]
    max_gain = max(float(gains.max()), 1.0)
    if finite_costs.size == 0 or float(finite_costs.max() - finite_costs.min()) < 1e-6:
        cost_norm = np.zeros_like(costs)
    else:
        cost_norm = (costs - float(finite_costs.min())) / float(
            finite_costs.max() - finite_costs.min()
        )
        cost_norm[~np.isfinite(cost_norm)] = 1.0

    cost_weight = max(0.0, float(fmm_cost_weight))
    visit_weight = max(0.0, float(visited_ratio_penalty))
    for index, candidate in enumerate(enriched):
        gain_norm = float(gains[index]) / max_gain
        path_factor = 1.0 + cost_weight * float(cost_norm[index])
        utility = gain_norm / path_factor - visit_weight * float(
            candidate["visited_ratio"]
        )
        candidate["information_gain_normalized"] = gain_norm
        candidate["fmm_cost_normalized"] = float(cost_norm[index])
        candidate["frontier_utility"] = float(utility)

    ranked = sorted(
        enriched,
        key=lambda item: (
            -float(item["frontier_utility"]),
            -int(item["unknown_gain_cells"]),
            float(item.get("fmm_cost", np.inf)),
        ),
    )
    debug = {
        "enabled": True,
        "ranking": "unknown_gain_div_path_factor_minus_visited_ratio",
        "candidate_count": int(len(ranked)),
        "radius_px": int(max(1, radius_px)),
        "fmm_cost_weight": cost_weight,
        "visited_ratio_penalty": visit_weight,
        "top_candidates": [
            {
                "point": [int(item["point"][0]), int(item["point"][1])],
                "unknown_gain_cells": int(item["unknown_gain_cells"]),
                "unknown_gain_ratio": round(float(item["unknown_gain_ratio"]), 4),
                "fmm_cost": round(float(item.get("fmm_cost", np.inf)), 3),
                "fmm_cost_normalized": round(float(item["fmm_cost_normalized"]), 4),
                "visited_ratio": round(float(item["visited_ratio"]), 4),
                "utility": round(float(item["frontier_utility"]), 6),
            }
            for item in ranked[:5]
        ],
    }
    return ranked, debug

