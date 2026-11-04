"""Reachability-preserving recovery for disconnected FMM frontier candidates.

This is deliberately independent of the IEVE frontier implementation.  It
only operates on already-built local traversibility/frontier masks and never
uses an Euclidean path through obstacles.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _component_from_start(traversible: np.ndarray, start: Sequence[int]) -> np.ndarray:
    grid = np.asarray(traversible).astype(bool)
    component = np.zeros_like(grid, dtype=bool)
    if grid.ndim != 2 or grid.size == 0:
        return component
    sr, sc = int(start[0]), int(start[1])
    if not (0 <= sr < grid.shape[0] and 0 <= sc < grid.shape[1]):
        return component
    if not grid[sr, sc]:
        free = np.argwhere(grid)
        if free.size == 0:
            return component
        nearest = free[np.argmin(np.sum((free - np.asarray([sr, sc])) ** 2, axis=1))]
        sr, sc = int(nearest[0]), int(nearest[1])
    queue = deque([(sr, sc)])
    component[sr, sc] = True
    while queue:
        row, col = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if (
                    0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and grid[nr, nc]
                    and not component[nr, nc]
                ):
                    component[nr, nc] = True
                    queue.append((nr, nc))
    return component


def _boundary_cells(component: np.ndarray, frontier: np.ndarray) -> np.ndarray:
    """Component cells adjacent to the frontier mask (8-connected)."""
    comp = np.asarray(component).astype(bool)
    front = np.asarray(frontier).astype(bool)
    if comp.shape != front.shape:
        return np.zeros_like(comp)
    adjacent = np.zeros_like(comp)
    rows, cols = np.where(comp)
    for row, col in zip(rows.tolist(), cols.tolist()):
        r0, r1 = max(0, row - 1), min(comp.shape[0], row + 2)
        c0, c1 = max(0, col - 1), min(comp.shape[1], col + 2)
        if np.any(front[r0:r1, c0:c1]):
            adjacent[row, col] = True
    return adjacent


def recover_reachable_frontiers(
    traversible: np.ndarray,
    frontier_mask: np.ndarray,
    start: Sequence[int],
    representative_points: Iterable[Sequence[int]] = (),
    *,
    max_projection_px: float = 15.0,
    min_goal_dist_px: float = 10.0,
    allow_patrol_fallback: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return frontier goals in the start connected component.

    Existing representative points are preferred.  If their exact cells are
    disconnected, they are projected to a nearby cell in the start component;
    as a final recovery, component cells adjacent to unknown/frontier space
    are returned.  Distances are grid shortest-path distances, not straight
    lines, so obstacles are respected.
    """
    grid = np.asarray(traversible).astype(bool)
    front = np.asarray(frontier_mask).astype(bool)
    reps = list(representative_points)
    debug: Dict[str, Any] = {
        "input_count": int(len(reps)),
        "component_cells": 0,
        "projected_count": 0,
        "boundary_count": 0,
        "max_projection_px": float(max_projection_px),
        "used_boundary_fallback": False,
        "used_patrol_fallback": False,
    }
    if grid.ndim != 2 or front.shape != grid.shape:
        debug["reason"] = "shape_mismatch"
        return [], debug
    component = _component_from_start(grid, start)
    debug["component_cells"] = int(component.sum())
    if not component.any():
        debug["reason"] = "empty_start_component"
        return [], debug

    sr, sc = int(start[0]), int(start[1])
    # BFS gives a valid obstacle-aware cost for every reachable cell.
    distances = np.full(grid.shape, np.inf, dtype=np.float32)
    seed = np.argwhere(component)
    if seed.size:
        # The component may have been repaired to a nearby free cell.
        root = seed[np.argmin(np.sum((seed - np.asarray([sr, sc])) ** 2, axis=1))]
        queue = deque([(int(root[0]), int(root[1]))])
        distances[int(root[0]), int(root[1])] = 0.0
        while queue:
            row, col = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and component[nr, nc] and not np.isfinite(distances[nr, nc]):
                    distances[nr, nc] = distances[row, col] + 1.0
                    queue.append((nr, nc))

    selected: Dict[Tuple[int, int], Dict[str, Any]] = {}
    max_proj = max(0.0, float(max_projection_px))
    component_cells = np.argwhere(component)
    for point in reps:
        if len(point) < 2:
            continue
        row, col = int(point[0]), int(point[1])
        if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and component[row, col] and front[row, col]:
            projected = (row, col)
        else:
            if component_cells.size == 0:
                continue
            delta = component_cells.astype(np.float32) - np.asarray([row, col], dtype=np.float32)
            order = np.argsort(np.sum(delta * delta, axis=1))
            projected = None
            for idx in order.tolist():
                candidate = component_cells[idx]
                cr, cc = int(candidate[0]), int(candidate[1])
                # The representative itself may be in a disconnected FMM
                # component.  Project to the nearest free cell in the start
                # component; the BFS distance below still enforces an
                # obstacle-aware route.  Prefer a cell adjacent to frontier
                # space when one is available, but do not require it.
                if float(np.linalg.norm(delta[idx])) <= max_proj:
                    projected = (cr, cc)
                    break
            if projected is None:
                continue
            debug["projected_count"] += 1
        dist = float(distances[projected])
        if not np.isfinite(dist) or dist < float(min_goal_dist_px):
            continue
        selected[projected] = {
            "point": projected,
            "fmm_cost": dist,
            "euclidean_dist": float(np.linalg.norm(np.asarray(projected) - np.asarray([sr, sc]))),
            "detour_ratio": 1.0,
            "frontier_recovery": True,
            "frontier_projection": bool(projected != (row, col)),
        }

    if not selected:
        boundary = _boundary_cells(component, front)
        boundary_coords = np.argwhere(boundary)
        debug["boundary_count"] = int(boundary_coords.shape[0])
        debug["used_boundary_fallback"] = bool(boundary_coords.size)
        for row, col in boundary_coords.tolist():
            dist = float(distances[row, col])
            if np.isfinite(dist) and dist >= float(min_goal_dist_px):
                selected[(int(row), int(col))] = {
                    "point": (int(row), int(col)),
                    "fmm_cost": dist,
                    "euclidean_dist": float(np.linalg.norm(np.asarray([row, col]) - np.asarray([sr, sc]))),
                    "detour_ratio": 1.0,
                    "frontier_recovery": True,
                }
    if not selected and allow_patrol_fallback:
        # A fully explored connected component has no mathematical frontier.
        # Revisit a distant reachable area instead of emitting zero velocity
        # for the rest of the budget.  This uses only the occupancy map.
        patrol_mask = component & np.isfinite(distances)
        if patrol_mask.shape[0] > 2 and patrol_mask.shape[1] > 2:
            patrol_mask[[0, -1], :] = False
            patrol_mask[:, [0, -1]] = False
        patrol_mask &= distances >= float(min_goal_dist_px)
        patrol_cells = np.argwhere(patrol_mask)
        if patrol_cells.size:
            patrol_distances = distances[patrol_mask]
            best_index = int(np.argmax(patrol_distances))
            row, col = [int(value) for value in patrol_cells[best_index]]
            dist = float(distances[row, col])
            selected[(row, col)] = {
                "point": (row, col),
                "fmm_cost": dist,
                "euclidean_dist": float(
                    np.linalg.norm(np.asarray([row, col]) - np.asarray([sr, sc]))
                ),
                "detour_ratio": 1.0,
                "frontier_recovery": True,
                "frontier_patrol": True,
            }
            debug["used_patrol_fallback"] = True
    result = sorted(selected.values(), key=lambda item: float(item["fmm_cost"]))
    debug["output_count"] = int(len(result))
    debug["reason"] = "recovered" if result else "no_reachable_frontier"
    return result, debug
