import cv2
import numpy as np
import skimage.morphology

import envs.utils.pose as pu
from envs.utils.fmm_planner import FMMPlanner


def _add_boundary(mat, value=1):
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1:h + 1, 1:w + 1] = mat
    return new_mat


def compute_temp_goal_distance_ieve(
    grid,
    goal_map,
    start,
    planning_window,
    *,
    visited=None,
    collision_map=None,
    map_resolution=5.0,
    obstacle_selem=None,
    goal_dilation_radius=10,
):
    """Match IEVE's original temp-goal FMM reachability rule.

    Returns the FMM distance in centimeters when reachable, otherwise None.
    """
    if grid is None or goal_map is None or start is None:
        return None

    grid = np.rint(np.asarray(grid))
    goal = np.asarray(goal_map, dtype=np.float32) * 1
    if grid.ndim != 2 or goal.shape != grid.shape:
        return None

    gx1, gx2, gy1, gy2 = [int(v) for v in planning_window]
    x1, y1 = 0, 0
    x2, y2 = grid.shape
    start = [int(start[0]), int(start[1])]
    start = pu.threshold_poses(start, grid.shape)

    if obstacle_selem is None:
        obstacle_selem = skimage.morphology.disk(3)
    traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], obstacle_selem)

    if visited is not None:
        visited_arr = np.asarray(visited)
        if visited_arr.shape[0] >= gx2 and visited_arr.shape[1] >= gy2:
            traversible[visited_arr[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1

    if collision_map is not None:
        collision_arr = np.asarray(collision_map)
        if collision_arr.shape[0] >= gx2 and collision_arr.shape[1] >= gy2:
            traversible[
                collision_arr[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1
            ] = 0

    sr, sc = int(start[0] - x1), int(start[1] - y1)
    traversible[
        max(0, sr - 1): min(traversible.shape[0], sr + 2),
        max(0, sc - 1): min(traversible.shape[1], sc + 2),
    ] = 1

    st_dis = pu.get_l2_dis_point_map(start, goal) * float(map_resolution)
    traversible = _add_boundary(traversible)
    planner = FMMPlanner(traversible)
    goal = cv2.dilate(goal, skimage.morphology.disk(int(goal_dilation_radius)))
    goal = _add_boundary(goal, value=0)
    planner.set_multi_goal(goal)
    fmm_dist = planner.fmm_dist * float(map_resolution)
    dis = fmm_dist[start[0] + 1, start[1] + 1]

    if not np.isfinite(dis) or dis >= fmm_dist.max():
        return None
    if st_dis <= 1e-6:
        return dis
    if dis / st_dis < 2:
        return dis
    return None


def compute_temp_goal_reachability_ieve(*args, **kwargs):
    distance = compute_temp_goal_distance_ieve(*args, **kwargs)
    return {
        "reachable": distance is not None,
        "distance_cm": distance,
        "reason": "ieve_temp_goal_reachable"
        if distance is not None
        else "ieve_temp_goal_unreachable",
    }


def select_reachable_goal_ieve(
    grid,
    start,
    target_cell,
    *,
    visited=None,
    collision_map=None,
    map_resolution=5.0,
    preferred_radius_m=0.5,
    fallback_radius_m=1.0,
    max_radius_m=2.0,
    ring_step_m=0.25,
    obstacle_selem=None,
):
    """Select the closest FMM-connected cell around a target.

    The preferred and fallback tiers are terminal goal regions. Cells outside
    the fallback radius are staging goals and must not be treated as success.
    """
    result = {
        "selected": False,
        "reason": "invalid_input",
        "goal_cell": None,
        "goal_map": None,
        "goal_region_mask": None,
        "goal_tier": None,
        "effective_stop_radius_m": None,
        "may_terminal_stop": False,
        "search_radius_m": None,
        "target_center_distance_m": None,
        "fmm_cost_cm": None,
        "obstacle_clearance_m": None,
        "candidate_count": 0,
    }
    if grid is None or start is None or target_cell is None:
        return result

    grid_arr = np.rint(np.asarray(grid)).astype(np.uint8)
    if grid_arr.ndim != 2 or grid_arr.size == 0:
        return result
    h, w = grid_arr.shape
    try:
        start_row, start_col = int(start[0]), int(start[1])
        target_row, target_col = float(target_cell[0]), float(target_cell[1])
    except Exception:
        return result
    start_row = int(np.clip(start_row, 0, h - 1))
    start_col = int(np.clip(start_col, 0, w - 1))

    resolution_cm = max(float(map_resolution), 1e-6)
    resolution_m = resolution_cm / 100.0
    preferred_radius_m = max(0.0, float(preferred_radius_m))
    fallback_radius_m = max(preferred_radius_m, float(fallback_radius_m))
    max_radius_m = max(fallback_radius_m, float(max_radius_m))
    ring_step_m = max(resolution_m, float(ring_step_m))

    if obstacle_selem is None:
        obstacle_selem = skimage.morphology.disk(3)
    traversible = 1.0 - cv2.dilate(grid_arr, obstacle_selem)
    traversible = traversible > 0.5

    if visited is not None:
        visited_arr = np.asarray(visited)
        if visited_arr.shape == traversible.shape:
            traversible[visited_arr > 0] = True
    traversible_without_collision = traversible.copy()
    collision_present = False
    if collision_map is not None:
        collision_arr = np.asarray(collision_map)
        if collision_arr.shape == traversible.shape:
            collision_present = bool(np.any(collision_arr > 0))
            traversible[collision_arr > 0] = False

    traversible[
        max(0, start_row - 1): min(h, start_row + 2),
        max(0, start_col - 1): min(w, start_col + 2),
    ] = True
    traversible_without_collision[
        max(0, start_row - 1): min(h, start_row + 2),
        max(0, start_col - 1): min(w, start_col + 2),
    ] = True

    num_labels, labels = cv2.connectedComponents(
        traversible.astype(np.uint8), connectivity=8
    )
    del num_labels
    start_label = int(labels[start_row, start_col])
    if start_label <= 0:
        result["reason"] = "start_not_traversible"
        return result
    connected = labels == start_label

    padded_traversible = _add_boundary(traversible.astype(np.float32), value=0)
    planner = FMMPlanner(padded_traversible)
    planner.set_goal([start_row + 1, start_col + 1])
    fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)[1:-1, 1:-1]

    rows, cols = np.indices((h, w), dtype=np.float32)
    target_dist_m = (
        np.hypot(rows - target_row, cols - target_col) * resolution_m
    )
    if np.all(traversible):
        # OpenCV has no zero-valued boundary to measure against on an all-free
        # map, so use the map diagonal as a finite, deterministic clearance.
        clearance_m = np.full(
            (h, w), np.hypot(h, w) * resolution_m, dtype=np.float32
        )
    else:
        clearance_m = cv2.distanceTransform(
            traversible.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        ) * resolution_m

    tiers = [
        ("preferred", 0.0, preferred_radius_m, preferred_radius_m, True),
        (
            "fallback",
            preferred_radius_m,
            fallback_radius_m,
            fallback_radius_m,
            True,
        ),
    ]
    lower = fallback_radius_m
    while lower < max_radius_m - 1e-9:
        upper = min(max_radius_m, lower + ring_step_m)
        tiers.append(("staging", lower, upper, None, False))
        lower = upper

    for tier, lower_radius, upper_radius, stop_radius, may_stop in tiers:
        if tier == "preferred":
            radius_mask = target_dist_m <= upper_radius + 1e-6
        else:
            radius_mask = (
                (target_dist_m > lower_radius + 1e-6)
                & (target_dist_m <= upper_radius + 1e-6)
            )
        candidate_mask = (
            connected & traversible & np.isfinite(fmm_dist) & radius_mask
        )
        candidate_rows, candidate_cols = np.where(candidate_mask)
        if candidate_rows.size == 0:
            continue
        ranking = sorted(
            zip(candidate_rows.tolist(), candidate_cols.tolist()),
            key=lambda cell: (
                float(target_dist_m[cell[0], cell[1]]),
                float(fmm_dist[cell[0], cell[1]]),
                -float(clearance_m[cell[0], cell[1]]),
                int(cell[0]),
                int(cell[1]),
            ),
        )
        goal_row, goal_col = ranking[0]
        goal_map = np.zeros((h, w), dtype=np.float32)
        goal_map[goal_row, goal_col] = 1.0
        result.update(
            {
                "selected": True,
                "reason": "selected_{}_goal".format(tier),
                "goal_cell": (int(goal_row), int(goal_col)),
                "goal_map": goal_map,
                "goal_region_mask": target_dist_m <= upper_radius + 1e-6,
                "goal_tier": tier,
                "effective_stop_radius_m": (
                    None if stop_radius is None else float(stop_radius)
                ),
                "may_terminal_stop": bool(may_stop),
                "search_radius_m": float(upper_radius),
                "target_center_distance_m": float(
                    target_dist_m[goal_row, goal_col]
                ),
                "fmm_cost_cm": float(fmm_dist[goal_row, goal_col] * resolution_cm),
                "obstacle_clearance_m": float(clearance_m[goal_row, goal_col]),
                "candidate_count": int(candidate_rows.size),
                "connected_cell_count": int(np.count_nonzero(connected)),
                "traversible_cell_count": int(np.count_nonzero(traversible)),
            }
        )
        return result

    search_mask = target_dist_m <= max_radius_m + 1e-6
    free_before_collision = traversible_without_collision & search_mask
    free_after_collision = traversible & search_mask
    collision_removed_candidates = bool(
        np.any(free_before_collision & ~traversible)
    )
    if not np.any(free_before_collision):
        reason = "target_region_no_free_cell"
    elif not np.any(free_after_collision) and collision_removed_candidates:
        reason = "target_region_blocked_by_collision_map"
    elif not np.any(connected & free_after_collision):
        if collision_present:
            _, labels_without_collision = cv2.connectedComponents(
                traversible_without_collision.astype(np.uint8), connectivity=8
            )
            start_label_without_collision = int(
                labels_without_collision[start_row, start_col]
            )
            connected_without_collision = (
                start_label_without_collision > 0
            ) & (labels_without_collision == start_label_without_collision)
            reason = (
                "path_blocked_by_collision_map"
                if np.any(connected_without_collision & free_before_collision)
                else "target_region_disconnected"
            )
        else:
            reason = "target_region_disconnected"
    else:
        reason = "no_connected_goal_within_max_radius"
    result.update(
        {
            "reason": reason,
            "search_radius_m": float(max_radius_m),
            "connected_cell_count": int(np.count_nonzero(connected)),
            "traversible_cell_count": int(np.count_nonzero(traversible)),
            "free_candidate_count_before_collision": int(
                np.count_nonzero(free_before_collision)
            ),
            "free_candidate_count_after_collision": int(
                np.count_nonzero(free_after_collision)
            ),
        }
    )
    return result
