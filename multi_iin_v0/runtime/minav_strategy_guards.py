"""Additive safety guards for the Multi-IIN MinAV manager.

The wrappers are installed per manager instance.  The baseline IEVE manager
and its source files remain untouched.
"""

from __future__ import annotations

import types
import time
from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np
import skimage.morphology

from envs.utils.fmm_planner import FMMPlanner


def _category_radius(category: Any, default: float) -> float:
    name = str(category or "").lower()
    if "bed" in name or "sofa" in name:
        return 1.15
    if "chair" in name or "plant" in name:
        return 0.80
    if "toilet" in name or "tv" in name or "monitor" in name:
        return 0.65
    return float(default)


def _has_terminal_evidence(manager: Any) -> bool:
    threshold = float(getattr(manager, "confirm_match_threshold", getattr(manager.args, "minav_confirm_match_threshold", 20.0)))
    min_support = int(getattr(manager, "min_support_count", getattr(manager.args, "minav_confirm_min_support", 2)))
    for record in getattr(manager, "phase1_confirmation_evidence", []) or []:
        if bool(record.get("confirmed")):
            return True
        scores = [float(record.get("best_score", 0.0) or 0.0)]
        supports = [int(record.get("support_count", 0) or 0), int(record.get("support", 0) or 0)]
        for cluster in record.get("all_cluster_scores") or []:
            if isinstance(cluster, dict):
                scores.append(float(cluster.get("best_score", 0.0) or 0.0))
                supports.append(int(cluster.get("support_count", cluster.get("support", 0)) or 0))
        if max(scores, default=0.0) >= threshold and max(supports, default=0) >= min_support:
            return True
    return False


def _refresh_fem_foreground_features(fem: Any, mask: Any) -> bool:
    """Build FEM's semantic branch from an already-known goal-image mask.

    ``FeatureMatcher.refresh_target_features`` historically exits early when
    the matcher has no segmenter, even when a valid ``precomputed_mask`` is
    supplied. Multi-IIN normally reuses the manager's goal-image mask and its
    confirmation FEM is intentionally created without another segmenter, so
    that early return silently left matching in ``uniform`` mode. Keep the
    compatibility path when a segmenter is present and provide the equivalent
    mask setup locally otherwise.
    """
    mask_np = np.asarray(mask, dtype=bool)
    target_image = getattr(fem, "target_image", None)
    if target_image is None or mask_np.ndim != 2:
        return False
    target_shape = tuple(np.asarray(target_image).shape[:2])
    if len(target_shape) != 2 or min(target_shape) <= 0:
        return False
    if mask_np.shape != target_shape:
        mask_np = cv2.resize(
            mask_np.astype(np.uint8),
            (int(target_shape[1]), int(target_shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    if not bool(np.any(mask_np)):
        return False

    if (
        getattr(fem, "segmenter", None) is not None
        and getattr(fem, "target_category", None) is not None
        and hasattr(fem, "refresh_target_features")
    ):
        fem.refresh_target_features(
            precomputed_mask=mask_np,
            default_feature_mode="semantic",
        )
    else:
        # Mirror FeatureMatcher._init_semantic_mask for a precomputed binary
        # mask. No semantic model is needed (or rerun) on the goal image.
        fem.default_feature_mode = "semantic"
        fem.target_semantic_mask = mask_np.copy()
        fem.target_sampling_mask = cv2.dilate(
            mask_np.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        component_count, component_labels = cv2.connectedComponents(
            mask_np.astype(np.uint8), connectivity=8
        )
        kept_labels = np.zeros_like(component_labels, dtype=np.int32)
        kept_count = 0
        for component_id in range(1, int(component_count)):
            component = component_labels == component_id
            if int(component.sum()) < 500:
                continue
            kept_count += 1
            kept_labels[component] = kept_count
        fem._target_instance_labels = kept_labels
        fem._n_target_instances = kept_count
        fem._build_target_feature_sets()

    return getattr(fem, "target_features_semantic", None) is not None


def _select_centered_goal_instance_mask(
    manager: Any,
    labels: Any,
    image_shape: Any,
) -> Optional[Dict[str, Any]]:
    """Choose the foreground instance nearest the Instance-ImageNav center.

    Goal renderings are target-centred by construction.  When the detector
    assigns the wrong class (for example plant -> chair), discarding all
    instance masks and using a very large rectangular crop admits background
    furniture into LightGlue.  This fallback uses only image observations: it
    selects a compact detected instance at the image centre, without any
    Habitat object id or target position.
    """
    try:
        height, width = int(image_shape[0]), int(image_shape[1])
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None
    cy, cx = height // 2, width // 2
    diagonal = max(float(np.hypot(height, width)), 1.0)
    target_category = getattr(manager, "target_category", None)
    candidates = []
    for index, label in enumerate(labels or []):
        if not isinstance(label, dict):
            continue
        raw_mask = label.get("binary_mask")
        if raw_mask is None:
            continue
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (height, width):
            continue
        area = int(mask.sum())
        area_ratio = float(area) / float(height * width)
        if area <= 0 or area_ratio < 0.001 or area_ratio > 0.80:
            continue
        ys, xs = np.where(mask)
        center_distance = float(
            np.hypot(float(np.mean(ys)) - cy, float(np.mean(xs)) - cx)
            / diagonal
        )
        center_hit = bool(mask[cy, cx])
        radius_y = max(1, int(round(height * 0.18)))
        radius_x = max(1, int(round(width * 0.18)))
        central = mask[
            max(0, cy - radius_y): min(height, cy + radius_y + 1),
            max(0, cx - radius_x): min(width, cx + radius_x + 1),
        ]
        central_overlap = float(central.sum()) / float(max(area, 1))
        category_match = bool(
            getattr(manager, "_category_matches")(
                label.get("category", ""), target_category or ""
            )
        )
        # Lexicographic ranking intentionally prioritizes image-centred
        # foreground over mask size.  Large background objects therefore
        # cannot win merely by covering more pixels.
        rank = (
            int(category_match),
            int(center_hit),
            -center_distance,
            central_overlap,
            -max(0.0, area_ratio - 0.45),
            min(area_ratio, 0.30),
            -int(index),
        )
        candidates.append(
            (
                rank,
                {
                    "mask": mask.copy(),
                    "category": label.get("category", ""),
                    "pixel_count": area,
                    "area_ratio": area_ratio,
                    "center_hit": center_hit,
                    "center_distance_normalized": center_distance,
                    "central_overlap": central_overlap,
                    "source": "multi_goal_centered_detected_instance",
                },
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _compact_center_prior_mask(category: Any, image_shape: Any) -> Optional[np.ndarray]:
    """Build a conservative image-only fallback when no instance mask exists."""
    try:
        height, width = int(image_shape[0]), int(image_shape[1])
    except Exception:
        return None
    if height <= 0 or width <= 0:
        return None
    name = str(category or "").lower()
    if "bed" in name or "sofa" in name:
        half_w, half_h = 0.34, 0.24
    elif "chair" in name or "plant" in name or "toilet" in name:
        half_w, half_h = 0.24, 0.34
    else:
        half_w, half_h = 0.28, 0.28
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (
        max(1, int(round(width * half_w))),
        max(1, int(round(height * half_h))),
    )
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1, thickness=-1)
    return mask.astype(bool)


def _alternate_reachable_navigation_goal(
    manager: Any,
    raw_target_world: Any,
    context: Dict[str, Any],
    attempts: Any,
    phase3: Optional[bool],
) -> Optional[Dict[str, Any]]:
    """Select a reachable viewpoint while excluding old goals, not paths."""
    grid = context.get("map_pred")
    if grid is None:
        return None
    grid_arr = np.rint(np.asarray(grid)).astype(np.uint8)
    if grid_arr.ndim != 2 or grid_arr.size == 0:
        return None
    raw_goal = manager._raw_ieve_goal_from_world(raw_target_world, context)
    if not raw_goal or not raw_goal.get("ok"):
        return None

    h, w = grid_arr.shape
    start_row, start_col = (int(v) for v in raw_goal["start_cell"])
    target_row, target_col = (float(v) for v in raw_goal["local_cell"])
    start_row = int(np.clip(start_row, 0, h - 1))
    start_col = int(np.clip(start_col, 0, w - 1))
    resolution_cm = max(
        1e-6,
        float(
            raw_goal["projection"].get(
                "map_resolution_cm",
                context.get("map_resolution_cm", getattr(manager.args, "map_resolution", 5.0)),
            )
        ),
    )
    resolution_m = resolution_cm / 100.0

    traversible = 1.0 - cv2.dilate(grid_arr, skimage.morphology.disk(3))
    traversible = traversible > 0.5
    visited = context.get("visited_pred")
    if visited is not None and np.asarray(visited).shape == traversible.shape:
        traversible[np.asarray(visited) > 0] = True
    collision = context.get("collision_map")
    if collision is not None and np.asarray(collision).shape == traversible.shape:
        traversible[np.asarray(collision) > 0] = False
    traversible[
        max(0, start_row - 1): min(h, start_row + 2),
        max(0, start_col - 1): min(w, start_col + 2),
    ] = True

    _, labels = cv2.connectedComponents(traversible.astype(np.uint8), connectivity=8)
    start_label = int(labels[start_row, start_col])
    if start_label <= 0:
        return None
    connected = labels == start_label
    padded = np.pad(traversible.astype(np.float32), 1, constant_values=0)
    planner = FMMPlanner(padded)
    planner.set_goal([start_row + 1, start_col + 1])
    fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)[1:-1, 1:-1]
    rows, cols = np.indices((h, w), dtype=np.float32)
    target_dist_m = np.hypot(rows - target_row, cols - target_col) * resolution_m
    if np.all(traversible):
        clearance_m = np.full(
            (h, w), np.hypot(h, w) * resolution_m, dtype=np.float32
        )
    else:
        clearance_m = cv2.distanceTransform(
            traversible.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        ) * resolution_m

    exclusion = np.zeros((h, w), dtype=bool)
    block_radius_m = max(
        0.05,
        float(getattr(manager.args, "multi_navigation_retry_block_radius_m", 0.45)),
    )
    block_radius_px = max(1, int(round(block_radius_m / resolution_m)))
    blocked_cells = []
    for attempt in attempts:
        try:
            projection = manager._buffer_project_world_to_local(attempt["world"], context)
        except Exception:
            projection = None
        if not projection or not projection.get("ok") or projection.get("clipped_to_local_map", False):
            continue
        cell = tuple(int(v) for v in projection["raw_local_cell"])
        blocked_cells.append(cell)
        exclusion |= (
            (rows - float(cell[0])) ** 2 + (cols - float(cell[1])) ** 2
            <= float(block_radius_px ** 2)
        )
    if not blocked_cells:
        return None

    use_phase3 = manager._phase3_navigation_active() if phase3 is None else bool(phase3)
    if use_phase3:
        preferred = float(manager.strategy_phase3_fallback_stop_radius)
        fallback = preferred + float(manager.phase3_navigation_stuck_tolerance_m)
        maximum = float(manager.phase3_navigation_staging_max_radius)
    else:
        preferred = float(manager.navigation_preferred_stop_radius)
        fallback = max(preferred, float(manager.navigation_arrival_threshold))
        maximum = max(fallback, float(manager.navigation_staging_max_radius))
    ring_step = max(
        resolution_m, float(getattr(manager, "navigation_reachable_ring_step", 0.25))
    )
    tiers = [("preferred", 0.0, preferred, preferred, True)]
    if fallback > preferred + 1e-9:
        tiers.append(("fallback", preferred, fallback, fallback, True))
    lower = fallback
    while lower < maximum - 1e-9:
        upper = min(maximum, lower + ring_step)
        tiers.append(("staging", lower, upper, None, False))
        lower = upper

    for tier, lower, upper, stop_radius, may_stop in tiers:
        if tier == "preferred":
            ring = target_dist_m <= upper + 1e-6
        else:
            ring = (target_dist_m > lower + 1e-6) & (target_dist_m <= upper + 1e-6)
        candidate_mask = connected & traversible & np.isfinite(fmm_dist) & ring & ~exclusion
        candidate_rows, candidate_cols = np.where(candidate_mask)
        if candidate_rows.size == 0:
            continue
        ranking = sorted(
            zip(candidate_rows.tolist(), candidate_cols.tolist()),
            key=lambda cell: (
                float(target_dist_m[cell]),
                float(fmm_dist[cell]),
                -float(clearance_m[cell]),
                int(cell[0]),
                int(cell[1]),
            ),
        )
        goal_cell = tuple(int(v) for v in ranking[0])
        execution_goal = manager._buffer_local_cell_to_world(
            goal_cell,
            raw_goal["projection"],
            np.asarray(raw_target_world, dtype=np.float32).reshape(-1)[:3],
        )
        goal_map = np.zeros((h, w), dtype=np.float32)
        goal_map[goal_cell] = 1.0
        effective_stop_radius = stop_radius
        if use_phase3 and tier == "fallback" and not manager.phase3_navigation_stuck_tolerance_active:
            effective_stop_radius = preferred
        return {
            "selected": True,
            "reason": "selected_{}_alternate_goal".format(tier),
            "goal_cell": goal_cell,
            "goal_map": goal_map,
            "goal_region_mask": target_dist_m <= upper + 1e-6,
            "goal_tier": tier,
            "effective_stop_radius_m": effective_stop_radius,
            "may_terminal_stop": bool(may_stop),
            "search_radius_m": float(upper),
            "target_center_distance_m": float(target_dist_m[goal_cell]),
            "fmm_cost_cm": float(fmm_dist[goal_cell] * resolution_cm),
            "obstacle_clearance_m": float(clearance_m[goal_cell]),
            "candidate_count": int(candidate_rows.size),
            "connected_cell_count": int(np.count_nonzero(connected)),
            "traversible_cell_count": int(np.count_nonzero(traversible)),
            "phase3": bool(use_phase3),
            "raw_target_world": np.asarray(raw_target_world, dtype=np.float32).reshape(-1)[:3].copy(),
            "raw_full_cell": tuple(int(v) for v in raw_goal["full_cell"]),
            "raw_goal": raw_goal,
            "execution_goal_world": execution_goal,
            "full_cell": (
                int(goal_cell[0]) + int(raw_goal["projection"]["gx1"]),
                int(goal_cell[1]) + int(raw_goal["projection"]["gy1"]),
            ),
            "multi_alternate_standoff": True,
            "multi_blocked_goal_cells": blocked_cells,
            "multi_retry_block_radius_m": block_radius_m,
        }
    return None


def _target_directed_frontier_goal(
    manager: Any,
    target_world: Any,
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Choose an informative reachable frontier progressing toward evidence."""
    obstacle = context.get("map_pred")
    explored = context.get("exp_pred")
    if obstacle is None or explored is None:
        return None
    obstacle = np.rint(np.asarray(obstacle)).astype(np.uint8)
    explored = np.asarray(explored) > 0.5
    if obstacle.ndim != 2 or explored.shape != obstacle.shape:
        return None
    projection = manager._buffer_project_world_to_local(target_world, context)
    if not projection or not projection.get("ok"):
        return None
    h, w = obstacle.shape
    start_row, start_col = (int(value) for value in projection["start_cell"])
    target_row, target_col = (
        float(value) for value in projection.get(
            "target_local_cell", projection["raw_local_cell"]
        )
    )
    start_row = int(np.clip(start_row, 0, h - 1))
    start_col = int(np.clip(start_col, 0, w - 1))
    resolution_m = max(
        1e-6, float(projection.get("map_resolution_cm", 5.0)) / 100.0
    )

    traversible = 1.0 - cv2.dilate(obstacle, skimage.morphology.disk(2))
    traversible = traversible > 0.5
    visited = context.get("visited_pred")
    if visited is not None and np.asarray(visited).shape == traversible.shape:
        traversible[np.asarray(visited) > 0] = True
    collision = context.get("collision_map")
    if collision is not None and np.asarray(collision).shape == traversible.shape:
        traversible[np.asarray(collision) > 0] = False
    traversible[
        max(0, start_row - 1): min(h, start_row + 2),
        max(0, start_col - 1): min(w, start_col + 2),
    ] = True
    _, components = cv2.connectedComponents(
        traversible.astype(np.uint8), connectivity=8
    )
    start_component = int(components[start_row, start_col])
    if start_component <= 0:
        return None
    connected = components == start_component

    # Explored free cells bordering unknown space are executable frontiers.
    unknown = ~explored
    unknown_near = cv2.dilate(
        unknown.astype(np.uint8), skimage.morphology.disk(2)
    ) > 0
    frontier = connected & traversible & explored & unknown_near
    rows, cols = np.indices((h, w), dtype=np.float32)
    min_start_px = max(
        2.0,
        float(getattr(manager.args, "multi_phase3_frontier_min_start_m", 0.50))
        / resolution_m,
    )
    frontier &= np.hypot(rows - start_row, cols - start_col) >= min_start_px
    candidate_rows, candidate_cols = np.where(frontier)
    if candidate_rows.size == 0:
        return None

    padded = np.pad(traversible.astype(np.float32), 1, constant_values=0)
    planner = FMMPlanner(padded)
    planner.set_goal([start_row + 1, start_col + 1])
    fmm = np.asarray(planner.fmm_dist, dtype=np.float32)[1:-1, 1:-1]
    finite = np.isfinite(fmm[candidate_rows, candidate_cols])
    candidate_rows, candidate_cols = candidate_rows[finite], candidate_cols[finite]
    if candidate_rows.size == 0:
        return None

    target_distance = np.hypot(rows - target_row, cols - target_col)
    candidate_target = target_distance[candidate_rows, candidate_cols]
    directional_band_px = max(
        1.0,
        float(getattr(manager.args, "multi_phase3_frontier_direction_band_m", 1.50))
        / resolution_m,
    )
    directional = candidate_target <= float(candidate_target.min()) + directional_band_px
    candidate_rows = candidate_rows[directional]
    candidate_cols = candidate_cols[directional]

    gain_radius_px = max(
        1,
        int(round(
            float(getattr(manager.args, "multi_phase3_frontier_gain_radius_m", 0.75))
            / resolution_m
        )),
    )
    gain = cv2.boxFilter(
        unknown.astype(np.float32),
        ddepth=-1,
        ksize=(2 * gain_radius_px + 1, 2 * gain_radius_px + 1),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    ranking = sorted(
        zip(candidate_rows.tolist(), candidate_cols.tolist()),
        key=lambda cell: (
            -float(gain[cell]) / (1.0 + 0.15 * float(fmm[cell]) * resolution_m),
            float(target_distance[cell]),
            float(fmm[cell]),
            int(cell[0]),
            int(cell[1]),
        ),
    )
    goal_cell = tuple(int(value) for value in ranking[0])
    goal_map = np.zeros((h, w), dtype=np.float32)
    goal_map[goal_cell] = 1.0
    world = manager._local_cell_to_world_from_context(
        goal_cell[0], goal_cell[1], context
    )
    return {
        "goal_map": goal_map,
        "goal_cell": goal_cell,
        "goal_world": world,
        "target_world": np.asarray(target_world, dtype=np.float32).reshape(-1)[:3],
        "target_local_cell": (target_row, target_col),
        "information_gain_cells": float(gain[goal_cell]),
        "target_distance_m": float(target_distance[goal_cell] * resolution_m),
        "fmm_cost_m": float(fmm[goal_cell] * resolution_m),
        "projection_window": (
            int(projection.get("gx1", 0)),
            int(projection.get("gy1", 0)),
            h,
            w,
        ),
    }


def install_minav_strategy_guards(manager: Any) -> Any:
    """Install additive wrappers once and return ``manager``."""
    if getattr(manager, "_multi_strategy_guards_installed", False):
        return manager
    args = manager.args
    manager._multi_strategy_guards_installed = True
    manager._multi_original_methods = {}
    manager._multi_rejected_instances = []
    manager._multi_fem_retry_used = False
    manager._multi_last_standoff_selection = None
    manager._multi_navigation_attempted_goals = []
    manager._multi_navigation_last_position = None
    manager._multi_navigation_stuck_frames = 0
    manager._multi_navigation_execution_reached = False
    manager._multi_navigation_perception_guard_active = False
    manager._multi_navigation_perception_stop_accepted = False
    manager._multi_navigation_guard_last_rejection_step = -1
    manager._multi_stage_runtime = {
        "confirmation": {"calls": 0, "total_s": 0.0},
        "localization": {"calls": 0, "total_s": 0.0},
    }
    manager._multi_pipeline_audit = {
        "confirmation_attempts": 0,
        "confirmation_successes": 0,
        "localization_attempts": 0,
        "localization_successes": 0,
        "navigation_entries": 0,
        "strong_identity_unreachable_frames": 0,
        "stable_low_confidence_promotions": 0,
    }
    manager._multi_phase3_frontier_target = None
    manager._multi_phase3_frontier_selection = None
    manager._multi_phase3_frontier_last_step = -10 ** 9
    manager._multi_stable_detection_memory = []

    def wrap(name, function):
        original = getattr(manager, name)
        manager._multi_original_methods[name] = original
        setattr(manager, name, types.MethodType(function, manager))

    def desired(self, step):
        original = self._multi_original_methods["_desired_strategy_phase"]
        # Preserve fixed-step scheduling: a retry flag cannot move Phase 3
        # ahead of the configured start step.
        saved = bool(getattr(self, "targeted_phase2_force_phase3", False))
        respect_step = bool(int(getattr(args, "multi_phase3_respect_start_step", 1)))
        if respect_step and int(step) < int(getattr(self, "strategy_phase3_start_step", 800)):
            self.targeted_phase2_force_phase3 = False
        try:
            return original(int(step))
        finally:
            self.targeted_phase2_force_phase3 = saved

    def phase3_active(self, step):
        if bool(int(getattr(args, "multi_phase3_respect_start_step", 1))) and int(step) < int(getattr(self, "strategy_phase3_start_step", 800)):
            return False
        return self._multi_original_methods["_strategy_phase3_active"](int(step))

    def run_update(self, payload, step):
        original = self._multi_original_methods["_run_targeted_update"]
        threshold = int(getattr(self, "strategy_phase3_start_step", 800))
        saved_limit = getattr(self, "phase2_no_reachable_retry_limit", 3)
        saved_force = bool(getattr(self, "targeted_phase2_force_phase3", False))
        respect_step = bool(int(getattr(args, "multi_phase3_respect_start_step", 1)))
        if respect_step and int(step) < threshold:
            self.phase2_no_reachable_retry_limit = 10 ** 9
            self.targeted_phase2_force_phase3 = False
        try:
            return original(payload, int(step))
        finally:
            self.phase2_no_reachable_retry_limit = saved_limit
            if respect_step and int(step) < threshold:
                self.targeted_phase2_force_phase3 = saved_force and int(step) >= threshold

    def fallback_stop(self, step, reason):
        require_evidence = bool(int(getattr(self.args, "multi_phase3_terminal_requires_confirmed_evidence", 1)))
        if require_evidence and not _has_terminal_evidence(self):
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace("multi_phase3_stop_blocked_insufficient_evidence", step=int(step), reason=str(reason), evidence_count=len(getattr(self, "phase1_confirmation_evidence", []) or []), required_score=float(getattr(self, "confirm_match_threshold", 20.0)), required_support=int(getattr(self, "min_support_count", 2)))
            return False
        return self._multi_original_methods["_maybe_activate_strategy_phase3_fallback_stop"](int(step), reason)

    def record_failure(self, result, step):
        radius = float(getattr(self, "blacklist_radius", 0.5))
        if result and not result.get("confirmed"):
            default_radius = float(getattr(self, "blacklist_radius", 0.5))
            category = getattr(self, "target_category", None) or getattr(self, "target_category_name", None)
            if bool(int(getattr(args, "multi_failure_adaptive_radius_enabled", 1))):
                self.blacklist_radius = _category_radius(category, default_radius)
            radius = float(self.blacklist_radius)
            self.confirmation_failure_blacklist_threshold = int(getattr(args, "multi_failure_soft_threshold", 3))
            self.blacklist_cooldown = int(getattr(args, "multi_failure_soft_cooldown_steps", 80))
            try:
                anchor = self._confirmation_failure_anchor(result)
            except Exception:
                anchor = None
            if anchor is not None:
                match = None
                for entry in self._multi_rejected_instances:
                    if float(np.linalg.norm(np.asarray(entry["center"])[[0, 2]] - np.asarray(anchor)[[0, 2]])) <= radius:
                        match = entry
                        break
                if match is None:
                    match = {"center": np.asarray(anchor, dtype=np.float32).copy(), "count": 0, "persistent": False}
                    self._multi_rejected_instances.append(match)
                elif int(step) - int(match.get("last_step", step)) > int(getattr(args, "multi_failure_soft_cooldown_steps", 80)):
                    match["count"] = 0
                    match["persistent"] = False
                match["count"] = int(match.get("count", 0)) + 1
                match["last_step"] = int(step)
                if match["count"] >= int(getattr(args, "multi_failure_hard_threshold", 5)):
                    match["persistent"] = True
                    trace = getattr(self, "_trace", None)
                    if trace is not None and not bool(match.get("hard_traced")):
                        trace(
                            "multi_confirmation_instance_persistently_rejected",
                            step=int(step),
                            center=match["center"],
                            failure_count=int(match["count"]),
                            radius=float(radius),
                        )
                        match["hard_traced"] = True
        try:
            return self._multi_original_methods["_record_confirmation_failure_for_blacklist"](result, int(step))
        finally:
            # Keep the adaptive values for _is_blacklisted during this target;
            # reset_episode restores the baseline state through the wrapper.
            pass

    def sync_anchor(self, localizer, step, source):
        original = self._multi_original_methods["_sync_full_localizer_candidate_anchor"]
        result = original(localizer, int(step), source)
        try:
            from multi_iin_v0.runtime.localization_runtime import set_confirmed_center_lock

            locked = set_confirmed_center_lock(
                localizer,
                self.confirmed_target_position,
                trace=getattr(self, "_trace", None),
            )
            if locked:
                self._trace(
                    "confirmed_center_lock_synced",
                    step=int(step),
                    source=str(source),
                    center=self.confirmed_target_position,
                    contract="filtered_cloud_must_not_replace_confirmed_center",
                )
        except Exception as exc:
            self.logger.debug("Multi-IIN center lock sync failed: %s", exc)
        return result

    def _run_close_guarded(self, payload):
        """Retry one transient FEM failure using the same confirmed center."""
        original = self._multi_original_methods["run_close_localization"]
        localizer = getattr(self, "full_localizer", None)
        saved_state = getattr(localizer, "current_state", None) if localizer is not None else None
        saved_object = getattr(localizer, "current_object", None) if localizer is not None else None
        saved_images = list(getattr(localizer, "rendered_images", []) or []) if localizer is not None else []
        result = original(payload)
        retryable = (
            isinstance(result, dict)
            and not bool(result.get("localized"))
            and not bool(result.get("system_not_ready"))
            and self.confirmed_target_position is not None
            and _has_terminal_evidence(self)
            and not bool(getattr(self, "_multi_fem_retry_used", False))
        )
        if not retryable:
            return result

        raw = result.get("raw_result") or {}
        raw_state = str(raw.get("state", "")).lower()
        reason = str(result.get("fail_reason", "")).lower()
        if raw_state not in ("failed", "fem") and "fem" not in reason and "match" not in reason:
            return result
        if localizer is None or not saved_images or saved_object is None:
            return result

        self._multi_fem_retry_used = True
        try:
            localizer.current_object = saved_object
            localizer.rendered_images = saved_images
            if saved_state is not None:
                localizer.current_state = saved_state
            # _execute_fem sets FAILED/current_object=None on a miss.  Restore
            # the exact FEM snapshot so the second attempt tests the same
            # render views and locked center, rather than a new candidate.
            from ieve_minav.localizer.localization_system import LocalizationState

            localizer.current_state = LocalizationState.FEM
            self._trace(
                "fem_retry_same_confirmed_center",
                step=int(payload.get("step", 0)),
                center=self.confirmed_target_position,
                first_fail_reason=result.get("fail_reason"),
                rendered_view_count=len(saved_images),
            )
            retry = original(payload)
            if isinstance(retry, dict):
                retry["fem_retry_same_center"] = True
                retry["fem_first_attempt"] = result
            return retry
        except Exception as exc:
            self._trace(
                "fem_retry_same_center_failed",
                step=int(payload.get("step", 0)),
                error=str(exc),
            )
            return result

    def run_close(self, payload):
        started = time.perf_counter()
        result = None
        try:
            result = _run_close_guarded(self, payload)
            return result
        finally:
            stats = self._multi_stage_runtime["localization"]
            stats["calls"] += 1
            stats["total_s"] += float(time.perf_counter() - started)
            audit = self._multi_pipeline_audit
            audit["localization_attempts"] += 1
            if isinstance(result, dict) and bool(result.get("localized")):
                audit["localization_successes"] += 1

    def run_target_confirmation(self, payload):
        started = time.perf_counter()
        result = None
        try:
            result = self._multi_original_methods["run_target_confirmation"](payload)
            return result
        finally:
            stats = self._multi_stage_runtime["confirmation"]
            stats["calls"] += 1
            stats["total_s"] += float(time.perf_counter() - started)
            audit = self._multi_pipeline_audit
            audit["confirmation_attempts"] += 1
            if isinstance(result, dict) and bool(result.get("confirmed")):
                audit["confirmation_successes"] += 1

    def prepare_buffer_goal(self, target_world_xyz, payload, source, search_radius_m,
                            allow_adjustment=True, commit=False,
                            avoid_goal_worlds=None, avoid_radius_m=None):
        """Select the smallest reachable standoff ring before FMM tie-breaks."""
        original = self._multi_original_methods["_prepare_buffer_goal"]
        if not allow_adjustment:
            return original(target_world_xyz, payload, source, search_radius_m,
                            allow_adjustment, commit, avoid_goal_worlds, avoid_radius_m)
        desired = float(max(0.0, getattr(self, "buffer_standoff_min_radius", 0.0)))
        maximum = float(max(desired, getattr(self, "buffer_standoff_max_radius", desired)))
        if search_radius_m and float(search_radius_m) > 0:
            maximum = min(maximum, max(desired, float(search_radius_m)))
        if maximum <= desired + 1e-3:
            return original(target_world_xyz, payload, source, search_radius_m,
                            allow_adjustment, commit, avoid_goal_worlds, avoid_radius_m)
        radii = np.linspace(desired, maximum, num=4).tolist()
        selected = None
        saved_min = self.buffer_standoff_min_radius
        saved_max = self.buffer_standoff_max_radius
        try:
            for radius in radii:
                self.buffer_standoff_min_radius = float(radius)
                self.buffer_standoff_max_radius = float(radius)
                candidate = original(
                    target_world_xyz, payload, source, float(radius),
                    allow_adjustment, False, avoid_goal_worlds, avoid_radius_m,
                )
                if candidate is not None:
                    selected = candidate
                    selected["standoff_selection_policy"] = "minimum_reachable_radius_first"
                    selected["standoff_search_radii_m"] = [float(r) for r in radii]
                    break
        finally:
            self.buffer_standoff_min_radius = saved_min
            self.buffer_standoff_max_radius = saved_max
        if selected is not None and commit:
            # Re-run the upstream commit path at the chosen radius so all
            # navigation fields are populated exactly as before.
            chosen_radius = float(selected.get("stand_off_radius_m", desired))
            self.buffer_standoff_min_radius = chosen_radius
            self.buffer_standoff_max_radius = chosen_radius
            try:
                committed = original(
                    target_world_xyz, payload, source, chosen_radius,
                    allow_adjustment, True, avoid_goal_worlds, avoid_radius_m,
                )
                if committed is not None:
                    committed["standoff_selection_policy"] = "minimum_reachable_radius_first"
                    committed["standoff_search_radii_m"] = [float(r) for r in radii]
                    selected = committed
            finally:
                self.buffer_standoff_min_radius = saved_min
                self.buffer_standoff_max_radius = saved_max
        if selected is not None:
            self._multi_last_standoff_selection = dict(selected)
            self._trace(
                "buffer_standoff_min_reachable_selected",
                step=int(payload.get("step", 0)),
                selected_radius_m=selected.get("stand_off_radius_m"),
                candidate_radii_m=radii,
                target_world=target_world_xyz,
            )
        return selected

    def switch_to_navigation(self, step, from_mode, target_position, loc_result):
        """Keep the confirmed object centre as the navigation/stop target.

        The localizer also returns a camera pose that is useful as an
        observation viewpoint.  The upstream manager already separates that
        pose (``navigation_localizer_suggested_goal``) from the object centre
        (``localized_target_position``).  Do not collapse the two here.
        """
        original = self._multi_original_methods[
            "_switch_to_navigation_from_localization"
        ]
        result = original(step, from_mode, target_position, loc_result)
        self._multi_pipeline_audit["navigation_entries"] += 1
        camera = None
        if isinstance(loc_result, dict):
            camera = loc_result.get("localized_camera_position")
            if camera is None and loc_result.get("position_source") == "localized_target_camera_pose":
                camera = loc_result.get("target_position")
        try:
            camera = np.asarray(camera, dtype=np.float32).reshape(-1)[:3]
            valid = camera.size >= 3 and np.all(np.isfinite(camera))
        except Exception:
            valid = False
        if valid:
            # Diagnostic/view-planning state only.  In particular, never
            # overwrite localized_target_position after the upstream method
            # populated it from target_object_center.
            self._multi_localized_camera_viewpoint = camera.copy()
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace(
                    "multi_iin_localized_viewpoint_recorded",
                    step=int(step),
                    localized_camera_position=camera,
                    navigation_target=getattr(self, "localized_target_position", None),
                    confirmed_object_center=getattr(
                        self, "confirmed_target_position", None
                    ),
                    contract=(
                        "camera_is_observation_viewpoint_object_center_is_navigation_target_"
                        "no_habitat_ground_truth"
                    ),
                )
        return result

    def select_phase3_evidence_object(self, payload, step):
        """Resolve evidence identity before applying reachability.

        The upstream implementation discards unreachable evidence first.  A
        weak, tiny but reachable cluster can consequently replace the much
        stronger target instance.  Select the strongest historical identity
        first; if it is not reachable in the current local map, keep
        exploring instead of navigating to a different object.
        """
        original = self._multi_original_methods[
            "_select_targeted_phase3_evidence_object"
        ]
        if not bool(int(getattr(self.args, "multi_phase3_identity_first", 1))):
            return original(payload, int(step))

        identities = []
        for record_index, record in enumerate(
            getattr(self, "phase1_confirmation_evidence", []) or []
        ):
            if not isinstance(record, dict):
                continue
            record_center = record.get("center")
            if record_center is not None:
                identities.append(
                    {
                        "center": record_center,
                        "score": float(record.get("best_score", 0.0) or 0.0),
                        "matches": int(record.get("best_match_count", 0) or 0),
                        "support": 1,
                        "size": 0,
                        "confirmed": bool(record.get("confirmed")),
                        "record_index": record_index,
                        "source": "record",
                    }
                )
            for cluster in record.get("all_cluster_scores") or []:
                if not isinstance(cluster, dict):
                    continue
                center = cluster.get("target_cluster_center")
                if center is None:
                    center = cluster.get("center_3d")
                if center is None:
                    continue
                identities.append(
                    {
                        "center": center,
                        "score": float(cluster.get("best_score", 0.0) or 0.0),
                        "matches": int(cluster.get("best_match_count", 0) or 0),
                        "support": int(cluster.get("support_count", 1) or 1),
                        "size": int(cluster.get("candidate_cluster_size", 0) or 0),
                        "confirmed": bool(record.get("confirmed")),
                        "record_index": record_index,
                        "source": "cluster",
                    }
                )
        valid = []
        for identity in identities:
            try:
                center = np.asarray(identity["center"], dtype=np.float32).reshape(-1)[:3]
            except Exception:
                continue
            if center.size < 3 or not np.all(np.isfinite(center)):
                continue
            identity["center"] = center
            valid.append(identity)
        if not valid:
            return original(payload, int(step))

        # Prefer identities supported by multiple spatially consistent
        # observations.  A single fortuitous high FEM score must not replace
        # a repeatedly observed target (the plant failure in ep093 exposed
        # this boundary).  Score remains primary inside the same stability
        # tier, and reachability is deliberately absent from this ordering.
        strongest = max(
            valid,
            key=lambda item: (
                int(int(item["support"]) >= 2),
                float(item["score"]),
                int(item["matches"]),
                int(item["support"]),
                int(item["size"]),
                int(item["confirmed"]),
                -int(item["record_index"]),
            ),
        )
        context = dict(payload)
        context.setdefault("current_world_xyz", payload.get("agent_position"))
        context.setdefault(
            "map_resolution_cm",
            getattr(
                self.args,
                "map_resolution",
                self.occupancy_adapter.map_resolution,
            ),
        )
        reachability = self._select_reachable_navigation_goal(
            strongest["center"], context, phase3=True
        )
        if not reachability.get("selected"):
            self._multi_phase3_frontier_target = strongest["center"].copy()
            self._multi_pipeline_audit[
                "strong_identity_unreachable_frames"
            ] += 1
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace(
                    "multi_phase3_strongest_identity_waiting_for_reachability",
                    step=int(step),
                    center=strongest["center"],
                    score=float(strongest["score"]),
                    match_count=int(strongest["matches"]),
                    support_count=int(strongest["support"]),
                    gaussian_count=int(strongest["size"]),
                    source=strongest["source"],
                    candidate_count=len(valid),
                    contract="identity_before_reachability_continue_exploration",
                )
            return None

        self._multi_phase3_frontier_target = None
        self._multi_phase3_frontier_selection = None

        # With the globally strongest identity reachable, the upstream method
        # produces the normal candidate payload and preserves all established
        # navigation bookkeeping.
        return original(payload, int(step))

    def record_detection(self, payload):
        """Accumulate non-consecutive, spatially stable plant/toilet views."""
        original = self._multi_original_methods["_record_detection"]
        original(payload)
        if not bool(int(getattr(self.args, "multi_stable_detection_memory", 1))):
            return
        category = str(getattr(self, "target_category", "") or "").lower()
        if not ("plant" in category or "toilet" in category):
            return
        instances = payload.get("minav_target_instances")
        if not instances:
            instances = payload.get("target_instances") or []
        if not instances:
            return
        best_conf = max(
            float(item.get("confidence", 0.0) or 0.0) for item in instances
        )
        relaxed = float(
            getattr(self.args, "multi_stable_detection_min_confidence", 0.18)
        )
        if best_conf < relaxed:
            return
        position = self._estimate_detection_position(payload, instances)
        if position is None:
            return
        position = np.asarray(position, dtype=np.float32).reshape(-1)[:3]
        if position.size < 3 or not np.all(np.isfinite(position)):
            return
        step = int(payload.get("step", 0))
        if self._is_blacklisted(position, step):
            self._multi_stable_detection_memory = [
                item for item in self._multi_stable_detection_memory
                if float(
                    np.linalg.norm(item["position"][[0, 2]] - position[[0, 2]])
                ) > float(getattr(self, "blacklist_radius", 0.5))
            ]
            return
        max_age = max(
            1, int(getattr(self.args, "multi_stable_detection_max_age_steps", 12))
        )
        radius = max(
            0.05,
            float(getattr(self.args, "multi_stable_detection_radius_m", 0.75)),
        )
        memory = [
            item for item in self._multi_stable_detection_memory
            if step - int(item["step"]) <= max_age
            and float(np.linalg.norm(item["position"][[0, 2]] - position[[0, 2]]))
            <= radius
        ]
        frame = dict(payload)
        frame["slam_camera"] = self._latest_slam_camera()
        frame["target_instances"] = self._copy_target_instances(instances)
        frame["detected_target_position"] = position.copy()
        memory.append(
            {
                "step": step,
                "position": position.copy(),
                "confidence": best_conf,
                "frame": frame,
            }
        )
        minimum = max(
            2, int(getattr(self.args, "multi_stable_detection_min_observations", 2))
        )
        memory = memory[-max(minimum, int(getattr(self, "detect_window", 2))):]
        self._multi_stable_detection_memory = memory
        if len(memory) < minimum:
            return
        # Populate the normal confirmation inputs without changing detector
        # confidence values. _ready_to_confirm owns the explicit stable-memory
        # exception below.
        self.conf_history.clear()
        self.frame_window.clear()
        for item in memory[-int(getattr(self, "detect_window", 2)):]:
            self.conf_history.append(float(item["confidence"]))
            self.frame_window.append(item["frame"])
        self.detected_target_position = np.mean(
            [item["position"] for item in memory], axis=0
        ).astype(np.float32)
        self._multi_pipeline_audit["stable_low_confidence_promotions"] += 1
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(
                "multi_stable_low_confidence_detection_promoted",
                step=step,
                target_category=category,
                observation_count=len(memory),
                confidences=[float(item["confidence"]) for item in memory],
                center=self.detected_target_position,
                radius_m=radius,
                max_age_steps=max_age,
                contract="image_depth_observations_only_no_habitat_target_metadata",
            )

    def ready_to_confirm(self):
        if self._multi_original_methods["_ready_to_confirm"]():
            return True
        category = str(getattr(self, "target_category", "") or "").lower()
        minimum = max(
            2, int(getattr(self.args, "multi_stable_detection_min_observations", 2))
        )
        return bool(
            ("plant" in category or "toilet" in category)
            and len(self._multi_stable_detection_memory) >= minimum
            and len(self.frame_window) >= min(
                minimum, int(getattr(self, "detect_window", 2))
            )
        )

    def clear_detection_window(self):
        result = self._multi_original_methods["_clear_detection_window"]()
        self._multi_stable_detection_memory = []
        return result

    def analyze_goal_center_mask(self, segmenter, step):
        """Use a centred detected instance when category prediction is wrong."""
        # This value belongs to one goal image only.  Clear it before every
        # fresh analysis so a fallback from target N cannot leak into target
        # N+1 when the detector succeeds normally there.
        self._multi_goal_foreground_fallback = None
        original = self._multi_original_methods["_analyze_goal_image_center_mask"]
        mask = original(segmenter, int(step))
        if mask is not None or not bool(
            int(getattr(self.args, "multi_goal_foreground_mask_guard", 1))
        ):
            return mask
        try:
            _semantic, labels, _visual = segmenter.segment_frame(
                self.target_image, largest_only=False
            )
            selected = _select_centered_goal_instance_mask(
                self, labels, self.target_image.shape
            )
        except Exception as exc:
            selected = None
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.debug("Centred goal-mask fallback failed: %s", exc)
        if selected is None:
            return None
        self._multi_goal_foreground_fallback = {
            **dict(selected),
            "goal_image_id": id(getattr(self, "target_image", None)),
            "task_category": str(getattr(self, "target_category", None) or ""),
        }
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(
                "multi_goal_centered_instance_mask_selected",
                step=int(step),
                task_category=getattr(self, "target_category", None),
                detected_category=selected.get("category"),
                pixels=selected.get("pixel_count"),
                area_ratio=selected.get("area_ratio"),
                center_hit=selected.get("center_hit"),
                center_distance_normalized=selected.get(
                    "center_distance_normalized"
                ),
                central_overlap=selected.get("central_overlap"),
                contract="image_observation_only_no_habitat_target_metadata",
            )
        return np.asarray(selected["mask"], dtype=bool)

    def analyze_all_goal_masks(self, segmenter, step):
        """Expose the centred ImageGoal fallback to MR under the task class.

        The segmenter can label the centred target as a nearby/background
        class (for example, chair -> sofa).  The centre-mask guard already
        resolves which image region is the Instance-ImageNav target, but MR
        consumes ``goal_all_masks`` and otherwise sees only the incorrect
        detector class.  Add a task-class proxy only for that exact fallback
        mask and current goal image.  This does not create a global class
        alias and does not use Habitat object positions or ids.
        """
        original = self._multi_original_methods["_analyze_all_goal_image_masks"]
        semantic_mask, all_masks = original(segmenter, int(step))
        masks = list(all_masks or [])
        if not bool(int(getattr(self.args, "multi_goal_foreground_mask_guard", 1))):
            return semantic_mask, masks

        fallback = getattr(self, "_multi_goal_foreground_fallback", None) or {}
        target_category = str(getattr(self, "target_category", None) or "")
        current_image_id = id(getattr(self, "target_image", None))
        fallback_current = (
            fallback.get("goal_image_id") == current_image_id
            and str(fallback.get("task_category", "")) == target_category
        )
        already_has_target = any(
            self._category_matches(item.get("category", ""), target_category)
            for item in masks
            if isinstance(item, dict)
        )
        fallback_mask = fallback.get("mask")
        if (
            not fallback_current
            or already_has_target
            or not target_category
            or fallback_mask is None
        ):
            return semantic_mask, masks

        fallback_mask = np.asarray(fallback_mask, dtype=bool)
        target_image = getattr(self, "target_image", None)
        if (
            target_image is None
            or fallback_mask.ndim != 2
            or fallback_mask.shape != tuple(np.asarray(target_image).shape[:2])
            or not bool(np.any(fallback_mask))
        ):
            return semantic_mask, masks

        masks.append(
            {
                "mask": fallback_mask.copy(),
                "category": target_category,
                "pixel_count": int(fallback_mask.sum()),
                "source": "multi_goal_centered_instance_task_category_proxy",
                "detected_category": fallback.get("category", ""),
            }
        )
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(
                "multi_goal_mr_target_mask_proxy_added",
                step=int(step),
                task_category=target_category,
                detected_category=fallback.get("category"),
                pixels=int(fallback_mask.sum()),
                contract=(
                    "task_category_plus_image_center_prior_no_habitat_object_metadata"
                ),
            )
        return semantic_mask, masks

    def build_goal_center_prior(self, step, reason):
        """Replace the broad 72%-wide box with a compact category prior."""
        original = self._multi_original_methods["_build_goal_image_center_prior_masks"]
        result = original(int(step), reason)
        if not bool(int(getattr(self.args, "multi_goal_compact_center_prior", 1))):
            return result
        target_image = getattr(self, "target_image", None)
        if target_image is None:
            return result
        mask = _compact_center_prior_mask(
            getattr(self, "target_category", None), target_image.shape
        )
        if mask is None:
            return result
        category = str(getattr(self, "target_category", None) or "target")
        self.goal_center_mask = mask
        self.goal_semantic_mask = mask
        self.goal_all_masks = [
            {
                "mask": mask.copy(),
                "category": category,
                "pixel_count": int(mask.sum()),
                "source": "multi_compact_center_prior_instance_imagegoal",
            }
        ]
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(
                "multi_goal_compact_center_prior_mask",
                step=int(step),
                reason=str(reason),
                target_category=category,
                pixels=int(mask.sum()),
                area_ratio=float(mask.mean()),
                image_shape=[int(target_image.shape[0]), int(target_image.shape[1])],
            )
        return result

    def filter_and_rank_candidates(self, clusters):
        """Rank valid same-class instances by anchor distance, then size."""
        original = self._multi_original_methods[
            "_filter_and_rank_target_confirm_candidates"
        ]
        ranked = list(original(clusters) or [])
        if not bool(int(getattr(self.args, "multi_confirm_anchor_first", 1))):
            return ranked
        def _anchor_distance(cluster):
            try:
                value = float(cluster.get("detection_dist_m", float("inf")))
                return value if np.isfinite(value) else float("inf")
            except (TypeError, ValueError):
                return float("inf")

        ranked.sort(
            key=lambda cluster: (
                _anchor_distance(cluster),
                -min(int(cluster.get("size", 0) or 0), 12000),
                -int(cluster.get("support_count", 1) or 1),
                -float(cluster.get("confidence", 0.0) or 0.0),
            )
        )
        return ranked

    def target_confirm_pass_rank(self, cluster, match):
        """Keep the currently observed anchor ahead of a larger lookalike."""
        if not bool(int(getattr(self.args, "multi_confirm_anchor_first", 1))):
            return self._multi_original_methods["_target_confirm_pass_rank"](
                cluster, match
            )
        distance = float(cluster.get("detection_dist_m", float("inf")))
        if not np.isfinite(distance):
            distance = 1e9
        return (
            -distance,
            min(int(cluster.get("size", 0) or 0), 12000),
            int(cluster.get("support_count", 1) or 1),
            float(match.get("score", 0.0) or 0.0),
            int(match.get("match_count", 0) or 0),
        )

    def ensure_target_confirm_nvsm_fem(self):
        """Make distant confirmation score the goal foreground, not background."""
        original = self._multi_original_methods["_ensure_target_confirm_nvsm_fem"]
        nvsm, fem, fail = original()
        if (
            fail is not None
            or fem is None
            or not bool(
                int(getattr(self.args, "multi_goal_masked_confirmation_matching", 1))
            )
        ):
            return nvsm, fem, fail
        payload = getattr(self, "_last_payload", None) or {}
        try:
            self._ensure_goal_image_masks(payload, int(payload.get("step", 0)))
            mask = getattr(self, "goal_semantic_mask", None)
            if mask is None or not bool(np.any(mask)):
                return nvsm, fem, fail
            mask_np = np.asarray(mask, dtype=bool)
            mask_key = (
                id(getattr(self, "target_image", None)),
                tuple(mask_np.shape),
                int(mask_np.sum()),
                str(getattr(self, "target_category", "")),
            )
            if getattr(fem, "_multi_goal_mask_key", None) != mask_key:
                semantic_ready = _refresh_fem_foreground_features(fem, mask_np)
                fem._multi_goal_mask_key = mask_key
            else:
                semantic_ready = getattr(
                    fem, "target_features_semantic", None
                ) is not None
            if not hasattr(fem, "_multi_original_match_rendered_views"):
                fem._multi_original_match_rendered_views = fem.match_rendered_views

                def _masked_match(fem_self, rendered_images, feature_mode=None):
                    semantic_ready = getattr(
                        fem_self, "target_features_semantic", None
                    ) is not None
                    return fem_self._multi_original_match_rendered_views(
                        rendered_images,
                        feature_mode="semantic" if semantic_ready else feature_mode,
                    )

                fem.match_rendered_views = types.MethodType(_masked_match, fem)
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace(
                    "multi_goal_masked_confirmation_ready",
                    step=int(payload.get("step", 0)),
                    target_category=getattr(self, "target_category", None),
                    mask_pixels=int(mask_np.sum()),
                    mask_area_ratio=float(mask_np.mean()),
                    semantic_features_ready=bool(semantic_ready),
                    requested_feature_mode="semantic",
                    mask_fallback_source=(
                        getattr(self, "_multi_goal_foreground_fallback", None) or {}
                    ).get("source"),
                )
        except Exception as exc:
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.debug("Masked target-confirmation setup failed: %s", exc)
        return nvsm, fem, fail

    def adaptive_target_confirmation_match(
        self,
        center,
        cluster_indices,
        stage=None,
        step=0,
        metadata=None,
    ):
        """Coarse-rank ordinary candidates before spending the full 80 views.

        The coarse pass is deliberately additive: it uses the same NVSM/FEM
        implementation, target mask and candidate ordering.  Only candidates
        with enough evidence receive the original full render budget.  If a
        full pass is unavailable or disabled, the coarse result is returned,
        preserving the manager's existing failure/retry behavior.
        """
        original = self._multi_original_methods["_match_candidate_with_nvsm_views"]
        if str(stage or "") != "target_confirmation":
            return original(center, cluster_indices, stage=stage, step=step, metadata=metadata)

        coarse_views = max(1, int(getattr(self.args, "multi_confirm_coarse_views_per_scale", 4)))
        coarse_min_score = float(getattr(self.args, "multi_confirm_coarse_min_score", 8.0))
        full_limit = max(0, int(getattr(self.args, "multi_confirm_full_candidate_limit", 2)))
        if coarse_views <= 0:
            return original(center, cluster_indices, stage=stage, step=step, metadata=metadata)

        # Ensure the renderer exists before changing its per-call view count.
        nvsm = getattr(self, "target_confirm_nvsm", None)
        if nvsm is None:
            try:
                _, _, fail = self._ensure_target_confirm_nvsm_fem()
                if fail is not None:
                    return original(center, cluster_indices, stage=stage, step=step, metadata=metadata)
            except Exception:
                return original(center, cluster_indices, stage=stage, step=step, metadata=metadata)
            nvsm = getattr(self, "target_confirm_nvsm", None)
        if nvsm is None or not hasattr(nvsm, "num_views"):
            return original(center, cluster_indices, stage=stage, step=step, metadata=metadata)

        full_views = max(1, int(getattr(nvsm, "num_views", coarse_views)))
        if coarse_views >= full_views:
            return original(center, cluster_indices, stage=stage, step=step, metadata=metadata)

        # Candidate state is episode-local and reset by the wrapper below.
        candidate_index = int((metadata or {}).get("candidate_rank", 0) or 0)

        saved_views = nvsm.num_views
        saved_render_method = getattr(nvsm, "render_novel_views", None)
        coarse_entries = []

        def capture_coarse_render(*render_args, **render_kwargs):
            render_result = saved_render_method(*render_args, **render_kwargs)
            if isinstance(render_result, dict):
                coarse_entries.extend(list(render_result.get("rendered_images") or []))
            return render_result

        nvsm.num_views = coarse_views
        if callable(saved_render_method):
            nvsm.render_novel_views = capture_coarse_render
        coarse_kwargs = dict(metadata or {})
        coarse_kwargs.update({"confirmation_pass": "coarse", "coarse_views_per_scale": coarse_views})
        try:
            coarse = original(
                center,
                cluster_indices,
                stage=None,
                step=step,
                metadata=coarse_kwargs,
            )
        finally:
            nvsm.num_views = saved_views
            if callable(saved_render_method):
                nvsm.render_novel_views = saved_render_method

        coarse_score = float((coarse or {}).get("score", 0.0) or 0.0)
        if not isinstance(coarse, dict):
            coarse = {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "message": "coarse matcher returned no result",
            }
        coarse_score = float(coarse.get("score", 0.0) or 0.0)
        coarse["coarse_score"] = coarse_score
        coarse["coarse_rendered_view_count"] = int(coarse.get("rendered_view_count", 0) or 0)
        coarse["confirmation_pass"] = "coarse"

        # Low-evidence candidates are discarded without expensive rendering.
        if not coarse.get("available") or coarse_score < coarse_min_score:
            coarse["message"] = "coarse score {:.2f} < {:.2f}".format(coarse_score, coarse_min_score)
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace(
                    "multi_confirmation_coarse_rejected",
                    step=int(step),
                    candidate_index=candidate_index,
                    coarse_score=coarse_score,
                    coarse_min_score=coarse_min_score,
                    rendered_view_count=coarse.get("rendered_view_count", 0),
                )
            return coarse

        # Preserve the old full matcher for the first few promising candidates.
        # Once the limit is reached, the coarse evidence remains available for
        # ranking and the manager can still continue exploration.
        if full_limit > 0 and candidate_index >= full_limit:
            coarse["message"] = "full confirmation candidate limit reached"
            coarse["full_confirmation_skipped"] = True
            # A coarse-only result is useful for diagnostics/ranking but must
            # never cross the unchanged full-confirmation threshold.
            threshold = float(getattr(self, "confirm_match_threshold", 20.0))
            coarse["score"] = min(coarse_score, np.nextafter(threshold, -np.inf))
            return coarse

        # The coarse angles are an exact subset when the full count is an
        # integer multiple (8-view angles are indices 0,2,... of 16). Reuse
        # those RGB renders and ask 3DGS only for the missing angles. This
        # keeps a promoted candidate at the original 80-render total instead
        # of paying 40+80 duplicate renders.
        can_reuse = bool(
            coarse_entries
            and full_views % coarse_views == 0
            and hasattr(nvsm, "_batch_render")
        )
        saved_batch_render = getattr(nvsm, "_batch_render", None)
        if can_reuse:
            stride = full_views // coarse_views
            cached_by_full_index = {}
            for entry in coarse_entries:
                if not isinstance(entry, dict) or entry.get("image") is None:
                    continue
                scale_index = int(entry.get("scale_index", -1))
                local_index = int(entry.get("local_index", -1))
                if scale_index >= 0 and 0 <= local_index < coarse_views:
                    cached_by_full_index[
                        scale_index * full_views + local_index * stride
                    ] = entry["image"]

            def render_missing_angles(all_poses):
                output = [None] * len(all_poses)
                missing_indices = []
                missing_poses = []
                for pose_index, pose in enumerate(all_poses):
                    cached = cached_by_full_index.get(pose_index)
                    if cached is not None:
                        output[pose_index] = cached
                    else:
                        missing_indices.append(pose_index)
                        missing_poses.append(pose)
                rendered_missing = list(saved_batch_render(missing_poses) or [])
                if len(rendered_missing) != len(missing_indices):
                    raise RuntimeError(
                        "NVSM complement render count mismatch: {} != {}".format(
                            len(rendered_missing), len(missing_indices)
                        )
                    )
                for pose_index, image in zip(missing_indices, rendered_missing):
                    output[pose_index] = image
                return output

            nvsm._batch_render = render_missing_angles
        nvsm.num_views = full_views
        try:
            full = original(center, cluster_indices, stage=stage, step=step, metadata=metadata)
        finally:
            nvsm.num_views = saved_views
            if can_reuse:
                nvsm._batch_render = saved_batch_render
        if isinstance(full, dict):
            full["coarse_score"] = coarse_score
            full["coarse_rendered_view_count"] = int(coarse.get("rendered_view_count", 0) or 0)
            full["confirmation_pass"] = "full"
            full["coarse_views_reused"] = int(len(cached_by_full_index)) if can_reuse else 0
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(
                "multi_confirmation_full_candidate",
                step=int(step),
                candidate_index=candidate_index,
                coarse_score=coarse_score,
                full_candidate_index=candidate_index,
                full_view_budget=full_views,
                coarse_views_reused=(
                    int(len(cached_by_full_index)) if can_reuse else 0
                ),
                newly_rendered_full_views=(
                    int(full_views * 5 - len(cached_by_full_index))
                    if can_reuse
                    else int(full_views * 5)
                ),
            )
        return full

    def _remember_navigation_attempt(self, payload: Dict[str, Any], reason: str) -> None:
        world = getattr(self, "navigation_goal_world", None)
        if world is None:
            return
        world = np.asarray(world, dtype=np.float32).reshape(-1)
        if world.size < 3 or not np.all(np.isfinite(world[:3])):
            return
        attempts = self._multi_navigation_attempted_goals
        duplicate = any(
            float(np.linalg.norm(np.asarray(item["world"], dtype=np.float32)[[0, 2]] - world[[0, 2]])) <= 0.05
            for item in attempts
        )
        if not duplicate:
            attempts.append(
                {
                    "world": world[:3].copy(),
                    "step": int(payload.get("step", 0)),
                    "reason": str(reason),
                    "goal_tier": getattr(self, "navigation_goal_tier", None),
                }
            )
            keep = max(1, int(getattr(self.args, "multi_navigation_retry_memory", 8)))
            if len(attempts) > keep:
                del attempts[:-keep]

    def _invalidate_unsuccessful_navigation_goal(self, payload: Dict[str, Any], reason: str) -> None:
        _remember_navigation_attempt(self, payload, reason)
        previous_goal = getattr(self, "navigation_goal_world", None)
        previous_tier = getattr(self, "navigation_goal_tier", None)
        self.navigation_complete = False
        self._multi_navigation_execution_reached = False
        self._multi_navigation_perception_guard_active = True
        self._multi_navigation_perception_stop_accepted = False
        self._multi_navigation_guard_last_rejection_step = int(
            payload.get("step", -1)
        )
        invalidate = getattr(self, "_invalidate_navigation_execution_goal", None)
        if invalidate is not None:
            invalidate()
        trace = getattr(self, "_trace", None)
        if trace is not None:
            trace(
                "multi_navigation_execution_goal_rejected",
                step=int(payload.get("step", 0)),
                reason=str(reason),
                rejected_goal_world=previous_goal,
                rejected_goal_tier=previous_tier,
                attempted_goal_count=len(self._multi_navigation_attempted_goals),
                contract="perception_center_same_instance_different_standoff_angle",
            )

    def update_frame(self, payload):
        return self._multi_original_methods["update_frame"](payload)

    def update_navigation_completion(self, payload):
        original = self._multi_original_methods["_update_navigation_completion"]
        original(payload)
        if not bool(int(getattr(self.args, "multi_navigation_perception_stop_guard", 1))):
            return

        execution_distance = float("inf")
        raw_target_distance = float("inf")
        distance_fn = getattr(self, "_navigation_map_distance_m", None)
        if distance_fn is not None:
            try:
                execution_distance = float(
                    distance_fn(
                        payload,
                        full_cell=getattr(self, "navigation_goal_full_cell", None),
                        local_cell=getattr(self, "navigation_goal_local_cell", None),
                    )
                )
            except Exception:
                execution_distance = float("inf")
            try:
                raw_target_distance = float(
                    distance_fn(
                        payload,
                        full_cell=getattr(self, "navigation_raw_target_full_cell", None),
                        local_cell=None,
                    )
                )
            except Exception:
                raw_target_distance = float("inf")
        execution_reached = bool(
            payload.get("minav_navigation_goal_cell_reached", False)
            or payload.get("minav_fmm_stop", False)
            or (
                np.isfinite(execution_distance)
                and execution_distance
                <= max(0.10, 3.0 * float(getattr(self.args, "map_resolution", 5.0)) / 100.0)
            )
        )
        self._multi_navigation_execution_reached = execution_reached
        # The upstream manager can independently mark the localized raw target
        # complete before the selected execution cell reports arrival.  It is
        # still an internal arrival signal and must pass the same guard.
        upstream_complete = bool(getattr(self, "navigation_complete", False))
        arrival_signal = bool(
            execution_reached
            or (
                upstream_complete
                and not bool(
                    getattr(self, "_multi_navigation_perception_guard_active", False)
                )
            )
        )

        position = payload.get("agent_position")
        stationary = False
        if position is not None:
            try:
                position = np.asarray(position, dtype=np.float32).reshape(-1)[:3]
                previous = self._multi_navigation_last_position
                if previous is not None and position.size >= 3:
                    delta = float(np.linalg.norm(position[[0, 2]] - np.asarray(previous)[[0, 2]]))
                    stationary = delta <= float(getattr(self.args, "multi_navigation_stuck_tolerance_m", 0.03))
                self._multi_navigation_last_position = position.copy()
            except Exception:
                stationary = False
        if stationary and not execution_reached:
            self._multi_navigation_stuck_frames += 1
        else:
            self._multi_navigation_stuck_frames = 0

        terminal_tier = bool(getattr(self, "navigation_may_terminal_stop", False))
        stuck_patience = max(1, int(getattr(self.args, "multi_navigation_stuck_patience", 20)))
        minimum_arrivals = max(
            1,
            int(getattr(self.args, "multi_navigation_min_standoff_arrivals", 2)),
        )
        arrival_count = len(getattr(self, "_multi_navigation_attempted_goals", []) or [])

        # A terminal STOP must be justified exclusively by estimated geometry:
        # the perception-derived locked center, the robot pose/map, and at
        # least ``minimum_arrivals`` distinct reachable standoff viewpoints.
        # Habitat distance/success is deliberately unavailable to this policy.
        if arrival_signal and terminal_tier:
            if arrival_count + 1 < minimum_arrivals:
                _invalidate_unsuccessful_navigation_goal(
                    self, payload, "internal_first_standoff_requires_alternate_view"
                )
                return
            estimated_threshold = float(getattr(self.args, "success_dist", 1.0))
            if np.isfinite(raw_target_distance) and raw_target_distance <= estimated_threshold:
                self.navigation_complete = True
                self._multi_navigation_perception_guard_active = False
                self._multi_navigation_perception_stop_accepted = True
                self._multi_navigation_stuck_frames = 0
                trace = getattr(self, "_trace", None)
                if trace is not None:
                    trace(
                        "multi_navigation_perception_stop_accepted",
                        step=int(payload.get("step", 0)),
                        estimated_target_distance=raw_target_distance,
                        estimated_success_dist=estimated_threshold,
                        distinct_standoff_arrivals=arrival_count + 1,
                        contract="no_habitat_metrics_used_for_navigation_or_stop",
                    )
                return
            # Reaching an execution cell that remains too far from the
            # perception-estimated target used to leave the robot issuing the
            # same command until the 1000-step budget.  Reject it immediately
            # and select another angle instead.
            _invalidate_unsuccessful_navigation_goal(
                self,
                payload,
                "internal_standoff_outside_estimated_success_radius",
            )
            return

        # While an alternate goal is in flight, the upstream manager can mark
        # the nearby estimated center complete again.  Suppress that STOP until
        # the newly selected execution point has actually been reached.
        if self._multi_navigation_perception_guard_active and not execution_reached:
            self.navigation_complete = False
        if (
            self._multi_navigation_stuck_frames >= stuck_patience
            and getattr(self, "navigation_goal_world", None) is not None
        ):
            _invalidate_unsuccessful_navigation_goal(
                self, payload, "internal_navigation_stationary"
            )

    def select_navigation_goal(self, raw_target_world, context, *, phase3=None):
        original = self._multi_original_methods["_select_reachable_navigation_goal"]
        attempts = list(getattr(self, "_multi_navigation_attempted_goals", []) or [])
        if not attempts:
            return original(raw_target_world, context, phase3=phase3)
        selection = _alternate_reachable_navigation_goal(
            self, raw_target_world, context, attempts, phase3
        )
        while selection is None and self._multi_navigation_attempted_goals:
            # Age out the oldest failed viewpoint if all nearby candidates
            # have been covered, so navigation cannot deadlock permanently.
            expired = self._multi_navigation_attempted_goals.pop(0)
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace(
                    "multi_navigation_retry_memory_relaxed",
                    step=int(context.get("step", 0)),
                    expired_goal_world=expired.get("world"),
                    remaining_attempt_count=len(self._multi_navigation_attempted_goals),
                )
            attempts = list(self._multi_navigation_attempted_goals)
            if attempts:
                selection = _alternate_reachable_navigation_goal(
                    self, raw_target_world, context, attempts, phase3
                )
        if selection is None:
            return original(raw_target_world, context, phase3=phase3)
        if selection.get("selected"):
            trace = getattr(self, "_trace", None)
            if trace is not None:
                trace(
                    "multi_navigation_alternate_standoff_selected",
                    step=int(context.get("step", 0)),
                    goal_cell=selection.get("goal_cell"),
                    goal_tier=selection.get("goal_tier"),
                    blocked_goal_cells=selection.get("multi_blocked_goal_cells"),
                    retry_block_radius_m=selection.get("multi_retry_block_radius_m"),
                    attempted_goal_count=len(attempts),
                )
        return selection

    def goal_override(self, context):
        result = self._multi_original_methods["get_goal_override"](context)
        mode = getattr(getattr(self, "mode", None), "value", getattr(self, "mode", None))
        pending_target = getattr(self, "_multi_phase3_frontier_target", None)
        if (
            str(mode).lower() == "exploration"
            and pending_target is not None
            and bool(int(getattr(self.args, "multi_phase3_target_frontier", 1)))
        ):
            step = int(context.get("step", 0))
            refresh = max(
                1,
                int(getattr(self.args, "multi_phase3_frontier_refresh_steps", 10)),
            )
            pose = np.asarray(context.get("pose_pred", []), dtype=np.float32).reshape(-1)
            window = (
                int(pose[3]) if pose.size >= 7 else None,
                int(pose[5]) if pose.size >= 7 else None,
                *tuple(np.asarray(context.get("map_pred", [])).shape[:2]),
            )
            cached = getattr(self, "_multi_phase3_frontier_selection", None)
            if (
                cached is None
                or step - int(self._multi_phase3_frontier_last_step) >= refresh
                or tuple(cached.get("projection_window", ())) != tuple(window)
            ):
                cached = _target_directed_frontier_goal(
                    self, pending_target, context
                )
                self._multi_phase3_frontier_selection = cached
                self._multi_phase3_frontier_last_step = step
                if cached is not None:
                    trace = getattr(self, "_trace", None)
                    if trace is not None:
                        trace(
                            "multi_phase3_target_directed_frontier_selected",
                            step=step,
                            goal_cell=cached.get("goal_cell"),
                            goal_world=cached.get("goal_world"),
                            target_world=cached.get("target_world"),
                            target_local_cell=cached.get("target_local_cell"),
                            information_gain_cells=cached.get("information_gain_cells"),
                            target_distance_m=cached.get("target_distance_m"),
                            fmm_cost_m=cached.get("fmm_cost_m"),
                            contract="perception_evidence_guides_unknown_frontier_no_habitat_ground_truth",
                        )
            if cached is not None:
                result.update(
                    {
                        "goal_world": cached.get("goal_world"),
                        "goal_map": cached["goal_map"],
                        "goal_source": "MULTI_PHASE3_TARGET_DIRECTED_FRONTIER",
                        "found_goal": False,
                        "hold_action": False,
                    }
                )
        if not bool(int(getattr(self.args, "multi_navigation_perception_stop_guard", 1))):
            return result
        if str(mode).lower() != "navigation":
            return result
        if bool(getattr(self, "_multi_navigation_perception_stop_accepted", False)):
            self.navigation_complete = True
            result["navigation_complete"] = True
            result["found_goal"] = True
            return result
        execution_reached = bool(getattr(self, "_multi_navigation_execution_reached", False))
        # get_goal_override() may set navigation_complete again after choosing
        # a reachable goal.  Intercept that late path as well.  The alternate
        # goal is rebuilt immediately so the next action cannot reuse a stale
        # map or sit at FMM stop for hundreds of frames.
        if (
            bool(result.get("navigation_complete", False))
            and not bool(getattr(self, "_multi_navigation_perception_guard_active", False))
        ):
            _invalidate_unsuccessful_navigation_goal(
                self,
                context,
                "goal_override_internal_completion_requires_alternate_view",
            )
            result = self._multi_original_methods["get_goal_override"](context)
            execution_reached = False
        if self._multi_navigation_perception_guard_active and not execution_reached:
            # Keep the alternate goal map, but do not let proximity to the
            # estimated center cause STOP before that viewpoint is reached.
            self.navigation_complete = False
            result["navigation_complete"] = False
            result["found_goal"] = False
            if result.get("force_stop_reason") == "minav_navigation_complete":
                result["force_stop"] = False
                result["force_stop_reason"] = None
        return result

    def reset_episode(self, *args_, **kwargs):
        result = self._multi_original_methods["reset_episode"](*args_, **kwargs)
        self._multi_failure_records = []
        self._multi_rejected_instances = []
        self._multi_fem_retry_used = False
        self._multi_last_standoff_selection = None
        self._multi_navigation_attempted_goals = []
        self._multi_navigation_last_position = None
        self._multi_navigation_stuck_frames = 0
        self._multi_navigation_execution_reached = False
        self._multi_navigation_perception_guard_active = False
        self._multi_navigation_perception_stop_accepted = False
        self._multi_navigation_guard_last_rejection_step = -1
        self._multi_goal_foreground_fallback = None
        self._multi_phase3_frontier_target = None
        self._multi_phase3_frontier_selection = None
        self._multi_phase3_frontier_last_step = -10 ** 9
        self._multi_stable_detection_memory = []
        self._multi_pipeline_audit = {
            "confirmation_attempts": 0,
            "confirmation_successes": 0,
            "localization_attempts": 0,
            "localization_successes": 0,
            "navigation_entries": 0,
            "strong_identity_unreachable_frames": 0,
            "stable_low_confidence_promotions": 0,
        }
        return result

    def is_blacklisted(self, position, step):
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.size >= 3:
            radius = float(getattr(self, "blacklist_radius", 0.5))
            for entry in self._multi_rejected_instances:
                if bool(entry.get("persistent")) and float(np.linalg.norm(pos[[0, 2]] - np.asarray(entry["center"])[[0, 2]])) <= radius:
                    return True
        return self._multi_original_methods["_is_blacklisted"](position, int(step))

    wrap("_desired_strategy_phase", desired)
    wrap("_strategy_phase3_active", phase3_active)
    wrap("_run_targeted_update", run_update)
    wrap("_maybe_activate_strategy_phase3_fallback_stop", fallback_stop)
    wrap("_record_confirmation_failure_for_blacklist", record_failure)
    # Methods below are all additive wrappers; upstream manager code remains
    # untouched on disk.
    if hasattr(manager, "_sync_full_localizer_candidate_anchor"):
        wrap("_sync_full_localizer_candidate_anchor", sync_anchor)
    if hasattr(manager, "run_close_localization"):
        wrap("run_close_localization", run_close)
    if hasattr(manager, "run_target_confirmation"):
        wrap("run_target_confirmation", run_target_confirmation)
    if hasattr(manager, "_prepare_buffer_goal"):
        wrap("_prepare_buffer_goal", prepare_buffer_goal)
    if hasattr(manager, "_switch_to_navigation_from_localization"):
        wrap("_switch_to_navigation_from_localization", switch_to_navigation)
    if hasattr(manager, "_select_targeted_phase3_evidence_object"):
        wrap("_select_targeted_phase3_evidence_object", select_phase3_evidence_object)
    if hasattr(manager, "_record_detection"):
        wrap("_record_detection", record_detection)
    if hasattr(manager, "_ready_to_confirm"):
        wrap("_ready_to_confirm", ready_to_confirm)
    if hasattr(manager, "_clear_detection_window"):
        wrap("_clear_detection_window", clear_detection_window)
    if hasattr(manager, "_analyze_goal_image_center_mask"):
        wrap("_analyze_goal_image_center_mask", analyze_goal_center_mask)
    if hasattr(manager, "_analyze_all_goal_image_masks"):
        wrap("_analyze_all_goal_image_masks", analyze_all_goal_masks)
    if hasattr(manager, "_build_goal_image_center_prior_masks"):
        wrap("_build_goal_image_center_prior_masks", build_goal_center_prior)
    if hasattr(manager, "_filter_and_rank_target_confirm_candidates"):
        wrap("_filter_and_rank_target_confirm_candidates", filter_and_rank_candidates)
    if hasattr(manager, "_target_confirm_pass_rank"):
        wrap("_target_confirm_pass_rank", target_confirm_pass_rank)
    if hasattr(manager, "_ensure_target_confirm_nvsm_fem"):
        wrap("_ensure_target_confirm_nvsm_fem", ensure_target_confirm_nvsm_fem)
    if hasattr(manager, "_match_candidate_with_nvsm_views"):
        wrap("_match_candidate_with_nvsm_views", adaptive_target_confirmation_match)
    if hasattr(manager, "update_frame"):
        wrap("update_frame", update_frame)
    if hasattr(manager, "_update_navigation_completion"):
        wrap("_update_navigation_completion", update_navigation_completion)
    if hasattr(manager, "_select_reachable_navigation_goal"):
        wrap("_select_reachable_navigation_goal", select_navigation_goal)
    if hasattr(manager, "get_goal_override"):
        wrap("get_goal_override", goal_override)
    wrap("_is_blacklisted", is_blacklisted)
    wrap("reset_episode", reset_episode)
    return manager
