import logging
import contextlib
import json
import os
import re
import sys
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.utils.ieve_reachability import (
    compute_temp_goal_reachability_ieve,
    select_reachable_goal_ieve,
)

from .ieve_segmenter import IEVEMinAVSegmenter, minav_semantic_class_from_category
from .strategy import (
    StrategyTransitionConfig,
    StrategyTransitionManager as FixedStepStrategyTransitionManager,
)


class IEVETargetMode(str, Enum):
    EXPLORATION = "exploration"
    TARGET_CONFIRMATION = "target_confirmation"
    TARGETED_EXPLORATION = "targeted_exploration"
    BUFFER = "buffer"
    LOCALIZATION = "localization"
    CLUSTER_RECOVERY = "cluster_recovery"
    NAVIGATION = "navigation"


class _IEVEFullLocalizerOccupancyAdapter:
    """Minimal occupancy mapper surface expected by MinAV LocalizationSystem."""

    def __init__(self, map_resolution_cm: float = 5.0):
        self.map_resolution = float(map_resolution_cm)
        self.map = np.zeros((1, 1, 3), dtype=np.float32)
        self.map_size_px = 1
        self.robot_px = (0, 0)
        self.pose_pred: Optional[np.ndarray] = None
        self.current_world: Optional[np.ndarray] = None
        self.anchor_pose_xy: Optional[np.ndarray] = None
        self.anchor_world_xz: Optional[np.ndarray] = None
        self.target_semantic_class: Optional[int] = None
        self.map_frame: str = "unknown"

    def update(self, payload: Dict[str, Any]) -> None:
        map_pred = payload.get("full_map_pred")
        exp_pred = payload.get("full_exp_pred")
        semantic_map = payload.get("full_semantic_target_map_pred")
        self.map_frame = "ieve_full_map" if map_pred is not None or semantic_map is not None else "ieve_local_map"
        if map_pred is None:
            map_pred = payload.get("map_pred")
        if exp_pred is None:
            exp_pred = payload.get("exp_pred")
        if semantic_map is None:
            semantic_map = payload.get("semantic_target_map_pred")
        semantic_mask = payload.get("semantic_target_mask_raw")
        if map_pred is None and semantic_map is None and semantic_mask is None:
            return
        if map_pred is not None:
            base = np.asarray(map_pred, dtype=np.float32)
            h, w = base.shape[:2]
        else:
            semantic_source = semantic_map if semantic_map is not None else semantic_mask
            base = np.zeros(np.asarray(semantic_source).shape[:2], dtype=np.float32)
            h, w = base.shape[:2]
        occ_map = np.zeros((h, w, 3), dtype=np.float32)
        occ_map[:, :, 0] = base
        if exp_pred is not None:
            exp = np.asarray(exp_pred, dtype=np.float32)
            if exp.shape[:2] == (h, w):
                occ_map[:, :, 1] = exp
        semantic_source = semantic_map
        if semantic_source is None:
            semantic_source = semantic_mask
        if semantic_source is not None:
            sem = np.asarray(semantic_source)
            if sem.ndim == 3:
                sem = sem[..., 0]
            if sem.shape[:2] == (h, w):
                semantic_id = self._target_semantic_id_from_payload(payload)
                self.target_semantic_class = semantic_id
                occ_map[:, :, 2] = (sem > 0).astype(np.float32) * float(semantic_id)
        self.map = occ_map
        self.map_size_px = int(max(h, w))
        pose_pred = payload.get("pose_pred")
        self.pose_pred = None
        if pose_pred is not None:
            try:
                pose = np.asarray(pose_pred, dtype=np.float32)
                if pose.size >= 7:
                    self.pose_pred = pose.reshape(-1)[:7].copy()
                gx1 = int(pose[3]) if self.map_frame != "ieve_full_map" else 0
                gy1 = int(pose[5]) if self.map_frame != "ieve_full_map" else 0
                row = int(round(float(pose[1]) * 100.0 / self.map_resolution)) - gx1
                col = int(round(float(pose[0]) * 100.0 / self.map_resolution)) - gy1
                self.robot_px = (
                    int(np.clip(col, 0, w - 1)),
                    int(np.clip(row, 0, h - 1)),
                )
            except Exception:
                self.robot_px = (w // 2, h // 2)
        else:
            self.robot_px = (w // 2, h // 2)

        current_world = payload.get("agent_position")
        if current_world is None:
            current_world = payload.get("current_world_xyz")
        self.current_world = None
        if current_world is not None:
            try:
                current = np.asarray(current_world, dtype=np.float32).reshape(-1)
                if current.size >= 3 and np.all(np.isfinite(current[:3])):
                    self.current_world = current[:3].copy()
            except Exception:
                self.current_world = None

        self._maybe_set_world_map_anchor()

    def observe_pose(self, payload: Dict[str, Any]) -> None:
        pose_pred = payload.get("pose_pred")
        if pose_pred is not None:
            try:
                pose = np.asarray(pose_pred, dtype=np.float32)
                if pose.size >= 7:
                    self.pose_pred = pose.reshape(-1)[:7].copy()
            except Exception:
                pass
        current_world = payload.get("agent_position")
        if current_world is None:
            current_world = payload.get("current_world_xyz")
        if current_world is not None:
            try:
                current = np.asarray(current_world, dtype=np.float32).reshape(-1)
                if current.size >= 3 and np.all(np.isfinite(current[:3])):
                    self.current_world = current[:3].copy()
            except Exception:
                pass
        self._maybe_set_world_map_anchor()

    def _maybe_set_world_map_anchor(self) -> None:
        if self.pose_pred is None or self.current_world is None:
            return
        try:
            pose = np.asarray(self.pose_pred, dtype=np.float32).reshape(-1)
            world = np.asarray(self.current_world, dtype=np.float32).reshape(-1)
            if pose.size < 2 or world.size < 3:
                return
            if not np.all(np.isfinite(pose[:2])) or not np.all(np.isfinite(world[[0, 2]])):
                return
            self.anchor_pose_xy = pose[:2].copy()
            self.anchor_world_xz = world[[0, 2]].copy()
        except Exception:
            self.anchor_pose_xy = None
            self.anchor_world_xz = None

    def _world_to_ieve_xy(self, world_x: float, world_z: float) -> Optional[Tuple[float, float]]:
        if self.pose_pred is not None and self.current_world is not None:
            start_x, start_y, _, _, _, _, _ = self.pose_pred
            ieve_x = float(start_x + (self.current_world[2] - float(world_z)))
            ieve_y = float(start_y + (self.current_world[0] - float(world_x)))
            return ieve_x, ieve_y
        if self.anchor_pose_xy is not None and self.anchor_world_xz is not None:
            ieve_x = float(self.anchor_pose_xy[0] + (self.anchor_world_xz[1] - float(world_z)))
            ieve_y = float(self.anchor_pose_xy[1] + (self.anchor_world_xz[0] - float(world_x)))
            return ieve_x, ieve_y
        return None

    def world_to_map(self, world_x: float, world_z: float) -> Tuple[Optional[int], Optional[int]]:
        if not np.isfinite(world_x) or not np.isfinite(world_z):
            return None, None
        if self.map is None or self.map.ndim != 3:
            return None, None
        h, w = self.map.shape[:2]
        ieve_xy = self._world_to_ieve_xy(world_x, world_z)
        if ieve_xy is not None:
            ieve_x, ieve_y = ieve_xy
            gx1 = int(self.pose_pred[3]) if self.pose_pred is not None and self.pose_pred.size >= 7 else 0
            gy1 = int(self.pose_pred[5]) if self.pose_pred is not None and self.pose_pred.size >= 7 else 0
            offset_row = gx1 if self.map_frame != "ieve_full_map" else 0
            offset_col = gy1 if self.map_frame != "ieve_full_map" else 0
            row = int(round(ieve_y * 100.0 / self.map_resolution)) - offset_row
            col = int(round(ieve_x * 100.0 / self.map_resolution)) - offset_col
        else:
            col = int(round(w / 2.0 + float(world_x) * 100.0 / self.map_resolution))
            row = int(round(h / 2.0 - float(world_z) * 100.0 / self.map_resolution))
        if row < 0 or row >= h or col < 0 or col >= w:
            return None, None
        return col, row

    def map_to_world(self, col: int, row: int) -> Tuple[float, float]:
        h, w = self.map.shape[:2]
        if self.pose_pred is not None and self.current_world is not None:
            start_x, start_y, _, gx1, _, gy1, _ = self.pose_pred
            offset_row = int(gx1) if self.map_frame != "ieve_full_map" else 0
            offset_col = int(gy1) if self.map_frame != "ieve_full_map" else 0
            ieve_x = (float(col) + offset_col) * self.map_resolution / 100.0
            ieve_y = (float(row) + offset_row) * self.map_resolution / 100.0
            world_x = float(self.current_world[0] + start_y - ieve_y)
            world_z = float(self.current_world[2] - (ieve_x - start_x))
        elif self.anchor_pose_xy is not None and self.anchor_world_xz is not None:
            gx1 = int(self.pose_pred[3]) if self.pose_pred is not None and self.pose_pred.size >= 7 else 0
            gy1 = int(self.pose_pred[5]) if self.pose_pred is not None and self.pose_pred.size >= 7 else 0
            offset_row = gx1 if self.map_frame != "ieve_full_map" else 0
            offset_col = gy1 if self.map_frame != "ieve_full_map" else 0
            ieve_x = (float(col) + offset_col) * self.map_resolution / 100.0
            ieve_y = (float(row) + offset_row) * self.map_resolution / 100.0
            world_x = float(self.anchor_world_xz[0] + self.anchor_pose_xy[1] - ieve_y)
            world_z = float(self.anchor_world_xz[1] - (ieve_x - self.anchor_pose_xy[0]))
        else:
            world_x = (float(col) - w / 2.0) * self.map_resolution / 100.0
            world_z = -(float(row) - h / 2.0) * self.map_resolution / 100.0
        return world_x, world_z

    def get_traversible_map(self) -> np.ndarray:
        occ = self.map[:, :, 0] if self.map.ndim == 3 else self.map
        exp = self.map[:, :, 1] if self.map.ndim == 3 and self.map.shape[2] > 1 else np.ones_like(occ)
        return ((occ < 0.5) & (exp > 0.0)).astype(np.float32)

    def get_occupancy_map(self, *_, **__) -> np.ndarray:
        return self.map[:, :, 0] if self.map.ndim == 3 else self.map

    @staticmethod
    def _target_semantic_id_from_payload(payload: Dict[str, Any]) -> int:
        raw = payload.get("minav_target_semantic_class")
        if raw is not None:
            try:
                value = int(raw)
            except Exception:
                value = 1
            return max(1, value)

        category = (
            payload.get("target_category")
            or payload.get("goal_name")
            or payload.get("task_name")
        )
        semantic_id = minav_semantic_class_from_category(category)
        if semantic_id > 0:
            return semantic_id

        raw = payload.get("goal_cat_id")
        try:
            value = int(raw) + 1
        except Exception:
            value = 1
        return max(1, value)


class MinAVConfirmationLocalizationManager:
    """Main-process target manager that keeps IEVE FMM as the executor.

    This is the first integration layer for the MinAV-style pipeline. It uses
    strict gates: missing SLAM/NVSM/FEM-style matching support or too few
    Gaussians never produces a positive confirmation.
    """

    def __init__(
        self,
        args,
        env_idx: int,
        slam_mapper=None,
        target_image: Optional[np.ndarray] = None,
        target_category: Optional[str] = None,
    ):
        self.args = args
        self.env_idx = env_idx
        self.logger = logging.getLogger(__name__)
        self.mode = IEVETargetMode.EXPLORATION
        self.slam_mapper = slam_mapper
        self.target_image = self._as_uint8_rgb(target_image)
        self.target_category = target_category

        self.detect_conf_threshold = float(
            getattr(args, "minav_target_detect_confidence", 0.3)
        )
        self.detect_window = int(getattr(args, "minav_target_detect_window", 2))
        self.min_support_count = int(getattr(args, "minav_confirm_min_support", 2))
        self.min_gaussians = int(getattr(args, "minav_confirm_min_gaussians", 200))
        self.confirm_match_threshold = float(
            getattr(args, "minav_confirm_match_threshold", 20.0)
        )
        self.localization_match_threshold = float(
            getattr(args, "minav_localization_match_threshold", 20.0)
        )
        self.cluster_recovery_enabled = bool(
            getattr(args, "minav_cluster_recovery_enabled", 1)
        )
        self.cluster_recovery_trigger_score = float(
            getattr(args, "minav_cluster_recovery_trigger_score", 100.0)
        )
        self.cluster_recovery_max_radius_m = max(
            0.1, float(getattr(args, "minav_cluster_recovery_max_radius_m", 3.0))
        )
        self.cluster_recovery_waypoint_count = max(
            8, int(getattr(args, "minav_cluster_recovery_waypoints", 12))
        )
        self.cluster_recovery_timeout = max(
            1, int(getattr(args, "minav_cluster_recovery_timeout", 200))
        )
        self.cluster_recovery_replan_stall_frames = max(
            1,
            int(
                getattr(
                    args,
                    "minav_cluster_recovery_replan_stall_frames",
                    20,
                )
            ),
        )
        self.cluster_recovery_replan_progress_epsilon_m = max(
            0.0,
            float(
                getattr(
                    args,
                    "minav_cluster_recovery_replan_progress_epsilon_m",
                    0.05,
                )
            ),
        )
        self.cluster_recovery_max_replans = max(
            0,
            int(getattr(args, "minav_cluster_recovery_max_replans", 3)),
        )
        self.cluster_recovery_approach_inner_radius_m = max(
            0.1,
            float(
                getattr(
                    args,
                    "minav_cluster_recovery_approach_inner_radius_m",
                    2.5,
                )
            ),
        )
        self.cluster_recovery_approach_outer_radius_m = min(
            self.cluster_recovery_max_radius_m,
            max(
                self.cluster_recovery_approach_inner_radius_m,
                float(
                    getattr(
                        args,
                        "minav_cluster_recovery_approach_outer_radius_m",
                        2.7,
                    )
                ),
            ),
        )
        self.cluster_recovery_cell_arrival_radius_m = max(
            0.0,
            float(
                getattr(
                    args,
                    "minav_cluster_recovery_cell_arrival_radius_m",
                    0.15,
                )
            ),
        )
        self.cluster_recovery_failure_cluster_center_tolerance_m = max(
            0.0,
            float(
                getattr(
                    args,
                    "minav_cluster_recovery_failure_cluster_center_tolerance_m",
                    0.25,
                )
            ),
        )
        self.cluster_recovery_failure_cluster_voxel_size_m = max(
            0.01,
            float(
                getattr(
                    args,
                    "minav_cluster_recovery_failure_cluster_voxel_size_m",
                    0.10,
                )
            ),
        )
        self.cluster_recovery_failure_cluster_overlap_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        args,
                        "minav_cluster_recovery_failure_cluster_overlap_ratio",
                        0.60,
                    )
                ),
            ),
        )
        self.cluster_recovery_failure_map_change_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        args,
                        "minav_cluster_recovery_failure_map_change_ratio",
                        0.05,
                    )
                ),
            ),
        )
        self.cluster_recovery_min_remaining_steps = max(
            0, int(getattr(args, "minav_cluster_recovery_min_remaining_steps", 50))
        )
        self.cluster_recovery_yaw_range_deg = abs(
            float(getattr(args, "minav_cluster_recovery_yaw_range_deg", 45.0))
        )
        self.cluster_recovery_yaw_step_deg = max(
            0.1, abs(float(getattr(args, "minav_cluster_recovery_yaw_step_deg", 5.0)))
        )
        self.buffer_distance_threshold = float(
            getattr(args, "minav_buffer_distance_threshold", 1.5)
        )
        self.phase2_arrival_threshold = max(
            0.0,
            float(
                getattr(
                    args,
                    "minav_phase2_arrival_threshold",
                    min(0.35, self.buffer_distance_threshold),
                )
            ),
        )
        self.buffer_goal_search_radius = float(
            getattr(args, "minav_buffer_goal_search_radius", 2.5)
        )
        self.buffer_goal_max_extra_radius = float(
            getattr(args, "minav_buffer_goal_max_extra_radius", 0.25)
        )
        self.buffer_standoff_min_radius = max(
            0.0,
            float(getattr(args, "minav_buffer_standoff_min_radius", 1.0)),
        )
        self.buffer_standoff_max_radius = max(
            self.buffer_standoff_min_radius,
            float(getattr(args, "minav_buffer_standoff_max_radius", 1.3)),
        )
        self.buffer_cached_goal_recheck_radius = float(
            getattr(args, "minav_buffer_cached_goal_recheck_radius", 0.10)
        )
        self.buffer_soim_wait_timeout = int(
            getattr(args, "minav_buffer_soim_wait_timeout", 15)
        )
        self.buffer_soim_anchor_radius_m = float(
            getattr(args, "minav_buffer_soim_anchor_radius_m", 3.0)
        )
        self.buffer_timeout = int(getattr(args, "minav_buffer_timeout", 150))
        self.buffer_no_progress_timeout = int(
            getattr(args, "minav_buffer_no_progress_timeout", self.buffer_timeout)
        )
        self.buffer_localization_wait_timeout = int(
            getattr(args, "minav_buffer_localization_wait_timeout", 80)
        )
        self.buffer_goal_update_interval = int(
            getattr(args, "minav_buffer_goal_update_interval", 5)
        )
        self.localization_timeout = int(
            getattr(args, "minav_localization_timeout", 80)
        )
        self.navigation_arrival_threshold = float(
            getattr(args, "minav_navigation_arrival_threshold", getattr(args, "success_dist", 1.0))
        )
        self.navigation_preferred_stop_radius = min(
            max(
                0.0,
                float(
                    getattr(args, "minav_navigation_preferred_stop_radius", 0.5)
                ),
            ),
            self.navigation_arrival_threshold,
        )
        self.navigation_staging_max_radius = max(
            self.navigation_arrival_threshold,
            float(getattr(args, "minav_navigation_staging_max_radius", 2.0)),
        )
        self.phase3_navigation_staging_max_radius = max(
            0.0,
            float(
                getattr(args, "minav_phase3_navigation_staging_max_radius", 1.2)
            ),
        )
        self.navigation_reachable_ring_step = max(
            float(getattr(args, "map_resolution", 5.0)) / 100.0,
            float(getattr(args, "minav_navigation_reachable_ring_step", 0.25)),
        )
        self.navigation_staging_replan_limit = max(
            1,
            int(getattr(args, "minav_navigation_staging_replan_limit", 1)),
        )
        self.navigation_execution_goal_dilation_radius = max(
            0,
            int(
                getattr(
                    args,
                    "minav_navigation_execution_goal_dilation_radius",
                    1,
                )
            ),
        )
        self.blacklist_radius = float(getattr(args, "minav_blacklist_radius", 0.5))
        self.blacklist_cooldown = int(
            getattr(args, "minav_blacklist_cooldown", 50)
        )
        self.confirmation_failure_blacklist_threshold = int(
            getattr(args, "minav_confirm_failure_blacklist_threshold", 10)
        )
        self.confirmation_failure_mask_dilation_radius = int(
            getattr(args, "minav_confirm_failure_mask_dilation_radius", 5)
        )
        self.cluster_eps = float(getattr(args, "minav_confirm_cluster_eps", 0.18))
        self.cluster_min_size = int(
            getattr(args, "minav_confirm_cluster_min_size", 20)
        )
        self.use_full_localizer = bool(getattr(args, "minav_use_full_localizer", 1))
        self.fine_localization_enabled = bool(
            getattr(args, "minav_fine_localization_enabled", 1)
        )
        self.verbose = bool(getattr(args, "minav_verbose", 0))
        self.save_stage_visualizations = bool(
            getattr(args, "minav_save_stage_visualizations", 1)
        )
        self.gaussian_projection_debug = bool(
            getattr(args, "minav_gaussian_projection_debug", 0)
        )
        self.gaussian_projection_debug_overlay = bool(
            getattr(args, "minav_gaussian_projection_debug_overlay", 0)
        )
        self.gaussian_projection_debug_ply = bool(
            getattr(args, "minav_gaussian_projection_debug_ply", 0)
        )
        self.gaussian_projection_debug_sample = int(
            getattr(args, "minav_gaussian_projection_debug_sample", 200)
        )
        self.initial_spin_enabled = bool(getattr(args, "minav_initial_spin", 1))
        self.initial_spin_total_steps = (
            self._full_turn_sweep_steps() if self.initial_spin_enabled else 0
        )
        self.initial_spin_remaining_steps = self.initial_spin_total_steps
        self.initial_spin_completed = not self.initial_spin_enabled
        self.buffer_standoff_sweep_enabled = bool(
            getattr(args, "minav_buffer_standoff_sweep", 1)
        )
        self.buffer_phase2_refine_frames = int(
            getattr(args, "minav_buffer_phase2_refine_frames", 12)
        )
        self.strategy_controller = FixedStepStrategyTransitionManager(
            StrategyTransitionConfig(
                enabled=bool(
                    getattr(
                        args,
                        "minav_secondary_exploration_enabled",
                        getattr(args, "minav_enable_strategy_transition", 1),
                    )
                ),
                use_area_growth_trigger=bool(
                    getattr(args, "minav_strategy_use_area_growth_trigger", 0)
                ),
                phase2_start_step=int(
                    getattr(args, "minav_strategy_phase2_start_step", 500)
                ),
                phase3_start_step=int(
                    getattr(args, "minav_strategy_phase3_start_step", 800)
                ),
                phase3_fallback_stop_radius=float(
                    getattr(args, "minav_strategy_phase3_fallback_stop_radius", 0.5)
                ),
            )
        )
        self.strategy_transition_enabled = self.strategy_controller.config.enabled
        self.strategy_use_area_growth_trigger = (
            self.strategy_controller.config.use_area_growth_trigger
        )
        self.strategy_phase2_start_step = (
            self.strategy_controller.config.phase2_start_step
        )
        self.strategy_phase3_start_step = (
            self.strategy_controller.config.phase3_start_step
        )
        self.strategy_phase3_fallback_stop_radius = (
            self.strategy_controller.config.phase3_fallback_stop_radius
        )
        self.phase2_standoff_retry_limit = max(
            0, int(getattr(args, "minav_phase2_standoff_retry_limit", 3))
        )
        self.phase2_standoff_stuck_patience = max(
            1,
            int(
                getattr(
                    args,
                    "minav_phase2_standoff_stuck_patience",
                    getattr(args, "minav_phase2_no_goal_timeout", 8),
                )
            ),
        )
        self.phase2_standoff_retry_block_radius = max(
            0.0,
            float(getattr(args, "minav_phase2_standoff_retry_block_radius", 0.45)),
        )
        self.targeted_quick_lock_max_dist_m = max(
            0.0,
            float(getattr(args, "minav_targeted_quick_lock_max_dist_m", 1.5)),
        )
        self.phase2_no_reachable_cooldown_steps = max(
            0,
            int(getattr(args, "minav_phase2_no_reachable_cooldown_steps", 40)),
        )
        self.phase2_no_reachable_retry_limit = max(
            1,
            int(getattr(args, "minav_phase2_no_reachable_retry_limit", 3)),
        )
        self.phase3_no_candidate_cooldown_steps = max(
            0,
            int(getattr(args, "minav_phase3_no_candidate_cooldown_steps", 40)),
        )
        self.phase3_navigation_stuck_patience = max(
            1,
            int(getattr(args, "minav_phase3_navigation_stuck_patience", 20)),
        )
        self.phase3_navigation_progress_epsilon_m = max(
            0.0,
            float(
                getattr(args, "minav_phase3_navigation_progress_epsilon_m", 0.02)
            ),
        )
        self.phase3_navigation_stuck_tolerance_m = max(
            0.0,
            float(
                getattr(args, "minav_phase3_navigation_stuck_tolerance_m", 0.1)
            ),
        )
        self.phase3_navigation_staging_max_radius = max(
            self.phase3_navigation_staging_max_radius,
            self.strategy_phase3_fallback_stop_radius
            + self.phase3_navigation_stuck_tolerance_m,
        )
        results_dir = getattr(args, "minav_results_dir", "")
        if results_dir:
            self.results_root = Path(results_dir).expanduser()
            if not self.results_root.is_absolute():
                self.results_root = Path(__file__).resolve().parents[1] / self.results_root
        else:
            self.results_root = Path(__file__).resolve().parents[1] / "results"

        self.conf_history: deque = deque(maxlen=self.detect_window)
        self.frame_window: deque = deque(maxlen=self.detect_window)
        self.goal_blacklist: List[Dict[str, Any]] = []
        self.confirmation_failure_cache: List[Dict[str, Any]] = []
        map_size = int(
            getattr(args, "map_size_cm", 3600)
            // max(1, int(getattr(args, "map_resolution", 5)))
        )
        self.full_map_shape: Tuple[int, int] = (map_size, map_size)
        self.confirmation_mask_entries: List[Dict[str, Any]] = []
        self.confirmation_blocked_full_mask = np.zeros(
            self.full_map_shape, dtype=np.uint8
        )
        self.targeted_area_samples: deque = deque(maxlen=11)
        self.targeted_area_frame_counter: int = 0
        self.targeted_revert_samples: deque = deque(maxlen=2)
        self.targeted_revert_frame_counter: int = 0
        self.targeted_exploration_complete: bool = False
        self.targeted_phase: int = 1
        self.targeted_objects: Dict[int, Dict[str, Any]] = {}
        self.targeted_visited_positions: List[Tuple[float, float]] = []
        self.targeted_failed_regions: List[Dict[str, Any]] = []
        self.targeted_current_obj_id: Optional[int] = None
        self.targeted_current_goal_world: Optional[np.ndarray] = None
        self.targeted_current_goal_map: Optional[np.ndarray] = None
        self.targeted_current_semantic_position: Optional[np.ndarray] = None
        self.targeted_current_standoff_retries: int = 0
        self.targeted_current_stuck_frames: int = 0
        self.targeted_current_failed_standoffs: List[np.ndarray] = []
        self.targeted_task_should_terminate: bool = False
        self.targeted_phase2_no_reachable_retries: int = 0
        self.targeted_phase2_no_reachable_cooldown_until: int = -1
        self.targeted_phase2_force_phase3: bool = False
        self.targeted_phase3_no_candidate_cooldown_until: int = -1
        self.strategy_last_desired_phase: int = 1
        self.strategy_force_stop: bool = False
        self.strategy_force_stop_reason: Optional[str] = None
        self.strategy_force_stop_step: int = -1

        self.confirmed_target_position: Optional[np.ndarray] = None
        self.confirmed_target_map_anchor_position: Optional[np.ndarray] = None
        self.confirmed_indices: Optional[np.ndarray] = None
        self.detected_target_position: Optional[np.ndarray] = None
        self.confirmation_detected_target_position: Optional[np.ndarray] = None
        self.localized_target_position: Optional[np.ndarray] = None
        self.buffer_detection_target_position: Optional[np.ndarray] = None
        self.navigation_goal_world: Optional[np.ndarray] = None
        self.navigation_goal_raw_world: Optional[np.ndarray] = None
        self.navigation_goal_local_cell: Optional[Tuple[int, int]] = None
        self.navigation_goal_full_cell: Optional[Tuple[int, int]] = None
        self.navigation_raw_target_full_cell: Optional[Tuple[int, int]] = None
        self.navigation_goal_adjusted: bool = False
        self.navigation_goal_tier: Optional[str] = None
        self.navigation_effective_stop_radius_m: Optional[float] = None
        self.navigation_may_terminal_stop: bool = False
        self.navigation_goal_selection: Dict[str, Any] = {}
        self.navigation_staging_replan_count: int = 0
        self.navigation_staging_observation_pending: bool = False
        self.navigation_fallback_force_stop: bool = False
        self.navigation_fallback_force_stop_reason: Optional[str] = None
        self.navigation_phase3_tolerance_hold: bool = False
        self.navigation_localizer_suggested_goal: Optional[np.ndarray] = None
        self.buffer_direct_nav_to_confirmed_target: bool = False
        self.last_confirm_result: Optional[Dict[str, Any]] = None
        self.last_localization_result: Optional[Dict[str, Any]] = None
        self.phase1_confirmation_evidence: List[Dict[str, Any]] = []
        self.buffer_start_step: int = -1
        self.buffer_frame_counter: int = 0
        self.buffer_phase: int = 1
        self.buffer_goal_update_counter: int = 0
        self.buffer_current_goal_world: Optional[np.ndarray] = None
        self.buffer_gaussmi_signaled: bool = False
        self.buffer_gaussmi_signaled_step: int = -1
        self.buffer_gaussmi_signaled_frame: int = -1
        self.buffer_standoff_sweep_active: bool = False
        self.buffer_standoff_sweep_just_completed: bool = False
        self.buffer_standoff_sweep_remaining_steps: int = 0
        self.buffer_standoff_sweep_total_steps: int = 0
        self.buffer_standoff_sweep_direction: str = "turn_left"
        self.localization_signal_frame_counter: int = -1
        self.localization_stage: str = "idle"
        self.localization_frame_counter: int = 0
        self.localization_trace: List[Dict[str, Any]] = []
        self.navigation_complete: bool = False
        self.phase3_navigation_best_distance: float = float("inf")
        self.phase3_navigation_stuck_frames: int = 0
        self.phase3_navigation_stuck_tolerance_active: bool = False
        self.pending_closed_loop_pose = None
        self.pending_closed_loop_score: Optional[float] = None
        self.cluster_recovery_used_in_chain: bool = False
        self.cluster_recovery_center: Optional[np.ndarray] = None
        self.cluster_recovery_center_full_cell: Optional[Tuple[int, int]] = None
        self.cluster_recovery_original_indices: Optional[np.ndarray] = None
        self.cluster_recovery_base_pose: Optional[np.ndarray] = None
        self.cluster_recovery_trigger: Dict[str, Any] = {}
        self.cluster_recovery_active_signature: Optional[Dict[str, Any]] = None
        self.cluster_recovery_route_cells: List[Tuple[int, int]] = []
        self.cluster_recovery_route_segments: List[List[Tuple[int, int]]] = []
        self.cluster_recovery_waypoint_cells: List[Tuple[int, int]] = []
        self.cluster_recovery_waypoints_world: List[np.ndarray] = []
        self.cluster_recovery_waypoint_index: int = 0
        self.cluster_recovery_start_step: int = -1
        self.cluster_recovery_frame_counter: int = 0
        self.cluster_recovery_waypoint_best_distance: float = float("inf")
        self.cluster_recovery_waypoint_stall_frames: int = 0
        self.cluster_recovery_replan_count: int = 0
        self.cluster_recovery_replan_requested_reason: Optional[str] = None
        self.cluster_recovery_approach_active: bool = False
        self.cluster_recovery_approach_goal_cell: Optional[Tuple[int, int]] = None
        self.cluster_recovery_approach_for_replan: bool = False
        self.cluster_recovery_approach_reason: Optional[str] = None
        self.cluster_recovery_candidates: List[Dict[str, Any]] = []
        self.cluster_recovery_last_result: Optional[Dict[str, Any]] = None
        self.cluster_recovery_route_failure_cache: List[Dict[str, Any]] = []
        self.mr_remap_pending: bool = False
        self.mr_remap_sweep_active: bool = False
        self.mr_remap_sweep_just_completed: bool = False
        self.mr_remap_sweep_remaining_steps: int = 0
        self.mr_remap_sweep_total_steps: int = 0
        self.mr_remap_sweep_direction: str = "turn_left"
        self.goal_center_mask: Optional[np.ndarray] = None
        self.goal_all_masks: List[Dict[str, Any]] = []
        self.goal_semantic_mask: Optional[np.ndarray] = None
        self.goal_mask_analysis_key: Optional[Tuple[Any, ...]] = None
        self.goal_mask_analysis_error: Optional[str] = None
        self.goal_segmenter = None
        self.goal_segmenter_error: Optional[str] = None
        self._last_payload: Optional[Dict[str, Any]] = None
        self.visualization_dir: Optional[Path] = None
        self.visualization_episode_key: Optional[Tuple[str, str, str]] = None
        self.visualization_counters: Dict[str, int] = {}
        self._target_image_saved = False
        self._last_match_visual: Optional[Dict[str, Any]] = None
        self.occupancy_adapter = _IEVEFullLocalizerOccupancyAdapter(
            getattr(args, "map_resolution", 5.0)
        )
        self.full_localizer = None
        self.full_localizer_started = False
        self.full_localizer_error: Optional[str] = None
        self.target_confirm_nvsm = None
        self.target_confirm_fem = None
        self.target_confirm_fem_key = None
        self.target_confirm_error: Optional[str] = None
        self.target_confirm_category_detection_dist_m = {
            "bed": 10.0,
            "sofa": 10.0,
            "chair": 10.0,
            "armchair": 10.0,
            "plant": 10.0,
            "toilet": 10.0,
            "tv_monitor": 10.0,
            "tv monitor": 10.0,
            "tv": 10.0,
        }
        self.target_confirm_category_xz_extent_m = {
            "bed": 3.20,
            "sofa": 2.60,
            "chair": 1.50,
            "armchair": 1.50,
            "plant": 1.20,
            "toilet": 1.20,
            "tv_monitor": 1.20,
            "tv monitor": 1.20,
            "tv": 1.20,
        }

        self._extractor = None
        self._matcher = None
        self._matcher_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_episode(
        self,
        info: Optional[Dict[str, Any]] = None,
        slam_mapper=None,
        target_image: Optional[np.ndarray] = None,
        target_category: Optional[str] = None,
    ) -> None:
        self.mode = IEVETargetMode.EXPLORATION
        if slam_mapper is not None:
            self.slam_mapper = slam_mapper
        if target_image is not None:
            self.target_image = self._as_uint8_rgb(target_image)
        elif info is not None and info.get("instance_imagegoal") is not None:
            self.target_image = self._as_uint8_rgb(info.get("instance_imagegoal"))
        if target_category is not None:
            self.target_category = target_category
        elif info is not None:
            self.target_category = info.get("goal_name", self.target_category)

        self.conf_history.clear()
        self.frame_window.clear()
        self.goal_blacklist = []
        self.confirmation_failure_cache = []
        self.confirmation_mask_entries = []
        self.confirmation_blocked_full_mask = np.zeros(
            self.full_map_shape, dtype=np.uint8
        )
        self.targeted_area_samples.clear()
        self.targeted_area_frame_counter = 0
        self.targeted_revert_samples.clear()
        self.targeted_revert_frame_counter = 0
        self.targeted_exploration_complete = False
        self.targeted_phase = 1
        self.targeted_objects = {}
        self.targeted_visited_positions = []
        self.targeted_failed_regions = []
        self.targeted_current_obj_id = None
        self.targeted_current_goal_world = None
        self.targeted_current_goal_map = None
        self.targeted_current_semantic_position = None
        self.targeted_current_standoff_retries = 0
        self.targeted_current_stuck_frames = 0
        self.targeted_current_failed_standoffs = []
        self.targeted_task_should_terminate = False
        self.targeted_phase2_no_reachable_retries = 0
        self.targeted_phase2_no_reachable_cooldown_until = -1
        self.targeted_phase2_force_phase3 = False
        self.targeted_phase3_no_candidate_cooldown_until = -1
        self.strategy_last_desired_phase = 1
        self.strategy_force_stop = False
        self.strategy_force_stop_reason = None
        self.strategy_force_stop_step = -1
        self.strategy_controller.reset()
        self.confirmed_target_position = None
        self.confirmed_target_map_anchor_position = None
        self.confirmed_indices = None
        self.detected_target_position = None
        self.confirmation_detected_target_position = None
        self.localized_target_position = None
        self.buffer_detection_target_position = None
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self.buffer_direct_nav_to_confirmed_target = False
        self.last_confirm_result = None
        self.last_localization_result = None
        self.phase1_confirmation_evidence = []
        self.buffer_start_step = -1
        self.buffer_frame_counter = 0
        self.buffer_phase = 1
        self.buffer_goal_update_counter = 0
        self.buffer_current_goal_world = None
        self.buffer_gaussmi_signaled = False
        self.buffer_gaussmi_signaled_step = -1
        self.buffer_gaussmi_signaled_frame = -1
        self._clear_buffer_standoff_sweep_state()
        self.localization_signal_frame_counter = -1
        self.localization_stage = "idle"
        self.localization_frame_counter = 0
        self.localization_trace = []
        self.navigation_complete = False
        self.phase3_navigation_best_distance = float("inf")
        self.phase3_navigation_stuck_frames = 0
        self.phase3_navigation_stuck_tolerance_active = False
        self.pending_closed_loop_pose = None
        self.pending_closed_loop_score = None
        self._clear_cluster_recovery_state(reset_used=True)
        self.cluster_recovery_route_failure_cache = []
        self.initial_spin_total_steps = (
            self._full_turn_sweep_steps() if self.initial_spin_enabled else 0
        )
        self.initial_spin_remaining_steps = self.initial_spin_total_steps
        self.initial_spin_completed = not self.initial_spin_enabled
        self.mr_remap_pending = False
        self.mr_remap_sweep_active = False
        self.mr_remap_sweep_just_completed = False
        self.mr_remap_sweep_remaining_steps = 0
        self.mr_remap_sweep_total_steps = 0
        self.mr_remap_sweep_direction = "turn_left"
        self.goal_center_mask = None
        self.goal_all_masks = []
        self.goal_semantic_mask = None
        self.goal_mask_analysis_key = None
        self.goal_mask_analysis_error = None
        self._last_payload = None
        self.visualization_dir = None
        self.visualization_episode_key = None
        self.visualization_counters = {}
        self._target_image_saved = False
        self._last_match_visual = None
        self.occupancy_adapter = _IEVEFullLocalizerOccupancyAdapter(
            getattr(self.args, "map_resolution", 5.0)
        )
        self.full_localizer = None
        self.full_localizer_started = False
        self.full_localizer_error = None
        self.target_confirm_nvsm = None
        self.target_confirm_fem = None
        self.target_confirm_fem_key = None
        self.target_confirm_error = None
        self._ensure_visualization_episode(info or {})

    def update_slam_mapper(self, slam_mapper) -> None:
        self.slam_mapper = slam_mapper

    def ensure_visualization_episode(self, info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        self._ensure_visualization_episode(info or {})
        return str(self.visualization_dir) if self.visualization_dir is not None else None

    @property
    def requires_force_3dgs(self) -> bool:
        return (
            self._initial_spin_active()
            or self.buffer_standoff_sweep_active
            or self.mr_remap_sweep_active
            or self.mode in (
                IEVETargetMode.BUFFER,
                IEVETargetMode.LOCALIZATION,
                IEVETargetMode.CLUSTER_RECOVERY,
            )
        )

    @property
    def has_confirmed_target(self) -> bool:
        return self.confirmed_target_position is not None

    # ------------------------------------------------------------------
    # Stage visualization
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_name(value: Any, default: str = "unknown") -> str:
        text = str(value if value is not None else default).strip()
        if not text:
            text = default
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
        return text.strip("_") or default

    @staticmethod
    def _scene_hash_name(scene_name: Any) -> str:
        scene_text = str(scene_name or "").strip()
        if not scene_text:
            return "unknown_hash"
        scene_path = Path(scene_text)
        hash_name = scene_path.name
        for suffix in (".basis.glb", ".glb"):
            if hash_name.endswith(suffix):
                hash_name = hash_name[: -len(suffix)]
                break
        if hash_name.endswith(".basis"):
            hash_name = hash_name[: -len(".basis")]
        if not hash_name:
            parent_name = scene_path.parent.name
            hash_name = parent_name.rsplit("-", 1)[-1] if "-" in parent_name else parent_name
        return MinAVConfirmationLocalizationManager._safe_name(
            hash_name, "unknown_hash"
        )

    @staticmethod
    def _habitat_episode_index(info: Dict[str, Any]) -> Optional[int]:
        if not info:
            return None
        habitat_episode_id = info.get("habitat_episode_id")
        if habitat_episode_id is None:
            return None
        try:
            return int(habitat_episode_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(k): MinAVConfirmationLocalizationManager._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                MinAVConfirmationLocalizationManager._json_safe(v) for v in value
            ]
        return value

    def _trace(self, event: str, **fields: Any) -> None:
        record = {
            "event": str(event),
            "env": int(self.env_idx),
            "mode": self.mode.value,
        }
        record.update(fields)
        try:
            msg = "[MINAV_TRACE] {}".format(
                json.dumps(self._json_safe(record), ensure_ascii=False, sort_keys=True)
            )
        except Exception as exc:
            msg = "[MINAV_TRACE] event={} env={} serialize_error={}".format(
                event, self.env_idx, exc
            )
        print(msg)
        self.logger.info(msg)

    @staticmethod
    def _result_summary(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}
        keys = (
            "confirmed",
            "localized",
            "mr_reposition",
            "system_not_ready",
            "fail_reason",
            "best_score",
            "best_match_count",
            "target_object_center",
            "candidate_count",
            "rendered_candidate_count",
            "best_view_source",
            "target_map_anchor_world",
            "position_source",
            "method",
        )
        summary = {k: result.get(k) for k in keys if k in result}
        raw = result.get("raw_result")
        if isinstance(raw, dict):
            summary["raw_status"] = raw.get("status")
            summary["raw_state"] = raw.get("state")
            summary["raw_message"] = raw.get("message")
        return summary

    @staticmethod
    def _localization_raw_state(result: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(result, dict):
            return None
        raw = result.get("raw_result")
        state = raw.get("state") if isinstance(raw, dict) else None
        if state is None:
            state = result.get("state")
        if state is None:
            return None
        return str(state).lower()

    @staticmethod
    def _target_semantic_class_from_payload(payload: Dict[str, Any]) -> int:
        raw = payload.get("minav_target_semantic_class")
        if raw is not None:
            try:
                return max(1, int(raw))
            except Exception:
                return 1

        category = (
            payload.get("target_category")
            or payload.get("goal_name")
            or payload.get("task_name")
        )
        semantic_id = minav_semantic_class_from_category(category)
        if semantic_id > 0:
            return semantic_id

        raw = payload.get("goal_cat_id")
        try:
            return max(1, int(raw) + 1)
        except Exception:
            return 1

    def _semantic_mask_for_full_localizer(
        self, payload: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        mask = payload.get("minav_sem_mask")
        if mask is not None:
            sem = np.asarray(mask)
            if sem.ndim == 3:
                sem = sem[..., 0]
            if sem.ndim == 2:
                return sem.astype(np.int32, copy=False)

        mask = payload.get("semantic_target_mask_raw")
        if mask is None:
            return None
        sem = np.asarray(mask)
        if sem.ndim == 3:
            sem = sem[..., 0]
        if sem.ndim != 2:
            return sem
        semantic_id = self._target_semantic_class_from_payload(payload)
        return (sem > 0).astype(np.int32) * int(semantic_id)

    def _ensure_visualization_episode(self, info: Dict[str, Any]) -> None:
        if not self.save_stage_visualizations:
            return
        info = info or {}
        scene_name = (
            info.get("scene_name")
            or info.get("scene_id")
            or info.get("scene")
            or "unknown_scene"
        )
        episode_no = info.get("episode_no", info.get("episode_id", "unknown_episode"))
        habitat_episode_id = info.get("habitat_episode_id")
        goal_object_id = info.get("goal_object_id", info.get("habitat_goal_object_id"))
        task_name = (
            info.get("task_name")
            or info.get("goal_name")
            or info.get("target_category")
            or self.target_category
            or info.get("goal_cat_id")
            or "unknown_task"
        )
        scene_key = self._safe_name(scene_name, "unknown_scene")
        task_key = self._safe_name(task_name, "unknown_task")
        episode_name = str(episode_no)
        if habitat_episode_id is not None:
            episode_name = "{}_hab{}".format(episode_name, habitat_episode_id)
        if goal_object_id is not None:
            episode_name = "{}_obj{}".format(episode_name, goal_object_id)
        episode_key = self._safe_name(episode_name, "unknown_episode")
        key = (scene_key, task_key, episode_key)
        if self.visualization_episode_key == key and self.visualization_dir is not None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_hash_key = self._scene_hash_name(scene_name)
        episode_index = self._habitat_episode_index(info)
        episode_index_key = (
            "{:03d}".format(episode_index)
            if episode_index is not None
            else self._safe_name(episode_no, "unknown_episode")
        )
        folder_name = "habitat_{}_{}_{}".format(
            episode_index_key, timestamp, scene_hash_key
        )
        base_dir = self.results_root / folder_name
        if base_dir.exists():
            base_dir = self.results_root / "{}_env{}".format(folder_name, self.env_idx)
        suffix = 1
        unique_dir = base_dir
        while unique_dir.exists():
            unique_dir = Path("{}_{:02d}".format(str(base_dir), suffix))
            suffix += 1

        try:
            for stage in ("target_confirmation", "NVSM", "PRE", "REF", "2d_grid"):
                (unique_dir / stage).mkdir(parents=True, exist_ok=True)
            self.visualization_dir = unique_dir
            self.visualization_episode_key = key
            self.visualization_counters = {
                "target_confirmation": 0,
                "NVSM": 0,
                "PRE": 0,
                "REF": 0,
                "2d_grid": 0,
            }
            self._target_image_saved = False
            self._write_episode_metadata(info)
            self._save_target_image_once()
        except Exception as exc:
            self.logger.debug("Failed to create MinAV visualization dir: %s", exc)
            self.visualization_dir = None

    def _write_episode_metadata(self, info: Dict[str, Any]) -> None:
        if self.visualization_dir is None:
            return
        metadata = {
            "env_idx": self.env_idx,
            "scene_name": info.get("scene_name") if info else None,
            "episode_no": info.get("episode_no") if info else None,
            "habitat_episode_id": info.get("habitat_episode_id") if info else None,
            "goal_object_id": (
                info.get("goal_object_id", info.get("habitat_goal_object_id"))
                if info
                else None
            ),
            "habitat_episode_dataset_index": (
                info.get("habitat_episode_dataset_index") if info else None
            ),
            "habitat_episode_index_in_1000": (
                self._habitat_episode_index(info) if info else None
            ),
            "scene_hash": (
                self._scene_hash_name(info.get("scene_name")) if info else None
            ),
            "start_position": info.get("start_position") if info else None,
            "start_rotation": info.get("start_rotation") if info else None,
            "geodesic_distance": info.get("geodesic_distance") if info else None,
            "euclidean_distance": info.get("euclidean_distance") if info else None,
            "task_name": (
                info.get("task_name")
                or info.get("goal_name")
                or info.get("target_category")
                or self.target_category
                if info
                else self.target_category
            ),
            "target_category": self.target_category,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            with open(self.visualization_dir / "episode_meta.json", "w") as f:
                json.dump(self._json_safe(metadata), f, indent=2)
        except Exception as exc:
            self.logger.debug("Failed to write MinAV episode metadata: %s", exc)

    def _save_target_image_once(self) -> None:
        if (
            not self.save_stage_visualizations
            or self.visualization_dir is None
            or self._target_image_saved
            or self.target_image is None
        ):
            return
        try:
            import cv2

            cv2.imwrite(
                str(self.visualization_dir / "target_image.png"),
                cv2.cvtColor(self.target_image, cv2.COLOR_RGB2BGR),
            )
            self._target_image_saved = True
        except Exception as exc:
            self.logger.debug("Failed to save target image visualization: %s", exc)

    def _save_stage_best_view(
        self,
        stage: str,
        step: int,
        match: Dict[str, Any],
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self.save_stage_visualizations or image is None:
            return None
        self._ensure_visualization_episode(self._last_payload or {})
        if self.visualization_dir is None:
            return None

        image = self._as_uint8_rgb(image)
        if image is None:
            return None
        stage_dir = self.visualization_dir / stage
        try:
            stage_dir.mkdir(parents=True, exist_ok=True)
            import cv2

            counter = int(self.visualization_counters.get(stage, 0)) + 1
            self.visualization_counters[stage] = counter
            source = self._safe_name(match.get("source", "unknown_source"))
            score = float(match.get("score", 0.0) or 0.0)
            filename_base = "{:04d}_step_{:06d}_score_{:07.2f}_{}".format(
                counter, int(step), score, source
            )
            image_path = stage_dir / "{}.png".format(filename_base)
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            meta = {
                "stage": stage,
                "step": int(step),
                "score": score,
                "match_count": int(match.get("match_count", 0) or 0),
                "source": match.get("source", "unknown_source"),
                "message": match.get("message", ""),
                "image_path": str(image_path),
            }
            if metadata:
                meta.update(metadata)
            with open(stage_dir / "{}.json".format(filename_base), "w") as f:
                json.dump(self._json_safe(meta), f, indent=2)
            return str(image_path)
        except Exception as exc:
            self.logger.debug("Failed to save %s visualization: %s", stage, exc)
            return None

    def _save_stage_from_last_match(
        self,
        stage: str,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not self._last_match_visual:
            return None
        return self._save_stage_best_view(
            stage,
            step,
            self._last_match_visual.get("match", {}),
            self._last_match_visual.get("image"),
            metadata=metadata,
        )

    def _project_confirmation_center_to_grid(
        self,
        center_world_xyz: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = context or self._last_payload or {}
        try:
            center = np.asarray(center_world_xyz, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if center.size < 3 or not np.all(np.isfinite(center[:3])):
            return None

        projection = self._buffer_project_world_to_local(center[:3], context)
        if projection is None or not projection.get("ok", False):
            return None
        return projection

    def _confirmation_overlay_center(
        self,
        result: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], str]:
        point = self._as_world_point(result.get("target_cluster_center"))
        if point is not None:
            return point, "target_cluster_center"
        return None, "none"

    def _target_confirmation_map_memory_overlay(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        result = self.last_confirm_result
        if not result:
            return None
        center, center_source = self._confirmation_overlay_center(result)
        if center is None:
            return None
        projection = self._project_confirmation_center_to_grid(center, context)
        if projection is None:
            return None
        return {
            "kind": "target_confirmation",
            "center_source": center_source,
            "center_world_xyz": np.asarray(center, dtype=np.float32).reshape(-1)[:3],
            "local_cell": projection.get("target_local_cell"),
            "target_local_cell": projection.get("target_local_cell"),
            "full_cell": projection.get("target_full_cell"),
            "score": float(result.get("best_score", 0.0) or 0.0),
            "threshold": float(self.confirm_match_threshold),
            "confirmed": bool(result.get("confirmed", False)),
            "fail_reason": result.get("fail_reason", ""),
            "best_match_count": int(result.get("best_match_count", 0) or 0),
            "step": int((context or self._last_payload or {}).get("step", 0)),
            "clipped_to_local_map": bool(projection.get("clipped_to_local_map", False)),
            "debug": self._debug_map_memory_overlay_extras(context),
        }

    def _debug_enabled(self) -> bool:
        return bool(self.gaussian_projection_debug)

    def _mask_projection_filter_debug_enabled(self) -> bool:
        return bool(
            getattr(self.args, "minav_confirm_mask_projection_filter_debug", 0)
        )

    def _debug_map_memory_overlay_extras(
        self, context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.gaussian_projection_debug_overlay:
            return None
        context = context or self._last_payload or {}
        extras: Dict[str, Any] = {}

        for key, point in (
            ("detected_target", self.detected_target_position),
            ("confirmed_target", self.confirmed_target_position),
            ("confirmed_target_map_anchor", self.confirmed_target_map_anchor_position),
            ("navigation_goal", self.navigation_goal_world),
            ("navigation_raw_target", self.navigation_goal_raw_world),
            ("buffer_goal", self.buffer_current_goal_world),
        ):
            projection = self._project_debug_point_to_grid(point, context)
            if projection is not None:
                extras[key] = projection

        sample_cells = self._debug_confirmed_cluster_sample_cells(context)
        if sample_cells:
            extras["confirmed_cluster_sample_cells"] = sample_cells
        return extras or None

    def _project_debug_point_to_grid(
        self, point_world_xyz: Optional[np.ndarray], context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if point_world_xyz is None:
            return None
        projection = self._project_confirmation_center_to_grid(point_world_xyz, context)
        if projection is None:
            return None
        return {
            "world": np.asarray(point_world_xyz, dtype=np.float32).reshape(-1)[:3],
            "local_cell": projection.get("raw_local_cell"),
            "target_local_cell": projection.get("target_local_cell"),
            "full_cell": projection.get("raw_full_cell"),
            "target_full_cell": projection.get("target_full_cell"),
            "clipped_to_local_map": bool(projection.get("clipped_to_local_map", False)),
        }

    def _debug_confirmed_cluster_sample_cells(
        self, context: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        if self.confirmed_indices is None or self.confirmed_indices.size == 0:
            return []
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            return []
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception:
            return []
        indices = np.asarray(self.confirmed_indices, dtype=np.int64).reshape(-1)
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            return []
        max_points = max(0, int(self.gaussian_projection_debug_sample))
        if max_points <= 0:
            return []
        if valid.size > max_points:
            sample_idx = np.linspace(0, valid.size - 1, max_points).astype(np.int64)
            valid = valid[sample_idx]
        cells: List[Tuple[int, int]] = []
        seen = set()
        for point in xyz_all[valid]:
            projection = self._project_confirmation_center_to_grid(point, context)
            if projection is None:
                continue
            cell = projection.get("raw_local_cell")
            if cell is None:
                continue
            cell_tuple = (int(cell[0]), int(cell[1]))
            if cell_tuple in seen:
                continue
            seen.add(cell_tuple)
            cells.append(cell_tuple)
        return cells

    def _debug_semantic_distance_to_projection(
        self,
        projection: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        empty = {
            "available": False,
            "hit": False,
            "cells": None,
            "meters": None,
            "semantic_value": None,
            "map_frame": None,
        }
        if projection is None:
            return empty
        context = context or self._last_payload or {}
        map_resolution_m = float(
            projection.get("map_resolution_cm", context.get("map_resolution_cm", 5.0))
        ) / 100.0

        semantic_map = context.get("full_semantic_target_map_pred")
        cell = projection.get("target_full_cell") or projection.get("raw_full_cell")
        map_frame = "full"
        if semantic_map is None:
            semantic_map = context.get("semantic_target_map_pred")
            cell = projection.get("target_local_cell") or projection.get("raw_local_cell")
            map_frame = "local"
        if semantic_map is None or cell is None:
            return empty

        sem = np.asarray(semantic_map)
        if sem.ndim == 3:
            sem = sem[..., 0]
        if sem.ndim != 2:
            return empty
        row, col = int(cell[0]), int(cell[1])
        h, w = sem.shape[:2]
        in_bounds = 0 <= row < h and 0 <= col < w
        mask = sem > 0
        if not bool(mask.any()):
            result = dict(empty)
            result.update({"available": True, "map_frame": map_frame})
            return result

        if in_bounds:
            semantic_value = float(sem[row, col])
            hit = bool(mask[row, col])
        else:
            semantic_value = None
            hit = False

        ys, xs = np.where(mask)
        if in_bounds:
            dist_cells = float(np.sqrt(np.min((ys - row) ** 2 + (xs - col) ** 2)))
        else:
            clipped_row = int(np.clip(row, 0, h - 1))
            clipped_col = int(np.clip(col, 0, w - 1))
            dist_cells = float(
                np.sqrt(np.min((ys - clipped_row) ** 2 + (xs - clipped_col) ** 2))
            )
        return {
            "available": True,
            "hit": hit,
            "cells": dist_cells,
            "meters": dist_cells * map_resolution_m,
            "semantic_value": semantic_value,
            "map_frame": map_frame,
            "cell": (row, col),
            "in_bounds": in_bounds,
        }

    def _semantic_projection_reference(
        self,
        projection: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if projection is None:
            return None
        context = context or self._last_payload or {}

        semantic_map = context.get("full_semantic_target_map_pred")
        map_frame = "full"
        cell = projection.get("target_full_cell") or projection.get("raw_full_cell")
        if semantic_map is None:
            semantic_map = context.get("semantic_target_map_pred")
            map_frame = "local"
            cell = projection.get("target_local_cell") or projection.get("raw_local_cell")
        if semantic_map is None or cell is None:
            return None

        sem = np.asarray(semantic_map)
        if sem.ndim == 3:
            sem = sem[..., 0]
        if sem.ndim != 2:
            return None

        try:
            row, col = int(cell[0]), int(cell[1])
        except Exception:
            return None
        h, w = sem.shape[:2]
        mask = sem > 0
        if not bool(mask.any()):
            return None

        in_bounds = 0 <= row < h and 0 <= col < w
        if in_bounds and bool(mask[row, col]):
            nearest_row, nearest_col = row, col
            dist_cells = 0.0
        else:
            query_row = int(np.clip(row, 0, h - 1))
            query_col = int(np.clip(col, 0, w - 1))
            ys, xs = np.where(mask)
            d2 = (ys - query_row) ** 2 + (xs - query_col) ** 2
            idx = int(np.argmin(d2))
            nearest_row, nearest_col = int(ys[idx]), int(xs[idx])
            dist_cells = float(np.sqrt(float(d2[idx])))

        gx1 = int(projection.get("gx1", 0))
        gy1 = int(projection.get("gy1", 0))
        if map_frame == "full":
            full_cell = (int(nearest_row), int(nearest_col))
            local_cell = (int(nearest_row) - gx1, int(nearest_col) - gy1)
        else:
            local_cell = (int(nearest_row), int(nearest_col))
            full_cell = (int(nearest_row) + gx1, int(nearest_col) + gy1)

        map_resolution_m = float(
            projection.get("map_resolution_cm", context.get("map_resolution_cm", 5.0))
        ) / 100.0
        return {
            "available": True,
            "map_frame": map_frame,
            "source_cell": (int(row), int(col)),
            "source_in_bounds": bool(in_bounds),
            "source_hit": bool(in_bounds and mask[row, col]),
            "semantic_cell": (int(nearest_row), int(nearest_col)),
            "full_cell": full_cell,
            "local_cell": local_cell,
            "cells": dist_cells,
            "meters": dist_cells * map_resolution_m,
            "semantic_value": float(sem[nearest_row, nearest_col]),
        }

    def _semantic_anchor_world_from_projection(
        self,
        projection: Optional[Dict[str, Any]],
        reference_world: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        semantic = self._semantic_projection_reference(projection, context)
        if not semantic:
            return None
        local_cell = semantic.get("local_cell")
        if local_cell is None:
            return None
        try:
            world = self._buffer_local_cell_to_world(
                tuple(int(v) for v in local_cell),
                projection,
                np.asarray(reference_world, dtype=np.float32).reshape(-1)[:3],
            )
        except Exception:
            return None
        if world is None:
            return None
        return {
            "world": np.asarray(world, dtype=np.float32).reshape(-1)[:3],
            "semantic": semantic,
        }

    def _target_map_anchor_from_world(
        self,
        point_world_xyz: Optional[np.ndarray],
        context: Optional[Dict[str, Any]],
        *,
        source: str,
        reference_world: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        if point_world_xyz is None:
            return None
        try:
            point = np.asarray(point_world_xyz, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if point.size < 3 or not np.all(np.isfinite(point[:3])):
            return None

        context = dict(context or self._last_payload or {})
        context["suppress_projection_debug"] = True
        projection = self._project_confirmation_center_to_grid(point[:3], context)
        if projection is None:
            return None

        ref = point[:3] if reference_world is None else reference_world
        anchor = self._semantic_anchor_world_from_projection(
            projection,
            np.asarray(ref, dtype=np.float32).reshape(-1)[:3],
            context,
        )
        if not anchor:
            return None

        semantic = anchor.get("semantic") or {}
        max_dist_m = float(
            getattr(self.args, "minav_confirm_semantic_anchor_max_dist_m", 2.0) or 0.0
        )
        semantic_m = semantic.get("meters")
        if (
            max_dist_m > 0.0
            and semantic_m is not None
            and float(semantic_m) > max_dist_m
        ):
            return None

        return {
            "source": source,
            "query_world": point[:3].copy(),
            "query_projection": projection,
            "anchor_world": np.asarray(anchor["world"], dtype=np.float32).reshape(-1)[:3],
            "semantic": semantic,
        }

    def _select_target_map_anchor(
        self,
        target_center: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            center = np.asarray(target_center, dtype=np.float32).reshape(-1)[:3]
        except Exception:
            return None
        if center.size < 3 or not np.all(np.isfinite(center[:3])):
            return None

        candidates: List[Dict[str, Any]] = []
        for source, point in (
            ("target_cluster_center", center),
            ("detected_target_position", self.detected_target_position),
            ("confirmation_detected_target_position", self.confirmation_detected_target_position),
        ):
            candidate = self._target_map_anchor_from_world(
                point,
                context,
                source=source,
                reference_world=center,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None

        def _rank(candidate: Dict[str, Any]) -> Tuple[float, float, int]:
            semantic = candidate.get("semantic") or {}
            semantic_m = semantic.get("meters")
            semantic_score = float(semantic_m) if semantic_m is not None else float("inf")
            anchor_world = candidate.get("anchor_world")
            center_dist = self._world_distance_xz(anchor_world, center)
            source_priority = {
                "target_cluster_center": 0,
                "detected_target_position": 1,
                "confirmation_detected_target_position": 2,
            }.get(str(candidate.get("source")), 9)
            return semantic_score, center_dist, source_priority

        selected = min(candidates, key=_rank)
        selected["candidate_count"] = len(candidates)
        selected["max_dist_m"] = float(
            getattr(self.args, "minav_confirm_semantic_anchor_max_dist_m", 2.0) or 0.0
        )
        return selected

    # ------------------------------------------------------------------
    # Public step API
    # ------------------------------------------------------------------
    def _initial_spin_active(self) -> bool:
        return bool(
            self.initial_spin_enabled
            and not self.initial_spin_completed
            and self.initial_spin_remaining_steps > 0
        )

    def _consume_initial_spin_action(self, context: Dict[str, Any]) -> str:
        step = int(context.get("step", 0))
        if self.initial_spin_remaining_steps == self.initial_spin_total_steps:
            self._trace(
                "initial_spin_start",
                step=step,
                total_steps=self.initial_spin_total_steps,
                turn_angle=float(getattr(self.args, "turn_angle", 30.0) or 30.0),
            )
        self.initial_spin_remaining_steps = max(
            0, int(self.initial_spin_remaining_steps) - 1
        )
        if self.initial_spin_remaining_steps == 0:
            self.initial_spin_completed = True
            self._trace(
                "initial_spin_complete",
                step=step,
                total_steps=self.initial_spin_total_steps,
            )
        else:
            self._trace(
                "initial_spin_turn",
                step=step,
                remaining_steps=self.initial_spin_remaining_steps,
                total_steps=self.initial_spin_total_steps,
            )
        return "turn_left"

    def update_frame(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._last_payload = payload
        step = int(payload.get("step", 0))
        self.occupancy_adapter.observe_pose(payload)
        self._ensure_frame_semantics(payload, step)
        mode_before = self.mode
        self._trace(
            "frame_start",
            step=step,
            target_category=payload.get("target_category") or self.target_category,
            detections=len(
                payload.get("minav_target_instances")
                or payload.get("target_instances")
                or []
            ),
            best_conf=max(
                [
                    float(inst.get("confidence", 0.0) or 0.0)
                    for inst in (
                        payload.get("minav_target_instances")
                        or payload.get("target_instances")
                        or []
                    )
                    if isinstance(inst, dict)
                ]
                or [0.0]
            ),
            confirmed=self.confirmed_target_position is not None,
            localized=self.localized_target_position is not None,
            buffer_phase=self.buffer_phase,
            buffer_frames=self.buffer_frame_counter,
            localization_frames=self.localization_frame_counter,
        )
        if payload.get("target_image") is not None and self.target_image is None:
            self.target_image = self._as_uint8_rgb(payload.get("target_image"))
        if payload.get("target_category") is not None:
            self.target_category = payload.get("target_category")
        self._ensure_visualization_episode(payload)
        self._save_target_image_once()

        self._expire_blacklist(step)
        self._expire_confirmation_failure_cache(step)
        self._expire_confirmation_mask_entries(step)

        if self._initial_spin_active():
            self._trace(
                "initial_spin_active",
                step=step,
                remaining_steps=self.initial_spin_remaining_steps,
                total_steps=self.initial_spin_total_steps,
                contract="minav_initial_spin_updates_mapping_before_navigation",
            )
            status = self.get_status()
            self._trace(
                "frame_end",
                step=step,
                mode_before=mode_before.value,
                mode_after=self.mode.value,
                status={
                    "initial_spin_remaining_steps": self.initial_spin_remaining_steps,
                    "initial_spin_completed": self.initial_spin_completed,
                    "confirmed": status.get("confirmed_target_position") is not None,
                    "localized": status.get("localized_target_position") is not None,
                },
            )
            return status

        if self.strategy_force_stop:
            status = self.get_status()
            self._trace(
                "strategy_force_stop_active",
                step=step,
                reason=self.strategy_force_stop_reason,
                force_stop_step=self.strategy_force_stop_step,
                mode=self.mode.value,
            )
            self._trace(
                "frame_end",
                step=step,
                mode_before=mode_before.value,
                mode_after=self.mode.value,
                status=status,
            )
            return status

        self._apply_fixed_strategy_phase_transition(payload, step)

        if self.mode == IEVETargetMode.EXPLORATION:
            if self.strategy_use_area_growth_trigger:
                self._update_targeted_area_tracker(payload, step, track_completion=True)
            self._record_detection(payload)
            self._trace(
                "exploration_detection",
                step=step,
                conf_window=list(self.conf_history),
                detect_window=self.detect_window,
                ready_to_confirm=self._ready_to_confirm(),
            )
            if self._ready_to_confirm():
                self.mode = IEVETargetMode.TARGET_CONFIRMATION
                self._trace("target_confirmation_start", step=step)
                self._refresh_slam_training_before_confirmation(step, payload)
                result = self.run_target_confirmation(payload)
                self.last_confirm_result = result
                self._record_phase1_confirmation_evidence(step, result)
                self._trace(
                    "target_confirmation_result",
                    step=step,
                    result=self._result_summary(result),
                )
                if result.get("confirmed"):
                    self._enter_buffer_from_confirmation(result, step, payload)
                elif result.get("system_not_ready"):
                    self._trace(
                        "target_confirmation_wait",
                        step=step,
                        reason=result.get("fail_reason"),
                    )
                    self.mode = IEVETargetMode.EXPLORATION
                    self._clear_detection_window()
                else:
                    if result.get("should_blacklist") and result.get(
                        "target_cluster_center"
                    ) is not None:
                        self._add_blacklist(
                            result["target_cluster_center"],
                            step,
                            result.get("fail_reason", "confirmation_failed"),
                        )
                    else:
                        self._record_confirmation_failure_for_blacklist(result, step)
                    self.detected_target_position = None
                    self._clear_detection_window()

            if (
                self.mode == IEVETargetMode.EXPLORATION
                and self._targeted_ready_to_enter(payload)
            ):
                desired_phase = self._desired_strategy_phase(step)
                if not self._targeted_entry_cooldown_active(
                    step, desired_phase, trace=False
                ):
                    self._clear_strategy_entry_target_blacklists(step, desired_phase)
                    self._enter_targeted_exploration(
                        step,
                        reason=(
                            "map_growth_stalled"
                            if self.strategy_use_area_growth_trigger
                            else self.strategy_controller.phase_reason(desired_phase)
                        ),
                        phase=max(2, desired_phase),
                    )

        elif self.mode == IEVETargetMode.TARGET_CONFIRMATION:
            self._trace("target_confirmation_idle_reset", step=step)
            self.mode = IEVETargetMode.EXPLORATION

        elif self.mode == IEVETargetMode.TARGETED_EXPLORATION:
            if self.strategy_use_area_growth_trigger:
                self._update_targeted_area_tracker(payload, step, track_revert=True)
            self._run_targeted_update(payload, step)

        elif self.mode == IEVETargetMode.BUFFER:
            if self.buffer_direct_nav_to_confirmed_target:
                if payload.get("minav_phase_goal_reached"):
                    self._run_direct_buffer_update(payload, step, force_arrived=True)
                else:
                    self._run_direct_buffer_update(payload, step)
            else:
                self._run_mr_remap_buffer_update(payload, step)

        elif self.mode == IEVETargetMode.LOCALIZATION:
            self._run_localization_update(payload, step)

        elif self.mode == IEVETargetMode.CLUSTER_RECOVERY:
            self._run_cluster_recovery_update(payload, step)

        elif self.mode == IEVETargetMode.NAVIGATION:
            self._update_navigation_completion(payload)

        status = self.get_status()
        self._trace(
            "frame_end",
            step=step,
            mode_before=mode_before.value,
            mode_after=self.mode.value,
            status={
                "buffer_phase": status.get("buffer_phase"),
                "buffer_gaussmi_signaled": status.get("buffer_gaussmi_signaled"),
                "localization_stage": status.get("localization_stage"),
                "navigation_complete": status.get("navigation_complete"),
                "confirmed": status.get("confirmed_target_position") is not None,
                "localized": status.get("localized_target_position") is not None,
            },
        )
        return status

    def get_goal_override(self, context: Dict[str, Any]) -> Dict[str, Any]:
        goal_world = None
        goal_source = "EXP_GOAL"
        found_goal = False
        hold_action = False
        action_override = None
        goal_map = None
        path_constraint_mask = None
        goal_region_mask = None
        force_stop = False
        force_stop_reason = None

        goal_dilation_radius = None

        if self.navigation_fallback_force_stop:
            goal_source = "MINAV_NAVIGATION_UNREACHABLE_FALLBACK_STOP"
            force_stop = True
            force_stop_reason = self.navigation_fallback_force_stop_reason
        elif self.strategy_force_stop:
            goal_source = "MINAV_STRATEGY_PHASE3_FALLBACK_STOP"
            force_stop = True
            force_stop_reason = self.strategy_force_stop_reason
        elif self._initial_spin_active():
            action_override = self._consume_initial_spin_action(context)
            goal_source = "MINAV_INITIAL_360_SPIN"
        elif self.mode == IEVETargetMode.BUFFER:
            if self.buffer_direct_nav_to_confirmed_target:
                goal_world, goal_map, goal_source, action_override, hold_action = (
                    self._get_direct_buffer_goal_override(context)
                )
            else:
                goal_world, goal_map, goal_source, action_override, hold_action = (
                    self._get_mr_remap_goal_override(context)
                )
        elif self.mode == IEVETargetMode.TARGETED_EXPLORATION:
            goal_world = self.targeted_current_goal_world
            goal_map = None
            goal_source = "MINAV_TARGETED_EXPLORATION"
            found_goal = False
        elif self.mode == IEVETargetMode.CLUSTER_RECOVERY:
            (
                goal_world,
                goal_map,
                path_constraint_mask,
                hold_action,
            ) = self._get_cluster_recovery_goal_override(context)
            goal_source = (
                "MINAV_CLUSTER_RECOVERY_APPROACH"
                if self.cluster_recovery_approach_active
                else "MINAV_CLUSTER_RECOVERY_ORBIT"
            )
            found_goal = goal_map is not None
        elif self.mode == IEVETargetMode.LOCALIZATION:
            goal_world = (
                self.buffer_current_goal_world
                if self.buffer_current_goal_world is not None
                else self._confirmed_target_goal_anchor()
            )
            goal_source = "MINAV_LOCALIZATION_WAIT"
            hold_action = True
        elif self.mode == IEVETargetMode.NAVIGATION:
            navigation_override = self._get_reachable_navigation_goal_override(
                context
            )
            goal_world = navigation_override.get("goal_world")
            goal_map = navigation_override.get("goal_map")
            goal_region_mask = navigation_override.get("goal_region_mask")
            goal_source = navigation_override.get(
                "goal_source", "LOCALIZED_WORLD_GOAL"
            )
            hold_action = bool(navigation_override.get("hold_action", False))
            action_override = navigation_override.get("action_override")
            goal_dilation_radius = navigation_override.get("goal_dilation_radius")
            if bool(navigation_override.get("force_stop", False)):
                force_stop = True
                force_stop_reason = navigation_override.get(
                    "force_stop_reason", "navigation_unreachable_fallback_stop"
                )
            found_goal = bool(self.navigation_complete)

        if goal_world is not None:
            if goal_map is None and self.mode not in (
                IEVETargetMode.BUFFER,
                IEVETargetMode.CLUSTER_RECOVERY,
            ):
                goal_map = self.world_target_to_goal_map(goal_world, context)
                if (
                    self.mode == IEVETargetMode.TARGETED_EXPLORATION
                    and goal_map is not None
                ):
                    self.targeted_current_goal_map = goal_map
            if goal_map is None:
                hold_action = self.mode in (
                    IEVETargetMode.LOCALIZATION,
                    IEVETargetMode.CLUSTER_RECOVERY,
                )
                found_goal = False
            elif self.mode == IEVETargetMode.NAVIGATION:
                found_goal = bool(self.navigation_complete)
            elif (
                self.mode == IEVETargetMode.BUFFER
                and self.buffer_direct_nav_to_confirmed_target
            ):
                found_goal = True

        self._trace(
            "goal_override",
            step=int(context.get("step", 0)),
            goal_source=goal_source,
            found_goal=bool(found_goal),
            hold_action=bool(hold_action),
            action_override=action_override,
            has_goal_world=goal_world is not None,
            has_goal_map=goal_map is not None,
            navigation_complete=self.navigation_complete,
            navigation_goal_world=self.navigation_goal_world,
            navigation_goal_local_cell=self.navigation_goal_local_cell,
            navigation_goal_adjusted=self.navigation_goal_adjusted,
            force_stop=bool(force_stop),
            force_stop_reason=force_stop_reason,
        )

        return {
            "mode": self.mode.value,
            "goal_world": goal_world,
            "goal_map": goal_map,
            "path_constraint_mask": path_constraint_mask,
            "goal_region_mask": goal_region_mask,
            "found_goal": bool(found_goal),
            "hold_action": bool(hold_action),
            "action_override": action_override,
            "navigation_complete": bool(self.navigation_complete),
            "force_stop": bool(force_stop),
            "force_stop_reason": force_stop_reason,
            "goal_source": goal_source,
            "goal_dilation_radius": goal_dilation_radius,
            "visualization_dir": (
                str(self.visualization_dir) if self.visualization_dir is not None else None
            ),
            "map_memory_overlay": self._target_confirmation_map_memory_overlay(context),
            "confirmed": self.confirmed_target_position is not None,
            "system_not_ready": bool(
                self.last_confirm_result
                and self.last_confirm_result.get("system_not_ready")
            ),
            "debug": {
                "last_confirm": self.last_confirm_result,
                "last_localization": self.last_localization_result,
                "buffer_frame_counter": self.buffer_frame_counter,
                "buffer_phase": self.buffer_phase,
                "buffer_gaussmi_signaled": self.buffer_gaussmi_signaled,
                "buffer_standoff_sweep_active": self.buffer_standoff_sweep_active,
                "buffer_standoff_sweep_just_completed": self.buffer_standoff_sweep_just_completed,
                "buffer_standoff_sweep_remaining_steps": self.buffer_standoff_sweep_remaining_steps,
                "initial_spin_remaining_steps": self.initial_spin_remaining_steps,
                "initial_spin_completed": self.initial_spin_completed,
                "mr_remap_pending": self.mr_remap_pending,
                "mr_remap_sweep_active": self.mr_remap_sweep_active,
                "mr_remap_sweep_remaining_steps": self.mr_remap_sweep_remaining_steps,
                "cluster_recovery_used_in_chain": self.cluster_recovery_used_in_chain,
                "cluster_recovery_waypoint_index": self.cluster_recovery_waypoint_index,
                "cluster_recovery_waypoint_count": len(self.cluster_recovery_waypoint_cells),
                "cluster_recovery_center": self.cluster_recovery_center,
                "targeted_phase": self.targeted_phase,
                "targeted_current_obj_id": self.targeted_current_obj_id,
                "targeted_object_count": len(self.targeted_objects),
                "targeted_failed_regions": len(self.targeted_failed_regions),
                "localization_stage": self.localization_stage,
                "localization_trace": self.localization_trace,
                "navigation_complete": self.navigation_complete,
                "buffer_direct_nav_to_confirmed_target": self.buffer_direct_nav_to_confirmed_target,
                "confirmed_target_map_anchor_position": self.confirmed_target_map_anchor_position,
                "navigation_goal_world": self.navigation_goal_world,
                "navigation_goal_raw_world": self.navigation_goal_raw_world,
                "navigation_goal_local_cell": self.navigation_goal_local_cell,
                "navigation_goal_adjusted": self.navigation_goal_adjusted,
                "navigation_goal_tier": self.navigation_goal_tier,
                "navigation_effective_stop_radius_m": self.navigation_effective_stop_radius_m,
                "navigation_may_terminal_stop": self.navigation_may_terminal_stop,
                "navigation_goal_selection": self.navigation_goal_selection,
                "navigation_staging_replan_count": self.navigation_staging_replan_count,
                "navigation_staging_observation_pending": self.navigation_staging_observation_pending,
                "navigation_fallback_force_stop": self.navigation_fallback_force_stop,
                "navigation_fallback_force_stop_reason": self.navigation_fallback_force_stop_reason,
                "navigation_phase3_tolerance_hold": self.navigation_phase3_tolerance_hold,
                "strategy_force_stop": self.strategy_force_stop,
                "strategy_force_stop_reason": self.strategy_force_stop_reason,
            },
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "buffer_phase": self.buffer_phase,
            "buffer_goal_world": self.buffer_current_goal_world,
            "buffer_direct_nav_to_confirmed_target": self.buffer_direct_nav_to_confirmed_target,
            "buffer_gaussmi_signaled": self.buffer_gaussmi_signaled,
            "buffer_standoff_sweep_active": self.buffer_standoff_sweep_active,
            "buffer_standoff_sweep_just_completed": self.buffer_standoff_sweep_just_completed,
            "buffer_standoff_sweep_remaining_steps": self.buffer_standoff_sweep_remaining_steps,
            "initial_spin_remaining_steps": self.initial_spin_remaining_steps,
            "initial_spin_completed": self.initial_spin_completed,
            "mr_remap_pending": self.mr_remap_pending,
            "mr_remap_sweep_active": self.mr_remap_sweep_active,
            "mr_remap_sweep_remaining_steps": self.mr_remap_sweep_remaining_steps,
            "cluster_recovery_used_in_chain": self.cluster_recovery_used_in_chain,
            "cluster_recovery_waypoint_index": self.cluster_recovery_waypoint_index,
            "cluster_recovery_waypoint_count": len(self.cluster_recovery_waypoint_cells),
            "cluster_recovery_center": self.cluster_recovery_center,
            "cluster_recovery_last_result": self.cluster_recovery_last_result,
            "targeted_phase": self.targeted_phase,
            "targeted_current_obj_id": self.targeted_current_obj_id,
            "targeted_current_goal_world": self.targeted_current_goal_world,
            "targeted_object_count": len(self.targeted_objects),
            "targeted_failed_regions": len(self.targeted_failed_regions),
            "strategy_transition_enabled": self.strategy_transition_enabled,
            "strategy_use_area_growth_trigger": self.strategy_use_area_growth_trigger,
            "strategy_force_stop": self.strategy_force_stop,
            "strategy_force_stop_reason": self.strategy_force_stop_reason,
            "localization_stage": self.localization_stage,
            "navigation_complete": self.navigation_complete,
            "confirmed_target_position": self.confirmed_target_position,
            "confirmed_target_map_anchor_position": self.confirmed_target_map_anchor_position,
            "localized_target_position": self.localized_target_position,
            "navigation_goal_world": self.navigation_goal_world,
            "navigation_goal_raw_world": self.navigation_goal_raw_world,
            "navigation_goal_local_cell": self.navigation_goal_local_cell,
            "navigation_goal_adjusted": self.navigation_goal_adjusted,
            "navigation_goal_tier": self.navigation_goal_tier,
            "navigation_effective_stop_radius_m": self.navigation_effective_stop_radius_m,
            "navigation_may_terminal_stop": self.navigation_may_terminal_stop,
            "navigation_staging_replan_count": self.navigation_staging_replan_count,
            "navigation_staging_observation_pending": self.navigation_staging_observation_pending,
            "navigation_fallback_force_stop": self.navigation_fallback_force_stop,
            "navigation_fallback_force_stop_reason": self.navigation_fallback_force_stop_reason,
            "navigation_phase3_tolerance_hold": self.navigation_phase3_tolerance_hold,
            "last_confirm_result": self.last_confirm_result,
            "last_localization_result": self.last_localization_result,
            "visualization_dir": (
                str(self.visualization_dir) if self.visualization_dir is not None else None
            ),
        }

    def _refresh_slam_training_before_confirmation(
        self, step: int, payload: Dict[str, Any]
    ):
        frame_window = list(self.frame_window)
        if not frame_window:
            self._trace(
                "slam_refresh_skipped",
                step=step,
                stage="target_confirm",
                reason="no_detection_window",
            )
            return None
        return self._refresh_slam_training_focus(
            step,
            source="EXP_CONFIRM",
            stage="target_confirm",
            preferred_window=frame_window[-3:],
            extra_iters_attr="confirmation_pre_refresh_iters",
        )

    def _refresh_slam_training_before_buffer(self, step: int, payload: Dict[str, Any]):
        return self._refresh_slam_training_focus(
            step,
            source="EXP_CONFIRM",
            stage="prebuffer",
            preferred_window=list(self.frame_window)[-3:],
            extra_iters_attr="buffer_pre_refresh_iters",
        )

    def _refresh_slam_training_before_localization(
        self, step: int, source: str = "BUFFER_PHASE2"
    ):
        slam = getattr(self.slam_mapper, "slam", None) if self.slam_mapper is not None else None
        if slam is not None and hasattr(slam, "target_focus_window_size"):
            try:
                slam.target_focus_window_size = max(1, int(self.buffer_phase2_refine_frames))
            except Exception:
                pass
        return self._refresh_slam_training_focus(
            step,
            source=source,
            stage="prelocalization",
            preferred_window=[],
            extra_iters_attr="buffer_pre_refresh_iters",
        )

    def _refresh_slam_training_focus(
        self,
        step: int,
        source: str,
        stage: str,
        preferred_window: List[Dict[str, Any]],
        extra_iters_attr: str,
    ):
        if self.slam_mapper is None or not hasattr(self.slam_mapper, "refresh_training_focus"):
            self._trace(
                "slam_refresh_skipped",
                step=step,
                stage=stage,
                source=source,
                reason="refresh_training_focus_unavailable",
            )
            return None

        slam = getattr(self.slam_mapper, "slam", None)
        extra_iters = None
        if slam is not None:
            extra_iters = int(getattr(slam, extra_iters_attr, 0) or 0)
        if extra_iters is not None and extra_iters <= 0:
            self._trace(
                "slam_refresh_skipped",
                step=step,
                stage=stage,
                source=source,
                reason="iters_disabled",
                iters=extra_iters,
            )
            return None

        preferred_kf_ids = self._preferred_refresh_keyframes(preferred_window)
        try:
            result = self.slam_mapper.refresh_training_focus(
                object_id=None,
                reason="{}_pre_{}_step_{}".format(source, stage, step),
                extra_iters=extra_iters,
                preferred_kf_ids=preferred_kf_ids or None,
                stage="target_confirm" if stage == "target_confirm" and preferred_kf_ids else stage,
            )
        except Exception as exc:
            self._trace(
                "slam_refresh_failed",
                step=step,
                stage=stage,
                source=source,
                error=str(exc),
            )
            return None

        self._trace(
            "slam_refresh",
            step=step,
            stage=stage,
            source=source,
            preferred_kf_ids=preferred_kf_ids,
            iters=extra_iters,
            result=result,
        )
        return result

    def _preferred_refresh_keyframes(
        self, preferred_window: List[Dict[str, Any]]
    ) -> List[int]:
        slam = getattr(self.slam_mapper, "slam", None) if self.slam_mapper is not None else None
        if slam is None:
            return []
        recent_steps = {
            int(item.get("step", -1))
            for item in preferred_window
            if isinstance(item, dict)
        }
        metadata = getattr(slam, "kf_metadata", None)
        preferred: List[int] = []
        if isinstance(metadata, dict) and recent_steps:
            for kf_id, meta in metadata.items():
                try:
                    if int(meta.get("step", -999999)) in recent_steps:
                        preferred.append(int(kf_id))
                except Exception:
                    continue
        if preferred:
            return sorted(set(preferred))
        kf_indices = list(getattr(slam, "kf_indices", []) or [])
        if not kf_indices:
            return []
        return [int(k) for k in kf_indices[-max(1, min(3, len(preferred_window) or 3)):]]

    def _ensure_goal_segmenter(self):
        if self.goal_segmenter is not None:
            return self.goal_segmenter
        if self.goal_segmenter_error is not None:
            return None

        use_grounded_sam = bool(
            getattr(self.args, "minav_use_grounded_sam_segmenter", 0)
        )
        ieve_error = None
        if not use_grounded_sam:
            try:
                self.goal_segmenter = IEVEMinAVSegmenter(self.args)
                self.goal_segmenter_error = None
                self._trace(
                    "goal_segmenter_ready",
                    step=int((self._last_payload or {}).get("step", 0)),
                    backend="ieve_maskrcnn",
                    sem_gpu_id=getattr(self.args, "sem_gpu_id", None),
                    sem_pred_prob_thr=getattr(self.args, "sem_pred_prob_thr", None),
                )
                return self.goal_segmenter
            except Exception as exc:
                ieve_error = str(exc)
                self._trace(
                    "goal_segmenter_unavailable",
                    step=int((self._last_payload or {}).get("step", 0)),
                    backend="ieve_maskrcnn",
                    error=ieve_error,
                )
                self.goal_segmenter_error = "ieve_maskrcnn: {}".format(ieve_error)
                return None

        minav_error = None
        try:
            minav_root = Path(__file__).resolve().parents[1] / "3rdparty" / "minav"
            if str(minav_root) not in sys.path:
                sys.path.insert(0, str(minav_root))
            import importlib.util

            segmenter_path = minav_root / "scripts" / "mapper" / "semantic_segmenter.py"
            spec = importlib.util.spec_from_file_location(
                "ieve_minav_thirdparty_semantic_segmenter",
                str(segmenter_path),
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load semantic_segmenter.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            HAS_SEMANTIC_LIBS = getattr(module, "HAS_SEMANTIC_LIBS", False)
            SemanticSegmenter = getattr(module, "SemanticSegmenter")
            if not HAS_SEMANTIC_LIBS:
                raise RuntimeError("Grounded-SAM semantic libraries unavailable")

            gd_config = getattr(
                self.args,
                "minav_grounding_dino_config",
                str(minav_root / "models" / "GroundingDINO_SwinB_cfg.py"),
            )
            gd_ckpt = getattr(
                self.args,
                "minav_grounding_dino_checkpoint",
                str(minav_root / "models" / "groundingdino_swinb_cogcoor.pth"),
            )
            sam_ckpt = getattr(
                self.args,
                "minav_sam_checkpoint",
                str(minav_root / "models" / "sam_vit_b_01ec64.pth"),
            )
            self.goal_segmenter = SemanticSegmenter(gd_config, gd_ckpt, sam_ckpt)
            self.goal_segmenter_error = None
            self._trace(
                "goal_segmenter_ready",
                step=int((self._last_payload or {}).get("step", 0)),
                backend="minav_grounded_sam",
                grounding_dino_config=gd_config,
                grounding_dino_checkpoint=gd_ckpt,
                sam_checkpoint=sam_ckpt,
            )
            return self.goal_segmenter
        except Exception as exc:
            minav_error = str(exc)
            self._trace(
                "goal_segmenter_unavailable",
                step=int((self._last_payload or {}).get("step", 0)),
                backend="minav_grounded_sam",
                error=minav_error,
            )

        try:
            self.goal_segmenter = IEVEMinAVSegmenter(self.args)
            self.goal_segmenter_error = None
            self._trace(
                "goal_segmenter_ready",
                step=int((self._last_payload or {}).get("step", 0)),
                backend="ieve_maskrcnn",
                minav_fallback_reason=minav_error,
                preferred_grounded_sam=use_grounded_sam,
                sem_gpu_id=getattr(self.args, "sem_gpu_id", None),
                sem_pred_prob_thr=getattr(self.args, "sem_pred_prob_thr", None),
            )
            return self.goal_segmenter
        except Exception as exc:
            ieve_error = str(exc)
            error = "minav_grounded_sam: {}; ieve_maskrcnn: {}".format(
                minav_error, ieve_error
            )
            self.goal_segmenter_error = error
            self._trace(
                "goal_segmenter_unavailable",
                step=int((self._last_payload or {}).get("step", 0)),
                backend="all",
                minav_error=minav_error,
                ieve_error=ieve_error,
            )
            return None

    def _ensure_frame_semantics(self, payload: Dict[str, Any], step: int) -> None:
        """Mirror MinAV runner's per-frame segment_frame() contract."""
        if payload.get("minav_sem_mask") is not None and payload.get("minav_sem_labels") is not None:
            return
        rgb = self._as_uint8_rgb(payload.get("raw_rgb"))
        if rgb is None:
            payload.setdefault("minav_sem_mask", None)
            payload.setdefault("minav_sem_labels", [])
            return

        reused = self._reuse_ieve_frame_semantics(payload, step, rgb.shape[:2])
        if reused:
            return

        segmenter = self._ensure_goal_segmenter()
        if segmenter is None:
            payload.setdefault("minav_sem_mask", None)
            payload.setdefault("minav_sem_labels", [])
            self._trace(
                "frame_semantic_segmentation_skipped",
                step=step,
                reason=self.goal_segmenter_error or "segmenter_unavailable",
            )
            return

        try:
            sem_mask, sem_labels, _ = segmenter.segment_frame(rgb, largest_only=False)
        except TypeError:
            sem_mask, sem_labels, _ = segmenter.segment_frame(rgb)
        except Exception as exc:
            payload.setdefault("minav_sem_mask", None)
            payload.setdefault("minav_sem_labels", [])
            self._trace(
                "frame_semantic_segmentation_failed",
                step=step,
                error=str(exc),
            )
            return

        sem_labels = list(sem_labels or [])
        payload["minav_sem_mask"] = sem_mask
        payload["minav_sem_labels"] = sem_labels
        payload["minav_target_instances"] = self._target_instances_from_sem_labels(
            sem_labels,
            target_category=payload.get("target_category") or self.target_category,
            min_confidence=self.detect_conf_threshold,
        )
        self._trace(
            "frame_semantic_segmentation",
            step=step,
            backend=getattr(segmenter, "__class__", type(segmenter)).__name__,
            sem_shape=list(np.asarray(sem_mask).shape) if sem_mask is not None else None,
            label_count=len(sem_labels),
            target_instance_count=len(payload.get("minav_target_instances") or []),
            target_categories=[
                item.get("category") for item in (payload.get("minav_target_instances") or [])
            ],
        )

    def _reuse_ieve_frame_semantics(
        self,
        payload: Dict[str, Any],
        step: int,
        image_shape: Tuple[int, int],
    ) -> bool:
        height, width = int(image_shape[0]), int(image_shape[1])
        target_category = payload.get("target_category") or self.target_category
        target_instances = list(payload.get("target_instances") or [])
        semantic_target_mask = payload.get("semantic_target_mask_raw")
        labels: List[Dict[str, Any]] = []

        for inst in target_instances:
            if not isinstance(inst, dict):
                continue
            category = (
                inst.get("category")
                or inst.get("category_name")
                or target_category
                or ""
            )
            if not self._category_matches(category, target_category or ""):
                continue
            mask = inst.get("binary_mask")
            if mask is None:
                continue
            mask_np = np.asarray(mask, dtype=bool)
            if mask_np.shape != (height, width):
                continue
            pixel_count = int(inst.get("pixel_count", int(mask_np.sum())) or 0)
            if pixel_count <= 0:
                continue
            labels.append(
                {
                    "category": category,
                    "confidence": float(inst.get("confidence", 1.0) or 0.0),
                    "pixel_count": pixel_count,
                    "bbox": inst.get("bbox") if inst.get("bbox") is not None else self._bbox_from_mask(mask_np),
                    "binary_mask": mask_np.copy(),
                    "source": inst.get("source", "ieve_info_target_instances"),
                }
            )

        if not labels and semantic_target_mask is not None:
            mask_np = np.asarray(semantic_target_mask, dtype=bool)
            if mask_np.shape == (height, width) and int(mask_np.sum()) > 0:
                labels.append(
                    {
                        "category": target_category or "",
                        "confidence": 1.0,
                        "pixel_count": int(mask_np.sum()),
                        "bbox": self._bbox_from_mask(mask_np),
                        "binary_mask": mask_np.copy(),
                        "source": "ieve_info_semantic_target_mask_raw",
                    }
                )

        if not labels:
            return False

        semantic_id = self._target_semantic_class_from_payload(payload)
        sem_mask = np.zeros((height, width), dtype=np.int32)
        for label in labels:
            sem_mask[np.asarray(label["binary_mask"], dtype=bool)] = int(semantic_id)

        payload["minav_sem_mask"] = sem_mask
        payload["minav_sem_labels"] = labels
        payload["minav_target_instances"] = self._target_instances_from_sem_labels(
            labels,
            target_category=target_category,
            min_confidence=self.detect_conf_threshold,
        )
        self._trace(
            "frame_semantic_segmentation_reused_ieve",
            step=step,
            source="info.target_instances/semantic_target_mask_raw",
            sem_shape=[height, width],
            label_count=len(labels),
            target_instance_count=len(payload.get("minav_target_instances") or []),
            target_categories=[
                item.get("category") for item in (payload.get("minav_target_instances") or [])
            ],
        )
        return True

    def _target_instances_from_sem_labels(
        self,
        sem_labels: List[Dict[str, Any]],
        *,
        target_category: Optional[str],
        min_confidence: float,
    ) -> List[Dict[str, Any]]:
        target_instances: List[Dict[str, Any]] = []
        for label in sem_labels or []:
            if not isinstance(label, dict):
                continue
            if not self._category_matches(label.get("category", ""), target_category or ""):
                continue
            confidence = float(label.get("confidence", 0.0) or 0.0)
            if confidence < float(min_confidence):
                continue
            mask = label.get("binary_mask")
            if mask is None:
                continue
            mask_np = np.asarray(mask, dtype=bool)
            if mask_np.ndim != 2:
                continue
            pixel_count = int(label.get("pixel_count", int(mask_np.sum())) or 0)
            if pixel_count <= 0:
                continue
            target_instances.append(
                {
                    "category": label.get("category", ""),
                    "confidence": confidence,
                    "pixel_count": pixel_count,
                    "bbox": label.get("bbox"),
                    "binary_mask": mask_np.copy(),
                    "source": label.get("source", "minav_segmenter"),
                }
            )
        return target_instances

    def _ensure_goal_image_masks(self, payload: Dict[str, Any], step: int) -> None:
        if self.target_image is None:
            self.goal_center_mask = None
            self.goal_all_masks = []
            self.goal_semantic_mask = None
            self.goal_mask_analysis_key = None
            self.goal_mask_analysis_error = "target_image_unavailable"
            return

        key = (
            id(self.target_image),
            tuple(self.target_image.shape),
            str(self.target_category),
        )
        if self.goal_mask_analysis_key == key:
            return

        self.goal_center_mask = None
        self.goal_all_masks = []
        self.goal_semantic_mask = None
        self.goal_mask_analysis_key = key
        self.goal_mask_analysis_error = None

        segmenter = self._ensure_goal_segmenter()
        if segmenter is None:
            self.goal_mask_analysis_error = (
                self.goal_segmenter_error or "goal_segmenter_unavailable"
            )
            self._build_goal_image_center_prior_masks(step, self.goal_mask_analysis_error)
            self._trace(
                "goal_image_mask_analysis_fallback",
                step=step,
                reason=self.goal_mask_analysis_error,
                target_category=self.target_category,
                fallback="center_prior_instance_imagegoal",
                center_mask_pixels=(
                    int(np.sum(self.goal_center_mask))
                    if self.goal_center_mask is not None else 0
                ),
                all_masks_count=len(self.goal_all_masks or []),
            )
            return

        try:
            center_mask = self._analyze_goal_image_center_mask(segmenter, step)
            semantic_mask, all_masks = self._analyze_all_goal_image_masks(segmenter, step)
            if center_mask is None:
                fallback_reason = "goal_image_target_mask_missing"
                self._build_goal_image_center_prior_masks(step, fallback_reason)
                self.goal_mask_analysis_error = fallback_reason
                self._trace(
                    "goal_image_mask_analysis_fallback",
                    step=step,
                    reason=fallback_reason,
                    target_category=self.target_category,
                    fallback="center_prior_instance_imagegoal",
                    center_mask_pixels=(
                        int(np.sum(self.goal_center_mask))
                        if self.goal_center_mask is not None else 0
                    ),
                    all_masks_count=len(self.goal_all_masks or []),
                    segmenter_labels_count=len(all_masks or []),
                )
                return
            if not all_masks:
                fallback_reason = "goal_image_all_masks_empty"
                self._build_goal_image_center_prior_masks(step, fallback_reason)
                self.goal_mask_analysis_error = fallback_reason
                self._trace(
                    "goal_image_mask_analysis_fallback",
                    step=step,
                    reason=fallback_reason,
                    target_category=self.target_category,
                    fallback="center_prior_instance_imagegoal",
                    center_mask_pixels=(
                        int(np.sum(self.goal_center_mask))
                        if self.goal_center_mask is not None else 0
                    ),
                    all_masks_count=len(self.goal_all_masks or []),
                )
                return
            self.goal_center_mask = center_mask
            self.goal_all_masks = list(all_masks or [])
            self.goal_semantic_mask = center_mask
            self._trace(
                "goal_image_mask_analysis",
                step=step,
                target_category=self.target_category,
                center_mask_pixels=(
                    int(np.sum(center_mask)) if center_mask is not None else 0
                ),
                all_masks_count=len(self.goal_all_masks),
                all_mask_categories=[
                    item.get("category") for item in self.goal_all_masks
                ],
                semantic_mask_shape=(
                    list(np.asarray(semantic_mask).shape)
                    if semantic_mask is not None else None
                ),
            )
        except Exception as exc:
            self.goal_center_mask = None
            self.goal_all_masks = []
            self.goal_semantic_mask = None
            self.goal_mask_analysis_error = str(exc)
            self._build_goal_image_center_prior_masks(step, self.goal_mask_analysis_error)
            self._trace(
                "goal_image_mask_analysis_failed_fallback",
                step=step,
                target_category=self.target_category,
                error=str(exc),
                fallback="center_prior_instance_imagegoal",
                center_mask_pixels=(
                    int(np.sum(self.goal_center_mask))
                    if self.goal_center_mask is not None else 0
                ),
                all_masks_count=len(self.goal_all_masks or []),
            )

    def _build_goal_image_center_prior_masks(self, step: int, reason: str) -> None:
        if self.target_image is None:
            return
        height, width = self.target_image.shape[:2]
        if height <= 0 or width <= 0:
            return

        mask = np.zeros((height, width), dtype=bool)
        half_w = max(1, int(round(width * 0.36)))
        half_h = max(1, int(round(height * 0.36)))
        cx = width // 2
        cy = height // 2
        x0 = max(0, cx - half_w)
        x1 = min(width, cx + half_w)
        y0 = max(0, cy - half_h)
        y1 = min(height, cy + half_h)
        mask[y0:y1, x0:x1] = True
        category = str(self.target_category or "target")
        self.goal_center_mask = mask
        self.goal_semantic_mask = mask
        self.goal_all_masks = [
            {
                "mask": mask,
                "category": category,
                "pixel_count": int(np.sum(mask)),
                "source": "center_prior_instance_imagegoal",
            }
        ]
        self._trace(
            "goal_image_center_prior_mask",
            step=step,
            reason=reason,
            target_category=category,
            bbox=[int(x0), int(y0), int(x1), int(y1)],
            pixels=int(np.sum(mask)),
            image_shape=[int(height), int(width)],
        )

    def _analyze_goal_image_center_mask(self, segmenter, step: int) -> Optional[np.ndarray]:
        if self.target_image is None:
            return None
        _, labels, vis_img = segmenter.segment_frame(
            self.target_image, largest_only=False
        )
        if not labels:
            self._trace(
                "goal_image_center_mask_missing",
                step=step,
                reason="no_semantic_objects",
            )
            return None

        semantic_matched = [
            label for label in labels
            if self._category_matches(
                label.get("category", ""), self.target_category or ""
            )
        ]
        if not semantic_matched:
            self._trace(
                "goal_image_center_mask_missing",
                step=step,
                reason="target_category_not_detected",
                target_category=self.target_category,
                detected_categories=[label.get("category", "?") for label in labels],
            )
            return None

        height, width = self.target_image.shape[:2]
        img_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        max_diag = float(np.sqrt(width ** 2 + height ** 2))
        best_label = None
        best_score = -1.0
        for label in semantic_matched:
            bbox = label.get("bbox")
            if not bbox:
                continue
            box_center = np.array(
                [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
                dtype=np.float32,
            )
            dist = float(np.linalg.norm(box_center - img_center))
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            score = area * (1.0 - dist / max(max_diag, 1e-6))
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        center_mask = best_label.get("binary_mask")
        if center_mask is not None:
            center_mask = np.asarray(center_mask).astype(bool)

        if (
            vis_img is not None
            and self.visualization_dir is not None
            and self.save_stage_visualizations
        ):
            try:
                import cv2

                bbox = best_label.get("bbox")
                if bbox:
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    cv2.drawMarker(
                        vis_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2
                    )
                    cv2.putText(
                        vis_img,
                        "Target: {} (semantic matched)".format(
                            best_label.get("category", "?")
                        ),
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                cv2.imwrite(
                    str(self.visualization_dir / "goal_image_analysis_result.png"),
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
                )
            except Exception as exc:
                self.logger.debug("Failed to save goal center mask vis: %s", exc)

        return center_mask

    def _analyze_all_goal_image_masks(self, segmenter, step: int):
        if self.target_image is None:
            return None, []
        semantic_mask, labels, vis_img = segmenter.segment_frame(
            self.target_image, largest_only=False
        )
        if not labels:
            self._trace(
                "goal_image_all_masks_missing",
                step=step,
                reason="no_semantic_objects",
            )
            return None, []

        all_masks = []
        for label in labels:
            bmask = label.get("binary_mask")
            if bmask is None:
                continue
            mask = np.asarray(bmask).astype(bool)
            all_masks.append(
                {
                    "mask": mask,
                    "category": label.get("category", ""),
                    "pixel_count": int(label.get("pixel_count", int(np.sum(mask)))),
                }
            )

        if (
            vis_img is not None
            and self.visualization_dir is not None
            and self.save_stage_visualizations
        ):
            try:
                import cv2

                for label in labels:
                    bbox = label.get("bbox")
                    if not bbox:
                        continue
                    cx = int((bbox[0] + bbox[2]) / 2)
                    cy = int((bbox[1] + bbox[3]) / 2)
                    cv2.drawMarker(
                        vis_img, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2
                    )
                cv2.putText(
                    vis_img,
                    "All objects ({} detected)".format(len(labels)),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(
                    str(self.visualization_dir / "goal_image_all_analysis_result.png"),
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
                )
            except Exception as exc:
                self.logger.debug("Failed to save goal all-mask vis: %s", exc)

        return semantic_mask, all_masks

    def _sync_goal_masks_to_localizer(
        self, localizer, payload: Dict[str, Any], step: int, source: str
    ) -> None:
        if localizer is None:
            return
        self._ensure_goal_image_masks(payload, step)
        localizer.goal_center_mask = self.goal_center_mask
        localizer.goal_all_masks = list(self.goal_all_masks or [])
        localizer.target_semantic_mask = self.goal_semantic_mask
        if hasattr(localizer, "fem") and localizer.fem is not None:
            try:
                fem = localizer.fem
                if hasattr(fem, "refresh_target_features"):
                    fem.refresh_target_features(
                        precomputed_mask=self.goal_semantic_mask,
                        default_feature_mode="uniform",
                    )
                else:
                    fem._init_semantic_mask(precomputed_mask=self.goal_semantic_mask)
            except Exception as exc:
                self.logger.debug("Failed to refresh FEM target masks: %s", exc)
        self._trace(
            "goal_masks_synced_to_localizer",
            step=step,
            source=source,
            center_mask_pixels=(
                int(np.sum(self.goal_center_mask))
                if self.goal_center_mask is not None else 0
            ),
            all_masks_count=len(self.goal_all_masks or []),
            target_semantic_mask_pixels=(
                int(np.sum(self.goal_semantic_mask))
                if self.goal_semantic_mask is not None else 0
            ),
            analysis_error=self.goal_mask_analysis_error,
        )

    def _category_matches(self, detected: str, target: str) -> bool:
        detected_lower = str(detected or "").lower().strip()
        target_lower = str(target or "").lower().strip()
        if not detected_lower or not target_lower:
            return False
        if detected_lower == target_lower:
            return True
        if target_lower in detected_lower or detected_lower in target_lower:
            return True
        aliases = {
            "chair": ["chair", "stool", "seat", "armchair"],
            "couch": ["couch", "sofa"],
            "sofa": ["couch", "sofa"],
            "bed": ["bed"],
            "table": ["table", "desk", "dining table", "coffee table"],
            "plant": ["plant", "potted plant", "houseplant"],
            "toilet": ["toilet"],
            "tv": ["tv", "television", "monitor", "screen", "tv_monitor"],
            "tv_monitor": ["tv", "television", "monitor", "screen", "tv_monitor"],
        }
        for canonical, alias_list in aliases.items():
            if target_lower in alias_list or target_lower == canonical:
                if detected_lower in alias_list or detected_lower == canonical:
                    return True
        return False

    def _record_phase1_confirmation_evidence(
        self, step: int, confirm_result: Dict[str, Any]
    ) -> None:
        if not confirm_result:
            return
        center = confirm_result.get("target_cluster_center")
        if center is None:
            center = self.detected_target_position
        record = self._record_strategy_evidence(
            step=step,
            center=center,
            best_score=float(confirm_result.get("best_score", 0.0) or 0.0),
            source="target_confirmation",
            confirmed=bool(confirm_result.get("confirmed")),
            best_match_count=int(confirm_result.get("best_match_count", 0) or 0),
            fail_reason=confirm_result.get("fail_reason", ""),
            all_cluster_scores=list(confirm_result.get("all_cluster_scores") or []),
        )
        if record is None:
            return
        self._trace(
            "phase1_confirmation_evidence",
            step=step,
            confirmed=record["confirmed"],
            center=record["center"],
            best_score=record["best_score"],
            evidence_count=len(self.phase1_confirmation_evidence),
        )

    def _record_strategy_evidence(
        self,
        *,
        step: int,
        center: Optional[np.ndarray],
        best_score: float,
        source: str,
        confirmed: bool = False,
        best_match_count: int = 0,
        fail_reason: str = "",
        all_cluster_scores: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if center is None:
            return None
        center_np = np.asarray(center, dtype=np.float32).reshape(-1)
        if center_np.size < 3 or np.any(~np.isfinite(center_np[:3])):
            return None
        score = float(best_score or 0.0)
        match_count = int(best_match_count or 0)
        all_scores = list(all_cluster_scores or [])
        record = None
        for existing in self.phase1_confirmation_evidence:
            existing_center = existing.get("center")
            if existing_center is None:
                continue
            if self._world_distance_xz(existing_center, center_np[:3]) <= 1.5:
                record = existing
                break

        created = record is None
        if created:
            record = {
                "step": int(step),
                "last_update_step": int(step),
                "confirmed": bool(confirmed),
                "center": center_np[:3].copy(),
                "best_score": score,
                "best_match_count": match_count,
                "fail_reason": fail_reason,
                "all_cluster_scores": all_scores,
                "sources": [str(source)],
            }
            self.phase1_confirmation_evidence.append(record)
            self.phase1_confirmation_evidence = self.phase1_confirmation_evidence[-32:]
        else:
            record["last_update_step"] = int(step)
            record["confirmed"] = bool(record.get("confirmed")) or bool(confirmed)
            if score >= float(record.get("best_score", 0.0) or 0.0):
                record["best_score"] = score
                record["center"] = center_np[:3].copy()
                record["fail_reason"] = fail_reason
            record["best_match_count"] = max(
                int(record.get("best_match_count", 0) or 0),
                match_count,
            )
            if all_scores:
                record["all_cluster_scores"] = list(
                    (record.get("all_cluster_scores") or []) + all_scores
                )[-16:]
            sources = list(record.get("sources") or [])
            if str(source) not in sources:
                sources.append(str(source))
            record["sources"] = sources

        self._trace(
            "strategy_evidence_recorded",
            step=step,
            source=source,
            created=created,
            confirmed=record.get("confirmed"),
            center=record.get("center"),
            best_score=record.get("best_score"),
            best_match_count=record.get("best_match_count"),
            evidence_count=len(self.phase1_confirmation_evidence),
            sources=record.get("sources"),
        )
        return record

    # ------------------------------------------------------------------
    # Targeted Exploration (MinAV StrategyTransition semantics)
    # ------------------------------------------------------------------
    def _desired_strategy_phase(self, step: int) -> int:
        desired = self.strategy_controller.desired_phase(
            step, current_phase=self.targeted_phase
        )
        if self.targeted_phase2_force_phase3 and desired == 2:
            return 3
        return desired

    def _targeted_entry_cooldown_active(
        self, step: int, desired_phase: int, *, trace: bool
    ) -> bool:
        if desired_phase >= 3:
            cooldown_until = self.targeted_phase3_no_candidate_cooldown_until
            event = "targeted_phase3_no_candidate_cooldown_active"
            details = {
                "cooldown_steps": self.phase3_no_candidate_cooldown_steps,
                "evidence_count": len(self.phase1_confirmation_evidence),
            }
        elif desired_phase == 2:
            cooldown_until = self.targeted_phase2_no_reachable_cooldown_until
            event = "targeted_phase2_cooldown_active"
            details = {
                "retry_count": self.targeted_phase2_no_reachable_retries,
                "retry_limit": self.phase2_no_reachable_retry_limit,
            }
        else:
            return False
        active = int(step) < int(cooldown_until)
        if active and trace:
            self._trace(
                event,
                step=step,
                phase=desired_phase,
                cooldown_until=cooldown_until,
                contract="phase1_frontier_explores_during_targeted_retry_cooldown",
                **details,
            )
        return active

    def _clear_strategy_entry_target_blacklists(self, step: int, phase: int) -> None:
        goal_blacklist_count = len(self.goal_blacklist)
        failure_cache_count = len(self.confirmation_failure_cache)
        mask_entry_count = len(self.confirmation_mask_entries)
        blocked_pixels = int(np.sum(self.confirmation_blocked_full_mask > 0))
        self.goal_blacklist.clear()
        self.confirmation_failure_cache.clear()
        self.confirmation_mask_entries.clear()
        self.confirmation_blocked_full_mask = np.zeros(
            self.full_map_shape, dtype=np.uint8
        )
        self._trace(
            "strategy_entry_target_blacklists_cleared",
            step=step,
            phase=phase,
            goal_blacklist_count=goal_blacklist_count,
            confirmation_failure_cache_count=failure_cache_count,
            confirmation_mask_entry_count=mask_entry_count,
            blocked_pixels=blocked_pixels,
            preserved_targeted_failed_regions=len(self.targeted_failed_regions),
            contract="fixed_step_strategy_clears_confirmation_localization_blacklists_but_keeps_exploration_blacklists",
        )

    def _clear_targeted_current_goal(self) -> None:
        self.targeted_current_obj_id = None
        self.targeted_current_goal_world = None
        self.targeted_current_goal_map = None
        self.targeted_current_semantic_position = None
        self.targeted_current_standoff_retries = 0
        self.targeted_current_stuck_frames = 0
        self.targeted_current_failed_standoffs = []

    def _apply_fixed_strategy_phase_transition(
        self, payload: Dict[str, Any], step: int
    ) -> None:
        del payload
        if (
            not self.strategy_transition_enabled
            or self.strategy_use_area_growth_trigger
            or self.navigation_complete
            or self.strategy_force_stop
        ):
            return
        if self.mode not in (
            IEVETargetMode.EXPLORATION,
            IEVETargetMode.TARGETED_EXPLORATION,
        ):
            return

        desired_phase = self._desired_strategy_phase(step)
        if desired_phase <= 1:
            self.strategy_last_desired_phase = desired_phase
            return
        if (
            self.mode == IEVETargetMode.EXPLORATION
            and self._targeted_entry_cooldown_active(
                step, desired_phase, trace=True
            )
        ):
            self.strategy_last_desired_phase = desired_phase
            return

        phase_changed = desired_phase != self.targeted_phase
        entering_targeted = self.mode == IEVETargetMode.EXPLORATION
        if not phase_changed and not entering_targeted:
            self.strategy_last_desired_phase = desired_phase
            return

        if phase_changed or entering_targeted:
            self._clear_targeted_current_goal()
            self._clear_strategy_entry_target_blacklists(step, desired_phase)

        if entering_targeted:
            self._enter_targeted_exploration(
                step,
                reason=self.strategy_controller.phase_reason(desired_phase),
                phase=desired_phase,
            )
        else:
            old_phase = self.targeted_phase
            self.targeted_phase = desired_phase
            self._trace(
                "strategy_phase_transition",
                step=step,
                old_phase=old_phase,
                new_phase=desired_phase,
                reason=self.strategy_controller.phase_reason(desired_phase),
                phase2_start_step=self.strategy_phase2_start_step,
                phase3_start_step=self.strategy_phase3_start_step,
            )
        self.strategy_last_desired_phase = desired_phase

    def _update_targeted_area_tracker(
        self,
        payload: Dict[str, Any],
        step: int,
        *,
        track_completion: bool = False,
        track_revert: bool = False,
    ) -> None:
        explored = payload.get("exp_pred")
        if explored is None:
            return
        explored_np = np.asarray(explored)
        if explored_np.ndim != 2:
            return
        area = int(np.sum(explored_np > 0.5))
        was_complete = bool(self.strategy_controller.area_tracker.exploration_complete)
        self.strategy_controller.area_tracker.update(
            explored_np,
            track_completion=track_completion,
            track_revert=track_revert,
        )

        tracker = self.strategy_controller.area_tracker
        self.targeted_exploration_complete = bool(tracker.exploration_complete)
        self.targeted_area_samples = deque(
            tracker.completion_samples,
            maxlen=tracker.window_samples,
        )
        self.targeted_area_frame_counter = tracker.completion_frame_counter
        self.targeted_revert_samples = deque(
            tracker.revert_samples,
            maxlen=2,
        )
        self.targeted_revert_frame_counter = tracker.revert_frame_counter

        if track_completion and self.targeted_exploration_complete and not was_complete:
            oldest = int(self.targeted_area_samples[0]) if self.targeted_area_samples else area
            newest = int(self.targeted_area_samples[-1]) if self.targeted_area_samples else area
            self._trace(
                "targeted_map_growth_complete",
                step=step,
                oldest_area=oldest,
                newest_area=newest,
                growth=tracker.last_completion_growth,
                threshold=tracker.growth_threshold,
            )

    def _targeted_ready_to_enter(self, payload: Dict[str, Any]) -> bool:
        step = int(payload.get("step", 0))
        return self.strategy_controller.should_enter_targeted(
            step,
            current_phase=self.targeted_phase,
            has_confirmed_target=self.confirmed_target_position is not None,
            has_localized_target=self.localized_target_position is not None,
        )

    def _enter_targeted_exploration(
        self, step: int, reason: str, phase: int = 2
    ) -> None:
        self.mode = IEVETargetMode.TARGETED_EXPLORATION
        self.targeted_phase = int(phase)
        self._clear_targeted_current_goal()
        self.targeted_revert_samples.clear()
        self.targeted_revert_frame_counter = 0
        self._trace(
            "mode_transition",
            step=step,
            from_mode=IEVETargetMode.EXPLORATION.value,
            to_mode=IEVETargetMode.TARGETED_EXPLORATION.value,
            reason=reason,
            phase=self.targeted_phase,
            evidence_count=len(self.phase1_confirmation_evidence),
            failed_regions=len(self.targeted_failed_regions),
        )

    def _targeted_should_revert_to_exploration(self) -> bool:
        should_revert = self.strategy_controller.area_tracker.should_revert()
        tracker = self.strategy_controller.area_tracker
        self.targeted_exploration_complete = bool(tracker.exploration_complete)
        self.targeted_area_samples = deque(
            tracker.completion_samples,
            maxlen=tracker.window_samples,
        )
        self.targeted_revert_samples = deque(
            tracker.revert_samples,
            maxlen=2,
        )
        self.targeted_area_frame_counter = tracker.completion_frame_counter
        self.targeted_revert_frame_counter = tracker.revert_frame_counter
        return bool(should_revert)

    def _run_targeted_update(self, payload: Dict[str, Any], step: int) -> None:
        if (
            self.strategy_use_area_growth_trigger
            and self._targeted_should_revert_to_exploration()
        ):
            self._trace(
                "targeted_revert_to_exploration",
                step=step,
                reason="map_growth_resumed",
                revert_samples=list(self.targeted_revert_samples),
            )
            self.mode = IEVETargetMode.EXPLORATION
            self.targeted_exploration_complete = False
            self.targeted_area_samples.clear()
            self.targeted_area_frame_counter = 0
            self._clear_targeted_current_goal()
            return

        if self.targeted_current_obj_id is not None:
            arrival_threshold = (
                self.phase2_arrival_threshold
                if self.targeted_phase == 2
                else self.buffer_distance_threshold
            )
            distance = self._world_distance_xz(
                payload.get("agent_position"),
                self.targeted_current_goal_world,
            )
            fmm_stop = bool(payload.get("minav_fmm_stop", False))
            been_stuck = bool(
                payload.get("minav_been_stuck", False)
                or payload.get("ieve_frontier_been_stuck", False)
            )
            far_from_standoff = (not np.isfinite(distance)) or (
                distance > arrival_threshold
            )
            stuck_signal = bool(far_from_standoff and (fmm_stop or been_stuck))
            if stuck_signal:
                self.targeted_current_stuck_frames += 1
            else:
                self.targeted_current_stuck_frames = 0
            self._trace(
                "targeted_current_goal",
                step=step,
                obj_id=self.targeted_current_obj_id,
                goal_world=self.targeted_current_goal_world,
                semantic_position=self.targeted_current_semantic_position,
                distance=distance,
                threshold=arrival_threshold,
                buffer_distance_threshold=self.buffer_distance_threshold,
                fmm_stop=fmm_stop,
                been_stuck=been_stuck,
                stuck_signal=stuck_signal,
                stuck_frames=self.targeted_current_stuck_frames,
                stuck_patience=self.phase2_standoff_stuck_patience,
                standoff_retries=self.targeted_current_standoff_retries,
                standoff_retry_limit=self.phase2_standoff_retry_limit,
            )
            if np.isfinite(distance) and distance <= arrival_threshold:
                self._targeted_mark_visited(self.targeted_current_semantic_position)
                self._enter_buffer_from_targeted(payload, step)
                return
            if self.targeted_current_stuck_frames >= self.phase2_standoff_stuck_patience:
                if self._retry_targeted_standoff_or_skip_instance(
                    payload,
                    step,
                    distance=distance,
                    fmm_stop=fmm_stop,
                    been_stuck=been_stuck,
                ):
                    return
            else:
                return

        selected = self._select_targeted_object(payload, step)
        if selected is None:
            if self.targeted_phase == 2:
                self.targeted_phase2_no_reachable_retries += 1
                retry_count = self.targeted_phase2_no_reachable_retries
                retry_limit_reached = (
                    retry_count >= self.phase2_no_reachable_retry_limit
                )
                has_phase3_evidence = bool(self.phase1_confirmation_evidence)
                if retry_limit_reached and has_phase3_evidence:
                    self.targeted_phase2_force_phase3 = True
                    self.targeted_phase2_no_reachable_cooldown_until = -1
                    self._trace(
                        "targeted_phase2_no_reachable_enter_phase3",
                        step=step,
                        retry_count=retry_count,
                        retry_limit=self.phase2_no_reachable_retry_limit,
                        object_count=len(self.targeted_objects),
                        visited_count=len(self.targeted_visited_positions),
                        failed_regions=len(self.targeted_failed_regions),
                        contract="phase2_candidate_pool_exhausted_enter_phase3_early",
                    )
                else:
                    if retry_limit_reached:
                        self.targeted_phase2_no_reachable_retries = 0
                    self.targeted_phase2_no_reachable_cooldown_until = (
                        int(step) + self.phase2_no_reachable_cooldown_steps
                    )
                    self._trace(
                        (
                            "targeted_phase2_no_reachable_continue_phase1"
                            if retry_limit_reached
                            else "targeted_phase2_no_reachable_cooldown"
                        ),
                        step=step,
                        retry_count=retry_count,
                        retry_limit=self.phase2_no_reachable_retry_limit,
                        cooldown_steps=self.phase2_no_reachable_cooldown_steps,
                        cooldown_until=self.targeted_phase2_no_reachable_cooldown_until,
                        object_count=len(self.targeted_objects),
                        visited_count=len(self.targeted_visited_positions),
                        failed_regions=len(self.targeted_failed_regions),
                        evidence_count=len(self.phase1_confirmation_evidence),
                        contract="frontier_explores_before_retrying_phase2_ranking",
                    )
            else:
                self.targeted_phase3_no_candidate_cooldown_until = (
                    int(step) + self.phase3_no_candidate_cooldown_steps
                )
                self._trace(
                    "targeted_phase3_no_candidate_cooldown",
                    step=step,
                    cooldown_steps=self.phase3_no_candidate_cooldown_steps,
                    cooldown_until=self.targeted_phase3_no_candidate_cooldown_until,
                    evidence_count=len(self.phase1_confirmation_evidence),
                    object_count=len(self.targeted_objects),
                    visited_count=len(self.targeted_visited_positions),
                    failed_regions=len(self.targeted_failed_regions),
                    contract="phase1_frontier_explores_before_retrying_phase3",
                )
            self._trace(
                "targeted_no_reachable_object",
                step=step,
                phase=self.targeted_phase,
                object_count=len(self.targeted_objects),
                visited_count=len(self.targeted_visited_positions),
                failed_regions=len(self.targeted_failed_regions),
            )
            self.mode = IEVETargetMode.EXPLORATION
            if self.strategy_use_area_growth_trigger:
                self.targeted_exploration_complete = False
                self.targeted_area_samples.clear()
                self.targeted_area_frame_counter = 0
            return

        if self.targeted_phase == 2 and (
            self.targeted_phase2_no_reachable_retries > 0
            or self.targeted_phase2_no_reachable_cooldown_until >= 0
        ):
            self._trace(
                "targeted_phase2_no_reachable_reset",
                step=step,
                retry_count=self.targeted_phase2_no_reachable_retries,
                cooldown_until=self.targeted_phase2_no_reachable_cooldown_until,
                selected_obj_id=selected.get("obj_id"),
            )
            self.targeted_phase2_no_reachable_retries = 0
            self.targeted_phase2_no_reachable_cooldown_until = -1

        if self.targeted_phase >= 3 and (
            self.targeted_phase3_no_candidate_cooldown_until >= 0
        ):
            self._trace(
                "targeted_phase3_no_candidate_cooldown_reset",
                step=step,
                cooldown_until=self.targeted_phase3_no_candidate_cooldown_until,
                selected_obj_id=selected.get("obj_id"),
            )
            self.targeted_phase3_no_candidate_cooldown_until = -1

        self.targeted_current_obj_id = int(selected["obj_id"])
        self.targeted_current_goal_world = np.asarray(
            selected["nav_goal_3d"], dtype=np.float32
        ).reshape(-1)[:3]
        self.targeted_current_goal_map = selected["goal_map"]
        self.targeted_current_semantic_position = np.asarray(
            selected["position_3d"], dtype=np.float32
        ).reshape(-1)[:3]
        self.targeted_current_standoff_retries = 0
        self.targeted_current_stuck_frames = 0
        self.targeted_current_failed_standoffs = []
        self.targeted_objects[self.targeted_current_obj_id] = selected
        self._trace(
            "targeted_object_selected",
            step=step,
            obj_id=self.targeted_current_obj_id,
            semantic_position=self.targeted_current_semantic_position,
            nav_goal_3d=self.targeted_current_goal_world,
            cost=selected.get("cost"),
            evidence=selected.get("evidence_summary"),
            area=selected.get("area"),
            source=(
                "StrategyTransition_phase3"
                if self.targeted_phase >= 3
                else "StrategyTransition_phase2"
            ),
        )
        if self.targeted_phase >= 3:
            self._switch_to_phase3_best_evidence_navigation(selected, step)
            return

    def _targeted_add_failed_standoff(
        self, goal_world: Optional[np.ndarray]
    ) -> None:
        if goal_world is None:
            return
        try:
            goal = np.asarray(goal_world, dtype=np.float32).reshape(-1)[:3]
        except Exception:
            return
        if goal.size < 3 or not np.all(np.isfinite(goal)):
            return
        for old in self.targeted_current_failed_standoffs:
            if self._world_distance_xz(goal, old) <= 0.10:
                return
        self.targeted_current_failed_standoffs.append(goal.copy())

    def _retry_targeted_standoff_or_skip_instance(
        self,
        payload: Dict[str, Any],
        step: int,
        distance: Optional[float],
        fmm_stop: bool,
        been_stuck: bool,
    ) -> bool:
        old_obj_id = self.targeted_current_obj_id
        old_goal = None
        if self.targeted_current_goal_world is not None:
            old_goal = (
                np.asarray(self.targeted_current_goal_world, dtype=np.float32)
                .reshape(-1)[:3]
                .copy()
            )
        self._targeted_add_failed_standoff(old_goal)

        can_retry = (
            self.targeted_current_semantic_position is not None
            and self.targeted_current_standoff_retries < self.phase2_standoff_retry_limit
        )
        if can_retry:
            prepared = self._prepare_buffer_goal(
                self.targeted_current_semantic_position,
                payload,
                source="MINAV_TARGETED_PHASE2_STANDOFF_RETRY",
                search_radius_m=self.buffer_goal_search_radius,
                allow_adjustment=True,
                commit=False,
                avoid_goal_worlds=self.targeted_current_failed_standoffs,
                avoid_radius_m=self.phase2_standoff_retry_block_radius,
            )
            if prepared is not None:
                self.targeted_current_standoff_retries += 1
                self.targeted_current_stuck_frames = 0
                self.targeted_current_goal_world = np.asarray(
                    prepared["goal_world"], dtype=np.float32
                ).reshape(-1)[:3]
                self.targeted_current_goal_map = prepared["goal_map"]
                if (
                    self.targeted_current_obj_id is not None
                    and self.targeted_current_obj_id in self.targeted_objects
                ):
                    self.targeted_objects[self.targeted_current_obj_id].update(
                        {
                            "nav_goal_3d": self.targeted_current_goal_world.copy(),
                            "anchor_position_3d": self.targeted_current_goal_world.copy(),
                            "goal_map": prepared["goal_map"],
                            "goal_cell": prepared["local_cell"],
                            "cost": prepared.get("cost"),
                        }
                    )
                self._trace(
                    "targeted_standoff_replanned",
                    step=step,
                    obj_id=old_obj_id,
                    retry_index=self.targeted_current_standoff_retries,
                    retry_limit=self.phase2_standoff_retry_limit,
                    old_goal_world=old_goal,
                    new_goal_world=self.targeted_current_goal_world,
                    semantic_position=self.targeted_current_semantic_position,
                    distance=distance,
                    fmm_stop=fmm_stop,
                    been_stuck=been_stuck,
                    failed_standoff_count=len(self.targeted_current_failed_standoffs),
                    avoid_radius_m=self.phase2_standoff_retry_block_radius,
                    validation=prepared.get("validation"),
                    cost=prepared.get("cost"),
                )
                return True

        reason = (
            "phase2_standoff_retry_limit"
            if self.targeted_current_standoff_retries >= self.phase2_standoff_retry_limit
            else "phase2_standoff_replan_unavailable"
        )
        self._trace(
            "targeted_instance_standoff_failed",
            step=step,
            obj_id=old_obj_id,
            reason=reason,
            retry_count=self.targeted_current_standoff_retries,
            retry_limit=self.phase2_standoff_retry_limit,
            semantic_position=self.targeted_current_semantic_position,
            last_goal_world=old_goal,
            distance=distance,
            fmm_stop=fmm_stop,
            been_stuck=been_stuck,
            failed_standoff_count=len(self.targeted_current_failed_standoffs),
        )
        self._targeted_register_failed_region(
            self.targeted_current_semantic_position,
            step,
            reason,
        )
        self._clear_targeted_current_goal()
        return False

    def _switch_to_phase3_best_evidence_navigation(
        self, selected: Dict[str, Any], step: int
    ) -> None:
        target = self._as_world_point(selected.get("position_3d"))
        if target is None:
            target = self._as_world_point(selected.get("semantic_center_3d"))
        if target is None:
            self._trace(
                "targeted_phase3_navigation_fallback_skipped",
                step=step,
                reason="missing_target_position",
                selected_obj=selected.get("obj_id"),
            )
            return

        from_mode = self.mode.value
        self.mode = IEVETargetMode.NAVIGATION
        self.navigation_complete = False
        self._reset_phase3_navigation_progress()
        self._reset_reachable_navigation_state()
        self.localized_target_position = target.copy()
        self.navigation_goal_world = None
        self.navigation_goal_raw_world = target.copy()
        self.navigation_goal_local_cell = None
        self.navigation_goal_full_cell = None
        self.navigation_raw_target_full_cell = None
        self.navigation_goal_adjusted = False
        self.buffer_direct_nav_to_confirmed_target = False
        self.buffer_start_step = -1
        self.buffer_frame_counter = 0
        self.buffer_phase = 1
        self.buffer_goal_update_counter = 0
        self.buffer_current_goal_world = None
        self.buffer_detection_target_position = None
        self.buffer_gaussmi_signaled = False
        self.buffer_gaussmi_signaled_step = -1
        self.buffer_gaussmi_signaled_frame = -1
        self._clear_buffer_standoff_sweep_state()
        self.localization_signal_frame_counter = -1
        self.localization_stage = "phase3_best_evidence_navigation"
        self.localization_frame_counter = 0
        self.pending_closed_loop_pose = None
        self.pending_closed_loop_score = None
        self._clear_mr_remap_state()
        self._clear_detection_window()
        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.NAVIGATION.value,
            reason="strategy_phase3_best_failed_candidate_navigation",
            target_position=target,
            selected_obj=selected.get("obj_id"),
            selected_score=selected.get("historical_score"),
            goal_cell=selected.get("goal_cell"),
            final_goal_dilation_radius=int(
                getattr(self.args, "minav_navigation_goal_dilation_radius", 5)
            ),
            expected_stop_radius_m=float(self.strategy_phase3_fallback_stop_radius),
            stop_radius_source="minav_strategy_phase3_fallback_stop_radius",
            contract="phase3_final_navigation_stops_near_best_failed_candidate",
        )

    def _extract_targeted_objects_from_map(
        self, payload: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        semantic_map = payload.get("full_semantic_target_map_pred")
        map_frame = "ieve_full_map"
        if semantic_map is None or not np.any(np.asarray(semantic_map) > 0):
            semantic_map = payload.get("semantic_target_map_pred")
            map_frame = "ieve_local_map"
        if semantic_map is None:
            return {}
        sem = np.asarray(semantic_map)
        if sem.ndim == 3:
            sem = sem[..., 0]
        if sem.ndim != 2:
            return {}
        mask = sem > 0
        if not np.any(mask):
            return {}

        try:
            from scipy import ndimage as ndi
            labeled, num_components = ndi.label(mask.astype(np.uint8))
        except Exception:
            labeled = mask.astype(np.int32)
            num_components = 1

        objects: Dict[int, Dict[str, Any]] = {}
        min_pixels = 20
        for comp_id in range(1, int(num_components) + 1):
            comp_mask = labeled == comp_id
            area = int(np.sum(comp_mask))
            if area < min_pixels:
                continue
            rows, cols = np.where(comp_mask)
            if rows.size == 0:
                continue
            row = int(round(float(np.mean(rows))))
            col = int(round(float(np.mean(cols))))
            if map_frame == "ieve_full_map":
                world = self._full_cell_to_world_from_context(row, col, payload)
            else:
                world = self._local_cell_to_world_from_context(row, col, payload)
            if world is None:
                continue
            objects[int(comp_id)] = {
                "obj_id": int(comp_id),
                "map_frame": map_frame,
                "position_2d": (row, col),
                "position_3d": world,
                "semantic_center_3d": world.copy(),
                "mask_2d": comp_mask,
                "area": area,
                "boundary_2d": {
                    "y_min": int(rows.min()),
                    "y_max": int(rows.max()),
                    "x_min": int(cols.min()),
                    "x_max": int(cols.max()),
                },
            }
        return objects

    def _select_targeted_object(
        self, payload: Dict[str, Any], step: int
    ) -> Optional[Dict[str, Any]]:
        if self.targeted_phase >= 3:
            return self._select_targeted_phase3_evidence_object(payload, step)

        objects = self._extract_targeted_objects_from_map(payload)
        self.targeted_objects = objects
        ranked: List[Dict[str, Any]] = []
        unreachable = 0
        skipped_visited = 0
        skipped_failed = 0
        for obj_id, obj in objects.items():
            pos = obj["position_3d"]
            if self._targeted_is_visited(pos):
                skipped_visited += 1
                continue
            if self._targeted_in_failed_region(pos):
                skipped_failed += 1
                continue
            prepared = self._prepare_buffer_goal(
                pos,
                payload,
                source="MINAV_TARGETED_PHASE2_OBJECT",
                search_radius_m=self.buffer_goal_search_radius,
                allow_adjustment=True,
                commit=False,
            )
            if prepared is None:
                unreachable += 1
                continue
            evidence, evidence_dist = self._targeted_match_evidence(pos)
            evidence_summary = self._targeted_evidence_summary(evidence, evidence_dist)
            validation = prepared.get("validation") or {}
            cost_raw = prepared.get(
                "cost", validation.get("cost", validation.get("distance_cm", 0.0))
            )
            try:
                cost = float(cost_raw or 0.0)
            except Exception:
                cost = 0.0
            candidate = dict(obj)
            candidate.update(
                {
                    "nav_goal_3d": prepared["goal_world"].copy(),
                    "anchor_position_3d": prepared["goal_world"].copy(),
                    "goal_map": prepared["goal_map"],
                    "goal_cell": prepared["local_cell"],
                    "cost": cost,
                    "evidence_summary": evidence_summary,
                }
            )
            ranked.append(candidate)

        ranked.sort(key=self.strategy_controller.phase2_sort_key)
        self._trace(
            "targeted_phase2_ranking",
            step=step,
            ranking_rule="fmm_cost_then_larger_area_then_obj_id",
            object_count=len(objects),
            reachable_count=len(ranked),
            unreachable_count=unreachable,
            skipped_visited=skipped_visited,
            skipped_failed=skipped_failed,
            evidence_count=len(self.phase1_confirmation_evidence),
            top=[
                {
                    "obj_id": item.get("obj_id"),
                    "area": item.get("area"),
                    "cost": item.get("cost"),
                    "evidence": item.get("evidence_summary"),
                    "map_frame": item.get("map_frame"),
                    "goal_cell": item.get("goal_cell"),
                }
                for item in ranked[:8]
            ],
        )
        if ranked:
            return ranked[0]
        return None

    def _select_targeted_phase3_evidence_object(
        self, payload: Dict[str, Any], step: int
    ) -> Optional[Dict[str, Any]]:
        best = None
        visited_candidate_count = 0
        failed_region_candidate_count = 0
        for idx, record in enumerate(self.phase1_confirmation_evidence):
            centers = []
            if record.get("center") is not None:
                centers.append((record.get("center"), float(record.get("best_score", 0.0) or 0.0)))
            for cluster in record.get("all_cluster_scores") or []:
                if not isinstance(cluster, dict):
                    continue
                center = cluster.get("target_cluster_center")
                if center is None:
                    center = cluster.get("center_3d")
                if center is None:
                    continue
                centers.append((center, float(cluster.get("best_score", 0.0) or 0.0)))
            for center, score in centers:
                center_np = np.asarray(center, dtype=np.float32).reshape(-1)
                if center_np.size < 3:
                    continue
                was_visited = self._targeted_is_visited(center_np[:3])
                in_failed_region = self._targeted_in_failed_region(center_np[:3])
                visited_candidate_count += int(was_visited)
                failed_region_candidate_count += int(in_failed_region)
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
                    center_np[:3], context, phase3=True
                )
                if not reachability.get("selected"):
                    continue
                cost = float(reachability.get("fmm_cost_cm", 0.0) or 0.0)
                candidate = {
                    "obj_id": 100000 + int(idx),
                    "position_2d": None,
                    "position_3d": center_np[:3].copy(),
                    "semantic_center_3d": center_np[:3].copy(),
                    "nav_goal_3d": np.asarray(
                        reachability["execution_goal_world"], dtype=np.float32
                    ).copy(),
                    "anchor_position_3d": center_np[:3].copy(),
                    "goal_map": reachability["goal_map"],
                    "goal_cell": reachability["goal_cell"],
                    "full_cell": reachability["full_cell"],
                    "start_cell": reachability["raw_goal"]["start_cell"],
                    "validation": {
                        key: value
                        for key, value in reachability.items()
                        if key not in (
                            "goal_map",
                            "goal_region_mask",
                            "raw_goal",
                        )
                    },
                    "cost": cost,
                    "priority": float(score),
                    "historical_score": float(score),
                    "was_visited": bool(was_visited),
                    "in_failed_region": bool(in_failed_region),
                    "final_goal_dilation_radius": self.navigation_execution_goal_dilation_radius,
                    "goal_tier": reachability.get("goal_tier"),
                    "effective_stop_radius_m": reachability.get(
                        "effective_stop_radius_m"
                    ),
                    "area": 0,
                    "evidence_summary": {
                        "label": "phase3_best_evidence",
                        "tier": 4,
                        "support": 1,
                        "best_score": score,
                        "distance": 0.0,
                    },
                }
                if best is None or self.strategy_controller.phase3_sort_key(
                    candidate
                ) > self.strategy_controller.phase3_sort_key(best):
                    best = candidate
        self._trace(
            "targeted_phase3_evidence_selection",
            step=step,
            selected=best is not None,
            ranking_rule="historical_best_score_then_lower_fmm_cost",
            evidence_count=len(self.phase1_confirmation_evidence),
            visited_candidate_count=visited_candidate_count,
            failed_region_candidate_count=failed_region_candidate_count,
            ignores_phase2_visit_failure_memory=True,
            selected_obj=None if best is None else best.get("obj_id"),
            selected_score=None if best is None else best.get("evidence_summary", {}).get("best_score"),
        )
        return best

    def _targeted_match_evidence(
        self, position_3d: np.ndarray
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        best_record = None
        best_dist = float("inf")
        for record in self.phase1_confirmation_evidence:
            centers = []
            if record.get("center") is not None:
                centers.append(record.get("center"))
            for cluster in record.get("all_cluster_scores") or []:
                center = (
                    cluster.get("target_cluster_center")
                    if isinstance(cluster, dict)
                    else None
                )
                if center is None and isinstance(cluster, dict):
                    center = cluster.get("center_3d")
                if center is not None:
                    centers.append(center)
            for center in centers:
                dist = self._world_distance_xz(position_3d, center)
                if dist <= 1.5 and dist < best_dist:
                    best_record = record
                    best_dist = dist
        return best_record, best_dist

    @staticmethod
    def _targeted_evidence_summary(
        evidence: Optional[Dict[str, Any]], distance: float
    ) -> Dict[str, Any]:
        if evidence is None:
            return {
                "label": "unseen",
                "tier": 1,
                "support": 0,
                "best_score": 0.0,
                "distance": float("inf"),
            }
        confirmed = bool(evidence.get("confirmed"))
        best_score = float(evidence.get("best_score", 0.0) or 0.0)
        all_scores = evidence.get("all_cluster_scores") or []
        support = 1 if confirmed else 0
        if all_scores:
            support = max(support, min(3, len(all_scores)))
            try:
                best_score = max(
                    best_score,
                    max(float(item.get("best_score", 0.0) or 0.0) for item in all_scores),
                )
            except Exception:
                pass
        if confirmed:
            tier = 4
            label = "confirmed"
        elif best_score >= 22.0 * 0.70:
            tier = 3
            label = "near_miss"
        elif best_score >= 22.0 * 0.45:
            tier = 2
            label = "weak"
        else:
            tier = 1
            label = "uncertain"
        return {
            "label": label,
            "tier": tier,
            "support": int(support),
            "best_score": best_score,
            "distance": distance,
        }

    def _targeted_mark_visited(self, position_3d: Optional[np.ndarray]) -> None:
        if position_3d is None:
            return
        pos = np.asarray(position_3d, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            return
        self.targeted_visited_positions.append((float(pos[0]), float(pos[2])))

    def _targeted_is_visited(self, position_3d: np.ndarray) -> bool:
        pos = np.asarray(position_3d, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            return False
        for x, z in self.targeted_visited_positions:
            if (float(pos[0]) - x) ** 2 + (float(pos[2]) - z) ** 2 <= 1.0 ** 2:
                return True
        return False

    def _targeted_register_failed_region(
        self, position_3d: Optional[np.ndarray], step: int, reason: str
    ) -> None:
        if position_3d is None:
            return
        pos = np.asarray(position_3d, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            return
        for entry in self.targeted_failed_regions:
            if self._world_distance_xz(pos, entry.get("center")) <= 1.2:
                return
        self.targeted_failed_regions.append(
            {
                "center": pos[:3].copy(),
                "step": int(step),
                "reason": reason,
            }
        )
        self._trace(
            "targeted_failed_region_registered",
            step=step,
            center=pos[:3],
            reason=reason,
            failed_regions=len(self.targeted_failed_regions),
        )

    def _targeted_in_failed_region(self, position_3d: np.ndarray) -> bool:
        for entry in self.targeted_failed_regions:
            if self._world_distance_xz(position_3d, entry.get("center")) <= 1.2:
                return True
        return False

    def _strategy_phase3_active(self, step: int) -> bool:
        if self.targeted_phase >= 3 or self.targeted_phase2_force_phase3:
            return True
        return self.strategy_controller.phase3_active(
            step, current_phase=self.targeted_phase
        )

    def _strategy_phase3_stop_distances(self) -> Dict[str, Optional[float]]:
        payload = self._last_payload or {}
        agent = payload.get("agent_position")
        return {
            "semantic_distance": self._world_distance_xz(
                agent, self.targeted_current_semantic_position
            ),
            "localized_distance": self._world_distance_xz(
                agent, self.localized_target_position
            ),
            "standoff_distance": self._world_distance_xz(
                agent, self.targeted_current_goal_world
            ),
            "confirmed_distance": self._world_distance_xz(
                agent, self.confirmed_target_position
            ),
            "buffer_goal_distance": self._world_distance_xz(
                agent, self.buffer_current_goal_world
            ),
        }

    def _maybe_activate_strategy_phase3_fallback_stop(
        self, step: int, reason: str
    ) -> bool:
        if self.strategy_force_stop or not self._strategy_phase3_active(step):
            return False
        distances = self._strategy_phase3_stop_distances()
        finite_distances = [
            float(value)
            for value in distances.values()
            if value is not None and np.isfinite(value)
        ]
        if not finite_distances:
            return False
        target_distance_values = [
            distances.get("semantic_distance"),
            distances.get("localized_distance"),
            distances.get("confirmed_distance"),
        ]
        target_distances = [
            float(value)
            for value in target_distance_values
            if value is not None and np.isfinite(value)
        ]
        best_distance = min(target_distances) if target_distances else min(finite_distances)
        radius = float(self.strategy_phase3_fallback_stop_radius)
        if not self.strategy_controller.should_force_stop_phase3(
            step,
            current_phase=self.targeted_phase,
            best_distance=best_distance,
        ):
            self._trace(
                "strategy_phase3_fallback_stop_wait",
                step=step,
                reason=reason,
                best_distance=best_distance,
                radius=radius,
                distances=distances,
            )
            return False

        self.strategy_force_stop = True
        self.strategy_force_stop_reason = (
            "strategy_phase3_arrived_localization_failed: {}".format(reason)
        )
        self.strategy_force_stop_step = int(step)
        self._trace(
            "strategy_phase3_fallback_stop",
            step=step,
            reason=reason,
            force_stop_reason=self.strategy_force_stop_reason,
            best_distance=best_distance,
            radius=radius,
            distances=distances,
            target_semantic_position=self.targeted_current_semantic_position,
            target_goal_world=self.targeted_current_goal_world,
            contract="non_success_stop_after_phase3_localization_failure_within_radius",
        )
        return True

    def _enter_buffer_from_targeted(self, payload: Dict[str, Any], step: int) -> None:
        semantic_pos = self.targeted_current_semantic_position
        phase1_nav = self.targeted_current_goal_world
        if semantic_pos is None and phase1_nav is None:
            self.mode = IEVETargetMode.EXPLORATION
            return
        target_pos = semantic_pos if semantic_pos is not None else phase1_nav
        quick_lock = self._quick_lock_targeted_cluster(payload, target_pos, step)
        locked_indices = None
        best_score = 0.0
        method = "targeted_phase_semantic_object"
        if not quick_lock.get("confirmed"):
            reason = str(quick_lock.get("reason", "targeted_quick_lock_failed"))
            target_arr = np.asarray(target_pos, dtype=np.float32).reshape(-1)[:3]
            self._record_strategy_evidence(
                step=step,
                center=target_arr,
                best_score=0.0,
                source="targeted_phase_quick_lock_failed",
                confirmed=False,
                best_match_count=0,
                fail_reason=reason,
            )
            self._trace(
                "targeted_quick_lock_buffer_aborted",
                step=step,
                obj_id=self.targeted_current_obj_id,
                target_position=target_arr,
                reason=reason,
                selected_dist_m=quick_lock.get("selected_dist_m"),
                max_dist_m=self.targeted_quick_lock_max_dist_m,
                contract="phase2_requires_nearby_gaussian_cluster_before_buffer",
            )
            self._targeted_register_failed_region(
                target_arr,
                step,
                reason=f"targeted_quick_lock_failed:{reason}",
            )
            self._clear_targeted_current_goal()
            self.mode = IEVETargetMode.EXPLORATION
            return

        locked_indices = quick_lock.get("best_cluster_global_indices")
        best_score = float(quick_lock.get("best_score", 1.0) or 1.0)
        method = "targeted_phase_quick_lock"
        cluster_center = quick_lock.get("target_cluster_center")
        if cluster_center is not None:
            target_pos = np.asarray(cluster_center, dtype=np.float32).reshape(-1)[:3]
        result = {
            "target_cluster_center": np.asarray(target_pos, dtype=np.float32).reshape(-1)[:3],
            "best_cluster_global_indices": locked_indices,
            "best_score": best_score,
            "best_match_count": 0,
            "method": method,
        }
        self._record_strategy_evidence(
            step=step,
            center=result["target_cluster_center"],
            best_score=best_score,
            source=method,
            confirmed=bool(quick_lock.get("confirmed")),
            best_match_count=0,
            fail_reason=str(quick_lock.get("reason", "")),
        )
        obj_id = self.targeted_current_obj_id
        self._trace(
            "targeted_switch_to_buffer",
            step=step,
            obj_id=obj_id,
            target_position=target_pos,
            phase1_nav_goal=phase1_nav,
            quick_locked=bool(quick_lock.get("confirmed")),
            locked_gaussians=0
            if locked_indices is None
            else int(np.asarray(locked_indices).reshape(-1).size),
        )
        self._enter_buffer_from_confirmation(
            result,
            step,
            payload,
        )

    def _quick_lock_targeted_cluster(
        self, payload: Dict[str, Any], target_position: Optional[np.ndarray], step: int
    ) -> Dict[str, Any]:
        if target_position is None:
            return {"confirmed": False, "reason": "missing_target_position"}
        target = np.asarray(target_position, dtype=np.float32).reshape(-1)
        if target.size < 3 or np.any(~np.isfinite(target[:3])):
            return {"confirmed": False, "reason": "invalid_target_position"}

        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            self._trace("targeted_quick_lock_skipped", step=step, reason=fail)
            return {"confirmed": False, "reason": fail}

        candidates, candidate_source, candidate_count = self._targeted_quick_lock_candidates(
            gaussians, payload, step
        )
        if not candidates:
            self._trace(
                "targeted_quick_lock_skipped",
                step=step,
                reason=candidate_source,
                candidate_indices=int(candidate_count),
            )
            return {"confirmed": False, "reason": candidate_source}

        ranked = []
        for cluster in candidates:
            reason = self._target_confirm_candidate_reject_reason(
                cluster,
                enforce_support=False,
                enforce_detection=False,
            )
            if reason is not None:
                self._trace(
                    "targeted_quick_lock_cluster_rejected",
                    step=step,
                    reason=reason,
                    size=int(cluster.get("size", 0) or 0),
                    center=cluster.get("center"),
                )
                continue
            center = self._cluster_center(cluster)
            if center.size < 3:
                continue
            dist = float(np.linalg.norm(center[[0, 2]] - target[:3][[0, 2]]))
            ranked.append((dist, -int(cluster.get("size", 0) or 0), cluster))

        if not ranked:
            self._trace(
                "targeted_quick_lock_skipped",
                step=step,
                reason="all_clusters_rejected",
                candidate_clusters=len(candidates),
            )
            return {"confirmed": False, "reason": "all_clusters_rejected"}

        ranked.sort(key=lambda item: (item[0], item[1]))
        dist, _, cluster = ranked[0]
        center = self._cluster_center(cluster)
        if (
            self.targeted_quick_lock_max_dist_m > 0.0
            and dist > self.targeted_quick_lock_max_dist_m
        ):
            self._trace(
                "targeted_quick_lock_rejected_far_cluster",
                step=step,
                selected_dist_m=dist,
                max_dist_m=self.targeted_quick_lock_max_dist_m,
                target_position=target[:3],
                nearest_cluster_center=center,
                candidate_clusters=len(candidates),
                candidate_source=candidate_source,
                reason="nearest_cluster_too_far",
            )
            return {
                "confirmed": False,
                "reason": "nearest_cluster_too_far",
                "selected_dist_m": float(dist),
                "target_cluster_center": center,
            }

        cluster_indices = self._cluster_indices(cluster)
        locked = self._lock_confirmed_cluster(
            cluster_indices,
            step=step,
            stage="targeted_quick_lock",
        )
        if locked <= 0:
            self._trace(
                "targeted_quick_lock_failed",
                step=step,
                reason="lock_failed",
                selected_size=int(cluster_indices.size),
                selected_dist_m=dist,
            )
            return {"confirmed": False, "reason": "lock_failed"}

        self._trace(
            "targeted_quick_lock_confirmed",
            step=step,
            locked_gaussians=int(locked),
            selected_size=int(cluster_indices.size),
            selected_dist_m=dist,
            target_position=target[:3],
            target_cluster_center=center,
            candidate_clusters=len(candidates),
            candidate_source=candidate_source,
        )
        self._save_confirmed_target_semantic_gaussians(
            gaussians,
            cluster_indices,
            step,
            metadata={
                "confirmation_path": "targeted_quick_lock",
                "candidate_source": candidate_source,
                "selected_dist_m": float(dist),
                "target_position": target[:3],
                "target_cluster_center": center,
                "locked_gaussians": int(locked),
            },
        )
        return {
            "confirmed": True,
            "target_cluster_center": center,
            "best_cluster_global_indices": cluster_indices.astype(np.int64),
            "best_score": 1.0,
            "best_match_count": 0,
        }

    def _targeted_quick_lock_candidates(
        self, gaussians, payload: Dict[str, Any], step: int
    ) -> Tuple[List[Dict[str, Any]], str, int]:
        eps = float(getattr(self.args, "minav_confirm_local_cluster_eps", self.cluster_eps))
        min_samples = int(getattr(self.args, "minav_confirm_local_cluster_min_samples", 10))
        if hasattr(gaussians, "cluster_target_gaussians"):
            try:
                try:
                    cluster_result = gaussians.cluster_target_gaussians(
                        threshold=0.4,
                        eps=eps,
                        min_samples=min_samples,
                    )
                except TypeError:
                    cluster_result = gaussians.cluster_target_gaussians(
                        eps=eps,
                        min_samples=min_samples,
                        min_cluster_size=max(1, int(self.cluster_min_size)),
                    )
                if cluster_result is not None:
                    num_clusters = int(cluster_result.get("num_clusters", 0) or 0)
                    target_indices = np.asarray(
                        cluster_result.get("target_indices", []), dtype=np.int64
                    ).reshape(-1)
                    labels = np.asarray(cluster_result.get("labels", []))
                    if num_clusters <= 0:
                        return [], "no_target_clusters", int(target_indices.size)
                    if target_indices.size == labels.size and labels.size > 0:
                        candidates = []
                        for label_id in sorted(set(labels.astype(int).tolist()) - {-1}):
                            cluster_indices = target_indices[labels == label_id]
                            record = self._build_target_confirm_cluster_record(
                                gaussians, cluster_indices
                            )
                            if record is not None:
                                candidates.append(record)
                        if candidates:
                            return candidates, "cluster_target_gaussians", int(target_indices.size)
                    self._trace(
                        "targeted_quick_lock_cluster_result_unusable",
                        step=step,
                        reason="missing_global_indices",
                        num_clusters=num_clusters,
                    )
            except Exception as exc:
                self._trace(
                    "targeted_quick_lock_cluster_result_failed",
                    step=step,
                    error=str(exc),
                )

        indices = self._collect_candidate_indices(gaussians, payload)
        if indices.size == 0:
            return [], "no_candidate_indices", 0
        candidates = self._select_candidate_clusters(
            gaussians, indices, payload, allow_single_fallback=False
        )
        if not candidates:
            return [], "no_candidate_clusters", int(indices.size)
        return candidates, "strict_candidate_clusters", int(indices.size)

    def _local_cell_to_world_from_context(
        self, row: int, col: int, context: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        pose_pred = context.get("pose_pred")
        current_world = context.get("current_world_xyz")
        if current_world is None:
            current_world = context.get("agent_position")
        if pose_pred is None or current_world is None:
            return None
        try:
            pose = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
            current = np.asarray(current_world, dtype=np.float32).reshape(-1)
            if pose.size < 7 or current.size < 3:
                return None
            start_x, start_y, _, gx1, _, gy1, _ = pose[:7]
            try:
                self.occupancy_adapter.update(context)
                map_frame = getattr(self.occupancy_adapter, "map_frame", "ieve_local_map")
                map_row = int(row) + int(gx1) if map_frame == "ieve_full_map" else int(row)
                map_col = int(col) + int(gy1) if map_frame == "ieve_full_map" else int(col)
                world_x, world_z = self.occupancy_adapter.map_to_world(map_col, map_row)
                if np.isfinite(world_x) and np.isfinite(world_z):
                    world = current[:3].copy()
                    world[0] = float(world_x)
                    world[2] = float(world_z)
                    return world.astype(np.float32)
            except Exception as exc:
                self.logger.debug("Local semantic cell adapter inverse failed: %s", exc)
            res_m = float(context.get("map_resolution_cm", 5.0)) / 100.0
            full_row = int(row) + int(gx1)
            full_col = int(col) + int(gy1)
            target_ieve_y = float(full_row) * res_m
            target_ieve_x = float(full_col) * res_m
            world = current[:3].copy()
            world[0] = current[0] - (target_ieve_y - float(start_y))
            world[2] = current[2] - (target_ieve_x - float(start_x))
            return world.astype(np.float32)
        except Exception:
            return None

    def _full_cell_to_world_from_context(
        self, row: int, col: int, context: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        current_world = context.get("current_world_xyz")
        if current_world is None:
            current_world = context.get("agent_position")
        if current_world is None:
            return None
        try:
            current = np.asarray(current_world, dtype=np.float32).reshape(-1)
            if current.size < 3:
                return None
            self.occupancy_adapter.update(context)
            world_x, world_z = self.occupancy_adapter.map_to_world(int(col), int(row))
            if not (np.isfinite(world_x) and np.isfinite(world_z)):
                return None
            world = current[:3].copy()
            world[0] = float(world_x)
            world[2] = float(world_z)
            return world.astype(np.float32)
        except Exception:
            return None

    def _start_full_localizer_for_buffer(
        self, payload: Dict[str, Any], step: int
    ) -> bool:
        if not self.use_full_localizer or self.confirmed_target_position is None:
            return False
        if self.target_image is None:
            self._trace(
                "localization_start_skipped",
                step=step,
                reason="target_image_unavailable",
            )
            return False
        if self.slam_mapper is None or not getattr(self.slam_mapper, "is_running", False):
            self._trace(
                "localization_start_skipped",
                step=step,
                reason="slam_mapper_not_running",
            )
            return False

        self.occupancy_adapter.update(payload)
        localizer = self._ensure_full_localizer(payload)
        if localizer is None:
            self._trace(
                "localization_start_skipped",
                step=step,
                reason=self.full_localizer_error or "full_localizer_unavailable",
            )
            return False
        try:
            with self._maybe_silence_full_localizer():
                localizer.update_mappers(self.slam_mapper, self.occupancy_adapter)
                if self.visualization_dir is not None:
                    localizer.update_results_dir(str(self.visualization_dir))
                self._sync_full_localizer_target_semantic_class(
                    localizer, payload, source="buffer_entry"
                )
                self._sync_goal_masks_to_localizer(
                    localizer, payload, step, source="buffer_entry"
                )
                self._sync_full_localizer_candidate_anchor(
                    localizer, step, source="buffer_entry"
                )
                localizer.start_localization()
                localizer.best_pose = None
                localizer.refined_pose = None
                localizer.rendered_images = []
            self.full_localizer_started = True
            self.localization_stage = "SOIM"
            self._trace(
                "localization_started_in_buffer",
                step=step,
                anchor=self.confirmed_target_position,
                gaussmi_ready=self.buffer_gaussmi_signaled,
            )
            return True
        except Exception as exc:
            self.full_localizer_error = str(exc)
            self._trace(
                "localization_start_failed",
                step=step,
                error=str(exc),
            )
            return False

    def _notify_full_localizer_gaussmi_ready(self, step: int) -> None:
        localizer = self.full_localizer
        if localizer is None:
            return
        try:
            with self._maybe_silence_full_localizer():
                localizer.notify_gaussmi_ready()
                if (
                    self.pending_closed_loop_pose is not None
                    and hasattr(localizer, "inject_closed_loop_pose_for_pre")
                ):
                    localizer.inject_closed_loop_pose_for_pre(
                        self.pending_closed_loop_pose,
                        score_hint=self.pending_closed_loop_score,
                        source="ieve_mr_remap_sweep",
                    )
                    self.pending_closed_loop_pose = None
                    self.pending_closed_loop_score = None
            self._trace(
                "localizer_gaussmi_notified",
                step=step,
                stage=self.localization_stage,
            )
        except Exception as exc:
            self._trace(
                "localizer_gaussmi_notify_failed",
                step=step,
                error=str(exc),
            )

    def _sync_full_localizer_candidate_anchor(
        self, localizer, step: int, source: str
    ) -> None:
        if localizer is None:
            return
        anchor = self.buffer_detection_target_position
        try:
            with self._maybe_silence_full_localizer():
                if anchor is not None and hasattr(localizer, "set_candidate_anchor"):
                    localizer.set_candidate_anchor(
                        np.asarray(anchor, dtype=np.float32).reshape(-1)[:3],
                        radius_m=self.buffer_soim_anchor_radius_m,
                    )
                    self._trace(
                        "soim_candidate_anchor_set",
                        step=step,
                        source=source,
                        anchor=anchor,
                        radius_m=self.buffer_soim_anchor_radius_m,
                        contract="minav_buffer_soim_candidate_anchor",
                    )
                elif hasattr(localizer, "clear_candidate_anchor"):
                    localizer.clear_candidate_anchor()
                    self._trace(
                        "soim_candidate_anchor_cleared",
                        step=step,
                        source=source,
                        reason="missing_buffer_detection_target_position",
                        contract="minav_buffer_soim_candidate_anchor",
                    )
        except Exception as exc:
            self._trace(
                "soim_candidate_anchor_sync_failed",
                step=step,
                source=source,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------
    def run_target_confirmation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = self._new_confirm_result()
        step = int(payload.get("step", 0))
        self._trace(
            "target_confirmation_prepare",
            step=step,
            frame_window=len(self.frame_window),
            conf_window=list(self.conf_history),
            min_gaussians=self.min_gaussians,
            threshold=self.confirm_match_threshold,
        )
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            return self._confirmation_skipped_as_confirmed(
                result,
                fail,
                step=step,
            )

        num_total = int(gaussians.get_xyz.shape[0])
        self._trace(
            "target_confirmation_gaussians",
            step=step,
            total_gaussians=num_total,
            min_gaussians=self.min_gaussians,
        )
        if num_total < self.min_gaussians:
            return self._confirmation_skipped_as_confirmed(
                result,
                "Too few gaussians ({} < {})".format(num_total, self.min_gaussians),
                step=step,
            )

        if self.target_image is None:
            return self._confirmation_skipped_as_confirmed(
                result,
                "Target image unavailable",
                step=step,
            )

        if self.detected_target_position is None:
            return self._confirmation_failed(
                result,
                "No detected target position",
                system_not_ready=False,
                should_blacklist=False,
            )

        clusters = self._assemble_target_confirm_candidates_from_window(
            gaussians,
            step,
        )
        self._trace(
            "target_confirmation_candidates",
            step=step,
            raw_candidate_count=len(clusters),
        )
        if not clusters:
            return self._confirmation_failed(
                result,
                "No target gaussian candidates from recent instance window",
                system_not_ready=False,
                should_blacklist=False,
            )

        clusters = self._filter_and_rank_target_confirm_candidates(clusters)
        self._trace(
            "target_confirmation_candidates_filtered",
            step=step,
            candidate_count=len(clusters),
            top_k=int(getattr(self.args, "minav_confirm_candidate_top_k", 0) or 0),
        )
        if not clusters:
            return self._confirmation_failed(
                result,
                "No target-confirm candidates passed support/geometry filters",
                system_not_ready=False,
                should_blacklist=False,
            )

        candidate_top_k = int(
            getattr(self.args, "minav_confirm_candidate_top_k", 0) or 0
        )
        render_clusters = (
            clusters[:candidate_top_k] if candidate_top_k > 0 else clusters
        )
        all_cluster_scores: List[Dict[str, Any]] = []
        unavailable_messages: List[str] = []
        best_pass_cluster = None
        best_pass_match = None
        best_observed_cluster = None
        best_observed_match = None

        for rank, cluster in enumerate(render_clusters):
            center = self._cluster_center(cluster)
            cluster_indices = self._cluster_indices(cluster)
            match = self._match_candidate_with_nvsm_views(
                center,
                cluster_indices,
                stage="target_confirmation",
                step=step,
                metadata={
                    "candidate_rank": rank,
                    "candidate_count": len(render_clusters),
                    "target_cluster_center": center,
                    "candidate_cluster_size": int(cluster_indices.size),
                    "support_count": int(cluster.get("support_count", 1) or 1),
                    "support_steps": cluster.get("support_steps", []),
                    "candidate_confidence": float(cluster.get("confidence", 0.0) or 0.0),
                    "detection_dist_m": float(cluster.get("detection_dist_m", float("inf"))),
                    "threshold": self.confirm_match_threshold,
                },
            )
            record = {
                "candidate_rank": rank,
                "target_cluster_center": center,
                "candidate_cluster_size": int(cluster_indices.size),
                "support_count": int(cluster.get("support_count", 1) or 1),
                "support_steps": cluster.get("support_steps", []),
                "candidate_confidence": float(cluster.get("confidence", 0.0) or 0.0),
                "detection_dist_m": float(cluster.get("detection_dist_m", float("inf"))),
                "available": bool(match.get("available")),
                "skipped": False,
                "message": match.get("message", ""),
                "best_score": float(match.get("score", 0.0)),
                "best_match_count": int(match.get("match_count", 0)),
                "best_view_source": match.get("source", "unknown"),
                "best_view_index": int(match.get("best_index", -1)),
                "rendered_view_count": int(match.get("rendered_view_count", 0)),
                "all_view_scores": match.get("all_scores", []),
                "all_view_match_counts": match.get("all_match_counts", []),
            }
            if self._debug_enabled():
                projection = self._project_confirmation_center_to_grid(
                    center,
                    payload,
                )
                semantic_distance = self._debug_semantic_distance_to_projection(
                    projection,
                    payload,
                )
                semantic_anchor = self._select_target_map_anchor(center, payload)
                self._trace(
                    "target_confirmation_candidate_geometry",
                    step=step,
                    rank=rank,
                    center=center,
                    size=int(cluster_indices.size),
                    bbox_min=cluster.get("bbox_min"),
                    bbox_max=cluster.get("bbox_max"),
                    bbox_extent=cluster.get("bbox_extent"),
                    xz_extent_m=float(cluster.get("xz_extent_m", 0.0) or 0.0),
                    y_extent_m=float(cluster.get("y_extent_m", 0.0) or 0.0),
                    detection_dist_m=float(
                        cluster.get("detection_dist_m", float("inf"))
                    ),
                    detected_target_position=self.detected_target_position,
                    support_count=int(cluster.get("support_count", 1) or 1),
                    support_steps=cluster.get("support_steps", []),
                    projected_local_cell=None
                    if projection is None
                    else projection.get("raw_local_cell"),
                    projected_full_cell=None
                    if projection is None
                    else projection.get("raw_full_cell"),
                    target_local_cell=None
                    if projection is None
                    else projection.get("target_local_cell"),
                    target_full_cell=None
                    if projection is None
                    else projection.get("target_full_cell"),
                    coordinate_source=None
                    if projection is None
                    else projection.get("coordinate_source"),
                    semantic_distance=semantic_distance,
                    semantic_anchor_source=None
                    if semantic_anchor is None
                    else semantic_anchor.get("source"),
                    semantic_anchor_world=None
                    if semantic_anchor is None
                    else semantic_anchor.get("anchor_world"),
                    semantic_anchor_full_cell=None
                    if semantic_anchor is None
                    else (semantic_anchor.get("semantic") or {}).get("full_cell"),
                    semantic_anchor_local_cell=None
                    if semantic_anchor is None
                    else (semantic_anchor.get("semantic") or {}).get("local_cell"),
                    semantic_anchor_distance_m=None
                    if semantic_anchor is None
                    else (semantic_anchor.get("semantic") or {}).get("meters"),
                )
                self._save_debug_cluster_ply(
                    "candidate_rank_{:02d}".format(rank),
                    gaussians,
                    cluster_indices,
                    step,
                    metadata={
                        "rank": int(rank),
                        "center": center,
                        "score": float(match.get("score", 0.0)),
                        "semantic_distance": semantic_distance,
                    },
                )
            all_cluster_scores.append(record)
            self._trace(
                "target_confirmation_candidate_result",
                step=step,
                rank=rank,
                available=record["available"],
                skipped=record["skipped"],
                score=record["best_score"],
                matches=record["best_match_count"],
                threshold=self.confirm_match_threshold,
                cluster_size=record["candidate_cluster_size"],
                message=record["message"],
                source=record["best_view_source"],
            )
            if not match.get("available"):
                unavailable_messages.append(match.get("message", "unavailable"))
                continue

            observed_rank = (
                float(match.get("score", 0.0)),
                int(match.get("match_count", 0)),
                int(cluster_indices.size),
            )
            current_observed_rank = (
                float(best_observed_match.get("score", 0.0)),
                int(best_observed_match.get("match_count", 0)),
                int(self._cluster_indices(best_observed_cluster).size),
            ) if best_observed_match is not None and best_observed_cluster is not None else None
            if current_observed_rank is None or observed_rank > current_observed_rank:
                best_observed_cluster = cluster
                best_observed_match = match

            if float(match.get("score", 0.0)) >= self.confirm_match_threshold:
                pass_rank = self._target_confirm_pass_rank(cluster, match)
                current_pass_rank = (
                    self._target_confirm_pass_rank(best_pass_cluster, best_pass_match)
                    if best_pass_cluster is not None and best_pass_match is not None
                    else None
                )
                if current_pass_rank is None or pass_rank > current_pass_rank:
                    best_pass_cluster = cluster
                    best_pass_match = match

        if best_observed_match is None or best_observed_cluster is None:
            fallback_cluster = render_clusters[0] if render_clusters else None
            fallback_center = (
                self._cluster_center(fallback_cluster)
                if fallback_cluster is not None
                else None
            )
            fallback_indices = (
                self._cluster_indices(fallback_cluster)
                if fallback_cluster is not None
                else np.array([], dtype=np.int64)
            )
            failed_snapshot_path = self._save_failed_target_confirmation_full_gaussian_snapshot(
                gaussians,
                fallback_indices,
                step,
                reason="no_available_candidate_matches",
                metadata={
                    "target_cluster_center": fallback_center,
                    "best_score": 0.0,
                    "best_match_count": 0,
                    "best_view_source": "none",
                    "candidate_count": len(clusters),
                    "rendered_candidate_count": len(render_clusters),
                    "all_cluster_scores": all_cluster_scores,
                },
            )
            skipped = self._confirmation_skipped_as_confirmed(
                result,
                "; ".join(unavailable_messages)
                or "No available target-confirmation candidate matches",
                step=step,
                target_cluster_center=fallback_center,
            )
            skipped.update(
                {
                    "all_cluster_scores": all_cluster_scores,
                    "candidate_count": len(clusters),
                    "rendered_candidate_count": len(render_clusters),
                    "failed_semantic_full_ply_path": failed_snapshot_path,
                }
            )
            return skipped

        if best_pass_match is None or best_pass_cluster is None:
            match = best_observed_match
            cluster = best_observed_cluster
            center = self._cluster_center(cluster)
            cluster_indices = self._cluster_indices(cluster)
            best_score = float(match.get("score", 0.0))
            failed = self._confirmation_failed(
                result,
                "FEM/LightGlue score {:.2f} < {:.2f}".format(
                    best_score, self.confirm_match_threshold
                ),
                system_not_ready=False,
                should_blacklist=False,
                target_cluster_center=center,
                best_score=best_score,
                best_match_count=int(match.get("match_count", 0)),
            )
            failed.update(
                {
                    "best_view_source": match.get("source", "unknown"),
                    "best_view_index": int(match.get("best_index", -1)),
                    "rendered_view_count": int(match.get("rendered_view_count", 0)),
                    "best_cluster_size": int(cluster_indices.size),
                    "best_cluster_global_indices": cluster_indices.astype(np.int64),
                    "all_view_scores": match.get("all_scores", []),
                    "all_view_match_counts": match.get("all_match_counts", []),
                    "all_cluster_scores": all_cluster_scores,
                    "candidate_count": len(clusters),
                    "rendered_candidate_count": len(render_clusters),
                }
            )
            failed["failed_semantic_full_ply_path"] = (
                self._save_failed_target_confirmation_full_gaussian_snapshot(
                    gaussians,
                    cluster_indices,
                    step,
                    reason="score_below_threshold",
                    metadata={
                        "target_cluster_center": center,
                        "best_score": best_score,
                        "best_match_count": int(match.get("match_count", 0)),
                        "best_view_source": match.get("source", "unknown"),
                        "best_view_index": int(match.get("best_index", -1)),
                        "rendered_view_count": int(
                            match.get("rendered_view_count", 0)
                        ),
                        "threshold": self.confirm_match_threshold,
                        "candidate_count": len(clusters),
                        "rendered_candidate_count": len(render_clusters),
                        "all_cluster_scores": all_cluster_scores,
                    },
                )
            )
            return failed

        match = best_pass_match
        cluster = best_pass_cluster
        center = self._cluster_center(cluster)
        cluster_indices = self._cluster_indices(cluster)
        best_score = float(match.get("score", 0.0))
        semantic_anchor = self._select_target_map_anchor(center, payload)

        reject_reason = self._target_confirm_candidate_reject_reason(
            cluster,
            enforce_support=False,
            enforce_detection=True,
        )
        if reject_reason is not None:
            failed = self._confirmation_failed(
                result,
                "Confirmed candidate rejected before lock: {}".format(reject_reason),
                system_not_ready=False,
                should_blacklist=False,
                target_cluster_center=center,
                best_score=best_score,
                best_match_count=int(match.get("match_count", 0)),
            )
            failed.update(
                {
                    "best_view_source": match.get("source", "unknown"),
                    "best_view_index": int(match.get("best_index", -1)),
                    "rendered_view_count": int(match.get("rendered_view_count", 0)),
                    "best_cluster_size": int(cluster_indices.size),
                    "best_cluster_global_indices": cluster_indices.astype(np.int64),
                    "all_view_scores": match.get("all_scores", []),
                    "all_view_match_counts": match.get("all_match_counts", []),
                    "all_cluster_scores": all_cluster_scores,
                    "candidate_count": len(clusters),
                    "rendered_candidate_count": len(render_clusters),
                }
            )
            failed["failed_semantic_full_ply_path"] = (
                self._save_failed_target_confirmation_full_gaussian_snapshot(
                    gaussians,
                    cluster_indices,
                    step,
                    reason="rejected_before_lock",
                    metadata={
                        "target_cluster_center": center,
                        "best_score": best_score,
                        "best_match_count": int(match.get("match_count", 0)),
                        "best_view_source": match.get("source", "unknown"),
                        "best_view_index": int(match.get("best_index", -1)),
                        "rendered_view_count": int(
                            match.get("rendered_view_count", 0)
                        ),
                        "reject_reason": reject_reason,
                        "candidate_count": len(clusters),
                        "rendered_candidate_count": len(render_clusters),
                        "all_cluster_scores": all_cluster_scores,
                    },
                )
            )
            return failed

        locked = self._lock_confirmed_cluster(
            cluster_indices,
            step=step,
            stage="target_confirmation",
        )
        if locked <= 0:
            if (
                self.cluster_recovery_enabled
                and best_score > self.cluster_recovery_trigger_score
            ):
                result.update(
                    {
                        "confirmed": True,
                        "provisional_lock_failed": True,
                        "target_cluster_center": center,
                        "target_map_anchor_world": None
                        if semantic_anchor is None
                        else semantic_anchor.get("anchor_world"),
                        "target_map_anchor": semantic_anchor,
                        "best_score": best_score,
                        "best_match_count": int(match.get("match_count", 0)),
                        "best_view_source": match.get("source", "unknown"),
                        "best_view_index": int(match.get("best_index", -1)),
                        "rendered_view_count": int(match.get("rendered_view_count", 0)),
                        "best_cluster_size": int(cluster_indices.size),
                        "best_cluster_global_indices": cluster_indices.astype(np.int64),
                        "best_pose": match.get("best_pose"),
                        "all_view_scores": match.get("all_scores", []),
                        "all_view_match_counts": match.get("all_match_counts", []),
                        "all_cluster_scores": all_cluster_scores,
                        "candidate_count": len(clusters),
                        "rendered_candidate_count": len(render_clusters),
                        "fail_reason": "",
                        "system_not_ready": False,
                        "should_blacklist": False,
                        "method": "provisional_high_score_cluster_for_recovery",
                    }
                )
                self._trace(
                    "target_confirmation_provisional_after_lock_failure",
                    step=step,
                    target_cluster_center=center,
                    best_score=best_score,
                    trigger_threshold=self.cluster_recovery_trigger_score,
                    candidate_size=int(cluster_indices.size),
                )
                return result
            failed = self._confirmation_failed(
                result,
                "Failed to lock confirmed gaussian cluster",
                system_not_ready=False,
                should_blacklist=False,
                target_cluster_center=center,
                best_score=best_score,
                best_match_count=int(match.get("match_count", 0)),
            )
            failed.update(
                {
                    "best_view_source": match.get("source", "unknown"),
                    "best_view_index": int(match.get("best_index", -1)),
                    "rendered_view_count": int(match.get("rendered_view_count", 0)),
                    "best_cluster_size": int(cluster_indices.size),
                    "best_cluster_global_indices": cluster_indices.astype(np.int64),
                    "all_view_scores": match.get("all_scores", []),
                    "all_view_match_counts": match.get("all_match_counts", []),
                    "all_cluster_scores": all_cluster_scores,
                    "candidate_count": len(clusters),
                    "rendered_candidate_count": len(render_clusters),
                }
            )
            failed["failed_semantic_full_ply_path"] = (
                self._save_failed_target_confirmation_full_gaussian_snapshot(
                    gaussians,
                    cluster_indices,
                    step,
                    reason="lock_failed",
                    metadata={
                        "target_cluster_center": center,
                        "best_score": best_score,
                        "best_match_count": int(match.get("match_count", 0)),
                        "best_view_source": match.get("source", "unknown"),
                        "best_view_index": int(match.get("best_index", -1)),
                        "rendered_view_count": int(
                            match.get("rendered_view_count", 0)
                        ),
                        "candidate_count": len(clusters),
                        "rendered_candidate_count": len(render_clusters),
                        "all_cluster_scores": all_cluster_scores,
                    },
                )
            )
            return failed

        self._save_debug_cluster_ply(
            "confirmed_cluster",
            gaussians,
            cluster_indices,
            step,
            metadata={
                "center": center,
                "best_score": best_score,
                "best_match_count": int(match.get("match_count", 0)),
                "best_view_source": match.get("source", "unknown"),
            },
        )
        confirmed_semantic_path = self._save_confirmed_target_semantic_gaussians(
            gaussians,
            cluster_indices,
            step,
            metadata={
                "confirmation_path": "strict_nvsm_fem",
                "target_cluster_center": center,
                "target_map_anchor_world": None
                if semantic_anchor is None
                else semantic_anchor.get("anchor_world"),
                "target_map_anchor": semantic_anchor,
                "best_score": best_score,
                "best_match_count": int(match.get("match_count", 0)),
                "best_view_source": match.get("source", "unknown"),
                "best_view_index": int(match.get("best_index", -1)),
                "rendered_view_count": int(match.get("rendered_view_count", 0)),
                "support_count": int(cluster.get("support_count", len(self.frame_window)) or 1),
                "support_steps": cluster.get("support_steps", [
                    int(frame.get("step", -1)) for frame in list(self.frame_window)
                ]),
            },
        )

        result.update(
            {
                "confirmed": True,
                "target_cluster_center": center,
                "target_map_anchor_world": None
                if semantic_anchor is None
                else semantic_anchor.get("anchor_world"),
                "target_map_anchor": semantic_anchor,
                "best_score": best_score,
                "best_match_count": int(match.get("match_count", 0)),
                "best_view_source": match.get("source", "unknown"),
                "best_view_index": int(match.get("best_index", -1)),
                "rendered_view_count": int(match.get("rendered_view_count", 0)),
                "best_cluster_size": int(cluster_indices.size),
                "best_cluster_global_indices": cluster_indices.astype(np.int64),
                "best_pose": match.get("best_pose"),
                "confirmed_semantic_ply_path": confirmed_semantic_path,
                "all_view_scores": match.get("all_scores", []),
                "all_view_match_counts": match.get("all_match_counts", []),
                "all_cluster_scores": all_cluster_scores,
                "candidate_count": len(clusters),
                "rendered_candidate_count": len(render_clusters),
                "support_count": int(cluster.get("support_count", len(self.frame_window)) or 1),
                "support_steps": cluster.get("support_steps", [
                    int(frame.get("step", -1)) for frame in list(self.frame_window)
                ]),
                "fail_reason": "",
                "system_not_ready": False,
                "should_blacklist": False,
                "method": "strict_gaussian_cluster_nvsm_multiview_lightglue_v3",
            }
        )
        self._trace(
            "target_confirmation_confirmed",
            step=step,
            target_cluster_center=center,
            target_map_anchor_world=None
            if semantic_anchor is None
            else semantic_anchor.get("anchor_world"),
            target_map_anchor_source=None
            if semantic_anchor is None
            else semantic_anchor.get("source"),
            target_map_anchor_full_cell=None
            if semantic_anchor is None
            else (semantic_anchor.get("semantic") or {}).get("full_cell"),
            target_map_anchor_distance_m=None
            if semantic_anchor is None
            else (semantic_anchor.get("semantic") or {}).get("meters"),
            locked_gaussians=int(cluster_indices.size),
            best_score=best_score,
            best_match_count=int(match.get("match_count", 0)),
            best_view_source=match.get("source", "unknown"),
        )
        return result

    def run_close_localization(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.use_full_localizer:
            return self._run_full_close_localization(payload)

        result = {
            "localized": False,
            "target_position": None,
            "best_score": 0.0,
            "best_match_count": 0,
            "fail_reason": "",
            "system_not_ready": False,
            "method": "soim_nvsm_fem_pre_ref_mr_v2",
            "stages": [],
        }
        if self.confirmed_target_position is None:
            result.update({"fail_reason": "No confirmed target"})
            return result
        if self.target_image is None:
            result.update(
                {"fail_reason": "Target image unavailable", "system_not_ready": True}
            )
            return result
        current_rgb = self._as_uint8_rgb(payload.get("raw_rgb"))
        if current_rgb is None:
            result.update(
                {"fail_reason": "Current RGB unavailable", "system_not_ready": True}
            )
            return result

        self.localization_trace = []
        self._trace_localization_stage("SOIM", True, "gaussmi_ready")
        self._trace_localization_stage("NVSM", True, "using recent 3DGS/current evidence views")

        match = self._match_with_evidence_views(
            current_rgb,
            stage="NVSM",
            step=int(payload.get("step", 0)),
            metadata={
                "target_position": self.confirmed_target_position,
                "threshold": self.localization_match_threshold,
            },
        )
        if not match.get("available"):
            result.update(
                {
                    "fail_reason": match.get("message", "LightGlue unavailable"),
                    "system_not_ready": True,
                }
            )
            return result

        score = float(match.get("score", 0.0))
        result["best_score"] = score
        result["best_match_count"] = int(match.get("match_count", 0))
        result["best_view_source"] = match.get("source", "unknown")
        if score < self.localization_match_threshold:
            self._trace_localization_stage(
                "FEM",
                False,
                "score {:.2f} < {:.2f}".format(score, self.localization_match_threshold),
            )
            result["stages"] = list(self.localization_trace)
            result["fail_reason"] = "Localization score {:.2f} < {:.2f}".format(
                score, self.localization_match_threshold
            )
            return result
        self._trace_localization_stage(
            "FEM",
            True,
            "score {:.2f}, matches {}".format(score, int(match.get("match_count", 0))),
        )

        target_position = self._confirmed_cluster_navigation_position()
        if target_position is None:
            target_position = self.confirmed_target_position
        if not self.fine_localization_enabled:
            result.update(
                {
                    "localized": True,
                    "target_position": np.asarray(target_position, dtype=np.float32),
                    "fail_reason": "",
                    "method": "soim_nvsm_fem_coarse_v1",
                    "stages": list(self.localization_trace),
                }
            )
            return result
        self._trace_localization_stage("PRE", True, "coarse target pose from locked cluster")
        self._save_stage_from_last_match(
            "PRE",
            int(payload.get("step", 0)),
            metadata={
                "pre_target_position": target_position,
                "threshold": self.localization_match_threshold,
            },
        )

        refined_position = self._select_navigation_position_from_ref_view(
            payload, target_position
        )
        if refined_position is not None:
            target_position = refined_position
            self._trace_localization_stage("REF", True, "target cluster refined from current view")
            ref_message = "target cluster refined from current view"
        else:
            self._trace_localization_stage("REF", True, "fallback to PRE/confirmed cluster")
            ref_message = "fallback to PRE/confirmed cluster"
        self._save_stage_from_last_match(
            "REF",
            int(payload.get("step", 0)),
            metadata={
                "ref_target_position": target_position,
                "ref_message": ref_message,
                "threshold": self.localization_match_threshold,
            },
        )

        closed_loop_threshold = float(
            getattr(self.args, "minav_mr_closed_loop_threshold", self.localization_match_threshold)
        )
        if score >= closed_loop_threshold:
            self._trace_localization_stage(
                "MR", True, "closed-loop skipped: score above threshold"
            )
        else:
            self._trace_localization_stage(
                "MR", True, "fallback accepted: PRE/FEM evidence already valid"
            )

        result.update(
            {
                "localized": True,
                "target_position": np.asarray(target_position, dtype=np.float32),
                "fail_reason": "",
                "stages": list(self.localization_trace),
            }
        )
        return result

    def _run_full_close_localization(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "localized": False,
            "target_position": None,
            "best_score": 0.0,
            "best_match_count": 0,
            "fail_reason": "",
            "system_not_ready": False,
            "method": "full_minav_localization_system",
            "stages": [],
        }
        if self.confirmed_target_position is None:
            result["fail_reason"] = "No confirmed target"
            return result
        if self.target_image is None:
            result.update(
                {"fail_reason": "Target image unavailable", "system_not_ready": True}
            )
            return result
        if self.slam_mapper is None or not getattr(self.slam_mapper, "is_running", False):
            result.update(
                {"fail_reason": "SLAM mapper not running", "system_not_ready": True}
            )
            return result

        current_rgb = self._as_uint8_rgb(payload.get("raw_rgb"))
        if current_rgb is None:
            result.update(
                {"fail_reason": "Current RGB unavailable", "system_not_ready": True}
            )
            return result

        self.occupancy_adapter.update(payload)
        try:
            semantic_id = self._target_semantic_class_from_payload(payload)
            bev_semantic = payload.get("semantic_target_map_pred")
            full_semantic = payload.get("full_semantic_target_map_pred")
            raw_semantic = payload.get("semantic_target_mask_raw")
            minav_semantic = payload.get("minav_sem_mask")
            adapter_semantic = getattr(self.occupancy_adapter, "map", None)
            adapter_pixels = 0
            if adapter_semantic is not None and adapter_semantic.ndim == 3:
                adapter_pixels = int(np.sum(adapter_semantic[:, :, 2] == semantic_id))
            self._trace(
                "full_localizer_semantic_input",
                step=int(payload.get("step", 0)),
                target_semantic_class=semantic_id,
                bev_target_pixels=int(np.sum(np.asarray(bev_semantic) > 0))
                if bev_semantic is not None
                else 0,
                bev_shape=list(np.asarray(bev_semantic).shape)
                if bev_semantic is not None
                else None,
                full_target_pixels=int(np.sum(np.asarray(full_semantic) > 0))
                if full_semantic is not None
                else 0,
                full_shape=list(np.asarray(full_semantic).shape)
                if full_semantic is not None
                else None,
                raw_target_pixels=int(np.sum(np.asarray(raw_semantic) > 0))
                if raw_semantic is not None
                else 0,
                raw_shape=list(np.asarray(raw_semantic).shape)
                if raw_semantic is not None
                else None,
                minav_target_pixels=int(np.sum(np.asarray(minav_semantic) == semantic_id))
                if minav_semantic is not None
                else 0,
                minav_semantic_pixels=int(np.sum(np.asarray(minav_semantic) > 0))
                if minav_semantic is not None
                else 0,
                minav_shape=list(np.asarray(minav_semantic).shape)
                if minav_semantic is not None
                else None,
                adapter_target_pixels=adapter_pixels,
                adapter_map_frame=getattr(self.occupancy_adapter, "map_frame", None),
            )
        except Exception as exc:
            self.logger.debug("Full localizer semantic-input trace failed: %s", exc)
        localizer = self._ensure_full_localizer(payload)
        if localizer is None:
            result.update(
                {
                    "fail_reason": self.full_localizer_error or "Full localizer unavailable",
                    "system_not_ready": True,
                }
            )
            return result

        try:
            confirm = self.last_confirm_result or {}
            localizer.set_cluster_recovery_context(
                tc_score=confirm.get("best_score", 0.0),
                tc_pose=confirm.get("best_pose"),
                tc_center=confirm.get("target_cluster_center", self.confirmed_target_position),
            )
        except Exception as exc:
            self.logger.debug("Failed to sync cluster-recovery TC evidence: %s", exc)

        try:
            with self._maybe_silence_full_localizer():
                localizer.update_mappers(self.slam_mapper, self.occupancy_adapter)
                if self.visualization_dir is not None:
                    localizer.update_results_dir(str(self.visualization_dir))
                self._sync_goal_masks_to_localizer(
                    localizer,
                    payload,
                    int(payload.get("step", 0)),
                    source="localization_update",
                )
        except Exception as exc:
            self.logger.debug("Full localizer mapper update failed: %s", exc)

        if not self.full_localizer_started:
            self._seed_full_localizer_object(localizer, payload)
            self.full_localizer_started = True

        keyframe_data = self._build_full_localizer_keyframe(payload, current_rgb)
        try:
            with self._maybe_silence_full_localizer():
                loc_raw = localizer.run_localization(keyframe_data)
        except Exception as exc:
            self.full_localizer_error = str(exc)
            result["fail_reason"] = "Full MinAV localization exception: {}".format(exc)
            return result

        self._log_full_localization_raw(payload, loc_raw)
        self.localization_stage = str(loc_raw.get("state", getattr(localizer.current_state, "value", "unknown"))).upper()
        self.localization_trace = list(getattr(localizer, "state_history", []) or [])
        result["stages"] = list(self.localization_trace)
        result["raw_result"] = loc_raw

        status = loc_raw.get("status")
        if status == "cluster_recovery_requested":
            result.update(
                {
                    "cluster_recovery_requested": True,
                    "best_score": float(loc_raw.get("trigger_score", 0.0) or 0.0),
                    "trigger_source": loc_raw.get("trigger_source"),
                    "trigger_score": float(loc_raw.get("trigger_score", 0.0) or 0.0),
                    "trigger_threshold": float(
                        loc_raw.get("trigger_threshold", self.cluster_recovery_trigger_score)
                        or self.cluster_recovery_trigger_score
                    ),
                    "base_pose": loc_raw.get("base_pose"),
                    "tc_score": float(loc_raw.get("tc_score", 0.0) or 0.0),
                    "fem_score": float(loc_raw.get("fem_score", 0.0) or 0.0),
                    "ref_score": float(loc_raw.get("ref_score", 0.0) or 0.0),
                    "reference_center": loc_raw.get("reference_center"),
                    "fail_reason": "",
                    "system_not_ready": True,
                }
            )
            return result
        if status == "localization_complete":
            target_position = loc_raw.get("target_position")
            if target_position is None:
                result["fail_reason"] = "Full MinAV localization completed without target_position"
                return result
            position_source = loc_raw.get("position_source")
            if position_source == "camera":
                result["fail_reason"] = (
                    "Full MinAV localization returned REF camera position; "
                    "object-derived target position required"
                )
                result["system_not_ready"] = True
                return result
            result.update(
                {
                    "localized": True,
                    "target_position": np.asarray(target_position, dtype=np.float32),
                    "target_object_center": np.asarray(
                        loc_raw.get("target_object_center", target_position),
                        dtype=np.float32,
                    ),
                    "target_yaw": loc_raw.get("target_yaw"),
                    "position_source": position_source,
                    "fail_reason": "",
                }
            )
            return result

        if status == "mr_closed_loop_done":
            target_position = loc_raw.get("target_position")
            if target_position is None:
                result["fail_reason"] = "MR closed-loop returned no target_position"
                return result
            result.update(
                {
                    "mr_reposition": True,
                    "target_position": np.asarray(target_position, dtype=np.float32),
                    "target_yaw": loc_raw.get("target_yaw"),
                    "new_pose": loc_raw.get("new_pose"),
                    "best_score": float(
                        loc_raw.get("score", self.localization_match_threshold) or 0.0
                    ),
                    "fail_reason": "",
                }
            )
            return result

        if status == "failed" or loc_raw.get("state") == "failed":
            result["fail_reason"] = loc_raw.get("message", "Full MinAV localization failed")
            return result

        result.update(
            {
                "fail_reason": loc_raw.get("message", "Full MinAV localization running"),
                "system_not_ready": True,
            }
        )
        return result

    def _ensure_full_localizer(self, payload: Dict[str, Any]):
        if self.full_localizer is not None:
            return self.full_localizer
        try:
            from ieve_minav.localizer.localization_system import LocalizationSystem

            target_semantic_class = self._target_semantic_class_from_payload(payload)
            segmenter = self._ensure_goal_segmenter()
            config = {
                "match_threshold": self.localization_match_threshold,
                "closed_loop_threshold": float(
                    getattr(self.args, "minav_mr_closed_loop_threshold", self.localization_match_threshold)
                ),
                "render_views": int(getattr(self.args, "minav_full_render_views", 48)),
                "render_radius": float(getattr(self.args, "minav_full_render_radius", 1.0)),
                "max_closed_loop_rounds": int(
                    getattr(self.args, "minav_full_max_closed_loop_rounds", 3)
                ),
                "gaussian_view_min_size_ratio": float(
                    getattr(self.args, "minav_gaussian_view_min_size_ratio", 0.10)
                ),
                "cluster_recovery_enabled": self.cluster_recovery_enabled,
                "cluster_recovery_trigger_score": self.cluster_recovery_trigger_score,
                "fine_localization_enabled": self.fine_localization_enabled,
                "target_semantic_class": target_semantic_class,
            }
            with self._maybe_silence_full_localizer():
                self.full_localizer = LocalizationSystem(
                    slam_mapper=self.slam_mapper,
                    occupancy_mapper=self.occupancy_adapter,
                    target_image=self.target_image,
                    habitat_sim=None,
                    config=config,
                    target_category=payload.get("target_category") or self.target_category,
                    results_dir=str(self.visualization_dir) if self.visualization_dir is not None else None,
                    segmenter=segmenter,
                    save_visualization_results=self.save_stage_visualizations,
                )
            self._sync_full_localizer_target_semantic_class(
                self.full_localizer, payload, source="init"
            )
            self._sync_goal_masks_to_localizer(
                self.full_localizer,
                payload,
                int(payload.get("step", 0)),
                source="init",
            )
            self._sync_full_localizer_candidate_anchor(
                self.full_localizer,
                int(payload.get("step", 0)),
                source="init",
            )
            self.full_localizer_error = None
            return self.full_localizer
        except Exception as exc:
            self.full_localizer = None
            self.full_localizer_error = str(exc)
            self.logger.warning("Full MinAV localizer unavailable: %s", exc)
            return None

    def _sync_full_localizer_target_semantic_class(
        self, localizer, payload: Dict[str, Any], source: str
    ) -> int:
        semantic_id = self._target_semantic_class_from_payload(payload)
        if hasattr(self.occupancy_adapter, "target_semantic_class"):
            self.occupancy_adapter.target_semantic_class = semantic_id
        if hasattr(localizer, "target_semantic_class"):
            localizer.target_semantic_class = semantic_id
        if hasattr(localizer, "soim") and localizer.soim is not None:
            try:
                localizer.soim.set_target_class(semantic_id)
            except Exception as exc:
                self.logger.debug("Failed to set SOIM target class: %s", exc)
        self._trace(
            "soim_target_semantic_class_set",
            step=int(payload.get("step", 0)),
            target_semantic_class=semantic_id,
            goal_cat_id=payload.get("goal_cat_id"),
            source=source,
            contract="occupancy_map_channel_2_equals_target_semantic_class",
        )
        return semantic_id

    def _seed_full_localizer_object(self, localizer, payload: Dict[str, Any]) -> None:
        self._sync_full_localizer_target_semantic_class(localizer, payload, source="seed")
        self._sync_goal_masks_to_localizer(
            localizer,
            payload,
            int(payload.get("step", 0)),
            source="seed",
        )
        self._sync_full_localizer_candidate_anchor(
            localizer,
            int(payload.get("step", 0)),
            source="seed",
        )
        if not self.full_localizer_started:
            localizer.start_localization()
            self.full_localizer_started = True
        localizer.best_pose = None
        localizer.refined_pose = None
        localizer.rendered_images = []
        try:
            with self._maybe_silence_full_localizer():
                if self.buffer_gaussmi_signaled:
                    localizer.notify_gaussmi_ready()
                if (
                    self.pending_closed_loop_pose is not None
                    and hasattr(localizer, "inject_closed_loop_pose_for_pre")
                ):
                    localizer.inject_closed_loop_pose_for_pre(
                        self.pending_closed_loop_pose,
                        score_hint=self.pending_closed_loop_score,
                        source="ieve_mr_remap_sweep",
                    )
                    self.pending_closed_loop_pose = None
                    self.pending_closed_loop_score = None
        except Exception:
            pass

    @contextlib.contextmanager
    def _maybe_silence_full_localizer(self):
        yield

    def _log_full_localization_raw(self, payload: Dict[str, Any], loc_raw: Dict[str, Any]) -> None:
        try:
            msg = "[MINAV_LOCALIZATION_RAW] env={} step={} mode={} raw={}".format(
                self.env_idx,
                int(payload.get("step", 0)),
                self.mode.value,
                json.dumps(self._json_safe(loc_raw), ensure_ascii=False, sort_keys=True),
            )
        except Exception as exc:
            msg = "[MINAV_LOCALIZATION_RAW] env={} step={} failed_to_serialize={}".format(
                self.env_idx,
                int((payload or {}).get("step", 0)),
                exc,
            )
        print(msg)
        self.logger.info(msg)

    def _build_full_localizer_keyframe(
        self, payload: Dict[str, Any], current_rgb: np.ndarray
    ) -> Dict[str, Any]:
        pose_w2c = self._build_full_localizer_pose_w2c(payload)
        return {
            "rgb": current_rgb,
            "depth": payload.get("depth_m", payload.get("raw_depth")),
            "semantic": self._semantic_mask_for_full_localizer(payload),
            "pose": pose_w2c,
            "agent_position": payload.get("agent_position"),
            "agent_rotation_xyzw": payload.get("agent_rotation_xyzw"),
            "step": int(payload.get("step", 0)),
            "timestamp": int(payload.get("step", 0)),
            "is_3dgs_keyframe": True,
        }

    def _build_full_localizer_pose_w2c(self, payload: Dict[str, Any]) -> Optional[np.ndarray]:
        position = payload.get("agent_position")
        rotation = payload.get("agent_rotation_xyzw")
        if position is None:
            return None
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return None
        rot = self._quat_xyzw_to_matrix(rotation) if rotation is not None else None
        if rot is None:
            rot = np.eye(3, dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot.astype(np.float32)
        c2w[:3, 3] = pos[:3]
        flip_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        c2w = c2w @ flip_mat
        try:
            return np.linalg.inv(c2w).astype(np.float32)
        except Exception:
            w2c = np.eye(4, dtype=np.float32)
            corrected_rot = c2w[:3, :3]
            w2c[:3, :3] = corrected_rot.T.astype(np.float32)
            w2c[:3, 3] = -(corrected_rot.T @ pos[:3]).astype(np.float32)
            return w2c

    def _trace_localization_stage(self, stage: str, ok: bool, message: str) -> None:
        self.localization_stage = stage
        self.localization_trace.append(
            {"stage": stage, "ok": bool(ok), "message": str(message)}
        )

    def _select_navigation_position_from_ref_view(self, payload, fallback_position):
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            return fallback_position
        indices = self.confirmed_indices
        if indices is None or indices.size == 0:
            try:
                indices = self._to_numpy_indices(gaussians.get_confirmed_target_indices())
            except Exception:
                return fallback_position
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception:
            return fallback_position
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            return fallback_position
        xyz = xyz_all[valid]

        current_world = payload.get("agent_position")
        if current_world is not None:
            current_world = np.asarray(current_world, dtype=np.float32).reshape(-1)
            if current_world.size >= 3:
                dists = np.linalg.norm(xyz[:, [0, 2]] - current_world[[0, 2]], axis=1)
                near = xyz[dists <= max(self.buffer_distance_threshold, 1.0)]
                if near.shape[0] >= max(3, self.cluster_min_size // 4):
                    return near.mean(axis=0).astype(np.float32)
        return xyz.mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Goal adapter
    # ------------------------------------------------------------------
    def _clear_navigation_goal_state(self) -> None:
        self.navigation_goal_world = None
        self.navigation_goal_raw_world = None
        self.navigation_goal_local_cell = None
        self.navigation_goal_full_cell = None
        self.navigation_raw_target_full_cell = None
        self.navigation_goal_adjusted = False

    def _reset_reachable_navigation_state(self) -> None:
        self.navigation_goal_tier = None
        self.navigation_effective_stop_radius_m = None
        self.navigation_may_terminal_stop = False
        self.navigation_goal_selection = {}
        self.navigation_staging_replan_count = 0
        self.navigation_staging_observation_pending = False
        self.navigation_fallback_force_stop = False
        self.navigation_fallback_force_stop_reason = None
        self.navigation_phase3_tolerance_hold = False
        self.navigation_localizer_suggested_goal = None

    def _invalidate_navigation_execution_goal(self) -> None:
        self._clear_navigation_goal_state()
        self.navigation_goal_tier = None
        self.navigation_effective_stop_radius_m = None
        self.navigation_may_terminal_stop = False
        self.navigation_goal_selection = {}
        self.navigation_phase3_tolerance_hold = False

    def _confirmed_target_goal_anchor(self) -> Optional[np.ndarray]:
        return self.confirmed_target_position

    @staticmethod
    def _as_world_point(value) -> Optional[np.ndarray]:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
            return None
        return arr[:3].copy()

    def _target_goal_anchor_from_confirm_result(
        self,
        result: Dict[str, Any],
        target_center: np.ndarray,
        payload: Dict[str, Any],
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        del result, payload
        return np.asarray(target_center, dtype=np.float32).reshape(-1)[:3].copy(), {
            "source": "target_cluster_center",
            "strict_gaussian_center": True,
        }

    def _raw_ieve_goal_from_world(
        self, target_world_xyz: np.ndarray, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        projection = self._buffer_project_world_to_local(target_world_xyz, context)
        if not projection or not projection.get("ok"):
            return None
        if bool(projection.get("clipped_to_local_map", False)):
            return {
                "ok": False,
                "reason": "target_outside_local_map",
                "projection": projection,
            }
        local_cell = tuple(int(v) for v in projection["raw_local_cell"])
        shape = tuple(int(v) for v in projection["shape"])
        goal_map = self._buffer_make_goal_map(shape, local_cell)
        return {
            "ok": True,
            "projection": projection,
            "goal_map": goal_map,
            "local_cell": local_cell,
            "full_cell": tuple(int(v) for v in projection["raw_full_cell"]),
            "start_cell": tuple(int(v) for v in projection["start_cell"]),
        }

    def _phase3_navigation_active(self) -> bool:
        return self.localization_stage == "phase3_best_evidence_navigation"

    def _navigation_raw_target(self) -> Optional[np.ndarray]:
        target = self.localized_target_position
        if target is None:
            target = self.navigation_goal_raw_world
        if target is None:
            target = self.confirmed_target_position
        return self._as_world_point(target)

    def _select_reachable_navigation_goal(
        self,
        raw_target_world: np.ndarray,
        context: Dict[str, Any],
        *,
        phase3: Optional[bool] = None,
    ) -> Dict[str, Any]:
        phase3 = self._phase3_navigation_active() if phase3 is None else bool(phase3)
        raw_goal = self._raw_ieve_goal_from_world(raw_target_world, context)
        if not raw_goal or not raw_goal.get("ok"):
            return {
                "selected": False,
                "reason": (raw_goal or {}).get("reason", "projection_failed"),
                "raw_goal": raw_goal,
            }

        if phase3:
            preferred_radius = float(self.strategy_phase3_fallback_stop_radius)
            fallback_radius = preferred_radius + float(
                self.phase3_navigation_stuck_tolerance_m
            )
            max_radius = float(self.phase3_navigation_staging_max_radius)
        else:
            preferred_radius = float(self.navigation_preferred_stop_radius)
            fallback_radius = float(self.navigation_arrival_threshold)
            max_radius = float(self.navigation_staging_max_radius)

        selection = select_reachable_goal_ieve(
            context.get("map_pred"),
            raw_goal["start_cell"],
            raw_goal["local_cell"],
            visited=context.get("visited_pred"),
            collision_map=context.get("collision_map"),
            map_resolution=float(
                raw_goal["projection"].get(
                    "map_resolution_cm",
                    context.get("map_resolution_cm", 5.0),
                )
            ),
            preferred_radius_m=preferred_radius,
            fallback_radius_m=fallback_radius,
            max_radius_m=max_radius,
            ring_step_m=float(self.navigation_reachable_ring_step),
        )
        selection["phase3"] = bool(phase3)
        selection["raw_target_world"] = np.asarray(
            raw_target_world, dtype=np.float32
        ).reshape(-1)[:3].copy()
        selection["raw_full_cell"] = tuple(int(v) for v in raw_goal["full_cell"])
        selection["raw_goal"] = raw_goal
        if not selection.get("selected"):
            return selection

        goal_cell = tuple(int(v) for v in selection["goal_cell"])
        projection = raw_goal["projection"]
        execution_goal = self._buffer_local_cell_to_world(
            goal_cell,
            projection,
            np.asarray(raw_target_world, dtype=np.float32).reshape(-1)[:3],
        )
        selection["execution_goal_world"] = execution_goal
        selection["goal_map"] = self._buffer_make_goal_map(
            tuple(int(v) for v in projection["shape"]), goal_cell
        )
        selection["full_cell"] = (
            int(goal_cell[0]) + int(projection["gx1"]),
            int(goal_cell[1]) + int(projection["gy1"]),
        )
        if phase3 and selection.get("goal_tier") == "fallback":
            selection["effective_stop_radius_m"] = (
                fallback_radius
                if self.phase3_navigation_stuck_tolerance_active
                else preferred_radius
            )
        return selection

    def _commit_reachable_navigation_goal(
        self, selection: Dict[str, Any], step: int
    ) -> None:
        raw_target = np.asarray(
            selection["raw_target_world"], dtype=np.float32
        ).reshape(-1)[:3]
        execution_goal = np.asarray(
            selection["execution_goal_world"], dtype=np.float32
        ).reshape(-1)[:3]
        self.navigation_goal_world = execution_goal.copy()
        self.navigation_goal_raw_world = raw_target.copy()
        self.navigation_goal_local_cell = tuple(
            int(v) for v in selection["goal_cell"]
        )
        self.navigation_goal_full_cell = tuple(
            int(v) for v in selection["full_cell"]
        )
        raw_full_cell = selection.get("raw_full_cell")
        if raw_full_cell is None:
            raw_full_cell = selection.get("full_cell")
        self.navigation_raw_target_full_cell = (
            None
            if raw_full_cell is None
            else tuple(int(v) for v in raw_full_cell)
        )
        self.navigation_goal_adjusted = bool(
            self._world_distance_xz(raw_target, execution_goal)
            > float(getattr(self.args, "map_resolution", 5.0)) / 200.0
        )
        self.navigation_goal_tier = str(selection.get("goal_tier"))
        stop_radius = selection.get("effective_stop_radius_m")
        self.navigation_effective_stop_radius_m = (
            None if stop_radius is None else float(stop_radius)
        )
        self.navigation_may_terminal_stop = bool(
            selection.get("may_terminal_stop", False)
        )
        self.navigation_goal_selection = {
            key: value
            for key, value in selection.items()
            if key not in ("goal_map", "goal_region_mask", "raw_goal")
        }
        self._trace(
            "navigation_reachable_goal_selected",
            step=int(step),
            raw_target_world=raw_target,
            execution_goal_world=execution_goal,
            goal_tier=self.navigation_goal_tier,
            effective_stop_radius_m=self.navigation_effective_stop_radius_m,
            may_terminal_stop=self.navigation_may_terminal_stop,
            search_radius_m=selection.get("search_radius_m"),
            candidate_count=selection.get("candidate_count"),
            fmm_cost_cm=selection.get("fmm_cost_cm"),
            target_center_distance_m=selection.get("target_center_distance_m"),
            obstacle_clearance_m=selection.get("obstacle_clearance_m"),
            staging_replan_count=self.navigation_staging_replan_count,
            replan_reason=selection.get("replan_reason"),
            fallback_stop=False,
            fallback_stop_reason=None,
        )

    def _navigation_staging_face_action(
        self, context: Dict[str, Any], raw_target: np.ndarray
    ) -> Optional[str]:
        desired = self._yaw_from_position_to_target(
            context.get("agent_position", context.get("current_world_xyz")),
            raw_target,
        )
        current = self._yaw_from_quat_xyzw(
            context.get("agent_rotation_xyzw", context.get("current_rotation_xyzw"))
        )
        if desired is None or current is None:
            return None
        error = self._angle_diff_rad(desired, current)
        turn_angle = max(
            1.0, abs(float(getattr(self.args, "turn_angle", 30.0) or 30.0))
        )
        if abs(np.rad2deg(error)) <= turn_angle * 0.5:
            return None
        return "turn_left" if error > 0.0 else "turn_right"

    def _activate_navigation_fallback_stop(self, step: int, reason: str) -> None:
        self.navigation_fallback_force_stop = True
        self.navigation_fallback_force_stop_reason = str(reason)
        self.navigation_may_terminal_stop = False
        self.navigation_complete = False
        self._trace(
            "navigation_unreachable_fallback_stop",
            step=int(step),
            reason=str(reason),
            raw_target_world=self._navigation_raw_target(),
            execution_goal_world=self.navigation_goal_world,
            goal_tier=self.navigation_goal_tier,
            effective_stop_radius_m=self.navigation_effective_stop_radius_m,
            staging_replan_count=self.navigation_staging_replan_count,
            contract="explicit_failure_stop_after_staging_replan",
        )

    def _recover_from_missing_navigation_staging(
        self, step: int, reason: str
    ) -> None:
        if self._phase3_navigation_active():
            self.targeted_phase3_no_candidate_cooldown_until = (
                int(step) + self.phase3_no_candidate_cooldown_steps
            )
            self._trace(
                "phase3_navigation_no_staging_recovery",
                step=int(step),
                reason=str(reason),
                cooldown_until=self.targeted_phase3_no_candidate_cooldown_until,
                cooldown_steps=self.phase3_no_candidate_cooldown_steps,
            )
        self._fail_to_exploration(int(step), str(reason), blacklist=False)

    def _get_reachable_navigation_goal_override(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        step = int(context.get("step", 0))
        raw_target = self._navigation_raw_target()
        if raw_target is None:
            self._recover_from_missing_navigation_staging(
                step, "navigation_raw_target_missing"
            )
            return {"goal_source": "MINAV_NAVIGATION_TARGET_MISSING"}

        replanning_after_staging = False
        if self.navigation_staging_observation_pending:
            face_action = self._navigation_staging_face_action(context, raw_target)
            if face_action is not None:
                return {
                    "goal_source": "MINAV_NAVIGATION_STAGING_FACE_TARGET",
                    "action_override": face_action,
                    "hold_action": False,
                }
            self.navigation_staging_observation_pending = False
            self.navigation_staging_replan_count += 1
            replanning_after_staging = True
            self._trace(
                "navigation_staging_replan",
                step=step,
                raw_target_world=raw_target,
                staging_goal_world=self.navigation_goal_world,
                staging_replan_count=self.navigation_staging_replan_count,
                staging_replan_limit=self.navigation_staging_replan_limit,
            )

        selection = self._select_reachable_navigation_goal(
            raw_target, context
        )
        selection["replan_reason"] = (
            "staging_arrival_replan"
            if replanning_after_staging
            else (
                "initial_selection"
                if self.navigation_goal_world is None
                else "map_or_pose_update"
            )
        )
        if not selection.get("selected"):
            reason = str(selection.get("reason", "navigation_goal_unavailable"))
            if replanning_after_staging:
                self._activate_navigation_fallback_stop(
                    step,
                    "phase3_unreachable_fallback_stop"
                    if self._phase3_navigation_active()
                    else "localized_target_unreachable_fallback_stop",
                )
                return {
                    "goal_source": "MINAV_NAVIGATION_UNREACHABLE_FALLBACK_STOP",
                    "force_stop": True,
                    "force_stop_reason": self.navigation_fallback_force_stop_reason,
                }
            self._trace(
                "navigation_reachable_goal_unavailable",
                step=step,
                reason=reason,
                raw_target_world=raw_target,
                selection=selection,
            )
            self._recover_from_missing_navigation_staging(step, reason)
            return {"goal_source": "MINAV_NAVIGATION_NO_STAGING_RECOVERY"}

        if (
            replanning_after_staging
            and selection.get("goal_tier") == "staging"
            and self.navigation_staging_replan_count
            >= self.navigation_staging_replan_limit
        ):
            self._activate_navigation_fallback_stop(
                step,
                "phase3_unreachable_fallback_stop"
                if self._phase3_navigation_active()
                else "localized_target_unreachable_fallback_stop",
            )
            return {
                "goal_source": "MINAV_NAVIGATION_UNREACHABLE_FALLBACK_STOP",
                "force_stop": True,
                "force_stop_reason": self.navigation_fallback_force_stop_reason,
            }

        if replanning_after_staging and selection.get("goal_tier") != "staging":
            self._trace(
                "navigation_staging_replan_recovered",
                step=step,
                recovered_goal_tier=selection.get("goal_tier"),
                staging_replan_count=self.navigation_staging_replan_count,
            )
            self.navigation_staging_replan_count = 0

        self._commit_reachable_navigation_goal(selection, step)
        raw_distance = self._navigation_map_distance_m(
            context,
            full_cell=self.navigation_raw_target_full_cell,
        )
        if (
            self.navigation_may_terminal_stop
            and self.navigation_effective_stop_radius_m is not None
            and raw_distance <= self.navigation_effective_stop_radius_m
        ):
            self.navigation_complete = True
            self.navigation_phase3_tolerance_hold = False
            self._trace(
                "navigation_completed_after_goal_selection",
                step=step,
                raw_target_world=raw_target,
                raw_target_distance_m=raw_distance,
                goal_tier=self.navigation_goal_tier,
                effective_stop_radius_m=self.navigation_effective_stop_radius_m,
            )
        if self.navigation_phase3_tolerance_hold:
            return {
                "goal_source": "MINAV_PHASE3_TOLERANCE_WAIT",
                "hold_action": True,
            }
        return {
            "goal_world": self.navigation_goal_world,
            "goal_map": selection["goal_map"],
            "goal_region_mask": selection.get("goal_region_mask"),
            "goal_source": "MINAV_NAVIGATION_REACHABLE_{}".format(
                str(self.navigation_goal_tier or "goal").upper()
            ),
            "goal_dilation_radius": self.navigation_execution_goal_dilation_radius,
        }

    def _ieve_temp_goal_reachability(
        self,
        map_pred,
        goal_map,
        start_cell,
        projection,
        context=None,
        goal_dilation_radius: Optional[int] = None,
    ) -> Dict[str, Any]:
        if map_pred is None:
            return {
                "reachable": False,
                "distance_cm": None,
                "reason": "missing_map_pred",
            }
        context = context or {}
        h, w = np.asarray(map_pred).shape[:2]
        if goal_dilation_radius is None:
            goal_dilation_radius = int(
                getattr(self.args, "minav_buffer_goal_dilation_radius", 10)
            )
        result = compute_temp_goal_reachability_ieve(
            map_pred,
            goal_map,
            start_cell,
            [0, h, 0, w],
            visited=context.get("visited_pred"),
            collision_map=context.get("collision_map"),
            map_resolution=float(projection.get("map_resolution_cm", 5.0)),
            goal_dilation_radius=max(0, int(goal_dilation_radius)),
        )
        result["rule"] = "ieve_compute_temp_goal_distance"
        result["goal_dilation_radius"] = max(0, int(goal_dilation_radius))
        if result.get("cost") is None and result.get("distance_cm") is not None:
            result["cost"] = result.get("distance_cm")
        return result

    def world_target_to_goal_map(
        self, target_world_xyz: np.ndarray, context: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        map_pred = context.get("map_pred")
        target_world = np.asarray(target_world_xyz, dtype=np.float32).reshape(-1)
        if target_world.size < 3:
            return None
        track_executor_goal = bool(
            self.mode == IEVETargetMode.NAVIGATION
            or (
                self.mode == IEVETargetMode.BUFFER
                and self.buffer_direct_nav_to_confirmed_target
            )
        )

        raw_goal = self._raw_ieve_goal_from_world(target_world[:3], context)
        if not raw_goal or not raw_goal.get("ok"):
            projection = (raw_goal or {}).get("projection", {})
            if track_executor_goal:
                self._clear_navigation_goal_state()
                self._trace(
                    "navigation_goal_projected",
                    step=int(context.get("step", 0)),
                    mode=self.mode.value,
                    ok=False,
                    reason=(raw_goal or {}).get("reason", "projection_failed"),
                    raw_local_cell=projection.get("raw_local_cell"),
                    raw_full_cell=projection.get("raw_full_cell"),
                    target_local_cell=projection.get("target_local_cell"),
                    target_full_cell=projection.get("target_full_cell"),
                    raw_target_world=target_world[:3],
                    coordinate_source=projection.get("coordinate_source"),
                    adapter_map_frame=projection.get("adapter_map_frame"),
                    rule="ieve_compute_temp_goal_distance",
                )
            return None

        projection = raw_goal["projection"]
        local_row, local_col = raw_goal["local_cell"]
        full_row, full_col = raw_goal["full_cell"]
        start_row, start_col = raw_goal["start_cell"]
        if self.mode == IEVETargetMode.NAVIGATION:
            reachability_goal_dilation_radius = int(
                getattr(self.args, "minav_navigation_goal_dilation_radius", 5)
            )
        else:
            reachability_goal_dilation_radius = int(
                getattr(self.args, "minav_buffer_goal_dilation_radius", 10)
            )
        reachability = self._ieve_temp_goal_reachability(
            map_pred,
            raw_goal["goal_map"],
            raw_goal["start_cell"],
            projection,
            context,
            goal_dilation_radius=reachability_goal_dilation_radius,
        )
        if not reachability.get("reachable"):
            if track_executor_goal:
                self._clear_navigation_goal_state()
                self._trace(
                    "navigation_goal_projected",
                    step=int(context.get("step", 0)),
                    mode=self.mode.value,
                    ok=False,
                    reason=reachability.get("reason", "ieve_temp_goal_unreachable"),
                    raw_local_cell=raw_goal["local_cell"],
                    raw_full_cell=raw_goal["full_cell"],
                    start_cell=raw_goal["start_cell"],
                    shape=projection.get("shape"),
                    raw_target_world=target_world[:3],
                    coordinate_source=projection.get("coordinate_source"),
                    adapter_map_frame=projection.get("adapter_map_frame"),
                    reachability=reachability,
                )
            return None

        if track_executor_goal:
            self.navigation_goal_world = target_world[:3].copy().astype(np.float32)
            self.navigation_goal_raw_world = target_world[:3].copy()
            self.navigation_goal_local_cell = (int(local_row), int(local_col))
            self.navigation_goal_full_cell = (int(full_row), int(full_col))
            self.navigation_raw_target_full_cell = (int(full_row), int(full_col))
            self.navigation_goal_adjusted = False
            self._trace(
                "navigation_goal_projected",
                step=int(context.get("step", 0)),
                mode=self.mode.value,
                ok=True,
                raw_target_world=target_world[:3],
                navigation_goal_world=self.navigation_goal_world,
                raw_local_cell=raw_goal["local_cell"],
                local_cell=self.navigation_goal_local_cell,
                raw_full_cell=raw_goal["full_cell"],
                full_cell=self.navigation_goal_full_cell,
                start_cell=(int(start_row), int(start_col)),
                adjusted=self.navigation_goal_adjusted,
                raw_to_goal_distance=0.0,
                coordinate_source=projection.get("coordinate_source"),
                adapter_map_frame=projection.get("adapter_map_frame"),
                reachability={
                    key: value
                    for key, value in reachability.items()
                    if key not in ("goal_map", "goal_region_mask", "raw_goal")
                },
            )
        return raw_goal["goal_map"]

    def _pose_delta_projection(
        self,
        target_world: np.ndarray,
        current_world: np.ndarray,
        pose_pred: np.ndarray,
        h: int,
        w: int,
        map_resolution_cm: float,
    ) -> Dict[str, Any]:
        start_x, start_y, _, gx1, _, gy1, _ = np.asarray(pose_pred, dtype=np.float32)
        target_ieve_x = float(start_x + (current_world[2] - target_world[2]))
        target_ieve_y = float(start_y + (current_world[0] - target_world[0]))
        full_row_f = target_ieve_y * 100.0 / map_resolution_cm
        full_col_f = target_ieve_x * 100.0 / map_resolution_cm
        full_row = int(round(full_row_f))
        full_col = int(round(full_col_f))
        local_row = full_row - int(gx1)
        local_col = full_col - int(gy1)
        unclipped_local_row = int(local_row)
        unclipped_local_col = int(local_col)
        clipped_local_row = int(np.clip(unclipped_local_row, 0, h - 1))
        clipped_local_col = int(np.clip(unclipped_local_col, 0, w - 1))
        clipped_to_local = (
            clipped_local_row != unclipped_local_row
            or clipped_local_col != unclipped_local_col
        )
        selected_full_row = clipped_local_row + int(gx1)
        selected_full_col = clipped_local_col + int(gy1)
        start_row = int(round(start_y * 100.0 / map_resolution_cm)) - int(gx1)
        start_col = int(round(start_x * 100.0 / map_resolution_cm)) - int(gy1)
        return {
            "ok": True,
            "target_world": target_world[:3].copy(),
            "raw_local_cell": (int(clipped_local_row), int(clipped_local_col)),
            "raw_full_cell": (int(selected_full_row), int(selected_full_col)),
            "target_local_cell": (int(unclipped_local_row), int(unclipped_local_col)),
            "target_full_cell": (int(full_row), int(full_col)),
            "start_cell": (
                int(np.clip(start_row, 0, h - 1)),
                int(np.clip(start_col, 0, w - 1)),
            ),
            "shape": (h, w),
            "gx1": int(gx1),
            "gy1": int(gy1),
            "map_resolution_cm": map_resolution_cm,
            "start_x": float(start_x),
            "start_y": float(start_y),
            "current_world": current_world[:3].copy(),
            "coordinate_source": "ieve_pose_delta_fallback",
            "clipped_to_local_map": bool(clipped_to_local),
            "unrounded_full_cell": (float(full_row_f), float(full_col_f)),
        }

    def _debug_projection_roundtrip(
        self, projection: Optional[Dict[str, Any]], target_world: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        if projection is None:
            return None
        try:
            map_frame = projection.get("adapter_map_frame")
            cell = (
                projection.get("raw_full_cell")
                if map_frame == "ieve_full_map"
                else projection.get("raw_local_cell")
            )
            if cell is None:
                return None
            row, col = int(cell[0]), int(cell[1])
            world_x, world_z = self.occupancy_adapter.map_to_world(col, row)
            error = float(
                np.linalg.norm(
                    [
                        float(world_x) - float(target_world[0]),
                        float(world_z) - float(target_world[2]),
                    ]
                )
            )
            return {
                "cell": (row, col),
                "world_xz": (float(world_x), float(world_z)),
                "error_m": error,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _trace_world_to_map_projection_debug(
        self,
        context: Dict[str, Any],
        target_world: np.ndarray,
        current_world: np.ndarray,
        pose_pred: np.ndarray,
        adapter_projection: Optional[Dict[str, Any]],
        fallback_projection: Optional[Dict[str, Any]],
        selected_projection: Optional[Dict[str, Any]],
    ) -> None:
        if not self._debug_enabled():
            return
        if bool(context.get("suppress_projection_debug", False)):
            return
        adapter_full = (
            adapter_projection.get("target_full_cell")
            if isinstance(adapter_projection, dict)
            else None
        )
        fallback_full = (
            fallback_projection.get("target_full_cell")
            if isinstance(fallback_projection, dict)
            else None
        )
        full_delta = None
        if adapter_full is not None and fallback_full is not None:
            full_delta = (
                int(adapter_full[0]) - int(fallback_full[0]),
                int(adapter_full[1]) - int(fallback_full[1]),
            )
        self._trace(
            "world_to_map_projection_debug",
            step=int(context.get("step", 0)),
            target_world=target_world[:3],
            current_world=current_world[:3],
            pose_pred=np.asarray(pose_pred, dtype=np.float32).reshape(-1)[:7],
            adapter_projection=adapter_projection,
            fallback_projection=fallback_projection,
            selected_coordinate_source=(
                selected_projection.get("coordinate_source")
                if isinstance(selected_projection, dict)
                else None
            ),
            adapter_vs_fallback_target_full_delta=full_delta,
            selected_semantic_distance=self._debug_semantic_distance_to_projection(
                selected_projection,
                context,
            ),
            selected_roundtrip=self._debug_projection_roundtrip(
                selected_projection,
                target_world,
            ),
        )

    @staticmethod
    def _projection_full_cell_delta(
        lhs: Optional[Dict[str, Any]], rhs: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[int, int]]:
        if not isinstance(lhs, dict) or not isinstance(rhs, dict):
            return None
        lhs_cell = lhs.get("target_full_cell")
        rhs_cell = rhs.get("target_full_cell")
        if lhs_cell is None or rhs_cell is None:
            return None
        try:
            return (
                int(lhs_cell[0]) - int(rhs_cell[0]),
                int(lhs_cell[1]) - int(rhs_cell[1]),
            )
        except Exception:
            return None

    def _select_world_to_local_projection(
        self,
        adapter_projection: Optional[Dict[str, Any]],
        fallback_projection: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not isinstance(adapter_projection, dict):
            return fallback_projection, "fallback_no_adapter"
        if not isinstance(fallback_projection, dict):
            return adapter_projection, "adapter_no_fallback"

        delta = self._projection_full_cell_delta(adapter_projection, fallback_projection)
        if delta is None:
            return adapter_projection, "adapter_no_delta"

        adapter_clipped = bool(adapter_projection.get("clipped_to_local_map", False))
        fallback_clipped = bool(fallback_projection.get("clipped_to_local_map", False))
        max_delta = max(abs(int(delta[0])), abs(int(delta[1])))
        if (adapter_clipped and not fallback_clipped) or max_delta > 5:
            selected = dict(fallback_projection)
            selected["coordinate_source"] = "ieve_pose_delta_fallback_selected"
            selected["adapter_projection_rejected"] = {
                "reason": (
                    "adapter_clipped_fallback_valid"
                    if adapter_clipped and not fallback_clipped
                    else "adapter_fallback_cell_mismatch"
                ),
                "adapter_target_full_cell": adapter_projection.get("target_full_cell"),
                "fallback_target_full_cell": fallback_projection.get("target_full_cell"),
                "target_full_cell_delta": delta,
                "adapter_clipped_to_local_map": adapter_clipped,
                "fallback_clipped_to_local_map": fallback_clipped,
            }
            return selected, "fallback_selected"
        return adapter_projection, "adapter_selected"

    def _buffer_project_world_to_local(
        self, target_world_xyz: np.ndarray, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        pose_pred = context.get("pose_pred")
        map_pred = context.get("map_pred")
        current_world = context.get("current_world_xyz")
        if current_world is None:
            current_world = context.get("agent_position")
        map_resolution_cm = float(context.get("map_resolution_cm", 5.0))
        if pose_pred is None or current_world is None or map_pred is None:
            return None

        target_world = np.asarray(target_world_xyz, dtype=np.float32).reshape(-1)
        current_world = np.asarray(current_world, dtype=np.float32).reshape(-1)
        if target_world.size < 3 or current_world.size < 3:
            return None
        if (
            not np.all(np.isfinite(target_world[:3]))
            or not np.all(np.isfinite(current_world[:3]))
        ):
            return None

        local_shape = np.asarray(map_pred).shape
        if len(local_shape) < 2:
            return None
        h, w = int(local_shape[0]), int(local_shape[1])
        if h <= 0 or w <= 0 or map_resolution_cm <= 0:
            return None

        pose_arr = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
        if pose_arr.size < 7:
            return None
        start_x, start_y, _, gx1, _, gy1, _ = pose_arr[:7]
        fallback_projection = self._pose_delta_projection(
            target_world[:3],
            current_world[:3],
            pose_arr[:7],
            h,
            w,
            map_resolution_cm,
        )
        adapter_projection = None
        try:
            self.occupancy_adapter.update(context)
            map_col, map_row = self.occupancy_adapter.world_to_map(
                float(target_world[0]), float(target_world[2])
            )
            if map_col is not None and map_row is not None:
                map_frame = getattr(self.occupancy_adapter, "map_frame", "ieve_local_map")
                if map_frame == "ieve_full_map":
                    full_row = int(map_row)
                    full_col = int(map_col)
                    local_row = full_row - int(gx1)
                    local_col = full_col - int(gy1)
                else:
                    local_row = int(map_row)
                    local_col = int(map_col)
                    full_row = local_row + int(gx1)
                    full_col = local_col + int(gy1)
                unclipped_local_row = int(local_row)
                unclipped_local_col = int(local_col)
                clipped_local_row = int(np.clip(unclipped_local_row, 0, h - 1))
                clipped_local_col = int(np.clip(unclipped_local_col, 0, w - 1))
                clipped_to_local = (
                    clipped_local_row != unclipped_local_row
                    or clipped_local_col != unclipped_local_col
                )
                selected_full_row = clipped_local_row + int(gx1)
                selected_full_col = clipped_local_col + int(gy1)
                start_row = int(round(start_y * 100.0 / map_resolution_cm)) - int(gx1)
                start_col = int(round(start_x * 100.0 / map_resolution_cm)) - int(gy1)
                adapter_projection = {
                    "target_world": target_world[:3].copy(),
                    "raw_local_cell": (int(clipped_local_row), int(clipped_local_col)),
                    "raw_full_cell": (int(selected_full_row), int(selected_full_col)),
                    "target_local_cell": (int(unclipped_local_row), int(unclipped_local_col)),
                    "target_full_cell": (int(full_row), int(full_col)),
                    "start_cell": (
                        int(np.clip(start_row, 0, h - 1)),
                        int(np.clip(start_col, 0, w - 1)),
                    ),
                    "shape": (h, w),
                    "gx1": int(gx1),
                    "gy1": int(gy1),
                    "map_resolution_cm": map_resolution_cm,
                    "start_x": float(start_x),
                    "start_y": float(start_y),
                    "current_world": current_world[:3].copy(),
                    "coordinate_source": "occupancy_adapter",
                    "adapter_map_frame": map_frame,
                    "clipped_to_local_map": bool(clipped_to_local),
                }
                adapter_projection["ok"] = True
                selected_projection, selection_reason = self._select_world_to_local_projection(
                    adapter_projection,
                    fallback_projection,
                )
                if isinstance(selected_projection, dict):
                    selected_projection.setdefault("projection_selection", selection_reason)
                self._trace_world_to_map_projection_debug(
                    context,
                    target_world[:3],
                    current_world[:3],
                    pose_arr[:7],
                    adapter_projection,
                    fallback_projection,
                    selected_projection,
                )
                return selected_projection
        except Exception as exc:
            self.logger.debug("Buffer adapter projection failed: %s", exc)

        self._trace_world_to_map_projection_debug(
            context,
            target_world[:3],
            current_world[:3],
            pose_arr[:7],
            None,
            fallback_projection,
            fallback_projection,
        )
        return fallback_projection

    def _buffer_local_cell_to_world(
        self,
        local_cell: Tuple[int, int],
        projection: Dict[str, Any],
        reference_world: np.ndarray,
    ) -> np.ndarray:
        row, col = int(local_cell[0]), int(local_cell[1])
        full_row = row + int(projection["gx1"])
        full_col = col + int(projection["gy1"])
        try:
            map_frame = projection.get("adapter_map_frame")
            if projection.get("coordinate_source") == "occupancy_adapter":
                if map_frame == "ieve_full_map":
                    adapter_row, adapter_col = full_row, full_col
                else:
                    adapter_row, adapter_col = row, col
                world_x, world_z = self.occupancy_adapter.map_to_world(
                    int(adapter_col), int(adapter_row)
                )
                if np.isfinite(world_x) and np.isfinite(world_z):
                    world = np.asarray(reference_world, dtype=np.float32).reshape(-1)[:3].copy()
                    world[0] = float(world_x)
                    world[2] = float(world_z)
                    return world.astype(np.float32)
        except Exception as exc:
            self.logger.debug("Buffer adapter inverse projection failed: %s", exc)

        res_m = float(projection["map_resolution_cm"]) / 100.0
        target_ieve_y = full_row * res_m
        target_ieve_x = full_col * res_m
        current_world = np.asarray(projection["current_world"], dtype=np.float32)
        world = np.asarray(reference_world, dtype=np.float32).reshape(-1)[:3].copy()
        world[0] = current_world[0] - (target_ieve_y - float(projection["start_y"]))
        world[2] = current_world[2] - (target_ieve_x - float(projection["start_x"]))
        return world.astype(np.float32)

    @staticmethod
    def _buffer_make_goal_map(
        shape: Tuple[int, int], local_cell: Tuple[int, int]
    ) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        row, col = int(local_cell[0]), int(local_cell[1])
        goal_map = np.zeros((h, w), dtype=np.float32)
        if 0 <= row < h and 0 <= col < w:
            goal_map[row, col] = 1.0
        return goal_map

    def _buffer_goal_fmm_diagnostics(
        self, target_world_xyz: Optional[np.ndarray], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            "has_goal": target_world_xyz is not None,
            "reachable": None,
            "reason": "no_current_goal" if target_world_xyz is None else "not_checked",
        }
        if target_world_xyz is None:
            return diag

        context = dict(payload)
        context.setdefault("current_world_xyz", payload.get("agent_position"))
        context.setdefault(
            "map_resolution_cm",
            getattr(self.args, "map_resolution", self.occupancy_adapter.map_resolution),
        )

        projection = self._buffer_project_world_to_local(target_world_xyz, context)
        if not projection or not projection.get("ok"):
            diag.update(
                {
                    "reachable": False,
                    "reason": (projection or {}).get("reason", "projection_failed"),
                    "raw_local_cell": (projection or {}).get("raw_local_cell"),
                    "raw_full_cell": (projection or {}).get("raw_full_cell"),
                }
            )
            return diag

        goal_map = self._buffer_make_goal_map(
            tuple(projection["shape"]),
            tuple(projection["raw_local_cell"]),
        )
        validation = self._ieve_temp_goal_reachability(
            context.get("map_pred"),
            goal_map,
            tuple(projection["start_cell"]),
            projection,
            context,
        )
        diag.update(
            {
                "reachable": bool(validation.get("reachable")),
                "reason": validation.get("reason", "unknown"),
                "raw_local_cell": tuple(projection["raw_local_cell"]),
                "raw_full_cell": tuple(projection["raw_full_cell"]),
                "start_cell": tuple(projection["start_cell"]),
                "validation": validation,
            }
        )
        return diag

    def _prepare_buffer_goal(
        self,
        target_world_xyz: np.ndarray,
        payload: Dict[str, Any],
        source: str,
        search_radius_m: float,
        allow_adjustment: bool = True,
        commit: bool = False,
        avoid_goal_worlds: Optional[List[np.ndarray]] = None,
        avoid_radius_m: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        target = np.asarray(target_world_xyz, dtype=np.float32).reshape(-1)
        if target.size < 3 or not np.all(np.isfinite(target[:3])):
            return None

        context = dict(payload)
        context.setdefault("current_world_xyz", payload.get("agent_position"))
        context.setdefault(
            "map_resolution_cm",
            getattr(self.args, "map_resolution", self.occupancy_adapter.map_resolution),
        )
        agent = context.get("agent_position")
        if agent is None:
            agent = context.get("current_world_xyz")
        agent_arr = None
        if agent is not None:
            try:
                agent_arr = np.asarray(agent, dtype=np.float32).reshape(-1)[:3]
                if agent_arr.size < 3 or not np.all(np.isfinite(agent_arr)):
                    agent_arr = None
            except Exception:
                agent_arr = None

        desired_radius = float(max(0.0, self.buffer_standoff_min_radius))
        max_radius = float(max(desired_radius, self.buffer_standoff_max_radius))
        configured_max_radius = float(search_radius_m)
        if configured_max_radius > 0.0:
            max_radius = min(max_radius, max(desired_radius, configured_max_radius))
        max_extra_radius = float(max_radius - desired_radius)

        base_yaw = 0.0
        if agent_arr is not None:
            delta = agent_arr[[0, 2]] - target[[0, 2]]
            if float(np.linalg.norm(delta)) > 1e-4:
                base_yaw = float(np.arctan2(delta[0], delta[1]))

        candidates: List[Tuple[float, np.ndarray, str]] = []
        if allow_adjustment:
            radii = []
            if max_radius <= desired_radius + 1e-3:
                radius_samples = [desired_radius]
            else:
                radius_samples = np.linspace(desired_radius, max_radius, num=4).tolist()
            for radius in radius_samples:
                if radius > 0 and all(abs(radius - r) > 1e-3 for r in radii):
                    radii.append(radius)
            angle_offsets = [0, -30, 30, -60, 60, -90, 90, 135, -135, 180]
            for radius in radii:
                for offset in angle_offsets:
                    yaw = base_yaw + np.deg2rad(float(offset))
                    cand = target[:3].copy()
                    cand[0] = float(target[0] + radius * np.sin(yaw))
                    cand[2] = float(target[2] + radius * np.cos(yaw))
                    candidates.append((radius, cand.astype(np.float32), "ring"))
        else:
            candidates.append((0.0, target[:3].copy().astype(np.float32), "exact"))

        best = None
        evaluated = 0
        skipped_avoided = 0
        avoid_points = []
        for avoid in avoid_goal_worlds or []:
            try:
                avoid_arr = np.asarray(avoid, dtype=np.float32).reshape(-1)[:3]
                if avoid_arr.size >= 3 and np.all(np.isfinite(avoid_arr)):
                    avoid_points.append(avoid_arr.copy())
            except Exception:
                continue
        if avoid_radius_m is None:
            avoid_radius_m = self.phase2_standoff_retry_block_radius
        avoid_radius_m = max(0.0, float(avoid_radius_m))
        for radius, cand, candidate_source in candidates:
            if avoid_points and avoid_radius_m > 0.0:
                if any(
                    self._world_distance_xz(cand, avoid) <= avoid_radius_m
                    for avoid in avoid_points
                ):
                    skipped_avoided += 1
                    continue
            raw_goal = self._raw_ieve_goal_from_world(cand, context)
            if not raw_goal or not raw_goal.get("ok"):
                continue
            reachability = self._ieve_temp_goal_reachability(
                context.get("map_pred"),
                raw_goal["goal_map"],
                raw_goal["start_cell"],
                raw_goal["projection"],
                context,
            )
            evaluated += 1
            if not reachability.get("reachable"):
                continue
            validation_cost = reachability.get("cost", reachability.get("distance_cm", 0.0))
            try:
                validation_cost = float(validation_cost or 0.0)
            except Exception:
                validation_cost = 0.0
            agent_dist = self._world_distance_xz(agent_arr, cand) if agent_arr is not None else 0.0
            target_dist = self._world_distance_xz(target[:3], cand)
            score = (
                validation_cost
                + 25.0 * abs(target_dist - desired_radius)
                + 5.0 * agent_dist
                + (0.0 if candidate_source == "ring" else 100.0)
            )
            prepared = {
                "goal_world": cand.copy(),
                "target_world": target[:3].copy(),
                "goal_map": raw_goal["goal_map"],
                "local_cell": raw_goal["local_cell"],
                "full_cell": raw_goal["full_cell"],
                "target_full_cell": raw_goal["full_cell"],
                "start_cell": raw_goal["start_cell"],
                "projection": raw_goal["projection"],
                "validation": reachability,
                "cost": float(validation_cost),
                "source": source,
                "candidate_source": candidate_source,
                "search_radius_m": float(search_radius_m),
                "stand_off_radius_m": float(target_dist),
                "raw_to_goal_distance": float(target_dist),
                "score": float(score),
            }
            if best is None or prepared["score"] < best["score"]:
                best = prepared

        self._trace(
            "buffer_goal_prepared",
            step=int(payload.get("step", 0)),
            source=source,
            target_world=target[:3],
            selected=best is not None,
            evaluated_candidates=evaluated,
            skipped_avoided=skipped_avoided,
            candidate_count=len(candidates),
            desired_radius_m=desired_radius,
            search_radius_m=max_radius,
            configured_search_radius_m=configured_max_radius,
            max_extra_radius_m=max_extra_radius,
            standoff_min_radius_m=desired_radius,
            standoff_max_radius_m=max_radius,
            goal_world=None if best is None else best.get("goal_world"),
            local_cell=None if best is None else best.get("local_cell"),
            full_cell=None if best is None else best.get("full_cell"),
            raw_to_goal_distance=None if best is None else best.get("raw_to_goal_distance"),
            validation=None if best is None else best.get("validation"),
            debug_metrics=self._debug_target_navigation_metrics(
                payload,
                confirmed_target=target[:3],
                buffer_goal=None if best is None else best.get("goal_world"),
                detected_target=self.detected_target_position,
            ),
        )

        if best is not None and commit:
            self.buffer_current_goal_world = np.asarray(best["goal_world"], dtype=np.float32).copy()
            self.navigation_goal_world = self.buffer_current_goal_world.copy()
            self.navigation_goal_raw_world = target[:3].copy()
            self.navigation_goal_local_cell = best["local_cell"]
            self.navigation_goal_full_cell = best["full_cell"]
            self.navigation_raw_target_full_cell = best.get(
                "target_full_cell", best["full_cell"]
            )
            self.navigation_goal_adjusted = True
        return best

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------
    def _record_detection(self, payload: Dict[str, Any]) -> None:
        instances = payload.get("minav_target_instances")
        instance_source = "minav_sem_labels"
        if not instances:
            instances = payload.get("target_instances") or []
            instance_source = "ieve_target_instances"
        best_conf = 0.0
        if instances:
            best_conf = max(float(inst.get("confidence", 0.0) or 0.0) for inst in instances)
        if best_conf > self.detect_conf_threshold:
            detected_position = self._estimate_detection_position(payload, instances)
            if detected_position is not None and self._is_blacklisted(
                detected_position, int(payload.get("step", 0))
            ):
                self.conf_history.clear()
                self.frame_window.clear()
                self.detected_target_position = None
                self._trace(
                    "detection_blacklisted",
                    step=int(payload.get("step", 0)),
                    detected_target_position=np.asarray(
                        detected_position, dtype=np.float32
                    ),
                    blacklist_radius=self.blacklist_radius,
                    blacklist_cooldown=self.blacklist_cooldown,
                )
                self._trace(
                    "detection_window_update",
                    step=int(payload.get("step", 0)),
                    instance_count=len(instances),
                    instance_source=instance_source,
                    best_conf=best_conf,
                    threshold=self.detect_conf_threshold,
                    conf_window=list(self.conf_history),
                    frame_window=len(self.frame_window),
                    detected_target_position=self.detected_target_position,
                    blacklisted=True,
                )
                return
            if detected_position is not None:
                self.detected_target_position = detected_position.astype(np.float32)
            self.conf_history.append(best_conf)
            entry = dict(payload)
            entry["slam_camera"] = self._latest_slam_camera()
            entry["target_instances"] = self._copy_target_instances(instances)
            entry["detected_target_position"] = (
                None
                if detected_position is None
                else np.asarray(detected_position, dtype=np.float32).copy()
            )
            self.frame_window.append(entry)
        else:
            self.conf_history.clear()
            self.frame_window.clear()
            self.detected_target_position = None
        self._trace(
            "detection_window_update",
            step=int(payload.get("step", 0)),
            instance_count=len(instances),
            instance_source=instance_source,
            best_conf=best_conf,
            threshold=self.detect_conf_threshold,
            conf_window=list(self.conf_history),
            frame_window=len(self.frame_window),
            detected_target_position=self.detected_target_position,
            blacklisted=False,
        )

    def _estimate_detection_position(
        self,
        payload: Dict[str, Any],
        instances: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        if not instances:
            return None
        best = max(
            instances,
            key=lambda inst: (
                float(inst.get("confidence", 0.0) or 0.0),
                int(inst.get("pixel_count", 0) or 0),
            ),
        )
        bbox = best.get("bbox")
        if bbox is None and best.get("binary_mask") is not None:
            bbox = self._bbox_from_mask(best.get("binary_mask"))
        depth = payload.get("depth_m")
        if depth is None:
            depth = payload.get("raw_depth")
        position = payload.get("agent_position")
        rotation = payload.get("agent_rotation_xyzw")
        return self._target_world_from_detection_bbox(
            bbox=bbox,
            depth=depth,
            position=position,
            rotation=rotation,
        )

    @staticmethod
    def _bbox_from_mask(mask) -> Optional[np.ndarray]:
        mask_np = np.asarray(mask)
        if mask_np.ndim != 2:
            return None
        ys, xs = np.where(mask_np > 0)
        if xs.size == 0 or ys.size == 0:
            return None
        return np.asarray(
            [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
            dtype=np.float32,
        )

    def _target_world_from_detection_bbox(
        self,
        *,
        bbox,
        depth,
        position,
        rotation,
    ) -> Optional[np.ndarray]:
        if bbox is None or depth is None or position is None or rotation is None:
            return None
        try:
            bbox_np = np.asarray(bbox, dtype=np.float32).reshape(-1)
            depth_np = np.asarray(depth, dtype=np.float32)
            if depth_np.ndim == 3:
                depth_np = depth_np[..., 0]
            if bbox_np.size < 4 or depth_np.ndim != 2:
                return None
            height, width = depth_np.shape[:2]
            cx_pix = int((float(bbox_np[0]) + float(bbox_np[2])) * 0.5)
            cy_pix = int((float(bbox_np[1]) + float(bbox_np[3])) * 0.5)
            cx_pix = int(np.clip(cx_pix, 0, max(width - 1, 0)))
            cy_pix = int(np.clip(cy_pix, 0, max(height - 1, 0)))
            x1 = max(0, cx_pix - 10)
            x2 = min(width, cx_pix + 10)
            y1 = max(0, cy_pix - 10)
            y2 = min(height, cy_pix + 10)
            depth_region = depth_np[y1:y2, x1:x2]
            valid_depths = depth_region[
                (depth_region > 0.1)
                & (depth_region < float(getattr(self.args, "max_depth", 10.0)))
                & np.isfinite(depth_region)
            ]
            if valid_depths.size == 0:
                return None
            target_depth = float(np.median(valid_depths))

            camera_params = getattr(self.slam_mapper, "camera_params", {}) or {}
            fx = float(camera_params.get("fx", 0.0) or 0.0)
            fy = float(camera_params.get("fy", 0.0) or 0.0)
            pp_x = float(camera_params.get("cx", width * 0.5))
            pp_y = float(camera_params.get("cy", height * 0.5))
            if fx <= 0.0 or fy <= 0.0:
                fov = np.deg2rad(float(getattr(self.args, "hfov", 58.0)))
                fx = width / (2.0 * np.tan(fov * 0.5))
                fy = fx

            cam_z = target_depth
            cam_x = (float(cx_pix) - pp_x) * cam_z / fx
            cam_y = (float(cy_pix) - pp_y) * cam_z / fy
            point_cam = np.asarray([cam_x, cam_y, cam_z], dtype=np.float32)
            camera_to_robot = np.asarray(
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
                dtype=np.float32,
            )
            point_robot = camera_to_robot @ point_cam
            point_robot[1] += float(
                getattr(self.args, "minav_confirm_camera_height", 0.88)
            )
            rot_matrix = self._quat_xyzw_to_matrix(rotation)
            pos = np.asarray(position, dtype=np.float32).reshape(-1)
            if rot_matrix is None or pos.size < 3:
                return None
            point_world = rot_matrix @ point_robot + pos[:3]
            if not np.all(np.isfinite(point_world[:3])):
                return None
            return point_world.astype(np.float32)
        except Exception as exc:
            self.logger.debug("target-confirm detection position failed: %s", exc)
            return None

    @staticmethod
    def _quat_xyzw_to_matrix(rotation) -> Optional[np.ndarray]:
        quat = np.asarray(rotation, dtype=np.float32).reshape(-1)
        if quat.size < 4 or not np.all(np.isfinite(quat[:4])):
            return None
        try:
            from scipy.spatial.transform import Rotation as R

            return R.from_quat(quat[:4]).as_matrix().astype(np.float32)
        except Exception:
            x, y, z, w = [float(v) for v in quat[:4]]
            norm = np.sqrt(x * x + y * y + z * z + w * w)
            if norm <= 0.0:
                return None
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
            return np.asarray(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ],
                dtype=np.float32,
            )

    @staticmethod
    def _copy_target_instances(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        copied = []
        for inst in instances or []:
            item = {}
            for key, value in inst.items():
                if key in ("binary_mask", "bbox"):
                    item[key] = np.asarray(value).copy()
                else:
                    item[key] = value
            copied.append(item)
        return copied

    def _ready_to_confirm(self) -> bool:
        if len(self.conf_history) < self.detect_window:
            return False
        return all(conf > self.detect_conf_threshold for conf in self.conf_history)

    def _clear_detection_window(self) -> None:
        self.conf_history.clear()
        self.frame_window.clear()

    def _enter_navigation_from_confirmation(
        self, result: Dict[str, Any], step: int, payload: Dict[str, Any]
    ) -> None:
        target_center = np.asarray(
            result["target_cluster_center"], dtype=np.float32
        ).reshape(-1)[:3]
        target_goal_anchor, anchor_debug = self._target_goal_anchor_from_confirm_result(
            result,
            target_center,
            payload,
        )
        reachability = self._select_reachable_navigation_goal(
            target_goal_anchor, payload, phase3=False
        )
        if not reachability.get("selected"):
            reason = str(
                reachability.get("reason", "navigation_goal_unavailable")
            )
            self._trace(
                "confirmed_target_unreachable_by_ieve_rule",
                step=step,
                reason=reason,
                target_position=target_center,
                target_map_anchor_position=target_goal_anchor,
                target_map_anchor=anchor_debug,
                reachability={
                    key: value
                    for key, value in reachability.items()
                    if key not in ("goal_map", "goal_region_mask", "raw_goal")
                },
                rule="ieve_select_reachable_goal",
            )
            self.confirmed_target_position = target_center.copy()
            self.confirmed_target_map_anchor_position = target_goal_anchor.copy()
            self._add_blacklist(target_center, step, reason)
            self._fail_to_exploration(step, reason, blacklist=False)
            self._clear_detection_window()
            return

        indices = result.get("best_cluster_global_indices")
        self.confirmed_target_position = target_center.copy()
        self.confirmed_target_map_anchor_position = target_goal_anchor.copy()
        self.confirmation_detected_target_position = (
            None
            if self.detected_target_position is None
            else np.asarray(self.detected_target_position, dtype=np.float32)
            .reshape(-1)[:3]
            .copy()
        )
        self.confirmed_indices = (
            np.asarray(indices, dtype=np.int64).reshape(-1)
            if indices is not None
            else None
        )
        self.localized_target_position = None
        self.buffer_detection_target_position = target_center.copy()
        self.last_confirm_result = result
        self.last_localization_result = None
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self._commit_reachable_navigation_goal(reachability, step)
        self.navigation_complete = False
        self.mode = IEVETargetMode.NAVIGATION
        self._trace(
            "mode_transition",
            step=step,
            from_mode=IEVETargetMode.TARGET_CONFIRMATION.value,
            to_mode=IEVETargetMode.NAVIGATION.value,
            reason="target_confirmed_ieve_reachable",
            target_position=self.confirmed_target_position,
            target_map_anchor_position=self.confirmed_target_map_anchor_position,
            target_map_anchor=anchor_debug,
            navigation_goal_world=self.navigation_goal_world,
            local_cell=self.navigation_goal_local_cell,
            full_cell=self.navigation_goal_full_cell,
            goal_tier=self.navigation_goal_tier,
            effective_stop_radius_m=self.navigation_effective_stop_radius_m,
            reachability={
                key: value
                for key, value in reachability.items()
                if key not in ("goal_map", "goal_region_mask", "raw_goal")
            },
            locked_gaussians=0
            if self.confirmed_indices is None
            else int(self.confirmed_indices.size),
            best_score=result.get("best_score"),
            best_match_count=result.get("best_match_count"),
            contract="minav_confirms_ieve_executes_reachable_2d_goal",
        )
        self._clear_detection_window()

    def _enter_buffer_from_confirmation(
        self, result: Dict[str, Any], step: int, payload: Dict[str, Any]
    ) -> None:
        from_mode = self.mode.value
        self._clear_cluster_recovery_state(reset_used=True)
        target_center = np.asarray(
            result["target_cluster_center"], dtype=np.float32
        ).reshape(-1)[:3]
        target_goal_anchor, anchor_debug = self._target_goal_anchor_from_confirm_result(
            result,
            target_center,
            payload,
        )
        raw_goal = self._raw_ieve_goal_from_world(target_goal_anchor, payload)
        reachability = None
        if raw_goal and raw_goal.get("ok"):
            reachability = self._ieve_temp_goal_reachability(
                payload.get("map_pred"),
                raw_goal["goal_map"],
                raw_goal["start_cell"],
                raw_goal["projection"],
                payload,
            )
        if (
            not raw_goal
            or not raw_goal.get("ok")
            or not (reachability or {}).get("reachable")
        ):
            reason = (
                (raw_goal or {}).get("reason")
                or (reachability or {}).get("reason")
                or "ieve_temp_goal_unreachable"
            )
            self._trace(
                "confirmed_target_center_not_directly_reachable",
                step=step,
                reason=reason,
                target_position=target_center,
                target_map_anchor_position=target_goal_anchor,
                target_map_anchor=anchor_debug,
                raw_goal=raw_goal,
                reachability=reachability,
                rule="ieve_compute_temp_goal_distance",
                requested_phase=IEVETargetMode.BUFFER.value,
                contract="buffer_uses_reachable_standoff_even_if_target_center_is_not_reachable",
            )

        indices = result.get("best_cluster_global_indices")
        self.confirmed_target_position = target_center.copy()
        self.confirmed_target_map_anchor_position = target_goal_anchor.copy()
        self.confirmation_detected_target_position = (
            None
            if self.detected_target_position is None
            else np.asarray(self.detected_target_position, dtype=np.float32)
            .reshape(-1)[:3]
            .copy()
        )
        self.confirmed_indices = (
            np.asarray(indices, dtype=np.int64).reshape(-1)
            if indices is not None
            else None
        )
        self.localized_target_position = None
        self.buffer_detection_target_position = target_center.copy()
        self.last_confirm_result = result
        self.last_localization_result = None
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()

        prepared = self._prepare_buffer_goal(
            target_goal_anchor,
            payload,
            source="MINAV_BUFFER_FROM_CONFIRMATION_STANDOFF",
            search_radius_m=self.buffer_goal_search_radius,
            allow_adjustment=True,
            commit=True,
        )
        if prepared is None:
            self.buffer_current_goal_world = None
            self.navigation_goal_world = None
            self.navigation_goal_raw_world = target_goal_anchor.copy()
            self.navigation_goal_local_cell = None
            self.navigation_goal_full_cell = None
            self.navigation_raw_target_full_cell = None
            self.navigation_goal_adjusted = False

        prepared_summary = None
        if prepared is not None:
            prepared_summary = {
                "source": prepared.get("source"),
                "candidate_source": prepared.get("candidate_source"),
                "goal_world": prepared.get("goal_world"),
                "target_world": prepared.get("target_world"),
                "local_cell": prepared.get("local_cell"),
                "full_cell": prepared.get("full_cell"),
                "stand_off_radius_m": prepared.get("stand_off_radius_m"),
                "raw_to_goal_distance": prepared.get("raw_to_goal_distance"),
                "score": prepared.get("score"),
                "validation": prepared.get("validation"),
            }

        self.buffer_direct_nav_to_confirmed_target = True
        self.buffer_start_step = int(step)
        self.buffer_frame_counter = 0
        self.buffer_phase = 1
        self.buffer_goal_update_counter = 0
        self.buffer_gaussmi_signaled = False
        self.buffer_gaussmi_signaled_step = -1
        self.buffer_gaussmi_signaled_frame = -1
        self._clear_buffer_standoff_sweep_state()
        self.localization_signal_frame_counter = -1
        self.localization_stage = "idle"
        self.localization_frame_counter = 0
        self.localization_trace = []
        self.navigation_complete = False
        self.mode = IEVETargetMode.BUFFER

        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.BUFFER.value,
            reason="target_confirmed_enter_buffer_for_localization",
            target_position=self.confirmed_target_position,
            target_map_anchor_position=self.confirmed_target_map_anchor_position,
            target_map_anchor=anchor_debug,
            buffer_goal_world=self.buffer_current_goal_world,
            navigation_goal_world=self.navigation_goal_world,
            local_cell=self.navigation_goal_local_cell,
            full_cell=self.navigation_goal_full_cell,
            navigation_goal_adjusted=self.navigation_goal_adjusted,
            reachability=reachability,
            buffer_goal_prepared=prepared_summary,
            locked_gaussians=0
            if self.confirmed_indices is None
            else int(self.confirmed_indices.size),
            best_score=result.get("best_score"),
            best_match_count=result.get("best_match_count"),
            contract="minav_confirms_ieve_buffers_then_localizes_before_navigation",
        )
        self._clear_detection_window()

    def _enter_localization_from_phase_goal(
        self, payload: Dict[str, Any], step: int
    ) -> None:
        from_mode = self.mode.value
        executor_goal_world = (
            self.navigation_goal_world
            if self.navigation_goal_world is not None
            else self._confirmed_target_goal_anchor()
        )
        localization_target_world = (
            self.confirmed_target_position
            if self.confirmed_target_position is not None
            else executor_goal_world
        )
        if localization_target_world is None:
            self._trace(
                "phase_goal_reached_ignored",
                step=step,
                reason="no_confirmed_or_navigation_target",
                minav_fmm_stop=payload.get("minav_fmm_stop"),
            )
            self.buffer_direct_nav_to_confirmed_target = False
            self.mode = IEVETargetMode.EXPLORATION
            return

        self.buffer_current_goal_world = np.asarray(
            localization_target_world, dtype=np.float32
        ).reshape(-1)[:3].copy()
        self.buffer_detection_target_position = (
            None
            if self.confirmed_target_position is None
            else np.asarray(self.confirmed_target_position, dtype=np.float32)
            .reshape(-1)[:3]
            .copy()
        )
        self.buffer_gaussmi_signaled = True
        self.buffer_gaussmi_signaled_step = int(step)
        self.buffer_gaussmi_signaled_frame = int(max(0, self.buffer_frame_counter))
        self.localization_signal_frame_counter = 0
        self.localization_frame_counter = 0
        self.localization_stage = "SOIM"
        self.navigation_complete = False
        self.buffer_direct_nav_to_confirmed_target = False
        self.mode = IEVETargetMode.LOCALIZATION

        if self.full_localizer is not None:
            self._notify_full_localizer_gaussmi_ready(step)

        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.LOCALIZATION.value,
            reason="ieve_fmm_phase_goal_reached",
            target_position=self.buffer_current_goal_world,
            confirmed_target_position=self.confirmed_target_position,
            executor_goal_world=executor_goal_world,
            navigation_goal_world=self.navigation_goal_world,
            navigation_goal_local_cell=self.navigation_goal_local_cell,
            debug_metrics=self._debug_target_navigation_metrics(
                payload,
                confirmed_target=self.confirmed_target_position,
                buffer_goal=executor_goal_world,
                detected_target=self.confirmation_detected_target_position,
            ),
            minav_fmm_stop=payload.get("minav_fmm_stop"),
            minav_found_goal=payload.get("minav_found_goal"),
            contract=(
                "fmm_stop_is_internal_phase_transition_not_habitat_final_stop"
            ),
        )
        if self._debug_enabled():
            self._trace(
                "target_navigation_geometry_debug",
                step=step,
                phase="phase_goal_reached_enter_localization",
                metrics=self._debug_target_navigation_metrics(
                    payload,
                    confirmed_target=self.confirmed_target_position,
                    buffer_goal=executor_goal_world,
                    detected_target=self.confirmation_detected_target_position,
                ),
            )

    def _clear_mr_remap_state(self) -> None:
        self.mr_remap_pending = False
        self.mr_remap_sweep_active = False
        self.mr_remap_sweep_just_completed = False
        self.mr_remap_sweep_remaining_steps = 0
        self.mr_remap_sweep_total_steps = 0
        self.mr_remap_sweep_direction = "turn_left"

    def _fail_to_exploration(self, step: int, reason: str, blacklist: bool) -> None:
        from_mode = self.mode.value
        if self._maybe_activate_strategy_phase3_fallback_stop(step, reason):
            return
        if (
            self.targeted_current_semantic_position is not None
            and from_mode
            in (
                IEVETargetMode.BUFFER.value,
                IEVETargetMode.LOCALIZATION.value,
                IEVETargetMode.NAVIGATION.value,
                IEVETargetMode.TARGETED_EXPLORATION.value,
            )
        ):
            self._targeted_register_failed_region(
                self.targeted_current_semantic_position,
                step,
                reason,
            )
        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.EXPLORATION.value,
            reason=reason,
            blacklist=bool(blacklist),
            confirmed=self.confirmed_target_position is not None,
            localized=self.localized_target_position is not None,
        )
        if blacklist and self.confirmed_target_position is not None:
            self._add_blacklist(self.confirmed_target_position, step, reason)
        self.mode = IEVETargetMode.EXPLORATION
        self.confirmed_target_position = None
        self.confirmed_target_map_anchor_position = None
        self.confirmed_indices = None
        self.localized_target_position = None
        self.confirmation_detected_target_position = None
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self.buffer_direct_nav_to_confirmed_target = False
        self.buffer_start_step = -1
        self.buffer_frame_counter = 0
        self.buffer_phase = 1
        self.buffer_goal_update_counter = 0
        self.buffer_current_goal_world = None
        self.buffer_detection_target_position = None
        self.buffer_gaussmi_signaled = False
        self.buffer_gaussmi_signaled_step = -1
        self.buffer_gaussmi_signaled_frame = -1
        self._clear_buffer_standoff_sweep_state()
        self.localization_signal_frame_counter = -1
        self.localization_stage = "idle"
        self.localization_frame_counter = 0
        self.localization_trace = []
        self.navigation_complete = False
        self.full_localizer = None
        self.full_localizer_started = False
        self.full_localizer_error = None
        self.pending_closed_loop_pose = None
        self.pending_closed_loop_score = None
        self._clear_mr_remap_state()
        self._clear_detection_window()
        self._clear_cluster_recovery_state(reset_used=True)
        self._reset_slam_target_semantic_state(step, reason)

    def _enter_buffer_reposition(self, loc_result: Dict[str, Any], step: int) -> None:
        target = loc_result.get("target_position")
        if target is None:
            self._fail_to_exploration(step, "mr_reposition_without_target", blacklist=True)
            return
        from_mode = self.mode.value
        target_arr = np.asarray(target, dtype=np.float32).reshape(-1)[:3]
        self.confirmed_target_position = target_arr.copy()
        self.confirmed_target_map_anchor_position = target_arr.copy()
        self.buffer_detection_target_position = target_arr.copy()
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self.buffer_phase = 1
        self.buffer_frame_counter = 0
        self.buffer_goal_update_counter = 0
        self.buffer_current_goal_world = target_arr.copy()
        new_pose = loc_result.get("new_pose")
        self.pending_closed_loop_pose = (
            np.asarray(new_pose, dtype=np.float64).copy()
            if new_pose is not None
            else None
        )
        self.pending_closed_loop_score = float(
            loc_result.get("best_score", self.localization_match_threshold) or 0.0
        )
        self.buffer_gaussmi_signaled = False
        self.buffer_gaussmi_signaled_step = -1
        self.buffer_gaussmi_signaled_frame = -1
        self._clear_buffer_standoff_sweep_state()
        self.localization_signal_frame_counter = -1
        self.localization_stage = "MR_REPOSITION"
        self.localization_frame_counter = 0
        self.full_localizer_started = False
        self._sync_full_localizer_candidate_anchor(
            self.full_localizer,
            step,
            source="mr_reposition",
        )
        self.mr_remap_pending = True
        self.mr_remap_sweep_active = False
        self.mr_remap_sweep_just_completed = False
        self.mr_remap_sweep_remaining_steps = 0
        self.mr_remap_sweep_total_steps = self._full_turn_sweep_steps()
        self.mr_remap_sweep_direction = "turn_left"
        self.mode = IEVETargetMode.BUFFER
        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.BUFFER.value,
            reason="mr_reposition",
            target_position=self.buffer_current_goal_world,
            best_score=loc_result.get("best_score"),
            phase=1,
            mr_remap_sweep_steps=self.mr_remap_sweep_total_steps,
        )

    def _reset_slam_target_semantic_state(self, step: int, reason: str) -> None:
        gaussians = None
        if self.slam_mapper is not None:
            slam = getattr(self.slam_mapper, "slam", None)
            gaussians = getattr(slam, "gaussians", None) if slam is not None else None
            if gaussians is None and hasattr(self.slam_mapper, "_gaussians"):
                try:
                    gaussians = self.slam_mapper._gaussians()
                except Exception:
                    gaussians = None
        if gaussians is None:
            self._trace(
                "slam_target_semantic_reset_skipped",
                step=step,
                reason=reason,
                detail="gaussians_unavailable",
            )
            return

        reset_semantic = False
        try:
            if hasattr(gaussians, "reset_semantic_target"):
                gaussians.reset_semantic_target()
                reset_semantic = True
        except Exception as exc:
            self.logger.debug("reset_semantic_target failed: %s", exc)

        for attr in (
            "_confirmed_cluster_locked",
            "_confirmed_target_indices",
            "_last_confirmed_cluster_indices",
        ):
            if hasattr(gaussians, attr):
                try:
                    setattr(
                        gaussians,
                        attr,
                        False if attr == "_confirmed_cluster_locked" else None,
                    )
                except Exception:
                    pass

        self._trace(
            "slam_target_semantic_reset",
            step=step,
            reason=reason,
            reset_semantic=reset_semantic,
            cleared_confirmed_lock=hasattr(gaussians, "_confirmed_cluster_locked"),
        )

    def _clear_cluster_recovery_state(self, reset_used: bool = False) -> None:
        if reset_used:
            self.cluster_recovery_used_in_chain = False
        self.cluster_recovery_center = None
        self.cluster_recovery_center_full_cell = None
        self.cluster_recovery_original_indices = None
        self.cluster_recovery_base_pose = None
        self.cluster_recovery_trigger = {}
        self.cluster_recovery_active_signature = None
        self.cluster_recovery_route_cells = []
        self.cluster_recovery_route_segments = []
        self.cluster_recovery_waypoint_cells = []
        self.cluster_recovery_waypoints_world = []
        self.cluster_recovery_waypoint_index = 0
        self.cluster_recovery_start_step = -1
        self.cluster_recovery_frame_counter = 0
        self.cluster_recovery_waypoint_best_distance = float("inf")
        self.cluster_recovery_waypoint_stall_frames = 0
        self.cluster_recovery_replan_count = 0
        self.cluster_recovery_replan_requested_reason = None
        self.cluster_recovery_approach_active = False
        self.cluster_recovery_approach_goal_cell = None
        self.cluster_recovery_approach_for_replan = False
        self.cluster_recovery_approach_reason = None
        self.cluster_recovery_candidates = []
        if reset_used:
            self.cluster_recovery_last_result = None

    @staticmethod
    def _validate_cluster_recovery_closed_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
        route = np.asarray(plan.get("route_cells", []), dtype=np.float64)
        center = np.asarray(plan.get("center_cell", []), dtype=np.float64).reshape(-1)
        if route.ndim != 2 or route.shape[0] < 4 or route.shape[1] != 2:
            return {"valid": False, "reason": "closed_route_too_short"}
        if center.size < 2:
            return {"valid": False, "reason": "closed_route_missing_center"}
        if not np.array_equal(route[0].astype(np.int64), route[-1].astype(np.int64)):
            return {"valid": False, "reason": "route_not_closed"}
        route_angles = np.unwrap(
            np.arctan2(route[:, 0] - center[0], route[:, 1] - center[1])
        )
        winding = float((route_angles[-1] - route_angles[0]) / (2.0 * np.pi))
        if not np.isfinite(winding) or abs(abs(winding) - 1.0) > 0.20:
            return {
                "valid": False,
                "reason": "route_winding_not_one",
                "winding": winding,
            }
        return {"valid": True, "reason": None, "winding": winding}

    def _install_cluster_recovery_closed_plan(self, plan: Dict[str, Any]) -> bool:
        validation = self._validate_cluster_recovery_closed_plan(plan)
        if not validation.get("valid"):
            return False
        self.cluster_recovery_route_cells = list(plan["route_cells"])
        self.cluster_recovery_route_segments = [
            list(segment) for segment in plan["route_segments"]
        ]
        self.cluster_recovery_waypoint_cells = list(plan["waypoint_cells"])
        self.cluster_recovery_waypoints_world = []
        self.cluster_recovery_center_full_cell = tuple(
            int(v) for v in plan["center_cell"]
        )
        self.cluster_recovery_waypoint_index = 0
        self.cluster_recovery_waypoint_best_distance = float("inf")
        self.cluster_recovery_waypoint_stall_frames = 0
        self.cluster_recovery_replan_requested_reason = None
        return True

    def _cluster_recovery_cluster_signature(self) -> Optional[Dict[str, Any]]:
        if self.confirmed_indices is None or self.confirmed_target_position is None:
            return None
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None or gaussians is None:
            return None
        try:
            xyz = getattr(gaussians, "get_xyz", None)
            if callable(xyz):
                xyz = xyz()
            if hasattr(xyz, "detach"):
                xyz = xyz.detach().cpu().numpy()
            xyz = np.asarray(xyz, dtype=np.float32)
            indices = np.asarray(self.confirmed_indices, dtype=np.int64).reshape(-1)
            indices = indices[(indices >= 0) & (indices < xyz.shape[0])]
            if indices.size == 0:
                return None
            points = xyz[indices, :3]
            voxel_size = self.cluster_recovery_failure_cluster_voxel_size_m
            voxels = np.unique(np.rint(points / voxel_size).astype(np.int32), axis=0)
            if voxels.size == 0:
                return None
            return {
                "center": np.asarray(
                    self.confirmed_target_position, dtype=np.float32
                ).reshape(-1)[:3].copy(),
                "voxels": frozenset(tuple(int(v) for v in row) for row in voxels),
                "gaussian_count": int(indices.size),
            }
        except Exception:
            return None

    def _cluster_recovery_map_snapshot(
        self, payload: Dict[str, Any], center_world: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        obstacle = payload.get("full_map_pred")
        explored = payload.get("full_exp_pred")
        if obstacle is None or explored is None:
            return None
        obstacle = np.asarray(obstacle, dtype=np.float32)
        explored = np.asarray(explored, dtype=np.float32)
        if obstacle.ndim != 2 or explored.shape != obstacle.shape:
            return None
        projection = self._buffer_project_world_to_local(center_world, payload)
        if not projection:
            return None
        center_cell = projection.get(
            "target_full_cell", projection.get("raw_full_cell")
        )
        if center_cell is None:
            return None
        center_row, center_col = (int(center_cell[0]), int(center_cell[1]))
        resolution_m = float(
            payload.get("map_resolution_cm", getattr(self.args, "map_resolution", 5.0))
        ) / 100.0
        radius = int(
            np.ceil(self.cluster_recovery_max_radius_m / max(resolution_m, 1e-6))
        )
        offsets = np.arange(-radius, radius + 1, dtype=np.int32)
        rows = center_row + offsets[:, None]
        cols = center_col + offsets[None, :]
        disk = np.hypot(offsets[:, None], offsets[None, :]) <= radius + 1e-6
        valid = (
            (rows >= 0)
            & (rows < obstacle.shape[0])
            & (cols >= 0)
            & (cols < obstacle.shape[1])
            & disk
        )
        sampled_explored = np.zeros(valid.shape, dtype=bool)
        sampled_blocked = np.zeros(valid.shape, dtype=bool)
        rr = np.broadcast_to(rows, valid.shape)[valid]
        cc = np.broadcast_to(cols, valid.shape)[valid]
        sampled_explored[valid] = explored[rr, cc] > 0.0
        sampled_blocked[valid] = obstacle[rr, cc] >= 0.5
        return {
            "center_cell": (center_row, center_col),
            "valid": valid,
            "explored": sampled_explored,
            "blocked": sampled_blocked,
        }

    def _same_cluster_recovery_failure_signature(
        self, cached: Dict[str, Any], current: Dict[str, Any]
    ) -> bool:
        center_distance = self._world_distance_xz(
            cached.get("center"), current.get("center")
        )
        if center_distance > self.cluster_recovery_failure_cluster_center_tolerance_m:
            return False
        cached_voxels = cached.get("voxels") or frozenset()
        current_voxels = current.get("voxels") or frozenset()
        if not cached_voxels or not current_voxels:
            return False
        overlap = len(cached_voxels & current_voxels) / max(
            min(len(cached_voxels), len(current_voxels)), 1
        )
        return overlap >= self.cluster_recovery_failure_cluster_overlap_ratio

    def _cluster_recovery_map_changed_significantly(
        self,
        cached_snapshot: Optional[Dict[str, Any]],
        current_snapshot: Optional[Dict[str, Any]],
    ) -> bool:
        if cached_snapshot is None or current_snapshot is None:
            return False
        cached_valid = np.asarray(cached_snapshot.get("valid"), dtype=bool)
        current_valid = np.asarray(current_snapshot.get("valid"), dtype=bool)
        if cached_valid.shape != current_valid.shape:
            return True
        valid = cached_valid & current_valid
        valid_count = int(valid.sum())
        if valid_count <= 0:
            return True
        changed = (
            np.asarray(cached_snapshot.get("explored"), dtype=bool)
            ^ np.asarray(current_snapshot.get("explored"), dtype=bool)
        ) | (
            np.asarray(cached_snapshot.get("blocked"), dtype=bool)
            ^ np.asarray(current_snapshot.get("blocked"), dtype=bool)
        )
        return (
            float(np.count_nonzero(changed & valid)) / float(valid_count)
            >= self.cluster_recovery_failure_map_change_ratio
        )

    def _cached_cluster_recovery_route_failure(
        self, payload: Dict[str, Any], signature: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if signature is None:
            return None
        current_snapshot = self._cluster_recovery_map_snapshot(
            payload, signature["center"]
        )
        for cached in list(self.cluster_recovery_route_failure_cache):
            cached_signature = cached.get("cluster_signature") or {}
            if not self._same_cluster_recovery_failure_signature(
                cached_signature, signature
            ):
                continue
            if self._cluster_recovery_map_changed_significantly(
                cached.get("map_snapshot"), current_snapshot
            ):
                self.cluster_recovery_route_failure_cache.remove(cached)
                return None
            return cached
        return None

    def _cache_cluster_recovery_route_failure(
        self,
        payload: Dict[str, Any],
        signature: Optional[Dict[str, Any]],
        reason: str,
        step: int,
    ) -> None:
        if signature is None:
            return
        map_snapshot = self._cluster_recovery_map_snapshot(
            payload, signature["center"]
        )
        if map_snapshot is None:
            return
        self.cluster_recovery_route_failure_cache = [
            cached
            for cached in self.cluster_recovery_route_failure_cache
            if not self._same_cluster_recovery_failure_signature(
                cached.get("cluster_signature") or {}, signature
            )
        ]
        self.cluster_recovery_route_failure_cache.append(
            {
                "cluster_signature": signature,
                "map_snapshot": map_snapshot,
                "reason": str(reason),
                "step": int(step),
            }
        )
        self.cluster_recovery_route_failure_cache = (
            self.cluster_recovery_route_failure_cache[-32:]
        )

    def _cluster_recovery_estimated_start_cell(
        self, payload: Dict[str, Any], shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        pose_pred = payload.get("pose_pred")
        if pose_pred is None:
            return None
        try:
            pose = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
            if pose.size < 2 or not np.all(np.isfinite(pose[:2])):
                return None
            resolution_cm = float(
                payload.get(
                    "map_resolution_cm", getattr(self.args, "map_resolution", 5.0)
                )
            )
            row = int(round(float(pose[1]) * 100.0 / resolution_cm))
            col = int(round(float(pose[0]) * 100.0 / resolution_cm))
            if not (0 <= row < int(shape[0]) and 0 <= col < int(shape[1])):
                return None
            return row, col
        except Exception:
            return None

    def _cluster_recovery_center_cell(
        self,
        payload: Dict[str, Any],
        center_world: np.ndarray,
        shape: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        projection = self._buffer_project_world_to_local(center_world, payload)
        if not projection:
            return None
        cell = projection.get("target_full_cell", projection.get("raw_full_cell"))
        if cell is None:
            return None
        row, col = int(cell[0]), int(cell[1])
        if not (0 <= row < int(shape[0]) and 0 <= col < int(shape[1])):
            return None
        return row, col

    def _select_cluster_recovery_approach_goal(
        self, payload: Dict[str, Any], center_world: np.ndarray
    ) -> Dict[str, Any]:
        obstacle = payload.get("full_map_pred")
        explored = payload.get("full_exp_pred")
        if obstacle is None or explored is None:
            return {"success": False, "reason": "missing_full_map"}
        obstacle = np.asarray(obstacle, dtype=np.float32)
        explored = np.asarray(explored, dtype=np.float32)
        if obstacle.ndim != 2 or explored.shape != obstacle.shape:
            return {"success": False, "reason": "invalid_full_map_shape"}
        start_cell = self._cluster_recovery_estimated_start_cell(
            payload, obstacle.shape
        )
        center_cell = self._cluster_recovery_center_cell(
            payload, center_world, obstacle.shape
        )
        if start_cell is None or center_cell is None:
            return {"success": False, "reason": "missing_estimated_map_pose"}

        resolution_m = float(
            payload.get("map_resolution_cm", getattr(self.args, "map_resolution", 5.0))
        ) / 100.0
        start_radius_m = float(
            np.hypot(
                start_cell[0] - center_cell[0],
                start_cell[1] - center_cell[1],
            )
            * resolution_m
        )
        if start_radius_m <= self.cluster_recovery_max_radius_m + 0.5 * resolution_m:
            return {
                "success": True,
                "needed": False,
                "start_cell": start_cell,
                "center_cell": center_cell,
                "start_radius_m": start_radius_m,
            }

        try:
            from skimage.morphology import binary_dilation, disk

            blocked = binary_dilation(obstacle >= 0.5, disk(3))
        except Exception:
            blocked = obstacle >= 0.5
        traversible = (~blocked) & (explored > 0.0)
        traversible[start_cell] = True
        component = self._cluster_recovery_component(traversible, start_cell)
        if component is None:
            return {
                "success": False,
                "reason": "approach_empty_start_component",
                "start_radius_m": start_radius_m,
            }

        rows, cols = np.indices(obstacle.shape)
        radii_m = (
            np.hypot(rows - center_cell[0], cols - center_cell[1]) * resolution_m
        )
        candidates = np.argwhere(
            component
            & (radii_m >= self.cluster_recovery_approach_inner_radius_m)
            & (radii_m <= self.cluster_recovery_approach_outer_radius_m)
        )
        if candidates.size == 0:
            return {
                "success": False,
                "reason": "approach_no_reachable_ingress_in_annulus",
                "start_radius_m": start_radius_m,
            }
        try:
            from envs.utils.fmm_planner import FMMPlanner

            planner = FMMPlanner(np.asarray(component, dtype=np.float32))
            planner.set_goal(start_cell)
            distances = np.asarray(planner.fmm_dist, dtype=np.float64)
            path_costs = distances[candidates[:, 0], candidates[:, 1]]
        except Exception:
            path_costs = np.hypot(
                candidates[:, 0] - start_cell[0],
                candidates[:, 1] - start_cell[1],
            )
        finite = np.isfinite(path_costs)
        if not np.any(finite):
            return {
                "success": False,
                "reason": "approach_ingress_fmm_unreachable",
                "start_radius_m": start_radius_m,
            }
        candidates = candidates[finite]
        path_costs = path_costs[finite]
        preferred_radius = 0.5 * (
            self.cluster_recovery_approach_inner_radius_m
            + self.cluster_recovery_approach_outer_radius_m
        )
        radial_error = np.abs(
            radii_m[candidates[:, 0], candidates[:, 1]] - preferred_radius
        )
        selected_index = int(np.lexsort((radial_error, path_costs))[0])
        goal_cell = tuple(int(v) for v in candidates[selected_index])
        return {
            "success": True,
            "needed": True,
            "goal_cell": goal_cell,
            "start_cell": start_cell,
            "center_cell": center_cell,
            "start_radius_m": start_radius_m,
            "goal_radius_m": float(radii_m[goal_cell]),
            "path_length_m": float(path_costs[selected_index] * resolution_m),
        }

    @staticmethod
    def _cluster_recovery_failure_proves_no_closed_route(reason: str) -> bool:
        return str(reason) not in {
            "start_outside_3m_constraint",
            "missing_estimated_map_pose",
            "world_to_full_projection_failed",
            "center_or_start_outside_full_map",
        } and not str(reason).startswith("approach_")

    def _start_cluster_recovery(
        self, payload: Dict[str, Any], loc_result: Dict[str, Any], step: int
    ) -> None:
        localizer = self.full_localizer
        if (
            not self.cluster_recovery_enabled
            or self.cluster_recovery_used_in_chain
            or localizer is None
        ):
            self._skip_cluster_recovery_to_mr(
                step,
                "disabled_or_already_used",
            )
            return

        center = self._as_world_point(self.confirmed_target_position)
        base_pose = loc_result.get("base_pose")
        try:
            base_pose = np.asarray(base_pose, dtype=np.float64)
        except Exception:
            base_pose = None
        if center is None or base_pose is None or base_pose.shape != (4, 4):
            self._skip_cluster_recovery_to_mr(step, "missing_center_or_base_pose")
            return

        cluster_signature = self._cluster_recovery_cluster_signature()
        cached_failure = self._cached_cluster_recovery_route_failure(
            payload, cluster_signature
        )
        if cached_failure is not None:
            self._trace(
                "cluster_recovery_cached_route_failure",
                step=step,
                reason=cached_failure.get("reason"),
                cached_step=cached_failure.get("step"),
                cluster_center=cluster_signature.get("center")
                if cluster_signature is not None
                else center,
                contract="same_gaussian_cluster_and_unchanged_nearby_map",
            )
            self._skip_cluster_recovery_to_mr(
                step,
                "cached_route_unavailable:{}".format(
                    cached_failure.get("reason", "unknown")
                ),
            )
            return

        max_steps = int(getattr(self.args, "max_episode_length", 1000))
        remaining_steps = max_steps - int(step)
        if remaining_steps < self.cluster_recovery_min_remaining_steps:
            self._skip_cluster_recovery_to_mr(
                step,
                "insufficient_remaining_steps_{}".format(remaining_steps),
            )
            return

        self.localization_stage = "CLUSTER_RECOVERY_PLAN"
        self.occupancy_adapter.update(payload)
        approach = self._select_cluster_recovery_approach_goal(payload, center)
        if not approach.get("success"):
            self._skip_cluster_recovery_to_mr(
                step,
                "approach_unavailable:{}".format(
                    approach.get("reason", "unknown")
                ),
            )
            return
        approach_needed = bool(approach.get("needed", False))
        plan = None
        if not approach_needed:
            plan = self._plan_cluster_recovery_closed_route(payload, center)
            if not plan.get("success"):
                failure_reason = str(plan.get("reason", "unknown"))
                if self._cluster_recovery_failure_proves_no_closed_route(
                    failure_reason
                ):
                    self._cache_cluster_recovery_route_failure(
                        payload,
                        cluster_signature,
                        failure_reason,
                        step,
                    )
                self._skip_cluster_recovery_to_mr(
                    step,
                    "route_unavailable:{}".format(failure_reason),
                )
                return

        estimated_length_m = float(
            approach.get("path_length_m", 0.0)
            if approach_needed
            else plan.get("length_m", 0.0)
        )
        estimated_actions = int(np.ceil(estimated_length_m / 0.25))
        required_remaining = max(
            self.cluster_recovery_min_remaining_steps,
            estimated_actions + 10,
        )
        if remaining_steps < required_remaining:
            self._skip_cluster_recovery_to_mr(
                step,
                "route_budget_{}_lt_{}".format(remaining_steps, required_remaining),
            )
            return

        self._clear_cluster_recovery_state(reset_used=False)
        self.cluster_recovery_used_in_chain = True
        self.cluster_recovery_center = center.copy()
        center_cell = approach.get("center_cell")
        if center_cell is None and plan is not None:
            center_cell = plan.get("center_cell")
        self.cluster_recovery_center_full_cell = (
            None
            if center_cell is None
            else tuple(int(v) for v in center_cell)
        )
        self.cluster_recovery_original_indices = (
            None
            if self.confirmed_indices is None
            else np.asarray(self.confirmed_indices, dtype=np.int64).reshape(-1).copy()
        )
        self.cluster_recovery_base_pose = base_pose.copy()
        self.cluster_recovery_trigger = {
            "source": loc_result.get("trigger_source"),
            "score": float(loc_result.get("trigger_score", 0.0) or 0.0),
            "threshold": float(
                loc_result.get("trigger_threshold", self.cluster_recovery_trigger_score)
                or self.cluster_recovery_trigger_score
            ),
            "tc_score": float(loc_result.get("tc_score", 0.0) or 0.0),
            "fem_score": float(loc_result.get("fem_score", 0.0) or 0.0),
            "ref_score": float(loc_result.get("ref_score", 0.0) or 0.0),
        }
        self.cluster_recovery_active_signature = cluster_signature
        if approach_needed:
            self.cluster_recovery_approach_active = True
            self.cluster_recovery_approach_goal_cell = tuple(
                int(v) for v in approach["goal_cell"]
            )
            self.cluster_recovery_approach_for_replan = False
            self.cluster_recovery_approach_reason = "initial_start_outside_3m"
        elif not self._install_cluster_recovery_closed_plan(plan):
            failure_reason = "closed_route_validation_failed"
            self._cache_cluster_recovery_route_failure(
                payload,
                cluster_signature,
                failure_reason,
                step,
            )
            self._skip_cluster_recovery_to_mr(
                step, "route_unavailable:{}".format(failure_reason)
            )
            return
        self.cluster_recovery_start_step = int(step)
        self.cluster_recovery_frame_counter = 0
        self.cluster_recovery_replan_count = 0
        self.localization_stage = (
            "CLUSTER_RECOVERY_APPROACH"
            if approach_needed
            else "CLUSTER_RECOVERY_ORBIT"
        )
        self.mode = IEVETargetMode.CLUSTER_RECOVERY
        localizer.mark_cluster_recovery_started()

        try:
            self.slam_mapper.set_target_semantic_growth_policy(
                target_category=self.target_category,
                soft_lock=True,
                chair_xz_radius=self.cluster_recovery_max_radius_m,
                chair_y_radius=3.0,
                default_xz_radius=self.cluster_recovery_max_radius_m,
                default_y_radius=3.0,
            )
        except Exception as exc:
            self.logger.debug("Recovery semantic growth policy failed: %s", exc)
        self._collect_cluster_recovery_candidates(payload, step)
        if approach_needed:
            self._trace(
                "cluster_recovery_approach_started",
                step=step,
                center=center,
                trigger=self.cluster_recovery_trigger,
                start_cell=approach.get("start_cell"),
                center_cell=approach.get("center_cell"),
                goal_cell=approach.get("goal_cell"),
                start_radius_m=approach.get("start_radius_m"),
                goal_radius_m=approach.get("goal_radius_m"),
                estimated_path_length_m=approach.get("path_length_m"),
                coordinate_contract="estimated_pose_and_occupancy_map_only",
                remaining_steps=remaining_steps,
            )
        else:
            self._trace(
                "cluster_recovery_started",
                step=step,
                center=center,
                trigger=self.cluster_recovery_trigger,
                route_nodes=len(self.cluster_recovery_route_cells),
                waypoint_count=len(self.cluster_recovery_waypoint_cells),
                route_length_m=float(plan.get("length_m", 0.0)),
                winding=float(plan.get("winding", 0.0)),
                max_node_radius_m=float(plan.get("max_node_radius_m", 0.0)),
                estimated_actions=estimated_actions,
                remaining_steps=remaining_steps,
            )

    def _skip_cluster_recovery_to_mr(self, step: int, reason: str) -> None:
        self.cluster_recovery_used_in_chain = True
        previous_result = self.cluster_recovery_last_result
        self.cluster_recovery_last_result = {
            "success": False,
            "reason": reason,
            "step": int(step),
            "fallback": "mr",
            "selection_result": previous_result,
        }
        if self.full_localizer is not None:
            self.full_localizer.skip_cluster_recovery_to_mr()
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self.mode = IEVETargetMode.LOCALIZATION
        self.localization_stage = "MR"
        self._trace(
            "cluster_recovery_skipped",
            step=step,
            reason=reason,
            fallback="mr",
        )

    def _plan_cluster_recovery_closed_route(
        self,
        payload: Dict[str, Any],
        center_world: np.ndarray,
        start_cell_override: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        obstacle = payload.get("full_map_pred")
        explored = payload.get("full_exp_pred")
        if obstacle is None or explored is None:
            return {"success": False, "reason": "missing_full_map"}
        obstacle = np.asarray(obstacle, dtype=np.float32)
        explored = np.asarray(explored, dtype=np.float32)
        if obstacle.ndim != 2 or explored.shape != obstacle.shape:
            return {"success": False, "reason": "invalid_full_map_shape"}

        center_cell = self._cluster_recovery_center_cell(
            payload, center_world, obstacle.shape
        )
        start_cell = (
            tuple(int(v) for v in start_cell_override)
            if start_cell_override is not None
            else self._cluster_recovery_estimated_start_cell(payload, obstacle.shape)
        )
        if center_cell is None:
            return {"success": False, "reason": "world_to_full_projection_failed"}
        if start_cell is None:
            return {"success": False, "reason": "missing_estimated_map_pose"}
        h, w = obstacle.shape
        if not (
            0 <= center_cell[0] < h
            and 0 <= center_cell[1] < w
            and 0 <= start_cell[0] < h
            and 0 <= start_cell[1] < w
        ):
            return {"success": False, "reason": "center_or_start_outside_full_map"}

        resolution_m = float(
            payload.get("map_resolution_cm", getattr(self.args, "map_resolution", 5.0))
        ) / 100.0
        max_radius_cells = self.cluster_recovery_max_radius_m / max(resolution_m, 1e-6)
        start_radius = float(
            np.hypot(start_cell[0] - center_cell[0], start_cell[1] - center_cell[1])
        )
        if start_radius > max_radius_cells + 0.5:
            return {"success": False, "reason": "start_outside_3m_constraint"}

        try:
            from skimage.morphology import binary_dilation, disk

            blocked = binary_dilation(obstacle >= 0.5, disk(3))
        except Exception:
            blocked = obstacle >= 0.5
        traversible_base = (~blocked) & (explored > 0.0)
        rows, cols = np.indices(obstacle.shape)
        distance_cells = np.hypot(rows - center_cell[0], cols - center_cell[1])
        max_disk = distance_cells <= max_radius_cells + 1e-6
        traversible_base &= max_disk
        traversible_base[start_cell] = True

        guide_count = int(self.cluster_recovery_waypoint_count)
        start_angle = float(
            np.arctan2(
                start_cell[0] - center_cell[0],
                start_cell[1] - center_cell[1],
            )
        )
        min_radius_cells = max(4.0, 0.6 / max(resolution_m, 1e-6))
        radius_caps = [
            max_radius_cells * factor for factor in (1.0, 0.9, 0.8, 0.7, 0.6)
        ]
        best_failure = "no_candidate_route"
        for radius_cap in radius_caps:
            if radius_cap < min_radius_cells:
                continue
            traversible = traversible_base & (distance_cells <= radius_cap + 1e-6)
            traversible[start_cell] = True
            component = self._cluster_recovery_component(traversible, start_cell)
            if component is None:
                best_failure = "empty_start_component"
                continue

            anchors = []
            valid = True
            for index in range(1, guide_count):
                target_angle = start_angle + 2.0 * np.pi * index / guide_count
                anchor = self._cluster_recovery_angular_anchor(
                    component,
                    center_cell,
                    target_angle,
                    radius_cap,
                    min_radius_cells,
                    angular_tolerance=np.pi / guide_count,
                )
                if anchor is None or anchor in anchors:
                    valid = False
                    best_failure = "missing_angular_anchor"
                    break
                anchors.append(anchor)
            if not valid:
                continue

            segment_goals = anchors + [start_cell]
            route = [start_cell]
            route_segments = []
            cursor = start_cell
            for goal in segment_goals:
                segment = self._extract_cluster_recovery_fmm_path(
                    component, cursor, goal
                )
                if not segment:
                    valid = False
                    best_failure = "fmm_segment_unreachable"
                    break
                route_segments.append(segment)
                route.extend(segment[1:])
                cursor = goal
            if not valid or len(route) < guide_count:
                continue

            route_arr = np.asarray(route, dtype=np.float64)
            route_angles = np.unwrap(
                np.arctan2(
                    route_arr[:, 0] - center_cell[0],
                    route_arr[:, 1] - center_cell[1],
                )
            )
            winding = float((route_angles[-1] - route_angles[0]) / (2.0 * np.pi))
            node_radii = np.hypot(
                route_arr[:, 0] - center_cell[0],
                route_arr[:, 1] - center_cell[1],
            )
            if abs(abs(winding) - 1.0) > 0.20:
                best_failure = "route_winding_not_one"
                continue
            if float(node_radii.max()) > max_radius_cells + 1e-6:
                best_failure = "route_node_outside_3m"
                continue

            waypoint_cells = segment_goals
            diffs = np.diff(route_arr, axis=0)
            length_m = float(np.linalg.norm(diffs, axis=1).sum() * resolution_m)
            plan = {
                "success": True,
                "route_cells": [tuple(int(v) for v in cell) for cell in route],
                "route_segments": [
                    [tuple(int(v) for v in cell) for cell in segment]
                    for segment in route_segments
                ],
                "waypoint_cells": [
                    tuple(int(v) for v in cell) for cell in waypoint_cells
                ],
                "length_m": length_m,
                "winding": winding,
                "max_node_radius_m": float(node_radii.max() * resolution_m),
                "center_cell": center_cell,
                "start_cell": start_cell,
            }
            validation = self._validate_cluster_recovery_closed_plan(plan)
            if not validation.get("valid"):
                best_failure = str(
                    validation.get("reason", "closed_route_validation_failed")
                )
                continue
            plan["winding"] = float(validation["winding"])
            plan["closed_curve_validated"] = True
            return plan
        return {"success": False, "reason": best_failure}

    @staticmethod
    def _cluster_recovery_component(
        traversible: np.ndarray, start: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        try:
            from scipy.ndimage import label

            labels, _ = label(np.asarray(traversible, dtype=np.uint8), structure=np.ones((3, 3)))
            label_id = int(labels[start])
            if label_id <= 0:
                return None
            return labels == label_id
        except Exception:
            return np.asarray(traversible, dtype=bool)

    @staticmethod
    def _cluster_recovery_angular_anchor(
        component: np.ndarray,
        center: Tuple[int, int],
        target_angle: float,
        radius_cap: float,
        min_radius: float,
        angular_tolerance: float,
    ) -> Optional[Tuple[int, int]]:
        cells = np.argwhere(component)
        if cells.size == 0:
            return None
        dr = cells[:, 0].astype(np.float64) - float(center[0])
        dc = cells[:, 1].astype(np.float64) - float(center[1])
        radii = np.hypot(dr, dc)
        angles = np.arctan2(dr, dc)
        delta = np.abs(np.arctan2(np.sin(angles - target_angle), np.cos(angles - target_angle)))
        mask = (
            (radii >= min_radius)
            & (radii <= radius_cap + 1e-6)
            & (delta <= angular_tolerance)
        )
        if not np.any(mask):
            return None
        selected = cells[mask]
        selected_radii = radii[mask]
        selected_delta = delta[mask]
        order = np.lexsort((selected_delta, -selected_radii))
        row, col = selected[int(order[0])]
        return int(row), int(col)

    @staticmethod
    def _extract_cluster_recovery_fmm_path(
        traversible: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        try:
            from envs.utils.fmm_planner import FMMPlanner

            planner = FMMPlanner(np.asarray(traversible, dtype=np.float32))
            planner.set_goal(goal)
            distance = np.asarray(planner.fmm_dist, dtype=np.float64)
        except Exception:
            return []
        current = tuple(int(v) for v in start)
        goal = tuple(int(v) for v in goal)
        path = [current]
        visited = {current}
        h, w = traversible.shape
        for _ in range(int(traversible.size)):
            if current == goal or np.hypot(current[0] - goal[0], current[1] - goal[1]) <= 1.0:
                if current != goal:
                    path.append(goal)
                return path
            candidates = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    row, col = current[0] + dr, current[1] + dc
                    if not (0 <= row < h and 0 <= col < w):
                        continue
                    if not traversible[row, col] or (row, col) in visited:
                        continue
                    candidates.append((float(distance[row, col]), row, col))
            if not candidates:
                return []
            candidates.sort()
            best_distance, row, col = candidates[0]
            if best_distance >= float(distance[current]) - 1e-6:
                return []
            current = (int(row), int(col))
            visited.add(current)
            path.append(current)
        return []

    def _replan_cluster_recovery_closed_route(
        self, payload: Dict[str, Any], step: int, reason: str
    ) -> bool:
        if self.cluster_recovery_center is None:
            self._skip_cluster_recovery_to_mr(
                step, "closed_route_replan_missing_center"
            )
            return False
        if self.cluster_recovery_replan_count >= self.cluster_recovery_max_replans:
            self._skip_cluster_recovery_to_mr(
                step,
                "closed_route_replan_limit_exhausted:{}".format(reason),
            )
            return False

        approach = self._select_cluster_recovery_approach_goal(
            payload, self.cluster_recovery_center
        )
        if not approach.get("success"):
            self._trace(
                "cluster_recovery_replan_approach_unavailable",
                step=step,
                request_reason=reason,
                approach_reason=approach.get("reason"),
            )
            self._skip_cluster_recovery_to_mr(
                step,
                "replan_approach_unavailable:{}".format(
                    approach.get("reason", "unknown")
                ),
            )
            return False
        if approach.get("needed"):
            self.cluster_recovery_route_cells = []
            self.cluster_recovery_route_segments = []
            self.cluster_recovery_waypoint_cells = []
            self.cluster_recovery_waypoints_world = []
            self.cluster_recovery_waypoint_index = 0
            self.cluster_recovery_approach_active = True
            self.cluster_recovery_approach_goal_cell = tuple(
                int(v) for v in approach["goal_cell"]
            )
            self.cluster_recovery_approach_for_replan = True
            self.cluster_recovery_approach_reason = str(reason)
            self.cluster_recovery_frame_counter = 0
            self.cluster_recovery_waypoint_best_distance = float("inf")
            self.cluster_recovery_waypoint_stall_frames = 0
            self.localization_stage = "CLUSTER_RECOVERY_APPROACH"
            self._trace(
                "cluster_recovery_replan_approach_started",
                step=step,
                request_reason=reason,
                start_cell=approach.get("start_cell"),
                center_cell=approach.get("center_cell"),
                goal_cell=approach.get("goal_cell"),
                start_radius_m=approach.get("start_radius_m"),
                goal_radius_m=approach.get("goal_radius_m"),
                coordinate_contract="estimated_pose_and_occupancy_map_only",
            )
            return True

        plan = self._plan_cluster_recovery_closed_route(
            payload, self.cluster_recovery_center
        )
        if not plan.get("success"):
            failure_reason = str(plan.get("reason", "unknown"))
            if self._cluster_recovery_failure_proves_no_closed_route(
                failure_reason
            ):
                self._cache_cluster_recovery_route_failure(
                    payload,
                    self.cluster_recovery_active_signature,
                    failure_reason,
                    step,
                )
            self._trace(
                "cluster_recovery_closed_route_replan_failed",
                step=step,
                request_reason=reason,
                route_reason=failure_reason,
                replan_count=self.cluster_recovery_replan_count,
            )
            self._skip_cluster_recovery_to_mr(
                step,
                "route_unavailable_after_replan:{}".format(failure_reason),
            )
            return False
        if not self._install_cluster_recovery_closed_plan(plan):
            failure_reason = "closed_route_validation_failed"
            self._cache_cluster_recovery_route_failure(
                payload,
                self.cluster_recovery_active_signature,
                failure_reason,
                step,
            )
            self._skip_cluster_recovery_to_mr(
                step,
                "route_unavailable_after_replan:{}".format(failure_reason),
            )
            return False

        self.cluster_recovery_replan_count += 1
        self._trace(
            "cluster_recovery_closed_route_replanned",
            step=step,
            reason=reason,
            replan_count=self.cluster_recovery_replan_count,
            route_nodes=len(self.cluster_recovery_route_cells),
            waypoint_count=len(self.cluster_recovery_waypoint_cells),
            route_length_m=float(plan.get("length_m", 0.0)),
            winding=float(plan.get("winding", 0.0)),
            closed_curve_validated=bool(plan.get("closed_curve_validated", False)),
            contract="every_replan_is_a_complete_closed_curve",
        )
        return True

    def _cluster_recovery_map_distance_to_cell(
        self, payload: Dict[str, Any], goal_cell: Tuple[int, int]
    ) -> Optional[float]:
        obstacle = payload.get("full_map_pred")
        if obstacle is None:
            return None
        shape = np.asarray(obstacle).shape[:2]
        if len(shape) != 2:
            return None
        start_cell = self._cluster_recovery_estimated_start_cell(payload, shape)
        if start_cell is None:
            return None
        resolution_m = float(
            payload.get("map_resolution_cm", getattr(self.args, "map_resolution", 5.0))
        ) / 100.0
        return float(
            np.hypot(
                start_cell[0] - int(goal_cell[0]),
                start_cell[1] - int(goal_cell[1]),
            )
            * resolution_m
        )

    def _finish_cluster_recovery_approach(
        self, payload: Dict[str, Any], step: int
    ) -> bool:
        if self.cluster_recovery_center is None:
            self._skip_cluster_recovery_to_mr(
                step, "approach_finished_without_cluster_center"
            )
            return False
        was_replan = bool(self.cluster_recovery_approach_for_replan)
        approach_reason = self.cluster_recovery_approach_reason
        plan = self._plan_cluster_recovery_closed_route(
            payload, self.cluster_recovery_center
        )
        if not plan.get("success"):
            failure_reason = str(plan.get("reason", "unknown"))
            if self._cluster_recovery_failure_proves_no_closed_route(
                failure_reason
            ):
                self._cache_cluster_recovery_route_failure(
                    payload,
                    self.cluster_recovery_active_signature,
                    failure_reason,
                    step,
                )
            self._trace(
                "cluster_recovery_route_after_approach_failed",
                step=step,
                approach_reason=approach_reason,
                route_reason=failure_reason,
                cached=self._cluster_recovery_failure_proves_no_closed_route(
                    failure_reason
                ),
            )
            self._skip_cluster_recovery_to_mr(
                step,
                "route_unavailable_after_approach:{}".format(failure_reason),
            )
            return False

        remaining_steps = int(getattr(self.args, "max_episode_length", 1000)) - int(
            step
        )
        estimated_actions = int(
            np.ceil(float(plan.get("length_m", 0.0)) / 0.25)
        )
        required_remaining = max(
            self.cluster_recovery_min_remaining_steps,
            estimated_actions + 10,
        )
        if remaining_steps < required_remaining:
            self._skip_cluster_recovery_to_mr(
                step,
                "route_budget_after_approach_{}_lt_{}".format(
                    remaining_steps, required_remaining
                ),
            )
            return False
        if not self._install_cluster_recovery_closed_plan(plan):
            failure_reason = "closed_route_validation_failed"
            self._cache_cluster_recovery_route_failure(
                payload,
                self.cluster_recovery_active_signature,
                failure_reason,
                step,
            )
            self._skip_cluster_recovery_to_mr(
                step,
                "route_unavailable_after_approach:{}".format(failure_reason),
            )
            return False

        self.cluster_recovery_approach_active = False
        self.cluster_recovery_approach_goal_cell = None
        self.cluster_recovery_approach_for_replan = False
        self.cluster_recovery_approach_reason = None
        self.cluster_recovery_frame_counter = 0
        self.localization_stage = "CLUSTER_RECOVERY_ORBIT"
        if was_replan:
            self.cluster_recovery_replan_count += 1
        self._trace(
            "cluster_recovery_closed_route_planned_after_approach",
            step=step,
            approach_reason=approach_reason,
            was_replan=was_replan,
            replan_count=self.cluster_recovery_replan_count,
            route_nodes=len(self.cluster_recovery_route_cells),
            waypoint_count=len(self.cluster_recovery_waypoint_cells),
            route_length_m=float(plan.get("length_m", 0.0)),
            winding=float(plan.get("winding", 0.0)),
            closed_curve_validated=bool(plan.get("closed_curve_validated", False)),
            contract="approach_then_plan_complete_closed_curve",
        )
        return True

    def _maybe_replan_stalled_cluster_recovery_route(
        self, payload: Dict[str, Any], step: int
    ) -> bool:
        requested_reason = self.cluster_recovery_replan_requested_reason
        if requested_reason:
            self.cluster_recovery_replan_requested_reason = None
            self._replan_cluster_recovery_closed_route(
                payload, step, requested_reason
            )
            return True
        if self.cluster_recovery_waypoint_index >= len(
            self.cluster_recovery_waypoint_cells
        ):
            return False
        waypoint_cell = self.cluster_recovery_waypoint_cells[
            self.cluster_recovery_waypoint_index
        ]
        distance = self._cluster_recovery_map_distance_to_cell(
            payload, waypoint_cell
        )
        if distance is None:
            return False
        if (
            distance
            < self.cluster_recovery_waypoint_best_distance
            - self.cluster_recovery_replan_progress_epsilon_m
        ):
            self.cluster_recovery_waypoint_best_distance = float(distance)
            self.cluster_recovery_waypoint_stall_frames = 0
            return False
        self.cluster_recovery_waypoint_stall_frames += 1
        if (
            self.cluster_recovery_waypoint_stall_frames
            < self.cluster_recovery_replan_stall_frames
        ):
            return False
        self._trace(
            "cluster_recovery_waypoint_stalled",
            step=step,
            waypoint_index=self.cluster_recovery_waypoint_index,
            waypoint_count=len(self.cluster_recovery_waypoint_cells),
            distance=float(distance),
            best_distance=float(self.cluster_recovery_waypoint_best_distance),
            stall_frames=self.cluster_recovery_waypoint_stall_frames,
            replan_count=self.cluster_recovery_replan_count,
        )
        self._replan_cluster_recovery_closed_route(
            payload,
            step,
            "waypoint_no_progress_{}".format(
                self.cluster_recovery_waypoint_stall_frames
            ),
        )
        return True

    def _run_cluster_recovery_update(
        self, payload: Dict[str, Any], step: int
    ) -> None:
        self.cluster_recovery_frame_counter += 1
        self._collect_cluster_recovery_candidates(payload, step)
        if self.cluster_recovery_frame_counter > self.cluster_recovery_timeout:
            self._skip_cluster_recovery_to_mr(
                step,
                "approach_timeout"
                if self.cluster_recovery_approach_active
                else "orbit_timeout",
            )
            return
        if self.cluster_recovery_approach_active:
            goal_cell = self.cluster_recovery_approach_goal_cell
            if goal_cell is None:
                self._skip_cluster_recovery_to_mr(
                    step, "approach_has_no_goal_cell"
                )
                return
            map_distance = self._cluster_recovery_map_distance_to_cell(
                payload, goal_cell
            )
            reached = bool(payload.get("minav_phase_goal_reached")) or (
                map_distance is not None
                and map_distance <= self.cluster_recovery_cell_arrival_radius_m
            )
            if reached:
                self._trace(
                    "cluster_recovery_approach_reached",
                    step=step,
                    goal_cell=goal_cell,
                    map_distance_m=map_distance,
                    fmm_goal_reached=bool(
                        payload.get("minav_phase_goal_reached")
                    ),
                    coordinate_contract="estimated_pose_and_occupancy_map_only",
                )
                self._finish_cluster_recovery_approach(payload, step)
            return
        if not self.cluster_recovery_waypoint_cells:
            self._skip_cluster_recovery_to_mr(step, "orbit_has_no_waypoints")
            return

        waypoint_map_distance = None
        if self.cluster_recovery_waypoint_index < len(
            self.cluster_recovery_waypoint_cells
        ):
            waypoint_map_distance = self._cluster_recovery_map_distance_to_cell(
                payload,
                self.cluster_recovery_waypoint_cells[
                    self.cluster_recovery_waypoint_index
                ],
            )
        waypoint_reached = bool(payload.get("minav_phase_goal_reached")) or (
            waypoint_map_distance is not None
            and waypoint_map_distance <= self.cluster_recovery_cell_arrival_radius_m
        )
        if waypoint_reached:
            reached_index = int(self.cluster_recovery_waypoint_index)
            self.cluster_recovery_waypoint_index += 1
            self._trace(
                "cluster_recovery_waypoint_reached",
                step=step,
                waypoint_index=reached_index,
                waypoint_count=len(self.cluster_recovery_waypoint_cells),
                candidate_count=len(self.cluster_recovery_candidates),
                map_distance_m=waypoint_map_distance,
                fmm_goal_reached=bool(payload.get("minav_phase_goal_reached")),
            )
            if self.cluster_recovery_waypoint_index >= len(
                self.cluster_recovery_waypoint_cells
            ):
                self._finish_cluster_recovery_orbit(payload, step)
            else:
                self.cluster_recovery_waypoint_best_distance = float("inf")
                self.cluster_recovery_waypoint_stall_frames = 0
            return

        if self._maybe_replan_stalled_cluster_recovery_route(payload, step):
            return

    def _cluster_recovery_goal_map_from_full_cell(
        self, context: Dict[str, Any], goal_cell: Optional[Tuple[int, int]]
    ) -> Optional[Dict[str, Any]]:
        pose_pred = context.get("pose_pred")
        map_pred = context.get("map_pred")
        if goal_cell is None or pose_pred is None or map_pred is None:
            return None
        try:
            pose = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
            shape = np.asarray(map_pred).shape[:2]
            if pose.size < 7 or len(shape) != 2:
                return None
            local_cell = (
                int(goal_cell[0]) - int(pose[3]),
                int(goal_cell[1]) - int(pose[5]),
            )
            if not (
                0 <= local_cell[0] < int(shape[0])
                and 0 <= local_cell[1] < int(shape[1])
            ):
                return None
            return {
                "goal_map": self._buffer_make_goal_map(shape, local_cell),
                "local_cell": local_cell,
                "full_cell": tuple(int(v) for v in goal_cell),
            }
        except Exception:
            return None

    def _get_cluster_recovery_goal_override(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        if self.cluster_recovery_approach_active:
            goal_cell = self.cluster_recovery_approach_goal_cell
        elif self.cluster_recovery_waypoint_index < len(
            self.cluster_recovery_waypoint_cells
        ):
            goal_cell = self.cluster_recovery_waypoint_cells[
                self.cluster_recovery_waypoint_index
            ]
        else:
            return None, None, None, True

        projected = self._cluster_recovery_goal_map_from_full_cell(
            context, goal_cell
        )
        if projected is None:
            self._trace(
                "cluster_recovery_map_goal_unavailable",
                step=int(context.get("step", 0)),
                approach=bool(self.cluster_recovery_approach_active),
                waypoint_index=self.cluster_recovery_waypoint_index,
                goal_cell=goal_cell,
                coordinate_contract="estimated_pose_and_map_cells_only",
            )
            if not self.cluster_recovery_approach_active:
                self.cluster_recovery_replan_requested_reason = (
                    "waypoint_map_goal_unavailable"
                )
            return None, None, None, True

        self.navigation_goal_world = None
        self.navigation_goal_raw_world = None
        self.navigation_goal_local_cell = tuple(projected["local_cell"])
        self.navigation_goal_full_cell = tuple(projected["full_cell"])
        self.navigation_goal_adjusted = False
        constraint = (
            None
            if self.cluster_recovery_approach_active
            else self._cluster_recovery_local_constraint(context)
        )
        return None, projected["goal_map"], constraint, False

    def _cluster_recovery_local_constraint(
        self, context: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        if (
            self.cluster_recovery_center_full_cell is None
            or context.get("map_pred") is None
        ):
            return None
        pose_pred = context.get("pose_pred")
        if pose_pred is None:
            return None
        shape = np.asarray(context["map_pred"]).shape[:2]
        try:
            pose = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
            if pose.size < 7:
                return None
            gx1, gy1 = int(pose[3]), int(pose[5])
        except Exception:
            return None
        center_row = int(self.cluster_recovery_center_full_cell[0]) - gx1
        center_col = int(self.cluster_recovery_center_full_cell[1]) - gy1
        resolution_m = float(
            context.get("map_resolution_cm", getattr(self.args, "map_resolution", 5.0))
        ) / 100.0
        rows, cols = np.indices(shape)
        radius_cells = self.cluster_recovery_max_radius_m / max(resolution_m, 1e-6)
        disk_constraint = (
            np.hypot(rows - float(center_row), cols - float(center_col))
            <= radius_cells + 1e-6
        )
        if self.cluster_recovery_waypoint_index >= len(
            self.cluster_recovery_route_segments
        ):
            return disk_constraint.astype(np.uint8)

        corridor = np.zeros(shape, dtype=np.uint8)
        for full_row, full_col in self.cluster_recovery_route_segments[
            self.cluster_recovery_waypoint_index
        ]:
            local_row = int(full_row) - gx1
            local_col = int(full_col) - gy1
            if 0 <= local_row < shape[0] and 0 <= local_col < shape[1]:
                corridor[local_row, local_col] = 1
        try:
            from skimage.morphology import binary_dilation, disk

            corridor = binary_dilation(corridor > 0, disk(6))
        except Exception:
            corridor = corridor > 0
        return (disk_constraint & corridor).astype(np.uint8)

    def _finish_cluster_recovery_orbit(
        self, payload: Dict[str, Any], step: int
    ) -> None:
        self.localization_stage = "CLUSTER_RECOVERY_RENDER"
        self._collect_cluster_recovery_candidates(payload, step)
        result = self._evaluate_cluster_recovery_candidates(payload, step)
        self.cluster_recovery_last_result = result
        if not result.get("success"):
            self._skip_cluster_recovery_to_mr(
                step,
                "candidate_selection_failed:{}".format(
                    result.get("reason", "unknown")
                ),
            )
            return

        winner = result["winner_cluster"]
        indices = self._cluster_indices(winner)
        center = self._cluster_center(winner)
        locked = self._lock_confirmed_cluster(
            indices,
            step=step,
            stage="cluster_recovery",
        )
        if locked <= 0:
            self._skip_cluster_recovery_to_mr(step, "winner_lock_failed")
            return

        self.confirmed_indices = indices.copy()
        self.confirmed_target_position = center.copy()
        anchor = self._select_target_map_anchor(center, payload)
        self.confirmed_target_map_anchor_position = (
            center.copy()
            if anchor is None or anchor.get("anchor_world") is None
            else np.asarray(anchor["anchor_world"], dtype=np.float32).reshape(-1)[:3]
        )
        self.buffer_current_goal_world = center.copy()
        localizer = self.full_localizer
        if localizer is None:
            self._skip_cluster_recovery_to_mr(step, "localizer_missing_after_orbit")
            return
        localizer.resume_after_cluster_recovery(
            best_pose=result["best_pose"],
            score=float(result["best_score"]),
            cluster_center=center,
            match_cache=result.get("match_cache"),
        )
        self._sync_full_localizer_candidate_anchor(
            localizer, step, source="cluster_recovery_winner"
        )
        self.mode = IEVETargetMode.LOCALIZATION
        self.localization_stage = "CLUSTER_RECOVERY_RELOCALIZE"
        self.localization_frame_counter = 0
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self._trace(
            "cluster_recovery_complete",
            step=step,
            trigger=self.cluster_recovery_trigger,
            candidate_count=len(self.cluster_recovery_candidates),
            winner_center=center,
            winner_size=int(indices.size),
            best_score=float(result["best_score"]),
            center_match_count=int(result["center_match_count"]),
            yaw_offset_deg=float(result["yaw_offset_deg"]),
            next_state="PRE",
        )

    def _collect_cluster_recovery_candidates(
        self, payload: Dict[str, Any], step: int
    ) -> None:
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None or self.cluster_recovery_center is None:
            return
        raw_candidates: List[Dict[str, Any]] = []
        if self.cluster_recovery_original_indices is not None:
            original = self._build_target_confirm_cluster_record(
                gaussians, self.cluster_recovery_original_indices
            )
            if original is not None:
                original.update({"source": "original_cluster", "support_steps": [int(step)]})
                raw_candidates.append(original)

        camera = self._latest_slam_camera()
        depth = payload.get("depth_m", payload.get("raw_depth"))
        if camera is not None:
            for instance in payload.get("minav_target_instances") or payload.get("target_instances") or []:
                mask = instance.get("binary_mask") if isinstance(instance, dict) else None
                if mask is None:
                    continue
                projected = self._project_instance_mask_to_gaussians(
                    gaussians, mask, camera, depth
                )
                clusters = self._expand_projected_indices_to_semantic_clusters(
                    gaussians, projected
                )
                if not clusters:
                    clusters = self._cluster_target_confirm_indices(
                        gaussians, projected
                    )
                for cluster in clusters:
                    cluster.update({"source": "orbit_observation", "support_steps": [int(step)]})
                    raw_candidates.append(cluster)

        for candidate in raw_candidates:
            center = self._cluster_center(candidate)
            if center.size < 3:
                continue
            self._merge_cluster_recovery_candidate(gaussians, candidate, step)

    def _merge_cluster_recovery_candidate(
        self, gaussians, candidate: Dict[str, Any], step: int
    ) -> None:
        indices = self._cluster_indices(candidate)
        center = self._cluster_center(candidate)
        if indices.size == 0 or center.size < 3:
            return
        merge_dist = float(getattr(self.args, "minav_confirm_merge_dist_m", 0.35))
        merge_overlap = float(
            getattr(self.args, "minav_confirm_merge_overlap_ratio", 0.25)
        )
        matched = None
        candidate_set = set(indices.tolist())
        for existing in self.cluster_recovery_candidates:
            existing_indices = self._cluster_indices(existing)
            existing_set = set(existing_indices.tolist())
            overlap = len(candidate_set & existing_set) / max(
                min(len(candidate_set), len(existing_set)), 1
            )
            distance = self._world_distance_xz(center, self._cluster_center(existing))
            if overlap >= merge_overlap or distance <= merge_dist:
                matched = existing
                break

        support_steps = {int(step)}
        support_steps.update(int(v) for v in candidate.get("support_steps", []) or [])
        if matched is None:
            record = self._build_target_confirm_cluster_record(gaussians, indices)
            if record is None:
                return
            record.update(
                {
                    "source": candidate.get("source", "recovery"),
                    "support_steps": sorted(support_steps),
                    "support_count": len(support_steps),
                }
            )
            self.cluster_recovery_candidates.append(record)
            return

        merged_indices = np.unique(
            np.concatenate([self._cluster_indices(matched), indices])
        )
        record = self._build_target_confirm_cluster_record(gaussians, merged_indices)
        if record is None:
            return
        support_steps.update(int(v) for v in matched.get("support_steps", []) or [])
        record.update(
            {
                "source": matched.get("source", candidate.get("source", "recovery")),
                "support_steps": sorted(support_steps),
                "support_count": len(support_steps),
            }
        )
        matched.clear()
        matched.update(record)

    def _evaluate_cluster_recovery_candidates(
        self, payload: Dict[str, Any], step: int
    ) -> Dict[str, Any]:
        if self.cluster_recovery_base_pose is None or self.cluster_recovery_center is None:
            return {"success": False, "reason": "missing_base_pose_or_center"}
        self._ensure_goal_image_masks(payload, step)
        if self.goal_center_mask is None:
            return {"success": False, "reason": "goal_center_object_mask_unavailable"}
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            return {"success": False, "reason": fail}
        self._refresh_cluster_recovery_candidate_indices(gaussians)
        nvsm, fem, fail = self._ensure_target_confirm_nvsm_fem()
        if fail is not None:
            return {"success": False, "reason": fail}

        all_view_records = []
        best = None
        for cluster_index, cluster in enumerate(self.cluster_recovery_candidates):
            center = self._cluster_center(cluster)
            indices = self._cluster_indices(cluster)
            render_result = nvsm.render_relative_yaw_views(
                object_center_3d=center,
                reference_center_3d=self.cluster_recovery_center,
                reference_pose_w2c=self.cluster_recovery_base_pose,
                yaw_range_deg=self.cluster_recovery_yaw_range_deg,
                yaw_step_deg=self.cluster_recovery_yaw_step_deg,
            )
            views = list(render_result.get("rendered_images") or [])
            if not render_result.get("success") or not views:
                all_view_records.append(
                    {
                        "cluster_index": cluster_index,
                        "center": center,
                        "cluster_size": int(indices.size),
                        "render_error": render_result.get("message", "render_failed"),
                        "views": [],
                    }
                )
                continue
            fem_result = fem.match_rendered_views(views, feature_mode="uniform")
            scores = [float(v) for v in fem_result.get("all_scores", [])]
            caches = list(getattr(fem, "_last_batch_match_caches", None) or [])
            cluster_views = []
            for view_index, view in enumerate(views):
                score = scores[view_index] if view_index < len(scores) else 0.0
                cache = caches[view_index] if view_index < len(caches) else None
                render_mask = self._project_cluster_to_render_mask(
                    gaussians,
                    indices,
                    view.get("pose"),
                    np.asarray(view.get("image")).shape[:2],
                )
                center_matches = self._count_center_object_matches(
                    cache,
                    self.goal_center_mask,
                    render_mask,
                )
                record = {
                    "cluster_index": int(cluster_index),
                    "view_index": int(view_index),
                    "yaw_offset_deg": float(view.get("yaw_offset_deg", 0.0)),
                    "score": float(score),
                    "center_match_count": int(center_matches),
                    "passes_threshold": bool(score >= self.localization_match_threshold),
                }
                cluster_views.append(record)
                if score < self.localization_match_threshold:
                    continue
                rank = (
                    int(center_matches),
                    float(score),
                    int(cluster.get("support_count", 1) or 1),
                )
                if best is None or rank > best["rank"]:
                    best = {
                        "rank": rank,
                        "cluster": cluster,
                        "pose": np.asarray(view["pose"], dtype=np.float64).copy(),
                        "score": float(score),
                        "center_match_count": int(center_matches),
                        "yaw_offset_deg": float(view.get("yaw_offset_deg", 0.0)),
                        "match_cache": cache,
                        "image": view.get("image"),
                    }
            all_view_records.append(
                {
                    "cluster_index": int(cluster_index),
                    "center": center,
                    "cluster_size": int(indices.size),
                    "support_count": int(cluster.get("support_count", 1) or 1),
                    "views": cluster_views,
                }
            )

        self._trace(
            "cluster_recovery_candidate_scores",
            step=step,
            candidate_count=len(self.cluster_recovery_candidates),
            localization_threshold=self.localization_match_threshold,
            view_records=all_view_records,
        )
        if best is None:
            return {
                "success": False,
                "reason": "no_view_passed_localization_threshold",
                "view_records": all_view_records,
            }

        return {
            "success": True,
            "winner_cluster": best["cluster"],
            "best_pose": best["pose"],
            "best_score": best["score"],
            "center_match_count": best["center_match_count"],
            "yaw_offset_deg": best["yaw_offset_deg"],
            "match_cache": best["match_cache"],
            "view_records": all_view_records,
        }

    def _refresh_cluster_recovery_candidate_indices(self, gaussians) -> None:
        """Reassociate observed candidates after Gaussian pruning compacts indices."""
        if not self.cluster_recovery_candidates:
            return
        try:
            semantic = getattr(gaussians, "get_semantic_target", None)
            if callable(semantic):
                semantic = semantic()
            if semantic is None or not hasattr(semantic, "numel") or semantic.numel() == 0:
                return
            import torch

            threshold = float(getattr(self.args, "minav_semantic_threshold", 0.6))
            indices = (
                torch.where(semantic > threshold)[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )
            current_clusters = self._cluster_target_confirm_indices(gaussians, indices)
        except Exception:
            return
        if not current_clusters:
            return

        refreshed = []
        max_distance = max(
            0.6,
            2.0 * float(getattr(self.args, "minav_confirm_merge_dist_m", 0.35)),
        )
        for candidate in self.cluster_recovery_candidates:
            center = self._cluster_center(candidate)
            nearest = min(
                current_clusters,
                key=lambda cluster: self._world_distance_xz(
                    center, self._cluster_center(cluster)
                ),
            )
            distance = self._world_distance_xz(center, self._cluster_center(nearest))
            if distance <= max_distance:
                record = dict(nearest)
                record.update(
                    {
                        "source": candidate.get("source", "recovery"),
                        "support_steps": list(candidate.get("support_steps", []) or []),
                        "support_count": int(candidate.get("support_count", 1) or 1),
                    }
                )
            else:
                record = candidate
            duplicate = next(
                (
                    item
                    for item in refreshed
                    if np.intersect1d(
                        self._cluster_indices(item), self._cluster_indices(record)
                    ).size
                    > 0
                ),
                None,
            )
            if duplicate is None:
                refreshed.append(record)
            else:
                steps = set(duplicate.get("support_steps", []) or [])
                steps.update(record.get("support_steps", []) or [])
                duplicate["support_steps"] = sorted(int(v) for v in steps)
                duplicate["support_count"] = len(steps)
        self.cluster_recovery_candidates = refreshed

    def _project_cluster_to_render_mask(
        self,
        gaussians,
        indices: np.ndarray,
        pose_w2c: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        try:
            xyz = gaussians.get_xyz.detach().cpu().numpy()
            valid_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            valid_indices = valid_indices[
                (valid_indices >= 0) & (valid_indices < xyz.shape[0])
            ]
            if valid_indices.size == 0:
                return None
            pose = np.asarray(pose_w2c, dtype=np.float64)
            points = xyz[valid_indices].astype(np.float64)
            points_h = np.concatenate(
                [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
            )
            camera = (pose @ points_h.T).T[:, :3]
            height, width = int(image_shape[0]), int(image_shape[1])
            params = getattr(self.slam_mapper, "camera_params", {}) or {}
            fx = float(params.get("fx", 0.0) or 0.0)
            fy = float(params.get("fy", 0.0) or 0.0)
            cx = float(params.get("cx", width * 0.5))
            cy = float(params.get("cy", height * 0.5))
            if fx <= 0.0 or fy <= 0.0:
                return None
            valid = camera[:, 2] > 0.1
            u = camera[:, 0] * fx / np.maximum(camera[:, 2], 1e-6) + cx
            v = camera[:, 1] * fy / np.maximum(camera[:, 2], 1e-6) + cy
            valid &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
            pixels = np.stack([u[valid], v[valid]], axis=1)
            if pixels.shape[0] == 0:
                return None
            mask = np.zeros((height, width), dtype=np.uint8)
            try:
                import cv2

                for x, y in np.rint(pixels).astype(np.int32):
                    cv2.circle(mask, (int(x), int(y)), 5, 1, thickness=-1)
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    np.ones((7, 7), dtype=np.uint8),
                )
            except Exception:
                coords = np.rint(pixels).astype(np.int32)
                mask[coords[:, 1], coords[:, 0]] = 1
            return mask
        except Exception as exc:
            self.logger.debug("Recovery cluster render-mask projection failed: %s", exc)
            return None

    @staticmethod
    def _count_center_object_matches(
        match_cache: Optional[Dict[str, Any]],
        target_center_mask: Optional[np.ndarray],
        render_cluster_mask: Optional[np.ndarray],
    ) -> int:
        if match_cache is None or target_center_mask is None or render_cluster_mask is None:
            return 0
        target_points = match_cache.get("kpts_ref")
        render_points = match_cache.get("kpts_query")
        if target_points is None or render_points is None:
            return 0
        target_points = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
        render_points = np.asarray(render_points, dtype=np.float32).reshape(-1, 2)
        count = 0
        target_h, target_w = np.asarray(target_center_mask).shape[:2]
        render_h, render_w = np.asarray(render_cluster_mask).shape[:2]
        for target_point, render_point in zip(target_points, render_points):
            tx, ty = int(round(float(target_point[0]))), int(round(float(target_point[1])))
            rx, ry = int(round(float(render_point[0]))), int(round(float(render_point[1])))
            if not (0 <= tx < target_w and 0 <= ty < target_h):
                continue
            if not (0 <= rx < render_w and 0 <= ry < render_h):
                continue
            if target_center_mask[ty, tx] and render_cluster_mask[ry, rx]:
                count += 1
        return int(count)

    def _run_localization_update(self, payload: Dict[str, Any], step: int) -> None:
        if self.confirmed_target_position is None:
            return
        self.localization_frame_counter += 1
        from_mode = self.mode.value
        self._trace(
            "localization_start",
            step=step,
            frame=self.localization_frame_counter,
            timeout=self.localization_timeout,
            gaussmi_ready=self.buffer_gaussmi_signaled,
            stage=self.localization_stage,
            owner_mode=from_mode,
        )
        loc_result = self.run_close_localization(payload)
        self.last_localization_result = loc_result
        loc_center = loc_result.get("target_position")
        if loc_center is None:
            loc_center = self.confirmed_target_position
        if loc_center is None:
            loc_center = self.buffer_current_goal_world
        self._record_strategy_evidence(
            step=step,
            center=loc_center,
            best_score=float(loc_result.get("best_score", 0.0) or 0.0),
            source="localization",
            confirmed=bool(loc_result.get("localized")),
            best_match_count=int(loc_result.get("best_match_count", 0) or 0),
            fail_reason=loc_result.get("fail_reason", ""),
        )
        self._trace(
            "localization_result",
            step=step,
            frame=self.localization_frame_counter,
            owner_mode=from_mode,
            result=self._result_summary(loc_result),
        )
        if loc_result.get("cluster_recovery_requested"):
            self._start_cluster_recovery(payload, loc_result, step)
            return
        if loc_result.get("localized"):
            self._switch_to_navigation_from_localization(
                step,
                from_mode,
                np.asarray(loc_result["target_position"], dtype=np.float32),
                loc_result,
            )
            return
        if loc_result.get("mr_reposition"):
            self._enter_buffer_reposition(loc_result, step)
            return
        if not loc_result.get("system_not_ready"):
            if self._debug_enabled():
                self._trace(
                    "localization_failure_geometry_debug",
                    step=step,
                    frame=self.localization_frame_counter,
                    fail_reason=loc_result.get("fail_reason", "localization_failed"),
                    score=loc_result.get("best_score"),
                    threshold=self.localization_match_threshold,
                    metrics=self._debug_target_navigation_metrics(
                        payload,
                        confirmed_target=self.confirmed_target_position,
                        buffer_goal=self.navigation_goal_world,
                        detected_target=self.confirmation_detected_target_position,
                    ),
                    result=self._result_summary(loc_result),
                )
            self._fail_to_exploration(
                step,
                loc_result.get("fail_reason", "localization_failed"),
                blacklist=True,
            )
            return
        if (
            self.buffer_gaussmi_signaled
            and self._localization_raw_state(loc_result) == "soim"
        ):
            frames_since_signal = self._frames_since_gaussmi_signal()
            if frames_since_signal >= self.buffer_soim_wait_timeout:
                self._trace(
                    "soim_post_signal_timeout",
                    step=step,
                    frame=self.localization_frame_counter,
                    frames_since_signal=frames_since_signal,
                    timeout=self.buffer_soim_wait_timeout,
                    fail_reason=loc_result.get("fail_reason"),
                )
                self._fail_to_exploration(
                    step,
                    "SOIM post-signal timeout ({} frames)".format(
                        frames_since_signal
                    ),
                    blacklist=True,
                )
                return
        if self.buffer_gaussmi_signaled:
            frames_since_signal = self._frames_since_gaussmi_signal()
            if frames_since_signal >= self.buffer_localization_wait_timeout:
                self._fail_to_exploration(
                    step,
                    "Localization wait timeout ({} frames)".format(
                        frames_since_signal
                    ),
                    blacklist=True,
                )
                return
        elif (
            self.buffer_frame_counter >= self.buffer_timeout
            and (
                self.buffer_frame_counter == self.buffer_timeout
                or self.buffer_frame_counter % max(1, self.buffer_timeout) == 0
            )
        ):
            self._trace(
                "buffer_localization_pre_signal_waiting",
                step=step,
                frame=self.localization_frame_counter,
                buffer_frame=self.buffer_frame_counter,
                timeout=self.buffer_timeout,
                timeout_applies_after_gaussmi_ready=True,
            )

    def _frames_since_gaussmi_signal(self) -> int:
        if not self.buffer_gaussmi_signaled:
            return 0
        if (
            self.mode == IEVETargetMode.LOCALIZATION
            and self.localization_signal_frame_counter >= 0
        ):
            return max(
                0,
                int(self.localization_frame_counter)
                - int(self.localization_signal_frame_counter),
            )
        if self.buffer_gaussmi_signaled_frame >= 0:
            return max(0, int(self.buffer_frame_counter) - int(self.buffer_gaussmi_signaled_frame))
        if self.localization_signal_frame_counter >= 0:
            return max(0, int(self.localization_frame_counter) - int(self.localization_signal_frame_counter))
        return 0

    def _switch_to_navigation_from_localization(
        self,
        step: int,
        from_mode: str,
        target_position: np.ndarray,
        loc_result: Dict[str, Any],
    ) -> None:
        suggested_goal = np.asarray(
            target_position, dtype=np.float32
        ).reshape(-1)[:3]
        raw_target = loc_result.get("target_object_center")
        if raw_target is None:
            raw_target = suggested_goal
        self.localized_target_position = np.asarray(
            raw_target, dtype=np.float32
        ).reshape(-1)[:3]
        self.mode = IEVETargetMode.NAVIGATION
        self.navigation_complete = False
        self._reset_phase3_navigation_progress()
        self._clear_navigation_goal_state()
        self._reset_reachable_navigation_state()
        self.navigation_localizer_suggested_goal = suggested_goal.copy()
        self.buffer_direct_nav_to_confirmed_target = False
        self.buffer_start_step = -1
        self.buffer_frame_counter = 0
        self.buffer_phase = 1
        self.buffer_goal_update_counter = 0
        self.buffer_current_goal_world = None
        self.buffer_detection_target_position = None
        self.buffer_gaussmi_signaled = False
        self.buffer_gaussmi_signaled_step = -1
        self.buffer_gaussmi_signaled_frame = -1
        self._clear_buffer_standoff_sweep_state()
        self.localization_signal_frame_counter = -1
        self.localization_stage = "localized"
        self.localization_frame_counter = 0
        self.localization_signal_frame_counter = -1
        self.pending_closed_loop_pose = None
        self.pending_closed_loop_score = None
        self._clear_mr_remap_state()
        if self.full_localizer is not None:
            try:
                with self._maybe_silence_full_localizer():
                    reset = getattr(self.full_localizer, "reset", None)
                    if callable(reset):
                        reset()
            except Exception as exc:
                self._trace(
                    "localizer_reset_after_navigation_failed",
                    step=step,
                    error=str(exc),
                )
        self.full_localizer = None
        self.full_localizer_started = False
        self.full_localizer_error = None
        self._clear_detection_window()
        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.NAVIGATION.value,
            reason="localized",
            target_position=self.localized_target_position,
            localizer_suggested_goal=suggested_goal,
            target_object_center=self.localized_target_position,
            best_score=loc_result.get("best_score"),
            best_match_count=loc_result.get("best_match_count"),
        )

    def _start_mr_remap_full_sweep(self, step: int) -> None:
        self.mr_remap_pending = False
        self.mr_remap_sweep_active = True
        self.mr_remap_sweep_just_completed = False
        self.mr_remap_sweep_total_steps = self._full_turn_sweep_steps()
        self.mr_remap_sweep_remaining_steps = self.mr_remap_sweep_total_steps
        self.mr_remap_sweep_direction = "turn_left"
        self._trace(
            "mr_remap_sweep_start",
            step=step,
            sweep_steps=self.mr_remap_sweep_total_steps,
            turn_angle=float(getattr(self.args, "turn_angle", 30.0) or 30.0),
            target_position=self.buffer_current_goal_world,
            contract="minav_mr_closed_loop_arrive_then_360_remap",
        )

    def _buffer_standoff_sweep_steps(self) -> int:
        explicit_steps = int(
            getattr(self.args, "minav_buffer_standoff_sweep_steps", 0) or 0
        )
        if explicit_steps > 0:
            return max(1, explicit_steps)
        return self._full_turn_sweep_steps()

    def _clear_buffer_standoff_sweep_state(self) -> None:
        self.buffer_standoff_sweep_active = False
        self.buffer_standoff_sweep_just_completed = False
        self.buffer_standoff_sweep_remaining_steps = 0
        self.buffer_standoff_sweep_total_steps = 0
        self.buffer_standoff_sweep_direction = "turn_left"

    def _start_buffer_standoff_full_sweep(self, step: int, target) -> None:
        self.buffer_phase = 2
        self.buffer_standoff_sweep_active = True
        self.buffer_standoff_sweep_just_completed = False
        self.buffer_standoff_sweep_total_steps = self._buffer_standoff_sweep_steps()
        self.buffer_standoff_sweep_remaining_steps = self.buffer_standoff_sweep_total_steps
        self.buffer_standoff_sweep_direction = "turn_left"
        self._trace(
            "buffer_standoff_sweep_start",
            step=step,
            sweep_steps=self.buffer_standoff_sweep_total_steps,
            turn_angle=float(getattr(self.args, "turn_angle", 30.0) or 30.0),
            target_position=target,
            confirmed_target=self.confirmed_target_position,
            contract="direct_buffer_arrive_standoff_then_360_refine_before_localization",
        )

    def _consume_buffer_standoff_sweep_action(self, context: Dict[str, Any]) -> str:
        step = int(context.get("step", 0))
        self.buffer_standoff_sweep_remaining_steps = max(
            0, int(self.buffer_standoff_sweep_remaining_steps) - 1
        )
        if self.buffer_standoff_sweep_remaining_steps <= 0:
            self.buffer_standoff_sweep_active = False
            self.buffer_standoff_sweep_just_completed = True
            self._trace(
                "buffer_standoff_sweep_action_complete",
                step=step,
                sweep_steps=self.buffer_standoff_sweep_total_steps,
                target_position=self.buffer_current_goal_world,
            )
        else:
            self._trace(
                "buffer_standoff_sweep_turn",
                step=step,
                remaining_steps=self.buffer_standoff_sweep_remaining_steps,
                total_steps=self.buffer_standoff_sweep_total_steps,
            )
        return self.buffer_standoff_sweep_direction

    def _finish_buffer_standoff_sweep_and_enter_localization(
        self, payload: Dict[str, Any], step: int
    ) -> None:
        self.buffer_standoff_sweep_active = False
        self.buffer_standoff_sweep_just_completed = False
        self._refresh_slam_training_before_localization(
            step, source="BUFFER_STANDOFF_SWEEP"
        )
        self._trace(
            "buffer_standoff_sweep_complete",
            step=step,
            sweep_steps=self.buffer_standoff_sweep_total_steps,
            refine_frames=self.buffer_phase2_refine_frames,
            target_position=self.buffer_current_goal_world,
        )
        self._enter_localization_from_phase_goal(payload, step)

    def _consume_mr_remap_sweep_action(self, context: Dict[str, Any]) -> str:
        step = int(context.get("step", 0))
        self.mr_remap_sweep_remaining_steps = max(
            0, int(self.mr_remap_sweep_remaining_steps) - 1
        )
        if self.mr_remap_sweep_remaining_steps <= 0:
            self.mr_remap_sweep_active = False
            self.mr_remap_sweep_just_completed = True
            self._trace(
                "mr_remap_sweep_action_complete",
                step=step,
                sweep_steps=self.mr_remap_sweep_total_steps,
                target_position=self.buffer_current_goal_world,
            )
        else:
            self._trace(
                "mr_remap_sweep_turn",
                step=step,
                remaining_steps=self.mr_remap_sweep_remaining_steps,
                total_steps=self.mr_remap_sweep_total_steps,
            )
        return self.mr_remap_sweep_direction

    def _finish_mr_remap_sweep_and_signal_pre(self, step: int) -> None:
        self.mr_remap_sweep_active = False
        self.mr_remap_sweep_just_completed = False
        self._refresh_slam_training_before_localization(
            step, source="MR_REMAP_SWEEP"
        )
        self.buffer_gaussmi_signaled = True
        self.buffer_gaussmi_signaled_step = int(step)
        self.buffer_gaussmi_signaled_frame = int(self.buffer_frame_counter)
        self.localization_signal_frame_counter = int(self.localization_frame_counter)
        localizer = self.full_localizer
        pnp_pose = self.pending_closed_loop_pose
        injected = False
        if localizer is not None and pnp_pose is not None and hasattr(
            localizer, "inject_closed_loop_pose_for_pre"
        ):
            try:
                with self._maybe_silence_full_localizer():
                    localizer.inject_closed_loop_pose_for_pre(
                        pnp_pose,
                        score_hint=self.pending_closed_loop_score,
                        source="ieve_mr_remap_full_sweep",
                    )
                injected = True
                self.pending_closed_loop_pose = None
                self.pending_closed_loop_score = None
            except Exception as exc:
                self._trace(
                    "mr_remap_sweep_inject_failed",
                    step=step,
                    error=str(exc),
                )
        elif localizer is not None:
            try:
                with self._maybe_silence_full_localizer():
                    localizer.notify_gaussmi_ready()
            except Exception as exc:
                self._trace(
                    "mr_remap_sweep_notify_failed",
                    step=step,
                    error=str(exc),
                )
        self._trace(
            "mr_remap_sweep_complete",
            step=step,
            injected_closed_loop_pose=bool(injected),
            has_localizer=localizer is not None,
            signaled_frame=self.buffer_gaussmi_signaled_frame,
        )

        # The PnP pose is now installed in the localizer's PRE state.  Keep
        # the outer manager in localization as well; leaving it in buffer
        # makes the next frame hit the legacy-buffer fallback before PRE runs.
        from_mode = self.mode.value
        self.mode = IEVETargetMode.LOCALIZATION
        self.localization_frame_counter = 0
        self.localization_signal_frame_counter = 0
        self.localization_stage = "PRE" if injected else "SOIM"
        self.buffer_direct_nav_to_confirmed_target = False
        self._trace(
            "mode_transition",
            step=step,
            from_mode=from_mode,
            to_mode=IEVETargetMode.LOCALIZATION.value,
            reason="mr_remap_sweep_complete_enter_localization",
            localization_stage=self.localization_stage,
            injected_closed_loop_pose=bool(injected),
            contract="mr_remap_must_execute_pre_then_ref_before_navigation",
        )

    def _run_direct_buffer_update(
        self, payload: Dict[str, Any], step: int, force_arrived: bool = False
    ) -> None:
        self.buffer_frame_counter += 1
        self.buffer_goal_update_counter += 1
        if self.buffer_standoff_sweep_just_completed:
            self._finish_buffer_standoff_sweep_and_enter_localization(payload, step)
            return
        if self.buffer_standoff_sweep_active:
            self._trace(
                "buffer_standoff_sweep_active",
                step=step,
                remaining_steps=self.buffer_standoff_sweep_remaining_steps,
                total_steps=self.buffer_standoff_sweep_total_steps,
            )
            return
        if self.confirmed_target_position is None:
            self._fail_to_exploration(
                step,
                "direct_buffer_without_confirmed_target",
                blacklist=False,
            )
            return

        target = self.buffer_current_goal_world
        if target is None:
            goal_anchor = self._confirmed_target_goal_anchor()
            if goal_anchor is not None:
                prepared = self._prepare_buffer_goal(
                    goal_anchor,
                    payload,
                    source="MINAV_BUFFER_CONFIRMED_TARGET_STANDOFF_UPDATE",
                    search_radius_m=self.buffer_goal_search_radius,
                    allow_adjustment=True,
                    commit=True,
                )
                if prepared is not None:
                    target = np.asarray(prepared["goal_world"], dtype=np.float32).reshape(-1)[:3]
        if target is None:
            self._trace(
                "direct_buffer_standoff_unavailable",
                step=step,
                frame=self.buffer_frame_counter,
                confirmed_target=self.confirmed_target_position,
                standoff_min_radius_m=self.buffer_standoff_min_radius,
                standoff_max_radius_m=self.buffer_standoff_max_radius,
                contract="buffer_requires_reachable_standoff_goal_before_localization",
            )
            if self.buffer_frame_counter >= self.buffer_timeout:
                self._fail_to_exploration(
                    step,
                    "direct_buffer_timeout_no_reachable_standoff_goal",
                    blacklist=True,
                )
            return
        current_goal_fmm = self._buffer_goal_fmm_diagnostics(target, payload)
        distance = self._world_distance_xz(payload.get("agent_position"), target)
        arrival_threshold = float(
            getattr(
                self.args,
                "minav_phase2_arrival_threshold",
                min(0.35, self.buffer_distance_threshold),
            )
        )
        arrived_by_distance = bool(
            force_arrived or (np.isfinite(distance) and distance <= arrival_threshold)
        )
        self._trace(
            "direct_buffer_update",
            step=step,
            frame=self.buffer_frame_counter,
            timeout=self.buffer_timeout,
            arrival_threshold=arrival_threshold,
            arrived_by_distance=arrived_by_distance,
            agent_position=payload.get("agent_position"),
            confirmed_target=self.confirmed_target_position,
            confirmed_target_map_anchor=self.confirmed_target_map_anchor_position,
            buffer_goal=target,
            distance=distance,
            goal_fmm=current_goal_fmm,
            navigation_goal_local_cell=self.navigation_goal_local_cell,
            contract="buffer_waits_for_ieve_fmm_phase_goal_before_localization",
            force_arrived=bool(force_arrived),
        )
        if arrived_by_distance:
            self._trace(
                "direct_buffer_distance_arrival",
                step=step,
                frame=self.buffer_frame_counter,
                distance=distance,
                threshold=arrival_threshold,
                target=target,
                minav_fmm_stop=payload.get("minav_fmm_stop"),
                force_arrived=bool(force_arrived),
            )
            if self.buffer_standoff_sweep_enabled:
                self._start_buffer_standoff_full_sweep(step, target)
            else:
                self._refresh_slam_training_before_localization(
                    step, source="BUFFER_DIRECT_ARRIVAL"
                )
                self._enter_localization_from_phase_goal(payload, step)
            return
        if self.buffer_frame_counter >= self.buffer_timeout:
            self._fail_to_exploration(
                step,
                "direct_buffer_timeout_waiting_for_ieve_fmm_goal",
                blacklist=True,
            )

    def _run_mr_remap_buffer_update(self, payload: Dict[str, Any], step: int) -> None:
        self.buffer_frame_counter += 1
        self.buffer_goal_update_counter += 1
        if not (
            self.mr_remap_pending
            or self.mr_remap_sweep_active
            or self.mr_remap_sweep_just_completed
        ):
            self._fail_to_exploration(
                step,
                "legacy_buffer_removed_non_direct_state",
                blacklist=False,
            )
            return

        agent_position = payload.get("agent_position")
        goal = self.buffer_current_goal_world
        goal_distance = self._world_distance_xz(agent_position, goal)
        arrived = bool(
            goal is not None
            and np.isfinite(goal_distance)
            and goal_distance <= self.buffer_distance_threshold
        )
        self._trace(
            "mr_remap_buffer_update",
            step=step,
            frame=self.buffer_frame_counter,
            agent_position=agent_position,
            goal=goal,
            goal_distance=goal_distance,
            arrived=arrived,
            mr_remap_pending=self.mr_remap_pending,
            mr_remap_sweep_active=self.mr_remap_sweep_active,
            mr_remap_sweep_just_completed=self.mr_remap_sweep_just_completed,
        )
        if self.confirmed_target_position is None:
            self._fail_to_exploration(
                step,
                "mr_remap_buffer_without_confirmed_target",
                blacklist=False,
            )
            return

        if self.mr_remap_sweep_just_completed:
            self._finish_mr_remap_sweep_and_signal_pre(step)
            return

        if self.mr_remap_sweep_active:
            self._trace(
                "mr_remap_sweep_active",
                step=step,
                remaining_steps=self.mr_remap_sweep_remaining_steps,
                total_steps=self.mr_remap_sweep_total_steps,
            )
            return

        if self.mr_remap_pending and arrived:
            self._start_mr_remap_full_sweep(step)
            return

        if self.buffer_frame_counter >= self.buffer_timeout:
            self._fail_to_exploration(
                step,
                "mr_remap_buffer_timeout",
                blacklist=True,
            )

    def _get_direct_buffer_goal_override(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Optional[str], bool]:
        if self.buffer_standoff_sweep_active:
            action = self._consume_buffer_standoff_sweep_action(context)
            return (
                self.buffer_current_goal_world,
                None,
                "MINAV_BUFFER_STANDOFF_360_SWEEP",
                action,
                False,
            )

        if self.buffer_standoff_sweep_just_completed:
            return (
                self.buffer_current_goal_world,
                None,
                "MINAV_BUFFER_STANDOFF_SWEEP_COMPLETE",
                "hold",
                True,
            )

        goal_anchor = self._confirmed_target_goal_anchor()
        target = self.buffer_current_goal_world
        if target is None and goal_anchor is not None:
            prepared = self._prepare_buffer_goal(
                goal_anchor,
                context,
                source="MINAV_BUFFER_CONFIRMED_TARGET_STANDOFF_INIT",
                search_radius_m=self.buffer_goal_search_radius,
                allow_adjustment=True,
                commit=True,
            )
            if prepared is not None:
                target = np.asarray(prepared["goal_world"], dtype=np.float32).reshape(-1)[:3]
        if target is None:
            reason = (
                "confirmed_target_missing"
                if goal_anchor is None
                else "no_reachable_standoff_goal"
            )
            goal_source = (
                "MINAV_BUFFER_CONFIRMED_TARGET_MISSING"
                if goal_anchor is None
                else "MINAV_BUFFER_STANDOFF_UNAVAILABLE"
            )
            self._trace(
                "direct_buffer_goal_unavailable",
                step=int(context.get("step", 0)),
                target=goal_anchor,
                reason=reason,
                standoff_min_radius_m=self.buffer_standoff_min_radius,
                standoff_max_radius_m=self.buffer_standoff_max_radius,
                navigation_goal_local_cell=self.navigation_goal_local_cell,
                rule="ieve_compute_temp_goal_distance",
            )
            return None, None, goal_source, "hold", True

        target = np.asarray(target, dtype=np.float32).reshape(-1)[:3]
        if (
            goal_anchor is not None
            and self._world_distance_xz(target, goal_anchor) < 0.25
        ):
            prepared = self._prepare_buffer_goal(
                goal_anchor,
                context,
                source="MINAV_BUFFER_CONFIRMED_TARGET_STANDOFF_REFRESH",
                search_radius_m=self.buffer_goal_search_radius,
                allow_adjustment=True,
                commit=True,
            )
            if prepared is not None:
                target = np.asarray(prepared["goal_world"], dtype=np.float32).reshape(-1)[:3]
            else:
                self._trace(
                    "direct_buffer_goal_unavailable",
                    step=int(context.get("step", 0)),
                    target=goal_anchor,
                    reason="no_reachable_standoff_goal",
                    standoff_min_radius_m=self.buffer_standoff_min_radius,
                    standoff_max_radius_m=self.buffer_standoff_max_radius,
                    navigation_goal_local_cell=self.navigation_goal_local_cell,
                    rule="ieve_compute_temp_goal_distance",
                )
                return (
                    None,
                    None,
                    "MINAV_BUFFER_STANDOFF_UNAVAILABLE",
                    "hold",
                    True,
                )

        goal_map = self.world_target_to_goal_map(target, context)
        if goal_map is None and goal_anchor is not None:
            prepared = self._prepare_buffer_goal(
                goal_anchor,
                context,
                source="MINAV_BUFFER_CONFIRMED_TARGET_STANDOFF_REPROJECT",
                search_radius_m=self.buffer_goal_search_radius,
                allow_adjustment=True,
                commit=True,
            )
            if prepared is not None:
                target = np.asarray(prepared["goal_world"], dtype=np.float32).reshape(-1)[:3]
                goal_map = self.world_target_to_goal_map(target, context)
            else:
                self._trace(
                    "direct_buffer_goal_unavailable",
                    step=int(context.get("step", 0)),
                    target=target,
                    reason="standoff_reproject_failed",
                    standoff_min_radius_m=self.buffer_standoff_min_radius,
                    standoff_max_radius_m=self.buffer_standoff_max_radius,
                    navigation_goal_local_cell=self.navigation_goal_local_cell,
                    rule="ieve_compute_temp_goal_distance",
                )
                return (
                    None,
                    None,
                    "MINAV_BUFFER_STANDOFF_UNAVAILABLE",
                    "hold",
                    True,
                )
        if goal_map is None:
            self._trace(
                "direct_buffer_goal_unavailable",
                step=int(context.get("step", 0)),
                target=target,
                navigation_goal_local_cell=self.navigation_goal_local_cell,
                rule="ieve_compute_temp_goal_distance",
            )
            return (
                None,
                None,
                "MINAV_BUFFER_STANDOFF_UNAVAILABLE",
                "hold",
                True,
            )

        self.buffer_current_goal_world = target.copy()
        return (
            self.buffer_current_goal_world,
            goal_map,
            "MINAV_BUFFER_CONFIRMED_TARGET_STANDOFF",
            None,
            False,
        )

    def _get_mr_remap_goal_override(
        self, context: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Optional[str], bool]:
        if self.mr_remap_sweep_active:
            action = self._consume_mr_remap_sweep_action(context)
            return (
                self.buffer_current_goal_world,
                None,
                "MINAV_MR_REMAP_360_SWEEP",
                action,
                False,
            )

        if self.mr_remap_sweep_just_completed:
            return (
                self.buffer_current_goal_world,
                None,
                "MINAV_MR_REMAP_SWEEP_COMPLETE",
                "hold",
                True,
            )

        target = self.buffer_current_goal_world
        if target is None:
            target = self._confirmed_target_goal_anchor()
        if target is None:
            return None, None, "MINAV_MR_REMAP_TARGET_MISSING", "hold", True
        target = np.asarray(target, dtype=np.float32).reshape(-1)[:3]
        goal_map = self.world_target_to_goal_map(target, context)
        if goal_map is None:
            return (
                target,
                None,
                "MINAV_MR_REMAP_TARGET_UNREACHABLE",
                "hold",
                True,
            )
        self.buffer_current_goal_world = target.copy()
        return (
            self.buffer_current_goal_world,
            goal_map,
            "MINAV_MR_REMAP_NAV_GOAL",
            None,
            False,
        )

    def _parse_habitat_completion_metrics(
        self, payload: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[bool], float]:
        habitat_distance = payload.get("distance_to_goal")
        habitat_success = payload.get("success")
        habitat_distance_f = None
        try:
            if habitat_distance is not None:
                habitat_distance_f = float(habitat_distance)
        except (TypeError, ValueError):
            habitat_distance_f = None

        habitat_success_b = None
        if habitat_success is not None:
            try:
                habitat_success_b = bool(float(habitat_success) > 0.0)
            except (TypeError, ValueError):
                habitat_success_b = bool(habitat_success)

        success_dist = float(getattr(self.args, "success_dist", 1.0))
        return habitat_distance_f, habitat_success_b, success_dist

    def _update_navigation_completion(self, payload: Dict[str, Any]) -> None:
        raw_target = self._navigation_raw_target()
        target = (
            self.navigation_goal_world
            if self.navigation_goal_world is not None
            else raw_target
        )
        if target is None:
            self.navigation_complete = False
            self._trace(
                "navigation_update",
                step=int(payload.get("step", 0)),
                has_target=False,
                navigation_complete=False,
            )
            return
        dist = self._navigation_map_distance_m(
            payload,
            full_cell=self.navigation_goal_full_cell,
            local_cell=self.navigation_goal_local_cell,
        )
        raw_dist = self._navigation_map_distance_m(
            payload,
            full_cell=self.navigation_raw_target_full_cell,
            local_cell=self.navigation_goal_local_cell,
        )
        habitat_distance_f, habitat_success_b, success_dist = (
            self._parse_habitat_completion_metrics(payload)
        )
        # The FMM goal can be an adjusted local waypoint when the localized
        # target is outside the current local map or falls on an obstacle.  That
        # waypoint is only an executor target; final STOP is gated by the raw
        # target's estimated map cell, not by Habitat GT metrics or coordinates.
        completion_distance = raw_dist if raw_target is not None else dist
        step = int(payload.get("step", 0))
        staging_goal = self.navigation_goal_tier == "staging"
        if staging_goal:
            completion_threshold = None
            completion_threshold_source = "staging_non_terminal"
        else:
            completion_threshold, completion_threshold_source = self._navigation_completion_threshold(
                payload=payload,
                completion_distance=completion_distance,
                step=step,
            )
        execution_reached = bool(
            payload.get("minav_navigation_goal_cell_reached", False)
            or payload.get("minav_fmm_stop", False)
            or (
                np.isfinite(dist)
                and dist
                <= max(
                    0.10,
                    3.0
                    * float(getattr(self.args, "map_resolution", 5.0))
                    / 100.0,
                )
            )
        )
        if (
            staging_goal
            and execution_reached
            and not self.navigation_staging_observation_pending
            and not self.navigation_fallback_force_stop
        ):
            self.navigation_staging_observation_pending = True
            self.navigation_phase3_tolerance_hold = False
            self._trace(
                "navigation_staging_arrived",
                step=step,
                raw_target_world=raw_target,
                execution_goal_world=target,
                execution_goal_distance_m=dist,
                raw_target_distance_m=raw_dist,
                staging_replan_count=self.navigation_staging_replan_count,
                staging_replan_limit=self.navigation_staging_replan_limit,
            )

        distance_complete = bool(
            self.navigation_may_terminal_stop
            and completion_threshold is not None
            and completion_distance <= completion_threshold
        )
        # In phase3, a fallback goal is the reachable execution point selected
        # around the best failed candidate. When that map goal is reached, the
        # raw candidate may still be outside the preferred radius because it is
        # obstructed or outside the local map. Stop on the estimated map goal
        # itself rather than waiting for an unreachable raw-target cell.
        phase3_fallback_execution_complete = bool(
            self._phase3_navigation_active()
            and self.navigation_goal_tier == "fallback"
            and self.navigation_may_terminal_stop
            and not self.navigation_staging_observation_pending
            and execution_reached
            and np.isfinite(dist)
            and dist
            <= max(
                0.10,
                3.0 * float(getattr(self.args, "map_resolution", 5.0)) / 100.0,
            )
        )
        if phase3_fallback_execution_complete:
            distance_complete = True
        relaxed_phase3_radius = (
            float(self.strategy_phase3_fallback_stop_radius)
            + float(self.phase3_navigation_stuck_tolerance_m)
        )
        self.navigation_phase3_tolerance_hold = bool(
            self._phase3_navigation_active()
            and self.navigation_goal_tier == "fallback"
            and not distance_complete
            and not self.phase3_navigation_stuck_tolerance_active
            and execution_reached
            and np.isfinite(raw_dist)
            and raw_dist <= relaxed_phase3_radius
        )
        habitat_distance_complete = bool(
            habitat_distance_f is not None and habitat_distance_f <= success_dist
        )
        habitat_success_complete = bool(habitat_success_b is True)
        self.navigation_complete = bool(distance_complete)
        if phase3_fallback_execution_complete:
            completion_source = "phase3_fallback_execution_goal"
        elif distance_complete and raw_target is not None:
            completion_source = "localized_target_distance"
        elif distance_complete:
            completion_source = "navigation_goal_distance"
        elif raw_target is not None:
            completion_source = "localized_target_distance_waiting"
        else:
            completion_source = "navigation_goal_distance_waiting"
        habitat_navigation_gap = None
        near_habitat_goal_without_nav_complete = False
        if habitat_distance_f is not None:
            try:
                habitat_navigation_gap = float(completion_distance) - float(
                    habitat_distance_f
                )
                near_habitat_goal_without_nav_complete = bool(
                    habitat_distance_f <= success_dist
                    and not self.navigation_complete
                )
            except (TypeError, ValueError):
                habitat_navigation_gap = None
                near_habitat_goal_without_nav_complete = False
        self._trace(
            "navigation_update",
            step=int(payload.get("step", 0)),
            has_target=True,
            agent_position=payload.get("agent_position"),
            distance=dist,
            raw_target_distance=raw_dist,
            completion_distance=completion_distance,
            threshold=completion_threshold,
            threshold_source=completion_threshold_source,
            distance_complete=distance_complete,
            phase3_fallback_execution_complete=phase3_fallback_execution_complete,
            habitat_distance=habitat_distance_f,
            habitat_success=habitat_success_b,
            habitat_success_dist=success_dist,
            habitat_distance_complete=habitat_distance_complete,
            habitat_success_complete=habitat_success_complete,
            habitat_navigation_gap=habitat_navigation_gap,
            near_habitat_goal_without_nav_complete=near_habitat_goal_without_nav_complete,
            habitat_metrics_used_for_stop=False,
            completion_source=completion_source,
            navigation_complete=self.navigation_complete,
            target=target,
            raw_target=raw_target,
            navigation_goal_world=self.navigation_goal_world,
            navigation_goal_local_cell=self.navigation_goal_local_cell,
            navigation_goal_adjusted=self.navigation_goal_adjusted,
            navigation_goal_tier=self.navigation_goal_tier,
            effective_stop_radius_m=self.navigation_effective_stop_radius_m,
            may_terminal_stop=self.navigation_may_terminal_stop,
            execution_goal_reached=execution_reached,
            staging_observation_pending=self.navigation_staging_observation_pending,
            staging_replan_count=self.navigation_staging_replan_count,
            fallback_stop=self.navigation_fallback_force_stop,
            fallback_stop_reason=self.navigation_fallback_force_stop_reason,
            phase3_tolerance_hold=self.navigation_phase3_tolerance_hold,
            phase3_navigation_best_distance=self.phase3_navigation_best_distance,
            phase3_navigation_stuck_frames=self.phase3_navigation_stuck_frames,
            phase3_navigation_stuck_tolerance_active=(
                self.phase3_navigation_stuck_tolerance_active
            ),
        )

    def _reset_phase3_navigation_progress(self) -> None:
        self.phase3_navigation_best_distance = float("inf")
        self.phase3_navigation_stuck_frames = 0
        self.phase3_navigation_stuck_tolerance_active = False

    def _navigation_completion_threshold(
        self,
        *,
        payload: Dict[str, Any],
        completion_distance: float,
        step: int,
    ) -> Tuple[float, str]:
        if self.localization_stage == "phase3_best_evidence_navigation":
            radius = float(self.strategy_phase3_fallback_stop_radius)
            if np.isfinite(radius) and radius > 0.0:
                relaxed_radius = (
                    radius + self.phase3_navigation_stuck_tolerance_m
                )
                distance = float(completion_distance)
                if np.isfinite(distance):
                    improved = bool(
                        distance
                        < self.phase3_navigation_best_distance
                        - self.phase3_navigation_progress_epsilon_m
                    )
                    if improved:
                        self.phase3_navigation_best_distance = distance
                        self.phase3_navigation_stuck_frames = 0
                    elif distance <= relaxed_radius:
                        self.phase3_navigation_stuck_frames += 1
                    else:
                        self.phase3_navigation_stuck_frames = 0
                    if (
                        not self.phase3_navigation_stuck_tolerance_active
                        and self.phase3_navigation_stuck_tolerance_m > 0.0
                        and distance <= relaxed_radius
                        and self.phase3_navigation_stuck_frames
                        >= self.phase3_navigation_stuck_patience
                    ):
                        self.phase3_navigation_stuck_tolerance_active = True
                        self._trace(
                            "phase3_navigation_stuck_tolerance_activated",
                            step=step,
                            distance=distance,
                            base_threshold=radius,
                            relaxed_threshold=relaxed_radius,
                            tolerance_m=self.phase3_navigation_stuck_tolerance_m,
                            stuck_frames=self.phase3_navigation_stuck_frames,
                            stuck_patience=self.phase3_navigation_stuck_patience,
                            progress_epsilon_m=(
                                self.phase3_navigation_progress_epsilon_m
                            ),
                            fmm_stop=bool(payload.get("minav_fmm_stop", False)),
                            been_stuck=bool(
                                payload.get("minav_been_stuck", False)
                                or payload.get("ieve_frontier_been_stuck", False)
                            ),
                            contract="phase3_near_target_stuck_enables_small_stop_tolerance",
                        )
                else:
                    self.phase3_navigation_stuck_frames = 0
                if self.phase3_navigation_stuck_tolerance_active:
                    self.navigation_effective_stop_radius_m = float(relaxed_radius)
                    return (
                        relaxed_radius,
                        "phase3_best_evidence_navigation_stuck_tolerance",
                    )
                self.navigation_effective_stop_radius_m = float(radius)
                return radius, "phase3_best_evidence_navigation"
        if self.navigation_effective_stop_radius_m is not None:
            return (
                float(self.navigation_effective_stop_radius_m),
                "navigation_reachable_{}_tier".format(
                    str(self.navigation_goal_tier or "goal")
                ),
            )
        return self.navigation_arrival_threshold, "navigation_arrival_threshold"

    def _estimated_navigation_start_cell(
        self, payload: Dict[str, Any]
    ) -> Optional[Tuple[int, int]]:
        pose_pred = payload.get("pose_pred")
        if pose_pred is None:
            return None
        try:
            pose = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
            if pose.size < 2 or not np.all(np.isfinite(pose[:2])):
                return None
            resolution_cm = float(
                payload.get(
                    "map_resolution_cm", getattr(self.args, "map_resolution", 5.0)
                )
            )
            row = int(round(float(pose[1]) * 100.0 / resolution_cm))
            col = int(round(float(pose[0]) * 100.0 / resolution_cm))
            return row, col
        except Exception:
            return None

    def _navigation_map_distance_m(
        self,
        payload: Dict[str, Any],
        *,
        full_cell: Optional[Tuple[int, int]] = None,
        local_cell: Optional[Tuple[int, int]] = None,
    ) -> float:
        start_cell = self._estimated_navigation_start_cell(payload)
        if start_cell is None:
            return float("inf")
        target_cell = full_cell if full_cell is not None else local_cell
        if target_cell is None:
            return float("inf")
        resolution_m = float(
            payload.get("map_resolution_cm", getattr(self.args, "map_resolution", 5.0))
        ) / 100.0
        return float(
            np.hypot(
                float(start_cell[0]) - float(target_cell[0]),
                float(start_cell[1]) - float(target_cell[1]),
            )
            * resolution_m
        )

    @staticmethod
    def _world_distance_xz(a, b) -> float:
        if a is None or b is None:
            return float("inf")
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.size < 3 or b.size < 3:
            return float("inf")
        return float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))

    def _debug_target_navigation_metrics(
        self,
        payload: Optional[Dict[str, Any]],
        confirmed_target: Optional[np.ndarray] = None,
        buffer_goal: Optional[np.ndarray] = None,
        detected_target: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self._debug_enabled():
            return None

        context = dict(payload or self._last_payload or {})
        context.setdefault("current_world_xyz", context.get("agent_position"))
        context.setdefault(
            "map_resolution_cm",
            getattr(self.args, "map_resolution", self.occupancy_adapter.map_resolution),
        )

        def _point(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if value is None:
                return None
            try:
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:
                return None
            if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
                return None
            return arr[:3].copy()

        confirmed = _point(confirmed_target)
        if confirmed is None:
            confirmed = _point(self.confirmed_target_position)

        detected = _point(detected_target)
        if detected is None:
            detected = _point(self.confirmation_detected_target_position)
        if detected is None:
            detected = _point(self.detected_target_position)

        goal = _point(buffer_goal)
        if goal is None:
            goal = _point(self.navigation_goal_world)

        agent = _point(context.get("agent_position"))
        if agent is None:
            agent = _point(context.get("current_world_xyz"))

        def _projection_fields(point: Optional[np.ndarray]) -> Dict[str, Any]:
            projection = self._project_confirmation_center_to_grid(point, context)
            semantic = self._debug_semantic_distance_to_projection(projection, context)
            return {
                "projection": projection,
                "semantic_distance": semantic,
                "semantic_m": semantic.get("meters"),
                "semantic_hit": semantic.get("hit"),
                "semantic_value": semantic.get("semantic_value"),
                "full_cell": None if projection is None else projection.get("raw_full_cell"),
                "local_cell": None if projection is None else projection.get("raw_local_cell"),
            }

        confirmed_fields = _projection_fields(confirmed)
        detected_fields = _projection_fields(detected)
        goal_fields = _projection_fields(goal)

        return {
            "agent_position": agent,
            "confirmed_target": confirmed,
            "detected_target": detected,
            "buffer_goal": goal,
            "confirmed_center_sem_m": confirmed_fields["semantic_m"],
            "confirmed_center_sem_hit": confirmed_fields["semantic_hit"],
            "confirmed_center_sem_value": confirmed_fields["semantic_value"],
            "confirmed_center_full_cell": confirmed_fields["full_cell"],
            "confirmed_center_local_cell": confirmed_fields["local_cell"],
            "detected_center_sem_m": detected_fields["semantic_m"],
            "detected_center_sem_hit": detected_fields["semantic_hit"],
            "detected_center_sem_value": detected_fields["semantic_value"],
            "detected_center_full_cell": detected_fields["full_cell"],
            "detected_center_local_cell": detected_fields["local_cell"],
            "buffer_goal_sem_m": goal_fields["semantic_m"],
            "buffer_goal_sem_hit": goal_fields["semantic_hit"],
            "buffer_goal_full_cell": goal_fields["full_cell"],
            "buffer_goal_local_cell": goal_fields["local_cell"],
            "detected_vs_confirmed_m": self._world_distance_xz(detected, confirmed),
            "buffer_goal_to_confirmed_m": self._world_distance_xz(goal, confirmed),
            "buffer_goal_to_detected_m": self._world_distance_xz(goal, detected),
            "agent_to_confirmed_m": self._world_distance_xz(agent, confirmed),
            "agent_to_detected_m": self._world_distance_xz(agent, detected),
            "agent_to_buffer_goal_m": self._world_distance_xz(agent, goal),
        }

    @staticmethod
    def _yaw_from_position_to_target(position, target) -> Optional[float]:
        if position is None or target is None:
            return None
        position = np.asarray(position, dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        if position.size < 3 or target.size < 3:
            return None
        return float(np.arctan2(target[0] - position[0], target[2] - position[2]))

    @staticmethod
    def _coerce_yaw_to_rad(yaw) -> Optional[float]:
        if yaw is None:
            return None
        try:
            value = float(yaw)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        if abs(value) > (2.0 * np.pi + 1e-3):
            value = np.deg2rad(value)
        return float(np.arctan2(np.sin(value), np.cos(value)))

    @staticmethod
    def _yaw_from_quat_xyzw(rotation) -> Optional[float]:
        if rotation is None:
            return None
        try:
            qx, qy, qz, qw = np.asarray(rotation, dtype=np.float64).reshape(-1)[:4]
        except Exception:
            return None
        if qw < 0:
            qx, qy, qz, qw = -qx, -qy, -qz, -qw
        siny_cosp = 2.0 * (qw * qy + qx * qz)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _angle_diff_rad(target: float, current: float) -> float:
        return float(np.arctan2(np.sin(target - current), np.cos(target - current)))

    def _full_turn_sweep_steps(self) -> int:
        explicit_steps = int(getattr(self.args, "initial_spin_steps", 0) or 0)
        if explicit_steps > 0:
            return max(1, explicit_steps)
        turn_angle = float(getattr(self.args, "turn_angle", 30.0) or 30.0)
        turn_angle = max(1.0, abs(turn_angle))
        return max(1, int(np.ceil(360.0 / turn_angle)))

    # ------------------------------------------------------------------
    # Gaussian helpers
    # ------------------------------------------------------------------
    def _get_gaussians_for_confirmation(self):
        if self.slam_mapper is None:
            return None, "SLAM mapper not available"
        if not getattr(self.slam_mapper, "is_running", False):
            return None, "SLAM mapper not running"
        slam = getattr(self.slam_mapper, "slam", None)
        if slam is None:
            return None, "SLAM not available"
        gaussians = getattr(slam, "gaussians", None)
        if gaussians is None or not hasattr(gaussians, "get_xyz"):
            return None, "Gaussians not available"
        return gaussians, None

    def _assemble_target_confirm_candidates_from_window(
        self, gaussians, step: int
    ) -> List[Dict[str, Any]]:
        frame_candidates: List[Dict[str, Any]] = []
        for frame in list(self.frame_window):
            camera = frame.get("slam_camera") or self._latest_slam_camera()
            if camera is None:
                continue
            depth = frame.get("depth_m", frame.get("raw_depth"))
            frame_step = int(frame.get("step", step))
            for instance_idx, instance in enumerate(frame.get("target_instances", []) or []):
                mask = instance.get("binary_mask")
                if mask is None:
                    continue
                projected = self._project_instance_mask_to_gaussians(
                    gaussians,
                    mask,
                    camera,
                    depth,
                )
                if self._mask_projection_filter_debug_enabled():
                    filter_debug = self._debug_mask_projection_filter_counts(
                        gaussians,
                        mask,
                        camera,
                        depth,
                    )
                    self._trace(
                        "target_confirmation_mask_projection_filter_debug",
                        step=step,
                        frame_step=frame_step,
                        instance_idx=int(instance_idx),
                        camera_uid=getattr(camera, "uid", None),
                        latest_slam_kf_id=self._latest_slam_keyframe_id(),
                        mask_pixel_count=int(np.asarray(mask).sum()) if mask is not None else 0,
                        instance_pixel_count=int(instance.get("pixel_count", 0) or 0),
                        projected_indices_count=int(projected.size),
                        depth_shape=None if depth is None else np.asarray(depth).shape,
                        depth_tolerance=float(
                            getattr(self.args, "minav_confirm_depth_tolerance", 0.3)
                        ),
                        **filter_debug,
                    )
                if self._debug_enabled():
                    self._trace(
                        "target_confirmation_mask_projection_debug",
                        step=step,
                        frame_step=frame_step,
                        instance_idx=int(instance_idx),
                        camera_uid=getattr(camera, "uid", None),
                        latest_slam_kf_id=self._latest_slam_keyframe_id(),
                        mask_pixel_count=int(np.asarray(mask).sum()) if mask is not None else 0,
                        instance_pixel_count=int(instance.get("pixel_count", 0) or 0),
                        projected_indices_count=int(projected.size),
                        depth_shape=None if depth is None else np.asarray(depth).shape,
                        depth_tolerance=float(
                            getattr(self.args, "minav_confirm_depth_tolerance", 0.3)
                        ),
                    )
                if projected.size == 0:
                    continue
                clusters = self._expand_projected_indices_to_full_target_clusters(
                    gaussians,
                    projected,
                )
                for cluster_idx, cluster in enumerate(clusters):
                    frame_candidates.append(
                        {
                            "frame_step": frame_step,
                            "instance_idx": int(instance_idx),
                            "cluster_idx_in_frame": int(cluster_idx),
                            "global_indices": cluster["global_indices"],
                            "center_3d": cluster["center_3d"],
                            "size": int(cluster["size"]),
                            "bbox_min": cluster.get("bbox_min"),
                            "bbox_max": cluster.get("bbox_max"),
                            "bbox_extent": cluster.get("bbox_extent"),
                            "xz_extent_m": float(cluster.get("xz_extent_m", 0.0) or 0.0),
                            "y_extent_m": float(cluster.get("y_extent_m", 0.0) or 0.0),
                            "confidence": float(instance.get("confidence", 0.0) or 0.0),
                            "pixel_count": int(instance.get("pixel_count", 0) or 0),
                        }
                    )
                    if self._debug_enabled():
                        cluster_indices = np.asarray(
                            cluster["global_indices"], dtype=np.int64
                        ).reshape(-1)
                        self._trace(
                            "target_confirmation_cluster_reprojection_debug",
                            step=step,
                            frame_step=frame_step,
                            instance_idx=int(instance_idx),
                            cluster_idx_in_frame=int(cluster_idx),
                            camera_uid=getattr(camera, "uid", None),
                            cluster_size=int(cluster_indices.size),
                            center_3d=cluster.get("center_3d"),
                            reprojection=self._debug_cluster_mask_reprojection_stats(
                                gaussians,
                                cluster_indices,
                                mask,
                                camera,
                                depth,
                            ),
                        )

        if not frame_candidates:
            payload = self._last_payload or {}
            return self._fallback_target_confirm_candidates_from_semantic(
                gaussians,
                payload,
                step,
                reason="recent_instance_projection_empty",
            )

        merge_dist_m = float(getattr(self.args, "minav_confirm_merge_dist_m", 0.35))
        merge_overlap_ratio = float(
            getattr(self.args, "minav_confirm_merge_overlap_ratio", 0.25)
        )
        merged_candidates: List[Dict[str, Any]] = []
        for candidate in sorted(
            frame_candidates,
            key=lambda c: (c["confidence"], c["size"], c["pixel_count"]),
            reverse=True,
        ):
            candidate_indices = np.asarray(candidate["global_indices"], dtype=np.int64)
            candidate_set = set(candidate_indices.tolist())
            matched = None
            for merged in merged_candidates:
                intersection = len(candidate_set & merged["global_index_set"])
                overlap_ratio = intersection / max(
                    min(len(candidate_set), len(merged["global_index_set"])),
                    1,
                )
                dist_xz = float(
                    np.linalg.norm(
                        candidate["center_3d"][[0, 2]] - merged["center_3d"][[0, 2]]
                    )
                )
                if overlap_ratio >= merge_overlap_ratio or dist_xz <= merge_dist_m:
                    matched = merged
                    break

            if matched is None:
                merged_candidates.append(
                    {
                        "global_index_set": set(candidate_set),
                        "support_steps": {int(candidate["frame_step"])},
                        "max_confidence": float(candidate["confidence"]),
                        "max_pixel_count": int(candidate["pixel_count"]),
                        "center_3d": np.asarray(candidate["center_3d"], dtype=np.float32).copy(),
                    }
                )
            else:
                matched["global_index_set"].update(candidate_set)
                matched["support_steps"].add(int(candidate["frame_step"]))
                matched["max_confidence"] = max(
                    float(matched["max_confidence"]),
                    float(candidate["confidence"]),
                )
                matched["max_pixel_count"] = max(
                    int(matched["max_pixel_count"]),
                    int(candidate["pixel_count"]),
                )

        finalized: List[Dict[str, Any]] = []
        for merged_idx, merged in enumerate(merged_candidates):
            indices = np.asarray(sorted(merged["global_index_set"]), dtype=np.int64)
            record = self._build_target_confirm_cluster_record(gaussians, indices)
            if record is None:
                continue
            record.update(
                {
                    "filtered_idx": int(merged_idx),
                    "raw_label": int(merged_idx),
                    "support_steps": sorted(int(s) for s in merged["support_steps"]),
                    "support_count": int(len(merged["support_steps"])),
                    "confidence": float(merged["max_confidence"]),
                    "pixel_count": int(merged["max_pixel_count"]),
                }
            )
            current = self._last_payload.get("agent_position") if self._last_payload else None
            if current is not None:
                current = np.asarray(current, dtype=np.float32).reshape(-1)
                if current.size >= 3:
                    record["detection_dist_m"] = float(
                        np.linalg.norm(record["center_3d"][[0, 2]] - current[[0, 2]])
                    )
            finalized.append(record)

        if not finalized:
            payload = self._last_payload or {}
            return self._fallback_target_confirm_candidates_from_semantic(
                gaussians,
                payload,
                step,
                reason="merged_window_candidates_empty",
            )

        finalized.sort(
            key=lambda c: (
                int(c.get("support_count", 1) or 1),
                float(c.get("confidence", 0.0) or 0.0),
                int(c.get("size", 0) or 0),
                int(c.get("pixel_count", 0) or 0),
            ),
            reverse=True,
        )
        return finalized

    def _project_instance_mask_to_gaussians(
        self,
        gaussians,
        binary_mask,
        camera,
        depth,
    ) -> np.ndarray:
        if gaussians is None or not hasattr(gaussians, "project_mask_to_gaussian_indices"):
            return np.array([], dtype=np.int64)
        mask_np = np.asarray(binary_mask, dtype=np.uint8)
        if mask_np.ndim != 2:
            return np.array([], dtype=np.int64)
        dilate_pixels = int(getattr(self.args, "minav_confirm_mask_dilate_pixels", 3))
        if dilate_pixels > 0:
            try:
                import cv2

                kernel = np.ones((2 * dilate_pixels + 1, 2 * dilate_pixels + 1), dtype=np.uint8)
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            except Exception:
                pass
        try:
            projected = gaussians.project_mask_to_gaussian_indices(
                target_mask_2d=mask_np.astype(np.float32),
                camera=camera,
                depth=depth,
                depth_tolerance=float(getattr(self.args, "minav_confirm_depth_tolerance", 0.3)),
            )
        except Exception as exc:
            self.logger.debug("target-confirm mask projection failed: %s", exc)
            return np.array([], dtype=np.int64)
        return np.unique(np.asarray(projected, dtype=np.int64).reshape(-1))

    def _debug_mask_projection_filter_counts(
        self,
        gaussians,
        binary_mask,
        camera,
        depth,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "total_gaussians": 0,
            "valid_z_count": 0,
            "valid_uv_count": 0,
            "valid_zuv_count": 0,
            "mask_hit_count": 0,
            "observed_depth_valid_count": None,
            "depth_valid_count": None,
            "final_projected_count": 0,
            "depth_diff_min": None,
            "depth_diff_p50": None,
            "depth_diff_p90": None,
            "depth_diff_count": 0,
            "debug_error": None,
        }
        try:
            import torch

            xyz_world = getattr(gaussians, "get_xyz", None)
            if callable(xyz_world):
                xyz_world = xyz_world()
            if xyz_world is None:
                xyz_world = getattr(gaussians, "_xyz", None)
            if xyz_world is None:
                stats["debug_error"] = "gaussian_xyz_unavailable"
                return stats
            if not hasattr(xyz_world, "shape"):
                stats["debug_error"] = "gaussian_xyz_invalid"
                return stats
            if not torch.is_tensor(xyz_world):
                xyz_world = torch.as_tensor(xyz_world, dtype=torch.float32)
            total_gaussians = int(xyz_world.shape[0])
            stats["total_gaussians"] = total_gaussians
            if total_gaussians <= 0:
                return stats

            xyz_world = xyz_world.detach()
            device = xyz_world.device
            dtype = xyz_world.dtype if xyz_world.is_floating_point() else torch.float32
            xyz_world = xyz_world.to(dtype=dtype)

            mask_np = np.asarray(binary_mask, dtype=np.uint8)
            if mask_np.ndim != 2:
                stats["debug_error"] = "mask_not_2d"
                return stats
            dilate_pixels = int(
                getattr(self.args, "minav_confirm_mask_dilate_pixels", 3)
            )
            stats["mask_dilate_pixels"] = dilate_pixels
            if dilate_pixels > 0:
                try:
                    import cv2

                    kernel = np.ones(
                        (2 * dilate_pixels + 1, 2 * dilate_pixels + 1),
                        dtype=np.uint8,
                    )
                    mask_np = cv2.dilate(mask_np, kernel, iterations=1)
                except Exception as exc:
                    stats["mask_dilate_error"] = str(exc)
            stats["dilated_mask_pixel_count"] = int(mask_np.sum())
            height, width = mask_np.shape[:2]

            if hasattr(camera, "world_view_transform"):
                w2c = camera.world_view_transform.T
            elif hasattr(camera, "R") and hasattr(camera, "T"):
                from gaussian_mapping.gaussian_splatting.utils.graphics_utils import (
                    getWorld2View2,
                )

                w2c = getWorld2View2(camera.R, camera.T)
            else:
                stats["debug_error"] = "camera_transform_unavailable"
                return stats
            if not torch.is_tensor(w2c):
                w2c = torch.as_tensor(w2c, dtype=dtype, device=device)
            else:
                w2c = w2c.to(device=device, dtype=dtype)

            fx = float(getattr(camera, "fx"))
            fy = float(getattr(camera, "fy"))
            cx = float(getattr(camera, "cx"))
            cy = float(getattr(camera, "cy"))

            ones = torch.ones((total_gaussians, 1), device=device, dtype=dtype)
            xyz_homo = torch.cat([xyz_world, ones], dim=1)
            xyz_cam = (w2c @ xyz_homo.T).T[:, :3]
            valid_z = xyz_cam[:, 2] > 0.1
            z = torch.clamp(xyz_cam[:, 2], min=1e-6)
            u = xyz_cam[:, 0] * fx / z + cx
            v = xyz_cam[:, 1] * fy / z + cy
            valid_u = (u >= 0) & (u < width)
            valid_v = (v >= 0) & (v < height)
            valid_uv = valid_u & valid_v
            valid_zuv = valid_z & valid_uv
            stats["valid_z_count"] = int(valid_z.sum().item())
            stats["valid_uv_count"] = int(valid_uv.sum().item())
            stats["valid_zuv_count"] = int(valid_zuv.sum().item())

            u_int = u.long().clamp(0, width - 1)
            v_int = v.long().clamp(0, height - 1)
            mask_tensor = torch.from_numpy(mask_np).float().to(device=device)
            mask_hit = mask_tensor[v_int, u_int] > 0.5
            mask_valid = valid_zuv & mask_hit
            stats["mask_hit_count"] = int(mask_valid.sum().item())

            if depth is None:
                stats["final_projected_count"] = int(mask_valid.sum().item())
                return stats

            if torch.is_tensor(depth):
                depth_tensor = depth.detach().to(device=device, dtype=dtype)
            else:
                depth_tensor = torch.as_tensor(
                    np.asarray(depth), device=device, dtype=dtype
                )
            if depth_tensor.ndim > 2:
                depth_tensor = depth_tensor.squeeze()
            if depth_tensor.ndim != 2:
                stats["debug_error"] = "depth_not_2d"
                return stats
            if int(depth_tensor.shape[0]) != height or int(depth_tensor.shape[1]) != width:
                stats["debug_error"] = "depth_mask_shape_mismatch"
                stats["depth_shape_after_squeeze"] = [
                    int(depth_tensor.shape[0]),
                    int(depth_tensor.shape[1]),
                ]
                return stats

            rendered_depth = xyz_cam[:, 2]
            observed_depth = depth_tensor[v_int, u_int]
            observed_valid = torch.isfinite(observed_depth) & (observed_depth > 0)
            stats["observed_depth_valid_count"] = int(
                (mask_valid & observed_valid).sum().item()
            )
            depth_diff = torch.abs(rendered_depth - observed_depth)
            depth_valid = depth_diff < float(
                getattr(self.args, "minav_confirm_depth_tolerance", 0.3)
            )
            final_valid = mask_valid & depth_valid
            stats["depth_valid_count"] = int(final_valid.sum().item())
            stats["final_projected_count"] = int(final_valid.sum().item())

            diff_mask = mask_valid & observed_valid & torch.isfinite(depth_diff)
            if bool(diff_mask.any().item()):
                diffs = depth_diff[diff_mask].detach().cpu().numpy().astype(np.float64)
                stats["depth_diff_count"] = int(diffs.size)
                stats["depth_diff_min"] = float(np.min(diffs))
                stats["depth_diff_p50"] = float(np.percentile(diffs, 50))
                stats["depth_diff_p90"] = float(np.percentile(diffs, 90))
            return stats
        except Exception as exc:
            stats["debug_error"] = str(exc)
            return stats

    def _build_target_confirm_cluster_record(
        self, gaussians, global_indices: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        indices = np.unique(np.asarray(global_indices, dtype=np.int64).reshape(-1))
        if indices.size == 0:
            return None
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception:
            return None
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            return None
        xyz = xyz_all[valid]
        bbox_min = xyz.min(axis=0).astype(np.float32)
        bbox_max = xyz.max(axis=0).astype(np.float32)
        bbox_extent = (bbox_max - bbox_min).astype(np.float32)
        return {
            "global_indices": valid.astype(np.int64),
            "indices": valid.astype(np.int64),
            "center_3d": xyz.mean(axis=0).astype(np.float32),
            "center": xyz.mean(axis=0).astype(np.float32),
            "size": int(valid.size),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_extent": bbox_extent,
            "xz_extent_m": float(np.linalg.norm(bbox_extent[[0, 2]])),
            "y_extent_m": float(bbox_extent[1]),
        }

    def _cluster_target_confirm_indices(
        self, gaussians, global_indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        indices = np.unique(np.asarray(global_indices, dtype=np.int64).reshape(-1))
        if indices.size == 0:
            return []
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception:
            return []
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            return []
        xyz = xyz_all[valid]
        eps = float(getattr(self.args, "minav_confirm_local_cluster_eps", self.cluster_eps))
        min_samples = int(getattr(self.args, "minav_confirm_local_cluster_min_samples", 4))
        min_cluster_size = int(
            getattr(self.args, "minav_confirm_local_cluster_min_size", self.cluster_min_size)
        )
        clusters: List[Dict[str, Any]] = []
        if valid.size >= max(min_samples, 2):
            try:
                from sklearn.cluster import DBSCAN

                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz)
            except Exception:
                labels = np.zeros((valid.size,), dtype=np.int32)
            for label in sorted(set(labels.tolist()) - {-1}):
                cluster_indices = valid[labels == label]
                if cluster_indices.size < min_cluster_size:
                    continue
                record = self._build_target_confirm_cluster_record(gaussians, cluster_indices)
                if record is not None:
                    clusters.append(record)
        if not clusters and valid.size >= min_cluster_size:
            record = self._build_target_confirm_cluster_record(gaussians, valid)
            if record is not None:
                clusters.append(record)
        return clusters

    def _expand_projected_indices_to_full_target_clusters(
        self, gaussians, global_indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        indices = np.unique(np.asarray(global_indices, dtype=np.int64).reshape(-1))
        if indices.size == 0:
            return []
        if not bool(getattr(self.args, "minav_confirm_expand_full_clusters", 0)):
            return self._cluster_target_confirm_indices(gaussians, indices)

        semantic_clusters = self._expand_projected_indices_to_semantic_clusters(
            gaussians, indices
        )
        if semantic_clusters:
            return semantic_clusters
        geometric_clusters = self._expand_projected_indices_to_geometric_clusters(
            gaussians, indices
        )
        if geometric_clusters:
            return geometric_clusters
        return self._cluster_target_confirm_indices(gaussians, indices)

    def _expand_projected_indices_to_semantic_clusters(
        self, gaussians, global_indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        if gaussians is None:
            return []
        semantic = getattr(gaussians, "get_semantic_target", None)
        if callable(semantic):
            semantic = semantic()
        if semantic is None or not hasattr(semantic, "numel") or semantic.numel() == 0:
            return []
        try:
            import torch

            threshold = float(
                getattr(
                    self.args,
                    "minav_confirm_semantic_threshold",
                    getattr(self.args, "minav_semantic_threshold", 0.6),
                )
            )
            semantic_indices = (
                torch.where(semantic > threshold)[0].detach().cpu().numpy().astype(np.int64)
            )
        except Exception:
            return []
        if semantic_indices.size < self.cluster_min_size:
            return []
        clusters = self._cluster_target_confirm_indices(gaussians, semantic_indices)
        expanded = []
        seeds = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        for cluster in clusters:
            cluster_indices = self._cluster_indices(cluster)
            if np.intersect1d(cluster_indices, seeds).size > 0:
                expanded.append(cluster)
        return expanded

    def _expand_projected_indices_to_geometric_clusters(
        self, gaussians, global_indices: np.ndarray
    ) -> List[Dict[str, Any]]:
        indices = np.unique(np.asarray(global_indices, dtype=np.int64).reshape(-1))
        if indices.size == 0:
            return []
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception:
            return []
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            return []

        seed_xyz = xyz_all[valid]
        pad_xz = float(getattr(self.args, "minav_confirm_full_cluster_bbox_pad_xz", 0.25))
        pad_y = float(getattr(self.args, "minav_confirm_full_cluster_bbox_pad_y", 0.35))
        bbox_min = seed_xyz.min(axis=0)
        bbox_max = seed_xyz.max(axis=0)
        roi_mask = (
            (xyz_all[:, 0] >= bbox_min[0] - pad_xz)
            & (xyz_all[:, 0] <= bbox_max[0] + pad_xz)
            & (xyz_all[:, 1] >= bbox_min[1] - pad_y)
            & (xyz_all[:, 1] <= bbox_max[1] + pad_y)
            & (xyz_all[:, 2] >= bbox_min[2] - pad_xz)
            & (xyz_all[:, 2] <= bbox_max[2] + pad_xz)
        )
        roi_indices = np.where(roi_mask)[0].astype(np.int64)
        if roi_indices.size == 0:
            return []

        roi_xyz = xyz_all[roi_indices]
        eps = float(
            getattr(
                self.args,
                "minav_confirm_full_cluster_eps",
                getattr(self.args, "minav_confirm_local_cluster_eps", self.cluster_eps),
            )
        )
        min_samples = int(
            getattr(
                self.args,
                "minav_confirm_full_cluster_min_samples",
                getattr(self.args, "minav_confirm_local_cluster_min_samples", 4),
            )
        )
        min_cluster_size = int(
            getattr(self.args, "minav_confirm_local_cluster_min_size", self.cluster_min_size)
        )

        expanded: List[Dict[str, Any]] = []
        if roi_indices.size >= max(min_samples, 2):
            try:
                from sklearn.cluster import DBSCAN

                labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(roi_xyz)
            except Exception:
                labels = np.zeros((roi_indices.size,), dtype=np.int32)
            for label in sorted(set(labels.tolist()) - {-1}):
                cluster_indices = roi_indices[labels == label]
                if cluster_indices.size < min_cluster_size:
                    continue
                if np.intersect1d(cluster_indices, valid).size == 0:
                    continue
                record = self._build_target_confirm_cluster_record(
                    gaussians, cluster_indices
                )
                if record is not None:
                    expanded.append(record)
        return expanded

    def _fallback_target_confirm_candidates_from_semantic(
        self,
        gaussians,
        payload: Dict[str, Any],
        step: int,
        reason: str,
    ) -> List[Dict[str, Any]]:
        indices = self._collect_candidate_indices(gaussians, payload)
        if indices.size == 0:
            self._trace(
                "target_confirmation_semantic_fallback_empty",
                step=step,
                reason=reason,
            )
            return []

        clusters = self._select_candidate_clusters(gaussians, indices, payload)
        finalized: List[Dict[str, Any]] = []
        best_conf = max((float(c) for c in self.conf_history), default=0.0)
        support_steps = [
            int(frame.get("step", step)) for frame in list(self.frame_window)
        ]
        support_count = max(
            int(self.min_support_count),
            len(set(support_steps)),
            1,
        )
        for cluster_idx, cluster in enumerate(clusters):
            cluster_indices = np.asarray(
                cluster.get("indices", []), dtype=np.int64
            ).reshape(-1)
            record = self._build_target_confirm_cluster_record(
                gaussians, cluster_indices
            )
            if record is None:
                continue
            record.update(
                {
                    "filtered_idx": int(cluster_idx),
                    "raw_label": int(cluster_idx),
                    "support_steps": sorted(set(support_steps)) or [int(step)],
                    "support_count": int(support_count),
                    "confidence": float(best_conf),
                    "pixel_count": 0,
                    "source": "semantic_target_fallback",
                }
            )
            finalized.append(record)

        self._trace(
            "target_confirmation_semantic_fallback",
            step=step,
            reason=reason,
            source_indices=int(indices.size),
            candidate_count=len(finalized),
            support_count=int(support_count),
            best_conf=float(best_conf),
        )
        return finalized

    def _filter_and_rank_target_confirm_candidates(
        self, clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        rejected_support: List[Dict[str, Any]] = []
        for cluster in clusters:
            reason = self._target_confirm_candidate_reject_reason(cluster, enforce_support=True)
            if reason is None:
                kept.append(cluster)
                continue

            if self._debug_enabled():
                center = self._cluster_center(cluster)
                self._trace(
                    "target_confirmation_candidate_rejected",
                    step=int((self._last_payload or {}).get("step", 0)),
                    reason=reason,
                    center=center,
                    size=int(cluster.get("size", self._cluster_indices(cluster).size) or 0),
                    support_count=int(cluster.get("support_count", 1) or 1),
                    detection_dist_m=float(cluster.get("detection_dist_m", float("inf"))),
                    xz_extent_m=float(cluster.get("xz_extent_m", 0.0) or 0.0),
                    y_extent_m=float(cluster.get("y_extent_m", 0.0) or 0.0),
                )

            if str(reason).startswith("support_too_low"):
                rejected_support.append(cluster)

        if not kept and bool(getattr(self.args, "minav_confirm_support_fallback_enabled", 1)):
            min_conf = float(getattr(self.args, "minav_confirm_support_fallback_min_confidence", 0.45))
            min_size = int(getattr(self.args, "minav_confirm_support_fallback_min_size", 50))
            fallback = [
                c for c in rejected_support
                if int(c.get("support_count", 1) or 1) == 1
                if float(c.get("confidence", 0.0) or 0.0) >= min_conf
                and int(c.get("size", 0) or 0) >= min_size
                and self._target_confirm_candidate_reject_reason(
                    c,
                    enforce_support=False,
                    enforce_detection=True,
                ) is None
            ]
            fallback.sort(
                key=lambda c: (
                    float(c.get("confidence", 0.0) or 0.0),
                    -float(c.get("detection_dist_m", 0.0) or 0.0),
                    min(int(c.get("size", 0) or 0), 12000),
                    int(c.get("pixel_count", 0) or 0),
                ),
                reverse=True,
            )
            kept.extend(fallback[:1])

        kept.sort(
            key=lambda c: (
                int(c.get("support_count", 1) or 1),
                -float(c.get("detection_dist_m", 0.0) or 0.0),
                float(c.get("confidence", 0.0) or 0.0),
                min(int(c.get("size", 0) or 0), 12000),
                int(c.get("pixel_count", 0) or 0),
            ),
            reverse=True,
        )
        return kept

    def _target_confirm_candidate_reject_reason(
        self,
        cluster: Dict[str, Any],
        enforce_support: bool = True,
        enforce_detection: bool = True,
    ) -> Optional[str]:
        if cluster is None:
            return "empty_candidate"
        if enforce_support:
            support = int(cluster.get("support_count", 1) or 1)
            if support < self.min_support_count:
                return "support_too_low:{}<{}".format(support, self.min_support_count)
        center = self._cluster_center(cluster)
        if center.size < 3 or np.any(~np.isfinite(center[:3])):
            return "invalid_center"
        if self._target_confirm_cluster_masked(cluster):
            self._trace(
                "target_confirmation_candidate_masked_by_failed_region",
                step=int((self._last_payload or {}).get("step", 0)),
                center=center[:3],
                active_mask_entries=len(self.confirmation_mask_entries),
                total_blocked_pixels=int(np.sum(self.confirmation_blocked_full_mask > 0)),
            )
            return "masked_by_failed_confirmation_region"

        det_pos = self.detected_target_position
        max_det_dist = self._get_target_confirm_max_detection_dist_m(cluster)
        if enforce_detection and det_pos is not None and max_det_dist > 0:
            det = np.asarray(det_pos, dtype=np.float32).reshape(-1)
            if det.size >= 3 and np.all(np.isfinite(det[:3])):
                dist = float(np.linalg.norm(center[[0, 2]] - det[[0, 2]]))
                cluster["detection_dist_m"] = dist
                if dist > max_det_dist:
                    return "far_from_detection:{:.2f}>{:.2f}m".format(
                        dist, max_det_dist
                    )

        size = int(cluster.get("size", self._cluster_indices(cluster).size) or 0)
        min_gaussians = int(
            getattr(self.args, "minav_confirm_min_locked_gaussians", 30) or 0
        )
        if min_gaussians > 0 and size < min_gaussians:
            return "cluster_too_small:{}<{}".format(size, min_gaussians)

        bbox_extent = cluster.get("bbox_extent")
        if bbox_extent is not None:
            extent = np.asarray(bbox_extent, dtype=np.float32).reshape(-1)
            if extent.size >= 3:
                cluster["xz_extent_m"] = float(np.linalg.norm(extent[[0, 2]]))
                cluster["y_extent_m"] = float(extent[1])

        xz_extent = float(cluster.get("xz_extent_m", 0.0) or 0.0)
        y_extent = float(cluster.get("y_extent_m", 0.0) or 0.0)
        # Disabled for debugging: do not reject target-confirm clusters by
        # horizontal or vertical extent.
        # max_xz = self._get_target_confirm_max_xz_extent_m()
        # if max_xz > 0 and xz_extent > max_xz:
        #     return "xz_extent_too_large:{:.2f}>{:.2f}m".format(xz_extent, max_xz)
        # max_y = float(getattr(self.args, "minav_confirm_max_y_extent_m", 1.40) or 0.0)
        # if max_y > 0 and y_extent > max_y:
        #     return "y_extent_too_large:{:.2f}>{:.2f}m".format(y_extent, max_y)
        return None

    def _get_category_value(self, mapping_attr: str, default_value: float) -> float:
        category = str(self.target_category or "").strip().lower()
        if not category:
            return default_value
        category_values = getattr(self, mapping_attr, None)
        if not isinstance(category_values, dict):
            return default_value
        normalized = category.replace("_", " ")
        for key in (category, normalized, normalized.replace(" ", "_")):
            if key in category_values:
                try:
                    return float(category_values[key])
                except (TypeError, ValueError):
                    return default_value
        return default_value

    def _get_target_confirm_max_detection_dist_m(
        self, cluster: Optional[Dict[str, Any]] = None
    ) -> float:
        default_limit = float(
            getattr(self.args, "minav_confirm_max_detection_dist_m", 10.0) or 0.0
        )
        category_limit = self._get_category_value(
            "target_confirm_category_detection_dist_m",
            default_limit,
        )
        if category_limit <= default_limit or cluster is None:
            return category_limit
        xz_extent = float(cluster.get("xz_extent_m", 0.0) or 0.0)
        max_xz_extent = self._get_target_confirm_max_xz_extent_m()
        if max_xz_extent > 0 and xz_extent > max_xz_extent:
            return default_limit
        return category_limit

    def _get_target_confirm_max_xz_extent_m(self) -> float:
        default_limit = float(
            getattr(self.args, "minav_confirm_max_xz_extent_m", 1.20) or 0.0
        )
        return self._get_category_value(
            "target_confirm_category_xz_extent_m",
            default_limit,
        )

    @staticmethod
    def _cluster_indices(cluster: Dict[str, Any]) -> np.ndarray:
        indices = cluster.get("global_indices", cluster.get("indices", []))
        return np.asarray(indices, dtype=np.int64).reshape(-1)

    @staticmethod
    def _cluster_center(cluster: Dict[str, Any]) -> np.ndarray:
        center = cluster.get("center_3d", cluster.get("center", []))
        return np.asarray(center, dtype=np.float32).reshape(-1)[:3]

    def _target_confirm_pass_rank(
        self, cluster: Dict[str, Any], match: Dict[str, Any]
    ) -> Tuple[int, float, float, int]:
        return (
            int(cluster.get("support_count", 1) or 1),
            -float(cluster.get("detection_dist_m", 0.0) or 0.0),
            float(match.get("score", 0.0) or 0.0),
            int(match.get("match_count", 0) or 0),
        )

    def _collect_candidate_indices(self, gaussians, payload: Dict[str, Any]) -> np.ndarray:
        indices = np.array([], dtype=np.int64)
        try:
            confirmed = (
                gaussians.get_confirmed_target_indices()
                if hasattr(gaussians, "get_confirmed_target_indices")
                else None
            )
            if confirmed is not None and len(confirmed) > 0:
                return self._to_numpy_indices(confirmed)
        except Exception:
            pass

        try:
            semantic = getattr(gaussians, "get_semantic_target", None)
            if semantic is not None and getattr(semantic, "numel", lambda: 0)() > 0:
                import torch

                threshold = float(getattr(self.args, "minav_semantic_threshold", 0.6))
                indices = (
                    torch.where(semantic > threshold)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
        except Exception:
            indices = np.array([], dtype=np.int64)

        if indices.size > 0:
            return np.unique(indices)

        mask = payload.get("semantic_target_mask_raw")
        if mask is None:
            minav_mask = payload.get("minav_sem_mask")
            if minav_mask is not None:
                try:
                    semantic_id = self._target_semantic_class_from_payload(payload)
                    mask = np.asarray(minav_mask) == int(semantic_id)
                except Exception:
                    mask = None
        latest_camera = self._latest_slam_camera()
        if mask is not None and latest_camera is not None:
            try:
                depth = payload.get("depth_m")
                projected = gaussians.project_mask_to_gaussian_indices(
                    np.asarray(mask, dtype=np.uint8),
                    latest_camera,
                    depth=depth,
                    depth_tolerance=float(
                        getattr(self.args, "minav_confirm_depth_tolerance", 0.3)
                    ),
                )
                return np.unique(np.asarray(projected, dtype=np.int64))
            except Exception as exc:
                self.logger.debug("mask projection failed: %s", exc)

        return np.array([], dtype=np.int64)

    def _select_candidate_clusters(
        self,
        gaussians,
        indices: np.ndarray,
        payload: Dict[str, Any],
        allow_single_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return []
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception:
            return []
        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            return []
        xyz = xyz_all[valid]

        labels = None
        try:
            from sklearn.cluster import DBSCAN

            labels = DBSCAN(eps=self.cluster_eps, min_samples=5).fit_predict(xyz)
        except Exception:
            if not allow_single_fallback:
                return []
            labels = np.zeros((xyz.shape[0],), dtype=np.int32)

        current_world = payload.get("agent_position")
        current_xz = None
        if current_world is not None:
            current_world = np.asarray(current_world, dtype=np.float32).reshape(-1)
            if current_world.size >= 3:
                current_xz = current_world[[0, 2]]

        candidates = []
        for label in sorted(set(labels.tolist()) - {-1}):
            mask = labels == label
            cluster_indices = valid[mask]
            if cluster_indices.size < self.cluster_min_size:
                continue
            center = xyz[mask].mean(axis=0)
            if current_xz is not None:
                dist = float(np.linalg.norm(center[[0, 2]] - current_xz))
            else:
                dist = float("inf")
            candidates.append(
                {
                    "indices": cluster_indices.astype(np.int64),
                    "center": center.astype(np.float32),
                    "size": int(cluster_indices.size),
                    "dist": dist,
                }
            )

        if (
            allow_single_fallback
            and not candidates
            and valid.size >= max(1, self.cluster_min_size)
        ):
            center = xyz.mean(axis=0)
            dist = (
                float(np.linalg.norm(center[[0, 2]] - current_xz))
                if current_xz is not None
                else float("inf")
            )
            candidates.append(
                {
                    "indices": valid.astype(np.int64),
                    "center": center.astype(np.float32),
                    "size": int(valid.size),
                    "dist": dist,
                }
            )

        if not candidates:
            return []
        candidates.sort(key=lambda c: (-c["size"], c["dist"]))
        return candidates

    def _select_candidate_cluster(
        self, gaussians, indices: np.ndarray, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        candidates = self._select_candidate_clusters(gaussians, indices, payload)
        return candidates[0] if candidates else None

    def _lock_confirmed_cluster(
        self,
        indices: np.ndarray,
        step: Optional[int] = None,
        stage: str = "target_confirmation",
    ) -> int:
        locked = 0
        if self.slam_mapper is not None and hasattr(
            self.slam_mapper, "set_confirmed_cluster_mask"
        ):
            try:
                locked = int(self.slam_mapper.set_confirmed_cluster_mask(indices))
            except Exception:
                pass
        if locked <= 0:
            gaussians, fail = self._get_gaussians_for_confirmation()
            if fail is not None:
                return 0
            if hasattr(gaussians, "set_confirmed_target_indices"):
                locked = int(gaussians.set_confirmed_target_indices(indices))
        if locked > 0:
            self._configure_target_semantic_growth_policy(step=step, stage=stage)
            self._save_confirmed_full_gaussian_snapshot(step, stage)
        return int(locked)

    def _configure_target_semantic_growth_policy(
        self,
        step: Optional[int] = None,
        stage: str = "target_confirmation",
    ) -> bool:
        category = str(self.target_category or "").strip() or None
        kwargs = {
            "soft_lock": True,
            "chair_xz_radius": float(
                getattr(self.args, "minav_target_growth_chair_xz_radius", 1.0)
            ),
            "chair_y_radius": float(
                getattr(self.args, "minav_target_growth_chair_y_radius", 1.2)
            ),
            "default_xz_radius": float(
                getattr(self.args, "minav_target_growth_default_xz_radius", 5.0)
            ),
            "default_y_radius": float(
                getattr(self.args, "minav_target_growth_default_y_radius", 3.0)
            ),
        }
        configured = False
        if self.slam_mapper is not None and hasattr(
            self.slam_mapper, "set_target_semantic_growth_policy"
        ):
            try:
                configured = bool(
                    self.slam_mapper.set_target_semantic_growth_policy(
                        target_category=category,
                        **kwargs,
                    )
                )
            except Exception:
                configured = False
        if not configured:
            try:
                gaussians, fail = self._get_gaussians_for_confirmation()
                if fail is None and hasattr(gaussians, "set_target_semantic_growth_policy"):
                    gaussians.set_target_semantic_growth_policy(
                        target_category=category,
                        **kwargs,
                    )
                    configured = True
            except Exception:
                configured = False
        self._trace(
            "target_semantic_growth_policy",
            step=step,
            stage=stage,
            configured=bool(configured),
            target_category=category,
            chair_xz_radius=kwargs["chair_xz_radius"],
            chair_y_radius=kwargs["chair_y_radius"],
            default_xz_radius=kwargs["default_xz_radius"],
            default_y_radius=kwargs["default_y_radius"],
        )
        return bool(configured)

    def _save_confirmed_full_gaussian_snapshot(
        self,
        step: Optional[int],
        stage: str,
    ) -> Optional[str]:
        if self.slam_mapper is None or not hasattr(
            self.slam_mapper, "save_full_ply_frame"
        ):
            return None
        try:
            ply_path = self.slam_mapper.save_full_ply_frame(
                env_step=step,
                stage="confirmed_{}".format(stage or "target"),
                output_subdir="3dgs_confirmed",
                highlight_confirmed=True,
            )
            if ply_path is not None:
                self._trace(
                    "confirmed_full_gaussian_ply_saved",
                    step=step,
                    stage=stage,
                    path=str(ply_path),
                )
                return str(ply_path)
        except Exception as exc:
            self._trace(
                "confirmed_full_gaussian_ply_save_failed",
                step=step,
                stage=stage,
                error=str(exc),
            )
        return None

    def _save_failed_target_confirmation_full_gaussian_snapshot(
        self,
        gaussians,
        cluster_indices: np.ndarray,
        step: Optional[int],
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if (
            not self.save_stage_visualizations
            or self.visualization_dir is None
            or gaussians is None
            or not hasattr(gaussians, "save_ply")
            or not hasattr(gaussians, "set_confirmed_target_indices")
        ):
            return None
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
            indices = np.asarray(cluster_indices, dtype=np.int64).reshape(-1)
            valid = np.unique(indices[(indices >= 0) & (indices < xyz_all.shape[0])])
            if valid.size == 0:
                return None

            out_dir = self.visualization_dir / "target_confirmation"
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_reason = self._safe_name(reason, "failed")
            slam = getattr(self.slam_mapper, "slam", None)
            frame_count = getattr(slam, "frame_count", None)
            frame_part = (
                "frame_{:06d}".format(int(frame_count))
                if frame_count is not None
                else "frame_unknown"
            )
            step_part = (
                "step_{:06d}".format(int(step))
                if step is not None
                else "step_unknown"
            )
            stem = "{}_{}_failed_{}_target_semantic_full".format(
                step_part,
                frame_part,
                safe_reason,
            )
            ply_path = out_dir / "{}.ply".format(stem)
            txt_path = out_dir / "{}.txt".format(stem)
            json_path = out_dir / "{}.json".format(stem)

            old_semantic = getattr(gaussians, "_semantic_target", None)
            old_confirmed = getattr(gaussians, "_confirmed_target_indices", None)
            old_locked = getattr(gaussians, "_confirmed_cluster_locked", False)

            if hasattr(old_semantic, "detach") and hasattr(old_semantic, "clone"):
                old_semantic = old_semantic.detach().clone()
            elif old_semantic is not None:
                old_semantic = np.array(old_semantic, copy=True)
            if hasattr(old_confirmed, "detach") and hasattr(old_confirmed, "clone"):
                old_confirmed = old_confirmed.detach().clone()
            elif old_confirmed is not None:
                old_confirmed = np.array(old_confirmed, copy=True)

            try:
                highlighted = int(gaussians.set_confirmed_target_indices(valid))
                if highlighted <= 0:
                    return None
                gaussians.save_ply(
                    str(ply_path),
                    include_color=True,
                    highlight_confirmed=True,
                    confirmed_color=(255, 0, 255),
                )
            finally:
                setattr(gaussians, "_semantic_target", old_semantic)
                setattr(gaussians, "_confirmed_target_indices", old_confirmed)
                setattr(gaussians, "_confirmed_cluster_locked", old_locked)

            xyz = xyz_all[valid]
            center = xyz.mean(axis=0).astype(np.float32)
            bbox_min = xyz.min(axis=0).astype(np.float32)
            bbox_max = xyz.max(axis=0).astype(np.float32)
            meta = {
                "step": int(step) if step is not None else None,
                "stage": "target_confirmation",
                "confirmation_result": "failed",
                "failure_reason": str(reason),
                "ply_path": str(ply_path),
                "txt_path": str(txt_path),
                "json_path": str(json_path),
                "count": int(valid.size),
                "center": center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_size": bbox_max - bbox_min,
                "target_category": self.target_category,
                "highlight_color_rgb": [255, 0, 255],
                "full_scene": True,
            }
            if metadata:
                meta.update(metadata)

            with open(txt_path, "w") as f:
                f.write("Stage: target_confirmation\n")
                f.write("Confirmation_result: failed\n")
                f.write("Failure_reason: {}\n".format(str(reason)))
                f.write("Step: {}\n".format(int(step) if step is not None else "unknown"))
                f.write("Target: {}\n".format(self.target_category))
                f.write("Highlighted_count: {}\n".format(int(valid.size)))
                f.write("Center: {}\n".format(center))
                f.write("Center_x: {:.6f}\n".format(float(center[0])))
                f.write("Center_y(height): {:.6f}\n".format(float(center[1])))
                f.write("Center_z: {:.6f}\n".format(float(center[2])))
                f.write("BBox_min: {}\n".format(bbox_min))
                f.write("BBox_max: {}\n".format(bbox_max))
                if metadata:
                    for key, value in sorted(metadata.items()):
                        f.write("{}: {}\n".format(key, self._json_safe(value)))

            with open(json_path, "w") as f:
                json.dump(self._json_safe(meta), f, indent=2)

            self._trace(
                "target_confirmation_failed_full_gaussian_ply_saved",
                step=step,
                reason=reason,
                path=str(ply_path),
                txt_path=str(txt_path),
                json_path=str(json_path),
                count=int(valid.size),
                center=center,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
            return str(ply_path)
        except Exception as exc:
            self._trace(
                "target_confirmation_failed_full_gaussian_ply_save_failed",
                step=step,
                reason=reason,
                error=str(exc),
            )
            return None

    def _confirmed_cluster_navigation_position(self) -> Optional[np.ndarray]:
        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            return None
        indices = self.confirmed_indices
        if indices is None or indices.size == 0:
            try:
                confirmed = gaussians.get_confirmed_target_indices()
                indices = self._to_numpy_indices(confirmed)
            except Exception:
                return self.confirmed_target_position
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
            valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
            if valid.size == 0:
                return self.confirmed_target_position
            return xyz_all[valid].mean(axis=0).astype(np.float32)
        except Exception:
            return self.confirmed_target_position

    def _latest_slam_camera(self):
        slam = getattr(self.slam_mapper, "slam", None)
        if slam is None:
            return None
        kf_indices = getattr(slam, "kf_indices", None)
        keyframes = getattr(slam, "keyframes", None)
        if not kf_indices or keyframes is None:
            return None
        return keyframes.get(kf_indices[-1])

    def _latest_slam_keyframe_id(self):
        slam = getattr(self.slam_mapper, "slam", None)
        if slam is None:
            return None
        kf_indices = getattr(slam, "kf_indices", None)
        if not kf_indices:
            return None
        try:
            return int(kf_indices[-1])
        except Exception:
            return kf_indices[-1]

    def _debug_cluster_mask_reprojection_stats(
        self,
        gaussians,
        cluster_indices: np.ndarray,
        binary_mask,
        camera,
        depth,
    ) -> Dict[str, Any]:
        if not self._debug_enabled():
            return {}
        empty = {
            "available": False,
            "visible_projected": 0,
            "inside_mask": 0,
            "hit_ratio": None,
            "depth_residual_mean": None,
            "depth_residual_p50": None,
            "depth_residual_p90": None,
        }
        if gaussians is None or camera is None or binary_mask is None:
            return empty
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
            indices = np.asarray(cluster_indices, dtype=np.int64).reshape(-1)
            valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
            if valid.size == 0:
                return empty
            xyz = xyz_all[valid]
            if hasattr(camera, "world_view_transform"):
                w2c = camera.world_view_transform.detach().cpu().numpy().T
            else:
                from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getWorld2View2

                w2c = getWorld2View2(camera.R, camera.T)
                if hasattr(w2c, "detach"):
                    w2c = w2c.detach().cpu().numpy()
            mask_np = np.asarray(binary_mask)
            if mask_np.ndim != 2:
                return empty
            h, w = mask_np.shape[:2]
            xyz_h = np.concatenate(
                [xyz.astype(np.float32), np.ones((xyz.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
            xyz_cam = (w2c @ xyz_h.T).T[:, :3]
            valid_z = xyz_cam[:, 2] > 0.1
            fx, fy = float(camera.fx), float(camera.fy)
            cx, cy = float(camera.cx), float(camera.cy)
            u = xyz_cam[:, 0] * fx / xyz_cam[:, 2] + cx
            v = xyz_cam[:, 1] * fy / xyz_cam[:, 2] + cy
            valid_proj = (
                valid_z
                & (u >= 0)
                & (u < w)
                & (v >= 0)
                & (v < h)
                & np.isfinite(u)
                & np.isfinite(v)
            )
            if not bool(valid_proj.any()):
                result = dict(empty)
                result["available"] = True
                return result
            u_int = np.clip(u[valid_proj].astype(np.int64), 0, w - 1)
            v_int = np.clip(v[valid_proj].astype(np.int64), 0, h - 1)
            hits = mask_np[v_int, u_int] > 0
            residuals = None
            if depth is not None:
                depth_np = np.asarray(depth, dtype=np.float32)
                if depth_np.ndim == 3:
                    depth_np = depth_np[..., 0]
                if depth_np.shape[:2] == (h, w):
                    observed = depth_np[v_int, u_int]
                    rendered = xyz_cam[valid_proj, 2]
                    depth_ok = np.isfinite(observed) & np.isfinite(rendered) & (observed > 0)
                    if bool(depth_ok.any()):
                        residuals = np.abs(rendered[depth_ok] - observed[depth_ok])
            return {
                "available": True,
                "camera_uid": getattr(camera, "uid", None),
                "cluster_indices": int(valid.size),
                "visible_projected": int(valid_proj.sum()),
                "inside_mask": int(hits.sum()),
                "hit_ratio": float(hits.sum()) / max(1, int(valid_proj.sum())),
                "depth_residual_mean": None
                if residuals is None
                else float(np.mean(residuals)),
                "depth_residual_p50": None
                if residuals is None
                else float(np.percentile(residuals, 50)),
                "depth_residual_p90": None
                if residuals is None
                else float(np.percentile(residuals, 90)),
            }
        except Exception as exc:
            result = dict(empty)
            result["error"] = str(exc)
            return result

    def _save_confirmed_target_semantic_gaussians(
        self,
        gaussians,
        cluster_indices: np.ndarray,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if (
            not self.save_stage_visualizations
            or self.visualization_dir is None
            or gaussians is None
        ):
            return None
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
            indices = np.asarray(cluster_indices, dtype=np.int64).reshape(-1)
            valid = np.unique(indices[(indices >= 0) & (indices < xyz_all.shape[0])])
            if valid.size == 0:
                return None

            out_dir = self.visualization_dir / "target_confirmation"
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = "step_{:06d}_confirmed_target_semantic".format(int(step))
            ply_path = out_dir / "{}.ply".format(stem)
            txt_path = out_dir / "{}.txt".format(stem)
            json_path = out_dir / "{}.json".format(stem)

            xyz = xyz_all[valid]
            center = xyz.mean(axis=0).astype(np.float32)
            bbox_min = xyz.min(axis=0).astype(np.float32)
            bbox_max = xyz.max(axis=0).astype(np.float32)
            with open(ply_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex {}\n".format(int(xyz.shape[0])))
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for point in xyz:
                    f.write(
                        "{:.6f} {:.6f} {:.6f} 255 0 0\n".format(
                            float(point[0]), float(point[1]), float(point[2])
                        )
                    )

            meta = {
                "step": int(step),
                "stage": "target_confirmation",
                "ply_path": str(ply_path),
                "txt_path": str(txt_path),
                "count": int(valid.size),
                "center": center,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "bbox_size": bbox_max - bbox_min,
                "target_category": self.target_category,
            }
            if metadata:
                meta.update(metadata)

            with open(txt_path, "w") as f:
                f.write("Stage: target_confirmation\n")
                f.write("Step: {}\n".format(int(step)))
                f.write("Target: {}\n".format(self.target_category))
                f.write("Count: {}\n".format(int(valid.size)))
                f.write("Center: {}\n".format(center))
                f.write("Center_x: {:.6f}\n".format(float(center[0])))
                f.write("Center_y(height): {:.6f}\n".format(float(center[1])))
                f.write("Center_z: {:.6f}\n".format(float(center[2])))
                f.write("BBox_min: {}\n".format(bbox_min))
                f.write("BBox_max: {}\n".format(bbox_max))
                if metadata:
                    for key, value in sorted(metadata.items()):
                        f.write("{}: {}\n".format(key, self._json_safe(value)))

            with open(json_path, "w") as f:
                json.dump(self._json_safe(meta), f, indent=2)

            self._trace(
                "target_confirmation_semantic_gaussians_saved",
                step=step,
                ply_path=str(ply_path),
                txt_path=str(txt_path),
                json_path=str(json_path),
                count=int(valid.size),
                center=center,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
            return str(ply_path)
        except Exception as exc:
            self.logger.debug(
                "Failed to save confirmed target semantic gaussians: %s", exc
            )
            return None

    def _save_debug_cluster_ply(
        self,
        name: str,
        gaussians,
        cluster_indices: np.ndarray,
        step: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if (
            not self.gaussian_projection_debug_ply
            or not self.save_stage_visualizations
            or self.visualization_dir is None
            or gaussians is None
        ):
            return None
        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
            indices = np.asarray(cluster_indices, dtype=np.int64).reshape(-1)
            valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
            if valid.size == 0:
                return None
            safe_name = self._safe_name(name, "cluster")
            out_dir = self.visualization_dir / "target_confirmation" / "cluster_debug"
            out_dir.mkdir(parents=True, exist_ok=True)
            ply_path = out_dir / "step_{:06d}_{}.ply".format(int(step), safe_name)
            xyz = xyz_all[valid]
            with open(ply_path, "w") as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex {}\n".format(int(xyz.shape[0])))
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for point in xyz:
                    f.write(
                        "{:.6f} {:.6f} {:.6f} 255 0 0\n".format(
                            float(point[0]), float(point[1]), float(point[2])
                        )
                    )
            meta = {
                "step": int(step),
                "name": safe_name,
                "ply_path": str(ply_path),
                "count": int(valid.size),
                "center": xyz.mean(axis=0).astype(np.float32),
                "bbox_min": xyz.min(axis=0).astype(np.float32),
                "bbox_max": xyz.max(axis=0).astype(np.float32),
            }
            if metadata:
                meta.update(metadata)
            with open(out_dir / "step_{:06d}_{}.json".format(int(step), safe_name), "w") as f:
                json.dump(self._json_safe(meta), f, indent=2)
            return str(ply_path)
        except Exception as exc:
            self.logger.debug("Failed to save debug cluster PLY: %s", exc)
            return None

    @staticmethod
    def _to_numpy_indices(indices) -> np.ndarray:
        if indices is None:
            return np.array([], dtype=np.int64)
        if hasattr(indices, "detach"):
            return indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
        return np.asarray(indices, dtype=np.int64).reshape(-1)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def _ensure_target_confirm_nvsm_fem(self):
        if self.target_image is None:
            return None, None, "Target image unavailable"
        if self.slam_mapper is None:
            return None, None, "SLAM mapper not available"

        results_dir = (
            str(self.visualization_dir) if self.visualization_dir is not None else None
        )
        try:
            with self._maybe_silence_full_localizer():
                if self.target_confirm_nvsm is None:
                    from ieve_minav.localizer.novel_view_renderer import (
                        NovelViewRenderer,
                    )

                    self.target_confirm_nvsm = NovelViewRenderer(
                        slam_mapper=self.slam_mapper,
                        num_views=int(
                            getattr(
                                self.args,
                                "minav_confirm_render_views",
                                getattr(self.args, "minav_full_render_views", 48),
                            )
                        ),
                        render_radius=float(
                            getattr(
                                self.args,
                                "minav_confirm_render_radius",
                                getattr(self.args, "minav_full_render_radius", 1.0),
                            )
                        ),
                        results_dir=results_dir,
                        save_visualization_results=self.save_stage_visualizations,
                    )
                else:
                    self.target_confirm_nvsm.slam_mapper = self.slam_mapper
                    self.target_confirm_nvsm.results_dir = results_dir
                    self.target_confirm_nvsm.save_visualization_results = (
                        self.save_stage_visualizations
                    )

                fem_key = (
                    id(self.target_image),
                    tuple(self.target_image.shape),
                    str(self.target_category),
                    float(self.confirm_match_threshold),
                    results_dir,
                )
                if (
                    self.target_confirm_fem is None
                    or self.target_confirm_fem_key != fem_key
                ):
                    from ieve_minav.localizer.feature_matcher import FeatureMatcher

                    self.target_confirm_fem = FeatureMatcher(
                        target_image=self.target_image,
                        match_threshold=self.confirm_match_threshold,
                        results_dir=results_dir,
                        segmenter=None,
                        target_category=self.target_category,
                        save_visualization_results=self.save_stage_visualizations,
                        default_feature_mode="uniform",
                    )
                    self.target_confirm_fem_key = fem_key
                else:
                    self.target_confirm_fem.match_threshold = (
                        self.confirm_match_threshold
                    )
                    self.target_confirm_fem.results_dir = (
                        Path(results_dir) if results_dir else None
                    )
                    self.target_confirm_fem.save_visualization_results = (
                        self.save_stage_visualizations
                    )
        except Exception as exc:
            self.target_confirm_error = str(exc)
            self.logger.warning("TargetConfirm NVSM/FEM unavailable: %s", exc)
            return None, None, str(exc)

        self.target_confirm_error = None
        return self.target_confirm_nvsm, self.target_confirm_fem, None

    def _match_candidate_with_nvsm_views(
        self,
        center: np.ndarray,
        cluster_indices: np.ndarray,
        stage: Optional[str] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nvsm, fem, fail = self._ensure_target_confirm_nvsm_fem()
        if fail is not None:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "source": "none",
                "message": fail,
            }

        center = np.asarray(center, dtype=np.float32).reshape(-1)
        if center.size < 3:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "source": "none",
                "message": "Invalid candidate center",
            }

        observation_positions: List[Tuple[float, float]] = []
        for frame in list(self.frame_window):
            pos = frame.get("agent_position") if isinstance(frame, dict) else None
            if pos is None:
                continue
            pos = np.asarray(pos, dtype=np.float32).reshape(-1)
            if pos.size >= 3:
                observation_positions.append((float(pos[0]), float(pos[2])))

        try:
            with self._maybe_silence_full_localizer():
                render_result = nvsm.render_novel_views(
                    object_center_2d=(0, 0),
                    object_center_3d=center[:3],
                    observation_positions=observation_positions or None,
                    target_indices=cluster_indices,
                )
        except Exception as exc:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "source": "none",
                "message": "NVSM render failed: {}".format(exc),
            }

        rendered_images = list(render_result.get("rendered_images") or [])
        if not render_result.get("success") or not rendered_images:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "source": "none",
                "message": "NVSM render unavailable: {}".format(
                    render_result.get("message", "no rendered views")
                ),
            }

        try:
            with self._maybe_silence_full_localizer():
                fem_result = fem.match_rendered_views(
                    rendered_images,
                    feature_mode="uniform",
                )
        except Exception as exc:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "source": "none",
                "message": "FEM/LightGlue match failed: {}".format(exc),
            }

        if not fem_result:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "source": "none",
                "message": "FEM/LightGlue returned no result",
            }

        scores = [float(score) for score in fem_result.get("all_scores", [])]
        match_counts = [
            int(count) for count in fem_result.get("all_match_counts", [])
        ]
        best_index = int(fem_result.get("best_index", -1))
        if best_index < 0 or best_index >= len(rendered_images):
            best_index = int(np.argmax(scores)) if scores else 0

        best_score = float(
            fem_result.get(
                "best_score",
                scores[best_index] if best_index < len(scores) else 0.0,
            )
        )
        best_view_match_count = int(
            fem_result.get(
                "best_view_match_count",
                match_counts[best_index] if best_index < len(match_counts) else 0,
            )
        )
        best_image = fem_result.get("best_image")
        best_view_data = (
            rendered_images[best_index] if 0 <= best_index < len(rendered_images) else {}
        )
        if best_image is None and isinstance(best_view_data, dict):
            best_image = best_view_data.get("image")

        source = "nvsm_view_{:03d}".format(best_index)
        if isinstance(best_view_data, dict):
            scale_index = best_view_data.get("scale_index")
            local_index = best_view_data.get("local_index")
            if scale_index is not None and local_index is not None:
                source = "nvsm_s{}_v{}".format(int(scale_index), int(local_index))

        match = {
            "available": True,
            "score": best_score,
            "match_count": best_view_match_count,
            "source": source,
            "message": "",
            "best_index": best_index,
            "rendered_view_count": len(rendered_images),
            "all_scores": scores,
            "all_match_counts": match_counts,
            "feature_mode": fem_result.get("feature_mode", "uniform"),
            "render_message": render_result.get("message", ""),
            "best_match_cache": fem_result.get("best_match_cache"),
            "best_pose": best_view_data.get("pose")
            if isinstance(best_view_data, dict)
            else None,
            "radius": best_view_data.get("radius") if isinstance(best_view_data, dict) else None,
            "scale_index": best_view_data.get("scale_index") if isinstance(best_view_data, dict) else None,
            "local_index": best_view_data.get("local_index") if isinstance(best_view_data, dict) else None,
            "radius_debug": render_result.get("radius_debug", {}),
        }
        self._last_match_visual = {
            "match": dict(match),
            "image": self._as_uint8_rgb(best_image),
        }

        if stage is not None and best_image is not None:
            meta = dict(metadata or {})
            meta.update(
                {
                    "target_cluster_center": center[:3],
                    "candidate_cluster_size": int(
                        np.asarray(cluster_indices).reshape(-1).size
                    ),
                    "threshold": self.confirm_match_threshold,
                    "best_view_index": best_index,
                    "rendered_view_count": len(rendered_images),
                    "all_view_scores": scores,
                    "all_view_match_counts": match_counts,
                    "feature_mode": fem_result.get("feature_mode", "uniform"),
                    "render_message": render_result.get("message", ""),
                    "radius": best_view_data.get("radius")
                    if isinstance(best_view_data, dict)
                    else None,
                    "scale_index": best_view_data.get("scale_index")
                    if isinstance(best_view_data, dict)
                    else None,
                    "local_index": best_view_data.get("local_index")
                    if isinstance(best_view_data, dict)
                    else None,
                    "scale_info": render_result.get("scale_info", []),
                    "radii_used": render_result.get("radii_used", []),
                    "radius_debug": render_result.get("radius_debug", {}),
                    "method": "nvsm_multiview_fem_uniform",
                }
            )
            saved_path = self._save_stage_best_view(
                stage, int(step), match, best_image, metadata=meta
            )
            if saved_path is not None:
                match["visualization_path"] = saved_path
            match_overlay_path = fem.save_best_match(
                best_image,
                best_index,
                score=best_score,
                match_count=best_view_match_count,
                source="{}_{}".format(stage, source),
                step=int(step),
                metadata=meta,
                match_cache=fem_result.get("best_match_cache"),
                save_subdir=stage,
            )
            if match_overlay_path is not None:
                match["nvsm_visualization_path"] = match_overlay_path
        return match

    def _ensure_matcher(self) -> bool:
        if self._extractor is not None and self._matcher is not None:
            return True
        if self._matcher_error is not None:
            return False
        try:
            import torch
            from lightglue import DISK, LightGlue

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._extractor = DISK(max_num_keypoints=2048).eval().to(device)
            self._matcher = LightGlue(features="disk").eval().to(device)
            self._matcher_device = device
            return True
        except Exception as exc:
            self._matcher_error = str(exc)
            self._extractor = None
            self._matcher = None
            return False

    def _match_current_to_target(self, current_rgb: np.ndarray) -> Dict[str, Any]:
        if not self._ensure_matcher():
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "message": "LightGlue unavailable: {}".format(
                    self._matcher_error or "unknown"
                ),
            }
        if self.target_image is None:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "message": "Target image unavailable",
            }
        try:
            import torch
            from lightglue.utils import match_pair, numpy_image_to_torch

            with torch.no_grad():
                ob = numpy_image_to_torch(current_rgb).to(self._matcher_device)
                gi = numpy_image_to_torch(self.target_image).to(self._matcher_device)
                _, _, matches01 = match_pair(self._extractor, self._matcher, ob, gi)
                matches = matches01.get("matches")
                if matches is None:
                    count = 0
                else:
                    valid = matches[..., 0] < 2048
                    count = int(valid.sum().item())
            return {
                "available": True,
                "score": float(count),
                "match_count": int(count),
                "message": "",
            }
        except Exception as exc:
            return {
                "available": False,
                "score": 0.0,
                "match_count": 0,
                "message": "LightGlue match failed: {}".format(exc),
            }

    def _match_with_evidence_views(
        self,
        current_rgb: np.ndarray,
        stage: Optional[str] = None,
        step: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        best = None
        best_image = None
        self._last_match_visual = None
        evidence_views = [("current_rgb", current_rgb)]
        evidence_views.extend(self._render_recent_slam_views())

        unavailable_messages = []
        for source, image in evidence_views:
            image = self._as_uint8_rgb(image)
            if image is None:
                continue
            match = self._match_current_to_target(image)
            if not match.get("available"):
                unavailable_messages.append(match.get("message", "unavailable"))
                continue
            match["source"] = source
            if best is None or float(match.get("score", 0.0)) > float(best.get("score", 0.0)):
                best = match
                best_image = image

        if best is not None:
            self._last_match_visual = {
                "match": dict(best),
                "image": self._as_uint8_rgb(best_image),
            }
            if stage is not None:
                saved_path = self._save_stage_best_view(
                    stage, int(step), best, best_image, metadata=metadata
                )
                if saved_path is not None:
                    best["visualization_path"] = saved_path
            return best
        return {
            "available": False,
            "score": 0.0,
            "match_count": 0,
            "source": "none",
            "message": "; ".join(unavailable_messages) or "No evidence views available",
        }

    def _render_recent_slam_views(self) -> List[Tuple[str, np.ndarray]]:
        views: List[Tuple[str, np.ndarray]] = []
        slam = getattr(self.slam_mapper, "slam", None)
        if slam is None or not hasattr(slam, "render_frame"):
            return views
        kf_indices = list(getattr(slam, "kf_indices", []) or [])
        max_views = int(getattr(self.args, "minav_nvsm_recent_views", 3))
        if max_views <= 0:
            return views
        for kf_idx in kf_indices[-max_views:]:
            try:
                render_result = slam.render_frame(kf_idx)
            except Exception as exc:
                self.logger.debug("3DGS evidence render failed for kf %s: %s", kf_idx, exc)
                continue
            image = None
            if isinstance(render_result, dict):
                for key in ("rendered", "image", "rgb", "render"):
                    if key in render_result:
                        image = render_result[key]
                        break
            else:
                image = render_result
            image = self._as_uint8_rgb(image)
            if image is not None:
                views.append(("nvsm_kf_{}".format(kf_idx), image))
        return views

    # ------------------------------------------------------------------
    # Blacklist
    # ------------------------------------------------------------------
    def _confirmation_failure_anchor(self, result: Dict[str, Any]) -> Optional[np.ndarray]:
        center = result.get("target_cluster_center") if result else None
        if center is None:
            center = self.detected_target_position
        if center is None:
            return None
        try:
            pos = np.asarray(center, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return None
        return pos[:3].copy()

    def _confirmation_failure_counts_for_blacklist(
        self, result: Dict[str, Any]
    ) -> bool:
        if not result or result.get("confirmed") or result.get("system_not_ready"):
            return False
        reason = str(result.get("fail_reason", "") or "")
        if not reason or reason == "No detected target position":
            return False
        return self._confirmation_failure_anchor(result) is not None

    def _confirmation_failure_indices(self, result: Dict[str, Any]) -> np.ndarray:
        if not result:
            return np.array([], dtype=np.int64)
        indices = result.get("best_cluster_global_indices")
        if indices is None:
            return np.array([], dtype=np.int64)
        try:
            return np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
        except Exception:
            return np.array([], dtype=np.int64)

    def _record_confirmation_failure_for_blacklist(
        self, result: Dict[str, Any], step: int
    ) -> bool:
        if not self._confirmation_failure_counts_for_blacklist(result):
            return False
        anchor = self._confirmation_failure_anchor(result)
        if anchor is None:
            return False

        self._expire_confirmation_failure_cache(step)
        matched = None
        for entry in self.confirmation_failure_cache:
            if (
                np.linalg.norm(entry["position"][[0, 2]] - anchor[[0, 2]])
                <= self.blacklist_radius
            ):
                matched = entry
                break

        if matched is None:
            matched = {
                "position": anchor.copy(),
                "count": 0,
                "first_step": int(step),
                "last_step": int(step),
                "best_score": 0.0,
                "reason": "",
            }
            self.confirmation_failure_cache.append(matched)

        matched["count"] = int(matched.get("count", 0)) + 1
        matched["last_step"] = int(step)
        matched["reason"] = str(result.get("fail_reason", "") or "")
        matched["best_score"] = max(
            float(matched.get("best_score", 0.0) or 0.0),
            float(result.get("best_score", 0.0) or 0.0),
        )

        self._trace(
            "target_confirmation_failure_count",
            step=step,
            position=matched["position"],
            count=matched["count"],
            threshold=self.confirmation_failure_blacklist_threshold,
            radius=self.blacklist_radius,
            reason=matched["reason"],
            best_score=matched["best_score"],
        )

        if matched["count"] < self.confirmation_failure_blacklist_threshold:
            return False

        reason = "confirmation_failed_repeated_{}x: {}".format(
            matched["count"], matched["reason"]
        )
        self._add_blacklist(matched["position"], step, reason)
        self._mask_failed_confirmation_gaussian_region(result, step, reason)
        self.confirmation_failure_cache = [
            entry
            for entry in self.confirmation_failure_cache
            if (
                np.linalg.norm(entry["position"][[0, 2]] - anchor[[0, 2]])
                > self.blacklist_radius
            )
        ]
        self._trace(
            "target_confirmation_failure_blacklisted",
            step=step,
            position=anchor,
            count=matched["count"],
            threshold=self.confirmation_failure_blacklist_threshold,
            radius=self.blacklist_radius,
            cooldown=self.blacklist_cooldown,
            reason=reason,
        )
        return True

    def _expire_confirmation_failure_cache(self, step: int) -> None:
        self.confirmation_failure_cache = [
            entry
            for entry in self.confirmation_failure_cache
            if int(step) - int(entry.get("last_step", 0)) <= self.blacklist_cooldown
        ]

    def _mask_failed_confirmation_gaussian_region(
        self, result: Dict[str, Any], step: int, reason: str
    ) -> bool:
        indices = self._confirmation_failure_indices(result)
        if indices.size == 0:
            self._trace(
                "target_confirmation_gaussian_region_mask_skipped",
                step=step,
                reason="missing_failed_cluster_indices",
                failure_reason=reason,
            )
            return False

        gaussians, fail = self._get_gaussians_for_confirmation()
        if fail is not None:
            self._trace(
                "target_confirmation_gaussian_region_mask_skipped",
                step=step,
                reason=fail,
                failure_reason=reason,
                failed_cluster_indices=int(indices.size),
            )
            return False

        try:
            xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        except Exception as exc:
            self._trace(
                "target_confirmation_gaussian_region_mask_skipped",
                step=step,
                reason="read_gaussian_xyz_failed",
                error=str(exc),
                failure_reason=reason,
                failed_cluster_indices=int(indices.size),
            )
            return False

        valid = indices[(indices >= 0) & (indices < xyz_all.shape[0])]
        if valid.size == 0:
            self._trace(
                "target_confirmation_gaussian_region_mask_skipped",
                step=step,
                reason="failed_cluster_indices_out_of_range",
                failure_reason=reason,
                failed_cluster_indices=int(indices.size),
            )
            return False

        payload = self._last_payload or {}
        mask = self._project_world_points_to_full_mask(xyz_all[valid], payload)
        projected_pixels = int(np.sum(mask > 0))
        if projected_pixels <= 0:
            self._trace(
                "target_confirmation_gaussian_region_mask_skipped",
                step=step,
                reason="projection_empty",
                failure_reason=reason,
                failed_cluster_indices=int(valid.size),
            )
            return False

        dilated = self._dilate_full_mask(
            mask, self.confirmation_failure_mask_dilation_radius
        )
        entry = {
            "mask": dilated.astype(np.uint8),
            "created_step": int(step),
            "reason": reason,
            "center": self._confirmation_failure_anchor(result),
            "indices_count": int(valid.size),
            "projected_pixels": projected_pixels,
            "masked_pixels": int(np.sum(dilated > 0)),
        }
        self.confirmation_mask_entries.append(entry)
        self._rebuild_confirmation_blocked_full_mask()
        self._trace(
            "target_confirmation_gaussian_region_masked",
            step=step,
            reason=reason,
            center=entry["center"],
            failed_cluster_indices=entry["indices_count"],
            projected_pixels=entry["projected_pixels"],
            masked_pixels=entry["masked_pixels"],
            dilation_radius=self.confirmation_failure_mask_dilation_radius,
            active_mask_entries=len(self.confirmation_mask_entries),
            total_blocked_pixels=int(np.sum(self.confirmation_blocked_full_mask > 0)),
            cooldown=self.blacklist_cooldown,
        )
        return True

    def _project_world_points_to_full_mask(
        self, points_world: np.ndarray, context: Dict[str, Any]
    ) -> np.ndarray:
        mask = np.zeros(self.full_map_shape, dtype=np.uint8)
        pose_pred = context.get("pose_pred")
        current_world = context.get("current_world_xyz")
        if current_world is None:
            current_world = context.get("agent_position")
        if pose_pred is None or current_world is None:
            return mask
        try:
            pose = np.asarray(pose_pred, dtype=np.float32).reshape(-1)
            current = np.asarray(current_world, dtype=np.float32).reshape(-1)
            points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        except Exception:
            return mask
        if pose.size < 7 or current.size < 3 or points.size == 0:
            return mask
        if not np.all(np.isfinite(current[:3])):
            return mask
        map_resolution_cm = float(context.get("map_resolution_cm", 5.0) or 5.0)
        if map_resolution_cm <= 0:
            return mask

        start_x, start_y = float(pose[0]), float(pose[1])
        h, w = self.full_map_shape
        finite = np.all(np.isfinite(points[:, :3]), axis=1)
        points = points[finite]
        if points.size == 0:
            return mask
        ieve_x = start_x + (float(current[2]) - points[:, 2])
        ieve_y = start_y + (float(current[0]) - points[:, 0])
        rows = np.rint(ieve_y * 100.0 / map_resolution_cm).astype(np.int64)
        cols = np.rint(ieve_x * 100.0 / map_resolution_cm).astype(np.int64)
        valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        if np.any(valid):
            mask[rows[valid], cols[valid]] = 1
        return mask

    @staticmethod
    def _dilate_full_mask(mask: np.ndarray, radius: int) -> np.ndarray:
        mask_np = (np.asarray(mask) > 0).astype(np.uint8)
        radius = int(max(0, radius))
        if radius <= 0 or not np.any(mask_np):
            return mask_np
        try:
            import cv2

            k = radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            return (cv2.dilate(mask_np, kernel, iterations=1) > 0).astype(np.uint8)
        except Exception:
            rows, cols = np.where(mask_np > 0)
            out = mask_np.copy()
            h, w = out.shape
            for row, col in zip(rows, cols):
                r0, r1 = max(0, row - radius), min(h, row + radius + 1)
                c0, c1 = max(0, col - radius), min(w, col + radius + 1)
                out[r0:r1, c0:c1] = 1
            return out

    def _rebuild_confirmation_blocked_full_mask(self) -> None:
        blocked = np.zeros(self.full_map_shape, dtype=np.uint8)
        for entry in self.confirmation_mask_entries:
            mask = entry.get("mask")
            if mask is None:
                continue
            mask_np = np.asarray(mask, dtype=np.uint8)
            if mask_np.shape == blocked.shape:
                blocked = np.maximum(blocked, (mask_np > 0).astype(np.uint8))
        self.confirmation_blocked_full_mask = blocked

    def _expire_confirmation_mask_entries(self, step: int) -> None:
        old_count = len(self.confirmation_mask_entries)
        self.confirmation_mask_entries = [
            entry
            for entry in self.confirmation_mask_entries
            if int(step) - int(entry.get("created_step", 0)) <= self.blacklist_cooldown
        ]
        if len(self.confirmation_mask_entries) != old_count:
            self._rebuild_confirmation_blocked_full_mask()
            self._trace(
                "target_confirmation_mask_entries_expired",
                step=step,
                expired=old_count - len(self.confirmation_mask_entries),
                remaining=len(self.confirmation_mask_entries),
                total_blocked_pixels=int(np.sum(self.confirmation_blocked_full_mask > 0)),
                cooldown=self.blacklist_cooldown,
            )

    def _world_position_in_confirmation_mask(
        self, position, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not self.confirmation_mask_entries:
            return False
        context = context or self._last_payload or {}
        point = np.asarray(position, dtype=np.float32).reshape(-1)
        if point.size < 3 or not np.all(np.isfinite(point[:3])):
            return False
        point_mask = self._project_world_points_to_full_mask(point[:3].reshape(1, 3), context)
        if not np.any(point_mask):
            return False
        return bool(np.any((point_mask > 0) & (self.confirmation_blocked_full_mask > 0)))

    def _target_confirm_cluster_masked(self, cluster: Dict[str, Any]) -> bool:
        center = self._cluster_center(cluster)
        if center.size < 3:
            return False
        return self._world_position_in_confirmation_mask(center[:3], self._last_payload or {})

    def _add_blacklist(self, position, step: int, reason: str) -> None:
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            return
        self.confirmation_failure_cache = [
            entry
            for entry in self.confirmation_failure_cache
            if (
                np.linalg.norm(entry["position"][[0, 2]] - pos[[0, 2]])
                > self.blacklist_radius
            )
        ]
        for entry in self.goal_blacklist:
            if (
                np.linalg.norm(entry["position"][[0, 2]] - pos[[0, 2]])
                <= self.blacklist_radius
            ):
                entry["created_step"] = int(step)
                entry["reason"] = reason
                return
        self.goal_blacklist.append(
            {
                "position": pos[:3].copy(),
                "created_step": int(step),
                "reason": reason,
            }
        )

    def _is_blacklisted(self, position, step: int) -> bool:
        pos = np.asarray(position, dtype=np.float32).reshape(-1)
        if pos.size < 3:
            return False
        self._expire_blacklist(step)
        self._expire_confirmation_mask_entries(step)
        if self._world_position_in_confirmation_mask(pos[:3], self._last_payload or {}):
            return True
        for entry in self.goal_blacklist:
            if (
                np.linalg.norm(entry["position"][[0, 2]] - pos[[0, 2]])
                <= self.blacklist_radius
            ):
                return True
        return False

    def _expire_blacklist(self, step: int) -> None:
        self.goal_blacklist = [
            entry
            for entry in self.goal_blacklist
            if int(step) - int(entry.get("created_step", 0)) <= self.blacklist_cooldown
        ]

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    def _new_confirm_result(self) -> Dict[str, Any]:
        return {
            "confirmed": False,
            "target_cluster_center": None,
            "best_score": 0.0,
            "best_match_count": 0,
            "best_view_source": "unknown",
            "best_view_index": -1,
            "rendered_view_count": 0,
            "all_view_scores": [],
            "all_view_match_counts": [],
            "all_cluster_scores": [],
            "candidate_count": 0,
            "rendered_candidate_count": 0,
            "best_cluster_size": 0,
            "best_cluster_global_indices": None,
            "fail_reason": "",
            "system_not_ready": False,
            "should_blacklist": False,
            "method": "strict_gaussian_cluster_nvsm_multiview_lightglue_v3",
        }

    @staticmethod
    def _confirmation_failed(
        result: Dict[str, Any],
        reason: str,
        *,
        system_not_ready: bool,
        should_blacklist: bool,
        target_cluster_center=None,
        best_score: float = 0.0,
        best_match_count: int = 0,
    ) -> Dict[str, Any]:
        result.update(
            {
                "confirmed": False,
                "target_cluster_center": target_cluster_center,
                "best_score": float(best_score),
                "best_match_count": int(best_match_count),
                "fail_reason": reason,
                "system_not_ready": bool(system_not_ready),
                "should_blacklist": bool(should_blacklist),
            }
        )
        return result

    def _confirmation_skipped_as_confirmed(
        self,
        result: Dict[str, Any],
        reason: str,
        *,
        step: int,
        target_cluster_center=None,
    ) -> Dict[str, Any]:
        failed = self._confirmation_failed(
            result,
            "skip_confirmation_disabled: {}".format(reason),
            system_not_ready=True,
            should_blacklist=False,
            target_cluster_center=target_cluster_center,
        )
        failed["method"] = "strict_system_not_ready"
        self._trace(
            "target_confirmation_skip_rejected",
            step=step,
            reason=reason,
            target_cluster_center=target_cluster_center,
            contract="migration_strict_confirmation_no_skip_confirmed",
        )
        return failed

    @staticmethod
    def _as_uint8_rgb(image) -> Optional[np.ndarray]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(1, 2, 0)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)
