"""Additive Multi-IIN agent wrapper around the current GS-IIN agent.

The existing agent is imported and subclassed without modification.  This
module owns the target-boundary operation and exposes it through VectorEnv's
generic ``call_at`` mechanism.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict

import numpy as np
import torch

from agents.instance_exp import Instance_Exp_Env_Agent
from multi_iin_v0.runtime.frontier_information_gain import (
    rank_frontier_candidates,
)
from multi_iin_v0.runtime.frontier_recovery import recover_reachable_frontiers


NAME_TO_INDEX = {
    "chair": 0,
    "sofa": 1,
    "plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv_monitor": 5,
}


class MultiIINEnvAgent(Instance_Exp_Env_Agent):
    """Current IEVE agent plus an explicit two-target lifecycle."""

    def _update_minav_detection_info(self, info):
        """Expose detector-only six-class masks for Gaussian label fusion."""
        info = super()._update_minav_detection_info(info)
        semantic_instances = []
        for instance in list(getattr(self, "pred_instances", []) or []):
            if not isinstance(instance, dict):
                continue
            try:
                category_id = int(instance.get("category_id", -1))
            except Exception:
                continue
            if category_id < 0 or category_id >= len(NAME_TO_INDEX):
                continue
            mask = instance.get("binary_mask")
            if mask is None:
                continue
            mask_np = np.asarray(mask, dtype=np.uint8)
            if mask_np.ndim != 2 or not bool(np.any(mask_np)):
                continue
            semantic_instances.append(
                {
                    "category_id": category_id,
                    "category_name": instance.get("category_name"),
                    "confidence": float(instance.get("confidence", 0.0) or 0.0),
                    "binary_mask": mask_np,
                    "pixel_count": int(mask_np.sum()),
                }
            )
        # This field comes exclusively from the runtime detector. Habitat's
        # semantic observation and goal object id are deliberately excluded.
        info["multi_semantic_instances"] = semantic_instances
        return info

    def get_frontier_map(self, planner_inputs):
        """Keep baseline reachability checks, then select by unknown gain."""

        enabled = bool(
            int(getattr(self.args, "multi_frontier_information_gain_enabled", 1))
        )
        self._multi_iin_frontier_exp_map = np.asarray(
            planner_inputs.get("exp_map"), dtype=np.float32
        )
        pose = planner_inputs.get("pose_pred")
        self._multi_iin_frontier_planning_window = (
            None if pose is None else [int(value) for value in pose[3:7]]
        )
        self._multi_iin_frontier_rank_debug = {}
        original_select = getattr(self.args, "frontier_select_by_fmm_cost", 1)
        if enabled:
            # The parent must retain all reachable candidates until our
            # overridden scorer has reordered them, then emit the first one.
            self.args.frontier_select_by_fmm_cost = 1
        try:
            frontier_map = super().get_frontier_map(planner_inputs)
        finally:
            self.args.frontier_select_by_fmm_cost = original_select
        if enabled and self._multi_iin_frontier_rank_debug:
            self.frontier_fmm_selection_debug["multi_iin_information_gain"] = dict(
                self._multi_iin_frontier_rank_debug
            )
            if bool(getattr(self.args, "frontier_debug_log", 1)):
                print(
                    "[MULTI_FRONTIER_INFO_GAIN] {}".format(
                        json.dumps(self._multi_iin_frontier_rank_debug, sort_keys=True)
                    )
                )
        return frontier_map

    def _score_frontier_representatives_by_fmm(
        self, representative_points, obs_map, frontier_mask, start, planning_window
    ):
        representative_points = list(representative_points)
        candidates, debug = super()._score_frontier_representatives_by_fmm(
            representative_points,
            obs_map,
            frontier_mask,
            start,
            planning_window,
        )
        if candidates:
            self._multi_frontier_recovery_empty_calls = 0
        else:
            self._multi_frontier_recovery_empty_calls = int(
                getattr(self, "_multi_frontier_recovery_empty_calls", 0)
            ) + 1
        recovery_enabled = bool(
            int(getattr(self.args, "multi_frontier_recovery_enabled", 1))
        )
        recovery_patience = max(
            1, int(getattr(self.args, "multi_frontier_recovery_patience", 3))
        )
        if recovery_enabled and not candidates and self._multi_frontier_recovery_empty_calls >= recovery_patience:
            # The baseline planner pads its traversibility map by one cell on
            # every side.  Keep the conversion local and return unpadded map
            # coordinates to the existing frontier caller.
            try:
                traversible = self._build_frontier_traversible(
                    obs_map, start, planning_window
                )
                frontier_padded = np.zeros_like(traversible, dtype=bool)
                frontier_arr = np.asarray(frontier_mask).astype(bool)
                if frontier_padded.shape == (frontier_arr.shape[0] + 2, frontier_arr.shape[1] + 2):
                    frontier_padded[1:-1, 1:-1] = frontier_arr
                    explored_arr = np.asarray(
                        getattr(self, "_multi_iin_frontier_exp_map", np.ones_like(frontier_arr))
                    ).astype(bool)
                    if explored_arr.shape == frontier_arr.shape:
                        # Safe-band filtering can erase every visible
                        # frontier cell even though the start component still
                        # borders unknown space.  Include that boundary only
                        # for recovery; normal frontier selection is unchanged.
                        frontier_padded[1:-1, 1:-1] |= ~explored_arr
                    recovered, recovery_debug = recover_reachable_frontiers(
                        traversible,
                        frontier_padded,
                        (int(start[0]) + 1, int(start[1]) + 1),
                        [(int(r) + 1, int(c) + 1) for r, c in representative_points],
                        max_projection_px=float(getattr(self.args, "multi_frontier_recovery_projection_max_px", 15.0)),
                        min_goal_dist_px=float(getattr(self.args, "frontier_min_goal_dist_px", 10.0)),
                        allow_patrol_fallback=bool(
                            int(getattr(self.args, "multi_frontier_patrol_fallback", 1))
                        ),
                    )
                    converted = []
                    for item in recovered:
                        row, col = item["point"]
                        if 1 <= row < frontier_arr.shape[0] + 1 and 1 <= col < frontier_arr.shape[1] + 1:
                            candidate = dict(item)
                            candidate["point"] = (int(row) - 1, int(col) - 1)
                            candidate["euclidean_dist"] = float(
                                np.linalg.norm(
                                    np.asarray(candidate["point"], dtype=np.float32)
                                    - np.asarray(start, dtype=np.float32)
                                )
                            )
                            converted.append(candidate)
                    if converted:
                        candidates = converted
                    debug["multi_frontier_recovery"] = recovery_debug
                    debug["multi_frontier_recovery"]["returned_unpadded"] = int(len(converted))
            except Exception as exc:
                debug["multi_frontier_recovery_error"] = str(exc)
        if not bool(
            int(getattr(self.args, "multi_frontier_information_gain_enabled", 1))
        ) or not candidates:
            return candidates, debug

        explored = getattr(self, "_multi_iin_frontier_exp_map", None)
        if explored is None or np.asarray(explored).ndim != 2:
            return candidates, debug
        visited_local = None
        visited_full = getattr(self, "visited", None)
        window = getattr(self, "_multi_iin_frontier_planning_window", None)
        if visited_full is not None and window is not None and len(window) == 4:
            gx1, gx2, gy1, gy2 = window
            candidate_visited = np.asarray(visited_full)[gx1:gx2, gy1:gy2]
            if candidate_visited.shape == np.asarray(explored).shape:
                visited_local = candidate_visited

        radius_m = max(
            0.05,
            float(getattr(self.args, "multi_frontier_information_radius_m", 1.5)),
        )
        resolution_cm = max(1e-6, float(getattr(self.args, "map_resolution", 5.0)))
        radius_px = max(1, int(round(radius_m * 100.0 / resolution_cm)))
        ranked, rank_debug = rank_frontier_candidates(
            candidates,
            np.asarray(explored),
            visited_map=visited_local,
            radius_px=radius_px,
            fmm_cost_weight=float(
                getattr(self.args, "multi_frontier_fmm_cost_weight", 0.20)
            ),
            visited_ratio_penalty=float(
                getattr(self.args, "multi_frontier_visited_ratio_penalty", 0.25)
            ),
        )
        rank_debug["radius_m"] = radius_m
        self._multi_iin_frontier_rank_debug = rank_debug
        debug["multi_iin_information_gain"] = rank_debug
        return ranked, debug

    def reset(self):
        obs, info = super().reset()
        self._multi_frontier_recovery_empty_calls = 0
        episode = self._env.current_episode
        spec = getattr(episode, "_multi_iin_spec", None)
        target_2 = getattr(episode, "_multi_iin_target_2_episode", None)
        if spec is None or target_2 is None:
            raise RuntimeError("Multi-IIN metadata was not attached to the episode")

        self._multi_iin_spec = spec
        self._multi_iin_target_2_episode = target_2
        self._multi_iin_target_index = 0
        sim_state = self._env.sim.get_agent_state(0)
        self._multi_iin_initial_position = np.asarray(
            sim_state.position, dtype=np.float32
        ).copy()
        self._multi_iin_initial_rotation = sim_state.rotation
        self._attach_multi_info(info)
        self.info = info
        return obs, info

    @staticmethod
    def _image_sha(image) -> str:
        if image is None:
            return ""
        return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()

    @staticmethod
    def _count_nonzero(value) -> int:
        if value is None:
            return 0
        return int(np.count_nonzero(np.asarray(value)))

    def multi_state_audit(self) -> Dict[str, Any]:
        sim_state = self._env.sim.get_agent_state(0)
        return {
            "multi_episode_id": self._multi_iin_spec["episode_id"],
            "target_index": int(self._multi_iin_target_index),
            "source_episode_id": str(self._env.current_episode.episode_id),
            "goal_object_id": str(self._env.current_episode.goal_object_id),
            "goal_name": self.goal_name,
            "policy_goal_cat_id": int(self.gt_goal_idx),
            "target_image_sha256": self._image_sha(self.instance_imagegoal),
            "sim_position": [float(v) for v in sim_state.position],
            "collision_cells": self._count_nonzero(self.collision_map),
            "visited_cells": self._count_nonzero(self.visited),
            "visited_vis_cells": self._count_nonzero(self.visited_vis),
            "frontier_visited_goal_count": len(self.frontier_visited_goals or []),
            "frontier_representative_count": len(
                self.frontier_representative_points or []
            ),
            "trajectory_state_count": len(self.trajectory_states or []),
            "timestep": int(self.timestep),
        }

    def _attach_multi_info(self, info: Dict[str, Any]) -> None:
        target = self._multi_iin_spec["goals"][self._multi_iin_target_index]
        info["multi_episode_id"] = self._multi_iin_spec["episode_id"]
        info["multi_target_index"] = int(self._multi_iin_target_index)
        info["multi_num_targets"] = 2
        info["multi_source_episode_id"] = str(target["source_episode_id"])
        info["multi_variant"] = str(getattr(self.args, "multi_variant", ""))

    def _clear_agent_navigation_memory(self) -> None:
        args = self.args
        map_shape = (
            args.map_size_cm // args.map_resolution,
            args.map_size_cm // args.map_resolution,
        )
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [
            args.map_size_cm / 100.0 / 2.0,
            args.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self._minav_debug_prev_map_pred = None
        self._minav_debug_prev_exp_pred = None
        self.been_stuck = False
        self.stuck_goal = None
        self.stuck_recovery_queue = []
        self.stuck_same_stg_count = 0
        self.stuck_last_stg = None
        self.stuck_last_position = None
        self.stuck_recovery_turn_sign = 1.0
        self._minav_stg_prev_record = None
        self._minav_stg_angle_stats = {}
        self._minav_wait_debug_counter = 0
        self.frontier_vis = None
        self.frontier_active_goal = None
        self.frontier_representative_points = []
        self.frontier_cluster_debug = []
        self.frontier_visited_goals = []
        self.frontier_reachability_debug = {}
        self.frontier_fmm_candidates = []
        self.frontier_fmm_selection_debug = {}
        center_cell = int(round((args.map_size_cm / 2.0) / args.map_resolution))
        self.frontier_initial_block_cell = (center_cell, center_cell)
        self._last_fmm_replan = False
        self._last_frontier_goal_reached = False
        self._last_frontier_goal_blocked = False

    def _clear_target_specific_agent_state(self) -> None:
        map_shape = (
            self.args.map_size_cm // self.args.map_resolution,
            self.args.map_size_cm // self.args.map_resolution,
        )
        self.global_goal = None
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
        self.pred_box = []
        self.pred_instances = []
        self.stuck_recovery_queue = []
        self.stuck_same_stg_count = 0
        self.stuck_last_stg = None
        self.stuck_last_position = None
        self._target_instance_debug_frame_idx = 0

    def _switch_habitat_target(self, preserve_pose: bool) -> Dict[str, Any]:
        env = self._env
        target_episode = self._multi_iin_target_2_episode
        current = env.current_episode

        if not preserve_pose:
            env.sim.set_agent_state(
                self._multi_iin_initial_position,
                self._multi_iin_initial_rotation,
            )

        current.episode_id = str(target_episode.episode_id)
        current.goal_object_id = str(target_episode.goal_object_id)
        current.goal_image_id = int(target_episode.goal_image_id)
        current.object_category = target_episode.object_category
        current.goals = target_episode.goals
        current._shortest_path_cache = None

        for sensor in env.task.sensor_suite.sensors.values():
            if hasattr(sensor, "_current_episode_id"):
                sensor._current_episode_id = None
                sensor._current_image_goal = None

        simulator_observations = env.sim.get_observations_at()
        if simulator_observations is None:
            raise RuntimeError("Habitat returned no observations at target switch")
        observations = dict(simulator_observations)
        observations.update(
            env.task.sensor_suite.get_observations(
                observations=simulator_observations,
                episode=current,
                task=env.task,
            )
        )

        for action_instance in env.task.actions.values():
            action_instance.reset(episode=current, task=env.task)
        env.task._is_episode_active = True
        env._episode_start_time = time.time()
        env._elapsed_steps = 0
        env._episode_over = False
        env._episode_force_changed = False
        # The next explicit Env.reset() happens only after target 2 and must
        # advance to the next manifest episode.  This flag does not itself
        # reset or move the simulator now.
        env._episode_from_iter_on_reset = True
        env.task.measurements.reset_measures(
            episode=current,
            task=env.task,
            observations=observations,
        )
        return observations

    def _update_wrapper_for_target_2(self, observations: Dict[str, Any]) -> None:
        current = self._env.current_episode
        category = str(current.object_category)
        self.goal_name = category
        self.episode_goal_idx = NAME_TO_INDEX[category]
        self.gt_goal_idx = self.episode_goal_idx
        self.goal_object_id = int(current.goal_object_id)
        self.habitat_episode_id = str(current.episode_id)
        self.habitat_goal_object_id = int(current.goal_object_id)
        self.habitat_episode_dataset_index = None

        sim_state = self._env.sim.get_agent_state(0)
        self.habitat_episode_start_position = [float(v) for v in sim_state.position]
        rotation = sim_state.rotation
        self.habitat_episode_start_rotation = [
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        ]

        metrics = self._env.get_metrics()
        self.episode_geo_distance = float(metrics["distance_to_goal"])
        current_position = np.asarray(sim_state.position, dtype=np.float32)
        view_points = [
            np.asarray(view.agent_state.position, dtype=np.float32)
            for goal in current.goals
            for view in goal.view_points
        ]
        self.episode_euc_distance = min(
            float(np.linalg.norm(current_position - point)) for point in view_points
        )

        self.global_step = 0
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        self.trajectory_states = []
        self.last_sim_location = self.get_sim_location()
        self.start_height = float(sim_state.position[1])
        self.agent_height = self.args.camera_height
        self.start_position = sim_state.position
        self.start_rotation = sim_state.rotation
        self.transform_matrix = self.get_transformation_matrix()

        self.instance_imagegoal = observations["instance_imagegoal"]
        goal_cat_source = getattr(self.args, "goal_cat_source", "oracle")
        grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        try:
            if goal_cat_source == "detector":
                detected = self.get_goal_id()
                if detected is not None:
                    self.gt_goal_idx = int(detected)
            elif goal_cat_source == "classifier":
                self.gt_goal_idx = self.get_classifier_goal_id()
            elif goal_cat_source == "oracle":
                self.gt_goal_idx = self.episode_goal_idx
            else:
                raise ValueError("Unknown goal_cat_source: {}".format(goal_cat_source))
        finally:
            torch.set_grad_enabled(grad_enabled)

        if "semantic" in observations:
            semantic = np.asarray(observations["semantic"])
            self.semantic_obs = np.where(semantic == self.goal_object_id, 1, 0)
            self.sign = bool(np.any(self.semantic_obs > 0))

        rgb = observations["rgb"].astype(np.uint8)
        depth = observations["depth"]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.raw_obs = state[:3].transpose(1, 2, 0)
        self.raw_depth = state[3:4]
        self.pred_box = []
        self.pred_instances = []
        processed = self._preprocess_obs(state)
        self.obs = processed
        self.obs_shape = processed.shape

        self.info = {
            "time": 0,
            "sensor_pose": [0.0, 0.0, 0.0],
            "goal_cat_id": int(self.gt_goal_idx),
            "goal_cat_source": goal_cat_source,
            "episode_goal_cat_id": int(self.episode_goal_idx),
            "goal_name": self.goal_name,
            "agent_height": self.agent_height,
            "instance_imagegoal": self.instance_imagegoal,
            "distance_to_goal": float(metrics["distance_to_goal"]),
            "spl": float(metrics["spl"]),
            "success": float(metrics["success"]),
            "soft_spl": float(metrics["soft_spl"]),
            "g_reward": 0.0,
        }
        self.info.update(self._get_3dgs_frame_info(observations))
        self.info = self._update_minav_detection_info(self.info)

    def switch_multi_goal(self, variant: str):
        """Move from target 1 to target 2 and return processed observation."""

        normalized = str(variant).strip().lower().replace("_", "-")
        aliases = {
            "independent-reset": "Independent-Reset",
            "continuous-clear": "Continuous-Clear",
            "continuous-shared": "Continuous-Shared",
        }
        if normalized not in aliases:
            raise ValueError("unknown Multi-IIN variant: {}".format(variant))
        if self._multi_iin_target_index != 0:
            raise RuntimeError("switch_multi_goal is only valid after target 1")

        before = self.multi_state_audit()
        canonical = aliases[normalized]
        observations = self._switch_habitat_target(
            preserve_pose=canonical != "Independent-Reset"
        )
        self._multi_iin_target_index = 1
        self._update_wrapper_for_target_2(observations)
        self._clear_target_specific_agent_state()
        if canonical != "Continuous-Shared":
            self._clear_agent_navigation_memory()
        self._attach_multi_info(self.info)
        self.info["multi_variant"] = canonical
        after = self.multi_state_audit()
        initial_position = [float(v) for v in self._multi_iin_initial_position]
        audit = {
            "variant": canonical,
            "agent_before": before,
            "agent_after": after,
            "initial_position": initial_position,
            "boundary_to_after_position_delta_m": float(
                np.linalg.norm(
                    np.asarray(before["sim_position"], dtype=np.float32)
                    - np.asarray(after["sim_position"], dtype=np.float32)
                )
            ),
            "initial_to_after_position_delta_m": float(
                np.linalg.norm(
                    np.asarray(initial_position, dtype=np.float32)
                    - np.asarray(after["sim_position"], dtype=np.float32)
                )
            ),
        }
        return self.obs, self.info, audit
