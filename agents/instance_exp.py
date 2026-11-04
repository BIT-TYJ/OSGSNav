import warnings
warnings.filterwarnings('ignore')
import logging
import json
import math
import os
import cv2
import numpy as np
import skimage.measure
import skimage.morphology
from PIL import Image
import math
import torch
from torchvision import transforms
import torch.nn.functional as F
from scipy import ndimage
from envs.utils.fmm_planner import FMMPlanner
from envs.utils.ieve_reachability import compute_temp_goal_distance_ieve
from envs.habitat.imagegoal_env import ObjectGoal_Env
from agents.utils.semantic_prediction import SemanticPredMaskRCNN
from constants import color_palette
import envs.utils.pose as pu
import agents.utils.visualization as vu
from collections import Counter
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
import numpy.ma as ma
import matplotlib.pyplot as plt
# from projects import *
from skimage.draw import line_aa, line
from argparse import ArgumentParser
import os.path as osp



class Instance_Exp_Env_Agent(ObjectGoal_Env):
    """The Sem_Exp environment agent class. A seperate Sem_Exp_Env_Agent class
    object is used for each environment thread.

    """

    def __init__(self, args, rank, config_env, dataset):
        self.args = args
        super().__init__(args, rank, config_env, dataset)
        

        # initialize transform for RGB observations
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])

        # initialize semantic segmentation prediction model
        if args.sem_gpu_id == -1:
            args.sem_gpu_id = config_env.habitat.simulator.habitat_sim_v0.gpu_device_id

        self.sem_pred = SemanticPredMaskRCNN(args)

        # initializations for planning:
        self.selem = skimage.morphology.disk(3)

        self.obs = None
        self.obs_shape = None
        self.collision_map = None
        self.visited = None
        self.visited_vis = None
        self.col_width = None
        self.curr_loc = None
        self.last_loc = None
        self.last_action = None
        self.count_forward_actions = None
        self.instance_imagegoal = None
        self._minav_debug_prev_map_pred = None
        self._minav_debug_prev_exp_pred = None
        self.stuck_recovery_queue = []
        self.stuck_same_stg_count = 0
        self.stuck_last_stg = None
        self.stuck_last_position = None
        self.stuck_recovery_turn_sign = 1.0
        self._minav_stg_prev_record = None
        self._minav_stg_angle_stats = {}
        self._minav_wait_debug_counter = 0
        self._target_instance_debug_frame_idx = 0
        self.frontier_vis = None
        self.frontier_active_goal = None
        self.frontier_representative_points = []
        self.frontier_cluster_debug = []
        self.frontier_visited_goals = []
        self.frontier_reachability_debug = {}
        self.frontier_fmm_candidates = []
        self.frontier_fmm_selection_debug = {}
        self.frontier_initial_block_cell = None
        self._last_fmm_replan = False
        self._last_frontier_goal_reached = False
        self._last_frontier_goal_blocked = False
        # Multi-IIN-only exact FMM memoization.  The cache is used only when
        # both the final traversible grid and dilated goal grid are bitwise
        # equal; all map/goal changes therefore keep the original replan path.
        self._multi_fmm_exact_cache = None
        self._multi_fmm_cache_hits = 0
        self._multi_fmm_cache_misses = 0
        self._last_multi_fmm_cache_hit = False

        self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='disk').eval().to(self.device)

        map_size = self.args.map_size_cm // self.args.map_resolution
        full_w, full_h = map_size, map_size
        self.full_w = full_w
        self.full_h = full_h
        self.local_w = int(full_w / self.args.global_downscaling)
        self.local_h = int(full_h / self.args.global_downscaling)
        self.global_goal = None
        # define a temporal goal with a living time
        self.temp_goal = None
        self.last_temp_goal = None # avoid choose one goal twice
        self.forbidden_temp_goal = []
        self.flag = 0
        self.goal_instance_whwh = None
        # define untraversible area of the goal: 0 means area can be goals, 1 means cannot be
        self.goal_map_mask = np.ones((full_w, full_h))
        self.pred_box = []
        self.pred_instances = []
        torch.set_grad_enabled(False)

        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None

    def reset(self):
        args = self.args

        obs, info = super().reset()
        self.raw_obs = obs[:3, :, :].transpose(1, 2, 0)
        self.raw_depth = obs[3:4, :, :]

        self.pred_box = []
        self.pred_instances = []
        obs = self._preprocess_obs(obs)
        self.obs = obs

        self.obs_shape = obs.shape

        # Episode initializations
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.collision_map = np.zeros(map_shape)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        self.col_width = 1
        self.count_forward_actions = 0
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_action = None
        self._minav_debug_prev_map_pred = None
        self._minav_debug_prev_exp_pred = None
        self.global_goal = None
        self.temp_goal = None
        self.last_temp_goal = None
        self.forbidden_temp_goal = []
        self.goal_map_mask = np.ones(map_shape)
        self.goal_instance_whwh = None
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
        self._target_instance_debug_frame_idx = 0
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
        self._multi_fmm_exact_cache = None
        self._multi_fmm_cache_hits = 0
        self._multi_fmm_cache_misses = 0
        self._last_multi_fmm_cache_hit = False

        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)

        info = self._update_minav_detection_info(info)
        self.info = info
        return obs, info

    def _update_minav_detection_info(self, info):
        if info is None:
            return info
        target_instances = []
        for inst in getattr(self, "pred_instances", []):
            try:
                if int(inst.get("category_id", -1)) != int(self.gt_goal_idx):
                    continue
            except Exception:
                continue
            inst_copy = {}
            for key, value in inst.items():
                if key == "binary_mask":
                    inst_copy[key] = np.asarray(value, dtype=np.uint8)
                elif key == "bbox":
                    inst_copy[key] = np.asarray(value, dtype=np.float32)
                else:
                    inst_copy[key] = value
            target_instances.append(inst_copy)

        semantic_target_mask = None
        if target_instances:
            best = max(
                target_instances,
                key=lambda x: (
                    float(x.get("confidence", 0.0) or 0.0),
                    int(x.get("pixel_count", 0) or 0),
                ),
            )
            semantic_target_mask = np.asarray(best["binary_mask"], dtype=np.uint8)

        info["pred_box"] = self.pred_box
        info["target_instances"] = target_instances
        info["semantic_target_mask_raw"] = semantic_target_mask
        self._save_target_instance_debug(
            info,
            all_instances=list(getattr(self, "pred_instances", []) or []),
            target_instances=target_instances,
            semantic_target_mask=semantic_target_mask,
        )
        return info

    @staticmethod
    def _non_terminal_hold_action():
        return {
            "action": ("velocity_control", "velocity_stop"),
            "action_args": {
                "angular_velocity": np.array([0]),
                "linear_velocity": np.array([0]),
                "camera_pitch_velocity": np.array([0]),
                "velocity_stop": np.array([0]),
            },
        }

    @staticmethod
    def _is_minav_localization_wait(planner_inputs):
        if not bool(planner_inputs.get("hold_action", False)):
            return False
        return (
            planner_inputs.get("goal_source") == "MINAV_LOCALIZATION_WAIT"
            or planner_inputs.get("target_mode") == "localization"
        )

    @staticmethod
    def _minav_display_mode_label(planner_inputs):
        mode = str(planner_inputs.get("target_mode", "") or "").strip().lower()
        labels = {
            "exploration": "Exploration",
            "target_confirmation": "Target Confirmation",
            "targeted_exploration": "Targeted Exploration",
            "buffer": "Buffer",
            "localization": "Localization",
            "cluster_recovery": "Cluster Recovery",
            "navigation": "Navigation",
        }
        return labels.get(mode)

    def _attach_minav_step_info(self, info, planner_inputs, frozen_hold=False):
        if info is None:
            info = {}
        info = self._update_minav_detection_info(info)
        info["minav_fmm_stop"] = bool(planner_inputs.get("_fmm_stop", False))
        info["minav_phase_goal_reached"] = bool(
            planner_inputs.get("_minav_phase_goal_reached", False)
        )
        info["minav_target_mode"] = planner_inputs.get("target_mode")
        info["minav_goal_source"] = planner_inputs.get("goal_source")
        info["minav_found_goal"] = int(bool(planner_inputs.get("found_goal", 0)))
        info["minav_navigation_goal_local_cell"] = planner_inputs.get(
            "minav_navigation_goal_local_cell"
        )
        info["minav_navigation_goal_cell_reached"] = bool(
            planner_inputs.get("_minav_nav_goal_cell_reached", False)
        )
        info["minav_collision_map_local"] = None
        try:
            pose_pred = np.asarray(
                planner_inputs.get("pose_pred", []), dtype=np.float32
            ).reshape(-1)
            if pose_pred.size >= 7 and self.collision_map is not None:
                gx1, gx2, gy1, gy2 = [int(v) for v in pose_pred[3:7]]
                collision_local = self.collision_map[gx1:gx2, gy1:gy2]
                goal_map = planner_inputs.get("goal")
                if goal_map is not None and collision_local.shape == np.shape(goal_map):
                    info["minav_collision_map_local"] = np.asarray(
                        collision_local > 0, dtype=np.uint8
                    )
        except Exception:
            info["minav_collision_map_local"] = None
        info["minav_frozen_hold"] = bool(frozen_hold)
        info["ieve_frontier_fmm_replan"] = bool(
            planner_inputs.get("_fmm_replan", False)
        )
        info["multi_fmm_exact_cache_hit"] = bool(
            planner_inputs.get("_multi_fmm_exact_cache_hit", False)
        )
        info["multi_fmm_exact_cache_hits"] = int(
            getattr(self, "_multi_fmm_cache_hits", 0)
        )
        info["multi_fmm_exact_cache_misses"] = int(
            getattr(self, "_multi_fmm_cache_misses", 0)
        )
        info["ieve_frontier_active_goal"] = planner_inputs.get(
            "_frontier_active_goal"
        )
        info["ieve_frontier_active_goal_reached"] = bool(
            planner_inputs.get("_frontier_active_goal_reached", False)
        )
        info["ieve_frontier_active_goal_blocked"] = bool(
            planner_inputs.get("_frontier_active_goal_blocked", False)
        )
        frontier_goal_count = planner_inputs.get("_frontier_goal_count")
        if frontier_goal_count is None:
            goal_map = planner_inputs.get("goal")
            frontier_goal_count = (
                0
                if goal_map is None
                else int(np.count_nonzero(np.asarray(goal_map) > 0))
            )
        info["ieve_frontier_goal_count"] = int(frontier_goal_count or 0)
        info["ieve_frontier_been_stuck"] = bool(getattr(self, "been_stuck", False))
        return info

    def _return_minav_frozen_hold(self, planner_inputs, action):
        obs = (
            np.array(self.obs, copy=True)
            if self.obs is not None
            else np.zeros(self.obs_shape)
        )
        info = dict(self.info) if self.info is not None else {}
        info["sensor_pose"] = [0., 0., 0.]
        info["time"] = self.timestep
        self._minav_wait_debug_counter += 1
        info["minav_wait_debug_counter"] = int(self._minav_wait_debug_counter)

        self.last_action = None
        info = self._attach_minav_step_info(
            info,
            planner_inputs,
            frozen_hold=True,
        )
        self.info = info
        self._log_minav_stop_debug(planner_inputs, action, False, info)
        return obs, 0., False, info

    def _non_terminal_turn_action(
        self, direction, in_place=False, turn_angle_deg=None
    ):
        direction = str(direction or "").lower()
        turn_speed = 1.0 if in_place else 0.6
        if direction in ("turn_right", "right", "sweep_right"):
            angular_velocity = np.array([-turn_speed])
        else:
            angular_velocity = np.array([turn_speed])
        # Habitat velocity-control inputs are normalized; with scale conversion
        # enabled, -1 maps to physical 0 m/s for linear velocity.
        linear_velocity = np.array([-1])
        action_args = {
            "angular_velocity": angular_velocity,
            "linear_velocity": linear_velocity,
            "camera_pitch_velocity": np.array([0]),
            "velocity_stop": np.array([0]),
        }
        if in_place:
            turn_angle = float(
                turn_angle_deg
                if turn_angle_deg is not None
                else (getattr(self.args, "turn_angle", 30.0) or 30.0)
            )
            turn_angle = max(1.0, abs(turn_angle))
            # Normalized angular velocity 1.0 maps to 5 rad/s in this Habitat
            # velocity controller. This gives one planner turn per turn_angle.
            action_args["time_step"] = float(np.deg2rad(turn_angle) / 5.0)
        return {
            "action": ("velocity_control", "velocity_stop"),
            "action_args": action_args,
        }

    @staticmethod
    def _non_terminal_velocity_action(linear_velocity=0.0, angular_velocity=0.0):
        return {
            "action": ("velocity_control", "velocity_stop"),
            "action_args": {
                "angular_velocity": np.array([angular_velocity]),
                "linear_velocity": np.array([linear_velocity]),
                "camera_pitch_velocity": np.array([0]),
                "velocity_stop": np.array([0]),
            },
        }

    @staticmethod
    def _terminal_stop_action():
        return {
            "action": ("velocity_control", "velocity_stop"),
            "action_args": {
                "angular_velocity": np.array([0]),
                "linear_velocity": np.array([0]),
                "camera_pitch_velocity": np.array([0]),
                "velocity_stop": np.array([1]),
            },
        }

    def _stuck_recovery_position(self, planner_inputs):
        position = planner_inputs.get("agent_position")
        if position is None:
            position = self.curr_loc[:2]
        try:
            arr = np.asarray(position, dtype=np.float32).reshape(-1)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return None
            return arr
        except Exception:
            return None

    def _block_stuck_stg(self, stg, planning_window, radius):
        if stg is None or self.collision_map is None:
            return 0
        gx1, gx2, gy1, gy2 = planning_window
        center_r = int(stg[0]) + int(gx1)
        center_c = int(stg[1]) + int(gy1)
        radius = max(0, int(radius))
        r1 = max(0, center_r - radius)
        r2 = min(self.collision_map.shape[0], center_r + radius + 1)
        c1 = max(0, center_c - radius)
        c2 = min(self.collision_map.shape[1], center_c + radius + 1)
        before = int(np.count_nonzero(self.collision_map[r1:r2, c1:c2]))
        self.collision_map[r1:r2, c1:c2] = 1
        after = int(np.count_nonzero(self.collision_map[r1:r2, c1:c2]))
        return max(0, after - before)

    def _make_stuck_recovery_queue(self):
        backward_steps = max(0, int(getattr(self.args, "stuck_recovery_backward_steps", 2)))
        turn_steps = max(1, int(getattr(self.args, "stuck_recovery_turn_steps", 4)))
        forward_steps = max(0, int(getattr(self.args, "stuck_recovery_forward_steps", 3)))
        turn_sign = float(getattr(self, "stuck_recovery_turn_sign", 1.0))
        self.stuck_recovery_turn_sign = -turn_sign
        queue = []
        for _ in range(backward_steps):
            queue.append(("reverse", -0.45, 0.0))
        for _ in range(turn_steps):
            queue.append(("turn", -1.0, 0.8 * turn_sign))
        for _ in range(forward_steps):
            queue.append(("forward_escape", 0.55, 0.0))
        return queue

    def _pop_stuck_recovery_action(self, planner_inputs):
        kind, linear_velocity, angular_velocity = self.stuck_recovery_queue.pop(0)
        if "_minav_planner_debug" in planner_inputs:
            planner_inputs["_minav_planner_debug"]["stuck_recovery_action"] = {
                "kind": kind,
                "linear_velocity": float(linear_velocity),
                "angular_velocity": float(angular_velocity),
                "remaining": len(self.stuck_recovery_queue),
            }
        return self._non_terminal_velocity_action(linear_velocity, angular_velocity)

    @staticmethod
    def _stuck_recovery_enabled():
        return False

    def _maybe_trigger_stuck_recovery(
        self,
        planner_inputs,
        stg,
        planning_window,
        minav_executor_mode,
    ):
        if not self._stuck_recovery_enabled():
            self.stuck_same_stg_count = 0
            self.stuck_last_stg = None
            self.stuck_last_position = None
            return False
        if minav_executor_mode:
            self.stuck_same_stg_count = 0
            return False
        if not getattr(self, "been_stuck", False) or stg is None:
            self.stuck_same_stg_count = 0
            self.stuck_last_stg = None
            self.stuck_last_position = None
            return False

        position = self._stuck_recovery_position(planner_inputs)
        stg_arr = np.asarray(stg, dtype=np.float32).reshape(-1)[:2]
        moved_small = False
        same_stg = False
        if position is not None and self.stuck_last_position is not None:
            prev = self.stuck_last_position
            dim = min(position.size, prev.size)
            moved_small = (
                float(np.linalg.norm(position[:dim] - prev[:dim]))
                < float(getattr(self.args, "stuck_recovery_move_eps", 0.03))
            )
        if self.stuck_last_stg is not None:
            same_stg = pu.get_l2_distance(
                float(stg_arr[0]),
                float(self.stuck_last_stg[0]),
                float(stg_arr[1]),
                float(self.stuck_last_stg[1]),
            ) <= float(getattr(self.args, "stuck_recovery_stg_eps", 1.0))

        if moved_small and same_stg:
            self.stuck_same_stg_count += 1
        else:
            self.stuck_same_stg_count = 1

        self.stuck_last_position = None if position is None else np.array(position, copy=True)
        self.stuck_last_stg = np.array(stg_arr, copy=True)

        patience = max(1, int(getattr(self.args, "stuck_recovery_patience", 3)))
        triggered = self.stuck_same_stg_count >= patience
        debug = {
            "enabled": True,
            "queued": len(getattr(self, "stuck_recovery_queue", [])),
            "same_stg_count": int(self.stuck_same_stg_count),
            "patience": int(patience),
            "moved_small": bool(moved_small),
            "same_stg": bool(same_stg),
            "triggered": bool(triggered),
        }

        if triggered:
            debug["blocked_cells"] = self._block_stuck_stg(
                stg,
                planning_window,
                getattr(self.args, "stuck_recovery_block_radius", 3),
            )
            self.stuck_recovery_queue = self._make_stuck_recovery_queue()
            debug["queued"] = len(self.stuck_recovery_queue)
            debug["sequence"] = [item[0] for item in self.stuck_recovery_queue]
            self.stuck_same_stg_count = 0
            self.stuck_last_position = None
            self.stuck_last_stg = None

        if "_minav_planner_debug" in planner_inputs:
            planner_inputs["_minav_planner_debug"]["stuck_recovery"] = debug
        return triggered

    def local_feature_match_lightglue(self, re_key2=False):
        with torch.set_grad_enabled(False):
            ob = numpy_image_to_torch(self.raw_obs[:, :, :3]).to(self.device)
            gi = numpy_image_to_torch(self.instance_imagegoal).to(self.device)
            try:
                feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                    )
                # indices with shape (K, 2)
                matches = matches01['matches']
                # in case that the matches collapse make a check
                b = torch.nonzero(matches[..., 0] < 2048, as_tuple=False)
                c = torch.index_select(matches[..., 0], dim=0, index=b.squeeze())
                points0 = feats0['keypoints'][c]
                if re_key2:
                    return (points0.numpy(), feats1['keypoints'][c].numpy())
                else:
                    return points0.numpy()  
            except:
                if re_key2:
                    # print(f'{self.rank}  {self.timestep}  h')
                    return (np.zeros((1, 2)), np.zeros((1, 2)))
                else:
                    # print(f'{self.rank}  {self.timestep}  h')
                    return np.zeros((1, 2))
                

    def get_mask_center(self, image_mask):
        '''
        image_mask: (N, N)
        '''
        center = ndimage.center_of_mass(image_mask)
        if np.isnan(center[0]) or np.isnan(center[1]):
            return None
        center = [int(center[0]), int(center[1])]
        return center

    def save_image(self, image, name):
        '''
        only supports save grayscale
        '''
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        fn1 = '{}/episodes/thread_{}/eps_{}/{}_{}.png'.format(
                dump_dir, self.rank, self.episode_no, name, self.timestep)
        cv2.imwrite(fn1, image*255)
    
    def count_overlap(self, ma1, index):
        '''
        count index in mask
        '''
        ma2 = np.zeros_like(ma1,dtype=int)
        ma1[ma1 != 0] = 1
        index = index.astype(np.int16)
        ma2[index[:, 1], index[:, 0]] = 1
        ma = (ma1 + ma2) // 2

        # self.save_image(ma1, "t1")
        # self.save_image(ma2, "t2")

        return ma.sum()

    def check_if_mask_big_enough(self, goal_mask, f):
        '''
        params : goal_mask: (H, W)
                f : thresh ratio, should less than 1 and greater than 0
        '''
        return goal_mask.sum() > goal_mask.shape[0] * goal_mask.shape[1] * ((f)**2)

    def compute_ins_dis_v1(self, depth, whwh, k=3):
        '''
        analyze the maxium depth points's pos
        make sure the object is within the range of 10m
        '''
        hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            bins=200,range=(0,2000))
        peak_indices = np.argsort(hist)[-k:]  # Get the indices of the top k peaks
        peak_values = hist[peak_indices] + hist[np.clip(peak_indices-1, 0, len(hist)-1)]  + \
            hist[np.clip(peak_indices+1, 0, len(hist)-1)]
        max_area_index = np.argmax(peak_values)  # Find the index of the peak with the largest area
        max_index = peak_indices[max_area_index]
        # max_index = np.argmax(hist)
        return bins[max_index]

    def compute_ins_goal_map(self, whwh, start, start_o):
        goal_mask = np.zeros_like(self.obs[3, :, :])
        goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1
        semantic_mask = (self.obs[4+self.gt_goal_idx, :, :] > 0) & (goal_mask > 0)

        depth_h, depth_w = np.where(semantic_mask > 0)
        goal_dis = self.obs[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.obs.shape[2]/2) \
        / (self.obs.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_w, self.local_h))
        goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
        goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        return goal_map

    def compute_ins_goal_map_mask(self, mask, start, start_o):

        semantic_mask = mask[:, :, 0] * 1
        selem = skimage.morphology.disk(3)
        semantic_mask = skimage.morphology.erosion(semantic_mask, selem)

        depth_h, depth_w = np.where(semantic_mask > 0)
        depth_h, depth_w = depth_h//4, depth_w//4
        goal_dis = self.obs[3, :, :][depth_h, depth_w] / self.args.map_resolution

        goal_angle = -self.args.hfov / 2 * (depth_w - self.obs.shape[2]/2) \
        / (self.obs.shape[2]/2)
        goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
            start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
        goal_map = np.zeros((self.local_w, self.local_h))
        goal[0] = np.clip(goal[0], 0, 240-1).astype(int)
        goal[1] = np.clip(goal[1], 0, 240-1).astype(int)
        goal_map[goal[0], goal[1]] = 1
        
        
        return goal_map


    def compute_object_distance(self, goal_mask, index, rgb_center):
        ma2 = np.zeros_like(goal_mask,dtype=int)
        goal_mask[goal_mask != 0] = 1
        index = index.astype(np.int16)
        ma2[index[:, 1], index[:, 0]] = 1
        maa = (goal_mask.astype(int) + ma2) // 2
        # if maa.sum() != 0:
        if False:
            masked_depth = ma.masked_array(self.obs[3, :, :], maa == 1.) / self.args.map_resolution
            goal_dis = np.clip(masked_depth, 0, 180).mean()
        else: 
            goal_dis = self.obs[3:4, :, :][0, int(rgb_center[0]), int(rgb_center[1])] / self.args.map_resolution
            # hist, bins = np.histogram(depth[whwh[1]:whwh[3], whwh[0]:whwh[2]].flatten(), \
            #     bins=100,range=(0,10))
            # max_index = np.argmax(hist)
            # return bins[max_index]
        return goal_dis


    def instance_discriminator_v2(self, planner_inputs, id_lo_whwh_speci):
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.semantic_obs

        if self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            return planner_inputs

        if np.any(goal_mask > 0):
            
            global_goal = np.zeros((self.full_w, self.full_h))
            goal_map = self.compute_ins_goal_map_mask(goal_mask, start, start_o)

            global_goal[gx1:gx2, gy1:gy2] = goal_map
            if np.any(goal_map > 0):
                planner_inputs['found_goal'] = 1
                self.global_goal = global_goal
                planner_inputs['goal'] = goal_map
            else:
                planner_inputs['found_goal'] = 0
                planner_inputs['goal'] = planner_inputs['exp_goal']
            
        else:
            planner_inputs['found_goal'] = 0
            planner_inputs['goal'] = planner_inputs['exp_goal']
        return planner_inputs



    def instance_discriminator_v1(self, planner_inputs, id_lo_whwh_speci):
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.obs[4+self.gt_goal_idx, :, :]
        if self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
        elif planner_inputs['found_goal'] == 1:

            # preprocess foundation model output
            # id_lo_whwh_speci = sorted(id_lo_whwh_speci, key=lambda s: s[1], reverse=True)
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            # index = self.local_feature_match()
            index = self.local_feature_match_lightglue()
            match_points = index.shape[0]
            

            if match_points > 100:
                planner_inputs['found_goal'] = 1
                global_goal = np.zeros((self.full_w, self.full_h))
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)

                global_goal[gx1:gx2, gy1:gy2] = goal_map
                self.global_goal = global_goal
                planner_inputs['goal'] = goal_map
            else:
                planner_inputs['found_goal'] = 0
                planner_inputs['goal'] = planner_inputs['exp_goal']
        else:
            planner_inputs['found_goal'] = 0
            planner_inputs['goal'] = planner_inputs['exp_goal']


    def instance_discriminator_auto(self, planner_inputs, id_lo_whwh_speci):
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.obs[4+self.gt_goal_idx, :, :]

        if self.instance_imagegoal is None:
            # not initialized
            return planner_inputs
        elif self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            return planner_inputs
        elif self.been_stuck:
            
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:

                navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
                goal = np.array([0, 0])
                for _ in range(100):
                    random_index = np.random.choice(len(navigable_indices))
                    goal = navigable_indices[random_index]
                    if pu.get_l2_distance(goal[0], start[0], goal[1], start[1]) > 16:
                        break

                goal = pu.threshold_poses(goal, map_pred.shape)                
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
        elif planner_inputs['found_goal'] == 1:

            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            # index = self.local_feature_match()
            index = self.local_feature_match_lightglue()
            match_points = index.shape[0]
            planner_inputs['found_goal'] = 0
            if self.temp_goal is not None:
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.obs[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.obs.shape[2]/2) \
                    / (self.obs.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                if not getattr(self.args, "quiet_log", 0):
                    print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
            else:
                if match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.full_w, self.full_h))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    with torch.no_grad():
                        dd = transforms.ToTensor()(self.raw_depth[0, :, :])
                        dd = transforms.Resize((160, 120))(dd)
                        depth_tensor = dd.unsqueeze(0).to(self.device)
                        extras_tensor = torch.tensor([match_points / 300.]).unsqueeze(0).to(self.device)
                        sign = self.switch_policy(depth_tensor, extras_tensor).item()
                        # print(f'{self.rank}  {self.timestep}  {sign}')
                        if sign > 0.9:
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.full_w, self.full_h))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        else:
                            if goal_dis < 150 :
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                self.temp_goal = None
                                selem = skimage.morphology.disk(1)
                                goal_map = skimage.morphology.dilation(goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                            else:
                                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                                if np.any(new_goal_map > 0):
                                    planner_inputs['goal'] = new_goal_map
                                    temp_goal = np.zeros((self.full_w, self.full_h))
                                    temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                                    self.temp_goal = temp_goal
                                else:
                                    planner_inputs['goal'] = planner_inputs['exp_goal']
                                    self.temp_goal = None

            return planner_inputs
        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            index = self.local_feature_match_lightglue()
                            match_points = index.shape[0]
                            if match_points < 60:
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        if not getattr(self.args, "quiet_log", 0):
                            print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                
            return planner_inputs

    def instance_discriminator(self, planner_inputs, id_lo_whwh_speci):
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        map_pred = np.rint(planner_inputs['map_pred'])
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r * 100.0 / self.args.map_resolution - gx1),
                 int(c * 100.0 / self.args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        goal_mask = self.obs[4+self.gt_goal_idx, :, :]
        # ds = 4
        # goal_mask = goal_mask[:, :, self.gt_goal_idx][ds // 2::ds, ds // 2::ds]

        # try:
        # now the logic here should be termed as get closer enough to check

        if self.instance_imagegoal is None:
            # not initialized
            return planner_inputs
        elif self.global_goal is not None:
            planner_inputs['found_goal'] = 1
            goal_map = pu.threshold_pose_map(self.global_goal, gx1, gx2, gy1, gy2)
            planner_inputs['goal'] = goal_map
            # goal = [self.global_goal[0] - gx1, self.global_goal[1] - gy1]
            # goal = pu.threshold_poses(goal, map_pred.shape)
            # planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            # planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
            return planner_inputs
        elif self.been_stuck:
            
            planner_inputs['found_goal'] = 0
            if self.stuck_goal is None:
                # goal_dis = -30 / self.args.map_resolution
                # goal_angle = 0
                # goal = [start[0]+goal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                #     start[1]+goal_dis*np.cos(np.deg2rad(start_o+goal_angle))]

                navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
                goal = np.array([0, 0])
                for _ in range(100):
                    random_index = np.random.choice(len(navigable_indices))
                    goal = navigable_indices[random_index]
                    if pu.get_l2_distance(goal[0], start[0], goal[1], start[1]) > 16:
                        break

                goal = pu.threshold_poses(goal, map_pred.shape)                
                self.stuck_goal = [int(goal[0])+gx1, int(goal[1])+gy1]
            else:
                goal = np.array([self.stuck_goal[0]-gx1, self.stuck_goal[1]-gy1])
                goal = pu.threshold_poses(goal, map_pred.shape)
            planner_inputs['goal'] = np.zeros((self.local_w, self.local_h))
            planner_inputs['goal'][int(goal[0]), int(goal[1])] = 1
        elif planner_inputs['found_goal'] == 1:

            # preprocess foundation model output
            # this order is not correct, we should follow the order of from near to further
            # id_lo_whwh_speci = sorted(id_lo_whwh_speci, key=lambda s: s[1], reverse=True)
            id_lo_whwh_speci = sorted(id_lo_whwh_speci, 
                key=lambda s: (s[2][2]-s[2][0])**2+(s[2][3]-s[2][1])**2, reverse=True)
            whwh = (id_lo_whwh_speci[0][2] / 4).astype(int)
            w, h = whwh[2]-whwh[0], whwh[3]-whwh[1]
            goal_mask = np.zeros_like(goal_mask)
            goal_mask[whwh[1]:whwh[3], whwh[0]:whwh[2]] = 1.

            index = self.local_feature_match_lightglue()
            match_points = index.shape[0]
            planner_inputs['found_goal'] = 0

            if self.temp_goal is not None:
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
            else:
                goal_map = self.compute_ins_goal_map(whwh, start, start_o)
                if not np.any(goal_map>0) :
                    tgoal_dis = self.compute_ins_dis_v1(self.obs[3, :, :], whwh) / self.args.map_resolution
                    rgb_center = np.array([whwh[3]+whwh[1], whwh[2]+whwh[0]])//2
                    goal_angle = -self.args.hfov / 2 * (rgb_center[1] - self.obs.shape[2]/2) \
                    / (self.obs.shape[2]/2)
                    goal = [start[0]+tgoal_dis*np.sin(np.deg2rad(start_o+goal_angle)), \
                        start[1]+tgoal_dis*np.cos(np.deg2rad(start_o+goal_angle))]
                    goal = pu.threshold_poses(goal, map_pred.shape)
                    rr,cc = skimage.draw.ellipse(goal[0], goal[1], 10, 10, shape=goal_map.shape)
                    goal_map[rr, cc] = 1


                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)

            if goal_dis is None:
                self.temp_goal = None
                planner_inputs['goal'] = planner_inputs['exp_goal']
                selem = skimage.morphology.disk(3)
                goal_map = skimage.morphology.dilation(goal_map, selem)
                self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                if not getattr(self.args, "quiet_log", 0):
                    print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
            else:
                if match_points > 100:
                    planner_inputs['found_goal'] = 1
                    global_goal = np.zeros((self.full_w, self.full_h))
                    global_goal[gx1:gx2, gy1:gy2] = goal_map
                    self.global_goal = global_goal
                    planner_inputs['goal'] = goal_map
                    self.temp_goal = None
                else:
                    if goal_dis < 150 :
                        if match_points > 60:
                            planner_inputs['found_goal'] = 1
                            global_goal = np.zeros((self.full_w, self.full_h))
                            global_goal[gx1:gx2, gy1:gy2] = goal_map
                            self.global_goal = global_goal
                            planner_inputs['goal'] = goal_map
                            self.temp_goal = None
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
                            selem = skimage.morphology.disk(1)
                            goal_map = skimage.morphology.dilation(goal_map, selem)
                            self.goal_map_mask[gx1:gx2, gy1:gy2][goal_map > 0] = 0
                    else:
                        new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                        if np.any(new_goal_map > 0):
                            planner_inputs['goal'] = new_goal_map
                            temp_goal = np.zeros((self.full_w, self.full_h))
                            temp_goal[gx1:gx2, gy1:gy2] = new_goal_map
                            self.temp_goal = temp_goal
                        else:
                            planner_inputs['goal'] = planner_inputs['exp_goal']
                            self.temp_goal = None
            return planner_inputs

        else:
            planner_inputs['goal'] = planner_inputs['exp_goal']
            if self.temp_goal is not None:  
                goal_map = pu.threshold_pose_map(self.temp_goal, gx1, gx2, gy1, gy2)
                goal_dis = self.compute_temp_goal_distance(map_pred, goal_map, start, planning_window)
                planner_inputs['found_goal'] = 0
                new_goal_map = goal_map * self.goal_map_mask[gx1:gx2, gy1:gy2]
                if np.any(new_goal_map > 0):
                    if goal_dis is not None:
                        planner_inputs['goal'] = new_goal_map
                        if goal_dis < 100:
                            index = self.local_feature_match_lightglue()
                            # index = self.local_feature_match_gluestick()
                            match_points = index.shape[0]
                            if match_points < 60:
                                planner_inputs['goal'] = planner_inputs['exp_goal']
                                selem = skimage.morphology.disk(3)
                                new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                                self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                                self.temp_goal = None
                    else:
                        selem = skimage.morphology.disk(3)
                        new_goal_map = skimage.morphology.dilation(new_goal_map, selem)
                        self.goal_map_mask[gx1:gx2, gy1:gy2][new_goal_map > 0] = 0
                        self.temp_goal = None
                        if not getattr(self.args, "quiet_log", 0):
                            print(f"Rank: {self.rank}, timestep: {self.timestep},  temp goal unavigable !")
                else:
                    self.temp_goal = None
                    
                    
            return planner_inputs


    def plan_act_and_preprocess(self, planner_inputs):
        """Function responsible for planning, taking the action and
        preprocessing observations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) mat denoting goal locations
                    'pose_pred' (ndarray): (7,) array denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                     'found_goal' (bool): whether the goal object is found

        Returns:
            obs (ndarray): preprocessed observations ((4+C) x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        # plan
        if planner_inputs["wait"]:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs.shape), 0., False, self.info

        # Reset reward if new long-term goal
        if planner_inputs["new_goal"]:
            self.info["g_reward"] = 0

        action_override = planner_inputs.get("action_override")
        localization_wait_hold = False
        if planner_inputs.get("force_stop", False):
            action = self._terminal_stop_action()
        elif action_override in ("turn_left", "turn_right", "sweep_left", "sweep_right"):
            action = self._non_terminal_turn_action(
                action_override,
                in_place=(
                    planner_inputs.get("goal_source")
                    in (
                        "MINAV_INITIAL_360_SPIN",
                        "INITIAL_360_SPIN",
                        "MINAV_BUFFER_STANDOFF_360_SWEEP",
                        "MINAV_MR_REMAP_360_SWEEP",
                        "MINAV_NAVIGATION_STAGING_FACE_TARGET",
                        "PHASE1_GAUSSMI_GAZE",
                    )
                ),
                turn_angle_deg=(
                    30.0
                    if planner_inputs.get("goal_source")
                    == "PHASE1_GAUSSMI_GAZE"
                    else None
                ),
            )
        elif planner_inputs.get("hold_action", False) or action_override == "hold":
            action = self._non_terminal_hold_action()
            localization_wait_hold = self._is_minav_localization_wait(planner_inputs)
        else:
            if not planner_inputs.get("skip_instance_discriminator", False):
                id_lo_whwh = self.pred_box

                id_lo_whwh_speci = [id_lo_whwh[i] for i in range(len(id_lo_whwh)) \
                            if id_lo_whwh[i][0] == self.gt_goal_idx]


                planner_inputs["found_goal"] = (id_lo_whwh_speci != [])

                self.instance_discriminator(planner_inputs, id_lo_whwh_speci)

            action = self._plan(planner_inputs)

        if self.args.visualize or self.args.print_images:
            self._visualize(planner_inputs)
            if 'sem_map_pred_3d' in planner_inputs:
                self._render_3d(planner_inputs)

        if action['action_args']['velocity_stop'] >= 0:
            if localization_wait_hold:
                return self._return_minav_frozen_hold(planner_inputs, action)

            obs, rew, done, info = super().step(action)
            self.raw_obs = obs[:3, :, :].transpose(1, 2, 0)
            self.raw_depth = obs[3:4, :, :]

            # preprocess obs
            obs = self._preprocess_obs(obs) 
            self.last_action = (action['action_args']['linear_velocity'] > 0)
            self.obs = obs
            info = self._attach_minav_step_info(info, planner_inputs)
            self.info = info

            info['g_reward'] += rew
            self._log_minav_stop_debug(planner_inputs, action, done, info)

            return obs, rew, done, info

        else:
            self.last_action = None
            self.info["sensor_pose"] = [0., 0., 0.]
            return np.zeros(self.obs_shape), 0., False, self.info

    def _log_minav_stop_debug(self, planner_inputs, action, done, info):
        if not getattr(self.args, "use_minav_confirmation_localization", 0):
            return

        action_args = action.get("action_args", {})
        velocity_stop = float(np.asarray(
            action_args.get("velocity_stop", np.array([0]))
        ).reshape(-1)[0])
        linear_velocity = float(np.asarray(
            action_args.get("linear_velocity", np.array([0]))
        ).reshape(-1)[0])
        angular_velocity = float(np.asarray(
            action_args.get("angular_velocity", np.array([0]))
        ).reshape(-1)[0])

        target_mode = planner_inputs.get("target_mode", "unknown")
        fmm_stop = planner_inputs.get("_fmm_stop")
        found_goal = int(bool(planner_inputs.get("found_goal", 0)))
        should_log = (
            bool(done)
            or velocity_stop > 0.0
            or bool(fmm_stop)
            or found_goal == 1
            or bool(planner_inputs.get("hold_action", False))
            or bool(planner_inputs.get("force_stop", False))
            or planner_inputs.get("action_override") is not None
            or target_mode != "exploration"
            or int(getattr(self, "timestep", 0)) % 50 == 0
            or self._minav_planner_debug_enabled(planner_inputs)
        )
        if not should_log:
            return

        dist = info.get("distance_to_goal")
        success = info.get("success")
        spl = info.get("spl")
        try:
            metrics = self.habitat_env.get_metrics()
            dist = metrics.get("distance_to_goal", dist)
            success = metrics.get("success", success)
            spl = metrics.get("spl", spl)
        except Exception:
            pass

        msg = (
            "[MINAV_STOP_DEBUG] rank={rank} ep={ep} t={t} "
            "mode={mode} goal_source={goal_source} found_goal={found_goal} "
            "fmm_stop={fmm_stop} velocity_stop={velocity_stop:.1f} "
            "lin={linear_velocity:.3f} ang={angular_velocity:.3f} "
            "hold={hold} frozen_hold={frozen_hold} "
            "override={override} done={done} "
            "nav_complete={nav_complete} "
            "force_stop={force_stop} force_reason={force_reason} "
            "dtg={dtg} success={success} spl={spl} "
            "stg={stg} start={start} nav_goal_cell={nav_goal_cell} "
            "nav_goal_cell_dist={nav_goal_cell_dist} "
            "nav_goal_cell_reached={nav_goal_cell_reached} "
            "nav_goal_world={nav_goal_world} nav_goal_adjusted={nav_goal_adjusted}"
        ).format(
            rank=self.rank,
            ep=getattr(self, "episode_no", None),
            t=getattr(self, "timestep", None),
            mode=target_mode,
            goal_source=planner_inputs.get("goal_source", "unknown"),
            found_goal=found_goal,
            fmm_stop=fmm_stop,
            velocity_stop=velocity_stop,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            hold=bool(planner_inputs.get("hold_action", False)),
            frozen_hold=bool(info.get("minav_frozen_hold", False)),
            override=planner_inputs.get("action_override"),
            done=bool(done),
            nav_complete=bool(planner_inputs.get("navigation_complete", False)),
            force_stop=bool(planner_inputs.get("force_stop", False)),
            force_reason=planner_inputs.get("force_stop_reason"),
            dtg=None if dist is None else "{:.4f}".format(float(dist)),
            success=success,
            spl=spl,
            stg=planner_inputs.get("_fmm_stg"),
            start=planner_inputs.get("_fmm_start"),
            nav_goal_cell=planner_inputs.get("minav_navigation_goal_local_cell"),
            nav_goal_cell_dist=planner_inputs.get("_minav_nav_goal_cell_distance"),
            nav_goal_cell_reached=planner_inputs.get("_minav_nav_goal_cell_reached"),
            nav_goal_world=planner_inputs.get("minav_navigation_goal_world"),
            nav_goal_adjusted=planner_inputs.get("minav_navigation_goal_adjusted"),
        )
        planner_debug = self._format_minav_planner_debug(planner_inputs, action)
        if planner_debug is not None:
            msg = "{} planner_debug={}".format(msg, planner_debug)
        print(msg)
        logging.info(msg)
        if (
            self._minav_navigation_complete_stop_debug_enabled()
            and bool(planner_inputs.get("navigation_complete", False))
        ):
            nav_payload = {
                "rank": int(self.rank),
                "episode": getattr(self, "episode_no", None),
                "t": getattr(self, "timestep", None),
                "mode": target_mode,
                "goal_source": planner_inputs.get("goal_source", "unknown"),
                "found_goal": found_goal,
                "done": bool(done),
                "navigation_complete": bool(
                    planner_inputs.get("navigation_complete", False)
                ),
                "force_stop": bool(planner_inputs.get("force_stop", False)),
                "force_stop_reason": planner_inputs.get("force_stop_reason"),
                "velocity_stop": velocity_stop,
                "linear_velocity": linear_velocity,
                "angular_velocity": angular_velocity,
                "habitat_distance": None if dist is None else float(dist),
                "habitat_success": success,
                "spl": spl,
                "minav_raw_target_distance": planner_inputs.get(
                    "minav_raw_target_distance"
                ),
                "completion_source": planner_inputs.get("completion_source"),
                "nav_goal_world": planner_inputs.get("minav_navigation_goal_world"),
                "nav_goal_cell": planner_inputs.get(
                    "minav_navigation_goal_local_cell"
                ),
                "nav_goal_cell_dist": planner_inputs.get(
                    "_minav_nav_goal_cell_distance"
                ),
            }
            nav_msg = "[MINAV_NAV_COMPLETE_DEBUG] {}".format(
                json.dumps(
                    nav_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            )
            print(nav_msg)
            logging.info(nav_msg)

    def _minav_planner_debug_enabled(self, planner_inputs=None):
        if not getattr(self.args, "use_minav_confirmation_localization", 0):
            return False
        if planner_inputs is not None and planner_inputs.get("minav_planner_debug"):
            return True
        if getattr(self.args, "minav_planner_debug", 0):
            return True
        return os.environ.get("MINAV_PLANNER_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _minav_stg_angle_debug_enabled(self, planner_inputs=None):
        if not getattr(self.args, "use_minav_confirmation_localization", 0):
            return False
        if getattr(self.args, "minav_stg_angle_debug", 0):
            return True
        return os.environ.get("MINAV_STG_ANGLE_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _minav_navigation_complete_stop_debug_enabled(self):
        if not getattr(self.args, "use_minav_confirmation_localization", 0):
            return False
        if getattr(self.args, "minav_navigation_complete_stop_debug", 0):
            return True
        return os.environ.get("MINAV_NAV_COMPLETE_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    @staticmethod
    def _normalize_minav_angle_deg(angle):
        try:
            value = float(angle)
        except Exception:
            return None
        value = value % 360.0
        if value > 180.0:
            value -= 360.0
        return value

    @staticmethod
    def _minav_float_list(value, max_len=None):
        if value is None:
            return None
        try:
            if isinstance(value, str):
                cleaned = value.strip().replace("[", " ").replace("]", " ")
                arr = np.asarray(
                    [float(v) for v in cleaned.replace(",", " ").split()],
                    dtype=np.float64,
                )
            else:
                arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if max_len is not None:
                arr = arr[:max_len]
            return [float(v) for v in arr.tolist()]
        except Exception:
            return None

    @staticmethod
    def _minav_action_floats(action):
        action_args = (action or {}).get("action_args", {})
        result = {}
        for key in ("linear_velocity", "angular_velocity", "velocity_stop"):
            try:
                result[key] = float(
                    np.asarray(action_args.get(key, np.array([0]))).reshape(-1)[0]
                )
            except Exception:
                result[key] = None
        return result

    def _build_minav_stg_angle_debug(self, start, stg, start_o):
        try:
            stg_x, stg_y = float(stg[0]), float(stg[1])
            start_r, start_c = float(start[0]), float(start[1])
            dr = stg_x - start_r
            dc = stg_y - start_c
            angle_agent = self._normalize_minav_angle_deg(start_o)
            if angle_agent is None:
                return None
            raw_candidates = {
                "current_atan2_dr_dc": math.degrees(math.atan2(dr, dc)),
                "rowcol_atan2_dc_negdr": math.degrees(math.atan2(dc, -dr)),
                "swap_atan2_dc_dr": math.degrees(math.atan2(dc, dr)),
                "negrow_atan2_negdr_dc": math.degrees(math.atan2(-dr, dc)),
                "negcol_atan2_negdc_dr": math.degrees(math.atan2(-dc, dr)),
            }
            candidates = {}
            for name, angle in raw_candidates.items():
                angle_norm = self._normalize_minav_angle_deg(angle)
                relative = self._normalize_minav_angle_deg(angle_agent - angle_norm)
                candidates[name] = {
                    "angle": None if angle_norm is None else round(float(angle_norm), 4),
                    "relative_angle": (
                        None if relative is None else round(float(relative), 4)
                    ),
                    "would_forward": (
                        False if relative is None else bool(abs(relative) <= 15.0)
                    ),
                }
            selected_formula = (
                "rowcol_atan2_dc_negdr"
                if getattr(self.args, "minav_stg_angle_fix", 0)
                else "current_atan2_dr_dc"
            )
            selected = candidates.get(selected_formula, candidates["current_atan2_dr_dc"])
            return {
                "start": [int(start[0]), int(start[1])],
                "stg": [int(stg[0]), int(stg[1])],
                "dr": round(float(dr), 4),
                "dc": round(float(dc), 4),
                "start_o": round(float(start_o), 4),
                "angle_agent": round(float(angle_agent), 4),
                "selected_formula": selected_formula,
                "selected_angle": selected.get("angle"),
                "selected_relative_angle": selected.get("relative_angle"),
                "selected_would_forward": selected.get("would_forward"),
                "fix_enabled": bool(getattr(self.args, "minav_stg_angle_fix", 0)),
                "candidates": candidates,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _select_minav_stg_angle(self, angle_debug, fallback_stg, fallback_start):
        if not angle_debug or angle_debug.get("error"):
            stg_x, stg_y = fallback_stg
            return math.degrees(
                math.atan2(stg_x - fallback_start[0], stg_y - fallback_start[1])
            )
        selected = angle_debug.get("selected_angle")
        if selected is None:
            stg_x, stg_y = fallback_stg
            return math.degrees(
                math.atan2(stg_x - fallback_start[0], stg_y - fallback_start[1])
            )
        return float(selected)

    @staticmethod
    def _classify_minav_action(action):
        values = Instance_Exp_Env_Agent._minav_action_floats(action)
        lin = values.get("linear_velocity")
        ang = values.get("angular_velocity")
        stop = values.get("velocity_stop")
        if stop is not None and stop > 0.0:
            kind = "terminal_stop"
        elif lin is not None and abs(lin) > 1e-6 and (ang is None or abs(ang) <= 1e-6):
            kind = "forward" if lin > 0.0 else "reverse"
        elif ang is not None and abs(ang) > 1e-6:
            kind = "turn"
        else:
            kind = "hold"
        values["kind"] = kind
        return values

    def _emit_minav_stg_angle_decision_debug(
        self,
        planner_inputs,
        action,
        angle_debug,
        target_mode,
    ):
        if not self._minav_stg_angle_debug_enabled(planner_inputs):
            return
        if target_mode not in ("navigation", "buffer"):
            return
        if not angle_debug:
            return
        payload = {
            "rank": int(self.rank),
            "episode": getattr(self, "episode_no", None),
            "t": getattr(self, "timestep", None),
            "mode": target_mode,
            "goal_source": planner_inputs.get("goal_source"),
            "found_goal": int(bool(planner_inputs.get("found_goal", 0))),
            "nav_goal_world": self._minav_float_list(
                planner_inputs.get("minav_navigation_goal_world"),
                max_len=3,
            ),
            "nav_goal_cell": planner_inputs.get("minav_navigation_goal_local_cell"),
            "action": self._classify_minav_action(action),
            "angle_debug": angle_debug,
        }
        msg = "[MINAV_STG_ANGLE_DEBUG] {}".format(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )
        print(msg)
        logging.info(msg)

    def _emit_minav_stg_progress_debug(
        self,
        planner_inputs,
        current_agent_position,
        target_mode,
    ):
        if not self._minav_stg_angle_debug_enabled(planner_inputs):
            return
        prev = self._minav_stg_prev_record
        if not prev:
            return
        if prev.get("mode") != "navigation":
            return
        prev_agent = self._minav_float_list(prev.get("agent_position"), max_len=3)
        curr_agent = self._minav_float_list(current_agent_position, max_len=3)
        nav_goal = self._minav_float_list(prev.get("nav_goal_world"), max_len=3)
        if not prev_agent or not curr_agent or not nav_goal:
            return
        try:
            prev_xz = np.asarray([prev_agent[0], prev_agent[2]], dtype=np.float64)
            curr_xz = np.asarray([curr_agent[0], curr_agent[2]], dtype=np.float64)
            goal_xz = np.asarray([nav_goal[0], nav_goal[2]], dtype=np.float64)
            move_vec = curr_xz - prev_xz
            goal_vec = goal_xz - prev_xz
            move_norm = float(np.linalg.norm(move_vec))
            goal_norm = float(np.linalg.norm(goal_vec))
            if move_norm <= 1e-6 or goal_norm <= 1e-6:
                progress = 0.0
                move_vs_goal_deg = None
            else:
                goal_unit = goal_vec / goal_norm
                progress = float(np.dot(move_vec, goal_unit))
                cos_value = float(
                    np.clip(np.dot(move_vec, goal_vec) / (move_norm * goal_norm), -1.0, 1.0)
                )
                move_vs_goal_deg = math.degrees(math.acos(cos_value))
            prev_action = prev.get("action", {})
            prev_angle_debug = prev.get("angle_debug", {})
            stats = self._update_minav_stg_angle_stats(
                prev_angle_debug,
                progress,
                move_norm,
                prev_action,
            )
            payload = {
                "rank": int(self.rank),
                "episode": getattr(self, "episode_no", None),
                "t": getattr(self, "timestep", None),
                "mode": target_mode,
                "prev_mode": prev.get("mode"),
                "prev_goal_source": prev.get("goal_source"),
                "prev_agent_position": prev_agent,
                "curr_agent_position": curr_agent,
                "nav_goal_world": nav_goal,
                "move_vec_xz": [round(float(v), 6) for v in move_vec.tolist()],
                "goal_vec_xz": [round(float(v), 6) for v in goal_vec.tolist()],
                "move_norm_m": round(float(move_norm), 6),
                "goal_dist_prev_m": round(float(goal_norm), 6),
                "progress_toward_goal_m": round(float(progress), 6),
                "move_vs_goal_deg": (
                    None if move_vs_goal_deg is None else round(float(move_vs_goal_deg), 4)
                ),
                "prev_action": prev_action,
                "prev_start": prev_angle_debug.get("start"),
                "prev_stg": prev_angle_debug.get("stg"),
                "prev_selected_formula": prev_angle_debug.get("selected_formula"),
                "prev_selected_relative_angle": prev_angle_debug.get(
                    "selected_relative_angle"
                ),
                "prev_candidates": prev_angle_debug.get("candidates"),
                "summary": stats,
            }
            msg = "[MINAV_STG_PROGRESS_DEBUG] {}".format(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
            )
            print(msg)
            logging.info(msg)
            if int(getattr(self, "timestep", 0) or 0) % 20 == 0:
                self._emit_minav_stg_angle_summary()
        except Exception as exc:
            msg = "[MINAV_STG_PROGRESS_DEBUG] {}".format(
                json.dumps(
                    {"error": str(exc), "t": getattr(self, "timestep", None)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(msg)
            logging.info(msg)

    def _update_minav_stg_angle_stats(self, angle_debug, progress, move_norm, action):
        candidates = (angle_debug or {}).get("candidates") or {}
        if move_norm <= 1e-6 or (action or {}).get("kind") != "forward":
            return self._summarize_minav_stg_angle_stats()
        for name, data in candidates.items():
            stat = self._minav_stg_angle_stats.setdefault(
                name,
                {
                    "forward_count": 0,
                    "forward_positive_progress": 0,
                    "forward_negative_progress": 0,
                    "forward_progress_sum": 0.0,
                },
            )
            if not data.get("would_forward"):
                continue
            stat["forward_count"] += 1
            stat["forward_progress_sum"] += float(progress)
            if progress >= 0.0:
                stat["forward_positive_progress"] += 1
            else:
                stat["forward_negative_progress"] += 1
        return self._summarize_minav_stg_angle_stats()

    def _summarize_minav_stg_angle_stats(self):
        summary = {}
        for name, stat in self._minav_stg_angle_stats.items():
            count = int(stat.get("forward_count", 0))
            progress_sum = float(stat.get("forward_progress_sum", 0.0))
            summary[name] = {
                "forward_count": count,
                "forward_positive_progress": int(
                    stat.get("forward_positive_progress", 0)
                ),
                "forward_negative_progress": int(
                    stat.get("forward_negative_progress", 0)
                ),
                "forward_progress_sum": round(progress_sum, 6),
                "forward_mean_progress": (
                    None if count <= 0 else round(progress_sum / float(count), 6)
                ),
            }
        return summary

    def _emit_minav_stg_angle_summary(self):
        payload = {
            "rank": int(self.rank),
            "episode": getattr(self, "episode_no", None),
            "t": getattr(self, "timestep", None),
            "summary": self._summarize_minav_stg_angle_stats(),
        }
        msg = "[MINAV_STG_ANGLE_SUMMARY] {}".format(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        )
        print(msg)
        logging.info(msg)

    def _store_minav_stg_prev_record(
        self,
        planner_inputs,
        action,
        angle_debug,
        current_agent_position,
        target_mode,
    ):
        if not self._minav_stg_angle_debug_enabled(planner_inputs):
            self._minav_stg_prev_record = None
            return
        self._minav_stg_prev_record = {
            "mode": target_mode,
            "goal_source": planner_inputs.get("goal_source"),
            "agent_position": self._minav_float_list(current_agent_position, max_len=3),
            "nav_goal_world": self._minav_float_list(
                planner_inputs.get("minav_navigation_goal_world"),
                max_len=3,
            ),
            "action": self._classify_minav_action(action),
            "angle_debug": angle_debug,
            "t": getattr(self, "timestep", None),
        }

    def _summarize_minav_goal_cells(self, goal):
        try:
            goal_arr = np.asarray(goal)
            cells = np.argwhere(goal_arr > 0)
            count = int(cells.shape[0])
            if count == 0:
                return {"count": 0}
            row_min, col_min = cells.min(axis=0)
            row_max, col_max = cells.max(axis=0)
            centroid = cells.mean(axis=0)
            sample = cells[: min(8, count)].astype(int).tolist()
            return {
                "count": count,
                "bbox": [
                    int(row_min),
                    int(col_min),
                    int(row_max),
                    int(col_max),
                ],
                "centroid": [
                    round(float(centroid[0]), 2),
                    round(float(centroid[1]), 2),
                ],
                "sample": sample,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _summarize_minav_map_delta(self, current, previous):
        try:
            curr = np.asarray(current)
            summary = {
                "shape": [int(v) for v in curr.shape[:2]],
                "nonzero": int(np.count_nonzero(curr)),
                "sum": round(float(np.sum(curr)), 3),
            }
            if previous is None or np.asarray(previous).shape != curr.shape:
                summary.update(
                    {
                        "changed": None,
                        "added": None,
                        "removed": None,
                        "abs_delta_sum": None,
                    }
                )
                return summary
            prev = np.asarray(previous)
            curr_occ = curr > 0
            prev_occ = prev > 0
            summary.update(
                {
                    "changed": int(np.count_nonzero(curr != prev)),
                    "added": int(np.count_nonzero(curr_occ & ~prev_occ)),
                    "removed": int(np.count_nonzero(~curr_occ & prev_occ)),
                    "abs_delta_sum": round(float(np.sum(np.abs(curr - prev))), 3),
                }
            )
            return summary
        except Exception as exc:
            return {"error": str(exc)}

    def _format_minav_planner_debug(self, planner_inputs, action):
        debug = planner_inputs.get("_minav_planner_debug")
        if not debug:
            return None
        try:
            action_args = action.get("action_args", {})
            debug = dict(debug)
            debug["planned_action"] = {
                "linear_velocity": float(
                    np.asarray(action_args.get("linear_velocity", [0])).reshape(-1)[0]
                ),
                "angular_velocity": float(
                    np.asarray(action_args.get("angular_velocity", [0])).reshape(-1)[0]
                ),
                "velocity_stop": float(
                    np.asarray(action_args.get("velocity_stop", [0])).reshape(-1)[0]
                ),
            }
            debug["last_action_after_step"] = self.last_action
            return json.dumps(debug, ensure_ascii=False, sort_keys=True, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False, default=str)

    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs.get('exp_pred', np.zeros_like(map_pred)))
        goal = planner_inputs['goal']
        minav_planner_debug = self._minav_planner_debug_enabled(planner_inputs)
        last_action_before_plan = self.last_action

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        # self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
        #                                start[1] - 0:start[1] + 1] = 1
        rr, cc, _ = line_aa(last_start[0], last_start[1], start[0], start[1])
        self.visited[gx1:gx2, gy1:gy2][rr, cc] += 1
        visited_local = self.visited[gx1:gx2, gy1:gy2]
        cur_r1 = max(0, int(start[0]) - 2)
        cur_r2 = min(visited_local.shape[0], int(start[0]) + 3)
        cur_c1 = max(0, int(start[1]) - 2)
        cur_c2 = min(visited_local.shape[1], int(start[1]) + 3)
        visited_local[cur_r1:cur_r2, cur_c1:cur_c2] = 1

        if args.visualize or args.print_images:            
            self.visited_vis[gx1:gx2, gy1:gy2] = \
                vu.draw_line(last_start, start,
                             self.visited_vis[gx1:gx2, gy1:gy2])
            self.visited_vis[gx1:gx2, gy1:gy2][
                cur_r1:cur_r2, cur_c1:cur_c2
            ] = 1

        # relieve the stuck goal
        x1, y1, t1 = self.last_loc
        x2, y2, _ = self.curr_loc
        if abs(x1 - x2) >= 0.05 or abs(y1 - y2) >= 0.05:
            self.been_stuck = False
            self.stuck_goal = None

        # Collision check
        collision_before = self.collision_map.copy() if minav_planner_debug else None
        collision_checked = False
        collision_detected = False
        collision_distance = None
        if self.last_action == 1:
            collision_checked = True
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                    self.been_stuck = True
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1                

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            collision_distance = float(dist)
            if dist < args.collision_threshold:  # Collision
                collision_detected = True
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                             + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                             - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1
        if minav_planner_debug:
            collision_after = self.collision_map
            collision_changed = int(np.count_nonzero(collision_after != collision_before))
            planner_inputs["_minav_planner_debug"] = {
                "agent_position": planner_inputs.get("agent_position"),
                "agent_rotation_xyzw": planner_inputs.get("agent_rotation_xyzw"),
                "last_action_before_plan": last_action_before_plan,
                "collision_map": {
                    "sum_before": int(np.sum(collision_before)),
                    "sum_after": int(np.sum(collision_after)),
                    "changed_cells": collision_changed,
                    "updated": bool(collision_changed > 0),
                    "checked": bool(collision_checked),
                    "detected": bool(collision_detected),
                    "distance": (
                        None
                        if collision_distance is None
                        else round(float(collision_distance), 4)
                    ),
                    "threshold": round(float(args.collision_threshold), 4),
                    "col_width": int(self.col_width),
                    "been_stuck": bool(getattr(self, "been_stuck", False)),
                },
                "goal_nonzero": self._summarize_minav_goal_cells(goal),
                "map_pred_delta": self._summarize_minav_map_delta(
                    map_pred,
                    self._minav_debug_prev_map_pred,
                ),
                "exp_pred_delta": self._summarize_minav_map_delta(
                    exp_pred,
                    self._minav_debug_prev_exp_pred,
                ),
                "pose_pred": [
                    round(float(v), 4)
                    for v in np.asarray(planner_inputs.get("pose_pred", [])).reshape(-1)
                ],
            }
            self._minav_debug_prev_map_pred = np.array(map_pred, copy=True)
            self._minav_debug_prev_exp_pred = np.array(exp_pred, copy=True)

        target_mode = planner_inputs.get("target_mode")
        minav_executor_mode = target_mode in (
            "navigation",
            "buffer",
            "cluster_recovery",
        )
        minav_goal_dilation_radius = None
        if (
            bool(getattr(args, "use_minav_confirmation_localization", 0))
            and minav_executor_mode
        ):
            requested_goal_dilation = planner_inputs.get(
                "minav_goal_dilation_radius"
            )
            if requested_goal_dilation is not None:
                minav_goal_dilation_radius = max(
                    0, int(requested_goal_dilation)
                )
            elif target_mode == "navigation":
                minav_goal_dilation_radius = int(
                    getattr(args, "minav_navigation_goal_dilation_radius", 5)
                )
            elif target_mode == "cluster_recovery":
                minav_goal_dilation_radius = int(
                    getattr(args, "minav_cluster_recovery_goal_dilation_radius", 2)
                )
            else:
                minav_goal_dilation_radius = int(
                    getattr(args, "minav_buffer_goal_dilation_radius", 10)
                )
            if "_minav_planner_debug" in planner_inputs:
                planner_inputs["_minav_planner_debug"][
                    "goal_dilation_radius"
                ] = int(minav_goal_dilation_radius)
        current_agent_position = planner_inputs.get("agent_position")
        if self._minav_stg_angle_debug_enabled(planner_inputs):
            self._emit_minav_stg_progress_debug(
                planner_inputs,
                current_agent_position,
                target_mode,
            )
        stg, stop = self._get_stg(
            map_pred,
            start,
            np.copy(goal),
            planning_window,
            use_large_goal_dilation=(
                bool(getattr(args, "use_minav_confirmation_localization", 0))
                and minav_executor_mode
            ),
            goal_dilation_radius=minav_goal_dilation_radius,
            traversible_constraint=planner_inputs.get("minav_path_constraint_mask"),
            goal_region_constraint=planner_inputs.get("minav_goal_region_mask"),
        )
        frontier_active_goal = getattr(self, "frontier_active_goal", None)
        frontier_active_goal_tuple = None
        if frontier_active_goal is not None:
            frontier_active_goal_tuple = (
                int(frontier_active_goal[0]),
                int(frontier_active_goal[1]),
            )
        self._last_frontier_goal_blocked = self._is_frontier_active_goal_blocked(
            map_pred,
            planning_window,
        )
        nav_goal_cell = planner_inputs.get("minav_navigation_goal_local_cell")
        nav_goal_cell_distance = None
        nav_goal_cell_reached = False
        if (
            minav_executor_mode
            and nav_goal_cell is not None
        ):
            try:
                nav_goal_arr = np.asarray(nav_goal_cell, dtype=np.float32).reshape(-1)
                if nav_goal_arr.size >= 2:
                    nav_goal_cell_distance = pu.get_l2_distance(
                        float(start[0]),
                        float(nav_goal_arr[0]),
                        float(start[1]),
                        float(nav_goal_arr[1]),
                    )
                    nav_goal_cell_reached = nav_goal_cell_distance < 3.0
            except Exception:
                nav_goal_cell_distance = None
                nav_goal_cell_reached = False
        planner_inputs["_fmm_stop"] = bool(stop)
        planner_inputs["_fmm_replan"] = bool(getattr(self, "_last_fmm_replan", False))
        planner_inputs["_multi_fmm_exact_cache_hit"] = bool(
            getattr(self, "_last_multi_fmm_cache_hit", False)
        )
        planner_inputs["_fmm_stg"] = (
            int(stg[0]) if stg is not None else None,
            int(stg[1]) if stg is not None else None,
        )
        planner_inputs["_frontier_active_goal"] = frontier_active_goal_tuple
        planner_inputs["_frontier_active_goal_reached"] = bool(
            getattr(self, "_last_frontier_goal_reached", False)
        )
        planner_inputs["_frontier_active_goal_blocked"] = bool(
            getattr(self, "_last_frontier_goal_blocked", False)
        )
        planner_inputs["_frontier_goal_count"] = int(np.count_nonzero(np.asarray(goal) > 0))
        minav_mode = bool(getattr(args, "use_minav_confirmation_localization", 0))
        planner_inputs["_fmm_start"] = (int(start[0]), int(start[1]))
        planner_inputs["_minav_nav_goal_cell_distance"] = nav_goal_cell_distance
        planner_inputs["_minav_nav_goal_cell_reached"] = bool(nav_goal_cell_reached)
        stg_angle_debug = None
        if minav_mode and stg is not None:
            stg_angle_debug = self._build_minav_stg_angle_debug(
                start,
                stg,
                start_o,
            )
        planner_inputs["_minav_phase_goal_reached"] = bool(
            minav_mode
            and minav_executor_mode
            and stop
            and planner_inputs["found_goal"] == 1
        )
        stuck_recovery_enabled = self._stuck_recovery_enabled()
        if not stuck_recovery_enabled:
            self.stuck_recovery_queue = []
            self.stuck_same_stg_count = 0
            self.stuck_last_stg = None
            self.stuck_last_position = None
        elif target_mode in ("navigation", "cluster_recovery"):
            self.stuck_recovery_queue = []
        elif self.stuck_recovery_queue:
            if "_minav_planner_debug" in planner_inputs:
                planner_inputs["_minav_planner_debug"]["stuck_recovery"] = {
                    "enabled": stuck_recovery_enabled,
                    "active": True,
                    "queued": len(self.stuck_recovery_queue),
                    "triggered": False,
                }
        elif not stop:
            self._maybe_trigger_stuck_recovery(
                planner_inputs,
                stg,
                planning_window,
                False,
            )
        # if self.global_goal is not None:
        #     stg = (self.global_goal[0]-gx1, self.global_goal[1]-gy1)


        # modify the code with velocity control
        # Deterministic Local Policy
        if (
            planner_inputs.get("_minav_phase_goal_reached", False)
            and target_mode in ("buffer", "cluster_recovery")
        ):
            self.stuck_recovery_queue = []
            action = self._non_terminal_hold_action()
        elif (
            stuck_recovery_enabled
            and self.stuck_recovery_queue
            and target_mode != "navigation"
        ):
            action = self._pop_stuck_recovery_action(planner_inputs)
        elif (
            stop and planner_inputs['found_goal'] == 1
        ) or (bool(nav_goal_cell_reached) and not minav_mode):
            action = {
                "action": ("velocity_control", "velocity_stop"),
                "action_args": {
                    "angular_velocity": np.array([0]),
                    "linear_velocity": np.array([0]),
                    "camera_pitch_velocity": np.array([0]),
                    "velocity_stop": np.array([1]),
                },
            }
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = self._select_minav_stg_angle(
                stg_angle_debug,
                stg,
                start,
            )
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            # if relative_angle > self.args.turn_angle / 2.:
            if relative_angle > 15.:
                ang_vel = np.array([abs(relative_angle) / 60.])
                ang_vel = np.clip(ang_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": -ang_vel,
                        "linear_velocity": np.array([-1]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }
            # elif relative_angle < -self.args.turn_angle / 2.:
            elif relative_angle < -15.:
                ang_vel = np.array([abs(relative_angle) / 60.])
                ang_vel = np.clip(ang_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": ang_vel,
                        "linear_velocity": np.array([-1]),
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }
            else:
                lin_vel = np.array([pu.get_l2_distance(stg_x, start[0], stg_y, start[1]) * 5 / 35.])
                lin_vel = np.clip(lin_vel, 0., 1.)
                action = {
                    "action": ("velocity_control", "velocity_stop"),
                    "action_args": {
                        "angular_velocity": np.array([0]),
                        "linear_velocity": lin_vel,
                        "camera_pitch_velocity": np.array([0]),
                        "velocity_stop": np.array([0]),
                    },
                }

        self._emit_minav_stg_angle_decision_debug(
            planner_inputs,
            action,
            stg_angle_debug,
            target_mode,
        )
        self._store_minav_stg_prev_record(
            planner_inputs,
            action,
            stg_angle_debug,
            current_agent_position,
            target_mode,
        )
        return action

    def _cluster_frontiers_with_dilation(
        self,
        frontier_map,
        dilation_radius=2.0,
        min_size=5,
    ):
        frontier_bool = np.asarray(frontier_map).astype(bool)
        if not np.any(frontier_bool):
            return []

        if dilation_radius > 0:
            radius = int(math.ceil(float(dilation_radius)))
            yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            footprint = (xx * xx + yy * yy) <= float(dilation_radius) ** 2
            clustered_mask = skimage.morphology.binary_dilation(
                frontier_bool,
                footprint,
            )
        else:
            clustered_mask = frontier_bool

        labeled, num_labels = skimage.measure.label(
            clustered_mask,
            return_num=True,
            connectivity=2,
        )
        clusters = []
        for label_id in range(1, num_labels + 1):
            cluster_mask = (labeled == label_id) & frontier_bool
            ys, xs = np.where(cluster_mask)
            if len(ys) < min_size:
                continue
            clusters.append([
                (int(y), int(x))
                for y, x in zip(ys.tolist(), xs.tolist())
            ])
        return clusters

    def _pick_frontier_representative_points(
        self,
        cluster,
        num_points=3,
        occupancy_map=None,
        explored_map=None,
    ):
        if not cluster or num_points <= 0:
            return []

        points = np.asarray(cluster, dtype=np.int32)
        if len(points) <= num_points:
            return [(int(row), int(col)) for row, col in points]

        center = points.astype(np.float32).mean(axis=0)
        center_dist = np.linalg.norm(points.astype(np.float32) - center, axis=1)
        center_scale = float(np.percentile(center_dist, 90))
        center_norm = center_dist / max(center_scale, 1.0)

        clearance_norm = np.ones(len(points), dtype=np.float32)
        if occupancy_map is not None and np.shape(occupancy_map) == np.shape(explored_map):
            free_mask = (np.asarray(occupancy_map) <= 0.5).astype(np.uint8)
            if free_mask.size > 0:
                clearance = cv2.distanceTransform(
                    free_mask,
                    cv2.DIST_L2,
                    cv2.DIST_MASK_PRECISE,
                )
                rows = np.clip(points[:, 0], 0, clearance.shape[0] - 1)
                cols = np.clip(points[:, 1], 0, clearance.shape[1] - 1)
                clearance_values = clearance[rows, cols].astype(np.float32)
                clearance_scale = float(np.percentile(clearance_values, 90))
                clearance_norm = np.clip(
                    clearance_values / max(clearance_scale, 1.0),
                    0.0,
                    2.0,
                )

        explored_bonus = np.zeros(len(points), dtype=np.float32)
        if explored_map is not None:
            explored_arr = np.asarray(explored_map)
            if explored_arr.shape == tuple(np.shape(occupancy_map)):
                rows = np.clip(points[:, 0], 0, explored_arr.shape[0] - 1)
                cols = np.clip(points[:, 1], 0, explored_arr.shape[1] - 1)
                explored_bonus = explored_arr[rows, cols].astype(np.float32)

        scores = center_norm - 0.45 * clearance_norm - 0.15 * explored_bonus
        order = np.argsort(scores)
        min_separation = max(6.0, min(20.0, center_scale * 0.35))

        selected = []
        selected_indices = set()
        for idx in order:
            candidate = points[int(idx)]
            if all(
                np.linalg.norm(candidate.astype(np.float32) - points[sel].astype(np.float32))
                >= min_separation
                for sel in selected_indices
            ):
                selected.append((int(candidate[0]), int(candidate[1])))
                selected_indices.add(int(idx))
                if len(selected) >= num_points:
                    break

        if len(selected) < num_points:
            for idx in order:
                idx = int(idx)
                if idx in selected_indices:
                    continue
                candidate = points[idx]
                selected.append((int(candidate[0]), int(candidate[1])))
                selected_indices.add(idx)
                if len(selected) >= num_points:
                    break

        return selected

    def _draw_frontier_active_goal(self, frontier_vis):
        if frontier_vis is None:
            return None
        active_goal = getattr(self, "frontier_active_goal", None)
        if active_goal is None:
            return frontier_vis

        row, col = int(active_goal[0]), int(active_goal[1])
        if row < 0 or col < 0 or row >= frontier_vis.shape[0] or col >= frontier_vis.shape[1]:
            return frontier_vis

        cv2.circle(frontier_vis, (col, row), 8, (255, 255, 255), 2)
        cv2.circle(frontier_vis, (col, row), 5, (255, 0, 0), -1)
        cv2.circle(frontier_vis, (col, row), 2, (0, 0, 0), -1)
        return frontier_vis

    @staticmethod
    def _add_map_boundary(mat, value=1):
        h, w = mat.shape
        new_mat = np.zeros((h + 2, w + 2), dtype=np.asarray(mat).dtype) + value
        new_mat[1:h + 1, 1:w + 1] = mat
        return new_mat

    def _force_footprint_and_visited_free(
        self,
        obs_map,
        exp_map,
        pose_pred=None,
        planning_window=None,
        start=None,
    ):
        """Mark the agent footprint and traversed path as explored free locally."""
        obs_out = np.asarray(obs_map).copy()
        exp_out = np.asarray(exp_map).copy()
        if obs_out.shape != exp_out.shape:
            return obs_out, exp_out

        free_mask = np.zeros(obs_out.shape, dtype=bool)
        gx1 = gx2 = gy1 = gy2 = None

        if planning_window is not None:
            try:
                gx1, gx2, gy1, gy2 = [int(v) for v in planning_window]
            except Exception:
                gx1 = gx2 = gy1 = gy2 = None
        elif pose_pred is not None and len(pose_pred) >= 7:
            try:
                gx1, gx2, gy1, gy2 = [int(v) for v in pose_pred[3:7]]
            except Exception:
                gx1 = gx2 = gy1 = gy2 = None

        if (
            gx1 is not None
            and self.visited is not None
            and 0 <= gx1 < gx2 <= self.visited.shape[0]
            and 0 <= gy1 < gy2 <= self.visited.shape[1]
        ):
            visited_local = self.visited[gx1:gx2, gy1:gy2]
            if visited_local.shape == free_mask.shape:
                free_mask |= visited_local > 0

        if start is None and pose_pred is not None and len(pose_pred) >= 7:
            try:
                start_x, start_y = float(pose_pred[0]), float(pose_pred[1])
                if gx1 is None or gy1 is None:
                    gx1, gy1 = int(pose_pred[3]), int(pose_pred[5])
                start = [
                    int(start_y * 100.0 / self.args.map_resolution - gx1),
                    int(start_x * 100.0 / self.args.map_resolution - gy1),
                ]
            except Exception:
                start = None

        if start is not None:
            try:
                row = int(np.clip(int(start[0]), 0, free_mask.shape[0] - 1))
                col = int(np.clip(int(start[1]), 0, free_mask.shape[1] - 1))
                footprint_radius = max(
                    1,
                    int(math.ceil(17.0 / max(float(self.args.map_resolution), 1e-6))),
                )
                footprint = np.zeros_like(free_mask, dtype=np.uint8)
                cv2.circle(footprint, (col, row), footprint_radius, 1, -1)
                free_mask |= footprint.astype(bool)
            except Exception:
                pass

        if (
            gx1 is not None
            and self.collision_map is not None
            and 0 <= gx1 < gx2 <= self.collision_map.shape[0]
            and 0 <= gy1 < gy2 <= self.collision_map.shape[1]
        ):
            collision_local = self.collision_map[gx1:gx2, gy1:gy2]
            if collision_local.shape == free_mask.shape:
                free_mask &= collision_local != 1

        obs_out[free_mask] = 0
        exp_out[free_mask] = 1
        return obs_out, exp_out

    def _mark_frontier_goal_visited(self, goal, max_history=100):
        if goal is None:
            return
        try:
            point = (int(goal[0]), int(goal[1]))
        except Exception:
            return
        radius = 1.0
        for old in getattr(self, "frontier_visited_goals", []) or []:
            if pu.get_l2_distance(point[0], old[0], point[1], old[1]) <= radius:
                return
        self.frontier_visited_goals.append(point)
        if len(self.frontier_visited_goals) > int(max_history):
            self.frontier_visited_goals = self.frontier_visited_goals[-int(max_history):]

    def _frontier_goal_in_cooldown(self, point, radius_px=8.0):
        visited_goals = getattr(self, "frontier_visited_goals", []) or []
        if not visited_goals:
            return False
        row, col = int(point[0]), int(point[1])
        for old_row, old_col in visited_goals:
            if pu.get_l2_distance(row, old_row, col, old_col) <= float(radius_px):
                return True
        return False

    def _frontier_initial_block_radius_px(self):
        radius_m = float(getattr(self.args, "frontier_initial_block_radius_m", 2.3))
        return radius_m * 100.0 / max(float(self.args.map_resolution), 1e-6)

    def _frontier_visited_path_block_radius_px(self):
        radius_m = float(
            getattr(self.args, "frontier_visited_path_block_radius_m", 0.5)
        )
        return radius_m * 100.0 / max(float(self.args.map_resolution), 1e-6)

    def _log_frontier_filter_debug(self, debug):
        if not bool(getattr(self.args, "frontier_debug_log", 1)):
            return
        try:
            payload = {
                "rank": int(getattr(self, "rank", -1)),
                "episode": (
                    None
                    if getattr(self, "episode_no", None) is None
                    else int(getattr(self, "episode_no"))
                ),
                "timestep": int(getattr(self, "timestep", -1)),
                "input_count": int(debug.get("input_count", 0) or 0),
                "kept_count": int(debug.get("kept_count", 0) or 0),
                "filtered_initial_start": int(
                    debug.get("filtered_initial_start", 0) or 0
                ),
                "filtered_visited_path": int(
                    debug.get("filtered_visited_path", 0) or 0
                ),
                "filtered_near": int(debug.get("filtered_near", 0) or 0),
                "filtered_cooldown": int(debug.get("filtered_cooldown", 0) or 0),
                "filtered_blocked": int(debug.get("filtered_blocked", 0) or 0),
                "filtered_unreachable": int(debug.get("filtered_unreachable", 0) or 0),
                "filtered_long_path": int(debug.get("filtered_long_path", 0) or 0),
                "filtered_detour": int(debug.get("filtered_detour", 0) or 0),
                "initial_block_cell": debug.get("initial_block_cell"),
                "initial_block_radius_px": debug.get("initial_block_radius_px"),
                "visited_path_block_radius_px": debug.get(
                    "visited_path_block_radius_px"
                ),
                "min_goal_dist_px": debug.get("min_goal_dist_px"),
                "max_fmm_dist_px": debug.get("max_fmm_dist_px"),
                "max_detour_ratio": debug.get("max_detour_ratio"),
                "fmm_dist_min": debug.get("fmm_dist_min"),
                "fmm_dist_max": debug.get("fmm_dist_max"),
                "fmm_error": debug.get("fmm_error"),
                "used_euclidean_fallback": bool(
                    debug.get("used_euclidean_fallback", False)
                ),
            }
            print("[FRONTIER_FILTER_DEBUG] {}".format(
                json.dumps(payload, sort_keys=True)
            ))
        except Exception:
            pass

    def _frontier_goal_in_episode_initial_block(
        self,
        point,
        planning_window,
        radius_px=None,
    ):
        initial_cell = getattr(self, "frontier_initial_block_cell", None)
        if initial_cell is None:
            return False
        try:
            gx1, _, gy1, _ = [int(v) for v in planning_window]
            row, col = int(point[0]), int(point[1])
            full_row = gx1 + row
            full_col = gy1 + col
            init_row, init_col = int(initial_cell[0]), int(initial_cell[1])
            if radius_px is None:
                radius_px = self._frontier_initial_block_radius_px()
            return (
                pu.get_l2_distance(full_row, init_row, full_col, init_col)
                <= float(radius_px)
            )
        except Exception:
            return False

    def _episode_initial_frontier_block_mask(
        self,
        shape,
        planning_window,
        radius_px=None,
    ):
        mask = np.zeros(shape, dtype=bool)
        initial_cell = getattr(self, "frontier_initial_block_cell", None)
        if initial_cell is None or len(shape) < 2:
            return mask
        try:
            gx1, _, gy1, _ = [int(v) for v in planning_window]
            init_row, init_col = int(initial_cell[0]), int(initial_cell[1])
            local_row = init_row - gx1
            local_col = init_col - gy1
            if radius_px is None:
                radius_px = self._frontier_initial_block_radius_px()
            radius = max(0, int(math.ceil(float(radius_px))))
            if (
                local_row + radius < 0
                or local_col + radius < 0
                or local_row - radius >= int(shape[0])
                or local_col - radius >= int(shape[1])
            ):
                return mask
            mask_u8 = np.zeros(shape, dtype=np.uint8)
            cv2.circle(mask_u8, (int(local_col), int(local_row)), radius, 1, -1)
            return mask_u8.astype(bool)
        except Exception:
            return mask

    def _visited_path_frontier_block_mask(
        self,
        shape,
        planning_window,
        radius_px=None,
    ):
        mask = np.zeros(shape, dtype=bool)
        if self.visited is None or len(shape) < 2:
            return mask
        try:
            gx1, gx2, gy1, gy2 = [int(v) for v in planning_window]
            if not (
                0 <= gx1 < gx2 <= self.visited.shape[0]
                and 0 <= gy1 < gy2 <= self.visited.shape[1]
            ):
                return mask
            visited_local = np.asarray(self.visited[gx1:gx2, gy1:gy2]) > 0
            if visited_local.shape != mask.shape or not np.any(visited_local):
                return mask
            if radius_px is None:
                radius_px = self._frontier_visited_path_block_radius_px()
            radius = int(math.ceil(float(radius_px)))
            if radius <= 0:
                return visited_local.astype(bool)
            return skimage.morphology.binary_dilation(
                visited_local,
                skimage.morphology.disk(radius),
            ).astype(bool)
        except Exception:
            return mask

    def _build_frontier_traversible(self, obs_map, start, planning_window):
        gx1, gx2, gy1, gy2 = [int(v) for v in planning_window]
        grid = np.rint(obs_map).astype(np.uint8)
        traversible = skimage.morphology.binary_dilation(
            grid,
            self.selem,
        ) != True

        if self.visited is not None:
            visited_local = self.visited[gx1:gx2, gy1:gy2]
            if visited_local.shape == traversible.shape:
                traversible[visited_local > 0] = 1

        if self.collision_map is not None:
            collision_local = self.collision_map[gx1:gx2, gy1:gy2]
            if collision_local.shape == traversible.shape:
                traversible[collision_local == 1] = 0

        row, col = int(start[0]), int(start[1])
        row1 = max(0, row - 1)
        row2 = min(traversible.shape[0], row + 2)
        col1 = max(0, col - 1)
        col2 = min(traversible.shape[1], col + 2)
        traversible[row1:row2, col1:col2] = 1
        return self._add_map_boundary(traversible.astype(np.float32), value=1)

    def _is_frontier_active_goal_blocked(self, map_pred, planning_window):
        active_goal = getattr(self, "frontier_active_goal", None)
        if active_goal is None:
            return False
        try:
            row, col = int(active_goal[0]), int(active_goal[1])
            map_arr = np.asarray(map_pred)
            if row < 0 or col < 0 or row >= map_arr.shape[0] or col >= map_arr.shape[1]:
                return True
            if map_arr[row, col] > 0:
                return True
            gx1, gx2, gy1, gy2 = [int(v) for v in planning_window]
            full_row = gx1 + row
            full_col = gy1 + col
            collision = getattr(self, "collision_map", None)
            if (
                collision is not None
                and 0 <= full_row < collision.shape[0]
                and 0 <= full_col < collision.shape[1]
                and collision[full_row, full_col] == 1
            ):
                return True
        except Exception:
            return False
        return False

    def _score_frontier_representatives_by_fmm(
        self,
        representative_points,
        obs_map,
        frontier_mask,
        start,
        planning_window,
    ):
        args = self.args
        min_goal_dist_px = float(getattr(args, "frontier_min_goal_dist_px", 10.0))
        cooldown_radius_px = float(getattr(args, "frontier_cooldown_radius_px", 8.0))
        initial_block_radius_px = self._frontier_initial_block_radius_px()
        visited_path_block_radius_px = self._frontier_visited_path_block_radius_px()
        max_fmm_dist_px = float(getattr(args, "frontier_max_fmm_dist_px", 300.0))
        max_detour_ratio = float(getattr(args, "frontier_max_detour_ratio", 5.0))
        min_detour_dist_px = float(getattr(args, "frontier_min_detour_dist_px", 30.0))
        frontier_bool = np.asarray(frontier_mask).astype(bool)
        visited_path_block_mask = self._visited_path_frontier_block_mask(
            frontier_bool.shape,
            planning_window,
            radius_px=visited_path_block_radius_px,
        )
        debug = {
            "input_count": int(len(representative_points)),
            "kept_count": 0,
            "filtered_near": 0,
            "filtered_initial_start": 0,
            "filtered_visited_path": 0,
            "filtered_cooldown": 0,
            "filtered_blocked": 0,
            "filtered_unreachable": 0,
            "filtered_long_path": 0,
            "filtered_detour": 0,
            "fmm_error": None,
            "min_goal_dist_px": float(min_goal_dist_px),
            "initial_block_radius_px": float(initial_block_radius_px),
            "initial_block_cell": (
                None
                if getattr(self, "frontier_initial_block_cell", None) is None
                else tuple(int(v) for v in self.frontier_initial_block_cell)
            ),
            "cooldown_radius_px": float(cooldown_radius_px),
            "visited_path_block_radius_px": float(visited_path_block_radius_px),
            "max_fmm_dist_px": float(max_fmm_dist_px),
            "max_detour_ratio": float(max_detour_ratio),
            "min_detour_dist_px": float(min_detour_dist_px),
            "visited_goal_count": int(len(getattr(self, "frontier_visited_goals", []) or [])),
        }

        prelim = []
        for row, col in representative_points:
            row, col = int(row), int(col)
            if row < 0 or col < 0 or row >= frontier_bool.shape[0] or col >= frontier_bool.shape[1]:
                debug["filtered_blocked"] += 1
                continue
            if not bool(frontier_bool[row, col]):
                debug["filtered_blocked"] += 1
                continue
            if self._frontier_goal_in_episode_initial_block(
                (row, col),
                planning_window,
                radius_px=initial_block_radius_px,
            ):
                debug["filtered_initial_start"] += 1
                continue
            if bool(visited_path_block_mask[row, col]):
                debug["filtered_visited_path"] += 1
                continue
            euclidean_dist = pu.get_l2_distance(row, start[0], col, start[1])
            if euclidean_dist < min_goal_dist_px:
                debug["filtered_near"] += 1
                continue
            if self._frontier_goal_in_cooldown(
                (row, col),
                radius_px=cooldown_radius_px,
            ):
                debug["filtered_cooldown"] += 1
                continue
            prelim.append((row, col, float(euclidean_dist)))

        if not prelim:
            self.frontier_reachability_debug = debug
            self._log_frontier_filter_debug(debug)
            return [], debug

        try:
            traversible = self._build_frontier_traversible(
                obs_map,
                start,
                planning_window,
            )
            planner = FMMPlanner(traversible)
            planner.set_goal([int(start[0]) + 1, int(start[1]) + 1])
            fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)
            unreachable_cutoff = np.inf
            if np.any(traversible <= 0):
                unreachable_cutoff = float(np.nanmax(fmm_dist)) - 1e-5
            scored_candidates = []
            for row, col, euclidean_dist in prelim:
                pad_row, pad_col = int(row) + 1, int(col) + 1
                if (
                    pad_row < 0
                    or pad_col < 0
                    or pad_row >= traversible.shape[0]
                    or pad_col >= traversible.shape[1]
                    or traversible[pad_row, pad_col] <= 0
                ):
                    debug["filtered_blocked"] += 1
                    continue
                dist = float(fmm_dist[pad_row, pad_col])
                if not np.isfinite(dist) or dist >= unreachable_cutoff:
                    debug["filtered_unreachable"] += 1
                    continue
                if dist < min_goal_dist_px:
                    debug["filtered_near"] += 1
                    continue
                if dist > max_fmm_dist_px:
                    debug["filtered_long_path"] += 1
                    continue
                detour_ratio = dist / max(float(euclidean_dist), 1.0)
                if dist > min_detour_dist_px and detour_ratio > max_detour_ratio:
                    debug["filtered_detour"] += 1
                    continue
                scored_candidates.append({
                    "point": (int(row), int(col)),
                    "fmm_cost": float(dist),
                    "euclidean_dist": float(euclidean_dist),
                    "detour_ratio": float(detour_ratio),
                })
            scored_candidates = sorted(
                scored_candidates,
                key=lambda item: float(item.get("fmm_cost", np.inf)),
            )
            fmm_values = [float(item["fmm_cost"]) for item in scored_candidates]
            debug["kept_count"] = int(len(scored_candidates))
            if fmm_values:
                debug["fmm_dist_min"] = round(float(np.min(fmm_values)), 3)
                debug["fmm_dist_max"] = round(float(np.max(fmm_values)), 3)
            self.frontier_reachability_debug = debug
            self._log_frontier_filter_debug(debug)
            return scored_candidates, debug
        except Exception as exc:
            debug["fmm_error"] = str(exc)
            scored_candidates = [
                {
                    "point": (int(row), int(col)),
                    "fmm_cost": float(euclidean_dist),
                    "euclidean_dist": float(euclidean_dist),
                    "detour_ratio": 1.0,
                    "fmm_error_fallback": True,
                }
                for row, col, euclidean_dist in prelim
            ]
            scored_candidates = sorted(
                scored_candidates,
                key=lambda item: float(item.get("fmm_cost", np.inf)),
            )
            debug["kept_count"] = int(len(scored_candidates))
            debug["used_euclidean_fallback"] = bool(scored_candidates)
            self.frontier_reachability_debug = debug
            self._log_frontier_filter_debug(debug)
            return scored_candidates, debug

    def _filter_frontier_representatives_by_reachability(
        self,
        representative_points,
        obs_map,
        frontiers,
        start,
        planning_window,
    ):
        scored_candidates, _ = self._score_frontier_representatives_by_fmm(
            representative_points,
            obs_map,
            frontiers,
            start,
            planning_window,
        )
        return [
            tuple(int(v) for v in candidate["point"])
            for candidate in scored_candidates
        ]

    def _get_stg(
        self,
        grid,
        start,
        goal,
        planning_window,
        use_large_goal_dilation=False,
        goal_dilation_radius=None,
        traversible_constraint=None,
        goal_region_constraint=None,
    ):
        """Get short-term goal"""

        [gx1, gx2, gy1, gy2] = planning_window
        self._last_fmm_replan = False
        self._last_frontier_goal_reached = False

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        ########################
        # selem = skimage.morphology.disk(3)
        # grid = skimage.morphology.dilation(grid, selem)
        # selem = skimage.morphology.disk(1)
        # grid = skimage.morphology.erosion(grid, selem)
        ########################

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2],
            self.selem) != True
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2]
                    [x1:x2, y1:y2] == 1] = 0
        

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1
        constraint = None
        if traversible_constraint is not None:
            candidate = np.asarray(traversible_constraint, dtype=bool)
            if candidate.shape == traversible.shape:
                constraint = candidate
                traversible &= constraint
                traversible[int(start[0] - x1), int(start[1] - y1)] = 1

        frontier_goal_points = None
        frontier_reps = set(getattr(self, "frontier_representative_points", []) or [])
        if frontier_reps:
            goal_points = np.argwhere(np.asarray(goal) > 0)
            matching_goal_points = [
                (int(row), int(col))
                for row, col in goal_points
                if (int(row), int(col)) in frontier_reps
            ]
            if matching_goal_points:
                frontier_goal_points = np.asarray(matching_goal_points, dtype=np.int32)
            else:
                if (
                    len(goal_points) == 1
                    and self.global_goal is None
                    and self.temp_goal is None
                ):
                    frontier_goal_points = np.asarray(goal_points, dtype=np.int32)
                else:
                    self.frontier_active_goal = None

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)
        visited = add_boundary(self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2], value=0)
        padded_constraint = (
            add_boundary(constraint.astype(np.uint8), value=0).astype(bool)
            if constraint is not None
            else None
        )
        padded_goal_region_constraint = None
        if goal_region_constraint is not None:
            candidate = np.asarray(goal_region_constraint, dtype=bool)
            unpadded_goal_shape = (goal.shape[0] - 2, goal.shape[1] - 2)
            if candidate.shape == unpadded_goal_shape:
                padded_goal_region_constraint = add_boundary(
                    candidate.astype(np.uint8), value=0
                ).astype(bool)

        if goal_dilation_radius is not None:
            selem = skimage.morphology.disk(max(0, int(goal_dilation_radius)))
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        elif (
            self.global_goal is not None
            or self.temp_goal is not None
            or use_large_goal_dilation
        ):
            selem = skimage.morphology.disk(10)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        elif self.stuck_goal is not None:
            selem = skimage.morphology.disk(1)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        else:
            selem = skimage.morphology.disk(3)
            goal = skimage.morphology.binary_dilation(
                goal, selem) != True
        goal = 1 - goal * 1.
        if padded_goal_region_constraint is not None:
            goal *= padded_goal_region_constraint.astype(goal.dtype)
            goal *= np.asarray(traversible, dtype=goal.dtype)
        if padded_constraint is not None:
            goal *= padded_constraint.astype(goal.dtype)
        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        self._last_multi_fmm_cache_hit = False
        if np.count_nonzero(np.asarray(goal) > 0) == 0:
            self._last_fmm_replan = True
            self.frontier_active_goal = None
            return (start[0], start[1]), False

        planner = FMMPlanner(traversible)
        use_exact_cache = bool(
            int(getattr(self.args, "multi_fmm_exact_cache", 0))
        )
        cache = getattr(self, "_multi_fmm_exact_cache", None)
        cache_hit = bool(
            use_exact_cache
            and isinstance(cache, dict)
            and np.array_equal(cache.get("traversible"), traversible)
            and np.array_equal(cache.get("goal"), goal)
        )
        if cache_hit:
            planner.fmm_dist = cache["fmm_dist"]
            self._multi_fmm_cache_hits += 1
            self._last_multi_fmm_cache_hit = True
        else:
            if use_exact_cache:
                self._multi_fmm_cache_misses += 1
            if not planner.set_multi_goal(goal):
                self._last_fmm_replan = True
                self.frontier_active_goal = None
                self._multi_fmm_exact_cache = None
                return (start[0], start[1]), False
            if use_exact_cache:
                self._multi_fmm_exact_cache = {
                    "traversible": np.array(traversible, copy=True),
                    "goal": np.array(goal, copy=True),
                    # get_short_term_goal pads its distance map and never
                    # mutates this array, so sharing it on cache hits is safe.
                    "fmm_dist": np.array(planner.fmm_dist, copy=True),
                }
        if frontier_goal_points is not None and len(frontier_goal_points) > 0:
            try:
                fmm_dist = np.asarray(planner.fmm_dist)
                cursor = np.asarray([
                    int(round(state[0])),
                    int(round(state[1])),
                ], dtype=np.int32)
                cursor[0] = int(np.clip(cursor[0], 0, fmm_dist.shape[0] - 1))
                cursor[1] = int(np.clip(cursor[1], 0, fmm_dist.shape[1] - 1))
                for _ in range(int(fmm_dist.shape[0] + fmm_dist.shape[1])):
                    row0 = max(0, int(cursor[0]) - 1)
                    row1 = min(fmm_dist.shape[0], int(cursor[0]) + 2)
                    col0 = max(0, int(cursor[1]) - 1)
                    col1 = min(fmm_dist.shape[1], int(cursor[1]) + 2)
                    local = fmm_dist[row0:row1, col0:col1]
                    if local.size == 0:
                        break
                    local_index = np.unravel_index(int(np.argmin(local)), local.shape)
                    next_cursor = np.asarray(
                        [row0 + local_index[0], col0 + local_index[1]],
                        dtype=np.int32,
                    )
                    if (
                        np.array_equal(next_cursor, cursor)
                        or fmm_dist[next_cursor[0], next_cursor[1]]
                        >= fmm_dist[cursor[0], cursor[1]] - 1e-6
                    ):
                        break
                    cursor = next_cursor
                    if fmm_dist[cursor[0], cursor[1]] <= 0.5:
                        break

                padded_points = frontier_goal_points + 1
                best_index = int(np.argmin(
                    np.linalg.norm(
                        padded_points.astype(np.float32) - cursor.astype(np.float32),
                        axis=1,
                    )
                ))
                self.frontier_active_goal = tuple(
                    int(v)
                    for v in frontier_goal_points[best_index]
                )
            except Exception:
                self.frontier_active_goal = None


        if self.global_goal is not None:
            st_dis = pu.get_l2_dis_point_map(state, goal) * self.args.map_resolution
            fmm_dist = planner.fmm_dist * self.args.map_resolution 
            dis = fmm_dist[start[0]+1, start[1]+1]
            if st_dis < 100 and dis/st_dis > 2:
                return (0, 0), True

        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)
        self._last_fmm_replan = bool(replan)
        if replan:
            stg_x, stg_y, _, stop = planner.get_short_term_goal(state, 2)
        # if self.rank == 0 and (self.timestep >= 50):
        #     np.set_printoptions(precision=3, suppress=True, linewidth=300)
        #     print(f"{self.timestep}    \n   {subset}")

        ########################
        # stg_x, stg_y, _, stop = planner.get_short_term_goal_v1(state, 48)
        # top_k_goal = np.zeros((48, 2))
        # top_k_goal[:, 0], top_k_goal[:, 1] = stg_x, stg_y
        # stg_goal = pu.ca_short_term_goal(np.rint(1-traversible), state, top_k_goal, 1)
        # if stg_goal is not None:
        #     stg_x = stg_goal[0]
        #     stg_y = stg_goal[1]
        # else:
        #     stg_x = top_k_goal[0, 0]
        #     stg_y = top_k_goal[0, 1]
        #     print(f"{self.rank}, {self.timestep}, error happens ! fall back to original")
        ########################

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        if bool(stop) and self.frontier_active_goal is not None:
            self._last_frontier_goal_reached = True
            self._mark_frontier_goal_visited(self.frontier_active_goal)

        return (stg_x, stg_y), stop

    def get_frontier_map(self, planner_inputs):
        """Function responsible for computing frontiers in the input map

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'obs_map' (ndarray): (M, M) map of obstacle locations
                    'exp_map' (ndarray): (M, M) map of explored locations

        Returns:
            frontier_map (ndarray): (M, M) binary map of frontier locations
        """
        args = self.args
        self.frontier_fmm_candidates = []
        self.frontier_fmm_selection_debug = {}

        obs_map = np.rint(planner_inputs["obs_map"]).astype(np.uint8)
        exp_map = np.rint(planner_inputs["exp_map"]).astype(np.uint8)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_inputs["pose_pred"]
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [
            int(r * 100.0 / args.map_resolution - gx1),
            int(c * 100.0 / args.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, exp_map.shape)
        obs_map, exp_map = self._force_footprint_and_visited_free(
            obs_map,
            exp_map,
            planning_window=planning_window,
            start=start,
        )

        is_explored = exp_map > 0
        is_occupied = obs_map > 0
        explored_free = is_explored & (~is_occupied)

        unseen = (~is_explored).astype(np.uint8)
        unseen_dilated = skimage.morphology.binary_dilation(
            unseen,
            skimage.morphology.disk(1),
        )

        # MinAV-style frontier seed: explored free cells adjacent to unseen space.
        raw_frontier_edge = explored_free & unseen_dilated

        obstacle_inflation_px = float(
            getattr(args, "frontier_safe_obstacle_inflation_cells", 4.0)
        )
        frontier_band_radius_px = float(
            getattr(args, "frontier_band_radius_cells", 4.0)
        )
        frontier_band_radius = max(0, int(math.ceil(frontier_band_radius_px)))

        free_region = (~is_occupied).astype(np.uint8)
        dist_to_obstacle = cv2.distanceTransform(
            free_region,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        safe_free = explored_free & (dist_to_obstacle > obstacle_inflation_px)

        filtered_frontier_edge = raw_frontier_edge.copy()
        removed_small_strip = 0
        if int(raw_frontier_edge.sum()) > 0:
            labeled = skimage.measure.label(
                raw_frontier_edge.astype(np.uint8),
                connectivity=2,
            )
            adaptive_small_component_max = 12
            adaptive_min_safe_support = 8
            support_radius_px = max(1, frontier_band_radius)
            support_kernel = skimage.morphology.disk(support_radius_px)
            component_slices = ndimage.find_objects(labeled)

            for cid, comp_slice in enumerate(component_slices, start=1):
                if comp_slice is None:
                    continue
                row_slice, col_slice = comp_slice
                comp_mask = labeled[row_slice, col_slice] == cid
                comp_size = int(comp_mask.sum())
                if comp_size == 0:
                    continue

                comp_clearance = dist_to_obstacle[row_slice, col_slice][comp_mask]
                row_start = max(row_slice.start - support_radius_px, 0)
                row_stop = min(row_slice.stop + support_radius_px, labeled.shape[0])
                col_start = max(col_slice.start - support_radius_px, 0)
                col_stop = min(col_slice.stop + support_radius_px, labeled.shape[1])
                comp_roi_mask = labeled[row_start:row_stop, col_start:col_stop] == cid
                comp_safe_support = (
                    skimage.morphology.binary_dilation(comp_roi_mask, support_kernel)
                    & safe_free[row_start:row_stop, col_start:col_stop]
                )
                safe_support_size = int(comp_safe_support.sum())
                if (
                    comp_size <= adaptive_small_component_max
                    and np.percentile(comp_clearance, 75) < obstacle_inflation_px
                    and safe_support_size < adaptive_min_safe_support
                ):
                    filtered_view = filtered_frontier_edge[row_slice, col_slice]
                    filtered_view[comp_mask] = False
                    removed_small_strip += comp_size

        initial_block_mask = self._episode_initial_frontier_block_mask(
            filtered_frontier_edge.shape,
            planning_window,
        )
        removed_initial_block = int(np.count_nonzero(
            filtered_frontier_edge & initial_block_mask
        ))
        if removed_initial_block > 0:
            filtered_frontier_edge[initial_block_mask] = False

        visited_path_block_mask = self._visited_path_frontier_block_mask(
            filtered_frontier_edge.shape,
            planning_window,
        )
        removed_visited_path_block = int(np.count_nonzero(
            filtered_frontier_edge & visited_path_block_mask
        ))
        if removed_visited_path_block > 0:
            filtered_frontier_edge[visited_path_block_mask] = False

        if frontier_band_radius > 0:
            frontier_band = skimage.morphology.binary_dilation(
                filtered_frontier_edge,
                skimage.morphology.disk(frontier_band_radius),
            )
        else:
            frontier_band = filtered_frontier_edge.copy()
        frontier_band = frontier_band & safe_free
        if removed_initial_block > 0:
            frontier_band[initial_block_mask] = False
        if np.any(visited_path_block_mask):
            frontier_band[visited_path_block_mask] = False

        frontiers = frontier_band.astype(np.uint8)

        frontier_vis = np.zeros((frontiers.shape[0], frontiers.shape[1], 3), np.uint8)
        frontier_colors = [
            (255, 80, 80),
            (80, 200, 255),
            (120, 255, 120),
            (255, 180, 60),
            (210, 120, 255),
            (80, 255, 220),
            (255, 120, 170),
            (170, 220, 80),
        ]
        frontier_cluster_dilation_radius = float(
            getattr(args, "frontier_cluster_dilation_radius", 2.0)
        )
        frontier_cluster_min_size = int(
            getattr(args, "frontier_cluster_min_size", 10)
        )
        frontier_min_representatives = max(
            1,
            int(getattr(args, "frontier_min_representatives", 2)),
        )
        frontier_max_representatives = max(
            frontier_min_representatives,
            int(getattr(args, "frontier_max_representatives", 4)),
        )

        clusters = self._cluster_frontiers_with_dilation(
            frontiers,
            dilation_radius=frontier_cluster_dilation_radius,
            min_size=frontier_cluster_min_size,
        )
        clusters = sorted(clusters, key=lambda x: len(x), reverse=True)

        new_frontiers = np.zeros_like(frontiers)
        representative_points = []
        rep_color_by_point = {}
        cluster_debug = []
        self.frontier_active_goal = None
        if removed_initial_block > 0:
            cluster_debug.append({
                "rank": -2,
                "size": int(removed_initial_block),
                "color": (0, 0, 0),
                "representatives": [],
                "representatives_after_agent_mask": [],
                "filtered_source": "episode_initial_block",
                "initial_block_radius_px": round(
                    float(self._frontier_initial_block_radius_px()),
                    3,
                ),
                "initial_block_cell": (
                    None
                    if getattr(self, "frontier_initial_block_cell", None) is None
                    else tuple(int(v) for v in self.frontier_initial_block_cell)
                ),
            })
        if removed_visited_path_block > 0:
            cluster_debug.append({
                "rank": -3,
                "size": int(removed_visited_path_block),
                "color": (0, 0, 0),
                "representatives": [],
                "representatives_after_agent_mask": [],
                "filtered_source": "visited_path_block",
                "visited_path_block_radius_px": round(
                    float(self._frontier_visited_path_block_radius_px()),
                    3,
                ),
            })

        for rank, cluster in enumerate(clusters):
            if not cluster:
                continue
            color = frontier_colors[rank % len(frontier_colors)]
            cluster_arr = np.asarray(cluster, dtype=np.int32)
            rows = cluster_arr[:, 0]
            cols = cluster_arr[:, 1]
            frontier_vis[rows, cols] = color

            representative_count = min(
                frontier_max_representatives,
                max(frontier_min_representatives, int(len(cluster) // 40)),
            )
            reps = self._pick_frontier_representative_points(
                cluster,
                num_points=representative_count,
                occupancy_map=obs_map,
                explored_map=exp_map,
            )

            clean_reps = []
            for row, col in reps:
                row, col = int(row), int(col)
                if row < 0 or col < 0 or row >= new_frontiers.shape[0] or col >= new_frontiers.shape[1]:
                    continue
                new_frontiers[row, col] = 1
                representative_points.append((row, col))
                rep_color_by_point[(row, col)] = color
                clean_reps.append((row, col))

            cluster_debug.append({
                "rank": int(rank),
                "size": int(len(cluster)),
                "color": tuple(int(v) for v in color),
                "representatives": clean_reps,
            })

        all_representative_points = list(representative_points)
        scored_candidates, score_debug = self._score_frontier_representatives_by_fmm(
            representative_points,
            obs_map,
            frontier_band,
            start,
            planning_window,
        )
        selection_score_debug = score_debug
        self.frontier_fmm_candidates = list(scored_candidates)
        selected_by_fmm = bool(getattr(args, "frontier_select_by_fmm_cost", 1))
        fallback_source = None
        if scored_candidates:
            if selected_by_fmm:
                representative_points = [
                    tuple(int(v) for v in scored_candidates[0]["point"])
                ]
            else:
                representative_points = [
                    tuple(int(v) for v in candidate["point"])
                    for candidate in scored_candidates
                ]
        else:
            representative_points = []

        # Handle edge cases where no original representative survives FMM filters.
        if not representative_points:
            def pick_fallback_points(source_mask, max_points=5):
                coords = np.argwhere(np.asarray(source_mask).astype(bool))
                if coords.size == 0:
                    return []
                dists = np.linalg.norm(
                    coords.astype(np.float32)
                    - np.asarray(start, dtype=np.float32).reshape(1, 2),
                    axis=1,
                )
                coords = coords[dists >= 10.0]
                if len(coords) == 0:
                    return []
                if len(coords) > max_points:
                    chosen = np.random.choice(len(coords), size=max_points, replace=False)
                    coords = coords[chosen]
                return [(int(row), int(col)) for row, col in coords]

            fallback_points = []
            fallback_scored = []
            for source_mask, source_name in (
                (frontier_band, "frontier_band"),
                (filtered_frontier_edge, "filtered_frontier_edge"),
                (explored_free, "explored_free"),
            ):
                fallback_source_mask = (
                    np.asarray(source_mask).astype(bool)
                    & (~visited_path_block_mask)
                )
                fallback_points = pick_fallback_points(fallback_source_mask)
                if not fallback_points:
                    continue
                fallback_mask = np.zeros_like(frontiers, dtype=bool)
                for row, col in fallback_points:
                    if 0 <= row < fallback_mask.shape[0] and 0 <= col < fallback_mask.shape[1]:
                        fallback_mask[row, col] = True
                fallback_scored, fallback_debug = self._score_frontier_representatives_by_fmm(
                    fallback_points,
                    obs_map,
                    fallback_mask,
                    start,
                    planning_window,
                )
                if fallback_scored:
                    fallback_source = source_name
                    selection_score_debug = fallback_debug
                    break
            if fallback_scored:
                self.frontier_fmm_candidates = list(fallback_scored)
                if selected_by_fmm:
                    representative_points = [
                        tuple(int(v) for v in fallback_scored[0]["point"])
                    ]
                else:
                    representative_points = [
                        tuple(int(v) for v in candidate["point"])
                        for candidate in fallback_scored
                    ]
                for fallback_point in representative_points:
                    rep_color_by_point.setdefault(fallback_point, (255, 255, 255))
                cluster_debug.append({
                    "rank": -1,
                    "size": int(len(fallback_points)),
                    "color": (255, 255, 255),
                    "representatives": fallback_points,
                    "representatives_after_agent_mask": representative_points,
                    "fallback_source": fallback_source,
                })

        frontiers = np.zeros_like(frontier_band, dtype=bool)
        for row, col in representative_points:
            row, col = int(row), int(col)
            if 0 <= row < frontiers.shape[0] and 0 <= col < frontiers.shape[1]:
                frontiers[row, col] = True

        final_rep_set = set(representative_points)
        for debug_info in cluster_debug:
            debug_info["representatives_after_agent_mask"] = [
                point
                for point in debug_info.get("representatives", [])
                if point in final_rep_set
            ]

        for row, col in all_representative_points:
            color = rep_color_by_point.get((row, col), (255, 255, 255))
            cv2.circle(frontier_vis, (int(col), int(row)), 3, color, -1)
            cv2.circle(frontier_vis, (int(col), int(row)), 5, (255, 255, 255), 1)
        for row, col in representative_points:
            cv2.circle(frontier_vis, (int(col), int(row)), 8, (255, 255, 255), 2)
            cv2.circle(frontier_vis, (int(col), int(row)), 5, (255, 0, 0), -1)
            cv2.circle(frontier_vis, (int(col), int(row)), 2, (0, 0, 0), -1)

        top_candidates = []
        for rank, candidate in enumerate(self.frontier_fmm_candidates[:5], start=1):
            point = candidate.get("point")
            top_candidates.append({
                "rank": int(rank),
                "point": (
                    None
                    if point is None
                    else [int(point[0]), int(point[1])]
                ),
                "fmm_cost": round(float(candidate.get("fmm_cost", np.inf)), 3),
                "euclidean_dist": round(float(candidate.get("euclidean_dist", np.inf)), 3),
                "detour_ratio": round(float(candidate.get("detour_ratio", np.inf)), 3),
            })

        best_point = representative_points[0] if representative_points else None
        best_cost = None
        if self.frontier_fmm_candidates:
            best_cost = float(self.frontier_fmm_candidates[0].get("fmm_cost", np.inf))
        self.frontier_fmm_selection_debug = {
            "raw_edge_cells": int(raw_frontier_edge.sum()),
            "filtered_edge_cells": int(filtered_frontier_edge.sum()),
            "safe_band_cells": int(frontier_band.sum()),
            "removed_small_strip": int(removed_small_strip),
            "removed_initial_block": int(removed_initial_block),
            "removed_visited_path_block": int(removed_visited_path_block),
            "cluster_count": int(len(clusters)),
            "representative_count_before_fmm": int(len(all_representative_points)),
            "valid_fmm_candidate_count": int(len(self.frontier_fmm_candidates)),
            "selected_by_fmm": bool(selected_by_fmm),
            "fallback_source": fallback_source,
            "best_point": (
                None
                if best_point is None
                else [int(best_point[0]), int(best_point[1])]
            ),
            "best_fmm_cost": (
                None
                if best_cost is None or not np.isfinite(best_cost)
                else round(float(best_cost), 3)
            ),
            "top_candidates": top_candidates,
            "score_debug": selection_score_debug,
            "obstacle_inflation_px": float(obstacle_inflation_px),
            "frontier_band_radius_px": float(frontier_band_radius_px),
            "visited_path_block_radius_px": float(
                self._frontier_visited_path_block_radius_px()
            ),
            "cluster_min_size": int(frontier_cluster_min_size),
            "max_representatives_per_cluster": int(frontier_max_representatives),
        }
        if bool(getattr(args, "frontier_debug_log", 1)):
            try:
                print("[FRONTIER_FMM_SELECT] {}".format(json.dumps(
                    self.frontier_fmm_selection_debug,
                    sort_keys=True,
                )))
            except Exception:
                pass

        self.frontier_vis = frontier_vis
        self.frontier_representative_points = representative_points
        self.frontier_cluster_debug = cluster_debug

        return frontiers

    def get_fmm_dists(self, planner_inputs):
        """Function responsible for planning, and identifying reachable locations

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'pred_map'   (ndarray): (N, H, W) map with 0 as floor, 1 - N as categories
                    'map_resolution' (int): size of grid-cell in pred_map

        Returns:
            fmm_dists (ndarray): (N, H, W) map of FMM dists per category
        """
        pred_map = planner_inputs["pred_map"]
        fmm_dists = np.zeros_like(pred_map)
        fmm_dists.fill(np.inf)
        map_resolution = planner_inputs["map_resolution"]
        orig_map_resolution = self.args.map_resolution
        assert orig_map_resolution == map_resolution

        # Setup planner
        traversible = pred_map[0]
        planner = FMMPlanner(traversible)
        # Get FMM dists to each category
        selem = skimage.morphology.disk(
            int(self.object_boundary / 4.0 * 100.0 / self.args.map_resolution)
        )
        for i in range(1, fmm_dists.shape[0]):
            if np.count_nonzero(pred_map[i]) == 0:
                continue
            goal_map = cv2.dilate(pred_map[i], selem)
            planner.set_multi_goal(goal_map)
            fmm_dist = planner.fmm_dist * map_resolution / 100.0
            fmm_dists[i] = fmm_dist

        return fmm_dists

    def check_temp_goal_reachability(self, grid, temp_goal, start, planning_window):
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape
        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(14)
        goal = np.zeros((self.local_w, self.local_h))
        goal[int(temp_goal[0] - gx1), int(temp_goal[1] - gy1)] = 1
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution / 100.0

        navigable_indices = np.argwhere(self.visited[gx1:gx2, gy1:gy2] > 0)
        navigable_point = np.array([0, 0])
        for _ in range(100):
            random_index = np.random.choice(len(navigable_indices))
            navigable_point = navigable_indices[random_index]
            if pu.get_l2_distance(navigable_point[0], start[0], navigable_point[1], start[1]) < 20:
                break

        return fmm_dist[navigable_point[0], navigable_point[1]] < fmm_dist.max()

    # cm
    def compute_temp_goal_distance(self, grid, goal_map, start, planning_window):
        return compute_temp_goal_distance_ieve(
            grid,
            goal_map,
            start,
            planning_window,
            visited=self.visited,
            collision_map=self.collision_map,
            map_resolution=self.args.map_resolution,
            obstacle_selem=self.selem,
            goal_dilation_radius=10,
        )




    def _get_reachability(self, grid, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = (
            0,
            0,
        )
        x2, y2 = grid.shape

        traversible = 1.0 - cv2.dilate(grid[x1:x2, y1:y2], self.selem)
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] > 0] = 1
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        

        # Note: Unlike _get_stg, no boundary is added here since we only want
        # to determine reachability.
        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(3)
        goal = cv2.dilate(goal, selem)
        planner.set_multi_goal(goal)
        fmm_dist = planner.fmm_dist * self.args.map_resolution / 100.0

        reachability = fmm_dist < fmm_dist.max()

        return reachability.astype(np.float32), fmm_dist.astype(np.float32)

    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        depth = obs[:, :, 3:4]

        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth=depth[:, :, 0], use_seg=use_seg)

        sem_seg_pred = self._get_sem_pred(
            rgb.astype(np.uint8), use_seg=use_seg)

        # sem_seg_pred =self._get_sem_pred_internimg(
        #     rgb.astype(np.uint8), depth[:, :, 0], use_seg=use_seg
        # )
        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)

        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        # lj dataset
        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        

        # up or down floor check
        # agent_state = self._env.sim.get_agent_state(0).position
        # if abs(self.start_height - agent_state[1]) > 0.2:
        #     depth[:, :] = 0.
        #     print(f"Rank: {self.rank}, timestep: {self.timestep},  Agent Height Change !")

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth

    def _get_sem_pred_internimg(self, rgb, depth, use_seg=True, pred_bbox=False):
        coco_categories_mapping_reduce = {
            56: 0,  # chair
            57: 1,  # couch
            58: 2,  # potted plant
            59: 3,  # bed
            61: 4,  # toilet
            62: 5,  # tv
        }
        conf = 0.3
        from constants import coco_categories_mapping
        if pred_bbox:
            bbox, mask = inference_detector(self.internimg, rgb)
            final = []
            for key in coco_categories_mapping_reduce:
                tt = bbox[key]
                select_index = np.where(tt[:, 4] > conf)
                select_tt = tt[select_index, :]
                select_tt = select_tt.squeeze(axis=0)
                for index in range(select_tt.shape[0]):
                    final.append([coco_categories_mapping_reduce[key], select_tt[index, 4], select_tt[index, :4]])
            return final
        else:
            if use_seg:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                bbox, mask = inference_detector(self.internimg, rgb)
                bbox_vis, mask_vis = bbox.copy(), mask.copy()
                final = []
                for key in coco_categories_mapping:
                    select_index = np.where(bbox[key][:, 4] > conf)
                    tt_mask = mask[key]
                    tt_mask = [tt_mask[i] for i in range(len(tt_mask)) if bbox[key][i, 4] > conf]
                    semantic_pred[:, :, coco_categories_mapping[key]] = np.logical_or.reduce(tt_mask, axis=0)
                    if key in coco_categories_mapping_reduce:
                        tt_box = bbox[key][select_index, :]
                        tt_box = tt_box.squeeze(axis=0)
                        for index in range(tt_box.shape[0]):
                            final.append([coco_categories_mapping[key], tt_box[index, 4], tt_box[index, :4]])

                self.pred_box = final
                # self.rgb_vis = self.internimg.show_result(
                #                     rgb,
                #                     (bbox_vis, mask_vis),
                #                     score_thr=0.3,
                #                     show=False,
                #                     bbox_color='coco',
                #                     text_color=(200, 200, 200),
                #                     mask_color='coco',
                #                     out_file=None
                #                 )[:, :, ::-1]
                # self.rgb_vis = rgb[:, :, ::-1]
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)

                semantic_pred = semantic_pred.astype(np.float32)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
            return semantic_pred

    def _get_sem_pred(self, rgb, depth=None, use_seg=True, pred_bbox=False):
        if pred_bbox:
            semantic_pred, self.rgb_vis, self.pred_box = self.sem_pred.get_prediction(rgb)
            self.pred_instances = getattr(self.sem_pred, "last_instances", [])
            return self.pred_box
        else:
            if use_seg:
                semantic_pred, self.rgb_vis, self.pred_box = self.sem_pred.get_prediction(rgb)
                self.pred_instances = getattr(self.sem_pred, "last_instances", [])
                semantic_pred = semantic_pred.astype(np.float32)
                if depth is not None:
                    normalize_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    self.rgb_vis = cv2.cvtColor(normalize_depth, cv2.COLOR_GRAY2BGR)
            else:
                semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
                self.rgb_vis = rgb[:, :, ::-1]
                self.pred_box = []
                self.pred_instances = []
            return semantic_pred

    def _visualize_mask(self, mask, points, sign):
        # mask should be 0, 1
        dump_dir = "{}/dump/{}/".format(self.dump_location,
                                        self.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        mask = (mask * 255).astype(int)
        if sign:
            fn = '{}/episodes/thread_{}/eps_{}/positive-{}-{}.png'.format(
                    dump_dir, self.rank, self.episode_no,
                     self.timestep, str(int(points )))
            cv2.imwrite(fn, mask)
        else:
            fn = '{}/episodes/thread_{}/eps_{}/negative-{}-{}.png'.format(
                    dump_dir, self.rank, self.episode_no,
                     self.timestep, str(int(points)))
            cv2.imwrite(fn, mask)

    def render_3d_semantic_occupancy(self, occupancy_map, file_name):
        try:
            from mayavi import mlab
        except ImportError as exc:
            raise ImportError(
                "Mayavi is required only for render_3d_semantic_occupancy. "
                "Install mayavi or use the default/non-Mayavi visualization path."
            ) from exc

        mlab.options.offscreen = True

        color_palette = np.array([
            [1.0, 1.0, 1.0],
            [0.6, 0.6, 0.6],
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.748199999999999]]) * 255
        
        color_palette = np.concatenate([color_palette.astype(int), np.ones([color_palette.shape[0], 1]) * 255], axis=1)

        xx, yy, zz, color = None, None, None, None
        for i in range(occupancy_map.shape[0]):
            x, y, z = np.where(occupancy_map[i, ...] == 1)
            if xx is not None:
                xx = np.concatenate((x, xx))
                yy = np.concatenate((y, yy))
                zz = np.concatenate((z, zz))
                color = np.concatenate((np.ones_like(x) * (i+1), color))
            else:
                xx = x
                yy = y
                zz = z
                color = np.ones_like(x) * (i+1)

        fig = mlab.figure(bgcolor=(1, 1, 1), size=(480, 480))
        plot = mlab.points3d(xx, yy, zz, color, mode="cube", figure=fig, scale_factor=1.0)

        # magic to modify lookup table
        plot.module_manager.scalar_lut_manager.lut._vtk_obj.SetTableRange(0, color_palette.shape[0])
        plot.module_manager.scalar_lut_manager.lut.number_of_colors = color_palette.shape[0]
        plot.module_manager.scalar_lut_manager.lut.table = color_palette

        mlab.view(azimuth=230, distance=200)
        mlab.savefig(filename=file_name)
        mlab.points3d

    def render_pyplot(self, occupancy_map, file_name):
        color_palette = [
            [1.0, 1.0, 1.0],
            [0.6, 0.6, 0.6],
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.748199999999999]]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # for i in range(occupancy_map.shape[0]):
        ax.voxels(occupancy_map[0, ...], facecolors=color_palette[1])
        ax.voxels(occupancy_map[1, ...], facecolors=color_palette[3])
        ax.voxels(occupancy_map[2, ...], facecolors=color_palette[4])
        ax.voxels(occupancy_map[3, ...], facecolors=color_palette[5])
        ax.voxels(occupancy_map[4, ...], facecolors=color_palette[6])
        ax.voxels(occupancy_map[5, ...], facecolors=color_palette[7])
        ax.voxels(occupancy_map[6, ...], facecolors=color_palette[8])
        ax.set_axis_off()
        plt.savefig(file_name)

    def render_voxels(self, voxels, output_path):
        # Get the coordinates of occupied voxels
        voxels = voxels.transpose(0,1,3,2)

        color_palette = [
            [0.95, 0.95, 0.95],
            [0.96, 0.36, 0.26],
            [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
            [0.9400000000000001, 0.7818, 0.66],
            [0.8882000000000001, 0.9400000000000001, 0.66],
            [0.66, 0.9400000000000001, 0.8518000000000001],
            [0.7117999999999999, 0.66, 0.9400000000000001],
            [0.9218, 0.66, 0.9400000000000001],
            [0.9400000000000001, 0.66, 0.748199999999999]]
        for i in range(voxels.shape[0]):
            if i == 0:
                occupied_voxels = np.argwhere(voxels[i] > 0.5)
                # colors = np.ones([len(occupied_voxels), 3]) * color_palette[i]
                max_height = np.max(occupied_voxels[:, 1])
                min_height = np.min(occupied_voxels[:, 1])
                scale_factor =  (occupied_voxels[:, 1] - min_height)/(max_height - min_height)
                base_color = np.tile([[0.6, 0.6, 0.6]], (len(occupied_voxels), 1))
                colors = base_color - [0.6, 0.6, 0.6] * scale_factor[:, None]
            elif i < 7:
                temp_occupied_voxels = np.argwhere(voxels[i] > 0.5)
                if temp_occupied_voxels.shape[0] != 0:
                    occupied_voxels = np.concatenate((occupied_voxels, temp_occupied_voxels))
                    colors = np.concatenate((colors, np.ones([len(temp_occupied_voxels), 3]) * color_palette[i]))



        # occupied_voxels = np.argwhere(voxels > 0.5)

        # Create a point cloud from the occupied voxel coordinates
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(occupied_voxels)
        # pcd.colors = o3d.utility.Vector3dVector(np.tile([[0.9, 0.9, 0.9]], (len(occupied_voxels), 1)))  # Set color for occupied voxels
        pcd.colors = o3d.utility.Vector3dVector(colors)  # Set color for occupied voxels

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, 0.8)

        # Create a visualizer and set the render options
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # Set the window size to match the desired image resolution
        vis.add_geometry(voxel_grid)

        ctr = vis.get_view_control()
        ctr.translate(0, 200)  # Move the camera up
        ctr.rotate(0, 200)  # Rotate the camera pitch
        # ctr.set_constant_z_near(20)

        img = vis.capture_screen_float_buffer(True)
        # Save the rendered images
        img = np.asarray(img) * 255  # Convert the image to a numpy array
        resize_img = cv2.resize(img,(960, 540),
                                interpolation=cv2.INTER_NEAREST)

        saved_image = np.zeros((540, 960+405, 3), np.uint8)
        tmp = cv2.resize(self.rgb_vis, (405, 540),interpolation=cv2.INTER_NEAREST)
        saved_image[:, :405, :] = tmp
        saved_image[:, 405:960+405, :] = resize_img
        

        plt.imsave(output_path,
                        saved_image,
                        dpi=1)

        vis.destroy_window()

    def _render_3d(self, inputs):

        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['sem_map_pred_3d']
        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}-3D.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            # self.render_pyplot(map_pred, fn)
            # self.render_3d_semantic_occupancy(map_pred, fn)
            self.render_voxels(map_pred, fn)
        


    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        goal = inputs['goal']
        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        # add a check with collision map
        map_pred[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 1

        sem_map += 5

        no_cat_mask = sem_map == 11
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        # vis_mask = self.visited[gx1:gx2, gy1:gy2] == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3

        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4

        color_pal = [int(x * 255.) for x in color_palette]
        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(np.asarray(sem_map_vis)).copy()

        if self.frontier_vis is not None:
            frontier_overlay_src = np.asarray(self.frontier_vis).copy()
            frontier_overlay_src = self._draw_frontier_active_goal(frontier_overlay_src)
            frontier_overlay = np.flipud(frontier_overlay_src).copy()
            if frontier_overlay.shape[:2] == sem_map_vis.shape[:2]:
                frontier_mask = np.any(frontier_overlay > 0, axis=2)
                sem_map_vis[frontier_mask] = frontier_overlay[frontier_mask]

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        tmp = cv2.resize(self.rgb_vis, (360, 480),
                                interpolation=cv2.INTER_NEAREST)
        tmp = self._overlay_observation_instance_masks(tmp)

        tmp_goal = cv2.resize(self.instance_imagegoal, (480, 480),
                                interpolation=cv2.INTER_NEAREST)
        tmp_goal = cv2.cvtColor(tmp_goal, cv2.COLOR_RGB2BGR)
        self.vis_image[50:50+480, 15:495] = tmp_goal
        self.vis_image[50:50+480, 510:870] = tmp
        self.vis_image[50:530, 885:1365] = sem_map_vis
        self.vis_image[50:530, 495:510] = [255,255,255]

        self._draw_minav_map_memory_overlay(inputs, map_pred.shape)

        minav_mode_label = None
        if bool(getattr(args, "use_minav_confirmation_localization", 0)):
            minav_mode_label = self._minav_display_mode_label(inputs)
        if minav_mode_label is not None:
            vu.update_text(self.vis_image, minav_mode_label)
        elif self.global_goal is not None:
            vu.update_text(self.vis_image, 'Exploitation')
        elif self.temp_goal is not None:
            vu.update_text(self.vis_image, 'Verificaiton')
        else:
            vu.update_text(self.vis_image, 'Exploration')

        # match0, match1 = self.local_feature_match_lightglue(re_key2=True)
        # try:
        #     for i in range(len(match0)):
        #         pt1 = match1[i]*480/512 + np.array([15, 50])
        #         pt2 = match0[i]*480/640 + np.array([510, 50])
        #         pt1 = pt1.astype(int)
        #         pt2 = pt2.astype(int)
        #         cv2.circle(self.vis_image, pt1, 3, (0, 255, 0), -1)  # Draw circle on image1
        #         cv2.circle(self.vis_image, pt2, 3, (0, 255, 0), -1)  # Draw circle on image2
        #         random_color = np.random.randint(0, 256, (3), np.uint8).tolist()
        #         random_color = tuple(random_color)
        #         cv2.line(self.vis_image, pt1, pt2, random_color, 2)
        # except:
        #     print("h")

        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(885, 50))
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            
            # try:
            #     import habitat
            #     top_down_map = habitat.utils.visualizations.maps.colorize_topdown_map(self.info['top_down_map']['map']) 
            #     top_fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Top-{}.png'.format(
            #     dump_dir, self.rank, self.episode_no,
            #     self.rank, self.episode_no, self.timestep)
            #     cv2.imwrite(top_fn, top_down_map)
            # except:
            #     import habitat
            #     from typing import cast
            #     from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
            #     top_down_map = habitat.utils.visualizations.maps.get_topdown_map_from_sim(
            #         cast("HabitatSim", self.habitat_env.sim), map_resolution=1024
            #     )
            #     top_fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Top-{}.png'.format(
            #     dump_dir, self.rank, self.episode_no,
            #     self.rank, self.episode_no, self.timestep)
            #     cv2.imwrite(top_fn, top_down_map)

            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            
            cv2.imwrite(fn, self.vis_image)

            minav_vis_dir = inputs.get("minav_visualization_dir")
            if minav_vis_dir:
                try:
                    grid_dir = os.path.join(str(minav_vis_dir), "2d_grid")
                    os.makedirs(grid_dir, exist_ok=True)
                    mode = str(inputs.get("target_mode", "unknown"))
                    source = str(inputs.get("goal_source", "unknown"))
                    mode = "".join(c if c.isalnum() or c in "._-" else "_" for c in mode)
                    source = "".join(c if c.isalnum() or c in "._-" else "_" for c in source)
                    wait_suffix = ""
                    if self._is_minav_localization_wait(inputs):
                        wait_suffix = "_wait_{:04d}".format(
                            int(getattr(self, "_minav_wait_debug_counter", 0)) + 1
                        )
                    grid_fn = os.path.join(
                        grid_dir,
                        "{:06d}{}_thread_{}_ep_{}_mode_{}_source_{}.png".format(
                            int(self.timestep),
                            wait_suffix,
                            self.rank,
                            self.episode_no,
                            mode,
                            source,
                        ),
                    )
                    cv2.imwrite(grid_fn, self.vis_image)
                except Exception as exc:
                    logging.warning("Failed to save MinAV 2D grid visualization: %s", exc)

    def _overlay_observation_instance_masks(self, observation_bgr):
        args = self.args
        if not bool(getattr(args, "vis_observation_masks", 1)):
            return observation_bgr

        instances = list(getattr(self, "pred_instances", []) or [])
        if not instances:
            return observation_bgr

        threshold = float(getattr(args, "vis_observation_mask_threshold", -1.0))
        if threshold < 0:
            threshold = float(getattr(args, "sem_pred_prob_thr", 0.5))
        target_only = bool(getattr(args, "vis_observation_target_only", 0))

        canvas = observation_bgr.copy()
        overlay = canvas.copy()
        out_h, out_w = canvas.shape[:2]
        draw_items = []
        drawn = 0
        for idx, inst in enumerate(instances):
            try:
                confidence = float(inst.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            if confidence < threshold:
                continue
            if target_only:
                try:
                    if int(inst.get("category_id", -1)) != int(self.gt_goal_idx):
                        continue
                except Exception:
                    continue

            mask = inst.get("binary_mask")
            if mask is None:
                continue
            mask_np = np.asarray(mask, dtype=np.uint8)
            if mask_np.ndim != 2 or int(mask_np.sum()) <= 0:
                continue

            mask_vis = cv2.resize(
                mask_np,
                (out_w, out_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if int(mask_vis.sum()) <= 0:
                continue

            color = self._observation_mask_color(inst, idx)
            overlay[mask_vis] = color
            contours, _ = cv2.findContours(
                mask_vis.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            draw_items.append((contours, inst, mask_vis, confidence, color))
            drawn += 1

        if drawn == 0:
            return observation_bgr

        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
        for contours, inst, mask_vis, confidence, color in draw_items:
            cv2.drawContours(canvas, contours, -1, color, 2)
            self._draw_observation_mask_label(canvas, inst, mask_vis, confidence, color)
        return canvas

    def _save_target_instance_debug(
        self,
        info,
        all_instances,
        target_instances,
        semantic_target_mask,
    ):
        if not bool(getattr(self.args, "target_instance_debug_dump", 0)):
            return
        try:
            debug_dir = self._target_instance_debug_dir()
            os.makedirs(debug_dir, exist_ok=True)
            step = int(info.get("time", getattr(self, "timestep", 0)) or 0)
            dump_idx = int(getattr(self, "_target_instance_debug_frame_idx", 0))
            self._target_instance_debug_frame_idx = dump_idx + 1
            prefix = "{:06d}_{:06d}".format(step, dump_idx)

            raw_rgb = getattr(self, "raw_obs", None)
            if raw_rgb is None:
                return
            raw_rgb = np.asarray(raw_rgb)
            if raw_rgb.ndim != 3 or raw_rgb.shape[2] < 3:
                return
            raw_rgb = raw_rgb[:, :, :3].astype(np.uint8, copy=False)

            all_png = "{}_all_pred_instances.png".format(prefix)
            target_png = "{}_target_instances.png".format(prefix)
            semantic_png = "{}_semantic_target_mask_raw.png".format(prefix)
            json_name = "{}_instances.json".format(prefix)

            all_overlay = self._make_instance_debug_overlay(raw_rgb, all_instances)
            target_overlay = self._make_instance_debug_overlay(raw_rgb, target_instances)
            cv2.imwrite(
                os.path.join(debug_dir, all_png),
                all_overlay,
            )
            cv2.imwrite(
                os.path.join(debug_dir, target_png),
                target_overlay,
            )

            semantic_vis = self._make_semantic_target_mask_debug_image(
                semantic_target_mask,
                raw_rgb.shape[:2],
            )
            cv2.imwrite(os.path.join(debug_dir, semantic_png), semantic_vis)

            record = {
                "rank": int(self.rank),
                "episode_no": int(getattr(self, "episode_no", -1)),
                "timestep": step,
                "dump_index": dump_idx,
                "goal_name": getattr(self, "goal_name", None),
                "gt_goal_idx": self._debug_json_value(getattr(self, "gt_goal_idx", None)),
                "goal_cat_id": self._debug_json_value(info.get("goal_cat_id")),
                "files": {
                    "all_pred_instances": all_png,
                    "target_instances": target_png,
                    "semantic_target_mask_raw": semantic_png,
                },
                "counts": {
                    "all_pred_instances": int(len(all_instances or [])),
                    "target_instances": int(len(target_instances or [])),
                    "semantic_target_mask_pixels": self._mask_pixel_count(
                        semantic_target_mask
                    ),
                },
                "all_pred_instances": [
                    self._instance_debug_metadata(inst, idx)
                    for idx, inst in enumerate(all_instances or [])
                ],
                "target_instances": [
                    self._instance_debug_metadata(inst, idx)
                    for idx, inst in enumerate(target_instances or [])
                ],
                "semantic_target_mask_raw": self._mask_debug_metadata(
                    semantic_target_mask
                ),
            }
            with open(os.path.join(debug_dir, json_name), "w") as f:
                json.dump(record, f, indent=2, sort_keys=True)
        except Exception as exc:
            logging.warning("Failed to save target instance debug dump: %s", exc)

    def _target_instance_debug_dir(self):
        dump_dir = "{}/dump/{}/".format(self.args.dump_location, self.args.exp_name)
        return os.path.join(
            dump_dir,
            "episodes",
            "thread_{}".format(self.rank),
            "eps_{}".format(self.episode_no),
            "target_instance_debug",
        )

    def _make_instance_debug_overlay(self, raw_rgb, instances):
        canvas = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        if not instances:
            return canvas

        overlay = canvas.copy()
        draw_items = []
        out_h, out_w = canvas.shape[:2]
        for idx, inst in enumerate(instances):
            if not isinstance(inst, dict):
                continue
            mask = inst.get("binary_mask")
            if mask is None:
                continue
            mask_np = np.asarray(mask, dtype=np.uint8)
            if mask_np.ndim != 2 or int(mask_np.sum()) <= 0:
                continue
            mask_vis = cv2.resize(
                mask_np,
                (out_w, out_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if int(mask_vis.sum()) <= 0:
                continue
            try:
                confidence = float(inst.get("confidence", 0.0) or 0.0)
            except Exception:
                confidence = 0.0
            color = self._observation_mask_color(inst, idx)
            overlay[mask_vis] = color
            contours, _ = cv2.findContours(
                mask_vis.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            draw_items.append((contours, inst, mask_vis, confidence, color))

        if not draw_items:
            return canvas
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
        for contours, inst, mask_vis, confidence, color in draw_items:
            cv2.drawContours(canvas, contours, -1, color, 2)
            self._draw_observation_mask_label(canvas, inst, mask_vis, confidence, color)
        return canvas

    def _make_semantic_target_mask_debug_image(self, semantic_target_mask, shape_hw):
        out_h, out_w = int(shape_hw[0]), int(shape_hw[1])
        vis = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        if semantic_target_mask is None:
            return vis
        mask = np.asarray(semantic_target_mask, dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            return vis
        mask_vis = cv2.resize(
            mask,
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        vis[mask_vis] = (255, 255, 255)
        return vis

    @staticmethod
    def _debug_json_value(value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [
                Instance_Exp_Env_Agent._debug_json_value(v)
                for v in value
            ]
        return value

    @staticmethod
    def _mask_pixel_count(mask):
        if mask is None:
            return 0
        try:
            return int(np.count_nonzero(np.asarray(mask)))
        except Exception:
            return 0

    def _mask_debug_metadata(self, mask):
        if mask is None:
            return {
                "present": False,
                "pixel_count": 0,
                "shape": None,
                "bbox": None,
            }
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        bbox = None
        if mask_np.ndim == 2:
            ys, xs = np.where(mask_np > 0)
            if xs.size > 0 and ys.size > 0:
                bbox = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ]
        return {
            "present": True,
            "pixel_count": int(np.count_nonzero(mask_np)),
            "shape": list(mask_np.shape),
            "bbox": bbox,
        }

    def _instance_debug_metadata(self, inst, idx):
        if not isinstance(inst, dict):
            return {"index": int(idx), "valid": False}
        mask = inst.get("binary_mask")
        meta = {
            "index": int(idx),
            "valid": True,
            "category_id": self._debug_json_value(inst.get("category_id")),
            "category_name": inst.get("category_name"),
            "category": inst.get("category"),
            "confidence": self._debug_json_value(inst.get("confidence")),
            "pixel_count": self._debug_json_value(inst.get("pixel_count")),
            "bbox": self._debug_json_value(inst.get("bbox")),
            "source": inst.get("source"),
            "is_target_category": False,
            "mask": self._mask_debug_metadata(mask),
        }
        try:
            meta["is_target_category"] = (
                int(inst.get("category_id", -1)) == int(self.gt_goal_idx)
            )
        except Exception:
            meta["is_target_category"] = False
        return meta

    def _observation_mask_color(self, inst, idx):
        palette = [
            (80, 180, 255),
            (80, 220, 120),
            (230, 130, 230),
            (255, 190, 70),
            (90, 220, 220),
            (230, 120, 120),
            (170, 150, 255),
            (120, 210, 180),
        ]
        try:
            key = int(inst.get("category_id", idx))
        except Exception:
            key = idx
        return palette[key % len(palette)]

    def _draw_observation_mask_label(self, canvas, inst, mask_vis, confidence, color):
        ys, xs = np.where(mask_vis)
        if len(xs) == 0 or len(ys) == 0:
            return

        category = str(inst.get("category_name") or inst.get("category_id") or "obj")
        text = "{} {:.2f}".format(category, confidence)
        x0 = int(xs.min())
        y0 = max(16, int(ys.min()) - 6)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x1 = min(canvas.shape[1] - 1, x0 + tw + 6)
        y1 = min(canvas.shape[0] - 1, y0 + baseline + 4)
        cv2.rectangle(canvas, (x0, y0 - th - 4), (x1, y1), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            text,
            (x0 + 3, y0),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


    def _draw_minav_map_memory_overlay(self, inputs, map_shape):
        overlay = inputs.get("minav_map_memory_overlay")
        if not overlay:
            return
        local_cell = overlay.get("local_cell")
        if local_cell is None:
            return
        try:
            row, col = [int(round(float(v))) for v in local_cell[:2]]
            h, w = int(map_shape[0]), int(map_shape[1])
            if h <= 0 or w <= 0:
                return
            if row < 0 or row >= h or col < 0 or col >= w:
                return
            origin_x, origin_y = 885, 50
            canvas = 480

            def cell_to_canvas(cell):
                rr, cc = [int(round(float(v))) for v in cell[:2]]
                rr = int(np.clip(rr, 0, h - 1))
                cc = int(np.clip(cc, 0, w - 1))
                return (
                    int(origin_x + (float(cc) + 0.5) * canvas / float(w)),
                    int(origin_y + (float(h - 1 - rr) + 0.5) * canvas / float(h)),
                )

            debug = overlay.get("debug") or {}
            sample_cells = debug.get("confirmed_cluster_sample_cells") or []
            for sample_cell in sample_cells:
                spx, spy = cell_to_canvas(sample_cell)
                cv2.circle(self.vis_image, (spx, spy), 2, (180, 0, 180), -1)

            debug_points = (
                ("detected_target", (255, 255, 0), cv2.MARKER_TILTED_CROSS),
                ("confirmed_target", (0, 200, 0), cv2.MARKER_CROSS),
                ("navigation_goal", (0, 255, 255), cv2.MARKER_DIAMOND),
                ("navigation_raw_target", (255, 0, 255), cv2.MARKER_STAR),
                ("buffer_goal", (0, 165, 255), cv2.MARKER_SQUARE),
            )
            for point_key, point_color, marker_type in debug_points:
                point_debug = debug.get(point_key) or {}
                point_cell = point_debug.get("local_cell")
                if point_cell is None:
                    continue
                dpx, dpy = cell_to_canvas(point_cell)
                cv2.drawMarker(
                    self.vis_image,
                    (dpx, dpy),
                    point_color,
                    markerType=marker_type,
                    markerSize=18,
                    thickness=2,
                )

            px, py = cell_to_canvas((row, col))
            confirmed = bool(overlay.get("confirmed", False))
            score = float(overlay.get("score", 0.0) or 0.0)
            threshold = float(overlay.get("threshold", 0.0) or 0.0)
            color = (0, 200, 0) if confirmed else (0, 0, 255)
            cv2.circle(self.vis_image, (px, py), 8, color, -1)
            cv2.circle(self.vis_image, (px, py), 12, (255, 255, 255), 2)
            cv2.drawMarker(
                self.vis_image,
                (px, py),
                (0, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=24,
                thickness=2,
            )
            label = "TC {:.2f}".format(score)
            if threshold > 0:
                label = "{} / {:.0f}".format(label, threshold)
            tx = min(max(px + 14, origin_x + 4), origin_x + canvas - 150)
            ty = min(max(py - 8, origin_y + 18), origin_y + canvas - 8)
            cv2.putText(
                self.vis_image,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                self.vis_image,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        except Exception as exc:
            logging.debug("Failed to draw MinAV map-memory overlay: %s", exc)
