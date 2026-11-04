# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from collections import deque, defaultdict
from dataclasses import asdict
import copy
import os
import sys
from pathlib import Path
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ["OMP_NUM_THREADS"] = "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import logging
import time
import json
import re
import gym
import torch.nn as nn
import torch
import numpy as np
import torch
from torchvision import transforms
from utils.model import Classifier1
from model import RL_Policy, Semantic_Mapping
from utils.storage import GlobalRolloutStorage
from five_goal_benchmark_20261026.runtime.multi_env import make_multi_vec_envs
from multi_iin_v0.runtime.multi_arguments import get_args
from multi_iin_v0.runtime.semantic_instance_filter import (
    attach_instance_filter_to_manager,
    install_instance_semantic_filter,
)
from multi_iin_v0.runtime.localization_runtime import install_fast_localization
from multi_iin_v0.runtime.runtime_timing import RuntimeTimingRecorder
from five_goal_benchmark_20261026.runtime.result_organizer import (
    CompactResultOrganizer,
    ensure_terminal_match_source,
)
from multi_iin_v0.runtime.frontier_information_gain import (
    resolve_target_local_step,
)
from multi_iin_v0.runtime.gaussian_capacity_manager import (
    GaussianCapacityManager,
    install_minav_full_snapshot_policy,
)
from multi_iin_v0.runtime.minav_strategy_guards import install_minav_strategy_guards
from five_goal_benchmark_20261026.runtime.target2_map_prevalidation import (
    install_target2_map_prevalidation,
)
from multi_iin_v0.runtime.gaussian_semantic_labels import (
    GaussianSemanticInstanceTrainer,
)
from multi_iin_v0.runtime.gaussian_quality_refinement import (
    GaussianQualityRefinementManager,
)
from multi_iin_v0.runtime.render_matching_acceleration import (
    install_render_matching_acceleration,
)
from ieve_minav import MinAVConfirmationLocalizationManager
from gaussian_mapping.phase1_gaussmi import Phase1GaussMIController
from five_goal_benchmark_20261026.runtime.instrumentation import install_capture
install_capture()
import envs.utils.depth_utils as du
import algo
from numpy import save


EVAL_METRIC_RE = re.compile(
    r"InstanceImageNav succ/spl/dtg:\s*"
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)\((\d+)\)"
)


def parse_eval_resume_log(log_path):
    resume = {
        "count": 0,
        "success_sum": 0.0,
        "spl_sum": 0.0,
        "dist_sum": 0.0,
    }
    if not log_path:
        return resume
    if not os.path.exists(log_path):
        raise FileNotFoundError("eval resume log not found: {}".format(log_path))

    with open(log_path, "r", errors="ignore") as f:
        text = f.read()
    matches = list(EVAL_METRIC_RE.finditer(text))
    if not matches:
        raise ValueError(
            "no InstanceImageNav metric line found in {}".format(log_path)
        )

    m = matches[-1]
    success_rate = float(m.group(1))
    spl = float(m.group(2))
    dist = float(m.group(3))
    count = int(m.group(4))
    resume["count"] = count
    resume["success_sum"] = float(round(success_rate * count))
    resume["spl_sum"] = spl * count
    resume["dist_sum"] = dist * count
    return resume


def collect_eval_totals(episode_success, episode_spl, episode_dist, resume):
    success_sum = resume["success_sum"]
    spl_sum = resume["spl_sum"]
    dist_sum = resume["dist_sum"]
    count = resume["count"]

    for e in range(len(episode_success)):
        success_sum += float(np.sum(episode_success[e]))
        spl_sum += float(np.sum(episode_spl[e]))
        dist_sum += float(np.sum(episode_dist[e]))
        count += len(episode_spl[e])

    return count, success_sum, spl_sum, dist_sum


def format_eval_metric(prefix, total_count, success_sum, spl_sum, dist_sum):
    return "{} {:.5f}/{:.5f}/{:.5f}({:.0f}),".format(
        prefix,
        success_sum / total_count,
        spl_sum / total_count,
        dist_sum / total_count,
        total_count,
    )


def resolve_project_path(path):
    if os.path.isabs(path):
        return path
    return str(REPO_ROOT / path)


def build_3dgs_camera_params(args):
    width = int(args.env_frame_width)
    height = int(args.env_frame_height)
    hfov = np.deg2rad(float(args.hfov))
    fx = width / (2.0 * np.tan(hfov / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    vfov = 2.0 * np.arctan(height / (2.0 * fy))
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "fovx": hfov,
        "fovy": vfov,
    }


def depth_to_3dgs_meters(raw_depth, min_depth, max_depth):
    depth = np.asarray(raw_depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    if finite.size > 0 and float(np.nanmax(finite)) <= 1.5:
        depth = float(min_depth) + depth * (float(max_depth) - float(min_depth))
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.01] = 0.0
    depth[depth >= float(max_depth) * 0.999] = 0.0
    return depth.astype(np.float32)


def normalize_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def synchronize_cuda_for_timing():
    """Drain queued CUDA work so stage wall times are attributed correctly."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ieve_full_pose_from_gt(info, episode_origin, map_center_m):
    pose = info.get("ieve_gt_pose")
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if pose.size < 3 or not np.all(np.isfinite(pose[:3])):
        return None
    origin = np.asarray(episode_origin, dtype=np.float32).reshape(-1)
    if origin.size < 3 or not np.all(np.isfinite(origin[:3])):
        return None
    return np.asarray(
        [
            float(map_center_m + pose[0] - origin[0]),
            float(map_center_m + pose[1] - origin[1]),
            normalize_angle_deg(np.rad2deg(float(pose[2]))),
        ],
        dtype=np.float32,
    )


def make_3dgs_slam_mapper(args, gs_config, camera_params, dump_dir, env_idx,
                          episode_idx, info=None, direct_output_parent=False):
    from gaussian_mapping.mapper.slam_mapper import SLAMMapper

    info = info or {}
    scene_name = os.path.basename(str(info.get("scene_name", "unknown")))
    scene_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene_name) or "unknown"
    episode_no = int(info.get("episode_no", episode_idx))
    multi_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(info.get("multi_episode_id", "")),
    )
    target_index = int(info.get("multi_target_index", 0) or 0)
    episode_dir = "episode_{:05d}_{}_{}_target{}".format(
        episode_no,
        scene_name,
        multi_id or "multi",
        target_index + 1,
    )
    if direct_output_parent:
        output_dir = os.path.join(dump_dir, episode_dir)
    else:
        output_dir = os.path.join(
            dump_dir,
            args.gs_output_dir,
            "thread_{}".format(env_idx),
            episode_dir,
        )
    mapper_config = copy.deepcopy(gs_config)
    capacity_manager = GaussianCapacityManager.from_args(args, output_dir)
    if capacity_manager.config.enabled:
        mapper_config.setdefault("Training", {})["max_gaussians"] = int(
            capacity_manager.config.hard_cap
        )
    mapper = SLAMMapper(
        config=mapper_config,
        output_dir=output_dir,
        camera_params=camera_params,
        enable_uaemi=False,
    )
    mapper.start_slam()
    capacity_manager.configure_mapper(mapper)
    mapper._multi_iin_capacity_manager = capacity_manager
    label_trainer = GaussianSemanticInstanceTrainer.from_args(args, output_dir)
    label_trainer.configure_mapper(mapper)
    mapper._multi_iin_label_trainer = label_trainer
    quality_manager = GaussianQualityRefinementManager.from_args(args, output_dir)
    quality_manager.configure_mapper(mapper)
    mapper._multi_iin_quality_refinement_manager = quality_manager
    return mapper


def resolve_3dgs_output_parent(args, default_dump_dir, env_idx, target_managers, info):
    if not bool(getattr(args, "use_minav_confirmation_localization", 0)):
        return default_dump_dir, False
    if not bool(getattr(args, "minav_save_stage_visualizations", 1)):
        return default_dump_dir, False
    if env_idx >= len(target_managers):
        return default_dump_dir, False
    manager = target_managers[env_idx]
    if manager is None or not hasattr(manager, "ensure_visualization_episode"):
        return default_dump_dir, False
    try:
        visualization_dir = manager.ensure_visualization_episode(info or {})
    except Exception as exc:
        logging.exception(
            "Failed to resolve MinAV visualization dir for 3DGS env %s: %s",
            env_idx,
            exc,
        )
        return default_dump_dir, False
    if not visualization_dir:
        return default_dump_dir, False
    return visualization_dir, True


def update_3dgs_mappers(
    args,
    slam_mappers,
    infos,
    step,
    wait_env,
    finished,
    force_train_envs=None,
    target_masks=None,
    semantic_instances=None,
    stage_labels=None,
    skip_train_envs=None,
):
    mapping_only_elapsed = [None for _ in slam_mappers]
    if not args.enable_3dgs:
        return mapping_only_elapsed
    force_train_envs = force_train_envs or [False for _ in slam_mappers]
    target_masks = target_masks or [None for _ in slam_mappers]
    semantic_instances = semantic_instances or [[] for _ in slam_mappers]
    stage_labels = stage_labels or ["ieve_online" for _ in slam_mappers]
    skip_train_envs = skip_train_envs or [False for _ in slam_mappers]

    def apply_capacity_manager(mapper, env_idx, stage, timing):
        capacity_manager = getattr(
            mapper, "_multi_iin_capacity_manager", None
        )
        if capacity_manager is None:
            return
        try:
            capacity_manager.maybe_prune(
                mapper,
                env_step=int(step),
                stage=stage,
                timing=timing,
            )
        except Exception as exc:
            logging.exception(
                "Gaussian capacity management failed env %s timing %s: %s",
                env_idx,
                timing,
                exc,
            )

    def update_interval_for_stage(stage_label):
        if int(getattr(args, "gs_save_full_ply_each_frame", 0)) > 0:
            return 1
        stage = str(stage_label or "")
        if stage == "minav_exploration":
            return max(1, int(getattr(args, "gs_exploration_update_interval", 1)))
        return max(1, int(args.gs_update_interval))

    due_envs = [
        bool(force_train_envs[env_idx])
        or step % update_interval_for_stage(stage_labels[env_idx]) == 0
        for env_idx in range(len(slam_mappers))
    ]
    if not any(due_envs):
        return mapping_only_elapsed

    for env_idx, mapper in enumerate(slam_mappers):
        if mapper is None or wait_env[env_idx] or finished[env_idx]:
            continue
        force_train = bool(force_train_envs[env_idx])
        if bool(skip_train_envs[env_idx]) and not force_train:
            continue
        interval = update_interval_for_stage(stage_labels[env_idx])
        if step % interval != 0 and not force_train:
            continue
        info = infos[env_idx]
        rgb = info.get("raw_rgb")
        raw_depth = info.get("raw_depth")
        position = info.get("camera_position", info.get("agent_position"))
        rotation = info.get("agent_rotation_xyzw")
        if rgb is None or raw_depth is None or position is None or rotation is None:
            logging.warning("3DGS skipped env %s step %s: missing RGBD/pose info", env_idx, step)
            continue

        try:
            depth_m = depth_to_3dgs_meters(raw_depth, args.min_depth, args.max_depth)
            apply_capacity_manager(
                mapper, env_idx, stage_labels[env_idx], "before_update"
            )
            synchronize_cuda_for_timing()
            mapping_started = time.perf_counter()
            mapper.incremental_train_3dgs({
                "rgb": np.asarray(rgb, dtype=np.uint8),
                "depth": depth_m,
                "position": np.asarray(position, dtype=np.float32),
                "rotation": np.asarray(rotation, dtype=np.float32),
                "target_mask": target_masks[env_idx],
                "training_context": {
                    "stage": stage_labels[env_idx],
                    "step": int(step),
                    "has_target": target_masks[env_idx] is not None,
                    "allow_current_frame_training": True,
                    "forced": force_train,
                },
            })
            label_trainer = getattr(mapper, "_multi_iin_label_trainer", None)
            if label_trainer is not None:
                label_trainer.observe(
                    mapper,
                    semantic_instances[env_idx],
                    depth_m,
                    env_step=int(step),
                    stage=stage_labels[env_idx],
                )
            synchronize_cuda_for_timing()
            mapping_only_elapsed[env_idx] = time.perf_counter() - mapping_started
            apply_capacity_manager(
                mapper, env_idx, stage_labels[env_idx], "after_update"
            )
            if int(getattr(args, "gs_save_full_ply_each_frame", 0)) > 0:
                try:
                    mapper.save_full_ply_frame(
                        env_step=int(step),
                        stage=stage_labels[env_idx],
                        output_subdir="3dgs_frames",
                        highlight_confirmed=bool(
                            getattr(args, "gs_highlight_confirmed_target", 1)
                        ),
                    )
                except Exception as save_exc:
                    logging.exception(
                        "3DGS per-frame PLY save failed for env %s step %s: %s",
                        env_idx,
                        step,
                        save_exc,
                    )
            if (
                int(args.gs_save_interval) > 0
                and step > 0
                and step % int(args.gs_save_interval) == 0
            ):
                mapper.save_slam_results(save_checkpoint=True)
            debug_interval = (
                int(args.gs_save_interval)
                if int(args.gs_save_interval) > 0
                else interval
            )
            if args.gs_render_debug and step > 0 and step % debug_interval == 0:
                result_dir = os.path.join(mapper.output_dir, "debug_renders")
                mapper.render_and_compute_psnr(np.asarray(rgb, dtype=np.uint8), step, result_dir)
        except Exception as exc:
            logging.exception("3DGS update failed for env %s step %s: %s", env_idx, step, exc)
    return mapping_only_elapsed


def main():
    args = get_args()
    if bool(args.cuda) and torch.cuda.is_available():
        torch.cuda.set_device(int(getattr(args, "sem_gpu_id", 0)))
    if int(args.num_processes) != 1:
        raise ValueError("Multi-IIN v0 real runner currently requires --num_processes 1")
    if str(args.goal_cat_source) != "oracle":
        raise ValueError(
            "Multi-IIN v0 requires --goal_cat_source oracle so the task-defined "
            "target category is not inferred incorrectly from the goal image"
        )
    args.num_eval_episodes = int(args.multi_num_episodes)
    args.eval_skip_episodes = 0
    args.eval_resume_log = ""
    args.eval_habitat_episode_id = ""
    args.eval_episode_start = 0
    args.eval_episode_end = 0
    if str(args.dump_location) == "./tmp/":
        args.dump_location = str(REPO_ROOT / "multi_iin_v0/results/runtime")
    if str(args.minav_results_dir) == "results":
        args.minav_results_dir = str(
            REPO_ROOT / "multi_iin_v0/results/runtime/minav"
        )

    instance_filter_config = install_instance_semantic_filter(args)
    fast_localization_config = install_fast_localization(args)
    render_matching_acceleration = install_render_matching_acceleration(args)

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # config = wandb.config
    # config = {
    #     "thresh_of_matched_points": 1/20,
    #     "success_dis_viewpoints": 0.5,
    #     "goal_selem": 14
    # }
    # wandb.init(project="InstanceImageNav",\
    #     dir="/instance_imagenav/wandb",config=config)    

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    compact_results = CompactResultOrganizer(args, REPO_ROOT)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    if not args.quiet_log:
        print("Dumping at {}".format(log_dir))
        print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device(
        "cuda:{}".format(int(getattr(args, "sem_gpu_id", 0)))
        if args.cuda
        else "cpu"
    )
    eval_resume = parse_eval_resume_log(args.eval_resume_log) if args.eval else {
        "count": 0,
        "success_sum": 0.0,
        "spl_sum": 0.0,
        "dist_sum": 0.0,
    }
    eval_skip_episodes = 0
    if args.eval:
        eval_habitat_episode_id = str(getattr(args, "eval_habitat_episode_id", "") or "").strip()
        eval_episode_start = int(getattr(args, "eval_episode_start", 0) or 0)
        eval_episode_end = int(getattr(args, "eval_episode_end", 0) or 0)
        if eval_habitat_episode_id:
            if eval_episode_start > 0 or eval_episode_end > 0:
                raise ValueError(
                    "eval_habitat_episode_id cannot be combined with "
                    "eval_episode_start/eval_episode_end"
                )
            num_episodes = 1
            eval_skip_episodes = 0
            args.eval_habitat_episode_id = eval_habitat_episode_id
            args.eval_skip_episodes = eval_skip_episodes
            args.num_eval_episodes = num_episodes
            msg = (
                "Using explicit Habitat episode_id: {} "
                "(official Habitat id, episodes={})"
            ).format(eval_habitat_episode_id, num_episodes)
            if not args.quiet_log:
                print(msg)
            logging.info(msg)
        elif eval_episode_start > 0:
            if eval_episode_end > 0 and eval_episode_end < eval_episode_start:
                raise ValueError(
                    "eval_episode_end ({}) must be >= eval_episode_start ({})".format(
                        eval_episode_end, eval_episode_start
                    )
                )
            if eval_episode_end > 0:
                num_episodes = eval_episode_end - eval_episode_start + 1
            eval_skip_episodes = 0
            args.eval_skip_episodes = eval_skip_episodes
            args.num_eval_episodes = num_episodes
            args.eval_dataset_slice_start = eval_episode_start
            args.eval_dataset_slice_end = eval_episode_end
            msg = (
                "Using explicit eval episode range: {}-{} "
                "(internal 1-based dataset slice, episodes={})"
            ).format(
                eval_episode_start,
                eval_episode_end if eval_episode_end > 0 else "open",
                num_episodes,
            )
            if not args.quiet_log:
                print(msg)
            logging.info(msg)
        else:
            eval_skip_episodes = args.eval_skip_episodes
        if eval_skip_episodes < 0:
            eval_skip_episodes = eval_resume["count"]
        if eval_resume["count"] > 0 or eval_skip_episodes > 0:
            assert args.num_processes == 1, \
                "eval resume/skip currently supports num_processes=1 only"
        if eval_resume["count"] >= num_episodes:
            raise ValueError(
                "resume count ({}) is already >= num_eval_episodes ({})".format(
                    eval_resume["count"], num_episodes
                )
            )
        if eval_resume["count"] > 0:
            msg = (
                "Resuming eval from {} completed episodes: "
                "success_sum={:.0f}, spl_sum={:.5f}, dist_sum={:.5f}"
            ).format(
                eval_resume["count"],
                eval_resume["success_sum"],
                eval_resume["spl_sum"],
                eval_resume["dist_sum"],
            )
            if not args.quiet_log:
                print(msg)
            logging.info(msg)

    g_masks = torch.ones(num_scenes).float().to(device)

    best_g_reward = -np.inf

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        eval_metrics = np.zeros((6, 10000)) #geo, euc, suc, spl, dtg, soft_spl
        eval_metrics_id = 0
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    else:
        episode_success = deque(maxlen=1000)
        episode_spl = deque(maxlen=1000)
        episode_dist = deque(maxlen=1000)

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_episode_rewards = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_multi_vec_envs(args)
    if args.eval and eval_skip_episodes > 0:
        msg = "Skipping first {} eval episodes before resuming...".format(
            eval_skip_episodes
        )
        if not args.quiet_log:
            print(msg)
        logging.info(msg)
        for skip_idx in range(eval_skip_episodes):
            envs.reset()
            if (skip_idx + 1) % 50 == 0 or skip_idx + 1 == eval_skip_episodes:
                msg = "Skipped {}/{} eval episodes".format(
                    skip_idx + 1, eval_skip_episodes
                )
                if not args.quiet_log:
                    print(msg)
                logging.info(msg)
    obs, infos = envs.reset()
    infos = list(infos)
    multi_completed_records = []
    multi_active_records = [None for _ in range(num_scenes)]
    multi_boundary_records = []

    torch.set_grad_enabled(False)

    slam_mappers = [None for _ in range(num_scenes)]
    gs_episode_counters = [0 for _ in range(num_scenes)]
    gs_config = None
    gs_camera_params = None
    use_minav_target = bool(getattr(args, "use_minav_confirmation_localization", 0))
    target_managers = [None for _ in range(num_scenes)]
    if use_minav_target:
        for env_idx in range(num_scenes):
            target_managers[env_idx] = MinAVConfirmationLocalizationManager(
                args,
                env_idx,
                None,
                infos[env_idx].get("instance_imagegoal"),
                infos[env_idx].get("goal_name"),
            )
            attach_instance_filter_to_manager(target_managers[env_idx])
            install_minav_strategy_guards(target_managers[env_idx])
            install_target2_map_prevalidation(target_managers[env_idx])
            install_minav_full_snapshot_policy(
                target_managers[env_idx],
                enabled=bool(
                    int(
                        getattr(
                            args,
                            "multi_save_repeated_full_gaussian_snapshots",
                            0,
                        )
                    )
                ),
            )
    if args.enable_3dgs:
        if not torch.cuda.is_available():
            msg = "3DGS requested but CUDA is unavailable; continuing with 3DGS disabled."
            print(msg)
            logging.warning(msg)
            args.enable_3dgs = 0
        else:
            try:
                from gaussian_mapping.utils.config import load_yaml_config

                gs_config = load_yaml_config(resolve_project_path(args.gs_config))
                gs_config.setdefault("Training", {})[
                    "semantic_guided_optimization_enabled"
                ] = bool(
                    getattr(args, "gs_semantic_guided_optimization_enabled", 1)
                )
                gs_camera_params = build_3dgs_camera_params(args)
                max_envs = num_scenes if args.gs_max_envs < 0 else min(num_scenes, args.gs_max_envs)
                for env_idx in range(max_envs):
                    gs_episode_counters[env_idx] += 1
                    gs_output_parent, gs_direct_output_parent = resolve_3dgs_output_parent(
                        args,
                        dump_dir,
                        env_idx,
                        target_managers,
                        infos[env_idx],
                    )
                    slam_mappers[env_idx] = make_3dgs_slam_mapper(
                        args,
                        gs_config,
                        gs_camera_params,
                        gs_output_parent,
                        env_idx,
                        gs_episode_counters[env_idx],
                        infos[env_idx],
                        direct_output_parent=gs_direct_output_parent,
                    )
                    if (
                        use_minav_target
                        and target_managers[env_idx] is not None
                    ):
                        target_managers[env_idx].update_slam_mapper(
                            slam_mappers[env_idx]
                        )
            except Exception as exc:
                logging.exception("Failed to initialize 3DGS mapping: %s", exc)
                print("Failed to initialize 3DGS mapping: {}".format(exc))
                args.enable_3dgs = 0

    if use_minav_target and not args.enable_3dgs:
        msg = (
            "MinAV confirmation/localization requested without enabled 3DGS; "
            "target confirmation will stay system_not_ready instead of falling back to confirmed=True."
        )
        print(msg)
        logging.warning(msg)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)
    minav_full_obstacle_count = np.zeros(
        (num_scenes, full_w, full_h), dtype=np.int32)
    minav_full_free_count = np.zeros(
        (num_scenes, full_w, full_h), dtype=np.int32)
    minav_full_map_pred = np.zeros(
        (num_scenes, full_w, full_h), dtype=np.float32)
    minav_full_exp_pred = np.zeros(
        (num_scenes, full_w, full_h), dtype=np.float32)
    minav_full_mapping_stats = [{} for _ in range(num_scenes)]
    minav_reward_explored_area = np.zeros(num_scenes, dtype=np.float32)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)
    map_center_m = args.map_size_cm / 100.0 / 2.0
    episode_gt_origins = np.zeros((num_scenes, 3), dtype=np.float32)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def clamp_full_pose_np(pose):
        pose = np.asarray(pose, dtype=np.float32).copy()
        pose[0] = float(np.clip(pose[0], 0.0, args.map_size_cm / 100.0))
        pose[1] = float(np.clip(pose[1], 0.0, args.map_size_cm / 100.0))
        pose[2] = normalize_angle_deg(pose[2])
        return pose

    def local_pose_from_full_np(full_pose_np, env_idx):
        return full_pose_np - origins[env_idx].astype(np.float32)

    def set_pose_from_gt_for_env(env_idx):
        full_pose_np = ieve_full_pose_from_gt(
            infos[env_idx],
            episode_gt_origins[env_idx],
            map_center_m,
        )
        if full_pose_np is None:
            full_pose_np = full_pose[env_idx].detach().cpu().numpy()
            full_pose_np = clamp_full_pose_np(full_pose_np)
            local_pose_np = local_pose_from_full_np(full_pose_np, env_idx)
            local_pose[env_idx] = torch.from_numpy(local_pose_np).to(device).float()
            planner_pose_inputs[env_idx, :3] = full_pose_np
            return False
        full_pose_np = clamp_full_pose_np(full_pose_np)
        full_pose[env_idx] = torch.from_numpy(full_pose_np).to(device).float()
        local_pose_np = local_pose_from_full_np(full_pose_np, env_idx)
        local_pose[env_idx] = torch.from_numpy(local_pose_np).to(device).float()
        planner_pose_inputs[env_idx, :3] = full_pose_np
        return True

    def set_all_poses_from_gt():
        for env_idx in range(num_scenes):
            set_pose_from_gt_for_env(env_idx)

    def gt_local_poses_tensor():
        set_all_poses_from_gt()
        return local_pose.clone()

    def init_map_and_pose():
        full_map.fill_(0.)
        minav_full_obstacle_count.fill(0)
        minav_full_free_count.fill(0)
        minav_full_map_pred.fill(0.0)
        minav_full_exp_pred.fill(0.0)
        minav_reward_explored_area.fill(0.0)
        for env_idx in range(num_scenes):
            minav_full_mapping_stats[env_idx] = {}
        full_pose.fill_(0.)
        full_pose[:, :2] = map_center_m

        for e in range(num_scenes):
            gt_pose = infos[e].get("ieve_gt_pose")
            if gt_pose is not None:
                gt_pose = np.asarray(gt_pose, dtype=np.float32).reshape(-1)
                if gt_pose.size >= 3 and np.all(np.isfinite(gt_pose[:3])):
                    episode_gt_origins[e] = gt_pose[:3]
        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            set_pose_from_gt_for_env(e)

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        minav_full_obstacle_count[e].fill(0)
        minav_full_free_count[e].fill(0)
        minav_full_map_pred[e].fill(0.0)
        minav_full_exp_pred[e].fill(0.0)
        minav_reward_explored_area[e] = 0.0
        minav_full_mapping_stats[e] = {}
        full_pose[e].fill_(0.)
        full_pose[e, :2] = map_center_m
        gt_pose = infos[e].get("ieve_gt_pose")
        if gt_pose is not None:
            gt_pose = np.asarray(gt_pose, dtype=np.float32).reshape(-1)
            if gt_pose.size >= 3 and np.all(np.isfinite(gt_pose[:3])):
                episode_gt_origins[e] = gt_pose[:3]

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        set_pose_from_gt_for_env(e)

    def update_intrinsic_rew(e):
        prev_explored_area = float(minav_reward_explored_area[e])
        full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
            local_map[e]
        curr_explored_area = float(full_map[e, 1].sum().item())
        minav_reward_explored_area[e] = curr_explored_area
        intrinsic_rews[e] = curr_explored_area - prev_explored_area
        intrinsic_rews[e] *= (args.map_resolution / 100.)**2  # to m^2

    init_map_and_pose()

    def multi_mapper_audit(mapper):
        if mapper is None:
            return {
                "enabled": False,
                "object_id": None,
                "frame_count": 0,
                "keyframe_count": 0,
                "num_gaussians": 0,
                "capacity": None,
            }
        slam = getattr(mapper, "slam", None)
        keyframes = getattr(slam, "keyframes", {}) if slam is not None else {}
        gaussians = getattr(slam, "gaussians", None) if slam is not None else None
        num_gaussians = 0
        if gaussians is not None and hasattr(gaussians, "get_xyz"):
            try:
                num_gaussians = int(gaussians.get_xyz.shape[0])
            except Exception:
                num_gaussians = 0
        capacity_manager = getattr(
            mapper, "_multi_iin_capacity_manager", None
        )
        capacity_snapshot = (
            None
            if capacity_manager is None
            else capacity_manager.snapshot(mapper)
        )
        return {
            "enabled": True,
            "object_id": int(id(mapper)),
            "frame_count": int(getattr(mapper, "frame_count", 0)),
            "keyframe_count": int(len(keyframes)),
            "num_gaussians": num_gaussians,
            "output_dir": str(getattr(mapper, "output_dir", "")),
            "capacity": capacity_snapshot,
        }

    from five_goal_benchmark_20261026.runtime.instrumentation import audit_extra, write_event, install_capture
    install_capture()

    def multi_runtime_audit(env_idx):
        local = local_map[env_idx].detach().cpu().numpy()
        full = full_map[env_idx].detach().cpu().numpy()
        manager = target_managers[env_idx]
        return {
            "supplementary": audit_extra(slam_mappers[env_idx], manager, infos[env_idx]),
            "full_map_nonzero": int(np.count_nonzero(full)),
            "full_explored_cells": int(np.count_nonzero(full[1] > 0)),
            "full_visited_cells": int(np.count_nonzero(full[3] > 0)),
            "local_map_nonzero": int(np.count_nonzero(local)),
            "minav_obstacle_observations": int(
                minav_full_obstacle_count[env_idx].sum()
            ),
            "minav_free_observations": int(minav_full_free_count[env_idx].sum()),
            "mapper": multi_mapper_audit(slam_mappers[env_idx]),
            "target_manager_conf_history": int(
                len(getattr(manager, "conf_history", [])) if manager is not None else 0
            ),
            "target_manager_frame_window": int(
                len(getattr(manager, "frame_window", [])) if manager is not None else 0
            ),
        }

    def compact_full_map_snapshot(env_idx):
        """Merge the live local window into a CPU copy of the full map."""
        merged = full_map[env_idx].detach().cpu().numpy().copy()
        gx1, gx2, gy1, gy2 = [int(value) for value in lmb[env_idx]]
        merged[:, gx1:gx2, gy1:gy2] = (
            local_map[env_idx].detach().cpu().numpy()
        )
        return merged

    def terminal_manager_summary(env_idx):
        manager = target_managers[env_idx]
        if manager is None:
            return {}
        result = (
            getattr(manager, "last_localization_result", None)
            or getattr(manager, "last_confirm_result", None)
        )
        try:
            summary = manager._result_summary(result)
            selection = getattr(manager, "navigation_goal_selection", {}) or {}
            summary.update(
                {
                    "confirmed_target_position": getattr(manager, "confirmed_target_position", None),
                    "navigation_goal_raw_world": getattr(manager, "navigation_goal_raw_world", None),
                    "navigation_goal_world": getattr(manager, "navigation_goal_world", None),
                    "navigation_effective_stop_radius_m": getattr(manager, "navigation_effective_stop_radius_m", None),
                    "internal_target_center_distance_m": selection.get("target_center_distance_m"),
                    "internal_execution_goal_distance_m": selection.get("execution_goal_distance_m"),
                    "standoff_selection_policy": selection.get("standoff_selection_policy"),
                    "standoff_selected_radius_m": selection.get("stand_off_radius_m", selection.get("target_center_distance_m")),
                }
            )
            standoff = getattr(manager, "_multi_last_standoff_selection", None)
            if isinstance(standoff, dict):
                summary.update(
                    {
                        "standoff_selection_policy": standoff.get(
                            "standoff_selection_policy", "minimum_reachable_radius_first"
                        ),
                        "standoff_selected_radius_m": standoff.get("stand_off_radius_m"),
                        "standoff_fmm_cost": standoff.get("cost"),
                        "standoff_goal_world": standoff.get("goal_world"),
                    }
                )
            return summary
        except Exception:
            return {}

    def save_terminal_gaussian_preview(env_idx, target_index, target_steps):
        mapper = slam_mappers[env_idx]
        rgb = infos[env_idx].get("raw_rgb") if infos[env_idx] is not None else None
        if mapper is None or rgb is None:
            return None, {}
        preview_dir = Path(mapper.output_dir) / "terminal_preview" / "target_{:02d}".format(
            int(target_index) + 1
        )
        try:
            result = mapper.render_and_compute_psnr(
                np.asarray(rgb, dtype=np.uint8), int(target_steps), preview_dir
            )
            if not isinstance(result, dict):
                return None, {}
            comparison = preview_dir / "comparison_{:05d}.png".format(int(target_steps))
            quality = {
                "psnr": float(result.get("psnr", 0.0)),
                "ssim": float(result.get("ssim", 0.0)),
                "keyframe_index": int(result.get("kf_idx", -1)),
            }
            return str(comparison) if comparison.is_file() else None, quality
        except Exception as exc:
            logging.exception(
                "Failed to render terminal compact 3DGS preview env %s: %s", env_idx, exc
            )
            return None, {}

    def reset_multi_mapper(env_idx, current_info, save_existing=True):
        if not args.enable_3dgs:
            return
        old_mapper = slam_mappers[env_idx]
        if old_mapper is not None and save_existing:
            try:
                old_mapper.save_slam_results(save_checkpoint=False)
            except Exception as exc:
                logging.exception(
                    "Failed to save Multi-IIN boundary 3DGS env %s: %s",
                    env_idx,
                    exc,
                )
        slam_mappers[env_idx] = None
        active_limit = (
            num_scenes
            if args.gs_max_envs < 0
            else min(num_scenes, args.gs_max_envs)
        )
        if (
            env_idx < active_limit
            and gs_config is not None
            and gs_camera_params is not None
        ):
            gs_episode_counters[env_idx] += 1
            gs_output_parent, gs_direct_output_parent = resolve_3dgs_output_parent(
                args,
                dump_dir,
                env_idx,
                target_managers,
                current_info,
            )
            slam_mappers[env_idx] = make_3dgs_slam_mapper(
                args,
                gs_config,
                gs_camera_params,
                gs_output_parent,
                env_idx,
                gs_episode_counters[env_idx],
                current_info,
                direct_output_parent=gs_direct_output_parent,
            )

    def get_minav_target_masks():
        if not use_minav_target:
            return [None for _ in range(num_scenes)]
        return [
            infos[env_idx].get("semantic_target_mask_raw")
            if infos[env_idx] is not None
            else None
            for env_idx in range(num_scenes)
        ]

    def get_gaussian_semantic_instances():
        return [
            list((infos[env_idx] or {}).get("multi_semantic_instances", []) or [])
            for env_idx in range(num_scenes)
        ]

    def get_minav_target_map(env_idx):
        info = infos[env_idx]
        if info is None:
            return None
        try:
            goal_cat_id = int(info.get("goal_cat_id", -1))
            channel = 4 + goal_cat_id
            if channel < 4 or channel >= local_map.shape[1]:
                return None
            return local_map[env_idx, channel, :, :].detach().cpu().numpy()
        except Exception:
            return None

    def get_minav_full_map_channel(env_idx, channel):
        try:
            if channel < 0 or channel >= full_map.shape[1]:
                return None
            full = full_map[env_idx, channel, :, :].detach().cpu().numpy().copy()
            gx1, gx2, gy1, gy2 = [int(v) for v in lmb[env_idx]]
            local = local_map[env_idx, channel, :, :].detach().cpu().numpy()
            full[gx1:gx2, gy1:gy2] = local
            return full
        except Exception:
            return None

    def get_minav_full_target_map(env_idx):
        info = infos[env_idx]
        if info is None:
            return None
        try:
            goal_cat_id = int(info.get("goal_cat_id", -1))
            channel = 4 + goal_cat_id
            return get_minav_full_map_channel(env_idx, channel)
        except Exception:
            return None

    def get_minav_mapping_depth_m(env_idx, obs_tensor):
        info = infos[env_idx] if env_idx < len(infos) else {}
        raw_depth = None if info is None else info.get("raw_depth")
        if raw_depth is not None:
            return depth_to_3dgs_meters(raw_depth, args.min_depth, args.max_depth)
        if obs_tensor is None:
            return None
        try:
            depth_cm = obs_tensor[env_idx, 3, :, :].detach().cpu().numpy()
        except Exception:
            return None
        depth_m = np.asarray(depth_cm, dtype=np.float32) / 100.0
        depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m[depth_m <= 0.01] = 0.0
        depth_m[depth_m >= float(args.max_depth) * 0.999] = 0.0
        return depth_m

    def sync_minav_full_grid_to_ieve_maps(env_idx):
        occ = torch.from_numpy(minav_full_map_pred[env_idx]).to(device).float()
        exp = torch.from_numpy(minav_full_exp_pred[env_idx]).to(device).float()
        full_map[env_idx, 0, :, :].copy_(occ)
        full_map[env_idx, 1, :, :].copy_(exp)
        gx1, gx2, gy1, gy2 = [int(v) for v in lmb[env_idx]]
        local_map[env_idx, 0, :, :].copy_(
            full_map[env_idx, 0, gx1:gx2, gy1:gy2])
        local_map[env_idx, 1, :, :].copy_(
            full_map[env_idx, 1, gx1:gx2, gy1:gy2])

    def apply_minav_full_grid_mapping(obs_tensor, step_idx):
        for env_idx in range(num_scenes):
            if wait_env[env_idx] or finished[env_idx]:
                continue
            info = infos[env_idx] if env_idx < len(infos) else {}
            if info is None:
                continue
            depth_m = get_minav_mapping_depth_m(env_idx, obs_tensor)
            stats = du.update_minav_full_grid_from_depth(
                minav_full_obstacle_count[env_idx],
                minav_full_free_count[env_idx],
                minav_full_map_pred[env_idx],
                minav_full_exp_pred[env_idx],
                depth_m,
                info.get("agent_position"),
                info.get("agent_rotation_xyzw"),
                episode_gt_origins[env_idx, :2],
                args.map_size_cm,
                args.map_resolution,
                args.hfov,
                getattr(args, "camera_height", 0.88),
            )
            stats["step"] = int(step_idx)
            minav_full_mapping_stats[env_idx] = stats
            sync_minav_full_grid_to_ieve_maps(env_idx)

    def get_minav_force_flags():
        if not use_minav_target:
            return [False for _ in range(num_scenes)]
        return [
            bool(manager.requires_force_3dgs) if manager is not None else False
            for manager in target_managers
        ]

    def get_minav_stage_labels():
        if not use_minav_target:
            return ["ieve_online" for _ in range(num_scenes)]
        labels = []
        for manager in target_managers:
            if manager is None:
                labels.append("ieve_online")
            else:
                mode_value = manager.mode.value
                if mode_value == "buffer":
                    status = manager.get_status()
                    if (
                        int(status.get("buffer_phase", 1) or 1) >= 2
                        or status.get("buffer_standoff_sweep_active")
                        or status.get("buffer_standoff_sweep_just_completed")
                        or status.get("mr_remap_sweep_active")
                    ):
                        labels.append("minav_buffer_phase2")
                    else:
                        labels.append("minav_buffer_phase1")
                else:
                    labels.append("minav_{}".format(mode_value))
        return labels

    initial_spin_enabled = bool(getattr(args, "minav_initial_spin", 1))
    initial_spin_steps_arg = int(getattr(args, "initial_spin_steps", 0) or 0)
    if initial_spin_steps_arg > 0:
        initial_spin_total_steps = initial_spin_steps_arg
    else:
        turn_angle = float(getattr(args, "turn_angle", 30.0) or 30.0)
        initial_spin_total_steps = max(1, int(np.ceil(360.0 / max(1.0, abs(turn_angle)))))
    initial_spin_remaining = np.zeros(num_scenes, dtype=np.int32)
    initial_spin_episode_no = [None for _ in range(num_scenes)]

    def reset_initial_spin_for_env(env_idx):
        if not initial_spin_enabled:
            initial_spin_remaining[env_idx] = 0
            return
        initial_spin_remaining[env_idx] = int(initial_spin_total_steps)
        info = infos[env_idx] if env_idx < len(infos) else {}
        initial_spin_episode_no[env_idx] = (
            None if info is None else info.get("episode_no")
        )
        log = (
            "[Initial Spin] env={} episode={} start total_steps={} turn_angle={}"
            .format(
                env_idx,
                initial_spin_episode_no[env_idx],
                initial_spin_total_steps,
                float(getattr(args, "turn_angle", 30.0) or 30.0),
            )
        )
        if not args.quiet_log:
            print(log)
        logging.info(log)

    def reset_initial_spin_all():
        for env_idx in range(num_scenes):
            reset_initial_spin_for_env(env_idx)

    def maybe_reset_initial_spin_for_new_episode(env_idx):
        if not initial_spin_enabled:
            return
        info = infos[env_idx] if env_idx < len(infos) else {}
        episode_no = None if info is None else info.get("episode_no")
        if episode_no != initial_spin_episode_no[env_idx]:
            reset_initial_spin_for_env(env_idx)

    def apply_initial_spin_overrides(planner_inputs, step_idx):
        nonlocal frontier_goal_maps_cache, frontier_goal_debug_cache
        nonlocal frontier_last_refresh_step
        nonlocal frontier_last_refresh_movement_steps
        nonlocal frontier_event_refresh_pending, frontier_event_refresh_reasons
        nonlocal frontier_last_explored_counts, frontier_refresh_reason_cache
        if not initial_spin_enabled:
            return
        for env_idx, p_input in enumerate(planner_inputs):
            if p_input.get("wait", False):
                continue
            maybe_reset_initial_spin_for_new_episode(env_idx)
            if initial_spin_remaining[env_idx] <= 0:
                continue
            remaining_before = int(initial_spin_remaining[env_idx])
            p_input["action_override"] = "turn_left"
            p_input["goal_source"] = "INITIAL_360_SPIN"
            p_input["skip_instance_discriminator"] = True
            p_input["found_goal"] = 0
            p_input["initial_spin_remaining_steps"] = remaining_before
            p_input["initial_spin_total_steps"] = int(initial_spin_total_steps)
            initial_spin_remaining[env_idx] = max(0, remaining_before - 1)
            if exploration_policy == "frontier":
                frontier_goal_maps_cache = None
                frontier_goal_debug_cache = None
                frontier_last_refresh_step = None
                frontier_last_refresh_movement_steps = None
                frontier_event_refresh_pending = False
                frontier_event_refresh_reasons = []
                for flags in frontier_event_previous_flags:
                    flags.clear()
                frontier_last_explored_counts = None
                frontier_refresh_reason_cache = []
            if (
                remaining_before == int(initial_spin_total_steps)
                or initial_spin_remaining[env_idx] == 0
            ):
                log = (
                    "[Initial Spin] env={} episode={} step={} remaining_after={}/{}"
                    .format(
                        env_idx,
                        initial_spin_episode_no[env_idx],
                        int(step_idx),
                        int(initial_spin_remaining[env_idx]),
                        int(initial_spin_total_steps),
                    )
                )
                if not args.quiet_log:
                    print(log)
                logging.info(log)

    reset_initial_spin_all()

    def build_minav_payload(env_idx, step_idx):
        info = infos[env_idx]
        target_step = resolve_target_local_step(info, step_idx)
        raw_depth = info.get("raw_depth")
        depth_m = None
        if raw_depth is not None:
            depth_m = depth_to_3dgs_meters(raw_depth, args.min_depth, args.max_depth)
        return {
            "step": int(target_step),
            "runner_global_step": int(step_idx),
            "raw_rgb": info.get("raw_rgb"),
            "raw_depth": raw_depth,
            "depth_m": depth_m,
            "agent_position": info.get("agent_position"),
            "agent_rotation_xyzw": info.get("agent_rotation_xyzw"),
            "pose_pred": planner_pose_inputs[env_idx],
            "map_pred": local_map[env_idx, 0, :, :].cpu().numpy(),
            "exp_pred": local_map[env_idx, 1, :, :].cpu().numpy(),
            "visited_pred": local_map[env_idx, 3, :, :].cpu().numpy(),
            "collision_map": info.get("minav_collision_map_local"),
            "full_map_pred": get_minav_full_map_channel(env_idx, 0),
            "full_exp_pred": get_minav_full_map_channel(env_idx, 1),
            "full_semantic_target_map_pred": get_minav_full_target_map(env_idx),
            "minav_occupancy_frame": "ieve_full_map",
            "minav_full_grid_mapping": minav_full_mapping_stats[env_idx],
            "map_resolution_cm": args.map_resolution,
            "target_instances": info.get("target_instances", []),
            "semantic_target_mask_raw": info.get("semantic_target_mask_raw"),
            "semantic_target_map_pred": get_minav_target_map(env_idx),
            "target_image": info.get("instance_imagegoal"),
            "target_category": info.get("goal_name"),
            "task_name": info.get("goal_name"),
            "scene_name": info.get("scene_name"),
            "episode_no": info.get("episode_no"),
            "habitat_episode_id": info.get("habitat_episode_id"),
            "habitat_episode_dataset_index": info.get(
                "habitat_episode_dataset_index"
            ),
            "goal_cat_id": info.get("goal_cat_id"),
            # Evaluator-only target distances, success/SPL, and episode-start
            # oracle metadata are intentionally excluded.  They remain in
            # ``infos`` for terminal reporting, but the online MinAV/navigation
            # policy never receives them.
            "minav_fmm_stop": info.get("minav_fmm_stop"),
            "minav_phase_goal_reached": info.get("minav_phase_goal_reached"),
            "minav_been_stuck": info.get("ieve_frontier_been_stuck"),
            "minav_target_mode": info.get("minav_target_mode"),
            "minav_goal_source": info.get("minav_goal_source"),
            "minav_found_goal": info.get("minav_found_goal"),
            "minav_navigation_goal_local_cell": info.get(
                "minav_navigation_goal_local_cell"
            ),
            "minav_navigation_goal_cell_reached": info.get(
                "minav_navigation_goal_cell_reached"
            ),
        }

    def apply_minav_goal_overrides(goal_maps, found_goal, step_idx):
        overrides = [None for _ in range(num_scenes)]
        if not use_minav_target:
            return overrides
        for env_idx, manager in enumerate(target_managers):
            if manager is None or wait_env[env_idx] or finished[env_idx]:
                continue
            manager.update_slam_mapper(slam_mappers[env_idx])
            target_step = resolve_target_local_step(infos[env_idx], step_idx)
            # Building the MinAV payload transfers local map channels from the
            # GPU and composes full-resolution occupancy/semantic snapshots.
            # The old path repeated all of that immediately below for the goal
            # override, even though update_frame does not mutate IEVE maps.
            # Reuse the exact same read-only frame snapshot.  The switch is
            # retained for controlled timing comparisons.
            payload = build_minav_payload(env_idx, step_idx)
            status = manager.update_frame(payload)
            reuse_frame_maps = bool(
                int(getattr(args, "multi_minav_reuse_frame_maps", 1))
            )
            context = {
                "step": int(target_step),
                "runner_global_step": int(step_idx),
                "pose_pred": planner_pose_inputs[env_idx],
                "map_pred": (
                    payload.get("map_pred")
                    if reuse_frame_maps
                    else local_map[env_idx, 0, :, :].cpu().numpy()
                ),
                "exp_pred": (
                    payload.get("exp_pred")
                    if reuse_frame_maps
                    else local_map[env_idx, 1, :, :].cpu().numpy()
                ),
                "visited_pred": (
                    payload.get("visited_pred")
                    if reuse_frame_maps
                    else local_map[env_idx, 3, :, :].cpu().numpy()
                ),
                "collision_map": infos[env_idx].get("minav_collision_map_local"),
                "full_map_pred": (
                    payload.get("full_map_pred")
                    if reuse_frame_maps
                    else get_minav_full_map_channel(env_idx, 0)
                ),
                "full_exp_pred": (
                    payload.get("full_exp_pred")
                    if reuse_frame_maps
                    else get_minav_full_map_channel(env_idx, 1)
                ),
                "full_semantic_target_map_pred": (
                    payload.get("full_semantic_target_map_pred")
                    if reuse_frame_maps
                    else get_minav_full_target_map(env_idx)
                ),
                "semantic_target_map_pred": (
                    payload.get("semantic_target_map_pred")
                    if reuse_frame_maps
                    else get_minav_target_map(env_idx)
                ),
                "minav_occupancy_frame": "ieve_full_map",
                "minav_full_grid_mapping": minav_full_mapping_stats[env_idx],
                "current_world_xyz": infos[env_idx].get("agent_position"),
                "agent_position": infos[env_idx].get("agent_position"),
                "current_rotation_xyzw": infos[env_idx].get("agent_rotation_xyzw"),
                "agent_rotation_xyzw": infos[env_idx].get("agent_rotation_xyzw"),
                "local_shape": (local_w, local_h),
                "map_resolution_cm": args.map_resolution,
            }
            override = manager.get_goal_override(context)
            override["status_mode"] = status.get("mode")
            override_goal = override.get("goal_map")
            if override_goal is not None:
                goal_maps[env_idx] = np.asarray(override_goal, dtype=np.float32)
                found_goal[env_idx] = int(bool(override.get("found_goal", False)))
            overrides[env_idx] = override
        return overrides

    def add_minav_planner_fields(env_idx, p_input, minav_overrides):
        if not use_minav_target:
            return
        override = minav_overrides[env_idx] or {}
        p_input["skip_instance_discriminator"] = True
        p_input["hold_action"] = bool(override.get("hold_action", False))
        p_input["action_override"] = override.get("action_override")
        navigation_complete = bool(override.get("navigation_complete", False))
        p_input["navigation_complete"] = navigation_complete
        p_input["target_mode"] = override.get("mode", "exploration")
        p_input["goal_source"] = override.get("goal_source", "EXP_GOAL")
        p_input["minav_visualization_dir"] = override.get("visualization_dir")
        p_input["minav_map_memory_overlay"] = override.get("map_memory_overlay")
        p_input["minav_path_constraint_mask"] = override.get(
            "path_constraint_mask"
        )
        p_input["minav_goal_region_mask"] = override.get("goal_region_mask")
        p_input["minav_goal_dilation_radius"] = override.get(
            "goal_dilation_radius"
        )
        if navigation_complete:
            p_input["force_stop"] = True
            p_input["force_stop_reason"] = "minav_navigation_complete"
        elif bool(override.get("force_stop", False)):
            p_input["force_stop"] = True
            p_input["force_stop_reason"] = override.get(
                "force_stop_reason", "minav_force_stop"
            )
        debug = override.get("debug") or {}
        p_input["minav_navigation_goal_world"] = debug.get("navigation_goal_world")
        p_input["minav_navigation_goal_local_cell"] = debug.get(
            "navigation_goal_local_cell"
        )
        p_input["minav_navigation_goal_adjusted"] = debug.get(
            "navigation_goal_adjusted"
        )
        p_input["minav_navigation_goal_tier"] = debug.get(
            "navigation_goal_tier"
        )
        p_input["minav_effective_stop_radius_m"] = debug.get(
            "navigation_effective_stop_radius_m"
        )
        p_input["minav_navigation_may_terminal_stop"] = debug.get(
            "navigation_may_terminal_stop"
        )
        minav_planner_debug = bool(
            getattr(args, "minav_planner_debug", 0)
            or os.environ.get("MINAV_PLANNER_DEBUG", "").lower()
            in ("1", "true", "yes", "on")
        )
        if minav_planner_debug:
            p_input["minav_planner_debug"] = True
            p_input["agent_position"] = infos[env_idx].get("agent_position")
            p_input["agent_rotation_xyzw"] = infos[env_idx].get(
                "agent_rotation_xyzw"
            )
            p_input["minav_full_grid_mapping"] = minav_full_mapping_stats[env_idx]

    def add_timeout_stop_fields(env_idx, p_input):
        p_input.setdefault("force_stop", False)
        p_input.setdefault("force_stop_reason", None)
        info = infos[env_idx] if env_idx < len(infos) else {}
        target_time = int((info or {}).get("time", 0) or 0)
        budget = max(1, int(args.multi_goal_step_budget))
        if target_time >= budget - 1:
            p_input["force_stop"] = True
            p_input["force_stop_reason"] = "multi_iin_target_budget"

    exploration_policy = str(getattr(args, "exploration_policy", "rl")).lower()
    use_rl_exploration = exploration_policy == "rl"
    if not args.quiet_log:
        print("Exploration policy: {}".format(exploration_policy))
    logging.info("Exploration policy: %s", exploration_policy)
    phase1_gaussmi_controllers = [
        Phase1GaussMIController(args, env_idx)
        for env_idx in range(num_scenes)
    ]
    for env_idx, controller in enumerate(phase1_gaussmi_controllers):
        controller.reset(infos[env_idx])
    phase1_gaussmi_skip_3dgs = [False for _ in range(num_scenes)]
    phase1_gaussmi_force_3dgs = [False for _ in range(num_scenes)]

    def phase1_gaussmi_eligible(env_idx, step_idx):
        if not (
            bool(getattr(args, "phase1_gaussmi_enabled", 1))
            and bool(args.enable_3dgs)
            and exploration_policy == "frontier"
            and slam_mappers[env_idx] is not None
            and not wait_env[env_idx]
            and not finished[env_idx]
            and int(initial_spin_remaining[env_idx]) <= 0
        ):
            return False
        info = infos[env_idx] if env_idx < len(infos) else {}
        if info is None:
            return False
        max_gaussians = int(
            getattr(args, "phase1_gaussmi_max_gaussians", 750000)
        )
        if max_gaussians > 0:
            try:
                gaussians = slam_mappers[env_idx]._gaussians()
                if (
                    gaussians is not None
                    and int(gaussians.get_xyz.shape[0]) > max_gaussians
                ):
                    return False
            except Exception:
                pass
        if (
            bool(info.get("ieve_frontier_been_stuck", False))
            or bool(info.get("ieve_frontier_fmm_replan", False))
            or bool(info.get("ieve_frontier_active_goal_blocked", False))
        ):
            return False
        if use_minav_target:
            manager = target_managers[env_idx]
            if manager is None or manager.mode.value != "exploration":
                return False
            status = manager.get_status()
            if int(status.get("initial_spin_remaining_steps", 0) or 0) > 0:
                return False
            if (
                bool(
                    getattr(
                        args,
                        "minav_secondary_exploration_enabled",
                        getattr(args, "minav_enable_strategy_transition", 0),
                    )
                )
                and not bool(
                    getattr(args, "minav_strategy_use_area_growth_trigger", 0)
                )
                and resolve_target_local_step(infos[env_idx], step_idx)
                >= int(getattr(args, "minav_strategy_phase2_start_step", 500))
            ):
                return False
        return True

    def observe_phase1_gaussmi(step_idx):
        nonlocal phase1_gaussmi_skip_3dgs, phase1_gaussmi_force_3dgs
        skip_flags = []
        force_flags = []
        for env_idx, controller in enumerate(phase1_gaussmi_controllers):
            controller.observe(
                infos[env_idx],
                planner_pose_inputs[env_idx, 2],
                phase1_gaussmi_eligible(env_idx, step_idx),
                step_idx,
                movement_eligible=(
                    exploration_policy == "frontier"
                    and not wait_env[env_idx]
                    and not finished[env_idx]
                ),
            )
            skip, force = controller.training_directive()
            skip_flags.append(bool(skip))
            force_flags.append(bool(force))
        phase1_gaussmi_skip_3dgs = skip_flags
        phase1_gaussmi_force_3dgs = force_flags

    def combined_3dgs_force_flags():
        minav_flags = get_minav_force_flags()
        return [
            bool(minav_flags[e]) or bool(phase1_gaussmi_force_3dgs[e])
            for e in range(num_scenes)
        ]

    def maybe_start_phase1_gaussmi(step_idx):
        for env_idx, controller in enumerate(phase1_gaussmi_controllers):
            controller.maybe_start(
                slam_mappers[env_idx],
                infos[env_idx],
                planner_pose_inputs[env_idx, 2],
                phase1_gaussmi_eligible(env_idx, step_idx),
                step_idx,
                long_term_goal=(
                    frontier_selected_goal_full_cache[env_idx]
                    if exploration_policy == "frontier"
                    else None
                ),
            )

    def apply_phase1_gaussmi_overrides(planner_inputs, step_idx):
        for env_idx, controller in enumerate(phase1_gaussmi_controllers):
            if not phase1_gaussmi_eligible(env_idx, step_idx):
                controller.cancel("phase1_ineligible", step=step_idx)
                continue
            controller.apply_action_override(
                planner_inputs[env_idx],
                planner_pose_inputs[env_idx, 2],
                step_idx,
            )

    frontier_goal_maps_cache = None
    frontier_goal_debug_cache = None
    frontier_last_refresh_step = None
    frontier_last_refresh_movement_steps = None
    frontier_candidate_refresh_interval = max(
        1,
        int(getattr(args, "frontier_candidate_refresh_interval", 5)),
    )
    frontier_goal_hold_steps = max(
        1,
        int(getattr(args, "frontier_goal_hold_steps", 10)),
    )
    frontier_refresh_interval = frontier_candidate_refresh_interval
    frontier_event_refresh_pending = False
    frontier_event_refresh_reasons = []
    frontier_event_previous_flags = [set() for _ in range(num_scenes)]
    frontier_event_observation_count = 0
    frontier_event_edge_count = 0
    frontier_event_suppressed_count = 0
    frontier_refresh_reason_cache = []
    frontier_last_explored_counts = None
    frontier_last_lmb_cache = None
    frontier_candidate_full_points_cache = [[] for _ in range(num_scenes)]
    frontier_selected_goal_full_cache = [None for _ in range(num_scenes)]
    frontier_goal_last_select_step = [None for _ in range(num_scenes)]
    frontier_low_explore_min_age = 5
    frontier_low_explore_gain_px = 10

    def current_frontier_explored_counts():
        return [
            int(torch.count_nonzero(local_map[e, 1, :, :] > 0).item())
            for e in range(num_scenes)
        ]

    def cached_frontier_goal_count():
        if frontier_goal_maps_cache is None:
            return None
        return int(sum(
            np.count_nonzero(np.asarray(goal_map) > 0)
            for goal_map in frontier_goal_maps_cache
            if goal_map is not None
        ))

    def current_frontier_lmb_snapshot():
        return [
            tuple(int(v) for v in lmb[e].tolist())
            for e in range(num_scenes)
        ]

    def frontier_local_points_to_full(env_idx, goal_map):
        if goal_map is None:
            return []
        coords = np.argwhere(np.asarray(goal_map) > 0)
        if coords.size == 0:
            return []
        gx1, _, gy1, _ = [int(v) for v in lmb[env_idx]]
        return [
            (int(gx1 + row), int(gy1 + col))
            for row, col in coords
        ]

    def frontier_full_goal_to_local_map(env_idx, full_goal):
        if full_goal is None:
            return None, None
        try:
            full_row, full_col = int(full_goal[0]), int(full_goal[1])
            gx1, gx2, gy1, gy2 = [int(v) for v in lmb[env_idx]]
            local_row = full_row - gx1
            local_col = full_col - gy1
            if not (0 <= local_row < local_w and 0 <= local_col < local_h):
                return None, None
            goal_map = np.zeros((local_w, local_h), dtype=np.float32)
            goal_map[local_row, local_col] = 1.0
            return goal_map, (int(local_row), int(local_col))
        except Exception:
            return None, None

    def select_frontier_goal_full(env_idx):
        candidates = list(frontier_candidate_full_points_cache[env_idx] or [])
        if not candidates:
            return None
        start_x, start_y = planner_pose_inputs[env_idx, 0], planner_pose_inputs[env_idx, 1]
        start_row = int(start_y * 100.0 / args.map_resolution)
        start_col = int(start_x * 100.0 / args.map_resolution)
        return min(
            candidates,
            key=lambda p: float(
                (int(p[0]) - start_row) ** 2
                + (int(p[1]) - start_col) ** 2
            ),
        )

    def frontier_initial_block_cell():
        center_cell = int(round((args.map_size_cm / 2.0) / args.map_resolution))
        return (center_cell, center_cell)

    def frontier_goal_distance_to_initial(full_goal):
        if full_goal is None:
            return None, None
        init_row, init_col = frontier_initial_block_cell()
        dist_px = float(np.linalg.norm(
            np.asarray([int(full_goal[0]), int(full_goal[1])], dtype=np.float32)
            - np.asarray([init_row, init_col], dtype=np.float32)
        ))
        return dist_px, float(dist_px * args.map_resolution / 100.0)

    def maybe_log_frontier_debug(env_idx, step_idx, debug, selected_full, selected_local):
        if not bool(getattr(args, "frontier_debug_log", 1)):
            return
        if not (
            debug.get("selected_goal_reselect_reasons")
            or debug.get("cache_age") in (None, 0)
        ):
            return
        dist_px, dist_m = frontier_goal_distance_to_initial(selected_full)
        payload = {
            "env": int(env_idx),
            "step": int(step_idx),
            "selected_goal_full": (
                None
                if selected_full is None
                else [int(selected_full[0]), int(selected_full[1])]
            ),
            "selected_goal_local": (
                None
                if selected_local is None
                else [int(selected_local[0]), int(selected_local[1])]
            ),
            "initial_block_cell": [
                int(v) for v in frontier_initial_block_cell()
            ],
            "distance_to_initial_px": (
                None if dist_px is None else round(float(dist_px), 3)
            ),
            "distance_to_initial_m": (
                None if dist_m is None else round(float(dist_m), 3)
            ),
            "initial_block_radius_px": round(
                float(
                    float(getattr(args, "frontier_initial_block_radius_m", 2.3))
                    * 100.0
                    / max(float(args.map_resolution), 1e-6)
                ),
                3,
            ),
            "candidate_count": int(debug.get("candidate_count", 0) or 0),
            "goal_count": int(debug.get("goal_count", 0) or 0),
            "refresh_reasons": list(debug.get("refresh_reasons", []) or []),
            "selected_goal_age": debug.get("selected_goal_age"),
            "selected_goal_reselect_reasons": list(
                debug.get("selected_goal_reselect_reasons", []) or []
            ),
            "cache_age": debug.get("cache_age"),
        }
        print("[FRONTIER_DEBUG] {}".format(json.dumps(payload, sort_keys=True)))

    def frontier_refresh_due(step_idx):
        nonlocal frontier_refresh_reason_cache
        if exploration_policy != "frontier":
            frontier_refresh_reason_cache = []
            return False
        reasons = []
        if (
            frontier_goal_maps_cache is None
            or frontier_last_refresh_step is None
            or any(goal_map is None for goal_map in frontier_goal_maps_cache)
        ):
            reasons.append("cache_empty")
        if frontier_event_refresh_pending:
            reasons.extend(frontier_event_refresh_reasons)

        cache_goal_count = cached_frontier_goal_count()
        if cache_goal_count == 0:
            reasons.append("frontier_goal_count_zero")

        current_lmb = current_frontier_lmb_snapshot()
        if frontier_last_lmb_cache is not None and current_lmb != frontier_last_lmb_cache:
            reasons.append("local_window_changed")

        if frontier_last_refresh_movement_steps is not None:
            movement_ages = [
                int(phase1_gaussmi_controllers[e].movement_steps)
                - int(frontier_last_refresh_movement_steps[e])
                for e in range(num_scenes)
            ]
            max_movement_age = max(movement_ages) if movement_ages else 0
            if max_movement_age >= frontier_refresh_interval:
                reasons.append(
                    "candidate_interval_{}".format(
                        int(frontier_candidate_refresh_interval)
                    )
                )
            if (
                frontier_last_explored_counts is not None
                and max_movement_age >= frontier_low_explore_min_age
            ):
                explored_counts = current_frontier_explored_counts()
                gains = [
                    int(cur) - int(prev)
                    for cur, prev in zip(explored_counts, frontier_last_explored_counts)
                ]
                if gains and max(gains) <= frontier_low_explore_gain_px:
                    reasons.append("low_explored_gain")

        frontier_refresh_reason_cache = sorted(set(reasons))
        return bool(frontier_refresh_reason_cache)

    def collect_frontier_event_refresh(infos_after_step):
        nonlocal frontier_event_refresh_pending, frontier_event_refresh_reasons
        nonlocal frontier_event_observation_count
        nonlocal frontier_event_edge_count, frontier_event_suppressed_count
        if exploration_policy != "frontier":
            return
        reasons = []
        for env_idx, info in enumerate(infos_after_step):
            if info is None:
                continue
            current = set()
            if bool(info.get("ieve_frontier_fmm_replan", False)):
                current.add("fmm_replan")
            if bool(info.get("ieve_frontier_active_goal_reached", False)):
                current.add("active_goal_reached")
            if bool(info.get("ieve_frontier_active_goal_blocked", False)):
                current.add("active_goal_blocked")
            if int(info.get("ieve_frontier_goal_count", 1) or 0) == 0:
                current.add("frontier_goal_count_zero")
            if bool(info.get("ieve_frontier_been_stuck", False)):
                current.add("been_stuck")
            previous = frontier_event_previous_flags[env_idx]
            edges = current - previous
            reasons.extend(sorted(edges))
            frontier_event_observation_count += len(current)
            frontier_event_edge_count += len(edges)
            frontier_event_suppressed_count += len(current & previous)
            frontier_event_previous_flags[env_idx] = current
        if reasons:
            frontier_event_refresh_pending = True
            frontier_event_refresh_reasons = sorted(set(
                list(frontier_event_refresh_reasons) + reasons
            ))
        if frontier_event_observation_count and frontier_event_observation_count % 250 == 0:
            print("[FRONTIER_EVENT_EDGE_V5] {}".format(json.dumps({
                "observations": int(frontier_event_observation_count),
                "edges": int(frontier_event_edge_count),
                "suppressed_repeats": int(frontier_event_suppressed_count),
            }, sort_keys=True)))

    def build_frontier_inputs():
        frontier_inputs = [{} for _ in range(num_scenes)]
        for e, p_input in enumerate(frontier_inputs):
            p_input["obs_map"] = local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_map"] = local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = planner_pose_inputs[e]
        return frontier_inputs

    def build_exploration_goal_maps(global_goals, step_idx, refresh=True):
        nonlocal frontier_goal_maps_cache, frontier_goal_debug_cache
        nonlocal frontier_last_refresh_step
        nonlocal frontier_last_refresh_movement_steps
        nonlocal frontier_event_refresh_pending, frontier_event_refresh_reasons
        nonlocal frontier_last_explored_counts, frontier_last_lmb_cache
        nonlocal frontier_candidate_full_points_cache
        nonlocal frontier_selected_goal_full_cache, frontier_goal_last_select_step
        goal_maps = [np.zeros((local_w, local_h), dtype=np.float32)
                     for _ in range(num_scenes)]
        goal_debug = []

        if exploration_policy == "frontier":
            refresh_reasons = list(frontier_refresh_reason_cache)
            if refresh or frontier_goal_maps_cache is None:
                if not refresh_reasons:
                    refresh_reasons = ["initial_or_forced"]
                explored_counts_before = current_frontier_explored_counts()
                frontier_maps = envs.get_frontier_map(build_frontier_inputs())
                refreshed_goal_maps = []
                refreshed_debug = []
                for e in range(num_scenes):
                    frontier_map = frontier_maps[e]
                    if hasattr(frontier_map, "detach"):
                        frontier_map = frontier_map.detach().cpu().numpy()
                    frontier_map = np.asarray(frontier_map, dtype=np.float32)
                    if frontier_map.shape != (local_w, local_h):
                        corrected = np.zeros((local_w, local_h), dtype=np.float32)
                        h = min(local_w, frontier_map.shape[0])
                        w = min(local_h, frontier_map.shape[1])
                        corrected[:h, :w] = frontier_map[:h, :w]
                        frontier_map = corrected
                    candidate_full_points = frontier_local_points_to_full(e, frontier_map)
                    refreshed_goal_maps.append(frontier_map)
                    frontier_candidate_full_points_cache[e] = candidate_full_points
                    refreshed_debug.append({
                        "policy": "frontier",
                        "refresh_step": int(step_idx),
                        "candidate_refresh_interval": int(frontier_candidate_refresh_interval),
                        "goal_hold_steps": int(frontier_goal_hold_steps),
                        "refresh_reasons": list(refresh_reasons),
                        "candidate_count": int(np.count_nonzero(frontier_map > 0)),
                        "explored_count": int(explored_counts_before[e]),
                    })
                frontier_goal_maps_cache = [
                    goal_map.copy() for goal_map in refreshed_goal_maps
                ]
                frontier_goal_debug_cache = [dict(d) for d in refreshed_debug]
                frontier_last_refresh_step = int(step_idx)
                frontier_last_refresh_movement_steps = [
                    int(controller.movement_steps)
                    for controller in phase1_gaussmi_controllers
                ]
                frontier_last_explored_counts = explored_counts_before
                frontier_last_lmb_cache = current_frontier_lmb_snapshot()
                frontier_event_refresh_pending = False
                frontier_event_refresh_reasons = []

            event_reselect_reasons = {
                "fmm_replan",
                "active_goal_reached",
                "active_goal_blocked",
                "frontier_goal_count_zero",
                "been_stuck",
            }
            goal_debug = []
            for e, debug in enumerate(frontier_goal_debug_cache):
                candidate_map = (
                    np.zeros((local_w, local_h), dtype=np.float32)
                    if frontier_goal_maps_cache is None
                    or frontier_goal_maps_cache[e] is None
                    else np.asarray(frontier_goal_maps_cache[e], dtype=np.float32)
                )
                selected_full = frontier_selected_goal_full_cache[e]
                selected_goal_map, selected_local = frontier_full_goal_to_local_map(
                    e,
                    selected_full,
                )
                select_age = (
                    None
                    if frontier_goal_last_select_step[e] is None
                    else int(phase1_gaussmi_controllers[e].movement_steps)
                    - int(frontier_goal_last_select_step[e])
                )
                reselect_reasons = []
                if selected_full is None:
                    reselect_reasons.append("selected_goal_empty")
                elif selected_goal_map is None:
                    reselect_reasons.append("selected_goal_outside_local_window")
                if select_age is not None and select_age >= frontier_goal_hold_steps:
                    reselect_reasons.append(
                        "goal_hold_{}".format(int(frontier_goal_hold_steps))
                    )
                for reason in refresh_reasons:
                    if reason in event_reselect_reasons:
                        reselect_reasons.append(reason)

                if reselect_reasons:
                    new_selected_full = select_frontier_goal_full(e)
                    frontier_selected_goal_full_cache[e] = new_selected_full
                    frontier_goal_last_select_step[e] = int(
                        phase1_gaussmi_controllers[e].movement_steps
                    )
                    selected_full = new_selected_full
                    selected_goal_map, selected_local = frontier_full_goal_to_local_map(
                        e,
                        selected_full,
                    )
                    select_age = 0

                if selected_goal_map is not None:
                    goal_maps[e] = selected_goal_map
                else:
                    goal_maps[e] = candidate_map.copy()

                current_debug = dict(debug)
                current_debug["step"] = int(step_idx)
                current_debug["cache_age"] = (
                    None
                    if frontier_last_refresh_step is None
                    else int(step_idx) - int(frontier_last_refresh_step)
                )
                current_debug["goal_count"] = int(np.count_nonzero(goal_maps[e] > 0))
                current_debug["selected_goal_full"] = (
                    None
                    if selected_full is None
                    else [int(selected_full[0]), int(selected_full[1])]
                )
                current_debug["selected_goal_local"] = (
                    None
                    if selected_local is None
                    else [int(selected_local[0]), int(selected_local[1])]
                )
                current_debug["selected_goal_age"] = select_age
                current_debug["selected_goal_reselect_reasons"] = sorted(set(reselect_reasons))
                maybe_log_frontier_debug(
                    e,
                    step_idx,
                    current_debug,
                    selected_full,
                    selected_local,
                )
                goal_debug.append(current_debug)
            return goal_maps, [goal_map.copy() for goal_map in goal_maps], goal_debug

        for e in range(num_scenes):
            gx, gy = global_goals[e]
            gx = min(max(int(gx), 0), int(local_w - 1))
            gy = min(max(int(gy), 0), int(local_h - 1))
            goal_maps[e][gx, gy] = 1.0
            goal_debug.append({
                "policy": "rl",
                "step": int(step_idx),
                "goal_cell": [gx, gy],
                "goal_count": 1,
            })
        return goal_maps, [goal_map.copy() for goal_map in goal_maps], goal_debug

    # Global policy observation space
    ngc = 8 + args.num_sem_categories
    es = 2
    g_observation_space = gym.spaces.Box(0, 1,
                                         (ngc,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=0.99,
                                    shape=(2,), dtype=np.float32)

    # Global policy recurrent layer size
    g_hidden_size = args.global_hidden_size


    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         model_type=1,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'num_sem_categories': ngc - 8
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    intrinsic_rews = torch.zeros(num_scenes).to(device)
    extras = torch.zeros(num_scenes, 2)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      es).to(device)

    if args.load != "0":
        if not args.quiet_log:
            print("Loading model {}".format(args.load))
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if args.eval:
        g_policy.eval()

    # Predict semantic map from frame 1
    poses = gt_local_poses_tensor()
    agent_heights = torch.from_numpy(np.asarray(
        [infos[env_idx]['agent_height'] for env_idx in range(num_scenes)])
    ).float().to(device)

    _, local_map, _, local_pose = \
        sem_map_module(
            obs,
            poses,
            local_map,
            local_pose,
            agent_heights,
            pose_mode="absolute",
        )
    set_all_poses_from_gt()
    apply_minav_full_grid_mapping(obs, 0)
    update_3dgs_mappers(
        args,
        slam_mappers,
        infos,
        0,
        wait_env,
        finished,
        force_train_envs=combined_3dgs_force_flags(),
        target_masks=get_minav_target_masks(),
        semantic_instances=get_gaussian_semantic_instances(),
        stage_labels=get_minav_stage_labels(),
        skip_train_envs=phase1_gaussmi_skip_3dgs,
    )

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, ngc, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :].detach()
    global_input[:, 4:8, :, :] = nn.MaxPool2d(args.global_downscaling)(
        full_map[:, 0:4, :, :])
    global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
    goal_cat_id = torch.from_numpy(np.asarray(
        [int(infos[env_idx]['goal_cat_id']) for env_idx
         in range(num_scenes)]))

    

    extras = torch.zeros(num_scenes, 2)
    extras[:, 0] = global_orientation[:, 0]
    extras[:, 1] = goal_cat_id

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(extras)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=True
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
    global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                    for x, y in global_goals]

    goal_maps, exp_goal_maps, exploration_goal_debug = \
        build_exploration_goal_maps(global_goals, 0)
    found_goal = [0 for _ in range(num_scenes)]
    minav_overrides = apply_minav_goal_overrides(goal_maps, found_goal, 0)
    maybe_start_phase1_gaussmi(0)

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]
        p_input['exp_goal'] = exp_goal_maps[e]
        p_input['exploration_policy'] = exploration_policy
        p_input['exploration_goal_debug'] = exploration_goal_debug[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = found_goal[e]
        p_input['wait'] = wait_env[e] or finished[e]
        p_input['sem_map'] = local_map[e, 4:11, :, :
                                            ].cpu().numpy()
        add_minav_planner_fields(e, p_input, minav_overrides)
        add_timeout_stop_fields(e, p_input)
        if args.visualize or args.print_images:
            local_map[e, 10, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:11, :, :
                                            ].argmax(0).cpu().numpy()

    apply_initial_spin_overrides(planner_inputs, 0)
    apply_phase1_gaussmi_overrides(planner_inputs, 0)
    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
    infos = list(infos)
    collect_frontier_event_refresh(infos)

    start = time.time()
    runtime_timing = RuntimeTimingRecorder()
    g_reward = 0
    last_result_only_count = int(eval_resume.get("count", 0))

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    write_event("initial_target", {"runtime": multi_runtime_audit(0), "agent": envs.multi_state_audit(0)})
    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        frame_started = time.perf_counter()
        frame_was_terminal = bool(np.any(done))
        semantic_mapping_elapsed = 0.0
        gs_update_elapsed = 0.0
        gs_mapping_only_elapsed = None
        goal_override_elapsed = 0.0
        gaussmi_elapsed = 0.0
        simulator_planner_elapsed = 0.0

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                terminal_info = dict(infos[e])
                target_index = int(terminal_info.get("multi_target_index", -1))
                if target_index not in range(5):
                    raise RuntimeError(
                        "terminal step missing Multi-IIN target index: {}".format(
                            target_index
                        )
                    )
                runtime_before = multi_runtime_audit(e)
                manifest_goal = compact_results.goal_metadata(terminal_info, target_index)
                target_record = {
                    "target_index": target_index,
                    "source_episode_id": str(
                        terminal_info.get("multi_source_episode_id", "")
                    ),
                    "goal_object_id": str(
                        manifest_goal.get("goal_object_id", "unknown")
                    ),
                    "goal_name": terminal_info.get("goal_name"),
                    "policy_goal_cat_id": int(
                        terminal_info.get("goal_cat_id", -1)
                    ),
                    "goal_cat_source": terminal_info.get("goal_cat_source"),
                    "episode_goal_cat_id": int(
                        terminal_info.get("episode_goal_cat_id", -1)
                    ),
                    "steps": int(terminal_info.get("time", 0)),
                    "success": float(terminal_info.get("success", 0.0)),
                    "spl": float(terminal_info.get("spl", 0.0)),
                    "soft_spl": float(terminal_info.get("soft_spl", 0.0)),
                    "distance_to_goal_m": float(
                        terminal_info.get("distance_to_goal", 0.0)
                    ),
                    "runtime_at_terminal": runtime_before,
                }
                manager_summary = terminal_manager_summary(e)
                # Keep the official Habitat metric separate from internal
                # Gaussian/FMM bookkeeping.  Internal fallback distances are
                # useful diagnostics, but never substitute for Habitat's
                # success/distance computation.
                manager = target_managers[e]
                final_agent = terminal_info.get("agent_position")
                def _xz_distance(a, b):
                    try:
                        aa = np.asarray(a, dtype=np.float32).reshape(-1)
                        bb = np.asarray(b, dtype=np.float32).reshape(-1)
                        if aa.size < 3 or bb.size < 3:
                            return None
                        value = float(np.linalg.norm(aa[[0, 2]] - bb[[0, 2]]))
                        return value if np.isfinite(value) else None
                    except Exception:
                        return None
                raw_target = None
                execution_goal = None
                if manager is not None:
                    raw_target = getattr(manager, "navigation_goal_raw_world", None)
                    execution_goal = getattr(manager, "navigation_goal_world", None)
                    if raw_target is None:
                        raw_target = getattr(manager, "localized_target_position", None)
                    if raw_target is None:
                        raw_target = getattr(manager, "confirmed_target_position", None)
                target_record.update(
                    {
                        "internal_raw_target_distance_m": _xz_distance(final_agent, raw_target),
                        "execution_goal_distance_m": _xz_distance(final_agent, execution_goal),
                        "habitat_distance_m": float(terminal_info.get("distance_to_goal", 0.0)),
                        "habitat_success_distance_m": terminal_info.get(
                            "success_distance", terminal_info.get("success_distance_m")
                        ),
                        "official_success": bool(float(terminal_info.get("success", 0.0)) > 0.0),
                        "distance_source": "habitat_official",
                    }
                )
                pipeline_audit = dict(
                    getattr(manager, "_multi_pipeline_audit", {}) or {}
                ) if manager is not None else {}
                if target_record["success"] > 0.0:
                    pipeline_stage = "success"
                elif int(pipeline_audit.get("confirmation_attempts", 0)) <= 0:
                    pipeline_stage = "not_discovered_or_no_confirm_candidate"
                elif int(pipeline_audit.get("confirmation_successes", 0)) <= 0:
                    pipeline_stage = "confirmation_failed"
                elif int(pipeline_audit.get("navigation_entries", 0)) <= 0:
                    pipeline_stage = "localization_or_reachability_failed"
                else:
                    pipeline_stage = "navigation_not_reached"
                target_record["pipeline_stage"] = pipeline_stage
                target_record["pipeline_audit"] = pipeline_audit
                if target_record["success"] <= 0.0:
                    target_record["failure_reason"] = (
                        manager_summary.get("fail_reason")
                        or (
                            "step_budget_exhausted"
                            if target_record["steps"] >= int(args.multi_goal_step_budget)
                            else "target_not_reached"
                        )
                    )
                if multi_active_records[e] is None:
                    multi_active_records[e] = {
                        "multi_episode_id": terminal_info.get("multi_episode_id"),
                        "scene_name": terminal_info.get("scene_name"),
                        "variant": args.multi_variant,
                        "goals": [],
                    }
                multi_active_records[e]["goals"].append(target_record)
                write_event("target_terminal", target_record)

                mapper = slam_mappers[e]
                mapper_output_dir = (
                    str(getattr(mapper, "output_dir", "")) if mapper is not None else None
                )
                if mapper is not None:
                    try:
                        mapper.save_slam_results(save_checkpoint=False)
                    except Exception as exc:
                        logging.exception(
                            "Failed to save terminal compact 3DGS env %s: %s", e, exc
                        )
                gaussian_preview_path, gaussian_quality = save_terminal_gaussian_preview(
                    e, target_index, target_record["steps"]
                )
                manager = target_managers[e]
                if compact_results.enabled and manager is not None:
                    try:
                        ensure_terminal_match_source(
                            manager,
                            terminal_info,
                            target_record["steps"],
                        )
                    except Exception as exc:
                        logging.exception(
                            "Failed to save terminal compact match env %s: %s", e, exc
                        )
                visualization_dir = (
                    str(getattr(manager, "visualization_dir", ""))
                    if manager is not None and getattr(manager, "visualization_dir", None)
                    else None
                )
                try:
                    target_archive_dir = compact_results.archive_target(
                        terminal_info,
                        target_record,
                        compact_full_map_snapshot(e),
                        mapper_output_dir,
                        visualization_dir,
                        terminal_localization=manager_summary,
                        gaussian_preview_path=gaussian_preview_path,
                        gaussian_quality=gaussian_quality,
                    )
                    target_record["compact_result_dir"] = (
                        str(target_archive_dir) if target_archive_dir is not None else None
                    )
                except Exception as exc:
                    logging.exception(
                        "Failed to archive compact Multi-IIN target env %s: %s", e, exc
                    )
                    target_record["compact_result_error"] = str(exc)

                if target_index < 4:
                    switched_obs, switched_info, agent_audit = \
                        envs.switch_multi_goal(e, args.multi_variant)
                    obs[e].copy_(switched_obs)
                    infos[e] = switched_info
                    reset_reusable_state = args.multi_variant != "Continuous-Shared"

                    if use_minav_target and target_managers[e] is not None:
                        target_managers[e].reset_episode(
                            infos[e],
                            slam_mappers[e] if not reset_reusable_state else None,
                            infos[e].get("instance_imagegoal"),
                            infos[e].get("goal_name"),
                        )
                    if reset_reusable_state:
                        update_intrinsic_rew(e)
                        init_map_and_pose_for_env(e)
                        reset_multi_mapper(e, infos[e], save_existing=True)
                        reset_initial_spin_for_env(e)
                    if use_minav_target and target_managers[e] is not None:
                        target_managers[e].update_slam_mapper(slam_mappers[e])
                    phase1_gaussmi_controllers[e].reset(infos[e])
                    # Event levels belong to the prior target request even when
                    # the persistent map is shared across the boundary.
                    frontier_event_previous_flags[e].clear()

                    runtime_after = multi_runtime_audit(e)
                    boundary_record = {
                        "multi_episode_id": infos[e].get("multi_episode_id"),
                        "variant": args.multi_variant,
                        "runtime_before": runtime_before,
                        "runtime_after": runtime_after,
                        "agent": agent_audit,
                    }
                    multi_boundary_records.append(boundary_record)
                    multi_active_records[e]["target_boundary"] = boundary_record
                    multi_active_records[e].setdefault("target_boundaries", []).append(boundary_record)
                    write_event("boundary", boundary_record)
                    done[e] = False
                    l_masks[e] = 1.0
                    g_masks[e] = 1.0
                    wait_env[e] = 0.0
                    if reset_reusable_state and exploration_policy == "frontier":
                        frontier_goal_maps_cache = None
                        frontier_goal_debug_cache = None
                        frontier_last_refresh_step = None
                        frontier_last_refresh_movement_steps = None
                        frontier_event_refresh_pending = False
                        frontier_event_refresh_reasons = []
                        frontier_last_explored_counts = None
                        frontier_last_lmb_cache = None
                        frontier_candidate_full_points_cache = [
                            [] for _ in range(num_scenes)
                        ]
                        frontier_selected_goal_full_cache = [
                            None for _ in range(num_scenes)
                        ]
                        frontier_goal_last_select_step = [
                            None for _ in range(num_scenes)
                        ]
                        frontier_refresh_reason_cache = []
                    continue

                completed_record = multi_active_records[e]
                completed_record["ordered_success"] = float(
                    all(goal["success"] > 0.0 for goal in completed_record["goals"])
                )
                completed_record["total_steps"] = int(
                    sum(goal["steps"] for goal in completed_record["goals"])
                )
                multi_completed_records.append(completed_record)
                multi_active_records[e] = None

                success = float(completed_record["ordered_success"])
                spl = float(target_record["spl"])
                dist = float(target_record["distance_to_goal_m"])
                spl_per_category[target_record["goal_name"]].append(spl)
                success_per_category[target_record["goal_name"]].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)

                if len(multi_completed_records) >= num_episodes:
                    finished[e] = 1
                    wait_env[e] = 1.0
                    continue

                next_obs, next_info = envs.reset_at(e)
                obs[e].copy_(next_obs)
                infos[e] = next_info
                if use_minav_target and target_managers[e] is not None:
                    target_managers[e].reset_episode(
                        infos[e],
                        None,
                        infos[e].get("instance_imagegoal"),
                        infos[e].get("goal_name"),
                    )
                reset_multi_mapper(e, infos[e], save_existing=True)
                if use_minav_target and target_managers[e] is not None:
                    target_managers[e].update_slam_mapper(slam_mappers[e])
                init_map_and_pose_for_env(e)
                reset_initial_spin_for_env(e)
                phase1_gaussmi_controllers[e].reset(infos[e])
                frontier_event_previous_flags[e].clear()
                wait_env[e] = 1.0
                if exploration_policy == "frontier":
                    frontier_goal_maps_cache = None
                    frontier_goal_debug_cache = None
                    frontier_last_refresh_step = None
                    frontier_last_refresh_movement_steps = None
                    frontier_event_refresh_pending = False
                    frontier_event_refresh_reasons = []
                    frontier_last_explored_counts = None
                    frontier_last_lmb_cache = None
                    frontier_candidate_full_points_cache = [
                        [] for _ in range(num_scenes)
                    ]
                    frontier_selected_goal_full_cache = [
                        None for _ in range(num_scenes)
                    ]
                    frontier_goal_last_select_step = [
                        None for _ in range(num_scenes)
                    ]
                    frontier_refresh_reason_cache = []
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        synchronize_cuda_for_timing()
        semantic_mapping_started = time.perf_counter()
        poses = gt_local_poses_tensor()
        agent_heights = torch.from_numpy(np.asarray(
            [infos[env_idx]['agent_height'] for env_idx in range(num_scenes)])
        ).float().to(device)

        _, local_map, _, local_pose = \
            sem_map_module(
                obs,
                poses,
                local_map,
                local_pose,
                agent_heights,
                pose_mode="absolute",
            )
        set_all_poses_from_gt()
        apply_minav_full_grid_mapping(obs, step + 1)
        synchronize_cuda_for_timing()
        semantic_mapping_elapsed = time.perf_counter() - semantic_mapping_started

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
        observe_phase1_gaussmi(step + 1)
        gs_update_started = time.perf_counter()
        gs_mapping_only_by_env = update_3dgs_mappers(
            args,
            slam_mappers,
            infos,
            step + 1,
            wait_env,
            finished,
            force_train_envs=combined_3dgs_force_flags(),
            target_masks=get_minav_target_masks(),
            semantic_instances=get_gaussian_semantic_instances(),
            stage_labels=get_minav_stage_labels(),
            skip_train_envs=phase1_gaussmi_skip_3dgs,
        )
        active_mapping_times = [
            value for value in gs_mapping_only_by_env if value is not None
        ]
        if active_mapping_times:
            gs_mapping_only_elapsed = float(sum(active_mapping_times))
        gs_update_elapsed = time.perf_counter() - gs_update_started

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):
                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.
                else:
                    update_intrinsic_rew(e)

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                set_pose_from_gt_for_env(e)

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()
                set_pose_from_gt_for_env(e)

            locs = local_pose.cpu().numpy()
            for e in range(num_scenes):
                global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
            global_input[:, 0:4, :, :] = local_map[:, 0:4, :, :]
            global_input[:, 4:8, :, :] = \
                nn.MaxPool2d(args.global_downscaling)(
                    full_map[:, 0:4, :, :])
            global_input[:, 8:, :, :] = local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(np.asarray(
                [int(infos[env_idx]['goal_cat_id']) for env_idx
                 in range(num_scenes)]))
            extras[:, 0] = global_orientation[:, 0]
            extras[:, 1] = goal_cat_id

            # Get exploration reward and metrics
            g_reward = torch.from_numpy(np.asarray(
                [infos[env_idx]['g_reward'] for env_idx in range(num_scenes)])
            ).float().to(device)
            g_reward += args.intrinsic_rew_coeff * intrinsic_rews.detach()

            g_process_rewards += g_reward.cpu().numpy()
            g_total_rewards = g_process_rewards * \
                (1 - g_masks.cpu().numpy())
            g_process_rewards *= g_masks.cpu().numpy()
            per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

            if np.sum(g_total_rewards) != 0:
                for total_rew in g_total_rewards:
                    if total_rew != 0:
                        g_episode_rewards.append(total_rew)

            # Add samples to global policy storage
            if step == 0:
                g_rollouts.obs[0].copy_(global_input)
                g_rollouts.extras[0].copy_(extras)
            else:
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, extras
                )

            # Sample long-term goal from global policy
            g_value, g_action, g_action_log_prob, g_rec_states = \
                g_policy.act(
                    g_rollouts.obs[g_step + 1],
                    g_rollouts.rec_states[g_step + 1],
                    g_rollouts.masks[g_step + 1],
                    extras=g_rollouts.extras[g_step + 1],
                    deterministic=True
                )
            cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
            global_goals = [[int(action[0] * local_w),
                             int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                             min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_reward = 0
            g_masks = torch.ones(num_scenes).float().to(device)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
        frontier_refresh_now = (
            frontier_refresh_due(step + 1)
            if exploration_policy == "frontier"
            else (l_step == args.num_local_steps - 1)
        )
        goal_maps, exp_goal_maps, exploration_goal_debug = \
            build_exploration_goal_maps(
                global_goals,
                step + 1,
                refresh=frontier_refresh_now,
            )

        # recompute found_goal[e] in each instance_exp
        # for e in range(num_scenes):
        #     cn = infos[e]['goal_cat_id'] + 4
        #     if local_map[e, cn, :, :].sum() != 0.:
        #         cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
        #         cat_semantic_scores = cat_semantic_map
        #         cat_semantic_scores[cat_semantic_scores > 0] = 1.
        #         goal_maps[e] = cat_semantic_scores
        #         found_goal[e] = 1
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        goal_override_started = time.perf_counter()
        minav_overrides = apply_minav_goal_overrides(goal_maps, found_goal, step + 1)
        goal_override_elapsed = time.perf_counter() - goal_override_started
        gaussmi_started = time.perf_counter()
        maybe_start_phase1_gaussmi(step + 1)
        gaussmi_elapsed = time.perf_counter() - gaussmi_started

        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e]
            p_input['exp_goal'] = exp_goal_maps[e]
            p_input['exploration_policy'] = exploration_policy
            p_input['exploration_goal_debug'] = exploration_goal_debug[e]
            if exploration_policy == "frontier":
                p_input['new_goal'] = bool(
                    exploration_goal_debug[e].get("selected_goal_reselect_reasons")
                    and exploration_goal_debug[e].get("goal_count", 0) > 0
                )
            else:
                p_input['new_goal'] = bool(frontier_refresh_now)
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            p_input['sem_map'] = local_map[e, 4:11, :, :
                                            ].cpu().numpy()
            add_minav_planner_fields(e, p_input, minav_overrides)
            add_timeout_stop_fields(e, p_input)
            if args.visualize or args.print_images:
                local_map[e, 10, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:11, :,
                                                    :].argmax(0).cpu().numpy()

        apply_initial_spin_overrides(planner_inputs, step + 1)
        apply_phase1_gaussmi_overrides(planner_inputs, step + 1)
        simulator_planner_started = time.perf_counter()
        obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        simulator_planner_elapsed = time.perf_counter() - simulator_planner_started
        infos = list(infos)
        collect_frontier_event_refresh(infos)
        timing_sample = {
            "semantic_and_2d_mapping_s": semantic_mapping_elapsed,
            "3dgs_update_s": gs_update_elapsed,
            "3dgs_mapping_only_s": gs_mapping_only_elapsed,
            "goal_override_localization_s": goal_override_elapsed,
            "gaussmi_s": gaussmi_elapsed,
            "simulator_and_local_planner_s": simulator_planner_elapsed,
            "total_frame_s": time.perf_counter() - frame_started,
        }
        if gs_mapping_only_elapsed is not None:
            timing_sample["mapping_only_total_s"] = (
                semantic_mapping_elapsed + gs_mapping_only_elapsed
            )
        runtime_timing.record_frame(
            timing_sample,
            terminal=frame_was_terminal,
        )


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Training
        torch.set_grad_enabled(True)
        if g_step % args.num_global_steps == args.num_global_steps - 1 \
                and l_step == args.num_local_steps - 1:
            if not args.eval and use_rl_exploration:
                g_next_value = g_policy.get_value(
                    g_rollouts.obs[-1],
                    g_rollouts.rec_states[-1],
                    g_rollouts.masks[-1],
                    extras=g_rollouts.extras[-1]
                ).detach()

                g_rollouts.compute_returns(g_next_value, args.use_gae,
                                           args.gamma, args.tau)
                g_value_loss, g_action_loss, g_dist_entropy = \
                    g_agent.update(g_rollouts)
                g_value_losses.append(g_value_loss)
                g_action_losses.append(g_action_loss)
                g_dist_entropies.append(g_dist_entropy)
            g_rollouts.after_update()

        torch.set_grad_enabled(False)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Logging
        if step % args.log_interval == 0:
            if args.eval and args.result_only_log:
                total_count, success_sum, spl_sum, dist_sum = \
                    collect_eval_totals(
                        episode_success, episode_spl, episode_dist, eval_resume
                    )
                if total_count > 0:
                    if total_count != last_result_only_count:
                        log = format_eval_metric(
                            "InstanceImageNav succ/spl/dtg:",
                            total_count,
                            success_sum,
                            spl_sum,
                            dist_sum,
                        )
                        print(log)
                        logging.info(log)
                        last_result_only_count = total_count
            else:
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(step * num_scenes),
                    "FPS {},".format(int(step * num_scenes / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])

                if args.eval:
                    total_count, success_sum, spl_sum, dist_sum = \
                        collect_eval_totals(
                            episode_success, episode_spl, episode_dist, eval_resume
                        )

                    if total_count > 0:
                        log += " " + format_eval_metric(
                            "InstanceImageNav succ/spl/dtg:",
                            total_count,
                            success_sum,
                            spl_sum,
                            dist_sum,
                        )
                else:
                    if len(episode_success) > 100:
                        log += " InstanceImageNav succ/spl/dtg:"
                        log += " {:.5f}/{:.5f}/{:.5f}({:.0f}),".format(
                            np.mean(episode_success),
                            np.mean(episode_spl),
                            np.mean(episode_dist),
                            len(episode_spl))

                log += "\n\tLosses:"
                if len(g_value_losses) > 0 and not args.eval:
                    log += " ".join([
                        " Policy Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])

                print(log)
                logging.info(log)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Save best models
        if (step * num_scenes) % args.save_interval < \
                num_scenes:
            if len(g_episode_rewards) >= 1000 and \
                    (np.mean(g_episode_rewards) >= best_g_reward) \
                    and not args.eval and use_rl_exploration:
                torch.save(g_policy.state_dict(),
                           os.path.join(log_dir, "model_best.pth"))
                best_g_reward = np.mean(g_episode_rewards)

        # Save periodic models
        if (step * num_scenes) % args.save_periodic < \
                num_scenes:
            total_steps = step * num_scenes
            if not args.eval and use_rl_exploration:
                torch.save(g_policy.state_dict(),
                           os.path.join(dump_dir,
                                        "periodic_{}.pth".format(total_steps)))
            
        # ------------------------------------------------------------------

    if args.enable_3dgs:
        for env_idx, mapper in enumerate(slam_mappers):
            if mapper is not None:
                try:
                    mapper.save_slam_results(save_checkpoint=False)
                except Exception as exc:
                    logging.exception("Failed to save final 3DGS for env %s: %s", env_idx, exc)

    boundary_checks = []
    for boundary in multi_boundary_records:
        before = boundary["runtime_before"]
        after = boundary["runtime_after"]
        agent = boundary["agent"]
        if args.multi_variant == "Continuous-Shared":
            state_ok = (
                before["full_map_nonzero"] == after["full_map_nonzero"]
                and before["minav_obstacle_observations"]
                == after["minav_obstacle_observations"]
                and before["minav_free_observations"]
                == after["minav_free_observations"]
                and before["mapper"]["object_id"]
                == after["mapper"]["object_id"]
                and before["mapper"]["frame_count"]
                == after["mapper"]["frame_count"]
                and agent["agent_before"]["visited_cells"]
                == agent["agent_after"]["visited_cells"]
                and agent["boundary_to_after_position_delta_m"] <= 1e-5
            )
        else:
            mapper_reset = (
                not bool(args.enable_3dgs)
                or (
                    before["mapper"]["object_id"]
                    != after["mapper"]["object_id"]
                    and after["mapper"]["frame_count"] == 0
                )
            )
            pose_ok = (
                agent["initial_to_after_position_delta_m"] <= 1e-5
                if args.multi_variant == "Independent-Reset"
                else agent["boundary_to_after_position_delta_m"] <= 1e-5
            )
            state_ok = (
                after["minav_obstacle_observations"] == 0
                and after["minav_free_observations"] == 0
                and agent["agent_after"]["visited_cells"] == 0
                and mapper_reset
                and pose_ok
            )
        boundary_checks.append(bool(state_ok))

    records_complete = (
        len(multi_completed_records) == int(args.multi_num_episodes)
        and all(len(record.get("goals", [])) == 5 for record in multi_completed_records)
    )
    task_category_only = all(
        goal.get("goal_cat_source") == "oracle"
        for record in multi_completed_records
        for goal in record.get("goals", [])
    )
    budget_respected = all(
        int(goal.get("steps", 0)) <= int(args.multi_goal_step_budget)
        for record in multi_completed_records
        for goal in record.get("goals", [])
    )
    result_payload = {
        "schema_version": "multi-iin-real-smoke-v0.1",
        "passed": bool(
            records_complete
            and task_category_only
            and budget_respected
            and len(boundary_checks) == 4 * int(args.multi_num_episodes)
            and all(boundary_checks)
        ),
        "variant": args.multi_variant,
        "manifest": os.path.abspath(args.multi_manifest),
        "episode_start": int(args.multi_episode_start),
        "num_episodes_requested": int(args.multi_num_episodes),
        "goal_step_budget": int(args.multi_goal_step_budget),
        "goal_category_source": args.goal_cat_source,
        "enable_3dgs_requested_and_active": bool(args.enable_3dgs),
        "semantic_instance_filter": asdict(instance_filter_config),
        "fast_localization": asdict(fast_localization_config),
        "render_matching_acceleration": render_matching_acceleration.snapshot(),
        "planner_acceleration": {
            "config": {
                "reuse_minav_frame_maps": bool(
                    int(getattr(args, "multi_minav_reuse_frame_maps", 1))
                ),
                "fmm_exact_cache": bool(
                    int(getattr(args, "multi_fmm_exact_cache", 1))
                ),
            },
            "last_env_stats": [
                {
                    "hits": int((info or {}).get("multi_fmm_exact_cache_hits", 0)),
                    "misses": int((info or {}).get("multi_fmm_exact_cache_misses", 0)),
                    "last_hit": bool((info or {}).get("multi_fmm_exact_cache_hit", False)),
                }
                for info in infos
            ],
        },
        "runtime_timing": runtime_timing.summary(),
        "stage_runtime": {
            "confirmation": {
                "calls": int(sum(
                    (getattr(manager, "_multi_stage_runtime", {}) or {})
                    .get("confirmation", {}).get("calls", 0)
                    for manager in target_managers if manager is not None
                )),
                "total_s": float(sum(
                    (getattr(manager, "_multi_stage_runtime", {}) or {})
                    .get("confirmation", {}).get("total_s", 0.0)
                    for manager in target_managers if manager is not None
                )),
            },
            "localization": {
                "calls": int(sum(
                    (getattr(manager, "_multi_stage_runtime", {}) or {})
                    .get("localization", {}).get("calls", 0)
                    for manager in target_managers if manager is not None
                )),
                "total_s": float(sum(
                    (getattr(manager, "_multi_stage_runtime", {}) or {})
                    .get("localization", {}).get("total_s", 0.0)
                    for manager in target_managers if manager is not None
                )),
            },
            "wall_clock_total_s": float(time.time() - start),
        },
        "gaussian_capacity": asdict(
            GaussianCapacityManager.from_args(args).config
        ),
        "gaussian_semantic_instance_training": {
            "config": asdict(
                GaussianSemanticInstanceTrainer.from_args(args).config
            ),
            "mappers": [
                (
                    mapper._multi_iin_label_trainer.snapshot(mapper)
                    if mapper is not None
                    and getattr(mapper, "_multi_iin_label_trainer", None) is not None
                    else None
                )
                for mapper in slam_mappers
            ],
        },
        "gaussian_quality_refinement": {
            "config": asdict(
                GaussianQualityRefinementManager.from_args(args).config
            ),
            "mappers": [
                (
                    mapper._multi_iin_quality_refinement_manager.snapshot()
                    if mapper is not None
                    and getattr(
                        mapper,
                        "_multi_iin_quality_refinement_manager",
                        None,
                    )
                    is not None
                    else None
                )
                for mapper in slam_mappers
            ],
        },
        "gaussian_storage_policy": {
            "periodic_checkpoint_interval": int(args.gs_save_interval),
            "save_full_ply_each_frame": bool(
                int(getattr(args, "gs_save_full_ply_each_frame", 0))
            ),
            "save_repeated_confirmation_full_scene_ply": bool(
                int(
                    getattr(
                        args,
                        "multi_save_repeated_full_gaussian_snapshots",
                        0,
                    )
                )
            ),
        },
        "checks": {
            "records_complete": bool(records_complete),
            "task_category_only": bool(task_category_only),
            "budget_respected": bool(budget_respected),
            "state_boundary_checks": boundary_checks,
        },
        "episodes": multi_completed_records,
        "boundaries": multi_boundary_records,
    }
    result_path = os.path.abspath(args.multi_result_json)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result_payload, result_file, ensure_ascii=False, indent=2)
        result_file.write("\n")
    result_payload["result_json"] = result_path
    compact_results.finish(result_payload)
    print(
        "Multi-IIN real smoke: variant={} passed={} episodes={} output={}".format(
            args.multi_variant,
            result_payload["passed"],
            len(multi_completed_records),
            result_path,
        )
    )
    if not result_payload["passed"]:
        logging.error("Multi-IIN real smoke checks failed: %s", result_payload["checks"])

    # Print and save model performance numbers during evaluation
    if args.eval:
        if not args.quiet_log:
            print("Dumping eval details...")
        
        total_count, success_sum, spl_sum, dist_sum = collect_eval_totals(
            episode_success, episode_spl, episode_dist, eval_resume
        )

        if total_count > 0:
            log = format_eval_metric(
                "Final InstanceImageNav succ/spl/dtg:",
                total_count,
                success_sum,
                spl_sum,
                dist_sum,
            )
            print(log)
            logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        if not args.quiet_log:
            print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)

    if not result_payload["passed"]:
        raise RuntimeError(
            "Multi-IIN real smoke lifecycle checks failed; inspect {}".format(
                result_path
            )
        )


if __name__ == "__main__":
    main()
