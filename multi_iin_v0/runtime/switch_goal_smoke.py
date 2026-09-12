#!/usr/bin/env python3
"""Smoke-test an in-place Instance-ImageNav goal switch in Habitat.

This probe deliberately bypasses the IEVE policy and mapping stack.  It tests
the lowest-level lifecycle operation needed by a real Multi-IIN runner:
finishing target 1, changing to target 2 without resetting the simulator, and
continuing to step the same Habitat environment.

Only this new ``multi_iin_v0`` namespace is used for outputs.  No existing
IEVE source or configuration file is modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import habitat  # noqa: E402
from habitat.config.default import get_config as get_habitat_config  # noqa: E402


DEFAULT_MANIFEST = REPO_ROOT / "multi_iin_v0/manifests/miin_v0_val_k2.json"
DEFAULT_OUTPUT = REPO_ROOT / "multi_iin_v0/results/switch_goal_smoke.json"
DEFAULT_TASK_CONFIG = REPO_ROOT / "envs/habitat/configs/tasks/instance_imagenav.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def scene_content_name(
    scene_key: str,
    dataset_root: Path,
    split: str,
) -> str:
    content_file = dataset_root / split / "content/{}.json.gz".format(scene_key)
    if not content_file.is_file():
        raise FileNotFoundError("missing scene content: {}".format(content_file))
    return scene_key


def episode_matches(episode: Any, goal: Mapping[str, Any]) -> bool:
    return (
        str(episode.episode_id) == str(goal["source_episode_id"])
        and str(episode.goal_object_id) == str(goal["goal_object_id"])
    )


def find_episode(episodes: Iterable[Any], goal: Mapping[str, Any]) -> Any:
    matches = [episode for episode in episodes if episode_matches(episode, goal)]
    if len(matches) != 1:
        raise RuntimeError(
            "expected one source episode for id={} object={}, found {}".format(
                goal["source_episode_id"], goal["goal_object_id"], len(matches)
            )
        )
    return matches[0]


def vector(value: Sequence[Any]) -> list[float]:
    return [float(item) for item in value]


def rotation_vector(rotation: Any) -> list[float]:
    return [
        float(rotation.x),
        float(rotation.y),
        float(rotation.z),
        float(rotation.w),
    ]


def pose_snapshot(env: habitat.Env) -> Dict[str, list[float]]:
    state = env.sim.get_agent_state(0)
    return {
        "position": vector(state.position),
        "rotation_xyzw": rotation_vector(state.rotation),
    }


def quaternion_distance(a: Sequence[float], b: Sequence[float]) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    return float(1.0 - abs(np.dot(qa, qb)))


def pose_delta(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    return {
        "position_m": float(
            np.linalg.norm(
                np.asarray(a["position"], dtype=np.float64)
                - np.asarray(b["position"], dtype=np.float64)
            )
        ),
        "rotation_one_minus_abs_dot": quaternion_distance(
            a["rotation_xyzw"], b["rotation_xyzw"]
        ),
    }


def image_fingerprint(image: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(image)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "mean": float(array.mean()),
    }


def goal_view_points(episode: Any) -> list[Any]:
    return [
        view_point.agent_state.position
        for goal in episode.goals
        for view_point in goal.view_points
    ]


def terminal_stop_action() -> Dict[str, Any]:
    return {
        "action": ("velocity_control", "velocity_stop"),
        "action_args": {
            "linear_velocity": np.array([-1.0], dtype=np.float32),
            "angular_velocity": np.array([0.0], dtype=np.float32),
            "camera_pitch_velocity": np.array([0.0], dtype=np.float32),
            "velocity_stop": np.array([1.0], dtype=np.float32),
        },
    }


def nonterminal_turn_action() -> Dict[str, Any]:
    return {
        "action": ("velocity_control", "velocity_stop"),
        "action_args": {
            "linear_velocity": np.array([-1.0], dtype=np.float32),
            "angular_velocity": np.array([0.2], dtype=np.float32),
            "camera_pitch_velocity": np.array([0.0], dtype=np.float32),
            "velocity_stop": np.array([0.0], dtype=np.float32),
        },
    }


def switch_goal_without_sim_reset(
    env: habitat.Env,
    target_episode: Any,
) -> Dict[str, Any]:
    """Replace target-specific Habitat state while preserving simulator pose."""

    current = env.current_episode
    current.episode_id = str(target_episode.episode_id)
    current.goal_object_id = str(target_episode.goal_object_id)
    current.goal_image_id = int(target_episode.goal_image_id)
    current.object_category = target_episode.object_category
    current.goals = target_episode.goals
    current._shortest_path_cache = None

    # Force the rendered instance-image sensor to invalidate its cache even if
    # a future manifest happens to reuse a source episode identifier.
    for sensor in env.task.sensor_suite.sensors.values():
        if hasattr(sensor, "_current_episode_id"):
            sensor._current_episode_id = None
            sensor._current_image_goal = None

    simulator_observations = env.sim.get_observations_at()
    if simulator_observations is None:
        raise RuntimeError("Habitat returned no simulator observations")
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

    # Equivalent to Env._reset_stats(), without calling Env.reset() or
    # EmbodiedTask.reset(), both of which reset Habitat-Sim.
    env._episode_start_time = time.time()
    env._elapsed_steps = 0
    env._episode_over = False
    env._episode_force_changed = False
    env._episode_from_iter_on_reset = False

    env.task.measurements.reset_measures(
        episode=current,
        task=env.task,
        observations=observations,
    )
    return observations


def build_environment(
    manifest: Mapping[str, Any],
    episode_spec: Mapping[str, Any],
    gpu_id: int,
    width: int,
    height: int,
) -> tuple[habitat.Env, Any]:
    config = get_habitat_config(config_path=str(DEFAULT_TASK_CONFIG))
    source = manifest["source"]
    scenes_root = (REPO_ROOT / source["scenes_root"]).resolve()
    dataset_root = (REPO_ROOT / source["dataset_root"]).resolve()
    content_scene = scene_content_name(
        episode_spec["scene_key"], dataset_root, source["split"]
    )

    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = source["split"]
    config.habitat.dataset.data_path = str(
        (dataset_root / "{split}/{split}.json.gz").resolve()
    )
    config.habitat.dataset.scenes_dir = str(scenes_root)
    config.habitat.dataset.content_scenes = [content_scene]
    config.habitat.environment.iterator_options.shuffle = False
    config.habitat.environment.max_episode_steps = 50
    config.habitat.environment.max_episode_seconds = 10000000
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_id)

    sensors = config.habitat.simulator.agents.main_agent.sim_sensors
    for sensor_name in ("rgb_sensor", "depth_sensor", "semantic_sensor"):
        sensors[sensor_name].width = int(width)
        sensors[sensor_name].height = int(height)
    OmegaConf.set_readonly(config, True)

    dataset = habitat.make_dataset(
        config.habitat.dataset.type,
        config=config.habitat.dataset,
    )
    goal_1, goal_2 = episode_spec["goals"]
    source_1 = copy.deepcopy(find_episode(dataset.episodes, goal_1))
    source_2 = copy.deepcopy(find_episode(dataset.episodes, goal_2))

    # The Multi-IIN manifest owns the initial pose.  Keep target 1's goal data
    # but make the reset pose exactly match the new benchmark episode.
    source_1.start_position = vector(episode_spec["start_position"])
    source_1.start_rotation = vector(episode_spec["start_rotation"])
    dataset.episodes = [source_1]
    return habitat.Env(config=config, dataset=dataset), source_2


def run(args: argparse.Namespace) -> Dict[str, Any]:
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    episodes = manifest["episodes"]
    if args.episode_index < 0 or args.episode_index >= len(episodes):
        raise IndexError(
            "episode index {} outside [0, {})".format(
                args.episode_index, len(episodes)
            )
        )
    episode_spec = episodes[args.episode_index]

    env, target_2_episode = build_environment(
        manifest,
        episode_spec,
        args.gpu_id,
        args.width,
        args.height,
    )
    try:
        observations_1 = env.reset()
        pose_after_reset = pose_snapshot(env)
        target_1_image = image_fingerprint(observations_1["instance_imagegoal"])
        metrics_1 = dict(env.get_metrics())

        expected_start = np.asarray(episode_spec["start_position"], dtype=np.float64)
        actual_start = np.asarray(pose_after_reset["position"], dtype=np.float64)
        start_position_error = float(np.linalg.norm(expected_start - actual_start))

        env.step(terminal_stop_action())
        pose_at_boundary = pose_snapshot(env)
        goal_1_was_terminal = bool(env.episode_over)

        observations_2 = switch_goal_without_sim_reset(env, target_2_episode)
        pose_after_switch = pose_snapshot(env)
        target_2_image = image_fingerprint(observations_2["instance_imagegoal"])
        metrics_2 = dict(env.get_metrics())

        target_2_expected_distance = float(
            env.sim.geodesic_distance(
                env.sim.get_agent_state(0).position,
                goal_view_points(env.current_episode),
                env.current_episode,
            )
        )
        target_2_metric_distance = float(metrics_2["distance_to_goal"])

        env.step(nonterminal_turn_action())
        continued_step_was_nonterminal = not bool(env.episode_over)
        pose_after_continued_step = pose_snapshot(env)

        boundary_delta = pose_delta(pose_at_boundary, pose_after_switch)
        checks = {
            "manifest_start_pose_used": start_position_error <= 1e-5,
            "goal_1_stop_was_terminal": goal_1_was_terminal,
            "simulator_pose_preserved_at_switch": (
                boundary_delta["position_m"] <= 1e-6
                and boundary_delta["rotation_one_minus_abs_dot"] <= 1e-8
            ),
            "target_image_changed": target_1_image["sha256"] != target_2_image["sha256"],
            "distance_metric_rebound_to_target_2": abs(
                target_2_metric_distance - target_2_expected_distance
            ) <= 1e-5,
            "continued_step_was_nonterminal": continued_step_was_nonterminal,
            "current_episode_is_target_2": (
                str(env.current_episode.episode_id)
                == str(episode_spec["goals"][1]["source_episode_id"])
                and str(env.current_episode.goal_object_id)
                == str(episode_spec["goals"][1]["goal_object_id"])
            ),
        }
        passed = all(checks.values())

        result = {
            "schema_version": "miin-switch-smoke-v0.1",
            "passed": passed,
            "manifest": str(manifest_path),
            "manifest_episode_index": int(args.episode_index),
            "multi_episode_id": episode_spec["episode_id"],
            "scene_key": episode_spec["scene_key"],
            "goal_1": episode_spec["goals"][0],
            "goal_2": episode_spec["goals"][1],
            "checks": checks,
            "evidence": {
                "start_position_error_m": start_position_error,
                "pose_at_boundary": pose_at_boundary,
                "pose_after_switch": pose_after_switch,
                "pose_switch_delta": boundary_delta,
                "pose_after_continued_step": pose_after_continued_step,
                "target_1_image": target_1_image,
                "target_2_image": target_2_image,
                "target_1_initial_distance_m": float(metrics_1["distance_to_goal"]),
                "target_2_initial_distance_m": target_2_metric_distance,
                "target_2_independent_distance_m": target_2_expected_distance,
            },
        }
        return result
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    result = run(args)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
