"""Independent vector-environment construction for Multi-IIN v0."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from habitat import make_dataset
from habitat.config.default import get_config as get_habitat_config
from omegaconf import OmegaConf

from envs.habitat.utils.vector_env import VectorEnv
from five_goal_benchmark_20261026.runtime.multi_agent import MultiIINEnvAgent


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_manifest(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scene_key(episode):
    name = Path(str(episode.scene_id)).name
    for suffix in (".basis.glb", ".glb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _find_source_episode(episodes, scene_key, goal):
    matches = [
        episode
        for episode in episodes
        if _scene_key(episode) == scene_key
        and str(episode.episode_id) == str(goal["source_episode_id"])
        and str(episode.goal_object_id) == str(goal["goal_object_id"])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "source episode lookup failed for scene={} id={} object={}: {} matches"
            .format(
                scene_key,
                goal["source_episode_id"],
                goal["goal_object_id"],
                len(matches),
            )
        )
    return matches[0]


def _build_multi_dataset(config_env, specs):
    dataset = make_dataset(
        config_env.habitat.dataset.type,
        config=config_env.habitat.dataset,
    )
    source_episodes = list(dataset.episodes)
    multi_episodes = []
    for spec in specs:
        targets = [copy.deepcopy(_find_source_episode(source_episodes, spec["scene_key"], goal)) for goal in spec["goals"]]
        assert len(targets) == 5
        episode_1 = copy.deepcopy(targets[0])
        episode_2 = targets[1]
        episode_1._multi_iin_all_target_episodes = targets
        episode_1.start_position = [float(v) for v in spec["start_position"]]
        episode_1.start_rotation = [float(v) for v in spec["start_rotation"]]
        episode_1._multi_iin_spec = copy.deepcopy(spec)
        episode_1._multi_iin_target_2_episode = episode_2
        multi_episodes.append(episode_1)
    dataset.episodes = multi_episodes
    return dataset


def _make_multi_env(args, config_env, rank, specs):
    dataset = _build_multi_dataset(config_env, specs)
    OmegaConf.set_readonly(config_env, False)
    config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)
    env = MultiIINEnvAgent(
        args=args,
        rank=rank,
        config_env=config_env,
        dataset=dataset,
    )
    env.seed(int(args.seed) + rank)
    return env


def _configure(args, manifest, specs):
    config = get_habitat_config(
        config_path=str(REPO_ROOT / "envs/habitat/configs" / args.task_config)
    )
    source = manifest["source"]
    dataset_root = (REPO_ROOT / source["dataset_root"]).resolve()
    scenes_root = (REPO_ROOT / source["scenes_root"]).resolve()
    content_scenes = sorted({spec["scene_key"] for spec in specs})

    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = source["split"]
    config.habitat.dataset.data_path = str(
        (dataset_root / "{split}/{split}.json.gz").resolve()
    )
    config.habitat.dataset.scenes_dir = str(scenes_root)
    config.habitat.dataset.content_scenes = content_scenes
    config.habitat.environment.iterator_options.shuffle = False
    config.habitat.environment.max_episode_steps = max(
        int(args.multi_goal_step_budget) + 10, 20
    )
    config.habitat.environment.max_episode_seconds = 10000000
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(args.sim_gpu_id)

    sensors = config.habitat.simulator.agents.main_agent.sim_sensors
    for sensor_name in ("rgb_sensor", "depth_sensor", "semantic_sensor"):
        sensors[sensor_name].width = int(args.env_frame_width)
        sensors[sensor_name].height = int(args.env_frame_height)
        sensors[sensor_name].hfov = int(round(float(args.hfov)))
        sensors[sensor_name].position = [0, float(args.camera_height), 0]
    sensors.depth_sensor.min_depth = float(args.min_depth)
    sensors.depth_sensor.max_depth = float(args.max_depth)
    config.habitat.simulator.agents.main_agent.height = float(args.camera_height)
    OmegaConf.set_readonly(config, True)
    return config


class MultiVecPyTorch:
    def __init__(self, vector_env, device):
        self.venv = vector_env
        self.num_envs = vector_env.num_envs
        self.observation_space = vector_env.observation_space
        self.action_space = vector_env.action_space
        self.device = device

    def reset(self):
        obs, info = self.venv.reset()
        return torch.from_numpy(obs).float().to(self.device), info

    def plan_act_and_preprocess(self, inputs):
        obs, reward, done, info = self.venv.plan_act_and_preprocess(inputs)
        return (
            torch.from_numpy(obs).float().to(self.device),
            torch.from_numpy(reward).float(),
            done,
            info,
        )

    def get_frontier_map(self, inputs):
        maps = self.venv.get_frontier_map(inputs)
        return torch.from_numpy(maps).to(self.device)

    def switch_multi_goal(self, index, variant):
        obs, info, audit = self.venv.call_at(
            index,
            "switch_multi_goal",
            {"variant": variant},
        )
        return torch.from_numpy(np.asarray(obs)).float().to(self.device), info, audit

    def reset_at(self, index):
        result = self.venv.reset_at(index)
        obs, info = result[0]
        return torch.from_numpy(np.asarray(obs)).float().to(self.device), info

    def multi_state_audit(self, index):
        return self.venv.call_at(index, "multi_state_audit")

    def close(self):
        return self.venv.close()


def make_multi_vec_envs(args):
    if int(args.num_processes) != 1:
        raise ValueError("Multi-IIN v0 real smoke currently requires --num_processes 1")
    manifest = _read_manifest(args.multi_manifest)
    start = int(args.multi_episode_start)
    count = int(args.multi_num_episodes)
    specs = list(manifest["episodes"])[start : start + count]
    if len(specs) != count:
        raise ValueError(
            "requested {} Multi-IIN episodes at start {}, only {} available".format(
                count, start, len(specs)
            )
        )
    config = _configure(args, manifest, specs)
    vector_env = VectorEnv(
        make_env_fn=_make_multi_env,
        env_fn_args=((args, config, 0, specs),),
        auto_reset_done=False,
    )
    return MultiVecPyTorch(vector_env, args.device)
