"""Sequential ordered two-goal adapter for the unmodified UniGoal sources.

This module deliberately lives outside `/code1/tyj/UniGoal`: it reuses its
environment and agent implementation but owns the manifest selection and the
target-1 -> target-2 transition.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
from habitat import make_dataset
from habitat.config.default import get_config
from omegaconf import OmegaConf

from src_unigoal.envs.instanceimagegoal_env import InstanceImageGoal_Env


def _scene_key(episode):
    return Path(str(episode.scene_id)).name.replace(".basis.glb", "").replace(".glb", "")


def _load_specs(path, start, count):
    with open(path, "r") as handle:
        manifest = json.load(handle)
    specs = manifest.get("episodes", [])[start:start + count]
    if manifest.get("task") != "SequentialOrderedMultiIIN" or len(specs) != count:
        raise ValueError("invalid manifest or requested episode range")
    return manifest, specs


def _source_episode(episodes, scene_key, goal):
    matches = [episode for episode in episodes
               if _scene_key(episode) == scene_key
               and str(episode.episode_id) == str(goal["source_episode_id"])
               and str(episode.goal_object_id) == str(goal["goal_object_id"])]
    if len(matches) != 1:
        raise RuntimeError("cannot uniquely resolve source episode: {}".format(goal))
    return matches[0]


class SequentialUniGoalEnv(InstanceImageGoal_Env):
    """Keep simulator pose while replacing only active target metadata."""
    def reset(self):
        obs, info = super().reset()
        episode = self._env.current_episode
        self._multi_spec = getattr(episode, "_multi_spec", None)
        self._target_two = getattr(episode, "_target_two", None)
        if self._multi_spec is None or self._target_two is None:
            raise RuntimeError("multi target metadata missing")
        self._target_index = 0
        self.task_goal_idx = self.gt_goal_idx
        self._attach(info)
        return obs, info

    def _attach(self, info):
        goal = self._multi_spec["goals"][self._target_index]
        info.update({"multi_episode_id": self._multi_spec["episode_id"],
                     "multi_target_index": self._target_index,
                     "multi_source_episode_id": str(goal["source_episode_id"])})

    def switch_multi_goal(self):
        if self._target_index != 0:
            raise RuntimeError("only one target switch is supported")
        env = self._env
        before = np.asarray(env.sim.get_agent_state(0).position, dtype=np.float32)
        target = self._target_two
        current = env.current_episode
        current.episode_id = str(target.episode_id)
        current.goal_object_id = str(target.goal_object_id)
        current.goal_image_id = int(target.goal_image_id)
        current.object_category = target.object_category
        current.goals = target.goals
        current._shortest_path_cache = None
        for sensor in env.task.sensor_suite.sensors.values():
            if hasattr(sensor, "_current_episode_id"):
                sensor._current_episode_id = None
                sensor._current_image_goal = None
        raw = env.sim.get_observations_at()
        if raw is None:
            raise RuntimeError("Habitat returned no observation on target switch")
        observations = dict(raw)
        observations.update(env.task.sensor_suite.get_observations(
            observations=raw, episode=current, task=env.task))
        for action in env.task.actions.values():
            action.reset(episode=current, task=env.task)
        env.task._is_episode_active = True
        env._episode_start_time = time.time()
        env._elapsed_steps = 0
        env._episode_over = False
        env._episode_force_changed = False
        env._episode_from_iter_on_reset = True
        env.task.measurements.reset_measures(episode=current, task=env.task, observations=observations)

        self._target_index = 1
        self.update_after_reset()
        self.task_goal_idx = self.gt_goal_idx
        self.instance_imagegoal = observations["instance_imagegoal"]
        self.timestep, self.stopped, self.path_length = 0, False, 1e-5
        self.trajectory_states = []
        self.last_agent_location = self.get_agent_location()
        state = self.get_agent_pose()
        self.start_height = state["position"][1]
        self.agent_height = self.args.camera_height
        self.start_position, self.start_rotation = state["position"], state["rotation"]
        self.info = {"time": 0, "sensor_pose": [0., 0., 0.],
                     "goal_cat_id": self.gt_goal_idx, "goal_name": self.goal_name,
                     "agent_height": self.agent_height,
                     "instance_imagegoal": self.instance_imagegoal}
        self._attach(self.info)
        after = np.asarray(env.sim.get_agent_state(0).position, dtype=np.float32)
        return observations, self.info, {"pose_delta_m": float(np.linalg.norm(before - after))}


def make_multi_env(args):
    manifest, specs = _load_specs(args.multi_manifest, args.multi_episode_start,
                                  args.multi_num_episodes)
    source = manifest["source"]
    config = get_config(config_path=str(Path(args.unigoal_root) / "configs" / args.task_config))
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = source["split"]
    config.habitat.dataset.data_path = str(Path(args.assets_root) / source["dataset_root"] / "{split}/{split}.json.gz")
    config.habitat.dataset.scenes_dir = str(Path(args.assets_root) / source["scenes_root"])
    config.habitat.dataset.content_scenes = sorted({spec["scene_key"] for spec in specs})
    config.habitat.environment.iterator_options.shuffle = False
    config.habitat.environment.max_episode_steps = int(args.multi_goal_step_budget)
    config.habitat.environment.max_episode_seconds = 10000000
    config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(args.sim_gpu_id)
    for name in ("rgb_sensor", "depth_sensor", "semantic_sensor"):
        sensor = getattr(config.habitat.simulator.agents.main_agent.sim_sensors, name)
        sensor.width, sensor.height, sensor.hfov = args.env_frame_width, args.env_frame_height, args.hfov
        sensor.position = [0, args.camera_height, 0]
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = args.min_depth
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = args.max_depth
    config.habitat.simulator.agents.main_agent.height = args.camera_height
    OmegaConf.set_readonly(config, True)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    source_episodes = list(dataset.episodes)
    selected = []
    for spec in specs:
        first, second = spec["goals"]
        episode = copy.deepcopy(_source_episode(source_episodes, spec["scene_key"], first))
        episode.start_position, episode.start_rotation = list(spec["start_position"]), list(spec["start_rotation"])
        episode._multi_spec = copy.deepcopy(spec)
        episode._target_two = copy.deepcopy(_source_episode(source_episodes, spec["scene_key"], second))
        selected.append(episode)
    dataset.episodes = selected
    OmegaConf.set_readonly(config, False)
    config.habitat.simulator.scene = selected[0].scene_id
    OmegaConf.set_readonly(config, True)
    env = SequentialUniGoalEnv(args=args, config_env=config, dataset=dataset)
    env.seed(args.seed)
    return env
