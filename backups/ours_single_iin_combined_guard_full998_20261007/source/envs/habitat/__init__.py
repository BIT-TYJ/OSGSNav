# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api
import os
import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat import Env, RLEnv, VectorEnv, make_dataset

from agents.sem_exp import Sem_Exp_Env_Agent
from agents.instance_exp import Instance_Exp_Env_Agent
# from agents.train_exp import Instance_Exp_Env_Agent
# from agents.t_dis import Instance_Exp_Env_Agent
from .objectgoal_env import ObjectGoal_Env
from omegaconf import OmegaConf
from .utils.vector_env import VectorEnv
from tqdm import tqdm
# from train_agents import MCCAgent


def _apply_eval_episode_slice(args, dataset):
    habitat_episode_id = str(getattr(args, "eval_habitat_episode_id", "") or "")
    if getattr(args, "eval", 0) and habitat_episode_id:
        matches = [
            episode for episode in dataset.episodes
            if str(getattr(episode, "episode_id", "")) == habitat_episode_id
        ]
        if not matches:
            raise ValueError(
                "Habitat episode_id {} was not found in dataset episode ids".format(
                    habitat_episode_id
                )
            )
        if len(matches) > 1:
            raise ValueError(
                "Habitat episode_id {} matched {} dataset episodes; "
                "use internal eval_episode_start/end to disambiguate".format(
                    habitat_episode_id, len(matches)
                )
            )
        dataset.episodes = [matches[0]]
        return

    start = int(getattr(args, "eval_dataset_slice_start", 0) or 0)
    if not getattr(args, "eval", 0) or start <= 0:
        return

    end = int(getattr(args, "eval_dataset_slice_end", 0) or 0)
    num_episodes = int(getattr(args, "num_eval_episodes", 0) or 0)
    start_idx = start - 1
    end_idx = end if end > 0 else start_idx + num_episodes
    total = len(dataset.episodes)
    if start_idx < 0 or start_idx >= total:
        raise ValueError(
            "eval_episode_start {} is outside dataset episode count {}".format(
                start, total
            )
        )
    if end_idx <= start_idx:
        raise ValueError(
            "empty eval episode range: start={}, end={}".format(start, end)
        )
    if end_idx > total:
        raise ValueError(
            "eval episode end {} exceeds dataset episode count {}".format(
                end_idx, total
            )
        )
    dataset.episodes = list(dataset.episodes[start_idx:end_idx])


def make_env_fn(args, config_env, rank):
    dataset = make_dataset(config_env.habitat.dataset.type, config=config_env.habitat.dataset)
    _apply_eval_episode_slice(args, dataset)
    OmegaConf.set_readonly(config_env, False)
    config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    OmegaConf.set_readonly(config_env, True)

    if args.agent == "sem_exp":
        env = Instance_Exp_Env_Agent(args=args, rank=rank,
                                config_env=config_env,
                                dataset=dataset
                                )
    else:
        env = ObjectGoal_Env(args=args, rank=rank,
                             config_env=config_env,
                             dataset=dataset
                             )

    env.seed(rank)
    return env


def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            # scene = filename[: -len(scene_dataset_ext) + 4]
            scene = filename[: -len(scene_dataset_ext) ]
            scenes.append(scene)
    scenes.sort()
    return scenes


def construct_envs(args):
    env_configs = []
    args_list = []

    basic_config = cfg_env(config_path="envs/habitat/configs/"
                                         + args.task_config)
    OmegaConf.set_readonly(basic_config, False)
    basic_config.habitat.dataset.split = args.split

    OmegaConf.set_readonly(basic_config, True)

    dataset = make_dataset(basic_config.habitat.dataset.type)
    scenes = basic_config.habitat.dataset.content_scenes
    if "*" in basic_config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)



    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / args.num_processes))
                             for _ in range(args.num_processes)]
        for i in range(len(scenes) % args.num_processes):
            scene_split_sizes[i] += 1

    quiet_log = bool(getattr(args, "quiet_log", 0))
    if not quiet_log:
        print("Scenes per thread:")
    iterator = range(args.num_processes) if quiet_log else tqdm(range(args.num_processes))
    for i in iterator:
        config_env = cfg_env(config_path="envs/habitat/configs/"
                                           + args.task_config)
        OmegaConf.set_readonly(config_env, False)

        if len(scenes) > 0:
            contentss = scenes[
                sum(scene_split_sizes[:i]):
                sum(scene_split_sizes[:i + 1])
            ]

            bad_scense = []
            good_scense = [contentss[i] for i in range(len(contentss)) if contentss[i] not in bad_scense]

            config_env.habitat.dataset.content_scenes = good_scense
            if not quiet_log:
                print("Thread {}: {}".format(i, config_env.habitat.dataset.content_scenes))

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        config_env.habitat.environment.iterator_options.shuffle = False
        max_episode_length = int(getattr(args, "max_episode_length", 0) or 0)
        if max_episode_length > 0:
            # MinAV controls the cutoff and sends the authoritative STOP.
            # Keep Habitat's own limits behind that so it cannot end buffer /
            # localization episodes before the MinAV runner logic does.
            config_env.habitat.environment.max_episode_steps = max_episode_length + 50
            config_env.habitat.environment.max_episode_seconds = 10000000

        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.hfov
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = args.hfov
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = args.min_depth
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = args.max_depth
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = args.hfov
        config_env.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.agents.main_agent.height = args.camera_height

        config_env.habitat.dataset.split = args.split

        # config_env.freeze()
        OmegaConf.set_readonly(config_env, True)
        env_configs.append(config_env)

        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, range(args.num_processes))
            )
        ),
    )

    return envs
