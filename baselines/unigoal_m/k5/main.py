import os
import sys
import json
import logging
import time
import yaml
from collections import deque, defaultdict
from types import SimpleNamespace
import numpy as np
import torch
import argparse
from pathlib import Path
_REPO = Path(__file__).resolve().parents[3]
_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument('--unigoal_root', default=os.environ.get('UNIGOAL_ROOT', str(_REPO / 'external/UniGoal')))
_boot_args, _ = _boot.parse_known_args()
ROOT = str(Path(_boot_args.unigoal_root).expanduser().resolve())
if not (Path(ROOT) / 'src_unigoal').is_dir():
    raise RuntimeError('Set UNIGOAL_ROOT or --unigoal_root to the pinned UniGoal checkout; see baselines/unigoal_m/README.md')
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, ROOT)
from baselines.unigoal_m.k5.env import make_multi_env
from baselines.unigoal_m.k5.agent import SequentialUniGoalAgent
from baselines.unigoal_m.k5.state import (retarget_graph_preserving_memory,
                         save_terminal_global_maps)
from src_unigoal.map.bev_mapping import BEV_Map
from src_unigoal.graph.graph import Graph
import gzip
sys.path.append(str(Path(ROOT) / "third_party/Grounded-Segment-Anything/segment-anything"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=str(_REPO / "baselines/unigoal_m/configs/k5.yaml"),
                        metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--goal_type", default="ins-image", type=str)
    parser.add_argument("--episode_id", default=-1, type=int, help="episode id, 0~999")
    parser.add_argument("--goal", default="", type=str)
    parser.add_argument("--real_world", action="store_true")
    parser.add_argument(
        "--start_task_id",
        type=int,
        default=0,
        help="从第 i 个 episode 开始执行 (默认 0 表示从头开始)"
    )
    parser.add_argument("--multi_manifest", required=True)
    parser.add_argument("--multi_episode_start", default=0, type=int)
    parser.add_argument("--multi_num_episodes", default=1, type=int)
    parser.add_argument("--multi_goal_step_budget", default=1000, type=int)
    parser.add_argument("--assets_root", default=str(_REPO))
    parser.add_argument("--unigoal_root", default=ROOT)

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
    args = vars(args)
    args.update(config)
    # Launcher-only overrides keep the copied baseline config reusable while
    # allowing one local Ollama endpoint and result folder per GPU worker.
    for key, env_key in (("base_url", "UNIGOAL_OLLAMA_BASE_URL"),
                         ("dump_location", "UNIGOAL_DUMP_LOCATION"),
                         ("experiment_id", "UNIGOAL_EXPERIMENT_ID")):
        if os.environ.get(env_key):
            args[key] = os.environ[env_key]

    args = SimpleNamespace(**args)

    args.is_debugging = sys.gettrace() is not None
    if args.is_debugging:
        args.experiment_id = "debug"

    args.log_dir = os.path.join(args.dump_location, args.experiment_id, 'log')
    args.visualization_dir = os.path.join(args.dump_location, args.experiment_id, 'visualization')

    args.map_size = args.map_size_cm // args.map_resolution
    args.global_width, args.global_height = args.map_size, args.map_size
    args.local_width = int(args.global_width / args.global_downscaling)
    args.local_height = int(args.global_height / args.global_downscaling)

    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    args.num_scenes = args.num_processes
    args.num_episodes = int(args.multi_num_episodes)

    return args


def main():
    args = get_config()
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(Path(args.log_dir).parent, exist_ok=True)
    with open(Path(args.log_dir).parent / "resolved_args.json", "w") as f:
        json.dump(vars(args), f, default=str, indent=2)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, 'eval.log'),
        level=logging.INFO)
    logging.info(args)

    eval_metrics_id = 0

    episode_success = deque(maxlen=args.num_episodes)
    episode_spl = deque(maxlen=args.num_episodes)

    finished = False
    wait_env = False

    if args.goal_type == 'text':
        with gzip.open(args.text_goal_dataset, 'rt') as f:
            text_goal_dataset = json.load(f)

    BEV_map = BEV_Map(args)
    graph = Graph(args)
    envs = make_multi_env(args)
    agent = SequentialUniGoalAgent(args, envs)

    BEV_map.init_map_and_pose()
    obs, rgbd, infos = agent.reset()

    def get_logged_episode_id(info):
        if args.episode_id != -1:
            return int(info['episode_no'])
        return int(getattr(args, "start_task_id", 0)) + int(info['episode_no'])

    current_episode_id = infos['multi_episode_id']
    current_target = 0
    target_records = []
    switch_audits = []

    BEV_map.mapping(rgbd, infos)

    global_goals = [args.local_width // 2, args.local_height // 2]

    goal_maps = np.zeros((args.local_width, args.local_height))

    goal_maps[global_goals[0], global_goals[1]] = 1

    agent_input = {}
    agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
    agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
    agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
    agent_input['goal'] = goal_maps
    agent_input['exp_goal'] = goal_maps * 1
    agent_input['new_goal'] = 1
    agent_input['found_goal'] = 0
    agent_input['wait'] = wait_env or finished
    agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()
    if args.visualize:
        BEV_map.local_map[0, 10, :, :] = 1e-5
        agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :, :
                                            ].argmax(0).cpu().numpy()

    obs, rgbd, done, infos = agent.step(agent_input)

    graph.reset()
    graph.set_obj_goal(infos['goal_name'])
    if args.goal_type == 'ins-image':
        graph.set_image_goal(infos['instance_imagegoal'])
    elif args.goal_type == 'text':
        graph.set_text_goal(infos['text_goal'])

    step = 0
    target_step = 0
    force_replan = False

    while True:
        if finished == True:
            break

        global_step = (target_step // args.num_local_steps) % args.num_global_steps
        local_step = target_step % args.num_local_steps

        if done:
            completed_episode_id = current_episode_id
            spl = infos['spl']
            success = infos['success']
            success = success if success is not None else 0.0
            target_records.append({'episode_id': completed_episode_id,
                                   'target_index': current_target,
                                   'success': float(success), 'spl': float(spl),
                                   'steps': int(infos.get('time', target_step)),
                                   'goal_name': infos.get('goal_name'),
                                   'goal_object_id': infos.get('multi_goal_object_id'),
                                   'source_episode_id': infos.get('multi_source_episode_id')})
            if args.visualize:
                try:
                    save_terminal_global_maps(
                        BEV_map, agent,
                        os.path.join(args.dump_location, args.experiment_id),
                        completed_episode_id, current_target, success, spl)
                except Exception:
                    logging.exception("failed to save terminal global map")
            if current_target < 4:
                observations, switch_info, audit = envs.switch_multi_goal()
                if audit['pose_delta_m'] > 1e-5:
                    raise RuntimeError('target switch moved pose: {}'.format(audit))
                obs, rgbd, infos = agent.switch_target(observations, switch_info)
                current_target += 1
                retained_nodes = len(graph.nodes)
                retained_objects = len(graph.objects)
                retained_edges = len(graph.get_edges())
                retarget_graph_preserving_memory(graph)
                graph.set_obj_goal(infos['goal_name'])
                graph.set_image_goal(infos['instance_imagegoal'])
                # Deliberately retain BEV_map and simulator pose for target2.
                wait_env = False
                target_step = 0
                force_replan = True
                switch_audits.append({
                    'episode_id': completed_episode_id,
                    'pose_delta_m': float(audit['pose_delta_m']),
                    'retained_scene_nodes': retained_nodes,
                    'retained_scene_objects': retained_objects,
                    'retained_scene_edges': retained_edges,
                    'forced_immediate_replan': True,
                })
            else:
                eval_metrics_id += 1
                episode_success.append(success)
                episode_spl.append(spl)
                if len(episode_success) == args.num_episodes:
                    finished = True
                if args.visualize:
                    video_path = os.path.join(args.visualization_dir, 'eps_{}.mp4'.format(completed_episode_id))
                    agent.save_visualization(video_path)
                if not finished:
                    BEV_map.update_intrinsic_rew()
                    BEV_map.init_map_and_pose_for_env()
                    obs, rgbd, infos = agent.reset()
                    current_episode_id, current_target = infos['multi_episode_id'], 0
                    graph.reset()
                    graph.set_obj_goal(infos['goal_name'])
                    graph.set_image_goal(infos['instance_imagegoal'])
                    wait_env = True
                    target_step = 0
                    force_replan = True

        BEV_map.mapping(rgbd, infos)

        navigate_steps = global_step * args.num_local_steps + local_step
        graph.set_navigate_steps(navigate_steps)
        if not agent_input['wait'] and navigate_steps % 2 == 0:
            graph.set_observations(obs)
            graph.update_scenegraph()

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        replanned = (force_replan or local_step == args.num_local_steps - 1 or
                     np.linalg.norm(np.array([BEV_map.local_row, BEV_map.local_col]) -
                                    np.array(global_goals)) < 10)
        if replanned:
            if wait_env == True:
                wait_env = False
            else:
                BEV_map.update_intrinsic_rew()

            BEV_map.move_local_map()

            graph.set_full_map(BEV_map.full_map)
            graph.set_full_pose(BEV_map.full_pose)
            goal = graph.explore()
            if hasattr(graph, 'frontier_locations_16'):
                graph.frontier_locations_16[:, 0] = graph.frontier_locations_16[:, 0] - BEV_map.local_map_boundary[0, 0]
                graph.frontier_locations_16[:, 1] = graph.frontier_locations_16[:, 1] - BEV_map.local_map_boundary[0, 2]
            if isinstance(goal, list) or isinstance(goal, np.ndarray):
                goal = list(goal)
                goal[0] = goal[0] - BEV_map.local_map_boundary[0, 0]
                goal[1] = goal[1] - BEV_map.local_map_boundary[0, 2]
                if 0 <= goal[0] < args.local_width and 0 <= goal[1] < args.local_height:
                    global_goals = goal
            force_replan = False


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        found_goal = False
        goal_maps = np.zeros((args.local_width, args.local_height))

        goal_maps[global_goals[0], global_goals[1]] = 1

        exp_goal_maps = goal_maps.copy()

        agent_input = {}
        agent_input['map_pred'] = BEV_map.local_map[0, 0, :, :].cpu().numpy()
        agent_input['exp_pred'] = BEV_map.local_map[0, 1, :, :].cpu().numpy()
        agent_input['pose_pred'] = BEV_map.planner_pose_inputs[0]
        agent_input['goal'] = goal_maps
        agent_input['exp_goal'] = exp_goal_maps
        agent_input['new_goal'] = replanned
        agent_input['found_goal'] = found_goal
        agent_input['wait'] = wait_env or finished
        agent_input['sem_map'] = BEV_map.local_map[0, 4:11, :, :
                                        ].cpu().numpy()

        if args.visualize:
            BEV_map.local_map[0, 10, :, :] = 1e-5
            agent_input['sem_map_pred'] = BEV_map.local_map[0, 4:11, :,
                                                :].argmax(0).cpu().numpy()

        obs, rgbd, done, infos = agent.step(agent_input)

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # log
        if step % args.log_interval == 0:
            log = " ".join([
                "num timesteps {},".format(step),
                "episode_id {}".format(current_episode_id),
            ])

            total_success = []
            total_spl = []
            for acc in episode_success:
                total_success.append(acc)
            for spl in episode_spl:
                total_spl.append(spl)

            if len(total_spl) > 0:
                log += " Average SR/SPL:"
                log += " {:.5f}/{:.5f},".format(
                    np.mean(total_success),
                    np.mean(total_spl))

            print(log)
            logging.info(log)
        # ------------------------------------------------------------------
        step += 1
        target_step += 1

    total_success = []
    total_spl = []
    for acc in episode_success:
        total_success.append(acc)
    for spl in episode_spl:
        total_spl.append(spl)

    if len(total_spl) > 0:
        log = "Average SR/SPL:"
        log += " {:.5f}/{:.5f},".format(
            np.mean(total_success),
            np.mean(total_spl))

    print(log)
    logging.info(log)

    total = {'succ': total_success, 'spl': total_spl, 'targets': target_records,
             'switch_audits': switch_audits,
             'protocol': 'SequentialOrderedMultiIIN'}

    with open('{}/total.json'.format(
            args.log_dir), 'w') as f:
        json.dump(total, f)


if __name__ == "__main__":
    main()
