import argparse
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_args():
    parser = argparse.ArgumentParser(
        description='Goal-Oriented-Semantic-Exploration')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument(
        '--multi_manifest',
        type=str,
        default='multi_iin_v0/manifests/miin_v0_val_k2.json',
        help='Multi-IIN v0 manifest consumed only by the independent runner',
    )
    parser.add_argument(
        '--multi_variant',
        type=str,
        default='Continuous-Shared',
        choices=['Independent-Reset', 'Continuous-Clear', 'Continuous-Shared'],
    )
    parser.add_argument('--multi_episode_start', type=int, default=0)
    parser.add_argument('--multi_num_episodes', type=int, default=1)
    parser.add_argument(
        '--multi_goal_step_budget', type=int, default=1000,
        help='maximum navigation steps for each ordered target (default: 1000)',
    )
    parser.add_argument(
        '--multi_gaussian_capacity_enabled', type=int, default=1,
        help='1: enable target-aware watermarked Gaussian capacity pruning',
    )
    parser.add_argument(
        '--multi_gaussian_hard_cap', type=int, default=1500000,
        help='absolute Gaussian admission cap for the additive Multi-IIN mapper',
    )
    parser.add_argument(
        '--multi_gaussian_prune_trigger', type=int, default=1350000,
        help='run capacity pruning at or above this Gaussian count',
    )
    parser.add_argument(
        '--multi_gaussian_prune_target', type=int, default=1100000,
        help='target Gaussian count after one capacity-pruning event',
    )
    parser.add_argument(
        '--multi_gaussian_capacity_voxel_size_m', type=float, default=0.03,
        help='voxel size used to identify spatially redundant Gaussians',
    )
    parser.add_argument(
        '--multi_gaussian_capacity_max_voxel_size_m', type=float, default=0.10,
        help='largest adaptive voxel size used when preserving every 3 cm cell cannot meet the cap',
    )
    parser.add_argument(
        '--multi_gaussian_dense_voxel_threshold', type=int, default=4,
        help='Gaussian count above which a voxel is prioritized for pruning',
    )
    parser.add_argument(
        '--multi_gaussian_recent_keyframe_grace', type=int, default=5,
        help='protect Gaussians from this many most recent keyframes',
    )
    parser.add_argument(
        '--multi_gaussian_semantic_protect_threshold', type=float, default=0.70,
        help='protect target-semantic Gaussians at or above this score',
    )
    parser.add_argument(
        '--multi_gaussian_low_opacity_threshold', type=float, default=0.05,
        help='prioritize non-protected Gaussians below this opacity for pruning',
    )
    parser.add_argument(
        '--multi_save_repeated_full_gaussian_snapshots', type=int, default=0,
        help='1: retain repeated confirmed/failed full-scene debug PLY snapshots',
    )
    parser.add_argument(
        '--multi_instance_filter', type=int, default=1,
        help='1: filter semantic Gaussian instances before NVSM rendering',
    )
    parser.add_argument('--multi_instance_filter_voxel_size_m', type=float, default=0.03)
    parser.add_argument('--multi_instance_filter_eps_m', type=float, default=0.10)
    parser.add_argument('--multi_instance_filter_min_samples', type=int, default=5)
    parser.add_argument('--multi_instance_filter_min_cluster_points', type=int, default=50)
    parser.add_argument('--multi_instance_filter_min_keep_ratio', type=float, default=0.02)
    parser.add_argument(
        '--multi_instance_filter_anchor_neighbourhood_slack_m',
        type=float,
        default=0.25,
        help=(
            'after finding the cluster nearest to the SOIM anchor, allow clusters '
            'within this additional XZ distance to compete by support size'
        ),
    )
    parser.add_argument('--multi_failure_soft_threshold', type=int, default=3,
                        help='failed confirmations near one instance before additive blacklisting')
    parser.add_argument('--multi_failure_hard_threshold', type=int, default=5,
                        help='reserved persistent-rejection threshold')
    parser.add_argument('--multi_failure_soft_cooldown_steps', type=int, default=80,
                        help='steps for which a failed instance remains blacklisted')
    parser.add_argument('--multi_failure_adaptive_radius_enabled', type=int, default=1,
                        help='use category-aware XZ association radii')
    parser.add_argument('--multi_frontier_recovery_enabled', type=int, default=1,
                        help='recover frontiers in the robot connected component when FMM rejects all')
    parser.add_argument('--multi_frontier_recovery_patience', type=int, default=3,
                        help='consecutive all-unreachable frontier calls before recovery is attempted')
    parser.add_argument('--multi_frontier_recovery_projection_max_px', type=float, default=15.0,
                        help='maximum grid distance for projecting a disconnected representative')
    parser.add_argument('--multi_frontier_patrol_fallback', type=int, default=1,
                        help='when no unknown frontier remains, revisit the farthest reachable cell instead of idling')
    parser.add_argument('--multi_phase3_respect_start_step', type=int, default=1,
                        help='do not enter or stop from Phase 3 before its fixed start step')
    parser.add_argument('--multi_phase3_terminal_requires_confirmed_evidence', type=int, default=1,
                        help='require threshold-level multi-view evidence for fallback stop')
    parser.add_argument('--multi_phase3_identity_first', type=int, default=1,
                        help='select the strongest historical target identity before reachability')
    parser.add_argument('--multi_phase3_target_frontier', type=int, default=1,
                        help='guide frontier exploration toward strong but currently unreachable evidence')
    parser.add_argument('--multi_phase3_frontier_refresh_steps', type=int, default=10)
    parser.add_argument('--multi_phase3_frontier_gain_radius_m', type=float, default=0.75)
    parser.add_argument('--multi_phase3_frontier_direction_band_m', type=float, default=1.50)
    parser.add_argument('--multi_phase3_frontier_min_start_m', type=float, default=0.50)
    parser.add_argument('--multi_stable_detection_memory', type=int, default=1,
                        help='accumulate spatially stable low-confidence plant/toilet observations')
    parser.add_argument('--multi_stable_detection_min_confidence', type=float, default=0.18)
    parser.add_argument('--multi_stable_detection_min_observations', type=int, default=2)
    parser.add_argument('--multi_stable_detection_max_age_steps', type=int, default=12)
    parser.add_argument('--multi_stable_detection_radius_m', type=float, default=0.75)
    parser.add_argument(
        '--multi_result_json',
        type=str,
        default='multi_iin_v0/results/real_smoke.json',
    )
    parser.add_argument(
        '--multi_save_compact_results', type=int, default=1,
        help='1: additionally archive only terminal, human-readable result artefacts',
    )
    parser.add_argument(
        '--multi_compact_results_dir', type=str,
        default='multi_iin_v0/results/compact',
        help='compact archive root; relative paths are resolved under the IEVE repo root',
    )
    parser.add_argument('--auto_gpu_config', type=int, default=0)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=5,
                        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""")
    parser.add_argument('--num_processes_per_gpu', type=int, default=6)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=1)
    parser.add_argument('--eval', type=int, default=1,
                        help='0: Train, 1: Evaluate (default: 0)')
    parser.add_argument('--num_training_frames', type=int, default=10000000,
                        help='total number of training frames')
    parser.add_argument('--num_eval_episodes', type=int, default=125,
                        help="number of test episodes per scene")
    parser.add_argument('--eval_resume_log', type=str, default="",
                        help="resume evaluation from the latest metric line in this log")
    parser.add_argument('--eval_skip_episodes', type=int, default=-1,
                        help="episodes to skip before evaluation; -1 uses eval_resume_log count")
    parser.add_argument('--eval_habitat_episode_id', type=str, default="",
                        help="official Habitat episode_id to evaluate; empty disables explicit id selection")
    parser.add_argument('--eval_episode_start', type=int, default=0,
                        help="internal 1-based eval slice start; 0 disables explicit range")
    parser.add_argument('--eval_episode_end', type=int, default=0,
                        help="internal 1-based inclusive eval slice end; 0 keeps num_eval_episodes")
    parser.add_argument('--num_train_episodes', type=int, default=10000,
                        help="""number of train episodes per scene
                                before loading the next scene""")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=-1,
                        help="""gpu id for semantic model,
                                -1: same as sim gpu, -2: cpu""")

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--quiet_log', type=int, default=0,
                        help='1: suppress non-metric stdout prints')
    parser.add_argument('--result_only_log', type=int, default=0,
                        help='1: in eval, print only one-line InstanceImageNav metrics to stdout')
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=50000,
                        help='Model save frequency in number of updates')
    parser.add_argument('--load', type=str, default="pretrained_models/sem_exp.pth",
    # parser.add_argument('--load', type=str, default="pretrained_models/periodic_1250000.pth",
    # parser.add_argument('--load', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=1,
                        help='1: save visualization as images')
    parser.add_argument('--vis_observation_masks', type=int, default=1,
                        help='1: overlay detector instance masks on the observation panel')
    parser.add_argument('--vis_observation_mask_threshold', type=float, default=-1.0,
                        help='minimum confidence for observation mask overlay; <0 uses sem_pred_prob_thr')
    parser.add_argument('--vis_observation_target_only', type=int, default=0,
                        help='1: overlay only the current target category masks')
    parser.add_argument('--target_instance_debug_dump', type=int, default=0,
                        help='1: save per-frame all/target instance mask debug PNGs and JSON')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=480,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=640,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=120,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=160,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=1000,
                        help="""Maximum episode length""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/instance_imagenav.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="val",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('--camera_height', type=float, default=0.88,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.00,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=30.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--success_dist', type=float, default=1.0,
                        help="success distance threshold in meters")
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor threshold in cm")
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")
    parser.add_argument('--version', type=str, default="v3",
                        help="dataset version")

    # Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--exploration_policy', type=str, default="frontier",
                        choices=["rl", "frontier"],
                        help="""high-level exploration goal policy:
                                rl uses the current learned global policy,
                                frontier uses IEVE frontier exploration""")
    parser.add_argument('--frontier_candidate_refresh_interval', type=int, default=5,
                        help='refresh frontier candidate points and visualization every N steps')
    parser.add_argument('--frontier_goal_hold_steps', type=int, default=10,
                        help='hold the selected frontier goal for N steps unless reached/blocked/stuck')
    parser.add_argument('--frontier_debug_log', type=int, default=1,
                        help='1: print frontier candidate/selected-goal debug lines')
    parser.add_argument('--frontier_initial_block_radius_m', type=float, default=2.3,
                        help='permanently ignore frontier goals within this many meters of the episode initial position')
    parser.add_argument('--frontier_visited_path_block_radius_m', type=float, default=0.5,
                        help='ignore frontier goals within this many meters of the visited path')
    parser.add_argument('--frontier_safe_obstacle_inflation_cells', type=float, default=4.0,
                        help='frontier safety clearance from obstacles in map cells')
    parser.add_argument('--frontier_band_radius_cells', type=float, default=4.0,
                        help='dilate raw frontier edge into an inner safe frontier band by this many cells')
    parser.add_argument('--frontier_cluster_dilation_radius', type=float, default=2.0,
                        help='dilation radius in cells before connected-component frontier clustering')
    parser.add_argument('--frontier_cluster_min_size', type=int, default=10,
                        help='minimum safe frontier-band cells required to keep a cluster')
    parser.add_argument('--frontier_min_representatives', type=int, default=2,
                        help='minimum representative points sampled per kept frontier cluster')
    parser.add_argument('--frontier_max_representatives', type=int, default=4,
                        help='maximum representative points sampled per kept frontier cluster')
    parser.add_argument('--frontier_min_goal_dist_px', type=float, default=10.0,
                        help='minimum FMM-selected frontier goal distance from robot in map cells')
    parser.add_argument('--frontier_cooldown_radius_px', type=float, default=8.0,
                        help='skip frontier goals within this cell radius of recently reached goals')
    parser.add_argument('--frontier_max_fmm_dist_px', type=float, default=300.0,
                        help='maximum allowed FMM path cost to a frontier candidate in map cells')
    parser.add_argument('--frontier_max_detour_ratio', type=float, default=5.0,
                        help='maximum allowed FMM path cost divided by Euclidean distance')
    parser.add_argument('--frontier_min_detour_dist_px', type=float, default=30.0,
                        help='minimum FMM distance before applying the detour-ratio filter')
    parser.add_argument('--frontier_select_by_fmm_cost', type=int, default=1,
                        help='1: return only the lowest-FMM-cost frontier representative as long-term goal')
    parser.add_argument(
        '--multi_fmm_exact_cache', type=int, default=1,
        help='1: reuse an FMM distance field only when traversible and goal grids are exactly unchanged',
    )
    parser.add_argument(
        '--multi_minav_reuse_frame_maps', type=int, default=1,
        help='1: share one immutable per-frame IEVE map snapshot between MinAV update and goal override',
    )
    parser.add_argument(
        '--multi_frontier_information_gain_enabled', type=int, default=1,
        help='1: rank reachable frontiers by nearby unknown cells before FMM cost',
    )
    parser.add_argument(
        '--multi_frontier_information_radius_m', type=float, default=1.5,
        help='radius around each frontier used to estimate unknown-area gain',
    )
    parser.add_argument(
        '--multi_frontier_fmm_cost_weight', type=float, default=0.20,
        help='normalized FMM path-cost weight in information-gain utility',
    )
    parser.add_argument(
        '--multi_frontier_visited_ratio_penalty', type=float, default=0.25,
        help='penalty for already visited cells near a frontier candidate',
    )
    parser.add_argument('--gaussmi_exploration_enabled',
                        '--phase1_gaussmi_enabled',
                        dest='phase1_gaussmi_enabled', type=int, default=1,
                        help='1: augment Phase-1 frontier exploration with fixed-position GauSS-MI gaze selection')
    parser.add_argument('--phase1_gaussmi_trigger_move_steps', type=int, default=25,
                        help='successful translation frames between Phase-1 GauSS-MI gaze selections')
    parser.add_argument('--phase1_gaussmi_move_eps_m', type=float, default=0.03,
                        help='minimum measured horizontal displacement counted as one movement step')
    parser.add_argument('--phase1_gaussmi_yaw_offsets', type=float, nargs='+',
                        default=[-60., 0., 60.],
                        help='relative yaw candidates in degrees at the current position')
    parser.add_argument('--phase1_gaussmi_angle_penalty_k', type=float, default=0.5)
    parser.add_argument('--phase1_gaussmi_theta_max_deg', type=float, default=90.0)
    parser.add_argument('--phase1_gaussmi_target_weight', type=float, default=2.0)
    parser.add_argument('--phase1_gaussmi_nearby_weight', type=float, default=1.5)
    parser.add_argument('--phase1_gaussmi_nearby_radius_m', type=float, default=2.0)
    parser.add_argument('--phase1_gaussmi_yaw_tolerance_deg', type=float, default=5.0)
    parser.add_argument('--phase1_gaussmi_min_frames', type=int, default=5)
    parser.add_argument('--phase1_gaussmi_min_gaussians', type=int, default=10000)
    parser.add_argument('--phase1_gaussmi_reliability_delta', type=float, default=0.02)
    parser.add_argument('--phase1_gaussmi_min_updated_ratio', type=float, default=0.10)
    parser.add_argument('--phase1_gaussmi_min_valid_candidates', type=int, default=3)
    parser.add_argument('--phase1_gaussmi_max_gaussians', type=int, default=750000,
                        help='disable costly exploration yaw scoring above this map size; <=0 disables the cap')
    parser.add_argument('--phase1_gaussmi_min_info_gain', type=float, default=0.0)
    parser.add_argument('--phase1_gaussmi_min_relative_gain', type=float, default=0.0)
    parser.add_argument('--phase1_gaussmi_debug', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='global_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')
    parser.add_argument('--num_local_steps', type=int, default=10,
                        help="""Number of steps the local policy
                                between each global step""")
    parser.add_argument('--reward_coeff', type=float, default=0.1,
                        help="Object goal reward coefficient")
    parser.add_argument('--intrinsic_rew_coeff', type=float, default=0.02,
                        help="intrinsic exploration reward coefficient")
    parser.add_argument('--num_sem_categories', type=float, default=16)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.50,
                        help="Semantic prediction confidence threshold")
    parser.add_argument('--goal_cat_source', type=str, default="oracle",
                        choices=["detector", "classifier", "oracle"],
                        help="""goal category source for InstanceImageNav:
                                detector infers the class from the goal image,
                                classifier uses pretrained_models/cla3_model.pth,
                                oracle reads the benchmark task object_category (Multi-IIN default)""")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=3)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=3600)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.2)

    # 3D Gaussian Splatting Mapping
    parser.add_argument('--enable_3dgs', dest='enable_3dgs', type=int, default=0,
                        help='1: enable online 3D Gaussian Splatting mapping')
    parser.add_argument('--3dgs_config', dest='gs_config', type=str,
                        default='gaussian_mapping/configs/base_config.yaml',
                        help='3DGS mapping config yaml')
    parser.add_argument('--3dgs_update_interval', dest='gs_update_interval',
                        type=int, default=1,
                        help='train/update 3DGS every N env steps')
    parser.add_argument('--3dgs_exploration_update_interval',
                        dest='gs_exploration_update_interval',
                        type=int, default=1,
                        help='train/update 3DGS every N env steps while exploring')
    parser.add_argument('--3dgs_save_interval', dest='gs_save_interval',
                        type=int, default=0,
                        help='save 3DGS checkpoint every N env steps; <=0 disables')
    parser.add_argument('--3dgs_output_dir', dest='gs_output_dir', type=str,
                        default='3dgs',
                        help='relative output directory under dump/<exp_name>/')
    parser.add_argument('--3dgs_max_envs', dest='gs_max_envs', type=int,
                        default=-1,
                        help='-1: all vector envs; otherwise first K envs only')
    parser.add_argument('--3dgs_render_debug', dest='gs_render_debug', type=int,
                        default=1,
                        help='1: save periodic 3DGS render/PSNR debug images')
    parser.add_argument('--3dgs_save_full_ply_each_frame',
                        dest='gs_save_full_ply_each_frame',
                        type=int, default=0,
                        help='1: save a complete colored 3DGS PLY after every 3DGS frame update')
    parser.add_argument('--3dgs_highlight_confirmed_target',
                        dest='gs_highlight_confirmed_target',
                        type=int, default=1,
                        help='1: color confirmed target Gaussians differently in complete per-frame PLY exports')
    parser.add_argument('--3dgs_semantic_guided_optimization_enabled',
                        dest='gs_semantic_guided_optimization_enabled',
                        type=int, default=1,
                        help='1: use semantic replay, target-mask loss, and target-focused 3DGS refresh training')
    parser.add_argument('--multi_gaussian_quality_refinement', type=int, default=1,
                        help='1: run sparse current-view refinement when an exploration render has a large dark hole')
    parser.add_argument('--multi_gaussian_quality_excess_black_threshold', type=float, default=0.02,
                        help='fraction of visible RGB pixels rendered nearly black before adaptive refinement')
    parser.add_argument('--multi_gaussian_quality_darkening_threshold', type=float, default=0.10,
                        help='strongly darkened pixel fraction before adaptive refinement')
    parser.add_argument('--multi_gaussian_quality_extra_current_iters', type=int, default=6,
                        help='current-keyframe-only iterations used after the quality gate triggers')
    parser.add_argument('--multi_gaussian_quality_min_visible_fraction', type=float, default=0.20,
                        help='minimum valid bright RGB fraction required by the quality gate')
    parser.add_argument('--multi_gaussian_quality_log_all_checks', type=int, default=0,
                        help='1: persist every quality check; 0: persist trigger events only')
    parser.add_argument('--multi_gaussian_quality_rollback_on_regression', type=int, default=1,
                        help='1: reject and restore adaptive 3DGS updates that regress observable RGB quality')
    parser.add_argument('--multi_gaussian_quality_max_psnr_drop_db', type=float, default=0.25,
                        help='largest tolerated current-view PSNR drop before adaptive-update rollback')
    parser.add_argument('--multi_gaussian_quality_max_excess_black_increase', type=float, default=0.002,
                        help='largest tolerated excess-black fraction increase before rollback')
    parser.add_argument('--multi_gaussian_quality_max_darkening_increase', type=float, default=0.01,
                        help='largest tolerated strong-darkening fraction increase before rollback')
    parser.add_argument('--multi_gaussian_quality_history_validation_views', type=int, default=3,
                        help='recent held-out keyframes checked for regression after adaptive refinement')
    parser.add_argument('--multi_gaussian_quality_history_validation_stride', type=int, default=5,
                        help='keyframe stride between recent held-out validation views')

    # MinAV-style target confirmation/localization replacement
    parser.add_argument('--use_minav_confirmation_localization', type=int, default=0,
                        help='1: use MinAV-style strict Gaussian target confirmation/localization')
    parser.add_argument('--minav_target_detect_confidence', type=float, default=0.3,
                        help='minimum detector confidence for target pre-confirmation frames')
    parser.add_argument('--minav_target_detect_window', type=int, default=2,
                        help='consecutive target detection frames required before Gaussian confirmation')
    parser.add_argument('--minav_use_grounded_sam_segmenter', type=int, default=0,
                        help='1: use original MinAV GroundedSAM segmenter instead of IEVE Mask R-CNN')
    parser.add_argument('--minav_confirm_min_support', type=int, default=2,
                        help='reserved support count for MinAV confirmation compatibility')
    parser.add_argument('--minav_confirm_min_gaussians', type=int, default=200,
                        help='minimum total Gaussian count before target confirmation may pass')
    parser.add_argument('--minav_confirm_match_threshold', type=float, default=20.0,
                        help='FEM/LightGlue match threshold for distant Gaussian confirmation')
    parser.add_argument('--minav_confirm_candidate_top_k', type=int, default=0,
                        help='max target-confirmation Gaussian candidates to render; 0 renders all candidates')
    parser.add_argument('--minav_confirm_render_views', type=int, default=16,
                        help='number of NVSM render views per scale for distant target confirmation')
    parser.add_argument('--minav_confirm_render_radius', type=float, default=1.0,
                        help='base NVSM render radius in meters for distant target confirmation')
    parser.add_argument('--multi_confirm_coarse_views_per_scale', type=int, default=8,
                        help='views per scale for the low-cost first pass of ordinary target confirmation')
    parser.add_argument('--multi_confirm_coarse_min_score', type=float, default=5.0,
                        help='minimum coarse FEM score before an ordinary confirmation candidate gets full rendering')
    parser.add_argument('--multi_confirm_full_candidate_limit', type=int, default=0,
                        help='maximum ordinary confirmation candidates receiving the full view budget; 0 disables the limit')
    parser.add_argument('--minav_confirm_mask_dilate_pixels', type=int, default=3,
                        help='target-confirm instance mask dilation before Gaussian projection')
    parser.add_argument('--minav_confirm_merge_dist_m', type=float, default=0.35,
                        help='merge target-confirm candidates across frames within this XZ distance')
    parser.add_argument('--minav_confirm_merge_overlap_ratio', type=float, default=0.25,
                        help='merge target-confirm candidates across frames above this Gaussian overlap ratio')
    parser.add_argument('--minav_confirm_expand_full_clusters', type=int, default=0,
                        help='1: expand projected seed Gaussians to full current target-semantic clusters')
    parser.add_argument('--minav_confirm_support_fallback_enabled', type=int, default=1,
                        help='1: allow single-frame high-confidence target-confirm fallback candidates')
    parser.add_argument('--minav_confirm_max_detection_dist_m', type=float, default=10.0,
                        help='max XZ distance from bbox-derived detection position for target-confirm candidates')
    parser.add_argument('--minav_confirm_camera_height', type=float, default=0.88,
                        help='camera height used by MinAV-style bbox-depth target position projection')
    parser.add_argument('--minav_confirm_min_locked_gaussians', type=int, default=30,
                        help='minimum Gaussian count for a target-confirm candidate; <=0 disables')
    parser.add_argument('--minav_confirm_max_locked_gaussians', type=int, default=50000,
                        help='optional max Gaussian count for a confirmed target cluster; 0 disables')
    parser.add_argument('--minav_confirm_max_xz_extent_m', type=float, default=1.20,
                        help='optional max horizontal extent for target-confirm clusters; 0 disables')
    parser.add_argument('--minav_confirm_max_y_extent_m', type=float, default=1.40,
                        help='optional max vertical extent for target-confirm clusters; 0 disables')
    parser.add_argument('--minav_localization_match_threshold', type=float, default=40.0,
                        help='LightGlue/DISK match threshold for close localization')
    parser.add_argument('--minav_cluster_recovery_enabled', type=int, default=1,
                        help='1: run one target-cluster recovery orbit after REF when TC/FEM/REF evidence is exceptionally high')
    parser.add_argument('--minav_cluster_recovery_trigger_score', type=float, default=100.0,
                        help='strict TC/FEM/REF evidence threshold; any score above it requests cluster recovery')
    parser.add_argument('--minav_cluster_recovery_max_radius_m', type=float, default=3.0,
                        help='maximum XZ distance from every recovery FMM path node to the original cluster center')
    parser.add_argument('--minav_cluster_recovery_waypoints', type=int, default=12,
                        help='angular guide count used to obtain an obstacle-shaped closed FMM route')
    parser.add_argument('--minav_cluster_recovery_goal_dilation_radius', type=int, default=2,
                        help='FMM goal dilation radius in cells for cluster-recovery waypoints')
    parser.add_argument('--minav_cluster_recovery_timeout', type=int, default=200,
                        help='maximum action frames allowed for one cluster-recovery orbit')
    parser.add_argument('--minav_cluster_recovery_replan_stall_frames', type=int, default=20,
                        help='frames without waypoint-distance progress before replanning a complete closed orbit')
    parser.add_argument('--minav_cluster_recovery_replan_progress_epsilon_m', type=float, default=0.05,
                        help='minimum waypoint-distance improvement that resets the cluster-recovery stall counter')
    parser.add_argument('--minav_cluster_recovery_max_replans', type=int, default=3,
                        help='maximum complete closed-orbit replans during one cluster-recovery attempt')
    parser.add_argument('--minav_cluster_recovery_approach_inner_radius_m', type=float, default=2.5,
                        help='inner map radius for a reachable ingress when recovery starts outside 3m')
    parser.add_argument('--minav_cluster_recovery_approach_outer_radius_m', type=float, default=2.7,
                        help='outer map radius for a reachable ingress when recovery starts outside 3m')
    parser.add_argument('--minav_cluster_recovery_cell_arrival_radius_m', type=float, default=0.15,
                        help='map-cell arrival radius for recovery ingress and orbit waypoints')
    parser.add_argument('--minav_cluster_recovery_failure_cluster_center_tolerance_m', type=float, default=0.25,
                        help='maximum center drift when matching the same cached Gaussian cluster')
    parser.add_argument('--minav_cluster_recovery_failure_cluster_voxel_size_m', type=float, default=0.10,
                        help='voxel size used to build pruning-stable Gaussian-cluster failure fingerprints')
    parser.add_argument('--minav_cluster_recovery_failure_cluster_overlap_ratio', type=float, default=0.60,
                        help='minimum voxel overlap required to treat a confirmed cluster as the cached failed cluster')
    parser.add_argument('--minav_cluster_recovery_failure_map_change_ratio', type=float, default=0.05,
                        help='fraction of the nearby explored/obstacle map that must change before retrying a cached route failure')
    parser.add_argument('--minav_cluster_recovery_min_remaining_steps', type=int, default=50,
                        help='minimum episode steps that must remain before starting cluster recovery')
    parser.add_argument('--minav_cluster_recovery_yaw_range_deg', type=float, default=45.0,
                        help='candidate rendering yaw range on each side of the strongest original view')
    parser.add_argument('--minav_cluster_recovery_yaw_step_deg', type=float, default=5.0,
                        help='candidate rendering yaw interval; defaults produce 19 views from -45 to +45 degrees')
    parser.add_argument('--minav_buffer_distance_threshold', type=float, default=1.5,
                        help='distance in meters for entering close localization after buffer navigation')
    parser.add_argument('--minav_phase2_arrival_threshold', type=float, default=0.35,
                        help='distance in meters for arriving at GaussMI/NBV phase-2 goal')
    parser.add_argument('--minav_targeted_quick_lock_max_dist_m', type=float, default=1.5,
                        help='maximum distance from a targeted semantic instance to the quick-locked Gaussian cluster; 0 disables')
    parser.add_argument('--minav_gaussmi_search_radius', type=float, default=1.5,
                        help='search radius in meters for MinAV-style semantic GaussMI NBV')
    parser.add_argument('--minav_gaussmi_num_candidates', type=int, default=6,
                        help='number of semantic NBV candidates sampled around the target')
    parser.add_argument('--minav_buffer_goal_search_radius', type=float, default=2.5,
                        help='search radius in meters for MinAV-style reachable Buffer-Fix goals')
    parser.add_argument('--minav_buffer_goal_max_extra_radius', type=float, default=0.25,
                        help='maximum extra meters beyond minav_buffer_distance_threshold allowed for Buffer-Fix standoff goals')
    parser.add_argument('--minav_buffer_standoff_min_radius', type=float, default=1.0,
                        help='minimum target-to-standpoint radius in meters for Buffer-Fix standoff goals')
    parser.add_argument('--minav_buffer_standoff_max_radius', type=float, default=1.3,
                        help='maximum target-to-standpoint radius in meters for Buffer-Fix standoff goals')
    parser.add_argument('--minav_buffer_cached_goal_recheck_radius', type=float, default=0.10,
                        help='max adjustment radius in meters when rechecking cached buffer goals')
    parser.add_argument('--minav_buffer_sweep_steps', type=int, default=6,
                        help='deprecated compatibility flag; direct buffer pre-localization sweep is disabled')
    parser.add_argument('--minav_buffer_standoff_sweep', type=int, default=1,
                        help='1: run a full in-place sweep after direct buffer reaches the standoff goal')
    parser.add_argument('--minav_buffer_standoff_sweep_steps', type=int, default=0,
                        help='turn steps for direct buffer standoff sweep; 0 uses ceil(360 / turn_angle)')
    parser.add_argument('--minav_buffer_phase2_refine_frames', type=int, default=12,
                        help='target-focused keyframe window used for buffer phase-2 pre-localization refresh')
    parser.add_argument('--minav_initial_spin', type=int, default=1,
                        help='1: force an initial 360 degree in-place turn before exploration')
    parser.add_argument('--initial_spin_steps', type=int, default=0,
                        help='initial in-place turn steps per episode; 0 uses ceil(360 / turn_angle)')
    parser.add_argument('--minav_mr_remap_sweep_steps', type=int, default=12,
                        help='number of non-terminal turn actions for MR closed-loop remap sweep')
    parser.add_argument('--minav_buffer_timeout', type=int, default=150,
                        help='maximum BUFFER frames before rejecting a confirmation')
    parser.add_argument('--minav_buffer_soim_wait_timeout', type=int, default=15,
                        help='maximum post-GaussMI frames allowed while full localizer remains in SOIM')
    parser.add_argument('--minav_localization_timeout', type=int, default=80,
                        help='maximum localization frames after GaussMI ready')
    parser.add_argument('--minav_phase2_no_goal_timeout', type=int, default=12,
                        help='maximum consecutive frames without a reachable GaussMI/NBV goal')
    parser.add_argument('--minav_phase2_standoff_retry_limit', type=int, default=3,
                        help='maximum same-instance standoff resamples before phase2 switches to the next instance')
    parser.add_argument('--minav_phase2_standoff_stuck_patience', type=int, default=8,
                        help='consecutive stuck/FMM-stop frames before phase2 resamples the current standoff')
    parser.add_argument('--minav_phase2_standoff_retry_block_radius', type=float, default=0.45,
                        help='meters around failed phase2 standoff points to avoid when resampling')
    parser.add_argument('--minav_phase2_no_reachable_cooldown_steps', type=int, default=40,
                        help='frontier exploration cooldown frames after Phase 2 finds no reachable semantic instance')
    parser.add_argument('--minav_phase2_no_reachable_retry_limit', type=int, default=3,
                        help='number of Phase 2 no-reachable retries before entering Phase 3 early')
    parser.add_argument('--minav_phase3_no_candidate_cooldown_steps', type=int, default=40,
                        help='frontier exploration frames before retrying Phase 3 when no historical candidate is reachable')
    parser.add_argument('--minav_phase3_navigation_stuck_patience', type=int, default=20,
                        help='near-target no-progress frames before Phase 3 navigation enables its small stop tolerance')
    parser.add_argument('--minav_phase3_navigation_progress_epsilon_m', type=float, default=0.02,
                        help='minimum Phase 3 target-distance improvement that resets the stuck counter')
    parser.add_argument('--minav_phase3_navigation_stuck_tolerance_m', type=float, default=0.1,
                        help='extra Phase 3 stop radius enabled only after near-target navigation remains stuck')
    parser.add_argument('--minav_navigation_arrival_threshold', type=float, default=1.0,
                        help='manager-side navigation-complete threshold in meters')
    parser.add_argument('--minav_navigation_preferred_stop_radius', type=float, default=0.5,
                        help='preferred successful stop radius for localized-target navigation')
    parser.add_argument('--minav_navigation_staging_max_radius', type=float, default=2.0,
                        help='maximum target-centered radius for localized-navigation staging goals')
    parser.add_argument('--minav_phase3_navigation_staging_max_radius', type=float, default=1.2,
                        help='maximum target-centered radius for Phase-3 staging goals')
    parser.add_argument('--minav_navigation_reachable_ring_step', type=float, default=0.25,
                        help='outward search increment in meters for reachable navigation goals')
    parser.add_argument('--minav_navigation_staging_replan_limit', type=int, default=1,
                        help='target-facing replans after reaching a staging goal before fallback STOP')
    parser.add_argument('--minav_navigation_execution_goal_dilation_radius', type=int, default=1,
                        help='low-level FMM dilation cells around a selected reachable execution goal')
    parser.add_argument('--multi_navigation_official_stop_guard', type=int, default=0,
                        help='deprecated compatibility flag; evaluator metrics are never used by the online policy')
    parser.add_argument('--multi_navigation_perception_stop_guard', type=int, default=1,
                        help='1: require distinct perception/map-derived standoff arrivals before STOP')
    parser.add_argument('--multi_navigation_min_standoff_arrivals', type=int, default=2,
                        help='distinct internally selected standoff viewpoints required before final STOP')
    parser.add_argument('--multi_navigation_retry_block_radius_m', type=float, default=0.45,
                        help='radius around an arrived-but-unsuccessful navigation goal excluded during the next standoff selection')
    parser.add_argument('--multi_navigation_retry_memory', type=int, default=8,
                        help='maximum recent unsuccessful navigation viewpoints retained for alternate-angle replanning')
    parser.add_argument('--multi_navigation_stuck_patience', type=int, default=20,
                        help='stationary navigation frames before the current execution goal is rejected and replanned')
    parser.add_argument('--multi_navigation_stuck_tolerance_m', type=float, default=0.03,
                        help='maximum position change counted as stationary during final navigation')
    parser.add_argument('--multi_goal_foreground_mask_guard', type=int, default=1,
                        help='1: if goal-category detection fails, use the image-centred detected instance mask')
    parser.add_argument('--multi_goal_compact_center_prior', type=int, default=1,
                        help='1: use a compact category-shaped goal-image prior when no instance mask exists')
    parser.add_argument('--multi_goal_masked_confirmation_matching', type=int, default=1,
                        help='1: use goal foreground features for distant NVSM/FEM confirmation')
    parser.add_argument('--multi_confirm_anchor_first', type=int, default=1,
                        help='1: rank same-class Gaussian candidates by current detection-anchor distance, then cluster size')
    parser.add_argument('--multi_target2_map_prevalidation', type=int, default=1,
                        help='1: prevalidate target 2 from retained 2D semantics and RGB Gaussians before moving')
    parser.add_argument('--multi_target2_map_prevalidation_grace_steps', type=int, default=5,
                        help='initial target-2 frames allowed for a retained semantic-map candidate to appear')
    parser.add_argument('--multi_target2_map_match_threshold', type=float, default=40.0,
                        help='conservative semantic-FEM score required to accept a target-2 map-only candidate')
    parser.add_argument('--multi_target2_map_max_candidates', type=int, default=3,
                        help='maximum retained 2D semantic components considered during target-2 prevalidation')
    parser.add_argument('--multi_target2_map_clusters_per_component', type=int, default=2,
                        help='maximum local Gaussian clusters tested for each retained semantic component')
    parser.add_argument('--multi_target2_map_shortlist_candidates', type=int, default=2,
                        help='maximum semantic/instance-ranked Gaussian candidates entering coarse FEM')
    parser.add_argument('--multi_target2_map_coarse_views_per_scale', type=int, default=4,
                        help='NVSM views per scale used for low-cost target-2 candidate ranking')
    parser.add_argument('--multi_target2_map_coarse_min_score', type=float, default=8.0,
                        help='minimum coarse semantic-FEM score required before full Top-1 validation')
    parser.add_argument('--multi_target2_map_coarse_save_visualizations', type=int, default=0,
                        help='1: save coarse-ranking renders/matches; disabled by default for speed')
    parser.add_argument('--multi_target2_map_full_views_per_scale', type=int, default=16,
                        help='NVSM views per scale used for full validation of the coarse Top-1 candidate')
    parser.add_argument('--multi_target2_map_gaussian_radius_m', type=float, default=1.0,
                        help='XZ radius used to collect RGB Gaussians around a retained target-2 semantic component')
    parser.add_argument('--multi_target2_map_min_gaussians', type=int, default=100,
                        help='minimum local Gaussian support required for target-2 map prevalidation')
    parser.add_argument('--multi_target2_map_max_gaussians', type=int, default=50000,
                        help='maximum local Gaussians clustered around one target-2 map component')
    parser.add_argument('--multi_target2_map_use_gaussian_labels', type=int, default=1,
                        help='1: prefer detector-trained target-class Gaussians during map prevalidation')
    parser.add_argument('--multi_target2_map_gaussian_label_confidence', type=float, default=0.55,
                        help='minimum fused class confidence used to restrict target-2 map candidates')
    parser.add_argument('--multi_target2_map_gaussian_label_min_observations', type=int, default=1,
                        help='minimum detector-mask updates for a Gaussian to guide target-2 prevalidation')
    parser.add_argument('--multi_gaussian_label_training', type=int, default=1,
                        help='1: fuse detector semantic classes and persistent instance ids into Gaussian sidecars')
    parser.add_argument('--multi_gaussian_label_min_confidence', type=float, default=0.5,
                        help='minimum detector confidence used for Gaussian semantic/instance label training')
    parser.add_argument('--multi_gaussian_label_depth_tolerance_m', type=float, default=0.25,
                        help='RGB-D depth consistency tolerance for detector-to-Gaussian label projection')
    parser.add_argument('--multi_gaussian_instance_vote_ratio', type=float, default=0.15,
                        help='minimum existing Gaussian-id vote ratio for persistent instance association')
    parser.add_argument('--multi_gaussian_instance_association_radius_m', type=float, default=0.75,
                        help='maximum XZ centroid distance for associating a new mask with an existing instance')
    parser.add_argument('--multi_gaussian_label_evidence_cap', type=float, default=32.0,
                        help='per-class accumulated detector evidence cap for each Gaussian')
    parser.add_argument('--multi_gaussian_label_log_interval', type=int, default=20,
                        help='labelled frames between concise Gaussian semantic-training log records')
    parser.add_argument('--minav_buffer_goal_dilation_radius', type=int, default=10,
                        help='FMM goal dilation radius in map cells for MinAV buffer/standoff navigation')
    parser.add_argument('--minav_navigation_goal_dilation_radius', type=int, default=5,
                        help='FMM goal dilation radius in map cells for final MinAV localized-target navigation')
    parser.add_argument('--minav_secondary_exploration_enabled',
                        '--minav_enable_strategy_transition',
                        dest='minav_secondary_exploration_enabled',
                        type=int, default=1,
                        help='1: enable Phase-2/3 targeted secondary exploration after normal exploration')
    parser.add_argument('--minav_strategy_phase2_start_step', type=int, default=500,
                        help='episode step where StrategyTransition Phase 2 starts visiting semantic instances')
    parser.add_argument('--minav_strategy_phase3_start_step', type=int, default=800,
                        help='episode step where StrategyTransition Phase 3 starts going to the historical best-scoring instance')
    parser.add_argument('--minav_strategy_phase3_fallback_stop_radius', type=float, default=0.5,
                        help='meters from the Phase 3 semantic/standoff target within which localization failure triggers a non-success force stop')
    parser.add_argument('--minav_strategy_use_area_growth_trigger', type=int, default=0,
                        help='1: use legacy map-growth StrategyTransition trigger; 0: use fixed 500/800 step trigger')
    parser.add_argument('--minav_nvsm_recent_views', type=int, default=3,
                        help='number of recent 3DGS rendered keyframes used as NVSM evidence views')
    parser.add_argument('--minav_mr_closed_loop_threshold', type=float, default=40.0,
                        help='MR closed-loop evidence threshold after REF')
    parser.add_argument('--minav_use_full_localizer', type=int, default=1,
                        help='1: use migrated full MinAV NVSM/FEM/PRE/REF/MR localizer')
    parser.add_argument('--minav_fine_localization_enabled', type=int, default=1,
                        help='1: run PRE/REF/MR after FEM; 0: finish from the FEM coarse pose')
    parser.add_argument('--minav_full_render_views', type=int, default=16,
                        help='number of NVSM render views per scale for full MinAV localizer')
    parser.add_argument('--multi_render_matching_acceleration', type=int, default=1,
                        help='1: enable validated inference-only NVSM/PRE cameras and FEM timing wrappers')
    parser.add_argument('--multi_fem_superpoint_batch_size', type=int, default=1,
                        help='query-image batch size for Multi-IIN SuperPoint extraction')
    parser.add_argument('--multi_fem_lightglue_cuda_streams', type=int, default=1,
                        help='independent CUDA streams used for numerically equivalent per-view LightGlue matching')
    parser.add_argument('--multi_defer_render_cpu_transfer', type=int, default=0,
                        help='experimental: 1 transfers a completed NVSM render group together; 0 uses faster per-view transfer')
    parser.add_argument('--multi_pre_score_only_matching', type=int, default=1,
                        help='1: PRE candidate search computes identical LightGlue scores without unused MR/visualization caches')
    parser.add_argument('--multi_pre_gpu_direct_matching', type=int, default=1,
                        help='1: PRE sends an OpenCV-equivalent GPU grayscale render directly to SuperPoint')
    parser.add_argument('--multi_fast_localization', type=int, default=1,
                        help='1: use bounded Multi-IIN PRE search and release NVSM images after FEM')
    parser.add_argument('--multi_pre_stage1_z_samples', type=int, default=12,
                        help='vertical samples in bounded PRE stage 1')
    parser.add_argument('--multi_pre_stage1_pitch_samples', type=int, default=5,
                        help='pitch samples per vertical sample in bounded PRE stage 1')
    parser.add_argument('--multi_pre_stage1_y_half_range_m', type=float, default=1.2,
                        help='vertical half range in metres in bounded PRE stage 1')
    parser.add_argument('--multi_pre_stage2_position_samples', type=int, default=16,
                        help='positions including centre in each bounded PRE coarse round')
    parser.add_argument('--multi_pre_stage3_position_samples', type=int, default=4,
                        help='positions including centre in bounded PRE fine stage')
    parser.add_argument('--multi_pre_adaptive_enabled', type=int, default=1,
                        help='1: use a lighter PRE search for exceptionally high FEM confidence, with full-search fallback')
    parser.add_argument('--multi_pre_adaptive_score_threshold', type=float, default=100.0,
                        help='FEM score at or above which adaptive PRE starts with the light search')
    parser.add_argument('--multi_pre_light_stage1_z_samples', type=int, default=6,
                        help='vertical samples in adaptive light PRE stage 1')
    parser.add_argument('--multi_pre_light_stage1_pitch_samples', type=int, default=3,
                        help='pitch samples per vertical sample in adaptive light PRE stage 1')
    parser.add_argument('--multi_pre_light_stage2_position_samples', type=int, default=4,
                        help='positions including centre in adaptive light PRE coarse rounds')
    parser.add_argument('--multi_pre_light_stage3_position_samples', type=int, default=2,
                        help='positions including centre in adaptive light PRE fine stage')
    parser.add_argument('--minav_full_render_radius', type=float, default=1.0,
                        help='base NVSM render radius in meters for full MinAV localizer')
    parser.add_argument('--minav_full_max_closed_loop_rounds', type=int, default=3,
                        help='maximum MR closed-loop rounds for full MinAV localizer')
    parser.add_argument('--minav_gaussian_view_min_size_ratio', type=float, default=0.10,
                        help='minimum size ratio to the largest REF-view target Gaussian cluster for navigation target selection')
    parser.add_argument('--minav_verbose', type=int, default=1,
                        help='1: print verbose MinAV localizer internals to stdout')
    parser.add_argument('--minav_planner_debug', type=int, default=0,
                        help='1: print extra MinAV planner debug fields including pose, goal/map deltas, and collision-map updates')
    parser.add_argument('--minav_gaussian_projection_debug', type=int, default=0,
                        help='1: emit diagnostic traces for target Gaussian cluster projection; does not change decisions')
    parser.add_argument('--minav_gaussian_projection_debug_overlay', type=int, default=0,
                        help='1: add extra target/detection/navigation/cluster footprint markers to MinAV 2D grid images')
    parser.add_argument('--minav_gaussian_projection_debug_ply', type=int, default=0,
                        help='1: save candidate/confirmed target Gaussian cluster PLY files for projection debugging')
    parser.add_argument('--minav_gaussian_projection_debug_sample', type=int, default=200,
                        help='max confirmed-cluster Gaussian points sampled for 2D debug overlay')
    parser.add_argument('--minav_stg_angle_debug', type=int, default=0,
                        help='1: print MinAV STG angle candidates and real movement progress; does not change behavior')
    parser.add_argument('--minav_stg_angle_fix', type=int, default=0,
                        help='1: use row/col-to-yaw corrected MinAV STG angle formula for local navigation')
    parser.add_argument('--minav_sweep_turn_in_place_fix', type=int, default=0,
                        help='deprecated compatibility flag; MinAV turn/sweep overrides use normalized -1 linear velocity for in-place rotation')
    parser.add_argument('--minav_navigation_complete_stop_debug', type=int, default=0,
                        help='1: print why MinAV navigation_complete frames continue producing actions/visualizations')
    parser.add_argument('--stuck_recovery_enabled', type=int, default=0,
                        help='reserved; stuck escape action sequence is disabled in the planner')
    parser.add_argument('--stuck_recovery_patience', type=int, default=3,
                        help='number of consecutive stuck planner steps with unchanged position/STG before escape')
    parser.add_argument('--stuck_recovery_move_eps', type=float, default=0.03,
                        help='meters below which agent_position is treated as unchanged for stuck recovery')
    parser.add_argument('--stuck_recovery_stg_eps', type=float, default=1.0,
                        help='map cells below which the short-term goal is treated as unchanged for stuck recovery')
    parser.add_argument('--stuck_recovery_block_radius', type=int, default=3,
                        help='collision-map radius in cells to block around the stuck short-term goal before replanning')
    parser.add_argument('--stuck_recovery_backward_steps', type=int, default=2,
                        help='number of reverse velocity steps in a stuck escape sequence')
    parser.add_argument('--stuck_recovery_turn_steps', type=int, default=4,
                        help='number of in-place turn steps in a stuck escape sequence')
    parser.add_argument('--stuck_recovery_forward_steps', type=int, default=3,
                        help='number of forced forward steps after turning away from a stuck short-term goal')
    parser.add_argument('--minav_save_stage_visualizations', type=int, default=1,
                        help='1: save MinAV stage visualizations under results/')
    parser.add_argument('--minav_results_dir', type=str, default='results',
                        help='directory for MinAV stage visualizations; relative paths are under the IEVE repo root')
    parser.add_argument('--minav_blacklist_radius', type=float, default=0.5,
                        help='radius in meters for rejecting recently failed target clusters')
    parser.add_argument('--minav_blacklist_cooldown', type=int, default=50,
                        help='frames to keep failed target clusters blacklisted')
    parser.add_argument('--minav_confirm_failure_blacklist_threshold', type=int, default=10,
                        help='confirmation failures near the same target before blacklisting')
    parser.add_argument('--minav_confirm_cluster_eps', type=float, default=0.18,
                        help='DBSCAN eps in meters for candidate Gaussian target clustering')
    parser.add_argument('--minav_confirm_cluster_min_size', type=int, default=20,
                        help='minimum Gaussian count in a candidate target cluster')
    parser.add_argument('--minav_confirm_full_cluster_eps', type=float, default=0.18,
                        help='DBSCAN eps in meters for optional full target-confirm cluster expansion')
    parser.add_argument('--minav_confirm_full_cluster_min_samples', type=int, default=4,
                        help='DBSCAN min samples for optional full target-confirm cluster expansion')
    parser.add_argument('--minav_confirm_full_cluster_bbox_pad_xz', type=float, default=0.25,
                        help='XZ ROI padding for optional geometric full cluster expansion')
    parser.add_argument('--minav_confirm_full_cluster_bbox_pad_y', type=float, default=0.35,
                        help='Y ROI padding for optional geometric full cluster expansion')
    parser.add_argument('--minav_confirm_semantic_threshold', type=float, default=0.6,
                        help='semantic target score threshold on Gaussians for target confirmation')
    parser.add_argument('--minav_semantic_threshold', type=float, default=0.6,
                        help='semantic target score threshold on Gaussians')
    parser.add_argument('--minav_confirm_depth_tolerance', type=float, default=0.3,
                        help='depth tolerance in meters for mask-to-Gaussian projection')
    parser.add_argument('--minav_confirm_mask_projection_filter_debug', type=int, default=0,
                        help='1: log per-filter mask-to-Gaussian projection counts during target confirmation')

    # Density Mapping
    parser.add_argument('--kernel_size', type=int, default=91) # ji shu
    parser.add_argument('--low_dense', type=int, default=8)
    parser.add_argument('--succeed_match_points', type=int, default=200)
    parser.add_argument('--big_enough_f', type=float, default=1/6)
    parser.add_argument('--big_enough_thresh', type=int, default=4)

    # parse arguments
    args = parser.parse_args()

    # Keep the legacy attribute available to out-of-tree callers while all
    # in-tree code uses the ablation-facing name above.
    args.minav_enable_strategy_transition = (
        args.minav_secondary_exploration_enabled
    )

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.result_only_log:
        args.quiet_log = 1

    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "train" in args.split:
                # args.total_num_scenes = 145
                args.total_num_scenes = 36
            elif "val" in args.split:
                args.total_num_scenes = 36
                # args.total_num_scenes = 15
            else:
                assert False, "Unknown task config, please specify" + \
                    " total_num_scenes"

            # GPU Memory required for the SemExp model:
            #       0.8 + 0.4 * args.total_num_scenes (GB)
            # GPU Memory required per thread: 2.6 (GB)
            min_memory_required = max(0.8 + 0.4 * args.total_num_scenes, 6)
            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                                 torch.cuda.get_device_properties(
                                     i).total_memory
                                 / 1024 / 1024 / 1024)
                assert gpu_memory > min_memory_required, \
                    """Insufficient GPU memory for GPU {}, gpu memory ({}GB)
                    needs to be greater than {}GB""".format(
                        i, gpu_memory, min_memory_required)

            num_processes_per_gpu = int(gpu_memory / 6)
            num_processes_on_first_gpu = \
                int((gpu_memory - min_memory_required) / 6)

            # if args.eval:
            #     max_threads = num_processes_per_gpu * (num_gpus - 1) \
            #         + num_processes_on_first_gpu
            #     assert max_threads >= args.total_num_scenes, \
            #         """Insufficient GPU memory for evaluation"""

            if num_gpus == 1:
                args.num_processes_on_first_gpu = num_processes_on_first_gpu
                args.num_processes_per_gpu = 0
                args.num_processes = num_processes_on_first_gpu
                assert args.num_processes > 0, "Insufficient GPU memory"
            else:
                num_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                num_threads = min(num_threads, args.total_num_scenes)
                args.num_processes_per_gpu = num_processes_per_gpu
                args.num_processes_on_first_gpu = max(
                    0,
                    num_threads - args.num_processes_per_gpu * (num_gpus - 1))
                args.num_processes = num_threads

            args.sim_gpu_id = 1

            if not args.quiet_log:
                print("Auto GPU config:")
                print("Number of processes: {}".format(args.num_processes))
                print("Number of processes on GPU 0: {}".format(
                    args.num_processes_on_first_gpu))
                print("Number of processes per GPU: {}".format(
                    args.num_processes_per_gpu))
    else:
        args.sem_gpu_id = -2

    if args.num_mini_batch == "auto":
        args.num_mini_batch = max(args.num_processes // 2, 1)
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    return args
