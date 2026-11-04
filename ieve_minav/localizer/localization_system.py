import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .semantic_object_identifier import SemanticObjectIdentifier
from .novel_view_renderer import NovelViewRenderer
from .feature_matcher import FeatureMatcher
# from .exploration_planner import ExplorationPlanner  # ★★★ EP已废弃 ★★★
from .pose_refiner import PoseRefiner
from .pose_pre_refiner import PosePreRefiner
from .map_regulator import MapRegulator

# ★★★ 新增：导入彩色输出 ★★★
from ieve_minav.localizer.colors import Colors


class LocalizationState(Enum):
    """定位系统状态枚举"""
    IDLE = "idle"
    SOIM = "soim"
    NVSM = "nvsm"
    FEM = "fem"
    PRE = "pre"       # ★★★ 新增：前置位姿优化状态 ★★★
    REF = "ref"       # ★★★ 新增：细定位状态 ★★★
    RECOVERY_WAIT = "recovery_wait"
    MR = "mr"        # ★★★ 新增：地图调控状态 ★★★
    EP = "ep"
    SUCCESS = "success"
    FAILED = "failed"


class LocalizationSystem:
    """主定位系统 - 协调各个子模块完成定位任务"""
    
    def __init__(self, 
                 slam_mapper,
                 occupancy_mapper,
                 target_image,
                 habitat_sim,
                 config: Optional[Dict[str, Any]] = None,
                 target_category: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 segmenter=None,
                 save_visualization_results: bool = True):
        """
        初始化定位系统
        
        Args:
            slam_mapper: SLAMMapper实例，提供3D高斯地图和渲染器
            occupancy_mapper: OccupancyMapper实例，提供2D栅格地图
            target_image: 目标图像
            habitat_sim: Habitat仿真环境
            config: 配置参数
            target_category: 目标类别名称
            results_dir: ★★★ 新增：当前运行的结果目录，防止多进程保存错乱 ★★★
            segmenter: SemanticSegmenter实例（可选），用于FEM语义加权匹配
        """
        self.logger = logging.getLogger(__name__)
        self.save_visualization_results = bool(save_visualization_results)
        
        # ★★★ 新增：保存 results_dir 以便传递给子模块 ★★★
        self._results_dir = str(results_dir) if results_dir is not None else None
        
        # 保存建图模块引用
        self.slam_mapper = slam_mapper
        self.occupancy_mapper = occupancy_mapper
        
        # 将 occupancy mapper 引用同步给 NVSM 使用的 slam_mapper。
        slam_mapper.occupancy_mapper = occupancy_mapper
        
        # 默认配置
        default_config = {
            'observation_threshold': 3, 
            'match_threshold': 20,       # FEM 粗定位匹配证据阈值（Σ confidence）
            'closed_loop_threshold': 40, # ★★★ 新增：REF后闭环定位阈值 ★★★
            'pre_detection_min_matches': 10,  # ★★★ 新增：预检测最少匹配点数 ★★★
            'cooldown_period': 38,       # 冷却期（秒）
            'render_views': 48,          # ★★★ 修改：增加到48个视角 ★★★
            'render_radius': 1.0,        # ★★★ 修改：基础渲染半径改为1m ★★★
            'max_closed_loop_rounds': 3,  # ★★★ 新增：MR闭环最大轮次，超过则直接定位完成 ★★★
            'fine_localization_enabled': True,
            'cluster_recovery_enabled': True,
            'cluster_recovery_trigger_score': 100.0,
        }
        self.config = {**default_config, **(config or {})}
        self.fine_localization_enabled = bool(
            self.config.get('fine_localization_enabled', True)
        )
        
        # 检查SLAM是否可用
        if not slam_mapper.slam or not slam_mapper.is_running:
            self.logger.warning("SLAM mapper not running, localization may fail")
        
        # ★★★ 添加参数验证 ★★★
        try:
            # 检查 target_image 是否为有效的 numpy 数组
            if not isinstance(target_image, np.ndarray):
                if hasattr(target_image, 'numpy'):
                    target_image = target_image.numpy()
                else:
                    raise ValueError(f"target_image must be numpy array, got {type(target_image)}")
            
            # 确保是 RGB 格式 (H, W, 3)
            if target_image.ndim != 3 or target_image.shape[2] != 3:
                raise ValueError(f"target_image must be (H, W, 3), got {target_image.shape}")
            
            # 确保是 uint8 类型
            if target_image.dtype != np.uint8:
                if target_image.max() <= 1.0:
                    target_image = (target_image * 255).astype(np.uint8)
                else:
                    target_image = target_image.astype(np.uint8)
            
            print(f"{Colors.CYAN}[Init] Target image validated: shape={target_image.shape}, dtype={target_image.dtype}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}[Init] Target image validation failed: {e}{Colors.END}")
            raise
        
        # IEVE 适配：外层已把 IEVE 的 goal_cat_id 转成 MinAV 语义类别 id。
        target_semantic_class = self.config.get('target_semantic_class')
        if target_semantic_class is not None:
            try:
                target_semantic_class = int(target_semantic_class)
            except (TypeError, ValueError):
                target_semantic_class = None
        
        if target_category is not None and target_semantic_class is None:
            # 尝试获取语义类别（可能失败，这是正常的）
            if segmenter is not None:
                if hasattr(segmenter, 'indoor_categories'):
                    target_semantic_class = segmenter.get_category_id(target_category.lower())
                    # target_semantic_class = segmenter.indoor_categories.get(target_category.lower())
                    if target_semantic_class is not None:
                        print(f"{Colors.CYAN}[Init] Target category '{target_category}' mapped to semantic class {target_semantic_class}{Colors.END}")
            
            # ★★★ 修改：如果获取失败，只打印提示，不再报错 ★★★
            if target_semantic_class is None:
                print(f"{Colors.YELLOW}[Init] Note: Target semantic class will be set later after goal image analysis{Colors.END}")
                print(f"{Colors.YELLOW}  Target category: {target_category}{Colors.END}")
        
        # ★★★ 保存 target_category 以便后续使用 ★★★
        self.target_category = target_category
        self.segmenter = segmenter
        self.target_semantic_class = target_semantic_class
        
        # 初始化各个子模块
        try:
            # ★★★ 改进：从 slam_mapper 获取相机参数并传递给 SOIM ★★★
            camera_params_for_soim = {
                'fx': slam_mapper.camera_params.get('fx'),
                'fy': slam_mapper.camera_params.get('fy'),
                'cx': slam_mapper.camera_params.get('cx'),
                'cy': slam_mapper.camera_params.get('cy'),
                'width': slam_mapper.camera_params.get('width'),
                'height': slam_mapper.camera_params.get('height')
            }
            
            self.soim = SemanticObjectIdentifier(
                occupancy_mapper=occupancy_mapper,
                observation_threshold=self.config['observation_threshold'],
                cooldown_period=self.config['cooldown_period'],
                target_semantic_class=target_semantic_class,
                min_observation_distance=self.config.get('min_observation_distance', 0.5),
                min_angle_threshold=self.config.get('min_angle_threshold', 38.0),
                camera_params=camera_params_for_soim,  # ★★★ 新增：传递相机参数 ★★★
                fov_margin=self.config.get('fov_margin', 0.1)  # ★★★ 新增：视野边界裕度 ★★★
            )
            if target_semantic_class is not None:
                print(f"{Colors.GREEN}[Init] SOIM initialized with target_class={target_semantic_class}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[Init] SOIM initialized (target_class pending){Colors.END}")
            
            # ★★★ 修改：传递多尺度渲染参数 ★★★
            multi_scale_radii = self.config.get('multi_scale_radii', None)
            self.nvsm = NovelViewRenderer(
                slam_mapper=slam_mapper,
                num_views=self.config['render_views'],
                render_radius=self.config['render_radius'],
                multi_scale_radii=multi_scale_radii,
                results_dir=self._results_dir,  # ★★★ 修复：直接传入 results_dir ★★★
                save_visualization_results=self.save_visualization_results,
            )
            print(f"{Colors.GREEN}[Init] NVSM initialized with multi-scale rendering{Colors.END}")
            
            # ★★★ 改进：传递语义分割器和目标类别，用于语义加权匹配 ★★★
            _segmenter = segmenter  # 由外部直接传入，不再从 occupancy_mapper 上取
            if _segmenter is not None:
                print(f"{Colors.CYAN}[Init] FEM will use semantic-weighted matching{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[Init] FEM semantic weighting disabled (no segmenter provided){Colors.END}")
            
            self.fem = FeatureMatcher(
                target_image=target_image,
                match_threshold=self.config['match_threshold'],
                results_dir=self._results_dir,
                segmenter=_segmenter,
                target_category=target_category,
                semantic_density_ratio=self.config.get('semantic_density_ratio', 3.0),
                save_visualization_results=self.save_visualization_results,
                default_feature_mode=self.config.get('feature_mode', 'uniform'),
            )
            print(f"{Colors.GREEN}[Init] FEM initialized{Colors.END}")
            
            # ★★★ EP已废弃：不再初始化 ExplorationPlanner ★★★
            # self.ep = ExplorationPlanner(habitat_sim=habitat_sim)
            self.ep = None
            print(f"{Colors.GREEN}[Init] EP abolished (not initialized){Colors.END}")
            
            ref_config = self.config.get('refinement', {})
            self.ref = None
            self.pre = None
            self.mr = None
            if self.fine_localization_enabled:
                camera_params_for_ref = dict(slam_mapper.camera_params)
                self.ref = PoseRefiner(
                    slam_mapper=slam_mapper,
                    camera_params=camera_params_for_ref,
                    config=ref_config,
                    save_visualization_results=self.save_visualization_results,
                )
                print(f"{Colors.GREEN}[Init] REF (PoseRefiner) initialized{Colors.END}")
                self.pre = PosePreRefiner(
                    slam_mapper=slam_mapper,
                    feature_matcher=self.fem,
                    results_dir=self._results_dir,
                    save_visualization_results=self.save_visualization_results,
                )
                print(f"{Colors.GREEN}[Init] PRE (PosePreRefiner) initialized{Colors.END}")
                mr_config = self.config.get('map_regulator', {})
                self.mr = MapRegulator(
                    slam_mapper=slam_mapper,
                    segmenter=_segmenter,
                    target_category=target_category,
                    results_dir=self._results_dir,
                    config=mr_config
                )
                print(f"{Colors.GREEN}[Init] MR (MapRegulator) initialized{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[Init] PRE/REF/MR disabled; FEM coarse pose is final{Colors.END}")
            
            # ★★★ 新增：立即将 results_dir 传播到所有子模块 ★★★
            if self._results_dir is not None:
                self.update_results_dir(self._results_dir)
                print(f"{Colors.GREEN}[Init] results_dir propagated to all sub-modules: {self._results_dir}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}[Init] Submodule initialization failed: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            raise
        
        # 状态管理
        self.current_state = LocalizationState.IDLE
        self.current_object = None  # 当前处理的物体
        self.best_pose = None       # FEM 粗定位位姿
        self.fem_best_pose = None   # FEM 原始最高分视角，PRE 不得覆盖
        self.refined_pose = None        # ★★★ 新增：REF 细定位位姿 ★★★
        self.fem_best_score = 0.0       # ★★★ 新增：FEM最佳分数（供PRE使用）★★★
        self.rendered_images = []   # 渲染的图像列表
        self.target_depth = None    # ★★★ 新增：深度目标图像（由外部设置）★★★
        self.target_keypoints = None  # ★★★ 新增：预提取特征点（由外部设置）★★★
        self.target_semantic_mask = None  # ★★★ 新增：目标语义掩码（用于判断是否需要闭环）★★★
        self._last_keyframe_step = None

        # ★★★ 新增：定位系统状态跟踪 ★★★
        self.localization_attempts = 0  # 定位尝试次数
        self.state_history = []  # 状态历史记录
        self._mr_trigger_source = None  # ★★★ 新增：MR 触发来源 ('fem'/'ref') ★★★
        self._mr_trigger_source = None  # ★★★ 新增：MR 触发来源 ('fem'/'ref') ★★★
        self.cluster_recovery_used = False
        self.cluster_recovery_requested = False
        self.cluster_recovery_tc_evidence = {
            'score': 0.0,
            'pose': None,
            'center': None,
        }
        
        # ★★★ 启动时的彩色输出 ★★★
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}★ LocalizationSystem Initialized ★{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}  Config:{Colors.END}")
        print(f"    - Observation threshold: {self.config['observation_threshold']}")
        print(f"    - Match threshold: {self.config['match_threshold']}")
        print(f"    - Render views: {self.config['render_views']}")
        print(f"    - Render radius: {self.config['render_radius']}m")
        print(f"    - Refinement max_iter: {ref_config.get('max_iterations', 50)}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info("LocalizationSystem initialized with real mapping modules")
    
    def run_localization(self, keyframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行定位流程 - 主入口函数（增量式执行）
        
        每次调用只执行当前状态的一步，需要外部循环持续调用
        
        Args:
            keyframe_data: 关键帧数据，包含：
                - rgb: RGB图像
                - depth: 深度图像
                - semantic: 语义分割
                - pose: 相机位姿
                - timestamp: 时间戳
                - is_3dgs_keyframe: 是否是3DGS关键帧
        
        Returns:
            结果字典，包含状态和相关信息
        """
        result = {
            'status': 'running',
            'state': self.current_state.value,
            'message': ''
        }
        if isinstance(keyframe_data, dict):
            for key in ('step', 'frame', 'timestamp'):
                if key in keyframe_data and keyframe_data.get(key) is not None:
                    self._last_keyframe_step = keyframe_data.get(key)
                    break
        
        # ★★★ 记录状态历史 ★★★
        if len(self.state_history) == 0 or self.state_history[-1] != self.current_state.value:
            self.state_history.append(self.current_state.value)
        
        try:
            if self.current_state == LocalizationState.IDLE:
                # IDLE状态：等待启动
                return {
                    'status': 'idle',
                    'state': self.current_state.value,
                    'message': 'System idle, waiting for start_localization()'
                }
                
            elif self.current_state == LocalizationState.SOIM:
                result = self._execute_soim(keyframe_data)
                
            elif self.current_state == LocalizationState.NVSM:
                result = self._execute_nvsm()
                
            elif self.current_state == LocalizationState.FEM:
                result = self._execute_fem()
            
            # ★★★ 新增：PRE 状态处理 ★★★
            elif self.current_state == LocalizationState.PRE:
                result = self._execute_pre()
            
            # ★★★ 新增：REF 状态处理 ★★★
            elif self.current_state == LocalizationState.REF:
                result = self._execute_ref()

            elif self.current_state == LocalizationState.RECOVERY_WAIT:
                result = {
                    'status': 'cluster_recovery_waiting',
                    'state': self.current_state.value,
                    'message': 'Waiting for IEVE cluster-recovery orbit',
                }
            
            # ★★★ 新增：MR 状态处理 ★★★
            elif self.current_state == LocalizationState.MR:
                result = self._execute_mr()
                
            elif self.current_state == LocalizationState.EP:
                # ★★★ EP已废弃：直接返回定位完成结果 ★★★
                result = self._make_localization_complete_result(
                    'EP state reached (EP module abolished, returning pose directly)',
                    ref_rendered_rgb=None,
                    ref_depth=None
                )
            elif self.current_state in [LocalizationState.SUCCESS, LocalizationState.FAILED]:
                # 终止状态
                result = {
                    'status': self.current_state.value,
                    'state': self.current_state.value,
                    'message': f'Localization {self.current_state.value}'
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Localization error: {e}", exc_info=True)
            self.current_state = LocalizationState.FAILED
            
            # ★★★ 错误输出 ★★★
            print(f"\n{Colors.RED}{'='*70}{Colors.END}")
            print(f"{Colors.RED}[Localization] ✗ ERROR ✗{Colors.END}")
            print(f"{Colors.RED}  {e}{Colors.END}")
            print(f"{Colors.RED}{'='*70}{Colors.END}\n")
            
            return {
                'status': 'failed',
                'state': self.current_state.value,
                'message': str(e)
            }
    
    def _execute_soim(self, keyframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行语义物体识别模块"""
        self.logger.info("Executing SOIM module")
        
        # 更新物体观测
        soim_result = self.soim.process_keyframe(keyframe_data)
        
        # ★★★ 新增：检查是否跳过了非 3DGS 关键帧 ★★★
        if soim_result.get('skipped', False):
            # 非 3DGS 关键帧，不打印详细输出，静默返回
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'Skipped non-3DGS frame',
                'observed_objects': soim_result['observed_objects']
            }
        
        # ★★★ 状态输出 (只对 3DGS 关键帧) ★★★
        print(f"\n{Colors.MAGENTA}[SOIM] 处理关键帧 #{int(keyframe_data['timestamp'])}...{Colors.END}")
        
        # ★★★ 观测统计输出 - 增强版 ★★★
        num_observed = len(soim_result['observed_objects'])
        print(f"{Colors.CYAN}[SOIM] 当前追踪 {num_observed} 个目标物体{Colors.END}")
        
        # 打印所有活跃物体的详细信息
        active_objects = {k: v for k, v in soim_result['observed_objects'].items() if v['status'] == 'active'}
        if active_objects:
            print(f"{Colors.CYAN}[SOIM] 活跃物体列表:{Colors.END}")
            for obj_id, obj_data in active_objects.items():
                obs_count = obj_data['observations']
                progress = obs_count / self.config['observation_threshold']
                progress_bar = '█' * int(progress * 10) + '░' * (10 - int(progress * 10))
                
                color = Colors.GREEN if obs_count >= self.config['observation_threshold'] else Colors.CYAN
                print(f"  {color}- {obj_id}: {obs_count}/{self.config['observation_threshold']} "
                      f"[{progress_bar}] 类别={obj_data.get('semantic_class', 'N/A')}{Colors.END}")
        
        # 打印冷却中的物体
        abandoned_objects = {k: v for k, v in soim_result['observed_objects'].items() if v['status'] == 'abandoned'}
        if abandoned_objects:
            print(f"{Colors.YELLOW}[SOIM] 冷却中的物体: {len(abandoned_objects)} 个{Colors.END}")
        
        if soim_result['trigger_nvsm']:
            # 达到观测阈值，切换到NVSM
            self.current_object = soim_result['object_info']
            self.current_state = LocalizationState.NVSM
            
            # ★★★ 触发输出（区分首次触发和再次触发）★★★
            obs_count = self.current_object.get('observations', 0)
            is_retry = obs_count > self.config['observation_threshold']
            
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            if is_retry:
                print(f"{Colors.GREEN}[SOIM] ✓ TRIGGERING NVSM (第{obs_count - self.config['observation_threshold'] + 1}次尝试) ✓{Colors.END}")
            else:
                print(f"{Colors.GREEN}[SOIM] ✓ THRESHOLD REACHED ✓{Colors.END}")
            print(f"{Colors.GREEN}  Object ID: {self.current_object['id']}{Colors.END}")
            print(f"{Colors.GREEN}  Observations: {obs_count}{Colors.END}")
            print(f"{Colors.GREEN}  Position 2D: {self.current_object['position_2d']}{Colors.END}")
            print(f"{Colors.GREEN}  Position 3D: {self.current_object.get('position_3d')}{Colors.END}")
            print(f"{Colors.GREEN}  → Switching to NVSM (Novel View Rendering){Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
            
            self.logger.info(f"Object {self.current_object['id']} triggering NVSM (obs={obs_count})")
            
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f"Object {self.current_object['id']} ready for rendering (obs={obs_count})"
            }
        else:
            # 继续观测
            print(f"{Colors.CYAN}[SOIM] Continuing observation...{Colors.END}\n")
            
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'Still observing objects',
                'observed_objects': soim_result['observed_objects']
            }
    
    def _execute_nvsm(self) -> Dict[str, Any]:
        """执行新视角渲染模块"""
        # ★★★ 状态输出 ★★★
        observation_positions = self.current_object.get('observation_positions', [])
        
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}[NVSM] Starting Multi-Scale Novel View Rendering{Colors.END}")
        print(f"{Colors.MAGENTA}  Target Object: {self.current_object['id']}{Colors.END}")
        print(f"{Colors.MAGENTA}  Observation Positions: {len(observation_positions)}{Colors.END}")
        print(f"{Colors.MAGENTA}  Base Render Views: {self.config['render_views']}{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info(f"Executing NVSM module for object {self.current_object['id']}")
        
        # ★★★ 核心修改：根据SOIM触发的实例重新标记语义高斯，确保 ★★★
        # .ply / .txt / 渲染中心 三者使用完全相同的一组实例级高斯
        gaussian_center_3d = None
        try:
            # --- Step 1: 从SOIM触发物体的boundary构建实例级2D掩码 ---
            boundary = self.current_object.get('boundary_2d')
            semantic_class = self.current_object.get('semantic_class', 1)
            
            if boundary is not None and hasattr(self.occupancy_mapper, 'map'):
                semantic_channel = self.occupancy_mapper.map[:, :, 2]
                map_h, map_w = semantic_channel.shape
                
                # 在boundary范围（+ padding）内提取属于该语义类别的像素
                pad = 10  # 像素padding，给形态学闭运算留空间
                r_min = max(0, boundary['y_min'] - pad)
                r_max = min(map_h - 1, boundary['y_max'] + pad)
                c_min = max(0, boundary['x_min'] - pad)
                c_max = min(map_w - 1, boundary['x_max'] + pad)
                
                instance_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                roi = semantic_channel[r_min:r_max+1, c_min:c_max+1]
                instance_mask[r_min:r_max+1, c_min:c_max+1] = (roi == semantic_class).astype(np.uint8)
                
                # 形态学闭运算弥合稀疏间隙
                try:
                    from skimage.morphology import disk, binary_closing
                    instance_mask = binary_closing(
                        instance_mask.astype(bool), footprint=disk(7)
                    ).astype(np.uint8)
                except ImportError:
                    pass
                
                num_instance_pixels = int(np.sum(instance_mask))
                pos_2d = self.current_object['position_2d']  # (col, row)
                
                instance_region = {
                    'bbox': (r_min, r_max, c_min, c_max),
                    'mask': instance_mask,
                    'center': (int(pos_2d[1]), int(pos_2d[0])),  # (row, col)
                    'semantic_id': int(semantic_class),
                    'map_size': map_h,
                }
                
                # --- Step 2: 用实例级掩码重新标记3DGS中的目标语义高斯 ---
                map_resolution = self.occupancy_mapper.map_resolution / 100.0  # cm/px → m/px
                num_marked = self.slam_mapper.set_target_semantic_region(
                    instance_region, map_resolution, dilate_pixels=3
                )
                
                print(f"{Colors.CYAN}[NVSM] ★ Instance-level semantic Gaussian re-marking:{Colors.END}")
                print(f"{Colors.CYAN}  SOIM object: {self.current_object['id']}{Colors.END}")
                print(f"{Colors.CYAN}  Boundary: x=[{boundary['x_min']},{boundary['x_max']}], "
                      f"y=[{boundary['y_min']},{boundary['y_max']}]{Colors.END}")
                print(f"{Colors.CYAN}  Instance 2D pixels: {num_instance_pixels}{Colors.END}")
                print(f"{Colors.CYAN}  Marked 3D Gaussians: {num_marked}{Colors.END}")
            
            # --- Step 3: 从重新标记的高斯中获取质心（与.ply/.txt完全一致）---
            stats = self.slam_mapper.get_target_semantic_stats()
            if stats is not None and stats.get('center') is not None:
                slam_center = stats['center']  # get_xyz 已是 Habitat 世界坐标
                # ★★★ 直接使用质心坐标，不加任何偏移 ★★★
                # 与 visualize_target_semantic_gaussians() 写入.ply/.txt的计算方式完全一致
                gaussian_center_3d = np.array([
                    float(slam_center[0]),
                    float(slam_center[1]),
                    float(slam_center[2])
                ], dtype=np.float32)
                
                print(f"{Colors.CYAN}[NVSM] Render center (= PLY/TXT centroid, Habitat): "
                      f"[{gaussian_center_3d[0]:.3f}, {gaussian_center_3d[1]:.3f}, {gaussian_center_3d[2]:.3f}]{Colors.END}")
                if stats.get('bbox_min') is not None:
                    print(f"{Colors.CYAN}[NVSM] BBox_min: {stats['bbox_min']}{Colors.END}")
                    print(f"{Colors.CYAN}[NVSM] BBox_max: {stats['bbox_max']}{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[NVSM] No target semantic gaussians after re-marking, "
                      f"falling back to SOIM position_3d{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}[NVSM] Failed to re-mark/get gaussian center: {e}{Colors.END}")
            import traceback; traceback.print_exc()
        
        # 如果无法从高斯获取中心，回退到 SOIM 的 position_3d
        render_center_3d = gaussian_center_3d if gaussian_center_3d is not None else self.current_object.get('position_3d')
        
        # ★★★ 保存重标记后的实例级语义高斯 .ply 和 .txt ★★★
        # 确保 .ply/.txt/渲染中心 三者使用完全一致的高斯点集
        try:
            if (
                self.save_visualization_results
                and gaussian_center_3d is not None
                and hasattr(self, 'nvsm')
                and self.nvsm.results_dir is not None
            ):
                import os
                from datetime import datetime
                nvsm_save_dir = os.path.join(str(self.nvsm.results_dir), "NVSM")
                os.makedirs(nvsm_save_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 保存 .ply（与 visualize_target_semantic_gaussians 同源同数据）
                ply_path = os.path.join(nvsm_save_dir, f"instance_semantic_{timestamp}.ply")
                ply_result = self.slam_mapper.visualize_target_semantic_gaussians(ply_path)
                
                # 保存 .txt
                txt_path = os.path.join(nvsm_save_dir, f"instance_semantic_{timestamp}.txt")
                with open(txt_path, "w") as f:
                    f.write(f"Object: {self.current_object['id']}\n")
                    f.write(f"Target: {getattr(self, 'target_category', 'unknown')}\n")
                    f.write(f"Count: {ply_result['count']}\n")
                    if ply_result.get('center') is not None:
                        c = ply_result['center']
                        f.write(f"Center: {c}\n")
                        f.write(f"Center_x: {c[0]:.6f}\n")
                        f.write(f"Center_y(height): {c[1]:.6f}\n")
                        f.write(f"Center_z: {c[2]:.6f}\n")
                    if ply_result.get('bbox_min') is not None:
                        f.write(f"BBox_min: {ply_result['bbox_min']}\n")
                    if ply_result.get('bbox_max') is not None:
                        f.write(f"BBox_max: {ply_result['bbox_max']}\n")
                    f.write(f"Render_center: [{render_center_3d[0]:.6f}, {render_center_3d[1]:.6f}, {render_center_3d[2]:.6f}]\n")
                
                print(f"{Colors.GREEN}[NVSM] ✓ Saved instance PLY/TXT → {nvsm_save_dir}{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}[NVSM] Warning: Failed to save instance PLY/TXT: {e}{Colors.END}")
        
        # ★★★ 核心改进：从语义高斯聚类获取所有 Instance Center，多中心渲染 ★★★
        instance_centers = None
        try:
            cluster_info = self.slam_mapper.get_instance_cluster_centers(
                threshold=0.4, eps=0.15, min_samples=10, min_cluster_size=50)
            if cluster_info['num_clusters'] > 0:
                instance_centers = cluster_info['centers_3d']
                cluster_sizes = cluster_info['cluster_sizes']
                print(f"{Colors.CYAN}[NVSM] ★ Found {cluster_info['num_clusters']} "
                      f"instance clusters for multi-center rendering{Colors.END}")
            else:
                print(f"{Colors.YELLOW}[NVSM] No valid clusters, "
                      f"falling back to single-center rendering{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}[NVSM] Clustering failed ({e}), "
                  f"falling back to single-center rendering{Colors.END}")
        
        if instance_centers is not None and len(instance_centers) > 1:
            # ★ 多中心渲染：绕每个 Instance Center 渲染
            nvsm_result = self.nvsm.render_novel_views_multi_center(
                instance_centers_3d=instance_centers,
                cluster_sizes=cluster_sizes,
                object_center_2d=self.current_object['position_2d'],
                observation_positions=observation_positions,
            )
        else:
            # ★ 单中心渲染（降级 / 只有一个聚类）
            single_center = (instance_centers[0]
                             if instance_centers and len(instance_centers) == 1
                             else render_center_3d)
            nvsm_result = self.nvsm.render_novel_views(
                object_center_2d=self.current_object['position_2d'],
                object_center_3d=single_center,
                observation_positions=observation_positions,
            )
        
        if nvsm_result['success']:
            self.rendered_images = nvsm_result['rendered_images']
            self.current_state = LocalizationState.FEM
            
            # ★★★ 成功输出（包含多中心 + 多尺度信息）★★★
            scale_info = nvsm_result.get('scale_info', [])
            radii_used = nvsm_result.get('radii_used', [self.config['render_radius']])
            num_centers_used = nvsm_result.get('num_instance_centers', 1)
            
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}[NVSM] ✓ MULTI-CENTER MULTI-SCALE RENDERING SUCCESS ✓{Colors.END}")
            print(f"{Colors.GREEN}  Instance Centers: {num_centers_used}{Colors.END}")
            print(f"{Colors.GREEN}  Total Views: {len(self.rendered_images)}{Colors.END}")
            print(f"{Colors.GREEN}  Scales Used: {len(radii_used)}{Colors.END}")
            for i, radius in enumerate(radii_used):
                scale_views = sum(1 for img in self.rendered_images if img.get('scale_index', 0) == i)
                print(f"{Colors.GREEN}    Scale {i}: radius={radius:.2f}m, views={scale_views}{Colors.END}")
            print(f"{Colors.GREEN}  → Switching to FEM (Feature Matching){Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
            
            self.logger.info(f"Rendered {len(self.rendered_images)} views across {len(radii_used)} scales")
            
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f"Rendered {len(self.rendered_images)} views across {len(radii_used)} scales",
                'scale_info': scale_info
            }
        else:
            # 渲染失败，退出定位模式
            self.current_state = LocalizationState.FAILED
            
            # ★★★ 失败输出 ★★★
            print(f"\n{Colors.RED}{'='*70}{Colors.END}")
            print(f"{Colors.RED}[NVSM] ✗ RENDERING FAILED ✗{Colors.END}")
            print(f"{Colors.RED}  Reason: {nvsm_result['message']}{Colors.END}")
            print(f"{Colors.RED}  → Exiting Localization Mode{Colors.END}")
            print(f"{Colors.RED}{'='*70}{Colors.END}\n")
            
            self.logger.warning(f"Rendering failed: {nvsm_result['message']}")
            
            return {
                'status': 'failed',
                'state': self.current_state.value,
                'message': f"Rendering failed: {nvsm_result['message']}"
            }
    
    def _execute_fem(self) -> Dict[str, Any]:
        """执行特征匹配模块"""
        # ★★★ 状态输出 ★★★
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}[FEM] Starting Feature Matching{Colors.END}")
        print(f"{Colors.MAGENTA}  Candidates: {len(self.rendered_images)} views{Colors.END}")
        print(f"{Colors.MAGENTA}  Threshold: {self.config['match_threshold']}{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info("Executing FEM module")
        
        # ★★★ 新增：检查渲染图像列表是否有效 ★★★
        if not self.rendered_images or len(self.rendered_images) == 0:
            print(f"{Colors.RED}[FEM] Error: No rendered images available{Colors.END}")
            self.soim.set_object_abandoned(self.current_object['id'])
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'No rendered images, returning to SOIM'
            }
        
        # ★★★ 修复：在 FEM 匹配前同步 CUDA，确保 3DGS 渲染的异步错误在此暴露 ★★★
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except RuntimeError as cuda_err:
            print(f"{Colors.RED}[FEM] CUDA error from prior operation (likely 3DGS render): {cuda_err}{Colors.END}")
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f'CUDA error before FEM: {cuda_err}'
            }

        # 执行特征匹配
        try:
            fem_result = self.fem.match_rendered_views(
                self.rendered_images, feature_mode="uniform"
            )
        except Exception as e:
            print(f"{Colors.RED}[FEM] Exception during matching: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            
            self.soim.set_object_abandoned(self.current_object['id'])
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f'Match error: {str(e)}'
            }
        
        # ★★★ 新增：验证返回结果 ★★★
        if fem_result is None:
            print(f"{Colors.RED}[FEM] Error: match_rendered_views returned None{Colors.END}")
            self.soim.set_object_abandoned(self.current_object['id'])
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'Match returned None, returning to SOIM'
            }
        
        if not isinstance(fem_result, dict) or 'match_success' not in fem_result:
            print(f"{Colors.RED}[FEM] Error: Invalid result structure{Colors.END}")
            self.soim.set_object_abandoned(self.current_object['id'])
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'Invalid match result, returning to SOIM'
            }
        self._print_lightglue_match_details(fem_result)
        nvsm_visualization_path = self._save_fem_best_match_visualization(fem_result)
        
        if fem_result['match_success']:
            best_pose = fem_result.get('best_pose')
            if best_pose is None:
                print(f"{Colors.RED}[FEM] Error: Match success but best_pose is None{Colors.END}")
                self.soim.set_object_abandoned(self.current_object['id'])
                self.current_state = LocalizationState.SOIM
                self.current_object = None
                return {
                    'status': 'running',
                    'state': self.current_state.value,
                    'message': 'Match success but no pose, returning to SOIM'
                }
            
            # ★★★ 关键修改：记录粗定位位姿，切换到 REF 而不是 EP ★★★
            self.best_pose = best_pose
            self.fem_best_pose = np.asarray(best_pose, dtype=np.float64).copy()
            self.refined_pose = None  # 清空之前的细定位结果

            # ★★★ 计算目标位置信息（用于输出）★★★
            c2w = np.linalg.inv(best_pose)
            camera_position = c2w[:3, 3].copy()
            camera_forward = -c2w[:3, 2]
            target_yaw = np.arctan2(camera_forward[0], -camera_forward[2])
            target_yaw_deg = np.rad2deg(target_yaw)
            
            object_position = None
            if self.current_object is not None and 'position_3d' in self.current_object:
                object_position = self.current_object['position_3d']

            self.fem_best_score = fem_result['best_score']
            if not self.fine_localization_enabled:
                coarse_result = self._make_localization_complete_result(
                    message=(
                        f"FEM coarse localization complete "
                        f"(score={fem_result['best_score']:.3f}); PRE/REF/MR disabled"
                    )
                )
                coarse_result['best_score'] = fem_result['best_score']
                coarse_result['nvsm_visualization_path'] = nvsm_visualization_path
                return coarse_result

            # ★★★ 关键修改：切换到 PRE（前置位姿优化）而非直接 REF ★★★
            self.current_state = LocalizationState.PRE
            
            # ★★★ 新增：从最佳匹配视角中获取其所属的 Instance Center 信息 ★★★
            best_idx = fem_result['best_index']
            best_view_data = self.rendered_images[best_idx] if 0 <= best_idx < len(self.rendered_images) else {}
            instance_center_idx = best_view_data.get('instance_center_idx', None)
            instance_center_3d = best_view_data.get('instance_center_3d', None)
            
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}[FEM] ✓ MATCH SUCCESS ✓{Colors.END}")
            print(f"{Colors.GREEN}  Best View: #{fem_result['best_index']}{Colors.END}")
            if instance_center_idx is not None:
                print(f"{Colors.GREEN}  From Instance Center: #{instance_center_idx} "
                      f"[{instance_center_3d[0]:.3f}, {instance_center_3d[1]:.3f}, {instance_center_3d[2]:.3f}]{Colors.END}")
            print(f"{Colors.GREEN}  Match Score: {fem_result['best_score']:.4f} (threshold: {self.config['match_threshold']}){Colors.END}")
            print(f"{Colors.GREEN}  Coarse Position: [{camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f}]{Colors.END}")
            print(f"{Colors.GREEN}  Coarse Yaw: {target_yaw_deg:.1f}°{Colors.END}")
            print(f"{Colors.GREEN}  → Switching to PRE (PosePreRefiner){Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
            
            self.logger.info(f"Match success with score {fem_result['best_score']:.3f}, switching to PRE")
            
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f"Match found with score {fem_result['best_score']:.3f}, entering PRE",
                'best_score': fem_result['best_score'],
                'best_pose': self.best_pose,
                'camera_position': camera_position,
                'target_yaw': target_yaw_deg,
                'object_position': object_position,
                'nvsm_visualization_path': nvsm_visualization_path
            }
        else:
            # ★★★ 修改：匹配失败 → 全部低于20直接舍弃，不再进入 MR ★★★
            object_id = self.current_object['id'] if self.current_object else 'unknown'
            best_score = fem_result.get('best_score', 0)
            
            self.logger.info(f"Match failed for object {object_id}, score={best_score:.4f}")
            
            print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
            print(f"{Colors.YELLOW}[FEM] ✗ MATCH FAILED ✗{Colors.END}")
            print(f"{Colors.YELLOW}  Object: {object_id}{Colors.END}")
            print(f"{Colors.YELLOW}  Best Score: {best_score:.4f} < {self.config['match_threshold']}{Colors.END}")
            print(f"{Colors.YELLOW}  → 全部匹配度低于{self.config['match_threshold']}，直接舍弃{Colors.END}")
            print(f"{Colors.YELLOW}{'='*70}{Colors.END}\n")
            
            self.soim.set_object_abandoned(object_id)
            self.current_state = LocalizationState.FAILED
            self.current_object = None
            
            return {
                'status': 'failed',
                'state': self.current_state.value,
                'message': f'FEM match failed (score={best_score:.4f} < {self.config["match_threshold"]}), exiting buffer',
                'best_score': best_score,
                'nvsm_visualization_path': nvsm_visualization_path,
            }

    def _save_fem_best_match_visualization(self, fem_result: Dict[str, Any]) -> Optional[str]:
        """Save the best localization FEM render, including below-threshold failures."""
        if not self.save_visualization_results or fem_result is None:
            return None
        try:
            best_image = fem_result.get('best_image')
            best_index = fem_result.get('best_index')
            if best_image is None or best_index is None:
                return None
            best_index = int(best_index)
            best_view_data = (
                self.rendered_images[best_index]
                if 0 <= best_index < len(self.rendered_images)
                else fem_result.get('best_view_data', {})
            )
            source = f"view{best_index}"
            if isinstance(best_view_data, dict):
                scale_index = best_view_data.get('scale_index')
                local_index = best_view_data.get('local_index')
                if scale_index is not None and local_index is not None:
                    source = f"nvsm_s{int(scale_index)}_v{int(local_index)}"
                elif best_view_data.get('index') is not None:
                    source = f"view{int(best_view_data.get('index'))}"

            metadata = {
                'match_success': bool(fem_result.get('match_success', False)),
                'threshold': float(self.config.get('match_threshold', 0.0)),
                'feature_mode': fem_result.get('feature_mode'),
                'best_match_count': int(fem_result.get('best_match_count', 0) or 0),
                'best_match_count_index': self._safe_int_or_default(
                    fem_result.get('best_match_count_index'), -1
                ),
                'all_scores': fem_result.get('all_scores', []),
                'all_match_counts': fem_result.get('all_match_counts', []),
            }
            if isinstance(best_view_data, dict):
                metadata.update(
                    {
                        'scale_index': best_view_data.get('scale_index'),
                        'local_index': best_view_data.get('local_index'),
                        'radius': best_view_data.get('radius'),
                        'instance_center_idx': best_view_data.get('instance_center_idx'),
                        'instance_center_3d': best_view_data.get('instance_center_3d'),
                    }
                )
            saved_path = self.fem.save_best_match(
                best_image,
                best_index,
                score=float(fem_result.get('best_score', 0.0) or 0.0),
                match_count=int(fem_result.get('best_view_match_count', 0) or 0),
                source=source,
                step=self._safe_int_or_none(self._last_keyframe_step),
                metadata=metadata,
                match_cache=fem_result.get('best_match_cache'),
            )
            if saved_path:
                print(f"{Colors.GREEN}[FEM] Saved localization best render: {saved_path}{Colors.END}")
            return saved_path
        except Exception as e:
            print(f"{Colors.YELLOW}[FEM] Warning: Failed to save localization best render: {e}{Colors.END}")
            return None

    @staticmethod
    def _safe_int_or_none(value) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int_or_default(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _execute_pre(self) -> Dict[str, Any]:
        """
        ★★★ 新增：执行前置位姿优化模块 (PRE) ★★★
        
        在FEM粗定位之后、REF细定位之前运行。
        通过多阶段采样+LightGlue匹配来修正粗定位位姿的误差。
        """
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}[PRE] Starting PosePreRefiner{Colors.END}")
        print(f"{Colors.MAGENTA}  Coarse pose available: {self.best_pose is not None}{Colors.END}")
        print(f"{Colors.MAGENTA}  Coarse FEM score: {self.fem_best_score:.2f}{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info("Executing PRE (PosePreRefiner) module")
        
        if self.best_pose is None:
            print(f"{Colors.RED}[PRE] Error: No coarse pose available{Colors.END}")
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'No coarse pose for PRE, returning to SOIM'
            }
        
        try:
            pre_result = self.pre.refine(
                coarse_pose=self.best_pose,
                coarse_score=self.fem_best_score
            )
        except Exception as e:
            print(f"{Colors.RED}[PRE] Exception during pre-refinement: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            
            # PRE失败，直接用原粗定位位姿进入REF
            self.current_state = LocalizationState.REF
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f'PRE failed ({e}), using original coarse pose for REF'
            }
        
        if pre_result.get('success') and pre_result.get('improved'):
            # PRE成功且有改进，更新 best_pose
            self.best_pose = pre_result['refined_pose']
            
            c2w = np.linalg.inv(self.best_pose)
            pre_position = c2w[:3, 3].copy()
            pre_forward = -c2w[:3, 2]
            pre_yaw_deg = np.rad2deg(np.arctan2(pre_forward[0], -pre_forward[2]))
            
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}[PRE] ✓ PRE-REFINEMENT IMPROVED POSE ✓{Colors.END}")
            print(f"{Colors.GREEN}  Score: {pre_result['coarse_score']:.2f} → {pre_result['refined_score']:.2f}{Colors.END}")
            print(f"{Colors.GREEN}  PRE Position: [{pre_position[0]:.3f}, {pre_position[1]:.3f}, {pre_position[2]:.3f}]{Colors.END}")
            print(f"{Colors.GREEN}  PRE Yaw: {pre_yaw_deg:.1f}°{Colors.END}")
            print(f"{Colors.GREEN}  Total renders: {pre_result['total_renders']}, Time: {pre_result['elapsed']:.1f}s{Colors.END}")
            print(f"{Colors.GREEN}  → Switching to REF (Pose Refinement){Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
        else:
            # PRE没有改进，使用原始粗定位位姿
            print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
            print(f"{Colors.YELLOW}[PRE] ⚠ PRE did not improve pose ⚠{Colors.END}")
            print(f"{Colors.YELLOW}  → Using original coarse pose for REF{Colors.END}")
            print(f"{Colors.YELLOW}{'='*70}{Colors.END}\n")
        
        # 无论是否改进，都切换到REF
        self.current_state = LocalizationState.REF
        
        return {
            'status': 'running',
            'state': self.current_state.value,
            'message': pre_result.get('message', 'PRE complete'),
            'best_pose': self.best_pose,
            'pre_result': pre_result
        }

    def _execute_ref(self) -> Dict[str, Any]:
        """
        ★★★ 新增：执行细定位模块 ★★★
        
        对 FEM 得到的粗定位位姿进行梯度下降优化
        """
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}[REF] Starting Pose Refinement{Colors.END}")
        print(f"{Colors.MAGENTA}  Coarse pose from FEM available: {self.best_pose is not None}{Colors.END}")
        print(f"{Colors.MAGENTA}  Refinement tools available: {self.ref._refinement_tools_available}{Colors.END}")
        print(f"{Colors.MAGENTA}  Max iterations: {self.ref.config['max_iterations']}{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info("Executing REF (Pose Refinement) module")
        
        if self.best_pose is None:
            print(f"{Colors.RED}[REF] Error: No coarse pose available{Colors.END}")
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': 'No coarse pose for refinement, returning to SOIM'
            }
        
        # ★★★ 快速检查：如果工具不可用，直接跳过，不假装成功 ★★★
        if not self.ref._refinement_tools_available:
            print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
            print(f"{Colors.YELLOW}[REF] ⚠ REFINEMENT TOOLS NOT AVAILABLE ⚠{Colors.END}")
            print(f"{Colors.YELLOW}  Skipping pose refinement, using coarse pose directly.{Colors.END}")
            print(f"{Colors.YELLOW}  To enable: install diff_gaussian_rasterization and camera_utils{Colors.END}")
            print(f"{Colors.YELLOW}  → Switching to EP with coarse pose{Colors.END}")
            print(f"{Colors.YELLOW}{'='*70}{Colors.END}\n")
            
            self.refined_pose = self.best_pose.copy()
            recovery_request = self._maybe_request_cluster_recovery(
                ref_match_score=0.0,
                ref_pose=self.refined_pose,
            )
            if recovery_request is not None:
                return recovery_request
            
            c2w = np.linalg.inv(self.refined_pose)
            fallback_position = c2w[:3, 3].copy()
            fallback_forward = -c2w[:3, 2]
            fallback_yaw_deg = np.rad2deg(np.arctan2(fallback_forward[0], -fallback_forward[2]))
            
            return self._make_localization_complete_result(
                'Refinement tools not available (fallback: using coarse pose)'
            )
        
        # 获取目标图像
        target_image = self.fem.target_image
        
        # ★★★ 修改：使用深度目标图像和预提取特征点 ★★★
        depth_input = self.target_depth       # 深度目标图像（可为 None）
        keypoints_input = self.target_keypoints  # 预提取特征点（可为 None）
        
        if depth_input is not None:
            print(f"{Colors.CYAN}[REF] Using goal depth image: shape={depth_input.shape}, "
                  f"range=[{depth_input.min():.3f}, {depth_input.max():.3f}]m{Colors.END}")
        else:
            print(f"{Colors.YELLOW}[REF] No goal depth image, running RGB-only refinement{Colors.END}")
        
        if keypoints_input is not None:
            kp_len = len(keypoints_input) if hasattr(keypoints_input, '__len__') else 'unknown'
            print(f"{Colors.CYAN}[REF] Using pre-extracted keypoints: {kp_len} points{Colors.END}")
        else:
            print(f"{Colors.YELLOW}[REF] No pre-extracted keypoints, using edge-based grad_mask only{Colors.END}")
        
        try:
            ref_result = self.ref.refine_pose(
                coarse_pose=self.best_pose,
                target_image=target_image,
                depth_input=depth_input,
                keypoints_input=keypoints_input
            )
        except Exception as e:
            print(f"{Colors.RED}[REF] Exception during refinement: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
            
            # 回退：使用粗定位位姿继续
            self.refined_pose = self.best_pose.copy()
            recovery_request = self._maybe_request_cluster_recovery(
                ref_match_score=0.0,
                ref_pose=self.refined_pose,
            )
            if recovery_request is not None:
                return recovery_request
            return self._make_localization_complete_result(
                f'Refinement failed ({e}), using coarse pose as fallback'
            )
        
        if ref_result['success'] and ref_result['iterations'] > 0:
            self.refined_pose = ref_result['refined_pose']
            
            # 计算细定位后的目标位置和朝向
            c2w = np.linalg.inv(self.refined_pose)
            refined_position = c2w[:3, 3].copy()
            refined_forward = -c2w[:3, 2]
            refined_yaw_deg = np.rad2deg(np.arctan2(refined_forward[0], -refined_forward[2]))
            
            # 粗定位位姿信息
            c2w_coarse = np.linalg.inv(self.best_pose)
            coarse_position = c2w_coarse[:3, 3].copy()
            coarse_forward = -c2w_coarse[:3, 2]
            coarse_yaw_deg = np.rad2deg(np.arctan2(coarse_forward[0], -coarse_forward[2]))
            
            trans_change = ref_result.get('translation_change', 0.0)
            rot_change = ref_result.get('rotation_change', 0.0)
            
            # ★★★ REF 成功后：从细定位位姿重新渲染 + 匹配，决定是否进入闭环定位 ★★★
            _ref_need_mr = False
            _ref_match_score = 0.0
            if hasattr(self, 'mr'):
                try:
                    render_result = self.mr.render_from_pose(self.refined_pose)
                    ref_rgb = render_result.get('rgb')
                    ref_depth = render_result.get('depth')
                    if ref_rgb is not None:
                        _ref_match_score = self._compute_ref_match_score(ref_rgb)
                        cl_threshold = self.config.get('closed_loop_threshold', 40)
                        print(f"{Colors.CYAN}[REF] 闭环评估: 匹配分数={_ref_match_score:.2f}, "
                              f"阈值={cl_threshold}{Colors.END}")
                        if _ref_match_score >= cl_threshold:
                            _ref_need_mr = True
                            self._mr_ref_rgb = ref_rgb
                            self._mr_ref_depth = ref_depth
                            self._mr_ref_pose = self.refined_pose.copy()
                        else:
                            print(f"{Colors.YELLOW}[REF] 匹配分数 {_ref_match_score:.2f} < {cl_threshold}, "
                                  f"将回退到 PRE 位姿导航{Colors.END}")
                except Exception as _mr_e:
                    self.logger.warning(f"[REF] Closed-loop eval failed: {_mr_e}")
                    import traceback; traceback.print_exc()

            # ★★★ 改革：EP已废弃，增加最大闭环轮次检查 ★★★
            max_rounds = self.config.get('max_closed_loop_rounds', 3)

            # ========== 方法1: 目标语义掩码内匹配点检查（优先） ==========
            _goal_mask_matches = None
            _skip_mr_reason = None

            if self.target_semantic_mask is not None:
                # 从 FEM 获取目标图像上的匹配点
                _last_match_cache = getattr(self.fem, '_last_match_cache', None)
                if _last_match_cache is not None:
                    _kpts_target = _last_match_cache.get('kpts_ref')
                    if _kpts_target is not None and len(_kpts_target) > 0:
                        # 检查有多少匹配点落在目标语义掩码内
                        H, W = self.target_semantic_mask.shape
                        _mask_inliers = 0
                        for (x, y) in _kpts_target:
                            ix, iy = int(round(x)), int(round(y))
                            if 0 <= ix < W and 0 <= iy < H and self.target_semantic_mask[iy, ix]:
                                _mask_inliers += 1

                        _goal_mask_matches = _mask_inliers
                        _sem_threshold = 20

                        print(f"{Colors.CYAN}[REF] 目标语义掩码检查: {_mask_inliers} 点在掩码内 (阈值={_sem_threshold}){Colors.END}")

                        if _mask_inliers >= _sem_threshold:
                            _skip_mr_reason = f'goal_semantic_mask ({_mask_inliers}/{_sem_threshold})'

            # ========== 方法2: 逐实例匹配点检查（fallback） ==========
            if _skip_mr_reason is None:
                _per_inst = getattr(self.fem, '_last_per_instance_semantic_matches', [])
                _sem_threshold = 20  # 与 MR Step3b 阈值一致
                _all_covered = (len(_per_inst) > 0 and all(c >= _sem_threshold for c in _per_inst))

                if _all_covered:
                    _skip_mr_reason = f'per-instance ({_per_inst}, threshold={_sem_threshold})'

            # 中心目标语义证据足够时，必须先于 recovery/MR 直接完成定位。
            # 这段门控不能被高 REF/FEM/TC 分数的 recovery 提前返回绕过。
            if _ref_need_mr and _skip_mr_reason is not None:
                print(f"{Colors.GREEN}[REF] ★ 匹配质量满足要求，跳过MR闭环 ★{Colors.END}")
                print(f"{Colors.GREEN}  原因: {_skip_mr_reason}{Colors.END}")
                return self._make_localization_complete_result(
                    f'Match quality sufficient: {_skip_mr_reason}, skipping closed loop',
                    ref_rendered_rgb=ref_rgb,
                    ref_depth=ref_depth
                )

            # The semantic gate has now been evaluated. Preserve the existing
            # recovery/MR behavior only for hypotheses not accepted by it.
            if _skip_mr_reason is None:
                _recovery_request = self._maybe_request_cluster_recovery(
                    ref_match_score=_ref_match_score,
                    ref_pose=self.refined_pose,
                )
                if _recovery_request is not None:
                    return _recovery_request

            # ========== 决策：是否进入MR闭环 ==========
            if _ref_need_mr and self.localization_attempts < max_rounds:
                self.current_state = LocalizationState.MR
            elif _ref_need_mr and self.localization_attempts >= max_rounds:
                # 达到最大闭环轮次 → 直接定位完成
                print(f"{Colors.YELLOW}[REF] 闭环已达最大轮次 {self.localization_attempts}>= {max_rounds}，"
                      f"直接使用当前位姿定位完成{Colors.END}")
                return self._make_localization_complete_result(
                    f'Max closed loop rounds ({self.localization_attempts}) reached, using current pose',
                    ref_rendered_rgb=ref_rgb,
                    ref_depth=ref_depth
                )
            elif _ref_match_score < self.config.get('closed_loop_threshold', 40):
                # REF 的 photometric 优化可能压低 LightGlue 分数；PRE/FEM 已验证过该位姿，
                # 因此低分时回退到 PRE 位姿导航，而不是把目标判成假阳性。
                self.refined_pose = self.best_pose.copy()
                print(f"{Colors.YELLOW}[REF] 匹配分数 {_ref_match_score:.2f} < 阈值, "
                      f"回退到 PRE 位姿并直接定位完成{Colors.END}")
                return self._make_localization_complete_result(
                    f'REF match score {_ref_match_score:.2f} below threshold; '
                    f'fallback to PRE pose'
                )
            
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}[REF] ✓ POSE REFINEMENT COMPLETE ✓{Colors.END}")
            print(f"{Colors.GREEN}  Iterations: {ref_result['iterations']}{Colors.END}")
            print(f"{Colors.GREEN}  Converged: {ref_result['converged']}{Colors.END}")
            print(f"{Colors.GREEN}  Translation change: {trans_change:.4f}m{Colors.END}")
            print(f"{Colors.GREEN}  Rotation change: {rot_change:.4f}°{Colors.END}")
            print(f"{Colors.CYAN}  Coarse  → Position: [{coarse_position[0]:.3f}, {coarse_position[1]:.3f}, {coarse_position[2]:.3f}], Yaw: {coarse_yaw_deg:.1f}°{Colors.END}")
            print(f"{Colors.GREEN}  Refined → Position: [{refined_position[0]:.3f}, {refined_position[1]:.3f}, {refined_position[2]:.3f}], Yaw: {refined_yaw_deg:.1f}°{Colors.END}")
            print(f"{Colors.GREEN}  → Switching to EP (Navigation + Alignment){Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
            
            self.logger.info(
                f"Refinement complete: iter={ref_result['iterations']}, "
                f"Δt={trans_change:.4f}m, Δr={rot_change:.4f}°"
            )
            
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': ref_result['message'],
                'coarse_pose': self.best_pose,
                'refined_pose': self.refined_pose,
                'camera_position': refined_position,
                'target_yaw': refined_yaw_deg,
                'translation_change': trans_change,
                'rotation_change': rot_change,
                'object_position': self.current_object.get('position_3d') if self.current_object else None,
            }
        else:
            # 细定位失败或没有实际执行优化，回退使用粗定位位姿
            self.refined_pose = self.best_pose.copy()
            
            # ★★★ REF 失败时也做闭环评估 ★★★
            _ref_fail_mr = False
            _fail_match_score = 0.0
            if hasattr(self, 'mr'):
                try:
                    render_result = self.mr.render_from_pose(self.refined_pose)
                    ref_rgb = render_result.get('rgb')
                    ref_depth = render_result.get('depth')
                    if ref_rgb is not None:
                        _fail_match_score = self._compute_ref_match_score(ref_rgb)
                        cl_threshold = self.config.get('closed_loop_threshold', 40)
                        if _fail_match_score >= cl_threshold:
                            _ref_fail_mr = True
                            self._mr_ref_rgb = ref_rgb
                            self._mr_ref_depth = ref_depth
                            self._mr_ref_pose = self.refined_pose.copy()
                except Exception:
                    pass

            _recovery_request = self._maybe_request_cluster_recovery(
                ref_match_score=_fail_match_score,
                ref_pose=self.refined_pose,
            )
            if _recovery_request is not None:
                return _recovery_request
            
            # ★★★ 改革：EP已废弃，增加最大闭环轮次检查 ★★★
            max_rounds = self.config.get('max_closed_loop_rounds', 3)
            # ★★★ 语义匹配点充足 → 逐实例检查，全部覆盖才跳过MR闭环 ★★★
            _per_inst = getattr(self.fem, '_last_per_instance_semantic_matches', [])
            _sem_threshold = 20
            _all_covered = (len(_per_inst) > 0 and all(c >= _sem_threshold for c in _per_inst))
            # ★★★ 可视化：保存闭环决策图（REF失败路径）★★★
            try:
                _need_cl = not _all_covered
                _vis_dir = getattr(self, '_results_dir', None) or (self.fem.results_dir if hasattr(self.fem, 'results_dir') else None)
                if self.save_visualization_results and _vis_dir is not None:
                    import os
                    _vis_path = os.path.join(str(_vis_dir), f"closed_loop_decision_ref_fail_attempt{self.localization_attempts}.png")
                    self.fem.save_closed_loop_decision_visualization(
                        save_path=_vis_path,
                        per_instance_matches=_per_inst,
                        sem_threshold=_sem_threshold,
                        need_closed_loop=_need_cl,
                        match_score=_fail_match_score,
                    )
            except Exception as _viz_e:
                self.logger.warning(f"[REF] Closed-loop decision visualization (fail path) failed: {_viz_e}")
            if _ref_fail_mr and _all_covered:
                print(f"{Colors.GREEN}[REF] 逐实例匹配点 {_per_inst}，全部 >= {_sem_threshold} "
                      f"→ 跳过MR闭环，直接定位完成{Colors.END}")
                return self._make_localization_complete_result(
                    f'All {len(_per_inst)} instance(s) covered (per-inst={_per_inst}, '
                    f'threshold={_sem_threshold}), skipping closed loop',
                    ref_rendered_rgb=ref_rgb,
                    ref_depth=ref_depth
                )
            elif _ref_fail_mr and self.localization_attempts < max_rounds:
                self.current_state = LocalizationState.MR
            elif _ref_fail_mr and self.localization_attempts >= max_rounds:
                # 达到最大闭环轮次 → 直接定位完成
                print(f"{Colors.YELLOW}[REF] 闭环已达最大轮次 {self.localization_attempts}>= {max_rounds}，"
                      f"直接使用当前位姿定位完成{Colors.END}")
                return self._make_localization_complete_result(
                    f'Max closed loop rounds ({self.localization_attempts}) reached, using current pose',
                    ref_rendered_rgb=ref_rgb,
                    ref_depth=ref_depth
                )
            elif _fail_match_score < self.config.get('closed_loop_threshold', 40):
                print(f"{Colors.YELLOW}[REF] (失败路径) 匹配分数 {_fail_match_score:.2f} < 阈值, "
                      f"回退到 PRE 位姿并直接定位完成{Colors.END}")
                return self._make_localization_complete_result(
                    f'REF failed/low-score path {_fail_match_score:.2f} below threshold; '
                    f'fallback to PRE pose'
                )

            reason = ref_result.get('message', 'Unknown reason')
            if ref_result.get('success') and ref_result.get('iterations', 0) == 0:
                reason = f"No iterations executed ({reason})"
            
            print(f"\n{Colors.YELLOW}{'='*70}{Colors.END}")
            print(f"{Colors.YELLOW}[REF] ⚠ REFINEMENT DID NOT IMPROVE POSE ⚠{Colors.END}")
            print(f"{Colors.YELLOW}  Success: {ref_result.get('success', False)}, Iterations: {ref_result.get('iterations', 0)}{Colors.END}")
            print(f"{Colors.YELLOW}  Reason: {reason}{Colors.END}")
            print(f"{Colors.YELLOW}  → Switching to EP with coarse pose{Colors.END}")
            print(f"{Colors.YELLOW}{'='*70}{Colors.END}\n")
            
            c2w = np.linalg.inv(self.refined_pose)
            fallback_position = c2w[:3, 3].copy()
            fallback_forward = -c2w[:3, 2]
            fallback_yaw_deg = np.rad2deg(np.arctan2(fallback_forward[0], -fallback_forward[2]))
            
            return {
                'status': 'running',
                'state': self.current_state.value,
                'message': f'Refinement not effective, using coarse pose: {reason}',
                'refined_pose': self.refined_pose,
                'camera_position': fallback_position,
                'target_yaw': fallback_yaw_deg,
            }
    
    def _execute_mr(self) -> Dict[str, Any]:
        """
        ★★★ 修改：闭环定位模块（替换原模糊检测+GaussMI方案）★★★
        
        流程：
        1. 用 REF 渲染图 + 目标图像做 LightGlue 匹配
        2. 语义分割 + 聚类筛选
        3. RANSAC PnP 求解下一探索位姿
        4. 返回位姿供智能体在 buffer 模式中前进
        """
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}[MR] ★ 闭环定位模块启动 ★{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info("Executing MR (closed-loop localization)")
        
        # 获取 REF 阶段缓存的渲染结果
        ref_rgb = getattr(self, '_mr_ref_rgb', None)
        ref_depth = getattr(self, '_mr_ref_depth', None)
        ref_pose = getattr(self, '_mr_ref_pose', None)
        
        if ref_rgb is None or ref_pose is None:
            # 没有缓存，尝试从 refined_pose 重新渲染
            pose_to_use = self.refined_pose if self.refined_pose is not None else self.best_pose
            if pose_to_use is None:
                msg = "No pose available for closed-loop rendering"
                print(f"{Colors.RED}[MR] {msg}{Colors.END}")
                self.soim.set_object_abandoned(
                    self.current_object['id'] if self.current_object else 'unknown')
                self.current_state = LocalizationState.SOIM
                self.current_object = None
                return {'status': 'running', 'state': self.current_state.value, 'message': msg}
            
            render_result = self.mr.render_from_pose(pose_to_use)
            ref_rgb = render_result.get('rgb')
            ref_depth = render_result.get('depth')
            ref_pose = pose_to_use
        
        if ref_rgb is None:
            msg = "Failed to render from REF pose"
            print(f"{Colors.RED}[MR] {msg}{Colors.END}")
            self.soim.set_object_abandoned(
                self.current_object['id'] if self.current_object else 'unknown')
            self.current_state = LocalizationState.SOIM
            self.current_object = None
            return {'status': 'running', 'state': self.current_state.value, 'message': msg}
        
        if ref_depth is None:
            msg = "No depth available from REF render (required for PnP)"
            print(f"{Colors.RED}[MR] {msg}{Colors.END}")
            # 没有深度，无法做 PnP，回退到 EP
            self.current_state = LocalizationState.SUCCESS
            return {'status': 'localization_complete', 'state': self.current_state.value, 'message': msg}
        
        # 获取目标图像和相机参数
        target_image = self.fem.target_image
        camera_params = dict(self.slam_mapper.camera_params)
        cached_match = getattr(self.fem, '_last_match_cache', None)
        goal_all_masks = getattr(self, 'goal_all_masks', None)
        
        # 调用闭环定位主流程
        mr_result = self.mr.run_closed_loop(
            ref_rendered_image=ref_rgb,
            ref_depth=ref_depth,
            ref_pose_w2c=ref_pose,
            target_image=target_image,
            camera_params=camera_params,
            cached_match=cached_match,
            goal_all_masks=goal_all_masks,
        )
        
        # 清理缓存
        self._mr_ref_rgb = None
        self._mr_ref_depth = None
        self._mr_ref_pose = None
        
        if not mr_result.get('success', False):
            # 闭环定位失败 → 通知 runner 退出 buffer 切回探索
            obj_id = self.current_object['id'] if self.current_object else 'unknown'
            print(f"{Colors.YELLOW}[MR] 闭环定位失败: {mr_result['message']}{Colors.END}")
            print(f"{Colors.YELLOW}[MR] 舍弃目标 {obj_id} → 定位失败，退出buffer{Colors.END}")
            self.soim.set_object_abandoned(obj_id)
            self.current_state = LocalizationState.FAILED
            self.current_object = None
            return {
                'status': 'failed',
                'state': self.current_state.value,
                'message': f"Closed-loop failed: {mr_result['message']}"
            }
        
        # 闭环定位成功，返回新位姿供 buffer 模式导航
        new_pose = mr_result['new_pose']
        new_position = mr_result.get('new_position')
        new_yaw = mr_result.get('new_yaw', 0)
        
        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}[MR] ★ 闭环定位成功 → 智能体将前往新位姿 ★{Colors.END}")
        if new_position is not None:
            print(f"{Colors.GREEN}  目标位置: [{new_position[0]:.3f}, "
                  f"{new_position[1]:.3f}, {new_position[2]:.3f}]{Colors.END}")
        print(f"{Colors.GREEN}  目标朝向: {new_yaw:.1f}°{Colors.END}")
        print(f"{Colors.GREEN}  PnP inliers: {mr_result.get('num_inliers', 0)}{Colors.END}")

        # ★★★ 新增：如果 MR 闭环位置距离很近（< 1.0m），使用缓存的 REF 渲染图完成定位（包括反投影）★★★
        # 这避免了绕过反投影逻辑的问题
        ref_rgb = getattr(self, '_mr_ref_rgb', None)
        ref_depth = getattr(self, '_mr_ref_depth', None)

        if ref_rgb is not None and ref_depth is not None:
            # 检查距离（需要知道原始位置，这里用 coarse_pose 的位置）
            if self.best_pose is not None:
                c2w_coarse = np.linalg.inv(self.best_pose)
                coarse_position = c2w_coarse[:3, 3]
                dist_to_new = float(np.linalg.norm(new_position[[0, 2]] - coarse_position[[0, 2]]))

                if dist_to_new < 1.0:
                    print(f"{Colors.GREEN}  → MR闭环距离 {dist_to_new:.2f}m < 1.0m，使用REF渲染图完成定位（含反投影）{Colors.END}")
                    print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")

                    # 使用缓存的 REF 渲染图完成定位，会尝试反投影
                    return self._make_localization_complete_result(
                        message=f'Closed-loop PnP success ({mr_result.get("num_inliers", 0)} inliers), '
                                f'distance {dist_to_new:.2f}m < 1.0m, using cached REF renders for backprojection',
                        ref_rendered_rgb=ref_rgb,
                        ref_depth=ref_depth,
                        target_semantic_mask=self.target_semantic_mask,
                        fem_match_keypoints=self.target_keypoints,
                    )

        print(f"{Colors.GREEN}  → 保持 buffer 模式，到达后重复定位流程{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")

        # 重置状态机，准备到达后的新一轮完整定位
        self.restart_for_closed_loop()

        # 返回 mr_closed_loop_done 状态
        # 外部 runner 应当：
        # 1. 将智能体移动到 new_position
        # 2. 到达后 GaussMI 重新积累 → 通知 SOIM → 完整定位流程
        return {
            'status': 'mr_closed_loop_done',
            'state': self.current_state.value,
            'message': mr_result['message'],
            'new_pose': new_pose,
            'target_position': new_position,
            'target_yaw': new_yaw,
            'num_inliers': mr_result.get('num_inliers', 0),
            'num_matches': mr_result.get('num_matches', 0),
        }

    def _make_localization_complete_result(
            self,
            message: str = "Localization complete",
            ref_rendered_rgb: Optional[np.ndarray] = None,
            ref_depth: Optional[np.ndarray] = None,
            target_semantic_mask: Optional[np.ndarray] = None,
            fem_match_keypoints: Optional[Dict] = None) -> Dict[str, Any]:
        """★★★ 新增：构造定位完成结果（取代原EP模块入口） ★★★

        直接返回定位位姿，由外部 runner 导航至目标。无需经过 EP 模块。

        ★★★ 修改：从REF相机视角内的目标语义高斯簇选择导航目标 ★★★
        """
        nav_pose = self.refined_pose if self.refined_pose is not None else self.best_pose
        self.current_state = LocalizationState.SUCCESS

        if nav_pose is not None:
            c2w = np.linalg.inv(nav_pose)
            camera_position = c2w[:3, 3].copy()
            # REF/NVSM poses are c2w_habitat @ flip_mat, so c2w[:, 2]
            # is the real Habitat forward direction after the SLAM flip.
            forward = c2w[:3, 2]
            yaw_deg = np.rad2deg(np.arctan2(forward[0], -forward[2]))
        else:
            camera_position = None
            yaw_deg = 0.0

        target_position = None
        position_source = "unresolved"
        cluster_info = None
        if nav_pose is not None:
            cluster_position, cluster_info = self._select_target_gaussian_cluster_from_ref_view(nav_pose)
            if cluster_position is not None:
                target_position = cluster_position
                position_source = "gaussian_cluster_view"
        if target_position is None:
            target_position, cluster_info = self._select_target_object_fallback_position()
            if target_position is not None:
                position_source = cluster_info.get("source", "target_object_fallback")
        if target_position is None:
            self.current_state = LocalizationState.FAILED
            fail_message = (
                f"{message}; target object position unavailable, refusing REF camera fallback"
            )
            print(f"\n{Colors.RED}{'='*70}{Colors.END}")
            print(f"{Colors.RED}✗ LOCALIZATION FAILED: NO TARGET OBJECT POSITION ✗{Colors.END}")
            print(f"{Colors.RED}  Message: {fail_message}{Colors.END}")
            if camera_position is not None:
                print(f"{Colors.RED}  REF camera fallback suppressed: "
                      f"[{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]{Colors.END}")
            print(f"{Colors.RED}  State History: {' → '.join(self.state_history)}{Colors.END}")
            print(f"{Colors.RED}{'='*70}{Colors.END}\n")
            return {
                'status': 'failed',
                'state': self.current_state.value,
                'message': fail_message,
                'best_pose': nav_pose,
                'target_position': None,
                'target_yaw': yaw_deg,
                'position_source': position_source,
                'camera_position_suppressed': camera_position,
            }

        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}★ LOCALIZATION COMPLETE (EP abolished) ★{Colors.END}")
        print(f"{Colors.GREEN}  Message: {message}{Colors.END}")
        print(f"{Colors.GREEN}  Position Source: {position_source}{Colors.END}")
        if target_position is not None:
            print(f"{Colors.GREEN}  Position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]{Colors.END}")
            if position_source != "camera" and camera_position is not None:
                offset = np.linalg.norm(target_position[[0, 2]] - camera_position[[0, 2]])
                print(f"{Colors.GREEN}  Camera pos offset: {offset:.3f}m{Colors.END}")
            if cluster_info is not None and cluster_info.get('center') is not None:
                center = cluster_info['center']
                details = [
                    f"center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]",
                    f"source={cluster_info.get('source', position_source)}",
                ]
                if 'size' in cluster_info:
                    details.append(f"size={cluster_info['size']}")
                if 'dist_xz' in cluster_info:
                    details.append(f"dist={cluster_info['dist_xz']:.2f}m")
                if 'angle_deg' in cluster_info:
                    details.append(f"angle={cluster_info['angle_deg']:.1f}°")
                details.append(f"reachable={cluster_info.get('reachable', False)}")
                print(f"{Colors.GREEN}  Target object: {' | '.join(details)}{Colors.END}")
            print(f"{Colors.GREEN}  Yaw: {yaw_deg:.1f}°{Colors.END}")
        print(f"{Colors.GREEN}  State History: {' → '.join(self.state_history)}{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")

        target_object_center = target_position
        if cluster_info is not None and cluster_info.get('center') is not None:
            center = np.asarray(cluster_info.get('center'), dtype=np.float32).reshape(-1)
            if center.size >= 3 and np.all(np.isfinite(center[:3])):
                target_object_center = center[:3].copy()

        return {
            'status': 'localization_complete',
            'state': self.current_state.value,
            'message': message,
            'best_pose': nav_pose,
            'target_position': target_position,
            'target_object_center': target_object_center,
            'target_yaw': yaw_deg,
            'position_source': position_source,  # 新增：记录位置来源
        }

    def _select_target_object_fallback_position(
            self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Return an object-derived navigation point without using REF camera position."""
        semantic_position, semantic_info = self._target_semantic_centroid_position()
        if semantic_position is not None:
            return semantic_position, semantic_info

        soim_position = None
        if self.current_object is not None:
            soim_position = self.current_object.get('position_3d')
        if soim_position is None:
            return None, None

        center = np.asarray(soim_position, dtype=np.float32).reshape(-1)
        if center.size < 3 or not np.all(np.isfinite(center[:3])):
            return None, None
        center = center[:3].astype(np.float32)
        reachable_pos = self._find_nearest_reachable_point(center)
        nav_position = reachable_pos if reachable_pos is not None else center.copy()
        info = {
            'source': 'soim_target_position',
            'center': center,
            'size': 0,
            'reachable': reachable_pos is not None,
        }
        print(f"{Colors.YELLOW}[TargetObjectFallback] Using SOIM target position "
              f"nav=[{nav_position[0]:.3f}, {nav_position[1]:.3f}, {nav_position[2]:.3f}], "
              f"reachable={info['reachable']}{Colors.END}")
        return nav_position, info

    def _target_semantic_centroid_position(
            self) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """Use all target semantic gaussians as an object fallback, independent of REF FOV."""
        if self.slam_mapper is None or not hasattr(self.slam_mapper, 'slam'):
            return None, None
        slam = self.slam_mapper.slam
        if slam is None or not hasattr(slam, 'gaussians'):
            return None, None
        gaussians = slam.gaussians
        if not hasattr(gaussians, 'get_xyz') or gaussians.get_xyz.shape[0] == 0:
            return None, None

        try:
            import torch
            target_indices = None
            if hasattr(slam, 'get_target_semantic_gaussians'):
                target_indices = slam.get_target_semantic_gaussians()
            if target_indices is None:
                semantic_threshold = float(self.config.get('gaussian_view_semantic_threshold', 0.4))
                semantic_scores = getattr(gaussians, '_semantic_target', None)
                if semantic_scores is None or semantic_scores.shape[0] != gaussians.get_xyz.shape[0]:
                    return None, None
                target_indices = torch.where(semantic_scores > semantic_threshold)[0]

            if hasattr(target_indices, 'detach'):
                target_indices_np = target_indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
                target_indices_tensor = target_indices.long().to(gaussians.get_xyz.device)
            else:
                target_indices_np = np.asarray(target_indices, dtype=np.int64).reshape(-1)
                target_indices_tensor = torch.as_tensor(
                    target_indices_np, dtype=torch.long, device=gaussians.get_xyz.device)
            if target_indices_np.size == 0:
                return None, None

            xyz = gaussians.get_xyz[target_indices_tensor].detach().cpu().numpy()
            if xyz.size == 0:
                return None, None
            center = xyz.mean(axis=0).astype(np.float32)
        except Exception as e:
            print(f"{Colors.YELLOW}[TargetObjectFallback] Failed to compute semantic centroid ({e}){Colors.END}")
            return None, None

        try:
            reachable_pos = self._find_nearest_reachable_point(center)
        except Exception as e:
            print(f"{Colors.YELLOW}[TargetObjectFallback] Reachable-point search failed ({e}); "
                  f"using semantic centroid{Colors.END}")
            reachable_pos = None
        nav_position = reachable_pos if reachable_pos is not None else center.copy()
        info = {
            'source': 'target_semantic_gaussian_centroid',
            'center': center,
            'size': int(xyz.shape[0]),
            'reachable': reachable_pos is not None,
        }
        print(f"{Colors.YELLOW}[TargetObjectFallback] Using target semantic Gaussian centroid "
              f"size={info['size']}, nav=[{nav_position[0]:.3f}, {nav_position[1]:.3f}, {nav_position[2]:.3f}], "
              f"reachable={info['reachable']}{Colors.END}")
        return nav_position, info

    def _select_target_gaussian_cluster_from_ref_view(
            self,
            ref_pose_w2c: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        从REF渲染相机前方视角内选择目标语义高斯簇作为导航目标。

        策略：
        1. 从当前目标语义高斯中 DBSCAN 聚类。
        2. 用簇中心判断是否在REF相机forward方向±60°内。
        3. 在视角内候选簇中选择XZ距离相机最近的簇。
        4. 对簇中心调用现有最近可达点逻辑；若找不到可达点，仍返回簇中心。
        """
        from ieve_minav.localizer.colors import Colors

        if self.slam_mapper is None or not hasattr(self.slam_mapper, 'slam'):
            print(f"{Colors.YELLOW}[GaussianViewFallback] SLAM mapper unavailable, trying object fallback{Colors.END}")
            return None, None
        slam = self.slam_mapper.slam
        if slam is None or not hasattr(slam, 'gaussians'):
            print(f"{Colors.YELLOW}[GaussianViewFallback] SLAM/gaussians unavailable, trying object fallback{Colors.END}")
            return None, None

        gaussians = slam.gaussians
        if not hasattr(gaussians, 'get_xyz') or gaussians.get_xyz.shape[0] == 0:
            print(f"{Colors.YELLOW}[GaussianViewFallback] No gaussians available, trying object fallback{Colors.END}")
            return None, None

        try:
            c2w = np.linalg.inv(ref_pose_w2c)
            camera_position = c2w[:3, 3].copy()
            # REF/NVSM poses are c2w_habitat @ flip_mat, so c2w[:, 2]
            # is the real Habitat forward direction after the SLAM flip.
            forward = c2w[:3, 2]
            forward_xz = np.array([forward[0], forward[2]], dtype=np.float64)
            forward_norm = np.linalg.norm(forward_xz)
            if forward_norm < 1e-6:
                print(f"{Colors.YELLOW}[GaussianViewFallback] Invalid camera forward, trying object fallback{Colors.END}")
                return None, None
            forward_xz = forward_xz / forward_norm
        except Exception as e:
            print(f"{Colors.YELLOW}[GaussianViewFallback] Failed to parse REF pose ({e}), trying object fallback{Colors.END}")
            return None, None

        target_indices = None
        if hasattr(slam, 'get_target_semantic_gaussians'):
            try:
                target_indices = slam.get_target_semantic_gaussians()
            except Exception as e:
                print(f"{Colors.YELLOW}[GaussianViewFallback] get_target_semantic_gaussians failed: {e}{Colors.END}")

        try:
            import torch
            if target_indices is None:
                semantic_threshold = float(self.config.get('gaussian_view_semantic_threshold', 0.4))
                semantic_scores = getattr(gaussians, '_semantic_target', None)
                if semantic_scores is None or semantic_scores.shape[0] != gaussians.get_xyz.shape[0]:
                    print(f"{Colors.YELLOW}[GaussianViewFallback] No target semantic scores, trying object fallback{Colors.END}")
                    return None, None
                target_indices = torch.where(semantic_scores > semantic_threshold)[0]

            if hasattr(target_indices, 'detach'):
                target_indices_np = target_indices.detach().cpu().numpy().astype(np.int64)
                target_indices_tensor = target_indices.long().to(gaussians.get_xyz.device)
            else:
                target_indices_np = np.asarray(target_indices, dtype=np.int64).reshape(-1)
                target_indices_tensor = torch.as_tensor(
                    target_indices_np, dtype=torch.long, device=gaussians.get_xyz.device)

            if len(target_indices_np) == 0:
                print(f"{Colors.YELLOW}[GaussianViewFallback] No target semantic gaussians, trying object fallback{Colors.END}")
                return None, None

            xyz = gaussians.get_xyz[target_indices_tensor].detach().cpu().numpy()
        except Exception as e:
            print(f"{Colors.YELLOW}[GaussianViewFallback] Failed to collect target gaussians ({e}), trying object fallback{Colors.END}")
            return None, None

        eps = float(self.config.get('gaussian_view_cluster_eps', 0.15))
        min_samples = int(self.config.get('gaussian_view_cluster_min_samples', 10))
        min_cluster_size = int(self.config.get('gaussian_view_cluster_min_size', 20))
        angle_deg = float(self.config.get('gaussian_view_angle_deg', 60.0))
        cos_threshold = float(np.cos(np.deg2rad(angle_deg)))

        try:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyz)
        except Exception as e:
            print(f"{Colors.YELLOW}[GaussianViewFallback] DBSCAN failed ({e}), trying object fallback{Colors.END}")
            return None, None

        print(f"{Colors.CYAN}[GaussianViewFallback] Selecting target cluster from REF view: "
              f"target_gaussians={len(xyz)}, eps={eps}, min_samples={min_samples}, "
              f"min_size={min_cluster_size}, fov=±{angle_deg:.0f}°{Colors.END}")

        candidates = []
        rejected_small = 0
        rejected_angle = 0
        unique_labels = sorted(set(labels.tolist()) - {-1})
        for label_id in unique_labels:
            cluster_xyz = xyz[labels == label_id]
            cluster_size = int(len(cluster_xyz))
            if cluster_size < min_cluster_size:
                rejected_small += 1
                continue

            center = cluster_xyz.mean(axis=0).astype(np.float32)
            to_cluster_xz = np.array([
                float(center[0] - camera_position[0]),
                float(center[2] - camera_position[2]),
            ], dtype=np.float64)
            dist_xz = float(np.linalg.norm(to_cluster_xz))
            if dist_xz < 1e-6:
                cos_angle = 1.0
            else:
                cos_angle = float(np.dot(forward_xz, to_cluster_xz / dist_xz))
                cos_angle = max(-1.0, min(1.0, cos_angle))
            cluster_angle_deg = float(np.rad2deg(np.arccos(cos_angle)))

            if cos_angle < cos_threshold:
                rejected_angle += 1
                print(f"{Colors.CYAN}[GaussianViewFallback] Reject cluster {label_id}: "
                      f"size={cluster_size}, dist={dist_xz:.2f}m, angle={cluster_angle_deg:.1f}° > {angle_deg:.0f}°{Colors.END}")
                continue

            print(f"{Colors.CYAN}[GaussianViewFallback] Candidate cluster {label_id}: "
                  f"size={cluster_size}, center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}], "
                  f"dist={dist_xz:.2f}m, angle={cluster_angle_deg:.1f}°{Colors.END}")
            candidates.append({
                'source': 'gaussian_cluster_view',
                'label': int(label_id),
                'center': center,
                'size': cluster_size,
                'dist_xz': dist_xz,
                'cos_angle': cos_angle,
                'angle_deg': cluster_angle_deg,
            })

        if not candidates:
            print(f"{Colors.YELLOW}[GaussianViewFallback] No target gaussian cluster inside REF view "
                  f"(clusters={len(unique_labels)}, small_reject={rejected_small}, angle_reject={rejected_angle}); "
                  f"trying object fallback{Colors.END}")
            return None, None

        selected = min(candidates, key=lambda c: (c['dist_xz'], -c['size'], -c['cos_angle']))
        selected_center = selected['center'].copy()
        try:
            reachable_pos = self._find_nearest_reachable_point(selected_center)
        except Exception as e:
            print(f"{Colors.YELLOW}[GaussianViewFallback] Reachable-point search failed ({e}); "
                  f"using cluster center{Colors.END}")
            reachable_pos = None
        selected['reachable'] = reachable_pos is not None
        nav_position = reachable_pos if reachable_pos is not None else selected_center

        print(f"{Colors.GREEN}[GaussianViewFallback] Selected cluster {selected['label']}: "
              f"size={selected['size']}, dist={selected['dist_xz']:.2f}m, "
              f"angle={selected['angle_deg']:.1f}°, "
              f"nav=[{nav_position[0]:.3f}, {nav_position[1]:.3f}, {nav_position[2]:.3f}], "
              f"reachable={selected['reachable']}{Colors.END}")

        return nav_position, selected

    def _resolve_ref_backprojection_inputs(
            self,
            target_semantic_mask: Optional[np.ndarray] = None,
            fem_match_keypoints: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """补齐 REF 反投影需要的语义 mask 和 FEM 匹配点。"""
        if target_semantic_mask is None:
            target_semantic_mask = getattr(self, 'target_semantic_mask', None)
        if target_semantic_mask is None:
            target_semantic_mask = getattr(self, 'goal_center_mask', None)

        if not self._has_ref_backprojection_keypoints(fem_match_keypoints):
            fem_match_keypoints = self._adapt_fem_match_cache_for_backprojection()

        return target_semantic_mask, fem_match_keypoints

    def _has_ref_backprojection_keypoints(self, keypoints: Optional[Dict]) -> bool:
        if not isinstance(keypoints, dict):
            return False
        return keypoints.get('kpts_ref') is not None and keypoints.get('kpts_target') is not None

    def _adapt_fem_match_cache_for_backprojection(self) -> Optional[Dict]:
        cache = getattr(getattr(self, 'fem', None), '_last_match_cache', None)
        if not isinstance(cache, dict):
            return None

        # FeatureMatcher stores target-image points as kpts_ref and rendered-query
        # points as kpts_query. REF backprojection needs the rendered points first.
        kpts_render = cache.get('kpts_query')
        kpts_target = cache.get('kpts_ref')
        if kpts_render is None or kpts_target is None:
            return None

        return {
            'kpts_ref': kpts_render,
            'kpts_target': kpts_target,
            'scores': cache.get('scores'),
            'num_matches': cache.get('num_matches'),
        }

    def _execute_ep(self) -> Dict[str, Any]:
        """执行探索规划模块"""
        # ★★★ 状态输出 ★★★
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}[EP] Starting Navigation to Target{Colors.END}")
        
        # ★★★ 修改：优先使用细定位位姿 ★★★
        nav_pose = self.refined_pose if self.refined_pose is not None else self.best_pose
        pose_source = "refined" if self.refined_pose is not None else "coarse"
        print(f"{Colors.MAGENTA}  Using {pose_source} pose{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info(f"Executing EP module with {pose_source} pose")
        
        # ★★★ 修改：使用选定的位姿进行导航 ★★★
        ep_result = self.ep.navigate_to_pose(nav_pose)
        
        if ep_result['success']:
            self.current_state = LocalizationState.SUCCESS
            
            print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
            print(f"{Colors.GREEN}{'★'*30} SUCCESS {'★'*30}{Colors.END}")
            print(f"{Colors.GREEN}[EP] ✓ NAVIGATION COMPLETED ✓{Colors.END}")
            print(f"{Colors.GREEN}  Pose source: {pose_source}{Colors.END}")
            print(f"{Colors.GREEN}{'★'*30} {'★'*30}{Colors.END}")
            print(f"{Colors.GREEN}  Final Pose: {ep_result.get('final_pose')}{Colors.END}")
            print(f"{Colors.GREEN}  Distance to Goal: {ep_result.get('distance_to_goal', 0):.3f}m{Colors.END}")
            print(f"{Colors.CYAN}  State History: {' → '.join(self.state_history)}{Colors.END}")
            print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")
            
            return {
                'status': 'success',
                'state': self.current_state.value,
                'message': 'Localization and navigation completed',
                'final_pose': ep_result.get('final_pose')
            }
        else:
            self.current_state = LocalizationState.FAILED
            
            print(f"\n{Colors.RED}{'='*70}{Colors.END}")
            print(f"{Colors.RED}[EP] ✗ NAVIGATION FAILED ✗{Colors.END}")
            print(f"{Colors.RED}  Reason: {ep_result['message']}{Colors.END}")
            print(f"{Colors.CYAN}  State History: {' → '.join(self.state_history)}{Colors.END}")
            print(f"{Colors.RED}{'='*70}{Colors.END}\n")
            
            return {
                'status': 'failed',
                'state': self.current_state.value,
                'message': f"Navigation failed: {ep_result['message']}"
            }
    
    def reset(self):
        """重置定位系统状态（episode 切换时调用）"""
        # 状态机
        self.current_state = LocalizationState.IDLE
        self.current_object = None
        self.best_pose = None
        self.fem_best_pose = None
        self.refined_pose = None
        self.rendered_images = []

        # ★★★ 修复：重置跨 episode 累积的变量 ★★★
        self.fem_best_score = 0.0
        self.localization_attempts = 0
        self.state_history = []
        self.target_depth = None
        self.target_keypoints = None
        self.target_semantic_mask = None  # ★★★ 新增：重置目标语义掩码 ★★
        self.cluster_recovery_used = False
        self.cluster_recovery_requested = False
        self.cluster_recovery_tc_evidence = {
            'score': 0.0,
            'pose': None,
            'center': None,
        }

        # ★★★ 清理闭环定位缓存 ★★★
        self._mr_ref_rgb = None
        self._mr_ref_depth = None
        self._mr_ref_pose = None
        
        # 重置所有子模块
        self.soim.reset()
        if self.pre is not None:
            self.pre.reset()
        if self.ref is not None:
            self.ref.reset()
        # self.ep.reset()  # ★★★ EP已废弃 ★★★
        if hasattr(self, 'mr') and self.mr is not None:
            self.mr.reset()
        
        print(f"\n{Colors.CYAN}[LocalizationSystem] System reset{Colors.END}\n")
        self.logger.info("LocalizationSystem reset")
    
    def _clear_closed_loop_attempt_state(self):
        """清理上一轮定位状态，但不改变闭环轮次计数。"""
        self.current_state = LocalizationState.SOIM
        self.current_object = None
        self.best_pose = None
        self.fem_best_pose = None
        self.refined_pose = None
        self.fem_best_score = 0.0
        self.rendered_images = []
        self._mr_ref_rgb = None
        self._mr_ref_depth = None
        self._mr_ref_pose = None
        
        # 重置SOIM的GaussMI就绪标志，等待新一轮GaussMI完成
        self.soim.gaussmi_ready = False
        self.soim.clear_candidate_anchor()
        
        # 重置PRE和REF子模块
        if self.pre is not None:
            self.pre.reset()
        if self.ref is not None:
            self.ref.reset()

    def restart_for_closed_loop(self):
        """MR闭环成功后，重置状态机以准备新一轮 SOIM→NVSM→FEM→PRE→REF→MR"""
        self._clear_closed_loop_attempt_state()
        
        self.localization_attempts += 1
        
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}[LocalizationSystem] ★ 闭环成功 → 重启定位流程 (第{self.localization_attempts}轮) ★{Colors.END}")
        print(f"{Colors.CYAN}  状态: SOIM (等待GaussMI就绪){Colors.END}")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

    def inject_closed_loop_pose_for_pre(
            self,
            pnp_pose: np.ndarray,
            score_hint: Optional[float] = None,
            source: str = "MR remap") -> None:
        """真正把闭环 PnP 位姿接入 PRE，而不是缓存后立刻丢弃。"""
        if pnp_pose is None:
            raise ValueError("pnp_pose is None")

        # MR成功时已经计入一轮；这里只清理状态并注入闭环位姿，避免同一次闭环重复计数。
        self._clear_closed_loop_attempt_state()

        pose = np.array(pnp_pose, dtype=np.float64, copy=True)
        self.best_pose = pose
        self.fem_best_pose = pose.copy()
        self.refined_pose = None
        self.current_state = LocalizationState.PRE

        if score_hint is None:
            score_hint = float(self.config.get('closed_loop_threshold', 40))
        self.fem_best_score = float(score_hint)

        c2w = np.linalg.inv(pose)
        camera_position = c2w[:3, 3].copy()
        camera_forward = -c2w[:3, 2]
        yaw_deg = np.rad2deg(np.arctan2(camera_forward[0], -camera_forward[2]))

        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}[LocalizationSystem] ★ 注入闭环PnP位姿到 PRE ★{Colors.END}")
        print(f"{Colors.GREEN}  Source: {source}{Colors.END}")
        print(f"{Colors.GREEN}  Coarse Position: "
              f"[{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}]{Colors.END}")
        print(f"{Colors.GREEN}  Coarse Yaw: {yaw_deg:.1f}°{Colors.END}")
        print(f"{Colors.GREEN}  PRE Score Hint: {self.fem_best_score:.2f}{Colors.END}")
        print(f"{Colors.GREEN}  → Next localization step will execute PRE{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")

        self.logger.info(
            f"Injected closed-loop pose into PRE from {source}: "
            f"pos={camera_position.tolist()}, yaw={yaw_deg:.1f}, score_hint={self.fem_best_score:.2f}"
        )

    def set_cluster_recovery_context(
            self,
            tc_score: Optional[float],
            tc_pose: Optional[np.ndarray],
            tc_center: Optional[np.ndarray] = None) -> None:
        """Provide the TC evidence that must compete with FEM and REF evidence."""
        pose = None
        if tc_pose is not None:
            candidate = np.asarray(tc_pose, dtype=np.float64)
            if candidate.shape == (4, 4) and np.all(np.isfinite(candidate)):
                pose = candidate.copy()
        center = None
        if tc_center is not None:
            candidate = np.asarray(tc_center, dtype=np.float32).reshape(-1)
            if candidate.size >= 3 and np.all(np.isfinite(candidate[:3])):
                center = candidate[:3].copy()
        self.cluster_recovery_tc_evidence = {
            'score': float(tc_score or 0.0),
            'pose': pose,
            'center': center,
        }

    def _maybe_request_cluster_recovery(
            self,
            ref_match_score: float,
            ref_pose: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if not bool(self.config.get('cluster_recovery_enabled', True)):
            return None
        if self.cluster_recovery_used or self.cluster_recovery_requested:
            return None

        tc = dict(self.cluster_recovery_tc_evidence or {})
        evidence = []
        for source, score, pose in (
                ('target_confirmation', tc.get('score', 0.0), tc.get('pose')),
                (
                    'localization_fem',
                    self.fem_best_score,
                    getattr(self, 'fem_best_pose', None)
                    if getattr(self, 'fem_best_pose', None) is not None
                    else self.best_pose,
                ),
                ('ref', ref_match_score, ref_pose)):
            try:
                score = float(score or 0.0)
                pose_array = np.asarray(pose, dtype=np.float64)
            except Exception:
                continue
            if pose_array.shape != (4, 4) or not np.all(np.isfinite(pose_array)):
                continue
            evidence.append({
                'source': source,
                'score': score,
                'pose': pose_array.copy(),
            })

        if not evidence:
            return None
        strongest = max(evidence, key=lambda item: float(item['score']))
        threshold = float(self.config.get('cluster_recovery_trigger_score', 100.0))
        if float(strongest['score']) <= threshold:
            return None

        self.cluster_recovery_requested = True
        self.current_state = LocalizationState.RECOVERY_WAIT
        return {
            'status': 'cluster_recovery_requested',
            'state': self.current_state.value,
            'message': '{} score {:.2f} > {:.2f}'.format(
                strongest['source'], strongest['score'], threshold
            ),
            'trigger_source': strongest['source'],
            'trigger_score': float(strongest['score']),
            'trigger_threshold': threshold,
            'base_pose': strongest['pose'],
            'tc_score': float(tc.get('score', 0.0) or 0.0),
            'fem_score': float(self.fem_best_score or 0.0),
            'ref_score': float(ref_match_score or 0.0),
            'reference_center': tc.get('center'),
        }

    def mark_cluster_recovery_started(self) -> None:
        self.cluster_recovery_used = True
        self.cluster_recovery_requested = False
        self.current_state = LocalizationState.RECOVERY_WAIT

    def resume_after_cluster_recovery(
            self,
            best_pose: np.ndarray,
            score: float,
            cluster_center: Optional[np.ndarray] = None,
            match_cache: Optional[Dict[str, Any]] = None) -> None:
        pose = np.asarray(best_pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError('Invalid cluster-recovery pose')
        self.cluster_recovery_used = True
        self.cluster_recovery_requested = False
        self.best_pose = pose.copy()
        self.fem_best_pose = pose.copy()
        self.refined_pose = None
        self.fem_best_score = float(score)
        self._mr_ref_rgb = None
        self._mr_ref_depth = None
        self._mr_ref_pose = None
        if match_cache is not None:
            self.fem._last_match_cache = match_cache
        if cluster_center is not None and self.current_object is not None:
            center = np.asarray(cluster_center, dtype=np.float32).reshape(-1)
            if center.size >= 3:
                self.current_object['position_3d'] = center[:3].copy()
        if self.pre is not None:
            self.pre.reset()
        if self.ref is not None:
            self.ref.reset()
        self.current_state = LocalizationState.PRE

    def skip_cluster_recovery_to_mr(self) -> None:
        """Use the original REF hypothesis and continue with the existing MR path."""
        self.cluster_recovery_used = True
        self.cluster_recovery_requested = False
        self.current_state = LocalizationState.MR
    
    def update_results_dir(self, results_dir):
        """
        ★★★ 新增：更新所有子模块的 results_dir（每次 episode 切换后调用）★★★
        
        防止 NVSM/PRE/REF 使用 sorted(subdirs)[-1] 回退逻辑
        导致渲染结果保存到错误的（非当前 episode 的）目录。
        
        Args:
            results_dir: 当前 episode 的输出目录 (Path 或 str)
        """
        results_dir_str = str(results_dir) if results_dir is not None else None
        from pathlib import Path
        
        if hasattr(self, 'nvsm') and self.nvsm is not None:
            self.nvsm.results_dir = results_dir_str

        if hasattr(self, 'fem') and self.fem is not None:
            self.fem.results_dir = Path(results_dir_str) if results_dir_str is not None else None
            if (
                self.fem.results_dir is not None
                and getattr(self.fem, 'save_visualization_results', True)
            ):
                self.fem.results_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self, 'pre') and self.pre is not None:
            self.pre.results_dir = results_dir_str
        
        if hasattr(self, 'ref') and self.ref is not None:
            self.ref.results_dir = results_dir_str
        
        if hasattr(self, 'mr') and self.mr is not None:
            self.mr.results_dir = results_dir_str
        
        self.logger.info(f"LocalizationSystem: results_dir updated to {results_dir_str}")

    def update_mappers(self, slam_mapper, occupancy_mapper):
        """
        更新所有子模块中的 mapper 引用（在 reset_for_goal_image_task 后调用）
        
        Args:
            slam_mapper: 新的 SLAMMapper 实例
            occupancy_mapper: 新的 OccupancyMapper 实例
        """
        self.slam_mapper = slam_mapper
        self.occupancy_mapper = occupancy_mapper
        
        # 同步 occupancy_mapper 到 slam_mapper
        slam_mapper.occupancy_mapper = occupancy_mapper
        
        # 更新子模块引用
        if hasattr(self, 'nvsm') and self.nvsm is not None:
            self.nvsm.slam_mapper = slam_mapper
        
        if hasattr(self, 'soim') and self.soim is not None:
            self.soim.occupancy_mapper = occupancy_mapper
        
        if hasattr(self, 'ref') and self.ref is not None:
            self.ref.slam_mapper = slam_mapper
        
        if hasattr(self, 'pre') and self.pre is not None:
            self.pre.slam_mapper = slam_mapper
        
        if hasattr(self, 'mr') and self.mr is not None:
            self.mr.slam_mapper = slam_mapper
        
        self.logger.info("LocalizationSystem: mapper references updated")
    
    def start_localization(self):
        """启动定位模式"""
        self.current_state = LocalizationState.SOIM
        self.localization_attempts += 1
        self.state_history = ['idle']
        
        # ★★★ 启动输出 ★★★
        print(f"\n{Colors.MAGENTA}{'='*70}{Colors.END}")
        print(f"{Colors.MAGENTA}★ Localization Mode STARTED (Attempt #{self.localization_attempts}) ★{Colors.END}")
        print(f"{Colors.MAGENTA}  Initial State: {self.current_state.value.upper()}{Colors.END}")
        print(f"{Colors.MAGENTA}  SOIM awaiting GaussMI ready signal...{Colors.END}")
        print(f"{Colors.MAGENTA}{'='*70}{Colors.END}\n")
        
        self.logger.info("Localization mode started")
    
    def notify_gaussmi_ready(self):
        """
        ★★★ 新增：GaussMI 完成多视角采集后调用 ★★★
        
        通知 SOIM 解锁 NVSM 触发条件，
        下一帧 SOIM 将选择最佳追踪物体并触发 NVSM → FEM → ...
        """
        if hasattr(self, 'soim') and self.soim is not None:
            self.soim.set_gaussmi_ready()
            print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
            print(f"{Colors.GREEN}★ LocalizationSystem: GaussMI READY → SOIM unlocked ★{Colors.END}")
            print(f"{Colors.GREEN}  SOIM will trigger NVSM on next keyframe{Colors.END}")
            print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
        else:
            self.logger.warning("notify_gaussmi_ready called but SOIM not available")

    def set_candidate_anchor(self, anchor_position, radius_m: float = 3.0):
        """把当前 buffer 候选位置同步给 SOIM，限制后续只在该位置附近触发 NVSM。"""
        if hasattr(self, 'soim') and self.soim is not None:
            self.soim.set_candidate_anchor(anchor_position, radius_m=radius_m)

    def clear_candidate_anchor(self):
        """清除 SOIM 的 buffer 候选锚点。"""
        if hasattr(self, 'soim') and self.soim is not None:
            self.soim.clear_candidate_anchor()
        
    def _compute_ref_match_score(self, ref_rendered_image: np.ndarray) -> float:
        """
        ★★★ 新增：对 REF 渲染图和目标图像做 LightGlue 匹配，计算证据分数 ★★★
        
        用于 REF 后判断是否进入闭环定位（阈值 40）
        
        Args:
            ref_rendered_image: REF 渲染的 RGB 图像 (H, W, 3) uint8
            
        Returns:
            匹配证据分数（= FEM 同口径的 Σ(confidence)）
        """
        try:
            score = self.fem._compute_match_score(
                ref_rendered_image, feature_mode="uniform"
            )
            return score
        except Exception as e:
            self.logger.warning(f"_compute_ref_match_score failed: {e}")
            return 0.0

    def _print_lightglue_match_details(self, fem_result: Dict[str, Any]):
        """
        打印LightGlue特征匹配的详细结果
        
        Args:
            fem_result: 特征匹配结果字典
        """
        print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}[FEM] LightGlue Feature Matching Results{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        
        # 打印总体结果
        best_score = fem_result.get('best_score', 0)
        best_index = fem_result.get('best_index', -1)
        match_success = fem_result.get('match_success', False)
        threshold = self.config.get('match_threshold', 0.1)
        
        status_color = Colors.GREEN if match_success else Colors.RED
        status_text = "✓ SUCCESS" if match_success else "✗ FAILED"
        
        print(f"  {status_color}Match Status: {status_text}{Colors.END}")
        print(f"  Best Score: {best_score:.6f}")
        print(f"  Threshold: {threshold:.6f}")
        print(f"  Best View Index: {best_index}")
        print(f"  Score/Threshold Ratio: {best_score/threshold:.2f}x" if threshold > 0 else "  Score/Threshold Ratio: N/A")
        
        # 打印每个视角的匹配得分（如果有的话）
        all_scores = fem_result.get('all_scores', None)
        if all_scores is not None and len(all_scores) > 0:
            print(f"\n  {Colors.CYAN}Per-View Match Scores:{Colors.END}")
            print(f"  {'View':<6} {'Score':<12} {'Matches':<10} {'Status'}")
            print(f"  {'-'*50}")
            
            for i, score_info in enumerate(all_scores):
                if isinstance(score_info, dict):
                    score = score_info.get('score', 0)
                    num_matches = score_info.get('num_matches', 0)
                    view_idx = score_info.get('view_index', i)
                else:
                    score = score_info if isinstance(score_info, (int, float)) else 0
                    num_matches = 0
                    view_idx = i
                
                is_best = (view_idx == best_index)
                is_pass = (score >= threshold)
                
                if is_best:
                    color = Colors.GREEN
                    status = "★ BEST"
                elif is_pass:
                    color = Colors.YELLOW
                    status = "PASS"
                else:
                    color = Colors.END
                    status = ""
                
                print(f"  {color}{view_idx:<6} {score:<12.6f} {num_matches:<10} {status}{Colors.END}")
            
            # 统计信息
            if isinstance(all_scores[0], dict):
                scores_list = [s.get('score', 0) for s in all_scores]
            else:
                scores_list = [s if isinstance(s, (int, float)) else 0 for s in all_scores]
            
            print(f"\n  {Colors.CYAN}Statistics:{Colors.END}")
            print(f"    Total Views: {len(scores_list)}")
            print(f"    Max Score: {max(scores_list):.6f}")
            print(f"    Min Score: {min(scores_list):.6f}")
            print(f"    Mean Score: {np.mean(scores_list):.6f}")
            print(f"    Std Score: {np.std(scores_list):.6f}")
            
            num_pass = sum(1 for s in scores_list if s >= threshold)
            print(f"    Views Above Threshold: {num_pass}/{len(scores_list)}")
        
        # 打印匹配点数信息（如果有的话）
        match_counts = fem_result.get('match_counts', None)
        if match_counts is not None:
            print(f"\n  {Colors.CYAN}Match Point Counts:{Colors.END}")
            total_matches = sum(match_counts) if isinstance(match_counts, list) else 0
            print(f"    Total Matches: {total_matches}")
            if isinstance(match_counts, list) and len(match_counts) > 0:
                print(f"    Max Matches per View: {max(match_counts)}")
                print(f"    Mean Matches per View: {np.mean(match_counts):.1f}")
        
        # 打印最佳匹配的详细信息
        best_matches = fem_result.get('best_matches', None)
        if best_matches is not None:
            print(f"\n  {Colors.CYAN}Best View Match Details:{Colors.END}")
            print(f"    Number of Matched Keypoints: {len(best_matches) if hasattr(best_matches, '__len__') else 'N/A'}")
        
        print(f"{Colors.CYAN}{'='*80}{Colors.END}\n")

    # ================================================================
    # ★★★ 新增：REF渲染图反投影到栅格地图找目标物体 ★★★
    # ================================================================

    def _save_ref_render_visualization(
            self,
            ref_rendered_rgb: np.ndarray,
            method_name: str,
            semantic_result: Optional[Dict] = None,
            match_result: Optional[Dict] = None,
            final_position: Optional[np.ndarray] = None) -> None:
        """
        保存REF渲染图的可视化到 REF 文件夹

        Args:
            ref_rendered_rgb: REF渲染的RGB图像
            method_name: 'semantic' 或 'matchpoints'
            semantic_result: 语义分割结果 {labels, selected_label, ...}
            match_result: 匹配点结果 {vote_map, best_component, ...}
            final_position: 最终确定的目标位置 [x, y, z]
        """
        import os
        import cv2
        from datetime import datetime

        if not self.save_visualization_results or self._results_dir is None:
            return

        # 创建 REF 目录
        vis_dir = os.path.join(str(self._results_dir), "REF")
        os.makedirs(vis_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        base_filename = f"{method_name}_{timestamp}"

        # 转换为BGR格式
        if ref_rendered_rgb.ndim == 3 and ref_rendered_rgb.shape[2] == 3:
            vis_img = cv2.cvtColor(ref_rendered_rgb, cv2.COLOR_RGB2BGR)
        else:
            vis_img = ref_rendered_rgb.copy()

        H, W = vis_img.shape[:2]

        if method_name == 'semantic' and semantic_result is not None:
            # 标注语义分割结果
            labels = semantic_result.get('labels', [])
            selected = semantic_result.get('selected_label')

            # 绘制所有检测到的物体
            for label in labels:
                bbox = label.get('bbox')
                if bbox is None:
                    continue
                category = label.get('category', 'unknown')
                is_target = (label == selected)

                color = (0, 255, 0) if is_target else (255, 0, 0)  # 绿色=选中，红色=其他
                thickness = 3 if is_target else 1
                cv2.rectangle(vis_img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), color, thickness)

                # 标注类别
                label_pos = (int(bbox[0]), max(0, int(bbox[1]) - 5))
                cv2.putText(vis_img, category, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 高亮选中的mask
            if selected is not None:
                mask = selected.get('binary_mask')
                if mask is not None:
                    overlay = vis_img.copy()
                    overlay[mask] = [0, 255, 0]  # 绿色高亮
                    vis_img = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

            # 添加文字说明
            info_lines = [
                f"Method: Semantic Segmentation",
                f"Selected: {selected['category'] if selected else 'None'}",
                f"Pixels: {int(np.sum(selected.get('binary_mask', []))) if selected else 0}",
            ]
            for i, line in enumerate(info_lines):
                # 先画黑色轮廓
                cv2.putText(vis_img, line, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                # 再画白色文字
                cv2.putText(vis_img, line, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif method_name == 'matchpoints' and match_result is not None:
            # 标注匹配点反投影结果
            kpts_ref = match_result.get('kpts_ref', [])
            num_points = match_result.get('num_points', len(kpts_ref))
            best_component = match_result.get('best_component')

            # 绘制所有匹配点（用不同颜色区分密集度）
            for (x, y) in kpts_ref:
                ix, iy = int(x), int(y)
                if 0 <= ix < W and 0 <= iy < H:
                    # 红色圆圈标记匹配点
                    cv2.circle(vis_img, (ix, iy), 4, (0, 0, 255), 2)
                    cv2.circle(vis_img, (ix, iy), 2, (0, 0, 255), -1)

            # 在图像中心绘制十字标记，表示这是fallback方法
            center_x, center_y = W // 2, H // 2
            cv2.drawMarker(vis_img, (center_x, center_y), (255, 0, 255),
                          markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # 添加文字说明
            info_lines = [
                f"Method: Match Points Fallback",
                f"Total matches: {num_points}",
                f"Best component: {best_component if best_component else 'N/A'}",
            ]
            for i, line in enumerate(info_lines):
                # 先画黑色轮廓
                cv2.putText(vis_img, line, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                # 再画白色文字
                cv2.putText(vis_img, line, (10, 25 + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 标注最终目标位置
        if final_position is not None:
            pos_str = f"Target: [{final_position[0]:.2f}, {final_position[1]:.2f}, {final_position[2]:.2f}]"
            # 先画黑色轮廓
            cv2.putText(vis_img, pos_str, (10, H - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            # 再画黄色文字
            cv2.putText(vis_img, pos_str, (10, H - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 保存可视化结果
        save_path = os.path.join(vis_dir, f"{base_filename}.png")
        cv2.imwrite(save_path, vis_img)

    def find_target_from_ref_render(
            self,
            ref_rendered_rgb: np.ndarray,
            ref_depth: np.ndarray,
            ref_pose_w2c: np.ndarray,
            target_semantic_mask: Optional[np.ndarray] = None,
            fem_match_keypoints: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        ★★★ 从REF渲染图反投影找到目标物体在栅格地图中的位置 ★★★

        两种方法:
        1. Primary: 对REF渲染图做语义分割，反投影mask到栅格地图
        2. Fallback: 用匹配点反投影到栅格地图

        Args:
            ref_rendered_rgb: REF渲染的RGB图像 (H, W, 3)
            ref_depth: REF渲染的深度图 (H, W)
            ref_pose_w2c: REF渲染时的世界到相机变换矩阵 (4, 4)
            target_semantic_mask: 目标图像的语义掩码 (H, W) - 用于fallback
            fem_match_keypoints: FEM匹配点 {'kpts_ref': ..., 'kpts_target': ..., 'scores': ...}

        Returns:
            np.ndarray: 目标物体的世界坐标位置 [x, y, z]，或 None
        """
        from ieve_minav.localizer.colors import Colors

        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.CYAN}[REF] ★ 反投影REF渲染图找目标物体 ★{Colors.END}")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")

        result_position = None
        method_used = None

        # 检查输入参数状态
        has_segmenter = hasattr(self, 'segmenter') and self.segmenter is not None
        has_target_category = getattr(self, 'target_category', None) is not None
        has_semantic_mask = target_semantic_mask is not None
        has_keypoints = fem_match_keypoints is not None

        print(f"{Colors.CYAN}[REF] 输入参数状态:{Colors.END}")
        print(f"{Colors.CYAN}  has_segmenter: {has_segmenter}{Colors.END}")
        print(f"{Colors.CYAN}  has_target_category: {has_target_category} (cat={getattr(self, 'target_category', None)}){Colors.END}")
        print(f"{Colors.CYAN}  has_semantic_mask: {has_semantic_mask}{Colors.END}")
        print(f"{Colors.CYAN}  has_keypoints: {has_keypoints}{Colors.END}")

        # ========== 方法1: 语义掩码反投影 ==========
        if has_segmenter and has_target_category:
            target_category = getattr(self, 'target_category', None)
            result = self._backproject_semantic_mask_to_map(
                ref_rendered_rgb, ref_depth, ref_pose_w2c, target_category
            )
            if result is not None and result[0] is not None:
                result_position, semantic_info = result
                method_used = 'semantic'
                print(f"{Colors.GREEN}[REF] ★ 语义掩码反投影成功 ★{Colors.END}")
                print(f"{Colors.GREEN}  目标位置: {result_position}{Colors.END}")

                # 保存可视化
                self._save_ref_render_visualization(
                    ref_rendered_rgb, method_name='semantic',
                    semantic_result=semantic_info,
                    final_position=result_position
                )
                return result_position
            else:
                    print(f"{Colors.YELLOW}[REF] 语义掩码反投影失败，尝试匹配点fallback{Colors.END}")
        else:
            if not has_segmenter:
                print(f"{Colors.YELLOW}[REF] segmenter 不可用，跳过语义掩码反投影{Colors.END}")
            elif not has_target_category:
                print(f"{Colors.YELLOW}[REF] target_category 为空，跳过语义掩码反投影{Colors.END}")

        # ========== 方法2 (Fallback): 匹配点反投影 ==========
        if target_semantic_mask is not None and fem_match_keypoints is not None:
            result = self._backproject_match_points_to_map(
                ref_rendered_rgb, ref_depth, ref_pose_w2c, target_semantic_mask, fem_match_keypoints
            )
            if result is not None:
                if isinstance(result, tuple):
                    result_position, match_info = result
                else:
                    result_position = result
                    match_info = None

                if result_position is not None:
                    method_used = 'matchpoints'
                    print(f"{Colors.GREEN}[REF] ★ 匹配点反投影成功 ★{Colors.END}")
                    print(f"{Colors.GREEN}  目标位置: {result_position}{Colors.END}")

                    # 保存可视化
                    self._save_ref_render_visualization(
                        ref_rendered_rgb, method_name='matchpoints',
                        match_result=match_info,
                        final_position=result_position
                    )
                    return result_position
            else:
                print(f"{Colors.YELLOW}[REF] 匹配点反投影也失败{Colors.END}")
        else:
            print(f"{Colors.YELLOW}[REF] 匹配点反投影跳过: semantic_mask={target_semantic_mask is not None}, keypoints={fem_match_keypoints is not None}{Colors.END}")

        print(f"{Colors.YELLOW}[REF] 所有反投影方法都失败，使用相机位置{Colors.END}")

        # 即使失败也保存可视化（用于调试）
        if ref_rendered_rgb is not None:
            self._save_ref_render_visualization(
                ref_rendered_rgb, method_name='failed',
                final_position=None  # 表示失败
            )

        return None

    def _backproject_semantic_mask_to_map(
            self,
            ref_rendered_rgb: np.ndarray,
            ref_depth: np.ndarray,
            ref_pose_w2c: np.ndarray,
            target_category: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        对REF渲染图做语义分割，反投影目标语义mask到栅格地图

        Returns:
            目标世界坐标 [x, 0, z] 或 None
        """
        from ieve_minav.localizer.colors import Colors

        print(f"{Colors.CYAN}[REF Backproject] 开始语义掩码反投影，目标类别: {target_category}{Colors.END}")

        # 1. 对REF渲染图做语义分割
        try:
            _, labels, _ = self.segmenter.segment_frame(ref_rendered_rgb, largest_only=False)
            print(f"{Colors.CYAN}[REF Backproject] 语义分割完成，检测到 {len(labels)} 个物体{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}[REF Backproject] 语义分割失败: {e}{Colors.END}")
            return None, None

        if not labels:
            print(f"{Colors.YELLOW}[REF Backproject] 未检测到任何语义物体{Colors.END}")
            return None, None

        # 2. 筛选目标语义类别
        semantic_matched = []
        for label in labels:
            category = label.get('category', '')
            if self._category_matches(category, target_category):
                semantic_matched.append(label)

        print(f"{Colors.CYAN}[REF Backproject] 找到 {len(semantic_matched)} 个目标类别匹配{Colors.END}")

        if not semantic_matched:
            print(f"{Colors.YELLOW}[REF Backproject] 未找到目标类别 '{target_category}'{Colors.END}")
            print(f"{Colors.YELLOW}[REF Backproject] 检测到的类别: {[l.get('category', '?') for l in labels]}{Colors.END}")
            return None, None

        # 3. 选择最靠中心且最大的mask
        H, W = ref_rendered_rgb.shape[:2]
        img_center = np.array([W / 2, H / 2])
        max_diag = np.sqrt(W**2 + H**2)

        best_label = None
        best_score = -1.0

        for label in semantic_matched:
            bbox = label.get('bbox')
            if not bbox:
                continue

            # 检查是否太靠边或太小
            box_width = bbox[2] - bbox[0]
            box_height = bbox[3] - bbox[1]
            min_size = min(H, W) * 0.1  # 至少占图像10%

            if box_width < min_size or box_height < min_size:
                continue

            # 检查中心是否太靠边
            box_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            dist_to_center = np.linalg.norm(box_center - img_center)
            if dist_to_center > max_diag * 0.4:  # 中心点距离图像中心超过40%对角线
                continue

            # 计算综合评分
            area = box_width * box_height
            norm_dist = dist_to_center / max_diag
            score = area * (1.0 - norm_dist)

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            print(f"{Colors.YELLOW}[REF Backproject] 所有目标语义mask都不符合条件{Colors.END}")
            return None, None

        # 4. 反投影mask到栅格地图
        mask = best_label.get('binary_mask')
        if mask is None:
            return None, None

        position = self._backproject_mask_to_grid_map(mask, ref_depth, ref_pose_w2c)

        # 构建可视化信息
        semantic_info = {
            'labels': labels,
            'selected_label': best_label,
        }
        return position, semantic_info

    def _backproject_match_points_to_map(
            self,
            ref_rendered_rgb: np.ndarray,
            ref_depth: np.ndarray,
            ref_pose_w2c: np.ndarray,
            target_semantic_mask: np.ndarray,
            fem_match_keypoints: Dict) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Fallback: 用匹配点反投影到栅格地图

        策略:
        1. 提取目标语义掩码内的所有匹配点
        2. 对应的REF渲染图匹配点
        3. 反投影到栅格地图
        4. 深度过滤
        5. 找重合最多的连通域

        Returns:
            (目标世界坐标 [x, 0, z], 可视化信息字典) 或 None
            可视化信息包含: {kpts_ref, vote_map, best_component, labeled, num_points}
        """
        from ieve_minav.localizer.colors import Colors

        print(f"{Colors.CYAN}[REF Backproject] 开始匹配点反投影{Colors.END}")

        kpts_ref = fem_match_keypoints.get('kpts_ref')
        kpts_target = fem_match_keypoints.get('kpts_target')
        scores = fem_match_keypoints.get('scores')

        if kpts_ref is None or kpts_target is None:
            print(f"{Colors.YELLOW}[REF Backproject] 匹配点数据缺失: kpts_ref={kpts_ref is not None}, kpts_target={kpts_target is not None}{Colors.END}")
            return None

        print(f"{Colors.CYAN}[REF Backproject] 总匹配点数: {len(kpts_ref)}{Colors.END}")

        # 1. 筛选在目标语义掩码内的匹配点
        H, W = target_semantic_mask.shape
        valid_indices = []

        for i, (x, y) in enumerate(kpts_target):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < H and 0 <= ix < W and target_semantic_mask[iy, ix]:
                valid_indices.append(i)

        if len(valid_indices) < 5:  # 至少需要5个点
            print(f"{Colors.YELLOW}[REF Backproject] 目标语义掩码内匹配点不足: {len(valid_indices)}{Colors.END}")
            return None

        kpts_ref_filtered = kpts_ref[valid_indices]
        # scores_filtered = scores[valid_indices]  # 暂时不用

        # 2. 反投影到栅格地图
        map = self.occupancy_mapper.map
        semantic_channel = map[:, :, 2]

        # 获取目标语义类别
        target_class = getattr(self, 'target_category', None)
        if target_class is None:
            return None

        # 从segmenter获取目标语义类别ID
        if not hasattr(self.segmenter, 'get_category_id'):
            return None
        target_sem_id = self.segmenter.get_category_id(target_class.lower())
        if target_sem_id is None or target_sem_id == 0:
            return None

        # 3. 对每个匹配点反投影并投票
        vote_map = np.zeros_like(semantic_channel, dtype=np.int32)
        ref_depths = []

        for i, (x, y) in enumerate(kpts_ref_filtered):
            ix, iy = int(round(x)), int(round(y))

            # 边界检查
            if not (0 <= iy < ref_depth.shape[0] and 0 <= ix < ref_depth.shape[1]):
                continue

            depth = ref_depth[iy, ix]
            if depth <= 0.1 or depth > 10.0 or not np.isfinite(depth):
                continue

            # 计算世界坐标
            world_pos = self._unproject_pixel_to_world(ix, iy, depth, ref_pose_w2c)
            if world_pos is None:
                continue

            ref_depths.append(depth)

            # 转换到栅格地图坐标
            map_x, map_y = self.occupancy_mapper.world_to_map(world_pos[0], world_pos[2])
            if map_x is None or map_y is None:
                continue

            if 0 <= map_x < semantic_channel.shape[1] and 0 <= map_y < semantic_channel.shape[0]:
                vote_map[map_y, map_x] += 1

        if len(ref_depths) < 5:
            return None

        # 4. 找重合最多的连通域
        target_sem_mask = (semantic_channel == target_sem_id)

        # 只在目标语义区域内投票
        vote_map = vote_map * target_sem_mask.astype(np.int32)

        if np.sum(vote_map) == 0:
            print(f"{Colors.YELLOW}[REF Backproject] 投票点与目标语义无重合{Colors.END}")
            return None

        # 连通域分析
        try:
            from skimage.measure import label as sk_label
            labeled, num_components = sk_label(vote_map > 0, return_num=True, connectivity=2)
        except ImportError:
            # 无skimage，直接取最大值点
            max_idx = np.argmax(vote_map)
            max_pos = np.unravel_index(max_idx, vote_map.shape)
            map_y, map_x = max_pos
            world_x, world_z = self.occupancy_mapper.map_to_world(map_x, map_y)
            object_pos = np.array([world_x, 0.0, world_z])

            # ★★★ 找到最近的可达点 ★★★
            reachable_pos = self._find_nearest_reachable_point(object_pos)
            final_pos = reachable_pos if reachable_pos is not None else object_pos

            # 返回位置和可视化信息
            match_info = {
                'kpts_ref': kpts_ref_filtered,
                'vote_map': vote_map,
                'best_component': None,  # 无连通域信息
                'labeled': None,
                'num_points': len(kpts_ref_filtered),
            }
            return (final_pos, match_info)

        if num_components == 0:
            return None

        # 找投票数最多的连通域
        max_votes = 0
        best_label = 0
        for label_id in range(1, num_components + 1):
            comp_mask = (labeled == label_id)
            votes_in_comp = np.sum(vote_map[comp_mask])
            if votes_in_comp > max_votes:
                max_votes = votes_in_comp
                best_label = label_id

        # 计算最佳连通域的中心
        comp_mask = (labeled == best_label)
        positions = np.argwhere(comp_mask)
        center = np.mean(positions, axis=0)
        map_y, map_x = int(center[0]), int(center[1])

        world_x, world_z = self.occupancy_mapper.map_to_world(map_x, map_y)
        object_pos = np.array([world_x, 0.0, world_z])

        # ★★★ 找到最近的可达点 ★★★
        reachable_pos = self._find_nearest_reachable_point(object_pos)
        final_pos = reachable_pos if reachable_pos is not None else object_pos

        # 返回位置和可视化信息
        match_info = {
            'kpts_ref': kpts_ref_filtered,
            'vote_map': vote_map,
            'best_component': best_label,
            'labeled': labeled,
            'num_points': len(kpts_ref_filtered),
        }
        return (final_pos, match_info)

    def _get_target_semantic_id(self, target_category: Optional[str] = None) -> int:
        explicit_semantic_class = getattr(self, 'target_semantic_class', None)
        if explicit_semantic_class is not None:
            try:
                return int(explicit_semantic_class)
            except (TypeError, ValueError):
                pass

        category = target_category or getattr(self, 'target_category', '')
        category = str(category or '').lower()
        segmenter = getattr(self, 'segmenter', None)
        if not category or segmenter is None:
            return 0

        if hasattr(segmenter, 'get_category_id'):
            try:
                sem_id = segmenter.get_category_id(category)
                return int(sem_id) if sem_id is not None else 0
            except (TypeError, ValueError):
                return 0

        categories = getattr(segmenter, 'indoor_categories', {})
        if isinstance(categories, dict) and category in categories:
            return list(categories.keys()).index(category) + 1
        return 0

    def _backproject_mask_to_grid_map(
            self,
            mask: np.ndarray,
            ref_depth: np.ndarray,
            ref_pose_w2c: np.ndarray) -> Optional[np.ndarray]:
        """
        将语义掩码反投影到栅格地图，找重叠最多的连通域

        Returns:
            目标世界坐标 [x, 0, z] 或 None
        """
        from ieve_minav.localizer.colors import Colors
        import scipy.ndimage as ndi

        map = self.occupancy_mapper.map
        semantic_channel = map[:, :, 2]

        # 获取目标语义类别
        target_category = getattr(self, 'target_category', None)
        if target_category is None:
            return None

        target_sem_id = self._get_target_semantic_id(target_category)
        if target_sem_id <= 0:
            return None

        H, W = mask.shape
        vote_map = np.zeros_like(semantic_channel, dtype=np.int32)

        # 降采样以提高速度
        step = 4
        for y in range(0, H, step):
            for x in range(0, W, step):
                if not mask[y, x]:
                    continue

                depth = ref_depth[y, x]
                if depth <= 0.1 or depth > 10.0 or not np.isfinite(depth):
                    continue

                # 反投影到世界坐标
                world_pos = self._unproject_pixel_to_world(x, y, depth, ref_pose_w2c)
                if world_pos is None:
                    continue

                # 转换到栅格地图坐标
                map_x, map_y = self.occupancy_mapper.world_to_map(world_pos[0], world_pos[2])
                if map_x is None or map_y is None:
                    continue

                if 0 <= map_x < semantic_channel.shape[1] and 0 <= map_y < semantic_channel.shape[0]:
                    vote_map[map_y, map_x] += 1

        if np.sum(vote_map) == 0:
            return None

        # 找与目标语义重叠的区域
        target_sem_mask = (semantic_channel == target_sem_id)
        overlap_map = vote_map * target_sem_mask.astype(np.int32)

        if np.sum(overlap_map) == 0:
            # 没有直接重叠，尝试膨胀目标语义区域
            dilated_sem = ndi.binary_dilation(target_sem_mask, iterations=3)
            overlap_map = vote_map * dilated_sem.astype(np.int32)

            if np.sum(overlap_map) == 0:
                return None

        # 连通域分析找最大重叠区域
        try:
            from skimage.measure import label as sk_label
            labeled, num_components = sk_label(overlap_map > 0, return_num=True, connectivity=2)
        except ImportError:
            max_idx = np.argmax(overlap_map)
            max_pos = np.unravel_index(max_idx, overlap_map.shape)
            map_y, map_x = max_pos
            world_x, world_z = self.occupancy_mapper.map_to_world(map_x, map_y)
            object_pos = np.array([world_x, 0.0, world_z])

            # ★★★ 找到最近的可达点 ★★★
            reachable_pos = self._find_nearest_reachable_point(object_pos)
            if reachable_pos is not None:
                return reachable_pos
            return object_pos

        if num_components == 0:
            return None

        # 找投票数最多的连通域
        max_votes = 0
        best_label = 0
        for label_id in range(1, num_components + 1):
            comp_mask = (labeled == label_id)
            votes_in_comp = np.sum(overlap_map[comp_mask])
            if votes_in_comp > max_votes:
                max_votes = votes_in_comp
                best_label = label_id

        # 计算最佳连通域的中心
        comp_mask = (labeled == best_label)
        positions = np.argwhere(comp_mask)
        center = np.mean(positions, axis=0)
        map_y, map_x = int(center[0]), int(center[1])

        world_x, world_z = self.occupancy_mapper.map_to_world(map_x, map_y)
        object_pos = np.array([world_x, 0.0, world_z])

        # 找到最近的可达点
        reachable_pos = self._find_nearest_reachable_point(object_pos)
        if reachable_pos is not None:
            return reachable_pos
        return object_pos

    def _unproject_pixel_to_world(
            self,
            pixel_x: int,
            pixel_y: int,
            depth: float,
            w2c_pose: np.ndarray) -> Optional[np.ndarray]:
        """
        将像素坐标+深度反投影到世界坐标

        Args:
            pixel_x: 像素x坐标
            pixel_y: 像素y坐标
            depth: 深度值 (米)
            w2c_pose: 世界到相机变换矩阵 (4, 4)

        Returns:
            世界坐标 [x, y, z] 或 None
        """
        try:
            # 获取相机内参
            cp = self.slam_mapper.camera_params
            fx = cp.get('fx', 600)
            fy = cp.get('fy', 600)
            cx = cp.get('cx', 320)
            cy = cp.get('cy', 240)

            # 像素坐标转相机坐标
            z_c = depth
            x_c = (pixel_x - cx) * z_c / fx
            y_c = (pixel_y - cy) * z_c / fy

            # 相机坐标转世界坐标
            c2w = np.linalg.inv(w2c_pose)
            point_cam = np.array([x_c, y_c, z_c])
            point_world = c2w[:3, :3] @ point_cam + c2w[:3, 3]

            return point_world
        except Exception as e:
            return None

    def _find_nearest_reachable_point(
            self,
            object_position: np.ndarray,
            search_radius_m: float = 2.0) -> Optional[np.ndarray]:
        """
        找到物体连通域周围最近的可达点（用于导航）

        Args:
            object_position: 物体世界坐标 [x, y, z]
            search_radius_m: 搜索半径（米）

        Returns:
            最近可达点的世界坐标 [x, y, z]，或 None
        """
        import scipy.ndimage as ndi
        from ieve_minav.localizer.colors import Colors

        # 1. 将物体位置转换到栅格地图坐标
        map_x, map_y = self.occupancy_mapper.world_to_map(
            object_position[0], object_position[2])
        if map_x is None or map_y is None:
            return None

        map_size = self.occupancy_mapper.map_size_px
        map_resolution = self.occupancy_mapper.map_resolution / 100.0  # m/px
        search_radius_px = int(search_radius_m / map_resolution)

        # 2. 获取可通行地图
        traversible = self.occupancy_mapper.get_traversible_map()

        # 3. 检查物体位置本身是否可达
        if (0 <= map_y < traversible.shape[0] and
                0 <= map_x < traversible.shape[1] and
                traversible[map_y, map_x] > 0.5):
            return object_position.copy()

        # 4. 找到物体所在的语义连通域（用于膨胀搜索）
        semantic_channel = self.occupancy_mapper.map[:, :, 2]
        target_sem_id = self._get_target_semantic_id()

        if target_sem_id > 0:
            # 提取目标语义连通域
            object_mask = (semantic_channel == target_sem_id)

            # 膨胀连通域形成搜索带
            dilated = ndi.binary_dilation(object_mask, iterations=search_radius_px)
            search_band = dilated & (~object_mask)
        else:
            # 如果没有语义信息，以物体中心为圆心搜索
            h, w = traversible.shape
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - map_x)**2 + (y - map_y)**2
            search_band = dist_sq <= search_radius_px**2

        # 5. 在搜索带内找可通行且离物体最近的点
        candidate_mask = search_band & (traversible > 0.5)
        candidate_ys, candidate_xs = np.where(candidate_mask)

        if len(candidate_ys) == 0:
            print(f"{Colors.YELLOW}[Reachable] 物体周围{search_radius_m}m内无可达点{Colors.END}")
            return None

        # 找离物体中心最近的候选点
        dists_sq = (candidate_xs - map_x)**2 + (candidate_ys - map_y)**2
        best_idx = int(np.argmin(dists_sq))
        best_y, best_x = candidate_ys[best_idx], candidate_xs[best_idx]

        # 6. 转换回世界坐标
        world_x, world_z = self.occupancy_mapper.map_to_world(best_x, best_y)
        return np.array([world_x, object_position[1], world_z])

    def _category_matches(self, detected: str, target: str) -> bool:
        """检查检测到的类别是否与目标匹配"""
        detected_lower = detected.lower().strip()
        target_lower = target.lower().strip()

        if detected_lower == target_lower:
            return True
        if target_lower in detected_lower or detected_lower in target_lower:
            return True

        return False
