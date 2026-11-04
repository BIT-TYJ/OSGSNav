import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from ieve_minav.localizer.colors import Colors

class SemanticObjectIdentifier:
    """语义物体识别模块 - 在探索时持续观测目标物体"""
    
    def __init__(self, 
                 occupancy_mapper,
                 observation_threshold: int = 6,
                 cooldown_period: float = 0.0,
                 position_tolerance: float = 0.5,
                 target_semantic_class: Optional[int] = None,
                 min_observation_distance: float = 0.5,
                 min_angle_threshold: float = 45.0,
                 camera_params: Optional[Dict[str, Any]] = None,  # ★★★ 新增 ★★★
                 fov_margin: float = 0.1):  # ★★★ 新增：视野边界裕度 ★★★
        """
        初始化SOIM模块
        
        Args:
            occupancy_mapper: OccupancyMapper实例
            observation_threshold: 触发NVSM的最小观测次数
            cooldown_period: 放弃物体的冷却期（秒）
            position_tolerance: Z轴位置容忍度（米）
            target_semantic_class: 目标语义类别
            min_observation_distance: 最远两个观测点之间的最小距离（米）
            min_angle_threshold: 最远两个观测点与物体中心连线的最小夹角（度）
            camera_params: 相机内参字典，包含 fx, fy, cx, cy, width, height  # ★★★ 新增 ★★★
            fov_margin: 视野检查的边界裕度（相对于图像宽度的比例，0-1）  # ★★★ 新增 ★★★
        """
        self.logger = logging.getLogger(__name__)
        self.occupancy_mapper = occupancy_mapper
        self.observation_threshold = observation_threshold
        self.cooldown_period = cooldown_period
        self.z_tolerance = position_tolerance
        self.min_observation_distance = min_observation_distance
        self.min_angle_threshold = min_angle_threshold
        
        # ★★★ 新增：相机内参和视野参数 ★★★
        self.camera_params = camera_params or {}
        self.fov_margin = max(0.0, min(1.0, fov_margin))  # 限制在[0, 1]
        
        # 物体观测记录
        self.observed_objects: Dict[str, Dict[str, Any]] = {}
        
        self.current_keyframe_objects = set()
        self.object_counter = 0
        
        try:
            from ieve_minav.localizer.colors import Colors
            self.Colors = Colors
        except ImportError:
            class EmptyColors:
                CYAN = GREEN = YELLOW = RED = END = ""
            self.Colors = EmptyColors()
        
        self.target_semantic_class = target_semantic_class
        
        self.processed_objects: Dict[str, float] = {}
        self.trigger_cooldown: float = 0.0
        
        # ★★★ GaussMI 就绪标志：只有 GaussMI 到达指定次数后才允许触发 NVSM ★★★
        self.gaussmi_ready = False
        self.candidate_anchor_position: Optional[np.ndarray] = None
        self.candidate_anchor_radius_m: float = 3.0
        
        self.logger.info(
            f"SOIM initialized with threshold={observation_threshold}, "
            f"z_tolerance={position_tolerance}m, "
            f"min_obs_distance={min_observation_distance}m, "
            f"min_angle={min_angle_threshold}°, "
            f"target_class={target_semantic_class}, "
            f"fov_margin={fov_margin}"  # ★★★ 新增 ★★★
        )
    
    def set_camera_params(self, camera_params: Dict[str, Any]):
        """
        ★★★ 新增：动态设置相机内参 ★★★
        
        Args:
            camera_params: 相机参数，包含 fx, fy, cx, cy, width, height
        """
        self.camera_params = camera_params
        self.logger.info(f"Camera params updated: {camera_params}")
    
    def process_keyframe(self, keyframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理新的关键帧，更新物体观测
        
        Args:
            keyframe_data: 包含 rgb, depth, semantic, pose, timestamp, is_3dgs_keyframe
            
        Returns:
            dict: 包含 trigger_nvsm, object_info, observed_objects, skipped
        """
        is_3dgs_keyframe = keyframe_data.get('is_3dgs_keyframe', True)  # 默认True保持兼容性
        
        if not is_3dgs_keyframe:
            # 非3DGS关键帧，静默返回（不更新观测计数）
            return {
                'trigger_nvsm': False,
                'object_info': None,
                'observed_objects': self.observed_objects,
                'skipped': True  # 标记为跳过
            }
        
        # 清空当前关键帧的物体集合
        self.current_keyframe_objects.clear()
        
        timestamp = keyframe_data.get('timestamp', time.time())
        semantic_map = keyframe_data.get('semantic')
        camera_position = self._get_camera_position(keyframe_data)
        camera_pose = keyframe_data.get('pose')  # ★★★ 新增：获取相机位姿 ★★★
        
        if semantic_map is None:
            return {
                'trigger_nvsm': False,
                'observed_objects': self.observed_objects
            }
        
        # 检测语义物体
        detected_objects = self._detect_semantic_objects(semantic_map, keyframe_data)
        
        # ★★★ 修改：传递当前帧语义分割到观测更新 ★★★
        for obj_info in detected_objects:
            self._update_observation(
                obj_info, 
                timestamp, 
                camera_position, 
                camera_pose,
                current_semantic=semantic_map  # ★★★ 新增：传递当前帧语义分割 ★★★
            )
        
        # ★★★ 修改：检查所有活跃物体的触发条件 ★★★
        ready_objects = []
        current_time = time.time()
        
        for obj_id, obj_data in self.observed_objects.items():
            if obj_data['status'] != 'active':
                continue

            # ★★★ 修改：当 GaussMI 就绪时，允许对已观测物体触发（即使当前帧不可见）★★★
            if obj_id not in self.current_keyframe_objects:
                if not self.gaussmi_ready:
                    continue
                # GaussMI 就绪时，允许对已观测物体触发（即使当前帧不在视野中）

            anchor_distance = self._distance_to_candidate_anchor(
                obj_data.get('anchor_position_3d', obj_data.get('position_3d'))
            )
            if (self.gaussmi_ready
                    and self.candidate_anchor_position is not None
                    and anchor_distance is not None
                    and anchor_distance > self.candidate_anchor_radius_m):
                self.logger.debug(
                    f"Object {obj_id} outside candidate anchor radius: "
                    f"{anchor_distance:.2f}m > {self.candidate_anchor_radius_m:.2f}m"
                )
                continue
            
            # ★★★ 新增：检查是否在触发冷却期内 ★★★
            last_trigger_time = self.processed_objects.get(obj_id, 0)
            if current_time - last_trigger_time < self.trigger_cooldown:
                self.logger.debug(f"Object {obj_id} in trigger cooldown, skipping")
                continue
            
            trigger_result = self._check_multi_view_trigger(obj_id, obj_data)
            
            if trigger_result['should_trigger']:
                ready_objects.append({
                    'id': obj_id,
                    'position_2d': obj_data['position_2d'],
                    'position_3d': obj_data['position_3d'],
                    'semantic_class': obj_data['semantic_class'],
                    'observations': obj_data['observations'],
                    'observation_positions': obj_data.get('observation_positions', []),
                    'trigger_info': trigger_result,
                    'anchor_distance': anchor_distance if anchor_distance is not None else float('inf')
                })
        
        # 优先选离当前 buffer 候选最近的物体，其次再看观测次数。
        if self.candidate_anchor_position is not None:
            ready_objects.sort(key=lambda x: (x['anchor_distance'], -x['observations']))
        else:
            ready_objects.sort(key=lambda x: x['observations'], reverse=True)
        
        # 选择优先级最高的物体触发
        trigger_object = None
        if ready_objects:
            trigger_object = ready_objects[0]
            self.processed_objects[trigger_object['id']] = current_time
            
            ti = trigger_object.get('trigger_info', {})

            max_dist = ti.get('max_distance', 0.0)
            angle = ti.get('angle', 0.0)

            print(f"\n{self.Colors.GREEN}{'='*70}{self.Colors.END}")
            print(f"{self.Colors.GREEN}[SOIM] ✓ GaussMI-READY TRIGGER ✓{self.Colors.END}")
            print(f"{self.Colors.GREEN}  Object ID: {trigger_object['id']}{self.Colors.END}")
            print(f"{self.Colors.GREEN}  Observations: {trigger_object['observations']}{self.Colors.END}")
            print(f"{self.Colors.GREEN}  Max Distance: {max_dist:.2f}m (threshold: {self.min_observation_distance}m){self.Colors.END}")
            print(f"{self.Colors.GREEN}  Angle: {angle:.1f}° (threshold: {self.min_angle_threshold}°){self.Colors.END}")
            print(f"{self.Colors.GREEN}  Observation Positions: {len(trigger_object['observation_positions'])}{self.Colors.END}")
            if self.candidate_anchor_position is not None:
                print(f"{self.Colors.GREEN}  Anchor Distance: {trigger_object['anchor_distance']:.2f}m "
                      f"(limit: {self.candidate_anchor_radius_m:.2f}m){self.Colors.END}")
            print(f"{self.Colors.GREEN}{'='*70}{self.Colors.END}\n")
            
            self.logger.info(
                f"Object {trigger_object['id']} triggering NVSM: "
                f"obs={trigger_object['observations']}, "
                f"dist={max_dist:.2f}m, "
                f"angle={angle:.1f}°"
            )
        
        return {
            'trigger_nvsm': trigger_object is not None,
            'object_info': trigger_object,
            'observed_objects': self.observed_objects,
            'ready_objects_count': len(ready_objects)  # ★★★ 新增：返回就绪物体数量 ★★★
        }
    
    def _get_camera_position(self, keyframe_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """
        从关键帧数据中获取相机位置（世界坐标 X, Z）
        
        Args:
            keyframe_data: 关键帧数据
            
        Returns:
            (world_x, world_z) 或 None
        """
        pose = keyframe_data.get('pose')
        if pose is None:
            return None
        
        try:
            # pose 是 4x4 w2c 矩阵，需要转换为 c2w 获取相机位置
            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                c2w = np.linalg.inv(pose)
                camera_pos = c2w[:3, 3]
                return (float(camera_pos[0]), float(camera_pos[2]))
            else:
                self.logger.warning(f"Invalid pose format: {type(pose)}, shape: {getattr(pose, 'shape', 'N/A')}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to extract camera position: {e}")
            return None
    
    def set_gaussmi_ready(self):
        """
        ★★★ 新增：由外部（GaussMI 到达指定次数后）调用，允许 SOIM 触发 NVSM ★★★
        """
        self.gaussmi_ready = True
        print(f"{self.Colors.GREEN}[SOIM] ★ GaussMI ready signal received → "
              f"NVSM trigger UNLOCKED{self.Colors.END}")

    def set_candidate_anchor(self, anchor_position: np.ndarray, radius_m: float = 3.0):
        """设置当前 buffer 候选的锚点，GaussMI-ready 后只允许触发该锚点附近的物体。"""
        if anchor_position is None:
            self.clear_candidate_anchor()
            return

        self.candidate_anchor_position = np.array(anchor_position, dtype=np.float32).copy()
        self.candidate_anchor_radius_m = float(radius_m)
        self.logger.info(
            f"SOIM candidate anchor set to {self.candidate_anchor_position.tolist()} "
            f"(radius={self.candidate_anchor_radius_m:.2f}m)"
        )

    def clear_candidate_anchor(self):
        """清除当前 buffer 候选锚点。"""
        self.candidate_anchor_position = None

    def _distance_to_candidate_anchor(self, position_3d: Optional[np.ndarray]) -> Optional[float]:
        """返回物体到当前 buffer 候选锚点的 XZ 平面距离。"""
        if self.candidate_anchor_position is None or position_3d is None:
            return None

        pos = np.asarray(position_3d, dtype=np.float32)
        anchor = np.asarray(self.candidate_anchor_position, dtype=np.float32)
        return float(np.linalg.norm(pos[[0, 2]] - anchor[[0, 2]]))
    
    def _check_multi_view_trigger(self, obj_id: str, obj_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ★★★ 改进：由 GaussMI 就绪信号驱动的触发检查 ★★★
        
        - gaussmi_ready == False → 永远不触发（等待 GaussMI 完成多视角采集）
        - gaussmi_ready == True  → 只要有 >=1 次观测且有 3D 位置即可触发
        
        Args:
            obj_id: 物体ID
            obj_data: 物体数据
            
        Returns:
            {
                'should_trigger': bool,
                'max_distance': float,
                'angle': float,
                'farthest_positions': [(x1, z1), (x2, z2)]
            }
        """
        result = {
            'should_trigger': False,
            'max_distance': 0.0,
            'angle': 0.0,
            'farthest_positions': []
        }
        
        # ★★★ 核心门控：GaussMI 未就绪时，永远不触发 ★★★
        if not self.gaussmi_ready:
            return result
        
        # GaussMI 已就绪 → 只需最基本条件：有观测且有 3D 位置
        if obj_data['observations'] < 1:
            return result
        
        if obj_data.get('position_3d') is None and obj_data.get('anchor_position_3d') is None:
            return result
        
        # 计算观测统计信息（用于日志，不作为触发条件）
        observation_positions = obj_data.get('observation_positions', [])
        if len(observation_positions) >= 2:
            positions = [np.array(pos) for pos in observation_positions]
            max_distance = 0.0
            farthest_pair = None
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist > max_distance:
                        max_distance = dist
                        farthest_pair = (positions[i], positions[j])
            result['max_distance'] = max_distance
            if farthest_pair is not None:
                result['farthest_positions'] = [tuple(farthest_pair[0]), tuple(farthest_pair[1])]
                
                anchor_center = obj_data.get('anchor_position_3d', obj_data.get('position_3d'))
                if anchor_center is not None:
                    object_xz = np.array([anchor_center[0], anchor_center[2]])
                    vec1 = farthest_pair[0] - object_xz
                    vec2 = farthest_pair[1] - object_xz
                    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                    if norm1 > 1e-6 and norm2 > 1e-6:
                        cos_a = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
                        result['angle'] = np.rad2deg(np.arccos(cos_a))
        
        # ★ GaussMI 就绪 → 直接触发
        result['should_trigger'] = True
        
        self.logger.info(
            f"Object {obj_id} TRIGGER (GaussMI-ready): "
            f"obs={obj_data['observations']}, "
            f"positions={len(observation_positions)}, "
            f"max_dist={result['max_distance']:.2f}m, "
            f"angle={result['angle']:.1f}°"
        )
        
        return result
    
    def _detect_semantic_objects(self, semantic_map: np.ndarray, 
                                keyframe_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        从语义地图中检测目标物体
        
        Returns:
            检测到的物体列表，每个包含:
            - semantic_class
            - position_2d (中心)
            - position_3d (中心)
            - boundary_2d (边界范围)
        """
        detected = []
        
        # 获取目标语义类别
        target_classes = self._get_target_semantic_classes()
        
        # ★★★ 新增：调试输出 ★★★
        # print(f"{Colors.CYAN}[SOIM Debug] Target semantic classes: {target_classes}{Colors.END}")
        
        if not target_classes:
            print(f"{Colors.YELLOW}[SOIM Warning] No target classes configured!{Colors.END}")
            return []
        
        # 获取真实的2D栅格地图（语义通道）
        if hasattr(self.occupancy_mapper, 'map'):
            grid_map = self.occupancy_mapper.map
            semantic_channel = grid_map[:, :, 2]
            
            # ★★★ 新增：打印语义通道统计 ★★★
            unique_classes = np.unique(semantic_channel)
            # print(f"{Colors.CYAN}[SOIM Debug] Semantic channel unique values: {unique_classes}{Colors.END}")
        else:
            semantic_channel = semantic_map
        
        for semantic_class in target_classes:
            # 在语义通道中查找该类别
            mask = (semantic_channel == semantic_class)
            
            num_pixels = np.sum(mask)
            # print(f"{Colors.CYAN}[SOIM Debug] Class {semantic_class}: found {num_pixels} pixels{Colors.END}")
            
            if not np.any(mask):
                continue
            
            # ★★★ 实例级检测：使用连通域分析拆分同类别的不同实例 ★★★
            try:
                from skimage.morphology import disk, binary_closing
                from skimage.measure import label as sk_label
                
                # 形态学闭运算弥合稀疏语义地图的间隙
                closed = binary_closing(
                    mask.astype(bool), footprint=disk(5)
                ).astype(np.uint8)
                
                labeled, num_components = sk_label(closed, return_num=True, connectivity=2)
                
                MIN_INSTANCE_PIXELS = 20  # 过滤 2D 语义图中的小碎片噪声
                
                # print(f"{Colors.CYAN}[SOIM Debug] Class {semantic_class}: "
                #       f"{num_components} instances after connected-component analysis{Colors.END}")
                
                for comp_id in range(1, num_components + 1):
                    comp_mask = (labeled == comp_id)
                    # 取连通域范围内的原始语义像素（排除闭运算填充的像素）
                    instance_mask = mask & comp_mask
                    
                    instance_pixels = int(np.sum(instance_mask))
                    if instance_pixels < MIN_INSTANCE_PIXELS:
                        continue
                    
                    positions = np.argwhere(instance_mask)
                    rows = positions[:, 0]
                    cols = positions[:, 1]
                    boundary_2d = {
                        'y_min': int(rows.min()),
                        'y_max': int(rows.max()),
                        'x_min': int(cols.min()),
                        'x_max': int(cols.max()),
                    }
                    
                    center_grid = np.mean(positions, axis=0)  # (row, col)
                    grid_y = int(center_grid[0])
                    grid_x = int(center_grid[1])
                    
                    map_size = self.occupancy_mapper.map_size_px
                    if not (0 <= grid_x < map_size and 0 <= grid_y < map_size):
                        self.logger.warning(f"Instance center ({grid_x}, {grid_y}) out of bounds")
                        continue
                    
                    world_x, world_z = self._grid_to_world(grid_x, grid_y)
                    
                    detected.append({
                        'semantic_class': semantic_class,
                        'position_2d': (grid_x, grid_y),
                        'position_3d': np.array([world_x, 0.0, world_z]),
                        'boundary_2d': boundary_2d,
                    })
                    
                    self.logger.debug(
                        f"Detected instance #{comp_id} of class {semantic_class}: "
                        f"grid ({grid_x}, {grid_y}), pixels={instance_pixels}, "
                        f"boundary: x=[{boundary_2d['x_min']}, {boundary_2d['x_max']}], "
                        f"y=[{boundary_2d['y_min']}, {boundary_2d['y_max']}]"
                    )
                    # # print(f"{Colors.CYAN}[SOIM Debug]   Instance #{comp_id}: "
                    #       f"center=({grid_x},{grid_y}), pixels={instance_pixels}, "
                    #       f"boundary=[{boundary_2d['x_min']}:{boundary_2d['x_max']}, "
                    #       f"{boundary_2d['y_min']}:{boundary_2d['y_max']}]{Colors.END}")
                    
            except ImportError:
                # 回退：类别级检测（无skimage时）
                self.logger.warning("skimage not available, falling back to category-level detection")
                positions = np.argwhere(mask)
                rows = positions[:, 0]
                cols = positions[:, 1]
                boundary_2d = {
                    'y_min': int(rows.min()),
                    'y_max': int(rows.max()),
                    'x_min': int(cols.min()),
                    'x_max': int(cols.max()),
                }
                center_grid = np.mean(positions, axis=0)
                grid_y = int(center_grid[0])
                grid_x = int(center_grid[1])
                
                map_size = self.occupancy_mapper.map_size_px
                if 0 <= grid_x < map_size and 0 <= grid_y < map_size:
                    world_x, world_z = self._grid_to_world(grid_x, grid_y)
                    detected.append({
                        'semantic_class': semantic_class,
                        'position_2d': (grid_x, grid_y),
                        'position_3d': np.array([world_x, 0.0, world_z]),
                        'boundary_2d': boundary_2d,
                    })
        
        return detected
    
    def _update_observation(self, obj_info: Dict[str, Any], timestamp: float,
                            camera_position: Optional[Tuple[float, float]] = None,
                            camera_pose: Optional[np.ndarray] = None,
                            current_semantic: Optional[np.ndarray] = None) -> Optional[str]:
        """
        更新物体观测记录，并打印详细调试信息
        """
        position_2d = obj_info['position_2d']
        position_3d = obj_info['position_3d']
        boundary_2d = obj_info['boundary_2d']
        semantic_class = obj_info['semantic_class']

        if self.gaussmi_ready and self.candidate_anchor_position is not None:
            anchor_distance = self._distance_to_candidate_anchor(position_3d)
            if (anchor_distance is not None
                    and anchor_distance > self.candidate_anchor_radius_m):
                self.logger.debug(
                    f"Skip object at {position_2d}: outside candidate anchor radius "
                    f"({anchor_distance:.2f}m > {self.candidate_anchor_radius_m:.2f}m)"
                )
                return None

        # GaussMI-ready 后改为直接相信缓冲期前面建立的地图语义，不再要求当前帧再次看见目标。
        in_view = True
        visibility_gate_bypassed = bool(self.gaussmi_ready)
        if current_semantic is not None and not visibility_gate_bypassed:
            in_view = self._is_object_in_view(
                position_3d, camera_pose, current_semantic=current_semantic,
                target_semantic_class=semantic_class
            )
            # print(f"{self.Colors.YELLOW}[SOIM Debug] Object at {position_2d} "
            #     f"world Z={position_3d[2]:.2f} in view: {'是' if in_view else '否'}{self.Colors.END}")
            if not in_view:
                self.logger.debug(f"Object at {position_2d} not in current view, skipping observation")
                return None
        elif visibility_gate_bypassed:
            self.logger.debug(
                f"GaussMI-ready: bypass visibility gate for object at {position_2d}"
            )

        # ★★★ 使用边界匹配逻辑查找已知物体 ★★★
        obj_id = self._find_existing_object_by_boundary(position_2d, position_3d, boundary_2d)

        if obj_id is None:
            # 新物体
            self.object_counter += 1
            obj_id = f"{self.object_counter}({int(timestamp)})"
            self.observed_objects[obj_id] = {
                'position_2d': position_2d,
                'position_3d': position_3d,
                'anchor_position_3d': position_3d.copy(),
                'boundary_2d': boundary_2d,
                'observations': 0,
                'observation_positions': [],
                'last_seen': timestamp,
                'status': 'active',
                'semantic_class': semantic_class,
                'first_seen': timestamp
            }
            # print(f"{self.Colors.GREEN}[SOIM Debug] New object {obj_id} created:"
            #     f" position_2d={position_2d}, anchor Z={position_3d[2]:.2f}, boundary={boundary_2d}{self.Colors.END}")
        else:
            # 更新已知物体位置和边界，但保持 anchor 不变
            obj_data = self.observed_objects[obj_id]
            # print(f"{self.Colors.CYAN}[SOIM Debug] Updating existing object {obj_id}:"
            #     f" old_pos_3d={obj_data['position_3d']}, new_pos_3d={position_3d}{self.Colors.END}")
            obj_data['position_2d'] = position_2d
            obj_data['position_3d'] = position_3d
            obj_data['boundary_2d'] = boundary_2d

        # ★★★ 核心：记录观测、视角和关键帧信息 ★★★
        if obj_id not in self.current_keyframe_objects:
            obj_data = self.observed_objects[obj_id]

            if obj_data['status'] == 'abandoned':
                abandon_time = obj_data.get('abandon_time', 0)
                if timestamp - abandon_time >= self.cooldown_period:
                    obj_data['status'] = 'active'
                    print(f"{self.Colors.GREEN}[SOIM] Object {obj_id} reactivated{self.Colors.END}")
                else:
                    self.current_keyframe_objects.add(obj_id)
                    return obj_id

            if obj_data['status'] == 'active':
                if camera_position is not None:
                    self._add_observation_position(obj_id, camera_position, camera_pose)

                obj_data['observations'] += 1
                obs_count = obj_data['observations']
                obs_positions = obj_data.get('observation_positions', [])

                status_color = self.Colors.CYAN
                if obs_count >= self.observation_threshold:
                    status_color = self.Colors.GREEN
                elif obs_count >= self.observation_threshold - 1:
                    status_color = self.Colors.YELLOW

                # print(f"{status_color}[SOIM Debug] Object {obj_id} observed in keyframe {int(timestamp)}:"
                #     f" obs_count={obs_count}, obs_positions={len(obs_positions)}, semantic_class={semantic_class}{self.Colors.END}")
                if camera_position is not None:
                    print(f"  - Camera position: ({camera_position[0]:.2f}, {camera_position[1]:.2f})")
                if visibility_gate_bypassed:
                    print("  - In view gate: bypassed (GaussMI ready)")
                else:
                    print(f"  - In view: {'是' if in_view else '否'}")
                print(f"  - Anchor position Z: {obj_data['anchor_position_3d'][2]:.2f}, current Z: {position_3d[2]:.2f}")

                if len(obs_positions) >= 2:
                    trigger_result = self._check_multi_view_trigger(obj_id, obj_data)
                    print(f"  - Multi-view trigger check: max_distance={trigger_result['max_distance']:.2f}, "
                        f"angle={trigger_result['angle']:.1f}°, should_trigger={trigger_result['should_trigger']}")

            obj_data['last_seen'] = timestamp
            self.current_keyframe_objects.add(obj_id)

        return obj_id

    
    def _add_observation_position(self, obj_id: str, camera_position: Tuple[float, float],
                                 camera_pose: Optional[np.ndarray] = None):
        """
        ★★★ 改进：添加观测位置到物体记录，使用视角多样性判断去重 ★★★
        
        Args:
            obj_id: 物体ID
            camera_position: 相机位置 (world_x, world_z)
            camera_pose: 相机w2c位姿矩阵（用于计算视角相似度）
        """
        if obj_id not in self.observed_objects:
            return
        
        positions = self.observed_objects[obj_id].get('observation_positions', [])
        
        # ★★★ 改进：视角多样性判断 ★★★
        MIN_POSITION_DIFF = 0.01  # 最小位置差异（米）
        MIN_VIEWPOINT_ANGLE = 1.0  # 最小视角差异（度）
        
        new_pos = np.array(camera_position)
        
        # 检查是否与已有位置过于相似
        for i, existing_pos in enumerate(positions):
            existing_array = np.array(existing_pos)
            pos_dist = np.linalg.norm(new_pos - existing_array)
            
            # ★★★ 规则1：位置差异很小 ★★★
            if pos_dist < MIN_POSITION_DIFF:
                # 位置太近，需要检查视角差异
                # 如果位置非常近（<0.02m），则视为重复观测
                if pos_dist < MIN_POSITION_DIFF * 0.4:
                    self.logger.debug(
                        f"Object {obj_id}: position too close ({pos_dist:.4f}m < {MIN_POSITION_DIFF*0.4}m), skipped"
                    )
                    return
                
                # ★★★ 规则2：位置接近但视角多样吗？ ★★★
                # 如果有相机位姿信息，计算视角差异
                if camera_pose is not None and len(positions) > 0:
                    try:
                        # 获取前一个观测的相机位姿（如果可用）
                        # 这里使用简化版：计算相对位置向量的夹角
                        obj_pos = np.array(self.observed_objects[obj_id]['anchor_position_3d'][:2])
                        
                        current_vec = obj_pos - new_pos
                        existing_vec = obj_pos - existing_array
                        
                        current_norm = np.linalg.norm(current_vec)
                        existing_norm = np.linalg.norm(existing_vec)
                        
                        if current_norm > 1e-3 and existing_norm > 1e-3:
                            cos_angle = np.dot(current_vec, existing_vec) / (current_norm * existing_norm)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle_deg = np.rad2deg(np.arccos(cos_angle))
                            
                            # 视角差异足够大，保留新位置
                            if angle_deg >= MIN_VIEWPOINT_ANGLE:
                                self.logger.debug(
                                    f"Object {obj_id}: position close but viewpoint angle {angle_deg:.1f}° >= {MIN_VIEWPOINT_ANGLE}°, keeping"
                                )
                                break  # 视角差异足够，不跳过
                        
                    except Exception as e:
                        self.logger.debug(f"Viewpoint angle calculation failed: {e}")
                
                # 如果视角差异不足或无法计算，则视为重复
                self.logger.debug(
                    f"Object {obj_id}: position too close and viewpoint similarity insufficient, skipped"
                )
                return
        
        # 通过检查，添加新位置
        positions.append(camera_position)
        self.observed_objects[obj_id]['observation_positions'] = positions
        
        self.logger.info(
            f"Object {obj_id}: added observation position {camera_position}, "
            f"total positions: {len(positions)}"
        )
        print(f"{self.Colors.CYAN}  [SOIM] 记录观测位置: ({camera_position[0]:.2f}, {camera_position[1]:.2f}) "
              f"[总计: {len(positions)}]{self.Colors.END}")
    
    def _find_existing_object_by_boundary(
            self, 
            new_position_2d: Tuple[int, int],
            new_position_3d: np.ndarray,
            new_boundary_2d: Dict[str, int]
        ) -> Optional[str]:
        """
        ★★★ 改进：根据边界重叠判断是否为已知物体，使用锚点位置进行Z轴匹配 ★★★
        并打印调试信息
        
        Args:
            new_position_2d: 新物体中心（栅格坐标）(x, y)
            new_position_3d: 新物体中心（世界坐标）[x, y, z]
            new_boundary_2d: 新物体边界 {x_min, x_max, y_min, y_max}
            
        Returns:
            匹配的物体ID，如果没有匹配则返回None
        """
        new_x, new_y = new_position_2d
        new_z = new_position_3d[2]  # 世界坐标的Z
        best_match_id = None
        best_overlap_score = 0

        # print(f"{self.Colors.YELLOW}[SOIM Debug] Attempting to match new object at grid {new_position_2d}, "
        #     f"world Z={new_z:.2f}, boundary={new_boundary_2d}{self.Colors.END}")

        for obj_id, obj_data in self.observed_objects.items():
            # 只跳过语义类别不同的物体
            if obj_data.get('semantic_class') != self.target_semantic_class:
                continue
            
            old_x, old_y = obj_data['position_2d']
            old_z = obj_data['anchor_position_3d'][2]  # 使用锚点Z
            old_boundary = obj_data['boundary_2d']

            # 打印每个匹配尝试
            # print(f"{self.Colors.CYAN}[SOIM Debug] Checking against object {obj_id}: "
            #     f"anchor Z={old_z:.2f}, boundary={old_boundary}{self.Colors.END}")

            # 检查Z轴距离
            z_diff = abs(new_z - old_z)
            if z_diff > self.z_tolerance:
                # print(f"{self.Colors.RED}[SOIM Debug] Skipped {obj_id} due to Z diff ({z_diff:.2f} > {self.z_tolerance}){self.Colors.END}")
                continue

            # 计算边界重叠面积
            x_overlap = max(0, min(new_boundary_2d['x_max'], old_boundary['x_max']) - 
                        max(new_boundary_2d['x_min'], old_boundary['x_min']))
            y_overlap = max(0, min(new_boundary_2d['y_max'], old_boundary['y_max']) - 
                        max(new_boundary_2d['y_min'], old_boundary['y_min']))
            overlap_area = x_overlap * y_overlap

            new_center_in_old = (old_boundary['x_min'] <= new_x <= old_boundary['x_max'] and
                                old_boundary['y_min'] <= new_y <= old_boundary['y_max'])
            old_center_in_new = (new_boundary_2d['x_min'] <= old_x <= new_boundary_2d['x_max'] and
                                new_boundary_2d['y_min'] <= old_y <= new_boundary_2d['y_max'])

            if new_center_in_old or old_center_in_new or overlap_area > 0:
                match_score = overlap_area
                if new_center_in_old:
                    match_score += 100
                if old_center_in_new:
                    match_score += 100

                # print(f"{self.Colors.GREEN}[SOIM Debug] Potential match with {obj_id}: "
                #     f"match_score={match_score}, overlap_area={overlap_area}, "
                #     f"new_center_in_old={new_center_in_old}, old_center_in_new={old_center_in_new}{self.Colors.END}")

                if match_score > best_overlap_score:
                    best_overlap_score = match_score
                    best_match_id = obj_id
            else:
                # print(f"{self.Colors.RED}[SOIM Debug] Skipped {obj_id} due to no overlap or center inside{self.Colors.END}")
                pass

        if best_match_id:
            # print(f"{self.Colors.GREEN}[SOIM Debug] Final matched object: {best_match_id} (score={best_overlap_score}){self.Colors.END}")
            self.logger.info(
                f"Matched with existing object {best_match_id} (overlap_score={best_overlap_score})"
            )
        else:
            # print(f"{self.Colors.RED}[SOIM Debug] No match found for new object at {new_position_2d}{self.Colors.END}")
            pass

        return best_match_id

    
    def set_object_abandoned(self, obj_id: str):
        """
        ★★★ 改进：设置物体为放弃状态，但保留观测数据 ★★★
        """
        if obj_id in self.observed_objects:
            self.observed_objects[obj_id]['status'] = 'abandoned'
            self.observed_objects[obj_id]['abandon_time'] = time.time()
            # ★★★ 不清除观测位置，以便重新激活时可以继续使用 ★★★
            self.logger.info(f"Object {obj_id} set to abandoned state (data preserved)")
    
    def _get_target_semantic_classes(self) -> List[int]:
        """
        ★★★ 修改：只返回目标语义类别 ★★★
        
        Returns:
            只包含目标类别的列表，如果未设置则返回空列表
        """
        if self.target_semantic_class is not None:
            return [self.target_semantic_class]
        else:
            # 如果没有设置目标类别，返回空列表（不追踪任何物体）
            self.logger.warning("No target semantic class set, SOIM will not track any objects")
            return []
    
    def set_target_class(self, semantic_class: int):
        """
        ★★★ 动态设置目标类别 ★★★
        
        Args:
            semantic_class: 目标语义类别ID
        """
        self.target_semantic_class = semantic_class
        self.logger.info(f"Target semantic class set to: {semantic_class}")
        
        # ★★★ 新增：打印确认信息 ★★★
        print(f"{Colors.GREEN}[SOIM] Target semantic class updated to: {semantic_class}{Colors.END}")
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        将栅格地图坐标转换为世界坐标
        
        这是 OccupancyMapper.world_to_map 的逆运算:
        world_to_map:
            map_x = map_size_px / 2 + world_x * 100 / map_resolution
            map_y = map_size_px / 2 - world_z * 100 / map_resolution
        
        逆运算:
            world_x = (map_x - map_size_px/2) * map_resolution / 100
            world_z = -(map_y - map_size_px/2) * map_resolution / 100
        
        Args:
            grid_x: 栅格地图X坐标（列）
            grid_y: 栅格地图Y坐标（行）
            
        Returns:
            (world_x, world_z): 世界坐标 (米)
        """
        if hasattr(self.occupancy_mapper, 'map_to_world'):
            try:
                return self.occupancy_mapper.map_to_world(int(grid_x), int(grid_y))
            except Exception as exc:
                self.logger.debug(f"occupancy_mapper.map_to_world failed: {exc}")

        map_size_px = self.occupancy_mapper.map_size_px
        map_resolution = self.occupancy_mapper.map_resolution  # 厘米/像素
        
        # 执行逆转换
        world_x = (grid_x - map_size_px / 2.0) * map_resolution / 100.0
        world_z = -(grid_y - map_size_px / 2.0) * map_resolution / 100.0
        
        return world_x, world_z
    
    def reset(self):
        """重置观测记录"""
        self.observed_objects.clear()
        self.current_keyframe_objects.clear()
        self.object_counter = 0  # ★★★ 重置计数器 ★★★
        self.gaussmi_ready = False  # ★★★ 重置 GaussMI 就绪标志 ★★★
        self.clear_candidate_anchor()
        self.logger.info("SOIM reset")
    
    def _is_object_in_view(self, object_pos_3d: np.ndarray, 
                          camera_pose: np.ndarray,
                          current_semantic: Optional[np.ndarray] = None,
                          target_semantic_class: Optional[int] = None) -> bool:
        """
        ★★★ 新增：检查物体是否在相机的视野范围内 ★★★
        
        直接检查当前帧的语义分割图像中是否包含目标语义类别
        
        Args:
            object_pos_3d: 物体3D位置 [x, y, z]（世界坐标）- 本方法中不使用
            camera_pose: 相机位姿 4x4 w2c 矩阵 - 本方法中不使用
            current_semantic: 当前帧的语义分割图像 (H, W) 或 (H, W, C)
            target_semantic_class: 目标语义类别ID
            
        Returns:
            True 如果物体在视野内（即当前帧语义分割包含目标类别），False 否则
        """
        # 如果没有传入语义分割图像，默认认为在视野内（保持兼容性）
        if current_semantic is None:
            self.logger.debug("No semantic segmentation provided, assuming object in view")
            return True
        
        # 如果没有指定目标类别，使用实例变量
        if target_semantic_class is None:
            target_semantic_class = self.target_semantic_class
        
        if target_semantic_class is None:
            self.logger.debug("No target semantic class set, assuming object in view")
            return True
        
        try:
            # ★★★ 处理多维度语义分割 ★★★
            if current_semantic.ndim == 3:
                # 如果是3维数组 (H, W, C)，提取语义通道
                if current_semantic.shape[2] == 3:
                    # 假设语义在第3个通道（索引2）
                    semantic_channel = current_semantic[:, :, 2]
                else:
                    # 如果只有1个通道，直接使用
                    semantic_channel = current_semantic[:, :, 0]
            elif current_semantic.ndim == 2:
                # 单通道格式 (H, W)
                semantic_channel = current_semantic
            else:
                self.logger.warning(f"Unexpected semantic shape: {current_semantic.shape}")
                return True
            
            # ★★★ 检查当前帧语义分割中是否存在目标类别 ★★★
            has_target = np.any(semantic_channel == target_semantic_class)
            
            if has_target:
                num_pixels = np.sum(semantic_channel == target_semantic_class)
                self.logger.debug(
                    f"Target class {target_semantic_class} found in current frame: {num_pixels} pixels"
                )
            else:
                self.logger.debug(
                    f"Target class {target_semantic_class} NOT in current frame"
                )
            
            return has_target
            
        except Exception as e:
            self.logger.warning(f"FOV check failed: {e}")
            return True  # 检查失败时默认认为在视野内
