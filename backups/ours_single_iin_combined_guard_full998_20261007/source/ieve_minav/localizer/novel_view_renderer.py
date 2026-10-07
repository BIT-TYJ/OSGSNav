import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class NovelViewRenderer:
    """新视角渲染模块 - 多尺度环绕物体渲染"""
    
    def __init__(self,
                 slam_mapper,
                 num_views: int = 48,  # ★★★ 修改：增加到48个视角 ★★★
                 render_radius: float = 1.0,  # ★★★ 修改：基础渲染半径改为1m ★★★
                 camera_height: float = 0.88,  # Habitat robot sensor height
                 multi_scale_radii: Optional[List[float]] = None,
                 results_dir: Optional[str] = None,
                 save_visualization_results: bool = True):
        """
        初始化NVSM模块
        
        Args:
            slam_mapper: SLAMMapper实例，提供高斯地图和渲染器
            num_views: 每个尺度的渲染视角数量
            render_radius: 默认渲染半径（米）- 作为中尺度
            camera_height: 相机高度（米）
            multi_scale_radii: 多尺度渲染半径列表（米），如 [0.8, 1.5, 2.5]
        """
        self.logger = logging.getLogger(__name__)
        self.slam_mapper = slam_mapper
        self.num_views = num_views
        self.render_radius = render_radius
        self.camera_height = camera_height
        self.results_dir = results_dir  # ★★★ 新增：结果保存目录 ★★★
        self.save_visualization_results = bool(save_visualization_results)
        
        # ★★★ 多尺度渲染配置：改为5个尺度 ★★★
        if multi_scale_radii is not None:
            self.render_radii = sorted(multi_scale_radii)
        else:
            # 默认五个尺度：极近、近距离、中距离、远距离、极远
            self.render_radii = [
                render_radius * 0.4,   # 极近距离（40%）
                render_radius * 0.7,   # 近距离（70%）
                render_radius,          # 中距离（100%）
                render_radius * 1.5,   # 远距离（150%）
                render_radius * 2.2    # 极远距离（220%）
            ]
        
        self.logger.info(
            f"NVSM initialized with {num_views} views per scale, "
            f"multi-scale radii (5 scales): {self.render_radii}m"
        )
    
    def render_novel_views(self, 
                          object_center_2d: Tuple[int, int],
                          object_center_3d: Optional[np.ndarray] = None,
                          observation_positions: Optional[List[Tuple[float, float]]] = None,
                          target_indices: Optional[np.ndarray] = None,
                          ) -> Dict[str, Any]:
        """
        多尺度环绕物体渲染
        
        Args:
            object_center_2d: 物体在2D栅格地图上的中心坐标 (grid_x, grid_y)
            object_center_3d: 物体的3D世界坐标（可选）[x, y, z]
            observation_positions: 观测位置列表（可选），用于智能选择渲染半径
            target_indices: 可选目标高斯索引。提供时半径只按这些高斯的 bbox 计算。
            
        Returns:
            渲染结果字典
        """
        try:
            
            # 检查 SLAM 可用性
            slam = getattr(self.slam_mapper, 'slam', None)
            is_running = getattr(self.slam_mapper, 'is_running', False)
            
            if slam is None or not is_running:
                return {
                    'success': False,
                    'message': 'SLAM not available or not running',
                    'rendered_images': []
                }
            
            # 检查高斯点数量
            if hasattr(slam, 'gaussians'):
                num_gaussians = slam.gaussians.get_xyz.shape[0] if hasattr(slam.gaussians, 'get_xyz') else 0
                if num_gaussians < 100:
                    return {
                        'success': False,
                        'message': f'Not enough gaussians for rendering: {num_gaussians} < 100',
                        'rendered_images': []
                    }
                
                xyz = slam.gaussians.get_xyz.detach().cpu().numpy()
            # ★★★ 转换坐标：优先使用调用者传入的 3D 中心（已含正确高度）★★★
            if object_center_3d is None:
                object_center_3d = self._convert_grid_to_world(object_center_2d)
                # 仅在无外部中心时回退精化 y
                object_center_3d = self._refine_center_height(object_center_3d)
            
            
            # ★★★ 智能选择渲染半径（基于观测距离）★★★
            adaptive_radii, radius_debug = self._compute_adaptive_radii(
                object_center_3d,
                observation_positions,
                target_indices=target_indices,
            )
            
            
            self.logger.info(
                f"Multi-scale rendering {self.num_views} views × {len(adaptive_radii)} scales "
                f"around object at 3D: {object_center_3d}"
            )
            
            # ★★★ 多尺度批量渲染 ★★★
            all_rendered_images = []
            scale_info = []
            
            # Phase 1: 批量生成所有位姿
            all_poses = []
            pose_meta = []  # (scale_idx, local_idx, radius)
            for scale_idx, radius in enumerate(adaptive_radii):
                camera_poses = self._generate_circular_poses(object_center_3d, radius)
                for i, pose in enumerate(camera_poses):
                    all_poses.append(pose)
                    pose_meta.append((scale_idx, i, radius))
            
            # Phase 2: 批量渲染（tight loop, 避免逐个开销）
            rendered_results = self._batch_render(all_poses)
            
            # Phase 3: 组装结果
            scale_counts = {}
            for idx, (image, pose) in enumerate(zip(rendered_results, all_poses)):
                scale_idx, local_idx, radius = pose_meta[idx]
                if image is None:
                    continue
                entry = {
                    'image': image,
                    'pose': pose,
                    'index': len(all_rendered_images),
                    'local_index': local_idx,
                    'scale_index': scale_idx,
                    'radius': radius
                }
                all_rendered_images.append(entry)
                scale_counts[scale_idx] = scale_counts.get(scale_idx, 0) + 1
            
            for scale_idx, radius in enumerate(adaptive_radii):
                n = scale_counts.get(scale_idx, 0)
                scale_info.append({
                    'scale_index': scale_idx,
                    'radius': radius,
                    'num_views': n,
                    'success': n > 0
                })
            
            if len(all_rendered_images) == 0:
                return {
                    'success': False,
                    'message': 'No views successfully rendered at any scale',
                    'rendered_images': [],
                    'scale_info': scale_info
                }
            
            total_expected = len(adaptive_radii) * self.num_views
            # print(f"\n[NVSM] ★ Total rendered: {len(all_rendered_images)}/{total_expected} views across {len(adaptive_radii)} scales")
            
            self.logger.info(
                f"Successfully rendered {len(all_rendered_images)} views "
                f"across {len(adaptive_radii)} scales"
            )
            
            # ★★★ 不再保存全部渲染图，由调用者(FEM)决定保存最佳 ★★★
            # self._save_rendered_images(all_rendered_images, object_center_2d)
            
            return {
                'success': True,
                'message': f'Rendered {len(all_rendered_images)} views across {len(adaptive_radii)} scales',
                'rendered_images': all_rendered_images,
                'object_center': object_center_3d,
                'scale_info': scale_info,
                'radii_used': adaptive_radii,
                'radius_debug': radius_debug,
            }
            
        except Exception as e:
            self.logger.error(f"Rendering failed: {e}", exc_info=True)
            return {
                'success': False,
                'message': str(e),
                'rendered_images': []
            }
    
    def _compute_adaptive_radii(self, 
                               object_center_3d: np.ndarray,
                               observation_positions: Optional[List[Tuple[float, float]]] = None,
                               target_indices: Optional[np.ndarray] = None,
                               ):
        """
        ★★★ 基于目标语义高斯点的物理尺寸计算自适应渲染半径（5个尺度）★★★
        
        计算方法：
        1. 获取被标记为 target 的语义高斯点的 XZ 平面 bounding box
        2. 取最长边 / 2 作为基准半径 base_radius
        3. 使用 base_radius × [0.4, 0.7, 1.0, 1.5, 2.0] 作为5个渲染尺度
        
        Args:
            object_center_3d: 物体3D中心坐标
            observation_positions: 观测位置列表（此版本不再使用，保留参数兼容性）
            
        Returns:
            自适应渲染半径列表（5个尺度）
        """
        MIN_RADIUS = 0.5  # 最小基准半径（米）
        SCALE_FACTORS = [0.4, 0.7, 1.0, 1.5, 2.0]
        
        base_radius = None
        radius_debug = {
            'source': 'fallback_render_radius',
            'base_radius': None,
            'num_target_gaussians': 0,
            'x_span': None,
            'z_span': None,
            'longest_span': None,
        }

        try:
            slam = getattr(self.slam_mapper, 'slam', None)
            if (
                target_indices is not None
                and slam is not None
                and hasattr(slam, 'gaussians')
                and hasattr(slam.gaussians, 'get_xyz')
            ):
                indices = np.asarray(target_indices, dtype=np.int64).reshape(-1)
                num_gaussians = int(slam.gaussians.get_xyz.shape[0])
                indices = indices[(indices >= 0) & (indices < num_gaussians)]
                if indices.size > 0:
                    xyz = slam.gaussians.get_xyz.detach()
                    target_xyz = xyz[indices]
                    target_x = target_xyz[:, 0].detach().cpu().numpy()
                    target_z = target_xyz[:, 2].detach().cpu().numpy()
                    x_span = float(target_x.max() - target_x.min())
                    z_span = float(target_z.max() - target_z.min())
                    longest_span = max(x_span, z_span)
                    base_radius = longest_span / 2.0
                    radius_debug.update({
                        'source': 'target_indices_bbox',
                        'num_target_gaussians': int(indices.size),
                        'x_span': x_span,
                        'z_span': z_span,
                        'longest_span': longest_span,
                    })
        except Exception:
            base_radius = None
        
        # ★★★ 从 target semantic gaussians 的 bounding box 计算基准半径 ★★★
        # 优先用全部目标语义高斯（_semantic_target > 0.4），不受 confirmed 锁定影响
        try:
            slam = getattr(self.slam_mapper, 'slam', None)
            if base_radius is None and slam is not None:
                target_mask = getattr(slam, '_target_semantic_mask', None)
                try:
                    sem_target = slam.gaussians._semantic_target
                    all_target_mask = sem_target > 0.4
                    if all_target_mask.any():
                        target_mask = all_target_mask
                except Exception:
                    pass  # fallback to _target_semantic_mask
                if target_mask is not None and target_mask.any():
                    xyz = slam.gaussians.get_xyz.detach()
                    target_xyz = xyz[target_mask]  # (N_target, 3)
                    
                    # XZ 平面的 bounding box（X=水平, Z=水平, Y=高度）
                    target_x = target_xyz[:, 0].cpu().numpy()
                    target_z = target_xyz[:, 2].cpu().numpy()
                    
                    x_span = float(target_x.max() - target_x.min())
                    z_span = float(target_z.max() - target_z.min())
                    longest_span = max(x_span, z_span)
                    
                    base_radius = longest_span / 2.0
                    
                    num_target = int(target_mask.sum().item())
                    radius_debug.update({
                        'source': 'semantic_target_bbox',
                        'num_target_gaussians': num_target,
                        'x_span': x_span,
                        'z_span': z_span,
                        'longest_span': longest_span,
                    })
                    # print(f"[NVSM Adaptive] Target gaussians: {num_target}")
                    # print(f"[NVSM Adaptive] Target XZ bounding box: X=[{target_x.min():.2f}, {target_x.max():.2f}]m, "
                          # f"Z=[{target_z.min():.2f}, {target_z.max():.2f}]m")
                    # print(f"[NVSM Adaptive] Spans: X={x_span:.2f}m, Z={z_span:.2f}m, longest={longest_span:.2f}m")
                    # print(f"[NVSM Adaptive] Raw base_radius (longest/2): {base_radius:.2f}m")
                else:
                    pass
                    # print(f"[NVSM Adaptive] No target semantic mask available, using fallback")
            else:
                pass
                # print(f"[NVSM Adaptive] SLAM not available, using fallback")
        except Exception as e:
            pass
            # print(f"[NVSM Adaptive] Failed to get target gaussians: {e}, using fallback")
        
        # 如果无法从 target gaussians 获取，使用默认半径
        if base_radius is None:
            base_radius = self.render_radius  # 默认 1.0m
            # print(f"[NVSM Adaptive] Fallback base_radius: {base_radius:.2f}m")
        
        # 确保基准半径不小于最小值
        base_radius = max(MIN_RADIUS, base_radius)
        radius_debug['base_radius'] = float(base_radius)
        # print(f"[NVSM Adaptive] Final base_radius: {base_radius:.2f}m (min={MIN_RADIUS}m)")
        
        # ★★★ 生成5个渲染半径：base_radius × [0.4, 0.7, 1.0, 1.5, 2.0] ★★★
        filtered_radii = [round(base_radius * s, 2) for s in SCALE_FACTORS]
        
        # print(f"[NVSM Adaptive] Scale factors: {SCALE_FACTORS}")
        # print(f"[NVSM Adaptive] Selected {len(filtered_radii)} radii: {[f'{r:.2f}m' for r in filtered_radii]}")
        
        radius_debug['radii'] = filtered_radii
        return filtered_radii, radius_debug
          
    def _refine_center_height(self, center_3d: np.ndarray) -> np.ndarray:
        """
        ★★★ 修复：用 SLAM 目标语义高斯点精化渲染中心的 y 坐标 ★★★
        
        SOIM 基于 2D 栅格地图检测物体，无法获取高度信息，默认 y=0。
        此方法查询 SLAM 的目标语义高斯点，取其 Habitat-y 的中位数作为物体高度。
        如果查询失败，回退到 camera_height（机器人传感器高度）。
        
        坐标系约定：
            get_xyz 返回的高斯点坐标已是 Habitat 世界坐标（flip_mat 右乘不影响平移列）
        """
        import torch
        refined = center_3d.copy()
        original_y = float(center_3d[1])
        
        try:
            slam = getattr(self.slam_mapper, 'slam', None)
            if slam is not None and hasattr(slam, 'gaussians'):
                target_mask = getattr(slam, '_target_semantic_mask', None)
                # ★ 优先用全部目标语义高斯，不受 confirmed 锁定影响
                try:
                    sem_target = slam.gaussians._semantic_target
                    all_target_mask = sem_target > 0.4
                    if all_target_mask.any():
                        target_mask = all_target_mask
                except Exception:
                    pass
                if target_mask is not None and target_mask.any():
                    xyz = slam.gaussians.get_xyz.detach()
                    target_xyz = xyz[target_mask]
                    # get_xyz 的 y 坐标已是 Habitat 世界坐标，无需取反
                    target_y_habitat = target_xyz[:, 1].cpu().numpy()
                    median_y = float(np.median(target_y_habitat))
                    # median_y = float(np.median(target_y_habitat)) + 0.6
                    refined[1] = median_y
                    # print(f"[NVSM] Center y refined from {original_y:.2f} -> "
                          # f"{median_y:.2f} (median of {int(target_mask.sum().item())} target gaussians)")
                    return refined
        except Exception as e:
            pass
            # print(f"[NVSM] Warning: Failed to refine center y from gaussians: {e}")
        
        # 回退：使用机器人相机高度
        if abs(original_y) < 0.01:  # y approx 0 说明是 SOIM 的默认值
            refined[1] = self.camera_height
            # print(f"[NVSM] Center y fallback from {original_y:.2f} -> {self.camera_height:.2f} (camera_height)")
        
        return refined
    
    def _convert_grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """
        将栅格地图像素坐标转换为世界坐标
        
        Args:
            grid_pos: (grid_x, grid_y) 栅格地图像素坐标 (列, 行)
            
        Returns:
            [world_x, world_y, world_z] 世界坐标
        """
        # ★★★ 修复：确保输入是整数 ★★★
        grid_x = int(grid_pos[0])
        grid_y = int(grid_pos[1])
        
        # 尝试从 slam_mapper 获取 occupancy_mapper 引用
        occupancy_mapper = None
        if hasattr(self.slam_mapper, 'occupancy_mapper'):
            occupancy_mapper = self.slam_mapper.occupancy_mapper
        
        if occupancy_mapper is not None:
            if hasattr(occupancy_mapper, 'map_to_world'):
                try:
                    world_x, world_z = occupancy_mapper.map_to_world(grid_x, grid_y)
                    world_y = float(0.5)
                    self.logger.info(
                        f"Converted grid ({grid_x}, {grid_y}) via occupancy_mapper "
                        f"-> world ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})"
                    )
                    return np.array([world_x, world_y, world_z], dtype=np.float32)
                except Exception as exc:
                    self.logger.debug(f"occupancy_mapper.map_to_world failed: {exc}")
            map_size_px = int(occupancy_mapper.map_size_px)
            map_resolution = float(occupancy_mapper.map_resolution)
        else:
            # 默认值
            map_size_px = 480
            map_resolution = 5.0
        
        # world_to_map 的逆运算
        world_x = float((grid_x - map_size_px / 2.0) * map_resolution / 100.0)
        world_z = float(-(grid_y - map_size_px / 2.0) * map_resolution / 100.0)
        # ★★★ 修复：使用合理的 Y 坐标（物体高度），而不是相机高度 ★★★
        world_y = float(0.5)  # 假设物体在地面以上 0.5 米的高度
        
        self.logger.info(f"Converted grid ({grid_x}, {grid_y}) -> world ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")
        
        # ★★★ 修复：确保返回 numpy 数组 ★★★
        return np.array([world_x, world_y, world_z], dtype=np.float32)
    
    def _generate_circular_poses(self, center: np.ndarray, 
                                radius: Optional[float] = None) -> List[np.ndarray]:
        """
        在物体周围生成水平环绕的相机位姿
        
        Args:
            center: 物体中心世界坐标 [x, y, z]
            
        Returns:
            位姿列表，每个位姿为4x4变换矩阵 (w2c格式)
        """
        if radius is None:
            radius = self.render_radius
        
        poses = []
        angles = np.linspace(0, 2 * np.pi, self.num_views, endpoint=False)
        
        for angle in angles:
            # 在XZ平面水平环绕
            camera_x = center[0] + radius * np.cos(angle)
            camera_z = center[2] + radius * np.sin(angle)
            # ★★★ 修复：相机高度应相对于物体中心定位，而非使用绝对值 ★★★
            # 目标图像是从机器人高度（~0.88m above floor）拍摄的，
            # 但 center[1] 在 Habitat 世界坐标系下可能远离 0（例如 -2.5），
            # 如果仍用绝对 camera_height，相机可能比物体高出数米，
            # 导致极端俯视角，渲染出全是虚影/伪影的图像。
            # 修复：camera_y = center[1] + 相对偏移（机器人视角略高于物体中心）
            camera_y = center[1] + 0.3  # 相机比物体中心高 0.3m，略微俯视
            
            camera_pos = np.array([camera_x, camera_y, camera_z])
            
            # 相机朝向物体中心
            forward = center - camera_pos
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-6:
                self.logger.warning("Camera too close to object center, skipping")
                continue
            forward = forward / forward_norm
            
            # 构建相机坐标系
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                right = np.array([1, 0, 0])
            else:
                right = right / right_norm
            up = np.cross(right, forward)
            
            # 构建 c2w 矩阵
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = -forward
            c2w[:3, 3] = camera_pos
            
            # Habitat 坐标系转换
            flip_mat = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ], dtype=np.float32)
            
            c2w_correct = c2w @ flip_mat
            w2c = np.linalg.inv(c2w_correct)
            
            poses.append(w2c)
        
        return poses

    def render_relative_yaw_views(
            self,
            object_center_3d: np.ndarray,
            reference_center_3d: np.ndarray,
            reference_pose_w2c: np.ndarray,
            yaw_range_deg: float = 45.0,
            yaw_step_deg: float = 5.0) -> Dict[str, Any]:
        """Render a bounded yaw sweep using the strongest original view as template."""
        center = np.asarray(object_center_3d, dtype=np.float64).reshape(-1)
        reference_center = np.asarray(reference_center_3d, dtype=np.float64).reshape(-1)
        pose = np.asarray(reference_pose_w2c, dtype=np.float64)
        if center.size < 3 or reference_center.size < 3 or pose.shape != (4, 4):
            return {
                'success': False,
                'message': 'Invalid center/reference pose for relative yaw rendering',
                'rendered_images': [],
            }

        step = abs(float(yaw_step_deg))
        yaw_range = abs(float(yaw_range_deg))
        if step <= 0.0:
            return {
                'success': False,
                'message': 'yaw_step_deg must be positive',
                'rendered_images': [],
            }

        try:
            reference_c2w = np.linalg.inv(pose)
        except np.linalg.LinAlgError:
            return {
                'success': False,
                'message': 'Reference pose is singular',
                'rendered_images': [],
            }
        relative_camera = reference_c2w[:3, 3] - reference_center[:3]
        if float(np.linalg.norm(relative_camera[[0, 2]])) < 1e-4:
            return {
                'success': False,
                'message': 'Reference camera has no horizontal baseline to cluster',
                'rendered_images': [],
            }

        offsets = np.arange(-yaw_range, yaw_range + step * 0.5, step, dtype=np.float64)
        poses = []
        for offset_deg in offsets:
            angle = np.deg2rad(float(offset_deg))
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotated = relative_camera.copy()
            rotated[0] = cos_a * relative_camera[0] + sin_a * relative_camera[2]
            rotated[2] = -sin_a * relative_camera[0] + cos_a * relative_camera[2]
            camera_position = center[:3] + rotated
            poses.append(self._look_at_w2c(camera_position, center[:3]))

        rendered = self._batch_render(poses)
        entries = []
        for index, (image, rendered_pose, offset_deg) in enumerate(
                zip(rendered, poses, offsets.tolist())):
            if image is None:
                continue
            entries.append({
                'image': image,
                'pose': rendered_pose,
                'index': int(index),
                'yaw_offset_deg': float(offset_deg),
                'instance_center_3d': center[:3].astype(np.float32),
            })
        return {
            'success': bool(entries),
            'message': 'Rendered {}/{} relative yaw views'.format(len(entries), len(poses)),
            'rendered_images': entries,
            'yaw_offsets_deg': offsets.tolist(),
            'object_center': center[:3].astype(np.float32),
        }

    @staticmethod
    def _look_at_w2c(camera_position: np.ndarray, center: np.ndarray) -> np.ndarray:
        camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
        center = np.asarray(center, dtype=np.float64).reshape(3)
        forward = center - camera_position
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            raise ValueError('Camera position coincides with cluster center')
        forward /= norm
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= right_norm
        up = np.cross(right, forward)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = camera_position
        flip_mat = np.diag([1.0, -1.0, -1.0, 1.0])
        return np.linalg.inv(c2w @ flip_mat)
    
    def _batch_render(self, poses: list) -> list:
        """
        批量渲染多个视角 — 预创建所有相机，在 torch.no_grad() 下紧密循环渲染。
        
        Args:
            poses: 位姿列表，每个为 4x4 numpy w2c 矩阵
            
        Returns:
            渲染图像列表（与 poses 等长），失败的位置为 None
        """
        import torch
        from gaussian_mapping.gaussian_splatting.gaussian_renderer import render as gs_render
        
        slam = getattr(self.slam_mapper, 'slam', None)
        if slam is None or not getattr(self.slam_mapper, 'is_running', False):
            return [None] * len(poses)
        
        if not hasattr(slam, 'gaussians') or slam.gaussians.get_xyz.shape[0] == 0:
            return [None] * len(poses)
        
        height = getattr(slam, 'height', 480)
        width = getattr(slam, 'width', 640)
        dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        dummy_depth = np.zeros((height, width), dtype=np.float32)
        
        # Phase 1: 批量创建虚拟相机
        cameras = []
        for i, pose in enumerate(poses):
            try:
                cam = slam.create_camera(dummy_rgb, dummy_depth, pose, uid=-(i + 1))
                cameras.append(cam)
            except Exception:
                cameras.append(None)
        
        # Phase 2: 批量渲染（tight loop under no_grad）
        results = []
        with torch.no_grad():
            for cam in cameras:
                if cam is None:
                    results.append(None)
                    continue
                try:
                    render_pkg = gs_render(cam, slam.gaussians, slam.pipeline_params, slam.background)
                    rendered_np = (render_pkg['render'].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    results.append(rendered_np)
                except Exception:
                    results.append(None)
        
        return results

    def _render_view(self, pose: np.ndarray, view_index: int) -> Optional[np.ndarray]:
        """
        从指定位姿渲染图像（调用真实的SLAM渲染器）
        
        Args:
            pose: 4x4相机位姿矩阵 (w2c格式)
            view_index: 视角索引（用于调试）
            
        Returns:
            渲染的RGB图像 (H, W, 3) uint8
        """
        try:
            import torch
            
            # ★★★ 修复：获取底层 RealtimeSLAM 实例 ★★★
            # slam_mapper 是 SLAMMapper 包装类，实际的渲染方法在 slam_mapper.slam (RealtimeSLAM) 中
            slam = None
            if hasattr(self.slam_mapper, 'slam') and self.slam_mapper.slam is not None:
                slam = self.slam_mapper.slam
            else:
                self.logger.error("SLAM instance not available in slam_mapper")
                return None
            
            # 检查是否有足够的关键帧用于渲染
            if not hasattr(slam, 'kf_indices') or len(slam.kf_indices) < 1:
                self.logger.warning(f"View {view_index}: Not enough keyframes for rendering (need >= 1)")
                return None
            
            # 方案1: 使用 RealtimeSLAM 的 create_camera + render_frame
            if hasattr(slam, 'create_camera') and hasattr(slam, 'render_frame'):
                # 创建虚拟相机（使用dummy数据）
                height = getattr(slam, 'height', 480)
                width = getattr(slam, 'width', 640)
                
                dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                dummy_depth = np.zeros((height, width), dtype=np.float32)
                
                # 使用负数UID来标识这是虚拟相机，不会与真实关键帧冲突
                virtual_cam = slam.create_camera(
                    dummy_rgb, dummy_depth, pose, uid=-(view_index + 1)
                )
                
                # 使用 render_frame 方法
                render_result = slam.render_frame(virtual_cam)
                
                if render_result is not None:
                    if isinstance(render_result, dict) and 'rendered' in render_result:
                        self.logger.info(f"View {view_index}: Successfully rendered using render_frame")
                        return render_result['rendered']
                    elif isinstance(render_result, np.ndarray):
                        return render_result
                
                self.logger.warning(f"View {view_index}: render_frame returned invalid result")
            
            # 方案2: 直接调用底层渲染 (备选)
            if hasattr(slam, 'gaussians') and hasattr(slam, 'pipeline_params'):
                try:
                    from gaussian_mapping.gaussian_splatting.gaussian_renderer import render as gs_render
                    
                    # 确保有高斯点
                    if slam.gaussians.get_xyz.shape[0] == 0:
                        self.logger.warning(f"View {view_index}: No gaussians available for rendering")
                        return None
                    
                    # 创建虚拟相机
                    height = getattr(slam, 'height', 480)
                    width = getattr(slam, 'width', 640)
                    dummy_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                    dummy_depth = np.zeros((height, width), dtype=np.float32)
                    
                    virtual_cam = slam.create_camera(
                        dummy_rgb, dummy_depth, pose, uid=-(view_index + 1)
                    )
                    
                    with torch.no_grad():
                        render_pkg = gs_render(
                            virtual_cam,
                            slam.gaussians,
                            slam.pipeline_params,
                            slam.background
                        )
                    
                    rendered_tensor = render_pkg['render']
                    rendered_np = (rendered_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    self.logger.info(f"View {view_index}: Successfully rendered using direct gs_render")
                    return rendered_np
                    
                except Exception as e:
                    self.logger.error(f"View {view_index}: Direct render failed: {e}")
            
            self.logger.error(f"View {view_index}: No valid rendering method found")
            return None
            
        except Exception as e:
            self.logger.error(f"Render error for view {view_index}: {e}", exc_info=True)
            return None

    def _save_rendered_images(self, rendered_images: List[Dict], object_center_2d: Tuple[int, int],
                               best_index: int = -1):
        """
        ★★★ 修改：不再保存单独文件，只保存一张合并图和一张最优图到results目录 ★★★
        """
        import os
        import cv2
        from datetime import datetime
        
        try:
            # ★★★ 修改：保存到results/habitat_imagenav_日期_时间/ 下 ★★★
            # ★★★ 修复：不再使用 sorted(subdirs)[-1] 回退，防止多进程保存错乱 ★★★
            if self.results_dir is not None:
                session_dir = os.path.join(str(self.results_dir), "NVSM")
            else:
                # 使用进程ID创建隔离目录，避免多进程冲突
                pid = os.getpid()
                session_dir = os.path.join("output", f"pid_{pid}", "NVSM")
                self.logger.warning(
                    f"results_dir not set, falling back to process-isolated dir: {session_dir}. "
                    f"Call update_results_dir() to fix."
                )
            
            os.makedirs(session_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # print(f"\n{'='*70}")
            # print(f"[NVSM] 📸 Saving merged grid + best view to: {session_dir}")
            # print(f"{'='*70}")
            
            # ★★★ 只保存合并网格图 ★★★
            self._create_grid_visualization(rendered_images, session_dir, 
                                           filename=f"all_views_grid_{timestamp}.png",
                                           best_index=best_index)
            
            # ★★★ 保存最优匹配图（如果有的话）★★★
            if 0 <= best_index < len(rendered_images):
                best_item = rendered_images[best_index]
                best_image = best_item['image']
                if best_image is not None:
                    best_path = os.path.join(session_dir, f"best_view_{timestamp}_idx{best_index}.png")
                    if len(best_image.shape) == 3 and best_image.shape[2] == 3:
                        best_bgr = cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR)
                    else:
                        best_bgr = best_image
                    cv2.imwrite(best_path, best_bgr)
                    
                    c2w = np.linalg.inv(best_item['pose'])
                    pos = c2w[:3, 3]
                    # print(f"[NVSM] ✓ Best view #{best_index} saved: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # print(f"[NVSM] ✓ Saved to {session_dir}")
            # print(f"{'='*70}\n")
            
        except Exception as e:
            self.logger.error(f"Failed to save rendered images: {e}")
            # print(f"[NVSM] ⚠️ Warning: Failed to save images: {e}")
            
    def _create_grid_visualization(self, rendered_images: List[Dict], output_dir: str,
                                    filename: str = "all_views_grid.png",
                                    best_index: int = -1):
        """
        创建渲染图像的网格可视化
        
        Args:
            rendered_images: 渲染图像列表
            output_dir: 输出目录
            filename: 输出文件名
            best_index: 最佳匹配视角索引（高亮显示）
        """
        import cv2
        import os
        
        try:
            if len(rendered_images) == 0:
                return
            
            # 获取图像尺寸
            first_image = rendered_images[0]['image']
            if first_image is None:
                return
            
            h, w = first_image.shape[:2]
            
            # 计算网格尺寸（尽量接近正方形）
            n_images = len(rendered_images)
            n_cols = int(np.ceil(np.sqrt(n_images)))
            n_rows = int(np.ceil(n_images / n_cols))
            
            # 缩放因子（提高分辨率上限）
            scale = min(1.0, 3840 / (n_cols * w), 2160 / (n_rows * h))
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 创建网格图像
            grid_h = n_rows * new_h
            grid_w = n_cols * new_w
            grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            for idx, item in enumerate(rendered_images):
                image = item['image']
                view_idx = item['index']
                
                if image is None:
                    continue
                
                # 缩放图像
                resized = cv2.resize(image, (new_w, new_h))
                
                # 如果是RGB，转为BGR
                if len(resized.shape) == 3 and resized.shape[2] == 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                
                # 添加视角编号标签（最优视角高亮红色）
                is_best = (view_idx == best_index)
                label_color = (0, 0, 255) if is_best else (0, 255, 0)
                label = f"View {view_idx}" + (" *BEST" if is_best else "")
                cv2.putText(resized, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                
                # ★★★ 最优视角加红色边框 ★★★
                if is_best:
                    cv2.rectangle(resized, (0, 0), (new_w - 1, new_h - 1), (0, 0, 255), 3)
                
                # 放入网格
                row = idx // n_cols
                col = idx % n_cols
                y1, y2 = row * new_h, (row + 1) * new_h
                x1, x2 = col * new_w, (col + 1) * new_w
                grid_image[y1:y2, x1:x2] = resized
            
            # 保存网格图像
            grid_path = os.path.join(output_dir, filename)
            cv2.imwrite(grid_path, grid_image)
            # print(f"  Grid visualization saved: {grid_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create grid visualization: {e}")
    def render_novel_views_multi_center(
            self,
            instance_centers_3d: 'List[np.ndarray]',
            cluster_sizes: 'List[int]' = None,
            object_center_2d: 'Tuple[int, int]' = (0, 0),
            observation_positions=None,
    ) -> 'Dict[str, Any]':
        """
        ★★★ 新增：绕多个 Instance Center 渲染新视角 ★★★
        
        对每个聚类中心调用 render_novel_views，然后将所有渲染结果合并。
        后续 FEM 在合并的全部视角中挑选匹配度最高的。
        
        Args:
            instance_centers_3d: 实例聚类中心列表，每个 [x, y, z] (Habitat 世界坐标)
            cluster_sizes: 每个聚类的点数（用于日志，可选）
            object_center_2d: 物体在 2D 栅格地图上的中心坐标（用于保存可视化）
            observation_positions: 观测位置列表（用于自适应半径）
            
        Returns:
            与 render_novel_views 相同格式的合并结果
        """
        import numpy as np
        
        if not instance_centers_3d:
            return {
                'success': False,
                'message': 'No instance centers provided',
                'rendered_images': [],
            }
        
        num_centers = len(instance_centers_3d)
        cluster_sizes = cluster_sizes or [0] * num_centers
        
        # print(f"\n{'='*70}")
        # print(f"[NVSM] ★ Multi-Center Rendering: {num_centers} instance centers ★")
        for i, (c, s) in enumerate(zip(instance_centers_3d, cluster_sizes)):
            pass
            # print(f"  Center {i}: [{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}] (size={s})")
        # print(f"{'='*70}\n")
        
        all_rendered_images = []
        all_scale_info = []
        all_radii_used = []
        per_center_info = []
        
        for center_idx, center_3d in enumerate(instance_centers_3d):
            center_3d = np.asarray(center_3d, dtype=np.float32)
            
            # print(f"\n{'─'*60}")
            # print(f"[NVSM] Rendering around Instance Center {center_idx}/{num_centers-1} "
                  # f"[{center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}] "
                  # f"(cluster_size={cluster_sizes[center_idx]})")
            # print(f"{'─'*60}")
            
            sub_result = self.render_novel_views(
                object_center_2d=object_center_2d,
                object_center_3d=center_3d,
                observation_positions=observation_positions,
            )
            
            if sub_result['success'] and sub_result['rendered_images']:
                # 给每张渲染图标注其所属的 instance center 索引
                offset = len(all_rendered_images)
                for img_data in sub_result['rendered_images']:
                    img_data['instance_center_idx'] = center_idx
                    img_data['instance_center_3d'] = center_3d.copy()
                    img_data['index'] = offset  # 更新全局索引
                    offset += 1
                    all_rendered_images.append(img_data)
                
                all_scale_info.extend(sub_result.get('scale_info', []))
                all_radii_used.extend(sub_result.get('radii_used', []))
                per_center_info.append({
                    'center_idx': center_idx,
                    'center_3d': center_3d.copy(),
                    'num_views': len(sub_result['rendered_images']),
                    'success': True,
                })
            else:
                per_center_info.append({
                    'center_idx': center_idx,
                    'center_3d': center_3d.copy(),
                    'num_views': 0,
                    'success': False,
                    'message': sub_result.get('message', 'unknown'),
                })
                # print(f"[NVSM] ⚠ Center {center_idx} rendering failed: "
                      # f"{sub_result.get('message', 'unknown')}")
        
        # 重新编号全局索引（保证连续）
        for i, img_data in enumerate(all_rendered_images):
            img_data['index'] = i
        
        if len(all_rendered_images) == 0:
            return {
                'success': False,
                'message': f'No views rendered from any of {num_centers} centers',
                'rendered_images': [],
                'per_center_info': per_center_info,
            }
        
        # print(f"\n{'='*70}")
        # print(f"[NVSM] ★ Multi-Center Rendering Complete ★")
        # print(f"  Total views: {len(all_rendered_images)} from {num_centers} centers")
        for info in per_center_info:
            status = "✓" if info['success'] else "✗"
            # print(f"  {status} Center {info['center_idx']}: {info['num_views']} views")
        # print(f"{'='*70}\n")
        
        # ★★★ 不再保存旧目录下的网格图+最优图 ★★★
        # self._save_rendered_images(all_rendered_images, object_center_2d)
        
        return {
            'success': True,
            'message': f'Rendered {len(all_rendered_images)} views across '
                       f'{num_centers} instance centers',
            'rendered_images': all_rendered_images,
            'object_center': instance_centers_3d[0],  # 主中心（最大聚类）
            'scale_info': all_scale_info,
            'radii_used': sorted(set(all_radii_used)),
            'per_center_info': per_center_info,
            'num_instance_centers': num_centers,
        }
