"""
Pose Refiner (REF) 模块

对 FEM 匹配得到的粗定位位姿进行梯度下降优化，输出细定位位姿。
- 高斯模型：从 SLAM 增量建图中获取
- RGB：目标图像
- Depth：预处理后的目标深度图（可选）
- 特征点：预提取的目标特征点（可选）
"""

import logging
import copy
import numpy as np
import torch
from typing import Optional, Dict, Any
from math import atan

from ieve_minav.localizer.colors import Colors


def focal2fov(focal, pixels):
    return 2 * atan(pixels / (2 * focal))


class PoseRefiner:
    """
    位姿细定位模块 - 基于3DGS渲染的梯度下降位姿优化

    在 FEM 匹配成功后被触发，对粗定位位姿做精细优化。
    """

    def __init__(self,
                 slam_mapper,
                 camera_params: Dict[str, Any],
                 config: Optional[Dict[str, Any]] = None,
                 save_visualization_results: bool = True):
        """
        初始化细定位模块

        Args:
            slam_mapper: SLAMMapper 实例（提供增量式 3DGS 模型和渲染能力）
            camera_params: 相机内参 {'fx','fy','cx','cy','width','height'}
            config: 细定位配置参数
        """
        self.logger = logging.getLogger(__name__)
        self.slam_mapper = slam_mapper
        self.camera_params = camera_params
        self.results_dir = None  # ★★★ 新增：由 localization_system 设置 ★★★
        self.save_visualization_results = bool(save_visualization_results)

        default_config = {
            'max_iterations': 50,
            'lr_rotation': 0.001,
            'lr_translation': 0.001,
            'lr_exposure_a': 0.001,
            'lr_exposure_b': 0.001,
            'converged_threshold': 1e-4,
            'opacity_threshold': 0.99,
            'edge_threshold': 1.1,
        }
        self.config = {**default_config, **(config or {})}

        # 预计算 FOV
        self.fovx = focal2fov(camera_params['fx'], camera_params['width'])
        self.fovy = focal2fov(camera_params['fy'], camera_params['height'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 内部状态
        self._refinement_tools_available = False
        self._try_import_refinement_tools()

        self.logger.info(
            f"PoseRefiner initialized: "
            f"max_iter={self.config['max_iterations']}, "
            f"tools_available={self._refinement_tools_available}"
        )

    def _try_import_refinement_tools(self):
        """尝试导入细定位依赖"""
        try:
            # ★★★ 修复：使用正确的包路径 ★★★
            from ieve_minav.localizer.refinement_tools.camera_utils import Camera
            from ieve_minav.localizer.refinement_tools.graphics_utils import getProjectionMatrix2
            from ieve_minav.localizer.refinement_tools.descent_utils import get_loss_tracking
            from ieve_minav.localizer.refinement_tools.pose_utils import update_pose

            # ★★★ 单独导入 render（依赖 diff_gaussian_rasterization）★★★
            # __init__.py 中 eagerly 导入了 CUDA 扩展，所以需要单独 try
            try:
                from ieve_minav.localizer.refinement_tools import render as gs_render
            except ImportError as e_render:
                self.logger.warning(
                    f"diff_gaussian_rasterization not available: {e_render}. "
                    f"Refinement will be disabled."
                )
                self._refinement_tools_available = False
                return

            self._Camera = Camera
            self._gs_render = gs_render
            self._get_loss_tracking = get_loss_tracking
            self._update_pose = update_pose
            self._getProjectionMatrix2 = getProjectionMatrix2
            self._refinement_tools_available = True
            self.logger.info("Refinement tools imported successfully")
        except ImportError as e:
            self.logger.warning(f"Refinement tools not available: {e}")
            self._refinement_tools_available = False

    def refine_pose(self,
                    coarse_pose: np.ndarray,
                    target_image: np.ndarray,
                    depth_input=None,
                    keypoints_input=None) -> Dict[str, Any]:
        """
        对粗定位位姿进行梯度下降优化

        Args:
            coarse_pose: 粗定位位姿 (4x4 w2c 矩阵, numpy)
            target_image: 目标图像 (H, W, 3) uint8 RGB
            depth_input: 预处理后的深度图，可为 None
            keypoints_input: 预提取的特征点，可为 None

        Returns:
            dict: {
                'success': bool,
                'refined_pose': np.ndarray (4x4 w2c),
                'coarse_pose': np.ndarray (4x4 w2c),
                'iterations': int,
                'converged': bool,
                'message': str
            }
        """
        result = {
            'success': False,
            'refined_pose': coarse_pose.copy(),
            'coarse_pose': coarse_pose.copy(),
            'iterations': 0,
            'converged': False,
            'message': ''
        }

        # 检查依赖
        if not self._refinement_tools_available:
            result['message'] = 'Refinement tools not available'
            self.logger.warning(result['message'])
            # 回退：直接返回粗定位位姿
            result['success'] = True
            result['message'] += ' (fallback: using coarse pose)'
            return result

        # 检查 SLAM 是否可用
        if self.slam_mapper is None or self.slam_mapper.slam is None:
            result['message'] = 'SLAM mapper not available'
            self.logger.warning(result['message'])
            result['success'] = True
            result['message'] += ' (fallback: using coarse pose)'
            return result

        try:
            with torch.enable_grad():
                refined_pose, iterations, converged = self._gradient_descent(
                    coarse_pose, target_image, depth_input, keypoints_input
                )
            result['success'] = True
            result['refined_pose'] = refined_pose
            result['iterations'] = iterations
            result['converged'] = converged
            result['message'] = (
                f"Refined in {iterations} iterations, converged={converged}"
            )

            # 计算优化前后的位姿差异
            pose_diff = self._compute_pose_difference(coarse_pose, refined_pose)
            result['translation_change'] = pose_diff['translation']
            result['rotation_change'] = pose_diff['rotation']

            self.logger.info(
                f"Pose refinement: iter={iterations}, converged={converged}, "
                f"Δt={pose_diff['translation']:.4f}m, Δr={pose_diff['rotation']:.4f}°"
            )

        except Exception as e:
            result['message'] = f"Refinement failed: {e}"
            self.logger.error(result['message'], exc_info=True)
            # 回退：返回粗定位位姿
            result['success'] = True
            result['message'] += ' (fallback: using coarse pose)'

        # ★★★ 新增：保存REF优化后位姿的渲染图 ★★★
        self._save_refined_render(result['refined_pose'], result['coarse_pose'])

        return result

    def _gradient_descent(self,
                          coarse_pose: np.ndarray,
                          target_image: np.ndarray,
                          depth_input,
                          keypoints_input):
        """
        核心梯度下降优化

        Args:
            coarse_pose: 4x4 w2c numpy
            target_image: (H, W, 3) uint8 RGB
            depth_input: 目标深度图，可为 None
            keypoints_input: 目标特征点，可为 None

        Returns:
            (refined_pose_np, iterations, converged)
        """
        # ---- 1. 从 SLAM 获取高斯模型和渲染管线参数 ----
        gaussian_model = self._get_gaussian_model()
        pipeline_params = self._get_pipeline_params()
        background = self._get_background()

        # ---- 2. 构建投影矩阵 ----
        cp = self.camera_params
        projection_matrix = self._getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=cp['fx'], fy=cp['fy'],
            cx=cp['cx'], cy=cp['cy'],
            W=cp['width'], H=cp['height'],
        ).transpose(0, 1).to(device=self.device)

        # ---- 3. 构建 viewpoint（Camera 对象）----
        viewpoint = self._build_viewpoint(
            target_image, depth_input, keypoints_input,
            coarse_pose, projection_matrix
        )

        # ---- 4. 从粗定位位姿提取初始 R, T ----
        initial_R = torch.tensor(coarse_pose[:3, :3], dtype=torch.float32)
        initial_T = torch.tensor(coarse_pose[:3, 3], dtype=torch.float32)

        viewpoint.update_RT(initial_R, initial_T)

        # ---- 5. 构建优化器 ----
        opt_params = [
            {"params": [viewpoint.cam_rot_delta],
             "lr": self.config['lr_rotation'],
             "name": f"rot_{viewpoint.uid}"},
            {"params": [viewpoint.cam_trans_delta],
             "lr": self.config['lr_translation'],
             "name": f"trans_{viewpoint.uid}"},
            {"params": [viewpoint.exposure_a],
             "lr": self.config['lr_exposure_a'],
             "name": f"exposure_a_{viewpoint.uid}"},
            {"params": [viewpoint.exposure_b],
             "lr": self.config['lr_exposure_b'],
             "name": f"exposure_b_{viewpoint.uid}"},
        ]
        optimizer = torch.optim.Adam(opt_params)

        # ---- 6. 构建 refinement config ----
        has_depth = viewpoint.depth is not None
        ref_config = self._build_refinement_config(has_depth=has_depth)
        mode = "RGB-D" if has_depth else "RGB-only"
        print(f"{Colors.CYAN}[REF] Refinement loss mode: {mode}{Colors.END}")

        # ---- 7. 计算 grad_mask ----
        viewpoint.compute_grad_mask(ref_config)

        # 如果有特征点，用于增强 grad_mask
        if keypoints_input is not None:
            self._apply_keypoints_mask(viewpoint, keypoints_input)

        # ---- 8. 梯度下降循环 ----
        max_iter = self.config['max_iterations']
        converged = False
        final_iter = 0

        for itr in range(max_iter):
            render_pkg = self._gs_render(
                viewpoint, gaussian_model, pipeline_params, background
            )
            image = render_pkg["render"]
            depth = render_pkg["depth"]
            opacity = render_pkg["opacity"]

            optimizer.zero_grad()

            loss = self._get_loss_tracking(
                ref_config, image, depth, opacity, viewpoint
            )
            if not loss.requires_grad:
                raise RuntimeError(
                    "REF loss has no grad_fn; check torch grad mode and "
                    "differentiable render outputs"
                )
            loss.backward()

            with torch.no_grad():
                optimizer.step()
                converged = self._update_pose(
                    viewpoint, converged_threshold=self.config['converged_threshold']
                )

            final_iter = itr + 1
            if converged:
                break

        # ---- 9. 提取优化后的位姿 ----
        R_refined = viewpoint.R.cpu().numpy()
        T_refined = viewpoint.T.cpu().numpy()

        refined_pose = np.eye(4, dtype=np.float64)
        refined_pose[:3, :3] = R_refined
        refined_pose[:3, 3] = T_refined.flatten()

        return refined_pose, final_iter, converged

    # ------------------------------------------------------------------
    #  保存REF优化后的渲染图
    # ------------------------------------------------------------------
    def _save_refined_render(self, refined_pose: np.ndarray, coarse_pose: np.ndarray):
        """
        ★★★ 新增：渲染并保存REF优化前后的位姿渲染图 ★★★
        保存到 results/habitat_imagenav_日期_时间/REF/ 下
        """
        if not self.save_visualization_results:
            return
        import os
        import cv2
        from datetime import datetime

        try:
            # ★★★ 修复：优先使用 self.results_dir（由 localization_system 传入当前 episode 的目录）★★★
            # ★★★ 修复：不再使用 sorted(subdirs)[-1] 回退，防止多进程保存错乱 ★★★
            if self.results_dir is not None:
                save_dir = os.path.join(str(self.results_dir), "REF")
            else:
                # 使用进程ID创建隔离目录，避免多进程冲突
                pid = os.getpid()
                save_dir = os.path.join("output", f"pid_{pid}", "REF")
                self.logger.warning(
                    f"results_dir not set, falling back to process-isolated dir: {save_dir}. "
                    f"Call update_results_dir() to fix."
                )

            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 渲染优化后的位姿
            refined_image = self._render_single_pose(refined_pose)
            # 渲染优化前的位姿（用于对比）
            coarse_image = self._render_single_pose(coarse_pose)

            if refined_image is not None:
                # 保存优化后渲染图
                refined_bgr = cv2.cvtColor(refined_image, cv2.COLOR_RGB2BGR)
                refined_path = os.path.join(save_dir, f"ref_refined_{timestamp}.png")
                cv2.imwrite(refined_path, refined_bgr)

                c2w = np.linalg.inv(refined_pose)
                pos = c2w[:3, 3]

                # 如果有优化前的图，拼接对比图
                if coarse_image is not None:
                    coarse_bgr = cv2.cvtColor(coarse_image, cv2.COLOR_RGB2BGR)
                    c2w_c = np.linalg.inv(coarse_pose)
                    pos_c = c2w_c[:3, 3]

                    # 添加标注文字
                    coarse_labeled = coarse_bgr.copy()
                    refined_labeled = refined_bgr.copy()
                    cv2.putText(coarse_labeled, "BEFORE REF", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(coarse_labeled, f"pos=[{pos_c[0]:.2f},{pos_c[1]:.2f},{pos_c[2]:.2f}]", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(refined_labeled, "AFTER REF", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(refined_labeled, f"pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    comparison = np.hstack([coarse_labeled, refined_labeled])
                    comp_path = os.path.join(save_dir, f"ref_comparison_{timestamp}.png")
                    cv2.imwrite(comp_path, comparison)

                print(f"[REF] Saved refined render to: {save_dir}")
            else:
                print(f"[REF] Warning: Could not render refined pose for saving")

        except Exception as e:
            print(f"[REF] Warning: Failed to save refined render: {e}")

    def _render_single_pose(self, w2c_pose: np.ndarray):
        """
        渲染单个位姿的RGB图像（用于输出保存）

        Args:
            w2c_pose: 4x4 w2c矩阵

        Returns:
            numpy (H, W, 3) uint8 RGB，或 None
        """
        try:
            if not self._refinement_tools_available:
                # 尝试通过SLAM渲染
                return self._render_via_slam(w2c_pose)

            gaussian_model = self._get_gaussian_model()
            pipeline_params = self._get_pipeline_params()
            background = self._get_background()

            cp = self.camera_params
            projection_matrix = self._getProjectionMatrix2(
                znear=0.01, zfar=100.0,
                fx=cp['fx'], fy=cp['fy'],
                cx=cp['cx'], cy=cp['cy'],
                W=cp['width'], H=cp['height'],
            ).transpose(0, 1).to(device=self.device)

            # 创建一个最小的 dummy 图像（只需要shape对，不用于loss）
            dummy_image = np.zeros((cp['height'], cp['width'], 3), dtype=np.uint8)

            viewpoint = self._build_viewpoint(
                dummy_image, None, None, w2c_pose, projection_matrix
            )

            R = torch.tensor(w2c_pose[:3, :3], dtype=torch.float32)
            T = torch.tensor(w2c_pose[:3, 3], dtype=torch.float32)
            viewpoint.update_RT(R, T)

            with torch.no_grad():
                render_pkg = self._gs_render(
                    viewpoint, gaussian_model, pipeline_params, background
                )
                rendered = render_pkg["render"]  # (3, H, W) float tensor [0, 1]

            # 转为numpy uint8
            rendered_np = rendered.detach().cpu().permute(1, 2, 0).numpy()
            rendered_np = np.clip(rendered_np * 255, 0, 255).astype(np.uint8)
            return rendered_np

        except Exception as e:
            self.logger.warning(f"_render_single_pose failed with gs_render: {e}")
            return self._render_via_slam(w2c_pose)

    def _render_via_slam(self, w2c_pose: np.ndarray):
        """
        通过SLAM的render_frame接口渲染（备选方案）

        Args:
            w2c_pose: 4x4 w2c矩阵

        Returns:
            numpy (H, W, 3) uint8 RGB，或 None
        """
        try:
            slam = self.slam_mapper.slam
            cp = self.camera_params
            dummy_rgb = np.zeros((cp['height'], cp['width'], 3), dtype=np.uint8)
            dummy_depth = np.zeros((cp['height'], cp['width']), dtype=np.float32)

            cam = slam.create_camera(dummy_rgb, dummy_depth, w2c_pose, uid=999999)
            result = slam.render_frame(cam)
            if result is not None and 'rendered' in result:
                rendered = result['rendered']
                if isinstance(rendered, torch.Tensor):
                    rendered = rendered.detach().cpu().numpy()
                if rendered.ndim == 3 and rendered.shape[0] == 3:
                    rendered = rendered.transpose(1, 2, 0)
                if rendered.dtype != np.uint8:
                    rendered = np.clip(rendered * 255, 0, 255).astype(np.uint8)
                return rendered
        except Exception as e:
            self.logger.warning(f"_render_via_slam failed: {e}")
        return None

    def _get_gaussian_model(self):
        """从 SLAM 获取当前增量式建图的 3DGS 高斯模型"""
        slam = self.slam_mapper.slam

        # 尝试多种属性名以兼容不同 SLAM 实现
        if hasattr(slam, 'gaussians'):
            return slam.gaussians
        elif hasattr(slam, 'gaussian_model'):
            return slam.gaussian_model
        elif hasattr(slam, 'model'):
            return slam.model
        else:
            raise RuntimeError(
                "Cannot find gaussian model in SLAM. "
                "Expected attributes: gaussians, gaussian_model, or model"
            )

    def _get_pipeline_params(self):
        """获取渲染管线参数"""
        slam = self.slam_mapper.slam

        if hasattr(slam, 'pipeline_params'):
            return slam.pipeline_params
        elif hasattr(slam, 'pipe'):
            return slam.pipe
        else:
            # 使用默认管线参数
            from munch import munchify
            return munchify({
                'convert_SHs_python': False,
                'compute_cov3D_python': False,
                'debug': False,
            })

    def _get_background(self):
        """获取背景颜色张量"""
        return torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)

    def _build_viewpoint(self, target_image, depth_input, keypoints_input,
                         coarse_pose, projection_matrix):
        """
        构建 Camera viewpoint 对象

        Args:
            target_image: (H, W, 3) uint8 RGB
            depth_input: 目标深度图，可为 None
            keypoints_input: 目标特征点，可为 None
            coarse_pose: 4x4 w2c numpy
            projection_matrix: 投影矩阵 tensor
        """
        # 将目标图像转为 tensor: (3, H, W) float32 [0, 1]
        image_tensor = (
            torch.from_numpy(target_image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=torch.float32)
        )

        # 深度目标图像 → tensor
        depth_tensor = self._prepare_depth(depth_input)

        # ★★★ 新增：预处理特征点（缩放到当前分辨率）★★★
        # 注意：这里不直接用 keypoints_input，它在 _gradient_descent 中通过
        # _apply_keypoints_mask 使用。仅在此处做分辨率对齐。
        # keypoints_input 的处理在 _apply_keypoints_mask 中完成。

        # 位姿
        pose_tensor = torch.from_numpy(coarse_pose).to(device=self.device)

        # 构建 Camera
        cp = self.camera_params
        viewpoint = self._Camera(
            uid=0,
            color=image_tensor,
            depth=depth_tensor,
            gt_T=pose_tensor,
            projection_matrix=projection_matrix,
            fx=cp['fx'], fy=cp['fy'],
            cx=cp['cx'], cy=cp['cy'],
            fovx=self.fovx, fovy=self.fovy,
            image_height=cp['height'],
            image_width=cp['width'],
        )

        return viewpoint

    def _prepare_depth(self, depth_input):
        """
        将深度目标图像转换为 Camera 需要的 tensor 格式，
        并对齐到相机参数的分辨率
        
        Args:
            depth_input: (H, W) float32 numpy array（米），或 None
            
        Returns:
            (1, H, W) float32 tensor on device，或 None
        """
        if depth_input is None:
            return None

        import cv2

        target_h = self.camera_params['height']
        target_w = self.camera_params['width']

        # numpy 阶段做 resize（避免 torch interpolate 引入额外依赖）
        if not torch.is_tensor(depth_input):
            depth_np = depth_input.astype(np.float32)
            h, w = depth_np.shape[:2]
            if h != target_h or w != target_w:
                print(f"[REF] Resizing goal depth: ({h}, {w}) → ({target_h}, {target_w})")
                depth_np = cv2.resize(depth_np, (target_w, target_h),
                                      interpolation=cv2.INTER_NEAREST)
            depth_t = torch.from_numpy(depth_np)
        else:
            depth_t = depth_input.clone().float()
            h, w = depth_t.shape[-2], depth_t.shape[-1]
            if h != target_h or w != target_w:
                print(f"[REF] Resizing goal depth tensor: ({h}, {w}) → ({target_h}, {target_w})")
                if depth_t.ndim == 2:
                    depth_t = depth_t.unsqueeze(0).unsqueeze(0)
                elif depth_t.ndim == 3:
                    depth_t = depth_t.unsqueeze(0)
                depth_t = torch.nn.functional.interpolate(
                    depth_t, size=(target_h, target_w), mode='nearest'
                ).squeeze(0)

        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)  # (H, W) -> (1, H, W)

        return depth_t.to(device=self.device, dtype=torch.float32)

    def _build_refinement_config(self, has_depth: bool = False) -> dict:
        """构建 refinement 所需的 config 字典"""
        # 从 SLAM config 合并细定位参数
        config = {}
        if hasattr(self.slam_mapper, 'config') and self.slam_mapper.config is not None:
            config = copy.deepcopy(self.slam_mapper.config)

        # ★★★ 确保 Dataset 段存在（compute_grad_mask 需要 Dataset.type）★★★
        if 'Dataset' not in config:
            config['Dataset'] = {}
        if 'type' not in config['Dataset']:
            config['Dataset']['type'] = 'custom'  # 非 replica，走 else 分支

        # 覆盖/添加细定位特有参数
        if 'Training' not in config:
            config['Training'] = {}
        config['Training']['opacity_threshold'] = self.config.get('opacity_threshold', 0.99)
        config['Training']['edge_threshold'] = self.config.get('edge_threshold', 1.1)
        config['Training']['monocular'] = not has_depth
        config['Training']['alpha'] = config.get('Training', {}).get('alpha', 0.99)

        return config

    def _apply_keypoints_mask(self, viewpoint, keypoints_input):
        """
        将特征点应用到 viewpoint 的 grad_mask

        Args:
            viewpoint: Camera 对象
            keypoints_input: (N, 2) float32 numpy array，每行 (x, y)，
                             坐标可能基于原始图像分辨率，需缩放到相机分辨率
        """
        if keypoints_input is None:
            return

        try:
            cp = self.camera_params
            target_w = cp['width']
            target_h = cp['height']

            # ★★★ 新增：缩放关键点坐标到相机分辨率 ★★★
            keypoints = self._prepare_keypoints(keypoints_input, target_w, target_h)
            if keypoints is None or len(keypoints) == 0:
                return

            mask = self._create_keypoints_mask(
                keypoints,
                width=target_w,
                height=target_h,
                k=10
            )
            mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)

            # ★★★ 修复维度不匹配：将 keypoints mask resize 到 grad_mask 的尺寸 ★★★
            if mask_tensor.shape != viewpoint.grad_mask.shape:
                import torch.nn.functional as F
                # mask_tensor: (1, target_h, target_w)  grad_mask: (1, H', W')
                gm_shape = viewpoint.grad_mask.shape[-2:]  # (H', W')
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(0).float(),
                    size=gm_shape, mode='nearest'
                ).squeeze(0).bool()

            viewpoint.grad_mask = viewpoint.grad_mask | mask_tensor
            
            print(f"[REF] Applied keypoints mask: {len(keypoints)} points, "
                  f"mask coverage: {mask.sum()}/{target_h * target_w} pixels")
        except Exception as e:
            self.logger.warning(f"Failed to apply keypoints mask: {e}")

    def _prepare_keypoints(self, keypoints_input, target_w: int, target_h: int):
        """
        将 h5 中读取的关键点坐标缩放到相机参数的分辨率
        
        h5 中的 keypoints 已经是原始图像分辨率下的坐标。
        如果目标图像分辨率 != 相机分辨率，需要缩放。
        
        Args:
            keypoints_input: (N, 2) float32 ndarray，坐标 (x, y)
            target_w: 目标宽度（相机分辨率）
            target_h: 目标高度（相机分辨率）
            
        Returns:
            (N, 2) float32 ndarray，缩放后的坐标
        """
        if keypoints_input is None:
            return None

        kp = np.array(keypoints_input, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[1] != 2:
            self.logger.warning(f"Invalid keypoints shape: {kp.shape}, expected (N, 2)")
            return None

        if len(kp) == 0:
            return None

        # 推断原始图像分辨率（从关键点坐标范围）
        src_w = float(kp[:, 0].max()) + 1
        src_h = float(kp[:, 1].max()) + 1

        # 如果坐标范围远大于目标分辨率，说明需要缩放
        scale_x = target_w / max(src_w, 1.0)
        scale_y = target_h / max(src_h, 1.0)

        # 只在明显不匹配时缩放（允许 5% 误差）
        if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
            print(f"[REF] Scaling keypoints: src~({src_w:.0f}, {src_h:.0f}) → "
                  f"target ({target_w}, {target_h}), "
                  f"scale=({scale_x:.3f}, {scale_y:.3f})")
            kp[:, 0] *= scale_x
            kp[:, 1] *= scale_y

        # 裁剪到有效范围
        kp[:, 0] = np.clip(kp[:, 0], 0, target_w - 1)
        kp[:, 1] = np.clip(kp[:, 1], 0, target_h - 1)

        return kp

    @staticmethod
    def _create_keypoints_mask(keypoints, width, height, k=10):
        """
        从关键点坐标创建二值掩码

        Args:
            keypoints: (N, 2) ndarray, 每行 (x, y)
            width, height: 图像尺寸
            k: 掩码区域大小 k×k

        Returns:
            (1, H, W) bool ndarray
        """
        mask = np.zeros((height, width), dtype=bool)
        half_k = k // 2

        for pt in keypoints:
            x, y = int(pt[0]), int(pt[1])
            x_min = max(0, x - half_k)
            x_max = min(width, x + half_k + 1)
            y_min = max(0, y - half_k)
            y_max = min(height, y + half_k + 1)
            mask[y_min:y_max, x_min:x_max] = True

        return mask[np.newaxis, :, :]

    @staticmethod
    def _compute_pose_difference(pose1: np.ndarray, pose2: np.ndarray) -> Dict[str, float]:
        """
        计算两个位姿之间的差异

        Args:
            pose1, pose2: 4x4 w2c 矩阵

        Returns:
            {'translation': 平移差(米), 'rotation': 旋转差(度)}
        """
        R1, t1 = pose1[:3, :3], pose1[:3, 3]
        R2, t2 = pose2[:3, :3], pose2[:3, 3]

        # 平移差异：在世界坐标系下计算
        pos1 = -R1.T @ t1
        pos2 = -R2.T @ t2
        trans_diff = float(np.linalg.norm(pos1 - pos2))

        # 旋转差异
        cos_val = np.clip((np.trace(R1.T @ R2) - 1) / 2, -1.0, 1.0)
        rot_diff = float(np.rad2deg(np.abs(np.arccos(cos_val))))

        return {'translation': trans_diff, 'rotation': rot_diff}

    def reset(self):
        """重置细定位模块状态"""
        self.logger.info("PoseRefiner reset")
