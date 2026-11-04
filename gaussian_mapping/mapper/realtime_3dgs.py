import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from gaussian_mapping.gaussian_splatting.gaussian_renderer import render
from gaussian_mapping.gaussian_splatting.gaussmi_renderer import render_gaussmi
from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import GaussianModel
from gaussian_mapping.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2
from gaussian_mapping.utils.camera_utils import Camera
from gaussian_mapping.utils.config import to_namespace
from gaussian_mapping.utils.slam_utils import get_loss_mapping


logger = logging.getLogger(__name__)


class RealtimeSLAM:
    """Small online 3DGS mapper adapted from minav's RealtimeSLAM for IEVE.

    This class only owns RGB-D frame ingestion, Gaussian growth/training, rendering
    and checkpointing. IEVE's 2D map and planner remain unchanged.
    """

    def __init__(self, config: dict, camera_params: dict, output_dir, use_gui: bool = False,
                 enable_uaemi: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("3DGS mapping requires CUDA, but torch.cuda.is_available() is False")

        self.config = self._with_defaults(config)
        self.camera_params = camera_params
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gui = use_gui
        self.enable_uaemi = enable_uaemi

        self.fx = float(camera_params["fx"])
        self.fy = float(camera_params["fy"])
        self.cx = float(camera_params["cx"])
        self.cy = float(camera_params["cy"])
        self.width = int(camera_params["width"])
        self.height = int(camera_params["height"])
        self.fovx = float(camera_params.get("fovx", 2.0 * np.arctan(self.width / (2.0 * self.fx))))
        self.fovy = float(camera_params.get("fovy", 2.0 * np.arctan(self.height / (2.0 * self.fy))))

        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            cx=self.cx,
            cy=self.cy,
            fx=self.fx,
            fy=self.fy,
            W=self.width,
            H=self.height,
        ).transpose(0, 1).cuda()

        model_params = to_namespace(self.config.get("model_params", {}))
        opt_params = to_namespace(self.config.get("opt_params", {}))
        self.pipeline_params = to_namespace(self.config.get("pipeline_params", {}))

        self.gaussians = GaussianModel(int(getattr(model_params, "sh_degree", 1)), config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        training_cfg = self.config["Training"]
        self.semantic_guided_optimization_enabled = bool(
            training_cfg.get("semantic_guided_optimization_enabled", True)
        )
        self.init_itr_num = int(training_cfg.get("init_itr_num", 100))
        self.mapping_itr_num = int(training_cfg.get("mapping_itr_num", 20))
        self.training_window_size = max(
            1, int(training_cfg.get("window_size", 8))
        )
        self.phase1_current_frame_iters = max(
            0, int(training_cfg.get("phase1_current_frame_iters", 4))
        )
        self.phase1_recent_window_iters = max(
            0, int(training_cfg.get("phase1_recent_window_iters", 4))
        )
        self.phase1_target_replay_iters = max(
            0, int(training_cfg.get("phase1_target_replay_iters", 4))
        )
        self.phase1_other_replay_iters = max(
            0, int(training_cfg.get("phase1_other_replay_iters", 8))
        )
        self.buffer_pre_refresh_iters = int(training_cfg.get("buffer_pre_refresh_iters", 24))
        self.confirmation_pre_refresh_iters = int(
            training_cfg.get("confirmation_pre_refresh_iters", 32)
        )
        self.target_focus_window_size = int(training_cfg.get("target_focus_window_size", 12))
        dataset_cfg = self.config["Dataset"]
        self.buffer_phase2_pcd_downsample = float(
            training_cfg.get(
                "buffer_phase2_pcd_downsample",
                dataset_cfg.get(
                    "buffer_phase2_pcd_downsample",
                    dataset_cfg.get("pcd_downsample", 64.0),
                ),
            )
        )
        self.gaussian_update_every = int(training_cfg.get("gaussian_update_every", 20))
        self.gaussian_th = float(training_cfg.get("gaussian_th", 0.5))
        self.gaussian_extent = float(training_cfg.get("gaussian_extent", 1.0))
        self.size_threshold = int(training_cfg.get("size_threshold", 20))
        self.max_gaussians = int(training_cfg.get("max_gaussians", 1000000))
        self.cameras_extent = float(training_cfg.get("cameras_extent", 6.0))

        gaussmi_cfg = self.config["GauSS_MI"]
        self.gaussmi_touch_threshold = int(
            gaussmi_cfg.get("reli_touch_threshold", 5)
        )
        self.gaussmi_min_loss = float(
            gaussmi_cfg.get("reli_min_loss", 0.0001)
        )
        self.gaussmi_lambda_l = float(
            gaussmi_cfg.get("reli_lambda_L", 1.7)
        )
        self.gaussmi_lambda_t = float(
            gaussmi_cfg.get("reli_lambda_T", 10.0)
        )
        self.gaussmi_loss_rgb_alpha = float(
            gaussmi_cfg.get("loss_rgb_alpha", 0.95)
        )
        self._gaussmi_runtime_error = None

        self.frame_count = 0
        self.iteration_count = 0
        self.keyframes: Dict[int, Camera] = {}
        self.kf_indices = []
        self.kf_metadata: Dict[int, Dict[str, Any]] = {}
        # Opt-in diagnostic used to capture a scientifically matched SKGS
        # before/after pair.  It is environment-gated so normal benchmark runs
        # are bit-for-bit unaffected unless explicitly requested.
        self._skgs_pair_capture_enabled = str(
            os.environ.get("MULTI_IIN_SKGS_PAIR_CAPTURE", "0")
        ).lower() in {"1", "true", "yes", "on"}
        self._skgs_pair_capture_min_frame = int(
            os.environ.get("MULTI_IIN_SKGS_PAIR_MIN_FRAME", "0") or 0
        )
        self._skgs_pair_captured = False
        self._last_phase1_schedule_debug: Dict[str, Any] = {}

        logger.info(
            "Initialized IEVE 3DGS mapper: %sx%s, fx=%.2f, fy=%.2f, output=%s",
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.output_dir,
        )

    @staticmethod
    def _with_defaults(config: dict) -> dict:
        cfg = dict(config or {})
        cfg.setdefault("Dataset", {})
        cfg.setdefault("Training", {})
        cfg.setdefault("opt_params", {})
        cfg.setdefault("model_params", {})
        cfg.setdefault("pipeline_params", {})
        cfg.setdefault("GauSS_MI", {})

        cfg["Dataset"] = {
            "sensor_type": "rgbd",
            "pcd_downsample": 64.0,
            "pcd_downsample_init": 8.0,
            "buffer_phase2_pcd_downsample": 32.0,
            "adaptive_pointsize": True,
            "point_size": 0.05,
            **cfg["Dataset"],
        }
        cfg["Training"] = {
            "monocular": False,
            "alpha": 0.9,
            "rgb_boundary_threshold": 0.01,
            "semantic_guided_optimization_enabled": True,
            "init_itr_num": 100,
            "mapping_itr_num": 20,
            "window_size": 8,
            "phase1_current_frame_iters": 4,
            "phase1_recent_window_iters": 4,
            "phase1_target_replay_iters": 4,
            "phase1_other_replay_iters": 8,
            "buffer_pre_refresh_iters": 24,
            "confirmation_pre_refresh_iters": 32,
            "target_focus_window_size": 12,
            "target_loss_weight": 2.0,
            "protect_semantic_target_prune": True,
            "target_prune_protect_threshold": 0.7,
            "gaussian_update_every": 20,
            "gaussian_th": 0.5,
            "gaussian_extent": 1.0,
            "size_threshold": 20,
            **cfg["Training"],
        }
        cfg["opt_params"] = {
            "percent_dense": 0.01,
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.001,
            "rotation_lr": 0.001,
            **cfg["opt_params"],
        }
        cfg["model_params"] = {"sh_degree": 1, **cfg["model_params"]}
        cfg["pipeline_params"] = {
            "convert_SHs_python": False,
            "compute_cov3D_python": False,
            **cfg["pipeline_params"],
        }
        cfg["GauSS_MI"] = {
            "reli_touch_threshold": 5,
            "reli_min_loss": 0.0001,
            "reli_lambda_L": 1.7,
            "reli_lambda_T": 10.0,
            "loss_rgb_alpha": 0.95,
            **cfg["GauSS_MI"],
        }
        return cfg

    def add_frame(self, rgb, depth, position, rotation, target_mask=None,
                  region_type: str = "target", training_context: Optional[dict] = None):
        self.frame_count += 1
        training_context = self._normalize_training_context(
            training_context,
            region_type=region_type,
            fallback_step=self.frame_count,
            target_mask=target_mask,
        )
        pose = self._quat_to_matrix(position, rotation)
        cam = self.create_camera(rgb, depth, pose, self.frame_count)
        self.keyframes[self.frame_count] = cam
        self.kf_indices.append(self.frame_count)
        self._register_keyframe_metadata(self.frame_count, cam, training_context)
        if getattr(self, "semantic_guided_optimization_enabled", True):
            self._store_keyframe_target_loss_mask(self.frame_count, target_mask)

        allow_topology_growth = not self._is_buffer_phase1_stage(
            training_context.get("stage", "")
        )
        if allow_topology_growth:
            self._add_keyframe_gaussians(cam, training_context=training_context)
        else:
            logger.info(
                "3DGS buffer phase1 topology frozen for keyframe %s",
                self.frame_count,
            )
        if self.gaussians.get_xyz.shape[0] == 0:
            logger.warning("3DGS frame %s produced no Gaussians", self.frame_count)
            return True

        if target_mask is not None:
            try:
                updated = self.gaussians.update_target_semantic_from_mask(
                    np.asarray(target_mask, dtype=np.uint8),
                    cam,
                    depth=np.asarray(depth, dtype=np.float32),
                    update_mode="max",
                    depth_tolerance=float(
                        self.config.get("Training", {}).get(
                            "target_depth_tolerance", 0.35
                        )
                    ),
                )
                logger.info(
                    "3DGS semantic target update frame %s: gaussians=%s",
                    self.frame_count,
                    updated,
                )
            except Exception as exc:
                logger.warning(
                    "3DGS semantic target update failed at frame %s: %s",
                    self.frame_count,
                    exc,
                )

        self._train_mapping(
            initialization=len(self.kf_indices) == 1,
            allow_topology_growth=allow_topology_growth,
        )
        self._update_gaussmi_reliability(cam)
        return True

    def _gaussmi_loss_image(self, render_pkg, viewpoint):
        image = render_pkg["render"]
        depth = render_pkg["depth"]
        gt_image = viewpoint.original_image.cuda()
        gt_depth = torch.from_numpy(viewpoint.depth).to(
            dtype=torch.float32, device=image.device
        )[None]

        rgb_mask = (gt_image.sum(dim=0) > 0.01).view(*depth.shape)
        depth_mask = (gt_depth > 0.01).view(*depth.shape)
        l1_rgb = torch.abs(image * rgb_mask - gt_image * rgb_mask)
        l1_rgb = l1_rgb.mean(dim=0, keepdim=True)
        l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
        return (
            self.gaussmi_loss_rgb_alpha * l1_rgb
            + (1.0 - self.gaussmi_loss_rgb_alpha) * l1_depth
        )

    def _update_gaussmi_reliability(self, viewpoint):
        """Faithful GauSS-MI reliability update for the current keyframe."""

        if self.gaussians.get_xyz.shape[0] == 0:
            return False
        try:
            with torch.no_grad():
                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                )
                if render_pkg is None:
                    return False
                loss_image = (
                    self._gaussmi_loss_image(render_pkg, viewpoint).detach()
                    * self.gaussmi_lambda_l
                )
                loss_image[loss_image < self.gaussmi_min_loss] = (
                    self.gaussmi_min_loss
                )
                loss_image = -torch.log(loss_image)

                render_out = render_gaussmi(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    loss_image=loss_image,
                )
                if render_out is None:
                    return False
                logodds_zt = torch.clamp(
                    render_out["render_loss"].detach()
                    * self.gaussmi_lambda_t,
                    min=-100.0,
                    max=100.0,
                ).reshape(self.gaussians.get_xyz.shape[0], 4)
                n_touched = render_out["n_touched"].detach()
                logodds_zt[n_touched < self.gaussmi_touch_threshold] = 0.0
                self.gaussians.input_logodds(logodds_zt)
            self._gaussmi_runtime_error = None
            return True
        except Exception as exc:
            message = str(exc)
            if message != self._gaussmi_runtime_error:
                logger.warning("GauSS-MI reliability update unavailable: %s", exc)
                self._gaussmi_runtime_error = message
            return False

    @staticmethod
    def _is_buffer_phase1_stage(stage: str) -> bool:
        return str(stage or "") == "minav_buffer_phase1"

    @staticmethod
    def _is_buffer_phase2_stage(stage: str) -> bool:
        return str(stage or "") == "minav_buffer_phase2"

    def _normalize_training_context(
        self,
        training_context: Optional[dict],
        *,
        region_type: str = "target",
        fallback_step: int = 0,
        target_mask=None,
    ) -> Dict[str, Any]:
        ctx = dict(training_context or {})
        if not ctx.get("stage"):
            ctx["stage"] = "buffer" if region_type == "buffer" else "exploration"
        ctx.setdefault("mode_label", "")
        ctx.setdefault("has_target", target_mask is not None)
        ctx.setdefault("target_visible", bool(ctx.get("has_target", False)))
        if target_mask is not None and "target_pixel_count" not in ctx:
            try:
                ctx["target_pixel_count"] = int((np.asarray(target_mask) > 0).sum())
            except Exception:
                ctx["target_pixel_count"] = 0
        ctx.setdefault("target_confidence", 0.0)
        ctx.setdefault("target_instance_count", 1 if bool(ctx.get("has_target", False)) else 0)
        ctx.setdefault("is_replay_candidate", bool(ctx.get("has_target", False)))
        ctx.setdefault("current_object_id", None)
        ctx.setdefault("current_object_phase", None)
        ctx.setdefault("current_object_related", bool(ctx.get("has_target", False)))
        ctx.setdefault("step", int(fallback_step))
        return ctx

    def _register_keyframe_metadata(
        self,
        kf_id: int,
        viewpoint: Camera,
        training_context: Dict[str, Any],
    ) -> None:
        camera_center = None
        try:
            camera_center = viewpoint.camera_center.detach().cpu().numpy().copy()
        except Exception:
            pass
        self.kf_metadata[int(kf_id)] = {
            "kf_id": int(kf_id),
            "stage": training_context.get("stage", "exploration"),
            "mode_label": training_context.get("mode_label", ""),
            "has_target": bool(training_context.get("has_target", False)),
            "target_visible": bool(training_context.get("target_visible", False)),
            "target_confidence": float(training_context.get("target_confidence", 0.0) or 0.0),
            "target_instance_count": int(training_context.get("target_instance_count", 0) or 0),
            "target_pixel_count": int(training_context.get("target_pixel_count", 0) or 0),
            "is_replay_candidate": bool(training_context.get("is_replay_candidate", False)),
            "current_object_id": training_context.get("current_object_id"),
            "current_object_phase": training_context.get("current_object_phase"),
            "current_object_related": bool(training_context.get("current_object_related", False)),
            "step": int(training_context.get("step", kf_id) or kf_id),
            "camera_center": camera_center,
        }

    def _store_keyframe_target_loss_mask(self, kf_id: int, target_mask) -> int:
        meta = self.kf_metadata.get(int(kf_id))
        if meta is None or target_mask is None:
            return 0
        mask_np = np.asarray(target_mask, dtype=np.uint8)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        if mask_np.ndim != 2:
            return 0
        mask_np = np.ascontiguousarray((mask_np > 0).astype(np.uint8))
        pixel_count = int(mask_np.sum())
        if pixel_count <= 0:
            return 0
        meta["target_loss_mask"] = mask_np
        meta["target_loss_mask_pixels"] = pixel_count
        return pixel_count

    def create_camera(self, rgb, depth, pose, uid: int) -> Camera:
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
            depth_np = depth_np[..., 0]
        if rgb_np.shape[:2] != (self.height, self.width):
            rgb_np = cv2.resize(rgb_np, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if depth_np.shape[:2] != (self.height, self.width):
            depth_np = cv2.resize(depth_np, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        color = torch.from_numpy(rgb_np).float().permute(2, 0, 1).cuda() / 255.0
        gt_T = torch.from_numpy(np.asarray(pose, dtype=np.float32)).float().cuda()
        cam = Camera(
            uid=uid,
            color=color,
            depth=depth_np,
            gt_T=gt_T,
            projection_matrix=self.projection_matrix,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            fovx=self.fovx,
            fovy=self.fovy,
            image_height=self.height,
            image_width=self.width,
            device="cuda",
        )
        cam.update_RT(gt_T[:3, :3], gt_T[:3, 3])
        return cam

    @staticmethod
    def _quat_to_matrix(position, rotation) -> np.ndarray:
        from scipy.spatial.transform import Rotation as R

        rot_matrix = R.from_quat(np.asarray(rotation, dtype=np.float32)).as_matrix()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rot_matrix
        c2w[:3, 3] = np.asarray(position, dtype=np.float32)
        flip_mat = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        return np.linalg.inv(c2w @ flip_mat).astype(np.float32)

    def _add_keyframe_gaussians(
        self, viewpoint: Camera, training_context: Optional[dict] = None
    ):
        import open3d as o3d

        num_before = int(self.gaussians.get_xyz.shape[0])
        max_new_points = max(0, self.max_gaussians - num_before)
        if max_new_points <= 0:
            return

        color_np = (
            viewpoint.original_image.detach().cpu().numpy().transpose(1, 2, 0) * 255.0
        ).astype(np.uint8)
        rgb_o3d = o3d.geometry.Image(color_np)
        depth_o3d = o3d.geometry.Image(np.asarray(viewpoint.depth, dtype=np.float32))
        is_init = len(self.kf_indices) == 1
        stage = ""
        if training_context:
            stage = str(training_context.get("stage", "") or "")
        original_downsample = self.config["Dataset"].get("pcd_downsample", 64.0)
        use_phase2_downsample = (
            not is_init
            and self._is_buffer_phase2_stage(stage)
            and self.buffer_phase2_pcd_downsample > 0
        )

        try:
            if use_phase2_downsample:
                self.config["Dataset"]["pcd_downsample"] = min(
                    float(original_downsample),
                    float(self.buffer_phase2_pcd_downsample),
                )
            fused_point_cloud, features, scales, rots, opacities = (
                self.gaussians.create_pcd_from_image_and_depth(
                    viewpoint,
                    rgb_o3d,
                    depth_o3d,
                    init=is_init,
                )
            )
        finally:
            self.config["Dataset"]["pcd_downsample"] = original_downsample

        num_candidates = int(fused_point_cloud.shape[0])
        if num_candidates == 0:
            return
        if num_candidates > max_new_points:
            indices = torch.randperm(num_candidates, device=fused_point_cloud.device)[:max_new_points]
            fused_point_cloud = fused_point_cloud[indices]
            features = features[indices]
            scales = scales[indices]
            rots = rots[indices]
            opacities = opacities[indices]

        self.gaussians.extend_from_pcd(
            fused_point_cloud,
            features,
            scales,
            rots,
            opacities,
            kf_id=viewpoint.uid,
        )
        logger.info(
            "3DGS added keyframe %s: gaussians=%s new=%s",
            viewpoint.uid,
            int(self.gaussians.get_xyz.shape[0]),
            int(self.gaussians.get_xyz.shape[0]) - num_before,
        )

    def _mapping_keyframe_pool(self) -> List[int]:
        pool = [int(kf_id) for kf_id in self.kf_indices]
        if not getattr(self, "semantic_guided_optimization_enabled", True):
            pool = pool[-self.training_window_size :]
        return pool

    def _keyframe_target_loss_mask(self, kf_id: int):
        if not getattr(self, "semantic_guided_optimization_enabled", True):
            return None
        return self.kf_metadata.get(int(kf_id), {}).get("target_loss_mask")

    def _train_mapping(
        self,
        initialization: bool = False,
        override_iterations: Optional[int] = None,
        preferred_kf_ids: Optional[List[int]] = None,
        allow_topology_growth: bool = True,
    ):
        iterations = (
            int(override_iterations)
            if override_iterations is not None
            else self.init_itr_num if initialization else self.mapping_itr_num
        )
        if not self.kf_indices or self.gaussians.get_xyz.shape[0] == 0:
            return
        kf_pool = self._mapping_keyframe_pool()
        if preferred_kf_ids:
            preferred = [int(k) for k in preferred_kf_ids if int(k) in self.keyframes]
            if preferred:
                kf_pool = preferred
        phase1_schedule = None
        if (
            not initialization
            and override_iterations is None
            and not preferred_kf_ids
            and getattr(self, "semantic_guided_optimization_enabled", True)
            and self._latest_keyframe_is_phase1_exploration()
        ):
            phase1_schedule = self._phase1_training_schedule(iterations)
        capture_pair = bool(
            phase1_schedule is not None
            and self._should_capture_skgs_pair()
        )
        if capture_pair:
            self._save_skgs_pair_snapshot("before")
        with torch.enable_grad():
            for iteration in range(iterations):
                if initialization:
                    kf_id = self.kf_indices[-1]
                elif phase1_schedule is not None:
                    kf_id = phase1_schedule[iteration]
                else:
                    kf_id = int(np.random.choice(kf_pool))
                viewpoint = self.keyframes[kf_id]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                if render_pkg is None:
                    continue

                image = render_pkg["render"]
                depth = render_pkg["depth"]
                opacity = render_pkg["opacity"]
                visibility_filter = render_pkg["visibility_filter"]
                viewspace_point_tensor = render_pkg["viewspace_points"]

                target_loss_mask = self._keyframe_target_loss_mask(kf_id)
                loss = get_loss_mapping(
                    self.config,
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    initialization,
                    target_mask=target_loss_mask,
                )
                scaling = self.gaussians.get_scaling
                if scaling.shape[0] > 0 and scaling.shape[1] > 1:
                    isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
                    loss = loss + 10.0 * isotropic_loss.mean()
                loss.backward()

                with torch.no_grad():
                    if visibility_filter.shape[0] == self.gaussians.max_radii2D.shape[0]:
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter],
                            render_pkg["radii"][visibility_filter],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor,
                            visibility_filter,
                        )

                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.iteration_count += 1
                    self.gaussians.update_learning_rate(self.iteration_count)

                    if (
                        iteration > 0
                        and allow_topology_growth
                        and self.gaussian_update_every > 0
                        and iteration % self.gaussian_update_every == 0
                        and self.gaussians.get_xyz.shape[0] < self.max_gaussians
                    ):
                        self.gaussians.densify_and_prune(
                            0.0002,
                            self.gaussian_th,
                            self.cameras_extent * self.gaussian_extent,
                            self.size_threshold,
                        )
        if capture_pair:
            self._save_skgs_pair_snapshot("after")
            self._skgs_pair_captured = True
            metadata_path = self._skgs_pair_output_dir() / "metadata.json"
            metadata = dict(self._last_phase1_schedule_debug)
            metadata.update(
                {
                    "capture_kind": "real_skgs_before_after",
                    "capture_min_frame": int(self._skgs_pair_capture_min_frame),
                    "mapping_iterations": int(iterations),
                    "note": (
                        "Both snapshots use the same online run and bracket the "
                        "actual semantic-guided mapping schedule."
                    ),
                }
            )
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
            )
            logger.info("Saved real SKGS before/after pair to %s", metadata_path.parent)

    def _should_capture_skgs_pair(self) -> bool:
        debug = getattr(self, "_last_phase1_schedule_debug", {}) or {}
        return bool(
            self._skgs_pair_capture_enabled
            and not self._skgs_pair_captured
            and int(self.frame_count) >= int(self._skgs_pair_capture_min_frame)
            and debug.get("target_pool")
            and debug.get("target_replay_ids")
        )

    def _skgs_pair_output_dir(self) -> Path:
        path = self.output_dir / "skgs_real_pair_frame_{:06d}".format(
            int(self.frame_count)
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _rgb_tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
        rgb = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
        if rgb.ndim == 3 and rgb.shape[0] in (1, 3):
            rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _save_semantic_instance_ply(self, path: Path) -> None:
        """Write RGB plus the currently available semantic/instance sidecars."""

        from plyfile import PlyData, PlyElement

        gaussians = self.gaussians
        xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        count = int(xyz.shape[0])
        colors = gaussians._base_rgb_uint8()
        semantic = np.full((count,), -1, dtype=np.int16)
        confidence = np.zeros((count,), dtype=np.float32)
        instance = np.full((count,), -1, dtype=np.int32)
        observations = np.zeros((count,), dtype=np.uint16)
        evidence = getattr(gaussians, "_multi_semantic_evidence", None)
        observed = getattr(gaussians, "_multi_semantic_observations", None)
        instance_ids = getattr(gaussians, "_multi_instance_ids", None)
        if torch.is_tensor(evidence) and torch.is_tensor(observed):
            keep = min(count, int(evidence.shape[0]), int(observed.shape[0]))
            if keep > 0:
                ev = evidence[:keep].detach().float().cpu()
                obs = observed[:keep].detach().cpu().numpy()
                totals = ev.sum(dim=1)
                best_value, best_class = ev.max(dim=1)
                valid = (totals > 0.0).numpy() & (obs > 0)
                labels = best_class.numpy().astype(np.int16)
                labels[~valid] = -1
                semantic[:keep] = labels
                confidence[:keep] = np.where(
                    valid,
                    best_value.numpy() / np.maximum(totals.numpy(), 1e-8),
                    0.0,
                )
                observations[:keep] = np.clip(obs, 0, 65535).astype(np.uint16)
        if torch.is_tensor(instance_ids):
            keep = min(count, int(instance_ids.shape[0]))
            instance[:keep] = instance_ids[:keep].detach().cpu().numpy().astype(np.int32)

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("semantic_class", "i2"), ("semantic_confidence", "f4"),
            ("instance_id", "i4"), ("label_observations", "u2"),
        ]
        vertices = np.empty((count,), dtype=dtype)
        for axis, name in enumerate(("x", "y", "z")):
            vertices[name] = xyz[:, axis]
        for channel, name in enumerate(("red", "green", "blue")):
            vertices[name] = colors[:, channel]
        vertices["semantic_class"] = semantic
        vertices["semantic_confidence"] = confidence
        vertices["instance_id"] = instance
        vertices["label_observations"] = observations
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))

    def _save_skgs_pair_snapshot(self, label: str) -> None:
        output_dir = self._skgs_pair_output_dir()
        self.gaussians.save_ply(
            str(output_dir / "{}_full_gaussians.ply".format(label)),
            include_color=True,
        )
        self._save_semantic_instance_ply(
            output_dir / "{}_semantic_instance_gaussians.ply".format(label)
        )

        debug = self._last_phase1_schedule_debug
        replay_ids = [int(value) for value in debug.get("target_replay_ids", [])]
        if not replay_ids:
            return
        view_id = replay_ids[0]
        viewpoint = self.keyframes.get(view_id)
        if viewpoint is None:
            return
        with torch.no_grad():
            rendered = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )["render"]
        render_rgb = self._rgb_tensor_to_uint8(rendered)
        gt_rgb = self._rgb_tensor_to_uint8(viewpoint.original_image)
        cv2.imwrite(
            str(output_dir / "{}_target_replay_render.png".format(label)),
            cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR),
        )
        gt_path = output_dir / "target_replay_ground_truth.png"
        if not gt_path.exists():
            cv2.imwrite(str(gt_path), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
        mse = float(np.mean((render_rgb.astype(np.float32) - gt_rgb.astype(np.float32)) ** 2))
        psnr = float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-12)))
        target_mask = self.kf_metadata.get(view_id, {}).get("target_loss_mask")
        target_mse = None
        target_psnr = None
        target_bbox = None
        if target_mask is not None:
            mask = np.asarray(target_mask, dtype=np.uint8) > 0
            if mask.shape != render_rgb.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (render_rgb.shape[1], render_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            if bool(mask.any()):
                squared_error = (
                    render_rgb.astype(np.float32) - gt_rgb.astype(np.float32)
                ) ** 2
                target_mse = float(squared_error[mask].mean())
                target_psnr = float(
                    10.0 * np.log10((255.0 ** 2) / max(target_mse, 1e-12))
                )
                ys, xs = np.nonzero(mask)
                padding = 24
                x0 = max(0, int(xs.min()) - padding)
                y0 = max(0, int(ys.min()) - padding)
                x1 = min(render_rgb.shape[1], int(xs.max()) + padding + 1)
                y1 = min(render_rgb.shape[0], int(ys.max()) + padding + 1)
                target_bbox = [x0, y0, x1, y1]
                cv2.imwrite(
                    str(output_dir / "target_replay_mask.png"),
                    mask.astype(np.uint8) * 255,
                )
                cv2.imwrite(
                    str(output_dir / "{}_target_crop.png".format(label)),
                    cv2.cvtColor(render_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
                )
                crop_gt_path = output_dir / "target_replay_ground_truth_crop.png"
                if not crop_gt_path.exists():
                    cv2.imwrite(
                        str(crop_gt_path),
                        cv2.cvtColor(gt_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR),
                    )
        (output_dir / "{}_render_metrics.json".format(label)).write_text(
            json.dumps(
                {
                    "view_id": view_id,
                    "mse_uint8": mse,
                    "psnr_db": psnr,
                    "target_mse_uint8": target_mse,
                    "target_psnr_db": target_psnr,
                    "target_bbox_xyxy": target_bbox,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _latest_keyframe_is_phase1_exploration(self) -> bool:
        if not self.kf_indices:
            return False
        latest_meta = self.kf_metadata.get(int(self.kf_indices[-1]), {})
        return str(latest_meta.get("stage", "")) in {
            "exploration",
            "ieve_online",
            "minav_exploration",
        }

    @staticmethod
    def _sample_keyframes(pool, count):
        pool = [int(kf_id) for kf_id in pool]
        count = max(0, int(count))
        if count == 0 or not pool:
            return []
        return [int(np.random.choice(pool)) for _ in range(count)]

    def _phase1_training_schedule(self, iterations: int) -> List[int]:
        """Build the 4/4/4/8 Phase-1 replay schedule."""

        iterations = max(0, int(iterations))
        if iterations == 0 or not self.kf_indices:
            return []

        all_ids = [int(kf_id) for kf_id in self.kf_indices]
        current_id = all_ids[-1]
        recent_ids = all_ids[-self.training_window_size :]
        recent_set = set(recent_ids)
        recent_pool = [kf_id for kf_id in recent_ids if kf_id != current_id]

        target_ids = {
            int(kf_id)
            for kf_id, meta in self.kf_metadata.items()
            if int(
                meta.get(
                    "target_loss_mask_pixels",
                    meta.get("target_pixel_count", 0),
                )
                or 0
            )
            > 0
        }
        historical_targets = []
        for kf_id in all_ids:
            if kf_id in recent_set or kf_id not in target_ids:
                continue
            meta = self.kf_metadata.get(kf_id, {})
            historical_targets.append(
                (
                    int(
                        meta.get(
                            "target_loss_mask_pixels",
                            meta.get("target_pixel_count", 0),
                        )
                        or 0
                    ),
                    float(meta.get("target_confidence", 0.0) or 0.0),
                    kf_id,
                )
            )
        historical_targets.sort(reverse=True)
        target_pool = [
            item[2]
            for item in historical_targets[: self.target_focus_window_size]
        ]
        ordinary_pool = [
            kf_id
            for kf_id in all_ids
            if kf_id not in recent_set and kf_id not in target_ids
        ]
        historical_pool = [kf_id for kf_id in all_ids if kf_id != current_id]

        schedule = [current_id] * self.phase1_current_frame_iters
        schedule.extend(
            self._sample_keyframes(
                recent_pool or [current_id],
                self.phase1_recent_window_iters,
            )
        )
        sampled_target_replay = self._sample_keyframes(
            target_pool
            or ordinary_pool
            or recent_pool
            or [current_id],
            self.phase1_target_replay_iters,
        )
        schedule.extend(sampled_target_replay)
        schedule.extend(
            self._sample_keyframes(
                ordinary_pool
                or historical_pool
                or [current_id],
                self.phase1_other_replay_iters,
            )
        )

        if len(schedule) < iterations:
            schedule.extend(
                self._sample_keyframes(
                    ordinary_pool
                    or historical_pool
                    or [current_id],
                    iterations - len(schedule),
                )
            )
        schedule = schedule[:iterations]
        # Preserve the target-history selections before shuffling the complete
        # schedule, then record the actual execution order as well.
        self._last_phase1_schedule_debug = {
            "frame": int(self.frame_count),
            "current_id": int(current_id),
            "recent_pool": [int(value) for value in recent_pool],
            "target_pool": [int(value) for value in target_pool],
            "ordinary_pool": [int(value) for value in ordinary_pool],
            "target_replay_ids": [int(value) for value in sampled_target_replay],
        }
        np.random.shuffle(schedule)
        self._last_phase1_schedule_debug["schedule"] = [int(value) for value in schedule]
        return schedule

    def refresh_training_focus(
        self,
        object_id: Optional[int] = None,
        reason: str = "",
        extra_iters: Optional[int] = None,
        preferred_kf_ids: Optional[List[int]] = None,
        stage: Optional[str] = None,
    ):
        if not getattr(self, "semantic_guided_optimization_enabled", True):
            logger.info(
                "3DGS semantic-guided refresh disabled: stage=%s reason=%s",
                stage,
                reason,
            )
            return None
        if not self.kf_indices or self.gaussians.get_xyz.shape[0] == 0:
            return None
        if extra_iters is None:
            extra_iters = (
                self.confirmation_pre_refresh_iters
                if stage == "target_confirm"
                else self.buffer_pre_refresh_iters
            )
        extra_iters = int(extra_iters or 0)
        if extra_iters <= 0:
            return None
        if not preferred_kf_ids and stage in {
            "target_confirm",
            "prebuffer",
            "prelocalization",
        }:
            preferred_kf_ids = self._target_focus_preferred_keyframes(
                limit=self.target_focus_window_size
            )
        start = time.time()
        self._train_mapping(
            initialization=False,
            override_iterations=extra_iters,
            preferred_kf_ids=preferred_kf_ids,
            allow_topology_growth=True,
        )
        result = {
            "stage": stage or "prebuffer",
            "iters": extra_iters,
            "preferred_kf_ids": list(preferred_kf_ids or []),
            "elapsed_s": float(time.time() - start),
            "reason": reason,
        }
        logger.info("3DGS refresh_training_focus: %s", result)
        return result

    def _target_focus_preferred_keyframes(self, limit: int = 12) -> List[int]:
        ranked = []
        for kf_id, meta in self.kf_metadata.items():
            if not meta.get("has_target", False):
                continue
            pixels = int(
                meta.get(
                    "target_loss_mask_pixels",
                    meta.get("target_pixel_count", 0),
                )
                or 0
            )
            if pixels <= 0:
                continue
            ranked.append(
                (
                    pixels,
                    float(meta.get("target_confidence", 0.0) or 0.0),
                    int(kf_id),
                )
            )
        ranked.sort(reverse=True)
        return [int(item[2]) for item in ranked[: max(1, int(limit))]]

    def render_frame(self, viewpoint_or_idx):
        if isinstance(viewpoint_or_idx, Camera):
            viewpoint = viewpoint_or_idx
        else:
            viewpoint = self.keyframes.get(int(viewpoint_or_idx))
        if viewpoint is None or self.gaussians.get_xyz.shape[0] == 0:
            return None
        with torch.no_grad():
            render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        if render_pkg is None:
            return None
        image = render_pkg["render"].detach().clamp(0.0, 1.0).cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        return {"rendered": (image * 255.0).astype(np.uint8), "render_pkg": render_pkg}

    def get_next_best_view_gaussmi_semantic(
        self,
        current_position,
        current_rotation,
        target_region_2d: dict = None,
        occupancy_map=None,
        explored_map=None,
        map_resolution=0.05,
        batch_size=32,
        search_radius_m: float = 1.5,
        num_clusters: int = 8,
        num_yaw_samples: int = 8,
        use_multiview_fusion: bool = True,
        hint_target_world_pos=None,
    ):
        """Semantic GauSS-MI NBV using IEVE's 3DGS visibility.

        The candidate goal is still filtered by IEVE's FMM in the caller. This
        method only ranks candidate pose/yaw pairs by visible target semantic
        Gaussian information gain instead of distance.
        """
        del use_multiview_fusion

        target_mask = self._target_semantic_mask_tensor()
        if target_mask is None or not bool(target_mask.any().item()):
            logger.warning("GauSS-MI semantic NBV unavailable: no target semantic gaussians")
            return None

        target_world = self._target_centroid_from_mask(target_mask)
        if hint_target_world_pos is not None:
            hint = np.asarray(hint_target_world_pos, dtype=np.float32).reshape(-1)
            if hint.size >= 3 and np.all(np.isfinite(hint[:3])):
                target_world = hint[:3].astype(np.float32)
        if target_world is None:
            return None

        current = np.asarray(current_position, dtype=np.float32).reshape(-1)
        if current.size < 3 or not np.all(np.isfinite(current[:3])):
            return None

        candidates = []
        if target_region_2d is not None and occupancy_map is not None:
            candidates = self._generate_candidate_views_around_target_semantic(
                current_position=current[:3],
                target_region_2d=target_region_2d,
                occupancy_map=occupancy_map,
                explored_map=explored_map,
                map_resolution=map_resolution,
                search_radius_m=search_radius_m,
                num_clusters=num_clusters,
                num_yaw_samples=num_yaw_samples,
                min_distance_m=0.2,
                hint_yaw_target_world_pos=target_world,
            )
        if not candidates:
            candidates = self._generate_semantic_gaussmi_candidates(
                current[:3],
                target_world,
                search_radius_m=search_radius_m,
                num_positions=num_clusters,
                num_yaws=num_yaw_samples,
            )
        if not candidates:
            return None

        positions = np.asarray([c["position"] for c in candidates], dtype=np.float32)
        yaws = np.asarray([c["yaw"] for c in candidates], dtype=np.float32)
        scores = self.compute_mutual_info_score_semantic_batch(
            positions,
            yaws,
            target_mask=target_mask,
            batch_size=batch_size,
        )
        for candidate, score in zip(candidates, scores):
            candidate["score"] = float(score)
            candidate["mutual_info"] = float(score)
            candidate["num_target_gaussians"] = int(target_mask.sum().item())
            candidate["method"] = "IEVE-GaussMI-Semantic"

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        if not ranked or ranked[0]["score"] <= 0.0:
            logger.warning("GauSS-MI semantic NBV produced no positive-information view")
            return None

        result = dict(ranked[0])
        result["all_ranked_candidates"] = ranked
        return result

    def score_phase1_gaussmi_yaws(
        self,
        current_position,
        current_rotation,
        yaw_offsets_deg,
        *,
        angle_penalty_k: float = 0.5,
        theta_max_deg: float = 90.0,
        target_weight: float = 2.0,
        nearby_weight: float = 1.5,
        nearby_radius_m: float = 2.0,
        min_frames: int = 5,
        min_gaussians: int = 10000,
        reliability_delta: float = 0.02,
        min_updated_ratio: float = 0.10,
        min_valid_candidates: int = 4,
        min_info_gain: float = 0.0,
        min_relative_gain: float = 0.0,
    ):
        """Rank fixed-position Phase-1 headings with original GauSS-MI."""

        num_gaussians = int(self.gaussians.get_xyz.shape[0])
        offsets = np.asarray(yaw_offsets_deg, dtype=np.float32).reshape(-1)
        if self.frame_count < int(min_frames) or num_gaussians < int(min_gaussians):
            return {
                "ready": False,
                "reason": "map_size",
                "frame_count": int(self.frame_count),
                "num_gaussians": num_gaussians,
                "candidates": [],
            }

        reliability = self.gaussians.get_reliability.detach()
        if reliability.shape != (num_gaussians, 4):
            return {"ready": False, "reason": "reliability_shape", "candidates": []}
        updated_ratio = float(
            (torch.abs(reliability - float(self.gaussians.init_reliability))
             > float(reliability_delta))
            .any(dim=1)
            .float()
            .mean()
            .item()
        )
        if updated_ratio < float(min_updated_ratio):
            return {
                "ready": False,
                "reason": "reliability_not_discriminative",
                "updated_reliability_ratio": updated_ratio,
                "candidates": [],
            }

        position = np.asarray(current_position, dtype=np.float32).reshape(-1)
        rotation = np.asarray(current_rotation, dtype=np.float32).reshape(-1)
        if position.size < 3 or rotation.size < 4:
            return {"ready": False, "reason": "invalid_pose", "candidates": []}

        semantic_weights, semantic_debug = self._phase1_gaussmi_semantic_weights(
            target_weight=float(target_weight),
            nearby_weight=float(nearby_weight),
            nearby_radius_m=float(nearby_radius_m),
        )
        # Preserve the original information-gain expression. IEVE semantics
        # only multiply its result before the original alpha*T rasterization.
        information_gain = self.gaussians.get_infogain.detach()
        information_gain = information_gain * semantic_weights[:, None]

        dummy_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        dummy_depth = np.zeros((self.height, self.width), dtype=np.float32)
        candidates = []
        theta_max = max(abs(float(theta_max_deg)), 1e-6)
        with torch.no_grad():
            for offset in offsets:
                candidate = {"offset_deg": float(offset), "valid": False}
                try:
                    candidate_rotation = self._relative_habitat_yaw_quaternion(
                        rotation[:4], float(offset)
                    )
                    pose = self._quat_to_matrix(position[:3], candidate_rotation)
                    virtual_cam = self.create_camera(
                        dummy_rgb, dummy_depth, pose, uid=-1
                    )
                    render_pkg = render_gaussmi(
                        virtual_cam,
                        self.gaussians,
                        self.pipeline_params,
                        self.background,
                        information_gain=information_gain,
                    )
                    if render_pkg is None:
                        candidate["reason"] = "empty_render"
                    else:
                        mutual_info = render_pkg["mutual_info"][0]
                        # This is intentionally identical to GauSS-MI's
                        # compute_GauSS_MI aggregation.
                        raw_score = int(float(mutual_info.sum().item())) / 1000.0
                        penalty = 1.0 - float(angle_penalty_k) * (
                            abs(float(offset)) / theta_max
                        ) ** 2
                        final_score = raw_score * penalty
                        candidate.update(
                            {
                                "valid": bool(
                                    np.isfinite(raw_score)
                                    and np.isfinite(final_score)
                                ),
                                "info_gain": float(raw_score),
                                "angle_penalty": float(penalty),
                                "final_score": float(final_score),
                                "visible_gaussians": int(
                                    render_pkg["visibility_filter"].sum().item()
                                ),
                            }
                        )
                except Exception as exc:
                    candidate["reason"] = str(exc)
                candidates.append(candidate)

        valid = [candidate for candidate in candidates if candidate.get("valid")]
        if len(valid) < int(min_valid_candidates):
            return {
                "ready": False,
                "reason": "insufficient_valid_candidates",
                "valid_candidate_count": len(valid),
                "updated_reliability_ratio": updated_ratio,
                "candidates": candidates,
                "semantic": semantic_debug,
            }

        best = max(valid, key=lambda item: float(item["final_score"]))
        zero = min(valid, key=lambda item: abs(float(item["offset_deg"])))
        skip_reason = None
        if float(best["info_gain"]) <= max(0.0, float(min_info_gain)):
            skip_reason = "low_absolute_information_gain"
        elif (
            abs(float(best["offset_deg"])) > 1e-6
            and float(best["final_score"])
            <= float(zero["final_score"])
            * (1.0 + max(0.0, float(min_relative_gain)))
        ):
            skip_reason = "insufficient_gain_over_zero"

        return {
            "ready": True,
            "reason": skip_reason,
            "should_turn": skip_reason is None and abs(float(best["offset_deg"])) > 1e-6,
            "selected_offset_deg": float(best["offset_deg"]),
            "best_info_gain": float(best["info_gain"]),
            "best_final_score": float(best["final_score"]),
            "zero_final_score": float(zero["final_score"]),
            "updated_reliability_ratio": updated_ratio,
            "frame_count": int(self.frame_count),
            "num_gaussians": num_gaussians,
            "valid_candidate_count": len(valid),
            "semantic": semantic_debug,
            "candidates": candidates,
        }

    def _phase1_gaussmi_semantic_weights(
        self,
        *,
        target_weight: float,
        nearby_weight: float,
        nearby_radius_m: float,
    ):
        xyz = self.gaussians.get_xyz.detach()
        weights = torch.ones(
            xyz.shape[0], dtype=torch.float32, device=xyz.device
        )
        target_mask = self._target_semantic_mask_tensor()
        if target_mask is None or not bool(target_mask.any().item()):
            return weights, {
                "target_count": 0,
                "nearby_count": 0,
                "cluster_count": 0,
            }

        centers = []
        try:
            clusters = self.gaussians.cluster_target_gaussians(
                threshold=0.4, eps=0.15, min_samples=10
            )
            centers = list(clusters.get("cluster_centers", []) or [])
        except Exception:
            centers = []
        if not centers:
            centers = [
                xyz[target_mask].mean(dim=0).detach().cpu().numpy()
            ]

        centers_tensor = torch.as_tensor(
            np.asarray(centers, dtype=np.float32),
            dtype=torch.float32,
            device=xyz.device,
        ).reshape(-1, 3)
        # Habitat is y-up; proximity is horizontal x-z distance as specified.
        horizontal_delta = (
            xyz[:, None, :][:, :, [0, 2]]
            - centers_tensor[None, :, :][:, :, [0, 2]]
        )
        nearby_mask = (
            torch.linalg.norm(horizontal_delta, dim=2).amin(dim=1)
            <= float(nearby_radius_m)
        ) & (~target_mask)
        weights[nearby_mask] = max(1.0, float(nearby_weight))
        weights[target_mask] = max(1.0, float(target_weight))
        return weights, {
            "target_count": int(target_mask.sum().item()),
            "nearby_count": int(nearby_mask.sum().item()),
            "cluster_count": int(centers_tensor.shape[0]),
        }

    @staticmethod
    def _relative_habitat_yaw_quaternion(rotation_xyzw, offset_deg):
        from scipy.spatial.transform import Rotation as R

        current = R.from_quat(np.asarray(rotation_xyzw, dtype=np.float64))
        # IEVE's planner yaw and Habitat's world-y quaternion both increase
        # for a left turn; apply the relative heading in the world frame.
        delta = R.from_rotvec(
            np.asarray([0.0, np.deg2rad(float(offset_deg)), 0.0])
        )
        return (delta * current).as_quat().astype(np.float32)

    def _target_semantic_mask_tensor(self):
        if self.gaussians.get_xyz.shape[0] == 0:
            return None
        semantic = getattr(self.gaussians, "get_semantic_target", None)
        if callable(semantic):
            semantic = semantic()
        if semantic is None or not torch.is_tensor(semantic) or semantic.numel() == 0:
            return None
        if semantic.shape[0] != self.gaussians.get_xyz.shape[0]:
            return None
        return (semantic > 0.4).to(device=self.gaussians.get_xyz.device, dtype=torch.bool)

    def _target_centroid_from_mask(self, target_mask):
        try:
            xyz = self.gaussians.get_xyz[target_mask]
            if xyz.shape[0] == 0:
                return None
            return xyz.detach().mean(dim=0).cpu().numpy().astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _generate_semantic_gaussmi_candidates(
        current: np.ndarray,
        target: np.ndarray,
        search_radius_m: float,
        num_positions: int,
        num_yaws: int,
    ):
        current = np.asarray(current, dtype=np.float32).reshape(-1)[:3]
        target = np.asarray(target, dtype=np.float32).reshape(-1)[:3]
        radius = max(0.4, float(search_radius_m))
        num_positions = max(4, int(num_positions))
        num_yaws = max(1, int(num_yaws))

        base_angle = float(np.arctan2(current[0] - target[0], current[2] - target[2]))
        yaw_offsets = np.linspace(-np.pi / 4.0, np.pi / 4.0, num_yaws, dtype=np.float32)
        candidates = []
        for cluster_id in range(num_positions):
            angle = base_angle + (2.0 * np.pi * cluster_id / num_positions)
            pos = target.copy()
            pos[0] = target[0] + np.sin(angle) * radius
            pos[2] = target[2] + np.cos(angle) * radius
            pos[1] = current[1]
            yaw_to_target = float(np.arctan2(target[0] - pos[0], target[2] - pos[2]))
            dist = float(np.linalg.norm(pos[[0, 2]] - current[[0, 2]]))
            for yaw_idx, offset in enumerate(yaw_offsets):
                yaw = float(np.arctan2(np.sin(yaw_to_target + offset), np.cos(yaw_to_target + offset)))
                candidates.append(
                    {
                        "position": pos.astype(np.float32),
                        "yaw": yaw,
                        "yaw_deg": float(np.rad2deg(yaw)),
                        "distance": dist,
                        "cluster_id": int(cluster_id),
                        "yaw_idx": int(yaw_idx),
                        "points_towards_target": bool(abs(float(offset)) < 1e-6),
                    }
                )
        return candidates

    @staticmethod
    def _generate_candidate_views_around_target_semantic(
        current_position,
        target_region_2d: dict,
        occupancy_map,
        explored_map=None,
        map_resolution: float = 0.05,
        search_radius_m: float = 1.5,
        num_clusters: int = 8,
        num_yaw_samples: int = 9,
        min_distance_m: float = 0.2,
        hint_yaw_target_world_pos=None,
    ):
        if occupancy_map is None or target_region_2d is None:
            return []
        occ = np.asarray(occupancy_map, dtype=np.float32)
        if occ.ndim != 2 or occ.size == 0:
            return []
        h, w = occ.shape[:2]
        map_center_row = h // 2
        map_center_col = w // 2
        map_resolution = max(1e-4, float(map_resolution))
        search_radius_px = max(1, int(round(float(search_radius_m) / map_resolution)))
        min_distance_px = max(0, int(round(float(min_distance_m) / map_resolution)))

        target_center_row, target_center_col = RealtimeSLAM._target_region_center(
            target_region_2d,
            h,
            w,
        )
        if target_center_row is None or target_center_col is None:
            return []

        target_component = RealtimeSLAM._target_region_mask(
            target_region_2d,
            h,
            w,
            int(target_center_row),
            int(target_center_col),
        )

        try:
            from scipy import ndimage

            labeled_target, num_target_cc = ndimage.label(target_component.astype(np.uint8))
            if num_target_cc > 0:
                cc_id = int(labeled_target[int(target_center_row), int(target_center_col)])
                if cc_id == 0:
                    cc_sizes = ndimage.sum(
                        target_component,
                        labeled_target,
                        index=np.arange(1, num_target_cc + 1),
                    )
                    cc_id = int(np.argmax(cc_sizes) + 1)
                target_component = labeled_target == cc_id
            dist_to_component = ndimage.distance_transform_edt(~target_component)
            neighbor_of_component = (
                ndimage.binary_dilation(
                    target_component,
                    structure=np.ones((3, 3), dtype=np.uint8),
                )
                & (~target_component)
            )
            use_scipy = True
        except Exception:
            yy, xx = np.indices((h, w))
            dist_to_component = np.sqrt(
                (yy - float(target_center_row)) ** 2
                + (xx - float(target_center_col)) ** 2
            )
            neighbor_of_component = dist_to_component <= 2.0
            use_scipy = False

        if explored_map is not None:
            explored = np.asarray(explored_map, dtype=np.float32)
            if explored.shape[:2] == (h, w):
                free_explored = (occ < 0.5) & (explored > 0.5)
            else:
                free_explored = occ < 0.5
        else:
            free_explored = occ < 0.5

        if use_scipy:
            try:
                from scipy import ndimage

                seed_mask = neighbor_of_component & free_explored
                if np.any(seed_mask):
                    labeled_free, _ = ndimage.label(free_explored.astype(np.uint8))
                    seed_labels = np.unique(labeled_free[seed_mask])
                    seed_labels = seed_labels[seed_labels > 0]
                    connected_free = np.isin(labeled_free, seed_labels)
                else:
                    connected_free = np.zeros((h, w), dtype=bool)
            except Exception:
                connected_free = free_explored
        else:
            connected_free = free_explored

        candidate_mask = (
            connected_free
            & (dist_to_component <= search_radius_px)
            & (~target_component)
        )
        if min_distance_px > 0:
            candidate_mask &= dist_to_component >= min_distance_px

        valid_rows, valid_cols = np.where(candidate_mask)
        if valid_rows.size == 0:
            return []

        free_cell_coords = np.column_stack([valid_rows, valid_cols]).astype(np.float32)
        num_clusters = max(1, min(int(num_clusters), int(free_cell_coords.shape[0])))
        cluster_centers_px = RealtimeSLAM._cluster_candidate_cells(
            free_cell_coords,
            num_clusters,
        )

        start_cell = target_region_2d.get("start_cell")
        if start_cell is not None:
            try:
                robot_row = int(np.clip(int(start_cell[0]), 0, h - 1))
                robot_col = int(np.clip(int(start_cell[1]), 0, w - 1))
            except Exception:
                robot_row = None
                robot_col = None
        else:
            robot_row = None
            robot_col = None
        reachable_mask = RealtimeSLAM._reachable_free_mask_from_current(
            occ,
            current_position,
            map_resolution,
            map_center_row,
            map_center_col,
            robot_row=robot_row,
            robot_col=robot_col,
        )

        current = np.asarray(current_position, dtype=np.float32).reshape(-1)[:3]
        if current.size < 3:
            return []
        coord_meta = RealtimeSLAM._target_region_coordinate_metadata(
            target_region_2d,
            map_resolution,
            current,
        )
        if hint_yaw_target_world_pos is not None:
            hint = np.asarray(hint_yaw_target_world_pos, dtype=np.float32).reshape(-1)
            target_world_x = float(hint[0]) if hint.size >= 1 else 0.0
            target_world_z = float(hint[2]) if hint.size >= 3 else 0.0
        elif coord_meta is not None:
            target_world = RealtimeSLAM._local_cell_to_world(
                int(target_center_row),
                int(target_center_col),
                coord_meta,
                current,
            )
            target_world_x = float(target_world[0])
            target_world_z = float(target_world[2])
        else:
            target_world_x = (float(target_center_col) - map_center_col) * map_resolution
            target_world_z = (map_center_row - float(target_center_row)) * map_resolution

        yaw_offsets = np.deg2rad(
            np.linspace(-45.0, 45.0, max(1, int(num_yaw_samples)), dtype=np.float32)
        )
        center_rows = np.clip(np.round(cluster_centers_px[:, 0]).astype(np.int32), 0, h - 1)
        center_cols = np.clip(np.round(cluster_centers_px[:, 1]).astype(np.int32), 0, w - 1)
        candidates = []
        for cluster_id, (center_row, center_col) in enumerate(zip(center_rows, center_cols)):
            if reachable_mask is not None and not bool(reachable_mask[center_row, center_col]):
                center_row, center_col = RealtimeSLAM._nearest_reachable_candidate_cell(
                    free_cell_coords,
                    reachable_mask,
                    int(center_row),
                    int(center_col),
                )
                if center_row is None or center_col is None:
                    continue
            if coord_meta is not None:
                position = RealtimeSLAM._local_cell_to_world(
                    int(center_row),
                    int(center_col),
                    coord_meta,
                    current,
                )
                world_x = float(position[0])
                world_z = float(position[2])
            else:
                world_x = current[0] + (float(center_col) - map_center_col) * map_resolution
                world_z = current[2] + (map_center_row - float(center_row)) * map_resolution
                position = np.asarray([world_x, current[1], world_z], dtype=np.float32)
            distance = float(np.linalg.norm(position[[0, 2]] - current[[0, 2]]))
            yaw_to_target = float(np.arctan2(target_world_x - world_x, target_world_z - world_z))
            center_offset_idx = len(yaw_offsets) // 2
            for yaw_idx, yaw_offset in enumerate(yaw_offsets):
                yaw = float(
                    np.arctan2(
                        np.sin(yaw_to_target + float(yaw_offset)),
                        np.cos(yaw_to_target + float(yaw_offset)),
                    )
                )
                candidates.append(
                    {
                        "position": position.copy(),
                        "yaw": yaw,
                        "yaw_deg": float(np.rad2deg(yaw)),
                        "distance": distance,
                        "cluster_id": int(cluster_id),
                        "yaw_idx": int(yaw_idx),
                        "grid_pos": (int(center_row), int(center_col)),
                        "points_towards_target": bool(yaw_idx == center_offset_idx),
                        "candidate_source": "target_region_free_connected",
                    }
                )
        return candidates

    @staticmethod
    def _target_region_center(target_region_2d: dict, height: int, width: int):
        center = target_region_2d.get("center")
        if center is not None:
            row, col = center
            return (
                int(np.clip(int(round(float(row))), 0, height - 1)),
                int(np.clip(int(round(float(col))), 0, width - 1)),
            )
        mask = target_region_2d.get("mask")
        if mask is not None:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            ys, xs = np.where(mask_np > 0)
            if ys.size > 0:
                return int(np.mean(ys)), int(np.mean(xs))
        bbox = target_region_2d.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            row_min, row_max, col_min, col_max = [int(round(float(v))) for v in bbox[:4]]
            return (
                int(np.clip((row_min + row_max) // 2, 0, height - 1)),
                int(np.clip((col_min + col_max) // 2, 0, width - 1)),
            )
        return None, None

    @staticmethod
    def _target_region_mask(
        target_region_2d: dict,
        height: int,
        width: int,
        center_row: int,
        center_col: int,
    ) -> np.ndarray:
        mask = target_region_2d.get("mask")
        if mask is not None:
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
            if mask_np.shape[:2] == (height, width):
                return mask_np > 0
        target_mask = np.zeros((height, width), dtype=bool)
        bbox = target_region_2d.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            row_min, row_max, col_min, col_max = [int(round(float(v))) for v in bbox[:4]]
            row_min = int(np.clip(row_min, 0, height))
            row_max = int(np.clip(row_max + 1, 0, height))
            col_min = int(np.clip(col_min, 0, width))
            col_max = int(np.clip(col_max + 1, 0, width))
            if row_max > row_min and col_max > col_min:
                target_mask[row_min:row_max, col_min:col_max] = True
                return target_mask
        target_mask[int(center_row), int(center_col)] = True
        return target_mask

    @staticmethod
    def _cluster_candidate_cells(free_cell_coords: np.ndarray, num_clusters: int) -> np.ndarray:
        if free_cell_coords.shape[0] <= num_clusters:
            return free_cell_coords
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=int(num_clusters), random_state=42, n_init=10)
            kmeans.fit(free_cell_coords)
            return kmeans.cluster_centers_.astype(np.float32)
        except Exception:
            centers = [free_cell_coords[0]]
            min_dist = np.full((free_cell_coords.shape[0],), np.inf, dtype=np.float32)
            while len(centers) < num_clusters:
                last = centers[-1]
                dist = np.linalg.norm(free_cell_coords - last[None, :], axis=1)
                min_dist = np.minimum(min_dist, dist)
                centers.append(free_cell_coords[int(np.argmax(min_dist))])
            return np.asarray(centers, dtype=np.float32)

    @staticmethod
    def _target_region_coordinate_metadata(
        target_region_2d: dict,
        map_resolution: float,
        current: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        required = ("gx1", "gy1", "start_x", "start_y", "current_world")
        if not all(key in target_region_2d for key in required):
            return None
        try:
            current_world = np.asarray(
                target_region_2d.get("current_world"), dtype=np.float32
            ).reshape(-1)
            if current_world.size < 3:
                current_world = np.asarray(current, dtype=np.float32).reshape(-1)
            return {
                "gx1": int(target_region_2d.get("gx1")),
                "gy1": int(target_region_2d.get("gy1")),
                "start_x": float(target_region_2d.get("start_x")),
                "start_y": float(target_region_2d.get("start_y")),
                "current_world": current_world[:3].astype(np.float32),
                "map_resolution": float(
                    target_region_2d.get("map_resolution_cm", map_resolution * 100.0)
                )
                / 100.0,
            }
        except Exception:
            return None

    @staticmethod
    def _local_cell_to_world(
        row: int,
        col: int,
        coord_meta: Dict[str, Any],
        reference_world: np.ndarray,
    ) -> np.ndarray:
        res_m = max(1e-4, float(coord_meta["map_resolution"]))
        full_row = int(row) + int(coord_meta["gx1"])
        full_col = int(col) + int(coord_meta["gy1"])
        target_ieve_y = full_row * res_m
        target_ieve_x = full_col * res_m
        current_world = np.asarray(coord_meta["current_world"], dtype=np.float32)
        world = np.asarray(reference_world, dtype=np.float32).reshape(-1)[:3].copy()
        world[0] = current_world[0] - (target_ieve_y - float(coord_meta["start_y"]))
        world[2] = current_world[2] - (target_ieve_x - float(coord_meta["start_x"]))
        return world.astype(np.float32)

    @staticmethod
    def _reachable_free_mask_from_current(
        occupancy_map: np.ndarray,
        current_position,
        map_resolution: float,
        map_center_row: int,
        map_center_col: int,
        robot_row: Optional[int] = None,
        robot_col: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        try:
            import cv2 as _cv2

            h, w = occupancy_map.shape[:2]
            current = np.asarray(current_position, dtype=np.float32).reshape(-1)[:3]
            if robot_row is None or robot_col is None:
                robot_col = int(
                    np.clip(map_center_col + current[0] / map_resolution, 0, w - 1)
                )
                robot_row = int(
                    np.clip(map_center_row - current[2] / map_resolution, 0, h - 1)
                )
            traversible = (occupancy_map < 0.5).astype(np.uint8)
            if traversible[robot_row, robot_col] == 0:
                found_anchor = False
                for radius in range(1, 7):
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr = robot_row + dr
                            nc = robot_col + dc
                            if 0 <= nr < h and 0 <= nc < w and traversible[nr, nc] == 1:
                                robot_row, robot_col = nr, nc
                                found_anchor = True
                                break
                        if found_anchor:
                            break
                    if found_anchor:
                        break
            if traversible[robot_row, robot_col] == 0:
                return None
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            traversible_flood = traversible.copy()
            _cv2.floodFill(traversible_flood, flood_mask, (robot_col, robot_row), 255)
            return traversible_flood == 255
        except Exception:
            return None

    @staticmethod
    def _nearest_reachable_candidate_cell(
        free_cell_coords: np.ndarray,
        reachable_mask: np.ndarray,
        center_row: int,
        center_col: int,
    ):
        dists = np.sqrt(
            (free_cell_coords[:, 0] - float(center_row)) ** 2
            + (free_cell_coords[:, 1] - float(center_col)) ** 2
        )
        for idx in np.argsort(dists)[:50]:
            row = int(free_cell_coords[int(idx), 0])
            col = int(free_cell_coords[int(idx), 1])
            if 0 <= row < reachable_mask.shape[0] and 0 <= col < reachable_mask.shape[1]:
                if bool(reachable_mask[row, col]):
                    return row, col
        return None, None

    def compute_mutual_info_score_semantic_batch(
        self,
        positions: np.ndarray,
        yaws: np.ndarray,
        target_mask,
        batch_size: int = 32,
    ) -> np.ndarray:
        if self.gaussians.get_xyz.shape[0] == 0:
            return np.zeros((len(positions),), dtype=np.float32)
        if target_mask is None or not bool(target_mask.any().item()):
            return np.zeros((len(positions),), dtype=np.float32)

        positions = np.asarray(positions, dtype=np.float32)
        yaws = np.asarray(yaws, dtype=np.float32).reshape(-1)
        scores = np.zeros((len(positions),), dtype=np.float32)
        dummy_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        dummy_depth = np.zeros((self.height, self.width), dtype=np.float32)

        opacity = self.gaussians.get_opacity.detach().reshape(-1)
        uncertainty = -torch.log(torch.clamp(opacity, min=1e-4, max=0.9999))
        semantic = getattr(self.gaussians, "get_semantic_target", None)
        if callable(semantic):
            semantic = semantic()
        if torch.is_tensor(semantic) and semantic.shape[0] == uncertainty.shape[0]:
            semantic_weight = torch.clamp(semantic.detach().reshape(-1), 0.0, 1.0)
        else:
            semantic_weight = torch.ones_like(uncertainty)

        with torch.no_grad():
            for batch_start in range(0, len(positions), max(1, int(batch_size))):
                batch_end = min(batch_start + max(1, int(batch_size)), len(positions))
                for idx in range(batch_start, batch_end):
                    try:
                        rotation = self._yaw_to_quat(float(yaws[idx]))
                        pose = self._quat_to_matrix(positions[idx], rotation)
                        virtual_cam = self.create_camera(dummy_rgb, dummy_depth, pose, -1)
                        render_pkg = render(
                            virtual_cam,
                            self.gaussians,
                            self.pipeline_params,
                            self.background,
                        )
                        if render_pkg is None:
                            continue
                        visibility = render_pkg.get("visibility_filter")
                        if visibility is None or visibility.shape[0] != target_mask.shape[0]:
                            continue
                        visible_target = visibility.to(torch.bool) & target_mask
                        if not bool(visible_target.any().item()):
                            continue
                        info = uncertainty[visible_target] * semantic_weight[visible_target]
                        scores[idx] = float(info.sum().item())
                    except Exception as exc:
                        logger.debug("semantic GauSS-MI candidate render failed: %s", exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return scores

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half = float(yaw) * 0.5
        return np.asarray([0.0, np.sin(half), 0.0, np.cos(half)], dtype=np.float32)

    def get_reconstruction_quality(self) -> Dict[str, float]:
        num_gaussians = int(self.gaussians.get_xyz.shape[0])
        avg_reliability = 0.0
        reliable_percent = 0.0
        if num_gaussians > 0:
            reliability = self.gaussians.get_reliability_1dim.detach()
            avg_reliability = float(reliability.mean().item())
            reliable_percent = float(
                (reliability > 0.7).float().mean().item() * 100.0
            )
        return {
            "num_gaussians": num_gaussians,
            "avg_reliability": avg_reliability,
            "reliable_percent": reliable_percent,
        }

    def save_checkpoint(self, name: str = "checkpoint"):
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ply_path = checkpoint_dir / "point_cloud.ply"
        self.gaussians.save_ply(str(ply_path))
        stats_path = checkpoint_dir / "stats.txt"
        stats_path.write_text(
            "\n".join(
                [
                    f"frames={self.frame_count}",
                    f"keyframes={len(self.kf_indices)}",
                    f"gaussians={int(self.gaussians.get_xyz.shape[0])}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        logger.info("Saved 3DGS checkpoint to %s", checkpoint_dir)
        return checkpoint_dir

    def save_frame_ply(
        self,
        env_step: Optional[int] = None,
        stage: str = "online",
        output_subdir: str = "3dgs_frames",
        highlight_confirmed: bool = True,
        confirmed_color=(255, 0, 255),
    ):
        if self.gaussians is None or self.gaussians.get_xyz.shape[0] == 0:
            return None
        out_dir = self.output_dir / str(output_subdir or "3dgs_frames")
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_stage = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in str(stage or "online")
        ).strip("_") or "online"
        step_part = (
            "step_{:06d}".format(int(env_step))
            if env_step is not None
            else "step_unknown"
        )
        ply_path = out_dir / "{}_frame_{:06d}_{}.ply".format(
            step_part,
            int(self.frame_count),
            safe_stage,
        )
        self.gaussians.save_ply(
            str(ply_path),
            include_color=True,
            highlight_confirmed=bool(highlight_confirmed),
            confirmed_color=confirmed_color,
        )
        logger.info("Saved per-frame full 3DGS PLY to %s", ply_path)
        return ply_path
