#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import logging

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_mapping.gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_mapping.gaussian_splatting.utils.graphics_utils import (
    BasicPointCloud,
    getWorld2View2,
)
from gaussian_mapping.gaussian_splatting.utils.sh_utils import RGB2SH, SH2RGB
from gaussian_mapping.gaussian_splatting.utils.system_utils import mkdir_p

logger = logging.getLogger(__name__)


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        # GauSS-MI directional reliability: +x, +z, -x, -z in Habitat's
        # y-up world frame. The original implementation uses four horizontal
        # directions and initializes every newly created Gaussian at 0.5.
        self._reliability = torch.empty(
            (0, 4), dtype=torch.float32, device="cuda"
        )
        self.init_reliability = 0.5
        
        # =====================================================================
        # NEW: Semantic target attribute (binary: 1=target, 0=not target)
        # Following SemSplatam pattern for semantic Gaussian initialization
        # =====================================================================
        self._semantic_target = torch.empty(0, dtype=torch.float32, device="cuda")
        self._confirmed_target_indices = None
        self._confirmed_cluster_locked = False
        self._target_semantic_soft_lock = True
        self._target_category = None
        self._target_growth_chair_xz_radius = 1.0
        self._target_growth_chair_y_radius = 1.2
        self._target_growth_default_xz_radius = 5.0
        self._target_growth_default_y_radius = 3.0

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_reliability(self):
        self.clip_reliability()
        return self._reliability

    @property
    def get_reliability_1dim(self):
        self.clip_reliability()
        if self._reliability.shape[0] == 0:
            return torch.empty(0, dtype=torch.float32, device="cuda")
        return self._reliability.mean(dim=1)

    @property
    def get_infogain(self):
        self.clip_reliability()
        return -torch.log(self._reliability)

    @property
    def get_logodds_p(self):
        p = self.get_reliability
        odds = p / (1.0 - p)
        return torch.log(odds)

    @staticmethod
    def logodds2r(logodds):
        odds = torch.exp(logodds)
        return 1.0 - 1.0 / (1.0 + odds)

    def clip_reliability(self):
        self._reliability = torch.clamp(
            self._reliability, min=0.0001, max=0.9999
        )

    @staticmethod
    def clip_logodds(logodds):
        return torch.clamp(logodds, min=-10.0, max=10.0)

    def input_logodds(self, logodds_zk):
        logodds_zk = self.clip_logodds(logodds_zk)
        logodds_k = self.clip_logodds(self.get_logodds_p + logodds_zk)
        self._reliability = self.logodds2r(logodds_k)
        self.clip_reliability()

    # =====================================================================
    # NEW: Semantic target accessors
    # =====================================================================
    @property
    def get_semantic_target(self):
        """Get semantic target labels (1=target, 0=not target)"""
        return self._semantic_target

    @property
    def get_target_mask(self):
        """Get boolean mask for target Gaussians"""
        if self._semantic_target.shape[0] == 0:
            return torch.empty(0, dtype=torch.bool, device="cuda")
        return self._semantic_target > 0.5
    
    @property
    def num_target_gaussians(self):
        """Get number of target Gaussians"""
        return int(self.get_target_mask.sum().item())
    
    def reset_semantic_target_values(self):
        """Reset semantic target values without clearing confirmed-cluster lock."""
        num_gaussians = self._xyz.shape[0]
        self._semantic_target = torch.zeros(
            num_gaussians, dtype=torch.float32, device=self._xyz.device)

    def _ensure_semantic_target_size(self):
        """Resize semantic target values to match current Gaussians, preserving labels."""
        num_gaussians = int(self._xyz.shape[0])
        if self._semantic_target.shape[0] == num_gaussians:
            return
        resized = torch.zeros(
            num_gaussians, dtype=torch.float32, device=self._xyz.device)
        if self._semantic_target.numel() > 0:
            keep = min(int(self._semantic_target.shape[0]), num_gaussians)
            resized[:keep] = self._semantic_target[:keep].to(
                device=self._xyz.device, dtype=torch.float32)
        self._semantic_target = resized
        confirmed = self.get_confirmed_target_indices()
        if confirmed is not None and confirmed.numel() > 0:
            self._semantic_target[confirmed.long().to(self._xyz.device)] = 1.0

    def reset_semantic_target(self):
        """Reset all semantic targets to 0 (not target)"""
        self.reset_semantic_target_values()
        self._confirmed_target_indices = None
        self._confirmed_cluster_locked = False
    
    def set_semantic_target(self, indices, value=1.0):
        """Set semantic target value for specific Gaussian indices"""
        if self._semantic_target.shape[0] != self._xyz.shape[0]:
            self._ensure_semantic_target_size()
        self._semantic_target[indices] = value

    def set_confirmed_target_indices(self, indices):
        """Lock the confirmed target cluster as the semantic target mask.

        This mirrors minav's target-confirmation handoff: far target
        confirmation chooses one Gaussian cluster, and later semantic NBV /
        localization should use that fixed cluster instead of re-clustering all
        target-labeled Gaussians.
        """
        if self._xyz.shape[0] == 0:
            return 0

        if self._semantic_target.shape[0] != self._xyz.shape[0]:
            self._ensure_semantic_target_size()

        if torch.is_tensor(indices):
            idx = indices.detach().long().to(self._xyz.device).reshape(-1)
        else:
            idx = torch.as_tensor(indices, dtype=torch.long, device=self._xyz.device).reshape(-1)
        if idx.numel() == 0:
            return 0

        valid = (idx >= 0) & (idx < self._xyz.shape[0])
        idx = torch.unique(idx[valid])
        if idx.numel() == 0:
            return 0

        self._semantic_target.zero_()
        self._semantic_target[idx] = 1.0
        self._confirmed_target_indices = idx.detach().clone()
        self._confirmed_cluster_locked = True
        return int(idx.numel())

    def get_confirmed_target_indices(self):
        if self._confirmed_target_indices is None:
            return None
        valid = self._confirmed_target_indices < self._xyz.shape[0]
        return self._confirmed_target_indices[valid]

    def set_target_semantic_growth_policy(
        self,
        target_category=None,
        soft_lock=True,
        chair_xz_radius=1.0,
        chair_y_radius=1.2,
        default_xz_radius=5.0,
        default_y_radius=3.0,
    ):
        """Configure post-confirmation growth around the confirmed target cluster."""
        self._target_category = target_category
        self._target_semantic_soft_lock = bool(soft_lock)
        self._target_growth_chair_xz_radius = float(chair_xz_radius)
        self._target_growth_chair_y_radius = float(chair_y_radius)
        self._target_growth_default_xz_radius = float(default_xz_radius)
        self._target_growth_default_y_radius = float(default_y_radius)

    def _target_semantic_growth_limits(self):
        category = str(getattr(self, "_target_category", "") or "").strip().lower()
        if category == "chair":
            return (
                float(getattr(self, "_target_growth_chair_xz_radius", 1.0)),
                float(getattr(self, "_target_growth_chair_y_radius", 1.2)),
            )
        return (
            float(getattr(self, "_target_growth_default_xz_radius", 5.0)),
            float(getattr(self, "_target_growth_default_y_radius", 3.0)),
        )

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def _normalize_saliency(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return values
        values = np.clip(values, 0.0, None)
        scale = float(np.percentile(values, 95))
        if scale < 1e-6:
            scale = float(values.max()) if values.size > 0 else 0.0
        if scale < 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.clip(values / scale, 0.0, 1.0).astype(np.float32)

    def _sample_rgbd_points_edge_aware(self, cam, rgb_np, depth_np, downsample_factor):
        """
        Sample RGBD points with edge/depth discontinuity awareness instead of
        pure random downsampling.

        The total point budget stays close to the original `1/downsample_factor`
        policy, but pixels near strong RGB/depth edges receive higher sampling
        probability so difficult regions are less likely to be starved.
        """
        depth_np = np.asarray(depth_np, dtype=np.float32)
        rgb_np = np.asarray(rgb_np, dtype=np.uint8)

        valid_mask = np.isfinite(depth_np) & (depth_np > 0.01) & (depth_np < 100.0)
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        if downsample_factor <= 1:
            target_count = valid_count
        else:
            target_count = max(1, int(np.ceil(valid_count / float(downsample_factor))))

        gray = (
            0.299 * rgb_np[:, :, 0].astype(np.float32)
            + 0.587 * rgb_np[:, :, 1].astype(np.float32)
            + 0.114 * rgb_np[:, :, 2].astype(np.float32)
        ) / 255.0
        grad_gray_y, grad_gray_x = np.gradient(gray)
        rgb_edge = np.sqrt(grad_gray_x ** 2 + grad_gray_y ** 2)

        depth_safe = np.where(valid_mask, depth_np, 0.0).astype(np.float32)
        grad_depth_y, grad_depth_x = np.gradient(depth_safe)
        depth_edge = np.sqrt(grad_depth_x ** 2 + grad_depth_y ** 2)
        depth_edge[~valid_mask] = 0.0

        rgb_score = self._normalize_saliency(rgb_edge)
        depth_score = self._normalize_saliency(depth_edge)

        weights_map = 0.15 + 0.45 * rgb_score + 0.70 * depth_score
        rows, cols = np.where(valid_mask)
        weights = weights_map[rows, cols].astype(np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.ones_like(weights, dtype=np.float64)
            weight_sum = float(weights.sum())
        probs = weights / weight_sum

        if target_count < valid_count:
            chosen = np.random.choice(valid_count, size=target_count, replace=False, p=probs)
            rows = rows[chosen]
            cols = cols[chosen]

        sampled_depth = depth_np[rows, cols].astype(np.float32)
        x_cam = (cols.astype(np.float32) - float(cam.cx)) * sampled_depth / float(cam.fx)
        y_cam = (rows.astype(np.float32) - float(cam.cy)) * sampled_depth / float(cam.fy)
        z_cam = sampled_depth

        pts_cam = np.stack(
            [x_cam, y_cam, z_cam, np.ones_like(z_cam, dtype=np.float32)],
            axis=1,
        ).astype(np.float32)

        w2c = getWorld2View2(cam.R, cam.T).cpu().numpy()
        c2w = np.linalg.inv(w2c).astype(np.float32)
        pts_world = (c2w @ pts_cam.T).T[:, :3]
        colors = rgb_np[rows, cols].astype(np.float32) / 255.0
        return pts_world.astype(np.float32), colors.astype(np.float32)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgb_np = np.asarray(rgb)
        depth_np = np.asarray(depth)
        new_xyz, new_rgb = self._sample_rgbd_points_edge_aware(
            cam,
            rgb_np,
            depth_np,
            downsample_factor=downsample_factor,
        )

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()

        # Guard: 空点云时 distCUDA2 会以 gridDim.x=0 启动内核 -> CUDA invalid configuration
        if fused_point_cloud.shape[0] == 0:
            n_sh = (self.max_sh_degree + 1) ** 2
            features_e = torch.zeros((0, 3, n_sh), dtype=torch.float, device="cuda")
            n_scale = 1 if self.isotropic else 3
            scales_e = torch.zeros((0, n_scale), dtype=torch.float, device="cuda")
            rots_e = torch.zeros((0, 4), dtype=torch.float, device="cuda")
            opacities_e = torch.zeros((0, 1), dtype=torch.float, device="cuda")
            return fused_point_cloud, features_e, scales_e, rots_e, opacities_e

        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        torch.cuda.synchronize()  # DEBUG: 捕获distCUDA2 CUDA错误
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def _base_rgb_uint8(self) -> np.ndarray:
        if self._features_dc.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        rgb = SH2RGB(
            self._features_dc.detach()
            .transpose(1, 2)[:, :, 0]
            .contiguous()
            .cpu()
            .numpy()
        )
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    def _confirmed_target_mask_cpu(self, num_points: int) -> np.ndarray:
        mask = np.zeros((int(num_points),), dtype=bool)
        if not bool(getattr(self, "_confirmed_cluster_locked", False)):
            return mask
        confirmed = self.get_confirmed_target_indices()
        if confirmed is not None:
            if torch.is_tensor(confirmed):
                confirmed = confirmed.detach().cpu().numpy()
            confirmed = np.asarray(confirmed, dtype=np.int64).reshape(-1)
            valid = confirmed[(confirmed >= 0) & (confirmed < int(num_points))]
            if valid.size > 0:
                mask[np.unique(valid)] = True
        if self._semantic_target.shape[0] == int(num_points):
            semantic = self._semantic_target.detach().cpu().numpy()
            mask |= semantic > 0.4
        return mask

    def save_ply(
        self,
        path,
        include_color: bool = False,
        highlight_confirmed: bool = False,
        confirmed_color=(255, 0, 255),
    ):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        confirmed_mask = None
        if highlight_confirmed:
            confirmed_mask = self._confirmed_target_mask_cpu(xyz.shape[0])
            if bool(confirmed_mask.any()):
                f_dc = f_dc.copy()
                highlight_rgb = np.asarray(confirmed_color, dtype=np.float32) / 255.0
                f_dc[confirmed_mask, :3] = RGB2SH(highlight_rgb)

        attr_list = self.construct_list_of_attributes()
        dtype_full = [(attribute, "f4") for attribute in attr_list]
        if include_color:
            dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        for attr_idx, attr_name in enumerate(attr_list):
            elements[attr_name] = attributes[:, attr_idx]
        if include_color:
            colors = self._base_rgb_uint8()
            if highlight_confirmed and colors.shape[0] == xyz.shape[0]:
                if confirmed_mask is None:
                    confirmed_mask = self._confirmed_target_mask_cpu(xyz.shape[0])
                if bool(confirmed_mask.any()):
                    colors = colors.copy()
                    colors[confirmed_mask] = np.asarray(confirmed_color, dtype=np.uint8)
            elements["red"] = colors[:, 0]
            elements["green"] = colors[:, 1]
            elements["blue"] = colors[:, 2]
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters, reset_value=0.4, protected_mask=None
    ):
        """
        Reset opacity for only non-visible gaussians (with optional protection)
        
        Args:
            visibility_filters: List of boolean tensors indicating visible gaussians
            reset_value: The opacity value to reset non-visible gaussians to (default: 0.4)
                        Lower values (e.g., 0.15) for bright scenes (more aggressive pruning)
                        Higher values (e.g., 0.35) for dark scenes (more conservative)
            protected_mask: Optional boolean tensor. If provided, gaussians with True 
                           will NOT be reset (e.g., high reliability gaussians)
        """
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * reset_value)

        # 保留可见高斯的原始透明度
        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        
        # ★★★ 新增：保护高可靠性高斯 ★★★
        if protected_mask is not None:
            opacities_new[protected_mask] = self.get_opacity[protected_mask]
        
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()
        self._reliability = torch.full(
            (self._xyz.shape[0], 4),
            float(self.init_reliability),
            dtype=torch.float32,
            device=self._xyz.device,
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        confirmed_old = self._confirmed_target_indices
        confirmed_keep_mask = None
        if confirmed_old is not None and confirmed_old.numel() > 0:
            confirmed_keep_mask = torch.zeros(
                valid_points_mask.shape[0], dtype=torch.bool, device=valid_points_mask.device)
            old_valid = confirmed_old[confirmed_old < valid_points_mask.shape[0]]
            if old_valid.numel() > 0:
                confirmed_keep_mask[old_valid] = True

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
        if self._reliability.shape[0] > 0:
            self._reliability = self._reliability[valid_points_mask]
        
        # NEW: Prune semantic target accordingly
        if self._semantic_target.shape[0] > 0:
            self._semantic_target = self._semantic_target[valid_points_mask]
        if confirmed_keep_mask is not None:
            kept_confirmed = confirmed_keep_mask & valid_points_mask
            if kept_confirmed.any():
                old_to_new = torch.cumsum(valid_points_mask.long(), dim=0) - 1
                self._confirmed_target_indices = old_to_new[kept_confirmed].long()
                self._confirmed_cluster_locked = self._confirmed_target_indices.numel() > 0
            else:
                self._confirmed_target_indices = None
                self._confirmed_cluster_locked = False

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                try:
                    _zeros = torch.zeros_like(extension_tensor)
                    torch.cuda.synchronize()
                except RuntimeError as _dbg_e:
                    logger.exception(
                        "cat_tensors_to_optimizer zeros_like failed for %s",
                        group["name"],
                    )
                    raise
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], _zeros), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        new_semantic_target=None,
        new_reliability=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        if self._semantic_target.shape[0] != self._xyz.shape[0]:
            self._ensure_semantic_target_size()

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        # NEW: Extend semantic target
        num_new = new_xyz.shape[0]
        if new_semantic_target is not None:
            new_semantic = new_semantic_target.to(
                device=self._semantic_target.device, dtype=torch.float32)
        else:
            # Default: new points are not target (0)
            new_semantic = torch.zeros(
                num_new, dtype=torch.float32, device=self._semantic_target.device)
        self._semantic_target = torch.cat([self._semantic_target, new_semantic], dim=0)

        if new_reliability is None:
            new_reliability = torch.full(
                (num_new, 4),
                float(self.init_reliability),
                dtype=torch.float32,
                device=self._xyz.device,
            )
        else:
            new_reliability = new_reliability.to(
                device=self._xyz.device, dtype=torch.float32
            ).reshape(num_new, 4)
        self._reliability = torch.cat(
            (self._reliability, new_reliability), dim=0
        )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)
        
        # NEW: Propagate semantic target for split points
        new_semantic = None
        if self._semantic_target.shape[0] > 0:
            new_semantic = self._semantic_target[selected_pts_mask].repeat(N)
        new_reliability = self._reliability[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_semantic_target=new_semantic,
            new_reliability=new_reliability,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        
        # NEW: Propagate semantic target for cloned points
        new_semantic = None
        if self._semantic_target.shape[0] > 0:
            new_semantic = self._semantic_target[selected_pts_mask]
        new_reliability = self._reliability[selected_pts_mask]
        
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_semantic_target=new_semantic,
            new_reliability=new_reliability,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        # current_count = self.get_xyz.shape[0]
        # if max_gaussians is None or current_count < max_gaussians:
        #     self.densify_and_clone(grads, max_grad, extent)
        #     self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        training_cfg = {}
        if isinstance(self.config, dict):
            training_cfg = self.config.get("Training", {}) or {}
        protect_semantic = bool(training_cfg.get("protect_semantic_target_prune", True))
        protect_threshold = float(
            training_cfg.get("target_prune_protect_threshold", 0.7)
        )
        protected_mask = torch.zeros_like(prune_mask, dtype=torch.bool)
        confirmed = self.get_confirmed_target_indices()
        if confirmed is not None and confirmed.numel() > 0:
            confirmed = confirmed.long().to(prune_mask.device)
            confirmed = confirmed[
                (confirmed >= 0) & (confirmed < protected_mask.shape[0])
            ]
            if confirmed.numel() > 0:
                protected_mask[confirmed] = True
        if (
            protect_semantic
            and self._semantic_target.shape[0] == protected_mask.shape[0]
        ):
            protected_mask = torch.logical_or(
                protected_mask,
                self._semantic_target.to(prune_mask.device) >= protect_threshold,
            )
        if protected_mask.any():
            prune_mask = torch.logical_and(prune_mask, ~protected_mask)
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
    
    # =========================================================================
    # NEW: Semantic Target Methods (following SemSplatam pattern)
    # =========================================================================
    
    def save_target_gaussians_ply(self, path, threshold=0.4):
        """
        Save only target Gaussians (semantic_target > threshold) to PLY file.
        Used for visualization of detected target object.
        
        Args:
            path: Output PLY file path
            threshold: Threshold for target classification (default 0.5)
        
        Returns:
            int: Number of target Gaussians saved
        """
        if self._semantic_target.shape[0] == 0:
            logger.debug("No semantic target initialized")
            return 0
        
        target_mask = self._semantic_target > threshold
        num_target = target_mask.sum().item()
        
        if num_target == 0:
            logger.debug("No target Gaussians found")
            return 0
        
        mkdir_p(os.path.dirname(path))
        
        # Extract target Gaussian attributes
        xyz = self._xyz[target_mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc[target_mask].detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest[target_mask].detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity[target_mask].detach().cpu().numpy()
        scale = self._scaling[target_mask].detach().cpu().numpy()
        rotation = self._rotation[target_mask].detach().cpu().numpy()
        semantic = self._semantic_target[target_mask].detach().cpu().numpy()[:, np.newaxis]

        # Build attribute list with semantic
        attr_list = self.construct_list_of_attributes() + ["semantic_target"]
        
        dtype_full = [(attribute, "f4") for attribute in attr_list]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                opacities,
                scale,
                rotation,
                semantic,
            ),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

        logger.debug("Saved %s target Gaussians to %s", num_target, path)
        return int(num_target)
    
    def project_mask_to_gaussian_indices(self, target_mask_2d, camera,
                                         depth=None, depth_tolerance=0.3):
        """
        Project a binary 2D mask to current Gaussian indices without mutating
        `_semantic_target`.

        Returns:
            np.ndarray: Global Gaussian indices whose projections fall inside the
                        mask and pass optional depth filtering.
        """
        if self._xyz.shape[0] == 0:
            return np.array([], dtype=np.int64)

        if hasattr(camera, 'world_view_transform'):
            W2C = camera.world_view_transform.T
        else:
            W2C = getWorld2View2(camera.R, camera.T)

        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        H, W = target_mask_2d.shape[:2]

        xyz_world = self._xyz
        ones = torch.ones(xyz_world.shape[0], 1, device=xyz_world.device)
        xyz_homo = torch.cat([xyz_world, ones], dim=1)
        xyz_cam = (W2C @ xyz_homo.T).T[:, :3]

        valid_z = xyz_cam[:, 2] > 0.1
        u = (xyz_cam[:, 0] * fx / xyz_cam[:, 2] + cx)
        v = (xyz_cam[:, 1] * fy / xyz_cam[:, 2] + cy)
        valid_u = (u >= 0) & (u < W)
        valid_v = (v >= 0) & (v < H)
        valid_proj = valid_z & valid_u & valid_v

        u_int = u.long().clamp(0, W - 1)
        v_int = v.long().clamp(0, H - 1)

        if isinstance(target_mask_2d, np.ndarray):
            mask_tensor = torch.from_numpy(target_mask_2d).float().to(self._xyz.device)
        else:
            mask_tensor = target_mask_2d.float()

        mask_hit = mask_tensor[v_int, u_int] > 0.5
        valid_proj = valid_proj & mask_hit

        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth_tensor = torch.from_numpy(depth).float().to(self._xyz.device)
            else:
                depth_tensor = depth.float()

            rendered_depth = xyz_cam[:, 2]
            observed_depth = depth_tensor[v_int, u_int]
            depth_valid = torch.abs(rendered_depth - observed_depth) < depth_tolerance
            valid_proj = valid_proj & depth_valid

        return torch.where(valid_proj)[0].detach().cpu().numpy().astype(np.int64)

    def update_target_semantic_from_mask(self, target_mask_2d, camera, 
                                          depth=None, update_mode="max",
                                          depth_tolerance=0.3):
        """
        Update semantic target labels from 2D segmentation mask (from DINO SAM).
        Projects 3D Gaussian positions to 2D and checks mask.
        
        Multi-view fusion: accumulates observations across frames.
        
        Args:
            target_mask_2d: Binary mask [H, W] from DINO SAM segmentation
            camera: Camera object with intrinsics and extrinsics
            depth: Optional depth image for depth-based filtering
            update_mode: "max" (take max), "avg" (running average), "replace"
            depth_tolerance: Tolerance for depth matching (meters)
        
        Returns:
            int: Number of Gaussians updated
        """
        if self._xyz.shape[0] == 0:
            return 0

        locked = bool(getattr(self, "_confirmed_cluster_locked", False))
        soft_lock = bool(getattr(self, "_target_semantic_soft_lock", True))
        confirmed = None
        if locked and hasattr(self, "get_confirmed_target_indices"):
            confirmed = self.get_confirmed_target_indices()
            if confirmed is not None:
                confirmed = confirmed.long().to(self._xyz.device).reshape(-1)
                confirmed = confirmed[(confirmed >= 0) & (confirmed < self._xyz.shape[0])]

        if locked and not soft_lock:
            confirmed = self.get_confirmed_target_indices()
            if confirmed is None:
                return 0
            return int(confirmed.numel())
        if locked and (confirmed is None or confirmed.numel() == 0):
            return 0
        
        # Ensure semantic target is initialized
        if self._semantic_target.shape[0] != self._xyz.shape[0]:
            self._ensure_semantic_target_size()
        
        # Get camera parameters
        if hasattr(camera, 'world_view_transform'):
            W2C = camera.world_view_transform.T  # [4, 4]
        else:
            W2C = getWorld2View2(camera.R, camera.T)
        
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        H, W = target_mask_2d.shape[:2]
        
        # Project 3D points to 2D
        xyz_world = self._xyz  # [N, 3]
        ones = torch.ones(xyz_world.shape[0], 1, device=xyz_world.device)
        xyz_homo = torch.cat([xyz_world, ones], dim=1)  # [N, 4]
        xyz_cam = (W2C @ xyz_homo.T).T[:, :3]  # [N, 3] in camera space
        
        # Filter points in front of camera
        valid_z = xyz_cam[:, 2] > 0.1
        
        # Project to pixel coordinates
        u = (xyz_cam[:, 0] * fx / xyz_cam[:, 2] + cx)
        v = (xyz_cam[:, 1] * fy / xyz_cam[:, 2] + cy)
        
        # Check bounds
        valid_u = (u >= 0) & (u < W)
        valid_v = (v >= 0) & (v < H)
        valid_proj = valid_z & valid_u & valid_v
        
        # Get pixel indices
        u_int = u.long().clamp(0, W - 1)
        v_int = v.long().clamp(0, H - 1)
        
        # Convert mask to tensor
        if isinstance(target_mask_2d, np.ndarray):
            mask_tensor = torch.from_numpy(target_mask_2d).float().to(self._xyz.device)
        else:
            mask_tensor = target_mask_2d.float()
        
        # Sample mask values
        new_labels = torch.zeros(self._xyz.shape[0], device=self._xyz.device)
        mask_values = mask_tensor[v_int, u_int]
        
        # Depth filtering (optional)
        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth_tensor = torch.from_numpy(depth).float().to(self._xyz.device)
            else:
                depth_tensor = depth.float()
            
            rendered_depth = xyz_cam[:, 2]
            observed_depth = depth_tensor[v_int, u_int]
            depth_valid = torch.abs(rendered_depth - observed_depth) < depth_tolerance
            valid_proj = valid_proj & depth_valid
        
        new_labels[valid_proj] = mask_values[valid_proj]
        if locked:
            confirmed_center = self._xyz[confirmed].mean(dim=0)
            xz_radius, y_radius = self._target_semantic_growth_limits()
            xz_delta = self._xyz[:, [0, 2]] - confirmed_center[[0, 2]]
            xz_dist = torch.linalg.norm(xz_delta, dim=1)
            y_dist = torch.abs(self._xyz[:, 1] - confirmed_center[1])
            growth_valid = (xz_dist <= xz_radius) & (y_dist <= y_radius)
            new_labels = new_labels * growth_valid.to(dtype=new_labels.dtype)
            new_labels[confirmed] = 1.0
        
        # Multi-view fusion
        if update_mode == "max":
            self._semantic_target = torch.max(self._semantic_target, new_labels)
        elif update_mode == "avg":
            # Running average (simple version)
            alpha = 0.5
            update_mask = new_labels > 0
            self._semantic_target[update_mask] = (
                alpha * self._semantic_target[update_mask] + 
                (1 - alpha) * new_labels[update_mask]
            )
        elif update_mode == "replace":
            self._semantic_target = new_labels

        if locked:
            self._semantic_target[confirmed] = 1.0
        
        num_updated = (new_labels > 0).sum().item()
        return int(num_updated)
    
    def extract_target_gaussian_points(self, threshold=0.4):
        """
        Extract 3D points of target Gaussians.
        
        Args:
            threshold: Threshold for target classification
            
        Returns:
            np.ndarray: [N, 3] array of target Gaussian positions
        """
        if self._semantic_target.shape[0] == 0:
            return np.array([]).reshape(0, 3)
        
        target_mask = self._semantic_target > threshold
        target_xyz = self._xyz[target_mask].detach().cpu().numpy()
        return target_xyz
    
    def get_target_gaussian_centroid(self, threshold=0.4):
        """
        Get centroid of target Gaussians.
        
        Args:
            threshold: Threshold for target classification
            
        Returns:
            np.ndarray or None: [3,] centroid position
        """
        points = self.extract_target_gaussian_points(threshold)
        if len(points) == 0:
            return None
        return points.mean(axis=0)
    
    # =========================================================================
    # =========================================================================
    # Instance Clustering Methods (DBSCAN for multi-view semantic Gaussians)
    # =========================================================================
    
    def cluster_target_gaussians(self, threshold=0.4, eps=0.15, min_samples=10):
        """
        对目标语义高斯进行DBSCAN聚类，实现实例级分割。
        
        Args:
            threshold: 目标高斯的语义阈值
            eps: DBSCAN邻域半径（米）
            min_samples: DBSCAN最小邻居数
            
        Returns:
            dict: {
                'labels': np.ndarray, 聚类标签 (-1为噪声)
                'num_clusters': int, 聚类数量
                'cluster_centers': list of np.ndarray, 每个聚类的中心坐标
                'cluster_sizes': list of int, 每个聚类的点数
                'target_indices': np.ndarray, 目标高斯在原始数组中的索引
                'target_xyz': np.ndarray, 目标高斯的坐标
            }
        """
        from sklearn.cluster import DBSCAN
        
        if self._semantic_target.shape[0] == 0:
            return {
                'labels': np.array([]),
                'num_clusters': 0,
                'cluster_centers': [],
                'cluster_sizes': [],
                'target_indices': np.array([]),
                'target_xyz': np.array([]).reshape(0, 3)
            }
        
        # 获取目标高斯
        target_mask = self._semantic_target > threshold
        target_indices = torch.where(target_mask)[0].cpu().numpy()
        target_xyz = self._xyz[target_mask].detach().cpu().numpy()
        
        if len(target_xyz) == 0:
            return {
                'labels': np.array([]),
                'num_clusters': 0,
                'cluster_centers': [],
                'cluster_sizes': [],
                'target_indices': target_indices,
                'target_xyz': target_xyz
            }
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(target_xyz)
        labels = clustering.labels_
        
        # 统计聚类信息
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声标签
        num_clusters = len(unique_labels)
        
        cluster_centers = []
        cluster_sizes = []
        
        for label in sorted(unique_labels):
            cluster_mask = labels == label
            cluster_points = target_xyz[cluster_mask]
            cluster_centers.append(cluster_points.mean(axis=0))
            cluster_sizes.append(len(cluster_points))
        
        return {
            'labels': labels,
            'num_clusters': num_clusters,
            'cluster_centers': cluster_centers,
            'cluster_sizes': cluster_sizes,
            'target_indices': target_indices,
            'target_xyz': target_xyz
        }
    
    def save_clustered_target_gaussians_ply(self, path, threshold=0.4, 
                                             eps=0.15, min_samples=10,
                                             min_cluster_size=150):
        """
        保存聚类后的目标高斯PLY文件，每个实例用不同颜色表示。
        过滤掉太小的聚类和噪声点。
        
        Args:
            path: 保存路径
            threshold: 目标高斯的语义阈值
            eps: DBSCAN邻域半径（米）
            min_samples: DBSCAN最小邻居数
            min_cluster_size: 最小聚类大小（过滤小聚类）
            
        Returns:
            dict: {
                'num_instances': int,
                'instance_centers': list of np.ndarray,
                'instance_sizes': list of int,
                'total_points_saved': int,
                'noise_points_filtered': int
            }
        """
        # 执行聚类
        cluster_result = self.cluster_target_gaussians(threshold, eps, min_samples)
        
        if cluster_result['num_clusters'] == 0:
            return {
                'num_instances': 0,
                'instance_centers': [],
                'instance_sizes': [],
                'total_points_saved': 0,
                'noise_points_filtered': 0
            }
        
        labels = cluster_result['labels']
        target_xyz = cluster_result['target_xyz']
        target_indices = cluster_result['target_indices']
        
        # 生成实例颜色（使用HSV空间生成区分度高的颜色）
        import colorsys
        unique_labels = sorted(set(labels))
        unique_labels = [l for l in unique_labels if l >= 0]  # 排除噪声
        
        # 过滤小聚类
        valid_labels = []
        for label in unique_labels:
            cluster_size = (labels == label).sum()
            if cluster_size >= min_cluster_size:
                valid_labels.append(label)
        
        if not valid_labels:
            return {
                'num_instances': 0,
                'instance_centers': [],
                'instance_sizes': [],
                'total_points_saved': 0,
                'noise_points_filtered': len(target_xyz)
            }
        
        # 为每个有效聚类分配颜色
        num_valid = len(valid_labels)
        colors = {}
        for i, label in enumerate(valid_labels):
            hue = i / max(num_valid, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors[label] = (int(r * 255), int(g * 255), int(b * 255))
        
        # 收集要保存的点
        points_to_save = []
        colors_to_save = []
        instance_centers = []
        instance_sizes = []
        
        for label in valid_labels:
            cluster_mask = labels == label
            cluster_xyz = target_xyz[cluster_mask]
            cluster_color = colors[label]
            
            for xyz in cluster_xyz:
                points_to_save.append(xyz)
                colors_to_save.append(cluster_color)
            
            instance_centers.append(cluster_xyz.mean(axis=0))
            instance_sizes.append(len(cluster_xyz))
        
        noise_count = (labels == -1).sum()
        small_cluster_count = len(target_xyz) - len(points_to_save) - noise_count
        
        if not points_to_save:
            return {
                'num_instances': 0,
                'instance_centers': [],
                'instance_sizes': [],
                'total_points_saved': 0,
                'noise_points_filtered': len(target_xyz)
            }
        
        # 写入PLY文件
        mkdir_p(os.path.dirname(path))
        
        points_arr = np.array(points_to_save)
        colors_arr = np.array(colors_to_save, dtype=np.uint8)
        
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        
        elements = np.empty(len(points_arr), dtype=dtype)
        elements['x'] = points_arr[:, 0]
        elements['y'] = points_arr[:, 1]
        elements['z'] = points_arr[:, 2]
        elements['red'] = colors_arr[:, 0]
        elements['green'] = colors_arr[:, 1]
        elements['blue'] = colors_arr[:, 2]
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        return {
            'num_instances': len(valid_labels),
            'instance_centers': instance_centers,
            'instance_sizes': instance_sizes,
            'total_points_saved': len(points_to_save),
            'noise_points_filtered': noise_count + small_cluster_count
        }
    
    def save_target_gaussians_with_clustering(self, base_path, threshold=0.4,
                                               eps=0.15, min_samples=10,
                                               min_cluster_size=150):
        """
        同时保存聚类前和聚类后的目标高斯PLY文件，以及实例信息txt文件。
        
        Args:
            base_path: 基础路径（不含扩展名），将生成:
                       - {base_path}_raw.ply (聚类前)
                       - {base_path}_clustered.ply (聚类后，带颜色)
                       - {base_path}_instances.txt (实例中心坐标)
            threshold: 目标高斯的语义阈值
            eps: DBSCAN邻域半径
            min_samples: DBSCAN最小邻居数
            min_cluster_size: 最小聚类大小
            
        Returns:
            dict: 保存结果信息
        """
        import os
        
        # 确保目录存在
        base_dir = os.path.dirname(base_path)
        if base_dir:
            mkdir_p(base_dir)
        
        result = {
            'raw_ply_path': None,
            'clustered_ply_path': None,
            'instances_txt_path': None,
            'raw_count': 0,
            'clustered_count': 0,
            'num_instances': 0,
            'instance_centers': []
        }
        
        # 1. 保存聚类前的原始目标高斯
        raw_path = f"{base_path}_raw.ply"
        raw_count = self.save_target_gaussians_ply(raw_path, threshold)
        result['raw_ply_path'] = raw_path
        result['raw_count'] = raw_count
        
        # 2. 保存聚类后的目标高斯（带实例颜色）
        clustered_path = f"{base_path}_clustered.ply"
        cluster_info = self.save_clustered_target_gaussians_ply(
            clustered_path, threshold, eps, min_samples, min_cluster_size
        )
        result['clustered_ply_path'] = clustered_path
        result['clustered_count'] = cluster_info['total_points_saved']
        result['num_instances'] = cluster_info['num_instances']
        result['instance_centers'] = cluster_info['instance_centers']
        result['instance_sizes'] = cluster_info['instance_sizes']
        
        # 3. 保存实例信息txt文件
        txt_path = f"{base_path}_instances.txt"
        result['instances_txt_path'] = txt_path
        
        with open(txt_path, 'w') as f:
            f.write(f"# Target Semantic Gaussians Instance Information\n")
            f.write(f"# threshold={threshold}, eps={eps}, min_samples={min_samples}, min_cluster_size={min_cluster_size}\n")
            f.write(f"# ============================================\n")
            f.write(f"Total raw target Gaussians: {raw_count}\n")
            f.write(f"Total clustered Gaussians: {cluster_info['total_points_saved']}\n")
            f.write(f"Noise/small clusters filtered: {cluster_info['noise_points_filtered']}\n")
            f.write(f"Number of instances: {cluster_info['num_instances']}\n")
            f.write(f"\n")
            f.write(f"# Instance Centers (x, y, z) and sizes:\n")
            f.write(f"# ============================================\n")
            
            for i, (center, size) in enumerate(zip(
                cluster_info['instance_centers'], 
                cluster_info['instance_sizes']
            )):
                f.write(f"Instance {i}: center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), "
                        f"size={size} points\n")
        
        return result
