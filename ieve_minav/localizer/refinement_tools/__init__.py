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

import math
import logging

_logger = logging.getLogger(__name__)

try:
    import torch
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import GaussianModel
    from gaussian_mapping.gaussian_splatting.utils.sh_utils import eval_sh

    _RENDER_AVAILABLE = True
except ImportError as e:
    _logger.warning(
        f"refinement_tools: CUDA rendering dependencies not available: {e}. "
        f"render() will raise ImportError if called."
    )
    _RENDER_AVAILABLE = False


def _apply_se3_perturbation(means3D, viewpoint_camera):
    """
    在 Python 层将 cam_rot_delta / cam_trans_delta 以 SE(3) 微扰
    可微地作用到 3D 高斯位置上，使梯度能回传到位姿参数。

    等价变换：
        p_cam_new = delta · T_CW · p_world
      = T_CW · (T_CW^-1 · delta · T_CW) · p_world
      = T_CW · M · p_world

    因此只需将 means3D 替换为 M · means3D（可微），rasterizer 保持原 viewmatrix。

    Returns:
        means3D_out: 变换后的 3D 位置（若无微扰则返回原始 means3D）
    """
    theta = getattr(viewpoint_camera, 'cam_rot_delta', None)
    rho = getattr(viewpoint_camera, 'cam_trans_delta', None)

    # 没有 delta 或不需要梯度时跳过
    if theta is None or rho is None:
        return means3D
    if not (theta.requires_grad or rho.requires_grad):
        return means3D

    from ieve_minav.localizer.refinement_tools.pose_utils import SE3_exp

    tau = torch.cat([rho, theta])          # [rho(3), theta(3)]
    delta_T = SE3_exp(tau)                 # 4x4, 可微

    # W2C（行主序）— detach，不对 viewmatrix 本身求导
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1).detach()
    c2w = torch.linalg.inv(w2c)

    # 世界坐标系下的等价变换  M = C2W · delta · W2C
    M = c2w @ delta_T @ w2c                # 4x4, 梯度经 delta_T -> tau -> theta/rho

    # 齐次坐标变换
    ones = torch.ones(means3D.shape[0], 1,
                      device=means3D.device, dtype=means3D.dtype)
    pts_h = torch.cat([means3D.detach(), ones], dim=1)   # (N, 4)
    means3D_out = (M @ pts_h.T).T[:, :3].contiguous()    # (N, 3), CUDA kernel needs contiguous memory
    return means3D_out


def render(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    override_color=None,
    mask=None,
):
    """
    Render the scene.

    theta/rho 位姿微扰在 Python 层通过 SE(3) 变换可微地作用到 3D 高斯位置上，
    与 GauSS-MI rasterizer（不含 theta/rho 参数）兼容。
    Background tensor (bg_color) must be on GPU!
    """
    if not _RENDER_AVAILABLE:
        raise ImportError(
            "Cannot call render(): diff_gaussian_rasterization or "
            "gaussian_splatting packages are not installed."
        )

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if pc.get_xyz.shape[0] == 0:
        return None

    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # ---- 在 Python 层应用 SE(3) 微扰（可微） ----
    means3D = _apply_se3_perturbation(pc.get_xyz, viewpoint_camera)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if pc.get_scaling.shape[-1] == 1:
            scales = pc.get_scaling.repeat(1, 3)
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    def _rasterize(**kwargs):
        try:
            num_gaussians = kwargs["means3D"].shape[0]
            kwargs["unreliabilities"] = (
                torch.ones(num_gaussians, 4, dtype=torch.float32, device="cuda") * 0.5
            )
            kwargs["loss_image"] = torch.zeros(
                int(viewpoint_camera.image_height),
                int(viewpoint_camera.image_width),
                dtype=torch.float32,
                device="cuda",
            )
            return rasterizer(**kwargs)
        except TypeError:
            kwargs.pop("unreliabilities", None)
            kwargs.pop("loss_image", None)
            return rasterizer(**kwargs)

    def _unpack_render_result(render_result):
        if len(render_result) == 7:
            rendered_image, radii, depth, opacity_out, n_touched, _, _ = render_result
        elif len(render_result) == 5:
            rendered_image, radii, depth, opacity_out, n_touched = render_result
        elif len(render_result) == 4:
            rendered_image, radii, depth, opacity_out = render_result
            n_touched = torch.zeros_like(radii, dtype=torch.int32)
        else:
            raise RuntimeError(
                f"Unsupported diff_gaussian_rasterization return length: {len(render_result)}"
            )
        return rendered_image, radii, depth, opacity_out, n_touched

    if mask is not None:
        rast_out = _rasterize(
            means3D=means3D[mask],
            means2D=means2D[mask],
            shs=shs[mask] if shs is not None else None,
            colors_precomp=colors_precomp[mask] if colors_precomp is not None else None,
            opacities=opacity[mask],
            scales=scales[mask] if scales is not None else None,
            rotations=rotations[mask] if rotations is not None else None,
            cov3D_precomp=cov3D_precomp[mask] if cov3D_precomp is not None else None,
        )
    else:
        rast_out = _rasterize(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

    rendered_image, radii, depth, opacity_out, n_touched = _unpack_render_result(rast_out)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity_out,
        "n_touched": n_touched,
    }
