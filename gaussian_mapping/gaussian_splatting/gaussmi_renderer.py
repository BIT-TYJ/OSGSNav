"""Faithful GauSS-MI rasterization on IEVE's shared Gaussian map."""

import importlib.util
import math
import sys
from pathlib import Path

import torch

from .utils.sh_utils import eval_sh


_MODULE_NAME = "_ieve_gaussmi_diff_rasterization"


def _load_gaussmi_rasterization():
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module

    package_dir = (
        Path(__file__).resolve().parents[2]
        / "3rdparty"
        / "GauSS-MI"
        / "submodules"
        / "diff-gaussian-rasterization-gaussmi"
        / "diff_gaussian_rasterization"
    )
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise ImportError("GauSS-MI rasterizer source is not initialized")
    if not list(package_dir.glob("_C*.so")):
        raise ImportError(
            "GauSS-MI rasterizer is not built; run setup.py build_ext --inplace "
            "under 3rdparty/GauSS-MI/submodules/diff-gaussian-rasterization-gaussmi"
        )

    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        str(init_path),
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load GauSS-MI rasterizer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


def is_available():
    try:
        _load_gaussmi_rasterization()
        return True
    except Exception:
        return False


def render_gaussmi(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    *,
    information_gain=None,
    loss_image=None,
    scaling_modifier=1.0,
):
    """Render GauSS-MI exactly as the original renderer.

    ``information_gain`` is the original per-Gaussian ``-log(reliability)``
    tensor, optionally multiplied by IEVE semantic weights before rasterizing.
    """

    if pc.get_xyz.shape[0] == 0:
        return None

    rasterization = _load_gaussmi_rasterization()
    screenspace_points = (
        torch.zeros_like(
            pc.get_xyz,
            dtype=pc.get_xyz.dtype,
            requires_grad=True,
            device=pc.get_xyz.device,
        )
        + 0
    )

    raster_settings = rasterization.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
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
    rasterizer = rasterization.GaussianRasterizer(
        raster_settings=raster_settings
    )

    scales = None
    rotations = None
    cov3d_precomp = None
    if pipe.compute_cov3D_python:
        cov3d_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        if scales.shape[-1] == 1:
            scales = scales.repeat(1, 3)
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        direction = pc.get_xyz - viewpoint_camera.camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        direction = direction / direction.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(
            pc.active_sh_degree, shs_view, direction
        )
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        shs = pc.get_features
    if information_gain is None:
        information_gain = pc.get_infogain
    information_gain = information_gain.to(
        device=pc.get_xyz.device, dtype=torch.float32
    ).reshape(pc.get_xyz.shape[0], 4)

    if loss_image is None:
        loss_image = torch.zeros(
            (viewpoint_camera.image_height, viewpoint_camera.image_width),
            dtype=torch.float32,
            device=pc.get_xyz.device,
        )
    else:
        loss_image = loss_image.to(
            device=pc.get_xyz.device, dtype=torch.float32
        ).reshape(viewpoint_camera.image_height, viewpoint_camera.image_width)

    (
        rendered_image,
        radii,
        depth,
        opacity,
        n_touched,
        mutual_info,
        render_loss,
    ) = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3d_precomp,
        unreliabilities=information_gain,
        loss_image=loss_image,
    )
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth,
        "opacity": opacity,
        "n_touched": n_touched,
        "mutual_info": mutual_info,
        "render_loss": render_loss,
    }
