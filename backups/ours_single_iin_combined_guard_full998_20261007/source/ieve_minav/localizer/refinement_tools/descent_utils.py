# Copyright (c) 2024 Imperial College London. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This code is based on techniques described in the following publication:
#
# Hidenobu Matsuki, Riku Murai, Paul H.J. Kelly, & Andrew J. Davison. Gaussian Splatting SLAM. CVPR, 2024.

import torch
import torch.nn.functional as F


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def _resize_to_match(tensor, target_hw, mode='nearest'):
    """Resize tensor to target (H, W). Handles Bool and arbitrary dims."""
    if tensor.shape[-2:] == target_hw:
        return tensor
    was_bool = (tensor.dtype == torch.bool)
    t = tensor.float()
    while t.dim() < 4:
        t = t.unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode=mode)
    while t.dim() > tensor.dim():
        t = t.squeeze(0)
    if was_bool:
        t = t > 0.5
    return t


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    render_hw = image.shape[-2:]  # use rendered image resolution as reference

    # Resize gt_image and masks to match rendered image resolution
    gt_image = _resize_to_match(gt_image, render_hw, mode='bilinear')
    opacity_mask = (opacity > config["Training"]["opacity_threshold"]).view(*depth.shape)
    opacity_mask = _resize_to_match(opacity_mask, render_hw)
    _grad_mask = _resize_to_match(viewpoint.grad_mask, render_hw)
    l1 = opacity_mask * torch.abs(image * _grad_mask - gt_image * _grad_mask)
    return l1.mean()


def _viewpoint_depth_to_tensor(viewpoint_depth, device):
    if viewpoint_depth is None:
        return None

    if torch.is_tensor(viewpoint_depth):
        gt_depth = viewpoint_depth.detach().to(dtype=torch.float32, device=device)
    else:
        gt_depth = torch.as_tensor(viewpoint_depth, dtype=torch.float32, device=device)

    if gt_depth.dim() == 2:
        gt_depth = gt_depth.unsqueeze(0)
    elif gt_depth.dim() == 3:
        if gt_depth.shape[-1] == 1 and gt_depth.shape[0] != 1:
            gt_depth = gt_depth.permute(2, 0, 1)
        elif gt_depth.shape[0] != 1:
            gt_depth = gt_depth[:1]
    elif gt_depth.dim() == 4:
        gt_depth = gt_depth[0]
        if gt_depth.dim() == 2:
            gt_depth = gt_depth.unsqueeze(0)
        elif gt_depth.shape[0] != 1:
            gt_depth = gt_depth[:1]

    return gt_depth


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.98

    gt_depth = _viewpoint_depth_to_tensor(viewpoint.depth, image.device)
    if gt_depth is None:
        return get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)

    render_hw = image.shape[-2:]
    gt_depth = _resize_to_match(gt_depth, render_hw)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > config["Training"]["opacity_threshold"]).view(*depth.shape)
    opacity_mask = _resize_to_match(opacity_mask, render_hw)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    _grad_mask_d = _resize_to_match(viewpoint.grad_mask, render_hw)
    depth_mask = depth_pixel_mask * opacity_mask * _grad_mask_d
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return l1_rgb + (1 - alpha) * l1_depth.mean()

