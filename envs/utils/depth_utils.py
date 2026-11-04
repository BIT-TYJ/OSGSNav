# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for processing depth images.
"""
from argparse import Namespace

import itertools
import numpy as np
import torch

import envs.utils.rotation_utils as ru


def get_camera_matrix(width, height, fov):
    """Returns a camera matrix from image size and fov."""
    xc = (width - 1.) / 2.
    zc = (height - 1.) / 2.
    f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
    camera_matrix = {'xc': xc, 'zc': zc, 'f': f}
    camera_matrix = Namespace(**camera_matrix)
    return camera_matrix


def get_point_cloud_from_z(Y, camera_matrix, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW   
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    x, z = np.meshgrid(np.arange(Y.shape[-1]),
                       np.arange(Y.shape[-2] - 1, -1, -1))
    for _ in range(Y.ndim - 2):
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(z, axis=0)
    X = (x[::scale, ::scale] - camera_matrix.xc) * \
        Y[::scale, ::scale] / camera_matrix.f
    Z = (z[::scale, ::scale] - camera_matrix.zc) * \
        Y[::scale, ::scale] / camera_matrix.f
    XYZ = np.concatenate((X[..., np.newaxis],
                          Y[::scale, ::scale][..., np.newaxis],
                          Z[..., np.newaxis]), axis=X.ndim)
    return XYZ


def transform_camera_view(XYZ, sensor_height, camera_elevation_degree):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[..., 2] = XYZ[..., 2] + sensor_height
    return XYZ


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
    XYZ[:, :, 0] = XYZ[:, :, 0] + current_pose[0]
    XYZ[:, :, 1] = XYZ[:, :, 1] + current_pose[1]
    return XYZ


def bin_points(XYZ_cms, map_size, z_bins, xy_resolution):
    """Bins points into xy-z bins
    XYZ_cms is ... x H x W x3
    Outputs is ... x map_size x map_size x (len(z_bins)+1)
    """
    sh = XYZ_cms.shape
    XYZ_cms = XYZ_cms.reshape([-1, sh[-3], sh[-2], sh[-1]])
    n_z_bins = len(z_bins) + 1
    counts = []
    for XYZ_cm in XYZ_cms:
        isnotnan = np.logical_not(np.isnan(XYZ_cm[:, :, 0]))
        X_bin = np.round(XYZ_cm[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ_cm[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ_cm[:, :, 2], bins=z_bins).astype(np.int32)

        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0,
                            Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        count = np.bincount(ind.ravel(), isvalid.ravel().astype(np.int32),
                            minlength=map_size * map_size * n_z_bins)
        counts = np.reshape(count, [map_size, map_size, n_z_bins])

    counts = counts.reshape(list(sh[:-3]) + [map_size, map_size, n_z_bins])

    return counts


def get_point_cloud_from_z_t(Y_t, camera_matrix, device, scale=1):
    """Projects the depth image Y into a 3D point cloud.
    Inputs:
        Y is ...xHxW
        camera_matrix
    Outputs:
        X is positive going right
        Y is positive into the image
        Z is positive up in the image
        XYZ is ...xHxWx3
    """
    grid_x, grid_z = torch.meshgrid(torch.arange(Y_t.shape[-1]),
                                    torch.arange(Y_t.shape[-2] - 1, -1, -1))
    grid_x = grid_x.transpose(1, 0).to(device)
    grid_z = grid_z.transpose(1, 0).to(device)
    grid_x = grid_x.unsqueeze(0).expand(Y_t.size())
    grid_z = grid_z.unsqueeze(0).expand(Y_t.size())

    X_t = (grid_x[:, ::scale, ::scale] - camera_matrix.xc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f
    Z_t = (grid_z[:, ::scale, ::scale] - camera_matrix.zc) * \
        Y_t[:, ::scale, ::scale] / camera_matrix.f

    XYZ = torch.stack(
        (X_t, Y_t[:, ::scale, ::scale], Z_t), dim=len(Y_t.size()))
    
    return XYZ


def transform_camera_view_t(
        XYZ, sensor_height, camera_elevation_degree, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera elevation and angle
    Input:
        XYZ                     : ...x3
        sensor_height           : height of the sensor
        camera_elevation_degree : camera elevation to rectify.
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix(
        [1., 0., 0.], angle=np.deg2rad(camera_elevation_degree))
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    # XYZ[..., 2] = XYZ[..., 2] + sensor_height
    height_ex = sensor_height.view(sensor_height.shape[0], 1, 1, 1)
    height_ex = height_ex.expand(XYZ.shape[0], XYZ.shape[1], XYZ.shape[2], XYZ.shape[3])
    XYZ[..., 2] = XYZ[..., 2] + height_ex[..., 2]
    return XYZ


def transform_pose_t(XYZ, current_pose, device):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = ru.get_r_matrix([0., 0., 1.], angle=current_pose[2] - np.pi / 2.)
    XYZ = torch.matmul(XYZ.reshape(-1, 3),
                       torch.from_numpy(R).float().transpose(1, 0).to(device)
                       ).reshape(XYZ.shape)
    XYZ[..., 0] += current_pose[0]
    XYZ[..., 1] += current_pose[1]
    return XYZ


def splat_feat_nd(init_grid, feat, coords):
    """
    Args:
        init_grid: B X nF X W X H X D X ..
        feat: B X nF X nPt
        coords: B X nDims X nPt in [-1, 1]
    Returns:
        grid: B X nF X W X H X D X ..
    """
    wts_dim = []
    pos_dim = []
    grid_dims = init_grid.shape[2:]

    B = init_grid.shape[0]
    F = init_grid.shape[1]

    n_dims = len(grid_dims)

    grid_flat = init_grid.view(B, F, -1)

    for d in range(n_dims):
        pos = coords[:, [d], :] * grid_dims[d] / 2 + grid_dims[d] / 2
        pos_d = []
        wts_d = []

        for ix in [0, 1]:
            pos_ix = torch.floor(pos) + ix
            safe_ix = (pos_ix > 0) & (pos_ix < grid_dims[d])
            safe_ix = safe_ix.type(pos.dtype)

            wts_ix = 1 - torch.abs(pos - pos_ix)

            wts_ix = wts_ix * safe_ix
            pos_ix = pos_ix * safe_ix

            pos_d.append(pos_ix)
            wts_d.append(wts_ix)

        pos_dim.append(pos_d)
        wts_dim.append(wts_d)

    l_ix = [[0, 1] for d in range(n_dims)]

    for ix_d in itertools.product(*l_ix):
        wts = torch.ones_like(wts_dim[0][0])
        index = torch.zeros_like(wts_dim[0][0])
        for d in range(n_dims):
            index = index * grid_dims[d] + pos_dim[d][ix_d[d]]
            wts = wts * wts_dim[d][ix_d[d]]

        index = index.long()
        grid_flat.scatter_add_(2, index.expand(-1, F, -1), feat * wts)
        grid_flat = torch.round(grid_flat)

    return grid_flat.view(init_grid.shape)


def _draw_line_cells(mask, x0, y0, x1, y1):
    """Rasterize one 2D grid line into mask using integer Bresenham."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    h, w = mask.shape

    while True:
        if 0 <= x < w and 0 <= y < h:
            mask[y, x] = 1
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def raycast_local_2d_map_from_agent_view(
        agent_view_centered_t,
        vision_range,
        xy_resolution,
        obstacle_height_min_cm=5.0,
        obstacle_height_max_cm=150.0,
        ray_sample_rate=5):
    """
    Build IEVE's local obstacle/explored patch with MinAV-style 2D ray casting.

    Args:
        agent_view_centered_t: B x H x W x 3 point cloud in centimeters after
            transform_pose_t(..., shift_loc). X maps to local columns, Y maps
            to local rows in the current agent-view patch, and Z is height.
        vision_range: local patch size in grid cells.
        xy_resolution: map resolution in centimeters per grid cell.
        obstacle_height_min_cm: minimum height treated as obstacle evidence.
        obstacle_height_max_cm: maximum height treated as obstacle evidence.
        ray_sample_rate: endpoint subsampling for free-space ray casting.

    Returns:
        fp_map_pred: B x 1 x vision_range x vision_range obstacle patch.
        fp_exp_pred: B x 1 x vision_range x vision_range explored patch.
    """
    device = agent_view_centered_t.device
    dtype = agent_view_centered_t.dtype
    pc = agent_view_centered_t.detach().cpu().numpy()
    bs = pc.shape[0]
    vr = int(vision_range)
    res = float(xy_resolution)
    ray_sample_rate = max(1, int(ray_sample_rate))

    fp_map = np.zeros((bs, 1, vr, vr), dtype=np.float32)
    fp_exp = np.zeros((bs, 1, vr, vr), dtype=np.float32)
    agent_x = int(vr // 2)
    agent_y = 0

    try:
        import cv2
    except Exception:
        cv2 = None

    for b in range(bs):
        points = pc[b]
        finite = np.isfinite(points).all(axis=-1)
        if not np.any(finite):
            continue

        safe_x = np.where(finite, points[..., 0], 0.0)
        safe_y = np.where(finite, points[..., 1], 0.0)
        x_bin = np.rint(safe_x / res).astype(np.int32)
        y_bin = np.rint(safe_y / res).astype(np.int32)
        z_cm = points[..., 2]

        in_bounds = (
            finite
            & (x_bin >= 0) & (x_bin < vr)
            & (y_bin >= 0) & (y_bin < vr)
        )
        if not np.any(in_bounds):
            continue

        endpoint_x = x_bin[in_bounds].reshape(-1)
        endpoint_y = y_bin[in_bounds].reshape(-1)
        endpoint_x = endpoint_x[::ray_sample_rate]
        endpoint_y = endpoint_y[::ray_sample_rate]

        free_mask = np.zeros((vr, vr), dtype=np.uint8)
        if cv2 is not None:
            for x1, y1 in zip(endpoint_x, endpoint_y):
                cv2.line(free_mask, (agent_x, agent_y), (int(x1), int(y1)), 1, 1)
        else:
            for x1, y1 in zip(endpoint_x, endpoint_y):
                _draw_line_cells(free_mask, agent_x, agent_y, int(x1), int(y1))

        obstacle = (
            in_bounds
            & (z_cm >= float(obstacle_height_min_cm))
            & (z_cm <= float(obstacle_height_max_cm))
        )
        obs_y = y_bin[obstacle]
        obs_x = x_bin[obstacle]
        if obs_x.size > 0:
            fp_map[b, 0, obs_y, obs_x] = 1.0

        explored = free_mask > 0
        if obs_x.size > 0:
            explored[obs_y, obs_x] = True
        fp_exp[b, 0] = explored.astype(np.float32)

    return (
        torch.from_numpy(fp_map).to(device=device, dtype=dtype),
        torch.from_numpy(fp_exp).to(device=device, dtype=dtype),
    )


def _rotation_matrix_from_xyzw(rotation_xyzw):
    try:
        from scipy.spatial.transform import Rotation as R
        return R.from_quat(rotation_xyzw).as_matrix().astype(np.float32)
    except Exception:
        x, y, z, w = [float(v) for v in rotation_xyzw]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.asarray([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ], dtype=np.float32)


def update_minav_full_grid_from_depth(
        obstacle_count,
        free_count,
        occupancy_map,
        explored_map,
        depth_m,
        agent_position,
        rotation_xyzw,
        episode_origin_xy,
        map_size_cm,
        map_resolution,
        fov_deg,
        camera_height,
        obstacle_height_min_m=0.05,
        obstacle_height_max_m=1.5,
        valid_depth_min_m=0.05,
        valid_depth_max_m=10.0,
        point_sample_step=2,
        ray_sample_rate=5,
        robot_radius_m=0.17):
    """
    Update IEVE full-map channels with MinAV's direct 2D occupancy logic.

    The state arrays stay in IEVE full-map coordinates. Only the projection
    and evidence-fusion rules mirror MinAV's OccupancyMapper.
    """
    stats = {
        "updated": False,
        "reason": "",
        "free_pixels": 0,
        "obstacle_points": 0,
        "robot_cell": None,
    }

    if depth_m is None or agent_position is None or rotation_xyzw is None:
        stats["reason"] = "missing_depth_or_pose"
        return stats

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
    if depth.ndim != 2:
        stats["reason"] = "invalid_depth_shape"
        return stats

    position = np.asarray(agent_position, dtype=np.float32).reshape(-1)
    rotation = np.asarray(rotation_xyzw, dtype=np.float32).reshape(-1)
    origin = np.asarray(episode_origin_xy, dtype=np.float32).reshape(-1)
    if (
        position.size < 3
        or rotation.size < 4
        or origin.size < 2
        or not np.all(np.isfinite(position[:3]))
        or not np.all(np.isfinite(rotation[:4]))
        or not np.all(np.isfinite(origin[:2]))
    ):
        stats["reason"] = "invalid_pose"
        return stats

    H, W = depth.shape
    step = max(1, int(point_sample_step))
    ray_sample_rate = max(1, int(ray_sample_rate))
    valid_mask = (
        np.isfinite(depth)
        & (depth > float(valid_depth_min_m))
        & (depth < float(valid_depth_max_m))
    )
    if not np.any(valid_mask):
        stats["reason"] = "no_valid_depth"
        return stats

    sample_depth = depth[::step, ::step]
    sample_valid = valid_mask[::step, ::step]
    if not np.any(sample_valid):
        stats["reason"] = "no_valid_sampled_depth"
        return stats

    v_coords = np.arange(0, H, step, dtype=np.float32)
    u_coords = np.arange(0, W, step, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    fov = np.deg2rad(float(fov_deg))
    fx = W / (2.0 * np.tan(fov / 2.0))
    fy = fx
    cx = W / 2.0
    cy = H / 2.0

    z = sample_depth[sample_valid]
    x = (u_grid[sample_valid] - cx) * z / fx
    y = (v_grid[sample_valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    camera_to_robot = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    points_robot = points @ camera_to_robot.T
    points_robot[:, 1] += float(camera_height)
    rot_matrix = _rotation_matrix_from_xyzw(rotation[:4])
    points_world = points_robot @ rot_matrix.T + position[:3]

    full_h, full_w = occupancy_map.shape
    map_center_m = float(map_size_cm) / 100.0 / 2.0
    scaler = 100.0 / float(map_resolution)

    # IEVE full-map pose uses x=-habitat_z and y=-habitat_x relative to the
    # episode start. Keep that frame so visualization and planning stay native.
    map_x_m = map_center_m + (-points_world[:, 2]) - float(origin[0])
    map_y_m = map_center_m + (-points_world[:, 0]) - float(origin[1])
    map_x = (map_x_m * scaler).astype(np.int32)
    map_y = (map_y_m * scaler).astype(np.int32)

    robot_x_m = map_center_m + (-float(position[2])) - float(origin[0])
    robot_y_m = map_center_m + (-float(position[0])) - float(origin[1])
    robot_x = int(robot_x_m * scaler)
    robot_y = int(robot_y_m * scaler)
    stats["robot_cell"] = [int(robot_y), int(robot_x)]

    in_bounds = (
        (map_x >= 0) & (map_x < full_w)
        & (map_y >= 0) & (map_y < full_h)
    )
    if not np.any(in_bounds):
        stats["reason"] = "all_points_out_of_bounds"
        return stats

    map_x = map_x[in_bounds]
    map_y = map_y[in_bounds]
    rel_h = points_world[:, 1][in_bounds] - float(position[1])

    free_mask_img = np.zeros_like(occupancy_map, dtype=np.uint8)
    if 0 <= robot_x < full_w and 0 <= robot_y < full_h:
        target_x = map_x[::ray_sample_rate]
        target_y = map_y[::ray_sample_rate]
        try:
            import cv2
        except Exception:
            cv2 = None
        if cv2 is not None:
            for x1, y1 in zip(target_x, target_y):
                cv2.line(
                    free_mask_img,
                    (robot_x, robot_y),
                    (int(x1), int(y1)),
                    1,
                    1,
                )
        else:
            for x1, y1 in zip(target_x, target_y):
                _draw_line_cells(
                    free_mask_img,
                    robot_x,
                    robot_y,
                    int(x1),
                    int(y1),
                )

    free_pixels = free_mask_img > 0
    free_count[free_pixels] += 1
    stats["free_pixels"] = int(np.count_nonzero(free_pixels))

    obstacle = (
        (rel_h >= float(obstacle_height_min_m))
        & (rel_h <= float(obstacle_height_max_m))
    )
    if np.any(obstacle):
        obs_x = map_x[obstacle]
        obs_y = map_y[obstacle]
        np.add.at(obstacle_count, (obs_y, obs_x), 1)
        stats["obstacle_points"] = int(obs_x.size)

    obstacle_mask = (
        (obstacle_count >= 2)
        & (obstacle_count > free_count * 0.3)
    )
    occupancy_map[obstacle_mask] = 1.0

    free_mask = (
        (free_count > 0)
        & (obstacle_count < free_count * 0.1)
    )
    occupancy_map[free_mask] = 0.0

    explored_mask = (free_count > 0) | (obstacle_count > 0)
    explored_map[explored_mask] = 1.0

    robot_radius_cells = max(
        1,
        int(np.ceil(float(robot_radius_m) * 100.0 / float(map_resolution))),
    )
    if 0 <= robot_x < full_w and 0 <= robot_y < full_h:
        for dy in range(-robot_radius_cells, robot_radius_cells + 1):
            for dx in range(-robot_radius_cells, robot_radius_cells + 1):
                if dx * dx + dy * dy > robot_radius_cells * robot_radius_cells:
                    continue
                nx, ny = robot_x + dx, robot_y + dy
                if 0 <= nx < full_w and 0 <= ny < full_h:
                    if obstacle_count[ny, nx] > max(3, free_count[ny, nx]):
                        continue
                    occupancy_map[ny, nx] = 0.0
                    explored_map[ny, nx] = 1.0

    stats["updated"] = True
    stats["reason"] = "ok"
    return stats
