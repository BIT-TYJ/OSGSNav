"""State helpers for sequential UniGoal without modifying upstream UniGoal."""

import json
import math
import os

import numpy as np
from PIL import Image, ImageDraw


# RGB equivalents of UniGoal's semantic-map palette.
_PALETTE = np.asarray([
    [255, 255, 255], [153, 153, 153], [242, 242, 242],
    [245, 92, 66], [31, 120, 180], [240, 199, 168],
    [226, 240, 168], [168, 240, 217], [182, 168, 240],
    [235, 168, 240], [240, 168, 191],
], dtype=np.uint8)


_TARGET1_FULL_MAPS = {}


def _translate_mask(mask, dx, dy):
    import cv2

    matrix = np.asarray([[1.0, 0.0, float(dx)],
                         [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), matrix,
                          (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool)


def _stabilize_target2_obstacles(full, reference, max_shift=6, min_support=32):
    """Keep target-1 walls fixed and reject their translated re-projections."""
    import cv2

    obstacle_ref = np.rint(reference[0]) == 1
    obstacle_now = np.rint(full[0]) == 1
    old_explored = np.rint(reference[1]) == 1
    added = obstacle_now & ~obstacle_ref
    overlap_added = added & old_explored
    reference_band = cv2.dilate(
        obstacle_ref.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    best = (0, 0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            score = int(np.count_nonzero(
                _translate_mask(overlap_added, dx, dy) & reference_band))
            if (score, -abs(dx) - abs(dy), dx, dy) > (
                    best[2], -abs(best[0]) - abs(best[1]), best[0], best[1]):
                best = (dx, dy, score)
    dx, dy, score = best
    if score >= min_support:
        matching_positions = _translate_mask(reference_band, -dx, -dy)
        duplicates = overlap_added & matching_positions
    else:
        dx, dy = 0, 0
        duplicates = np.zeros_like(overlap_added)
    corrected = full.copy()
    corrected[0, duplicates] = reference[0, duplicates]
    return corrected, {
        "estimated_target2_correction_xy_px": [int(dx), int(dy)],
        "registration_support_pixels": int(score),
        "minimum_support_pixels": int(min_support),
        "removed_duplicate_obstacle_pixels": int(duplicates.sum()),
        "policy": "target1 obstacle boundary fixed in overlap",
    }


def retarget_graph_preserving_memory(graph):
    """Clear goal-conditioned scores while retaining the observed scene graph."""
    for node in getattr(graph, "nodes", []):
        node.reason = None
        node.distance = 2
        node.score = 0.5
        node.exploration_level = 0
    for room in getattr(graph, "room_nodes", []):
        room.exploration_level = 0
        room.group_nodes = []
    graph.group_nodes = []
    for name in ("matcher", "subgraph", "mid_term_goal", "frontier_locations",
                 "frontier_locations_16", "scores"):
        if hasattr(graph, name):
            delattr(graph, name)


def _compose_full_map(bev_map):
    """Return a synchronized copy; never mutate the planner's live map."""
    full = bev_map.full_map[0].detach().cpu().numpy().copy()
    local = bev_map.local_map[0].detach().cpu().numpy()
    r1, r2, c1, c2 = [int(v) for v in bev_map.local_map_boundary[0]]
    full[:, r1:r2, c1:c2] = local
    return full


def _semantic_rgb(full, path_mask=None):
    obstacle = np.rint(full[0]) == 1
    explored = np.rint(full[1]) == 1
    semantic_scores = full[4:11]
    semantic = semantic_scores.argmax(axis=0) + 5
    # Completely unobserved full-map cells have all-zero semantic channels.
    # UniGoal's local visualizer seeds its final channel with 1e-5, but old
    # global tiles do not, so treat both representations as no-category.
    no_category = np.logical_or(semantic_scores.max(axis=0) <= 0,
                                semantic == 11)
    semantic[no_category] = 0
    semantic[np.logical_and(no_category, explored)] = 2
    semantic[np.logical_and(no_category, obstacle)] = 1
    if path_mask is not None:
        semantic[np.asarray(path_mask) > 0] = 3
    semantic = np.clip(semantic, 0, len(_PALETTE) - 1)
    return _PALETTE[semantic][::-1].copy(), explored[::-1]


def _draw_agent(image, full_pose, map_resolution):
    draw = ImageDraw.Draw(image)
    x_m, y_m, heading_deg = [float(v) for v in full_pose]
    x = x_m * 100.0 / map_resolution
    y = image.height - 1 - y_m * 100.0 / map_resolution
    heading = math.radians(-heading_deg)
    size = 11
    points = [
        (x + size * math.cos(heading), y + size * math.sin(heading)),
        (x + size * 0.65 * math.cos(heading + 2.45),
         y + size * 0.65 * math.sin(heading + 2.45)),
        (x + size * 0.65 * math.cos(heading - 2.45),
         y + size * 0.65 * math.sin(heading - 2.45)),
    ]
    draw.polygon(points, fill=(0, 117, 170), outline=(0, 55, 85))


def save_terminal_global_maps(bev_map, agent, output_dir, episode_id,
                              target_index, success, spl):
    """Save fixed-global terminal maps, plus a paper-friendly tight crop."""
    episode_dir = os.path.join(output_dir, "map_snapshots", str(episode_id))
    os.makedirs(episode_dir, exist_ok=True)
    full = _compose_full_map(bev_map)
    reference_key = str(episode_id)
    stabilization = None
    if int(target_index) == 0:
        _TARGET1_FULL_MAPS[reference_key] = full.copy()
    else:
        reference = _TARGET1_FULL_MAPS.pop(reference_key, None)
        if reference is not None:
            full, stabilization = _stabilize_target2_obstacles(full, reference)
    path = getattr(agent, "visited_vis", None)
    rgb, explored = _semantic_rgb(full, path)
    image = Image.fromarray(rgb, mode="RGB")
    full_pose = bev_map.planner_pose_inputs[0, :3].copy()
    _draw_agent(image, full_pose, bev_map.args.map_resolution)

    stem = "unigoal_target_{:02d}_terminal_map".format(target_index + 1)
    full_path = os.path.join(episode_dir, stem + "_global.png")
    image.save(full_path)

    ys, xs = np.where(explored)
    if len(xs):
        pad = 24
        box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
               min(image.width, int(xs.max()) + pad + 1),
               min(image.height, int(ys.max()) + pad + 1))
    else:
        box = (0, 0, image.width, image.height)
    tight = image.crop(box)
    tight_path = os.path.join(episode_dir, stem + ".png")
    tight.save(tight_path)

    metadata = {
        "episode_id": str(episode_id),
        "target_index": int(target_index),
        "success": float(success),
        "spl": float(spl),
        "map_coordinates": "fixed full-map coordinates; north/up after vertical flip",
        "trajectory": "current target only",
        "crop_xyxy": list(box),
        "full_pose_xy_heading_deg": [float(v) for v in full_pose],
        "global_map": os.path.basename(full_path),
        "tight_map": os.path.basename(tight_path),
        "translation_stabilization": stabilization,
    }
    with open(os.path.join(episode_dir, stem + ".json"), "w") as handle:
        json.dump(metadata, handle, indent=2)
    return tight_path
