"""Additive semantic-Gaussian instance filtering for the Multi-IIN runner.

This module deliberately leaves the IEVE/MinAV implementation untouched.  The
Multi-IIN runner installs small wrappers around the existing NVSM extension
points so that the target semantic mask is filtered before its centroid is
computed and before novel views are rendered.

The filter is intentionally lightweight:

1. voxelise the selected Gaussian centres;
2. run DBSCAN on the occupied voxels (also acting as a radius outlier filter);
3. prefer the component supported by the image-verified confirmed-target
   indices; otherwise use the SOIM-anchor neighbourhood and component size;
4. map the selected voxels back to the original Gaussian indices.

The original semantic Gaussian set is retained for audit artefacts.  Runtime
outputs contain ``*_raw.ply``, ``*_filtered.ply`` and
``*_filter_stats.json`` next to the existing ``instance_semantic_*.ply``.
"""

from __future__ import annotations

import json
import logging
import shutil
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceFilterConfig:
    enabled: bool = True
    voxel_size_m: float = 0.03
    cluster_eps_m: float = 0.10
    cluster_min_samples: int = 5
    min_cluster_points: int = 50
    min_keep_ratio: float = 0.02
    anchor_neighbourhood_slack_m: float = 0.25

    @classmethod
    def from_args(cls, args: Any) -> "InstanceFilterConfig":
        return cls(
            enabled=bool(int(getattr(args, "multi_instance_filter", 1))),
            voxel_size_m=max(
                1e-3,
                float(getattr(args, "multi_instance_filter_voxel_size_m", 0.03)),
            ),
            cluster_eps_m=max(
                1e-3,
                float(getattr(args, "multi_instance_filter_eps_m", 0.10)),
            ),
            cluster_min_samples=max(
                1,
                int(getattr(args, "multi_instance_filter_min_samples", 5)),
            ),
            min_cluster_points=max(
                1,
                int(getattr(args, "multi_instance_filter_min_cluster_points", 50)),
            ),
            min_keep_ratio=min(
                1.0,
                max(
                    0.0,
                    float(getattr(args, "multi_instance_filter_min_keep_ratio", 0.02)),
                ),
            ),
            anchor_neighbourhood_slack_m=max(
                0.0,
                float(
                    getattr(
                        args,
                        "multi_instance_filter_anchor_neighbourhood_slack_m",
                        0.25,
                    )
                ),
            ),
        )


def _finite_xyz_and_indices(
    xyz_all: np.ndarray, indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
    valid = indices[(indices >= 0) & (indices < int(xyz_all.shape[0]))]
    if valid.size == 0:
        return valid, np.empty((0, 3), dtype=np.float32)
    xyz = np.asarray(xyz_all[valid, :3], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    return valid[finite], xyz[finite]


def _array_stats(xyz: np.ndarray) -> Dict[str, Any]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if xyz.size == 0:
        return {
            "count": 0,
            "center": None,
            "bbox_min": None,
            "bbox_max": None,
            "bbox_size": None,
        }
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    return {
        "count": int(xyz.shape[0]),
        "center": xyz.mean(axis=0).astype(float).tolist(),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "bbox_size": (bbox_max - bbox_min).astype(float).tolist(),
    }


def _category_size_prior(category: Any) -> Optional[Dict[str, Tuple[float, float]]]:
    """Loose scale bounds used only to reject tiny semantic satellites."""
    name = str(category or "").lower()
    if not name:
        return None
    if "bed" in name or "sofa" in name:
        return {"xz_long": (0.65, 4.5), "y": (0.25, 2.5)}
    if "chair" in name:
        return {"xz_long": (0.25, 2.5), "y": (0.25, 2.5)}
    if "plant" in name:
        return {"xz_long": (0.20, 3.0), "y": (0.20, 4.0)}
    if "toilet" in name:
        return {"xz_long": (0.25, 2.0), "y": (0.20, 2.0)}
    if "tv" in name or "monitor" in name:
        return {"xz_long": (0.25, 3.0), "y": (0.15, 2.5)}
    return None


def _category_size_compatible(
    bbox_size: Any, category: Any
) -> Tuple[bool, Dict[str, Any]]:
    """Return whether a partial reconstruction has a plausible object scale."""
    prior = _category_size_prior(category)
    dims = np.asarray(bbox_size, dtype=np.float32).reshape(-1)
    if prior is None:
        return True, {"applied": False, "reason": "unknown_category"}
    if dims.size < 3 or not np.isfinite(dims[:3]).all():
        return False, {"applied": True, "reason": "invalid_bbox"}
    xz_long = float(max(dims[0], dims[2]))
    y_size = float(dims[1])
    long_min, long_max = prior["xz_long"]
    y_min, y_max = prior["y"]
    compatible = long_min <= xz_long <= long_max and y_min <= y_size <= y_max
    return compatible, {
        "applied": True,
        "category": str(category),
        "xz_long_m": xz_long,
        "y_size_m": y_size,
        "xz_long_bounds_m": [long_min, long_max],
        "y_bounds_m": [y_min, y_max],
        "reason": "within_bounds" if compatible else "bbox_size_mismatch",
    }


def _prefilter_indices_by_instance_region(
    xyz_all: np.ndarray,
    indices: np.ndarray,
    region_2d: Any,
    occupancy_mapper: Any,
    min_points: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Keep Gaussian centres whose XZ projection lies in the SOIM mask."""

    diagnostics = {
        "available": False,
        "applied": False,
        "input_count": int(np.asarray(indices).size),
        "output_count": int(np.asarray(indices).size),
        "fallback_reason": None,
    }
    if not isinstance(region_2d, dict) or region_2d.get("mask") is None:
        diagnostics["fallback_reason"] = "missing_instance_mask"
        return np.asarray(indices, dtype=np.int64), diagnostics
    if occupancy_mapper is None or not hasattr(occupancy_mapper, "world_to_map"):
        diagnostics["fallback_reason"] = "missing_world_to_map"
        return np.asarray(indices, dtype=np.int64), diagnostics

    mask = np.asarray(region_2d["mask"], dtype=bool)
    if mask.ndim != 2 or not bool(mask.any()):
        diagnostics["fallback_reason"] = "empty_instance_mask"
        return np.asarray(indices, dtype=np.int64), diagnostics
    diagnostics["available"] = True
    valid, xyz = _finite_xyz_and_indices(xyz_all, indices)
    keep = np.zeros((valid.size,), dtype=bool)
    height, width = mask.shape
    for offset, point in enumerate(xyz):
        try:
            col, row = occupancy_mapper.world_to_map(
                float(point[0]), float(point[2])
            )
            if col is None or row is None:
                continue
            col = int(col)
            row = int(row)
            keep[offset] = (
                0 <= row < height and 0 <= col < width and bool(mask[row, col])
            )
        except Exception:
            continue
    selected = valid[keep]
    diagnostics["output_count"] = int(selected.size)
    diagnostics["keep_ratio"] = float(selected.size) / max(int(valid.size), 1)
    if selected.size < int(min_points):
        diagnostics["fallback_reason"] = "instance_mask_selected_too_few_points"
        return valid, diagnostics
    diagnostics["applied"] = True
    return selected, diagnostics


def filter_instance_indices(
    xyz_all: np.ndarray,
    candidate_indices: np.ndarray,
    config: InstanceFilterConfig,
    anchor_3d: Optional[np.ndarray] = None,
    seed_indices: Optional[np.ndarray] = None,
    target_category: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Filter target Gaussian indices and return serialisable diagnostics."""

    valid, xyz = _finite_xyz_and_indices(xyz_all, candidate_indices)
    raw_stats = _array_stats(xyz)
    diagnostics: Dict[str, Any] = {
        "algorithm": "voxel_dbscan_confirmed_seed_anchor_size_v3",
        "config": asdict(config),
        "raw": raw_stats,
        "anchor_3d": None,
        "applied": False,
        "fallback_reason": None,
        "clusters": [],
        "selected_label": None,
        "target_category": target_category,
    }
    if seed_indices is None:
        seed_indices = np.empty((0,), dtype=np.int64)
    else:
        seed_indices = np.unique(np.asarray(seed_indices, dtype=np.int64).reshape(-1))
    seed_in_candidates = np.intersect1d(valid, seed_indices, assume_unique=True)
    seed_membership = np.isin(valid, seed_in_candidates, assume_unique=True)
    diagnostics["seed_indices_count"] = int(seed_indices.size)
    diagnostics["seed_indices_in_candidates"] = int(seed_in_candidates.size)
    if anchor_3d is not None:
        anchor = np.asarray(anchor_3d, dtype=np.float32).reshape(-1)
        if anchor.size >= 3 and np.isfinite(anchor[:3]).all():
            anchor_3d = anchor[:3]
            diagnostics["anchor_3d"] = anchor_3d.astype(float).tolist()
        else:
            anchor_3d = None

    if not config.enabled:
        diagnostics["fallback_reason"] = "disabled"
        diagnostics["filtered"] = raw_stats
        return valid, diagnostics
    if valid.size < config.min_cluster_points:
        diagnostics["fallback_reason"] = "too_few_candidate_points"
        diagnostics["filtered"] = raw_stats
        return valid, diagnostics

    try:
        from sklearn.cluster import DBSCAN
    except Exception as exc:
        diagnostics["fallback_reason"] = "sklearn_unavailable: {}".format(exc)
        diagnostics["filtered"] = raw_stats
        return valid, diagnostics

    voxel_keys = np.floor(xyz / float(config.voxel_size_m)).astype(np.int64)
    unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)
    voxel_xyz = (
        unique_keys.astype(np.float32) + np.float32(0.5)
    ) * np.float32(config.voxel_size_m)
    try:
        voxel_labels = DBSCAN(
            eps=float(config.cluster_eps_m),
            min_samples=int(config.cluster_min_samples),
            n_jobs=1,
        ).fit_predict(voxel_xyz)
    except Exception as exc:
        diagnostics["fallback_reason"] = "dbscan_failed: {}".format(exc)
        diagnostics["filtered"] = raw_stats
        return valid, diagnostics

    point_labels = voxel_labels[inverse]
    labels = sorted(set(int(v) for v in voxel_labels.tolist()) - {-1})
    records = []
    max_size = 0
    for label in labels:
        mask = point_labels == label
        count = int(mask.sum())
        if count < int(config.min_cluster_points):
            continue
        points = xyz[mask]
        center = points.mean(axis=0)
        anchor_distance_xz = None
        if anchor_3d is not None:
            anchor_distance_xz = float(
                np.linalg.norm(center[[0, 2]] - anchor_3d[[0, 2]])
            )
        max_size = max(max_size, count)
        records.append(
            {
                "label": int(label),
                "count": count,
                "seed_overlap_count": int(np.count_nonzero(seed_membership & mask)),
                "center": center.astype(float).tolist(),
                "bbox_min": points.min(axis=0).astype(float).tolist(),
                "bbox_max": points.max(axis=0).astype(float).tolist(),
                "anchor_distance_xz_m": anchor_distance_xz,
            }
        )

    if not records:
        diagnostics["fallback_reason"] = "no_cluster_above_minimum_size"
        diagnostics["filtered"] = raw_stats
        diagnostics["noise_points"] = int(np.count_nonzero(point_labels < 0))
        diagnostics["voxel_count"] = int(voxel_xyz.shape[0])
        return valid, diagnostics

    for record in records:
        record["size_ratio_to_largest"] = float(record["count"]) / max(max_size, 1)
        compatible, size_debug = _category_size_compatible(
            np.asarray(record["bbox_max"]) - np.asarray(record["bbox_min"]),
            target_category,
        )
        record["size_compatible"] = bool(compatible)
        record["size_gate"] = size_debug

    # A cluster must first have enough global support to be a plausible object
    # body. This is important because a noisy satellite can lie exactly on a
    # coarse SOIM anchor (the recorded bed case has such a 305-point cluster)
    # while the actual object body is larger and its centroid is farther away.
    plausible = [
        record
        for record in records
        if float(record["count"]) / max(int(valid.size), 1)
        >= float(config.min_keep_ratio)
    ]
    if not plausible:
        plausible = records
        plausible_fallback = True
    else:
        plausible_fallback = False
    plausible_labels = {int(record["label"]) for record in plausible}
    for record in records:
        record["body_candidate"] = int(record["label"]) in plausible_labels

    seeded = [record for record in records if int(record["seed_overlap_count"]) > 0]
    selected_from_seed = bool(seeded)
    if seeded:
        # Target confirmation has already used rendered-image feature evidence
        # to lock these Gaussian indices. It is stronger instance identity
        # evidence than a coarse 2-D-map anchor or raw cluster size.
        selected = max(
            seeded,
            key=lambda item: (int(item["seed_overlap_count"]), int(item["count"])),
        )
        nearest_distance = None
        neighbourhood_limit = None
        selection_mode = "confirmed_seed_overlap_then_size"
        selected_label = int(selected["label"])
        for record in records:
            record["anchor_neighbourhood_eligible"] = None
            record["confirmed_seed_selected"] = int(record["label"]) == selected_label
    elif anchor_3d is not None:
        # The SOIM anchor identifies the intended instance. First restrict
        # selection to its nearest cluster neighbourhood, then use support
        # size to avoid choosing a tiny satellite/noise cluster that happens
        # to be a few centimetres closer than the object body. A much larger
        # cluster farther away cannot win merely because of its size.
        nearest_distance = min(
            float(record["anchor_distance_xz_m"]) for record in plausible
        )
        neighbourhood_limit = (
            nearest_distance + float(config.anchor_neighbourhood_slack_m)
        )
        eligible = [
            record
            for record in plausible
            if float(record["anchor_distance_xz_m"]) <= neighbourhood_limit
        ]
        eligible_labels = {int(record["label"]) for record in eligible}
        for record in records:
            record["anchor_neighbourhood_eligible"] = (
                int(record["label"]) in eligible_labels
            )
        selected = max(
            eligible,
            key=lambda item: (
                int(item["count"]),
                -float(item["anchor_distance_xz_m"]),
            ),
        )
        selection_mode = "anchor_neighbourhood_then_size"
    else:
        nearest_distance = None
        neighbourhood_limit = None
        selected = max(plausible, key=lambda item: int(item["count"]))
        selection_mode = "largest_cluster_without_anchor"

    initial_selected_label = int(selected["label"])
    selected_size_compatible = bool(selected.get("size_compatible", True))
    size_fallback_used = False
    if not selected_size_compatible:
        compatible_records = [
            record
            for record in plausible
            if bool(record.get("size_compatible", True))
        ]
        if anchor_3d is not None:
            eligible_compatible = [
                record
                for record in compatible_records
                if record.get("anchor_neighbourhood_eligible", False)
            ]
            if eligible_compatible:
                compatible_records = eligible_compatible
        if compatible_records:
            selected = max(
                compatible_records,
                key=lambda item: (
                    int(item.get("seed_overlap_count", 0)),
                    int(item["count"]),
                    -float(item.get("anchor_distance_xz_m", np.inf)),
                ),
            )
            selected_size_compatible = True
            size_fallback_used = True
        else:
            diagnostics["fallback_reason"] = "no_category_size_compatible_cluster"
            diagnostics["size_gate"] = {
                "selected_initial_label": initial_selected_label,
                "selected_initial": selected.get("size_gate"),
                "compatible_cluster_count": 0,
            }
            diagnostics["filtered"] = raw_stats
            return valid, diagnostics

    diagnostics["size_gate"] = {
        "selected_initial_label": initial_selected_label,
        "selected_final_label": int(selected["label"]),
        "selected_initial": selected.get("size_gate"),
        "selected_final": selected.get("size_gate"),
        "fallback_used": bool(size_fallback_used),
    }
    keep_mask = point_labels == int(selected["label"])
    filtered = valid[keep_mask]
    keep_ratio = float(filtered.size) / max(int(valid.size), 1)
    diagnostics.update(
        {
            "clusters": sorted(records, key=lambda item: item["count"], reverse=True),
            "selected_label": int(selected["label"]),
            "selection_mode": selection_mode,
            "body_candidate_min_keep_ratio": float(config.min_keep_ratio),
            "body_candidate_floor_fallback": plausible_fallback,
            "selected_seed_overlap_count": int(selected["seed_overlap_count"]),
            "nearest_anchor_distance_xz_m": nearest_distance,
            "anchor_neighbourhood_limit_m": neighbourhood_limit,
            "keep_ratio": keep_ratio,
            "noise_points": int(np.count_nonzero(point_labels < 0)),
            "voxel_count": int(voxel_xyz.shape[0]),
        }
    )
    if (
        filtered.size < int(config.min_cluster_points)
        or (
            keep_ratio < float(config.min_keep_ratio)
            and not selected_from_seed
        )
    ):
        diagnostics["fallback_reason"] = "selected_cluster_failed_safety_floor"
        diagnostics["filtered"] = raw_stats
        return valid, diagnostics

    diagnostics["applied"] = True
    diagnostics["filtered"] = _array_stats(xyz[keep_mask])
    diagnostics["removed_points"] = int(valid.size - filtered.size)
    return filtered, diagnostics


def _write_xyz_ply(path: Path, xyz: np.ndarray, color: Tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("element vertex {}\n".format(int(points.shape[0])))
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        r, g, b = (int(value) for value in color)
        for point in points:
            handle.write(
                "{:.8g} {:.8g} {:.8g} {} {} {}\n".format(
                    float(point[0]), float(point[1]), float(point[2]), r, g, b
                )
            )


def save_filter_projection(
    kept_xyz: np.ndarray,
    removed_xyz: np.ndarray,
    output_path: Path,
    target_category: Optional[str] = None,
) -> None:
    """Save an audit PNG showing retained and removed Gaussian centres.

    The two orthographic panels use identical colours: retained points are
    green and rejected points are red.  This is deliberately a point
    projection, not a Gaussian render, so filtering mistakes remain visible.
    """

    from PIL import Image, ImageDraw

    kept = np.asarray(kept_xyz, dtype=np.float32).reshape(-1, 3)
    removed = np.asarray(removed_xyz, dtype=np.float32).reshape(-1, 3)
    kept = kept[np.isfinite(kept).all(axis=1)]
    removed = removed[np.isfinite(removed).all(axis=1)]
    width, height = 1400, 720
    margin, header, gap = 42, 82, 36
    panel_w = (width - 2 * margin - gap) // 2
    panel_h = height - header - margin
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    category = str(target_category or "unknown")
    draw.text(
        (margin, 18),
        "Target Gaussian filter projection | {} | SAVE={}  REMOVE={}".format(
            category, int(kept.shape[0]), int(removed.shape[0])
        ),
        fill=(20, 20, 20),
    )
    draw.rectangle((margin, 48, margin + 18, 66), fill=(25, 180, 75))
    draw.text((margin + 25, 49), "SAVE / retained", fill=(20, 20, 20))
    draw.rectangle((margin + 190, 48, margin + 208, 66), fill=(220, 55, 45))
    draw.text((margin + 215, 49), "REMOVE / rejected", fill=(20, 20, 20))

    def draw_panel(origin_x: int, axes: Tuple[int, int], label: str) -> None:
        x0, y0 = origin_x, header
        x1, y1 = x0 + panel_w, y0 + panel_h
        draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(90, 90, 90))
        draw.text((x0 + 10, y0 + 8), label, fill=(25, 25, 25))
        arrays = [points[:, axes] for points in (kept, removed) if points.size]
        if not arrays:
            draw.text((x0 + 10, y0 + 30), "no points", fill=(80, 80, 80))
            return
        extent_points = np.concatenate(arrays, axis=0)
        low = np.percentile(extent_points, 0.5, axis=0)
        high = np.percentile(extent_points, 99.5, axis=0)
        span = np.maximum(high - low, 1e-4)
        pad = span * 0.06
        low, high = low - pad, high + pad
        span = high - low
        plot_pad = 20

        # Draw rejected points first so retained structure remains readable.
        for points, color in ((removed, (220, 55, 45)), (kept, (25, 180, 75))):
            if not points.size:
                continue
            projected = points[:, axes]
            inside = ((projected >= low) & (projected <= high)).all(axis=1)
            projected = projected[inside]
            if projected.shape[0] > 180000:
                stride = int(np.ceil(projected.shape[0] / 180000.0))
                projected = projected[::stride]
            px = x0 + plot_pad + (
                (projected[:, 0] - low[0]) / span[0] * (panel_w - 2 * plot_pad)
            ).astype(np.int32)
            py = y1 - plot_pad - (
                (projected[:, 1] - low[1]) / span[1] * (panel_h - 2 * plot_pad)
            ).astype(np.int32)
            draw.point(list(zip(px.tolist(), py.tolist())), fill=color)

    draw_panel(margin, (0, 2), "Top view: X-Z")
    draw_panel(margin + panel_w + gap, (0, 1), "Front view: X-Y (height)")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output))


_ACTIVE_CONFIG = InstanceFilterConfig()
_INSTALLED = False
_ORIGINAL_SET_REGION = None
_ORIGINAL_VISUALIZE = None
_ORIGINAL_EXECUTE_NVSM = None


def _apply_semantic_indices(mapper: Any, indices: np.ndarray) -> int:
    gaussians = mapper._gaussians()
    if gaussians is None:
        return 0
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if hasattr(gaussians, "reset_semantic_target_values"):
        gaussians.reset_semantic_target_values()
    else:
        gaussians.reset_semantic_target()
    if indices.size:
        gaussians.set_semantic_target(indices, value=1.0)
    mask = np.zeros((int(gaussians.get_xyz.shape[0]),), dtype=bool)
    mask[indices] = True
    slam = getattr(mapper, "slam", None)
    if slam is not None:
        try:
            import torch

            slam._target_semantic_mask = torch.from_numpy(mask).to(
                gaussians.get_xyz.device
            )
        except Exception:
            slam._target_semantic_mask = mask
    return int(indices.size)


def _filtered_set_target_semantic_region(self: Any, *args: Any, **kwargs: Any) -> int:
    context = getattr(self, "_multi_iin_instance_filter_context", None)
    if context and context.get("preapplied"):
        indices = np.asarray(
            getattr(self, "_multi_iin_instance_filter_last", {})
            .get("filtered_indices", np.empty((0,), dtype=np.int64)),
            dtype=np.int64,
        )
        return int(indices.size)
    marked = _ORIGINAL_SET_REGION(self, *args, **kwargs)
    config = getattr(self, "_multi_iin_instance_filter_config", _ACTIVE_CONFIG)
    self._multi_iin_instance_filter_last = None
    if not context or not config.enabled or marked <= 0:
        return marked
    try:
        gaussians = self._gaussians()
        raw_indices = self.get_target_semantic_gaussians()
        xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        region_2d = args[0] if args else kwargs.get("region_2d")
        region_indices, region_diagnostics = _prefilter_indices_by_instance_region(
            xyz_all,
            raw_indices,
            region_2d,
            context.get("occupancy_mapper"),
            config.min_cluster_points,
        )
        confirmed_indices = np.asarray(
            context.get("confirmed_indices", np.empty((0,), dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        confirmed_indices = confirmed_indices[
            (confirmed_indices >= 0) & (confirmed_indices < xyz_all.shape[0])
        ]
        if confirmed_indices.size == 0 and hasattr(
            gaussians, "get_confirmed_target_indices"
        ):
            confirmed = gaussians.get_confirmed_target_indices()
            if confirmed is not None:
                try:
                    import torch

                    if torch.is_tensor(confirmed):
                        confirmed = confirmed.detach().cpu().numpy()
                except Exception:
                    pass
                confirmed_indices = np.asarray(confirmed, dtype=np.int64).reshape(-1)
                confirmed_indices = confirmed_indices[
                    (confirmed_indices >= 0) & (confirmed_indices < xyz_all.shape[0])
                ]
        if confirmed_indices.size:
            # The SOIM projection is intentionally tight, but map drift can
            # move an already image-verified cluster just outside that mask.
            # Preserve it as a selection seed instead of silently clipping it.
            region_indices = np.unique(
                np.concatenate([region_indices, confirmed_indices])
            )
        filtered, diagnostics = filter_instance_indices(
            xyz_all,
            region_indices,
            config,
            anchor_3d=context.get("anchor_3d"),
            seed_indices=confirmed_indices,
            target_category=context.get("target_category"),
        )
        _, semantic_raw_xyz = _finite_xyz_and_indices(xyz_all, raw_indices)
        diagnostics["semantic_raw"] = _array_stats(semantic_raw_xyz)
        diagnostics["instance_region_prefilter"] = region_diagnostics
        diagnostics["target_category"] = context.get("target_category")
        diagnostics["object_id"] = str(context.get("object_id", ""))
        diagnostics["raw_indices_count"] = int(raw_indices.size)
        diagnostics["confirmed_seed_count"] = int(confirmed_indices.size)
        self._multi_iin_instance_filter_last = {
            "raw_indices": np.asarray(raw_indices, dtype=np.int64),
            "filtered_indices": np.asarray(filtered, dtype=np.int64),
            "diagnostics": diagnostics,
        }
        count = _apply_semantic_indices(self, filtered)
        LOGGER.info(
            "Multi-IIN semantic instance filter: target=%s raw=%d filtered=%d "
            "removed=%d applied=%s",
            context.get("target_category"),
            int(raw_indices.size),
            count,
            int(raw_indices.size) - count,
            diagnostics.get("applied"),
        )
        return count
    except Exception as exc:
        LOGGER.exception("Multi-IIN semantic instance filter failed; keeping raw mask: %s", exc)
        return marked


def _filtered_visualize_target_semantic_gaussians(
    self: Any, output_path: str
) -> Dict[str, Any]:
    result = _ORIGINAL_VISUALIZE(self, output_path)
    state = getattr(self, "_multi_iin_instance_filter_last", None)
    if not state:
        return result
    try:
        output = Path(output_path)
        raw_path = output.with_name(output.stem + "_raw.ply")
        filtered_path = output.with_name(output.stem + "_filtered.ply")
        stats_path = output.with_name(output.stem + "_filter_stats.json")
        projection_path = output.with_name(output.stem + "_filter_projection.png")
        gaussians = self._gaussians()
        xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        raw_indices, raw_xyz = _finite_xyz_and_indices(
            xyz_all, state["raw_indices"]
        )
        filtered_indices, filtered_xyz = _finite_xyz_and_indices(
            xyz_all, state["filtered_indices"]
        )
        removed_indices = np.setdiff1d(
            raw_indices, filtered_indices, assume_unique=False
        )
        _, removed_xyz = _finite_xyz_and_indices(xyz_all, removed_indices)
        _write_xyz_ply(raw_path, raw_xyz, (255, 0, 0))
        if output.exists():
            shutil.copyfile(str(output), str(filtered_path))
        else:
            _write_xyz_ply(filtered_path, filtered_xyz, (0, 255, 0))
        diagnostics = dict(state["diagnostics"])
        save_filter_projection(
            filtered_xyz,
            removed_xyz,
            projection_path,
            target_category=diagnostics.get("target_category"),
        )
        diagnostics["artifacts"] = {
            "raw_ply": str(raw_path),
            "filtered_ply": str(filtered_path),
            "compatibility_filtered_ply": str(output),
            "stats_json": str(stats_path),
            "projection_png": str(projection_path),
        }
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        result = dict(result)
        result.update(
            {
                "filter_applied": bool(diagnostics.get("applied")),
                "raw_count": int(raw_xyz.shape[0]),
                "filtered_count": int(filtered_xyz.shape[0]),
                "raw_ply": str(raw_path),
                "filtered_ply": str(filtered_path),
                "filter_stats_json": str(stats_path),
                "filter_projection_png": str(projection_path),
            }
        )
    except Exception as exc:
        LOGGER.exception("Failed to save Multi-IIN instance filter artifacts: %s", exc)
    return result


def _execute_nvsm_with_filter_context(self: Any) -> Dict[str, Any]:
    mapper = getattr(self, "slam_mapper", None)
    if mapper is None:
        return _ORIGINAL_EXECUTE_NVSM(self)
    current_object = getattr(self, "current_object", {}) or {}
    anchor = current_object.get("position_3d")
    if anchor is None:
        soim = getattr(self, "soim", None)
        anchor = getattr(soim, "candidate_anchor_position", None)
    confirmed_indices = np.empty((0,), dtype=np.int64)
    provider = getattr(self, "_multi_iin_confirmed_indices_provider", None)
    if callable(provider):
        try:
            provided = provider()
            if provided is not None:
                confirmed_indices = np.asarray(provided, dtype=np.int64).reshape(-1)
        except Exception as exc:
            LOGGER.warning("Multi-IIN confirmed-index bridge failed: %s", exc)
    mapper._multi_iin_instance_filter_config = _ACTIVE_CONFIG
    mapper._multi_iin_instance_filter_context = {
        "anchor_3d": anchor,
        "target_category": getattr(self, "target_category", None),
        "object_id": current_object.get("id"),
        "occupancy_mapper": getattr(self, "occupancy_mapper", None),
        "confirmed_indices": confirmed_indices,
    }
    # Some SOIM quick-lock objects do not carry a 2-D boundary.  The upstream
    # NVSM then skips set_target_semantic_region entirely.  Pre-apply the same
    # instance filter from the exact confirmed indices so rendering and the
    # saved PLY share one mask even in that branch.
    try:
        gaussians = mapper._gaussians()
        raw_indices = mapper.get_target_semantic_gaussians()
        xyz_all = gaussians.get_xyz.detach().cpu().numpy()
        region_indices = np.unique(
            np.concatenate(
                [
                    np.asarray(raw_indices, dtype=np.int64).reshape(-1),
                    confirmed_indices,
                ]
            )
        )
        filtered, diagnostics = filter_instance_indices(
            xyz_all,
            region_indices,
            _ACTIVE_CONFIG,
            anchor_3d=anchor,
            seed_indices=confirmed_indices,
            target_category=getattr(self, "target_category", None),
        )
        diagnostics["semantic_raw"] = _array_stats(
            _finite_xyz_and_indices(xyz_all, raw_indices)[1]
        )
        diagnostics["raw_indices_count"] = int(np.asarray(raw_indices).size)
        diagnostics["confirmed_seed_count"] = int(confirmed_indices.size)
        diagnostics["target_category"] = getattr(self, "target_category", None)
        diagnostics["object_id"] = str(current_object.get("id", ""))
        _apply_semantic_indices(mapper, filtered)
        mapper._multi_iin_instance_filter_last = {
            "raw_indices": np.asarray(raw_indices, dtype=np.int64),
            "filtered_indices": np.asarray(filtered, dtype=np.int64),
            "diagnostics": diagnostics,
        }
        mapper._multi_iin_instance_filter_context["preapplied"] = True
        LOGGER.info(
            "Multi-IIN preapplied semantic instance filter: raw=%d filtered=%d applied=%s",
            int(np.asarray(raw_indices).size),
            int(filtered.size),
            diagnostics.get("applied"),
        )
    except Exception as exc:
        LOGGER.warning("Multi-IIN pre-NVSM filter preparation failed: %s", exc)
    LOGGER.info(
        "Multi-IIN NVSM filter context: target=%s object=%s confirmed_indices=%d",
        getattr(self, "target_category", None),
        current_object.get("id"),
        int(confirmed_indices.size),
    )
    try:
        return _ORIGINAL_EXECUTE_NVSM(self)
    finally:
        mapper._multi_iin_instance_filter_context = None


def attach_instance_filter_to_manager(manager: Any) -> None:
    """Bridge the manager's exact quick-lock indices into its localizer.

    The MinAV localizer otherwise has access only to a coarse SOIM boundary
    and can re-expand that boundary to several same-class objects.  This
    instance-level bridge is installed on a Multi-IIN manager only; no source
    class in IEVE/MinAV is changed.
    """

    if manager is None or getattr(manager, "_multi_iin_filter_bridge", False):
        return
    original_ensure = manager._ensure_full_localizer

    def ensure_with_bridge(self, payload):
        localizer = original_ensure(payload)
        if localizer is None:
            return None
        localizer._multi_iin_confirmed_indices_provider = lambda: getattr(
            self, "confirmed_indices", None
        )
        mapper = getattr(localizer, "slam_mapper", None)
        if mapper is not None:
            mapper._multi_iin_instance_filter_config = _ACTIVE_CONFIG
        return localizer

    manager._ensure_full_localizer = types.MethodType(ensure_with_bridge, manager)
    manager._multi_iin_filter_bridge = True


def install_instance_semantic_filter(args: Any) -> InstanceFilterConfig:
    """Install process-local wrappers used only by the Multi-IIN runner."""

    global _ACTIVE_CONFIG
    global _INSTALLED
    global _ORIGINAL_SET_REGION
    global _ORIGINAL_VISUALIZE
    global _ORIGINAL_EXECUTE_NVSM

    _ACTIVE_CONFIG = InstanceFilterConfig.from_args(args)
    if not _ACTIVE_CONFIG.enabled:
        LOGGER.info("Multi-IIN semantic instance filter disabled")
        return _ACTIVE_CONFIG
    if _INSTALLED:
        return _ACTIVE_CONFIG

    from gaussian_mapping.mapper.slam_mapper import SLAMMapper
    from ieve_minav.localizer.localization_system import LocalizationSystem

    _ORIGINAL_SET_REGION = SLAMMapper.set_target_semantic_region
    _ORIGINAL_VISUALIZE = SLAMMapper.visualize_target_semantic_gaussians
    _ORIGINAL_EXECUTE_NVSM = LocalizationSystem._execute_nvsm
    SLAMMapper.set_target_semantic_region = _filtered_set_target_semantic_region
    SLAMMapper.visualize_target_semantic_gaussians = (
        _filtered_visualize_target_semantic_gaussians
    )
    LocalizationSystem._execute_nvsm = _execute_nvsm_with_filter_context
    _INSTALLED = True
    LOGGER.info("Installed Multi-IIN semantic instance filter: %s", _ACTIVE_CONFIG)
    return _ACTIVE_CONFIG
