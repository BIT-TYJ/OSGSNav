"""Online detector-label fusion for Multi-IIN Gaussians.

The RGB Gaussian implementation remains unchanged.  This additive trainer
projects its current Gaussians into each detector-labelled RGB-D frame,
accumulates six-class semantic evidence, and associates per-frame masks with
persistent instance ids.  Labels are sidecar state: RGB/geometry optimization
continues to use the existing mapper and Habitat semantic/object ground truth
is never consumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import types
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)

CLASS_NAMES = ("chair", "sofa", "plant", "bed", "toilet", "tv_monitor")


@dataclass(frozen=True)
class GaussianLabelConfig:
    enabled: bool = True
    class_count: int = 6
    min_confidence: float = 0.50
    depth_tolerance_m: float = 0.25
    instance_vote_ratio: float = 0.15
    instance_association_radius_m: float = 0.75
    evidence_cap: float = 32.0
    log_interval: int = 20

    @classmethod
    def from_args(cls, args: Any) -> "GaussianLabelConfig":
        return cls(
            enabled=bool(int(getattr(args, "multi_gaussian_label_training", 1))),
            class_count=6,
            min_confidence=float(
                getattr(args, "multi_gaussian_label_min_confidence", 0.50)
            ),
            depth_tolerance_m=max(
                0.01,
                float(
                    getattr(
                        args, "multi_gaussian_label_depth_tolerance_m", 0.25
                    )
                ),
            ),
            instance_vote_ratio=float(
                getattr(args, "multi_gaussian_instance_vote_ratio", 0.15)
            ),
            instance_association_radius_m=max(
                0.05,
                float(
                    getattr(
                        args,
                        "multi_gaussian_instance_association_radius_m",
                        0.75,
                    )
                ),
            ),
            evidence_cap=max(
                1.0, float(getattr(args, "multi_gaussian_label_evidence_cap", 32.0))
            ),
            log_interval=max(
                1, int(getattr(args, "multi_gaussian_label_log_interval", 20))
            ),
        )


def _resize_mask(mask: Any, height: int, width: int) -> Optional[np.ndarray]:
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2 or not bool(np.any(array)):
        return None
    if array.shape != (height, width):
        array = cv2.resize(
            array, (width, height), interpolation=cv2.INTER_NEAREST
        )
    result = array > 0
    return result if bool(np.any(result)) else None


def _normalized_instances(
    instances: Any,
    *,
    height: int,
    width: int,
    class_count: int,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    result = []
    for item in instances or []:
        if not isinstance(item, dict):
            continue
        try:
            category_id = int(item.get("category_id", -1))
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except Exception:
            continue
        if not (0 <= category_id < class_count) or confidence < min_confidence:
            continue
        mask = _resize_mask(item.get("binary_mask"), height, width)
        if mask is None:
            continue
        result.append(
            {
                "category_id": category_id,
                "confidence": min(max(confidence, 0.0), 1.0),
                "mask": mask,
                "pixel_count": int(mask.sum()),
            }
        )
    # Let confident, compact masks claim overlapping pixels first.
    result.sort(key=lambda x: (-x["confidence"], x["pixel_count"]))
    return result


def _semantic_summary(
    evidence: torch.Tensor, observations: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if evidence.numel() == 0:
        device = evidence.device
        return (
            torch.empty((0,), dtype=torch.int16, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    totals = evidence.float().sum(dim=1)
    best_value, best_class = evidence.float().max(dim=1)
    valid = (totals > 0.0) & (observations > 0)
    labels = best_class.to(dtype=torch.int16)
    labels[~valid] = -1
    confidence = torch.zeros_like(totals)
    confidence[valid] = best_value[valid] / totals[valid].clamp_min(1e-6)
    return labels, confidence


class GaussianSemanticInstanceTrainer:
    """Fuse detector masks into topology-aligned Gaussian sidecar tensors."""

    def __init__(self, config: GaussianLabelConfig, output_dir: Optional[Path]):
        self.config = config
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.next_instance_id = 1
        self.instance_registry: Dict[int, Dict[str, Any]] = {}
        self.frames_seen = 0
        self.frames_with_labels = 0
        self.last_stats: Dict[str, Any] = {}

    @classmethod
    def from_args(cls, args: Any, output_dir: Optional[Path] = None):
        return cls(GaussianLabelConfig.from_args(args), output_dir)

    def _ensure_sidecars(self, gaussians: Any) -> None:
        count = int(gaussians.get_xyz.shape[0])
        device = gaussians.get_xyz.device
        class_count = int(self.config.class_count)
        specifications = (
            ("_multi_semantic_evidence", (count, class_count), torch.float16),
            ("_multi_semantic_observations", (count,), torch.int32),
            ("_multi_instance_ids", (count,), torch.int32),
            ("_multi_instance_scores", (count,), torch.float16),
        )
        for name, shape, dtype in specifications:
            current = getattr(gaussians, name, None)
            if torch.is_tensor(current) and tuple(current.shape) == tuple(shape):
                continue
            replacement = torch.zeros(shape, dtype=dtype, device=device)
            if torch.is_tensor(current):
                keep = min(int(current.shape[0]), count)
                if keep > 0 and current.ndim == replacement.ndim:
                    replacement[:keep] = current[:keep].to(device=device, dtype=dtype)
            setattr(gaussians, name, replacement)

    def _install_topology_hooks(self, gaussians: Any) -> None:
        if getattr(gaussians, "_multi_label_topology_hooks", False):
            return
        trainer = self
        original_prune = gaussians.prune_points
        original_densify = gaussians.densification_postfix

        def prune_points(this, mask):
            trainer._ensure_sidecars(this)
            valid = (~mask).to(dtype=torch.bool)
            old = {
                name: getattr(this, name)
                for name in (
                    "_multi_semantic_evidence",
                    "_multi_semantic_observations",
                    "_multi_instance_ids",
                    "_multi_instance_scores",
                )
            }
            result = original_prune(mask)
            for name, tensor in old.items():
                setattr(this, name, tensor[valid])
            return result

        def densification_postfix(this, *args, **kwargs):
            trainer._ensure_sidecars(this)
            count_before = int(this.get_xyz.shape[0])
            result = original_densify(*args, **kwargs)
            count_after = int(this.get_xyz.shape[0])
            added = max(0, count_after - count_before)
            if added:
                device = this.get_xyz.device
                this._multi_semantic_evidence = torch.cat(
                    [
                        this._multi_semantic_evidence,
                        torch.zeros(
                            (added, trainer.config.class_count),
                            dtype=torch.float16,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                this._multi_semantic_observations = torch.cat(
                    [
                        this._multi_semantic_observations,
                        torch.zeros((added,), dtype=torch.int32, device=device),
                    ]
                )
                this._multi_instance_ids = torch.cat(
                    [
                        this._multi_instance_ids,
                        torch.zeros((added,), dtype=torch.int32, device=device),
                    ]
                )
                this._multi_instance_scores = torch.cat(
                    [
                        this._multi_instance_scores,
                        torch.zeros((added,), dtype=torch.float16, device=device),
                    ]
                )
            return result

        gaussians.prune_points = types.MethodType(prune_points, gaussians)
        gaussians.densification_postfix = types.MethodType(
            densification_postfix, gaussians
        )
        gaussians._multi_label_topology_hooks = True

    def configure_mapper(self, mapper: Any) -> None:
        if not self.config.enabled:
            return
        slam = getattr(mapper, "slam", None)
        gaussians = getattr(slam, "gaussians", None) if slam is not None else None
        if gaussians is None:
            return
        self._ensure_sidecars(gaussians)
        self._install_topology_hooks(gaussians)
        if not getattr(mapper, "_multi_label_save_hook", False):
            original_save = mapper.save_slam_results
            trainer = self

            def save_slam_results(this, *args, **kwargs):
                result = original_save(*args, **kwargs)
                output = Path(result) if result is not None else Path(this.output_dir) / "final"
                trainer.export(this, output)
                return result

            mapper.save_slam_results = types.MethodType(save_slam_results, mapper)
            mapper._multi_label_save_hook = True

    @staticmethod
    def _project_visible(
        gaussians: Any, camera: Any, depth: np.ndarray, tolerance: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz = gaussians.get_xyz.detach()
        device = xyz.device
        height, width = depth.shape[:2]
        w2c = camera.world_view_transform.T.to(device=device, dtype=xyz.dtype)
        xyz_h = torch.cat(
            [xyz, torch.ones((xyz.shape[0], 1), device=device, dtype=xyz.dtype)],
            dim=1,
        )
        cam_xyz = (w2c @ xyz_h.T).T[:, :3]
        z = cam_xyz[:, 2]
        safe_z = z.clamp_min(1e-6)
        u = cam_xyz[:, 0] * float(camera.fx) / safe_z + float(camera.cx)
        v = cam_xyz[:, 1] * float(camera.fy) / safe_z + float(camera.cy)
        valid = (
            (z > 0.1)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        u_int = u.long().clamp(0, width - 1)
        v_int = v.long().clamp(0, height - 1)
        depth_tensor = torch.from_numpy(depth).to(device=device, dtype=torch.float32)
        observed = depth_tensor[v_int, u_int]
        valid &= observed > 0.01
        valid &= torch.abs(z.float() - observed) <= float(tolerance)
        indices = torch.where(valid)[0]
        return indices, u_int[indices], v_int[indices]

    def _associate_instance(
        self, gaussians: Any, indices: torch.Tensor, category_id: int
    ) -> int:
        current_ids = gaussians._multi_instance_ids[indices]
        positive = current_ids[current_ids > 0]
        if positive.numel() > 0:
            ids, counts = torch.unique(positive, return_counts=True)
            best_pos = int(torch.argmax(counts).item())
            vote_ratio = float(counts[best_pos].item()) / float(max(indices.numel(), 1))
            candidate_id = int(ids[best_pos].item())
            registry = self.instance_registry.get(candidate_id)
            if (
                vote_ratio >= float(self.config.instance_vote_ratio)
                and registry is not None
                and int(registry["category_id"]) == int(category_id)
            ):
                return candidate_id

        centroid = gaussians.get_xyz.detach()[indices].mean(dim=0).cpu().numpy()
        best = None
        for instance_id, record in self.instance_registry.items():
            if int(record["category_id"]) != int(category_id):
                continue
            distance = float(
                np.linalg.norm(np.asarray(record["centroid"])[[0, 2]] - centroid[[0, 2]])
            )
            if distance <= float(self.config.instance_association_radius_m):
                if best is None or distance < best[0]:
                    best = (distance, int(instance_id))
        if best is not None:
            return best[1]
        instance_id = int(self.next_instance_id)
        self.next_instance_id += 1
        self.instance_registry[instance_id] = {
            "category_id": int(category_id),
            "centroid": centroid.astype(float).tolist(),
            "observations": 0,
            "gaussian_updates": 0,
        }
        return instance_id

    def observe(
        self,
        mapper: Any,
        instances: Any,
        depth: Any,
        *,
        env_step: int,
        stage: str,
    ) -> Dict[str, Any]:
        self.frames_seen += 1
        if not self.config.enabled:
            return {"enabled": False}
        slam = getattr(mapper, "slam", None)
        gaussians = getattr(slam, "gaussians", None) if slam is not None else None
        if gaussians is None or int(gaussians.get_xyz.shape[0]) == 0:
            return {"enabled": True, "updated": 0, "reason": "no_gaussians"}
        self.configure_mapper(mapper)
        frame_id = int(getattr(slam, "frame_count", 0))
        camera = getattr(slam, "keyframes", {}).get(frame_id)
        if camera is None or depth is None:
            return {"enabled": True, "updated": 0, "reason": "no_camera_or_depth"}
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]
        height = int(getattr(camera, "image_height", depth_np.shape[0]))
        width = int(getattr(camera, "image_width", depth_np.shape[1]))
        if depth_np.shape != (height, width):
            depth_np = cv2.resize(
                depth_np, (width, height), interpolation=cv2.INTER_NEAREST
            )
        items = _normalized_instances(
            instances,
            height=height,
            width=width,
            class_count=self.config.class_count,
            min_confidence=self.config.min_confidence,
        )
        if not items:
            return {"enabled": True, "updated": 0, "reason": "no_detector_instances"}

        with torch.no_grad():
            projected, u, v = self._project_visible(
                gaussians, camera, depth_np, self.config.depth_tolerance_m
            )
            if projected.numel() == 0:
                return {"enabled": True, "updated": 0, "reason": "no_depth_consistent_gaussians"}
            claimed = torch.zeros(
                (projected.numel(),), dtype=torch.bool, device=projected.device
            )
            updates = []
            total_updated = 0
            for item in items:
                mask_tensor = torch.from_numpy(item["mask"]).to(
                    device=projected.device, dtype=torch.bool
                )
                selected_local = mask_tensor[v, u] & (~claimed)
                selected = projected[selected_local]
                if selected.numel() == 0:
                    continue
                claimed |= selected_local
                category_id = int(item["category_id"])
                confidence = float(item["confidence"])
                evidence = gaussians._multi_semantic_evidence[:, category_id]
                evidence[selected] = torch.clamp(
                    evidence[selected].float() + confidence,
                    max=float(self.config.evidence_cap),
                ).to(dtype=evidence.dtype)
                gaussians._multi_semantic_observations[selected] += 1
                instance_id = self._associate_instance(
                    gaussians, selected, category_id
                )
                old_ids = gaussians._multi_instance_ids[selected]
                old_scores = gaussians._multi_instance_scores[selected].float()
                replace = (old_ids == instance_id) | (old_scores <= confidence)
                selected_replace = selected[replace]
                gaussians._multi_instance_ids[selected_replace] = instance_id
                gaussians._multi_instance_scores[selected_replace] = confidence
                centroid = gaussians.get_xyz.detach()[selected].mean(dim=0).cpu().numpy()
                record = self.instance_registry[instance_id]
                old_obs = int(record.get("observations", 0))
                old_center = np.asarray(record["centroid"], dtype=np.float32)
                record["centroid"] = (
                    (old_center * old_obs + centroid) / float(old_obs + 1)
                ).astype(float).tolist()
                record["observations"] = old_obs + 1
                record["gaussian_updates"] = int(
                    record.get("gaussian_updates", 0)
                ) + int(selected.numel())
                total_updated += int(selected.numel())
                updates.append(
                    {
                        "category_id": category_id,
                        "instance_id": instance_id,
                        "confidence": confidence,
                        "gaussians": int(selected.numel()),
                    }
                )

        self.frames_with_labels += 1
        labels, _confidence = _semantic_summary(
            gaussians._multi_semantic_evidence,
            gaussians._multi_semantic_observations,
        )
        stats = {
            "enabled": True,
            "env_step": int(env_step),
            "frame_id": frame_id,
            "stage": str(stage),
            "detector_instances": len(items),
            "updated": total_updated,
            "labelled_gaussians": int((labels >= 0).sum().item()),
            "persistent_instances": len(self.instance_registry),
            "updates": updates,
        }
        self.last_stats = stats
        if self.frames_with_labels % int(self.config.log_interval) == 0:
            LOGGER.info("[GAUSSIAN_LABEL_TRAINING] %s", json.dumps(stats, sort_keys=True))
        return stats

    def snapshot(self, mapper: Any) -> Dict[str, Any]:
        slam = getattr(mapper, "slam", None)
        gaussians = getattr(slam, "gaussians", None) if slam is not None else None
        if gaussians is None:
            return {"enabled": self.config.enabled, "gaussians": 0}
        self._ensure_sidecars(gaussians)
        labels, confidence = _semantic_summary(
            gaussians._multi_semantic_evidence,
            gaussians._multi_semantic_observations,
        )
        class_counts = {
            CLASS_NAMES[index]: int((labels == index).sum().item())
            for index in range(min(self.config.class_count, len(CLASS_NAMES)))
        }
        return {
            "enabled": self.config.enabled,
            "source": "detector_instance_masks_only",
            "uses_habitat_semantic_sensor": False,
            "uses_goal_ground_truth": False,
            "gaussians": int(gaussians.get_xyz.shape[0]),
            "labelled_gaussians": int((labels >= 0).sum().item()),
            "mean_label_confidence": float(
                confidence[labels >= 0].mean().item()
            ) if bool((labels >= 0).any()) else 0.0,
            "class_counts": class_counts,
            "persistent_instances": len(self.instance_registry),
            "frames_seen": int(self.frames_seen),
            "frames_with_labels": int(self.frames_with_labels),
            "config": asdict(self.config),
            "last_stats": self.last_stats,
        }

    def export(self, mapper: Any, output_dir: Path) -> Optional[Dict[str, str]]:
        if not self.config.enabled:
            return None
        slam = getattr(mapper, "slam", None)
        gaussians = getattr(slam, "gaussians", None) if slam is not None else None
        if gaussians is None:
            return None
        self._ensure_sidecars(gaussians)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        labels, confidence = _semantic_summary(
            gaussians._multi_semantic_evidence,
            gaussians._multi_semantic_observations,
        )
        xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy().astype(np.int16)
        confidence_np = confidence.cpu().numpy().astype(np.float32)
        instance_np = gaussians._multi_instance_ids.cpu().numpy().astype(np.int32)
        observations_np = np.clip(
            gaussians._multi_semantic_observations.cpu().numpy(), 0, 65535
        ).astype(np.uint16)
        evidence_np = gaussians._multi_semantic_evidence.cpu().numpy().astype(np.float16)

        sidecar_path = output_dir / "semantic_instance_labels.npz"
        np.savez_compressed(
            sidecar_path,
            semantic_evidence=evidence_np,
            semantic_class=labels_np,
            semantic_confidence=confidence_np,
            instance_id=instance_np,
            label_observations=observations_np,
            class_names=np.asarray(CLASS_NAMES),
        )
        summary = self.snapshot(mapper)
        summary["instance_registry"] = self.instance_registry
        summary_path = output_dir / "semantic_instance_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

        colors = (
            gaussians._base_rgb_uint8()
            if hasattr(gaussians, "_base_rgb_uint8")
            else np.zeros((xyz.shape[0], 3), dtype=np.uint8)
        )
        from plyfile import PlyData, PlyElement

        dtype = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("semantic_class", "i2"), ("semantic_confidence", "f4"),
            ("instance_id", "i4"), ("label_observations", "u2"),
        ]
        vertices = np.empty((xyz.shape[0],), dtype=dtype)
        for axis, name in enumerate(("x", "y", "z")):
            vertices[name] = xyz[:, axis]
        for channel, name in enumerate(("red", "green", "blue")):
            vertices[name] = colors[:, channel]
        vertices["semantic_class"] = labels_np
        vertices["semantic_confidence"] = confidence_np
        vertices["instance_id"] = instance_np
        vertices["label_observations"] = observations_np
        ply_path = output_dir / "semantic_instance_gaussians.ply"
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
            str(ply_path)
        )
        LOGGER.info("Saved Multi-IIN Gaussian semantic/instance labels to %s", output_dir)
        return {
            "sidecar": str(sidecar_path),
            "summary": str(summary_path),
            "ply": str(ply_path),
        }
