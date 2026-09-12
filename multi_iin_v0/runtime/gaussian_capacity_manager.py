"""Bounded, target-aware Gaussian capacity management for Multi-IIN.

This module is intentionally additive.  It operates on the public runtime
objects created by the existing Gaussian mapper and uses the mapper's own
``prune_points`` implementation so optimizer state and auxiliary tensors stay
aligned.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaussianCapacityConfig:
    enabled: bool = True
    hard_cap: int = 1_500_000
    prune_trigger: int = 1_350_000
    prune_target: int = 1_100_000
    voxel_size_m: float = 0.03
    max_voxel_size_m: float = 0.10
    dense_voxel_threshold: int = 4
    recent_keyframe_grace: int = 5
    semantic_protect_threshold: float = 0.70
    low_opacity_threshold: float = 0.05

    @classmethod
    def from_args(cls, args: Any) -> "GaussianCapacityConfig":
        hard_cap = max(
            1,
            int(getattr(args, "multi_gaussian_hard_cap", 1_500_000)),
        )
        trigger = min(
            hard_cap,
            max(1, int(getattr(args, "multi_gaussian_prune_trigger", 1_350_000))),
        )
        target = min(
            max(1, trigger - 1),
            max(1, int(getattr(args, "multi_gaussian_prune_target", 1_100_000))),
        )
        return cls(
            enabled=bool(
                int(getattr(args, "multi_gaussian_capacity_enabled", 1))
            ),
            hard_cap=hard_cap,
            prune_trigger=trigger,
            prune_target=target,
            voxel_size_m=max(
                1e-3,
                float(getattr(args, "multi_gaussian_capacity_voxel_size_m", 0.03)),
            ),
            max_voxel_size_m=max(
                max(
                    1e-3,
                    float(
                        getattr(
                            args, "multi_gaussian_capacity_voxel_size_m", 0.03
                        )
                    ),
                ),
                float(
                    getattr(
                        args,
                        "multi_gaussian_capacity_max_voxel_size_m",
                        0.10,
                    )
                ),
            ),
            dense_voxel_threshold=max(
                1,
                int(getattr(args, "multi_gaussian_dense_voxel_threshold", 4)),
            ),
            recent_keyframe_grace=max(
                0,
                int(getattr(args, "multi_gaussian_recent_keyframe_grace", 5)),
            ),
            semantic_protect_threshold=float(
                getattr(args, "multi_gaussian_semantic_protect_threshold", 0.70)
            ),
            low_opacity_threshold=max(
                0.0,
                float(getattr(args, "multi_gaussian_low_opacity_threshold", 0.05)),
            ),
        )


def _vector(values: Optional[Iterable], count: int, default: float) -> np.ndarray:
    if values is None:
        return np.full((count,), default, dtype=np.float32)
    result = np.asarray(values).reshape(-1)
    if result.shape[0] != count:
        return np.full((count,), default, dtype=np.float32)
    return result


def _normalized_observation_score(observations: np.ndarray) -> np.ndarray:
    observations = np.maximum(
        0.0,
        np.nan_to_num(observations.astype(np.float32), nan=0.0),
    )
    positive = observations[observations > 0]
    if positive.size == 0:
        return np.zeros_like(observations, dtype=np.float32)
    scale = max(1.0, float(np.percentile(positive, 95)))
    return np.clip(np.log1p(observations) / np.log1p(scale), 0.0, 1.0)


def choose_capacity_prune_mask(
    xyz: np.ndarray,
    opacity: np.ndarray,
    *,
    target_count: int,
    semantic: Optional[np.ndarray] = None,
    reliability: Optional[np.ndarray] = None,
    observations: Optional[np.ndarray] = None,
    keyframe_ids: Optional[np.ndarray] = None,
    confirmed_indices: Optional[np.ndarray] = None,
    current_keyframe_id: int = 0,
    voxel_size_m: float = 0.03,
    max_voxel_size_m: float = 0.10,
    dense_voxel_threshold: int = 4,
    recent_keyframe_grace: int = 5,
    semantic_protect_threshold: float = 0.70,
    low_opacity_threshold: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Choose low-value redundant Gaussians while preserving spatial coverage."""

    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("xyz must have shape (N, >=3)")
    count = int(points.shape[0])
    requested_target = min(count, max(0, int(target_count)))
    remove_requested = max(0, count - requested_target)
    prune = np.zeros((count,), dtype=bool)
    if count == 0 or remove_requested == 0:
        return prune, {
            "count_before": count,
            "requested_target": requested_target,
            "remove_requested": remove_requested,
            "remove_selected": 0,
            "effective_target": count,
        }

    opacity_arr = np.clip(
        np.nan_to_num(_vector(opacity, count, 0.0).astype(np.float32), nan=0.0),
        0.0,
        1.0,
    )
    semantic_arr = np.nan_to_num(
        _vector(semantic, count, 0.0).astype(np.float32), nan=0.0
    )
    reliability_arr = np.clip(
        np.nan_to_num(_vector(reliability, count, 0.5).astype(np.float32), nan=0.5),
        0.0,
        1.0,
    )
    observations_arr = _vector(observations, count, 0.0).astype(np.float32)
    keyframes_arr = _vector(keyframe_ids, count, -1).astype(np.int64)

    protected_confirmed = np.zeros((count,), dtype=bool)
    if confirmed_indices is not None:
        confirmed = np.asarray(confirmed_indices, dtype=np.int64).reshape(-1)
        confirmed = confirmed[(confirmed >= 0) & (confirmed < count)]
        if confirmed.size:
            protected_confirmed[np.unique(confirmed)] = True
    protected_semantic = semantic_arr >= float(semantic_protect_threshold)
    protected_recent = np.zeros((count,), dtype=bool)
    if int(current_keyframe_id) > 0 and int(recent_keyframe_grace) >= 0:
        recent_floor = int(current_keyframe_id) - int(recent_keyframe_grace)
        protected_recent = keyframes_arr >= recent_floor
    protected = protected_confirmed | protected_semantic | protected_recent

    finite_points = np.all(np.isfinite(points[:, :3]), axis=1)
    voxel_size = max(1e-3, float(voxel_size_m))
    maximum_voxel_size = max(voxel_size, float(max_voxel_size_m))
    safe_points = np.where(finite_points[:, None], points[:, :3], 0.0)

    obs_score = _normalized_observation_score(observations_arr)
    age = np.maximum(0, int(current_keyframe_id) - keyframes_arr)
    recency_score = np.exp(-age.astype(np.float32) / 50.0)
    quality = (
        0.40 * opacity_arr
        + 0.25 * obs_score
        + 0.25 * reliability_arr
        + 0.10 * recency_score
    )
    quality[~finite_points] = -1.0

    # Adapt the coverage grid only if the requested capacity could not hold one
    # representative per base-size voxel plus all protected points.  Real
    # indoor maps normally stay at 3 cm; the coarser fallback avoids a capacity
    # deadlock in unusually broad and sparse maps.
    adaptive_attempts = 0
    while True:
        voxels = np.floor(safe_points / voxel_size).astype(np.int64)
        _, inverse, voxel_counts = np.unique(
            voxels,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        protected_per_voxel = np.bincount(
            inverse,
            weights=protected.astype(np.int64),
            minlength=int(voxel_counts.shape[0]),
        )
        coverage_floor = int(protected.sum()) + int(
            np.count_nonzero(protected_per_voxel == 0)
        )
        if coverage_floor <= requested_target or voxel_size >= maximum_voxel_size:
            break
        ratio = float(coverage_floor) / max(float(requested_target), 1.0)
        next_size = (
            maximum_voxel_size
            if adaptive_attempts >= 1
            else min(
                maximum_voxel_size,
                voxel_size * max(1.25, float(np.sqrt(ratio)) * 1.05),
            )
        )
        if next_size <= voxel_size + 1e-9:
            break
        voxel_size = next_size
        adaptive_attempts += 1

    # If a voxel has no protected target/recent point, retain its best Gaussian
    # as a coverage anchor.  If even the maximum voxel size produces more
    # anchors than the hard target can hold, retain the highest-quality subset;
    # capacity remains bounded instead of silently refusing to prune.
    coverage_anchor = np.zeros((count,), dtype=bool)
    unprotected_indices = np.flatnonzero(~protected)
    if unprotected_indices.size:
        ordered = unprotected_indices[
            np.lexsort(
                (
                    -quality[unprotected_indices],
                    inverse[unprotected_indices],
                )
            )
        ]
        ordered_voxels = inverse[ordered]
        first = np.empty((ordered.shape[0],), dtype=bool)
        first[0] = True
        first[1:] = ordered_voxels[1:] != ordered_voxels[:-1]
        anchors = ordered[first]
        anchors = anchors[protected_per_voxel[inverse[anchors]] == 0]
        anchor_budget = max(0, requested_target - int(protected.sum()))
        anchors_before_budget = int(anchors.size)
        if anchors.size > anchor_budget:
            anchor_order = np.argsort(-quality[anchors], kind="stable")
            anchors = anchors[anchor_order[:anchor_budget]]
        coverage_anchor[anchors] = True
    else:
        anchors_before_budget = 0

    removable = ~(protected | coverage_anchor)
    dense_redundant = (
        voxel_counts[inverse] > max(1, int(dense_voxel_threshold))
    ) & removable
    low_opacity = (opacity_arr < float(low_opacity_threshold)) & removable
    invalid = (~finite_points) & removable

    removable_indices = np.flatnonzero(removable)
    if removable_indices.size:
        # Invalid and low-opacity points go first, then dense-voxel duplicates;
        # quality is the tie-breaker within each tier.
        priority = (
            4 * invalid.astype(np.int8)
            + 2 * low_opacity.astype(np.int8)
            + dense_redundant.astype(np.int8)
        )
        removal_order = removable_indices[
            np.lexsort(
                (
                    quality[removable_indices],
                    -priority[removable_indices],
                )
            )
        ]
        selected = removal_order[: min(remove_requested, removal_order.size)]
        prune[selected] = True

    selected_count = int(prune.sum())
    stats = {
        "count_before": count,
        "requested_target": requested_target,
        "remove_requested": remove_requested,
        "remove_selected": selected_count,
        "effective_target": int(count - selected_count),
        "capacity_shortfall": int(remove_requested - selected_count),
        "protected_total": int(protected.sum()),
        "protected_confirmed": int(protected_confirmed.sum()),
        "protected_semantic": int(protected_semantic.sum()),
        "protected_recent": int(protected_recent.sum()),
        "coverage_anchor_count": int(coverage_anchor.sum()),
        "coverage_anchor_candidates": int(anchors_before_budget),
        "coverage_anchor_dropped_for_capacity": int(
            anchors_before_budget - coverage_anchor.sum()
        ),
        "voxel_count": int(voxel_counts.shape[0]),
        "dense_redundant_candidates": int(dense_redundant.sum()),
        "low_opacity_candidates": int(low_opacity.sum()),
        "invalid_candidates": int(invalid.sum()),
        "voxel_size_m": voxel_size,
        "base_voxel_size_m": max(1e-3, float(voxel_size_m)),
        "max_voxel_size_m": maximum_voxel_size,
        "adaptive_voxel_attempts": int(adaptive_attempts),
        "dense_voxel_threshold": int(max(1, dense_voxel_threshold)),
    }
    return prune, stats


class GaussianCapacityManager:
    """Apply watermarked capacity pruning between complete mapping updates."""

    SAFE_PRUNE_STAGES = {"exploration", "ieve_online", "minav_exploration"}

    def __init__(self, config: GaussianCapacityConfig, output_dir: Optional[Path]):
        self.config = config
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.event_count = 0
        self.total_removed = 0
        self.last_event: Optional[Dict[str, Any]] = None

    @classmethod
    def from_args(cls, args: Any, output_dir: Optional[Path] = None):
        return cls(GaussianCapacityConfig.from_args(args), output_dir)

    def configure_mapper(self, mapper: Any) -> None:
        if not self.config.enabled:
            return
        slam = getattr(mapper, "slam", None)
        if slam is None:
            return
        slam.max_gaussians = int(self.config.hard_cap)
        training = getattr(slam, "config", {}).setdefault("Training", {})
        training["max_gaussians"] = int(self.config.hard_cap)
        # The legacy densifier can clone/split many points in one call.  Guard
        # topology growth once the high watermark is reached so that a single
        # mapping update cannot jump over the absolute cap before the after-
        # update pruning hook runs.
        if getattr(slam, "_multi_iin_capacity_growth_guard_installed", False):
            return
        original_train = getattr(slam, "_train_mapping", None)
        if original_train is None:
            return

        config = self.config

        def guarded_train(this, *args, **kwargs):
            gaussians = getattr(this, "gaussians", None)
            count = 0
            if gaussians is not None and hasattr(gaussians, "get_xyz"):
                count = int(gaussians.get_xyz.shape[0])
            if count >= int(config.prune_trigger):
                kwargs["allow_topology_growth"] = False
            return original_train(*args, **kwargs)

        slam._train_mapping = MethodType(guarded_train, slam)
        slam._multi_iin_capacity_growth_guard_installed = True

    @staticmethod
    def _numpy_tensor(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _write_event(self, event: Dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "gaussian_capacity_events.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception as exc:
            LOGGER.warning("Failed to write Gaussian capacity event: %s", exc)

    def maybe_prune(
        self,
        mapper: Any,
        *,
        env_step: int,
        stage: str,
        timing: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.config.enabled:
            return None
        slam = getattr(mapper, "slam", None)
        gaussians = getattr(slam, "gaussians", None) if slam is not None else None
        if gaussians is None or not hasattr(gaussians, "get_xyz"):
            return None
        count = int(gaussians.get_xyz.shape[0])
        if count < int(self.config.prune_trigger):
            return None

        stage_name = str(stage or "")
        if stage_name not in self.SAFE_PRUNE_STAGES:
            return None
        if getattr(gaussians, "optimizer", None) is None or not hasattr(
            gaussians, "prune_points"
        ):
            return None

        confirmed = None
        if hasattr(gaussians, "get_confirmed_target_indices"):
            confirmed = gaussians.get_confirmed_target_indices()
        semantic_for_prune = self._numpy_tensor(
            getattr(gaussians, "_semantic_target", None)
        )
        multi_evidence = self._numpy_tensor(
            getattr(gaussians, "_multi_semantic_evidence", None)
        )
        if multi_evidence is not None and multi_evidence.ndim == 2:
            totals = np.maximum(multi_evidence.sum(axis=1), 1e-6)
            multi_confidence = multi_evidence.max(axis=1) / totals
            if semantic_for_prune is None or semantic_for_prune.shape[0] != count:
                semantic_for_prune = multi_confidence
            else:
                semantic_for_prune = np.maximum(
                    np.asarray(semantic_for_prune).reshape(-1), multi_confidence
                )
        prune_mask, stats = choose_capacity_prune_mask(
            self._numpy_tensor(gaussians.get_xyz),
            self._numpy_tensor(gaussians.get_opacity),
            target_count=int(self.config.prune_target),
            semantic=semantic_for_prune,
            reliability=self._numpy_tensor(
                getattr(gaussians, "get_reliability_1dim", None)
            ),
            observations=self._numpy_tensor(getattr(gaussians, "n_obs", None)),
            keyframe_ids=self._numpy_tensor(
                getattr(gaussians, "unique_kfIDs", None)
            ),
            confirmed_indices=self._numpy_tensor(confirmed),
            current_keyframe_id=int(getattr(slam, "frame_count", 0)),
            voxel_size_m=self.config.voxel_size_m,
            max_voxel_size_m=self.config.max_voxel_size_m,
            dense_voxel_threshold=self.config.dense_voxel_threshold,
            recent_keyframe_grace=self.config.recent_keyframe_grace,
            semantic_protect_threshold=self.config.semantic_protect_threshold,
            low_opacity_threshold=self.config.low_opacity_threshold,
        )
        remove_count = int(prune_mask.sum())
        if remove_count <= 0:
            return None

        import torch

        device = gaussians.get_xyz.device
        mask_tensor = torch.from_numpy(prune_mask).to(device=device, dtype=torch.bool)
        with torch.no_grad():
            gaussians.prune_points(mask_tensor)
        count_after = int(gaussians.get_xyz.shape[0])
        self.event_count += 1
        self.total_removed += int(count - count_after)
        event = {
            "event": "gaussian_capacity_prune",
            "event_index": int(self.event_count),
            "env_step": int(env_step),
            "frame_count": int(getattr(slam, "frame_count", 0)),
            "stage": stage_name,
            "timing": str(timing),
            "count_after": count_after,
            "hard_cap": int(self.config.hard_cap),
            "prune_trigger": int(self.config.prune_trigger),
            "prune_target": int(self.config.prune_target),
            **stats,
        }
        self.last_event = event
        self._write_event(event)
        LOGGER.info("[GAUSSIAN_CAPACITY] %s", json.dumps(event, sort_keys=True))
        return event

    def snapshot(self, mapper: Any = None) -> Dict[str, Any]:
        count = None
        if mapper is not None:
            slam = getattr(mapper, "slam", None)
            gaussians = getattr(slam, "gaussians", None) if slam is not None else None
            if gaussians is not None and hasattr(gaussians, "get_xyz"):
                count = int(gaussians.get_xyz.shape[0])
        return {
            "config": asdict(self.config),
            "current_count": count,
            "event_count": int(self.event_count),
            "total_removed": int(self.total_removed),
            "last_event": self.last_event,
        }


def install_minav_full_snapshot_policy(manager: Any, enabled: bool) -> None:
    """Suppress repeated full-scene debug PLYs while retaining target-only data."""

    if manager is None or bool(enabled):
        return

    def _disabled_snapshot(self, *args, **kwargs):
        del self, args, kwargs
        return None

    for name in (
        "_save_confirmed_full_gaussian_snapshot",
        "_save_failed_target_confirmation_full_gaussian_snapshot",
    ):
        if hasattr(manager, name):
            setattr(manager, name, MethodType(_disabled_snapshot, manager))
    setattr(manager, "_multi_iin_repeated_full_snapshots_enabled", False)
