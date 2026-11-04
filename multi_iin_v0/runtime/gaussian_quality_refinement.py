"""Sparse quality-triggered refinement for the Multi-IIN Gaussian mapper.

This module is additive: it wraps a mapper instance owned by Multi-IIN and
does not change the default Gaussian mapper used by the original IEVE runner.
The gate compares the current RGB observation with a current-pose render.  It
does not consume semantic ground truth, target coordinates, or task metrics.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GaussianQualityRefinementConfig:
    enabled: bool = True
    excess_black_threshold: float = 0.02
    strong_darkening_threshold: float = 0.10
    extra_current_frame_iters: int = 6
    minimum_visible_fraction: float = 0.20
    log_all_checks: bool = False
    rollback_on_regression: bool = True
    max_psnr_drop_db: float = 0.25
    max_excess_black_increase: float = 0.002
    max_darkening_increase: float = 0.01
    history_validation_views: int = 3
    history_validation_stride: int = 5

    @classmethod
    def from_args(cls, args: Any) -> "GaussianQualityRefinementConfig":
        return cls(
            enabled=bool(
                int(getattr(args, "multi_gaussian_quality_refinement", 1))
            ),
            excess_black_threshold=max(
                0.0,
                float(
                    getattr(
                        args,
                        "multi_gaussian_quality_excess_black_threshold",
                        0.02,
                    )
                ),
            ),
            strong_darkening_threshold=max(
                0.0,
                float(
                    getattr(
                        args,
                        "multi_gaussian_quality_darkening_threshold",
                        0.10,
                    )
                ),
            ),
            extra_current_frame_iters=max(
                0,
                int(
                    getattr(
                        args,
                        "multi_gaussian_quality_extra_current_iters",
                        6,
                    )
                ),
            ),
            minimum_visible_fraction=min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            args,
                            "multi_gaussian_quality_min_visible_fraction",
                            0.20,
                        )
                    ),
                ),
            ),
            log_all_checks=bool(
                int(getattr(args, "multi_gaussian_quality_log_all_checks", 0))
            ),
            rollback_on_regression=bool(
                int(
                    getattr(
                        args,
                        "multi_gaussian_quality_rollback_on_regression",
                        1,
                    )
                )
            ),
            max_psnr_drop_db=max(
                0.0,
                float(
                    getattr(
                        args,
                        "multi_gaussian_quality_max_psnr_drop_db",
                        0.25,
                    )
                ),
            ),
            max_excess_black_increase=max(
                0.0,
                float(
                    getattr(
                        args,
                        "multi_gaussian_quality_max_excess_black_increase",
                        0.002,
                    )
                ),
            ),
            max_darkening_increase=max(
                0.0,
                float(
                    getattr(
                        args,
                        "multi_gaussian_quality_max_darkening_increase",
                        0.01,
                    )
                ),
            ),
            history_validation_views=max(
                0,
                int(
                    getattr(
                        args,
                        "multi_gaussian_quality_history_validation_views",
                        3,
                    )
                ),
            ),
            history_validation_stride=max(
                1,
                int(
                    getattr(
                        args,
                        "multi_gaussian_quality_history_validation_stride",
                        5,
                    )
                ),
            ),
        )


def current_render_quality(rgb: Any, rendered: Any) -> Dict[str, float]:
    observation = np.asarray(rgb, dtype=np.uint8)
    reconstruction = np.asarray(rendered, dtype=np.uint8)
    if observation.ndim != 3 or observation.shape[2] < 3:
        raise ValueError("rgb observation must have shape HxWx3")
    if reconstruction.ndim != 3 or reconstruction.shape[2] < 3:
        raise ValueError("rendered image must have shape HxWx3")
    if reconstruction.shape[:2] != observation.shape[:2]:
        reconstruction = cv2.resize(
            reconstruction,
            (observation.shape[1], observation.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    # Mean intensity is sufficient for the conservative artifact gate and is
    # independent of RGB/BGR channel ordering.
    gt_gray = observation[..., :3].astype(np.float32).mean(axis=2)
    render_gray = reconstruction[..., :3].astype(np.float32).mean(axis=2)
    visible = gt_gray >= 32.0
    excess_black = visible & (render_gray <= 8.0)
    strong_darkening = (gt_gray >= 48.0) & (render_gray < 0.35 * gt_gray)
    mse = float(
        np.mean(
            (
                observation[..., :3].astype(np.float32)
                - reconstruction[..., :3].astype(np.float32)
            )
            ** 2
        )
    )
    psnr = float("inf") if mse == 0.0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
    return {
        "visible_fraction": float(np.mean(visible)),
        "excess_black_fraction": float(np.mean(excess_black)),
        "strong_darkening_fraction": float(np.mean(strong_darkening)),
        "psnr_db": psnr,
    }


def should_refine(
    metrics: Dict[str, float], config: GaussianQualityRefinementConfig
) -> bool:
    if metrics["visible_fraction"] < config.minimum_visible_fraction:
        return False
    return bool(
        metrics["excess_black_fraction"] >= config.excess_black_threshold
        or metrics["strong_darkening_fraction"]
        >= config.strong_darkening_threshold
    )


def refinement_regressions(
    before: Dict[str, float],
    after: Dict[str, float],
    config: GaussianQualityRefinementConfig,
) -> List[str]:
    """Return observable RGB-quality regressions; no task truth is used."""

    regressions: List[str] = []
    before_psnr = float(before["psnr_db"])
    after_psnr = float(after["psnr_db"])
    if (
        np.isinf(before_psnr)
        and not np.isinf(after_psnr)
    ) or (
        np.isfinite(before_psnr)
        and after_psnr < before_psnr - config.max_psnr_drop_db
    ):
        regressions.append("psnr_drop")
    if (
        float(after["excess_black_fraction"])
        > float(before["excess_black_fraction"])
        + config.max_excess_black_increase
    ):
        regressions.append("excess_black_increase")
    if (
        float(after["strong_darkening_fraction"])
        > float(before["strong_darkening_fraction"])
        + config.max_darkening_increase
    ):
        regressions.append("strong_darkening_increase")
    return regressions


def refinement_is_acceptable(
    before: Dict[str, float],
    after: Dict[str, float],
    config: GaussianQualityRefinementConfig,
) -> bool:
    return not refinement_regressions(before, after, config)


def _cpu_tensor_snapshot(value: Any) -> Tuple[Any, Optional[str]]:
    """Clone a tensor to CPU while remembering its original device."""

    if not hasattr(value, "detach") or not hasattr(value, "device"):
        return copy.deepcopy(value), None
    return value.detach().cpu().clone(), str(value.device)


def _capture_training_state(slam: Any) -> Dict[str, Any]:
    """Capture state mutated by topology-preserving mapping iterations.

    Tensor payloads live on CPU so a rare quality check cannot double the
    long-lived GPU footprint of a large Gaussian map.
    """

    gaussians = getattr(slam, "gaussians", None)
    optimizer = getattr(gaussians, "optimizer", None)
    if gaussians is None or optimizer is None:
        raise RuntimeError("Gaussian optimizer is unavailable")

    groups = []
    for index, group in enumerate(optimizer.param_groups):
        params = list(group.get("params", []))
        if len(params) != 1:
            raise RuntimeError(
                f"unsupported optimizer group {index}: expected one parameter"
            )
        parameter = params[0]
        parameter_value, parameter_device = _cpu_tensor_snapshot(parameter)
        optimizer_state = {}
        for key, value in optimizer.state.get(parameter, {}).items():
            snapshot_value, snapshot_device = _cpu_tensor_snapshot(value)
            optimizer_state[key] = {
                "value": snapshot_value,
                "device": snapshot_device,
            }
        groups.append(
            {
                "index": index,
                "name": group.get("name", str(index)),
                "parameter": parameter_value,
                "parameter_device": parameter_device,
                "optimizer_state": optimizer_state,
                "lr": group.get("lr"),
            }
        )

    auxiliary = {}
    for name in ("max_radii2D", "xyz_gradient_accum", "denom"):
        value = getattr(gaussians, name, None)
        if value is not None:
            snapshot_value, snapshot_device = _cpu_tensor_snapshot(value)
            auxiliary[name] = {
                "value": snapshot_value,
                "device": snapshot_device,
            }
    return {
        "groups": groups,
        "auxiliary": auxiliary,
        "iteration_count": int(getattr(slam, "iteration_count", 0)),
    }


def _restore_tensor_value(current: Any, snapshot: Dict[str, Any]) -> Any:
    value = snapshot["value"]
    device = snapshot.get("device")
    if device is None:
        return copy.deepcopy(value)
    if (
        hasattr(current, "copy_")
        and tuple(current.shape) == tuple(value.shape)
        and str(current.dtype) == str(value.dtype)
    ):
        current.copy_(value)
        return current
    return value.to(device=device)


def _restore_training_state(slam: Any, snapshot: Dict[str, Any]) -> None:
    gaussians = slam.gaussians
    optimizer = gaussians.optimizer
    with torch.no_grad():
        for saved_group in snapshot["groups"]:
            group = optimizer.param_groups[int(saved_group["index"])]
            parameter = group["params"][0]
            parameter.copy_(saved_group["parameter"])
            current_state = optimizer.state.setdefault(parameter, {})
            saved_state = saved_group["optimizer_state"]
            for stale_key in set(current_state) - set(saved_state):
                del current_state[stale_key]
            for key, saved_value in saved_state.items():
                current_state[key] = _restore_tensor_value(
                    current_state.get(key), saved_value
                )
            group["lr"] = saved_group["lr"]
        for name, saved_value in snapshot["auxiliary"].items():
            current = getattr(gaussians, name, None)
            setattr(
                gaussians,
                name,
                _restore_tensor_value(current, saved_value),
            )
        slam.iteration_count = int(snapshot["iteration_count"])


def _camera_rgb_uint8(camera: Any) -> Optional[np.ndarray]:
    image = getattr(camera, "original_image", None)
    if image is None:
        return None
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3:
        return None
    if image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = image.transpose(1, 2, 0)
    image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.5:
            image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def _history_validation_inputs(
    slam: Any,
    current_kf: int,
    config: GaussianQualityRefinementConfig,
) -> List[Tuple[int, np.ndarray]]:
    count = int(config.history_validation_views)
    if count <= 0:
        return []
    ids = [int(kf_id) for kf_id in getattr(slam, "kf_indices", [])]
    candidates = [kf_id for kf_id in ids if kf_id != int(current_kf)]
    selected = list(reversed(candidates))[:: config.history_validation_stride][:count]
    result = []
    keyframes = getattr(slam, "keyframes", {})
    for kf_id in selected:
        reference = _camera_rgb_uint8(keyframes.get(kf_id))
        if reference is not None:
            result.append((kf_id, reference))
    return result


def _render_quality_for_kf(
    slam: Any, kf_id: int, reference: np.ndarray
) -> Optional[Dict[str, float]]:
    render_result = slam.render_frame(int(kf_id))
    if not isinstance(render_result, dict) or render_result.get("rendered") is None:
        return None
    return current_render_quality(reference, render_result["rendered"])


class GaussianQualityRefinementManager:
    EXPLORATION_STAGES = {"exploration", "ieve_online", "minav_exploration"}

    def __init__(
        self,
        config: GaussianQualityRefinementConfig,
        output_dir: Optional[Path],
    ):
        self.config = config
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.check_count = 0
        self.trigger_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.rollback_count = 0
        self.snapshot_failure_count = 0
        self.last_event: Optional[Dict[str, Any]] = None

    @classmethod
    def from_args(cls, args: Any, output_dir: Optional[Path] = None):
        return cls(GaussianQualityRefinementConfig.from_args(args), output_dir)

    def _write_event(self, event: Dict[str, Any]) -> None:
        self.last_event = event
        if self.output_dir is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "gaussian_quality_refinement.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception as exc:
            LOGGER.warning("Failed to write Gaussian quality event: %s", exc)

    def configure_mapper(self, mapper: Any) -> None:
        if not self.config.enabled:
            return
        slam = getattr(mapper, "slam", None)
        if slam is None or getattr(
            slam, "_multi_iin_quality_refinement_installed", False
        ):
            return
        original_add_frame = getattr(slam, "add_frame", None)
        if original_add_frame is None:
            return
        manager = self

        def quality_guarded_add_frame(
            this,
            rgb,
            depth,
            position,
            rotation,
            target_mask=None,
            region_type="target",
            training_context=None,
        ):
            result = original_add_frame(
                rgb,
                depth,
                position,
                rotation,
                target_mask=target_mask,
                region_type=region_type,
                training_context=training_context,
            )
            context = training_context or {}
            stage = str(context.get("stage", "") or "")
            if stage not in manager.EXPLORATION_STAGES:
                return result
            if not getattr(this, "kf_indices", None):
                return result
            current_kf = int(this.kf_indices[-1])
            render_result = this.render_frame(current_kf)
            if not isinstance(render_result, dict) or render_result.get("rendered") is None:
                return result
            manager.check_count += 1
            before = current_render_quality(rgb, render_result["rendered"])
            triggered = should_refine(before, manager.config)
            event: Dict[str, Any] = {
                "event": "gaussian_quality_refinement",
                "frame": current_kf,
                "env_step": int(context.get("step", -1) or -1),
                "stage": stage,
                "triggered": triggered,
                "extra_iters": 0,
                "before": before,
            }
            if triggered and manager.config.extra_current_frame_iters > 0:
                manager.trigger_count += 1
                history_inputs = _history_validation_inputs(
                    this, current_kf, manager.config
                )
                history_before = []
                for history_kf, history_rgb in history_inputs:
                    metrics = _render_quality_for_kf(
                        this, history_kf, history_rgb
                    )
                    if metrics is not None:
                        history_before.append((history_kf, history_rgb, metrics))
                event["validation_history_frames"] = [
                    int(item[0]) for item in history_before
                ]
                training_snapshot = None
                if manager.config.rollback_on_regression:
                    snapshot_start = time.perf_counter()
                    try:
                        training_snapshot = _capture_training_state(this)
                        event["snapshot_seconds"] = float(
                            time.perf_counter() - snapshot_start
                        )
                    except Exception as exc:
                        manager.snapshot_failure_count += 1
                        event.update(
                            {
                                "accepted": False,
                                "rolled_back": False,
                                "rejection_reasons": ["snapshot_failed"],
                                "snapshot_error": str(exc),
                            }
                        )
                        manager._write_event(event)
                        LOGGER.warning(
                            "[GAUSSIAN_QUALITY] skipped unsafe refinement: %s",
                            exc,
                        )
                        return result
                # Sampling a one-element keyframe pool still advances NumPy's
                # global RNG.  Restore it so the quality-only branch cannot
                # perturb later planner or evaluation randomness.
                numpy_rng_state = np.random.get_state()
                try:
                    this._train_mapping(
                        override_iterations=manager.config.extra_current_frame_iters,
                        preferred_kf_ids=[current_kf],
                        allow_topology_growth=False,
                    )
                finally:
                    np.random.set_state(numpy_rng_state)
                event["extra_iters"] = manager.config.extra_current_frame_iters
                post_result = this.render_frame(current_kf)
                if isinstance(post_result, dict) and post_result.get("rendered") is not None:
                    event["after"] = current_render_quality(
                        rgb, post_result["rendered"]
                    )
                    regressions = [
                        f"current:{reason}"
                        for reason in refinement_regressions(
                            before, event["after"], manager.config
                        )
                    ]
                else:
                    regressions = ["current:post_render_unavailable"]

                history_validation = []
                for history_kf, history_rgb, history_metrics_before in history_before:
                    history_metrics_after = _render_quality_for_kf(
                        this, history_kf, history_rgb
                    )
                    if history_metrics_after is None:
                        history_reasons = ["post_render_unavailable"]
                    else:
                        history_reasons = refinement_regressions(
                            history_metrics_before,
                            history_metrics_after,
                            manager.config,
                        )
                    history_validation.append(
                        {
                            "frame": int(history_kf),
                            "before": history_metrics_before,
                            "after": history_metrics_after,
                            "regressions": history_reasons,
                        }
                    )
                    regressions.extend(
                        f"history_{history_kf}:{reason}"
                        for reason in history_reasons
                    )
                event["history_validation"] = history_validation

                accepted = not regressions
                event["accepted"] = accepted
                event["rejection_reasons"] = regressions
                event["rolled_back"] = False
                if accepted:
                    manager.accepted_count += 1
                else:
                    manager.rejected_count += 1
                    if manager.config.rollback_on_regression:
                        rollback_start = time.perf_counter()
                        _restore_training_state(this, training_snapshot)
                        manager.rollback_count += 1
                        event["rolled_back"] = True
                        event["rollback_seconds"] = float(
                            time.perf_counter() - rollback_start
                        )
                        restored_result = this.render_frame(current_kf)
                        if (
                            isinstance(restored_result, dict)
                            and restored_result.get("rendered") is not None
                        ):
                            event["restored"] = current_render_quality(
                                rgb, restored_result["rendered"]
                            )
            if triggered or manager.config.log_all_checks:
                manager._write_event(event)
                LOGGER.info("[GAUSSIAN_QUALITY] %s", json.dumps(event, sort_keys=True))
            return result

        slam.add_frame = MethodType(quality_guarded_add_frame, slam)
        slam._multi_iin_quality_refinement_installed = True
        slam._multi_iin_quality_refinement_config = asdict(self.config)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "checks": int(self.check_count),
            "triggers": int(self.trigger_count),
            "accepted": int(self.accepted_count),
            "rejected": int(self.rejected_count),
            "rollbacks": int(self.rollback_count),
            "snapshot_failures": int(self.snapshot_failure_count),
            "last_event": self.last_event,
        }
