"""Bounded localization runtime used only by the Multi-IIN runner.

The upstream MinAV PRE implementation evaluates 3,010 virtual views.  That
search is useful as an exhaustive diagnostic, but it is disproportionate for
the same-floor Multi-IIN smoke benchmark.  This module installs process-local
wrappers which retain all PRE stages while bounding their sample counts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Any, Dict

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FastLocalizationConfig:
    enabled: bool = True
    stage1_z_samples: int = 12
    stage1_pitch_samples: int = 5
    stage1_y_half_range_m: float = 1.2
    stage2_position_samples: int = 16
    stage3_position_samples: int = 4
    adaptive_pre_enabled: bool = True
    adaptive_pre_score_threshold: float = 100.0
    light_stage1_z_samples: int = 6
    light_stage1_pitch_samples: int = 3
    light_stage2_position_samples: int = 4
    light_stage3_position_samples: int = 2

    @classmethod
    def from_args(cls, args: Any) -> "FastLocalizationConfig":
        return cls(
            enabled=bool(int(getattr(args, "multi_fast_localization", 1))),
            stage1_z_samples=max(
                2, int(getattr(args, "multi_pre_stage1_z_samples", 12))
            ),
            stage1_pitch_samples=max(
                1, int(getattr(args, "multi_pre_stage1_pitch_samples", 5))
            ),
            stage1_y_half_range_m=max(
                0.05,
                float(getattr(args, "multi_pre_stage1_y_half_range_m", 1.2)),
            ),
            stage2_position_samples=max(
                1,
                int(getattr(args, "multi_pre_stage2_position_samples", 16)),
            ),
            stage3_position_samples=max(
                1,
                int(getattr(args, "multi_pre_stage3_position_samples", 4)),
            ),
            adaptive_pre_enabled=bool(int(getattr(args, "multi_pre_adaptive_enabled", 1))),
            adaptive_pre_score_threshold=float(
                getattr(args, "multi_pre_adaptive_score_threshold", 100.0)
            ),
            light_stage1_z_samples=max(
                2, int(getattr(args, "multi_pre_light_stage1_z_samples", 6))
            ),
            light_stage1_pitch_samples=max(
                1, int(getattr(args, "multi_pre_light_stage1_pitch_samples", 3))
            ),
            light_stage2_position_samples=max(
                1, int(getattr(args, "multi_pre_light_stage2_position_samples", 4))
            ),
            light_stage3_position_samples=max(
                1, int(getattr(args, "multi_pre_light_stage3_position_samples", 2))
            ),
        )

    @property
    def expected_pre_renders(self) -> int:
        # Stage 1 + two 3x3 coarse rounds + one 3x3 fine round.
        return int(
            self.stage1_z_samples * self.stage1_pitch_samples
            + 2 * self.stage2_position_samples * 9
            + self.stage3_position_samples * 9
        )


_CONFIG = FastLocalizationConfig()
_INSTALLED = False
_ORIGINAL_STAGE2 = None
_ORIGINAL_STAGE3 = None
_ORIGINAL_NVSM = None
_ORIGINAL_COMPLETE = None
_ORIGINAL_SELECT_CLUSTER = None
_ORIGINAL_PRE = None


def _locked_center(localizer):
    """Return a manager-confirmed center, if one was supplied.

    The semantic Gaussian filter is intentionally still used for the render
    cloud.  It must not, however, replace a center that was already confirmed
    from multi-frame evidence.
    """
    value = getattr(localizer, "_multi_iin_confirmed_center", None)
    if value is None:
        return None
    try:
        center = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if center.size < 3 or not np.all(np.isfinite(center[:3])):
        return None
    return center[:3].copy()


def set_confirmed_center_lock(localizer: Any, center: Any, trace=None) -> bool:
    """Attach/clear the manager-confirmed center on a LocalizationSystem."""
    if localizer is None:
        return False
    if center is None:
        localizer._multi_iin_confirmed_center = None
        localizer._multi_iin_trace = trace
        return False
    try:
        value = np.asarray(center, dtype=np.float32).reshape(-1)
    except Exception:
        return False
    if value.size < 3 or not np.all(np.isfinite(value[:3])):
        return False
    localizer._multi_iin_confirmed_center = value[:3].copy()
    localizer._multi_iin_trace = trace
    return True


def _locked_nvsm(self):
    """Run NVSM with the confirmed center as an additive render anchor."""
    center = _locked_center(self)
    if center is None:
        return _ORIGINAL_NVSM(self)

    mapper = getattr(self, "slam_mapper", None)
    if mapper is None:
        return _ORIGINAL_NVSM(self)
    stats_method = getattr(mapper, "get_target_semantic_stats", None)
    clusters_method = getattr(mapper, "get_instance_cluster_centers", None)

    def locked_stats(*args, **kwargs):
        stats = stats_method(*args, **kwargs) if callable(stats_method) else {}
        if not isinstance(stats, dict):
            stats = {}
        result = dict(stats)
        result["center"] = center.copy()
        return result

    def locked_clusters(*args, **kwargs):
        # Returning no auxiliary clusters forces the existing NVSM code down
        # its single-center path, while preserving the filtered Gaussian cloud
        # and all visualization/audit output.
        del args, kwargs
        return {"num_clusters": 0, "centers_3d": [], "cluster_sizes": []}

    try:
        if callable(stats_method):
            mapper.get_target_semantic_stats = locked_stats
        if callable(clusters_method):
            mapper.get_instance_cluster_centers = locked_clusters
        LOGGER.info(
            "Multi-IIN confirmed center lock: NVSM render center=[%.4f, %.4f, %.4f]",
            float(center[0]), float(center[1]), float(center[2]),
        )
        trace = getattr(self, "_multi_iin_trace", None)
        if callable(trace):
            trace("confirmed_center_lock_nvsm", center=center)
        return _ORIGINAL_NVSM(self)
    finally:
        if callable(stats_method):
            mapper.get_target_semantic_stats = stats_method
        if callable(clusters_method):
            mapper.get_instance_cluster_centers = clusters_method


def _locked_complete(self, *args, **kwargs):
    """Keep the object identity locked without discarding the IIN view pose.

    The confirmed Gaussian center is an instance/render anchor.  Instance
    ImageNav, however, is evaluated against navigable view points, so the
    FEM/PRE/REF camera position must remain the navigation target.
    """
    result = _ORIGINAL_COMPLETE(self, *args, **kwargs)
    center = _locked_center(self)
    if center is None or not isinstance(result, dict):
        return result
    if result.get("status") != "localization_complete":
        return result
    result = dict(result)
    nav_pose = result.get("best_pose")
    camera_position = None
    if nav_pose is not None:
        try:
            pose = np.asarray(nav_pose, dtype=np.float64)
            if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
                camera_position = np.linalg.inv(pose)[:3, 3].astype(np.float32)
        except Exception:
            camera_position = None
    if camera_position is not None:
        result["target_position"] = camera_position.copy()
        result["localized_camera_position"] = camera_position.copy()
        result["position_source"] = "localized_target_camera_pose"
    else:
        # A missing localization pose is a real failure mode; retaining the
        # center fallback keeps existing behavior without inventing a pose.
        result["target_position"] = center.copy()
        result["position_source"] = "confirmed_target_center_fallback"
    result["target_object_center"] = center.copy()
    result["confirmed_center_lock"] = center.copy()
    LOGGER.info(
        "Multi-IIN localization split: object center=[%.4f, %.4f, %.4f], camera=%s",
        float(center[0]), float(center[1]), float(center[2]),
        None if camera_position is None else camera_position.tolist(),
    )
    return result


def _locked_select_cluster(self, *args, **kwargs):
    center = _locked_center(self)
    if center is None:
        return _ORIGINAL_SELECT_CLUSTER(self, *args, **kwargs)
    return center.copy(), {
        "source": "confirmed_target_center_lock",
        "center": center.copy(),
        "size": int(getattr(self, "_multi_iin_confirmed_index_count", 0) or 0),
        "reachable": False,
        "locked": True,
    }


def _fast_stage1_z_pitch(self, pos, yaw, pitch, roll):
    half_range = float(_CONFIG.stage1_y_half_range_m)
    offsets = np.linspace(
        -half_range,
        half_range,
        int(_CONFIG.stage1_z_samples),
        dtype=np.float32,
    )
    candidates = []
    n_renders = 0
    for y_offset in offsets:
        # Keep the original direction convention: a higher virtual camera
        # looks down; a lower camera looks up.  The same-floor range avoids
        # spending most of PRE on physically implausible ±4 m hypotheses.
        pitch_limit = -45.0 if float(y_offset) > 0.0 else 45.0
        pitch_offsets = np.linspace(
            0.0,
            np.deg2rad(pitch_limit),
            int(_CONFIG.stage1_pitch_samples),
            dtype=np.float32,
        )
        new_pos = np.asarray(pos, dtype=np.float64).copy()
        new_pos[1] += float(y_offset)
        for pitch_offset in pitch_offsets:
            pose = self._euler_pos_to_w2c(
                new_pos, yaw, pitch + float(pitch_offset), roll
            )
            image = self._render_view(pose)
            n_renders += 1
            if image is None:
                continue
            score = self.fem._compute_match_score(image, feature_mode="uniform")
            candidates.append(
                {"pose": pose, "score": score, "pos": new_pos.copy()}
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    LOGGER.info(
        "Multi-IIN fast PRE stage1: renders=%d valid=%d half_range=%.2fm",
        n_renders,
        len(candidates),
        half_range,
    )
    return candidates, n_renders


def _bounded_stage2(
    self,
    center_pos,
    yaw,
    pitch,
    roll,
    cube_half=0.5,
    n_samples=99,
    angle_offsets_deg=None,
):
    del n_samples
    positions = int(_CONFIG.stage2_position_samples)
    return _ORIGINAL_STAGE2(
        self,
        center_pos,
        yaw,
        pitch,
        roll,
        cube_half=cube_half,
        n_samples=max(0, positions - 1),
        angle_offsets_deg=[-8.0, 0.0, 8.0],
    )


def _bounded_stage3(
    self,
    center_pos,
    yaw,
    pitch,
    roll,
    cube_half=0.05,
    n_samples=9,
    angle_offsets_deg=None,
):
    del n_samples, angle_offsets_deg
    positions = int(_CONFIG.stage3_position_samples)
    return _ORIGINAL_STAGE3(
        self,
        center_pos,
        yaw,
        pitch,
        roll,
        cube_half=cube_half,
        n_samples=max(0, positions - 1),
        angle_offsets_deg=[-3.0, 0.0, 3.0],
    )


def _timed_stage(stage_name, original):
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            timings = getattr(self, "_multi_iin_stage_timings", None)
            if timings is None:
                timings = []
                self._multi_iin_stage_timings = timings
            timings.append({"stage": stage_name, "elapsed_s": float(elapsed)})
            LOGGER.info(
                "Multi-IIN localization timing: stage=%s elapsed_s=%.3f",
                stage_name,
                elapsed,
            )
            if stage_name == "FEM":
                # FEM has already copied the selected pose and saved its audit
                # image.  Hundreds of full-resolution NVSM arrays are no
                # longer needed by PRE/REF and caused large RSS accumulation.
                state = str(getattr(getattr(self, "current_state", None), "value", ""))
                if state != "fem":
                    self.rendered_images = []

    return wrapped


def _adaptive_pre(original):
    """Run a small PRE first only for very high FEM confidence.

    A failed/non-improving light pass is followed by the normal bounded PRE,
    so this optimization cannot turn a recoverable high-confidence case into
    a localization failure.  The upstream localizer remains untouched.
    """
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        score = float(getattr(self, "fem_best_score", 0.0) or 0.0)
        if not _CONFIG.adaptive_pre_enabled or score < _CONFIG.adaptive_pre_score_threshold:
            return original(self, *args, **kwargs)

        full_config = _CONFIG
        light_config = FastLocalizationConfig(
            enabled=True,
            stage1_z_samples=full_config.light_stage1_z_samples,
            stage1_pitch_samples=full_config.light_stage1_pitch_samples,
            stage1_y_half_range_m=full_config.stage1_y_half_range_m,
            stage2_position_samples=full_config.light_stage2_position_samples,
            stage3_position_samples=full_config.light_stage3_position_samples,
            adaptive_pre_enabled=full_config.adaptive_pre_enabled,
            adaptive_pre_score_threshold=full_config.adaptive_pre_score_threshold,
            light_stage1_z_samples=full_config.light_stage1_z_samples,
            light_stage1_pitch_samples=full_config.light_stage1_pitch_samples,
            light_stage2_position_samples=full_config.light_stage2_position_samples,
            light_stage3_position_samples=full_config.light_stage3_position_samples,
        )
        globals()["_CONFIG"] = light_config
        try:
            LOGGER.info(
                "Multi-IIN adaptive PRE: high FEM score %.2f, light budget=%d renders",
                score,
                light_config.expected_pre_renders,
            )
            light_result = original(self, *args, **kwargs)
        finally:
            globals()["_CONFIG"] = full_config

        # PRE's contract is success+improved.  If the light pass does not
        # improve the pose, rerun the normal bounded search for quality.
        light_pre = light_result.get("pre_result", {}) if isinstance(light_result, dict) else {}
        if isinstance(light_pre, dict) and light_pre.get("success") and light_pre.get("improved"):
            light_result = dict(light_result)
            light_result["adaptive_pre"] = "light"
            light_result["light_pre_renders"] = int(light_config.expected_pre_renders)
            return light_result

        LOGGER.info("Multi-IIN adaptive PRE: light pass did not improve; falling back to full bounded PRE")
        full_result = original(self, *args, **kwargs)
        if isinstance(full_result, dict):
            full_result = dict(full_result)
            full_result["adaptive_pre"] = "light_then_full_fallback"
            full_result["light_pre_renders"] = int(light_config.expected_pre_renders)
        return full_result

    return wrapped


def install_fast_localization(args: Any) -> FastLocalizationConfig:
    """Install additive Multi-IIN-only PRE bounds and stage timers."""

    global _CONFIG, _INSTALLED, _ORIGINAL_STAGE2, _ORIGINAL_STAGE3, _ORIGINAL_NVSM, _ORIGINAL_COMPLETE, _ORIGINAL_SELECT_CLUSTER, _ORIGINAL_PRE
    _CONFIG = FastLocalizationConfig.from_args(args)
    if _INSTALLED:
        return _CONFIG

    from ieve_minav.localizer.localization_system import LocalizationSystem

    for method_name, stage_name in (
        ("_execute_nvsm", "NVSM"),
        ("_execute_fem", "FEM"),
        ("_execute_pre", "PRE"),
        ("_execute_ref", "REF"),
    ):
        original = getattr(LocalizationSystem, method_name)
        setattr(LocalizationSystem, method_name, _timed_stage(stage_name, original))

    # Adaptive PRE wraps the already-timed method and changes only the
    # process-local sampling configuration for the first high-confidence pass.
    _ORIGINAL_PRE = LocalizationSystem._execute_pre
    LocalizationSystem._execute_pre = _adaptive_pre(_ORIGINAL_PRE)

    # Additive center-lock wrappers.  They are installed process-locally by
    # the Multi-IIN runner and do not alter the upstream localization source.
    _ORIGINAL_NVSM = LocalizationSystem._execute_nvsm
    _ORIGINAL_COMPLETE = LocalizationSystem._make_localization_complete_result
    _ORIGINAL_SELECT_CLUSTER = LocalizationSystem._select_target_gaussian_cluster_from_ref_view
    LocalizationSystem._execute_nvsm = _locked_nvsm
    LocalizationSystem._make_localization_complete_result = _locked_complete
    LocalizationSystem._select_target_gaussian_cluster_from_ref_view = _locked_select_cluster

    if _CONFIG.enabled:
        from ieve_minav.localizer.pose_pre_refiner import PosePreRefiner

        _ORIGINAL_STAGE2 = PosePreRefiner._stage2_cube_perturb
        _ORIGINAL_STAGE3 = PosePreRefiner._stage3_fine
        PosePreRefiner._stage1_z_pitch = _fast_stage1_z_pitch
        PosePreRefiner._stage2_cube_perturb = _bounded_stage2
        PosePreRefiner._stage3_fine = _bounded_stage3

    _INSTALLED = True
    LOGGER.info("Installed Multi-IIN localization runtime: %s", asdict(_CONFIG))
    return _CONFIG
