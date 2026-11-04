"""Perception-only prevalidation of target 2 from a shared map.

Continuous-Shared already contains RGB Gaussians and a 2D semantic map when
the second goal is revealed.  This module checks those existing map
candidates with NVSM + FEM before ordinary exploration moves the robot.  It
never consumes Habitat goal coordinates, object ids, distance, success, or
SPL; a successful prevalidation only skips the *discovery* motion.  The normal
Buffer/localization/navigation stages still choose and execute a reachable
observation pose.
"""

from __future__ import annotations

import time
import types
from typing import Any, Dict, List, Optional

import numpy as np


def _target_category_id(category: Any) -> Optional[int]:
    name = str(category or "").strip().lower().replace("-", "_")
    if "chair" in name:
        return 0
    if "sofa" in name or "couch" in name:
        return 1
    if "plant" in name:
        return 2
    if "bed" in name:
        return 3
    if "toilet" in name:
        return 4
    if "tv" in name or "monitor" in name:
        return 5
    return None


def _trained_category_mask(manager: Any, gaussians: Any) -> Optional[np.ndarray]:
    if not bool(
        int(getattr(manager.args, "multi_target2_map_use_gaussian_labels", 1))
    ):
        return None
    category_id = _target_category_id(getattr(manager, "target_category", None))
    evidence = getattr(gaussians, "_multi_semantic_evidence", None)
    observations = getattr(gaussians, "_multi_semantic_observations", None)
    if category_id is None or evidence is None or observations is None:
        return None
    try:
        if evidence.ndim != 2 or category_id >= int(evidence.shape[1]):
            return None
        if int(evidence.shape[0]) != int(gaussians.get_xyz.shape[0]):
            return None
        evidence_float = evidence.detach().float()
        totals = evidence_float.sum(dim=1)
        best_value, best_class = evidence_float.max(dim=1)
        confidence = best_value / totals.clamp_min(1e-6)
        minimum_confidence = float(
            getattr(manager.args, "multi_target2_map_gaussian_label_confidence", 0.55)
        )
        minimum_observations = max(
            1,
            int(
                getattr(
                    manager.args,
                    "multi_target2_map_gaussian_label_min_observations",
                    1,
                )
            ),
        )
        mask = (
            (best_class == int(category_id))
            & (confidence >= minimum_confidence)
            & (observations.to(device=evidence.device) >= minimum_observations)
        )
        return mask.detach().cpu().numpy().astype(bool)
    except Exception:
        return None


def _trace(manager: Any, event: str, **kwargs: Any) -> None:
    from five_goal_benchmark_20261026.runtime.instrumentation import record_retrieval
    record_retrieval(manager, event, kwargs)
    method = getattr(manager, "_trace", None)
    if method is not None:
        method(event, **kwargs)


def _local_gaussian_clusters(
    manager: Any,
    gaussians: Any,
    map_object: Dict[str, Any],
    payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    center = np.asarray(map_object["position_3d"], dtype=np.float32).reshape(-1)[:3]
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    if xyz.ndim != 2 or xyz.shape[1] < 3 or center.size < 3:
        return []

    radius = max(
        0.25, float(getattr(manager.args, "multi_target2_map_gaussian_radius_m", 1.0))
    )
    horizontal = np.linalg.norm(xyz[:, [0, 2]] - center[[0, 2]], axis=1)
    agent = payload.get("agent_position")
    if agent is not None:
        agent = np.asarray(agent, dtype=np.float32).reshape(-1)
    base_y = float(agent[1]) if agent is not None and agent.size >= 3 else float(center[1])
    category = str(getattr(manager, "target_category", "") or "").lower()
    top_height = 2.4 if ("plant" in category or "tv" in category) else 1.8
    keep = (
        (horizontal <= radius)
        & (xyz[:, 1] >= base_y - 0.30)
        & (xyz[:, 1] <= base_y + top_height)
    )
    candidate_source = "rgb_gaussians_near_2d_semantic_component"
    trained_mask = _trained_category_mask(manager, gaussians)
    minimum = max(
        1, int(getattr(manager.args, "multi_target2_map_min_gaussians", 100))
    )
    if trained_mask is not None and trained_mask.shape == keep.shape:
        trained_local = keep & trained_mask
        if int(trained_local.sum()) >= minimum:
            keep = trained_local
            candidate_source = "trained_semantic_gaussians_near_2d_component"
    indices = np.where(keep)[0].astype(np.int64)
    if indices.size < minimum:
        return []
    maximum = max(
        minimum,
        int(getattr(manager.args, "multi_target2_map_max_gaussians", 50000)),
    )
    if indices.size > maximum:
        nearest = np.argpartition(horizontal[indices], maximum - 1)[:maximum]
        indices = indices[nearest]

    clusters = list(manager._cluster_target_confirm_indices(gaussians, indices) or [])
    accepted = []
    for cluster in clusters:
        cluster_center = manager._cluster_center(cluster)
        if cluster_center.size < 3:
            continue
        map_distance = float(
            np.linalg.norm(cluster_center[[0, 2]] - center[[0, 2]])
        )
        if map_distance > radius:
            continue
        record = dict(cluster)
        record["map_anchor_world"] = center.copy()
        record["map_anchor_distance_m"] = map_distance
        record["map_component_id"] = map_object.get("obj_id")
        record["map_component_area"] = int(map_object.get("area", 0) or 0)
        record["candidate_source"] = candidate_source
        cluster_indices = manager._cluster_indices(record)
        if trained_mask is not None and cluster_indices.size:
            valid_indices = cluster_indices[
                (cluster_indices >= 0) & (cluster_indices < trained_mask.size)
            ]
            record["trained_semantic_ratio"] = (
                float(trained_mask[valid_indices].mean())
                if valid_indices.size
                else 0.0
            )
        else:
            record["trained_semantic_ratio"] = 0.0
        observations = getattr(gaussians, "_multi_semantic_observations", None)
        evidence = getattr(gaussians, "_multi_semantic_evidence", None)
        category_id = _target_category_id(getattr(manager, "target_category", None))
        try:
            selected_observations = observations[cluster_indices].detach().float()
            record["semantic_observation_sum"] = float(
                selected_observations.sum().item()
            )
            record["semantic_observation_mean"] = float(
                selected_observations.mean().item()
            )
            selected_evidence = evidence[cluster_indices].detach().float()
            totals = selected_evidence.sum(dim=1).clamp_min(1e-6)
            record["target_semantic_confidence_mean"] = float(
                (selected_evidence[:, int(category_id)] / totals).mean().item()
            )
        except Exception:
            record["semantic_observation_sum"] = 0.0
            record["semantic_observation_mean"] = 0.0
            record["target_semantic_confidence_mean"] = 0.0
        instance_ids = getattr(gaussians, "_multi_instance_ids", None)
        try:
            selected_ids = instance_ids[cluster_indices].detach().cpu().numpy()
            selected_ids = selected_ids[selected_ids > 0]
            if selected_ids.size:
                ids, counts = np.unique(selected_ids, return_counts=True)
                dominant = int(np.argmax(counts))
                record["dominant_instance_id"] = int(ids[dominant])
                record["dominant_instance_points"] = int(counts[dominant])
                record["dominant_instance_ratio"] = float(
                    counts[dominant] / max(cluster_indices.size, 1)
                )
            else:
                record["dominant_instance_id"] = 0
                record["dominant_instance_points"] = 0
                record["dominant_instance_ratio"] = 0.0
        except Exception:
            record["dominant_instance_id"] = 0
            record["dominant_instance_points"] = 0
            record["dominant_instance_ratio"] = 0.0
        accepted.append(record)
    accepted.sort(
        key=lambda item: (
            -int(
                item.get("candidate_source")
                == "trained_semantic_gaussians_near_2d_component"
            ),
            -float(item.get("semantic_observation_mean", 0.0) or 0.0),
            -min(int(item.get("dominant_instance_points", 0) or 0), 12000),
            -float(item.get("dominant_instance_ratio", 0.0) or 0.0),
            float(item.get("map_anchor_distance_m", float("inf"))),
            -min(int(item.get("size", 0) or 0), 12000),
        )
    )
    per_component = max(
        1,
        int(getattr(manager.args, "multi_target2_map_clusters_per_component", 2)),
    )
    return accepted[:per_component]


def _candidate_prior_rank(candidate: Dict[str, Any]) -> tuple:
    """Rank candidates using only detector-trained map evidence."""
    return (
        int(
            candidate.get("candidate_source")
            == "trained_semantic_gaussians_near_2d_component"
        ),
        float(candidate.get("semantic_observation_mean", 0.0) or 0.0),
        float(candidate.get("target_semantic_confidence_mean", 0.0) or 0.0),
        min(int(candidate.get("dominant_instance_points", 0) or 0), 12000),
        float(candidate.get("dominant_instance_ratio", 0.0) or 0.0),
        -float(candidate.get("map_anchor_distance_m", float("inf"))),
        min(int(candidate.get("size", 0) or 0), 12000),
    )


def _match_with_temporary_nvsm_budget(
    manager: Any,
    center: np.ndarray,
    indices: np.ndarray,
    *,
    views_per_scale: int,
    save_visualizations: bool,
    stage: str,
    step: int,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one match with a temporary view budget and restore all settings."""
    ensure = getattr(manager, "_ensure_target_confirm_nvsm_fem", None)
    if ensure is None:
        return manager._match_candidate_with_nvsm_views(
            center, indices, stage=stage, step=step, metadata=metadata
        )
    nvsm, fem, fail = ensure()
    if fail is not None or nvsm is None or fem is None:
        return {
            "available": False,
            "score": 0.0,
            "match_count": 0,
            "source": "none",
            "message": str(fail or "NVSM/FEM unavailable"),
        }

    old_views = getattr(nvsm, "num_views", None)
    old_nvsm_save = getattr(nvsm, "save_visualization_results", None)
    old_fem_save = getattr(fem, "save_visualization_results", None)
    has_manager_save = hasattr(manager, "save_stage_visualizations")
    old_manager_save = getattr(manager, "save_stage_visualizations", None)
    try:
        nvsm.num_views = max(1, int(views_per_scale))
        if has_manager_save:
            manager.save_stage_visualizations = bool(save_visualizations)
        if old_nvsm_save is not None:
            nvsm.save_visualization_results = bool(save_visualizations)
        if old_fem_save is not None:
            fem.save_visualization_results = bool(save_visualizations)
        return manager._match_candidate_with_nvsm_views(
            center, indices, stage=stage, step=step, metadata=metadata
        )
    finally:
        if old_views is not None:
            nvsm.num_views = old_views
        if old_nvsm_save is not None:
            nvsm.save_visualization_results = old_nvsm_save
        if old_fem_save is not None:
            fem.save_visualization_results = old_fem_save
        if has_manager_save:
            manager.save_stage_visualizations = old_manager_save


def _run_map_prevalidation(
    manager: Any, payload: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    step = int(payload.get("step", 0))
    objects = list(manager._extract_targeted_objects_from_map(payload).values())
    if not objects:
        _trace(
            manager,
            "target2_map_prevalidation_waiting",
            step=step,
            reason="no_existing_2d_semantic_candidate",
        )
        return None
    objects.sort(key=lambda item: int(item.get("area", 0) or 0), reverse=True)
    max_components = max(
        1, int(getattr(manager.args, "multi_target2_map_max_candidates", 3))
    )
    objects = objects[:max_components]

    gaussians, fail = manager._get_gaussians_for_confirmation()
    if fail is not None:
        _trace(
            manager,
            "target2_map_prevalidation_skipped",
            step=step,
            reason=str(fail),
        )
        return None
    manager._ensure_goal_image_masks(payload, step)

    candidates: List[Dict[str, Any]] = []
    for map_object in objects:
        candidates.extend(
            _local_gaussian_clusters(manager, gaussians, map_object, payload)
        )
    if not candidates:
        _trace(
            manager,
            "target2_map_prevalidation_skipped",
            step=step,
            reason="no_local_gaussian_cluster_near_2d_semantic_candidates",
            semantic_candidate_count=len(objects),
        )
        return None

    candidates.sort(key=_candidate_prior_rank, reverse=True)
    shortlist_limit = max(
        1,
        int(
            getattr(manager.args, "multi_target2_map_shortlist_candidates", 2)
        ),
    )
    candidates = candidates[:shortlist_limit]

    # A real candidate set is stable at the target boundary because the
    # robot has not moved yet.  Evaluate it once.  The grace window is only
    # for waiting until retained 2D semantics become available; repeatedly
    # rendering the same candidates wastes several NVSM/FEM passes and
    # cannot add evidence.
    manager._multi_target2_map_prevalidation_evaluated = True

    base_threshold = float(getattr(manager, "confirm_match_threshold", 20.0))
    # Map-only validation has no fresh robot observation to disambiguate
    # repeated furniture and textured background.  It must therefore be more
    # conservative than ordinary close-by confirmation: a weak map match is a
    # reason to continue exploration, never a reason to navigate blindly.
    threshold = max(
        base_threshold,
        float(getattr(manager.args, "multi_target2_map_match_threshold", 40.0)),
    )
    coarse_views = max(
        1,
        int(
            getattr(
                manager.args, "multi_target2_map_coarse_views_per_scale", 4
            )
        ),
    )
    coarse_min_score = max(
        0.0,
        float(getattr(manager.args, "multi_target2_map_coarse_min_score", 8.0)),
    )
    coarse_save = bool(
        int(
            getattr(
                manager.args,
                "multi_target2_map_coarse_save_visualizations",
                0,
            )
        )
    )
    coarse_ranked = []
    for rank, candidate in enumerate(candidates):
        center = manager._cluster_center(candidate)
        indices = manager._cluster_indices(candidate)
        started = time.perf_counter()
        match = _match_with_temporary_nvsm_budget(
            manager,
            center,
            indices,
            views_per_scale=coarse_views,
            save_visualizations=coarse_save,
            stage="target2_map_prevalidation_coarse",
            step=step,
            metadata={
                "candidate_rank": rank,
                "validation_pass": "coarse",
                "map_component_id": candidate.get("map_component_id"),
                "map_anchor_distance_m": candidate.get("map_anchor_distance_m"),
                "contract": "existing_2d_map_plus_rgb_gaussians_no_robot_motion",
            },
        )
        elapsed = float(time.perf_counter() - started)
        score = float(match.get("score", 0.0) or 0.0)
        fem = getattr(manager, "target_confirm_fem", None)
        semantic_ready = bool(
            fem is not None
            and getattr(fem, "target_features_semantic", None) is not None
        )
        feature_mode = "semantic" if semantic_ready else "uniform"
        _trace(
            manager,
            "target2_map_prevalidation_coarse_candidate",
            step=step,
            rank=rank,
            center=center,
            cluster_size=int(indices.size),
            score=score,
            match_count=int(match.get("match_count", 0) or 0),
            coarse_min_score=coarse_min_score,
            base_confirmation_threshold=base_threshold,
            available=bool(match.get("available", False)),
            feature_mode=feature_mode,
            semantic_features_ready=semantic_ready,
            candidate_source=candidate.get("candidate_source"),
            map_anchor_world=candidate.get("map_anchor_world"),
            map_anchor_distance_m=candidate.get("map_anchor_distance_m"),
            dominant_instance_id=candidate.get("dominant_instance_id"),
            dominant_instance_points=candidate.get("dominant_instance_points", 0),
            dominant_instance_ratio=candidate.get("dominant_instance_ratio", 0.0),
            semantic_observation_mean=candidate.get(
                "semantic_observation_mean", 0.0
            ),
            rendered_view_count=int(match.get("rendered_view_count", 0) or 0),
            elapsed_seconds=elapsed,
        )
        if (
            not match.get("available")
            or not semantic_ready
            or score < coarse_min_score
        ):
            continue
        coarse_ranked.append(
            (
                (
                    score,
                    int(match.get("match_count", 0) or 0),
                )
                + _candidate_prior_rank(candidate),
                candidate,
                match,
            )
        )

    if not coarse_ranked:
        _trace(
            manager,
            "target2_map_prevalidation_failed",
            step=step,
            reason="all_shortlisted_candidates_failed_coarse_ranking",
            candidate_count=len(candidates),
            threshold=threshold,
            coarse_min_score=coarse_min_score,
            base_confirmation_threshold=base_threshold,
            contract="weak_or_uniform_map_evidence_falls_back_to_exploration",
        )
        return None

    coarse_ranked.sort(key=lambda item: item[0], reverse=True)
    _coarse_rank, candidate, coarse_match = coarse_ranked[0]
    indices = manager._cluster_indices(candidate)
    center = manager._cluster_center(candidate)
    full_views = max(
        1,
        int(
            getattr(
                manager.args,
                "multi_target2_map_full_views_per_scale",
                getattr(
                    getattr(manager, "target_confirm_nvsm", None), "num_views", 16
                ),
            )
        ),
    )
    started = time.perf_counter()
    match = _match_with_temporary_nvsm_budget(
        manager,
        center,
        indices,
        views_per_scale=full_views,
        save_visualizations=bool(
            getattr(manager, "save_stage_visualizations", True)
        ),
        stage="target2_map_prevalidation",
        step=step,
        metadata={
            "validation_pass": "full",
            "coarse_score": float(coarse_match.get("score", 0.0) or 0.0),
            "map_component_id": candidate.get("map_component_id"),
            "map_anchor_distance_m": candidate.get("map_anchor_distance_m"),
            "contract": "coarse_top1_full_semantic_fem_no_robot_motion",
        },
    )
    full_elapsed = float(time.perf_counter() - started)
    score = float(match.get("score", 0.0) or 0.0)
    fem = getattr(manager, "target_confirm_fem", None)
    semantic_ready = bool(
        fem is not None and getattr(fem, "target_features_semantic", None) is not None
    )
    _trace(
        manager,
        "target2_map_prevalidation_full_candidate",
        step=step,
        center=center,
        cluster_size=int(indices.size),
        coarse_score=float(coarse_match.get("score", 0.0) or 0.0),
        score=score,
        match_count=int(match.get("match_count", 0) or 0),
        threshold=threshold,
        available=bool(match.get("available", False)),
        semantic_features_ready=semantic_ready,
        rendered_view_count=int(match.get("rendered_view_count", 0) or 0),
        elapsed_seconds=full_elapsed,
    )
    if not match.get("available") or not semantic_ready or score < threshold:
        _trace(
            manager,
            "target2_map_prevalidation_failed",
            step=step,
            reason="coarse_winner_failed_full_match_threshold",
            coarse_score=float(coarse_match.get("score", 0.0) or 0.0),
            full_score=score,
            threshold=threshold,
            contract="full_match_rejection_falls_back_to_exploration",
        )
        return None

    locked = int(
        manager._lock_confirmed_cluster(
            indices, step=step, stage="target2_map_prevalidation"
        )
    )
    if locked <= 0:
        _trace(
            manager,
            "target2_map_prevalidation_failed",
            step=step,
            reason="confirmed_map_candidate_lock_failed",
        )
        return None

    result = manager._new_confirm_result()
    result.update(
        {
            "confirmed": True,
            "target_cluster_center": center.copy(),
            "target_map_anchor_world": np.asarray(
                candidate.get("map_anchor_world", center), dtype=np.float32
            ).reshape(-1)[:3],
            "best_cluster_global_indices": indices.copy(),
            "best_score": float(match.get("score", 0.0) or 0.0),
            "best_match_count": int(match.get("match_count", 0) or 0),
            "best_view_source": match.get("source", "unknown"),
            "best_view_index": int(match.get("best_index", -1)),
            "best_pose": match.get("best_pose"),
            "fail_reason": "",
            "system_not_ready": False,
            "should_blacklist": False,
            "method": "target2_existing_map_coarse_top1_nvsm_fem_prevalidation_v1",
        }
    )
    manager.detected_target_position = np.asarray(
        result["target_map_anchor_world"], dtype=np.float32
    ).copy()
    manager._record_strategy_evidence(
        step=step,
        center=center,
        best_score=result["best_score"],
        source="target2_map_prevalidation",
        confirmed=True,
        best_match_count=result["best_match_count"],
    )
    manager._enter_buffer_from_confirmation(result, step, payload)
    _trace(
        manager,
        "target2_map_prevalidation_confirmed",
        step=step,
        center=center,
        map_anchor_world=result["target_map_anchor_world"],
        score=result["best_score"],
        match_count=result["best_match_count"],
        feature_mode="semantic",
        candidate_source=candidate.get("candidate_source"),
        threshold=threshold,
        locked_gaussians=locked,
        next_mode=getattr(getattr(manager, "mode", None), "value", None),
        contract="discovery_confirmed_without_motion_navigation_still_required",
    )
    return result


def install_target2_map_prevalidation(manager: Any) -> Any:
    if getattr(manager, "_multi_target2_map_prevalidation_installed", False):
        return manager
    manager._multi_target2_map_prevalidation_installed = True
    manager._multi_target2_map_prevalidation_pending = False
    manager._multi_target2_map_prevalidation_attempted = False
    manager._multi_target2_map_prevalidation_evaluated = False
    original_reset = manager.reset_episode
    original_update = manager.update_frame

    def reset_episode(self, *args, **kwargs):
        result = original_reset(*args, **kwargs)
        info = args[0] if args and isinstance(args[0], dict) else kwargs.get("info", {})
        target_index = int((info or {}).get("multi_target_index", 0) or 0)
        shared = str(getattr(self.args, "multi_variant", "")) == "Continuous-Shared"
        enabled = bool(int(getattr(self.args, "multi_target2_map_prevalidation", 1)))
        self._multi_target2_map_prevalidation_pending = bool(
            enabled and shared and target_index >= 1 and getattr(self, "slam_mapper", None) is not None
        )
        self._multi_target2_map_prevalidation_attempted = False
        self._multi_target2_map_prevalidation_evaluated = False
        if self._multi_target2_map_prevalidation_pending:
            _trace(
                self,
                "target2_map_prevalidation_armed",
                step=0,
                target_category=getattr(self, "target_category", None),
                variant=str(getattr(self.args, "multi_variant", "")),
            )
        return result

    def update_frame(self, payload):
        step = int(payload.get("step", 0))
        grace = max(
            0, int(getattr(self.args, "multi_target2_map_prevalidation_grace_steps", 5))
        )
        if self._multi_target2_map_prevalidation_pending and step <= grace:
            previous_payload = getattr(self, "_last_payload", None)
            self._last_payload = payload
            try:
                result = _run_map_prevalidation(self, payload)
            except Exception as exc:
                result = None
                _trace(
                    self,
                    "target2_map_prevalidation_failed",
                    step=step,
                    reason="exception:{}".format(exc),
                )
            finally:
                if previous_payload is not None and result is None:
                    self._last_payload = previous_payload
            if result is not None or self._multi_target2_map_prevalidation_evaluated:
                self._multi_target2_map_prevalidation_pending = False
                self._multi_target2_map_prevalidation_attempted = True
            elif step >= grace:
                self._multi_target2_map_prevalidation_pending = False
                self._multi_target2_map_prevalidation_attempted = True
        return original_update(payload)

    manager.reset_episode = types.MethodType(reset_episode, manager)
    manager.update_frame = types.MethodType(update_frame, manager)
    return manager
