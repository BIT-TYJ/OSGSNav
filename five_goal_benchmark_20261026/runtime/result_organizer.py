"""Compact, human-readable result archive for the additive Multi-IIN runner.

The MinAV and Gaussian components intentionally keep their original verbose
debug output.  This module copies only the terminal artefacts needed to inspect
or audit a Multi-IIN result into a second, stable directory layout.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw


SEMANTIC_COLORS = np.asarray(
    [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (188, 189, 34),
        (23, 190, 207),
        (127, 127, 127),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
    ],
    dtype=np.uint8,
)


def _safe_name(value: Any, default: str = "unknown") -> str:
    text = str(value if value is not None else "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text).strip("_.-")
    return text or default


def _scene_key(scene_name: Any) -> str:
    name = Path(str(scene_name or "unknown_scene")).name
    for suffix in (".basis.glb", ".glb", ".basis"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return _safe_name(name, "unknown_scene")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _copy(source: Optional[Path], destination: Path) -> Optional[str]:
    if source is None or not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(destination))
    return str(destination)


def _newest(paths: Iterable[Path]) -> Optional[Path]:
    files = [path for path in paths if path.is_file()]
    return max(files, key=lambda path: (path.stat().st_mtime_ns, str(path))) if files else None


def _rgb_uint8(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def ensure_terminal_match_source(manager: Any, info: Dict[str, Any], step: int) -> Optional[str]:
    """Ensure compact archiving has one honest terminal match record.

    This post-terminal diagnostic never feeds back into navigation.  A real
    FEM match is attempted first; if unavailable, a labelled zero-match panel
    is emitted so every target still has an auditable ``05_final_match``.
    """
    if manager is None or not bool(getattr(manager, "save_stage_visualizations", False)):
        return None
    visualization_dir = getattr(manager, "visualization_dir", None)
    if visualization_dir is None and hasattr(manager, "ensure_visualization_episode"):
        visualization_dir = manager.ensure_visualization_episode(info or {})
    if not visualization_dir:
        return None
    source_dir = Path(visualization_dir)
    nvsm_dir = source_dir / "NVSM"
    nvsm_dir.mkdir(parents=True, exist_ok=True)
    existing = _newest(nvsm_dir.glob("best_match_*.json"))
    if existing is not None:
        return str(existing)

    current_rgb = _rgb_uint8((info or {}).get("raw_rgb"))
    if current_rgb is not None and hasattr(manager, "_match_with_evidence_views"):
        try:
            manager._match_with_evidence_views(
                current_rgb,
                stage="NVSM",
                step=int(step),
                metadata={
                    "diagnostic_only": True,
                    "contract": "post_terminal_match_does_not_affect_navigation",
                },
            )
        except Exception:
            pass
    existing = _newest(nvsm_dir.glob("best_match_*.json"))
    if existing is not None:
        return str(existing)
    if current_rgb is None:
        return None

    target_rgb = _rgb_uint8(getattr(manager, "target_image", None))
    render_path = nvsm_dir / "best_match_step_{:06d}_score_0000.00_compact_terminal_no_match.png".format(int(step))
    Image.fromarray(current_rgb).save(str(render_path))
    query = Image.fromarray(current_rgb)
    target = Image.fromarray(target_rgb) if target_rgb is not None else Image.new("RGB", query.size, (32, 32, 32))
    height = min(512, max(target.height, query.height))
    resampling = getattr(Image, "Resampling", Image)
    target = target.resize((max(1, round(target.width * height / target.height)), height), resampling.BILINEAR)
    query = query.resize((max(1, round(query.width * height / query.height)), height), resampling.BILINEAR)
    panel = Image.new("RGB", (target.width + query.width + 16, height + 28), (245, 245, 245))
    panel.paste(target, (0, 28))
    panel.paste(query, (target.width + 16, 28))
    ImageDraw.Draw(panel).text((8, 7), "target | terminal view - no feature match available", fill=(0, 0, 0))
    match_path = render_path.with_name(render_path.stem + "_matches.png")
    panel.save(str(match_path))
    metadata_path = render_path.with_suffix(".json")
    _write_json(metadata_path, {
        "stage": "NVSM",
        "step": int(step),
        "score": 0.0,
        "match_count": 0,
        "source": "compact_terminal_no_match",
        "match_success": False,
        "diagnostic_only": True,
        "image_path": str(render_path),
        "match_visualization_path": str(match_path),
    })
    return str(metadata_path)


def _crop_bounds(map_layers: np.ndarray, padding: int = 12) -> Sequence[int]:
    occupied = np.any(np.asarray(map_layers) > 0, axis=0)
    rows, cols = np.where(occupied)
    height, width = occupied.shape
    if rows.size == 0:
        return 0, height, 0, width
    return (
        max(0, int(rows.min()) - padding),
        min(height, int(rows.max()) + padding + 1),
        max(0, int(cols.min()) - padding),
        min(width, int(cols.max()) + padding + 1),
    )


def save_composite_grid_map(map_layers: np.ndarray, output_path: Path) -> Dict[str, Any]:
    """Save obstacle/exploration/visited/semantic as one labelled 2x2 PNG."""

    layers = np.asarray(map_layers, dtype=np.float32)
    if layers.ndim != 3 or layers.shape[0] < 4:
        raise ValueError("map_layers must have shape [channels>=4, height, width]")
    r0, r1, c0, c1 = _crop_bounds(layers)
    layers = layers[:, r0:r1, c0:c1]
    obstacle = layers[0] > 0.5
    explored = layers[1] > 0.5
    current = layers[2] > 0.0
    visited = layers[3] > 0.0

    base = np.full((*obstacle.shape, 3), 78, dtype=np.uint8)
    occupancy = base.copy()
    occupancy[explored] = (238, 238, 238)
    occupancy[obstacle] = (20, 20, 20)

    exploration = np.full_like(base, (48, 48, 48))
    exploration[explored] = (76, 175, 80)
    exploration[obstacle] = (20, 20, 20)

    visited_image = np.full_like(base, (48, 48, 48))
    visited_image[explored] = (205, 205, 205)
    visited_image[obstacle] = (20, 20, 20)
    visited_image[visited] = (0, 188, 212)

    semantic = np.full_like(base, (48, 48, 48))
    semantic[explored] = (205, 205, 205)
    semantic[obstacle] = (20, 20, 20)
    semantic_counts = []
    if layers.shape[0] > 4:
        scores = layers[4:]
        labels = np.argmax(scores, axis=0)
        support = np.max(scores, axis=0) > 0.0
        for category_id in range(scores.shape[0]):
            mask = support & (labels == category_id)
            count = int(np.count_nonzero(mask))
            if count:
                semantic[mask] = SEMANTIC_COLORS[category_id % len(SEMANTIC_COLORS)]
                semantic_counts.append({"category_id": category_id, "cells": count})

    for panel in (occupancy, exploration, visited_image, semantic):
        panel[current] = (244, 67, 54)

    panels = [occupancy, exploration, visited_image, semantic]
    labels = ["occupancy", "exploration", "visited", "semantic"]
    max_side = 520
    height, width = obstacle.shape
    scale = min(max_side / max(height, 1), max_side / max(width, 1), 4.0)
    draw_width = max(1, int(round(width * scale)))
    draw_height = max(1, int(round(height * scale)))
    margin, label_height = 12, 28
    canvas = Image.new(
        "RGB",
        (2 * draw_width + 3 * margin, 2 * (draw_height + label_height) + 3 * margin),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        row, col = divmod(index, 2)
        x = margin + col * (draw_width + margin)
        y = margin + row * (draw_height + label_height + margin)
        draw.text((x, y + 5), label, fill=(0, 0, 0))
        resampling = getattr(Image, "Resampling", Image)
        image = Image.fromarray(panel).resize(
            (draw_width, draw_height), resample=resampling.NEAREST
        )
        canvas.paste(image, (x, y + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path))
    return {
        "source_shape": [int(value) for value in map_layers.shape],
        "crop_row_col": [int(r0), int(r1), int(c0), int(c1)],
        "obstacle_cells": int(np.count_nonzero(obstacle)),
        "explored_cells": int(np.count_nonzero(explored)),
        "visited_cells": int(np.count_nonzero(visited)),
        "semantic_cells": int(sum(item["cells"] for item in semantic_counts)),
        "semantic_categories": semantic_counts,
    }


class CompactResultOrganizer:
    """Write a compact archive without deleting or rewriting verbose outputs."""

    def __init__(self, args: Any, repo_root: Path):
        self.enabled = bool(int(getattr(args, "multi_save_compact_results", 1)))
        self.variant = str(getattr(args, "multi_variant", "unknown"))
        self.variant_slug = _safe_name(self.variant.replace("-", "_"))
        root = Path(getattr(args, "multi_compact_results_dir", "multi_iin_v0/results/compact"))
        if not root.is_absolute():
            root = Path(repo_root) / root
        start = int(getattr(args, "multi_episode_start", 0))
        count = int(getattr(args, "multi_num_episodes", 1))
        end = start + max(count, 1) - 1
        budget = int(getattr(args, "multi_goal_step_budget", 0))
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = "budget{}__episodes{:05d}-{:05d}__{}".format(
            budget, start, end, timestamp
        )
        base_run_dir = root / self.variant_slug / run_name
        self.run_dir = base_run_dir
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = Path("{}__{:02d}".format(base_run_dir, suffix))
            suffix += 1
        self.targets_by_episode: Dict[str, list] = {}
        self.episode_dir_by_id: Dict[str, Path] = {}
        self.goal_metadata_by_key: Dict[tuple, Dict[str, Any]] = {}
        manifest_path = Path(str(getattr(args, "multi_manifest", "")))
        if not manifest_path.is_absolute():
            manifest_path = Path(repo_root) / manifest_path
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            for episode in manifest.get("episodes", []):
                episode_id = str(episode.get("episode_id", ""))
                for goal_index, goal in enumerate(episode.get("goals", [])):
                    self.goal_metadata_by_key[(episode_id, int(goal_index))] = dict(goal)
        except Exception:
            pass
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            _write_json(
                self.run_dir / "run_summary.json",
                {
                    "schema_version": "multi-iin-compact-v1",
                    "status": "running",
                    "variant": self.variant,
                    "goal_step_budget": budget,
                    "episode_start": start,
                    "num_episodes": count,
                    "manifest": str(getattr(args, "multi_manifest", "")),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "contents": [
                        "target image and task-category audit",
                        "terminal 2D map (occupancy/exploration/visited/semantic)",
                        "terminal Gaussian map",
                        "final filtered target-instance Gaussian PLY",
                        "target-instance filter keep/remove projection PNG",
                        "final selected render and target/render feature matches",
                    ],
                },
            )

    def goal_metadata(self, info: Dict[str, Any], target_index: int) -> Dict[str, Any]:
        """Return manifest audit fields without placing them in policy-visible info."""
        return dict(
            self.goal_metadata_by_key.get(
                (str(info.get("multi_episode_id", "")), int(target_index)), {}
            )
        )

    @staticmethod
    def _episode_dir_name(info: Dict[str, Any]) -> str:
        episode_id = _safe_name(info.get("multi_episode_id"), "unknown_episode")
        scene = _scene_key(info.get("scene_name"))
        episode_no = int(info.get("episode_no", 0) or 0)
        return "episode_{:05d}__{}__{}".format(episode_no, scene, episode_id)

    @staticmethod
    def _target_dir_name(record: Dict[str, Any]) -> str:
        target_no = int(record.get("target_index", 0)) + 1
        category = _safe_name(record.get("goal_name"), "unknown_category")
        object_id = _safe_name(record.get("goal_object_id"), "unknown")
        outcome = "success" if float(record.get("success", 0.0)) > 0.0 else "failure"
        return "target_{:02d}__{}__obj{}__{}".format(
            target_no, category, object_id, outcome
        )

    @staticmethod
    def _select_final_match(source_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
        if source_dir is None:
            return None
        candidates = []
        for meta_path in (source_dir / "NVSM").glob("best_match_*.json"):
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                step = metadata.get("step")
                step = -1 if step is None else int(step)
                score = float(metadata.get("score", 0.0) or 0.0)
                candidates.append((step, score, meta_path.stat().st_mtime_ns, meta_path, metadata))
            except Exception:
                continue
        if not candidates:
            return None
        _, _, _, meta_path, metadata = max(candidates, key=lambda item: item[:3])
        image_path = Path(str(metadata.get("image_path", "")))
        if not image_path.is_file():
            image_path = meta_path.with_suffix(".png")
        match_path = Path(str(metadata.get("match_visualization_path", "")))
        if not match_path.is_file():
            conventional = meta_path.with_name(meta_path.stem + "_matches.png")
            match_path = conventional if conventional.is_file() else Path()
        return {
            "metadata_path": meta_path,
            "metadata": metadata,
            "image_path": image_path if image_path.is_file() else None,
            "match_path": match_path if match_path.is_file() else None,
        }

    def archive_target(
        self,
        info: Dict[str, Any],
        record: Dict[str, Any],
        map_layers: np.ndarray,
        mapper_output_dir: Optional[str],
        visualization_dir: Optional[str],
        terminal_localization: Optional[Dict[str, Any]] = None,
        gaussian_preview_path: Optional[str] = None,
        gaussian_quality: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if not self.enabled:
            return None
        episode_name = self._episode_dir_name(info)
        episode_dir = self.run_dir / episode_name
        self.episode_dir_by_id[str(info.get("multi_episode_id", ""))] = episode_dir
        target_dir = episode_dir / self._target_dir_name(record)
        target_dir.mkdir(parents=True, exist_ok=False)
        source_dir = Path(visualization_dir) if visualization_dir else None
        mapper_dir = Path(mapper_output_dir) if mapper_output_dir else None
        artefacts: Dict[str, Optional[str]] = {
            "target_image": None,
            "target_detector_analysis": None,
            "final_2d_map": None,
            "final_gaussian_map": None,
            "final_gaussian_rgb_comparison": None,
            "target_instance": None,
            "target_instance_filter_stats": None,
            "target_instance_filter_projection": None,
            "selected_render": None,
            "target_render_matches": None,
            "match_metrics": None,
        }

        if source_dir is not None:
            artefacts["target_image"] = _copy(
                source_dir / "target_image.png", target_dir / "01_goal" / "target_image.png"
            )
            analysis = _newest(
                source_dir.glob("goal_image*_analysis_result.png")
            )
            artefacts["target_detector_analysis"] = _copy(
                analysis, target_dir / "01_goal" / "target_detector_analysis.png"
            )
        if artefacts["target_image"] is None:
            target_image = _rgb_uint8(info.get("instance_imagegoal"))
            if target_image is not None:
                target_path = target_dir / "01_goal" / "target_image.png"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(target_image).save(str(target_path))
                artefacts["target_image"] = str(target_path)

        map_path = target_dir / "02_map" / "final_2d_map.png"
        map_stats = save_composite_grid_map(map_layers, map_path)
        artefacts["final_2d_map"] = str(map_path)

        gaussian_source = mapper_dir / "final" / "point_cloud.ply" if mapper_dir else None
        artefacts["final_gaussian_map"] = _copy(
            gaussian_source, target_dir / "03_gaussian_map" / "final_gaussian_map.ply"
        )
        artefacts["final_gaussian_rgb_comparison"] = _copy(
            Path(gaussian_preview_path) if gaussian_preview_path else None,
            target_dir / "03_gaussian_map" / "final_gaussian_rgb_comparison.png",
        )
        if mapper_dir is not None:
            label_dir = mapper_dir / "final"
            semantic_ply = _copy(
                label_dir / "semantic_instance_gaussians.ply",
                target_dir / "03_gaussian_map" / "semantic_instance_gaussians.ply",
            )
            semantic_labels = _copy(
                label_dir / "semantic_instance_labels.npz",
                target_dir / "03_gaussian_map" / "semantic_instance_labels.npz",
            )
            semantic_summary = _copy(
                label_dir / "semantic_instance_summary.json",
                target_dir / "03_gaussian_map" / "semantic_instance_summary.json",
            )
            if semantic_ply is not None:
                artefacts["gaussian_semantic_instance_ply"] = semantic_ply
            if semantic_labels is not None:
                artefacts["gaussian_semantic_instance_labels"] = semantic_labels
            if semantic_summary is not None:
                artefacts["gaussian_semantic_instance_summary"] = semantic_summary

        instance_source = None
        stats_source = None
        projection_source = None
        if source_dir is not None:
            instance_source = _newest((source_dir / "NVSM").glob("instance_semantic_*_filtered.ply"))
            if instance_source is not None:
                candidate = instance_source.with_name(
                    instance_source.name.replace("_filtered.ply", "_filter_stats.json")
                )
                stats_source = candidate if candidate.is_file() else None
                candidate = instance_source.with_name(
                    instance_source.name.replace(
                        "_filtered.ply", "_filter_projection.png"
                    )
                )
                projection_source = candidate if candidate.is_file() else None
            # Some MinAV/NVSM code paths emit the terminal semantic instance
            # cloud but do not invoke the additive filter visualizer (for
            # example, when localization exits directly after the semantic
            # re-marking step).  Keep the compact archive complete in that
            # case by falling back to the newest raw instance artifact.  The
            # accompanying stats file explicitly records that this is a raw
            # fallback so it is never mistaken for a filtered cloud.
            if instance_source is None:
                instance_source = _newest((source_dir / "NVSM").glob("instance_semantic_*.ply"))
                if instance_source is not None:
                    candidate = instance_source.with_name(
                        instance_source.stem + "_filter_stats.json"
                    )
                    stats_source = candidate if candidate.is_file() else None
        artefacts["target_instance"] = _copy(
            instance_source,
            target_dir / "04_target_instance" / "target_instance_filtered.ply",
        )
        artefacts["target_instance_filter_stats"] = _copy(
            stats_source, target_dir / "04_target_instance" / "filter_stats.json"
        )
        artefacts["target_instance_filter_projection"] = _copy(
            projection_source,
            target_dir / "04_target_instance" / "filter_projection.png",
        )
        if instance_source is not None and stats_source is None:
            fallback_stats = target_dir / "04_target_instance" / "filter_stats.json"
            _write_json(
                fallback_stats,
                {
                    "applied": False,
                    "fallback_reason": "runtime_filter_artifact_missing",
                    "source_ply": str(instance_source),
                    "note": "Raw NVSM instance cloud archived for audit; no filtered artifact was emitted by the runtime path.",
                },
            )
            artefacts["target_instance_filter_stats"] = str(fallback_stats)

        final_match = self._select_final_match(source_dir)
        match_metadata = None
        if final_match is not None:
            artefacts["selected_render"] = _copy(
                final_match["image_path"],
                target_dir / "05_final_match" / "selected_render.png",
            )
            artefacts["target_render_matches"] = _copy(
                final_match["match_path"],
                target_dir / "05_final_match" / "target_render_matches.png",
            )
            source_metadata = final_match["metadata"]
            keep_keys = (
                "stage",
                "step",
                "best_view_index",
                "score",
                "match_count",
                "source",
                "match_success",
                "threshold",
                "feature_mode",
            )
            match_metadata = {
                key: source_metadata.get(key)
                for key in keep_keys
                if key in source_metadata
            }
            match_metadata["archived_selected_render"] = artefacts.get("selected_render")
            match_metadata["archived_target_render_matches"] = artefacts.get("target_render_matches")
            _write_json(
                target_dir / "05_final_match" / "match_metrics.json", match_metadata
            )
            artefacts["match_metrics"] = str(
                target_dir / "05_final_match" / "match_metrics.json"
            )

        relative_artefacts = {
            key: (
                str(Path(value).relative_to(target_dir)) if value is not None else None
            )
            for key, value in artefacts.items()
        }
        missing = [key for key, value in relative_artefacts.items() if value is None]
        summary = dict(record)
        summary.update(
            {
                "schema_version": "multi-iin-target-compact-v1",
                "variant": self.variant,
                "outcome": "success" if float(record.get("success", 0.0)) > 0 else "failure",
                "failure_reason": (
                    None
                    if float(record.get("success", 0.0)) > 0
                    else record.get("failure_reason", "target_not_reached")
                ),
                "map_stats": map_stats,
                "gaussian_render_quality": gaussian_quality or {},
                "terminal_localization": terminal_localization or {},
                "artifacts": relative_artefacts,
                "missing_artifacts": missing,
            }
        )
        _write_json(target_dir / "target_summary.json", summary)
        self.targets_by_episode.setdefault(episode_name, []).append(summary)
        _write_json(
            episode_dir / "episode_summary.json",
            {
                "schema_version": "multi-iin-episode-compact-v1",
                "multi_episode_id": info.get("multi_episode_id"),
                "scene_name": info.get("scene_name"),
                "variant": self.variant,
                "target_order": [
                    {
                        "target_index": item.get("target_index"),
                        "goal_name": item.get("goal_name"),
                        "goal_object_id": item.get("goal_object_id"),
                        "success": item.get("success"),
                        "steps": item.get("steps"),
                    }
                    for item in self.targets_by_episode[episode_name]
                ],
                "ordered_success": bool(
                    len(self.targets_by_episode[episode_name]) == int(info.get("multi_num_targets", 5))
                    and all(float(item.get("success", 0.0)) > 0 for item in self.targets_by_episode[episode_name])
                ),
            },
        )
        return target_dir

    def finish(self, result_payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        summary_path = self.run_dir / "run_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary.update(
            {
                "status": "complete",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "passed": bool(result_payload.get("passed", False)),
                "episodes_archived": int(len(self.targets_by_episode)),
                "targets_archived": int(sum(len(items) for items in self.targets_by_episode.values())),
                "full_result_json": str(result_payload.get("result_json", "")),
            }
        )
        _write_json(summary_path, summary)
        for episode in result_payload.get("episodes", []):
            episode_id = str(episode.get("multi_episode_id", ""))
            episode_dir = self.episode_dir_by_id.get(episode_id)
            episode_summary_path = (
                episode_dir / "episode_summary.json" if episode_dir is not None else None
            )
            if episode_summary_path is None or not episode_summary_path.is_file():
                continue
            with episode_summary_path.open("r", encoding="utf-8") as handle:
                episode_summary = json.load(handle)
            boundary = episode.get("target_boundary", {})
            episode_summary.update(
                {
                    "ordered_success": bool(episode.get("ordered_success", False)),
                    "total_steps": int(episode.get("total_steps", 0)),
                    "target_boundary": {
                        "variant": boundary.get("variant"),
                        "runtime_before": boundary.get("runtime_before"),
                        "runtime_after": boundary.get("runtime_after"),
                        "agent": boundary.get("agent"),
                    },
                }
            )
            _write_json(episode_summary_path, episode_summary)
