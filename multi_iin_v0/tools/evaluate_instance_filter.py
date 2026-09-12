#!/usr/bin/env python3
"""Offline batch evaluation for Multi-IIN semantic-instance PLY files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from multi_iin_v0.runtime.semantic_instance_filter import (
    InstanceFilterConfig,
    _write_xyz_ply,
    filter_instance_indices,
)


INSTANCE_PATTERN = re.compile(r"instance_semantic_\d{8}_\d{6}\.ply$")


def _read_ascii_ply_xyz(path: Path) -> np.ndarray:
    vertex_count = None
    header_lines = 0
    with path.open("r", encoding="ascii") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        header_lines = 1
        for line in handle:
            header_lines += 1
            stripped = line.strip()
            if stripped == "format binary_little_endian 1.0":
                raise ValueError(f"binary PLY is not supported: {path}")
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.rsplit(" ", 1)[-1])
            if stripped == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"missing vertex count: {path}")
    xyz = np.loadtxt(
        path,
        dtype=np.float32,
        skiprows=header_lines,
        max_rows=vertex_count,
        usecols=(0, 1, 2),
    )
    return np.asarray(xyz, dtype=np.float32).reshape(-1, 3)


def _read_target(path: Path) -> str:
    metadata = path.with_suffix(".txt")
    if not metadata.exists():
        return "unknown"
    match = re.search(r"^Target:\s*(\S+)", metadata.read_text(), re.MULTILINE)
    return match.group(1) if match else "unknown"


def _load_anchors(path: Optional[Path]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    entries = payload.get("anchors", payload)
    return {
        key: np.asarray(value["xyz"] if isinstance(value, dict) else value, dtype=np.float32)
        for key, value in entries.items()
    }


def _recover_confirmed_seed_indices(
    instance_ply: Path,
    instance_xyz: np.ndarray,
    tolerance_m: float = 0.05,
) -> Tuple[np.ndarray, Optional[Path]]:
    confirmation_dir = instance_ply.parent.parent / "target_confirmation"
    candidates = []
    instance_mtime = instance_ply.stat().st_mtime
    if confirmation_dir.exists():
        candidates = [
            path
            for path in confirmation_dir.glob("step_*_confirmed_target_semantic.ply")
            if path.stat().st_mtime <= instance_mtime
        ]
    if not candidates:
        return np.empty((0,), dtype=np.int64), None
    source = max(candidates, key=lambda path: path.stat().st_mtime)
    confirmed_xyz = _read_ascii_ply_xyz(source)
    if confirmed_xyz.size == 0 or instance_xyz.size == 0:
        return np.empty((0,), dtype=np.int64), source
    from sklearn.neighbors import NearestNeighbors

    model = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(instance_xyz)
    distances, indices = model.kneighbors(confirmed_xyz, return_distance=True)
    keep = distances[:, 0] <= float(tolerance_m)
    return np.unique(indices[keep, 0].astype(np.int64)), source


def _bbox_volume(stats: dict) -> Optional[float]:
    size = stats.get("bbox_size")
    if size is None:
        return None
    return float(np.prod(np.maximum(np.asarray(size, dtype=np.float64), 0.0)))


def _save_topdown(
    output: Path,
    xyz: np.ndarray,
    selected: np.ndarray,
    anchor: Optional[np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = np.zeros((xyz.shape[0],), dtype=bool)
    keep[np.asarray(selected, dtype=np.int64)] = True
    max_points = 50000
    stride = max(1, int(np.ceil(xyz.shape[0] / max_points)))
    shown = np.arange(0, xyz.shape[0], stride)
    shown_keep = keep[shown]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(xyz[shown, 0], xyz[shown, 2], s=1, c="#888888", alpha=0.55)
    axes[0].set_title(f"Raw ({xyz.shape[0]:,})")
    axes[1].scatter(
        xyz[shown[~shown_keep], 0],
        xyz[shown[~shown_keep], 2],
        s=1,
        c="#d62728",
        alpha=0.45,
        label="removed",
    )
    axes[1].scatter(
        xyz[shown[shown_keep], 0],
        xyz[shown[shown_keep], 2],
        s=1,
        c="#2ca02c",
        alpha=0.65,
        label="kept",
    )
    axes[1].set_title(f"Filtered ({int(keep.sum()):,})")
    axes[1].legend(loc="best", markerscale=5)
    for axis in axes:
        if anchor is not None:
            axis.scatter(
                [anchor[0]], [anchor[2]], marker="*", s=100, c="#1f77b4", label="SOIM anchor"
            )
        axis.set_xlabel("Habitat X (m)")
        axis.set_ylabel("Habitat Z (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def evaluate(
    input_root: Path,
    output_root: Path,
    anchors: Dict[str, np.ndarray],
    config: InstanceFilterConfig,
    recover_confirmed_seeds: bool,
) -> dict:
    samples = []
    ply_paths = sorted(
        path
        for path in input_root.rglob("instance_semantic_*.ply")
        if INSTANCE_PATTERN.search(path.name)
    )
    for ply_path in ply_paths:
        relative = ply_path.relative_to(input_root).as_posix()
        target = _read_target(ply_path)
        anchor = anchors.get(relative)
        xyz = _read_ascii_ply_xyz(ply_path)
        if recover_confirmed_seeds:
            seed_indices, seed_source = _recover_confirmed_seed_indices(ply_path, xyz)
        else:
            seed_indices = np.empty((0,), dtype=np.int64)
            seed_source = None
        selected, diagnostics = filter_instance_indices(
            xyz,
            np.arange(xyz.shape[0], dtype=np.int64),
            config,
            anchor_3d=anchor,
            seed_indices=seed_indices,
            target_category=target,
        )
        run_name = ply_path.parent.parent.name
        sample_name = f"{run_name}__{ply_path.stem}"
        sample_dir = output_root / target / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        filtered_ply = sample_dir / "instance_semantic_filtered.ply"
        stats_json = sample_dir / "instance_filter_stats.json"
        comparison_png = sample_dir / "filter_comparison_topdown.png"
        _write_xyz_ply(filtered_ply, xyz[selected], (0, 255, 0))
        diagnostics["source_ply"] = str(ply_path)
        diagnostics["anchor_source"] = "manifest_soim_trace" if anchor is not None else None
        diagnostics["confirmed_seed_source_ply"] = (
            str(seed_source) if seed_source is not None else None
        )
        diagnostics["confirmed_seed_recovered_count"] = int(seed_indices.size)
        diagnostics["artifacts"] = {
            "source_raw_ply": str(ply_path),
            "filtered_ply": str(filtered_ply),
            "comparison_topdown_png": str(comparison_png),
            "stats_json": str(stats_json),
        }
        stats_json.write_text(json.dumps(diagnostics, indent=2) + "\n")
        _save_topdown(comparison_png, xyz, selected, anchor)

        raw = diagnostics["raw"]
        filtered = diagnostics["filtered"]
        raw_volume = _bbox_volume(raw)
        filtered_volume = _bbox_volume(filtered)
        sample = {
            "target": target,
            "sample": sample_name,
            "source_ply": str(ply_path),
            "anchor_available": anchor is not None,
            "confirmed_seed_available": bool(seed_indices.size),
            "confirmed_seed_count": int(seed_indices.size),
            "raw_count": int(raw["count"]),
            "filtered_count": int(filtered["count"]),
            "removed_count": int(raw["count"] - filtered["count"]),
            "keep_ratio": float(filtered["count"]) / max(int(raw["count"]), 1),
            "raw_bbox_size": raw["bbox_size"],
            "filtered_bbox_size": filtered["bbox_size"],
            "bbox_volume_ratio": (
                None
                if raw_volume in (None, 0.0) or filtered_volume is None
                else filtered_volume / raw_volume
            ),
            "cluster_count": len(diagnostics.get("clusters", [])),
            "selection_mode": diagnostics.get("selection_mode"),
            "fallback_reason": diagnostics.get("fallback_reason"),
            "stats_json": str(stats_json),
            "comparison_topdown_png": str(comparison_png),
        }
        samples.append(sample)
        print(
            f"{target:8s} {sample_name}: {sample['raw_count']:7d} -> "
            f"{sample['filtered_count']:7d} ({sample['keep_ratio']:.3f}), "
            f"clusters={sample['cluster_count']}, fallback={sample['fallback_reason']}"
        )

    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["target"]].append(sample)
    by_target = {}
    for target, target_samples in sorted(grouped.items()):
        by_target[target] = {
            "samples": len(target_samples),
            "raw_points": sum(item["raw_count"] for item in target_samples),
            "filtered_points": sum(item["filtered_count"] for item in target_samples),
            "mean_keep_ratio": float(np.mean([item["keep_ratio"] for item in target_samples])),
            "median_keep_ratio": float(np.median([item["keep_ratio"] for item in target_samples])),
            "fallback_count": sum(item["fallback_reason"] is not None for item in target_samples),
        }
    return {
        "schema_version": "multi-iin-instance-filter-offline-eval-v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "config": asdict(config),
        "sample_count": len(samples),
        "anchors_available": sum(item["anchor_available"] for item in samples),
        "confirmed_seeds_available": sum(
            item["confirmed_seed_available"] for item in samples
        ),
        "targets": by_target,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path)
    parser.add_argument(
        "--recover-confirmed-seeds",
        action="store_true",
        help=(
            "recover the most recent preceding target-confirmation PLY and map "
            "its points back to the instance PLY within 5 cm"
        ),
    )
    args = parser.parse_args()
    config = InstanceFilterConfig()
    summary = evaluate(
        args.input_root,
        args.output_root,
        _load_anchors(args.anchor_manifest),
        config,
        args.recover_confirmed_seeds,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output_root / "samples.csv"
    fields = [
        "target", "sample", "anchor_available", "confirmed_seed_available",
        "confirmed_seed_count", "raw_count", "filtered_count",
        "removed_count", "keep_ratio", "bbox_volume_ratio", "cluster_count",
        "selection_mode", "fallback_reason", "source_ply", "comparison_topdown_png",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["samples"])
    print(f"summary: {summary_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
