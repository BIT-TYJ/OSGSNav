#!/usr/bin/env python3
"""Aggregate sharded Multi-IIN ``full_result.json`` files.

The output is intentionally table-oriented: individual targets and episodes
remain auditable, while the JSON/Markdown summaries provide paper-ready
overall, target-order, six-class, and runtime statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SIX_CATEGORIES = ("bed", "chair", "plant", "sofa", "toilet", "tv_monitor")


def canonical_category(value: Any) -> str:
    name = str(value or "unknown").strip().lower()
    aliases = {"tv": "tv_monitor", "monitor": "tv_monitor", "couch": "sofa"}
    return aliases.get(name, name)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = [_number(row.get(key)) for row in rows]
    return None if not values else sum(values) / len(values)


def _metric_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "count": len(rows),
        "success": _mean(rows, "success"),
        "spl": _mean(rows, "spl"),
        "soft_spl": _mean(rows, "soft_spl"),
        "mean_steps": _mean(rows, "steps"),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(root: Path) -> Dict[str, Any]:
    result_files = sorted(root.rglob("full_result.json"))
    targets: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    runtimes: List[Dict[str, Any]] = []
    invalid_files = []

    for path in result_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid_files.append({"path": str(path), "error": str(exc)})
            continue
        run_id = path.parent.name
        timing = payload.get("runtime_timing") or {}
        ordinary = (
            (timing.get("ordinary_frames") or {}).get("total_frame_s") or {}
        )
        all_frames = (timing.get("all_frames") or {}).get("total_frame_s") or {}
        stage = payload.get("stage_runtime") or {}
        runtime_row = {
            "run_id": run_id,
            "result_json": str(path),
            "ordinary_frame_count": int(ordinary.get("count", 0) or 0),
            "ordinary_frame_total_s": ordinary.get("total_s"),
            "ordinary_frame_mean_s": ordinary.get("mean_s"),
            "all_frame_count": int(all_frames.get("count", 0) or 0),
            "all_frame_total_s": all_frames.get("total_s"),
            "confirmation_calls": int(
                (stage.get("confirmation") or {}).get("calls", 0) or 0
            ),
            "confirmation_total_s": (stage.get("confirmation") or {}).get("total_s"),
            "localization_calls": int(
                (stage.get("localization") or {}).get("calls", 0) or 0
            ),
            "localization_total_s": (stage.get("localization") or {}).get("total_s"),
            "wall_clock_total_s": stage.get("wall_clock_total_s"),
        }
        runtimes.append(runtime_row)

        for episode in payload.get("episodes") or []:
            episode_id = str(episode.get("multi_episode_id", ""))
            goals = list(episode.get("goals") or [])
            episode_row = {
                "run_id": run_id,
                "multi_episode_id": episode_id,
                "scene_name": episode.get("scene_name"),
                "variant": episode.get("variant", payload.get("variant")),
                "target_count": len(goals),
                "ordered_success": int(bool(episode.get("ordered_success"))),
                "total_steps": int(episode.get("total_steps", 0) or 0),
                "result_json": str(path),
            }
            episodes.append(episode_row)
            for ordinal, goal in enumerate(goals):
                target_index = int(goal.get("target_index", ordinal))
                targets.append(
                    {
                        "run_id": run_id,
                        "multi_episode_id": episode_id,
                        "scene_name": episode.get("scene_name"),
                        "target_index": target_index,
                        "target_position": "target{}".format(target_index + 1),
                        "category": canonical_category(goal.get("goal_name")),
                        "goal_object_id": goal.get("goal_object_id"),
                        "success": _number(goal.get("success")),
                        "spl": _number(goal.get("spl")),
                        "soft_spl": _number(goal.get("soft_spl")),
                        "steps": int(goal.get("steps", 0) or 0),
                        "distance_to_goal_m": goal.get("distance_to_goal_m"),
                        "failure_reason": goal.get("failure_reason", ""),
                        "pipeline_stage": goal.get("pipeline_stage", "unknown"),
                        "result_json": str(path),
                    }
                )

    by_position = {}
    for target_index in (0, 1):
        rows = [row for row in targets if row["target_index"] == target_index]
        by_position["target{}".format(target_index + 1)] = _metric_group(rows)
    by_category = {}
    for category in SIX_CATEGORIES:
        rows = [row for row in targets if row["category"] == category]
        by_category[category] = _metric_group(rows)
    by_pipeline_stage = {}
    for stage_name in sorted({row.get("pipeline_stage", "unknown") for row in targets}):
        stage_rows = [
            row for row in targets if row.get("pipeline_stage", "unknown") == stage_name
        ]
        by_pipeline_stage[stage_name] = _metric_group(stage_rows)

    ordinary_count = sum(row["ordinary_frame_count"] for row in runtimes)
    ordinary_total = sum(
        _number(row["ordinary_frame_total_s"])
        for row in runtimes if row["ordinary_frame_total_s"] is not None
    )
    confirmation_calls = sum(row["confirmation_calls"] for row in runtimes)
    confirmation_total = sum(
        _number(row["confirmation_total_s"])
        for row in runtimes if row["confirmation_total_s"] is not None
    )
    localization_calls = sum(row["localization_calls"] for row in runtimes)
    localization_total = sum(
        _number(row["localization_total_s"])
        for row in runtimes if row["localization_total_s"] is not None
    )
    wall_values = [
        _number(row["wall_clock_total_s"])
        for row in runtimes if row["wall_clock_total_s"] is not None
    ]
    ordered_successes = sum(row["ordered_success"] for row in episodes)
    summary = {
        "schema_version": "multi-iin-aggregate-v1",
        "root": str(root.resolve()),
        "result_file_count": len(result_files),
        "valid_result_file_count": len(runtimes),
        "invalid_files": invalid_files,
        "episode_count": len(episodes),
        "target_count": len(targets),
        "overall_targets": _metric_group(targets),
        "ordered": {
            "count": len(episodes),
            "successes": ordered_successes,
            "success": None if not episodes else ordered_successes / len(episodes),
        },
        "by_target_position": by_position,
        "by_category": by_category,
        "by_pipeline_stage": by_pipeline_stage,
        "runtime": {
            "ordinary_frames": {
                "count": ordinary_count,
                "total_s": ordinary_total,
                "mean_s": None if not ordinary_count else ordinary_total / ordinary_count,
            },
            "confirmation": {
                "calls": confirmation_calls,
                "total_s": confirmation_total,
                "mean_call_s": None if not confirmation_calls else confirmation_total / confirmation_calls,
            },
            "localization": {
                "calls": localization_calls,
                "total_s": localization_total,
                "mean_call_s": None if not localization_calls else localization_total / localization_calls,
            },
            "wall_clock_total_s": sum(wall_values) if wall_values else None,
        },
        "targets": targets,
        "episodes": episodes,
        "runtime_runs": runtimes,
    }
    return summary


def _percent(value: Optional[float]) -> str:
    return "N/A" if value is None else "{:.2f}%".format(100.0 * value)


def write_outputs(summary: Dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "per_target.csv", summary["targets"])
    _write_csv(output / "per_episode.csv", summary["episodes"])
    _write_csv(output / "runtime_per_run.csv", summary["runtime_runs"])
    _write_csv(
        output / "by_target_position.csv",
        [{"target_position": key, **value} for key, value in summary["by_target_position"].items()],
    )
    _write_csv(
        output / "by_category.csv",
        [{"category": key, **value} for key, value in summary["by_category"].items()],
    )
    _write_csv(
        output / "by_pipeline_stage.csv",
        [
            {"pipeline_stage": key, **value}
            for key, value in summary["by_pipeline_stage"].items()
        ],
    )

    lines = [
        "# Multi-IIN aggregate",
        "",
        "- Episodes: {}".format(summary["episode_count"]),
        "- Targets: {}".format(summary["target_count"]),
        "- Per-target Success: {}".format(_percent(summary["overall_targets"]["success"])),
        "- Per-target SPL: {:.4f}".format(summary["overall_targets"]["spl"] or 0.0),
        "- Ordered two-target success: {}".format(_percent(summary["ordered"]["success"])),
        "",
        "## Target order",
        "",
        "| Position | N | Success | SPL |",
        "|---|---:|---:|---:|",
    ]
    for name, item in summary["by_target_position"].items():
        lines.append("| {} | {} | {} | {:.4f} |".format(name, item["count"], _percent(item["success"]), item["spl"] or 0.0))
    lines.extend(["", "## Six categories", "", "| Category | N | Success | SPL |", "|---|---:|---:|---:|"])
    for name, item in summary["by_category"].items():
        lines.append("| {} | {} | {} | {:.4f} |".format(name, item["count"], _percent(item["success"]), item["spl"] or 0.0))
    runtime = summary["runtime"]
    lines.extend([
        "", "## Runtime", "",
        "| Component | Count | Total (s) | Mean (s) |",
        "|---|---:|---:|---:|",
        "| Ordinary frame | {} | {:.3f} | {:.3f} |".format(runtime["ordinary_frames"]["count"], runtime["ordinary_frames"]["total_s"] or 0.0, runtime["ordinary_frames"]["mean_s"] or 0.0),
        "| Confirmation | {} | {:.3f} | {:.3f} |".format(runtime["confirmation"]["calls"], runtime["confirmation"]["total_s"] or 0.0, runtime["confirmation"]["mean_call_s"] or 0.0),
        "| Localization | {} | {:.3f} | {:.3f} |".format(runtime["localization"]["calls"], runtime["localization"]["total_s"] or 0.0, runtime["localization"]["mean_call_s"] or 0.0),
        "| Total wall clock | {} runs | {:.3f} | - |".format(summary["valid_result_file_count"], runtime["wall_clock_total_s"] or 0.0),
        "",
    ])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="experiment root containing shard directories")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or (args.root / "aggregate")
    summary = aggregate(args.root)
    write_outputs(summary, output)
    print(json.dumps({
        "episodes": summary["episode_count"],
        "targets": summary["target_count"],
        "target_success": summary["overall_targets"]["success"],
        "target_spl": summary["overall_targets"]["spl"],
        "ordered_success": summary["ordered"]["success"],
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
