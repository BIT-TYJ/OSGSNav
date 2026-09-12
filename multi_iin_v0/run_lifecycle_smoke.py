#!/usr/bin/env python3
"""Run the additive Multi-IIN v0 oracle lifecycle smoke test.

This runner deliberately tests protocol mechanics rather than GS-IIN quality.
It uses real HM3D navmeshes for motion and an auditable state ledger for the
reset/preserve contract of Independent-Reset, Continuous-Clear, and
Continuous-Shared.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from common import find_navmesh, load_scene_content, view_positions


VARIANTS = (
    "Independent-Reset",
    "Continuous-Clear",
    "Continuous-Shared",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("multi_iin_v0/manifests/miin_v0_val_k2.json"),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--action-step-m",
        type=float,
        default=0.25,
        help="distance represented by one oracle movement step",
    )
    parser.add_argument(
        "--map-cell-m",
        type=float,
        default=0.25,
        help="cell size for the state-ledger visited map",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multi_iin_v0/results/lifecycle_smoke_10.json"),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close_xyz(a: Sequence[float], b: Sequence[float], tolerance: float = 1e-4) -> bool:
    return all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b))


@dataclass
class ReusableStateLedger:
    """Observable stand-in for the reset boundary of expensive runtime state."""

    visited_cells: set = field(default_factory=set)
    map_observations: int = 0
    gs_observations: int = 0
    history_frames: int = 0
    reset_count: int = 0

    def clear(self) -> None:
        self.visited_cells.clear()
        self.map_observations = 0
        self.gs_observations = 0
        self.history_frames = 0
        self.reset_count += 1

    def snapshot(self) -> Dict[str, int]:
        return {
            "visited_cells": len(self.visited_cells),
            "map_observations": self.map_observations,
            "gs_observations": self.gs_observations,
            "history_frames": self.history_frames,
            "reset_count": self.reset_count,
        }

    def observe_path(self, points: Iterable[Sequence[float]], cell_m: float) -> None:
        count = 0
        for point in points:
            x, _, z = (float(value) for value in point)
            self.visited_cells.add((round(x / cell_m), round(z / cell_m)))
            count += 1
        self.map_observations += count
        self.gs_observations += count
        self.history_frames += count


@dataclass
class TargetStateLedger:
    active_goal_key: str
    generation: int = 0
    candidate_count: int = 0
    match_attempts: int = 0

    def switch(self, goal_key: str) -> None:
        self.active_goal_key = goal_key
        self.generation += 1
        self.candidate_count = 0
        self.match_attempts = 0


class PathFinderCache:
    def __init__(self, scenes_root: Path):
        try:
            import habitat_sim
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("runner requires habitat-sim and numpy") from exc
        self.habitat_sim = habitat_sim
        self.np = np
        self.scenes_root = scenes_root
        self.cache: Dict[str, Any] = {}

    def get(self, scene_key: str, scene_id: str):
        if scene_key not in self.cache:
            pathfinder = self.habitat_sim.PathFinder()
            navmesh = find_navmesh(self.scenes_root, scene_key, scene_id)
            if not pathfinder.load_nav_mesh(str(navmesh)):
                raise RuntimeError("failed to load navmesh {}".format(navmesh))
            self.cache[scene_key] = pathfinder
        return self.cache[scene_key]

    def shortest_path(
        self,
        scene_key: str,
        scene_id: str,
        start: Sequence[float],
        ends: Sequence[Sequence[float]],
    ) -> Tuple[float, List[List[float]]]:
        pathfinder = self.get(scene_key, scene_id)
        query = self.habitat_sim.MultiGoalShortestPath()
        query.requested_start = self.np.asarray(start, dtype=self.np.float32)
        query.requested_ends = self.np.asarray(ends, dtype=self.np.float32)
        if not pathfinder.find_path(query):
            return math.inf, []
        points = [
            [float(value) for value in point]
            for point in query.points
        ]
        return float(query.geodesic_distance), points


def interpolate_path(
    points: Sequence[Sequence[float]], step_m: float
) -> List[List[float]]:
    if not points:
        return []
    samples = [[float(value) for value in points[0]]]
    for start, end in zip(points, points[1:]):
        delta = [float(end[i]) - float(start[i]) for i in range(3)]
        length = math.sqrt(sum(value * value for value in delta))
        pieces = max(1, int(math.ceil(length / step_m)))
        for index in range(1, pieces + 1):
            alpha = index / pieces
            samples.append(
                [float(start[i]) + alpha * delta[i] for i in range(3)]
            )
    return samples


def run_goal(
    path_cache: PathFinderCache,
    episode: Mapping[str, Any],
    target: Mapping[str, Any],
    source_goal: Mapping[str, Any],
    start_pose: Sequence[float],
    budget: int,
    action_step_m: float,
    map_cell_m: float,
    reusable: ReusableStateLedger,
    target_state: TargetStateLedger,
) -> Tuple[Dict[str, Any], List[float]]:
    distance, raw_points = path_cache.shortest_path(
        episode["scene_key"],
        episode["scene_id"],
        start_pose,
        view_positions(source_goal),
    )
    samples = interpolate_path(raw_points, action_step_m)
    required_steps = max(0, len(samples) - 1)
    success = math.isfinite(distance) and required_steps <= budget
    executed = samples[: budget + 1] if samples else []
    if executed:
        end_pose = executed[-1]
        reusable.observe_path(executed, map_cell_m)
    else:
        end_pose = [float(value) for value in start_pose]

    target_state.candidate_count += int(bool(executed))
    target_state.match_attempts += 1
    result = {
        "order": int(target["order"]),
        "goal_key": target["goal_key"],
        "source_episode_id": str(target["source_episode_id"]),
        "start_pose": [float(value) for value in start_pose],
        "end_pose": end_pose,
        "success": bool(success),
        "path_length_m": float(distance) if success else None,
        "required_steps": required_steps,
        "executed_steps": min(required_steps, budget),
        "budget": budget,
        "target_state_generation": target_state.generation,
    }
    return result, end_pose


def assert_boundary_contract(
    variant: str,
    original_start: Sequence[float],
    first_end: Sequence[float],
    second_start: Sequence[float],
    before_second: Mapping[str, int],
    target_state: TargetStateLedger,
) -> None:
    if variant == "Independent-Reset":
        assert close_xyz(second_start, original_start), "independent pose was not reset"
        assert before_second["map_observations"] == 0, "independent map leaked"
        assert before_second["gs_observations"] == 0, "independent 3DGS leaked"
        assert before_second["history_frames"] == 0, "independent history leaked"
        assert before_second["reset_count"] == 1, "independent reset not recorded"
    elif variant == "Continuous-Clear":
        assert close_xyz(second_start, first_end), "continuous pose was reset"
        assert before_second["map_observations"] == 0, "clear map was retained"
        assert before_second["gs_observations"] == 0, "clear 3DGS was retained"
        assert before_second["history_frames"] == 0, "clear history was retained"
        assert before_second["reset_count"] == 1, "clear reset not recorded"
    elif variant == "Continuous-Shared":
        assert close_xyz(second_start, first_end), "shared pose was reset"
        assert before_second["map_observations"] > 0, "shared map was empty"
        assert before_second["gs_observations"] > 0, "shared 3DGS ledger was empty"
        assert before_second["history_frames"] > 0, "shared history was empty"
        assert before_second["reset_count"] == 0, "shared state was cleared"
    else:
        raise AssertionError("unknown variant {}".format(variant))
    assert target_state.generation == 1, "target-specific state was not switched"


def run_variant_episode(
    variant: str,
    episode: Mapping[str, Any],
    source_scene: Mapping[str, Any],
    path_cache: PathFinderCache,
    budget: int,
    action_step_m: float,
    map_cell_m: float,
) -> Dict[str, Any]:
    original_start = [float(value) for value in episode["start_position"]]
    targets = episode["goals"]
    reusable = ReusableStateLedger()
    target_state = TargetStateLedger(active_goal_key=targets[0]["goal_key"])

    first, first_end = run_goal(
        path_cache,
        episode,
        targets[0],
        source_scene["goals"][targets[0]["goal_key"]],
        original_start,
        budget,
        action_step_m,
        map_cell_m,
        reusable,
        target_state,
    )
    state_after_first = reusable.snapshot()

    if variant == "Independent-Reset":
        second_start = list(original_start)
        reusable.clear()
    elif variant == "Continuous-Clear":
        second_start = list(first_end)
        reusable.clear()
    else:
        second_start = list(first_end)

    target_state.switch(targets[1]["goal_key"])
    state_before_second = reusable.snapshot()
    assert_boundary_contract(
        variant,
        original_start,
        first_end,
        second_start,
        state_before_second,
        target_state,
    )

    second, second_end = run_goal(
        path_cache,
        episode,
        targets[1],
        source_scene["goals"][targets[1]["goal_key"]],
        second_start,
        budget,
        action_step_m,
        map_cell_m,
        reusable,
        target_state,
    )
    successes = int(first["success"]) + int(second["success"])
    lengths = [
        goal["path_length_m"]
        for goal in (first, second)
        if goal["path_length_m"] is not None
    ]
    return {
        "episode_id": episode["episode_id"],
        "scene_key": episode["scene_key"],
        "variant": variant,
        "lifecycle_contract_passed": True,
        "goals": [first, second],
        "state_after_first": state_after_first,
        "state_before_second": state_before_second,
        "state_after_second": reusable.snapshot(),
        "final_pose": second_end,
        "goal_success_rate": successes / 2.0,
        "all_goal_success": successes == 2,
        "total_successful_path_m": sum(lengths),
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    output = {}
    for variant in VARIANTS:
        selected = [item for item in results if item["variant"] == variant]
        if not selected:
            continue
        second_lengths = [
            item["goals"][1]["path_length_m"]
            for item in selected
            if item["goals"][1]["path_length_m"] is not None
        ]
        output[variant] = {
            "episodes": len(selected),
            "goal_success_rate": mean(item["goal_success_rate"] for item in selected),
            "all_goal_success_rate": mean(
                float(item["all_goal_success"]) for item in selected
            ),
            "mean_total_successful_path_m": mean(
                item["total_successful_path_m"] for item in selected
            ),
            "mean_second_goal_path_m": mean(second_lengths) if second_lengths else None,
            "mean_second_goal_steps": mean(
                item["goals"][1]["executed_steps"] for item in selected
            ),
            "mean_memory_before_second": {
                key: mean(item["state_before_second"][key] for item in selected)
                for key in ("visited_cells", "map_observations", "gs_observations", "history_frames")
            },
            "lifecycle_contract_pass_rate": mean(
                float(item["lifecycle_contract_passed"]) for item in selected
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.action_step_m <= 0 or args.map_cell_m <= 0:
        raise SystemExit("step and cell sizes must be positive")

    started = time.time()
    manifest = load_json(args.manifest)
    episodes = manifest.get("episodes", [])[: args.episodes]
    if len(episodes) != args.episodes:
        raise RuntimeError(
            "manifest has only {} episodes, requested {}".format(
                len(episodes), args.episodes
            )
        )
    protocol = manifest["protocol"]
    if protocol.get("num_goals") != 2 or protocol.get("goal_order") != "fixed":
        raise RuntimeError("runner only supports ordered K=2 manifests")

    source = manifest["source"]
    source_scenes = load_scene_content(Path(source["dataset_root"]), source["split"])
    path_cache = PathFinderCache(Path(source["scenes_root"]))
    budget = int(protocol["per_goal_step_budget"])
    results: List[Dict[str, Any]] = []

    for episode_index, episode in enumerate(episodes, start=1):
        source_scene = source_scenes[episode["scene_key"]]
        for variant in args.variants:
            result = run_variant_episode(
                variant,
                episode,
                source_scene,
                path_cache,
                budget,
                args.action_step_m,
                args.map_cell_m,
            )
            results.append(result)
        print(
            "checked episode {}/{}: {}".format(
                episode_index, len(episodes), episode["episode_id"]
            ),
            flush=True,
        )

    payload = {
        "schema_version": "miin-lifecycle-smoke-v0.1",
        "runner_scope": "oracle_navmesh_and_state_boundary_only",
        "not_validated": [
            "GS-IIN policy quality",
            "real 2D mapper reuse",
            "real 3DGS reuse",
            "detector or target matching",
            "wall-clock reuse gain",
        ],
        "manifest": str(args.manifest),
        "episode_count": len(episodes),
        "variants": list(args.variants),
        "action_step_m": args.action_step_m,
        "map_cell_m": args.map_cell_m,
        "elapsed_s": time.time() - started,
        "aggregate": aggregate(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("wrote {} variant-runs to {}".format(len(results), args.output))
    for variant, metrics in payload["aggregate"].items():
        print(
            "{}: contract={:.1%} GSR={:.1%} all={:.1%} "
            "second_path={:.3f}m second_steps={:.1f}".format(
                variant,
                metrics["lifecycle_contract_pass_rate"],
                metrics["goal_success_rate"],
                metrics["all_goal_success_rate"],
                metrics["mean_second_goal_path_m"],
                metrics["mean_second_goal_steps"],
            )
        )


if __name__ == "__main__":
    main()

