#!/usr/bin/env python3
"""Build a K=2 Sequential Ordered Multi-IIN manifest from IIN-v3 data."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping

from common import (
    GeodesicOracle,
    distance_bin,
    find_navmesh,
    goal_key,
    load_scene_content,
    view_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/tyj/datasets/instance_imagenav/hm3d/v3"),
    )
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=Path("data/tyj/hm3d-v0.2"),
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--episodes-per-scene", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--allow-same-category",
        action="store_true",
        help="allow the two requested targets to share a category",
    )
    parser.add_argument(
        "--same-floor",
        dest="same_floor",
        action="store_true",
        default=True,
        help="only pair targets whose view-point heights belong to the same floor",
    )
    parser.add_argument(
        "--allow-cross-floor",
        dest="same_floor",
        action="store_false",
        help="disable same-floor filtering (for experiments with 3D navigation)",
    )
    parser.add_argument(
        "--floor-height-tolerance-m",
        type=float,
        default=0.75,
        help="maximum median target-view height difference for same-floor pairing",
    )
    parser.add_argument(
        "--min-first-to-second-m",
        type=float,
        default=3.0,
        help=(
            "minimum geodesic distance between the two target view-point regions; "
            "prevents target 2 from being completed almost at the target-1 stop pose"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multi_iin_v0/manifests/miin_v0_val_k2.json"),
    )
    return parser.parse_args()


def target_from_episode(
    source_episode: Mapping[str, Any], scene_key: str, order: int
) -> Dict[str, Any]:
    return {
        "order": order,
        "source_episode_id": str(source_episode["episode_id"]),
        "goal_key": goal_key(scene_key, source_episode["goal_object_id"]),
        "goal_object_id": str(source_episode["goal_object_id"]),
        "goal_image_id": int(source_episode["goal_image_id"]),
        # Multi-IIN uses this task-defined category instead of inferring the
        # class from a potentially ambiguous goal image.
        "object_category": str(source_episode["object_category"]),
    }


def median_view_height(goal: Mapping[str, Any]) -> float:
    """Robust representative floor height for an IIN target's view points."""
    heights = [
        float(position[1])
        for position in view_positions(goal)
        if len(position) >= 3 and math.isfinite(float(position[1]))
    ]
    if not heights:
        return math.nan
    return float(statistics.median(heights))


def same_floor_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    goals: Mapping[str, Any],
    scene_key: str,
    tolerance_m: float,
) -> bool:
    first_height = median_view_height(
        goals[goal_key(scene_key, first["goal_object_id"])]
    )
    second_height = median_view_height(
        goals[goal_key(scene_key, second["goal_object_id"])]
    )
    if not (math.isfinite(first_height) and math.isfinite(second_height)):
        return False
    start_height = float(first["start_position"][1])
    return (
        abs(first_height - second_height) <= float(tolerance_m)
        and abs(start_height - first_height) <= float(tolerance_m)
        and abs(start_height - second_height) <= float(tolerance_m)
    )


def build_scene_episodes(
    scene_key: str,
    payload: Mapping[str, Any],
    scenes_root: Path,
    count: int,
    rng: random.Random,
    require_distinct_categories: bool,
    require_same_floor: bool,
    floor_height_tolerance_m: float,
    min_first_to_second_m: float,
) -> List[Dict[str, Any]]:
    source_episodes = list(payload["episodes"])
    rng.shuffle(source_episodes)
    pairs = [
        (first, second)
        for first in source_episodes
        for second in source_episodes
        if str(first["goal_object_id"]) != str(second["goal_object_id"])
        and (
            not require_distinct_categories
            or first["object_category"] != second["object_category"]
        )
        and (
            not require_same_floor
            or same_floor_pair(
                first,
                second,
                payload["goals"],
                scene_key,
                floor_height_tolerance_m,
            )
        )
    ]
    rng.shuffle(pairs)

    scene_id = str(source_episodes[0]["scene_id"])
    oracle = GeodesicOracle(find_navmesh(scenes_root, scene_key, scene_id))
    accepted = []
    used_pairs = set()
    for first, second in pairs:
        pair_key = (str(first["episode_id"]), str(second["episode_id"]))
        if pair_key in used_pairs:
            continue

        first_key = goal_key(scene_key, first["goal_object_id"])
        second_key = goal_key(scene_key, second["goal_object_id"])
        first_views = view_positions(payload["goals"][first_key])
        second_views = view_positions(payload["goals"][second_key])
        route, start_to_first, _ = (
            oracle.ordered_two_goal_distance(
                first["start_position"], first_views, second_views
            )
        )
        start_to_second = oracle.distance_to_region(
            first["start_position"], second_views
        )
        pair_distance = min(
            oracle.distance_to_region(view, second_views) for view in first_views
        )
        if not all(
            math.isfinite(value)
            for value in (route, start_to_first, start_to_second, pair_distance)
        ):
            continue
        if pair_distance < float(min_first_to_second_m):
            continue

        accepted.append(
            {
                "scene_id": str(first["scene_id"]),
                "scene_key": scene_key,
                "start_position": [float(v) for v in first["start_position"]],
                "start_rotation": [float(v) for v in first["start_rotation"]],
                "goals": [
                    target_from_episode(first, scene_key, 0),
                    target_from_episode(second, scene_key, 1),
                ],
                "oracle": {
                    "definition": "view_point_center_path",
                    "ordered_length_m": route,
                    "start_to_first_m": start_to_first,
                    "start_to_second_m": start_to_second,
                    "first_to_second_m": pair_distance,
                },
                "difficulty": {
                    "same_category": (
                        first["object_category"] == second["object_category"]
                    ),
                    "target_pair_distance_bin": distance_bin(pair_distance),
                },
            }
        )
        used_pairs.add(pair_key)
        if len(accepted) == count:
            break

    return accepted


def main() -> None:
    args = parse_args()
    if args.num_scenes <= 0 or args.episodes_per_scene <= 0:
        raise SystemExit("num-scenes and episodes-per-scene must be positive")

    scenes = load_scene_content(args.dataset_root, args.split)
    rng = random.Random(args.seed)
    scene_keys = sorted(scenes)
    rng.shuffle(scene_keys)

    output_episodes: List[Dict[str, Any]] = []
    selected_scenes = []
    for scene_key in scene_keys:
        candidates = build_scene_episodes(
            scene_key=scene_key,
            payload=scenes[scene_key],
            scenes_root=args.scenes_root,
            count=args.episodes_per_scene,
            rng=rng,
            require_distinct_categories=not args.allow_same_category,
            require_same_floor=bool(args.same_floor),
            floor_height_tolerance_m=float(args.floor_height_tolerance_m),
            min_first_to_second_m=max(0.0, float(args.min_first_to_second_m)),
        )
        if len(candidates) != args.episodes_per_scene:
            continue
        selected_scenes.append(scene_key)
        for candidate in candidates:
            candidate["episode_id"] = "miin_v0_{}_{:06d}".format(
                args.split, len(output_episodes)
            )
            output_episodes.append(candidate)
        if len(selected_scenes) == args.num_scenes:
            break

    if len(selected_scenes) != args.num_scenes:
        raise RuntimeError(
            "only {} scenes could provide {} valid episodes".format(
                len(selected_scenes), args.episodes_per_scene
            )
        )

    manifest = {
        "schema_version": "miin-v0.1",
        "task": "SequentialOrderedMultiIIN",
        "protocol": {
            "num_goals": 2,
            "goal_order": "fixed",
            "goal_visibility": "current_goal_only",
            "goal_category_source": "task_object_category",
            "per_goal_step_budget": 1000,
            "success_distance_m": 1.0,
            "intermediate_stop_semantics": "declare_current_goal",
            "preserve_simulator_pose_between_goals": True,
            "policy_must_not_observe": [
                "goal_object_id",
                "view_points",
                "oracle",
            ],
        },
        "source": {
            "benchmark": "Habitat 2023 InstanceImageNav",
            "dataset": "HM3D InstanceImageNav v3",
            "dataset_root": str(args.dataset_root),
            "scenes_root": str(args.scenes_root),
            "split": args.split,
        },
        "generation": {
            "seed": args.seed,
            "num_scenes": args.num_scenes,
            "episodes_per_scene": args.episodes_per_scene,
            "require_distinct_categories": not args.allow_same_category,
            "require_same_floor": bool(args.same_floor),
            "floor_height_tolerance_m": float(args.floor_height_tolerance_m),
            "min_first_to_second_m": max(0.0, float(args.min_first_to_second_m)),
            "selected_scenes": selected_scenes,
        },
        "episodes": output_episodes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        "wrote {} episodes from {} scenes to {}".format(
            len(output_episodes), len(selected_scenes), args.output
        )
    )


if __name__ == "__main__":
    main()
