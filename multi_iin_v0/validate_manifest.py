#!/usr/bin/env python3
"""Validate source references and geodesics in a Multi-IIN v0 manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from common import (
    GeodesicOracle,
    find_navmesh,
    load_scene_content,
    view_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--check-geodesic",
        action="store_true",
        help="recompute all stored geodesic distances from navmeshes",
    )
    parser.add_argument("--tolerance-m", type=float, default=1e-4)
    return parser.parse_args()


def require(condition: bool, message: str, errors: List[str]) -> None:
    if not condition:
        errors.append(message)


def validate_episode(
    episode: Mapping[str, Any],
    source_scene: Mapping[str, Any],
    require_distinct_categories: bool,
    errors: List[str],
) -> None:
    prefix = str(episode.get("episode_id", "<missing-id>"))
    goals = episode.get("goals", [])
    require(len(goals) == 2, "{}: expected exactly two goals".format(prefix), errors)
    if len(goals) != 2:
        return

    require(
        goals[0].get("order") == 0 and goals[1].get("order") == 1,
        "{}: goal order must be [0, 1]".format(prefix),
        errors,
    )
    require(
        goals[0].get("goal_object_id") != goals[1].get("goal_object_id"),
        "{}: target instances must be distinct".format(prefix),
        errors,
    )
    if require_distinct_categories:
        require(
            goals[0].get("object_category") != goals[1].get("object_category"),
            "{}: target categories must be distinct".format(prefix),
            errors,
        )

    source_by_id = {
        str(item["episode_id"]): item for item in source_scene.get("episodes", [])
    }
    source_goals = source_scene.get("goals", {})
    for goal in goals:
        source_id = str(goal.get("source_episode_id"))
        require(
            source_id in source_by_id,
            "{}: missing source episode {}".format(prefix, source_id),
            errors,
        )
        require(
            goal.get("goal_key") in source_goals,
            "{}: missing source goal {}".format(prefix, goal.get("goal_key")),
            errors,
        )
        if source_id not in source_by_id or goal.get("goal_key") not in source_goals:
            continue
        source = source_by_id[source_id]
        source_goal = source_goals[goal["goal_key"]]
        require(
            str(source["goal_object_id"]) == str(goal["goal_object_id"]),
            "{}: goal_object_id differs from source".format(prefix),
            errors,
        )
        require(
            source["object_category"] == goal["object_category"],
            "{}: object_category differs from source".format(prefix),
            errors,
        )
        image_id = int(goal["goal_image_id"])
        require(
            0 <= image_id < len(source_goal.get("image_goals", [])),
            "{}: invalid goal_image_id {}".format(prefix, image_id),
            errors,
        )
        require(
            bool(view_positions(source_goal)),
            "{}: goal has no valid view points".format(prefix),
            errors,
        )


def check_geodesic(
    episode: Mapping[str, Any],
    source_scene: Mapping[str, Any],
    scenes_root: Path,
    tolerance: float,
    errors: List[str],
    oracle_cache: Dict[str, GeodesicOracle],
) -> None:
    scene_key = episode["scene_key"]
    if scene_key not in oracle_cache:
        oracle_cache[scene_key] = GeodesicOracle(
            find_navmesh(scenes_root, scene_key, str(episode["scene_id"]))
        )
    oracle = oracle_cache[scene_key]
    first = source_scene["goals"][episode["goals"][0]["goal_key"]]
    second = source_scene["goals"][episode["goals"][1]["goal_key"]]
    first_views = view_positions(first)
    second_views = view_positions(second)
    route, start_to_first, _ = oracle.ordered_two_goal_distance(
        episode["start_position"], first_views, second_views
    )
    start_to_second = oracle.distance_to_region(
        episode["start_position"], second_views
    )
    pair_distance = min(
        oracle.distance_to_region(view, second_views) for view in first_views
    )
    expected = {
        "ordered_length_m": route,
        "start_to_first_m": start_to_first,
        "start_to_second_m": start_to_second,
        "first_to_second_m": pair_distance,
    }
    for key, value in expected.items():
        stored = float(episode["oracle"][key])
        if not math.isfinite(value) or abs(stored - value) > tolerance:
            errors.append(
                "{}: {} stored={} recomputed={}".format(
                    episode["episode_id"], key, stored, value
                )
            )


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    errors: List[str] = []
    require(
        manifest.get("schema_version") == "miin-v0.1",
        "unsupported schema_version",
        errors,
    )
    protocol = manifest.get("protocol", {})
    require(protocol.get("num_goals") == 2, "protocol.num_goals must be 2", errors)
    require(protocol.get("goal_order") == "fixed", "goal_order must be fixed", errors)
    require(
        protocol.get("goal_visibility") == "current_goal_only",
        "goal_visibility must be current_goal_only",
        errors,
    )

    source = manifest.get("source", {})
    dataset_root = Path(source["dataset_root"])
    scenes_root = Path(source["scenes_root"])
    split = source["split"]
    source_scenes = load_scene_content(dataset_root, split)
    generation = manifest.get("generation", {})
    min_first_to_second_m = float(
        generation.get("min_first_to_second_m", 0.0)
    )
    expected_count = int(generation["num_scenes"]) * int(
        generation["episodes_per_scene"]
    )
    episodes = manifest.get("episodes", [])
    require(
        len(episodes) == expected_count,
        "episode count {} != expected {}".format(len(episodes), expected_count),
        errors,
    )

    ids = [str(episode.get("episode_id")) for episode in episodes]
    require(len(ids) == len(set(ids)), "episode ids are not unique", errors)
    scene_counts = Counter(episode.get("scene_key") for episode in episodes)
    selected = set(generation.get("selected_scenes", []))
    require(set(scene_counts) == selected, "selected scene list mismatch", errors)

    for episode in episodes:
        scene_key = episode.get("scene_key")
        if scene_key not in source_scenes:
            errors.append("{}: unknown scene {}".format(episode.get("episode_id"), scene_key))
            continue
        validate_episode(
            episode,
            source_scenes[scene_key],
            bool(generation.get("require_distinct_categories", False)),
            errors,
        )
        require(
            float(episode.get("oracle", {}).get("first_to_second_m", -1.0))
            >= min_first_to_second_m,
            "{}: first_to_second_m is below configured minimum {:.3f}m".format(
                episode.get("episode_id"), min_first_to_second_m
            ),
            errors,
        )
        goals = episode.get("goals", [])
        if goals:
            require(
                scene_key == str(goals[0].get("goal_key", "")).rsplit("_", 1)[0],
                "{}: first goal belongs to another scene".format(
                    episode["episode_id"]
                ),
                errors,
            )

    if args.check_geodesic and not errors:
        oracle_cache: Dict[str, GeodesicOracle] = {}
        for index, episode in enumerate(episodes, start=1):
            check_geodesic(
                episode,
                source_scenes[episode["scene_key"]],
                scenes_root,
                args.tolerance_m,
                errors,
                oracle_cache,
            )
            if index % 20 == 0:
                print("geodesic checked {}/{}".format(index, len(episodes)))

    if errors:
        for error in errors[:50]:
            print("ERROR: {}".format(error))
        raise SystemExit("validation failed with {} error(s)".format(len(errors)))

    bins = Counter(
        episode["difficulty"]["target_pair_distance_bin"] for episode in episodes
    )
    print(
        "valid: episodes={} scenes={} bins={} geodesic_recomputed={}".format(
            len(episodes), len(scene_counts), dict(sorted(bins.items())), args.check_geodesic
        )
    )


if __name__ == "__main__":
    main()
