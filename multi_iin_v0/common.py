#!/usr/bin/env python3
"""Shared, read-only helpers for the Multi-IIN v0 manifest tools."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def open_json(path: Path) -> Dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def scene_key_from_id(scene_id: str) -> str:
    name = Path(scene_id).name
    for suffix in (".basis.glb", ".glb"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError("unsupported scene_id: {}".format(scene_id))


def goal_key(scene_key: str, goal_object_id: Any) -> str:
    return "{}_{}".format(scene_key, goal_object_id)


def view_positions(goal: Mapping[str, Any]) -> List[List[float]]:
    positions = []
    for view in goal.get("view_points", []):
        position = view.get("agent_state", {}).get("position")
        if position is not None and len(position) == 3:
            positions.append([float(value) for value in position])
    return positions


def distance_bin(distance_m: float) -> str:
    if distance_m < 5.0:
        return "near"
    if distance_m < 10.0:
        return "medium"
    return "far"


class GeodesicOracle:
    """Small PathFinder wrapper that does not start Habitat Simulator."""

    def __init__(self, navmesh_path: Path):
        try:
            import habitat_sim
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "geodesic validation requires habitat-sim and numpy"
            ) from exc

        self._habitat_sim = habitat_sim
        self._np = np
        self.navmesh_path = navmesh_path
        self.pathfinder = habitat_sim.PathFinder()
        if not self.pathfinder.load_nav_mesh(str(navmesh_path)):
            raise RuntimeError("failed to load navmesh: {}".format(navmesh_path))

    def distance_to_region(
        self, start: Sequence[float], ends: Iterable[Sequence[float]]
    ) -> float:
        ends_array = self._np.asarray(list(ends), dtype=self._np.float32)
        if ends_array.size == 0:
            return math.inf
        query = self._habitat_sim.MultiGoalShortestPath()
        query.requested_start = self._np.asarray(start, dtype=self._np.float32)
        query.requested_ends = ends_array
        if not self.pathfinder.find_path(query):
            return math.inf
        return float(query.geodesic_distance)

    def ordered_two_goal_distance(
        self,
        start: Sequence[float],
        first_region: Sequence[Sequence[float]],
        second_region: Sequence[Sequence[float]],
    ) -> Tuple[float, float, float]:
        """Return exact view-point route and two diagnostic distances.

        The ordered route minimizes d(start, v1) + d(v1, V2) over the same
        first-goal endpoint v1.  This avoids the inconsistent-endpoint
        underestimate caused by independently minimizing the two segments.
        """

        start_to_first = self.distance_to_region(start, first_region)
        start_to_second = self.distance_to_region(start, second_region)
        best_route = math.inf
        first_to_second = math.inf

        for first_view in first_region:
            first_leg = self.distance_to_region(start, [first_view])
            if not math.isfinite(first_leg):
                continue
            second_leg = self.distance_to_region(first_view, second_region)
            if not math.isfinite(second_leg):
                continue
            first_to_second = min(first_to_second, second_leg)
            best_route = min(best_route, first_leg + second_leg)

        return best_route, start_to_first, min(start_to_second, first_to_second)


def find_navmesh(
    scenes_root: Path, scene_key: str, scene_id: str = ""
) -> Path:
    if scene_id:
        scene_path = Path(scene_id)
        if len(scene_path.parts) >= 3:
            relative_scene = Path(*scene_path.parts[-3:])
            direct = (scenes_root / relative_scene).with_suffix(".navmesh")
            if direct.is_file():
                return direct

    matches = sorted(scenes_root.glob("*/{}/*.navmesh".format(scene_key)))
    if not matches:
        matches = sorted(scenes_root.rglob("{}*.navmesh".format(scene_key)))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected one navmesh for {}, found {} under {}".format(
                scene_key, len(matches), scenes_root
            )
        )
    return matches[0]


def load_scene_content(dataset_root: Path, split: str) -> Dict[str, Dict[str, Any]]:
    content_dir = dataset_root / split / "content"
    files = sorted(content_dir.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError("no scene content files under {}".format(content_dir))
    result = {}
    for path in files:
        payload = open_json(path)
        if not payload.get("episodes"):
            continue
        scene_key = scene_key_from_id(payload["episodes"][0]["scene_id"])
        result[scene_key] = payload
    return result
