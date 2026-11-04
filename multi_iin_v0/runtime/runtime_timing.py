"""Small in-memory timing summary for Multi-IIN runtime validation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np


class RuntimeTimingRecorder:
    def __init__(self):
        self._all = defaultdict(list)
        self._ordinary = defaultdict(list)

    def record_frame(self, sample: Dict[str, float], terminal: bool = False) -> None:
        clean = {
            str(key): float(value)
            for key, value in sample.items()
            if value is not None and np.isfinite(float(value))
        }
        for key, value in clean.items():
            self._all[key].append(value)
        # A normal exploration/map-update frame excludes virtual-view scoring,
        # localization, and terminal archiving.  The 2 s gates tolerate normal
        # planner variation while excluding the minute-scale calls of interest.
        ordinary = (
            not terminal
            and clean.get("gaussmi_s", 0.0) < 2.0
            and clean.get("goal_override_localization_s", 0.0) < 2.0
        )
        if ordinary:
            for key, value in clean.items():
                self._ordinary[key].append(value)

    @staticmethod
    def _stats(values):
        data = np.asarray(values, dtype=np.float64)
        if data.size == 0:
            return {
                "count": 0,
                "total_s": None,
                "mean_s": None,
                "median_s": None,
                "p90_s": None,
                "max_s": None,
            }
        return {
            "count": int(data.size),
            "total_s": float(np.sum(data)),
            "mean_s": float(np.mean(data)),
            "median_s": float(np.median(data)),
            "p90_s": float(np.percentile(data, 90)),
            "max_s": float(np.max(data)),
        }

    def summary(self):
        return {
            "all_frames": {
                key: self._stats(values) for key, values in sorted(self._all.items())
            },
            "ordinary_frames": {
                key: self._stats(values)
                for key, values in sorted(self._ordinary.items())
            },
            "ordinary_definition": (
                "non-terminal frame with GaussMI and localization/goal-override "
                "each below 2 seconds"
            ),
        }
