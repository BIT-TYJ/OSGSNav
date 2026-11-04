"""
IEVE StrategyTransition adapter.

This module is intentionally small: the current IEVE integration keeps map
projection, FMM goal preparation, action execution, target confirmation, and
localization inside IEVEMinAVManager. StrategyTransition owns only the phase
schedule and ranking policy so it has no dependency on the original external
MinAV runtime modules.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np


PHASE_EXPLORATION = 1
PHASE_INSTANCE_VISIT = 2
PHASE_BEST_EVIDENCE = 3


@dataclass
class StrategyTransitionConfig:
    enabled: bool = True
    phase2_start_step: int = 500
    phase3_start_step: int = 800
    phase3_fallback_stop_radius: float = 0.5
    use_area_growth_trigger: bool = False
    sample_interval: int = 5
    window_samples: int = 11
    map_growth_rate_threshold: float = 0.01

    def __post_init__(self) -> None:
        self.phase2_start_step = int(self.phase2_start_step)
        self.phase3_start_step = int(self.phase3_start_step)
        if self.phase3_start_step < self.phase2_start_step:
            self.phase3_start_step = self.phase2_start_step
        self.phase3_fallback_stop_radius = max(
            0.0, float(self.phase3_fallback_stop_radius)
        )
        self.sample_interval = max(1, int(self.sample_interval))
        self.window_samples = max(2, int(self.window_samples))
        self.map_growth_rate_threshold = float(self.map_growth_rate_threshold)


class MapAreaTracker:
    """Optional legacy map-growth tracker kept for debug/ablation."""

    def __init__(
        self,
        sample_interval: int = 5,
        window_samples: int = 11,
        growth_threshold: float = 0.01,
    ) -> None:
        self.sample_interval = max(1, int(sample_interval))
        self.window_samples = max(2, int(window_samples))
        self.growth_threshold = float(growth_threshold)
        self.reset()

    def reset(self) -> None:
        self.completion_samples: Deque[int] = deque(maxlen=self.window_samples)
        self.completion_frame_counter = 0
        self.revert_samples: Deque[int] = deque(maxlen=2)
        self.revert_frame_counter = 0
        self.exploration_complete = False
        self.last_completion_growth: Optional[float] = None
        self.last_revert_growth: Optional[float] = None

    def reset_revert_tracking(self) -> None:
        self.revert_samples.clear()
        self.revert_frame_counter = 0
        self.last_revert_growth = None

    def mark_exploration_incomplete(self) -> None:
        self.exploration_complete = False
        self.completion_samples.clear()
        self.completion_frame_counter = 0
        self.last_completion_growth = None
        self.reset_revert_tracking()

    def _sample_area(
        self,
        explored_map: np.ndarray,
        frame_counter_attr: str,
        samples: Deque[int],
    ) -> Optional[int]:
        frame_counter = int(getattr(self, frame_counter_attr))
        if frame_counter % self.sample_interval != 0:
            setattr(self, frame_counter_attr, frame_counter + 1)
            return None
        setattr(self, frame_counter_attr, frame_counter + 1)
        area = int(np.sum(np.asarray(explored_map) > 0.5))
        samples.append(area)
        return area

    def update(
        self,
        explored_map: np.ndarray,
        *,
        track_completion: bool = False,
        track_revert: bool = False,
    ) -> None:
        if track_completion:
            self._sample_area(
                explored_map,
                "completion_frame_counter",
                self.completion_samples,
            )
            if len(self.completion_samples) == self.window_samples:
                oldest = int(self.completion_samples[0])
                newest = int(self.completion_samples[-1])
                growth = ((newest - oldest) / oldest) if oldest > 0 else 1.0
                self.last_completion_growth = float(growth)
                if growth < self.growth_threshold:
                    self.exploration_complete = True

        if track_revert:
            self._sample_area(
                explored_map,
                "revert_frame_counter",
                self.revert_samples,
            )

    def should_revert(self) -> bool:
        if len(self.revert_samples) < 2:
            return False
        prev_area = int(self.revert_samples[-2])
        cur_area = int(self.revert_samples[-1])
        if prev_area <= 0:
            return False
        growth = (cur_area - prev_area) / prev_area
        self.last_revert_growth = float(growth)
        if growth >= self.growth_threshold:
            self.mark_exploration_incomplete()
            return True
        return False


class StrategyTransitionManager:
    """Fixed-step phase scheduler and candidate-ranking policy for IEVE."""

    def __init__(self, config: Optional[StrategyTransitionConfig] = None) -> None:
        self.config = config or StrategyTransitionConfig()
        self.area_tracker = MapAreaTracker(
            sample_interval=self.config.sample_interval,
            window_samples=self.config.window_samples,
            growth_threshold=self.config.map_growth_rate_threshold,
        )
        self.reset()

    def reset(self) -> None:
        self.area_tracker.reset()
        self.last_desired_phase = PHASE_EXPLORATION

    def desired_phase(self, step: int, *, current_phase: int = PHASE_EXPLORATION) -> int:
        if not self.config.enabled:
            self.last_desired_phase = PHASE_EXPLORATION
            return PHASE_EXPLORATION
        if self.config.use_area_growth_trigger:
            phase = int(current_phase)
        elif int(step) < self.config.phase2_start_step:
            phase = PHASE_EXPLORATION
        elif int(step) < self.config.phase3_start_step:
            phase = PHASE_INSTANCE_VISIT
        else:
            phase = PHASE_BEST_EVIDENCE
        self.last_desired_phase = phase
        return phase

    def phase_reason(self, phase: int) -> str:
        if int(phase) >= PHASE_BEST_EVIDENCE:
            return "fixed_step_phase3"
        if int(phase) >= PHASE_INSTANCE_VISIT:
            return "fixed_step_phase2"
        return "fixed_step_phase1"

    def phase3_active(
        self,
        step: int,
        *,
        current_phase: int = PHASE_EXPLORATION,
    ) -> bool:
        if not self.config.enabled:
            return False
        if self.config.use_area_growth_trigger:
            return int(current_phase) >= PHASE_BEST_EVIDENCE
        return int(step) >= self.config.phase3_start_step

    def should_enter_targeted(
        self,
        step: int,
        *,
        current_phase: int = PHASE_EXPLORATION,
        has_confirmed_target: bool = False,
        has_localized_target: bool = False,
    ) -> bool:
        if not self.config.enabled:
            return False
        if has_confirmed_target or has_localized_target:
            return False
        if self.config.use_area_growth_trigger:
            return bool(self.area_tracker.exploration_complete)
        return (
            self.desired_phase(step, current_phase=current_phase)
            >= PHASE_INSTANCE_VISIT
        )

    @staticmethod
    def phase2_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, int]:
        return (
            float(candidate.get("cost", float("inf"))),
            -float(candidate.get("area", 0) or 0),
            int(candidate.get("obj_id", 0) or 0),
        )

    @staticmethod
    def phase3_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float]:
        return (
            float(candidate.get("historical_score", 0.0) or 0.0),
            -float(candidate.get("cost", float("inf"))),
        )

    def should_force_stop_phase3(
        self,
        step: int,
        *,
        current_phase: int,
        best_distance: Optional[float],
    ) -> bool:
        if best_distance is None or not np.isfinite(float(best_distance)):
            return False
        if not self.phase3_active(step, current_phase=current_phase):
            return False
        return float(best_distance) <= self.config.phase3_fallback_stop_radius
