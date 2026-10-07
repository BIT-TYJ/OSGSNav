import json
import logging
import time

import numpy as np


logger = logging.getLogger(__name__)


def _normalize_angle_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


class Phase1GaussMIController:
    IDLE = "idle"
    TURN_TO = "turn_to"
    RETURN = "return"

    def __init__(self, args, env_idx):
        self.args = args
        self.env_idx = int(env_idx)
        self.enabled = bool(getattr(args, "phase1_gaussmi_enabled", 1))
        self.trigger_steps = max(
            1, int(getattr(args, "phase1_gaussmi_trigger_move_steps", 5))
        )
        self.move_eps_m = max(
            0.0, float(getattr(args, "phase1_gaussmi_move_eps_m", 0.03))
        )
        self.yaw_tolerance_deg = max(
            1.0, float(getattr(args, "phase1_gaussmi_yaw_tolerance_deg", 5.0))
        )
        self.yaw_offsets = np.asarray(
            getattr(
                args,
                "phase1_gaussmi_yaw_offsets",
                [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0],
            ),
            dtype=np.float32,
        ).reshape(-1)
        self.debug = bool(getattr(args, "phase1_gaussmi_debug", 1))
        self.reset()

    def reset(self, info=None):
        self.state = self.IDLE
        self.anchor_position = None
        self.anchor_yaw_deg = None
        self.saved_long_term_goal = None
        self.target_yaw_deg = None
        self.selected_offset_deg = None
        self.last_command = None
        self.last_position = self._position(info)
        self.movement_steps = 0
        self.movement_steps_since_trigger = 0
        self.last_score = None
        self._training_skip = False
        self._training_force = False

    @staticmethod
    def _position(info):
        if not info:
            return None
        position = info.get("agent_position")
        if position is None:
            return None
        position = np.asarray(position, dtype=np.float32).reshape(-1)
        if position.size < 3 or not np.all(np.isfinite(position[:3])):
            return None
        return position[:3].copy()

    def cancel(self, reason, step=None):
        if self.state != self.IDLE:
            self._log(
                "cancel",
                step=step,
                reason=str(reason),
                state=self.state,
            )
        self.state = self.IDLE
        self.anchor_position = None
        self.anchor_yaw_deg = None
        self.saved_long_term_goal = None
        self.target_yaw_deg = None
        self.selected_offset_deg = None
        self.last_command = None

    def observe(
        self,
        info,
        current_yaw_deg,
        eligible,
        step,
        movement_eligible=None,
    ):
        """Consume the observation produced by the previous action."""

        previous_command = self.last_command
        self._training_skip = False
        self._training_force = False

        position = self._position(info)
        if position is not None and self.last_position is not None:
            moved = float(
                np.linalg.norm(position[[0, 2]] - self.last_position[[0, 2]])
            )
            if (
                previous_command is None
                and bool(eligible if movement_eligible is None else movement_eligible)
                and moved >= self.move_eps_m
            ):
                self.movement_steps += 1
                self.movement_steps_since_trigger += 1
        if position is not None:
            self.last_position = position

        if not eligible:
            self.cancel("phase1_ineligible", step=step)
            return

        yaw = _normalize_angle_deg(current_yaw_deg)
        if previous_command == self.TURN_TO and self.target_yaw_deg is not None:
            arrived = (
                abs(_normalize_angle_deg(self.target_yaw_deg - yaw))
                <= self.yaw_tolerance_deg
            )
            if arrived:
                # The post-turn observation at the selected yaw is already a
                # stable frame. Train it directly instead of spending a hold.
                self._training_force = True
                self.state = self.RETURN
            else:
                self._training_skip = True
        elif previous_command == self.RETURN and self.anchor_yaw_deg is not None:
            self._training_skip = True
            if abs(_normalize_angle_deg(self.anchor_yaw_deg - yaw)) <= self.yaw_tolerance_deg:
                self._log(
                    "complete",
                    step=step,
                    selected_offset_deg=self.selected_offset_deg,
                )
                self.state = self.IDLE
                self.anchor_position = None
                self.anchor_yaw_deg = None
                self.saved_long_term_goal = None
                self.target_yaw_deg = None
                self.selected_offset_deg = None
        self.last_command = None

    def training_directive(self):
        return bool(self._training_skip), bool(self._training_force)

    def maybe_start(
        self,
        mapper,
        info,
        current_yaw_deg,
        eligible,
        step,
        long_term_goal=None,
    ):
        if not self.enabled or not eligible or self.state != self.IDLE:
            if not eligible:
                self.cancel("phase1_ineligible", step=step)
            return None
        if self.movement_steps_since_trigger < self.trigger_steps:
            return None
        self.movement_steps_since_trigger = 0

        if mapper is None:
            self._log("skip", step=step, reason="mapper_unavailable")
            return None
        position = info.get("camera_position", info.get("agent_position")) if info else None
        rotation = info.get("agent_rotation_xyzw") if info else None
        if position is None or rotation is None:
            self._log("skip", step=step, reason="pose_unavailable")
            return None

        started = time.perf_counter()
        result = mapper.score_phase1_gaussmi_yaws(
            position,
            rotation,
            self.yaw_offsets,
            angle_penalty_k=float(
                getattr(self.args, "phase1_gaussmi_angle_penalty_k", 0.5)
            ),
            theta_max_deg=float(
                getattr(self.args, "phase1_gaussmi_theta_max_deg", 90.0)
            ),
            target_weight=float(
                getattr(self.args, "phase1_gaussmi_target_weight", 2.0)
            ),
            nearby_weight=float(
                getattr(self.args, "phase1_gaussmi_nearby_weight", 1.5)
            ),
            nearby_radius_m=float(
                getattr(self.args, "phase1_gaussmi_nearby_radius_m", 2.0)
            ),
            min_frames=int(
                getattr(self.args, "phase1_gaussmi_min_frames", 5)
            ),
            min_gaussians=int(
                getattr(self.args, "phase1_gaussmi_min_gaussians", 10000)
            ),
            reliability_delta=float(
                getattr(self.args, "phase1_gaussmi_reliability_delta", 0.02)
            ),
            min_updated_ratio=float(
                getattr(self.args, "phase1_gaussmi_min_updated_ratio", 0.10)
            ),
            min_valid_candidates=int(
                getattr(self.args, "phase1_gaussmi_min_valid_candidates", 4)
            ),
            min_info_gain=float(
                getattr(self.args, "phase1_gaussmi_min_info_gain", 0.0)
            ),
            min_relative_gain=float(
                getattr(self.args, "phase1_gaussmi_min_relative_gain", 0.0)
            ),
        )
        result = dict(result or {})
        result["elapsed_ms"] = round(
            (time.perf_counter() - started) * 1000.0, 3
        )
        self.last_score = result

        selected = result.get("selected_offset_deg")
        if not result.get("ready", False) or selected is None:
            self._log("skip", step=step, **result)
            return result
        if not result.get("should_turn", False):
            self._log("score", step=step, action="stay", **result)
            return result

        self.anchor_position = np.asarray(position, dtype=np.float32).reshape(-1)[
            :3
        ].copy()
        self.anchor_yaw_deg = _normalize_angle_deg(current_yaw_deg)
        self.saved_long_term_goal = (
            None
            if long_term_goal is None
            else np.asarray(long_term_goal).copy()
        )
        self.selected_offset_deg = float(selected)
        self.target_yaw_deg = _normalize_angle_deg(
            self.anchor_yaw_deg + self.selected_offset_deg
        )
        self.state = self.TURN_TO
        self._log(
            "score",
            step=step,
            action="turn",
            anchor_position=self.anchor_position.tolist(),
            anchor_yaw_deg=self.anchor_yaw_deg,
            target_yaw_deg=self.target_yaw_deg,
            saved_long_term_goal=(
                None
                if self.saved_long_term_goal is None
                else self.saved_long_term_goal.tolist()
            ),
            **result,
        )
        return result

    def apply_action_override(self, planner_input, current_yaw_deg, step):
        if self.state == self.IDLE:
            return False
        if (
            planner_input.get("force_stop", False)
            or planner_input.get("hold_action", False)
            or planner_input.get("action_override") is not None
        ):
            self.cancel("higher_priority_action", step=step)
            return False

        desired = (
            self.target_yaw_deg
            if self.state == self.TURN_TO
            else self.anchor_yaw_deg
        )
        error = _normalize_angle_deg(float(desired) - float(current_yaw_deg))
        if abs(error) <= self.yaw_tolerance_deg:
            if self.state == self.TURN_TO:
                self.state = self.RETURN
                return False
            else:
                self.state = self.IDLE
                return False
        else:
            action = "turn_left" if error > 0.0 else "turn_right"

        planner_input["action_override"] = action
        planner_input["goal_source"] = "PHASE1_GAUSSMI_GAZE"
        planner_input["skip_instance_discriminator"] = True
        planner_input["found_goal"] = 0
        self.last_command = self.state
        return True

    def _log(self, event, **payload):
        if not self.debug:
            return
        data = {
            "event": str(event),
            "env": self.env_idx,
            "state": self.state,
            "movement_steps": int(self.movement_steps),
            "movement_steps_since_trigger": int(
                self.movement_steps_since_trigger
            ),
        }
        data.update(payload)
        line = "[PHASE1_GAUSSMI] {}".format(
            json.dumps(data, sort_keys=True, default=str)
        )
        print(line)
        logger.info(line)
