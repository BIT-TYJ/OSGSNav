from types import SimpleNamespace
import unittest

import numpy as np

from gaussian_mapping.phase1_gaussmi import Phase1GaussMIController


class _Mapper:
    def __init__(self, offset=60.0):
        self.offset = float(offset)
        self.calls = 0

    def score_phase1_gaussmi_yaws(self, *_args, **_kwargs):
        self.calls += 1
        return {
            "ready": True,
            "should_turn": True,
            "selected_offset_deg": self.offset,
            "best_info_gain": 2.0,
            "best_final_score": 1.5,
            "candidates": [],
        }


def _info(x, z=0.0):
    return {
        "agent_position": np.asarray([x, 0.0, z], dtype=np.float32),
        "camera_position": np.asarray([x, 0.8, z], dtype=np.float32),
        "agent_rotation_xyzw": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


class Phase1GaussMIControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = Phase1GaussMIController(
            SimpleNamespace(
                phase1_gaussmi_enabled=1,
                phase1_gaussmi_trigger_move_steps=5,
                phase1_gaussmi_move_eps_m=0.03,
                phase1_gaussmi_yaw_tolerance_deg=5.0,
                phase1_gaussmi_debug=0,
            ),
            env_idx=0,
        )
        self.controller.reset(_info(0.0))

    def test_five_real_moves_then_turn_selected_frame_return(self):
        for step in range(1, 6):
            self.controller.observe(
                _info(step * 0.05),
                current_yaw_deg=0.0,
                eligible=True,
                step=step,
                movement_eligible=True,
            )
        self.assertEqual(self.controller.movement_steps_since_trigger, 5)

        self.controller.maybe_start(
            _Mapper(), _info(0.25), 0.0, True, step=5
        )
        planner_input = {}
        self.assertTrue(
            self.controller.apply_action_override(planner_input, 0.0, step=5)
        )
        self.assertEqual(planner_input["action_override"], "turn_left")

        self.controller.observe(_info(0.25), 30.0, True, step=6)
        self.assertEqual(self.controller.state, self.controller.TURN_TO)
        self.assertEqual(self.controller.training_directive(), (True, False))
        planner_input = {}
        self.controller.apply_action_override(planner_input, 30.0, step=6)
        self.assertEqual(planner_input["action_override"], "turn_left")

        self.controller.observe(_info(0.25), 60.0, True, step=7)
        self.assertEqual(self.controller.state, self.controller.RETURN)
        self.assertEqual(self.controller.training_directive(), (False, True))

        planner_input = {}
        self.controller.apply_action_override(planner_input, 60.0, step=7)
        self.assertEqual(planner_input["action_override"], "turn_right")
        self.controller.observe(_info(0.25), 30.0, True, step=8)
        self.assertEqual(self.controller.state, self.controller.RETURN)
        self.assertEqual(self.controller.training_directive(), (True, False))
        planner_input = {}
        self.controller.apply_action_override(planner_input, 30.0, step=8)
        self.assertEqual(planner_input["action_override"], "turn_right")

        self.controller.observe(_info(0.25), 0.0, True, step=9)
        self.assertEqual(self.controller.state, self.controller.IDLE)
        self.assertEqual(self.controller.movement_steps, 5)

    def test_thirty_degree_target_has_no_intermediate_frame(self):
        self.controller.movement_steps_since_trigger = 5
        self.controller.maybe_start(
            _Mapper(offset=30.0), _info(0.0), 0.0, True, step=0
        )
        planner_input = {}
        self.controller.apply_action_override(planner_input, 0.0, step=0)
        self.assertEqual(planner_input["action_override"], "turn_left")

        self.controller.observe(_info(0.0), 30.0, True, step=1)
        self.assertEqual(self.controller.state, self.controller.RETURN)
        self.assertEqual(self.controller.training_directive(), (False, True))

    def test_disabled_never_calls_mapper_scoring(self):
        self.controller.enabled = False
        self.controller.movement_steps_since_trigger = 5
        mapper = _Mapper()

        result = self.controller.maybe_start(
            mapper, _info(0.0), 0.0, True, step=0
        )

        self.assertIsNone(result)
        self.assertEqual(mapper.calls, 0)
        self.assertEqual(self.controller.state, self.controller.IDLE)


if __name__ == "__main__":
    unittest.main()
