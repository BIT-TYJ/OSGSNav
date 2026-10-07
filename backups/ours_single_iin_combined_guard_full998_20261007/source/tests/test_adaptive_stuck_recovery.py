import unittest
from types import SimpleNamespace

from agents.instance_exp import Instance_Exp_Env_Agent


class AdaptiveStuckRecoveryTest(unittest.TestCase):
    def _agent(self, enabled):
        agent = Instance_Exp_Env_Agent.__new__(Instance_Exp_Env_Agent)
        agent.args = SimpleNamespace(stuck_recovery_enabled=int(enabled))
        agent.stuck_same_stg_count = 9
        agent.stuck_last_stg = object()
        agent.stuck_last_position = object()
        return agent

    def test_switch_controls_recovery(self):
        self.assertFalse(self._agent(False)._stuck_recovery_enabled())
        self.assertTrue(self._agent(True)._stuck_recovery_enabled())

    def test_executor_modes_never_trigger_escape_actions(self):
        agent = self._agent(True)
        triggered = agent._maybe_trigger_stuck_recovery(
            planner_inputs={},
            stg=(5, 5),
            planning_window=(0, 10, 0, 10),
            minav_executor_mode=True,
        )
        self.assertFalse(triggered)
        self.assertEqual(agent.stuck_same_stg_count, 0)


if __name__ == "__main__":
    unittest.main()
