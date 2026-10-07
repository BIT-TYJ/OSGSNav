import logging
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from arguments import get_args
from ieve_minav.localizer.localization_system import (
    LocalizationState,
    LocalizationSystem,
)
from ieve_minav.manager import MinAVConfirmationLocalizationManager
from ieve_minav.strategy.strategy_transition import (
    StrategyTransitionConfig,
    StrategyTransitionManager,
)


class AblationArgumentTest(unittest.TestCase):
    def test_canonical_switches_and_legacy_gaussmi_alias(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "prog",
                "--no_cuda",
                "--3dgs_semantic_guided_optimization_enabled",
                "0",
                "--gaussmi_exploration_enabled",
                "0",
                "--minav_fine_localization_enabled",
                "0",
                "--minav_secondary_exploration_enabled",
                "0",
            ],
        ):
            args = get_args()
        self.assertEqual(args.gs_semantic_guided_optimization_enabled, 0)
        self.assertEqual(args.phase1_gaussmi_enabled, 0)
        self.assertEqual(args.minav_fine_localization_enabled, 0)
        self.assertEqual(args.minav_secondary_exploration_enabled, 0)
        self.assertEqual(args.minav_enable_strategy_transition, 0)

        with mock.patch.object(
            sys,
            "argv",
            ["prog", "--no_cuda", "--phase1_gaussmi_enabled", "0"],
        ):
            legacy_args = get_args()
        self.assertEqual(legacy_args.phase1_gaussmi_enabled, 0)

        with mock.patch.object(
            sys,
            "argv",
            ["prog", "--no_cuda", "--minav_enable_strategy_transition", "0"],
        ):
            legacy_strategy_args = get_args()
        self.assertEqual(
            legacy_strategy_args.minav_secondary_exploration_enabled, 0
        )
        self.assertEqual(legacy_strategy_args.minav_enable_strategy_transition, 0)


class SecondaryExplorationAblationTest(unittest.TestCase):
    def test_disabled_switch_never_enters_phase2_or_phase3(self):
        controller = StrategyTransitionManager(
            StrategyTransitionConfig(
                enabled=False,
                phase2_start_step=500,
                phase3_start_step=800,
            )
        )

        self.assertEqual(controller.desired_phase(499), 1)
        self.assertEqual(controller.desired_phase(500), 1)
        self.assertEqual(controller.desired_phase(800), 1)
        self.assertEqual(controller.desired_phase(999), 1)


class FineLocalizationAblationTest(unittest.TestCase):
    def test_full_localizer_finishes_directly_from_fem(self):
        system = LocalizationSystem.__new__(LocalizationSystem)
        system.logger = logging.getLogger(__name__)
        system.rendered_images = [{"instance_center_idx": 0}]
        system.fem = SimpleNamespace(
            match_rendered_views=lambda *_args, **_kwargs: {
                "match_success": True,
                "best_pose": np.eye(4, dtype=np.float64),
                "best_score": 42.0,
                "best_index": 0,
            }
        )
        system.current_object = {
            "id": 1,
            "position_3d": np.asarray([1.0, 0.0, 2.0], dtype=np.float32),
        }
        system.current_state = LocalizationState.FEM
        system.config = {"match_threshold": 20.0}
        system.fine_localization_enabled = False
        system.best_pose = None
        system.fem_best_pose = None
        system.refined_pose = None
        system._print_lightglue_match_details = lambda _result: None
        system._save_fem_best_match_visualization = lambda _result: None
        system._make_localization_complete_result = lambda message: {
            "status": "localization_complete",
            "message": message,
        }

        result = system._execute_fem()

        self.assertEqual(result["status"], "localization_complete")
        self.assertIn("PRE/REF/MR disabled", result["message"])
        self.assertEqual(result["best_score"], 42.0)
        np.testing.assert_array_equal(system.best_pose, np.eye(4))
        self.assertNotEqual(system.current_state, LocalizationState.PRE)

    def test_simplified_localizer_stops_after_fem(self):
        manager = MinAVConfirmationLocalizationManager.__new__(
            MinAVConfirmationLocalizationManager
        )
        manager.use_full_localizer = False
        manager.fine_localization_enabled = False
        manager.confirmed_target_position = np.asarray(
            [1.0, 0.0, 2.0], dtype=np.float32
        )
        manager.target_image = np.zeros((4, 4, 3), dtype=np.uint8)
        manager.localization_match_threshold = 20.0
        manager.localization_trace = []
        manager._match_with_evidence_views = lambda *_args, **_kwargs: {
            "available": True,
            "score": 42.0,
            "match_count": 12,
            "source": "current_rgb",
        }
        manager._confirmed_cluster_navigation_position = lambda: np.asarray(
            [1.2, 0.0, 2.2], dtype=np.float32
        )

        result = manager.run_close_localization(
            {"raw_rgb": np.zeros((4, 4, 3), dtype=np.uint8), "step": 5}
        )

        self.assertTrue(result["localized"])
        self.assertEqual(result["method"], "soim_nvsm_fem_coarse_v1")
        stage_names = [item["stage"] for item in result["stages"]]
        self.assertEqual(stage_names, ["SOIM", "NVSM", "FEM"])
        self.assertFalse(any(stage in stage_names for stage in ("PRE", "REF", "MR")))


if __name__ == "__main__":
    unittest.main()
