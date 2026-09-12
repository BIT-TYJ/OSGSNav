import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from multi_iin_v0.runtime.result_organizer import (
    CompactResultOrganizer,
    ensure_terminal_match_source,
)


def _write_png(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(str(path))


class CompactResultOrganizerTest(unittest.TestCase):
    def _args(self, root: Path, variant="Continuous-Shared"):
        return SimpleNamespace(
            multi_save_compact_results=1,
            multi_variant=variant,
            multi_compact_results_dir=str(root / "compact"),
            multi_episode_start=0,
            multi_num_episodes=1,
            multi_goal_step_budget=1000,
            multi_manifest="multi_iin_v0/manifests/miin_v0_val_k2.json",
        )

    def test_human_readable_layout_and_minimal_terminal_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "verbose_minav"
            mapper = root / "verbose_mapper"
            _write_png(source / "target_image.png", (1, 2, 3))
            _write_png(source / "goal_image_analysis_result.png", (4, 5, 6))
            (mapper / "final").mkdir(parents=True)
            (mapper / "final" / "point_cloud.ply").write_text("gaussian", encoding="utf-8")
            gaussian_preview = mapper / "terminal_preview" / "target_01" / "comparison_00321.png"
            _write_png(gaussian_preview, (8, 9, 10))

            nvsm = source / "NVSM"
            nvsm.mkdir(parents=True)
            instance = nvsm / "instance_semantic_20260904_120000_filtered.ply"
            instance.write_text("instance", encoding="utf-8")
            stats = nvsm / "instance_semantic_20260904_120000_filter_stats.json"
            stats.write_text('{"filtered": {"count": 123}}', encoding="utf-8")
            projection = nvsm / "instance_semantic_20260904_120000_filter_projection.png"
            _write_png(projection, (7, 8, 9))

            # A higher historical score must not replace the later terminal attempt.
            for step, score, color in ((10, 99.0, (10, 0, 0)), (11, 12.0, (11, 0, 0))):
                base = "best_match_step_{:06d}_score_{:07.2f}_view".format(step, score)
                render = nvsm / (base + ".png")
                matches = nvsm / (base + "_matches.png")
                metadata = nvsm / (base + ".json")
                _write_png(render, color)
                _write_png(matches, (0, step, 0))
                metadata.write_text(
                    json.dumps(
                        {
                            "step": step,
                            "score": score,
                            "match_count": step,
                            "image_path": str(render),
                            "match_visualization_path": str(matches),
                        }
                    ),
                    encoding="utf-8",
                )
            noisy = source / "target_confirmation" / "failed_intermediate.png"
            _write_png(noisy, (255, 0, 0))

            layers = np.zeros((8, 40, 50), dtype=np.float32)
            layers[1, 10:30, 12:38] = 1
            layers[0, 10:12, 12:38] = 1
            layers[3, 18:22, 20:30] = 1
            layers[2, 21, 29] = 1
            layers[5, 14:17, 15:19] = 1

            organizer = CompactResultOrganizer(self._args(root), root)
            target_dir = organizer.archive_target(
                {
                    "episode_no": 1,
                    "scene_name": "/scenes/wcojb4TFT35.basis.glb",
                    "multi_episode_id": "miin_v0_val_000000",
                },
                {
                    "target_index": 0,
                    "goal_name": "toilet",
                    "goal_object_id": "270",
                    "success": 1.0,
                    "steps": 321,
                },
                layers,
                str(mapper),
                str(source),
                terminal_localization={"best_score": 12.0},
                gaussian_preview_path=str(gaussian_preview),
                gaussian_quality={"psnr": 21.5, "ssim": 0.72},
            )

            self.assertIn("continuous_shared", str(target_dir))
            self.assertIn("episode_00001__wcojb4tft35__miin_v0_val_000000", str(target_dir))
            self.assertTrue(target_dir.name.endswith("target_01__toilet__obj270__success"))
            expected = (
                "01_goal/target_image.png",
                "01_goal/target_detector_analysis.png",
                "02_map/final_2d_map.png",
                "03_gaussian_map/final_gaussian_map.ply",
                "03_gaussian_map/final_gaussian_rgb_comparison.png",
                "04_target_instance/target_instance_filtered.ply",
                "04_target_instance/filter_stats.json",
                "04_target_instance/filter_projection.png",
                "05_final_match/selected_render.png",
                "05_final_match/target_render_matches.png",
                "05_final_match/match_metrics.json",
                "target_summary.json",
            )
            for relative in expected:
                self.assertTrue((target_dir / relative).is_file(), relative)
            self.assertFalse(any(path.name == noisy.name for path in target_dir.rglob("*")))
            with (target_dir / "05_final_match" / "match_metrics.json").open() as handle:
                self.assertEqual(json.load(handle)["step"], 11)
            with (target_dir / "target_summary.json").open() as handle:
                self.assertEqual(json.load(handle)["missing_artifacts"], [])

    def test_variant_directory_names_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for variant, expected in (
                ("Independent-Reset", "independent_reset"),
                ("Continuous-Clear", "continuous_clear"),
                ("Continuous-Shared", "continuous_shared"),
            ):
                organizer = CompactResultOrganizer(self._args(root, variant), root)
                self.assertEqual(organizer.run_dir.parent.name, expected)

    def test_terminal_no_match_still_produces_auditable_goal_and_match_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "minav"
            source.mkdir()

            class Manager:
                save_stage_visualizations = True
                visualization_dir = source
                target_image = np.full((12, 16, 3), 40, dtype=np.uint8)

                def _match_with_evidence_views(self, *args, **kwargs):
                    return {"available": False}

            info = {
                "episode_no": 1,
                "scene_name": "/scenes/test.glb",
                "multi_episode_id": "miin_test",
                "instance_imagegoal": Manager.target_image,
                "raw_rgb": np.full((12, 16, 3), 90, dtype=np.uint8),
            }
            ensure_terminal_match_source(Manager(), info, 25)
            organizer = CompactResultOrganizer(self._args(root), root)
            target_dir = organizer.archive_target(
                info,
                {
                    "target_index": 0,
                    "goal_name": "chair",
                    "goal_object_id": "1",
                    "success": 0.0,
                    "steps": 25,
                },
                np.zeros((8, 10, 10), dtype=np.float32),
                None,
                str(source),
            )
            self.assertTrue((target_dir / "01_goal" / "target_image.png").is_file())
            self.assertTrue((target_dir / "05_final_match" / "selected_render.png").is_file())
            self.assertTrue((target_dir / "05_final_match" / "target_render_matches.png").is_file())


if __name__ == "__main__":
    unittest.main()
