import unittest

import numpy as np
import torch

from multi_iin_v0.runtime.gaussian_semantic_labels import (
    GaussianLabelConfig,
    GaussianSemanticInstanceTrainer,
    _normalized_instances,
    _semantic_summary,
)


class _Gaussians:
    def __init__(self):
        self._xyz = torch.tensor(
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [2.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

    @property
    def get_xyz(self):
        return self._xyz

    def prune_points(self, mask):
        self._xyz = self._xyz[~mask]

    def densification_postfix(self, new_xyz, *args, **kwargs):
        self._xyz = torch.cat([self._xyz, new_xyz], dim=0)


class _Camera:
    world_view_transform = torch.eye(4)
    fx = 1.0
    fy = 1.0
    cx = 1.0
    cy = 1.0
    image_height = 4
    image_width = 4


class _Slam:
    def __init__(self):
        self.gaussians = _Gaussians()
        self.frame_count = 1
        self.keyframes = {1: _Camera()}


class _Mapper:
    def __init__(self):
        self.slam = _Slam()
        self.output_dir = "/tmp/multi_iin_gaussian_label_test"

    def save_slam_results(self, *args, **kwargs):
        return None


class GaussianSemanticLabelsTest(unittest.TestCase):
    def test_detector_mask_trains_class_and_persistent_instance(self):
        mapper = _Mapper()
        trainer = GaussianSemanticInstanceTrainer(
            GaussianLabelConfig(min_confidence=0.5), None
        )
        trainer.configure_mapper(mapper)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 1
        stats = trainer.observe(
            mapper,
            [{"category_id": 2, "confidence": 0.9, "binary_mask": mask}],
            np.ones((4, 4), dtype=np.float32),
            env_step=1,
            stage="exploration",
        )
        self.assertEqual(stats["updated"], 2)
        gaussians = mapper.slam.gaussians
        labels, confidence = _semantic_summary(
            gaussians._multi_semantic_evidence,
            gaussians._multi_semantic_observations,
        )
        self.assertEqual(labels.tolist(), [2, 2, -1])
        self.assertGreater(float(confidence[0]), 0.99)
        self.assertEqual(gaussians._multi_instance_ids[0].item(), 1)
        self.assertEqual(gaussians._multi_instance_ids[1].item(), 1)

    def test_sidecars_follow_prune_and_growth(self):
        mapper = _Mapper()
        trainer = GaussianSemanticInstanceTrainer(GaussianLabelConfig(), None)
        trainer.configure_mapper(mapper)
        gaussians = mapper.slam.gaussians
        gaussians._multi_instance_ids[:] = torch.tensor([10, 20, 30])
        gaussians.prune_points(torch.tensor([False, True, False]))
        self.assertEqual(gaussians._multi_instance_ids.tolist(), [10, 30])
        gaussians.densification_postfix(torch.tensor([[3.0, 0.0, 1.0]]))
        self.assertEqual(gaussians._multi_instance_ids.tolist(), [10, 30, 0])
        self.assertEqual(gaussians._multi_semantic_evidence.shape, (3, 6))

    def test_only_six_class_confident_detector_masks_are_used(self):
        mask = np.ones((2, 2), dtype=np.uint8)
        result = _normalized_instances(
            [
                {"category_id": 0, "confidence": 0.8, "binary_mask": mask},
                {"category_id": 6, "confidence": 0.9, "binary_mask": mask},
                {"category_id": 1, "confidence": 0.2, "binary_mask": mask},
            ],
            height=4,
            width=4,
            class_count=6,
            min_confidence=0.5,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["category_id"], 0)
        self.assertEqual(result[0]["mask"].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()

