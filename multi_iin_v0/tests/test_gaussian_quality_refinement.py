from types import SimpleNamespace

import numpy as np
import torch

from multi_iin_v0.runtime.gaussian_quality_refinement import (
    GaussianQualityRefinementConfig,
    GaussianQualityRefinementManager,
    current_render_quality,
    refinement_is_acceptable,
    refinement_regressions,
    should_refine,
)


def test_quality_gate_detects_black_hole_without_semantic_inputs():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    rendered = rgb.copy()
    rendered[:, :10] = 0
    metrics = current_render_quality(rgb, rendered)
    assert metrics["excess_black_fraction"] == 0.5
    assert should_refine(metrics, GaussianQualityRefinementConfig())


def test_quality_gate_ignores_a_good_render():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    metrics = current_render_quality(rgb, rgb.copy())
    assert not should_refine(metrics, GaussianQualityRefinementConfig())


def test_refinement_acceptance_rejects_observable_regressions():
    config = GaussianQualityRefinementConfig()
    before = {
        "visible_fraction": 1.0,
        "excess_black_fraction": 0.02,
        "strong_darkening_fraction": 0.10,
        "psnr_db": 16.0,
    }
    improved = dict(
        before,
        excess_black_fraction=0.005,
        strong_darkening_fraction=0.04,
        psnr_db=17.0,
    )
    regressed = dict(
        before,
        excess_black_fraction=0.03,
        strong_darkening_fraction=0.15,
        psnr_db=15.0,
    )
    assert refinement_is_acceptable(before, improved, config)
    assert set(refinement_regressions(before, regressed, config)) == {
        "psnr_drop",
        "excess_black_increase",
        "strong_darkening_increase",
    }


class _FakeGaussians:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.optim.Adam(
            [{"params": [self.parameter], "name": "xyz", "lr": 0.1}],
            lr=0.0,
        )
        self.max_radii2D = torch.tensor([2.0])
        self.xyz_gradient_accum = torch.tensor([[3.0]])
        self.denom = torch.tensor([[4.0]])


class _FakeSLAM:
    def __init__(self):
        self.kf_indices = []
        self.train_calls = []
        self.rendered = np.zeros((20, 20, 3), dtype=np.uint8)
        self.gaussians = _FakeGaussians()
        self.iteration_count = 10

    def add_frame(self, rgb, depth, position, rotation, **kwargs):
        self.kf_indices.append(len(self.kf_indices) + 1)
        return True

    def render_frame(self, kf_id):
        return {"rendered": self.rendered}

    def _train_mapping(self, **kwargs):
        self.train_calls.append(kwargs)
        self.gaussians.parameter.data.add_(5.0)
        self.iteration_count += int(kwargs["override_iterations"])
        self.rendered[:] = 128


def test_manager_refines_only_the_current_keyframe():
    slam = _FakeSLAM()
    mapper = SimpleNamespace(slam=slam)
    manager = GaussianQualityRefinementManager(
        GaussianQualityRefinementConfig(extra_current_frame_iters=6), None
    )
    manager.configure_mapper(mapper)
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    slam.add_frame(
        rgb,
        np.ones((20, 20), dtype=np.float32),
        np.zeros(3),
        np.array([0, 0, 0, 1]),
        training_context={"stage": "exploration", "step": 7},
    )
    assert manager.trigger_count == 1
    assert slam.train_calls == [
        {
            "override_iterations": 6,
            "preferred_kf_ids": [1],
            "allow_topology_growth": False,
        }
    ]
    assert manager.last_event["after"]["excess_black_fraction"] == 0.0
    assert manager.last_event["accepted"] is True
    assert manager.last_event["rolled_back"] is False
    assert manager.snapshot()["checks"] == 1
    assert manager.snapshot()["triggers"] == 1
    assert manager.snapshot()["accepted"] == 1


class _RegressingFakeSLAM(_FakeSLAM):
    def __init__(self):
        super().__init__()
        self.rendered[:] = 128
        self.rendered[:, :10] = 0

    def _train_mapping(self, **kwargs):
        self.train_calls.append(kwargs)
        self.gaussians.parameter.data.add_(5.0)
        self.gaussians.max_radii2D.add_(7.0)
        self.iteration_count += int(kwargs["override_iterations"])
        self.rendered[:] = 0

    def render_frame(self, kf_id):
        # A real rollback changes the render through restored Gaussian values.
        if float(self.gaussians.parameter.detach()) == 1.0:
            restored = np.full((20, 20, 3), 128, dtype=np.uint8)
            restored[:, :10] = 0
            return {"rendered": restored}
        return {"rendered": self.rendered}


def test_manager_rolls_back_a_regressing_refinement():
    slam = _RegressingFakeSLAM()
    mapper = SimpleNamespace(slam=slam)
    manager = GaussianQualityRefinementManager(
        GaussianQualityRefinementConfig(extra_current_frame_iters=6), None
    )
    manager.configure_mapper(mapper)
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    slam.add_frame(
        rgb,
        np.ones((20, 20), dtype=np.float32),
        np.zeros(3),
        np.array([0, 0, 0, 1]),
        training_context={"stage": "exploration", "step": 9},
    )
    assert float(slam.gaussians.parameter.detach()) == 1.0
    assert float(slam.gaussians.max_radii2D) == 2.0
    assert slam.iteration_count == 10
    assert manager.last_event["accepted"] is False
    assert manager.last_event["rolled_back"] is True
    assert "current:psnr_drop" in manager.last_event["rejection_reasons"]
    assert manager.snapshot()["rejected"] == 1
    assert manager.snapshot()["rollbacks"] == 1


class _Camera:
    def __init__(self, value=128):
        self.original_image = torch.full((3, 20, 20), value / 255.0)


class _HistoryRegressingFakeSLAM(_FakeSLAM):
    def __init__(self):
        super().__init__()
        self.kf_indices = [1, 2]
        self.keyframes = {1: _Camera(), 2: _Camera()}
        self.current_id = None

    def add_frame(self, rgb, depth, position, rotation, **kwargs):
        self.current_id = 3
        self.kf_indices.append(self.current_id)
        return True

    def render_frame(self, kf_id):
        changed = float(self.gaussians.parameter.detach()) != 1.0
        if kf_id == self.current_id:
            image = np.full((20, 20, 3), 128, dtype=np.uint8)
            if not changed:
                image[:, :10] = 0
            return {"rendered": image}
        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        if changed:
            image[:] = 0
        return {"rendered": image}

    def _train_mapping(self, **kwargs):
        self.train_calls.append(kwargs)
        self.gaussians.parameter.data.add_(5.0)
        self.iteration_count += int(kwargs["override_iterations"])


def test_manager_rolls_back_when_a_recent_held_out_view_regresses():
    slam = _HistoryRegressingFakeSLAM()
    manager = GaussianQualityRefinementManager(
        GaussianQualityRefinementConfig(
            extra_current_frame_iters=6,
            history_validation_views=2,
            history_validation_stride=1,
        ),
        None,
    )
    manager.configure_mapper(SimpleNamespace(slam=slam))
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    slam.add_frame(
        rgb,
        np.ones((20, 20), dtype=np.float32),
        np.zeros(3),
        np.array([0, 0, 0, 1]),
        training_context={"stage": "exploration", "step": 11},
    )
    assert float(slam.gaussians.parameter.detach()) == 1.0
    assert manager.last_event["rolled_back"] is True
    assert manager.last_event["validation_history_frames"] == [2, 1]
    assert any(
        reason.startswith("history_")
        for reason in manager.last_event["rejection_reasons"]
    )
