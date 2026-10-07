"""Replay-only hook that captures a PLY and aligned GauSS-MI reliability."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def _install_snapshot_hook():
    output_value = os.environ.get("IEVE_GAUSSMI_SNAPSHOT_DIR", "").strip()
    if not output_value:
        return

    from gaussian_mapping.gaussian_splatting.scene.old_gaussian_model import (
        GaussianModel,
    )

    if getattr(GaussianModel.save_ply, "_ieve_snapshot_hook", False):
        return

    output_dir = Path(output_value).resolve()
    match_text = os.environ.get(
        "IEVE_GAUSSMI_SNAPSHOT_MATCH",
        "step_000039_frame_000040",
    )
    original_save_ply = GaussianModel.save_ply
    captured = {"done": False}

    def capture(model, source_path=None, trigger="save_ply", update_result=None):
        if captured["done"]:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        ply_target = output_dir / "step40_gaussian_map.ply"
        if source_path is None:
            original_save_ply(model, str(ply_target))
            source_description = str(ply_target)
        else:
            source_path = Path(source_path)
            if source_path.resolve() != ply_target.resolve():
                shutil.copy2(str(source_path), str(ply_target))
            source_description = str(source_path.resolve())

        xyz = np.ascontiguousarray(
            model.get_xyz.detach().cpu().numpy(), dtype=np.float32
        )
        reliability = np.ascontiguousarray(
            model.get_reliability.detach().cpu().numpy(), dtype=np.float32
        )
        if reliability.shape != (xyz.shape[0], 4):
            raise RuntimeError(
                "GauSS-MI reliability is not aligned with the saved PLY: "
                f"xyz={xyz.shape}, reliability={reliability.shape}"
            )

        npz_target = output_dir / "step40_gaussmi_state.npz"
        json_target = output_dir / "step40_gaussmi_state.json"
        np.savez_compressed(
            str(npz_target),
            xyz=xyz,
            reliability=reliability,
        )

        delta = np.abs(reliability - float(model.init_reliability))
        metadata = {
            "source_ply": source_description,
            "captured_ply": str(ply_target),
            "state_npz": str(npz_target),
            "gaussians": int(xyz.shape[0]),
            "reliability_shape": list(reliability.shape),
            "reliability_min": float(reliability.min()),
            "reliability_max": float(reliability.max()),
            "reliability_mean": float(reliability.mean()),
            "updated_reliability_ratio": float(
                np.any(delta > 0.02, axis=1).mean()
            ),
            "xyz_sha256": hashlib.sha256(xyz.tobytes()).hexdigest(),
            "direction_order": ["+x", "+z", "-x", "-z"],
            "match_text": match_text,
            "trigger": trigger,
        }
        if update_result is not None:
            metadata["reliability_update_result"] = bool(update_result)
        json_target.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        captured["done"] = True
        print(
            "[GauSS-MI Snapshot] captured "
            f"{xyz.shape[0]} gaussians at {output_dir} ({trigger})",
            flush=True,
        )

    def save_ply_and_snapshot(self, path, *args, **kwargs):
        result = original_save_ply(self, path, *args, **kwargs)
        source_path = Path(path)
        if captured["done"] or match_text not in source_path.name:
            return result
        capture(self, source_path, trigger="matching_save_ply")
        return result

    save_ply_and_snapshot._ieve_snapshot_hook = True
    GaussianModel.save_ply = save_ply_and_snapshot

    from gaussian_mapping.mapper.realtime_3dgs import RealtimeSLAM

    original_update = RealtimeSLAM._update_gaussmi_reliability
    target_frame = int(os.environ.get("IEVE_GAUSSMI_SNAPSHOT_FRAME", "40"))

    def update_and_snapshot(self, viewpoint):
        result = original_update(self, viewpoint)
        if not captured["done"] and int(self.frame_count) == target_frame:
            capture(
                self.gaussians,
                trigger=f"mapper_frame_{target_frame}_post_update",
                update_result=result,
            )
        return result

    update_and_snapshot._ieve_snapshot_hook = True
    RealtimeSLAM._update_gaussmi_reliability = update_and_snapshot


_install_snapshot_hook()
