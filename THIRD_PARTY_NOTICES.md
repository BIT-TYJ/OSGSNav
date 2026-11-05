# Third-party source and attribution

This snapshot does not replace existing copyright or license terms.

- **IEVE:** the root navigation architecture and portions of `agents`, `envs`, `algo`, `model.py` and utilities derive from https://github.com/XiaohanLei/IEVE. The upstream README and paper citation are retained at `docs/IEVE_UPSTREAM_README.md`. Do not infer a new blanket license from this upload.
- **Habitat-Lab:** modified version 0.2.3 source under `3rdparty/habitat-lab`; Meta copyright headers retained. The matching v0.2.3 MIT LICENSE was recovered from the official upstream `https://raw.githubusercontent.com/facebookresearch/habitat-lab/v0.2.3/LICENSE` because the workspace copy lacked it.
- **Gaussian Splatting / normal rasterizer:** source notices and `LICENSE.md` retained under `gaussian_mapping/submodules/diff-gaussian-rasterization`; the license contains research/non-commercial restrictions. Other Gaussian mapping source retains its own upstream headers.
- **simple-knn:** source under `gaussian_mapping/submodules/simple-knn` retains Inria / GRAPHDECO headers specifying non-commercial research/evaluation terms and referring to an upstream LICENSE.md. The workspace did not contain that file; this release does not manufacture a license or assert permissive rights. Consult the original authors before redistribution/use beyond those headers.
- **GauSS-MI:** the required rasterizer source is retained under `3rdparty/GauSS-MI/submodules/diff-gaussian-rasterization-gaussmi`, including its available license files. The parent GauSS-MI GPL-3.0 license is retained at `3rdparty/GauSS-MI/LICENSE`. Nested rasterizer and parent terms must both be reviewed as applicable.
- **GLM:** vendored header dependencies retain their upstream copyright and `copying.txt` notices.
- **LightGlue, SuperPoint, Detectron2:** installed externally using the reference revisions in `docs/INSTALL.md`; no model weights are distributed. Their own licenses and model terms apply.
- **Datasets and checkpoints:** no HM3D scene data, target images, learned model weights or experimental outcomes are included. Sequence manifests contain only source references and task metadata.

The historical optional Grounded-SAM fallback rooted at `3rdparty/minav` is not included. Its code is not required by the default IEVE Mask R-CNN backend used by this release's launch recipe.

- **UniGoal-M adapter dependency:** pinned to `BIT-TYJ/omnigoal` revision `6e581d269e246d60cb1276b9af812f4d864eb2a4`, the actual experimental fork. The upstream source/weights are not vendored by this update; its own notices apply. Adapter origin and path-only changes are recorded in `baselines/unigoal_m/adapter_provenance.json` and `portability_changes.diff`.
