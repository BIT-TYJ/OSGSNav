# Installation and reference simulation

This is a source snapshot, not a prebuilt environment. The reference environment uses Linux, Python 3.8, PyTorch 2.0.1 / torchvision 0.15.2, CUDA 11.7 and habitat-sim 0.2.3. A compatible NVIDIA driver and CUDA compiler are required to build extensions. No fresh environment installation was performed for this upload.

## 1. Python and simulation dependencies

Create and activate your environment, then install PyTorch appropriate to your GPU/driver. One reference combination is:

```bash
conda create -n osgsnav python=3.8
conda activate osgsnav
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
conda install habitat-sim=0.2.3 headless -c conda-forge -c aihabitat
pip install -r requirements-runtime.txt
```

`requirements-runtime.txt` is an installation starting point, not an independently validated lockfile. `requirements.txt` and `conda-explicit.txt` preserve historical environment information and can contain old editable-source URLs or machine-specific package references: do not blindly use them to overwrite a new environment. In particular, install the local CUDA and Habitat sources below instead of their historical editable URLs.

Install the bundled modified Habitat-Lab from its directory because its setup script reads a relative requirements file:

```bash
(cd 3rdparty/habitat-lab && pip install -e .)
```

Install external feature/detection libraries matching the reference revisions:

```bash
pip install 'git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8'
pip install 'git+https://github.com/facebookresearch/detectron2.git@7684ebce0790bf3b8b9330b69f63c44c74a08ead'
```

## 2. CUDA extensions

The normal rasterizer and GauSS-MI rasterizer have colliding Python package names. Install the normal renderer conventionally, but build GauSS-MI **in place**; `gaussmi_renderer.py` loads it under a separate module name from its source directory.

```bash
pip install --no-build-isolation ./gaussian_mapping/submodules/diff-gaussian-rasterization
pip install --no-build-isolation ./gaussian_mapping/submodules/simple-knn
(cd 3rdparty/GauSS-MI/submodules/diff-gaussian-rasterization-gaussmi && python setup.py build_ext --inplace)
```

Use a CUDA toolkit compatible with PyTorch. Compiled `.so` files are deliberately excluded and must be rebuilt on the deployment machine.

## 3. Assets

Obtain HM3D scenes and the HM3D InstanceImageNav-v3 validation dataset under their original access terms. Arrange paths as referenced by the bundled task config and manifest:

```text
data/scene_datasets/...
data/tyj/datasets/instance_imagenav/hm3d/v3/val/content/<scene>.json.gz
pretrained_models/sem_exp.pth
```

The manifest contains sequence metadata and references, not images or scene meshes. Inspect its `source` field and `envs/habitat/configs/tasks/instance_imagenav.yaml` for the exact source layout. If your data lives elsewhere, create local symlinks or a separate local config; retain the original manifest for comparisons.

The original IEVE pretrained-weight download is linked in `docs/IEVE_UPSTREAM_README.md`. Mask R-CNN uses the Detectron2 model-zoo URL configured in `agents/utils/semantic_prediction.py`; LightGlue/SuperPoint can also download weights on first use. Populate caches before an offline robot session. The five-goal simulation recipe uses oracle category metadata, so an independently configured classifier and its weights are required only if choosing that different category source.

## 4. Checks and launch

```bash
python tools/check_release.py
python tools/run_multi_goal.py --gpu 0 --episode 0 --output results/shared_ep000
# After checking assets and CUDA dependencies:
python tools/run_multi_goal.py --gpu 0 --episode 0 --output results/shared_ep000 --execute
```

Static release checks do not prove that a new machine has a working Habitat/CUDA environment. A first simulator episode is still required on that machine before a larger evaluation or hardware port.
