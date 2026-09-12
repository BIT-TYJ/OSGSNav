# Sequential Ordered Multi-IIN v0

This directory is an additive prototype. It does not import or modify the
existing IEVE runtime. Its first purpose is to establish a reproducible,
geodesically valid K=2 dataset before a multi-goal runtime is implemented.

## Frozen v0 protocol

- Two ordered target instances are sampled from one HM3D scene.
- Only the current target image is visible to the policy.
- A successful intermediate stop declares the current target and advances to
  the next target without resetting simulator pose.
- Each target has a 1000-step budget and a 1.0 m success threshold.
- `object_category`, object IDs, view points, and oracle paths are evaluator
  metadata and must not be exposed to the policy.
- The prototype defaults to different target categories. Same-category pairs
  are deferred until the basic task lifecycle is verified.

The generated manifest references the existing HM3D InstanceImageNav-v3
goals and image parameters. It does not duplicate or alter source data.

## Build the 5-scene validation manifest

Run from the IEVE repository root with the existing IEVE Python environment:

```bash
/code1/tyj/anaconda3/envs/ieve/bin/python \
  multi_iin_v0/build_manifest.py
```

The deterministic default build creates 100 episodes: five scenes and 20
episodes per scene. Change `--seed`, `--num-scenes`, or
`--episodes-per-scene` to generate another development manifest.

## Validate

Fast structural and source-reference validation:

```bash
/code1/tyj/anaconda3/envs/ieve/bin/python \
  multi_iin_v0/validate_manifest.py \
  multi_iin_v0/manifests/miin_v0_val_k2.json
```

Independent full geodesic recomputation:

```bash
/code1/tyj/anaconda3/envs/ieve/bin/python \
  multi_iin_v0/validate_manifest.py \
  multi_iin_v0/manifests/miin_v0_val_k2.json \
  --check-geodesic
```

## Lifecycle smoke runner

Run the first ten manifest episodes under all three reset contracts:

```bash
/code1/tyj/anaconda3/envs/ieve/bin/python \
  multi_iin_v0/run_lifecycle_smoke.py
```

The output is written to `multi_iin_v0/results/lifecycle_smoke_10.json`.
This runner uses real HM3D navmeshes and asserts the pose/map/3DGS/history
boundary contract. The reusable components are an auditable state ledger, not
the real GS-IIN mapper. Therefore this is a lifecycle smoke test and cannot be
used to claim map-reuse performance gains.

## First full runtime comparison (not implemented yet)

The first experiment should keep target order and episodes identical:

1. `Independent-Reset`: reset pose and all maps for each target.
2. `Continuous-Clear`: preserve pose but clear maps at the target boundary.
3. `Continuous-Shared`: preserve pose, 2D map, 3DGS, and history.

`Continuous-Shared - Continuous-Clear` isolates reuse better than comparing
only against `Independent-Reset`, because it controls for the first target's
ending position.

The next implementation step is a copied, separately namespaced GS-IIN runtime
that consumes this manifest. It should remain beside the existing runtime
rather than changing `main.py`, `agents/`, `envs/`, `ieve_minav/`, or
`3rdparty/`.
