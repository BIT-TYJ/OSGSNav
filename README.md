# OSGSNav

Online Semantic Gaussian Splatting and Persistent Scene Memory for Instance Image Goal Navigation.

This release contains the current **multi-goal navigation source snapshot**, including the five-goal V5 runtime used for the Shared/Clear experiments. It provides the algorithm implementation and simulation reference for subsequent real-robot integration.

**Deployment status:** the executable navigation backend is Habitat. ROS RGB-D / external-pose ingestion and a hardware motion controller are not implemented by this release. The real-robot documents describe the interfaces to replace; they are integration plans, not a tested robot driver.

## Included code

| Component | Location |
|---|---|
| Five-goal V5 runtime | `five_goal_benchmark_20261026/runtime/main_multi_iin_frontier_v5.py` |
| Two-goal reference runtime and shared runtime adapters | `multi_iin_v0/runtime/` |
| Online RGB-D Gaussian mapping | `gaussian_mapping/` |
| Semantic memory, candidate confirmation, localization and recovery | `ieve_minav/` |
| 2D mapping and FMM navigation | `model.py`, `agents/`, `envs/` |
| Memory-safe clustering | `memory_safe_clustering/fast_v2/` |
| Exact-input FMM reachability cache and timers | `navigation_acceleration_20261027/` |
| Fixed five-goal sequence definitions | `five_goal_benchmark_20261026/configs/five_goal_100.json` |

The five-goal runner supports `Continuous-Shared` and `Continuous-Clear`. Each sequence contains five different physical instances, including same-category transitions. Each target has a 1,000-step budget; failure of one target advances to the next without resetting the robot pose. Shared retains scene memory; Clear rebuilds its mapping state at target changes. The goal is RGB; online mapping uses RGB-D. The simulation recipe uses task-provided category metadata (`--goal_cat_source oracle`) and simulator pose: these must be replaced explicitly for real-robot deployment.

## Install and run

See [installation](docs/INSTALL.md) for dependencies, CUDA extensions, datasets and weights. From the repository root, inspect a portable launch command without starting simulation:

```bash
python tools/run_multi_goal.py --gpu 0 --episode 0 --output results/shared_ep000
```

After installation and asset setup, execute the five-goal Shared sequence:

```bash
python tools/run_multi_goal.py --gpu 0 --episode 0 --output results/shared_ep000 --execute
```

For Clear, use a different output directory:

```bash
python tools/run_multi_goal.py --gpu 0 --episode 0 --variant Continuous-Clear \
  --output results/clear_ep000 --execute
```

The launcher exposes only the requested physical GPU, resolves repository/output paths, and preserves the source algorithm flags. Episode indices are 0--99. Use a fresh output directory for every run. With a headless Habitat installation that requires a display, prefix the command with `xvfb-run -a`. This launcher does not impose a process-memory limit: monitor memory externally when preparing long runs.

## Real-robot integration

Start with [the real-robot handoff](docs/REAL_ROBOT_HANDOFF.md). Supporting interface and sensor/control plans:

- [Habitat-to-hardware interface inventory](docs/habitat_to_real_robot_interface_summary.md)
- [Mid-360 / RGB-D / LIO integration](docs/real_robot/real_robot_mid360_integration.md)
- [Latest-frame buffering and stop-and-go control](docs/real_robot/real_robot_30hz_stop_go_implementation.md)

## Snapshot and validation

The workspace source commit alone does not identify this release: the workspace includes uncommitted source changes. [The source manifest](release/source_manifest.json) records SHA256 hashes of the exported working-tree files. [The release manifest](release/release_manifest.json) hashes all delivered files. [Validation results](release/validation.json) report checks actually run; no new simulation, model training or hardware execution is claimed.

Datasets, checkpoints, compiled CUDA binaries, experiment logs and results are excluded. Optional historical Grounded-SAM fallback under `3rdparty/minav` is not bundled; the reference configuration uses the IEVE Mask R-CNN backend. No UniGoal/IEVE comparison results or external LLM services are needed for this OSGSNav launch.

## Upstream acknowledgements and licenses

This project builds on [IEVE](https://github.com/XiaohanLei/IEVE), [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), [LightGlue](https://github.com/cvg/LightGlue), Detectron2, Gaussian Splatting and GauSS-MI. Original source notices and available vendored licenses are retained. See [third-party notices](THIRD_PARTY_NOTICES.md). No new blanket license is asserted for code with different upstream terms.

The original IEVE overview and citation are retained in [docs/IEVE_UPSTREAM_README.md](docs/IEVE_UPSTREAM_README.md).
