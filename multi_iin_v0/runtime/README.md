# Multi-IIN v0 independent runtime

This directory is an additive runner around the current IEVE implementation.
It imports the existing agent and mapping modules but does not modify their
source files.  The runner consumes the K=2 manifest and performs an explicit
target-1 to target-2 transition inside one Habitat scene.

## Lifecycle variants

| Variant | Pose at target 2 | 2D map/navigation memory | Online 3DGS/MinAV state |
| --- | --- | --- | --- |
| `Independent-Reset` | Original episode start | New/cleared | New/cleared |
| `Continuous-Clear` | Target-1 terminal pose | New/cleared | New/cleared |
| `Continuous-Shared` | Target-1 terminal pose | Preserved | Preserved |

All variants clear target-specific state and replace the visible target image.
The comparison between `Continuous-Shared` and `Continuous-Clear` isolates
state reuse while controlling for the target-1 terminal pose.

## Reproduce a 10-episode smoke run

Run from the IEVE repository root.  The command below reproduces the
`Continuous-Shared` configuration used for the recorded result:

```bash
/code1/tyj/anaconda3/envs/ieve/bin/python -u \
  multi_iin_v0/runtime/main_multi_iin.py \
  --num_processes 1 \
  --num_eval_episodes 10 \
  --num_training_frames 1000 \
  --eval_skip_episodes 0 \
  --multi_num_episodes 10 \
  --multi_goal_step_budget 2 \
  --multi_variant Continuous-Shared \
  --multi_result_json multi_iin_v0/results/real_smoke_3dgs_shared_10.json \
  --enable_3dgs 1 \
  --use_minav_confirmation_localization 1 \
  --minav_initial_spin 0 \
  --sim_gpu_id 0 \
  --sem_gpu_id 0 \
  --dump_location multi_iin_v0/results/runtime_3dgs \
  --exp_name shared_10 \
  --minav_results_dir multi_iin_v0/results/runtime_3dgs/minav \
  --print_images 0 \
  --quiet_log 1 \
  --log_interval 1000
```

For the other contracts, change the variant, experiment name, and result file
as follows:

| Variant | `--exp_name` | `--multi_result_json` |
| --- | --- | --- |
| `Continuous-Clear` | `clear_10` | `multi_iin_v0/results/real_smoke_3dgs_clear_10.json` |
| `Independent-Reset` | `independent_10` | `multi_iin_v0/results/real_smoke_3dgs_independent_10.json` |

The original lifecycle smoke deliberately allows only two real navigation
steps per target.  A subsequent single-episode integration run used the
intended 1000-step maximum for each target; its combined audit is
`multi_iin_v0/results/real_nav_3dgs_1_budget1000_summary.json`.  One episode is
still not a navigation-performance benchmark.  Performance claims require the
full evaluation set and repeated/paired statistical analysis.

## Per-target strategy clock

The runner passes the Habitat wrapper's per-target `info["time"]` to MinAV as
`step`.  This clock is reset to zero at the target-1/target-2 boundary, while a
separate `runner_global_step` field retains the monotonically increasing run
clock for diagnostics.  Consequently, with the defaults below, *each* target
receives the same exploration schedule independently:

```text
target-local steps   0-499: Phase 1 frontier exploration
target-local steps 500-799: Phase 2 instance-visit exploration
target-local steps 800-999: Phase 3 best-evidence fallback
```

The 1000-step target budget is also checked against the same target-local
clock.  Target 2 therefore no longer inherits the Phase 2/3 progress already
consumed by target 1.

## Information-gain-first frontier selection

The Multi-IIN subclass keeps the original frontier construction and FMM
checks.  Obstacle clearance, reachability, maximum path length, detour ratio,
initial-area exclusion, visited-path exclusion, and cooldown filtering all run
before ranking.  Only the surviving reachable candidates are reordered.

For each candidate, the runner counts unexplored cells in a local disk and
computes:

```text
utility = normalized_unknown_gain
          / (1 + fmm_cost_weight * normalized_fmm_cost)
          - visited_penalty * local_visited_ratio
```

Thus unknown-area gain is the main signal, FMM cost mildly discourages a very
long reachable route, and the visited-ratio term discourages repeatedly
selecting an already traversed neighbourhood.  The defaults are:

```text
--multi_frontier_information_gain_enabled 1
--multi_frontier_information_radius_m 1.5
--multi_frontier_fmm_cost_weight 0.20
--multi_frontier_visited_ratio_penalty 0.25
```

This hook exists only in `multi_iin_v0/runtime/multi_agent.py`; the original
`agents/instance_exp.py` is unchanged.  Selection diagnostics are emitted as
`[MULTI_FRONTIER_INFO_GAIN]` JSON records alongside the original FMM logs.

## Gaussian capacity and disk policy

The additive runner raises the Gaussian admission cap to 1.5 million, but does
not allow the map to grow to that cap permanently.  At 1.35 million Gaussians,
the capacity manager prunes back toward 1.1 million, leaving roughly 250,000
slots for newly explored regions:

```text
--multi_gaussian_capacity_enabled 1
--multi_gaussian_hard_cap 1500000
--multi_gaussian_prune_trigger 1350000
--multi_gaussian_prune_target 1100000
--multi_gaussian_capacity_voxel_size_m 0.03
--multi_gaussian_capacity_max_voxel_size_m 0.10
--multi_gaussian_dense_voxel_threshold 4
--multi_gaussian_recent_keyframe_grace 5
--multi_gaussian_semantic_protect_threshold 0.70
--multi_gaussian_low_opacity_threshold 0.05
```

Capacity pruning is restricted to ordinary exploration stages.  Confirmed
target Gaussians, high target-semantic scores, the most recent keyframes, and
at least one representative per occupied 3 cm voxel are protected.  Removal
prioritizes invalid/low-opacity points and dense-voxel duplicates, then uses
opacity, observation count, GauSS-MI reliability and recency to remove the
lowest-value remaining points.  If an unusually sparse map cannot meet the
capacity target while retaining every 3 cm cell, the coverage grid adapts up
to 10 cm; if necessary, capacity wins over non-target coverage while all
target/recent protections remain hard constraints.  The original model's `prune_points()` method
performs the mutation, keeping optimizer tensors, semantic scores,
reliability, and confirmed-index remapping aligned.  Each actual prune is
recorded in `gaussian_capacity_events.jsonl` under that mapper's output folder
and summarized in the target runtime audit.

Periodic complete-map checkpoints are disabled by default in this runner
(`--3dgs_save_interval 0`).  Repeated full-scene PLYs for successful and failed
confirmation attempts are also suppressed by default while target-only PLYs,
matching evidence, terminal renders, and each target's final complete map are
retained.  The old verbose behavior can be restored with:

```text
--3dgs_save_interval 50
--multi_save_repeated_full_gaussian_snapshots 1
```

These changes are installed from
`multi_iin_v0/runtime/gaussian_capacity_manager.py`; the original Gaussian
mapper and MinAV manager sources are unchanged.

## Target-2 retained-map prevalidation

For `Continuous-Shared`, target 2 can be discovered and confirmed from the
retained map before the robot moves.  The prevalidator starts from target-class
components in the retained 2D semantic map, prefers detector-trained
target-class Gaussians around each component, and verifies their NVSM renders
with semantic FEM.  Acceptance uses a conservative score threshold of 40;
weak evidence simply falls back to ordinary exploration, and confirmation
never skips the subsequent physical navigation/standoff stage.

The grace window waits for a retained 2D component to become available.  Once
an actual candidate set exists it is rendered only once, because the robot has
not moved and repeated evaluations would add no evidence.  This path does not
read goal coordinates, object ids, Habitat distance, success, or SPL.

## Gaussian semantic and instance label training

The shared Gaussian map now maintains an additive six-class label sidecar for
chair, sofa, plant, bed, toilet and TV.  Every mapping frame reuses the
detector's instance masks, projects depth-consistent Gaussians into those
masks, and accumulates multi-view semantic evidence.  Per-frame masks are
associated with persistent instance ids using existing Gaussian-id votes and
an XZ-centroid gate.  This path explicitly does not read Habitat semantic
sensor labels, goal object ids or oracle target coordinates.

The label tensors follow Gaussian pruning and densification.  Capacity pruning
also treats high-confidence multi-class labels as protected semantic content.
The controls are:

```text
--multi_gaussian_label_training 1
--multi_gaussian_label_min_confidence 0.50
--multi_gaussian_label_depth_tolerance_m 0.25
--multi_gaussian_instance_vote_ratio 0.15
--multi_gaussian_instance_association_radius_m 0.75
--multi_gaussian_label_evidence_cap 32
```

At checkpoint time the mapper writes `semantic_instance_gaussians.ply`, an
index-aligned `semantic_instance_labels.npz`, and
`semantic_instance_summary.json` under the Gaussian `final/` directory.  The
implementation lives in `multi_iin_v0/runtime/gaussian_semantic_labels.py`;
the original Gaussian model and mapper files remain untouched.

## Compact result archive

The runner now creates a second, inspection-oriented archive by default.  It
does not replace or delete the verbose MinAV/3DGS debug directories.  Its path
states the lifecycle contract, budget, episode range and run time explicitly:

```text
multi_iin_v0/results/compact/
└── continuous_shared/                 # continuous_clear / independent_reset
    └── budget1000__episodes00000-00009__20260904-120000/
        ├── run_summary.json
        └── episode_00001__wcojb4tft35__miin_v0_val_000000/
            ├── episode_summary.json
            ├── target_01__toilet__obj270__success/
            └── target_02__bed__obj17__failure/
```

Each target folder retains only the terminal evidence needed to inspect the
result:

```text
01_goal/target_image.png
01_goal/target_detector_analysis.png
02_map/final_2d_map.png
03_gaussian_map/final_gaussian_map.ply
03_gaussian_map/final_gaussian_rgb_comparison.png
03_gaussian_map/semantic_instance_gaussians.ply
03_gaussian_map/semantic_instance_labels.npz
03_gaussian_map/semantic_instance_summary.json
04_target_instance/target_instance_filtered.ply
04_target_instance/filter_stats.json
05_final_match/selected_render.png
05_final_match/target_render_matches.png
05_final_match/match_metrics.json
target_summary.json
```

`final_2d_map.png` is a labelled 2x2 rendering of occupancy, exploration,
visited trajectory and semantic layers at that target's terminal step.  The
Gaussian folder keeps both the complete splat-parameter PLY and a terminal
RGB-versus-render comparison with PSNR/SSIM recorded in `target_summary.json`.
Opening the PLY in a generic point-cloud viewer still shows points because
that viewer ignores Gaussian scale, rotation, opacity and spherical-harmonic
colour; the comparison PNG is the directly viewable rendered result.  The
NVSM match is selected from the latest completed NVSM step (highest score only
breaks ties at that step), so an older high-scoring attempt cannot be mistaken
for the terminal result.  Repeated failed confirmations, candidate views,
PRE/REF iteration images, raw duplicate instance clouds and periodic Gaussian
checkpoints remain available only in the verbose debug tree.

The archive can be disabled with `--multi_save_compact_results 0`, or moved
with `--multi_compact_results_dir PATH`.

## Semantic Gaussian instance filtering

Only this independent runner installs the additive filter in
`semantic_instance_filter.py`; the existing IEVE, MinAV, and Gaussian mapper
source files remain unchanged.  Filtering runs after NVSM marks the current
semantic instance and before centroid calculation, multi-centre rendering,
and feature matching.  It first projects Gaussian centres into the SOIM
instance region (important when a room contains two objects of the same
category), then voxelises the remaining centres and applies DBSCAN. The SOIM
mask is deliberately not dilated: the goal is a clean object-body location
for orbit rendering rather than complete geometry recovery, and a tight mask
reduces background leakage for sparse plants. If target confirmation has
already locked image-verified Gaussian indices, the component overlapping
those indices has first priority and is preserved even when the tight SOIM
projection is slightly displaced. Without a confirmed seed, component
selection first keeps the neighbourhood of the cluster nearest to the SOIM
3-D anchor and then chooses the largest supported component there. Components
below the configured global keep-ratio are not treated as plausible object
bodies in this fallback path. The resulting evidence order is therefore
confirmed target, SOIM anchor, then component size.

Filtering is enabled by default.  The main controls are:

```text
--multi_instance_filter 1
--multi_instance_filter_voxel_size_m 0.03
--multi_instance_filter_eps_m 0.10
--multi_instance_filter_min_samples 5
--multi_instance_filter_min_cluster_points 50
--multi_instance_filter_anchor_neighbourhood_slack_m 0.25
```

Every filtered `instance_semantic_<timestamp>.ply` is accompanied by:

```text
instance_semantic_<timestamp>_raw.ply
instance_semantic_<timestamp>_filtered.ply
instance_semantic_<timestamp>_filter_stats.json
instance_semantic_<timestamp>_filter_projection.png
```

The unsuffixed compatibility PLY and its TXT metadata describe the filtered
set.  The JSON records filter parameters, all component sizes and centres,
the selected component, point counts, bounding boxes, and fallback reason.
The projection PNG shows retained Gaussians in green and removed Gaussians in
red in both X-Z top view and X-Y height view.
If clustering fails or violates the configured safety floor, the runner keeps
the unfiltered set instead of dropping the target.

### Offline filter validation

Historical instance PLY files can be evaluated without changing them:

```bash
MPLCONFIGDIR=/tmp/multi_iin_matplotlib \
/code1/tyj/anaconda3/envs/ieve/bin/python -m \
  multi_iin_v0.tools.evaluate_instance_filter \
  --input-root multi_iin_v0/results/runtime_3dgs_budget1000 \
  --output-root multi_iin_v0/results/filter_validation/batch_v3_seeded_budget1000 \
  --anchor-manifest multi_iin_v0/config/filter_validation_anchors_budget1000.json \
  --recover-confirmed-seeds
```

The evaluator pairs each NVSM instance PLY with its most recent preceding
target-confirmation PLY and maps those confirmed points back into the later
snapshot. Each sample receives a filtered PLY, full JSON diagnostics, and a
top-down raw/removed/kept comparison. The batch summary and CSV are written at
the output root. These artefacts validate spatial isolation, not target
identity: if upstream target confirmation locks the wrong object, filtering
will clean that object but cannot turn it into the correct navigation target.

## Task category, detector, and Habitat semantic sensor

Multi-IIN reads the current target category from the benchmark episode's
`object_category`, so an ambiguous goal image cannot change a plant task into
a chair task.  Every new result must therefore record
`goal_cat_source: "oracle"`; the runner rejects other category sources.

This only changes how the requested category is obtained.  Mask R-CNN still
processes the live RGB observations and supplies the observation-frame
semantic predictions used by exploration and MinAV.  Habitat's
`semantic_sensor` remains available to the existing runtime and audit path; it
is not used to replace the live-image detector during navigation.

## Recorded evidence

The combined audit is in
`multi_iin_v0/results/real_smoke_3dgs_10_summary.json`.  The three source JSON
files contain per-goal metrics and per-boundary snapshots of simulator pose,
2D map counters, MinAV counters, target-image hashes, and 3DGS mapper state.
