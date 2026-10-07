# One-factor component screen (2026-10-05)

This is an isolated source snapshot for identifying why the combined boost
regressed against the current single-target baseline. The runtime repository is
never edited.

The nine episode panel contains every baseline/boost flip from the preceding
12-episode screen (`4,25,61,212,249,517,635`) and two stable baseline successes
(`581,614`). Each profile changes exactly one control from the baseline:

- `baseline`: Phase-2 at step 500; downstairs guard off; stuck recovery off.
- `phase2_400`: only Phase-2 moves to step 400.
- `downstairs_guard`: only the persistent downward-drop guard is on.
- `stuck_recovery`: only ordinary-exploration stuck recovery is on.

`downstairs_guard` is deliberately named narrowly: it is not yet the final
bidirectional no-floor-change guard. The public 1000-episode audit established
that every task has a same-floor target viewpoint; a robust bidirectional guard
will be designed only after this regression attribution screen.

## Independent 100-episode validation

`stuck_recovery_independent100_seed20261006.txt` is a fixed random 100-episode
sample from the valid 998 episodes after excluding the nine screening episodes.
The selected candidate enables only ordinary-exploration stuck recovery:
`PREVENT_DOWNSTAIRS=0`, `PHASE2_START_STEP=500`, and
`STUCK_RECOVERY_ENABLED=1`.
## Combined guarded full-998 run (2026-10-07)

This backup adds a 500k Gaussian hard capacity guard, budget-aware
clone/split densification, and a metrics-only batch storage profile. The full
launcher combines the recovered 500-step Phase-2 setting with downstairs
protection, adaptive stuck recovery, semantic Gaussian mapping, GaussMI
frontier guidance, fine localization, secondary exploration, cluster recovery,
and the sweep turn-in-place fix. Official episodes 774 and 778 are excluded.

Launch with:

```bash
source/launch_combined_guard_full998_tmux.sh
```
