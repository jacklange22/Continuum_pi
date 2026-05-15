# Registration Trial Workflow

`registration_trial` is a registration *diagnosis* tool, not a replacement
for the standard 4-point session. Use it when the standard FRE varies
run-to-run, when you want to know whether more landmarks materially
improve the fit, or when you want measured (not intuited) evidence that
the rig's capture protocol is good enough for thesis-grade data.

The experiment captures **N landmarks × K samples** once, then runs the
analysis four ways and reports an operator-readable recommendation. It
never auto-modifies the active registration; promotion is a separate
explicit step.

## When to use it

- Your latest `latest_registration.json` shows FRE > ~0.7 mm and you
  want to know which landmarks are dragging it.
- You suspect a single landmark is mis-targeted (touched the wrong jig
  point, ferrous interference, etc).
- You want to defend "we used N points with K samples each" in a thesis
  methodology section with measurement-driven evidence instead of an
  arbitrary choice.
- You are debugging tracker noise that does not show up cleanly in the
  `aurora_grid_accuracy` workflow.

Do **not** use it as your routine registration step. The standard 4-point
flow on the Registration tab is faster and produces the active artifact
on its own. The trial is an analysis layer, not a production capture.

## What it computes

For one captured pool `{label: K samples}` and the truth coordinates from
`config/registration.yaml`, the experiment runs:

1. **Method sweep.** For each averaging method in `mean`, `median`,
   `trimmed_mean`, `mad_filtered_mean`, collapse each landmark's K
   captures to a single point and solve the full registration. Reports
   FRE, per-landmark residuals, leave-one-out FRE per excluded
   landmark, and a "max LOO drop" diagnostic that surfaces a dominant
   bad point.
2. **Subset search.** Using the best-method averaged points, solve
   *every* subset of sizes 4..N (configurable) exhaustively. Reports
   the best-FRE subset per size, the global best, and rank /
   condition-number geometry diagnostics — coplanar landmark sets get
   flagged here.
3. **Samples-per-point study.** Draw random k-of-K subsamples from each
   landmark's pool (mean averaging fixed, so the k axis is not stacked
   on top of the averaging-method axis), solve the full-N-point
   registration, and report FRE vs k. Recommends the smallest k whose
   mean FRE lies within an operator-tunable epsilon (default 0.02 mm)
   of the captured pool's best.
4. **Leave-one-out.** Already produced as part of (1). Useful to find
   one landmark that dominates the residual.

All math is the same Kabsch solve used everywhere else in the codebase.
Nothing is invented; this is honest re-analysis of one capture set.

## Operator workflow

1. **Restart** the GUI on the Pi (or kill+relaunch). The Trial Mode
   button is on the Registration tab's secondary action row.
2. Click **Run Registration Trial →**.
3. **Setup phase**: select the landmarks you want to capture (default:
   all enabled in `config/registration.yaml`). Pick K samples/landmark
   (default 50; 20–30 is often enough on a quiet rig). Click **Start
   Trial**.
4. **Capture phase**: position the pen probe on the current landmark,
   wait for the count to fill, click **Next Landmark**. **Capture
   Batch** auto-captures at a fixed rate until the target is reached.
   **Skip Landmark** drops the current landmark's collected samples and
   moves on. **Stop & Analyze** finalizes whatever has been captured.
5. **Result phase**: the dialog reports the best averaging method, the
   best subset, and the top recommendations. The full
   `trial_report.md` (markdown) and `trial_report.json` are in the run
   directory. Four PNG reports are in the same directory for hand-off.

## Promoting a trial result

After reviewing `trial_report.md` (or the dialog's result phase), if a
better subset emerges, write it to the active registration slot:

```bash
.venv/bin/python -m continuum_robot.data.promote_registration_trial \
  --run-dir data/experiments/registration_trial/<run> \
  [--subset L1,L2,L4,L7] \
  [--averaging mean] \
  [--operator-note "Trial T3 lowers FRE 0.4 mm by dropping L3"]
```

If `--subset` is omitted, the report's `global_best` subset is used.
`--averaging` defaults to `mean`. The current `latest_registration.json`
is copied to `latest_registration_backup_<timestamp>.json` before the
new artifact is written, so promotion is always reversible.

Pass `--dry-run` to validate the candidate without writing anything.

## What the report tells you

| Field | Meaning |
|---|---|
| `method_summary.best_method` | Which averaging method had the lowest full-N-points FRE on this dataset. If all methods agree within 0.01 mm, capture more samples per landmark; the averaging knob is not the bottleneck. |
| `subset_search_summary.global_best` | Which subset (any size 4..N) had the lowest FRE. The labels are the recommended capture set for the production 4-point session. |
| `subset_search_summary.per_size_best[size].best_geometry_rank` | Should be 3. If lower, your landmarks are coplanar — add a landmark at a different height. |
| `subset_search_summary.per_size_best[size].best_geometry_condition_number` | Larger = more ill-conditioned fit. Track this run-over-run; sudden spikes usually mean a landmark drift. |
| `samples_per_point_recommendation.recommended_k` | Smallest k whose mean FRE is within `epsilon_mm` of the captured pool's best. Use this as the new default samples-per-landmark target. |
| `samples_per_point_summary[*].fre_std_mm` | Capture-to-capture noise at that k. Compare to your thesis budget. |
| `trial_recommendations` | Plain-English findings the operator can act on. |

## Limits

- Trial captures DO NOT auto-promote. Nothing about the trial changes
  `latest_registration.json` until you run the promote tool.
- The samples-per-point study uses mean averaging and bootstrap
  iterations. Other averaging methods may scale differently with k;
  the sweep already covers that axis, but they are not stacked
  together to keep results comparable.
- Subset search is exhaustive at sizes 4..N for N ≤ 12 (default).
  At larger N the combinatorics grow; cap `subset_sizes` if you push
  past 12 landmarks.
- Trial outputs are **never** marked `valid_for_thesis_repeatability`
  or `valid_for_model_training`. The trial is a registration-protocol
  diagnosis; thesis repeatability runs come out of the
  `single_segment_repeatability` experiment, using whatever
  registration was active at the time.

## Pitfalls

- If the operator selects 3 landmarks the precheck refuses; need ≥ 3.
- If the dataset has fewer than 4 *shared* labels (capture set ∩ truth
  set), leave-one-out is skipped. Subset search still runs at the
  available size only.
- Mock mode: the dialog's **Capture One** button uses whatever the
  tracking service returns. In offscreen / no-tracker mock mode, this
  is the latest `0B` reading from the registration service, which
  defaults to zeros until a real tracker frame arrives. Use the
  experiment's replay path (`source_record_path` pointing at a saved
  raw-captures JSON) for headless analysis.
