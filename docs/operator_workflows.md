# Operator Workflows

## Goal

Keep the Pi GUI as the single operator surface for calibration, validation, registration, and experiments. These workflows reflect the current repo and the target lab sequence.

## Current Hardware-Day Guardrails

- Use `single_segment`, Segment B, servo IDs `[5,6,7,8]` for the current distal hardware setup.
- Missing Segment A IDs `[1,2,3,4]` are not a readiness blocker while Segment B is the active single segment.
- Keep torque enabled unless the operator intentionally disables torque. Normal reload/disconnect/GUI close should preserve torque by default.
- Do not use move-to-4095 untensioned reference with tendons attached.
- If any present position is near the raw 0/4095 discontinuity, stop, manually reset if needed, then re-capture neutral/startup.
- Before automatic pretension, verify tiny jog sign/mapping, capture neutral/safe bounds, and use `current_position` mode.
- Segment B mapping: `5 = +x`, `6 = +y`, `7 = -x`, `8 = -y`; pairs are `axis_a = 5/7`, `axis_b = 6/8`.
- Use `Confirm Configured Mapping` on the Servos tab when the active Segment B hardware matches the configured mapping. The saved artifact records `lower_tick_means_tension=true`.
- Penprobe chasing is demo-only: 0A coil-origin chases 0B in XY. Tune `Gain (ticks/mm)` and `Max Step / Cycle` carefully: start at 25 ticks, then 50, then 100 only if motion direction and tracker freshness are stable.
- System readiness uses cached servo telemetry during normal background refresh and active experiments. Use explicit refresh/scan buttons for live bus reads.

## Workflow 1: One-Servo Bring-Up

Applies now.

1. Launch the GUI.
2. In `System`, save the intended bring-up parameters:
   - `robot_1servo.yaml`
   - OpenRB port
   - baudrate
   - fine/coarse jog size
   - default pretension threshold
   - tightening direction
3. Connect external power on the OpenRB / DYNAMIXEL side.
4. Confirm tracker status in the System/Tracking surfaces.
5. Connect OpenRB and verify the prepared bridge status.
6. Scan one `XC330-M288-T` servo in `Servos`.
7. Verify model / firmware / position / current / voltage / temperature / error before any motion.
8. Use fine jog only first, then coarse jog if the first steps look correct.

Success criteria:

- Aurora backend is healthy or mock mode is explicitly enabled.
- OpenRB is connected and prepared.
- One servo is reachable and reports healthy telemetry.
- Fine and coarse jog both work through the canonical GUI / service path.

## Workflow 2: Startup Calibration And Pretension

Applies now for one-servo bring-up and scales to whole-segment one-click pretension. All pretension controls live on the `Servos` tab (there is no separate `Pretension` tab).

1. Put the robot or single tendon path in the intended neutral starting pose.
2. In `Servos`, use fine/coarse jog to reach the intended neutral position conservatively.
3. Save startup calibration for the active servo:
   - current Present Position becomes the neutral setpoint
   - conservative software min/max bounds are saved around that point
   - the pretension/current threshold is saved
4. Start the cautious pretension routine. For a whole segment, click **Run Pretension Trial** for one-click segment pretension.
5. Let the routine stop on threshold, cancel it, or retry as needed.
6. Accept the pretension result only after reviewing the final current / position.
7. The run writes an algorithm-vs-manual comparison report (markdown + figures) so the chosen policy is auditable.

Success criteria:

- startup calibration artifact is saved with neutral, bounds, threshold, and direction
- pretension run stops safely on threshold, overcurrent, travel limit, timeout, cancel, or telemetry failure
- accepted result is visible in the calibration summary
- algorithm-vs-manual comparison report is reviewable before promoting to thesis-trusted use

## Workflow 3: Pivot Calibration

Applies now in `Tracking`.

1. Open `Tracking`.
2. Validate tracker health and confirm `0B` is tracked.
3. Start `0B` pivot collection.
4. Move the probe through a wide range of orientations while the sample counter rises.
5. Stop collection.
6. Solve pivot calibration.
7. Review RMSE, used/rejected sample counts, and the staged tip file.
8. Accept the tip file.

Success criteria:

- raw capture CSV saved under `data/pivot_calibration/captures/`
- review run bundle saved under `data/pivot_calibration/`
- staged tip file reviewed before acceptance
- accepted tip vector file generated

## Workflow 4: Tracker Validation

Applies now.

1. Run tracker doctor / smoke / benchmark before registration.
2. Confirm tool visibility for `0A` and `0B`.
3. If needed, run `aurora_grid_accuracy` to measure tracker-only error.

Success criteria:

- tracker healthy before registration
- grid RMS metrics available when truth data exists

## Workflow 5: 4-Point Registration

Applies now in `Registration`.

1. Move to `Registration`.
2. Confirm the accepted `0B` tip file is loaded.
3. Choose 4 landmarks from the configured candidate set using the top-view map or the landmark list.
4. Capture repeated `0B` samples for each selected landmark.
5. Mark each selected point complete.
6. Solve.
7. Review the validation summary before saving:
   - FRE / RMSE
   - max landmark residual
   - worst landmark residual
   - landmark geometry / spacing diagnostics
   - live transform-chain status
   - repeated-run comparison if prior registrations already exist
8. Save the accepted registration.

Rules:

- exactly 4 unique landmarks
- only enabled landmarks may be selected
- the configured candidate landmarks come from the protected lab model points, not placeholder coordinates
- solve remains blocked until all selected points have enough samples

Success criteria:

- accepted registration file saved
- GUI shows selected landmarks, measured centroids, FRE, residuals, and output path
- saved registration artifact shows the accepted `0B` tip provenance used during capture
- saved registration artifact shows the live-pose `T_coil_tip` source explicitly
- repeated registration-validation summary is written under `data/registrations/validation/`
- tracking pipeline can use the saved registration on the next refresh

## Workflow 5A: Registration Trial (diagnosis)

Applies when standard 4-point FRE is varying run-to-run or you want
measured evidence for which landmarks and how many samples are right.

1. In `Registration`, click **Run Registration Trial →**.
2. Select N landmarks (default: all enabled candidates in
   `config/registration.yaml`) and K samples per landmark.
3. Capture all landmarks. Use **Capture Batch** to auto-fill at the
   configured rate or **Capture One** for manual control.
4. **Stop & Analyze** when done. The dialog's result phase reports:
   - best averaging method and its FRE
   - best landmark subset (any size 4..N) and its FRE
   - samples-per-point recommendation (smallest k within
     `samples_per_point_epsilon_mm` of the captured pool's best)
   - other recommendations: coplanarity, diminishing returns, dominant
     landmark
5. Inspect the run directory's `trial_report.md`, `.json`, and four
   PNG reports.
6. To replace the active registration with the trial's recommendation:
   ```bash
   python -m continuum_robot.data.promote_registration_trial \
     --run-dir <run> [--subset L1,L2,L4,L7] \
     --operator-note "Trial T3 lowers FRE 0.4 mm"
   ```

Rules:

- the trial NEVER auto-modifies `latest_registration.json`
- promotion is explicit; current artifact is backed up to
  `latest_registration_backup_<timestamp>.json` before the new one is
  written
- trial outputs are NOT marked thesis-valid or model-training-valid;
  they are protocol diagnosis only

See `docs/registration_trial_workflow.md` for the full reference.

## Workflow 5B: Runtime Tip Calibration

Applies now as an advanced workflow launched from `Registration`.

1. In `Registration`, click `Open Runtime Tip Calibration`.
2. Confirm the accepted `0B` tip file is loaded and the hat truth geometry is ready.
3. Begin the runtime tip calibration session.
4. Capture the configured hat points with the calibrated `0B` pen probe.
5. Mark each hat point complete after enough samples.
6. Collect stationary `0A` samples while the hat defines the fixed Tip frame.
7. Solve the calibration.
8. Review:
   - hat fit RMSE
   - max hat residual
   - per-point residuals and spreads
   - `0A` translation spread
   - `0A` rotation spread
   - runtime chain status
9. Save the accepted runtime tip calibration artifact.

Success criteria:

- canonical artifact saved under `data/runtime_tip_calibration/`
- artifact records `T_coil_tip`, `T_tip_aurora`, and `T_aurora_coil_avg`
- tracking reports that the live chain is using the accepted runtime tip calibration
- live robot-frame tip pose is no longer on identity fallback

## Workflow 6: Single-Segment Repeatability

Applies now in dry-run and later with live hardware.

1. Open the Experiment workspace.
2. Select `single_segment_repeatability`.
3. Review preflight, output path, run-validity thresholds, and config summary.
4. Run live only after neutral calibration, accepted pretension, accepted registration, and accepted runtime tip calibration are complete.
5. Review run-validity, repeatability RMS, per-target spread, and any baseline comparison deltas.

Target acceptance:

- logs commanded motion plus measured pose
- robot-frame metrics available only when the full live transform chain is trusted
- repeatability summary is comparable to the `< 1 mm` target

## Workflow 7: Modeling Dataset Collection (Random Data Collection)

Applies now in `Experiment`.

1. Open the Experiment workspace.
2. Select `Random Data Collection` (operator label for `collect_pose_command_dataset`).
3. Choose the `dataset_mode`:
   - `workspace_coverage` — first-pass forward-model data
   - `hysteresis_path_dependence` — ordered state-history datasets
   - `repeatability_linked` — trusted startup-state comparison blocks
   - `angular_test_mesh` — Wolfe §3.2.3 cross-acquisition test mesh
4. Review the collection summary:
   - runtime tip mode
   - pretension source
   - preflight trust summary
   - planned commands/captures
   - output destination
5. Run live only after accepted registration, accepted runtime tip, accepted pretension, and healthy tracker/servo state are confirmed.
6. After the run, review:
   - accepted vs rejected captures
   - workspace coverage plot
   - command distribution plot
   - export JSONL / optional legacy DAT
   - saved provenance in `summary.json` and `modeling_dataset_summary.txt`

Target acceptance:

- ordered command history is preserved
- accepted samples use fresh valid tracker data only
- robot-frame tip pose includes tangent/orientation when trusted
- saved output is ready for later offline ANN / state-aware model training

## Workflow 8: Two-Segment Startup And Dataset Foundation

Applies in `dual_segment` mode only. This is pre-control foundation work.

For a condensed bench-day cheat sheet that covers all six stages
(startup → babble → Mike probe → repeatability → modeling → handoff), see
[`docs/two_segment_bench_day_quickref.md`](two_segment_bench_day_quickref.md).

### Step 0: Confirm physical bottom/top assignment

The two fixed servo groups are hardware-identified by their servo IDs:

- Segment A: servos `[1, 2, 3, 4]`
- Segment B: servos `[5, 6, 7, 8]`

Their *physical role* (which one sits at the bottom/proximal and which at the
top/distal of the stacked rig) is selectable in the System page for each
dual-segment session, then persisted into the robot config override. The same
fields live in `config/robot_8servo.yaml` under the `physical_assembly` block:

```yaml
physical_assembly:
  bottom_segment: "segment_a"   # or "segment_b" if you stacked it the other way
  top_segment: "segment_b"
  lower_tick_means_tension: true
  notes: ""
```

The GUI mode summary chip displays the resolved bottom/top assignment and an
operator confirmation flag. Trusted two-segment startup/dataset runs should not
start until the operator confirms the physical stack for that session. If the
assignment is invalid (same segment selected twice, or unknown key) the GUI
preflight will block two-segment experiments with a clear error.

The kinematic convention is `T_distal = T_bottom(q_bottom) * T_top(q_top)` —
top tendons physically pass through the bottom segment, so bottom motion moves
the top base. This is metadata-only; no two-segment control is implemented.

### Step 1: Run the validation experiment

1. Select `dual_segment` in System.
2. Use Servos to confirm all 8 servos are visible (`[1..8]`).
3. Confirm the bottom/top assignment shown in System/Experiment.
4. Run `two_segment_startup_validation`.
5. Capture the staged manual workflow (stage names use bottom/top, not A/B):
   - `baseline`
   - `bottom_pretensioned` (proximal)
   - `top_pretensioned` (distal)
   - `bottom_recheck` (because top tendons pass through bottom)
   - `final_accept`
6. Save the all-8 manual startup artifact.
7. Run `two_segment_collect_pose_command_dataset` only after the all-8 startup
   artifact exists.

Legacy stage names `segment_a_pretensioned` / `segment_b_pretensioned` /
`segment_a_recheck` are still accepted on input for backwards compatibility and
are normalized to `bottom_*` / `top_*` in outputs.

### Step 2: Pick a collection schedule

Available `schedule_type` values for the collect-pose dataset:

- `zero`: no motion (capture only)
- `single_axis_micro`: small single-axis sweeps on bottom and top
- `segment_isolation`: a minimal subset that isolates each segment
- `small_combined`: small simultaneous bottom+top commands
- `bottom_only_sweep`: cardinal directions on the bottom segment only
- `top_only_sweep`: cardinal directions on the top segment only
- `workspace_coverage`: cardinals + diagonals across both segments
- `random_babble`: reproducible random sampling (seed-controlled)
- `structured_grid`: grid sweep along each tendon axis
- `mixed_training`: combination of structured + random for training data

The command amplitude is set by `max_segment_displacement_cm` (no silent cap).
Start at `0.25 cm`, ramp up after hardware safety is confirmed:

- `0.25 cm` (conservative bring-up)
- `0.50 cm` (early data)
- `0.75 cm`
- `1.00 cm` (target range for thesis-quality coverage)

If reliability is poor at higher amplitudes, slow/ramp the motion rather than
shrinking the final range. Long-run features:

- `continue_until_valid_samples: true` + `target_valid_sample_count: 1000` runs
  the schedule in cycles until enough accepted samples are collected.
- `long_run_recovery_enabled: true` together with `drop_sample_on_transport_error`
  drops individual bad samples and continues rather than aborting.
- The run writes `long_run_health.json`, `transport_recovery_report.json`, and
  `sample_failure_events.jsonl` to the run directory for analysis.

### Trust rules

- Servo-only/dry-run two-segment datasets are useful for software rehearsal, but
  are not model-training valid.
- Trusted two-segment modeling data needs an accepted all-8 startup artifact,
  confirmed bottom/top assembly, and a robot-frame `distal_tip` pose label.
- Missing orientation/tangent labels do not block XYZ position modeling.
- Missing `distal_tip` labels block trusted ANN model-training use.
- Distal-only runs are valid for ANN distal-tip mapping when `distal_tip` XYZ is
  present. They are clearly marked `distal_only=true`; intermediate/two-coil
  labels are optional unless the operator explicitly chooses a two-coil label
  mode.

### Tracker role selection

The two-segment collect-pose page has role selectors for `distal_tip`,
`intermediate_segment`, and `debug_tool`. Use these to set, for example, `0A`
as distal and `0C` as intermediate without editing YAML. The selected mapping is
stored in the run config/provenance. If only the distal role is live, the run is
`distal_only` and still usable for distal-tip ANN training.

### Current limitations

- No automatic two-segment pretension.
- No live two-segment control.
- No two-segment penprobe chasing.
- Mike/Camarillo comparison models remain scaffolded until active two-segment
  physics adapters are validated against hardware.

## Workflow 9: Two-Segment Modeling Analysis

Applies after `two_segment_collect_pose_command_dataset` has produced trusted labeled samples.
This workflow is offline data analysis only. It does not enable live two-segment control,
penprobe chasing, or automatic two-segment pretension.

Required input data:

- `operating_mode=dual_segment`
- accepted all-8 manual startup provenance
- successful accepted commands
- non-servo-only trusted samples by default
- `distal_tip` pose role in robot frame
- `valid_for_two_segment_ann_training=true` for ANN distal-tip or two-coil
  training. `valid_for_two_segment_model_training` remains false for mock/lower
  trust runs.

Servo-only or dry-run data is rejected by default because it has no robot-frame distal-tip
pose label to train against. Use lower-trust analysis only for debugging labeled data that
does not meet the trusted-run criteria.

CLI:

```bash
.venv/bin/python -m continuum_robot.modeling.two_segment.cli \
  --latest \
  --config config/modeling_two_segment.example.yaml \
  --label-mode auto \
  --models linear_baseline ann camarillo mike_constant_curvature
```

GUI:

1. Open the `2-Segment Modeling` tab (separate from the single-segment `Modeling` tab).
2. Select one or more `two_segment_collect_pose_command_dataset` runs.
3. Check trainability status: accepted/rejected samples, rejection reasons, and orientation availability.
4. Choose the label mode. `Auto` uses `two_coil_xyz` when every accepted sample has `intermediate_segment` and `distal_tip` robot-frame XYZ labels; otherwise it uses `distal_xyz`.
5. Keep `Strict` enabled for thesis-facing work.
6. Choose model families and run analysis.
7. Open or export the output bundle.

Data tab shortcut:

1. Select a `two_segment_collect_pose_command_dataset` run.
2. Click `Run Two-Segment Modeling` for a quick linear-baseline analysis.
3. Select the resulting `two_segment_modeling` run to open its summary or export the bundle.

Outputs:

- `two_segment_model_comparison_report.png`: model XYZ RMSE comparison
- `two_segment_distal_measured_vs_predicted_xy_report.png`: distal-tip XY measured vs predicted
- `two_segment_intermediate_measured_vs_predicted_xy_report.png`: intermediate coil XY measured vs predicted when labels exist
- `two_segment_two_coil_error_report.png`: distal/intermediate position error comparison
- `two_segment_position_error_distribution_report.png`: position error distribution in mm
- `two_segment_axis_error_report.png`: X/Y/Z RMSE
- `two_segment_orientation_error_report.png`: angular tangent/orientation error when explicit tangent labels exist

Composition semantics:

- Segment A is the lower/proximal segment by default, and Segment B is the upper/distal segment by default; the modeling metadata records the configured segment order and roles used for a run.
- Segment B rides on Segment A. Moving Segment A changes the global pose of Segment B even when Segment B tendon displacement is zero.
- The active scaffold fits a direct global forward model: `[dA, dB] -> robot-frame pose labels`.
- Future structured control may use `base -> intermediate -> distal` composition, but this workflow does not implement control.

Model status:

- `linear_baseline` is implemented and should be used as the first sanity check.
- `ann` runs when PyTorch is available; otherwise it is reported as unavailable and analysis continues.
- `mike_constant_curvature` is a config-gated geometric baseline. It only generates predictions when segment lengths, tendon positions, sign convention, frame convention, and curvature convention are explicitly configured.
- `camarillo` remains unavailable until segment/cable stiffness values, additional cable length, tendon routing, sign convention, and frame convention are measured and validated for the current stacked robot.

Physics model readiness:

- Physics models compose Segment A and Segment B transforms explicitly. This is different from direct ANN/linear models, which learn `[dA, dB] -> global pose` from data.
- Required geometry belongs in `config/modeling_two_segment.example.yaml` under `physics_models`.
- Known design geometry is recorded there with provenance/status fields. Current design entries include 66 mm Segment A/B lengths, 2 mm spacer after Segment A, 4.12 mm tendon-hole center radius, nominal 0/90/180/270 degree tendon angular positions, and Smooth-On PMC-780 / 80A material.
- The 1.65 mm hole-size entry is intentionally named `hole_radius_or_diameter_mm` because the radius-versus-diameter wording still needs confirmation.
- Positive tendon displacement is recorded as tendon shortening / more tension, and shortening is expected to decrease encoder ticks. Servo current remains a load proxy only, not a tendon stiffness measurement.
- Missing or unvalidated physics parameters produce `unavailable_missing_parameters` or `unavailable_unvalidated_convention`; they do not produce zeros or placeholder predictions.
- Each modeling run writes `model_status.json`, `physics_model_parameter_report.json`, and `physics_model_parameter_report.txt` so thesis evidence can distinguish completed numeric models from unavailable physics adapters.
- To enable real physics comparison later, measure segment lengths, tendon positions/radii, neutral/startup reference convention, displacement sign convention, and Camarillo stiffness/cable stiffness values, then validate the resulting prediction frames against tracked coils.

## Recommended Lab Order

1. tracker doctor / smoke
2. `Tracking` connect + validation
3. guided `0B` pivot calibration in `Tracking`
4. 4-point registration in `Registration`
5. runtime tip calibration from `Registration`
6. confirm live robot-frame pose
7. one-servo OpenRB bring-up
8. startup calibration and pretension
9. single-segment repeatability
10. Modeling dataset collection ("Random Data Collection" / `collect_pose_command_dataset`)
11. Two-segment startup validation and dataset collection
12. Two-segment modeling analysis only after trusted two-segment dataset labels exist

## Tracker Verdicts

Strict validation pass:

- all configured tracker thresholds pass

Operational with warning:

- tracker frames, tool visibility, and rigid transforms are good enough to use
- one strict target, usually FPS, is below the configured threshold
- this is a warning state, not the same as a hard tracker failure

Hard failure:

- tracker is not connected, no frames arrive, required tools are missing, transforms are invalid, or data is stale
- do not proceed until the tracker returns to an operational state

## Workflow 10: Two-Segment Repeatability (Bonus / Scaffold)

Applies in `dual_segment` mode after the all-8 startup artifact exists. The
experiment is `two_segment_repeatability`. It is **open-loop** and provides a
structured target set: center, inner/outer rings on bottom alone, inner/outer
rings on top alone, and a few combined targets. Each target is revisited
multiple times. Per-target scatter and per-run aggregate RMS are reported for
both distal and intermediate poses (when intermediate role is available).

Honest interpretation:

- This is **not** validated closed-loop control accuracy. The result is the
  repeatability of open-loop tendon commands plus the tracking pipeline.
- No <1 mm thesis target is hardcoded. Configure
  `target_distal_rms_mm` / `target_intermediate_rms_mm` to record an operator
  acceptance criterion; the run reports whether measurements meet it but does
  not auto-fail.
- Lower-trust runs are allowed via `allow_servo_only_test_run=true`.

Outputs:

- `two_segment_repeatability_summary.txt`
- `two_segment_repeatability_scatter_metrics.json`
- `two_segment_repeatability_per_target.csv`
- `two_segment_repeatability_distal_scatter.png`
- `two_segment_repeatability_per_target_rms.png`

## Workflow 10b: Mike Constant-Curvature Convention Probe (Evidence Before Flipping the Flag)

The Mike CC physics adapter stays `unavailable_unvalidated_convention` until
`physics_models.mike_constant_curvature.required_conventions_confirmed: true`
is set in the modeling config. Before flipping that flag, run the probe to
check that the predicted distal XYZ direction and magnitude agree with the
bench:

```bash
.venv/bin/python -m continuum_robot.modeling.two_segment.validate_mike_cc \
    --runs data/experiments/two_segment_collect_pose_command_dataset/<run> \
    --config config/modeling_two_segment.example.yaml \
    --output-dir data/experiments/two_segment_mike_convention_probe/<probe>
```

Inputs:
- A small `two_segment_collect_pose_command_dataset` run. `bottom_only_sweep`
  or `workspace_coverage` with conservative amplitude is ideal — it gives
  per-axis sign signal.
- The same modeling config you intend to use for ANN/Mike comparison.

What the probe checks:
- Sign of predicted vs measured distal X/Y/Z over all samples. A flipped sign
  in any axis is a strong "your conventions are off" signal.
- Magnitude of the predicted-vs-measured distal residual (mean, p95, max).
- Per-sample residuals so you can spot one bad capture vs systematic bias.

Outputs:
- `mike_cc_convention_report.json` — full numeric report.
- `mike_cc_convention_report.txt` — human-readable summary.

The probe never edits the config. After reading the report, the operator
decides whether to set `required_conventions_confirmed: true`. Treat the
"safe_to_confirm" recommendation as a sufficient — not exhaustive — check;
ANN training/comparison still surfaces measured-vs-predicted residuals.

Exit code is 0 when `recommendation` is `conventions_consistent_with_evidence_safe_to_confirm`
and 2 otherwise — useful when wiring into CI/smoke gates later.

## Workflow 11: 1 Mbps Servo Bus Migration (Optional / Two-Segment Throughput)

Two-segment work uses 8 servos; the default 57 600 baud is enough for slow
sweeps but limits the maximum sample rate. The repository supports raising the
DYNAMIXEL bus to 1 Mbps once every servo is reflashed at that baud. Do not flip
the configured baud without reflashing — the OpenRB will only see servos at
their currently-flashed baud and will report "no servos responded".

Migration checklist:

1. Power-down the rig.
2. With DYNAMIXEL Wizard or a known-good tool, set each of the 8 servos to
   1 000 000 baud, one at a time. Confirm each servo's baud setting is saved
   to EEPROM and survives a reboot.
3. Update the project config to match. Either edit `config/system.yaml`:

   ```yaml
   baudrate: 1000000
   ```

   …or place a local override in `config/system.local.yaml`:

   ```yaml
   baudrate: 1000000
   ```

4. Run the transport diagnostic on the full bus to confirm latency:

   ```bash
   .venv/bin/python -m continuum_robot.diagnostics.servo_transport_diagnostic \
     --servo-ids 1,2,3,4,5,6,7,8 \
     --baud 1000000 \
     --duration 10 \
     --read-rate-hz 20 \
     --fields minimal
   ```

   Watch:
   - `success_count_by_servo`: should equal the achieved sample count for every servo
   - `mean_read_duration_ms_by_servo`: should drop versus 57 600 baseline
   - `failure_count_by_type`: should remain zero or very small
5. Re-run `two_segment_startup_validation` to record an all-8 manual startup at
   the new baud.
6. Run a tiny `two_segment_collect_pose_command_dataset` with
   `schedule_type=zero` and `dry_run=false` to confirm read/write at the new
   baud before doing any motion.

Roll back to 57 600 the same way (DYNAMIXEL Wizard first, then config).

## Workflow 12: 20-Second Sci-Fi Spine Video Demo

Demo-only open-loop waypoint relay for slide-video recordings. The
relay sends a new waypoint before the spine fully settles at the
previous one, producing a "drift, almost reach, change my mind"
motion that reads as creepy / organic on camera. NOT data, NOT
closed-loop, NOT thesis evidence — every artifact stamps
`demo_only=True`, `closed_loop_control=False`,
`valid_for_model_training=False`,
`valid_for_thesis_repeatability=False`.

### One-click preset

GUI: Experiments → **Two-Segment Slow Motion Demo** → click **20s
Sci-Fi Spine** under "Presets". That fills:

| Field | Value |
|---|---|
| Pattern | `sci_fi_waypoint_relay` |
| Video duration | 20.0 s |
| Waypoint count | 9 |
| Amplitude | 0.35 cm |
| Early switch fraction | 0.72 (metadata; relay timing comes from duration/count) |
| Waypoint source | `preset_weird` |
| Auto-select seed | true |
| Profile velocity | 45 |
| Profile acceleration | 18 |
| Command rate (bookkeeping) | 10 Hz |
| Ramp in / out | 1.0 / 1.0 s |
| Hold start / end | 1.0 / 1.5 s |
| Dry run | **true** (operator flips off to record) |

### Recommended first live settings

Always run dry-run first. The preview card shows waypoint count,
seconds per waypoint, advisory warnings. First live pass should be
the conservative defaults above (0.35 cm). If you want the smallest
possible safe motion, drop amplitude to 0.10 or 0.25 cm before
trying 0.35.

### Tuning guide

If motion looks **too slow**:
- raise `profile_velocity` (try 60 → 90, watch current)
- reduce `video_duration_s` (try 15 s)
- raise amplitude carefully (≤ 0.50 cm)

If motion looks **too jerky** or hits the soft tick cap:
- lower amplitude (try 0.25 cm)
- lower `profile_velocity` (try 30)
- raise `profile_acceleration` carefully so each new goal ramps in
  smoothly
- try a different `relay_seed` (auto-select usually picks a good one
  but specific seeds can have one short segment)

If motion **pauses too long** between drifts (the relay should
overlap, not stop and start):
- lower `profile_velocity` so the spine is genuinely mid-motion when
  the next command arrives
- raise `waypoint_count` (try 12) to shorten each segment
- enable `auto_select_seed` so consecutive waypoints aren't clustered

### Hardware checklist (every live run)

- operating mode: `dual_segment`
- all-8 startup artifact accepted (`two_segment_startup_validation` ran)
- bottom/top physical assembly confirmed
- bus at **1 Mbps** for cleaner profile-velocity behaviour
- current / load proxy visible on the GUI
- dry-run preview rendered without warnings
- first live pass at **0.10 or 0.25 cm** before 0.35
- stop button reachable; hand on the e-stop

### Output bundle

Each run writes a normal slow-motion-demo bundle plus four
sci-fi-specific files:

- `sci_fi_waypoint_relay_summary.txt` — operator-facing summary
- `waypoints.json` — resolved 4D waypoint list + resolved seed
- `waypoint_schedule.csv` — issue_time_s + waypoint_index rows
- `sci_fi_waypoint_preview.png` — bottom + top XY waypoint preview
- `sci_fi_servo_goal_trace.png` — per-servo goal-tick timeline

The validator FAILs if any of these are missing on a relay run,
WARNs on amplitude > 0.50 cm, and WARNs loudly if
`profile_restore_success=False` (which would mean the bus was left
at demo speed — run a profile_restore / neutral jog before any
non-demo experiment).

## Legacy Surface

The Python legacy-bridge compatibility modules remain available only for explicit bridge fallback/debug work.
Normal operation should use the current `Tracking` plus `Registration` workflows.
