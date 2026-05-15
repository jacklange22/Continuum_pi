# Hardware-Day Runbook

This is the short operator workflow for the next Pi / OpenRB / Aurora hardware session.

## 1. Pre-Launch Checklist

Run from the repo root on the Pi:

```bash
git status --short
scripts/run_tests.sh quick
scripts/run_tests.sh hardware-safe
.venv/bin/python -m continuum_robot.diagnostics.prehardware_dry_run
.venv/bin/python -m continuum_robot.diagnostics.hardware_readiness_check
```

Confirm:

- `config/system.local.yaml` points at the intended OpenRB and Aurora ports.
- Normal hardware profile is `robot_8servo.yaml`.
- Operating mode matches the task: `one_servo`, `single_segment`, `dual_segment`, or `parallel_single`.
- `data/experiments/` and `data/exports/` are writable.
- `prehardware_dry_run` passes the no-hardware export/validator/operator fixture checks.
- `hardware_readiness_check` is PASS overall, or any WARN item has a clear hardware-day action.

## 2. Startup Workflow

1. Launch the GUI with `scripts/run_gui.sh`.
2. On System, select the operating mode.
3. For `single_segment`, choose Segment A `[1,2,3,4]` or Segment B `[5,6,7,8]`.
4. Apply settings explicitly.
5. Connect OpenRB and verify expected servo IDs match the selected mode.
6. Connect tracker and verify live tool status before trusting pose-dependent experiments.

## 3. Wednesday Single-Segment First Workflow

Current hardware-day default is `single_segment`, Segment B, servo IDs `[5,6,7,8]`.
The other segment being disconnected is informational only in this mode.

1. Before connecting hardware, confirm `mock_mode=false` in `config/system.local.yaml`.
2. Select `single_segment` on System.
3. Select Segment B `[5,6,7,8]` and apply settings unless the hardware has been deliberately rerouted.
4. Connect OpenRB.
5. On Servos, confirm the expected four active servo IDs respond.
6. Keep torque on unless intentionally disabling it. Normal GUI close/reload/disconnect should not be used as a torque-off control.
7. Use tiny manual jogs to confirm servo sign and tendon routing before any automated motion.
8. Do not command a move-to-4095 untensioned reference while tendons are attached.
9. If any servo position is near 0 or 4095, stop, manually reset the spool if needed, and re-capture neutral/startup.
10. Capture the Segment B sign/mapping checklist:
    - `5 = +x`, `6 = +y`, `7 = -x`, `8 = -y`
    - `axis_a = 5/7`, `axis_b = 6/8`
    - lower ticks mean more tension / tendon shortening
    - use `Confirm Configured Mapping` on Servos once the hardware matches this configured mapping
11. Capture valid neutral/safe bounds before calibrated motion.
12. Connect tracker.
13. Confirm 0A / runtime-tip visibility and load or validate registration if the run needs pose trust.
14. Do not use automatic pretension until sign/mapping and neutral/safe bounds pass.
15. When ready, run conservative `pretension_validation` from `current_position` mode only.
16. Repeat pretension validation three times before thesis-style repeatability claims.
17. Run a tiny `single_segment_repeatability` or `collect_pose_command_dataset` session.
18. Validate/export the run bundle from Data, then mark the run review status.

Stop immediately if:

- the wrong servo moves
- a tendon unwinds or visibly loses routing
- the spine bends aggressively
- position telemetry is missing
- no-status or incorrect-status packets repeat
- tracker 0A/runtime-tip data is stale or missing for pose-dependent runs

Pretension hardware ladder:

1. Manual visual check with torque off and no automated motion.
2. Current-only / servo-only characterization if tracking is unavailable.
3. Tracker-enabled conservative pretension with tiny steps and low travel caps.
4. Repeat pretension three times and compare final positions, load proxy spread, stop reasons, and final XY error when tracking is available.
5. Export all runs before changing hardware or routing.

## 4. Tracking And Registration

1. Verify 0A and 0B are live.
2. Check or load the accepted registration artifact.
3. Check runtime tip policy.
4. Remember: `coil_as_tip` means the 0A coil origin / position path in robot frame, not a separately calibrated physical tip offset.

## 5. Servo And Pretension

1. Start with manual jog/readiness checks.
2. Use manual startup capture as the fallback baseline.
3. For automatic pretension, use `single_segment` only.
4. Start with current-only or very small-travel validation before full tracker-enabled pretension.
5. Treat current as servo-reported current estimate / load proxy, not tendon force.

Stop reasons to take seriously:

- `stale_telemetry`
- `packet_retry_budget_exhausted`
- `partial_pair_failure`
- `tip_response_wrong_direction`
- `safety_limit_rejected`
- `current_noise_too_high`

## 6. Experiment Workflow

- Repeatability: use only trusted tracker/registration/runtime-tip state for thesis claims.
- Collect-pose / babble: no-tracker servo-only runs are hardware/debug runs, not model-training data.
- Penprobe chasing demo: `single_segment` only; 0B target wording depends on whether a pivot-calibrated tool tip is actually active.
- Penprobe chasing demo is a hardware demo only: the 0A coil origin chases 0B in XY using the active single-segment pairs. Start with max step 25 ticks/cycle, then 50, then at most 100 after sign/mapping is confirmed. Stop on stale tracker, wrap risk, servo hardware error, or persistent saturation at the startup cap.
- Background readiness uses cached telemetry during active experiments. Use explicit `Refresh Readiness` or `Discover / Read Servo` when you need a fresh bus read.
- `parallel_single` is mirrored single-segment babble/testing only, not full two-segment kinematics.
- For a two-segment bench day, keep [`docs/two_segment_bench_day_quickref.md`](two_segment_bench_day_quickref.md) open. It's a one-page cheat sheet for the full pipeline (startup → babble → Mike CC probe → repeatability → modeling → handoff) including what to watch for in the GUI chips at each stage.
- `dual_segment` mode is the true two-segment foundation. Before any two-segment work:
  1. Confirm the `physical_assembly` block in `config/robot_8servo.yaml` matches the rig — which fixed segment is at the bottom and which at the top.
  2. The GUI experiment-tab summary will show e.g. `Bottom: Segment A [1,2,3,4], Top: Segment B [5,6,7,8]`.
  3. Run `two_segment_startup_validation` (stages: baseline → bottom_pretensioned → top_pretensioned → bottom_recheck → final_accept).
  4. Only then run `two_segment_collect_pose_command_dataset` or `two_segment_repeatability`.
- Two-segment kinematics control, automatic two-segment pretension, and two-segment penprobe chasing are NOT implemented. The foundation is data/metadata only.
- The 1 Mbps baud migration is optional. Default 57 600 is fine for slow collection; raise to 1 000 000 only after every servo is reflashed (see `docs/operator_workflows.md` Workflow 11).

1 Mbps all-8 transport diagnostic:

```bash
python -m continuum_robot.diagnostics.servo_transport_diagnostic \
  --port /dev/ttyACM0 \
  --baud 1000000 \
  --servo-ids 1,2,3,4,5,6,7,8 \
  --duration 30 \
  --read-rate-hz 10 \
  --fields minimal
```

- If `bus_ready_for_parallel_single=true`, the all-8 bus profile is ready for `parallel_single` demo runs.
- If one servo reports repeated failures, inspect that servo’s cable/connector path first.
- If voltage dips are reported, inspect power headroom and mechanical/tendon loading before larger motions.

## 7. Export Workflow

CLI export:

```bash
.venv/bin/python -m continuum_robot.data.export_run_bundle --latest single_segment_repeatability --zip
.venv/bin/python -m continuum_robot.data.export_run_bundle --latest pretension_validation --include-samples --zip
```

Validate a run folder:

```bash
.venv/bin/python -m continuum_robot.data.validate_run_bundle data/experiments/<experiment>/<run_folder>
```

Data tab export:

1. Select a run.
2. Use `Validate Selected Run` to check trust/provenance completeness.
3. Mark the run as `thesis_candidate`, `advisor_share`, `debug`, or `garbage`.
4. Use `Export Selected Run` or `Export Latest Run`.
5. Keep `Zip` enabled for advisor/Mac handoff.
6. Use `Copy Transfer Command` or inspect `transfer_commands.txt` in the bundle.

CLI review and cleanup:

```bash
.venv/bin/python -m continuum_robot.data.manage_runs --mark data/experiments/<experiment>/<run_folder> --status thesis_candidate
.venv/bin/python -m continuum_robot.data.manage_runs --archive data/experiments/<experiment>/<run_folder>
.venv/bin/python -m continuum_robot.data.manage_runs --trash data/experiments/<experiment>/<run_folder>
.venv/bin/python -m continuum_robot.data.build_thesis_evidence_index
```

Use archive/trash instead of permanent deletion. Archive moves runs to `data/experiments_archived/`; trash moves runs to `data/trash/`. Curated important runs in `data/experiments/` and `data/experiments_archived/` can be committed when needed for thesis reproducibility or Pi/Mac handoff. Do not commit `data/exports/`, `data/trash/`, logs, temporary diagnostics, or generated zip bundles.

Before committing data, run:

```bash
scripts/check_data_for_git.py
```

GitHub rejects regular Git files over 100 MB. Use Git LFS or another transfer method for very large raw samples. Export bundles are the preferred quick handoff unit, but they should usually remain ignored rather than committed.

Transfer from the Mac:

```bash
rsync -av <pi-user>@<pi-host>:/path/to/pi_code/data/exports/<bundle>.zip ~/Downloads/
```

## 8. Do-Not-Trust Warnings

- Servo-only/no-tracker runs are not model-training valid.
- Lower-trust runs are not thesis-valid repeatability evidence.
- `success=true` does not imply thesis validity; inspect `run_trust_mode`, `valid_for_model_training`, and `valid_for_thesis_repeatability`.
- Debug/dashboard figures are for review; use `_report.png` figures for thesis and slides.
