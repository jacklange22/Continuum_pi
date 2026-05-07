# Hardware-Day Runbook

This is the short operator workflow for the next Pi / OpenRB / Aurora hardware session.

## 1. Pre-Launch Checklist

Run from the repo root on the Pi:

```bash
git status --short
scripts/run_tests.sh quick
.venv/bin/python -m continuum_robot.diagnostics.prehardware_dry_run
```

Confirm:

- `config/system.local.yaml` points at the intended OpenRB and Aurora ports.
- Normal hardware profile is `robot_8servo.yaml`.
- Operating mode matches the task: `one_servo`, `single_segment`, `dual_segment`, or `parallel_single`.
- `data/experiments/` and `data/exports/` are writable.

## 2. Startup Workflow

1. Launch the GUI with `scripts/run_gui.sh`.
2. On System, select the operating mode.
3. For `single_segment`, choose Segment A `[1,2,3,4]` or Segment B `[5,6,7,8]`.
4. Apply settings explicitly.
5. Connect OpenRB and verify expected servo IDs match the selected mode.
6. Connect tracker and verify live tool status before trusting pose-dependent experiments.

## 3. Tracking And Registration

1. Verify 0A and 0B are live.
2. Check or load the accepted registration artifact.
3. Check runtime tip policy.
4. Remember: `coil_as_tip` means the 0A coil origin / position path in robot frame, not a separately calibrated physical tip offset.

## 4. Servo And Pretension

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

## 5. Experiment Workflow

- Repeatability: use only trusted tracker/registration/runtime-tip state for thesis claims.
- Collect-pose / babble: no-tracker servo-only runs are hardware/debug runs, not model-training data.
- Penprobe chasing demo: `single_segment` only; 0B target wording depends on whether a pivot-calibrated tool tip is actually active.
- `parallel_single` is mirrored single-segment babble/testing only, not full two-segment kinematics.

## 6. Export Workflow

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

## 7. Do-Not-Trust Warnings

- Servo-only/no-tracker runs are not model-training valid.
- Lower-trust runs are not thesis-valid repeatability evidence.
- `success=true` does not imply thesis validity; inspect `run_trust_mode`, `valid_for_model_training`, and `valid_for_thesis_repeatability`.
- Debug/dashboard figures are for review; use `_report.png` figures for thesis and slides.
