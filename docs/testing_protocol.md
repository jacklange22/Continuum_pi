# Testing Protocol

This is the shortest reliable validation sequence for the current repo. Use the unified wrapper first:

```bash
python scripts/run_lab_workflow.py list
```

## 1. Environment And Config

1. Bootstrap the repo:
   `PYTHON_BIN=python3 scripts/bootstrap.sh`
2. Copy local machine config if needed:
   `cp config/system.local.example.yaml config/system.local.yaml`
3. Confirm the intended `mock_mode`, `aurora_port`, `openrb_port`, and `robot_config`.

Pass criteria:

- `.venv` exists
- `config/system.local.yaml` matches the current machine
- `./.venv/bin/pytest -q` passes before hardware testing

## 2. Tracker Bring-Up

1. Run doctor:
   `python scripts/run_lab_workflow.py tracker-doctor`
2. Run smoke:
   `python scripts/run_lab_workflow.py tracker-smoke -- --tracker-port /dev/ttyUSB0`
3. Run benchmark:
   `python scripts/run_lab_workflow.py tracker-benchmark -- --tracker-port /dev/ttyUSB0`

Pass criteria:

- Preferred backend starts or fallback is explicitly reported
- Live tool ids resolve to runtime roles `0A` and `0B`
- Tracker smoke reports `tracker_ready=True`
- Benchmark passes configured FPS and stale-data thresholds

## 3. OpenRB / Servo Bring-Up

1. Launch the GUI:
   `python scripts/run_lab_workflow.py gui`
2. In `System`, save runtime parameters for the active robot config and OpenRB port.
3. Connect OpenRB, then scan one servo in `Servos`.
4. Verify telemetry before motion.
5. Use fine jog only first.

Pass criteria:

- OpenRB validates and prepares cleanly
- DYNAMIXEL bus connects
- Servo telemetry includes position, current, voltage, temperature, and no hardware error
- Fine jog succeeds through the GUI path

## 4. Startup Calibration And Pretension

1. Capture neutral setpoints in `Servos`.
2. Save startup calibration for the active servo.
3. Run pretension cautiously and accept only the reviewed result.

Pass criteria:

- Calibration artifact is saved under `data/calibrations/`
- Safe bounds and pretension threshold are populated
- Pretension exits cleanly on threshold, timeout, cancel, or a protected failure mode

## 5. Registration Validation

1. Validate existing saved registration output:
   `python scripts/run_lab_workflow.py registration-runtime-sanity -- --live --tracker-port /dev/ttyUSB0`
2. Validate legacy CSV or session artifacts when comparing workflows:
   `python scripts/run_lab_workflow.py registration-validation -- --session-json path/to/session.json`
3. Solve from CSV if needed:
   `python scripts/run_lab_workflow.py registration-from-csv -- path/to/registration.csv`

Pass criteria:

- `tip_pose_status` is valid after loading registration
- Runtime coil tool id matches the saved registration
- Validation metrics stay inside the expected FRE bounds

## 6. Experiment Runs

1. Use the GUI for operator-led runs, or the CLI for scripted runs:
   `python scripts/run_lab_workflow.py experiment -- --help`
2. Run `pivot_calibration` before live registration when the tip file changes.
3. Run `aurora_grid_accuracy` before full robot experiments if tracker quality is in doubt.
4. Run `repeatability_dataset` only after tracker, calibration, and registration are clean.

Pass criteria:

- Every run writes `metadata.json`, `summary.json`, and samples under `data/experiments/`
- Tracker-driven pivot review datasets land under `data/experiments/pivot/runs/`
- Tracker-driven raw pivot capture CSVs land under `data/experiments/pivot/captures/`
- Preflight shows no blockers
- Result metrics are readable and reloadable

## 7. Regression Test Gate

Run this before and after hardware-focused edits:

```bash
./.venv/bin/pytest -q
```

For GUI-specific work:

```bash
./.venv/bin/pytest -q tests/test_gui_controllers.py tests/test_gui_bootstrap.py
```

## Notes

- Prefer the wrapper over individual scripts for normal lab use.
- Keep `references/` and `tools/` read-only.
- Treat tracker doctor, tracker smoke, servo telemetry validation, and registration sanity as hard gates before repeatability experiments.
