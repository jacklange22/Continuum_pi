# Continuum Robot Operator Platform

This repository is the Raspberry Pi operator stack for the continuum robot. The intended live runtime is:

- Aurora connected directly to the Pi
- OpenRB/DYNAMIXEL connected directly to the Pi
- the Python GUI running locally on the Pi

The current live tracking path is Python-native:

- Aurora hardware
- `scikit-surgerynditracker.NDITracker`
- [TrackerBackendNDI](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py)
- [TrackingService](/Users/jacklange/Continuum/pi_code/continuum_robot/services/tracking_service.py)
- diagnostics, benchmark, registration, GUI, and later servo control

`tracker_bridge` remains in the repo only as a legacy compatibility/comparison path. It is not the default live backend.

## Core Conventions

Transform convention is strict throughout this repo:

- `T_A_B` means transform coordinates from frame `B` into frame `A`
- `T_A_C = T_A_B @ T_B_C`

Runtime tool roles:

- `0A`: coil tool used for runtime tip pose
- `0B`: probe/measurement tool used for registration

Current Raspberry Pi Aurora mapping:

- raw live id `10` maps to runtime role `0A`
- raw live id `11` maps to runtime role `0B`

Tip pose chain after registration:

- `T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip`

Legacy registration compatibility is preserved:

- `T_aurora_2_model`
- `T_tip_2_coil`
- `T_coil_tip`

## Repository Layout

- `continuum_robot/`: Python app, services, tracking, registration, GUI, config loading
- `config/`: runtime YAML configuration
- `docs/`: migration notes and operator-facing design notes
- `scripts/`: bootstrap, diagnostics, benchmark, validation, launch helpers
- `tests/`: unit and mock-backed integration coverage
- `data/`: runtime outputs for registrations, captures, runs, logs, calibrations
- `tracker_bridge/`: legacy C++ Aurora bridge, retained for comparison only
- `references/`: read-only legacy reference material
- `tools/`: read-only registration assets and lab inputs

Do not modify `references/` or `tools/` unless you intentionally want to change protected reference material.

## Project Docs

The current project specification and phased validation docs live here:

- [system_spec.md](/Users/jacklange/Continuum/pi_code/docs/system_spec.md)
- [architecture.md](/Users/jacklange/Continuum/pi_code/docs/architecture.md)
- [operator_workflows.md](/Users/jacklange/Continuum/pi_code/docs/operator_workflows.md)
- [tracker_mvp_workflow.md](/Users/jacklange/Continuum/pi_code/docs/tracker_mvp_workflow.md)
- [testing_protocol.md](/Users/jacklange/Continuum/pi_code/docs/testing_protocol.md)
- [validation_plan.md](/Users/jacklange/Continuum/pi_code/docs/validation_plan.md)
- [registration_trace.md](/Users/jacklange/Continuum/pi_code/docs/registration_trace.md)
- [servo_interface_contract.md](/Users/jacklange/Continuum/pi_code/docs/servo_interface_contract.md)
- [servo_workflow_trace.md](/Users/jacklange/Continuum/pi_code/docs/servo_workflow_trace.md)

## Live Tracking Architecture

Primary live tracking path:

- [bootstrap.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/bootstrap.py) wires the configured tracker backend into the app context
- [backend_router.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/backend_router.py) is the single live-backend selector and fallback policy seam
- [ndi_backend.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py) owns `NDITracker`, live polling, raw-id normalization, payload extraction, and first-frame debug instrumentation
- [tracking_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/tracking_service.py) is the app-visible source of truth for tool state, freshness, faults, and `T_robot_tip`
- [diagnostics.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/diagnostics.py) owns staged validation, failure classification, and doctor reports
- [run_tracker_doctor.py](/Users/jacklange/Continuum/pi_code/scripts/run_tracker_doctor.py), [run_tracker_smoke.py](/Users/jacklange/Continuum/pi_code/scripts/run_tracker_smoke.py), and [run_tracker_benchmark.py](/Users/jacklange/Continuum/pi_code/scripts/run_tracker_benchmark.py) all consume `TrackingService`
- [registration_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/registration_service.py) consumes live `0B`
- [tip_pose_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tip_pose_service.py) consumes live `0A` plus saved registration

Architecture diagram:

```text
Aurora USB/serial
  -> TrackingBackendRouter
      -> TrackerBackendNDI (preferred)
      -> TrackerServiceManager / tracker_bridge (fallback/debug only)
  -> TrackingService
      -> GUI Tracking/System panels
      -> RegistrationService
      -> ExperimentRunner
      -> tracker doctor / smoke / benchmark
```

What the backend does:

- opens Aurora through `scikit-surgerynditracker`
- polls `NDITracker.get_frame()`
- keeps raw live ids for debug
- maps raw ids into app runtime roles
- extracts pose data from common live payload forms
- validates rigid transforms strictly
- publishes `tracked`, `missing`, `invalid`, or `unknown`
- logs only the first few raw payload summaries to stderr for bring-up debugging

Canonical backend states:

- `disabled`
- `mock`
- `connecting`
- `streaming_healthy`
- `streaming_degraded`
- `disconnected`
- `error`

## Hardware And Software Prerequisites

Hardware:

- Raspberry Pi with local display/input or remote desktop workflow
- Aurora SCU connected by USB/serial
- tracked tools visible to Aurora
- OpenRB-150 and servos only if you are validating servo/runtime paths later

System packages:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv build-essential
```

Python/runtime dependencies:

- Python 3.10+
- `numpy`
- `PyYAML`
- `pyserial`
- `dynamixel-sdk`
- `PySide6`
- `scikit-surgerynditracker`

Important:

- this repo installs the Python package dependency, but it does not bundle or auto-detect whatever low-level vendor/runtime dependencies your local `scikit-surgerynditracker` install needs
- live servo access uses the Robotis Python `dynamixel_sdk` module provided by the `dynamixel-sdk` package
- if `tracker_backend: "bridge"` is used instead, you also need the external NDI SDK and the legacy C++ bridge build

## Bootstrap

From the repo root:

```bash
PYTHON_BIN=python3 scripts/bootstrap.sh
```

What bootstrap does:

- creates `.venv/`
- installs the package and dev dependencies
- creates runtime directories under `data/`
- optionally builds `tracker_bridge` when `BUILD_TRACKER_BRIDGE=1`

Unified workflow wrapper:

```bash
python scripts/run_lab_workflow.py list
```

Use that wrapper for the common GUI, tracker, registration, and experiment commands instead of memorizing each script path.

Focused tracker-first launcher for tomorrow's Pi session:

```bash
python scripts/run_tracker_mvp.py
```

That launcher now opens the permanent `Tracking` tab. The permanent GUI split is:

- `Tracking`: Aurora connect, validation, live tool state, staged `0B` pivot calibration, and accepted tip review
- `Registration`: 4-point landmark capture, solve/review/save, and live robot-frame pose summary
- `Tracker Legacy`: compatibility/diagnostic copy of the old combined workspace

Current tracker-first MVP operator sequence:

1. connect Aurora
2. validate tracker health and transforms for `0A` / `0B`
3. collect, solve, review, and accept `0B` pivot calibration
4. capture, solve, review, and save 4-point registration
5. confirm live robot-frame pose

The simple 4-point registration config in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml) now uses candidate landmarks derived from the protected lab model points in `tools/12_model_registration_points_in_sw` transformed by `tools/T_sw_2_model`.

## Live Aurora Config

Copy the local config template:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

For live Aurora on the Pi, the important fields are:

```yaml
mock_mode: false
visualization_mode: "auto"
visualization_safe_effects: true
aurora_port: "/dev/ttyUSB0"
tracker_backend: "ndi"
tracker_fallback_backend: "bridge"
tracker_fallback_enabled: true
tracker_type: "aurora"
tracker_poll_ms: 20
tracker_freshness_timeout_s: 0.5
tracker_tool_id_aliases:
  "10": "0A"
  "11": "0B"
```

Notes:

- backend selection is explicit: the router tries `tracker_backend` first and only tries `tracker_fallback_backend` if fallback is enabled and the primary backend is unavailable or fails during startup
- fallback is always recorded in startup messages and diagnostics; it is never silent
- `visualization_mode: "auto"` uses the safest renderer for the current platform. On macOS the GUI stays on the projection fallback by default because QtDataVisualization is unstable there.
- `visualization_safe_effects: true` keeps the native 3D path on conservative settings and disables risky QtDataVisualization effects.
- raw live ids `10` and `11` are the current observed Aurora tool ids on the Pi
- the app runtime still uses `0A` and `0B`
- `system.local.yaml` overrides `system.yaml`

## Live OpenRB / DYNAMIXEL Config

For real OpenRB and servo bring-up, the important local config fields are:

```yaml
mock_mode: false
robot_config: "robot_1servo.yaml"
openrb_port: "/dev/ttyUSB1"
openrb_settings:
  connect_timeout_s: 0.5
  port_settle_time_s: 0.15
  require_usb_to_dynamixel_firmware: true
  require_external_power_for_motion: true
dynamixel_settings:
  protocol_version: 2.0
  positive_tick_rotation: "ccw"
  expected_operating_mode: 3
  allowed_operating_modes: [3]
  auto_torque_enable_on_write: true
  torque_disable_for_eeprom_write: true
  require_current_for_motion: true
  require_voltage_for_motion: true
  require_temperature_for_motion: true
  voltage_scale_mv_per_unit: 100.0
  current_scale_ma_per_unit: 1.0
  control_table:
    operating_mode: 11
    current_limit: 38
    max_position_limit: 48
    min_position_limit: 52
    servo_id: 7
    torque_enable: 64
    hardware_error_status: 70
    goal_position: 116
    present_current: 126
    present_position: 132
    present_input_voltage: 144
    present_temperature: 146
safety_overrides:
  fine_jog_step_ticks: 5
  coarse_jog_step_ticks: 25
  default_pretension_current_threshold_ma: 220
```

Current hardware assumptions:

- the OpenRB is reachable as a USB serial device from the Pi
- the OpenRB is running the Robotis `usb_to_dynamixel` bridge firmware, or equivalent pass-through firmware
- the OpenRB DYNAMIXEL side is externally powered for real movement tests
- the XC330/XC333 servos use the X-series control table addresses shown above
- the servos are already in a position-control-compatible operating mode for goal-position writes
- tightening direction is configured explicitly per servo instead of assumed implicitly

Permissions / serial notes:

- the current user must be able to open the OpenRB serial port
- if the OpenRB enumerates as a different path, update `openrb_port` in `config/system.local.yaml`
- if another process already owns the serial device, both OpenRB validation and DYNAMIXEL SDK access will fail

Environment overrides for visualization safety:

```bash
export CONTINUUM_VISUALIZATION_MODE=auto        # auto | 2d | placeholder | 3d
export CONTINUUM_VISUALIZATION_SAFE_EFFECTS=1   # 1 keeps conservative native-3D settings
```

## Recommended Order Of Operations

1. Verify Python environment and config
2. Run tracker doctor to verify backend selection and startup preflight
3. Run tracker smoke to verify pre-registration tracking readiness
4. Run the tracker benchmark for timing and freshness
5. Do one-servo OpenRB bring-up with external power
6. Save startup calibration and run cautious pretension on one servo
7. Perform registration and create `data/registrations/latest_registration.json`
8. Run registration readiness validation to confirm `T_robot_tip`
9. Launch the full GUI/app

## Stage 1: Tracker Bring-Up Without Registration

Tracker doctor:

```bash
.venv/bin/python scripts/run_tracker_doctor.py --tracker-port /dev/ttyUSB0
```

What good output looks like:

- `selected_backend=ndi`
- `backend_identity=ndi_tracker_python`
- `tracker_ready=True`
- `full_pose_pipeline_ready=False`
- `Stage 1` through `Stage 4` pass
- `Stage 5` reports `pending` when registration has not been done yet
- `raw_live_tool_ids=['10', '11']`
- `normalized_live_tool_ids=['0A', '0B']`
- `runtime_role_mappings={'0A': '10', '0B': '11'}`

Pre-registration smoke test:

```bash
.venv/bin/python scripts/run_tracker_smoke.py --tracker-port /dev/ttyUSB0
```

This is the quickest acceptance check before touching registration. It exits success when the tracker is healthy even if registration is still pending.

Tracker benchmark:

```bash
.venv/bin/python scripts/run_tracker_benchmark.py --tracker-port /dev/ttyUSB0 --duration-s 5
```

What good output looks like:

- `Configured backend: ndi`
- `Selected backend: ndi`
- `Canonical state: streaming_healthy` or `streaming_degraded`
- `Unique frames observed` is greater than zero
- `Backend frame counter` is greater than zero
- raw/normalized ids and runtime mappings are populated
- tracked counts for `0A` and `0B` are greater than zero
- failures list is empty

Benchmark behavior note:

- the benchmark now waits briefly for the first live frame before starting the timed sample window so it does not falsely report zero frames while the backend is still starting

## Stage 2: Registration Workflow

Registration is expected to happen only after tracker-only bring-up is healthy.

Primary operator workflow:

1. launch the GUI
2. open the `Registration` tab
3. start a 4-point session
4. capture one or more `0B` samples for `L1`, then mark the point complete
5. repeat for `L2`, `L3`, and `L4`
6. solve the registration, review RMSE/FRE, then save the accepted result

Expected output file:

- `data/registrations/latest_registration.json`

Registration validation from a saved Aurora CSV remains available:

```bash
.venv/bin/python scripts/run_registration_validation.py \
  --registration-csv references/RegistrationPoints.csv \
  --save-report data/logs/registration_validation.json
```

Saved CSV-to-registration flow remains available:

```bash
.venv/bin/python scripts/run_registration_from_csv.py \
  references/RegistrationPoints.csv \
  --output-dir data/registrations
```

## Stage 3: Runtime Tip-Pose Validation After Registration

Registration readiness validation:

```bash
.venv/bin/python scripts/run_tracker_smoke.py \
  --tracker-port /dev/ttyUSB0 \
  --require-registration \
  --registration-file data/registrations/latest_registration.json
```

What good output looks like:

- `tracker_ready=True`
- `full_pose_pipeline_ready=True`
- `Stage 5: T_robot_tip computable: passed`

Deeper runtime tip-pose debugging remains available:

```bash
.venv/bin/python scripts/run_registration_runtime_sanity.py --live --tracker-port /dev/ttyUSB0
```

If registration has not been done yet, failure is expected and should explicitly report `missing_registration`.

## Stage 4: Full App Launch

```bash
scripts/run_gui.sh
```

In the full app, the expected progression is:

1. `System` tab shows live backend/config identity
2. `Tracking` tab shows `0A` and `0B` updating
3. `Registration` tab guides the 4-point body-alignment process
4. after registration is loaded, `Tracking` can compute `T_robot_tip`
5. `System` + `Servos` can connect OpenRB, scan servo IDs, read telemetry, assign a test ID, and jog a servo carefully
6. experiments can build on the validated live tracking and servo paths

## Stage 5: OpenRB / DYNAMIXEL Bring-Up

Use the GUI as the canonical bring-up path:

1. launch `scripts/run_gui.sh`
2. in `System`, save one-servo bring-up parameters first
3. connect external power to the OpenRB / DYNAMIXEL side
4. in `System`, set the OpenRB port and click `Connect OpenRB`
5. confirm the prepared OpenRB status before touching the bus
6. open `Servos`
7. click `Scan Servos` to list responding servo IDs
8. verify telemetry values populate for model, firmware, mode, position, current, voltage, temperature, and error
9. if needed, assign a new ID on a test servo
10. use `Fine -/+` first, then `Coarse -/+`, and confirm the readback updates
11. save startup calibration for the active servo
12. run cautious pretension and accept the result only after reviewing the final current / position

Bring-up cautions:

- start with a single test servo whenever possible
- use external power for real movement tests
- verify the saved servo calibration artifact before larger motions
- the current GUI now shows whether calibration exists, whether it matches the current robot config, and whether per-servo bounds/thresholds are present
- motion is blocked when telemetry is stale/missing, the operating mode is wrong, the hardware error state is unsafe, or bounds are unavailable for the requested action

Servo calibration artifact:

- the servo subsystem now uses one canonical calibration file through the same `neutral_setpoints_path` config seam for backward compatibility
- the file stores per-servo neutral setpoint, safe min/max bounds, pretension/current threshold, tightening direction, pretension result, calibration timestamp, validity, and robot compatibility metadata
- older flat neutral-setpoint JSON files are still readable and are treated as legacy input to the richer artifact path

## Troubleshooting

### `missing_from_live_backend`

Meaning:

- the app runtime role exists (`0A` or `0B`), but the live backend did not publish a sample for it on that frame

Typical causes:

- tracker connected but tool not visible
- raw live ids were seen but not mapped into runtime roles
- backend returned no tool samples for that frame

Checks:

```bash
.venv/bin/python scripts/run_tracker_doctor.py --tracker-port /dev/ttyUSB0
```

Look for:

- `raw_live_tool_ids`
- `normalized_live_tool_ids`
- `runtime_role_mappings`

### `invalid_transform: missing quaternion/translation payload`

Meaning:

- the runtime received a tool sample marked as non-missing, but the backend did not publish a complete pose

Typical causes:

- live backend extraction bug
- `NDITracker.get_frame()` returning a payload form the backend does not yet decode
- the backend already marked the sample invalid and the service preserved that failure state

Checks:

- rerun diagnostics and inspect the first few stderr lines from `ndi_backend`
- those lines now include concise raw payload summaries for the first live frames
- if tools are visible but invalid, compare the payload summary with [ndi_backend.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py)

### `missing_registration`

Meaning:

- tracker bring-up may be fine, but `data/registrations/latest_registration.json` does not exist yet

This should not block:

- tracker diagnostics
- tracker benchmark
- tool visibility testing
- transform extraction debugging

It should block:

- runtime `T_robot_tip`
- registration-dependent experiment/runtime paths

Next step:

- complete the registration workflow and create `data/registrations/latest_registration.json`

### Benchmark sees zero frames while diagnostics sees frames

Meaning:

- there is a startup/warmup mismatch or the benchmark sampled before frames started advancing

Current mitigation in this repo:

- benchmark waits for the first live frame before starting the timed sample window
- benchmark and diagnostics both consume `TrackingService`
- benchmark reports final connection state, backend frame counter, raw ids, normalized ids, and role mappings

If this still happens:

1. run diagnostics first
2. confirm backend reaches `tracking`
3. confirm `backend_frames` increments
4. rerun benchmark immediately after

### Tracker connected but no tools

Meaning:

- Aurora connection is alive but no active tool samples are being returned

Checks:

- confirm tools are physically present and enabled in Aurora
- confirm the SCU sees both tools
- inspect `raw_live_tool_ids` from diagnostics

### Tools mapped but invalid

Meaning:

- raw ids are correct, but transform extraction is failing

Checks:

- inspect bounded stderr logs from `ndi_backend`
- verify whether the payload summary indicates a `4x4`, `N x 8`, named quaternion/translation fields, or an unsupported object form

## Exact Bring-Up Commands From Scratch

1. Install system packages:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv build-essential
```

2. Clone and enter the repo:

```bash
git clone <repo-url> ~/Continuum_pi
cd ~/Continuum_pi
```

3. Bootstrap:

```bash
PYTHON_BIN=python3 scripts/bootstrap.sh
```

4. Create local config:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

5. Edit `config/system.local.yaml` for live Aurora + OpenRB:

```yaml
mock_mode: false
aurora_port: "/dev/ttyUSB0"
openrb_port: "/dev/ttyUSB1"
tracker_backend: "ndi"
tracker_tool_id_aliases:
  "10": "0A"
  "11": "0B"
```

6. Verify tracker-only diagnostics:

```bash
.venv/bin/python scripts/run_tracker_doctor.py --tracker-port /dev/ttyUSB0
```

7. Run pre-registration smoke:

```bash
.venv/bin/python scripts/run_tracker_smoke.py --tracker-port /dev/ttyUSB0
```

8. Run tracker benchmark:

```bash
.venv/bin/python scripts/run_tracker_benchmark.py --tracker-port /dev/ttyUSB0 --duration-s 5
```

9. Perform registration in the GUI and create `data/registrations/latest_registration.json`:

```bash
scripts/run_gui.sh
```

10. Validate runtime tip pose after registration:

```bash
.venv/bin/python scripts/run_tracker_smoke.py \
  --tracker-port /dev/ttyUSB0 \
  --require-registration \
  --registration-file data/registrations/latest_registration.json
```

11. Launch the full operator app:

```bash
scripts/run_gui.sh
```

12. In the GUI, bring up OpenRB and the DYNAMIXEL bus:

- `System` -> `Connect OpenRB`
- `System` -> `Prepare OpenRB`
- `Servos` -> `Scan`
- verify telemetry for position / current / voltage
- jog a single test servo carefully

## Mock Mode

Mock mode remains available and should not be broken by live-backend changes.

In mock mode:

- backend identity is `mock_tracker_manager`
- tools `0A` and `0B` are synthetic but valid
- GUI, registration flow, and experiments can be exercised without Aurora hardware

## Canonical Experiment Framework

The experiment subsystem now uses one shared path instead of one-off scripts:

- [framework.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/framework.py) defines experiment lifecycle hooks and declared hardware requirements
- [schemas.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/schemas.py) defines canonical metadata, timeseries, and summary schemas
- [schedules.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/schedules.py) generates deterministic sweep, grid, trajectory, and babble schedules
- [dataset_io.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/dataset_io.py) writes and reloads canonical datasets
- [builtins.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/builtins.py) contains the generic built-in diagnostics and compatibility experiment
- [critical_experiments.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/critical_experiments.py) contains the project-critical first-class experiments
- [metrics.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/metrics.py), [pivot_utils.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/pivot_utils.py), and [validation.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/validation.py) provide shared analysis and status classification
- [experiment_runner.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/experiment_runner.py) is the canonical runner and compatibility seam for the current GUI
- [run_experiment.py](/Users/jacklange/Continuum/pi_code/scripts/run_experiment.py) is the canonical CLI entry path

Architecture:

```text
Experiment CLI / GUI
  -> ExperimentRunner
      -> registered experiment definition
      -> TrackingService / ServoService as needed
      -> canonical dataset writer
          -> metadata.json
          -> samples.jsonl
          -> summary.json
```

Every run writes one directory under `data/experiments/` containing:

- `metadata.json`
- `samples.jsonl`
- `summary.json`

### Project-Critical Experiments

- `pivot_calibration`
  - purpose: generate the pen-probe tip file before registration
  - prerequisites: no registration required; can run from an existing CSV or canonical dataset
  - inspect: `tip_vector_local_mm`, `rmse_mm`, `sample_count_used`, `sample_count_rejected`
  - outputs: canonical dataset bundle plus the configured tip vector file
- `aurora_grid_accuracy`
  - purpose: characterize Aurora tracker bias, RMS error, and spread independently of robot repeatability
  - prerequisites: no registration if the truth grid is in tracker frame; tip calibration optional/configurable
  - inspect: `overall_rms_error_mm`, `per_axis_bias_mm`, `per_point_metrics`, `outlier_count`
  - outputs: canonical raw samples in `samples.jsonl` and reduced per-point metrics in `summary.json`
- `repeatability_dataset`
  - purpose: the main command-plus-pose dataset experiment for robot repeatability
  - prerequisites: none for tracker-frame dry-run/mock datasets; registration required only for robot-frame pose analysis
  - inspect: `per_target_metrics`, `overall_repeatability_rms_mm`, `approach_conditioned_spread_mm`, dropped/invalid counts
  - outputs: canonical metadata, phase-aware timeseries samples, and repeatability summary metrics

Recommended hardware workflow:

1. Run `pivot_calibration` to generate or refresh the pen-probe tip file.
2. Run `aurora_grid_accuracy` to verify the tracker is healthy before doing registration.
3. Run the 4-point `Registration` workflow in the GUI.
4. Save and verify the servo calibration artifact in `Servos`.
5. Run `repeatability_dataset` to collect the main robot dataset.

### GUI Experiment Workspace

Launch the full operator app:

```bash
scripts/run_gui.sh
```

The `Experiment` tab is now the canonical operator workspace for:

- `repeatability_dataset`
- `aurora_grid_accuracy`
- `pivot_calibration`

What the workspace shows:

- a dedicated selector for the three critical experiments
- a YAML config panel backed by the same canonical experiment runner used by the CLI
- a preflight panel with per-check status and one overall run state:
  - `ok_to_run`
  - `ok_with_warning`
  - `blocked`
- a compact run checklist card showing experiment, backend, dry-run/live mode, tool ids, tip file, registration file, and planned output path
- a visualization pane that prefers native 3D on stable platforms and falls back to a safe 2D projection or placeholder when advanced OpenGL/QtDataVisualization is unavailable
- an embedded results viewer with summary text and experiment-specific plots
- run history loading so prior runs can be reviewed without hardware attached
- export actions for the current plot/view plus direct open-folder access for the current run

What the 3D view shows:

- `repeatability_dataset`
  - logged sample points
  - target centroids
  - coloring by target, validity, phase, or revisit index
- `aurora_grid_accuracy`
  - measured sample points
  - truth grid points when available
  - coloring by point, validity, phase, or repetition
- `pivot_calibration`
  - pivot sample cloud
  - inlier/outlier coloring after a solved run

How preflight validation works:

- runs are blocked for invalid config, missing required files, missing live prerequisites in live mode, or dimensionality mismatches
- warnings are used only for optional-but-important conditions such as missing registration for tracker-frame repeatability runs or coil-origin fallback in grid accuracy
- informational notes are shown for dry-run/mock conditions and other non-blocking runtime facts
- pivot runs require explicit overwrite confirmation when the configured tip output file already exists

How to review prior runs:

1. Launch the GUI.
2. Open the `Experiment` tab.
3. Use `Open Run Folder` or double-click a run in `Run History`.
4. The config snapshot, summary, plots, and 3D sample view are repopulated from the saved dataset bundle.

Visualization stability note:

- on macOS the workspace defaults to the projection fallback because `PySide6.QtDataVisualization` has shown native crashes in development
- on the Pi/Linux desktop, `visualization_mode: "auto"` keeps the native 3D view when the platform looks safe
- if you want the most conservative behavior everywhere, set `visualization_mode: "2d"` or export `CONTINUUM_VISUALIZATION_MODE=2d`

### Run Without Hardware

List available experiments:

```bash
.venv/bin/python scripts/run_experiment.py --list
```

Diagnostic experiments that work in mock/offline mode:

```bash
.venv/bin/python scripts/run_experiment.py --experiment tracker_pipeline_mock
.venv/bin/python scripts/run_experiment.py --experiment transform_chain_validation
.venv/bin/python scripts/run_experiment.py --experiment command_schedule_validation --config config/experiment_command_schedule_validation.example.yaml
.venv/bin/python scripts/run_experiment.py --experiment dataset_schema_roundtrip
```

Replay a previous dataset:

```bash
.venv/bin/python scripts/run_experiment.py --experiment replay_runner --config config/experiment_replay_runner.example.yaml
```

Project-critical experiments that work before hardware is back:

```bash
.venv/bin/python scripts/run_experiment.py \
  --experiment repeatability_dataset \
  --config config/experiment_repeatability_dataset.example.yaml

.venv/bin/python scripts/run_experiment.py \
  --experiment aurora_grid_accuracy \
  --config config/experiment_aurora_grid_accuracy.example.yaml

.venv/bin/python scripts/run_experiment.py \
  --experiment pivot_calibration \
  --config config/experiment_pivot_calibration.example.yaml
```

What the repeatability and grid datasets record:

- lifecycle phases such as `setup`, `neutral_home`, `command_sequence`, `settle`, `sample`, and `finalize`
- commanded cable deltas
- commanded motor values when available or computable in dry-run
- tracker frame ids, tool ids seen, transform validity, freshness, and backend health
- pose in tracker frame
- pose in robot/model frame when registration exists
- explicit status flags when registration is missing versus when full pose is available

The older `collect_pose_command_dataset` experiment remains available as a generic compatibility path for the current GUI and for ad hoc command schedules, but the three experiments above are now the canonical project-facing workflow.

### Hardware-Backed Fit Later

When Aurora and OpenRB are available again:

- `repeatability_dataset` can switch from `dry_run: true` to `dry_run: false`
- `aurora_grid_accuracy` can run against the real pen probe and a measured grid truth set
- `pivot_calibration` can collect live pivot poses instead of replaying an existing file
- servo-connected runs send real commands through `ServoService` and the SDK-backed `DxlBus`
- tracker-backed runs will continue to consume the canonical `TrackingService`
- registration will remain optional for tracker-frame datasets but required for robot-frame pose outputs
- future timing, hysteresis, transient, and tensioning studies should be implemented as new registered experiments or schedule/config variants, not as standalone helper scripts

Migration notes for the old experiment ideas are in [experiments_migration.md](/Users/jacklange/Continuum/pi_code/docs/experiments_migration.md).

## Legacy Compatibility Path

Retained but not default:

- [tracker_bridge.cpp](/Users/jacklange/Continuum/pi_code/tracker_bridge/tracker_bridge.cpp)
- [tracker_service_manager.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tracker_service_manager.py)
- [aurora_framer.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_framer.py)
- [aurora_parser.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_parser.py)

Use that path only for comparison, legacy replay, or migration debugging. The production live Aurora path is the Python-native `NDITracker` backend.
