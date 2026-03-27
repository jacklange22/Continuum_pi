# Continuum Robot Operator Platform

This repository is a Raspberry Pi-oriented local operator stack for a tendon-driven continuum robot. The intended runtime model is:

- Raspberry Pi is the only host computer
- Aurora connects directly to the Pi
- OpenRB-150 connects directly to the Pi
- the GUI runs locally on the Pi with HDMI, keyboard, and mouse attached

The repository now contains a real PySide6 desktop application with five operator tabs:

- `System`
- `Servos`
- `Tracking`
- `Registration`
- `Experiment`

It also contains a validated mock mode so the full operator flow can be exercised without hardware.

## Current Status

Implemented and validated now:

- real PySide6 desktop GUI with five usable tabs
- mock-mode end-to-end bootstrap and operator workflow
- synthetic live tracker backend with tools `0A` and `0B`
- tracker diagnostics through the configured backend
- guided registration flow with persisted `latest_registration.json`
- tip-pose computation from `T_robot_aurora @ T_aurora_coil @ T_coil_tip`
- tendon-displacement command flow against a mock DYNAMIXEL backend
- neutral calibration persistence with archival of previous latest files
- experiment CSV loading and one `.dat` file per run in mock mode
- tests covering config, GUI bootstrap, controllers, tracking, registration, experiments, transform math, and hardening seams

Still hardware-pending:

- real DYNAMIXEL/OpenRB transport in [continuum_robot/hardware/dxl_bus.py](/Users/jacklange/Continuum/pi_code/continuum_robot/hardware/dxl_bus.py)
- real OpenRB prep/status transport in [continuum_robot/hardware/openrb_client.py](/Users/jacklange/Continuum/pi_code/continuum_robot/hardware/openrb_client.py)
- physical pretension stepping algorithm
- on-hardware validation of Aurora tracking through `scikit-surgerynditracker`
- measurement and acceptance of the registration pen tip transform, if the pen tip is offset from the tracked coil

Important safety change:

- hardware OpenRB/DYNAMIXEL connect paths now fail closed with explicit `not implemented` errors instead of pretending to connect successfully

## Repository Layout

- `continuum_robot/`
  Python application code: bootstrap, GUI, controllers, tracking, registration, servo services, experiments, config loading, and utilities
- `tracker_bridge/`
  legacy compatibility Aurora bridge using the NDI SDK and streaming JSON over a Unix socket
- `config/`
  runtime YAML configuration templates
- `scripts/`
  bootstrap, GUI launch, tracker diagnostics, and bridge helper scripts
- `tests/`
  unit and mock-backed integration coverage
- `data/`
  runtime outputs: calibrations, registrations, tracker captures, logs, and experiment runs
- `references/`
  read-only legacy reference scripts and documents
- `tools/`
  read-only geometry and lab reference inputs

Do not modify `references/` or `tools/` unless you intentionally want to change protected reference material.

## Architecture

Primary operator/runtime path:

- [bootstrap.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/bootstrap.py)
- [main.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/main.py)
- [app_window.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/app_window.py)
- [controllers](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers)

Tracker integration path:

- [ndi_backend.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py)
- [tracking_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/tracking_service.py)
- [tip_pose_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tip_pose_service.py)

Actual live runtime path:

- Aurora hardware is owned directly by `scikit-surgerynditracker.NDITracker`
- `TrackerBackendNDI` owns tracker configuration, start/stop, polling, and tool-state conversion
- `TrackingService` is the app-visible source of truth for live `0A` and `0B` poses, freshness, faults, and `T_robot_tip`
- registration, diagnostics, and the GUI consume `TrackingService`

Legacy compatibility paths retained but not default:

- [aurora_packet.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_packet.py)
- [aurora_parser.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_parser.py)
- [aurora_framer.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_framer.py)
- [tracker_bridge.cpp](/Users/jacklange/Continuum/pi_code/tracker_bridge/tracker_bridge.cpp)
- [tracker_service_manager.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tracker_service_manager.py)

Those modules are retained for replay, regression tests, historical client-packet compatibility, and optional bridge comparison. They are not the default live hardware backend.

Transform convention:

- `T_A_B` means “transform coordinates from frame B into frame A”
- `T_A_C = T_A_B @ T_B_C`

Tip pose chain:

- `T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip`

## Hardware Overview

Aurora:

- Aurora serial connection goes directly to the Pi
- tool `0A` is the robot coil used for runtime tip pose
- tool `0B` is the registration probe
- the registration backend can also reproduce the legacy `0A`-probe / `0B`-coil mapping when driven from saved CSV data

OpenRB-150:

- connects directly to the Pi
- serves as the intended DYNAMIXEL interface/controller
- GUI has connection and prep controls, but the real hardware transport is still pending

Robotis XC330-M288:

- supported robot modes are 4-servo and 8-servo
- operator-facing motion command is tendon displacement in centimeters
- displacement is mapped to servo ticks around saved neutral setpoints using spool diameter and ticks/revolution

## Software Requirements

Required:

- Python 3.10 or newer
- `venv`
- `pip`
- `numpy`
- `PyYAML`
- `pyserial`
- `scikit-surgerynditracker`
- `PySide6`

Default Python-native Aurora mode also depends on whatever low-level tracker bindings your local `scikit-surgerynditracker` installation requires. This repo does not bundle or auto-detect those vendor/runtime dependencies.

Legacy bridge compatibility mode also requires:

- NDI SDK installed outside this repo
- working `CombinedApi` headers and libraries

Practical note:

- a clean bootstrap needs internet access or a local wheel/cache mirror for Python dependencies

## Fresh Install

From the repo root:

```bash
PYTHON_BIN=python3 scripts/bootstrap.sh
```

What it does:

- creates `.venv/`
- installs the package and dev dependencies
- creates `data/calibrations`, `data/logs`, `data/registrations`, `data/tracker_captures`, and `data/runs`
- optionally builds `tracker_bridge` if `BUILD_TRACKER_BRIDGE=1` for legacy bridge compatibility

Bootstrap hardening:

- fails early if `PYTHON_BIN` is older than Python 3.10
- recreates the virtualenv if the existing env was created with a different Python major/minor version

Fresh-bootstrap smoke without touching `.venv`:

```bash
VENV_DIR=/tmp/pi_code_bootstrap_smoke_20260326 PYTHON_BIN=python3 scripts/bootstrap.sh
```

## Raspberry Pi Bring-Up

Recommended Pi sequence:

1. Clone the repo onto the Pi.
2. Check the system Python version:

```bash
python3 --version
```

3. If `python3` is 3.10 or newer, run `PYTHON_BIN=python3 scripts/bootstrap.sh`.
4. If `python3` is older than 3.10, upgrade the Pi OS image or install a newer Python before continuing.
5. Copy the machine-local config:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

4. Edit `config/system.local.yaml`:

- set `mock_mode`
- set `aurora_port`
- set `openrb_port`
- choose `robot_config`
- adjust any local paths if needed

5. Install and validate the local dependency chain required by `scikit-surgerynditracker`.
6. Only if you need bridge compatibility mode, build `tracker_bridge`.
7. Launch the GUI with `scripts/run_gui.sh`.

Repo-relative paths are resolved from the project root inside the app, so launch behavior no longer depends on the shell cwd.

## Python Aurora Backend

Default live hardware tracking now uses [ndi_backend.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py), not `tracker_bridge`.

Config fields controlling this path:

- `tracker_backend: "ndi"`
- `tracker_type: "aurora"`
- `aurora_port`
- `tracker_poll_ms`
- `tracker_freshness_timeout_s`
- `tracker_ports_to_probe`
- `tracker_settings_overrides`
- `tracker_min_effective_fps`
- `tracker_max_stale_interval_s`
- `tracker_max_consecutive_missing_frames`
- `tracker_require_valid_transforms`

The backend converts library output into the app transform model, validates 4x4 rigid transforms, normalizes tool ids, and feeds `TrackingService`.

## Tracker Bridge Build

Needed only for legacy comparison or compatibility mode.

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
scripts/build_tracker_bridge.sh
```

Or as part of bootstrap:

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
BUILD_TRACKER_BRIDGE=1 PYTHON_BIN=python3 scripts/bootstrap.sh
```

Expected output:

- `Built tracker_bridge at: .../bin/tracker_bridge`

## Configuration

Main files:

- [system.yaml](/Users/jacklange/Continuum/pi_code/config/system.yaml)
- [system.local.example.yaml](/Users/jacklange/Continuum/pi_code/config/system.local.example.yaml)
- [robot_4servo.yaml](/Users/jacklange/Continuum/pi_code/config/robot_4servo.yaml)
- [robot_8servo.yaml](/Users/jacklange/Continuum/pi_code/config/robot_8servo.yaml)
- [safety.yaml](/Users/jacklange/Continuum/pi_code/config/safety.yaml)
- [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml)
- [experiment.yaml](/Users/jacklange/Continuum/pi_code/config/experiment.yaml)
- [experiment_points.example.csv](/Users/jacklange/Continuum/pi_code/config/experiment_points.example.csv)

Important fields:

- `robot_config`
- `mock_mode`
- `aurora_port`
- `openrb_port`
- `tracker_backend`
- `tracker_type`
- `tracker_freshness_timeout_s`
- `tracker_ports_to_probe`
- `tracker_settings_overrides`
- `tracker_min_effective_fps`
- `tracker_max_stale_interval_s`
- `tracker_max_consecutive_missing_frames`
- `tracker_require_valid_transforms`
- `tracker_socket_path`
- `tracker_bridge_executable`
- `neutral_setpoints_path`
- `latest_registration_path`
- `capture_tool_id`
- `coil_tool_id`
- `model_points_file`
- `tip_points_file`
- `T_sw_2_model_file`
- `T_sw_2_tip_file`
- `penprobe_file`
- `capture_tool_tip_transform`

Registration now has a rigorous protected-asset path:

- load protected model/tip point files from `tools/`
- load `T_sw_2_model` and `T_sw_2_tip`
- load the protected penprobe file
- capture repeated measurement-tool and coil-tool samples
- solve `T_aurora_2_model`, `T_aurora_2_tip`, and `T_tip_2_coil`
- save strict keys plus legacy aliases in `latest_registration.json`

`capture_tool_tip_transform` is optional and lives in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml). When present, it overrides the translation-only penprobe file with an explicit 4x4 measurement-tool-to-point transform.

## Launching Mock Mode

Set `mock_mode: true`, then run:

```bash
scripts/run_gui.sh
```

Validated mock-mode behavior:

- System tab shows mock ports
- Tracking tab shows live synthetic `0A` and `0B`
- Registration can capture and save `latest_registration.json`
- Servos tab reads/writes mock telemetry
- Experiment tab can load a CSV and save one `.dat` file per run

## Launching Hardware Mode

Tracker-only hardware mode:

1. set `mock_mode: false`
2. set `tracker_backend: "ndi"`
3. set `aurora_port`
4. launch `scripts/run_gui.sh`

Optional legacy bridge comparison mode:

1. set `mock_mode: false`
2. set `tracker_backend: "bridge"`
3. build `tracker_bridge`
4. set `aurora_port`
5. launch `scripts/run_gui.sh`

Full servo hardware mode:

- not complete yet
- OpenRB/DYNAMIXEL connect attempts now fail clearly with `not implemented`
- do not treat the current servo hardware path as lab-ready

## GUI Workflow

**System**

- choose Aurora and OpenRB ports
- connect/disconnect tracker
- connect/disconnect OpenRB/DYNAMIXEL
- run OpenRB prep action
- inspect config summary and diagnostics

**Servos**

- scan IDs
- rename IDs
- jog by ticks
- capture/save/load neutral setpoints
- command tendon displacement in centimeters
- inspect position/current/voltage/fault telemetry
- run current-balance pretension validation

Current limitation:

- validated end-to-end only with the mock DYNAMIXEL backend

**Tracking**

- connect/disconnect tracker backend
- inspect tool state for `0A` and `0B`
- monitor frame count, freshness, and backend identity
- inspect tracked/missing/invalid/unknown state per tool
- see whether validity is known or still unknown from the live backend
- inspect tip status and tip position
- view a simple XY plot of tools and tip

**Registration**

- begin a guided session
- capture repeated landmark samples
- monitor counts per landmark
- capture paired measurement-tool and coil-tool poses
- see whether capture uses the protected penprobe file or an explicit tip transform
- solve/save registration
- inspect overall/model/tip FRE and residuals

Latest accepted registration:

- `data/registrations/latest_registration.json`

**Experiment**

- load a CSV
- inspect prerequisites
- start/stop a run
- watch progress
- inspect output path and last error

The controller refuses to run unless all of these are present:

- experiment file
- neutral calibration
- registration
- OpenRB/DYNAMIXEL connection
- tracker connection
- valid `0A` sample

## Neutral Calibration Workflow

Implemented flow:

1. connect the servo backend
2. jog each tendon drive to the desired neutral state
3. click `Capture Neutral`
4. click `Save Neutral`

Files:

- latest: `data/calibrations/neutral_setpoints.json`
- archived previous latest files: `data/calibrations/neutral_setpoints_<timestamp>.json`

## Pretension Workflow

Implemented now:

- current-balance validation from telemetry

Not implemented yet:

- the full stepwise algorithm that walks each servo until it reaches a configured pretension threshold window

## Registration Workflow

Default runtime config:

- 5 captures per landmark
- measurement tool `0B`
- coil tool `0A`
- protected model points from [12_model_registration_points_in_sw](/Users/jacklange/Continuum/pi_code/tools/12_model_registration_points_in_sw)
- protected tip points from [all_tip_registration_points_in_sw](/Users/jacklange/Continuum/pi_code/tools/all_tip_registration_points_in_sw)
- protected penprobe vector from [penprobe_08_09_24c](/Users/jacklange/Continuum/pi_code/tools/penprobe_08_09_24c)

Legacy reference note:

- the historical script used `0A` as the measured pen-probe tool and `0B` as the averaged coil tool
- the new backend keeps those roles configurable, so the old mapping can still be reproduced exactly when needed

Process:

1. start tracker
2. open Registration tab
3. click `Begin Session`
4. move the measurement tool through the ordered model and tip landmarks
5. click `Capture Sample` for each repetition
6. click `Solve + Save`
7. review overall/model/tip FRE and residuals

Backend behavior:

- expands truth points by repetition count using the same contiguous-per-landmark order as the legacy script
- splits measured points into model and tip groups
- performs two rigid SVD solves
- averages the coil-tool transform explicitly
- computes and saves `T_aurora_2_model`, `T_aurora_2_tip`, `T_tip_2_coil`, and strict `T_coil_tip`
- saves raw measured points, raw tool poses, grouped labels, and validation metrics

If the measurement point is offset from the tracked measurement tool:

- leave the protected `penprobe_file` configured for translation-only legacy behavior, or
- set `capture_tool_tip_transform` in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml) to override it with a full 4x4 transform

## Experiment Workflow

Process:

1. connect tracker and servo backend
2. load or capture neutral setpoints
3. complete registration
4. load an experiment CSV
5. click `Run`

Runner behavior:

- commands tendon displacement
- waits for settle time
- requires a live valid `0A` sample
- samples servo and tracker state
- computes tip pose in robot frame
- writes one `.dat` file per run

Example input:

- [experiment_points.example.csv](/Users/jacklange/Continuum/pi_code/config/experiment_points.example.csv)

## Output Files

Runtime artifacts live under `data/`:

- `data/calibrations/neutral_setpoints.json`
- `data/calibrations/neutral_setpoints_<timestamp>.json`
- `data/registrations/latest_registration.json`
- `data/registrations/registration_<timestamp>.json`
- `data/tracker_captures/`
- `data/runs/*.dat`
- `data/logs/`

## Diagnostics

Tracker diagnostics:

```bash
.venv/bin/python scripts/run_diagnostics.py --frames 3
```

Tracker benchmark with acceptance thresholds:

```bash
.venv/bin/python scripts/run_tracker_benchmark.py \
  --tracker-port /dev/ttyUSB0 \
  --duration-s 5 \
  --save-report data/logs/tracker_benchmark.json
```

Registration from a saved Aurora CSV:

```bash
.venv/bin/python scripts/run_registration_from_csv.py references/RegistrationPoints.csv \
  --measurement-tool-id 0A \
  --coil-tool-id 0B
```

This path is the easiest way to validate compatibility with the legacy CSV-driven workflow without touching the GUI.

Rigorous registration validation from saved data:

```bash
.venv/bin/python scripts/run_registration_validation.py \
  --registration-csv references/RegistrationPoints.csv \
  --measurement-tool-id 0A \
  --coil-tool-id 0B \
  --save-report data/registrations/validation_reference.json
```

Rerun from a saved registration/session artifact:

```bash
.venv/bin/python scripts/run_registration_validation.py \
  --session-json data/registrations/registration_<timestamp>.json \
  --save-report data/registrations/validation_rerun.json
```

Compare legacy-style outputs against the new path:

```bash
.venv/bin/python scripts/compare_registration_outputs.py \
  /path/to/legacy_output_dir \
  data/registrations/validation_reference.json
```

Runtime sanity from replayed Aurora packets:

```bash
.venv/bin/python scripts/run_registration_runtime_sanity.py \
  --registration-file data/registrations/latest_registration.json \
  --capture-jsonl data/tracker_captures/<capture>.jsonl
```

Runtime sanity from live Aurora data on the Pi:

```bash
.venv/bin/python scripts/run_registration_runtime_sanity.py \
  --registration-file data/registrations/latest_registration.json \
  --live \
  --tracker-port /dev/ttyUSB0
```

Expected output includes:

- backend name
- connection-state transitions
- tool frames
- per-tool state for `0A` and `0B`
- explicit `valid=unknown` when the live backend cannot prove a validity bit
- freshness and stale-data status from `TrackingService`
- registration role assignment loaded from the saved registration
- optional `T_robot_tip` output when a registration file exists

GUI smoke:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -c 'from PySide6.QtWidgets import QApplication; from continuum_robot.app.bootstrap import build_app_context; from continuum_robot.gui.app_window import AppWindow; app = QApplication([]); window = AppWindow(build_app_context()); print(window.windowTitle()); print(window.tab_widget.count()); window.shutdown()'
```

Headless note:

- `scripts/run_gui.sh` warns if no local display session is detected
- for an SSH smoke test, use `QT_QPA_PLATFORM=offscreen`

## Testing

Full suite:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Currently revalidated:

- `72 passed`
- offscreen GUI smoke shows title `Continuum Robot Operator Console` and `5` tabs
- mock-mode diagnostics run
- fresh bootstrap smoke with `VENV_DIR=/tmp/pi_code_bootstrap_smoke_20260326`
- repo code and scripts parse under Python 3.10 grammar

## Tomorrow Hardware Test

Validated already:

- test suite passes with `72 passed`
- GUI bootstraps in offscreen mode
- mock tracker diagnostics work
- fresh bootstrap path succeeds on a clean env

Requires tomorrow’s hardware acceptance:

- Aurora serial connectivity through `scikit-surgerynditracker`
- real `0A` and `0B` tool visibility on the Pi
- confirmation of the actual measurement-tool / coil-tool mapping on hardware
- physical `capture_tool_tip_transform` value only if you intend to override the protected penprobe file
- real OpenRB/DYNAMIXEL transport
- physical pretension stepping behavior

Recommended command sequence on the Pi:

```bash
cd /path/to/pi_code
sudo apt update
sudo apt install -y git python3 python3-venv build-essential
python3 --version
PYTHON_BIN=python3 scripts/bootstrap.sh
cp config/system.local.example.yaml config/system.local.yaml
```

If `python3 --version` prints lower than `3.10`, stop there and upgrade the Pi OS image or install a newer Python first.

Edit `config/system.local.yaml`:

- set `mock_mode: false`
- set `aurora_port`
- set `openrb_port`

If you need to override the protected penprobe file, edit [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml) and set `capture_tool_tip_transform`.

Tracker-only preflight:

```bash
.venv/bin/python scripts/run_diagnostics.py --tracker-port /dev/ttyUSB0 --frames 5
```

Tracker acceptance benchmark:

```bash
.venv/bin/python scripts/run_tracker_benchmark.py \
  --tracker-port /dev/ttyUSB0 \
  --duration-s 5
```

Registration validation procedure on the Pi:

1. Run the tracker preflight above and confirm `0A` and `0B` are visible.
2. Capture and save one Aurora registration CSV from the real registration sequence.
3. Run the validation CLI on that CSV:

```bash
.venv/bin/python scripts/run_registration_validation.py \
  --registration-csv /path/to/reg_capture.csv \
  --measurement-tool-id 0A \
  --coil-tool-id 0B \
  --save-report data/registrations/reg_capture_validation.json
```

4. Save the runtime-usable registration JSON:

```bash
.venv/bin/python scripts/run_registration_from_csv.py \
  /path/to/reg_capture.csv \
  --measurement-tool-id 0A \
  --coil-tool-id 0B
```

5. If you have legacy outputs for the same run, compare them:

```bash
.venv/bin/python scripts/compare_registration_outputs.py \
  /path/to/legacy_output_dir \
  data/registrations/reg_capture_validation.json
```

6. Replay a packet capture or use live Aurora data to verify `T_robot_tip`:

```bash
.venv/bin/python scripts/run_registration_runtime_sanity.py \
  --registration-file data/registrations/latest_registration.json \
  --capture-jsonl data/tracker_captures/<capture>.jsonl
```

Expected success signals:

- `Tracker backend: ndi`
- `State: tracking`
- repeated frame lines for `0A`
- `T_robot_tip translation: ...` if registration exists
- benchmark prints `passed=True`
- registration validation prints `T_aurora_2_model`, `T_aurora_2_tip`, `T_tip_2_coil`, and `T_coil_tip`
- registration validation prints `repetition_count`, per-label counts, and tool-role assignment
- comparison utility prints `passed=True`
- runtime sanity prints `passed=True` and `tip_pose_status=ok`

Archive these files from one real registration run:

- the raw Aurora registration CSV
- the validation report JSON from `scripts/run_registration_validation.py`
- the accepted registration JSON: `data/registrations/registration_<timestamp>.json`
- the current `data/registrations/latest_registration.json`
- the packet capture JSONL used for runtime sanity, if you recorded one
- the comparison report JSON, if you ran the comparison utility with `--save-report`

Reasonable default comparison thresholds for one run on identical data:

- translation difference `<= 0.25 mm`
- rotation difference `<= 0.25 deg`
- FRE difference `<= 0.05 mm`

What "pass" looks like:

- the validation CLI reports the expected measurement tool, coil tool, point counts, and repetition count
- all four saved transforms are present and finite
- overall/model/tip FRE values are finite and consistent with the run quality you expect
- the comparison utility passes within tolerance against the legacy result on the same dataset
- runtime sanity passes with the saved registration and valid `T_robot_tip` from live or replayed `0A`

Launch GUI:

```bash
scripts/run_gui.sh
```

Expected GUI success signals:

- System tab tracker state becomes `tracking`
- Tracking tab frame count increments
- tool table shows `0A` and `0B`
- Registration tab can begin and capture samples

Likely failure modes:

- `ERROR: no Aurora port is configured`
  fix `config/system.local.yaml` or pass `--tracker-port`
- tracker state stays `connecting` or `reconnecting`
  wrong serial port, Aurora not responding, or Python tracker dependency/runtime issue
- Registration shows `protected penprobe file`
  this is expected default behavior for the rigorous legacy-compatible path
- Registration solve fails on count mismatch or missing tool poses
  the current backend now validates repetition counts and paired tool-pose availability explicitly
- runtime sanity fails with `role_mismatch`
  the saved registration was solved with a different coil-tool assignment than the runtime currently assumes
- comparison utility fails on translation/rotation tolerance
  the new output does not yet match the legacy output closely enough on the same dataset
- OpenRB connect reports `not implemented`
  expected with the current codebase; servo hardware transport is still pending
- no local display warning from `scripts/run_gui.sh`
  launch from the Pi desktop session or use offscreen mode only for smoke testing

Fallback sanity check:

- set `mock_mode: true`
- rerun `scripts/run_gui.sh`

That should restore the validated synthetic workflow immediately.

## Troubleshooting

`pytest -q` fails with syntax/runtime issues:

```bash
.venv/bin/python -m pytest -q
```

Bootstrap keeps the wrong Python version:

- `scripts/bootstrap.sh` now recreates the virtualenv when the existing env uses a different Python major/minor

Pi cannot install `python3.11` packages by name:

- use `python3` and `python3-venv`
- run `python3 --version`
- this repo now requires Python 3.10 or newer, not specifically Python 3.11

GUI has no tip pose:

- registration is missing or invalid
- inspect `data/registrations/latest_registration.json`

Diagnostics say registration is missing:

- expected until a registration file exists
- also verify `capture_tool_id`, `coil_tool_id`, and the protected registration asset paths in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml)

Runtime sanity says `role_mismatch`:

- the saved registration was created with a different `coil_tool_id` than runtime tip-pose currently expects
- inspect `coil_tool_id` in the saved registration JSON and in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml)

Comparison utility fails:

- rerun both pipelines on the exact same registration capture
- verify measurement/coil tool ids match between the two outputs
- verify the legacy output directory contains the right `T_aurora_2_model` and `T_tip_2_coil`

Hardware tracker mode does not start:

- check `mock_mode: false`
- check `aurora_port`
- check `tracker_backend: "ndi"`
- check that `scikit-surgerynditracker` imports inside `.venv`
- if the library expects extra vendor/runtime components on your Pi, verify those outside the repo first
- if you intentionally selected bridge mode, check `tracker_bridge_executable`, NDI SDK paths, and `bin/tracker_bridge`

OpenRB/DYNAMIXEL reports `not implemented`:

- expected current boundary
- the hardware seam is intentionally blocked rather than faked

GUI launched over SSH does not open:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/run_gui.py
```

Use that only for a smoke test. Real operator use should be from the Pi desktop session.

## Recovery Notes

- registrations are safe to redo; timestamped copies are stored alongside `latest_registration.json`
- neutral calibration can be recaptured; previous latest files are archived with timestamps
- runtime outputs live under `data/` and can be deleted without affecting source code
- local machine overrides belong in `config/system.local.yaml`, not `config/system.yaml`
