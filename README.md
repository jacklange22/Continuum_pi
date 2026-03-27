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
- on-hardware validation of Aurora tracking through `tracker_bridge`
- measurement and acceptance of the registration pen tip transform, if the pen tip is offset from the tracked coil

Important safety change:

- hardware OpenRB/DYNAMIXEL connect paths now fail closed with explicit `not implemented` errors instead of pretending to connect successfully

## Repository Layout

- `continuum_robot/`
  Python application code: bootstrap, GUI, controllers, tracking, registration, servo services, experiments, config loading, and utilities
- `tracker_bridge/`
  C++ Aurora bridge using the NDI SDK and streaming JSON over a Unix socket
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

- [tracker_bridge.cpp](/Users/jacklange/Continuum/pi_code/tracker_bridge/tracker_bridge.cpp)
- [tracker_service_manager.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tracker_service_manager.py)
- [tip_pose_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tip_pose_service.py)

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

- Python 3.11 or newer
- `venv`
- `pip`
- `numpy`
- `PyYAML`
- `pyserial`
- `PySide6`

Tracker bridge hardware mode also requires:

- NDI SDK installed outside this repo
- working `CombinedApi` headers and libraries

Practical note:

- a clean bootstrap needs internet access or a local wheel/cache mirror for Python dependencies

## Fresh Install

From the repo root:

```bash
PYTHON_BIN=python3.11 scripts/bootstrap.sh
```

What it does:

- creates `.venv/`
- installs the package and dev dependencies
- creates `data/calibrations`, `data/logs`, `data/registrations`, `data/tracker_captures`, and `data/runs`
- optionally builds `tracker_bridge` if `BUILD_TRACKER_BRIDGE=1`

Bootstrap hardening:

- fails early if `PYTHON_BIN` is older than Python 3.11
- recreates the virtualenv if the existing env was created with a different Python major/minor version

Fresh-bootstrap smoke without touching `.venv`:

```bash
VENV_DIR=/tmp/pi_code_bootstrap_smoke_20260326 PYTHON_BIN=python3.11 scripts/bootstrap.sh
```

## Raspberry Pi Bring-Up

Recommended Pi sequence:

1. Clone the repo onto the Pi.
2. Run `PYTHON_BIN=python3.11 scripts/bootstrap.sh`.
3. Copy the machine-local config:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

4. Edit `config/system.local.yaml`:

- set `mock_mode`
- set `aurora_port`
- set `openrb_port`
- choose `robot_config`
- adjust any local paths if needed

5. If using Aurora hardware mode, build `tracker_bridge`.
6. Launch the GUI with `scripts/run_gui.sh`.

Repo-relative paths are resolved from the project root inside the app, so launch behavior no longer depends on the shell cwd.

## Tracker Bridge Build

Needed only for hardware tracker mode.

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
scripts/build_tracker_bridge.sh
```

Or as part of bootstrap:

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
BUILD_TRACKER_BRIDGE=1 PYTHON_BIN=python3.11 scripts/bootstrap.sh
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
- `tracker_socket_path`
- `tracker_bridge_executable`
- `neutral_setpoints_path`
- `latest_registration_path`
- `capture_tool_tip_transform`

`capture_tool_tip_transform` is optional and lives in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml). It is a 4x4 transform from the tracked registration-coil frame into the physical pen-tip frame.

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
2. build `tracker_bridge`
3. set `aurora_port`
4. launch `scripts/run_gui.sh`

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
- monitor frame count and quality
- inspect tip status and tip position
- view a simple XY plot of tools and tip

**Registration**

- begin a guided session
- capture repeated landmark samples
- monitor counts per landmark
- see whether capture uses coil origin or an explicit tip transform
- solve/save registration
- inspect FRE and residuals

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

Default config:

- 4 landmarks
- 5 captures per landmark
- capture tool `0B`

Process:

1. start tracker
2. open Registration tab
3. click `Begin Session`
4. move the probe to each landmark
5. click `Capture Sample` for each repetition
6. click `Solve + Save`
7. review FRE and residuals

If the probe tip is offset from the tracked coil:

- set `capture_tool_tip_transform` in [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml)

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
.venv/bin/python scripts/run_diagnostics.py --packets 3
```

Expected output includes:

- backend name
- connection-state transitions
- tool frames
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

- `51 passed`
- offscreen GUI smoke shows title `Continuum Robot Operator Console` and `5` tabs
- mock-mode diagnostics run
- fresh bootstrap smoke with `VENV_DIR=/tmp/pi_code_bootstrap_smoke_20260326`

## Tomorrow Hardware Test

Validated already:

- test suite passes with `51 passed`
- GUI bootstraps in offscreen mode
- mock tracker diagnostics work
- fresh bootstrap path succeeds with Python 3.11

Requires tomorrow’s hardware acceptance:

- Aurora serial connectivity through `tracker_bridge`
- real `0A` and `0B` tool visibility on the Pi
- physical `capture_tool_tip_transform` value if the pen tip is offset
- real OpenRB/DYNAMIXEL transport
- physical pretension stepping behavior

Recommended command sequence on the Pi:

```bash
cd /path/to/pi_code
PYTHON_BIN=python3.11 scripts/bootstrap.sh
cp config/system.local.example.yaml config/system.local.yaml
```

Edit `config/system.local.yaml`:

- set `mock_mode: false`
- set `aurora_port`
- set `openrb_port`

If the registration pen tip is offset, edit [registration.yaml](/Users/jacklange/Continuum/pi_code/config/registration.yaml) and set `capture_tool_tip_transform`.

Build bridge:

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
scripts/build_tracker_bridge.sh
```

Tracker-only preflight:

```bash
.venv/bin/python scripts/run_diagnostics.py --tracker-port /dev/ttyUSB0 --packets 5
```

Expected success signals:

- `Tracker backend: TrackerServiceManager`
- `State: tracking`
- repeated frame lines for `0A`
- `T_robot_tip translation: ...` if registration exists

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
- `tracker_bridge executable not found`
  build failed or `tracker_bridge_executable` is wrong
- tracker state stays `connecting` or `reconnecting`
  wrong serial port, Aurora not responding, or NDI runtime issue
- Registration shows `coil origin / no explicit tip offset`
  no pen-tip transform is configured
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

GUI has no tip pose:

- registration is missing or invalid
- inspect `data/registrations/latest_registration.json`

Diagnostics say registration is missing:

- expected until a registration file exists
- also verify `capture_tool_tip_transform` if the probe tip is offset

Hardware tracker mode does not start:

- check `mock_mode: false`
- check `aurora_port`
- check `tracker_bridge_executable`
- check NDI SDK paths and runtime libraries
- check `bin/tracker_bridge` exists

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
