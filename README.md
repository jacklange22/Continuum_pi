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
- `scripts/`: bootstrap, diagnostics, benchmark, validation, launch helpers
- `tests/`: unit and mock-backed integration coverage
- `data/`: runtime outputs for registrations, captures, runs, logs, calibrations
- `tracker_bridge/`: legacy C++ Aurora bridge, retained for comparison only
- `references/`: read-only legacy reference material
- `tools/`: read-only registration assets and lab inputs

Do not modify `references/` or `tools/` unless you intentionally want to change protected reference material.

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
- `PySide6`
- `scikit-surgerynditracker`

Important:

- this repo installs the Python package dependency, but it does not bundle or auto-detect whatever low-level vendor/runtime dependencies your local `scikit-surgerynditracker` install needs
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

## Live Aurora Config

Copy the local config template:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

For live Aurora on the Pi, the important fields are:

```yaml
mock_mode: false
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
- raw live ids `10` and `11` are the current observed Aurora tool ids on the Pi
- the app runtime still uses `0A` and `0B`
- `system.local.yaml` overrides `system.yaml`

## Recommended Order Of Operations

1. Verify Python environment and config
2. Run tracker doctor to verify backend selection and startup preflight
3. Run tracker smoke to verify pre-registration tracking readiness
4. Run the tracker benchmark for timing and freshness
5. Perform registration and create `data/registrations/latest_registration.json`
6. Run registration readiness validation to confirm `T_robot_tip`
6. Launch the full GUI/app

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
3. start a session
4. capture repeated `0B` samples for each landmark
5. finish and save

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
3. `Registration` tab can capture and solve
4. after registration is loaded, `Tracking` can compute `T_robot_tip`
5. servo/runtime work can build on the validated live tracking path

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

5. Edit `config/system.local.yaml` for live Aurora:

```yaml
mock_mode: false
aurora_port: "/dev/ttyUSB0"
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

## Mock Mode

Mock mode remains available and should not be broken by live-backend changes.

In mock mode:

- backend identity is `mock_tracker_manager`
- tools `0A` and `0B` are synthetic but valid
- GUI, registration flow, and experiments can be exercised without Aurora hardware

## Legacy Compatibility Path

Retained but not default:

- [tracker_bridge.cpp](/Users/jacklange/Continuum/pi_code/tracker_bridge/tracker_bridge.cpp)
- [tracker_service_manager.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/tracker_service_manager.py)
- [aurora_framer.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_framer.py)
- [aurora_parser.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/aurora_parser.py)

Use that path only for comparison, legacy replay, or migration debugging. The production live Aurora path is the Python-native `NDITracker` backend.
