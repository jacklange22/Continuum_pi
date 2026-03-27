# Continuum Robot (Scaffold)

This repository contains a Raspberry Pi-focused continuum robot stack where:

- `tracker_bridge` (C++) owns the Aurora lifecycle through NDI `CombinedApi`
- Python remains the top-level operator app (GUI/controllers/registration/diagnostics)

## Transform Convention

All transforms follow:

- `T_A_B` transforms coordinates from frame **B** into frame **A**
- `T_A_C = T_A_B @ T_B_C`

## Tracker Architecture

Aurora tracking is handled by `tracker_bridge`:

1. connect to Aurora serial device (for example `/dev/ttyUSB0`)
2. `initialize()`
3. initialize/enable tool handles
4. `startTracking()`
5. poll `getTrackingDataBX(...)`
6. emit line-delimited JSON over a Unix domain socket

Python consumes that socket stream via:

- `continuum_robot/tracking/tracker_socket_client.py`
- `continuum_robot/tracking/tracker_service_manager.py`

## tracker_bridge Message Format

Each line is a JSON object.

Status message fields:

- `type: "status"`
- `timestamp`
- `level` (`info|warning|error`)
- `state` (`connecting|initialized|tools_found|tool_enabled|tracking_started|tracking_stopped|...`)
- `message`
- `details` object

Transform message fields:

- `type: "transform"`
- `timestamp`
- `frame_number`
- `tool_id`
- `valid`
- `status`
- `quaternion` (`[w, x, y, z]`)
- `translation_mm` (`[x, y, z]`)
- `quality`

## Build tracker_bridge (Raspberry Pi)

Set SDK paths and build:

```bash
export NDI_SDK_INCLUDE_DIR=/path/to/ndi/include
export NDI_SDK_LIB_DIR=/path/to/ndi/lib
# Optional, default is CombinedApi:
# export NDI_SDK_LIBS="CombinedApi ndicapi"

scripts/build_tracker_bridge.sh
```

Binary output:

- `bin/tracker_bridge`

## Run tracker_bridge

```bash
AURORA_PORT=/dev/ttyUSB0 scripts/run_tracker_bridge.sh
```

Optional env vars:

- `TRACKER_BRIDGE_BIN`
- `TRACKER_SOCKET_PATH` (default `/tmp/tracker_bridge.sock`)
- `TRACKER_POLL_MS` (default `20`)

## Python Diagnostics

Start diagnostics (spawns and monitors `tracker_bridge`):

```bash
python3 scripts/run_diagnostics.py --tracker-port /dev/ttyUSB0 --packets 10
```

Print raw socket stream (bridge must already be running):

```bash
python3 scripts/print_tracker_stream.py --socket-path /tmp/tracker_bridge.sock
```

## Registration Outputs

Registration results are saved under:

- `data/registrations/`
- latest file: `data/registrations/latest_registration.json`

Diagnostics and GUI continue running when registration is missing, and clearly report `T_robot_tip` as unavailable.

## Config

Key tracker settings are in `config/system.yaml`:

- `aurora_port`
- `tracker_socket_path`
- `tracker_bridge_executable`
- `tracker_poll_ms`

## GUI Entry

```bash
python3 scripts/run_gui.py
```

Current GUI classes are scaffolded but now wired to tracker manager/controller state so a PySide view layer can subscribe without blocking.
