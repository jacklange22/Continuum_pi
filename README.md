# Continuum Robot (Scaffold)

This repository contains a Raspberry Pi-focused continuum robot stack where:

- `tracker_bridge` (C++) owns the Aurora lifecycle through NDI `CombinedApi`
- Python remains the top-level operator app (GUI/controllers/registration/diagnostics)

## GitHub Sync

GitHub only syncs files that are committed and pushed.

- Source code, scripts, config templates, and tracked reference assets sync through git
- Local virtual environments, the built `bin/tracker_bridge` binary, and runtime output under `data/` do not
- Machine-specific serial settings can live in `config/system.local.yaml`, which is intentionally ignored by git

Typical workflow:

```bash
git add -A
git commit -m "describe your change"
git push origin main
```

On another machine:

```bash
git pull origin main
```

## Bootstrap A New Machine

Clone the repo, create the virtualenv, install Python dependencies, and optionally build the Aurora bridge:

```bash
git clone https://github.com/jacklange22/Continuum_pi.git
cd Continuum_pi
scripts/bootstrap.sh
```

That script:

- creates `.venv/`
- installs the package and dev dependencies from `pyproject.toml`
- creates expected runtime directories under `data/`
- optionally builds `tracker_bridge` if you set `BUILD_TRACKER_BRIDGE=1`

## Host Prerequisites

Before running the bootstrap script on a fresh Raspberry Pi or Linux workstation, install the basic host tools you need for Python virtualenvs and C++ builds:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv g++
```

## Local Machine Overrides

For machine-specific ports and paths:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

Then edit `config/system.local.yaml` for the local Pi or workstation. The loader automatically merges it over `config/system.yaml`.

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

`tracker_bridge` links against the NDI SDK and cannot be installed from `requirements.txt` or `pyproject.toml` alone. The Python dependencies can be auto-installed, but the NDI SDK is a separate vendor C++ dependency that must already exist on the machine.

Recommended layout on the target machine:

- install or unpack the NDI SDK outside the repo, for example under `/opt/ndi_sdk`
- point the build at its `include/` and `lib/` directories

Set SDK paths and build:

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
# Optional, default is CombinedApi:
# export NDI_SDK_LIBS="CombinedApi ndicapi"

scripts/build_tracker_bridge.sh
```

You can also have `scripts/bootstrap.sh` build it in the same step:

```bash
export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
BUILD_TRACKER_BRIDGE=1 scripts/bootstrap.sh
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
scripts/run_gui.sh
```

Current GUI classes are scaffolded but now wired to tracker manager/controller state so a PySide view layer can subscribe without blocking.
