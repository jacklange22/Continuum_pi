# Continuum Robot (Scaffold)

This repository contains a modular scaffold for a Raspberry Pi-hosted tendon-driven continuum robot stack.

## Transform Convention (strict)

All transforms follow:

- `T_A_B` means **transform from frame B into frame A**.
- Composition rule: `T_A_C = T_A_B @ T_B_C`.

Example used throughout tracking:

- `T_robot_aurora`
- `T_aurora_coil`
- `T_coil_tip`
- `T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip`

## Current Scope

Implemented in this phase:

- real Aurora serial client (open/close/read/write)
- Aurora framing and DLE unstuffing
- transform packet parsing for tools `0A` and `0B`
- CRC validation for transform packets
- `T_robot_tip` computation from `0A` + saved calibration transforms
- tracker diagnostics script (no GUI coupling)

Not implemented yet:

- OpenRB/DYNAMIXEL hardware control
- GUI integration for live tracking

## Aurora Packet Assumptions (explicit)

These assumptions are based on reference behavior and should be validated against live captures:

- Frame format: `DLE STX <stuffed payload> DLE ETX`
- Payload format:
  - byte `0`: packet type (`0x01` expected)
  - byte `1`: tool record count
  - bytes `2..5`: frame number (`uint32`, little-endian)
  - bytes `6..N-2`: tool records (`36` bytes each)
  - byte `N-1`: CRC-8 over payload bytes except CRC byte
- Tool record layout (`36` bytes):
  - bytes `0..1`: tool id ASCII (`0A`, `0B`, ...)
  - byte `2`: status byte
  - byte `3`: reserved
  - bytes `4..35`: eight `float32` values (`quat[4]`, `translation[3]`, `quality_or_error`)

If your Aurora firmware stream differs, parser constants may need updates.

## Registration/Calibration Files Expected

Tracker diagnostics expects this JSON by default:

- `/Users/jacklange/Continuum/pi_code/data/registrations/latest_registration.json`

Required keys:

- `T_robot_aurora`: `4x4` homogeneous transform
- `T_coil_tip`: `4x4` homogeneous transform

Diagnostics **fails clearly** when this file is missing/malformed and reports that `T_robot_tip` is unavailable.

## Experiment Input File Format

`CSV` with header. Required and optional columns:

- required: `index`
- required: tendon displacement columns (`dl_1`, `dl_2`, ..., `dl_N`)
- optional: `settle_time_s`
- optional: `repeat`

Example:

```csv
index,dl_1,dl_2,dl_3,dl_4,settle_time_s,repeat
0,0.0,0.0,0.0,0.0,2.0,1
1,-6.0,0.0,6.0,0.0,3.0,2
```

## Run

### GUI scaffold

```bash
python3 /Users/jacklange/Continuum/pi_code/scripts/run_gui.py
```

### Tracker diagnostics

```bash
python3 /Users/jacklange/Continuum/pi_code/scripts/run_diagnostics.py --tracker-port /dev/ttyUSB0 --packets 10
```

Useful options:

- `--baudrate 115200`
- `--timeout 1.0`
- `--registration-file /path/to/latest_registration.json`

Expected diagnostics output includes:

- connection status
- packet frame number and CRC
- tool presence/status for `0A` and `0B`
- `T_aurora_coil` translation from `0A`
- `T_robot_tip` translation when registration/calibration is available

## Install

```bash
python3 -m pip install -e ".[dev]"
```
