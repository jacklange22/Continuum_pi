# Repo Instructions

## Overview

- `continuum_robot/` contains the current Python application code for tracking, registration, servo control, experiments, config loading, and GUI/controller scaffolding.
- `legacy/tracker_bridge/` contains the retired C++ Aurora bridge source kept only for legacy comparison/debugging. The active tracking runtime is the Python NDI path under `continuum_robot/tracking/`.
- `config/` contains editable runtime YAML configuration.
- `tests/` contains unit and integration-style coverage for transform math, tracker parsing, registration persistence, servo workflows, and related services.
- `references/` contains read-only legacy reference scripts, captured data, vendor docs, and prior validation assets.
- `tools/` contains read-only lab inputs such as registration-point assets, transform exports, and pen-probe geometry files.
- `docs/` contains the current project spec, architecture, operator workflows, validation plan, testing protocol, and trace documents.

## Protected Paths

- Treat everything under `references/` as read-only reference material. Do not modify, rename, delete, or reformat files in this directory unless the user explicitly asks for it.
- Treat everything under `tools/` as read-only reference input. Do not modify, rename, delete, or reformat files in this directory unless the user explicitly asks for it.
- This protection includes SolidWorks-derived registration-point files, transform exports, pen-probe geometry files, and any other lab artifacts stored in `tools/`.
- The following files are especially critical to registration and must stay read-only unless the user explicitly overrides this:
  - `tools/12_model_registration_points_in_sw`
  - `tools/5_model_registration_points_in_sw`
  - `tools/all_tip_registration_points_in_sw`
  - `tools/T_sw_2_model`
  - `tools/T_sw_2_tip`
  - `tools/camarillo_stiffness`
  - `tools/penprobe_08_09_24c`

## Canonical Runtime Paths

Prefer the current application architecture over legacy scripts.

- Tracking runtime: current tracking service / parser / GUI paths under `continuum_robot/`
- Legacy bridge compatibility/reference only: `continuum_robot/tracking/legacy_bridge/` and `legacy/tracker_bridge/`
- Registration runtime: current registration service / repository / GUI controller paths under `continuum_robot/`
- Servo runtime: current OpenRB / DYNAMIXEL hardware seam and `ServoService` paths under `continuum_robot/`
- Experiment runtime: current `ExperimentRunner` and GUI experiment workflow under `continuum_robot/`
- Persistence/runtime outputs: write new generated artifacts to normal runtime locations such as `data/`, configured output folders, `config/`, or source-controlled files under `continuum_robot/` and `docs/`

Do not create new primary workflows in `references/` or `tools/`.

## How To Use `references/`

Use `references/` as a source of:
- proven math
- packet parsing ideas
- transform conventions
- hardware contract details
- validation ideas
- artifact/file-format conventions

Do NOT use `references/` as the target architecture.
Do NOT recreate old experiment-folder sprawl, symlink-heavy layout, or Arduino-era architecture.

If current application behavior conflicts with a legacy reference, preserve the protected files and adapt the application code or config instead.

## Reference Routing By Subsystem

When working on a subsystem, consult these reference files first:

### Tracking / Aurora
- `references/continuum_aurora.py`
- `references/test_continuum_aurora.py`
- `references/aurora_timing.py`
- `references/track_server.cpp`

Use them for:
- packet request/read behavior
- DLE stuffing/unstuffing logic
- transform parsing by tool ID
- timing/throughput expectations
- prior tracker validation ideas

### Pivot Calibration / Tip Files
- `references/pivot_cal_lsq.m`
- `references/new_pivot_cal.py`

Use them for:
- pivot-calibration math
- outlier handling ideas
- tip-file conventions
- RMSE/error reporting

### Registration
- `references/rigid_registration.py`
- `references/RegistrationPoints.csv`
- `references/grid_line_reg.txt`

Use them for:
- rigid registration math
- landmark conventions
- validation/error metrics
- saved artifact conventions

Two registration experiments live side by side and serve different purposes:

- `registration_validation` measures repeatability across saved
  registrations and flags FRE drift over time.
- `registration_trial` (in `continuum_robot/experiments/registration_trial.py`
  + `continuum_robot/registration/trial_analysis.py`) captures
  N landmarks × K samples once and sweeps averaging methods, label
  subsets, leave-one-out residuals, and samples-per-point diminishing
  returns. The GUI launcher is **Run Registration Trial →** on the
  Registration tab. Trial runs never auto-replace
  `latest_registration.json`; use
  `continuum_robot/data/promote_registration_trial.py` after manual
  review of `trial_report.md`.

### Servo / Hardware Interface
- `references/openrb-150.md`
- `references/xc330-m288.md`
- `references/continuum_arduino.py` (conceptual only; not target architecture)

Use them for:
- OpenRB setup assumptions
- DYNAMIXEL control-table/register assumptions
- safe hardware workflow details
- old cable-length / setpoint concepts when useful

Do NOT port `references/continuum_arduino.py` directly into the new OpenRB/XC330 path.

### Old Experiment Behavior
- `references/repeatability.py`
- `references/data_2024_07_31_10_59_15.dat`

Use them for:
- experiment behavior expectations
- logging conventions
- repeatability dataset interpretation

## MVP Priority

Unless the user explicitly asks otherwise, prioritize the minimal lab product in this order:

1. Tracker connect and validation
2. Pivot calibration for tool `0B`
3. 4-point registration into robot/body frame
4. Servo bring-up and safe calibration
5. Pretension
6. Repeatability / babble experiment logging

Prefer finishing the current MVP layer over broad refactors or extra GUI polish.

## Working Rules

- When implementing features, read from `references/` and `tools/` if needed, but write new outputs to normal runtime locations such as `data/`, configured output folders, `config/`, `docs/`, or source files under `continuum_robot/`.
- Reuse legacy math and validation ideas where clearly helpful, but keep the current canonical service/controller/runtime architecture.
- If there are multiple ways to implement something, prefer the path that:
  1. preserves canonical services,
  2. improves operator validation,
  3. reduces ambiguity,
  4. is easier to test.
- When simplifying, remove or de-emphasize redundant operator-facing paths rather than adding more parallel workflows.
- When the user reports a repeated misconception or recurring bug source, update `AGENTS.md` so the correction persists in future sessions.

## Validation Expectations

- Prefer targeted tests for the exact workflow being modified.
- If a change affects tracker/registration/servo/operator flow, update or add tests in `tests/`.
- For hardware-facing work, distinguish clearly between:
  - dry/mock validation
  - bench validation still required on the Pi / real hardware
- Do not claim hardware truth unless it has actually been bench-validated.
