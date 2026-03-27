# Build The Raspberry Pi Continuum Robot Platform

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not contain a separate `PLANS.md`. This file itself is the living execution plan and must remain self-contained.

## Purpose / Big Picture

After this plan is complete, a lab operator can boot a Raspberry Pi, launch a local GUI, connect Aurora and OpenRB-150 directly from the Pi, calibrate tendon neutrals, verify pretension, register the robot frame with Aurora tool `0B`, command tendon displacements safely in either 4-servo or 8-servo mode, run repeatability-style experiments, and save one `.dat` file per run with synchronized tracker and servo data. The visible proof is that the GUI shows live tracker and servo health, registration metrics, and experiment progress, and the run artifacts appear under `data/registrations/` and `data/runs/`.

## Progress

- [x] (2026-03-27 02:25Z) Audited the repository baseline. Tracking, transform math, registration solving, registration persistence, experiment CSV loading, and `.dat` writing already exist. Servo bus control, OpenRB preparation, experiment execution, and most GUI tabs/controllers are still scaffolds.
- [x] (2026-03-27 02:25Z) Verified the repo boundaries and protected assets. `references/` and `tools/` are read-only reference inputs and must not be edited as part of this build.
- [x] (2026-03-27 02:25Z) Identified environment blockers. The host `pytest -q` path resolved to Python 3.9 and failed during test collection; later audit work confirmed the repo is compatible with Python 3.10+.
- [x] (2026-03-27 03:10Z) Completed Milestone 1. Config loading now includes runtime, registration, experiment, and calibration settings; bootstrap wires mock and hardware-facing services; and the application launches end-to-end in mock mode.
- [x] (2026-03-27 03:10Z) Replaced the placeholder GUI shell with a real PySide6 `QMainWindow` and usable System, Servos, Tracking, Registration, and Experiment tabs backed by real controllers.
- [x] (2026-03-27 03:10Z) Implemented mock-mode tracker, servo, registration, and experiment flows deeply enough to validate the operator workflow without hardware. This includes synthetic `0A`/`0B` tracking, neutral calibration persistence, current-balance pretension validation, registration save/load, and `.dat` run output.
- [x] (2026-03-27 03:10Z) Added validation coverage for config loading, mock tracker streaming, servo service, experiment runner, controller flows, and GUI bootstrap. Current verified result: `43 passed`.
- [x] (2026-03-27 03:10Z) Improved diagnostics and bootstrap portability. `scripts/run_diagnostics.py` now uses the configured backend so mock mode works, and `scripts/bootstrap.sh` now recreates the virtualenv when the requested Python version does not match the existing env.
- [x] (2026-03-27 03:10Z) Rewrote `README.md` into an end-to-end setup and usage guide with explicit validated scope and hardware-dependent gaps.
- [x] (2026-03-27 03:10Z) Re-ran the final validation pass after the docs/bootstrap updates. Verified outputs: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` -> `43 passed`, `.venv/bin/python scripts/run_diagnostics.py --packets 2` in mock mode, and an offscreen `AppWindow` smoke launch showing the expected window title and 5 tabs.
- [x] (2026-03-27 04:35Z) Completed a production-readiness audit across bootstrap, GUI/runtime wiring, diagnostics, experiment gating, and hardware seams. The app now resolves repo-relative runtime paths from the project root, the main bootstrap no longer wires the unused legacy tracking-service stack into the GUI path, and the operator-facing GUI exposes clearer state-dependent controls and fault messaging.
- [x] (2026-03-27 04:35Z) Hardened the hardware boundaries for tomorrow's bench test. `DxlBus` and `OpenRbClient` now fail closed in hardware mode instead of pretending to connect successfully, while dedicated mock implementations keep the validated operator workflow available in mock mode.
- [x] (2026-03-27 04:35Z) Added registration capture support for an optional probe-tip transform via `capture_tool_tip_transform` in `config/registration.yaml`, surfaced that geometry state in the Registration tab, and added regression coverage for the transform seam.
- [x] (2026-03-27 04:35Z) Strengthened experiment and calibration safety behavior. Experiment runs now require servo connection, tracker connection, registration, neutral calibration, and a valid `0A` sample. Neutral calibration saves now archive the previous latest file before overwrite.
- [x] (2026-03-27 04:35Z) Expanded regression coverage around hardware seams, tracker-manager startup guards, neutral-calibration archival, registration tip-transform capture, and experiment prerequisite gating. Verified result at that stage: `51 passed`.
- [x] (2026-03-27 05:15Z) Revalidated the final Pi preflight state. Verified outputs: `python3.11 -m compileall continuum_robot scripts tests`, `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` -> `53 passed`, `.venv/bin/python scripts/run_diagnostics.py --packets 2` in mock mode, offscreen `AppWindow` smoke launch with the expected title and 5 tabs, and a fresh bootstrap run via `VENV_DIR=/tmp/pi_code_bootstrap_smoke_py3 PYTHON_BIN=python3 scripts/bootstrap.sh`.
- [x] (2026-03-27 05:05Z) Final Pi preflight adjustment: lowered the supported Python floor from 3.11 to 3.10 after verifying that every file in `continuum_robot/`, `scripts/`, and `tests/` parses with Python 3.10 grammar. Updated bootstrap and README so Raspberry Pi bring-up uses `python3`/`python3-venv` instead of assuming `python3.11` packages exist.
- [x] Milestone 1: Harden configuration, bootstrap, and service wiring so the app runs end-to-end in mock mode on a Pi or laptop.
- [ ] Milestone 2: Implement production DYNAMIXEL/OpenRB servo control, neutral calibration, and a current-safe pretension routine for 4-servo and 8-servo robots.
- [ ] Milestone 3: Complete the tracker-to-tip runtime chain and guided registration using tool `0B` with persisted acceptance metrics.
- [x] Milestone 4: Replace GUI scaffolds with a local PySide6 operator application.
- [x] Milestone 5: Implement experiment execution, synchronized logging, and one `.dat` file per run.
- [ ] Milestone 6: Finish Raspberry Pi deployment docs, diagnostics, and on-hardware acceptance.

## Surprises & Discoveries

- Observation: The original validation failure was caused by Python 3.9 being too old, not by a hard Python 3.11 requirement in the repo.
  Evidence: `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` from `tests/test_live_registration_service.py` when run with system Python 3.9, followed later by a full Python 3.10 grammar parse of the repo.
- Observation: Current registration capture stores raw tracker translation samples only, which is insufficient if the physical registration point is the pen tip rather than the sensor coil origin.
  Evidence: `LiveRegistrationService.capture_current_sample()` in `continuum_robot/registration/live_registration_service.py` records `tool.translation_mm` directly without applying a probe-tip transform.
- Observation: The tracking runtime is already Pi-native even though it is split across C++ and Python.
  Evidence: `tracker_bridge/tracker_bridge.cpp` opens Aurora on a local serial port and publishes to a Unix socket, and Python consumes that socket on the same host.
- Observation: The original unit suite did not cover the real GUI bootstrap path, so controller/app-window signature mismatches survived until an offscreen smoke run.
  Evidence: `TypeError: TrackingController.__init__() got an unexpected keyword argument 'tracker_manager'` during the first `QT_QPA_PLATFORM=offscreen` window instantiation.
- Observation: Re-running bootstrap against an existing `.venv` can silently preserve the old interpreter unless the script checks the version.
  Evidence: the first bootstrap run reused `.venv/lib/python3.13/...` even though the requested interpreter was `python3.11`; a clean temporary bootstrap under `/tmp/pi_code_bootstrap_smoke` then created `/private/tmp/pi_code_bootstrap_smoke/lib/python3.11/...`.
- Observation: The original diagnostics script bypassed the configured backend and therefore skipped mock mode entirely.
  Evidence: `scripts/run_diagnostics.py` had to be rewritten to use `build_app_context()` and the registered tracker manager before mock-mode diagnostics could run successfully.
- Observation: A clean bootstrap on a fresh machine still depends on network access or a pre-populated wheel cache.
  Evidence: the first fresh `VENV_DIR=/tmp/pi_code_bootstrap_smoke_20260326 PYTHON_BIN=python3.11 scripts/bootstrap.sh` attempt failed under sandboxed network restrictions while trying to resolve `setuptools>=69`; the same command succeeded once network access was allowed.
- Observation: The pre-audit hardware seam was too optimistic for a real bench session.
  Evidence: prior to hardening, `OpenRbClient.connect()` and `DxlBus.connect()` reported success without any real transport implementation, which could have misled tomorrow's hardware bring-up.
- Observation: The GUI runtime still had a hidden launch-cwd dependency.
  Evidence: relative registration/output paths were already resolved correctly in bootstrap for some services but `TrackingController` and related GUI surfaces still built paths from raw config strings until the audit pass fixed them.
- Observation: The experiment tab previously allowed runs with no live tracker sample or servo connection.
  Evidence: `ExperimentController.refresh_prerequisites()` only checked for file, neutral calibration, and registration before the audit pass tightened it to require tracker connection, valid `0A`, and servo connection.
- Observation: The repo no longer appears to require Python 3.11 specifically.
  Evidence: every file under `continuum_robot/`, `scripts/`, and `tests/` parsed successfully under Python 3.10 grammar using `ast.parse(..., feature_version=(3, 10))`.
- Observation: A Pi OS image may not provide `python3.11` packages by name even when the default `python3` is sufficient.
  Evidence: the user's Raspberry Pi apt output failed to locate `python3.11` and `python3.11-venv`, which made the older bring-up instructions too specific for the actual deployment target.

## Decision Log

- Decision: Keep `tracker_bridge` as the runtime Aurora integration path instead of replacing it with a pure-Python serial stack.
  Rationale: the user wants the Pi to be the only host, not necessarily a Python-only Aurora driver. The existing C++ bridge already satisfies direct Pi integration, owns the NDI `CombinedApi` lifecycle cleanly, and reduces risk compared with swapping tracker runtimes mid-build.
  Date/Author: 2026-03-27 / Codex
- Decision: Treat the current repository as an evolutionary build, not a rewrite.
  Rationale: tracking transforms, registration persistence, tests, config templates, and bootstrap scripts already exist and should be extended instead of discarded.
  Date/Author: 2026-03-27 / Codex
- Decision: Interpret “closed loop” for v1 as servo position control with readback confirmation and current/voltage safety monitoring, not full tracker-in-the-loop shape control.
  Rationale: the requirements explicitly prioritize safety and reliability and state that v1 should remain position-based while using current thresholds to prevent over-tension and string pull-through.
  Date/Author: 2026-03-27 / Codex
- Decision: Registration must default to tool `0B` and use a probe-tip transform when one is available.
  Rationale: the requested workflow uses `0B` as the registration pen. Using raw coil translation would bias landmark capture whenever the probe tip is offset from the sensor coil.
  Date/Author: 2026-03-27 / Codex
- Decision: Keep the controller/view-state split for the GUI, but implement real PySide6 widgets and timers instead of placeholder state-holder classes.
  Rationale: the existing controller structure is a useful seam for hardware-independent tests; the missing piece is the actual operator-facing widget layer.
  Date/Author: 2026-03-27 / Codex
- Decision: Save one primary `.dat` file per experiment run and allow optional JSON sidecars for rich metadata.
  Rationale: this preserves compatibility with the requested repeatability-style workflow while still allowing richer provenance without overloading the main `.dat` format.
  Date/Author: 2026-03-27 / Codex
- Decision: Use a synthetic tracker manager rather than the lower-level serial tracking service for the GUI’s validated mock-mode path.
  Rationale: the operator GUI needed live, continuously changing `0A` and `0B` state immediately; the synthetic manager gives that behavior without requiring Aurora packet replay or hardware.
  Date/Author: 2026-03-27 / Codex
- Decision: Keep potentially failing GUI actions inside tab-level safe-call wrappers.
  Rationale: operator errors such as missing neutral setpoints or incomplete registration should update controller state and status panels, not surface as uncaught Qt callback exceptions.
  Date/Author: 2026-03-27 / Codex
- Decision: Make bootstrap recreate the virtualenv when the requested interpreter major/minor differs from the existing environment.
  Rationale: this is the safest way to keep the documented bring-up path portable across machines and across repeated bootstrap runs.
  Date/Author: 2026-03-27 / Codex
- Decision: Fail closed for unimplemented OpenRB and DYNAMIXEL hardware seams.
  Rationale: for bench safety, an explicit "not implemented" error is preferable to falsely reporting successful hardware connection.
  Date/Author: 2026-03-27 / Codex
- Decision: Resolve runtime output and calibration paths from the project root inside the application context.
  Rationale: the Pi launch path must not depend on whichever directory the operator happened to call `scripts/run_gui.sh` from.
  Date/Author: 2026-03-27 / Codex
- Decision: Gate experiment execution on a live valid `0A` sample and an active servo connection.
  Rationale: a repeatability run without live tracking or a connected bus is not operationally useful and should fail before data collection starts.
  Date/Author: 2026-03-27 / Codex
- Decision: Archive the previous neutral calibration before overwriting `neutral_setpoints.json`.
  Rationale: bench calibration is expected to be repeated, and silent overwrite makes recovery harder after a bad setup.
  Date/Author: 2026-03-27 / Codex
- Decision: Support an optional `capture_tool_tip_transform` directly in `config/registration.yaml`.
  Rationale: this matches the intent of the legacy rigid-registration workflow while keeping the transform explicit, testable, and operator-visible.
  Date/Author: 2026-03-27 / Codex
- Decision: Lower the documented and packaging minimum from Python 3.11 to Python 3.10.
  Rationale: this reduces Raspberry Pi bring-up friction while remaining consistent with the repo's current syntax and type-annotation usage.
  Date/Author: 2026-03-27 / Codex

## Outcomes & Retrospective

This plan now has a first implementation round plus a serious pre-flight audit behind it. The repository no longer stops at scaffolding: it has a real PySide6 operator GUI, a validated mock-mode end-to-end workflow, richer config/bootstrap behavior, diagnostics that work against the configured backend, explicit path resolution for Pi launch contexts, and expanded test coverage through 51 passing tests. The biggest remaining bench risk is still the real DYNAMIXEL/OpenRB transport. Tracker/registration bring-up is materially closer to tomorrow-ready: the tracker bridge path is preserved, diagnostics are aligned with the real backend model, the GUI no longer hides several important prerequisite failures, and the registration workflow now has an explicit seam for a probe-tip transform. The remaining hardware acceptance items should be treated as real bench work, not assumed complete because the software layer is greener.

## Context and Orientation

This repository is organized around the right major concerns. `continuum_robot/tracking/` contains Aurora packet framing, parsing, transform math, socket-client code, mock tracking, and `TipPoseService`. `tracker_bridge/tracker_bridge.cpp` is the current Aurora runtime bridge: it connects to the Aurora device on the Raspberry Pi, uses the NDI SDK locally, and streams line-delimited JSON over a Unix domain socket. `continuum_robot/registration/` contains rigid registration solving, capture session state, validation helpers, and persistence of registration outputs under `data/registrations/`. `continuum_robot/servos/` now contains usable displacement mapping, neutral calibration persistence, current-balance pretension validation, and a higher-level servo service, but the low-level real hardware bus layer in `continuum_robot/hardware/dxl_bus.py` is still a scaffold. `continuum_robot/experiments/` now contains CSV loading, `.dat` writing, and a mock-validated `ExperimentRunner`. `continuum_robot/gui/` and `continuum_robot/app/` now contain a real PySide6 application with five operator tabs instead of placeholders.

The protected reference inputs live under `references/` and `tools/`. They may be read during implementation, but they must not be edited, renamed, reformatted, or deleted. This matters because the requested platform must reuse the behavior of legacy scripts such as `references/repeatability.py` and `references/rigid_registration.py` without modifying those files.

Throughout this repository, transforms follow the convention `T_A_B`, meaning “transform coordinates from frame B into frame A.” For example, `T_robot_aurora @ T_aurora_coil @ T_coil_tip` yields `T_robot_tip`. The current code already follows this convention in `README.md` and `continuum_robot/tracking/tip_pose_service.py`, and the plan preserves it.

The target runtime is a Raspberry Pi that directly hosts all three responsibilities: local GUI, Aurora tracker connection, and OpenRB-150 / DYNAMIXEL control. There is no relay Linux box and no Arduino in the v1 architecture. The Aurora cable plugs into the Pi. The OpenRB-150 plugs into the Pi. The operator uses a monitor, keyboard, and mouse directly attached to the Pi. The GUI therefore must be a local desktop application, not a web app.

The user-visible hardware roles are fixed for v1. Tool `0A` is the robot coil used to compute robot tip pose in the robot frame. Tool `0B` is the pen probe used during registration. The servo hardware is XC330-M288 on an OpenRB-150 controller. The robot can be configured as either one segment with 4 servos or two segments with 8 servos. Tendon displacement in centimeters is the operator-facing motion command, and each displacement is converted into DYNAMIXEL position ticks around a saved neutral setpoint using the configured spool diameter and tendon-to-servo mapping.

The most important gap between current code and the requested platform is that the pieces do not yet run as one operator workflow. Tracking and registration logic exist but are only partially surfaced in controllers. Servo safety primitives exist but there is no real DYNAMIXEL integration. Experiment logging exists but there is no executor. The plan below closes those gaps in a sequence that keeps the application testable after every milestone.

## Milestones

### Milestone 1: Make the scaffold runnable as a real application in mock mode

At the end of this milestone, a contributor can bootstrap the repo with Python 3.10 or newer, launch a real PySide6 window on a Pi or laptop, switch between System, Servos, Tracking, Registration, and Experiment tabs, and exercise the full UI flow in mock mode without hardware attached. Nothing should block on missing serial devices. The acceptance proof is that the GUI launches, mock tracker data updates live, mock servo telemetry refreshes, and `.venv/bin/python -m pytest -q` passes on a bootstrapped environment.

This milestone starts by turning configuration into a real runtime model. Extend `continuum_robot/config/schemas.py`, `continuum_robot/config/settings.py`, and `continuum_robot/config/config_loader.py` so they load not just the current robot, serial, and safety settings, but also explicit mock-mode, registration, experiment, and calibration settings. `config/system.yaml`, `config/safety.yaml`, `config/registration.yaml`, `config/experiment.yaml`, `config/robot_4servo.yaml`, and `config/robot_8servo.yaml` should become the authoritative templates. `config/system.local.example.yaml` should document machine-specific serial ports and any Pi-specific overrides.

Then rework `continuum_robot/app/bootstrap.py` and `continuum_robot/app/service_registry.py` so they wire real services based on config. In mock mode, the registry should supply a mock tracker source and `MockDxlBus` implementation; in hardware mode it should supply `TrackerServiceManager`, real `DxlBus`, and `OpenRbClient`. The goal is that controllers do not care whether they are talking to hardware or mocks.

Finally, replace the placeholder `AppWindow` in `continuum_robot/gui/app_window.py` and the placeholder tab classes under `continuum_robot/gui/tabs/` with real PySide6 widgets. Use a `QMainWindow` with a `QTabWidget` and a timer-driven refresh loop. Keep view-state models in the controllers so unit tests can exercise logic without launching Qt. The UI can be visually simple, but it must be real and usable.

### Milestone 2: Finish servo control, safety, neutral calibration, and pretension workflow

At the end of this milestone, the operator can connect to the OpenRB-150 from the GUI, scan servos, assign IDs in a safe single-servo workflow, jog a servo, capture neutral setpoints from Present Position address 132, define safe travel bounds around neutral, command tendon displacement in centimeters, and run a startup pretension routine that steps tendons until they reach a current target window without exceeding configured safety limits. The acceptance proof is a mock-backed test suite plus on-hardware manual verification that unsafe commands are rejected and valid displacement commands land consistently near the requested positions.

Implement the low-level DYNAMIXEL integration in `continuum_robot/hardware/dxl_bus.py`. This layer must own raw register reads and writes only. It must expose connection management, ID scanning, goal position writes, Present Position reads, Present Current reads, Present Input Voltage reads, and ID reassignment. If the Python DYNAMIXEL SDK is used, wrap it here and keep the rest of the application SDK-agnostic. `continuum_robot/hardware/openrb_client.py` should remain board-specific and limited to safe setup or mode-preparation actions; it must not implement risky firmware flashing.

Expand `continuum_robot/servos/displacement_mapper.py` so it converts tendon displacement to servo ticks using spool circumference, optional direction reversal, per-servo offsets, and 4-servo or 8-servo mapping. Expand `continuum_robot/servos/safety_guard.py` so it validates both static command bounds and live current/voltage limits. `continuum_robot/servos/servo_service.py` must become the single orchestration layer for jog, absolute-neutral capture, tendon displacement command, telemetry polling, and motion-completion checks. Motion completion for v1 means the service verifies that present position converged within tolerance or raises a clear error.

Implement the neutral calibration workflow in `continuum_robot/servos/neutral_calibration_service.py`. The operator-visible sequence is fixed: place the robot in neutral backbone pose, manually jog each motor until the tendon tension feels correct, read Present Position, save that tick value as the neutral setpoint, and persist safe min/max offsets around it. Persist these calibrations under `data/calibrations/` and also write a human-readable export under `config/` only if the team decides those values should be source-controlled. Do not overwrite calibration data silently.

Implement the startup pretension routine in `continuum_robot/servos/pretension_validation_service.py` and its caller in `ServoService`. The algorithm should start from the saved neutral setpoints, increment each servo by small steps, measure current after each step, stop when the servo enters the configured current target window, and abort if the servo exceeds current, voltage, or travel limits. After all tendons are tensioned, compute current spread and require it to fall within a configured balance tolerance. Record the final currents, positions, and pass/fail reason so the GUI can explain what happened.

### Milestone 3: Complete runtime tracking and guided registration

At the end of this milestone, the operator can connect to Aurora, observe live status for tools `0A` and `0B`, see `T_robot_tip` update in the robot frame, launch a guided registration workflow that uses `0B`, collect repeated landmark captures, inspect residual error, accept or reject the solve, and save the resulting transforms so the runtime tracking chain uses them automatically. The acceptance proof is a test suite that covers the transform chain and an interactive registration run that produces a `latest_registration.json` file and immediately enables `T_robot_tip` display.

Start by aligning terminology and persistence. `config/registration.yaml` must default to `capture_tool_id: "0B"` and express nominal landmark coordinates in the robot frame. Extend the config to include the default number of landmarks, captures per landmark, and any tip-offset transform required for the registration probe. If the physical pen tip is offset from the tracked coil, introduce an explicit transform such as `T_probecoil_probetip` or a more precise name that matches the hardware. Do not hide this offset inside ad hoc math.

Then extend `continuum_robot/registration/live_registration_service.py`, `continuum_robot/registration/capture_session.py`, `continuum_robot/registration/repository.py`, and `continuum_robot/registration/validation.py`. The service must capture repeated samples for each landmark, apply the registration probe tip transform if configured, average samples, solve the rigid transform, compute residuals and FRE, and persist both the raw captures and the accepted transforms. Because the current `RegistrationRecord` field names blur Aurora-frame measurements and robot-frame truth, rename or version the record schema so the stored data names are honest. Keep backward-compatible reads for existing files when practical.

Integrate live tip-pose computation into tracking. `continuum_robot/tracking/tip_pose_service.py` already computes `T_robot_tip` from registration and tool `0A`; wire this into a higher-level service or controller so the tracking tab shows live numeric position and orientation, valid/invalid tool state, and an explicit “registration missing” indicator. `continuum_robot/tracking/tracker_service_manager.py` should remain responsible for bridge lifecycle and latest-sample state; avoid burying transform math inside it.

If the Python raw packet parser remains in the repository, keep it as a tested unit-level parser and diagnostic fallback. The main runtime path should still be `tracker_bridge` unless on-hardware evidence later proves that a different path is required.

### Milestone 4: Build the operator GUI around safe workflows

At the end of this milestone, the GUI is the primary operator surface on the Raspberry Pi. The System tab connects and diagnoses Aurora and OpenRB-150. The Servos tab scans, assigns IDs, jogs motors, saves neutral calibration, shows live telemetry, and runs pretension. The Tracking tab shows live tool state and robot tip pose. The Registration tab guides landmark capture and accept/retry flow. The Experiment tab loads a point file, runs it, and shows progress. The acceptance proof is that a lab operator can complete a full session from boot to registration to experiment without dropping to the terminal except for initial installation.

Keep the controller-driven architecture but make the controllers real. `continuum_robot/gui/controllers/system_controller.py` must own port selection, connect/disconnect, serial port enumeration, and OpenRB prepare actions. `continuum_robot/gui/controllers/servos_controller.py` must own jog, displacement command, calibration save/load, and pretension actions. `continuum_robot/gui/controllers/tracking_controller.py` must surface the live state already tracked by `TrackerServiceManager` plus the computed tip pose and tool quality. `continuum_robot/gui/controllers/registration_controller.py` must grow beyond start/capture/finish calls to expose capture counts, current landmark, residual metrics, accept/retry commands, and saved result path. `continuum_robot/gui/controllers/experiment_controller.py` must own experiment load, run, stop, progress, and output path state.

The tabs under `continuum_robot/gui/tabs/` should become concrete PySide6 widgets. They do not need fancy graphics. Simple forms, live numeric labels, status banners, a small 3D or 2D plot widget, and clearly labeled buttons are enough. Use color and layout to highlight invalid tool state, motion rejections, missing registration, and pretension failures. The GUI must never silently swallow an unsafe request. Every failed connect, failed move, or failed registration must surface a readable reason.

Add a thin visualization layer that is intentionally modest. For v1, a simple live scatter or axes view of landmarks, the current pen position, and the current tip position is enough. If Qt-native plotting is awkward on the Pi, a lightweight Matplotlib or pyqtgraph view is acceptable, but keep performance and installation complexity in mind.

### Milestone 5: Implement experiment execution and synchronized run logging

At the end of this milestone, the operator can load an experiment file, execute each commanded tendon displacement state in sequence, wait the configured settle time, sample tracker and servo state, compute the robot tip pose in the robot frame, and write exactly one `.dat` file per run. The acceptance proof is a repeatability-style run that produces a readable `.dat` file under `data/runs/` and optionally a sidecar metadata file with the calibration and registration identifiers used for that run.

Implement the runtime loop in `continuum_robot/experiments/experiment_runner.py`. It must use `ExperimentLoader` to load CSV points, `ServoService` to command motion, the tracking stack to sample `0A` and derive `T_robot_tip`, and `DatRunWriter` to write the output. Honor per-point settle overrides from the CSV and fall back to `config/experiment.yaml` for defaults. Keep the main run loop single-threaded and explicit unless timing demands otherwise; the first goal is reliable logs, not maximum throughput.

Expand `continuum_robot/experiments/dat_writer.py` so each row includes, at minimum, timestamp, point index, commanded tendon displacement, measured servo positions, measured current and voltage when available, raw tool data needed for later debugging, computed tip pose in robot frame, and enough metadata to identify the registration and neutral calibration used. The main `.dat` file should stay line-oriented and easy to parse. If this makes the file too verbose, keep the primary columns in the `.dat` and write richer structured metadata to a sidecar JSON file.

Use `references/repeatability.py` as the behavioral reference, not as a file to duplicate. The new runner should preserve the useful behavior from the reference script: move to a point, wait, sample, log, and return to neutral when the run is complete or aborted.

### Milestone 6: Finish deployment, diagnostics, documentation, and hardware acceptance

At the end of this milestone, a fresh Raspberry Pi can be prepared by following the README, the bridge can be built after the NDI SDK is installed, the GUI and diagnostic scripts run with clear instructions, mock mode remains available for development, and the repo includes example configs for 4-servo and 8-servo modes plus an example experiment file. The acceptance proof is that a new operator can follow the documented setup on a clean Pi and reach the GUI without tribal knowledge.

Update `README.md`, `requirements.txt`, `pyproject.toml`, `scripts/bootstrap.sh`, `scripts/run_gui.sh`, `scripts/run_diagnostics.py`, and any additional launch scripts needed for production use. The README must document the Pi wiring model, how tendon displacement converts to servo ticks, how neutral setpoints are established, how registration is performed, how experiment files are structured, and what the `.dat` outputs contain. It must also document the Python 3.10+ requirement explicitly so contributors do not repeat the current validation failure.

Create or update tests across `tests/` to cover every new service boundary. The minimum expected additions are unit tests for the DYNAMIXEL mapping and safety logic, neutral calibration persistence, pretension routine behavior, controller state transitions, experiment execution with mock services, and compatibility loading for registration outputs. Use mock classes or replayed tracker data so most tests run without hardware. Reserve only a small manual acceptance checklist for the actual Pi and robot.

## Plan of Work

Begin with configuration and runtime composition because that unblocks every later subsystem. Add explicit config dataclasses for robot mode, serial and OpenRB settings, tracker settings, servo safety limits, neutral-calibration paths, registration workflow defaults, experiment defaults, and mock-mode toggles. Update `build_app_context()` so every controller gets a complete service bundle and so the bundle can be swapped between mock and real backends without changing controller code.

After that, implement the real DYNAMIXEL bus and servo orchestration. Keep the low-level SDK or packet code in `continuum_robot/hardware/dxl_bus.py` only. Place all human-facing behavior in `continuum_robot/servos/`. This separation matters because servo math, safety, and workflow logic will need tests that run without hardware. Do not let GUI code reach directly into registers or serial ports.

Next, complete the tracking-registration chain. Preserve `TrackerServiceManager` as the process and socket manager, preserve `TipPoseService` as the transform-chain owner, and keep registration solving in `registration/rigid_solver.py`. The implementation task here is to make the interfaces honest and operator-usable: correct tool IDs, explicit tip transforms, accurate persisted naming, and live controller state that explains what is missing when tracking or registration is unavailable.

Once the services are solid, build the real PySide6 GUI. Keep the UI thin. Controllers should expose structured state objects, and widgets should translate that state into labels, buttons, plots, and dialogs. Use Qt timers or signals so the UI stays responsive. Every connect, disconnect, jog, capture, and run action must be non-blocking from the operator’s perspective, with long-running work handled by threads or worker objects only where needed.

Then implement the experiment runner on top of the finished servo and tracking services. The runner should refuse to start when required prerequisites are missing, such as a missing registration or neutral calibration. It should also capture enough metadata at run start to make the resulting files traceable later.

Finish by tightening docs, tests, and launch scripts around the actual Raspberry Pi deployment path. The result should be a codebase that is immediately usable in the lab, debuggable when hardware is misbehaving, and safe enough to trust for repetitive registration and repeatability studies.

## Concrete Steps

Work from the repository root:

    cd /Users/jacklange/Continuum/pi_code

Bootstrap a Python 3.10+ environment before running tests or the GUI:

    PYTHON_BIN=python3 scripts/bootstrap.sh

Expected result:

    Project root: /Users/jacklange/Continuum/pi_code
    Using Python: python3
    ...
    Bootstrap complete.

Run the unit test suite from the bootstrapped environment, not the system Python:

    .venv/bin/python -m pytest -q

Before `scripts/bootstrap.sh`, it is expected that `python3 -m pytest -q` may fail with `No module named pytest`. After bootstrap, the same command through `.venv/bin/python` should succeed.

When working on tracker integration and the NDI SDK is installed, build the bridge like this:

    export NDI_SDK_INCLUDE_DIR=/opt/ndi_sdk/include
    export NDI_SDK_LIB_DIR=/opt/ndi_sdk/lib
    BUILD_TRACKER_BRIDGE=1 PYTHON_BIN=python3 scripts/bootstrap.sh

For machine-local serial paths, copy the local override file once:

    cp config/system.local.example.yaml config/system.local.yaml

Then edit `config/system.local.yaml` for the Pi’s Aurora and OpenRB ports. This is safe to repeat because the file is machine-local and git-ignored.

During tracking work, use the diagnostics script to verify bridge output and optional tip-pose computation:

    .venv/bin/python scripts/run_diagnostics.py --tracker-port /dev/ttyUSB0 --packets 10

Expected results include a sequence of tracker states, tool samples, and either a `T_robot_tip translation` line or a clear message that registration is unavailable.

Use the GUI launch path during every milestone after mock mode is wired:

    scripts/run_gui.sh

The window should open locally on the Pi. In mock mode it must not require hardware. In hardware mode it should show connection errors as readable status text instead of crashing.

## Validation and Acceptance

A milestone is not complete when the code merely imports. It is complete when a human can observe the intended behavior.

For Milestone 1, acceptance means a contributor can bootstrap with Python 3.10 or newer, open the GUI, switch tabs, see live mock data, and run the test suite successfully from `.venv/bin/python`.

For Milestone 2, acceptance means the Servos tab can connect to a real or mock bus, scan IDs, jog a servo, save neutral setpoints, map a tendon displacement to predictable goal ticks, and reject any command that exceeds configured travel or current thresholds. On hardware, intentionally commanding an out-of-bounds displacement must produce a visible error and no motion.

For Milestone 3, acceptance means the Tracking tab shows live `0A` and `0B` state, the Registration tab can collect repeated `0B` captures, the solver computes a transform with residual metrics, `data/registrations/latest_registration.json` is updated, and the Tracking tab begins showing `T_robot_tip` in the robot frame without restarting the app.

For Milestone 4, acceptance means a lab operator can connect devices, calibrate, register, and troubleshoot from the GUI alone, with no placeholder prints or missing buttons.

For Milestone 5, acceptance means an experiment CSV can be loaded, executed, and logged into exactly one `.dat` file per run under `data/runs/`, with the file containing enough servo, tracker, and pose data to audit the run later.

For Milestone 6, acceptance means a new Raspberry Pi can be prepared by following the README, and a short manual checklist confirms Aurora connectivity, OpenRB connectivity, neutral calibration, registration, and one saved experiment run.

## Idempotence and Recovery

All runtime outputs must stay under `data/` so they can be deleted or regenerated without damaging source code. Never write generated files into `references/` or `tools/`. Safe retries matter in this project because registration and calibration are expected to be repeated often.

Configuration edits under `config/` should be additive and human-readable. If a calibration or registration run is bad, the operator should be able to reject it in the GUI and leave the previously accepted `latest_*.json` files untouched. When a new calibration or registration is accepted, write a timestamped archival file first and only then update the `latest_*.json` pointer file.

Servo ID assignment is the riskiest recovery case. Only support ID reassignment in an isolated workflow where exactly one target servo is powered or selected at a time. If ID assignment fails, instruct the operator to disconnect the rest of the chain or rerun scan mode before retrying. Do not broadcast ID changes blindly.

Experiment runs must fail closed. If registration is missing, neutral calibration is missing, pretension fails, or current limits are exceeded, the run should stop, write an explicit failure reason, and leave the robot either at neutral or in the safest reachable state defined by the servo service. Never continue logging as if the run succeeded.

## Artifacts and Notes

Current evidence gathered while writing this plan:

    $ pytest -q
    ...
    ERROR tests/test_live_registration_service.py - TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'

This proves the system Python on this machine is too old for the repo’s current type syntax.

    $ python3.11 --version
    Python 3.11.13

This confirms the correct interpreter is available.

    $ python3.11 -m pytest -q
    /opt/homebrew/opt/python@3.11/bin/python3.11: No module named pytest

This proves the repo must be bootstrapped before validation.

Relevant current files that already contain useful behavior and should be extended rather than replaced are:

- `tracker_bridge/tracker_bridge.cpp`
- `continuum_robot/tracking/tip_pose_service.py`
- `continuum_robot/registration/live_registration_service.py`
- `continuum_robot/registration/repository.py`
- `continuum_robot/experiments/experiment_loader.py`
- `continuum_robot/experiments/dat_writer.py`

Relevant current scaffolds that must be completed are:

- `continuum_robot/hardware/dxl_bus.py`
- `continuum_robot/hardware/openrb_client.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/experiments/experiment_runner.py`
- `continuum_robot/gui/app_window.py`
- `continuum_robot/gui/controllers/system_controller.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `continuum_robot/gui/controllers/experiment_controller.py`
- most classes in `continuum_robot/gui/tabs/`

## Interfaces and Dependencies

Use Python 3.10+ and PySide6 for the local GUI. Keep `numpy`, `PyYAML`, and `pyserial` as the core Python dependencies already present in `pyproject.toml`. If the DYNAMIXEL Python SDK is needed, add it explicitly and wrap it inside `continuum_robot/hardware/dxl_bus.py` rather than exposing SDK types across the codebase.

At the end of Milestone 1, `continuum_robot/config/settings.py` should define a single `Settings` object that includes, directly or via nested dataclasses, serial settings, robot mode settings, safety limits, registration settings, experiment defaults, calibration paths, and a `mock_mode` flag.

At the end of Milestone 2, `continuum_robot/hardware/dxl_bus.py` must provide a stable `DxlBus` interface with methods equivalent to:

    connect(port: str, baudrate: int) -> None
    disconnect() -> None
    scan_ids(min_id: int = 1, max_id: int = 20) -> list[int]
    read_telemetry(servo_ids: list[int]) -> dict[int, ServoTelemetry]
    write_goal_positions(positions_by_id: dict[int, int]) -> None
    write_servo_id(current_id: int, new_id: int) -> None

`ServoTelemetry` must include present position, current, voltage, and any hardware fault information available from the XC330-M288 control table. `ServoService` should expose high-level methods for jog, displacement command, neutral capture, pretension routine, and state polling. The GUI must depend on `ServoService`, not `DxlBus`.

At the end of Milestone 3, the tracking-registration boundary must be explicit. `TrackerServiceManager` owns bridge lifecycle and latest tool samples. `TipPoseService` owns `T_robot_tip` computation. `LiveRegistrationService` owns repeated capture and solving. The registration repository must write enough data to reconstruct a solve later and must keep backward-compatible reads for existing saved registrations where practical.

At the end of Milestone 4, each GUI controller should expose a structured state object that can be inspected by tests without launching Qt. The widgets under `continuum_robot/gui/tabs/` should depend on controller state and callbacks only.

At the end of Milestone 5, `ExperimentRunner.run(...)` should return the path of the created `.dat` file and a run summary object or raise a clear exception. The `.dat` writer should be the only code that formats run rows for disk.

Revision note: 2026-03-27. Replaced the generic ExecPlan template in `plans.md` with a repository-specific implementation plan because the current repository already contains a partial Raspberry Pi continuum robot scaffold and the user asked for a full-platform build plan grounded in the existing code.
Revision note: 2026-03-27. Updated the plan after the first implementation pass to record completed mock-mode GUI/bootstrap/diagnostics work, new validation evidence, portability fixes, and the remaining hardware-dependent gaps.
