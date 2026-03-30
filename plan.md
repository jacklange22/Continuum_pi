# Continuum Robot Implementation Plan

Last updated: 2026-03-28

This document is a grounded audit and phased plan for the current repo state. It intentionally focuses on what is actually implemented today in this codebase, what is still missing relative to the project target, and what the most disciplined next steps are.

This `plan.md` supersedes the older `plans.md` as the primary planning document.

## Project Target

The target system is a Raspberry Pi based operator app for a tendon-driven continuum robot that can:

- control 4 or 8 Robotis XC330/XC333 smart servos through an OpenRB-150
- use current-aware safety and a guided pretension / neutral calibration workflow
- ingest Aurora tracking directly on the Pi and compute live tool and robot-tip pose
- perform GUI-based 4-point robot-body registration with a pen probe
- run calibration, validation, and repeatability experiments from one operator app
- log synchronized servo command/state data and tracking/pose data for later analysis

The main scientific outcome is repeatability, so servo safety, tracking trustworthiness, and clean experiment logging matter more than additional GUI polish.

## Executive Summary

The repo is structurally strong as a software platform. Tracking, registration, experiment architecture, data schemas, and GUI workflow are much more mature than the hardware-control path. The current codebase is already a credible mock-mode operator app, but it is not yet a lab-ready robot operator system because the real OpenRB/DYNAMIXEL servo stack is still missing.

Current high-level status:

- Tracking: architecturally strong, diagnostics-rich, still needs bench validation
- Registration: correct direction and mostly aligned to the actual 4-point workflow
- Experiments: canonical framework and critical experiments are in good shape
- GUI: broad, cohesive, and fairly polished
- Servo/OpenRB: still the main blocker to end-to-end real robot operation

Current automated validation:

- `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`
- Result on 2026-03-28: `116 passed`

## 1. Current-State Audit

### 1.1 Servo Subsystem

Current implementation status: conceptually scaffolded, mock-mode usable, hardware path not complete

What exists now:

- High-level servo coordination lives in `continuum_robot/servos/servo_service.py`
- Manual jog, displacement command dispatch, neutral setpoint capture/load/save, telemetry reads, servo ID scan/assign hooks, and pretension validation hooks all exist at the service/controller level
- Tendon displacement to goal-position mapping exists in `continuum_robot/servos/displacement_mapper.py`
- Safety threshold checks exist in `continuum_robot/servos/safety_guard.py`
- Neutral setpoints persist to `data/calibrations/neutral_setpoints.json` via `continuum_robot/servos/neutral_calibration_service.py`
- The config model supports both 4-servo and 8-servo robots via `config/robot_4servo.yaml`, `config/robot_8servo.yaml`, and `RobotConfig`
- `tendon_to_servo` config and a mapping helper exist in `continuum_robot/servos/tendon_mapping.py`
- GUI surfaces for servo connect, scan, ID assign, jog, neutral capture, and displacement commands exist

What is not implemented yet:

- Real OpenRB/DYNAMIXEL transport is not implemented
- `continuum_robot/hardware/dxl_bus.py` intentionally fails in hardware mode for connect, telemetry, scan, ID assignment, and goal writes
- `continuum_robot/hardware/openrb_client.py` intentionally fails in hardware mode for connect and prepare
- No verified reads of Present Position(132), Present Current, or Present Voltage from real XC330/XC333 hardware exist in the repo today
- No firmware upload path or real field-service workflow exists for OpenRB-150
- No active closed-loop safe control exists; commands are still position commands with post-write current checking
- No current-based backoff or hold logic exists
- No guided pretension routine exists; only a current spread validator exists
- Neutral persistence stores only setpoint ticks, not calibrated min/max limits or richer safety metadata
- The live command path still does not clearly use `tendon_to_servo` mapping as a canonical part of actuation

Bottom line:

- The servo layer is still the main missing subsystem relative to the target system

### 1.2 Tracking Subsystem

Current implementation status: strong architecture, strong diagnostics, still needs hardware acceptance

What exists now:

- `continuum_robot/services/tracking_service.py` is the canonical app-visible tracking service
- `continuum_robot/tracking/backend_router.py` implements deterministic backend selection and fallback
- Canonical backend states are implemented: disabled, mock, connecting, streaming healthy/degraded, disconnected, error
- Preferred live path is the Python NDI backend
- Bridge backend remains available as explicit fallback/debug path
- Tool normalization, tool ID aliasing, backend health metadata, freshness, FPS, stale-frame handling, and role mapping exist
- Tracking diagnostics and staged validation exist in `continuum_robot/tracking/diagnostics.py`
- CLI validation tooling exists for doctor / smoke / benchmark workflows
- GUI tracking surfaces already expose connection state, backend selection, tool visibility, readiness, and diagnostic details
- Mock mode is supported
- Tip pose composition exists through the canonical transform chain

What is not yet proven:

- Real Aurora-on-Pi behavior is not yet proven in this repo by hardware acceptance
- Current live tool ID mapping assumptions still need real bench confirmation
- Bridge fallback remains optional and depends on external build/runtime setup

Bottom line:

- Tracking is one of the strongest subsystems in the codebase, but it is still “software-complete, hardware-pending”

### 1.3 Registration Subsystem

Current implementation status: good workflow direction, real persistence, still needs live acceptance

What exists now:

- Canonical registration service is in `continuum_robot/services/registration_service.py`
- Registration persistence is in `continuum_robot/registration/repository.py`
- The repo now defaults to a 4-point registration workflow through `config/registration.yaml`
- Repeated captures per landmark are supported
- Live current-point preview exists
- Solve, review, save, retry, and overwrite confirmation flows exist in the controller/UI
- Registration quality metrics including FRE/RMSE and residuals are surfaced
- Accepted registrations persist to `data/registrations/latest_registration.json` plus timestamped registration records
- Pen-probe file support remains available through config without modifying protected `tools/`

What is still open:

- Final live acceptance on real Aurora + pen probe hardware
- Confirmation that the nominal 4 body landmarks in config match the actual robot geometry used in the lab
- Bench confirmation that the saved transform chain gives reliable robot-tip-in-body-frame every run

Bottom line:

- Registration is no longer the main design problem; it is now primarily a correctness/acceptance problem on live hardware

### 1.4 GUI / Operator Workflow

Current implementation status: broad and coherent, but maturity is uneven across subsystems

What exists now:

- Single PySide6 operator app with system, servos, tracking, registration, and experiment tabs
- Canonical experiment workspace for:
  - `repeatability_dataset`
  - `aurora_grid_accuracy`
  - `pivot_calibration`
- Strong preflight, run-history reload, results viewing, and visualization-safe fallbacks
- Tracking and registration UIs are much more guided than earlier repo versions
- System tab surfaces serial ports, tracker status, and OpenRB/DYNAMIXEL status
- Experiment workspace uses the canonical experiment runner rather than a GUI-only execution path

Current limitations:

- The GUI is ahead of the hardware-control layer
- In hardware mode, the system and servo tabs still run into intentionally unimplemented OpenRB/DYNAMIXEL seams
- That creates a risk that the operator experience appears more complete than the underlying robot-control capability actually is

Bottom line:

- GUI scope is broad enough for a real operator app, but its real-world value is currently constrained by missing servo hardware integration

### 1.5 Experiment Framework and Critical Experiments

Current implementation status: strong

What exists now:

- Canonical experiment framework in `continuum_robot/experiments/`
- Shared lifecycle abstraction, metadata schema, timeseries schema, summary schema, dataset writer/loader, registry, and CLI runner
- Canonical outputs:
  - `metadata.json`
  - `samples.jsonl`
  - `summary.json`
  - `config_snapshot.yaml`
- First-class critical experiments exist:
  - `repeatability_dataset`
  - `aurora_grid_accuracy`
  - `pivot_calibration`
- Shared metrics, pivot solving, validation, replay, and analysis helpers exist
- Dry-run/offline/mock execution paths exist where appropriate
- GUI experiment workspace consumes the same canonical runner/data path

What is still missing:

- Live servo-backed repeatability execution is blocked by the missing real OpenRB/DYNAMIXEL path
- Main repeatability logging will remain partial until real servo command/state telemetry is available
- Some acceptance thresholds still need bench tuning on real hardware

Bottom line:

- The experiment subsystem is ready to support the real robot once the servo stack becomes real

### 1.6 Data Logging / Output Formats

Current implementation status: strong for experiments and registration, thin for servo calibration state

What exists now:

- Canonical experiment datasets with metadata + timeseries + summary + config snapshot
- Registration records with transforms, raw captures, averages, residuals, config-used, and validation metrics
- Neutral setpoints persisted to JSON
- Tracking packet capture directory and diagnostic/report paths exist

Current gaps:

- Neutral calibration artifact is too thin for safe live use
- No richer persisted servo calibration state for:
  - safe min/max per servo
  - pretension acceptance status
  - current/torque thresholds
  - calibration provenance
- Servo-side run provenance in experiments will remain incomplete until live transport + telemetry exists

### 1.7 Raspberry Pi Deployment Assumptions

Current implementation status: reasonable development assumptions, not yet full deployment packaging

What exists now:

- `scripts/bootstrap.sh` creates the venv, installs deps, and prepares runtime directories
- `config/system.local.example.yaml` defines machine-local serial-port and backend overrides
- `scripts/run_gui.sh` starts the GUI and gives desktop/offscreen guidance
- Python 3.10+ is required
- Main dependencies are modest:
  - `numpy`
  - `PyYAML`
  - `pyserial`
  - `scikit-surgerynditracker`
  - `PySide6`

What is still missing:

- No systemd/service packaging
- No installer or one-command hardware bring-up flow
- No documented real OpenRB udev/permissions workflow
- No hardware acceptance checklist for the Pi image itself

Bottom line:

- Pi deployment assumptions are fine for development, but not yet packaged into a fully repeatable lab deployment story

## 2. Gap Analysis Against Target Requirements

### 2.1 Servo Requirements vs Current State

Requirement | Current state | Gap
--- | --- | ---
OpenRB150 working with XC330/XC333 servos | Not implemented in repo hardware path | Critical blocker
Connect / troubleshoot servos | GUI scaffolding exists; real transport absent | Critical blocker
Firmware upload / ID assignment workflow | ID assignment hook exists; no real transport or firmware path | Major gap
Manual verify / jog motion | Mock mode only, real jog not possible yet | Major gap
Tendon length change command path | Software mapping exists | Needs real transport and mapping integration
Read Present Position(132) and related state | Interface exists, real bus read not implemented | Major gap
Current and voltage visibility | Telemetry model exists | Needs real readback
Closed-loop safe control foundation | Not present | Major gap
Current-based string protection | Post-write current threshold only, not fail-closed | Major gap
Pretension routine to target torque/current | Not implemented | Major gap
Save neutral setpoints | Implemented | Incomplete artifact
Safe min/max around each setpoint | Not persisted today | Major gap
Use calibrated values in future commands | Partially via global offsets; not per-servo calibrated state | Major gap
Support 4 and 8 servos | Config supports both | Needs real live-path verification and mapping closure

Conclusion:

- The repo has the right top-level abstractions, but the servo subsystem is still far from the target end state because the hardware seam and the calibration/safety model are not complete

### 2.2 Tracking Requirements vs Current State

Requirement | Current state | Gap
--- | --- | ---
Raw Aurora data to tool transforms | Implemented via NDI or bridge backend | Needs hardware proof
Reliable tip/orientation computation | Implemented in canonical service | Needs hardware proof
Transform into robot frame every run | Implemented when registration exists | Needs registration + live validation
GUI validation of tracker connection/tool visibility | Implemented | Good
Pivot calibration / tip-file generation for 0B | Implemented as first-class experiment | Needs live validation
4-point GUI registration with pen probe | Implemented | Needs live acceptance
Saved registration data and outputs | Implemented | Good
Live robot-tip-in-body-frame availability | Implemented when registration + live tracking are valid | Needs live validation

Conclusion:

- Tracking is close to target architecturally; the remaining work is mostly hardware acceptance and tuning

### 2.3 GUI Requirements vs Current State

Requirement | Current state | Gap
--- | --- | ---
Single operator app on Raspberry Pi | Implemented | Good
Device boot/connect status | Implemented | Good
Servo control and troubleshooting | GUI present, real transport absent | Major hardware gap
Manual servo control / tendon-length entry | GUI and service path exist | Blocked by real transport
Pretension and neutral calibration workflow | Partial | Major gap
Aurora connection and validation | Implemented | Good
Registration workflow | Implemented | Needs live validation
Live visualization | Implemented with safe fallback path | Good enough for now
Experiment workflow | Implemented | Good
Indicators for state validation | Implemented strongly in tracking/experiments | Needs servo-side readiness closure
Clean, trustworthy operator UX | Mixed: strong UI, but servo core still missing | Risk of false completeness

Conclusion:

- GUI requirements are mostly met at the workflow level except where servo functionality depends on the missing hardware-control layer

### 2.4 Experiment Requirements vs Current State

Requirement | Current state | Gap
--- | --- | ---
Main repeatability experiment logging servo + Aurora data | Architecture exists | Real live servo data blocked
Aurora grid accuracy test with RMS error | Implemented | Needs live bench use
Pivot calibration / tip file generation | Implemented | Needs live bench use
Clean saved outputs for later analysis | Implemented | Good

Conclusion:

- Experiment design is in good shape; the missing live servo foundation prevents full scientific use of repeatability experiments

## 3. Mismatch / Risk Areas

### 3.1 Conceptually Strong but Not Hardware-Proven

- Tracking backend routing, diagnostics, staged validation, and GUI surfaces are strong
- Registration workflow and persistence are strong
- Experiment architecture and schemas are strong
- None of that is the same as proving the real Aurora + pen probe + OpenRB + XC330/XC333 loop on the Pi

### 3.2 Missing Real Servo Foundation

- The biggest project risk is not UI polish, plotting, or experiment design
- It is the absence of a real, verified OpenRB/DYNAMIXEL hardware path
- Until that exists, main project requirements are blocked no matter how polished the rest of the app becomes

### 3.3 Safety Semantics Are Not Yet Appropriate for Live Tendon Hardware

- Missing current telemetry is not treated as unsafe
- Current checks happen after command writes rather than as part of a safer live control strategy
- Neutral calibration is too thin for a real safety envelope
- Pretension is only validated, not performed

### 3.4 GUI Is More Mature Than Core Actuation Capability

- The app now looks like a plausible operator cockpit
- That is good, but it also creates operator-risk if the plan does not clearly prioritize the underlying hardware-control gap
- Future work should favor real robot correctness before additional cosmetic feature expansion

### 3.5 Potential Operator Confusion

Likely confusion areas if the repo is used on hardware right now:

- system tab suggests hardware connectivity surfaces exist, but the servo transport still intentionally fails
- servo workflows look available but are only mock-validated
- repeatability looks first-class, but live repeatability is still blocked on the servo stack

### 3.6 Missing Persistent Servo Calibration Model

- There is no canonical persistent artifact yet for a real servo calibration state
- That means no clean ownership of:
  - neutral setpoints
  - safe min/max per servo
  - pretension acceptance state
  - calibration timestamp / operator / provenance
  - hardware IDs and robot configuration compatibility

### 3.7 Deployment / Pi Readiness Still Shallow

- Development bootstrap is good enough
- Real lab deployment still needs a more explicit machine-setup and permissions story

## 4. Recommended Architecture Boundaries Going Forward

These are the canonical paths the repo should preserve.

### 4.1 Servo Control Service

Canonical boundary:

- `ServoService` remains the only high-level motion/calibration interface used by controllers and experiments

Responsibilities that belong here:

- safe command dispatch
- tendon-length / tendon-delta to servo-goal conversion
- neutral capture/load/save
- live telemetry reads
- safety-bound validation
- pretension routine orchestration

Responsibilities that do not belong in GUI code:

- servo math
- transport calls
- current/voltage safety logic
- calibration persistence logic

Recommended sub-boundaries:

- `hardware/dxl_bus.py`: raw DYNAMIXEL/OpenRB transport
- `servos/displacement_mapper.py`: deterministic conversion math
- `servos/tendon_mapping.py`: tendon-order to servo-order mapping
- `servos/safety_guard.py`: bounds + telemetry safety policy
- `servos/neutral_calibration_service.py`: richer calibration artifact persistence
- new `servos/pretension_service.py`: active pretension/centering routine

### 4.2 Tracking Service

Canonical boundary:

- `TrackingService` stays the only app-visible live tracking path

Responsibilities:

- backend selection / fallback through router
- snapshot normalization
- freshness / stale handling
- tool identity normalization
- tip pose availability
- diagnostics surfaces used by GUI, CLI, registration, and experiments

No subsystem should bypass it to talk directly to lower-level tracker backends unless there is a very narrow diagnostics-only reason.

### 4.3 Registration Service

Canonical boundary:

- `RegistrationService` remains the only registration-session and solve/save interface

Responsibilities:

- session state
- capture accumulation
- repeated sample handling
- solve and validation
- persistence
- loading latest accepted registration

### 4.4 Experiment Runner

Canonical boundary:

- `ExperimentRunner` stays the one experiment execution path for CLI and GUI

Responsibilities:

- lifecycle orchestration
- metadata injection
- summary classification
- output writing
- stop/cancel handling
- config snapshotting

### 4.5 GUI Controllers and Widgets

Canonical boundary:

- controllers own presentation state and call services/runners
- widgets render state and send user intents back to controllers

Avoid:

- GUI-only experiment execution logic
- GUI-only registration math
- GUI-side safety checks that duplicate service-layer policy

### 4.6 Persistent Calibration / Config / State Files

Recommended durable file ownership:

- `config/system.yaml`: repo defaults
- `config/system.local.yaml`: machine-local ports and runtime overrides
- `data/calibrations/neutral_setpoints.json`:
  - should evolve into a richer servo calibration artifact
- `data/tip_cals/*.csv`:
  - pen-probe tip calibrations
- `data/registrations/latest_registration.json` and timestamped registration records:
  - accepted registration state
- `data/experiments/<run>/`:
  - canonical experiment outputs

Recommendation:

- Introduce one richer servo calibration state file rather than scattering neutral, safety, and pretension artifacts across multiple ad hoc files

## 5. Recommended Phased Plan

The best roadmap is driven by blockers, not by UI completeness. The current repo does not need another broad framework rewrite. It needs disciplined closure of the servo hardware and safety path, while preserving the stronger parts that already exist.

### Phase 0: Audit Freeze and Planning

Goal:

- Freeze the architectural direction and document the real current state

Why now:

- The repo has accumulated several large refactors and feature passes
- The next work should be driven by the actual blocker rather than by another broad redesign

Dependencies:

- None

Deliverables:

- this `plan.md`
- explicit subsystem status and risk framing
- ordered implementation phases

Acceptance criteria:

- plan is grounded in the current repo
- next coding prompt is narrow and high leverage

Hardware-dependent:

- No

Likely files:

- `plan.md`

### Phase 1: Real OpenRB150 / DYNAMIXEL Hardware Foundation

Goal:

- Make the repo capable of talking to real OpenRB/XC330/XC333 hardware safely enough to support basic bring-up

Why it comes now:

- This is the main blocker to the project target
- Tracking, registration, experiments, and GUI are already ahead of this subsystem

Dependencies:

- OpenRB hardware available
- chosen DYNAMIXEL SDK / serial path confirmed
- concrete servo control table addresses confirmed for XC330/XC333 mode in use

Exact deliverables:

- implement real `OpenRbClient.connect()` and `prepare_for_dynamixel_use()`
- implement real `DxlBus.connect()`, `disconnect()`, `scan_ids()`, `read_telemetry()`, `write_goal_positions()`, `write_servo_id()`
- verify readback of at least:
  - Present Position
  - Present Current
  - Present Input Voltage
- verify goal-position writes on real hardware
- wire real hardware status into System and Servo tabs without mock-only wording
- document real serial port / permission assumptions in README

Acceptance criteria:

- on the Pi, the app can connect to OpenRB
- scan and list real servo IDs
- assign a servo ID on a test servo
- read telemetry reliably from connected servos
- jog a single servo in a controlled way on real hardware
- all of the above work from the canonical service/controller path, not a one-off script

Hardware-dependent:

- Yes

Likely files/modules:

- `continuum_robot/hardware/openrb_client.py`
- `continuum_robot/hardware/dxl_bus.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/gui/controllers/system_controller.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `README.md`
- `tests/` for hardware-seam unit coverage and mock-preservation tests

### Phase 2: Servo Safety, Tendon Mapping, and Calibration State Closure

Goal:

- Make live actuation semantics defensible for tendon hardware

Why it comes now:

- Basic transport alone is not safe enough for tendon-driven experiments
- This phase closes the biggest correctness gap after transport exists

Dependencies:

- Phase 1

Exact deliverables:

- make safety fail closed when current telemetry is missing
- enforce neutral-relative bounds for both jog and displacement paths before writes
- wire `tendon_to_servo` mapping into the live actuation path
- expand neutral calibration persistence to include:
  - neutral setpoint
  - calibrated safe min/max
  - calibration metadata
- require calibration compatibility with current robot config / servo IDs
- expose richer calibration state in GUI and preflight surfaces

Acceptance criteria:

- live commands are blocked when telemetry needed for safety is missing
- jog and displacement commands respect calibrated bounds
- 4-servo and 8-servo mappings behave deterministically in tests
- calibration artifact round-trips and is used by all future commands

Hardware-dependent:

- Partly
- artifact design and tests can be done now
- final acceptance of live bounds behavior requires hardware

Likely files/modules:

- `continuum_robot/servos/safety_guard.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/servos/tendon_mapping.py`
- `continuum_robot/servos/neutral_calibration_service.py`
- `continuum_robot/config/schemas.py`
- `continuum_robot/gui/experiment_preflight.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `config/safety.yaml`
- `tests/test_servos*`

### Phase 3: Guided Pretension / Centering Workflow

Goal:

- Implement the actual repeatable tendon pretension and centered-start workflow

Why it comes now:

- Repeatability depends on reproducible starting tension
- This is the main missing calibration behavior described in the project requirements

Dependencies:

- Phase 1 for telemetry and writes
- Phase 2 for calibrated bounds and calibration artifact

Exact deliverables:

- implement a guided pretension routine that steps each tendon until current/torque target criteria are met
- persist pretension acceptance status and relevant metrics
- integrate pretension state into experiment preflight and run gating
- provide GUI workflow for:
  - start pretension
  - show currents live
  - accept/retry/save state

Acceptance criteria:

- routine can tension all configured tendons to within threshold
- current spread and absolute target criteria are both visible
- pretension state persists and is required before live repeatability runs

Hardware-dependent:

- Yes

Likely files/modules:

- new `continuum_robot/servos/pretension_service.py`
- `continuum_robot/servos/pretension_validation_service.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `continuum_robot/gui/tabs/servos_tab.py`
- `continuum_robot/gui/experiment_preflight.py`
- `tests/test_pretension*`

### Phase 4: Tracking and Registration Bench Closure

Goal:

- Turn tracking + registration from architecturally strong into accepted bench workflows

Why it comes now:

- The software path is already in good shape
- What remains is live correctness and workflow acceptance

Dependencies:

- Aurora hardware available
- pen probe available
- tip calibration workflow available

Exact deliverables:

- bench-validate Python NDI backend on the Pi
- confirm raw tool IDs and alias mapping
- produce accepted pivot calibration output for tool `0B`
- bench-validate 4-point registration workflow and RMSE acceptance criteria
- confirm that saved registration produces stable `T_robot_tip`
- tighten README bring-up order around actual lab workflow:
  - pivot calibration
  - tracking validation
  - 4-point registration

Acceptance criteria:

- tracker doctor / smoke / benchmark pass on live hardware
- pivot calibration produces acceptable RMSE and usable tip file
- registration produces accepted FRE/RMSE below configured threshold
- live tip pose in robot frame updates reliably after registration

Hardware-dependent:

- Yes

Likely files/modules:

- `continuum_robot/services/tracking_service.py`
- `continuum_robot/tracking/backend_router.py`
- `continuum_robot/tracking/diagnostics.py`
- `continuum_robot/services/registration_service.py`
- `continuum_robot/registration/repository.py`
- `continuum_robot/gui/controllers/tracking_controller.py`
- `continuum_robot/gui/controllers/registration_controller.py`
- `continuum_robot/gui/tabs/registration_tab.py`
- `README.md`
- tracking/registration tests and possibly fixture updates

### Phase 5: Live Repeatability Dataset Execution

Goal:

- Make `repeatability_dataset` the real scientific data-collection path for the robot

Why it comes now:

- It depends on real servo actuation, safe startup state, trusted tracking, and accepted registration

Dependencies:

- Phases 1 through 4

Exact deliverables:

- ensure experiments use the live servo service and canonical tracking snapshot path
- confirm servo command data and telemetry are logged alongside pose data
- require calibration / pretension / registration prerequisites appropriately
- return the robot to neutral safely on success and stop/failure
- confirm repeatability analysis reflects real live data

Acceptance criteria:

- live repeatability run can execute on the robot
- dataset contains command, telemetry, tracking, and pose outputs needed for later analysis
- run summaries include repeatability metrics and acceptance thresholds
- canceled or partial runs still save usable metadata and summary state

Hardware-dependent:

- Yes

Likely files/modules:

- `continuum_robot/experiments/critical_experiments.py`
- `continuum_robot/experiments/sample_builders.py`
- `continuum_robot/experiments/metrics.py`
- `continuum_robot/experiments/validation.py`
- `continuum_robot/gui/controllers/experiment_controller.py`
- `continuum_robot/gui/experiment_preflight.py`
- `tests/test_critical_experiments.py`

### Phase 6: Pi Operator Readiness and Lab Hardening

Goal:

- Turn the repo from a strong development system into a repeatable lab operator package

Why it comes last:

- It should stabilize around already-accepted real hardware workflows

Dependencies:

- Phases 1 through 5 largely complete

Exact deliverables:

- finalize machine-local config guidance
- add Pi bring-up checklist
- document serial permissions / USB device assumptions
- document exact run order for the lab
- optionally add lightweight launch / service conveniences if needed
- remove or clearly de-emphasize dead compatibility paths that remain unused

Acceptance criteria:

- a new operator can bring up the Pi from docs and GUI without guesswork
- operator state before experiments is explicit and trustworthy
- no major lab workflow still depends on hidden knowledge in legacy scripts

Hardware-dependent:

- Partly

Likely files/modules:

- `README.md`
- `config/system.local.example.yaml`
- `scripts/bootstrap.sh`
- `scripts/run_gui.sh`
- potentially `docs/`

## 6. Acceptance Tests and Validation by Subsystem

### 6.1 Servo Safety and Bounds

Should validate:

- unsafe commands are blocked before writes
- missing current telemetry fails closed when safety requires it
- neutral-relative min/max bounds are enforced
- tendon-order to servo-order mapping is deterministic
- 4-servo and 8-servo configs both work

Test types:

- unit tests for mapping and guard logic
- mock bus tests for service behavior
- live hardware smoke tests for single-servo jog and telemetry reads

### 6.2 Pretensioning

Should validate:

- current/torque target routine converges or fails clearly
- accepted pretension state persists
- experiment preflight blocks live runs if pretension is missing/invalid

Test types:

- synthetic unit tests for algorithm behavior
- mock service tests for state persistence and GUI gating
- live bench tests for convergence and repeatability

### 6.3 Tracking Diagnostics

Should validate:

- backend selection and fallback
- import failure and serial-port failure classification
- stale-frame and no-frame detection
- expected-tool visibility checks
- distinction between tracker healthy and full pose pipeline healthy

Test types:

- unit tests for router/diagnostics
- CLI smoke tests in mock mode
- live Pi bring-up with doctor / benchmark / smoke

### 6.4 Pivot Calibration

Should validate:

- least-squares solve correctness on synthetic transforms
- outlier rejection behavior
- offline replay path
- output tip file format
- live sample count and RMSE acceptance on real tool `0B`

### 6.5 Registration RMSE / Validity

Should validate:

- 4-point repeated-capture workflow
- solve/review/save/overwrite flow
- accepted registration persistence
- FRE/RMSE threshold evaluation
- resulting `T_robot_tip` availability after save

### 6.6 Experiment Logging

Should validate:

- canonical metadata/samples/summary/config snapshot
- sample count consistency
- partial/canceled runs remain reloadable
- servo + tracking provenance are recorded in live runs

### 6.7 Repeatability Metrics

Should validate:

- per-target centroid and spread
- overall RMS repeatability
- approach-conditioned spread
- thresholded pass/warn/fail reporting
- same metrics reload correctly from saved datasets

### 6.8 GUI / Operator Workflow

Should validate:

- preflight blocks when required prerequisites are missing
- registration save requires overwrite confirmation
- fallback visualization remains usable
- experiment workspace still works in mock/offscreen mode
- run-history reload remains functional without hardware attached

## 7. Likely Future Touch Points by Phase

This section is intended as a scope guide before future coding passes.

### Phase 1 likely touch points

- `continuum_robot/hardware/`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/gui/controllers/system_controller.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `README.md`
- servo hardware tests

### Phase 2 likely touch points

- `continuum_robot/servos/safety_guard.py`
- `continuum_robot/servos/tendon_mapping.py`
- `continuum_robot/servos/neutral_calibration_service.py`
- `continuum_robot/config/schemas.py`
- `continuum_robot/gui/experiment_preflight.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- calibration tests

### Phase 3 likely touch points

- `continuum_robot/servos/`
- servo GUI tab/controller
- experiment preflight
- calibration persistence files and tests

### Phase 4 likely touch points

- `continuum_robot/tracking/`
- `continuum_robot/services/tracking_service.py`
- `continuum_robot/services/registration_service.py`
- `continuum_robot/registration/repository.py`
- tracking/registration GUI tabs/controllers
- README and diagnostics scripts

### Phase 5 likely touch points

- `continuum_robot/experiments/`
- experiment GUI controller/preflight
- metrics / validation helpers
- experiment tests

### Phase 6 likely touch points

- `README.md`
- `docs/`
- `config/system.local.example.yaml`
- launch/bootstrap scripts

## 8. Recommended Immediate Next Phase

The single best next implementation phase is:

## Phase 1: Real OpenRB150 / DYNAMIXEL Hardware Foundation

Reason:

- It is the main blocker to the actual project target
- Tracking, registration, experiments, and GUI are already in materially better shape
- More UI work before this would improve appearance, not project readiness
- Pretension, neutral calibration hardening, and live repeatability all depend on a real servo transport and telemetry path first

## 9. Recommended Next Coding Prompt

Use this as the next implementation prompt:

```text
Implement Phase 1 from plan.md: the real OpenRB150 / DYNAMIXEL hardware foundation.

Scope:
1. Implement the real hardware path in `continuum_robot/hardware/dxl_bus.py` for:
   - connect/disconnect
   - scan_ids
   - read_telemetry
   - write_goal_positions
   - write_servo_id
2. Implement the real hardware path in `continuum_robot/hardware/openrb_client.py` for:
   - connect/disconnect
   - prepare_for_dynamixel_use
   - status reporting needed by the GUI
3. Keep mock-mode behavior intact.
4. Do not change `references/` or `tools/`.
5. Wire the new hardware behavior through the existing canonical services/controllers rather than adding one-off scripts.
6. Add/update tests that cover:
   - mock-mode preservation
   - hardware-seam behavior with fake/stubbed SDK calls
   - system/servo controller status updates
7. Update README with the real hardware bring-up steps, assumptions, and any required config fields.

Constraints:
- Do not do a broad GUI redesign.
- Do not implement pretension or richer calibration persistence in this pass.
- Keep the canonical experiment runner and tracking/registration paths unchanged.

Acceptance target:
- On a real Pi with OpenRB connected, the app can connect, scan servo IDs, read telemetry, assign a servo ID, and issue a controlled jog through the existing GUI/service path.
```

## 10. Manual Assumptions to Verify Before Coding Phase 1

These should be confirmed before or during the next implementation pass:

- exact XC330/XC333 model variant and control table addresses in the mode being used
- whether the real project uses current-based control mode, position mode with current monitoring, or another DYNAMIXEL operating mode
- expected OpenRB serial behavior and whether any board-specific initialization is required before DYNAMIXEL access
- whether servo ID assignment and firmware workflows should be exposed directly in the GUI or kept as maintenance-only utilities
- actual spool diameter and sign conventions for tendon displacement on the current hardware build
- whether the lab will use 4-servo and 8-servo configurations interchangeably or treat one as primary first
