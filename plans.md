# Raspberry Pi Continuum Robot Platform Plan

Last updated: 2026-03-27

## Goal

Build a Raspberry Pi hosted operator stack that:

- reads Aurora tracker data directly on the Pi
- commands 4 or 8 Robotis XC330-M288 servos through an OpenRB-150
- calibrates tendon neutral points and safe operating ranges
- computes robot tip pose in the robot body frame reliably
- supports registration, visualization, manual control, and experiments through one GUI
- logs synchronized tracking and servo data for repeatability and other studies

This document is a living status report plus execution plan. It replaces the earlier milestone list with the current verified state of the repo.

## Current Verified State

The software platform is structurally real and usable in mock mode.

- The PySide6 operator GUI exists and boots with tabs for system, servos, tracking, registration, and experiments.
- The application has a coherent bootstrap path in `continuum_robot/app/bootstrap.py`.
- Tracking has two runtime backends:
  - `ndi` backend in Python using `scikit-surgerynditracker`
  - `bridge` backend using the C++ Aurora socket bridge
- `config/system.yaml` currently defaults to `tracker_backend: "ndi"` and `mock_mode: true`.
- A `TrackingService` layer now handles pose snapshots, stale-data checks, registration transform application, and tool-role lookup.
- Registration has a modern service path and a legacy-compatible solver that uses the protected SolidWorks and pen-probe assets.
- Experiment execution and `.dat` logging exist and work in software.
- The automated test suite currently passes: `59 passed`.

## What Is Working Well

### 1. GUI and Application Structure

The repo is no longer just scaffolding.

- The GUI is wired through controllers and services rather than ad hoc scripts.
- The system tab can manage tracker and OpenRB connection state.
- The servo tab can capture neutral positions, jog motors, send displacement commands, and show telemetry in mock mode.
- The tracking tab can display tool state, role assignment, stale-data status, and derived tip pose.
- The registration tab can collect landmarks and solve a transform.
- The experiment tab can load a CSV experiment file, validate basic prerequisites, and log results.

### 2. Tracking Stack

Tracking is in better shape than the original plan assumed.

- The default live-hardware path is now the Python NDI backend, not the C++ bridge.
- The C++ bridge still exists as a fallback backend and remains useful on the Pi if the Python backend proves fragile.
- Tool aliasing is supported so hardware handles can map into logical roles like `0A` and `0B`.
- `TrackingService` centralizes snapshot generation and tip-pose computation, which is the right architecture for the GUI and for safety checks.

### 3. Registration Math

Registration is the strongest part of the repo relative to the original reference scripts.

- The code clearly carries forward the rigid registration workflow from `references/rigid_registration.py`.
- The legacy-compatible path uses protected model and tip landmark assets, pen-probe geometry, repeated point capture, and validation metrics.
- The resulting transform chain is good enough to support derived robot-tip pose in the robot frame.

### 4. Mock-Mode Integration

Mock mode is strong enough to support software development without hardware connected.

- Mock DYNAMIXEL and OpenRB clients exist.
- The GUI, tracking service, registration flow, and experiment logging can all be exercised in software.
- Test coverage is respectable for math, parsing, persistence, and service behavior.

## Critical Gaps and Shortcomings

These are the main reasons the project is not yet lab-ready for the real robot.

### 1. Real servo hardware transport is still missing

This is the biggest blocker.

- `continuum_robot/hardware/dxl_bus.py` still raises `NotImplementedError` for real bus operations.
- `continuum_robot/hardware/openrb_client.py` still raises `NotImplementedError` for real OpenRB connection and preparation.
- The GUI can expose servo and OpenRB actions, but in real-hardware mode the actual low-level transport is not implemented.

Impact:

- No verified control path from the Pi to the XC330-M288 servos exists in this repo yet.
- Closed-loop behavior cannot be validated because the basic hardware command path is absent.

### 2. Servo safety is not fail-closed

The current safety behavior is too permissive for tendon protection.

- `SafetyGuard.validate_currents()` ignores `None` current readings instead of treating missing telemetry as unsafe.
- `ServoService.command_displacement()` validates current after goal positions are written, not before or during motion.
- `ServoService.jog_servo()` does not enforce neutral-relative safe position limits before issuing a jog command.

Impact:

- A lost or missing current reading can silently bypass the intended protection logic.
- Over-tension conditions are only detected after a command is sent.
- Manual jogging can move outside a calibrated safe window once hardware exists.

### 3. The pretension workflow is only a validator, not an algorithm

The repo does not yet implement the pretension behavior described in the spec.

- The current pretension service only checks whether measured currents are sufficiently balanced.
- There is no closed-loop routine that steps each servo until a target torque/current threshold is reached.
- There is no robust centering workflow that guarantees a repeatable equalized starting condition on every run.

Impact:

- The code cannot yet establish a reliable neutral tendon state automatically.
- The operator still lacks the key tendon-protection and repeatability workflow that motivated current feedback in the first place.

### 4. Neutral calibration is incomplete

Neutral capture exists, but the persistence model is too thin.

- The current neutral calibration service stores only neutral setpoint ticks.
- It does not save per-servo safe minimum and maximum limits around that neutral point.
- It does not persist tension thresholds, pretension acceptance data, or other safety-relevant calibration metadata.

Impact:

- Future commands cannot be validated against calibrated servo-specific bounds.
- The current calibration artifact is not rich enough to support safe repeatable runs.

### 5. Tendon-to-servo mapping is not wired into the live control path

There is a helper for tendon mapping, but it is not part of the command flow.

- The config schema supports `tendon_to_servo`.
- `continuum_robot/servos/tendon_mapping.py` exists.
- The active servo command path still assumes incoming displacement vectors already match servo order.

Impact:

- The code is not yet robust to real 4-tendon versus 8-tendon layouts.
- Misordered commands remain an integration risk when hardware is brought online.

### 6. Registration workflow does not match the default operator spec

The current default registration workflow is more complex than the requested `0B` four-point flow.

- The protected-asset path expects model and tip landmark sets derived from the legacy workflow.
- In practice that means the default flow is based on the larger SolidWorks asset set, not a simple four-point body-frame registration.
- A simpler path exists in code, but it is not the default operator experience.

Impact:

- The GUI does not yet align cleanly with the "collect 4 points with pen `0B`" procedure described in the requirements.
- The current default is powerful but not operator-simple.

### 7. Registration acceptance is too automatic

The current GUI accepts a solved registration immediately.

- The controller calls solve and accept back-to-back.
- There is no explicit operator review step after metrics are shown.

Impact:

- A bad registration can be accepted without an intentional review action.
- The workflow does not yet support "solve, inspect, retry, then accept" behavior cleanly.

### 8. Experiment execution bypasses the richer tracking service

The experiment path still talks to the raw tracker manager.

- `ExperimentController` and `ExperimentRunner` depend on the lower-level tracker interface.
- They do not consume the richer `TrackingService` snapshot model used by the rest of the GUI.

Impact:

- Experiments do not benefit from the same stale-data handling and consistency checks used elsewhere.
- There is duplicated tracking-state logic in the codebase.

### 9. Experiment safety and traceability are incomplete

The experiment system exists, but it is not yet strong enough for reliable lab studies.

- The prerequisite check does not require a successful pretension/calibration state.
- The run logic does not return the robot to neutral on completion, abort, or failure.
- The `.dat` writer does not record registration identifiers, calibration identifiers, or other provenance metadata.
- `sample_count_per_point` exists in config but is not used in the runtime execution path.

Impact:

- Runs are less repeatable than the legacy references intended.
- Output files are harder to audit later.
- Experiment behavior is not yet aligned with the expected repeatability workflow.

### 10. Visualization is still minimal

The GUI currently provides lightweight plotting rather than a full robot/model visualization.

- The tracking and registration widgets are simple 2D plotting surfaces.
- There is no true 3D rendering of the robot, tools, and registered body geometry together.

Impact:

- Operator validation is weaker than the target user experience.
- The GUI does not yet help the user confirm geometry and transforms visually at the level described in the spec.

### 11. OpenRB firmware and field-service tooling are absent

The spec calls for setup and troubleshooting support that the repo does not yet provide.

- No firmware upload or OpenRB bootstrapping workflow is implemented.
- No servo ID assignment or scan-and-program workflow is fully implemented against real hardware.

Impact:

- The GUI is not yet a complete bring-up tool for a new hardware setup.

## Bugs to Treat as Immediate Engineering Issues

These should be fixed before spending much more time on UI polish.

1. Missing current telemetry is treated as acceptable in servo safety checks.
2. Manual jog commands are not guarded by neutral-relative safe bounds.
3. Pretension state is not enforced before experiment execution.
4. Experiment execution bypasses `TrackingService`, which creates inconsistent tracking-health behavior across the app.
5. Registration solve is auto-accepted instead of requiring explicit operator confirmation.
6. The simple registration path depends on an existing tip calibration transform rather than solving the full operator workflow cleanly.

## Status Relative to the Original Spec

### Servo requirements

Status: partially scaffolded, not complete

- Neutral point capture exists.
- Displacement-to-position mapping exists.
- Basic current and voltage telemetry surfaces exist.
- Real transport, active closed-loop protection, pretension control, and calibrated safe envelopes are still missing.

### Tracking requirements

Status: mostly implemented in software, still needs bench validation

- Raw Aurora data ingestion path exists on the Pi.
- Tool poses, frame transforms, and tip-pose derivation exist.
- The service architecture is strong.
- Real-hardware validation and operator-proofing remain to be done.

### Registration requirements

Status: mathematically strong, operator workflow still misaligned

- Legacy-compatible registration math is implemented.
- Persistence and transform application exist.
- The default workflow is more complex than the requested four-point registration procedure.
- Acceptance UX should be improved.

### GUI requirements

Status: broad coverage present, depth uneven

- Core tabs and workflows exist.
- Mock-mode operator experience is real.
- Visualization and hardware bring-up tooling are still too thin.

### Experiment requirements

Status: functional first version, not yet robust enough

- CSV-driven execution and logging exist.
- Logging includes commanded displacement, servo telemetry, tool positions, and derived tip pose.
- Neutral return, stronger gating, and run metadata are still missing.

## Execution Priorities

The next work should focus on making the stack safe and coherent before adding more UI breadth.

### Priority 1: Fix servo calibration and safety semantics

Implement the software behaviors that must exist before real hardware is trusted.

- persist per-servo neutral, min, max, and tension limits
- make missing current telemetry fail closed
- enforce safe position bounds for jog and displacement commands
- wire `tendon_to_servo` into the command path
- add an explicit pretension state model that later experiments can require

### Priority 2: Implement the real OpenRB and DYNAMIXEL path

Bring the servo stack out of mock mode.

- implement OpenRB connection and preparation flow
- implement DYNAMIXEL bus connect, scan, telemetry read, and goal write operations
- support servo discovery and ID assignment
- verify current and voltage reads against the XC330-M288 hardware

### Priority 3: Align registration with the operator workflow

Make registration match how the system will actually be used in the lab.

- support a first-class four-point `0B` registration flow in the GUI
- keep the protected-asset legacy workflow as an advanced or alternate mode
- separate solve from accept in the UI
- make registration metrics visible before acceptance

### Priority 4: Make experiments repeatable and auditable

Use the more robust service layer and improve output quality.

- route experiment execution through `TrackingService`
- require valid registration, neutral calibration, and pretension state
- return to neutral on stop, failure, and normal completion
- record calibration and registration IDs in experiment outputs
- either implement or remove `sample_count_per_point`

### Priority 5: Improve operator validation and docs

Finish the usability layer after the core control semantics are correct.

- add stronger registration and tracking status indicators
- improve visualization of robot, tools, and body frame
- document the actual hardware bring-up path on the Pi

## Immediate Next Slice

This is the best next implementation chunk to execute now:

1. Expand neutral calibration persistence to include safe min/max limits and safety metadata.
2. Make servo safety fail closed on missing current data and enforce safe bounds on jog commands.
3. Wire `tendon_to_servo` into the live displacement command path.
4. Add a persisted pretension status artifact and require it before experiments.
5. Update experiments to use `TrackingService`, return to neutral at the end, and write run metadata.

This slice does not depend on real hardware transport being complete, so it can be implemented and tested immediately in mock mode.

## Validation Checklist For The Next Slice

- unit tests for neutral calibration persistence schema changes
- unit tests proving jog and displacement commands reject out-of-bounds requests
- unit tests proving missing current telemetry raises a safety violation
- unit tests for tendon-to-servo mapping in the live command path
- unit tests proving experiments refuse to start without pretension state
- integration test proving experiments attempt neutral return on completion and stop

## Completion Definition

The project should be considered ready for first serious hardware trials only when:

- real OpenRB and DYNAMIXEL communication is implemented
- neutral and pretension calibration are persisted and enforced
- current-based safety is fail-closed
- registration can be solved, reviewed, and accepted intentionally
- experiment runs produce traceable outputs and leave the robot in a known state
- the full stack runs on the Raspberry Pi with Aurora on one USB port and OpenRB on another
