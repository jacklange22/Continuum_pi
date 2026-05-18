# System Specification

## Scope

This repo is the Raspberry Pi operator stack for the tendon-driven continuum robot. The first validated hardware target is the 4-servo robot using Robotis `XC330-M288-T` servos on an OpenRB-150, Aurora tracking, GUI-guided registration, and repeatability-focused experiments.

Repeatability target: `< 1.0 mm` tip-position RMS in robot frame for the main repeatability workflow.

## Subsystems

### Servo Subsystem

- `SRV-001` The system shall validate the 4-servo robot first, with 8-servo support retained as a later-compatible configuration.
- `SRV-001a` One-servo bring-up shall be a first-class workflow before the full 4-servo validation phase.
- `SRV-002` The primary live servo target shall be `XC330-M288-T` on OpenRB-150.
- `SRV-003` Experimental motion shall use position-based commands as the primary actuation path.
- `SRV-003a` External power shall be the expected motion-testing path for OpenRB / DYNAMIXEL movement validation.
- `SRV-004` The software shall expose present position, present current, input voltage, present temperature, operating mode, and hardware-error status through the canonical servo service.
- `SRV-005` The GUI shall support connect, scan, ID assignment on a test servo, telemetry refresh, and conservative fine/coarse jog commands through the same service path used by the runtime.
- `SRV-006` Neutral calibration shall be GUI-driven and shall capture per-servo neutral setpoints.
- `SRV-006a` The canonical calibration artifact shall store neutral setpoint, safe min/max bounds, pretension/current threshold, calibration timestamp, validity, and robot compatibility metadata in one machine-readable file.
- `SRV-007` Pretension shall be GUI-driven and shall use operator-tunable current thresholds for tensioning and safety checks.
- `SRV-008` Current feedback is primarily for pretensioning and safety enforcement, not as the primary experiment command mode.
- `SRV-009` Motion shall be blocked when telemetry is missing or stale, operating mode is incompatible, hardware error is nonzero, or safe bounds are unavailable for the requested action.

Current repo state:
- Real OpenRB / DYNAMIXEL transport exists.
- Position writes, richer telemetry reads, and torque-disabled ID assignment exist.
- A richer single-file servo calibration artifact exists and is surfaced in the GUI.
- One-click whole-segment pretension lives on the Servos tab; an algorithm-vs-manual comparison report is generated alongside each pretension run.
- A safety guard layer (`SafetyGuard`, `SegmentReadiness`, `MotorControlSupervisor`) enforces telemetry freshness, operating mode, current/voltage/temperature, hardware-error, and bounds checks on coordinated motion. Workspace-boundary command rejections are now skipped-and-counted rather than terminating runs.
- Full multi-servo closed-loop control beyond position commands is still ahead.

### Tracking Subsystem

- `TRK-001` The canonical live tracking path shall be the Python NDI backend.
- `TRK-002` The system shall expose normalized Aurora tool transforms, tool visibility, freshness, and diagnostics through `TrackingService`.
- `TRK-003` The GUI shall surface tracker readiness and tool visibility before registration or experiments.
- `TRK-004` Live tip pose shall be available as `T_robot_tip` once registration exists.
- `TRK-005` Mock tracking shall remain available for offline development and experiment dry runs.

Current repo state:
- Canonical tracking service, diagnostics, benchmark, and doctor flows exist.
- Live bench validation is still hardware-dependent.

### Registration Subsystem

- `REG-001` Registration shall use a 4-point robot-body alignment workflow through the GUI.
- `REG-002` The operator shall choose the best 4 landmarks from a larger configured candidate set.
- `REG-003` Candidate landmarks shall include ID, coordinates, optional display label, and optional enabled flag.
- `REG-004` The operator shall be able to choose landmarks from a simple 2D top-view map and a list/table view.
- `REG-005` Registration shall support repeated measurements per point.
- `REG-006` The solve path shall remain the canonical rigid-registration math already used by the repo.
- `REG-007` The GUI shall block solve until 4 unique points are selected and each has the required samples.
- `REG-008` Saved registration output shall include FRE / RMSE and remain reloadable by the GUI and tracking pipeline.

Current repo state:
- 4-point GUI workflow exists.
- Arbitrary 4-of-N landmark selection now exists.
- Candidate-landmark config and top-view selection now exist.

### GUI / Operator Workflow

- `GUI-001` The Pi GUI shall remain the canonical operator application.
- `GUI-002` The GUI shall expose system health, tracking readiness, registration workflow, servo bring-up, and canonical experiments without side scripts.
- `GUI-003` The GUI shall stay readable on Pi-class displays and external monitors without dense or clipped forms.
- `GUI-004` Preflight and status messaging shall stay concise, actionable, and validation-focused.
- `GUI-005` Advanced visualization must not be allowed to crash the operator workflow; safe fallback visualization must remain available.

Current repo state:
- Canonical experiment workspace exists.
- Registration and experiment tabs are present and usable.
- Stability protections for visualization are in place.
- System and Servo tabs now support one-servo bring-up parameters, startup calibration visibility, and cautious pretension control.

### Experiment Subsystem

- `EXP-001` The canonical experiment runner shall remain the only execution path for experiments.
- `EXP-002` The critical experiments are `pivot_calibration`, `pivot_validation`, `aurora_grid_accuracy`, `registration_validation`, `registration_trial`, `pretension_validation`, `tracker_timing_validation`, `servo_tracker_sync_validation`, and `single_segment_repeatability`.
- `EXP-003` The main scientific outcome is the single-segment repeatability experiment, which must log commanded motion plus measured pose under the legacy 17-target revisit protocol.
- `EXP-004` Experiments shall support dry-run or offline execution where logically possible.
- `EXP-005` Repeatability datasets shall be analyzable against the `< 1 mm` target. The Modeling tab's headline RMSE chart now visualizes this target alongside a Wolfe baseline reference line.
- `EXP-006` Thesis-facing experiments shall emit a **two-figure thesis contract** (one headline + one supporting view), separate from dashboard/debug figures. Already applied to `pivot_validation`, `registration_validation`, `tracker_timing_validation`, and `servo_tracker_sync_validation`.

Current repo state:
- Canonical framework and the listed critical experiments exist.
- **`single_segment_repeatability` has zero live bench runs as of 2026-05-17 — it is the gating experiment for the acceptance target below.**
- Two-segment counterparts (`two_segment_startup_validation`, `two_segment_collect_pose_command_dataset`, `two_segment_repeatability`) exist as foundation; full two-segment control is still ahead.

### Data / Logging

- `DAT-001` Runtime outputs shall live under `data/`.
- `DAT-002` Registration outputs shall live under `data/registrations/`.
- `DAT-003` Experiment outputs shall include metadata, samples, summary, and a config snapshot.
- `DAT-004` Persistent calibration paths shall remain explicit in config.
- `DAT-005` The servo subsystem shall use one canonical persisted calibration file, even while maintaining backward compatibility with the older neutral-setpoint format.

## Initial Acceptance Target

The initial integrated acceptance target for this repo is:

1. Bring up the 4-servo OpenRB / `XC330-M288-T` system — **done** (one-servo validation, multi-servo coordinated motion, safety guard).
   - first through one-servo validation with external power and conservative jog / pretension
2. Validate tracker health and tool visibility — **done** (dry/diagnostic); bench validation re-confirmed per session.
3. Generate or load a valid pen-probe tip file — **done** (artifact present, lower-trust until validation re-run).
4. Perform 4-point body registration from a larger candidate landmark set — **done** (artifact present; `registration_trial` available for sweep-replay comparison).
5. Run the repeatability dataset experiment and compute robot-frame repeatability against the `< 1 mm` target — **not yet met**. `data/experiments/single_segment_repeatability/` is empty; this is the top open item on this spec.
