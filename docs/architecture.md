# Architecture

## Canonical Runtime Paths

This repo already has the correct high-level split: services own subsystem state and I/O, controllers adapt those services to the GUI, and the experiment runner is the only canonical experiment execution path.

## Servo Subsystem

Primary files:

- [bootstrap.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/bootstrap.py)
- [openrb_client.py](/Users/jacklange/Continuum/pi_code/continuum_robot/hardware/openrb_client.py)
- [dxl_bus.py](/Users/jacklange/Continuum/pi_code/continuum_robot/hardware/dxl_bus.py)
- [servo_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/servo_service.py)
- [system_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/system_controller.py)
- [servos_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/servos_controller.py)

Responsibilities:

- `OpenRbClient`: board/port readiness, connection state, OpenRB preparation.
- `DxlBus`: low-level DYNAMIXEL SDK transport, scan, telemetry, goal writes, ID writes.
- `ServoService`: canonical high-level servo API for GUI and future experiment use.
- GUI controllers: user-facing actions and status.

Current architectural rule:

- The GUI must not bypass `ServoService` for motion or telemetry.

## Tracking Subsystem

Primary files:

- [backend_router.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/backend_router.py)
- [ndi_backend.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py)
- [tracking_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/tracking_service.py)
- [diagnostics.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/diagnostics.py)

Responsibilities:

- backend selection and fallback
- normalized tool state
- tracker health, freshness, and diagnostics
- robot-tip pose chain after registration

Current architectural rule:

- all GUI, registration, diagnostics, and experiments consume `TrackingService`, not backend-specific objects

## Registration Subsystem

Primary files:

- [registration_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/registration_service.py)
- [registration_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/registration_controller.py)
- [registration_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/registration_tab.py)
- [registration_landmark_map_widget.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/widgets/registration_landmark_map_widget.py)
- [rigid_solver.py](/Users/jacklange/Continuum/pi_code/continuum_robot/registration/rigid_solver.py)
- [repository.py](/Users/jacklange/Continuum/pi_code/continuum_robot/registration/repository.py)

Responsibilities:

- `RegistrationService`: canonical session state, sample capture, solve, acceptance, persistence
- controller/tab: operator workflow, 4-of-N landmark selection, solve gating, overwrite confirmation
- repository: persistent accepted registration artifacts

Current architectural rule:

- landmark selection may change the 4-point subset, but the math path remains the canonical rigid-registration solve already used by the service

## Experiment Subsystem

Primary files:

- [experiment_runner.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/experiment_runner.py)
- [framework.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/framework.py)
- [critical_experiments.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/critical_experiments.py)
- [experiment_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/experiment_controller.py)
- [experiment_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/experiment_tab.py)

Responsibilities:

- canonical experiment execution
- dataset writing and summaries
- live visualization and result reloads
- preflight gating before operator runs

Current architectural rule:

- no GUI-only experiment execution path

## Configuration And Persistent State

Runtime config:

- `config/system.yaml`
- `config/system.local.yaml`
- `config/robot_4servo.yaml`
- `config/safety.yaml`
- `config/registration.yaml`
- `config/experiment.yaml`

Persistent runtime artifacts:

- `config/` for durable servo startup calibration state
- `data/pivot_calibration/` for accepted and staged 0B tip-calibration artifacts plus pivot review data
- `data/registrations/` for durable accepted/latest registration artifacts
- `data/experiments/tracker_validation/` for saved tracker validation reports
- `data/experiments/` for canonical experiment datasets
- `data/experiments/<experiment_name>/` for canonical run bundles grouped by experiment type
- `data/pivot_calibration/captures/` for raw pivot-review capture CSVs

Current architectural rule:

- configuration defines defaults and machine-local startup state; accepted calibrations and experiment outputs live under their canonical `data/` categories
- repo-root `runs/` is retired and should not be used as an active runtime artifact location

## Recommended Next Architecture Boundary Work

The next architecture work should stay narrow:

1. keep `ServoService` as the only servo command/telemetry seam
2. add GUI-driven neutral calibration and pretension on top of that service
3. keep registration selection/UI logic in the controller/tab layer, not in the rigid solver
4. keep experiment requirements and logging inside the canonical experiment runner
