# Architecture

## Canonical Runtime Paths

This repo already has the correct high-level split: services own subsystem state and I/O, controllers adapt those services to the GUI, the experiment runner is the only canonical experiment execution path, and the modeling stack is a separate offline analysis layer that consumes saved experiment bundles.

## Application Bootstrap

Primary files:

- [main.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/main.py)
- [bootstrap.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/bootstrap.py)
- [service_registry.py](/Users/jacklange/Continuum/pi_code/continuum_robot/app/service_registry.py)

Responsibilities:

- `main.py`: CLI entry point that wires settings, services, and the GUI.
- `bootstrap.py`: constructs the service graph from settings.
- `service_registry.py`: canonical service-wiring seam used by GUI controllers and CLI tools.

## Servo Subsystem

Primary files:

- [openrb_client.py](/Users/jacklange/Continuum/pi_code/continuum_robot/hardware/openrb_client.py)
- [dxl_bus.py](/Users/jacklange/Continuum/pi_code/continuum_robot/hardware/dxl_bus.py)
- [servo_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/servo_service.py)
- [safety_guard.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/safety_guard.py)
- [motor_control_supervisor.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/motor_control_supervisor.py)
- [segment_readiness.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/segment_readiness.py)
- [neutral_calibration_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/neutral_calibration_service.py)
- [pretension_validation_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/pretension_validation_service.py)
- [displacement_mapper.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/displacement_mapper.py)
- [telemetry_diagnostics.py](/Users/jacklange/Continuum/pi_code/continuum_robot/servos/telemetry_diagnostics.py)
- [system_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/system_controller.py)
- [servos_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/servos_controller.py)

Responsibilities:

- `OpenRbClient`: board/port readiness, connection state, OpenRB preparation.
- `DxlBus`: low-level DYNAMIXEL SDK transport, scan, telemetry, goal writes, ID writes.
- `ServoService`: canonical high-level servo API for GUI and experiment use (discovery, jogging, motion planning, pretension routines, neutral capture, runtime state, bench debug, bus ownership). This is the single seam — and currently a large one (~6k LOC). A future refactor may extract subsurfaces; for now treat the class as the boundary.
- `SafetyGuard`, `SegmentReadiness`, `MotorControlSupervisor`: layered checks on telemetry freshness, operating mode, current/voltage/temperature, hardware-error, and bounds before coordinated motion. Workspace-boundary command rejections are skipped-and-counted rather than terminating runs.
- `NeutralCalibrationService` and `PretensionValidationService`: calibration artifact lifecycle and pretension routines, including one-click whole-segment pretension on the Servos tab.
- GUI controllers: user-facing actions and status.

Current architectural rule:

- The GUI must not bypass `ServoService` for motion or telemetry.

## Tracking Subsystem

Primary files:

- [backend_router.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/backend_router.py)
- [ndi_backend.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/ndi_backend.py)
- [tracking_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/tracking_service.py)
- [diagnostics.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/diagnostics.py)
- [runtime_tip_policy.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/runtime_tip_policy.py)
- [two_segment_roles.py](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/two_segment_roles.py)
- [legacy_bridge/](/Users/jacklange/Continuum/pi_code/continuum_robot/tracking/legacy_bridge/) — compatibility-only retained shim; not for new work

Responsibilities:

- backend selection and fallback
- normalized tool state with `frame_number` plumb-through and an opt-in validity heuristic
- tracker health, freshness, and diagnostics
- robot-tip pose chain after registration
- shared runtime tip trust policy (`coil_as_tip` thesis-trusted; accepted calibration artifacts lower-trust until validated)
- tracker-displacement gate that blocks experiment motion when the tracker reports implausible jumps

Current architectural rule:

- All GUI, registration, diagnostics, and experiments consume `TrackingService`, not backend-specific objects.

## Registration Subsystem

Primary files:

- [registration_service.py](/Users/jacklange/Continuum/pi_code/continuum_robot/services/registration_service.py)
- [registration_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/registration_controller.py)
- [registration_trial_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/registration_trial_controller.py)
- [registration_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/registration_tab.py)
- [registration_landmark_map_widget.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/widgets/registration_landmark_map_widget.py)
- [rigid_solver.py](/Users/jacklange/Continuum/pi_code/continuum_robot/registration/rigid_solver.py) — SVD solver + opt-in RANSAC outlier-rejection variant
- [trial_analysis.py](/Users/jacklange/Continuum/pi_code/continuum_robot/registration/trial_analysis.py)
- [trial_cli.py](/Users/jacklange/Continuum/pi_code/continuum_robot/registration/trial_cli.py)
- [repository.py](/Users/jacklange/Continuum/pi_code/continuum_robot/registration/repository.py)
- [promote_registration_trial.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data/promote_registration_trial.py)

Responsibilities:

- `RegistrationService`: canonical session state, sample capture, solve, acceptance, persistence
- `RigidRegistrationSolver`: SVD-based 4-point solve plus opt-in RANSAC outlier rejection
- controller/tab: operator workflow, 4-of-N landmark selection, solve gating, overwrite confirmation
- `RegistrationTrialController` + trial dialog: capture N landmarks × K samples, sweep averaging methods, exhaustively try 4..8-of-N subsets, per-landmark leave-one-out FRE, samples-per-point ladder
- `promote_registration_trial`: explicit operator promotion of a trial result to `latest_registration.json` (never automatic)
- repository: persistent accepted registration artifacts

Current architectural rule:

- Landmark selection may change the 4-point subset and the averaging/outlier method (mean / median / trimmed_mean / mad_filtered_mean / RANSAC), but the math path remains the canonical rigid-registration solve in the service.

## Experiment Subsystem

Primary files:

- [experiment_runner.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/experiment_runner.py)
- [framework.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/framework.py)
- [registry.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/registry.py)
- [builtins.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/builtins.py) — large registrar and execution surface for most built-in experiments
- [critical_experiments.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/critical_experiments.py)
- [calibration_validation.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/calibration_validation.py) (registers `pivot_validation`, `registration_validation`)
- [single_segment_repeatability.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/single_segment_repeatability.py)
- [single_segment_repeatability_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/single_segment_repeatability_outputs.py)
- [registration_trial.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/registration_trial.py)
- [registration_trial_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/registration_trial_outputs.py)
- [penprobe_chasing_demo.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/penprobe_chasing_demo.py)
- [two_segment_startup_validation.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/two_segment_startup_validation.py)
- [two_segment_collect_pose_dataset.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/two_segment_collect_pose_dataset.py)
- [two_segment_repeatability.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/two_segment_repeatability.py)
- [pretension_validation_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/pretension_validation_outputs.py)
- [modeling_dataset_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/modeling_dataset_outputs.py)
- [transform_chain_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/transform_chain_outputs.py)
- [tracker_timing_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/tracker_timing_outputs.py)
- [servo_tracker_sync_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/servo_tracker_sync_outputs.py)
- [grid_accuracy_outputs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/experiments/grid_accuracy_outputs.py)
- [experiment_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/experiment_controller.py)
- [experiment_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/experiment_tab.py)

Responsibilities:

- canonical experiment execution
- dataset writing and summaries
- live visualization and result reloads
- preflight gating before operator runs
- thesis-figure contract: a small set of canonical headline + supporting figures per experiment (already applied to `pivot_validation`, `registration_validation`, `tracker_timing_validation`, `servo_tracker_sync_validation`)
- workspace-boundary command rejections are skipped-and-counted rather than terminating runs

Current architectural rule:

- No GUI-only experiment execution path. All experiment runs go through `ExperimentRunner`.

## Modeling Subsystem

Primary files:

- [ann_training.py](/Users/jacklange/Continuum/pi_code/continuum_robot/modeling/ann_training.py)
- [analysis.py](/Users/jacklange/Continuum/pi_code/continuum_robot/modeling/analysis.py)
- [two_segment/](/Users/jacklange/Continuum/pi_code/continuum_robot/modeling/two_segment/) — `cli.py`, `dataset.py`, `features.py`, `physics.py`, `models.py` (`HybridResidualModel` lives here), `validate_mike_cc.py`
- [modeling_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/modeling_tab.py)
- [two_segment_modeling_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/two_segment_modeling_tab.py)
- [modeling_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/modeling_controller.py)
- [ann_training_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/ann_training_controller.py)

Responsibilities:

- offline analysis of saved experiment bundles
- ANN training with multi-seed sweeps (Wolfe-style `seeds_per_architecture`)
- `HybridResidualModel`: physics baseline + learned residual, with before/after visualization bundle
- linear baselines + Mike/Camarillo constant-curvature models for two-segment data
- separate test-dataset picker for Wolfe §3.2.3 cross-acquisition evaluation
- thesis-grade gate chip (6 hard gates), 1 mm target reference line, Wolfe baseline reference line, top-K worst predictions

Current architectural rule:

- The modeling stack reads from `data/experiments/` and writes to `data/models/` and `data/modeling_results/` — it never reaches into hardware or live services. The Modeling and 2-Segment Modeling tabs are separate UI surfaces.

## Data Lifecycle Subsystem

Primary files:

- [data_management.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data_management.py)
- [run_management.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data/run_management.py)
- [export_run_bundle.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data/export_run_bundle.py)
- [validate_run_bundle.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data/validate_run_bundle.py)
- [build_thesis_evidence_index.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data/build_thesis_evidence_index.py)
- [manage_runs.py](/Users/jacklange/Continuum/pi_code/continuum_robot/data/manage_runs.py)
- [data_management_controller.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/controllers/data_management_controller.py)
- [data_management_tab.py](/Users/jacklange/Continuum/pi_code/continuum_robot/gui/tabs/data_management_tab.py)

Responsibilities:

- discover runs across `data/experiments/`, `data/mock_experiments/`, `data/experiments_archived/`, `data/trash/`
- `run_review.json` sidecars mark runs as `keep`, `thesis_candidate`, `advisor_share`, `debug`, `garbage`, or `archived` without touching raw outputs
- validate/export run bundles
- build thesis evidence index for advisor/Mac/ChatGPT handoff
- Data tab provides grouped tree view + bulk quick-clean

## Configuration And Persistent State

Runtime config:

- `config/system.yaml`
- `config/system.local.yaml`
- `config/robot_8servo.yaml` for the normal full-platform hardware profile
- `config/robot_4servo.yaml`
- `config/safety.yaml`
- `config/registration.yaml`
- `config/experiment.yaml`

Persistent runtime artifacts:

- `config/` for runtime YAML and the active servo startup calibration singleton
- `data/pivot_calibration/` for accepted and staged 0B tip-calibration artifacts plus pivot review data
- `data/pivot_calibration/captures/` for raw pivot-review capture CSVs
- `data/registrations/` for durable accepted/latest registration artifacts
- `data/runtime_tip_calibration/` for accepted `0A` runtime tip calibration
- `data/diagnostics/tracker_validation/` for saved tracker validation reports
- `data/diagnostics/data_management_migration/` for dry-run and applied migration ledgers
- `data/diagnostics/registration_trial/` for trial-CLI replay outputs
- `data/calibration/servo_calibration/` for archived servo startup calibration history
- `data/experiments/` for canonical experiment datasets
- `data/experiments/<experiment_name>/` for canonical run bundles grouped by experiment type
- `data/mock_experiments/` for mock-mode counterparts (kept out of thesis-evidence promotion paths)
- `data/experiments_archived/` and `data/trash/` for Data-tab lifecycle moves
- `data/exports/` for export bundles and the thesis evidence index
- `data/models/` and `data/modeling_results/` for trained models and modeling outputs

Current architectural rule:

- configuration defines defaults and machine-local startup state; accepted calibrations and experiment outputs live under their canonical `data/` categories
- repo-root `runs/` is retired and should not be used as an active runtime artifact location

## Recommended Next Architecture Boundary Work

In priority order:

1. **Land the gating bench run.** `data/experiments/single_segment_repeatability/` is empty. Until at least one `single_segment_repeatability` run is captured and promoted to `thesis_candidate`, modeling/two-segment claims sit on an empty foundation.
2. **Decompose `ServoService`.** ~6k LOC and ~145 methods in one class is the highest blast-radius surface in the repo. A clean split into discovery / motion / pretension / bus-ownership / telemetry sub-services would shrink the average change footprint without breaking the single-seam rule for callers.
3. **Split `experiments/builtins.py`** (~10k LOC) into per-experiment modules. The pattern is already established by `single_segment_repeatability.py`, `registration_trial.py`, `two_segment_collect_pose_dataset.py`, etc. — most remaining experiments could move out and re-register through `registry.py`.
4. **GUI test coverage.** `gui/` is the largest subsystem and the lowest test-to-code ratio (~28%). The recent Modeling tab build-out adds behavior faster than tests; the marker work in CI helps, but the underlying ratio should rise.
5. **Reduce silent `except Exception:` density** in `experiments/builtins.py`, `tracking/ndi_backend.py`, `modeling/analysis.py`, and `gui/widgets/experiment_pages.py`. At minimum route swallowed exceptions through `LOG.debug(..., exc_info=True)` so bench debugging has a trail.
