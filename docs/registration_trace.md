# Registration Trace

This document traces the canonical 4-point registration workflow from the GUI to persistence. It is intended as a narrow correctness map for the current repo, not a design proposal.

## Canonical Path

`RegistrationTab` -> `RegistrationController` -> `RegistrationService` -> `RegistrationRepository`

Primary files:

- `continuum_robot/gui/tabs/registration_tab.py`
- `continuum_robot/gui/controllers/registration_controller.py`
- `continuum_robot/services/registration_service.py`
- `continuum_robot/registration/repository.py`
- `continuum_robot/services/models.py`

## Selection

The operator chooses exactly 4 enabled model points before starting capture.

- `RegistrationTab.landmark_map.pointToggled` and `RegistrationTab.available_points_table.cellClicked`
  call `RegistrationController.toggle_selected_model_point(...)`.
- `RegistrationController.set_selected_model_point(...)` and
  `RegistrationController.toggle_selected_model_point(...)` enforce:
  - enabled point only
  - at most 4 points
  - no duplicates
- `RegistrationController.selection_is_ready()` is the canonical readiness check for starting a session.

Relevant functions:

- `RegistrationTab._on_available_point_clicked`
- `RegistrationTab._safe_call`
- `RegistrationController.toggle_selected_model_point`
- `RegistrationController.set_selected_model_point`
- `RegistrationController.selection_is_ready`

## Begin Session

The GUI never creates a session locally. It always asks `RegistrationService` to do it.

- `RegistrationTab.begin_button` -> `RegistrationController.begin_session()`
- In simple 4-point mode, the controller passes:
  - the 4 selected labels
  - the matching nominal robot-frame landmarks
- `RegistrationService.begin_session(...)` creates the active `RegistrationSnapshot`:
  - `active=True`
  - `labels`
  - `current_label`
  - empty per-label raw sample lists
  - empty per-label captured counts

Relevant functions:

- `RegistrationController.begin_session`
- `RegistrationService.begin_session`

## Capture One Sample

Capture is the critical canonical action. The GUI must not append samples on its own.

- `RegistrationTab.capture_button` -> `RegistrationController.capture_current_label_sample()`
- `RegistrationController.capture_current_label_sample()` delegates to
  `RegistrationController.capture_label_sample(label)`
- `RegistrationController.capture_label_sample(...)` delegates to
  `RegistrationService.capture_sample(label)`
- `RegistrationService.capture_sample(...)`:
  - checks that a session is active
  - resolves the measurement tool through `TrackingService.get_latest_tool(...)`
  - converts the live tool transform into the measurement point
  - appends the point into `RegistrationSnapshot.raw_points_by_label[label]`
  - appends the raw measurement-tool pose into `raw_measurement_tool_samples_by_label[label]`
  - appends coil-tool pose samples too when legacy asset mode is active
  - updates `captured_counts[label]`

Relevant functions:

- `RegistrationController.capture_current_label_sample`
- `RegistrationController.capture_label_sample`
- `RegistrationService.capture_sample`
- `RegistrationService._require_tool_snapshot`
- `RegistrationService._measurement_point_from_tool_snapshot`

## Per-Point Completion

The controller does not advance labels locally.

- `RegistrationTab.complete_button` -> `RegistrationController.complete_current_label()`
- `RegistrationController.complete_current_label()` delegates to
  `RegistrationService.complete_landmark()`
- `RegistrationService.complete_landmark()` blocks until the current label has
  at least `captures_per_landmark` samples, then advances `current_label`

Relevant functions:

- `RegistrationController.complete_current_label`
- `RegistrationService.complete_landmark`

## Solve Readiness

Two layers must agree before solve:

- GUI/controller gate:
  - `RegistrationController.is_ready_to_solve()`
  - requires exactly 4 selected points
  - requires enough samples for every active label
- service gate:
  - `RegistrationService._solve_simple_registration()` or
    `RegistrationService._solve_legacy_compatible_registration()`
  - rechecks capture counts before solving

This means the solve button is disabled when incomplete, and the service still
refuses invalid direct calls.

Relevant functions:

- `RegistrationController.is_ready_to_solve`
- `RegistrationController.solve_session`
- `RegistrationService.solve_registration`

## Solve Execution

For the current simple 4-point GUI flow:

- `RegistrationController.solve_session()` -> `RegistrationService.solve_registration()`
- `RegistrationService._solve_simple_registration()`:
  - averages repeated samples per selected label
  - builds measured and nominal point sets in label order
  - solves `T_robot_aurora` through `RigidRegistrationSolver.solve_T_robot_aurora(...)`
  - computes residuals and FRE
  - stores a pending `RegistrationRecord`
  - updates `RegistrationSnapshot.pending_accept=True`

Relevant functions:

- `RegistrationController.solve_session`
- `RegistrationService._solve_simple_registration`
- `continuum_robot/registration/rigid_solver.py`

## Save / Overwrite

Saving is a separate explicit action after solve.

- `RegistrationTab.save_button` -> `RegistrationController.save_registration(...)`
- `RegistrationController.save_registration(...)` checks whether
  `latest_registration.json` already exists and requires explicit overwrite confirmation
- `RegistrationService.accept_registration()` is the canonical persistence step
- `RegistrationRepository.save_record(...)` writes:
  - timestamped `registration_*.json`
  - `latest_registration.json`

Relevant functions:

- `RegistrationTab._save_registration`
- `RegistrationController.save_registration`
- `RegistrationService.accept_registration`
- `RegistrationRepository.save_record`

## Reload Latest Accepted Registration

- `RegistrationTab.load_button` -> `RegistrationController.load_latest_result()`
- `RegistrationController.load_latest_result()` delegates to
  `RegistrationService.load_latest_accepted()`
- `RegistrationService.load_latest_accepted()` restores the GUI-facing snapshot fields needed for display:
  - labels
  - raw points
  - averaged points
  - captured counts
  - residuals
  - FRE
  - latest accepted path

Relevant functions:

- `RegistrationController.load_latest_result`
- `RegistrationService.load_latest_accepted`

## Persistence Artifact

Canonical persistence path:

- directory: `data/registrations/`
- latest file: `data/registrations/latest_registration.json`
- timestamped files: `data/registrations/registration_*.json`

Current record contents are defined by `RegistrationRecord` in
`continuum_robot/registration/repository.py`.

## Dry-Validation Coverage

The dry tests that validate this path live in:

- `tests/test_gui_controllers.py`
- `tests/test_registration_service.py`

They cover:

- selection and duplicate prevention
- widget-driven capture
- solve gating
- overwrite confirmation
- save artifact creation
- reload of the latest accepted registration
