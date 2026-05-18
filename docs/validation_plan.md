# Validation Plan

## Principle

Validate the 4-servo `XC330-M288-T` system first. Use mock-mode and offline paths for software confidence, then close hardware-dependent gaps on the Pi bench.

## Servo Validation

### `SRV-V001` OpenRB / DYNAMIXEL Bring-Up

Method:

- save one-servo bring-up parameters in `System`
- connect OpenRB through the GUI with external power attached
- verify the prepared bridge state
- scan one test servo ID
- read model / firmware / position / current / voltage / temperature / error
- jog one test servo with conservative fine and coarse steps

Acceptance:

- OpenRB is prepared and the DYNAMIXEL bus is connected
- the expected test-servo ID appears
- telemetry fields are populated
- fine and coarse jog succeed through the canonical service/controller path
- EEPROM-only maintenance actions still require torque disabled

### `SRV-V002` Startup Calibration Artifact

Method:

- GUI-only startup calibration workflow
- save and reload the single-file servo calibration artifact
- verify per-servo bounds, threshold, direction, and compatibility fields are present

Acceptance:

- saved artifact exists
- neutral setpoint matches the saved present position
- calibration summary reports compatible robot metadata
- per-servo safe min/max bounds are available for the current config
- per-servo pretension threshold and tightening direction are visible

Status:

- implemented in the canonical GUI / service path
- still needs real bench validation on OpenRB + `XC330-M288-T`

### `SRV-V003` Pretension Foundation

Method:

- GUI-driven threshold entry
- run the cautious startup pretension routine on one servo
- verify stop-on-threshold, stop-on-overcurrent, stop-on-travel-limit, timeout, and cancel behavior

Acceptance:

- threshold is operator-tunable
- stop condition is driven by current and bounded travel
- result status and final position/current are persisted in the calibration artifact
- accepted results are visible in the GUI calibration summary

Status:

- implemented in dry validation
- still hardware-dependent on real tension/current behavior

## Tracking Validation

### `TRK-V001` Tracker Doctor / Smoke / Benchmark

Method (preferred):

- `scripts/run_lab_workflow.py tracker-doctor -- ...`
- `scripts/run_lab_workflow.py tracker-smoke -- ...`
- `scripts/run_lab_workflow.py tracker-benchmark -- ...`

(Direct scripts `scripts/run_tracker_doctor.py`, `scripts/run_tracker_smoke.py`, `scripts/run_tracker_benchmark.py` still exist as the underlying entry points.)

Acceptance:

- correct backend selected
- tools visible
- freshness and FPS within configured thresholds

### `TRK-V002` Aurora Grid Accuracy

Method:

- run `aurora_grid_accuracy`
- compare measured points to truth grid

Acceptance:

- per-point RMS and overall RMS saved
- outliers classified clearly

## Registration Validation

### `REG-V001` Candidate Landmark Config

Method:

- load `config/registration.yaml`
- confirm candidate landmark metadata appears in the GUI

Acceptance:

- IDs, display labels, XYZ coordinates, and enabled flags load correctly

### `REG-V002` 4-Point Selection And Solve Gating

Method:

- select landmarks from the map/list
- attempt solve with missing selections or missing samples

Acceptance:

- duplicates prevented
- solve blocked until 4 unique points and enough samples exist

### `REG-V003` Registration Quality

Method:

- perform repeated sample capture and solve
- review FRE / RMSE, max landmark residual, worst landmark, and geometry diagnostics before accepting

Acceptance:

- accepted registration file is saved
- saved result reloads into the GUI
- tracking pipeline consumes the saved file on refresh
- per-landmark residual norms are saved
- transform-chain status reports whether tracking is using the accepted artifact

### `REG-V004` Repeated Registration Validation

Method:

- repeat the full registration workflow multiple times with the same landmark set
- save each accepted result
- review the repeated-validation summary under `data/registrations/validation/`

Acceptance:

- repeated runs report transform deltas between solves
- repeated runs report FRE summary across runs
- repeated runs report per-landmark residual trends across runs
- the GUI shows whether any landmark is consistently worse than the others

### `REG-V006` Registration Trial — Sweep Replay

Method:

- click **Run Registration Trial →** on the Registration tab to capture N landmarks × K samples
- review `trial_report.md` for the averaging-method sweep (mean / median / trimmed_mean / mad_filtered_mean), 4..8-of-N subset solves, per-landmark leave-one-out FRE, and samples-per-point ladder
- alternatively, replay on already-captured data via `python -m continuum_robot.registration.trial_cli data/registrations/latest_registration.json --output-dir data/diagnostics/registration_trial`

Acceptance:

- four trial output reports written (point_spread, subset_rms, samples_per_point, method_comparison)
- recommendations surface coplanar-truth-geometry warnings, outlier-landmark callouts (LOO drop > 0.05 mm), and capture-count warnings (all methods agreeing within 0.01 mm)
- promotion to `latest_registration.json` is explicit via `python -m continuum_robot.data.promote_registration_trial --run-dir <trial>` after operator review (never automatic)

### `REG-V007` Runtime Tip Calibration Chain Validation

Method:

- launch `Open Runtime Tip Calibration` from `Registration`
- capture the hat truth points with the accepted `0B` pen probe calibration
- collect stationary `0A` samples
- solve and save the runtime tip artifact
- verify tracking reports the selected runtime tip policy and does not present lower-trust calibration artifacts as thesis-trusted

Acceptance:

- saved artifact includes `T_coil_tip`, `T_tip_aurora`, and `T_aurora_coil_avg`
- hat-fit RMSE and max residual are saved
- `0A` translation and rotation spread summaries are saved
- tracking clearly distinguishes:
  - `coil_as_tip` thesis-trusted position path
  - accepted runtime tip artifact loaded as `lower_trust`
  - missing runtime tip artifact
  - identity fallback
  - invalid runtime tip artifact
- `transform_chain_summary.json`, `transform_chain_summary.txt`, and `transform_chain_overview.png` are written for current-state diagnostics and experiment runs

## Experiment Validation

### `EXP-V001` Pivot Calibration

Method:

- run `pivot_calibration` from a sample file and later live hardware

Acceptance:

- tip file written
- review dataset bundle written under `data/pivot_calibration/`
- RMSE and inlier/outlier counts saved

### `EXP-V002` Single-Segment Repeatability — **OPEN / GATING**

Method:

- run `single_segment_repeatability` live after preflight is fully ready

Acceptance:

- canonical dataset written
- summary includes repeatability metrics, run-validity coverage, and provenance
- target comparison possible against `< 1 mm`

Status: `data/experiments/single_segment_repeatability/` is empty as of 2026-05-17. This is the gating thesis experiment; every claim that depends on rung 9 of the validation ladder is structurally lower-trust until at least one run lands here.

### `EXP-V003` Modeling Dataset (Random Data Collection)

Method:

- run `collect_pose_command_dataset` (operator label: "Random Data Collection")
- collect `workspace_coverage`, `hysteresis_path_dependence`, `repeatability_linked`, or `angular_test_mesh` (Wolfe §3.2.3 cross-acquisition) datasets under trusted preflight

Acceptance:

- canonical run bundle written under `data/experiments/collect_pose_command_dataset/`
- summary records registration/runtime-tip/pretension provenance
- accepted vs rejected captures are explicit
- ordered export rows are preserved for later offline ANN / state-aware training
- robot-frame tip tangent/orientation are present when the transform chain is trusted
- workspace-boundary command rejections are skipped-and-counted (`workspace_boundary_skip_count` metric + `command_skipped_workspace_boundary` event) rather than aborting the run

## Modeling Validation

### `MOD-V001` Thesis-Grade Gate Chip

Method:

- train an ANN run from the Modeling tab
- inspect the thesis-grade chip and per-dataset mode chip

Acceptance:

- chip enforces all 6 hard gates (dataset trust, sample count, runtime-tip trust, registration trust, pretension provenance, and tracker freshness)
- chip explicitly distinguishes thesis-grade, near-thesis, and debug-only states

### `MOD-V002` Wolfe Cross-Acquisition Evaluation

Method:

- in the Modeling tab, pick a separate test dataset distinct from the training dataset
- train and review the headline RMSE chart

Acceptance:

- the RMSE chart shows the 1 mm target line and the Wolfe baseline reference
- top-K worst predictions are surfaced
- the mismatch chip is shown if training and test datasets use incompatible modes
- a per-axis tooltip and copy-summary-to-clipboard action are available

### `MOD-V003` Multi-Seed Sweep

Method:

- run the ANN sweep with `seeds_per_architecture` > 1

Acceptance:

- per-architecture variance is reported across seeds
- sweep result rows are sorted/filterable in the popout
- small-eval-dataset warning chip is surfaced when the eval dataset is too small to be informative

### `MOD-V004` HybridResidualModel Before/After

Method:

- train a `HybridResidualModel` from the Modeling tab

Acceptance:

- before/after visualization bundle is written
- both the baseline physics prediction and the residual-corrected prediction are reviewable

## GUI Validation

### `GUI-V001` Registration Usability

Method:

- operate the Registration tab on multiple window sizes
- verify map selection, table selection, capture progression, solve, save, reload

Acceptance:

- key controls remain visible
- status text is readable
- no clipped critical text or unusable pane sizes

### `GUI-V002` Experiment Workspace Safety

Method:

- verify preflight, output-path preview, run history reload, and visualization fallback

Acceptance:

- blocked vs warning states are clear
- run history reload works without hardware
- the GUI remains usable when advanced 3D is unavailable

### `GUI-V003` Servo Calibration Visibility

Method:

- open the `Servos` tab after loading or saving calibration
- review the calibration summary card and per-servo rows

Acceptance:

- calibration existence, compatibility, path, and update time are visible
- neutral values, bounds availability, and threshold availability are visible per servo

## Current Priority Order

1. bench-validate one-servo OpenRB bring-up with external power
2. bench-validate startup calibration and cautious pretension on one servo
3. expand from one-servo validation to the real 4-servo build
4. bench-validate pivot calibration and registration
5. run live repeatability datasets and compare against the `< 1 mm` target
