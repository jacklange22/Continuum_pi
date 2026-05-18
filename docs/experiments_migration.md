# Experiment Migration Note

The old experiment approach mixed valuable lab concepts with duplicated helper code and one-off script structure. The new canonical experiment framework keeps the concepts but moves execution, schemas, scheduling, and writing into shared modules under `continuum_robot/experiments/`.

## Old Concept To New Mapping

- `repeatability`
  - now represented authoritatively by `single_segment_repeatability`
  - `repeatability_dataset` remains only as hidden compatibility infrastructure where older data handling still depends on it
  - the revisit schedule explicitly revisits the same target from different prior states instead of flattening everything into one generic command list
- `aurora grid accuracy`
  - now represented directly by `aurora_grid_accuracy`
  - this isolates tracker bias, RMS error, and spread before registration or robot repeatability analysis
- `pivot calibration`
  - now represented directly by `pivot_calibration`
  - this replaces the old one-off tip-generation flow with a canonical offline/live experiment that writes both a dataset bundle and the resulting tip file
- `sweep`
  - now represented by `command_schedule_validation` and, when needed, by specific experiments using deterministic command schedules
- `hysteresis`
  - now represented conceptually by `single_segment_repeatability` approach-conditioned metrics and by future schedule-driven experiments
- `timing`
  - now represented by `tracker_pipeline_mock`, `command_schedule_validation`, and the per-sample timing fields in the canonical schema
- `transient`
  - now represented by the phase-aware dataset model in `single_segment_repeatability` and other canonical experiments
  - the settle/sample phases make transient-vs-steady-state analysis explicit instead of implicit in ad hoc scripts
- `tensioning`
  - not yet hardware-complete
  - when servo/current workflows are ready, this should become either a dedicated diagnostic experiment or a pretension/calibration service that the main experiments declare as a requirement
- `dataset collection`
  - now represented directly by `single_segment_repeatability`
  - `collect_pose_command_dataset` (GUI-labeled "Random Data Collection") is the canonical single-segment modeling dataset workspace, with `workspace_coverage`, `hysteresis_path_dependence`, `repeatability_linked`, and `angular_test_mesh` (Wolfe-style) dataset modes

## Canonical Experiments Registered Today

For the avoidance of doubt, this is the full set of experiments registered in the framework as of this writing. Names are the canonical strings consumed by `run_experiment.py`, `ExperimentRegistry`, and the GUI runner:

Tracking / calibration / registration:
- `tracker_pipeline_mock`
- `tracker_timing_validation`
- `servo_tracker_sync_validation`
- `aurora_grid_accuracy`
- `pivot_calibration`
- `pivot_validation`
- `registration_validation`
- `registration_trial` — sweep replay/comparison of saved registration captures (see [`registration_trial_workflow.md`](registration_trial_workflow.md))

Pretension / startup:
- `pretension_validation`

Single-segment runs and datasets:
- `single_segment_repeatability` — the thesis-gating repeatability experiment
- `collect_pose_command_dataset` — modeling dataset collection
- `repeatability_dataset` — hidden compatibility infrastructure; do not use for new work
- `penprobe_chasing_demo` — pen-probe demo / smoke

Two-segment:
- `two_segment_startup_validation`
- `two_segment_collect_pose_command_dataset`
- `two_segment_repeatability`

Utility / framework-internal:
- `command_schedule_validation`
- `dataset_schema_roundtrip`
- `replay_runner`
- `transform_chain_validation`

`command_schedule_validation` and `replay_runner` are hidden from the operator dropdown but still runnable via CLI.

## Legacy Files To Treat As Reference Inputs

These are still useful for comparison, but they should not be extended as active experiment infrastructure:

- `references/repeatability.py`
- `references/continuum_aurora.py`
- `references/continuum_arduino.py`
- `references/kinematics.py`

## Compatibility Modules In The Main Package

The following modules remain mainly to preserve compatibility with the current GUI/controller flow and should eventually be simplified further or archived once the CLI and GUI are fully migrated:

- `continuum_robot/experiments/experiment_loader.py`
- `continuum_robot/experiments/dat_writer.py`
- `continuum_robot/gui/controllers/experiment_controller.py`
- `continuum_robot/gui/tabs/experiment_tab.py`

## Canonical Path Going Forward

Use these modules for new experiment work:

- `continuum_robot/experiments/framework.py`
- `continuum_robot/experiments/schemas.py`
- `continuum_robot/experiments/schedules.py`
- `continuum_robot/experiments/dataset_io.py`
- `continuum_robot/experiments/critical_experiments.py`
- `continuum_robot/experiments/metrics.py`
- `continuum_robot/experiments/pivot_utils.py`
- `continuum_robot/experiments/validation.py`
- `continuum_robot/experiments/builtins.py`
- `continuum_robot/experiments/experiment_runner.py`
- `scripts/run_experiment.py`
