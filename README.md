# Continuum Robot Operator Platform

This repository contains the current Raspberry Pi operator stack for a tendon-driven continuum robot. It integrates Aurora tracking, robot/body registration, runtime tip calibration, OpenRB/DYNAMIXEL servo control, pretension/startup-state handling, experiment execution, logging, and analysis-oriented data management.

The project is a research and thesis platform, not just a control GUI. Its central job is to expose, quantify, and reduce error across the whole robot pipeline: tracker consistency, transform-chain correctness, registration, base motion, servo positioning, pretension state, hysteresis, and repeatability.

## Why This Exists

The previous architecture depended on scattered scripts, legacy bridge paths, weak validation boundaries, and startup assumptions that made it hard to know where error entered the system. This repo moves the workflow into one Python application with explicit services, repeatable calibration artifacts, GUI-guided operator steps, canonical experiment datasets, and validation outputs.

The immediate research focus is validated single-segment repeatability and startup-state characterization. The system supports Segment A `[1,2,3,4]`, Segment B `[5,6,7,8]`, and all-8 readiness/manual-startup foundation work, but current validation should prioritize the single-segment ladder before full two-segment control.

## System At A Glance

Canonical live runtime:

```text
Aurora tracker
  -> Python NDI backend
  -> TrackingService
  -> registration/runtime-tip chain
  -> GUI validation + experiments

OpenRB-150 + XC330 servos
  -> DYNAMIXEL SDK bus
  -> ServoService
  -> calibration, safety, pretension, motion commands
  -> GUI validation + experiments

ExperimentRunner
  -> built-in experiment registry
  -> canonical JSONL/YAML/summary datasets
  -> plots and analysis artifacts
```

Primary transform convention:

- `T_A_B` maps coordinates from frame `B` into frame `A`.
- Composition is `T_A_C = T_A_B @ T_B_C`.
- Live runtime tip pose is `T_robot_tip = T_robot_aurora @ T_aurora_0A @ T_coil_tip`.
- Current runtime tip trust policy treats `coil_as_tip` as the thesis-trusted robot-frame position path when the live tip pose is available. `latest_accepted` runtime tip calibration artifacts are still useful, but are recorded as `lower_trust` until that workflow is independently validated.

Runtime tool roles:

- `0A`: coil/runtime tool used for the robot tip chain.
- `0B`: probe/measurement tool used for pivot calibration and registration.
- Current Pi Aurora aliases map raw live IDs `10 -> 0A` and `11 -> 0B`.

## Core Capabilities

Working baseline workflows:

- PySide6 operator GUI with `System`, `Tracking`, `Registration`, `Servos`, `Experiment`, `Modeling`, `2-Segment Modeling`, and `Data` tabs.
- Python-native Aurora path through `scikit-surgerynditracker`, `TrackingBackendRouter`, `TrackerBackendNDI`, and `TrackingService`, with `frame_number` plumb-through, opt-in validity heuristic, and a tracker-displacement gate for safer hardware runs.
- Tracker diagnostics, smoke tests, and timing benchmarks.
- GUI-guided 4-point robot/body registration using `0B`, candidate landmarks from `config/registration.yaml`, repeated captures, solve/review/save, persisted registration artifacts, optional RANSAC outlier rejection, and a **Run Registration Trial** sweep that replays captures across averaging methods and label subsets to recommend the lowest-FRE protocol.
- Runtime tip policy support for `coil_as_tip`, accepted runtime tip calibration artifacts, and quick 4-point overrides, with trust status recorded in preflight, metadata, and transform-chain outputs.
- OpenRB/DYNAMIXEL transport through the Robotis SDK, including connect, scan, telemetry, ID assignment, operating mode writes, goal position writes, and conservative one-servo bring-up tools.
- Servo calibration artifacts with neutral ticks, safe bounds, pretension thresholds, tightening metadata, and compatibility checks.
- One-click segment pretension on the `Servos` tab plus pretension validation, with manual/automatic state persistence and an algorithm-vs-manual comparison report used by experiment preflight.
- Canonical experiment runner with structured metadata, samples, summaries, thesis-grade figures (two-figure contract per experiment), and config snapshots. Workspace-boundary command rejections are skipped-and-continued instead of aborting the run.
- Modeling tab with thesis-grade chip, headline RMSE chart against the 1 mm target and Wolfe baseline reference, top-K worst predictions, separate test-dataset picker (Wolfe §3.2.3 cross-acquisition eval), and a multi-seed sweep. `HybridResidualModel` ships with before/after visualization. Two-segment modeling lives in its own dedicated tab.
- Data-management GUI with grouped tree view, bulk quick-clean, and discover/migrate/review/safely-delete support for runtime artifacts.
- CI runs the full non-GUI test suite on every push and PR (see `.github/workflows/tests.yml`).

In-progress or still bench-sensitive workflows:

- Pretension is central but still being characterized. It should be treated as a measured startup-state workflow, not solved tendon-force sensing.
- Single-segment repeatability is the main thesis experiment, but run quality depends on accepted registration, the shared runtime tip trust policy, servo calibration, pretension state, and live tracker freshness.
- Modeling/babble dataset collection exists, but it should follow repeatability and pretension validation rather than displace them.
- `dual_segment` is currently all-8 readiness/manual-startup foundation only; it is not full two-segment control. `parallel_single` is mirrored single-segment testing only.

Legacy/reference paths:

- `legacy/tracker_bridge/` and `continuum_robot/tracking/legacy_bridge/` are retained for comparison/debugging only. The active live tracking path is Python NDI.
- `references/` and `tools/` contain protected historical scripts, captured inputs, SolidWorks-derived assets, transform exports, and vendor notes. Treat them as read-only reference material.

## Hardware And Software Stack

Hardware target:

- Raspberry Pi host with local GUI or remote desktop.
- NDI/Aurora tracker connected directly to the Pi.
- Tracked coil/probe tools visible to Aurora.
- OpenRB-150 connected directly to the Pi.
- Robotis XC330-M288-T / X-series DYNAMIXEL servos, currently validated first as a 4-servo single segment.
- External DYNAMIXEL power for real motion tests.

Software target:

- Python 3.10+
- `numpy`, `PyYAML`, `pyserial`, `dynamixel-sdk`, `PySide6`, `scikit-surgerynditracker`
- `pytest` for tests

The repository installs Python dependencies, but hardware-side vendor/runtime setup still matters. The Python NDI tracker path depends on a working local `scikit-surgerynditracker` installation for the Aurora system. The retired C++ bridge additionally requires external NDI SDK setup and should only be used deliberately.

## Repository Structure

- `continuum_robot/app/`: application bootstrap and service registry.
- `continuum_robot/tracking/`: Aurora backends, parser/framing utilities, diagnostics, timing, runtime models, and transform helpers.
- `continuum_robot/services/`: app-facing tracking, registration, runtime tip calibration, health, and capture services.
- `continuum_robot/registration/`: rigid registration, capture sessions, persistence, validation, legacy compatibility, and runtime tip repositories.
- `continuum_robot/servos/`: high-level servo service, safety guard, neutral/pretension calibration, displacement mapping, tendon mapping, telemetry diagnostics, and motion supervision.
- `continuum_robot/hardware/`: OpenRB, DYNAMIXEL, Aurora, and mock hardware clients.
- `continuum_robot/experiments/`: experiment framework, registry, built-ins, schemas, output writers, validation experiments, repeatability workflow, and dataset tools.
- `continuum_robot/gui/`: PySide6 tabs, controllers, widgets, theme, preflight checks, and visualization helpers.
- `continuum_robot/modeling/`: ANN training, modeling analysis helpers, `HybridResidualModel`, and the `two_segment/` modeling stack.
- `config/`: runtime YAML/JSON configuration and example experiment configs.
- `data/`: runtime outputs, diagnostics, calibration artifacts, experiment datasets, logs, models, and modeling results.
- `docs/`: deeper architecture, workflow, trace, validation, and testing notes.
- `scripts/`: bootstrap, GUI launchers, tracker diagnostics, registration tools, experiment CLI, workflow wrapper, and migration utilities.
- `tests/`: unit and integration-style coverage for math, parsing, persistence, services, calibration, safety, experiments, and GUI-adjacent logic.
- `references/`, `tools/`: protected read-only reference inputs.

## Key Runtime Artifacts

- `config/system.yaml`: default runtime settings, mock/live mode, serial ports, tracker aliases, safety overrides, and artifact paths.
- `config/system.local.yaml`: machine-specific override copied from `config/system.local.example.yaml`; should remain local.
- `config/registration.yaml`: canonical 4-point registration setup, candidate landmarks, `0A`/`0B` roles, and runtime-tip calibration references.
- `config/neutral_setpoints.json`: active servo startup calibration artifact path for neutral, bounds, and pretension source metadata.
- `data/registrations/latest_registration.json`: accepted robot/body registration.
- `data/runtime_tip_calibration/latest_runtime_tip_calibration.json`: accepted `0A` runtime tip calibration.
- `data/pivot_calibration/generated_penprobe_tip.csv`: generated `0B` pivot tip file used by registration.
- `data/experiments/<experiment_name>/<timestamp>_<experiment_name>/`: canonical experiment bundles with `metadata.json`, `summary.json`, `samples.jsonl`, config snapshots, and generated plots/text reports.
- `data/logs/current/` and `data/logs/recent/`: operator GUI/session logs.

## Operator Workflow

Typical live sequence:

1. Configure `config/system.local.yaml` for the current machine, ports, robot config, and `mock_mode`.
2. Launch the GUI or run tracker diagnostics first.
3. In `Tracking`, connect Aurora and verify `0A`/`0B` visibility, freshness, valid transforms, and timing.
4. Run or accept `0B` pivot calibration so the pen-probe tip file is current.
5. In `Registration`, select four suitable landmarks, capture repeated `0B` samples, solve, review FRE/RMSE, and save the accepted registration. To diagnose which landmarks and how many samples actually drive FRE on this rig, click **Run Registration Trial →** on the Registration tab; it captures N landmarks × K samples, sweeps averaging methods and subsets, and produces a report. After review, `python -m continuum_robot.data.promote_registration_trial --run-dir <trial>` writes the chosen subset as the new active registration (backing up the previous one first).
6. Confirm `T_robot_tip` is computable from live `0A`, registration, and the selected runtime tip policy. For thesis repeatability, the current trusted path is `coil_as_tip`.
7. In `System`/`Servos`, connect OpenRB, prepare the DYNAMIXEL bus, scan IDs, verify telemetry, capture neutral/bounds, and only then jog or command motion.
8. On the `Servos` tab, click **Run Pretension Trial** for one-click segment pretension; characterize or capture the startup pretension state and accept the source before thesis-grade experiments. The algorithm-vs-manual comparison report is generated alongside.
9. In `Experiment`, run validation or repeatability experiments only after preflight checks pass.
10. Use `Data` to inspect/migrate/review/safely-delete datasets and outputs; use `Modeling` and `2-Segment Modeling` to train models and review thesis-grade plots.

## Validation And Experiments

Important built-in experiment/workflow names:

- `aurora_grid_accuracy`: tracker grid consistency before registration or robot motion.
- `pivot_calibration`: `0B` pivot calibration and generated tip-file workflow.
- `pivot_validation`: offline/live validation of pivot calibration quality.
- `registration_validation`: registration artifact validation and repeatability analysis.
- `registration_trial`: capture N landmarks × K samples, sweep averaging methods and label subsets, recommend the protocol that minimizes FRE on the actual rig. See `docs/registration_trial_workflow.md`.
- `tracker_timing_validation`: tracker timing/freshness characterization.
- `servo_tracker_sync_validation`: timing relationship between servo commands and tracker samples.
- `pretension_validation`: current/travel and startup-state pretension characterization.
- `single_segment_repeatability`: canonical live thesis repeatability protocol using the legacy 17-target all-other-approaches structure. **Status: zero live bench runs as of 2026-05-17 — this is the gating thesis experiment.**
- `collect_pose_command_dataset`: GUI-labeled "Random Data Collection" — modeling dataset collection with `workspace_coverage`, `hysteresis_path_dependence`, `repeatability_linked`, and `angular_test_mesh` (Wolfe-style) modes.
- `two_segment_startup_validation`, `two_segment_collect_pose_command_dataset`, `two_segment_repeatability`: two-segment counterparts. Foundational; full two-segment kinematics/control still ahead.
- `penprobe_chasing_demo`: pen-probe demo / smoke.
- `command_schedule_validation`, `replay_runner`, `dataset_schema_roundtrip`, `transform_chain_validation`, `tracker_pipeline_mock`: support/diagnostic workflows. `command_schedule_validation` and `replay_runner` are hidden from the operator dropdown but still runnable via CLI.

Current validation ladder:

1. Tracker connectivity, tool IDs, transform validity, and timing.
2. `0B` pivot calibration and pivot validation.
3. 4-point registration and robot-frame tip-pose validation.
4. One-servo OpenRB/DYNAMIXEL bring-up with real telemetry and bounded jogs.
5. 4-servo neutral capture, calibrated bounds, and safe motion semantics.
6. Pretension characterization and accepted startup-state artifact.
7. Single-segment repeatability and path-dependence/hysteresis metrics.
8. Modeling dataset collection and later two-segment scaling.

## How To Run

Bootstrap from the repository root:

```bash
PYTHON_BIN=python3 scripts/bootstrap.sh
```

Create and edit machine-local config:

```bash
cp config/system.local.example.yaml config/system.local.yaml
```

Useful local config fields:

```yaml
mock_mode: false
robot_config: "robot_8servo.yaml"
aurora_port: "/dev/ttyUSB0"
openrb_port: "/dev/ttyACM0"
tracker_backend: "ndi"
tracker_fallback_enabled: false
tracker_tool_id_aliases:
  "10": "0A"
  "11": "0B"
```

List common workflows:

```bash
.venv/bin/python scripts/run_lab_workflow.py list
```

Launch the full GUI:

```bash
scripts/run_gui.sh
```

Launch the focused tracker-first workflow:

```bash
.venv/bin/python scripts/run_lab_workflow.py tracker-mvp
```

Run tracker diagnostics:

```bash
.venv/bin/python scripts/run_lab_workflow.py tracker-doctor -- --tracker-port /dev/ttyUSB0
.venv/bin/python scripts/run_lab_workflow.py tracker-smoke -- --tracker-port /dev/ttyUSB0
.venv/bin/python scripts/run_lab_workflow.py tracker-benchmark -- --tracker-port /dev/ttyUSB0 --duration-s 5
```

Run an experiment from the CLI:

```bash
.venv/bin/python scripts/run_lab_workflow.py experiment -- --experiment single_segment_repeatability --config config/experiment_single_segment_repeatability.example.yaml
.venv/bin/python scripts/run_lab_workflow.py experiment -- --experiment registration_trial --config config/experiment_registration_trial.example.yaml
```

Promote a `registration_trial` result into the active registration slot only after you have reviewed its `trial_report.md`:

```bash
.venv/bin/python -m continuum_robot.data.promote_registration_trial \
  --run-dir data/experiments/registration_trial/<run> \
  [--subset L1,L2,L4,L7] \
  --operator-note "Trial T3: L4/L7 lowers FRE by 0.4 mm"
```

The promote tool backs up the current `latest_registration.json` to a timestamped sibling before overwriting. Pass `--dry-run` to validate inputs without writing anything.

Run tests:

```bash
scripts/run_tests.sh quick
scripts/run_tests.sh hardware-safe
```

Use the canonical runner instead of bare `pytest` so the active `.venv` Python is used and Python-version skips are not missed. It requires Python 3.10+; Python 3.11+ is recommended. Useful modes are `quick`, `hardware-safe`, `full-nongui`, and `gui`.

The `gui` pytest marker is registered in `pyproject.toml` and applied at module scope to the truly Qt-dependent test files. `scripts/run_tests.sh full-nongui` therefore actually skips the GUI suite (`pytest -m "not gui"`), and `scripts/run_tests.sh gui` runs just the marked subset. If you add a new test file that imports PySide6 at module level and exercises real Qt widgets, set `pytestmark = pytest.mark.gui` near the top so the split keeps working.

Run the no-hardware operator readiness checks before a bench session:

```bash
.venv/bin/python -m continuum_robot.diagnostics.prehardware_dry_run
.venv/bin/python -m continuum_robot.diagnostics.hardware_readiness_check
```

`prehardware_dry_run` exercises the fixture/export/validator path. `hardware_readiness_check` adds hardware-day acceptance checks for config, serial-port setup, tracking-role config, GUI page construction, two-segment physics config readiness, and the nested dry run.

Use `docs/hardware_day_runbook.md` as the concise bench-session checklist.

### Thesis data export workflow

Experiment run folders under `data/experiments/<experiment>/...` are the canonical source of summaries, metrics, figures, samples, and provenance. To package one run for a Mac, ChatGPT review, or advisor handoff, use the bounded exporter instead of copying the whole `data/` tree:

```bash
.venv/bin/python -m continuum_robot.data.export_run_bundle --latest single_segment_repeatability --zip
.venv/bin/python -m continuum_robot.data.export_run_bundle --latest pretension_validation --include-samples --zip
.venv/bin/python -m continuum_robot.data.export_run_bundle --run-dir data/experiments/registration_validation/<run_folder> --zip
```

The wrapper script is equivalent:

```bash
scripts/export_run_bundle.py --latest collect_pose_command_dataset --zip
```

Bundles are written to `data/exports/` by default. They include summaries, trust/provenance metadata, report figures, metrics, config snapshots, a manifest with checksums, and a short README. Raw `samples.jsonl` files are excluded unless `--include-samples` is passed, and debug-heavy files are excluded unless `--include-debug` is passed.

To transfer from the Pi to a Mac, run from the Mac and replace the host/path:

```bash
rsync -av pi@<pi-host>:/home/pi/Continuum/pi_code/data/exports/<bundle>.zip ~/Downloads/
```

Before using a run in thesis/modeling work, inspect `trust_provenance.json` or `summary.json` for `run_trust_mode`, `valid_for_model_training`, `valid_for_thesis_repeatability`, and any `data_quality_warnings`. A successful run status is not by itself a thesis-valid label.

You can also validate an existing run folder directly:

```bash
.venv/bin/python -m continuum_robot.data.validate_run_bundle data/experiments/<experiment>/<run_folder>
```

Use the Data tab, or the CLI below, to review runs without modifying raw experiment files. Review state is stored in a small `run_review.json` sidecar:

```bash
.venv/bin/python -m continuum_robot.data.manage_runs --mark data/experiments/<experiment>/<run_folder> --status thesis_candidate --notes "Good registration and repeatability run"
.venv/bin/python -m continuum_robot.data.manage_runs --archive data/experiments/<experiment>/<run_folder>
.venv/bin/python -m continuum_robot.data.manage_runs --trash data/experiments/<experiment>/<run_folder>
.venv/bin/python -m continuum_robot.data.build_thesis_evidence_index
```

Archive moves runs to `data/experiments_archived/`; trash moves runs to `data/trash/`. Neither action deletes exported bundles, and protected review statuses (`keep`, `thesis_candidate`, `advisor_share`) require explicit force/confirmation before moving. Important curated runs under `data/experiments/` and `data/experiments_archived/` are intentionally trackable so they can move through GitHub when needed. Generated exports, trash, logs, temporary diagnostics, caches, and archive bundles stay ignored.

Before committing experiment data, run the data hygiene checker:

```bash
scripts/check_data_for_git.py
```

The checker reports unreviewed runs, review statuses, thesis candidates missing report/trust evidence, export/trash artifacts that should not be committed, and large files. GitHub rejects normal Git files over 100 MB; use Git LFS or a separate transfer method for large raw artifacts. Export bundles are convenient for handoff, but they should generally stay out of source control unless intentionally small and explicitly reviewed.

## Current Project Status

Stable enough for software development:

- mock-mode app bootstrap and GUI flow
- tracking service abstractions and diagnostics
- registration math, persistence, and GUI capture path
- DYNAMIXEL SDK transport layer structure
- servo calibration artifact model and safety checks
- experiment registry, runner, datasets, summaries, and many analysis outputs
- repeatability and pretension experiment infrastructure

Still exploratory or hardware-dependent:

- actual Aurora and OpenRB behavior on a given Pi/bench setup
- pretension thresholds and startup-state repeatability
- true motion repeatability under tendon load
- current-aware servo modes beyond the default position-control experiment path
- two-segment robot behavior
- quantitative precision claims beyond measured validation outputs

## Development Priorities

Current practical priorities:

1. Make startup state repeatable: neutral capture, bounds, pretension source, and preflight enforcement.
2. Treat repeatability as the central thesis experiment, with clear gating and provenance.
3. Strengthen connection robustness, logging, and validation reports so failed runs explain why they failed.
4. Quantify each pipeline stage before attributing error to the robot body.
5. Keep GUI work tied to operator validation, not decorative polish.
6. Avoid broad refactors while the 4-servo validation ladder is still being closed.

See `plans.md` for the execution strategy and milestone definitions.

## Important Notes And Limits

- Do not modify `references/` or `tools/` unless intentionally changing protected lab/reference assets.
- Do not treat old reference scripts as the target architecture. Use them for math, conventions, and validation ideas.
- Do not claim hardware precision from mock-mode tests.
- `tracker_bridge` is retired compatibility infrastructure, not the normal live path.
- `system.local.yaml` may contain machine-specific values that differ from committed defaults.
- The repo contains real runtime data under `data/`; use the `Data` tab and migration tools cautiously rather than manually reorganizing datasets.
- Some docs in `docs/` are narrow trace/runbook documents from specific development phases. They are useful, but the README and `plans.md` should be treated as the top-level orientation and current strategy.
