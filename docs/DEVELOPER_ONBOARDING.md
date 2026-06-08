# Developer Onboarding

**Start here if you are the new developer inheriting this project.** This is the
human-facing counterpart to `AGENTS.md` (which is written for AI assistants). It
gets you from "fresh clone" to "I can make and verify a change" and points you at
the right deep-dive docs in the right order.

---

## 1. What this project is (in three sentences)

A Raspberry-Pi operator stack for a **tendon-driven continuum robot**: NDI Aurora
EM tracking → robot/body registration → runtime tip pose, and OpenRB-150 +
DYNAMIXEL XC330 servos → calibration / pretension / motion. It is a **research and
thesis platform**, not just a control GUI — its job is to *quantify and reduce
error* across the whole pipeline (tracker, transforms, registration, servo
positioning, pretension, hysteresis, repeatability). The headline experiments are
single-segment repeatability and pretension/startup-state characterization.

Read the top of `README.md` and `plans.md` for the authoritative orientation —
those two are the canonical top-level docs. Everything in `docs/` else is a
narrower trace/runbook from a specific phase.

## 2. Read order (first day)

1. `README.md` — system at a glance, transform conventions, how to run, current
   status, limits. **The single most important doc.**
2. `docs/architecture.md` — subsystem boundaries and the rules (e.g. *the GUI must
   not bypass `ServoService`*).
3. This file — setup, the dev loop, how to extend, tech debt.
4. `docs/operator_workflows.md` — what the operator actually does in the GUI
   (registration → pretension → experiment), so you understand what the code serves.
5. `docs/testing_protocol.md` — how tests are organized and the markers.
6. Then, by subsystem as needed: `docs/servo_interface_contract.md`,
   `docs/registration_trace.md`, `docs/tracker_mvp_workflow.md`,
   `docs/two_segment_bench_day_quickref.md`, `docs/hardware_day_runbook.md`.

## 3. First-time setup

```bash
# From the repo root:
scripts/bootstrap.sh          # creates .venv, installs the package editable + dev deps,
                              # validates Python 3.10+ (3.11+ recommended)

# Sanity check — everything runs in MOCK mode with no hardware attached:
.venv/bin/python -m pytest -q tests -m "not gui"   # the CI suite (no Qt/hardware)
```

There is **no hardware required for development**. The app, tracking, servos, and
experiments all have mock backends (`mock_mode: true` in `config/system.yaml`).
Never claim hardware precision from mock-mode results (see `README.md` → *Important
Notes And Limits*).

### Run the GUI (mock mode)

```bash
.venv/bin/python scripts/run_gui.py
```

Entry chain: `scripts/run_gui.py` → `continuum_robot/app/main.py:main()` →
`app/bootstrap.py` (builds the service graph from config) →
`app/service_registry.py` (the canonical wiring seam) → `AppWindow`.

## 4. The dev loop

```bash
# Fast feedback while editing (sub-second):
scripts/run_tests.sh quick           # core config/servo/policy tests

# Before committing servo/pretension/experiment changes:
scripts/run_tests.sh hardware-safe   # the mock-safe experiment + servo suite

# Before any two-segment bench day:
scripts/run_tests.sh two-segment

# GUI tests (need an offscreen Qt platform; slower):
scripts/run_tests.sh gui

# The full suite (~5 min):
.venv/bin/python -m pytest tests -q
```

Other conventions:

- **Compile check** after large edits: `.venv/bin/python -m compileall -q continuum_robot scripts`.
- The repo uses **squash-merge PRs** (commits on `main` carry a `(#N)` suffix). Open
  a PR, let CI run `pytest -m "not gui"`, squash-merge.
- **Protected, do-not-touch:** `references/` and `tools/` are lab/reference assets.
  Don't refactor them; mine them for math and conventions only.
- `system.local.yaml` holds machine-specific overrides and may differ from the
  committed `system.yaml`.

## 5. Mental model: services → controllers → experiments

```
config/*.yaml ──> app/bootstrap ──> Services (the ONLY seam to hardware/state)
                                       ServoService, TrackingService,
                                       RegistrationService, NeutralCalibrationService
                                          │
                          ┌───────────────┴────────────────┐
                     GUI controllers                  ExperimentRunner
                  (gui/controllers/*)            (experiments/framework + registry)
                          │                                │
                     GUI tabs/pages                  built-in experiments
                  (gui/tabs, gui/widgets)           (experiments/builtins.py + friends)
```

Rules that matter:

- **Controllers and experiments go through Services.** Never reach into the
  DYNAMIXEL bus or tracker backend directly from the GUI or an experiment.
- **Experiments are pure-ish, registered units.** Each has a config dataclass
  (`from_dict`), a `precheck`/`execute`/`write_outputs` lifecycle, and is registered
  in `experiments/builtins.py:register_builtin_experiments`.
- **Data carries trust.** Every run records `run_trust_mode`,
  `valid_for_thesis_repeatability`, `valid_for_model_training`, and
  `data_quality_warnings`. Un-trustworthy data is *gated out of thesis claims at
  capture time* — see `data/build_thesis_evidence_index.py` and the runtime-tip
  policy (`tracking/runtime_tip_policy.py`). This provenance system is one of the
  most important and unusual parts of the codebase; respect it.

## 6. How to add a new experiment (the concrete recipe)

The most recently added experiment, `neutral_setpoint_drift_validation`, is the
cleanest worked example — copy its shape. Steps:

1. **Analysis + experiment class** — create
   `continuum_robot/experiments/<name>.py` with:
   - a `@dataclass <Name>Config` + `from_dict(payload)` (clamp/normalize inputs here),
   - a `<Name>Experiment(BaseExperiment)` with `name`, `description`,
     `hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)`,
     and `precheck` / `execute` / `write_outputs`,
   - a `register_<name>_experiment(registry)` that calls `registry.register(...)`
     with `title`, `category`, `tags`, `default_config_path`, `factory=...from_dict`.
   - Keep the heavy analysis in pure functions (no I/O) so unit tests stay fast.
2. **Output writer** — `continuum_robot/experiments/<name>_outputs.py` that writes
   the thesis figures + a `debug.json`, using the helpers in
   `experiments/plotting.py` (`create_figure`, `style_axes`, `save_figure`,
   `add_metric_box`, `color`, `legend`). Wrap each figure write so a failure drops a
   placeholder rather than crashing (`_write_plot_placeholder`).
3. **Register it** — import and call `register_<name>_experiment(registry)` inside
   `experiments/builtins.py:register_builtin_experiments`.
4. **GUI page** — add a page class in `gui/widgets/experiment_pages.py` (extend
   `ExperimentPageBase`, or `_ValidationSelectionPage` for offline multi-run
   analyses) and route it in `build_experiment_page`'s factory dict.
5. **Surface it in the dropdown** — this is the step people miss. The Experiments
   tab dropdown is filtered by three explicit lists in
   `gui/controllers/experiment_controller.py`:
   `MODE_EXPERIMENT_VISIBILITY` (per operating mode), `MANUAL_REFRESH_EXPERIMENTS`
   (if the page uses manual refresh), and `preferred_order` (sort). Add the name to
   the relevant ones.
6. **Preflight branch** — add an `elif experiment_name == "<name>":` block in
   `gui/experiment_preflight.py` so the Run button enables with a sensible check.
7. **Example YAML** — `config/experiment_<name>.example.yaml` matching
   `default_config_path`, with operator-facing comments on each knob.
8. **Tests** — `tests/test_<name>.py`: config normalization, the pure analysis
   functions, output-file generation, and registry presence
   (`register_builtin_experiments` includes it).

See git history for the `neutral_setpoint_drift_validation` and
`full_factorial_grid` additions as end-to-end examples.

## 7. Pretension: which mode for what

Pretension is the most actively-iterated subsystem and has several start modes.
For the **single-segment automatic workflow**, the canonical default is
`soft_release_to_zero_current`. Decision table:

| Start mode | When to use |
|---|---|
| `soft_release_to_zero_current` | **Default.** Release each tendon by current until slack, then take up. Proves each tendon is genuinely slack rather than trusting a position endpoint. |
| `current_position` | Start from wherever the servos are — no release. Fast, for quick re-runs from a known state. |
| `release_200_from_current` | Quick rough backoff (200 ticks). Cheap but doesn't prove slack. |
| `manual_startup_artifact` | Use the operator's saved hand-tensioned startup state. |
| `full_release_4095` | **Legacy.** Walks to the safe-max tick. Sits at the position-wrap edge where the PID hunts; kept only for backward compat. Avoid for new runs. |

Key pretension concepts the code relies on (all in `experiments/builtins.py` and
`config/experiment_pretension_validation.example.yaml`):

- **Signed current is the tension proxy** (no load cell). On this rig, `tension_ma
  = max(0, -signed_current_ma)`; operator scale is `-15 mA` light, `-30 mA` tight,
  `-50 mA` a lot, `> -5 mA` slack.
- **Holding current ≠ motion current.** Tension decisions read a *settled* burst
  average (`post_move_settle_s` + `holding_current_burst_count`), not an
  instantaneous read taken while the motor is still driving. This was a real bug
  source — an in-motion read says `+17 mA` where the settled holding current is
  `-30 mA`.
- **Take-up targets balanced holding tension, then equalizes (tighten-only).** The
  tension-equalization pass drives all four servos to within
  `equalize_tensions_tolerance_ma` of each other by only *tightening* the low ones —
  never releasing — because spine friction makes balanced tendon force more
  repeatable than tip XY.
- **Repeatability verdict.** A 5-repeat "one-rig proof" grades consistency
  high/medium/low/failed (`_compute_consistency_verdict`).

## 8. Known tech debt / cleanup opportunities

Honest list for the next person (none are blocking; all are safe-to-defer):

- **`experiments/builtins.py` is ~13k lines.** It registers ~40 experiments and
  carries most pretension orchestration. A future refactor should extract the
  pretension routines and the per-experiment registration into smaller modules.
  Do this only when the validation ladder is closed (see `README.md` priority 6:
  *avoid broad refactors while the 4-servo ladder is being closed*).
- **Duplicate private helpers** across modules: `_utc_now` / `_utc_now_iso`
  (several definitions, and they **diverge** — some use `.isoformat()`, some
  `.strftime("%Y-%m-%dT%H:%M:%SZ")`), plus `_summarize_scalars`, `_resolve_path`,
  `_display_path`, `_as_float`. Consolidating into a `utils/` module is good, but
  **the timestamp ones differ in output format** — unifying them changes on-disk
  artifact timestamps, so do it deliberately with a format decision, not a blind
  merge.
- **No `pyflakes`/lint in CI.** Unused imports have crept in before. Consider adding
  `ruff` (fast, catches unused imports + obvious bugs) to the dev deps and CI.
- **GUI controllers are large** (`experiment_controller.py` ~2.2k lines). State flow
  tab → controller → service is implicit; a short state-flow note would help.

## 9. Current state awareness (read before you trust anything)

- **This has been a multi-developer, multi-worktree checkout.** Before assuming the
  working tree is clean, run `git status`. There may be uncommitted parallel work.
- Per `README.md` → *Current Project Status*: much of the system is **validated
  infrastructure with little-to-no live bench data**. The pretension/repeatability
  code is built and tested in mock mode; the actual hardware datasets it is designed
  to produce are the next milestone. Treat precision numbers as targets, not results,
  until real bench data exists.

---

*If something here is wrong or stale, fix it — this doc is meant to be the living
"start here" for whoever holds the project next.*
