# Continuum Robot Platform Plan

Last updated: 2026-04-25

This is the living execution plan for the continuum robot operator platform. It is deliberately more candid than the README: it should help choose the next slice of work, reject distractions, and keep the research validation ladder intact.

## Project Mission

Build a robust Raspberry Pi operator system that connects tracking, registration, servo control, pretensioning, experiment execution, logging, and analysis into one validated workflow for the continuum robot.

The platform should make the robot usable in the lab, but its deeper purpose is measurement. Every major subsystem should either reduce error or make error visible enough to explain.

## Thesis / Research Framing

The research problem is not only "move the continuum robot." The harder problem is establishing a repeatable, auditable experiment stack where tip-position error can be assigned to specific sources instead of hidden inside a fragile startup routine.

The system exists to improve on the prior architecture by:

- eliminating scattered script-only workflows
- replacing ambiguous tracker/registration assumptions with explicit artifacts
- making startup pretension state measurable and repeatable
- synchronizing servo commands, telemetry, tracker samples, and registration provenance
- producing datasets that can support repeatability, hysteresis, and model-learning claims

The near-term thesis target is the 4-servo single-segment robot. Two-segment scaling is a later milestone, not the current critical path.

## Core Hypothesis

If the platform validates each stage of the pipeline in order, records the state used by each run, and controls startup pretension consistently, then single-segment tip repeatability can be improved and measured credibly.

Success does not mean every source of error disappears. Success means the system can answer:

- Is the tracker stable before robot motion is involved?
- Is the `0B` probe tip calibration good enough for registration?
- Is the robot/body registration good enough to trust robot-frame tip measurements?
- Is the active runtime tip policy explicit and appropriate for the run?
- Are servo commands bounded by real calibration artifacts?
- Is pretension/startup state repeatable enough to compare runs?
- Does repeatability error depend on prior target/path history?
- Which stage dominates the observed error?

## Error Sources To Quantify

Primary error sources:

- Aurora tracking consistency, visibility, sample freshness, and transform validity.
- Tool-role aliasing and transform-chain mistakes between raw tracker IDs and runtime roles.
- `0B` pivot calibration error.
- Four-point base/robot registration error and landmark choice.
- Physical base movement after registration.
- runtime tip policy error: `coil_as_tip` is the current trusted robot-frame position path, while accepted runtime tip calibration artifacts remain lower-trust until validated.
- Servo position read/write behavior, operating mode, current/voltage/temperature telemetry, and stale readings.
- Neutral capture and calibrated safe bounds.
- Pretension threshold choice and accepted startup-state source.
- Tendon hysteresis and path dependence.
- Experiment timing between servo commands, settling, and tracker capture.

Secondary risks:

- machine-specific serial-port naming and permissions
- OpenRB firmware/power assumptions
- data organization drift
- old reference workflows reappearing as primary workflows
- GUI paths that look valid but skip validation gates

## Validation Ladder

The project should move up this ladder in order. A later rung is lower-trust if an earlier rung is missing or stale.

1. Tracker bring-up
   Done when the Python NDI backend connects, publishes `0A`/`0B`, reports valid transforms, meets practical freshness/timing thresholds, and diagnostics explain failures clearly.

2. Tracker grid consistency
   Done when `aurora_grid_accuracy` produces repeatable grid metrics and plots that bound tracker-only error before robot registration is involved.

3. `0B` pivot calibration
   Done when pivot calibration produces an accepted pen-probe tip file, validation metrics/plots are saved, and the GUI can show whether the active tip file is present and trusted.

4. Four-point registration
   Done when the operator can select four visible landmarks, capture repeated `0B` samples, solve, review FRE/RMSE, save intentionally, and produce `data/registrations/latest_registration.json`.

5. Runtime tip validation
   Done when `T_robot_tip = T_robot_aurora @ T_aurora_0A @ T_coil_tip` is live, registration and runtime-tip policy states are visible, and failures distinguish missing registration, stale tracker data, lower-trust calibration artifacts, debug overrides, and unavailable tip pose.

6. One-servo OpenRB/DYNAMIXEL bring-up
   Done when a single real servo can be discovered, identified, read for telemetry, assigned an ID if needed, captured for neutral/bounds, and jogged inside conservative limits with external power and fresh telemetry.

7. Four-servo startup calibration
   Done when all configured tendons/servos have compatible neutral ticks, safe min/max bounds, thresholds, tightening metadata, and the GUI blocks motion when calibration is missing or incompatible.

8. Pretension/startup-state characterization
   Done when pretension validation or manual pretension capture creates an accepted startup-state source for every configured servo, with current/travel evidence and clear lower-trust flags where needed.

9. Single-segment repeatability
   Done when `single_segment_repeatability` runs with live tracker, accepted registration, a thesis-trusted runtime tip policy outcome, accepted pretension source, neutral return on finalize, and outputs path-dependence/repeatability plots and summary metrics.

10. Hysteresis and modeling datasets
    Done when repeatability has established a trusted baseline and `collect_pose_command_dataset` can collect model-training data with provenance good enough to compare against repeatability/hysteresis observations.

11. Two-segment scaling
    Done only after the single-segment ladder is credible. Scaling should reuse the same validation gates rather than adding a parallel architecture.

## Current Status By Subsystem

Tracking:

- Active path is Python NDI through `TrackerBackendNDI`, `TrackingBackendRouter`, and `TrackingService`.
- Mock and diagnostics paths are developed.
- Raw Aurora IDs are mapped into logical roles (`10 -> 0A`, `11 -> 0B` in current Pi config).
- Live bench health is still configuration- and setup-dependent; do not infer tracker precision from mock tests.

Registration and runtime tip:

- The default operator path is now 4-point body registration using `config/registration.yaml`.
- Legacy SolidWorks/reference assets remain protected inputs, not the default runtime architecture.
- Runtime tip calibration is separated from `0B` pivot calibration and base registration; accepted calibration artifacts are currently lower-trust until their own validation is complete.
- Registration quality still depends on physical landmark choice, base stability, and declared tip policy.

Servo and hardware:

- OpenRB port validation and DYNAMIXEL SDK transport are implemented in current code.
- One-servo bring-up, scan, telemetry, ID assignment, mode writes, goal writes, neutral capture, and conservative jogging have service/GUI paths.
- Safety is much stronger than the early plan: telemetry freshness, current/voltage/temperature, operating mode, hardware errors, and safe bounds are represented.
- Real hardware behavior still must be treated as bench-validated only when actually run on the rig.

Pretension:

- Pretension validation, current/travel capture, automatic threshold stepping, manual startup-state capture, acceptance, and calibration-artifact integration exist.
- The hard research problem remains open: choosing and validating a pretension/startup policy that is repeatable and does not mask hysteresis.
- Pretension should be described as startup-state control/characterization, not true tendon-force measurement.

Experiments:

- The canonical path is `ExperimentRunner` plus built-in registry entries.
- Important current experiments include `aurora_grid_accuracy`, `pivot_validation`, `registration_validation`, `tracker_timing_validation`, `servo_tracker_sync_validation`, `pretension_validation`, `single_segment_repeatability`, and `collect_pose_command_dataset`.
- `single_segment_repeatability` is now the central live thesis experiment and includes strict preflight/provenance expectations.
- Older `repeatability_dataset` remains hidden compatibility infrastructure.

GUI:

- The GUI is broad and useful: `System`, `Tracking`, `Registration`, `Servos`, `Pretension`, `Experiment`, `Modeling`, and `Data`.
- The GUI should continue to prioritize validation status, preflight checks, artifact trust, and operator-visible failure reasons.
- Visualization is helpful but should not become a substitute for numeric validation.

Data and docs:

- Runtime data is now organized under `data/` with experiment, diagnostic, calibration, registration, runtime-tip, log, model, and modeling-result roots.
- The `Data` tab and migration tools exist because historical data layout has been messy.
- Some docs are phase-specific traces/runbooks; they should be maintained or retired as the top-level strategy changes.

## Top Risks

1. Pretension/startup inconsistency
   This is the highest research risk. If startup state varies, repeatability and hysteresis claims become hard to interpret.

2. Base or registration drift
   The system can compute robot-frame tip pose, but a moved base or stale registration invalidates downstream claims.

3. False confidence from GUI completeness
   A tab existing is not proof that the stage is validated on hardware. Docs and GUI labels should keep that distinction visible.

4. Tracker trust without grid/pivot validation
   Repeatability results are not meaningful if tracker consistency, `0B` pivot calibration, and runtime tip calibration are not current.

5. Servo telemetry gaps
   Missing or stale current/voltage/temperature data should block thesis-grade motion. Relaxing these gates for convenience will hide failures.

6. Scope drift into modeling or two-segment work
   Modeling and two-segment scaling depend on trusted single-segment data. They should not outrun validation.

7. Data provenance gaps
   Every run should say which registration, runtime tip calibration, neutral/bounds artifact, pretension source, config, and backend state it used.

## Current Priorities

Priority 1: Make repeatability runs trustworthy.

- Keep `single_segment_repeatability` as the central experiment.
- Ensure preflight blocks missing registration, runtime tip calibration, neutral/bounds, pretension, stale tracker data, and mock mode for thesis runs.
- Preserve lower-trust/debug overrides only when the output clearly marks the run as lower-trust.
- Keep finalize behavior returning toward the center/neutral state when configured.

Priority 2: Characterize pretension/startup state.

- Use `pretension_validation` and manual pretension capture to compare startup policies.
- Record current/travel traces and accepted source metadata.
- Decide whether the working policy is automatic threshold stepping, manual startup capture, or a hybrid.
- Treat "repeatable enough to support the next repeatability run" as the near-term goal.

Priority 3: Keep the validation ladder clean.

- Run tracker grid, pivot validation, registration validation, and timing checks before attributing error to robot mechanics.
- Make failure reasons specific enough to guide bench action.
- Avoid broad architecture changes unless they remove a known validation ambiguity.

Priority 4: Tighten provenance and logging.

- Ensure each experiment output carries registration/runtime-tip/calibration/pretension/backend context.
- Keep session logs discoverable.
- Keep `data/` migration and cleanup conservative.

Priority 5: Defer two-segment and extra UI polish.

- Two-segment support should stay compatible but not become the main branch of work yet.
- GUI additions should directly improve calibration, validation, preflight, or run interpretation.

## Near-Term Milestones

Milestone A: Trusted tracker and registration baseline

- Pass tracker doctor/smoke/benchmark on the Pi using the Python NDI backend.
- Produce or confirm current `0B` pivot calibration.
- Save accepted 4-point registration.
- Confirm live `T_robot_tip` is available without debug identity fallback.
- Done when repeatability preflight no longer blocks on tracker, pivot, registration, or runtime tip artifacts.

Milestone B: Repeatable startup state

- Capture compatible 4-servo neutral/bounds.
- Run pretension validation across several startup attempts.
- Accept a pretension source for all configured servos.
- Document the chosen startup policy and its limitations.
- Done when repeated startup attempts lead to comparable pretension artifact state and do not require ad hoc operator interpretation.

Milestone C: Thesis-grade single-segment repeatability run

- Run `single_segment_repeatability` with live tracker, accepted registration, accepted runtime tip calibration, accepted pretension source, and fresh servo telemetry.
- Save complete outputs: samples, metadata, summary, config snapshot, cluster/path-dependence plots, RMSE summary, and text summary.
- Compare against previous baseline when available.
- Done when the run can be reviewed later and the trust level is clear without reconstructing operator memory.

Milestone D: Error-source attribution

- Compare tracker grid error, pivot validation error, registration error, servo-tracker sync, pretension variation, and repeatability/path-dependence metrics.
- Identify the dominant current limitation.
- Done when the next engineering slice can be justified by measured evidence, not suspicion.

## Medium-Term Roadmap

1. Improve pretension policy based on measured repeatability/startup data.
2. Add or refine hysteresis-specific schedules only after the repeatability baseline is trusted.
3. Use `collect_pose_command_dataset` for model-learning datasets once provenance and startup state are reliable.
4. Improve analysis outputs that connect per-target repeatability, approach history, and servo telemetry.
5. Promote two-segment configuration only after single-segment validation gates are reusable and documented.

## Scope Guardrails

Do:

- Prefer current canonical services and controllers over new scripts.
- Add tests for changed validation, calibration, servo safety, or experiment behavior.
- Keep reference/tool assets read-only.
- Make trust levels explicit when using debug fallbacks.
- Optimize for runs that future-you can interpret months later.

Do not:

- Rebuild the retired tracker bridge as the main path unless the Python NDI path is proven unusable.
- Move primary workflows into `references/` or `tools/`.
- Treat mock-mode success as hardware validation.
- Add a second experiment framework.
- Start two-segment work before the 4-servo validation ladder is credible.
- Polish visualization while pretension, repeatability, or provenance is ambiguous.
- Delete or reorganize runtime data manually when the `Data` tab/migration tooling can represent it safely.

## Open Questions

- What pretension/startup policy gives the best repeatability without over-constraining or damaging the tendon system?
- How often must registration be repeated when the base or fixture is disturbed?
- What acceptance thresholds should be used for tracker grid, pivot calibration, registration FRE, and runtime tip calibration before thesis-grade runs?
- How much repeatability error is explained by path dependence versus servo/pretension startup state?
- Is current-aware motion useful for characterization, or should position control remain the default for thesis runs?
- Which outputs should become the standard advisor-facing summary for each major run?
- What is the minimum evidence needed before beginning two-segment validation?

## Decision Rule

When choosing what to work on next, prefer the task that removes the largest ambiguity from the validation ladder. If two tasks are close, choose the one that improves repeatability provenance or startup-state consistency before adding new capabilities.
