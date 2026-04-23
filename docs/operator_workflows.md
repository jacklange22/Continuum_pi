# Operator Workflows

## Goal

Keep the Pi GUI as the single operator surface for calibration, validation, registration, and experiments. These workflows reflect the current repo and the target lab sequence.

## Workflow 1: One-Servo Bring-Up

Applies now.

1. Launch the GUI.
2. In `System`, save the intended bring-up parameters:
   - `robot_1servo.yaml`
   - OpenRB port
   - baudrate
   - fine/coarse jog size
   - default pretension threshold
   - tightening direction
3. Connect external power on the OpenRB / DYNAMIXEL side.
4. Confirm tracker status in the System/Tracking surfaces.
5. Connect OpenRB and verify the prepared bridge status.
6. Scan one `XC330-M288-T` servo in `Servos`.
7. Verify model / firmware / position / current / voltage / temperature / error before any motion.
8. Use fine jog only first, then coarse jog if the first steps look correct.

Success criteria:

- Aurora backend is healthy or mock mode is explicitly enabled.
- OpenRB is connected and prepared.
- One servo is reachable and reports healthy telemetry.
- Fine and coarse jog both work through the canonical GUI / service path.

## Workflow 2: Startup Calibration And Pretension

Applies now for one-servo bring-up and scales later to 4 servos.

1. Put the robot or single tendon path in the intended neutral starting pose.
2. In `Servos`, use fine/coarse jog to reach the intended neutral position conservatively.
3. Save startup calibration for the active servo:
   - current Present Position becomes the neutral setpoint
   - conservative software min/max bounds are saved around that point
   - the pretension/current threshold is saved
4. Start the cautious pretension routine.
5. Let the routine stop on threshold, cancel it, or retry as needed.
6. Accept the pretension result only after reviewing the final current / position.

Success criteria:

- startup calibration artifact is saved with neutral, bounds, threshold, and direction
- pretension run stops safely on threshold, overcurrent, travel limit, timeout, cancel, or telemetry failure
- accepted result is visible in the calibration summary

## Workflow 3: Pivot Calibration

Applies now in `Tracking`.

1. Open `Tracking`.
2. Validate tracker health and confirm `0B` is tracked.
3. Start `0B` pivot collection.
4. Move the probe through a wide range of orientations while the sample counter rises.
5. Stop collection.
6. Solve pivot calibration.
7. Review RMSE, used/rejected sample counts, and the staged tip file.
8. Accept the tip file.

Success criteria:

- raw capture CSV saved under `data/pivot_calibration/captures/`
- review run bundle saved under `data/pivot_calibration/`
- staged tip file reviewed before acceptance
- accepted tip vector file generated

## Workflow 4: Tracker Validation

Applies now.

1. Run tracker doctor / smoke / benchmark before registration.
2. Confirm tool visibility for `0A` and `0B`.
3. If needed, run `aurora_grid_accuracy` to measure tracker-only error.

Success criteria:

- tracker healthy before registration
- grid RMS metrics available when truth data exists

## Workflow 5: 4-Point Registration

Applies now in `Registration`.

1. Move to `Registration`.
2. Confirm the accepted `0B` tip file is loaded.
3. Choose 4 landmarks from the configured candidate set using the top-view map or the landmark list.
4. Capture repeated `0B` samples for each selected landmark.
5. Mark each selected point complete.
6. Solve.
7. Review the validation summary before saving:
   - FRE / RMSE
   - max landmark residual
   - worst landmark residual
   - landmark geometry / spacing diagnostics
   - live transform-chain status
   - repeated-run comparison if prior registrations already exist
8. Save the accepted registration.

Rules:

- exactly 4 unique landmarks
- only enabled landmarks may be selected
- the configured candidate landmarks come from the protected lab model points, not placeholder coordinates
- solve remains blocked until all selected points have enough samples

Success criteria:

- accepted registration file saved
- GUI shows selected landmarks, measured centroids, FRE, residuals, and output path
- saved registration artifact shows the accepted `0B` tip provenance used during capture
- saved registration artifact shows the live-pose `T_coil_tip` source explicitly
- repeated registration-validation summary is written under `data/registrations/validation/`
- tracking pipeline can use the saved registration on the next refresh

## Workflow 5B: Runtime Tip Calibration

Applies now as an advanced workflow launched from `Registration`.

1. In `Registration`, click `Open Runtime Tip Calibration`.
2. Confirm the accepted `0B` tip file is loaded and the hat truth geometry is ready.
3. Begin the runtime tip calibration session.
4. Capture the configured hat points with the calibrated `0B` pen probe.
5. Mark each hat point complete after enough samples.
6. Collect stationary `0A` samples while the hat defines the fixed Tip frame.
7. Solve the calibration.
8. Review:
   - hat fit RMSE
   - max hat residual
   - per-point residuals and spreads
   - `0A` translation spread
   - `0A` rotation spread
   - runtime chain status
9. Save the accepted runtime tip calibration artifact.

Success criteria:

- canonical artifact saved under `data/runtime_tip_calibration/`
- artifact records `T_coil_tip`, `T_tip_aurora`, and `T_aurora_coil_avg`
- tracking reports that the live chain is using the accepted runtime tip calibration
- live robot-frame tip pose is no longer on identity fallback

## Workflow 6: Single-Segment Repeatability

Applies now in dry-run and later with live hardware.

1. Open the Experiment workspace.
2. Select `single_segment_repeatability`.
3. Review preflight, output path, run-validity thresholds, and config summary.
4. Run live only after neutral calibration, accepted pretension, accepted registration, and accepted runtime tip calibration are complete.
5. Review run-validity, repeatability RMS, per-target spread, and any baseline comparison deltas.

Target acceptance:

- logs commanded motion plus measured pose
- robot-frame metrics available only when the full live transform chain is trusted
- repeatability summary is comparable to the `< 1 mm` target

## Workflow 7: Motor Babble Modeling Dataset

Applies now in `Experiment`.

1. Open the Experiment workspace.
2. Select `collect_pose_command_dataset`.
3. Choose the dataset mode:
   - `Workspace Coverage` for first-pass forward-model data
   - `Hysteresis / Path Dependence` for ordered state-history datasets
   - `Repeatability Linked` for trusted startup-state comparison blocks
4. Review the collection summary:
   - runtime tip mode
   - pretension source
   - preflight trust summary
   - planned commands/captures
   - output destination
5. Run live only after accepted registration, accepted runtime tip, accepted pretension, and healthy tracker/servo state are confirmed.
6. After the run, review:
   - accepted vs rejected captures
   - workspace coverage plot
   - command distribution plot
   - export JSONL / optional legacy DAT
   - saved provenance in `summary.json` and `modeling_dataset_summary.txt`

Target acceptance:

- ordered command history is preserved
- accepted samples use fresh valid tracker data only
- robot-frame tip pose includes tangent/orientation when trusted
- saved output is ready for later offline ANN / state-aware model training

## Recommended Lab Order

1. tracker doctor / smoke
2. `Tracking` connect + validation
3. guided `0B` pivot calibration in `Tracking`
4. 4-point registration in `Registration`
5. runtime tip calibration from `Registration`
6. confirm live robot-frame pose
7. one-servo OpenRB bring-up
8. startup calibration and pretension
9. single-segment repeatability
10. Motor Babble modeling dataset collection

## Tracker Verdicts

Strict validation pass:

- all configured tracker thresholds pass

Operational with warning:

- tracker frames, tool visibility, and rigid transforms are good enough to use
- one strict target, usually FPS, is below the configured threshold
- this is a warning state, not the same as a hard tracker failure

Hard failure:

- tracker is not connected, no frames arrive, required tools are missing, transforms are invalid, or data is stale
- do not proceed until the tracker returns to an operational state

## Legacy Surface

The Python legacy-bridge compatibility modules remain available only for explicit bridge fallback/debug work.
Normal operation should use the current `Tracking` plus `Registration` workflows.
