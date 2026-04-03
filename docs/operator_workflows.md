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

- raw capture CSV saved under `data/experiments/pivot/captures/`
- review run bundle saved under `data/experiments/pivot/runs/`
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
7. Review FRE / RMSE and the per-landmark residuals.
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
- tracking pipeline can use the saved registration on the next refresh

## Workflow 6: Repeatability Dataset

Applies now in dry-run and later with live hardware.

1. Open the Experiment workspace.
2. Select `repeatability_dataset`.
3. Review preflight, output path, and config summary.
4. Run in dry-run now, then live once neutral calibration and safety flows are complete.
5. Review repeatability RMS and per-target spread.

Target acceptance:

- logs commanded motion plus measured pose
- robot-frame metrics available when registration exists
- repeatability summary is comparable to the `< 1 mm` target

## Recommended Lab Order

1. tracker doctor / smoke
2. `Tracking` connect + validation
3. guided `0B` pivot calibration in `Tracking`
4. 4-point registration in `Registration`
5. confirm live robot-frame pose
6. one-servo OpenRB bring-up
7. startup calibration and pretension
8. repeatability dataset

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

`Tracker Legacy` remains available only for compatibility checks and deeper tracker-first diagnostics.
Normal operation should use `Tracking` plus `Registration`.
