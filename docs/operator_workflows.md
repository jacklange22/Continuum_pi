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

Applies now in offline/mock mode and later with hardware.

1. Open the Experiment workspace.
2. Select `pivot_calibration`.
3. Run from a recorded file or live tracker samples.
4. Review RMSE, sample count, and rejected samples.
5. Save the generated pen-probe tip file.

Success criteria:

- tip vector file generated
- residual summary visible in GUI/CLI outputs
- result is reloadable from run history

## Workflow 4: Tracker Validation

Applies now.

1. Run tracker doctor / smoke / benchmark before registration.
2. Confirm tool visibility for `0A` and `0B`.
3. If needed, run `aurora_grid_accuracy` to measure tracker-only error.

Success criteria:

- tracker healthy before registration
- grid RMS metrics available when truth data exists

## Workflow 5: 4-Point Registration

Applies now.

1. Open the Registration tab.
2. Choose 4 landmarks from the configured candidate set using the top-view map or the landmark list.
3. Capture repeated `0B` samples for each selected landmark.
4. Mark each selected point complete.
5. Solve, review FRE / RMSE, and save the accepted registration.

Rules:

- exactly 4 unique landmarks
- only enabled landmarks may be selected
- solve remains blocked until all selected points have enough samples

Success criteria:

- accepted registration file saved
- GUI shows selected landmarks, measured centroids, FRE, and output path
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
2. one-servo OpenRB bring-up
3. startup calibration and pretension
4. pivot calibration
5. aurora grid accuracy
6. 4-point registration
7. repeatability dataset
