# One-Servo Servo MVP Runbook

This runbook is the narrow OpenRB-150 / XC330-M288-T bring-up path for the current branch.

Scope:

- one servo first
- OpenRB serial bridge validation
- conservative discovery and telemetry readback
- maintenance-only ID assignment
- neutral capture with persisted software bounds
- tiny bounded jog
- threshold-based startup pretension

Out of scope:

- tracker, pivot calibration, and registration workflows
- broad GUI redesign
- full 4-servo or 8-servo orchestration
- experiment scheduling
- tendon displacement commands during one-servo bring-up
- the broader startup-calibration draft UI in one-servo mode

## Hardware Assumptions

- Board: `OpenRB-150`
- Servo: `XC330-M288-T`
- Real motion requires DYNAMIXEL-side power. USB-only must not be treated as a motion-ready condition.
- When facing the servo, clockwise rotation tightens. Software keeps the tightening direction explicit through `tightening_rotation_by_servo`.
- Present current is used as a practical startup threshold signal. It is not treated as true tendon tension.

## Canonical Files

- `continuum_robot/hardware/openrb_client.py`
- `continuum_robot/hardware/dxl_bus.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/servos/neutral_calibration_service.py`
- `continuum_robot/gui/controllers/system_controller.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `continuum_robot/gui/tabs/system_tab.py`
- `continuum_robot/gui/tabs/servos_tab.py`

## Manual Bench Steps

### Phase 0: Dry / No Motion

1. Set `robot_config` to `robot_1servo.yaml` in `config/system.local.yaml` or through the System tab.
2. Confirm `openrb_port`, expected servo ID, jog ticks, neutral offsets, software margin, pretension threshold, and telemetry freshness timeout.
3. Connect the OpenRB USB cable.
4. Connect DYNAMIXEL-side external power only when the bench setup is mechanically safe.
5. Use `Connect OpenRB`.
6. Use `Refresh Readiness`.
7. Confirm:
   - OpenRB is connected and prepared.
   - the DYNAMIXEL bus is connected.
   - the expected servo ID responds, or the system clearly reports that it does not.
   - model number, firmware version, ID, position, current, voltage, temperature, and hardware error status are readable.
   - `motion ready` remains false until calibrated bounds exist. Before neutral capture, the expected block is missing saved safe bounds.
8. Do not jog if readiness reports stale telemetry, missing voltage, a disallowed operating mode, high temperature, or any hardware error.

### Phase 1: Neutral Capture

1. Put the single servo in a physically safe unloaded state.
2. Use `Capture Neutral`.
3. Confirm the artifact is written to `config/neutral_setpoints.json`.
4. Confirm the artifact contains:
   - timestamp
   - robot mode and robot config metadata
   - servo ID
   - capture source
   - neutral tick
   - software min/max bounds around neutral
5. Confirm the Servos tab motion-blocking summary clears the missing-bounds condition before jog is enabled.

### Phase 2: Tiny Jog Validation

1. Keep the mechanism clear and unloaded enough for small movement checks.
2. Use `Tighten Fine` and `Loosen Fine` first.
3. Confirm physical direction matches the configured tightening direction.
4. Use `Tighten Coarse` and `Loosen Coarse` only after fine jogs look correct.
5. Confirm jogs stop at saved software bounds and reject commands outside them.
6. Confirm motion is blocked when:
   - telemetry is stale
   - telemetry is missing
   - voltage is below the configured minimum
   - temperature is at or above the configured maximum
   - operating mode is not allowed
   - hardware error status is non-zero
7. In one-servo mode, the tab should not offer tendon-displacement or startup-draft controls as part of this workflow.

### Phase 3: Pretension / Startup Calibration

1. Start from neutral or another clearly safe state inside the saved bounds.
2. Use `Start Pretension`.
3. Confirm the routine only steps in the configured tightening direction and only in tiny increments.
4. Confirm the routine stops on:
   - threshold reached
   - stale or missing telemetry
   - hardware error
   - unsafe voltage
   - unsafe temperature
   - safe travel limit reached
   - timeout
5. Confirm the pretension result is written back into the calibration artifact.
6. Use `Accept Result` only after bench review.

## Mock-Tested In This Branch

- safety blocking for missing telemetry, stale telemetry, hardware error, unsafe voltage, and unsafe temperature
- calibrated software bounds enforcement
- directional jog sign mapping from configured tightening direction
- persisted neutral capture metadata
- maintenance-only ID assignment routing
- threshold-based pretension stopping and failure modes

## Still Requires Bench Confirmation

- OpenRB bridge behavior on the target Pi and board firmware
- actual XC330 present-voltage/current behavior on the lab wiring
- real tightening direction for the installed tendon wrap
- safe pretension threshold for the physical mechanism
- acceptable software margin around neutral on the real build
- whether the external power path is stable under repeated connect / jog / pretension cycles

## Known Limitations

- bus reachability is inferred from servo response, not from a dedicated OpenRB power-status API
- GUI coverage for this pass is intentionally minimal
- full multi-servo coordination is not implemented here
- GUI tests were not runnable in this workspace because `PySide6` is not installed

## Preconditions Before 4-Servo Work

- one-servo discovery and readiness are repeatable on the bench
- neutral capture and persisted bounds are trusted
- tighten/loosen direction is bench-verified
- pretension threshold behavior is stable and mechanically safe
- no unexplained hardware errors or voltage dropouts appear during repeated one-servo runs
