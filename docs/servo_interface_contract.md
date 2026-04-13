# Servo Interface Contract

This document records the current OpenRB150 + DYNAMIXEL interface assumptions used by the repo. It is grounded in the current implementation and should be reviewed before bench bring-up.

## Targeted Hardware

- Controller board: OpenRB-150
- One-servo bring-up is the first-class validation workflow for this phase
- Primary validation target: 4-servo robot
- Primary servo target: `XC330-M288-T`
- Supported family assumption in code: X-series style DYNAMIXEL protocol 2.0 devices

## Canonical Transport Path

The canonical runtime path is:

`SystemController` -> `OpenRbClient` + `ServoService` -> `DxlBus`

Primary files:

- `continuum_robot/gui/controllers/system_controller.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/hardware/openrb_client.py`
- `continuum_robot/hardware/dxl_bus.py`
- `continuum_robot/app/bootstrap.py`

## OpenRB Assumptions

Current code assumes:

- OpenRB appears as a normal serial device on the Pi host
- the board is running Robotis `usb_to_dynamixel` bridge firmware or equivalent serial pass-through behavior
- OpenRB DYNAMIXEL-side power is gated and not directly observable over the bridge path
- external power is the expected motion-testing path; USB-only should not be assumed safe for real movement
- `OpenRbClient` validates the port, but does not keep it open
- `DxlBus` owns the serial port while connected to the DYNAMIXEL bus

Relevant implementation:

- `OpenRbClient.connect(...)`
- `OpenRbClient.prepare_for_dynamixel_use(...)`
- `DxlBus.connect(...)`

## Servo Operating Mode And Register Assumptions

The current command path assumes position-based goal writes are valid.

- experiments and manual GUI motion use goal position writes
- current feedback is treated as a practical safety / pretension signal, not ideal torque truth
- the expected initial operating mode is position mode (`Operating Mode(11) = 3`)
- motion is blocked when the live operating mode is not in the configured allow-list
- EEPROM writes such as servo ID changes must happen with `Torque Enable(64) = 0`

This means the bench setup must confirm the connected servos are already in a position-control-compatible mode before jog or pretension tests.

## Required Telemetry

The higher-level services currently require at minimum:

- Operating Mode
- Min / Max Position Limit
- Present Position
- Present Current
- Present Input Voltage
- Present Temperature
- Hardware Error Status

`ServoTelemetry` in `continuum_robot/hardware/dxl_bus.py` is the canonical readback shape used above the hardware seam.

## Current Control-Table Assumptions

`DxlBusConfig.control_table` currently defaults to:

- `model_number`: `0`
- `firmware_version`: `6`
- `servo_id`: `7`
- `operating_mode`: `11`
- `current_limit`: `38`
- `max_position_limit`: `48`
- `min_position_limit`: `52`
- `torque_enable`: `64`
- `hardware_error_status`: `70`
- `bus_watchdog`: `98`
- `profile_acceleration`: `108`
- `profile_velocity`: `112`
- `goal_position`: `116`
- `present_current`: `126`
- `present_position`: `132`
- `present_input_voltage`: `144`
- `present_temperature`: `146`

These values are configurable through `settings.serial.dynamixel_settings`.

## Scale / Unit Assumptions

Current config defaults:

- `protocol_version`: `2.0`
- `positive_tick_rotation`: `ccw`
- `expected_operating_mode`: `3`
- `allowed_operating_modes`: `[3]`
- `voltage_scale_mv_per_unit`: `100.0`
- `current_scale_ma_per_unit`: `1.0`
- `auto_torque_enable_on_write`: `true`
- `torque_disable_for_eeprom_write`: `true`
- `require_current_for_motion`: `true`
- `require_voltage_for_motion`: `true`
- `require_temperature_for_motion`: `true`
- `require_fresh_telemetry_for_motion`: `true`

Interpretation:

- voltage register values are multiplied by `100.0` to report millivolts
- current register values are multiplied by `1.0` to report milliamps
- goal positions are integer DYNAMIXEL ticks
- tightening direction is stored separately per servo as `cw` / `ccw` and is resolved against `positive_tick_rotation`

## Authoritative Config Fields

The main authoritative config inputs are:

- `config/system.yaml`
- `config/system.local.yaml`
- `config/robot_1servo.yaml`
- `config/robot_4servo.yaml`
- `config/safety.yaml`

Loaded through:

- `continuum_robot/config/config_loader.py`
- `continuum_robot/config/schemas.py`

Most relevant fields:

- `serial.openrb_port`
- `serial.baudrate`
- `serial.openrb_settings`
- `serial.dynamixel_settings`
- `robot.servo_ids`
- `robot.tendon_to_servo`
- `robot.tightening_rotation_by_servo`
- `robot.spool_diameter_cm`
- `robot.ticks_per_revolution`
- `safety.position_min_offset_ticks`
- `safety.position_max_offset_ticks`
- `safety.max_current_ma`
- `safety.default_pretension_current_threshold_ma`
- `safety.fine_jog_step_ticks`
- `safety.coarse_jog_step_ticks`
- `safety.software_position_margin_ticks`
- `serial.openrb_settings.require_external_power_for_motion`
- `serial.dynamixel_settings.allowed_operating_modes`

## Canonical Higher-Level Service Surface

All live servo motion and telemetry should flow through `ServoService`.

Public methods currently relied on by the GUI and experiments:

- `connect(...)`
- `disconnect()`
- `scan_ids(...)`
- `discover_one_servo(...)`
- `assign_servo_id(...)`
- `assign_servo_id_safely(...)`
- `read_telemetry(...)`
- `assess_motion(...)`
- `capture_neutral_setpoints(...)`
- `capture_and_save_neutral_setpoints(...)`
- `save_neutral_setpoints(...)`
- `load_neutral_setpoints(...)`
- `load_calibration_artifact()`
- `get_calibration_summary()`
- `save_startup_calibration(...)`
- `jog_servo(...)`
- `jog_servo_directional(...)`
- `command_displacement(...)`
- `run_pretension_routine(...)`
- `accept_pretension_result(...)`

The GUI should not talk to `DxlBus` directly.

## Calibration Artifact Contract

The canonical servo calibration artifact lives at the configured
`neutral_setpoints_path`, currently defaulting to:

- `config/neutral_setpoints.json`

It is owned by:

- `continuum_robot/servos/neutral_calibration_service.py`

It stores:

- neutral setpoint
- safe min/max bound
- pretension/current threshold
- tightening direction
- hardware min/max bounds when known
- last pretension result, final position, and acceptance state
- per-servo validity and timestamp
- robot compatibility metadata

## GUI Surfaces That Depend On This Contract

- `SystemController` / `SystemTab`
  - OpenRB validation
  - DYNAMIXEL connection status
- `ServosController` / `ServosTab`
  - scan IDs
  - assign ID
  - read telemetry
  - jog
  - tendon displacement commands
  - calibration summary

## Hardware-Dependent / Unproven Items

These assumptions are still not proven in dry mode:

- exact OpenRB firmware behavior on the real board
- actual XC330-M288-T operating mode on startup
- current and voltage scale agreement with the real servo registers
- whether the configured minimum motion voltage is appropriate for the real wiring path
- control-table address agreement for the exact servo model/firmware in use
- DYNAMIXEL bus behavior under real wiring/power conditions
- real safe current limits for tendon protection
- whether the configured tightening direction matches the actual tendon winding for each servo
- whether the conservative software safety margin should be widened or narrowed on the bench

Until that bench validation is done, this document is an implementation contract, not final hardware truth.
