# Servo Workflow Trace

This document traces the canonical servo GUI actions through the current controller, service, hardware, and persistence layers.

## Canonical Path

`SystemTab` / `ServosTab` -> controller -> `ServoService` -> hardware / calibration persistence

Primary files:

- `continuum_robot/gui/tabs/system_tab.py`
- `continuum_robot/gui/controllers/system_controller.py`
- `continuum_robot/gui/tabs/servos_tab.py`
- `continuum_robot/gui/controllers/servos_controller.py`
- `continuum_robot/servos/servo_service.py`
- `continuum_robot/servos/neutral_calibration_service.py`
- `continuum_robot/hardware/openrb_client.py`
- `continuum_robot/hardware/dxl_bus.py`

## Connect OpenRB + DYNAMIXEL

The connection button does not touch the bus directly from the widget.

- `SystemTab._connect_openrb()` reads the selected port from the combo box
- `SystemController.set_openrb_port(...)`
- `SystemController.connect_openrb()`
  - `OpenRbClient.connect(...)` validates the board serial device
  - `ServoService.connect(...)` opens the DYNAMIXEL transport
  - `ServoService.connect(...)` delegates to `DxlBus.connect(...)`

Relevant functions:

- `SystemTab._connect_openrb`
- `SystemController.connect_openrb`
- `OpenRbClient.connect`
- `ServoService.connect`
- `DxlBus.connect`

## Scan Servo IDs

- `ServosTab.scan_button` -> `ServosController.scan()`
- `ServosController.scan()` -> `ServoService.discover_one_servo(...)`
- `ServoService.discover_one_servo(...)`
  - tries the configured expected servo ID first
  - falls back to a conservative bounded scan only when needed
  - returns structured discovery + motion-readiness status
- `ServoService.scan_ids()` -> `DxlBus.scan_ids(...)` when the bounded scan path is used

The controller owns the displayed ID list. The widget just refreshes it.

## Read Telemetry

Telemetry refresh follows the same canonical path everywhere in the GUI.

- `ServosController.refresh()`
- `ServoService.read_telemetry(self.state.servo_ids)`
- `DxlBus.read_telemetry(...)`

Returned fields are normalized into `ServoTelemetry` and then into
`ServosViewState.telemetry`.

## Assign Servo ID

- `ServosTab._assign_id()`
- `ServosController.assign_servo_id(current_id, new_id)`
- `ServoService.assign_servo_id_safely(...)`
- `DxlBus.write_servo_id(...)`

After success the controller re-scans through the canonical scan path.

## Manual Jog

- `ServosTab._jog(...)`
- `ServosController.jog_servo(...)`
- `ServosController.fine_jog(...)` / `ServosController.coarse_jog(...)`
- `ServoService.jog_servo_directional(...)`
  - resolves tighten vs loosen through the configured tightening direction
- `ServoService.jog_servo(...)`
  - reads present telemetry
  - computes the goal relative to present position
  - validates saved calibrated bounds around neutral
  - writes goal positions through `DxlBus.write_goal_positions(...)`
  - re-reads telemetry
  - validates current / voltage / temperature / fault status

There is no widget-side bus write path.

## Tendon Displacement Command

- `ServosTab._apply_displacement()`
- `ServosController.set_tendon_displacements(...)`
- `ServosController.apply_displacement()`
- `ServosController._neutral_ticks_for_current_ids()`
  - loads the current neutral/calibration state already present in controller state
  - blocks if calibration is incompatible
- `ServoService.command_displacement(...)`
  - maps tendon displacements through `TendonDisplacementMapper`
  - validates calibrated bounds
  - writes positions through `DxlBus.write_goal_positions(...)`
  - re-reads telemetry
  - validates currents

Relevant functions:

- `ServosController.apply_displacement`
- `ServosController._neutral_ticks_for_current_ids`
- `ServoService.command_displacement`

## Calibration Artifact Load / Save / Use

The canonical persistence path is `NeutralCalibrationService`.

- capture neutral:
  - `ServosController.capture_neutral_setpoints()`
  - `ServoService.capture_and_save_neutral_setpoints(...)`
  - reads present positions from telemetry
  - persists the artifact immediately with capture metadata and neutral-centered software bounds
- save artifact:
  - `ServosController.save_neutral_setpoints()`
  - `ServoService.save_neutral_setpoints(...)`
  - `NeutralCalibrationService.save_neutral_setpoints(...)`
- load artifact:
  - `ServosController.load_neutral_setpoints()`
  - `ServoService.load_neutral_setpoints()`
  - `NeutralCalibrationService.load_neutral_setpoints()`
- GUI summary:
  - `ServosController._refresh_calibration_summary()`
  - `ServoService.get_calibration_summary()`
  - `NeutralCalibrationService.get_calibration_summary()`

## Experiment Preflight Dependency

The canonical experiment preflight does not read ad hoc GUI state. It reads:

- neutral calibration through `servo_service.load_neutral_setpoints()`
- registration presence through the configured registration path

Relevant functions:

- `ExperimentController.refresh`
- `evaluate_preflight` in `continuum_robot/gui/experiment_preflight.py`

## Dry-Validation Coverage

The dry tests that validate these paths live in:

- `tests/test_gui_controllers.py`
- `tests/test_servo_service.py`
- `tests/test_hardware_seams.py`

They cover:

- controller-to-service routing for jog and displacement
- fake hardware seam status propagation
- scan / telemetry / ID assignment through the canonical path
- calibration summary and compatibility use
