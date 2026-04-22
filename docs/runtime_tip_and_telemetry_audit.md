# Runtime Tip And Servo Telemetry Audit

## Runtime tip truth

The live tip display uses this chain when registration is loaded:

`T_robot_tip = T_robot_aurora @ T_aurora_0A @ T_coil_tip`

Where:

- `T_robot_aurora` comes from the accepted registration artifact.
- `T_aurora_0A` comes from the live `0A` coil tracker sample.
- `T_coil_tip` comes from the active runtime tip mode.

Runtime tip mode is stored in `TrackingService._runtime_tip_mode` and exposed in every tracking snapshot as:

- `runtime_tip_mode`
- `runtime_tip_trust_level`
- `runtime_tip_mode_message`
- `runtime_tip_calibration_state`
- `runtime_tip_identity_fallback`

Supported operator modes:

- `latest_accepted`: use the accepted runtime tip artifact when present.
- `quick_4_point`: use the saved quick override artifact when present.
- `coil_as_tip`: use identity `T_coil_tip`; the `0A` coil pose itself is shown directly as the tip.

The Tracking scene only draws the tip glyph when:

- registration is loaded, and
- `T_robot_tip` was computed successfully.

If the chain is not available, the raw `0A` and `0B` tool glyphs can still be shown, but the derived tip glyph stays hidden.

## Servo telemetry truth

Current low-level read behavior:

- `read_live_telemetry()` performs 7 separate register reads per servo:
  - operating mode
  - torque enable
  - present position
  - present current
  - present input voltage
  - present temperature
  - hardware error status
- `read_telemetry()` performs 14 separate register reads per servo:
  - the 7 live fields above
  - servo ID
  - model number
  - firmware version
  - current limit
  - max position limit
  - min position limit
  - bus watchdog

Current GUI policy:

- GUI timer target comes from `poll_rate_hz`.
- Only the active tab refreshes hardware state.
- Servos tab aims for selected-servo updates every GUI tick and all-servo refresh every fourth tick.
- System tab now throttles automatic servo-summary reads and avoids background scan pings on every GUI tick.
- Manual readiness refresh remains the place that forces a discovery-style scan.

Current practical throughput note:

- The current config still defaults to `57600` baud.
- OpenRB-150 and XC330 support `1 Mbps`.
- With the current per-servo transaction count, baudrate is the first thing to remeasure before chasing GUI timer changes.

## Real measurement commands

Measure direct servo telemetry throughput on the current rig:

```bash
python3 -m continuum_robot.servos.telemetry_diagnostics --profile both --iterations 200
```

Measure a smaller servo set:

```bash
python3 -m continuum_robot.servos.telemetry_diagnostics --profile live --iterations 200 --servo-ids 1 2
```

Saved outputs are written under:

`data/diagnostics/servo_telemetry/`

For end-to-end timing alignment between servo telemetry and tracker timing, use the existing canonical experiment:

- `servo_tracker_sync_validation`
