# Two-Segment Bench Day Quick Reference

This is the operator's one-page cheat sheet for a bench-day two-segment session.
It complements (does not replace) `docs/operator_workflows.md` and
`docs/hardware_day_runbook.md`.

## Pre-launch (10 min)

Edit `config/system.local.yaml` (or `config/system.yaml`):

```yaml
mock_mode: false                # IMPORTANT: real-hardware run
baudrate: 1000000               # ONLY after every servo is reflashed at 1 Mbps
openrb_port: /dev/ttyACM0       # or your bench's OpenRB device
aurora_port: /dev/ttyUSB0       # or your bench's Aurora device
```

Confirm the stack in the GUI System page every dual-segment session. The same
mapping is persisted in `config/robot_8servo.yaml` / local overrides:

```yaml
physical_assembly:
  bottom_segment: "segment_a"   # which fixed segment is at the bottom of the stack
  top_segment: "segment_b"      # the other one
  confirmed_by_operator: true   # set by GUI when you confirm the session stack
  lower_tick_means_tension: true
  notes: ""                     # OPTIONAL free-form context (e.g., "top mount loose"); appears in every run summary
```

The `notes` field is operator-grade context that travels with every run. Use
it to record anything the bench operator should know — known hardware quirks,
intentional pretension state, transient mounting issues. The string appears
in the GUI mode summary, the Data tab Two-Segment Foundation row, the
session log, and every run's foundation metadata.

Sanity checks:
- DYNAMIXEL Wizard reads all 8 servos at the configured baud.
- Aurora reports 0A, 0B (and 0C if you're doing two-coil).
- GUI launches, shows `dual_segment` mode, mode summary chip shows the bottom/top
  assignment you set above and `operator_confirmed=true`.

If any preflight check is yellow or red, fix it before proceeding. Yellow on
`Bus Baud` at 57600 is informational only for single-segment/debug work.
Trusted all-8 two-segment collection is designed around 1 Mbps after every
servo is reflashed and verified at that baud.

## Stage 1: Startup validation (~5 min)

Run `two_segment_startup_validation` from the GUI. Capture each stage:

| Stage | What to do |
|---|---|
| `baseline` | Power-on snapshot, no tension applied. |
| `bottom_pretensioned` | Manually pretension the bottom (proximal) segment, capture. |
| `top_pretensioned` | Manually pretension the top (distal) segment, capture. |
| `bottom_recheck` | Top tendons pass through bottom; re-check bottom didn't drift. |
| `final_accept` | Save the all-8 startup artifact. |

What to watch in the GUI:
- Each captured stage adds a row to the staged positions table.
- The `Load Proxy By Stage` figure should show modest baseline-subtracted current
  on pretensioned segments; sustained >800 mA on any servo is a stop-now signal.
- `Stage Drift Report` should show bounded drift between `bottom_pretensioned`
  and `bottom_recheck` — large drift means top pretensioning is yanking the
  bottom segment too hard.

After `final_accept`, confirm `data/.../two_segment_startup_artifact_metadata.json`
shows `manual_two_segment_startup: true` and the correct bottom/top keys.

## Stage 2: First two-segment motor babble (~10 min)

Run `two_segment_collect_pose_command_dataset`. Recommended first-run config:

```yaml
schedule_type: "single_axis_micro"     # cardinal bottom + cardinal top sweeps
max_segment_displacement_cm: 0.25      # CONSERVATIVE - ramp up later
max_tick_delta_from_startup: 320       # safety cap on tick travel
samples_per_pattern: 1
capture_repeats: 1
allow_servo_only_test_run: false       # we want trusted data
run_trust_mode: "thesis_trusted"
long_run_recovery_enabled: true        # drop dropped samples, continue
drop_sample_on_transport_error: true
physical_assembly_confirmed_by_operator: true
```

Set tracker roles in the GUI collect-pose page:
- `distal_tip`: required for ANN training; usually 0A for the distal/top coil.
- `intermediate_segment`: optional; use 0C if installed for two-coil labels.
- `debug_tool`: optional, never a training requirement.

Distal-only datasets are valid for ANN distal-tip mapping when `distal_tip` XYZ
is present. They are marked `distal_only=true`; missing orientation and missing
intermediate labels do not block distal XYZ training.

What to watch:
- The Data tab `Long-Run Health` row should show
  `stop=scheduled_repeat_count_reached` with `accepted_samples` ≈ command count
  and `transport_failures` = 0.
- The Data tab `Current/Load Summary` row should NOT show any
  `sustained_servos=[...]` chip. If it does → stop and inspect that servo's
  tendon path; sustained means ≥3 consecutive samples at warning threshold.
- `samples.jsonl` should have `capture_accepted: true` and
  `missing_measured_servo_ids: []` on every accepted sample.

Ramp plan (subsequent runs only after the previous range passed):
- 0.25 cm → 0.5 cm → 0.75 cm → 1.0 cm.

## Stage 3: Mike CC convention probe (~2 min, offline)

Once you have a small `bottom_only_sweep` or `workspace_coverage` run with a
tracker:

```bash
.venv/bin/python -m continuum_robot.modeling.two_segment.validate_mike_cc \
    --latest \
    --config config/modeling_two_segment.example.yaml \
    --output-dir data/experiments/two_segment_mike_convention_probe/$(date +%Y%m%d_%H%M%S)
```

Read `mike_cc_convention_report.txt`. Possible recommendations:

| Recommendation | What it means | What to do |
|---|---|---|
| `conventions_consistent_with_evidence_safe_to_confirm` | Sign-match all axes + residuals under threshold | Set `physics_models.mike_constant_curvature.required_conventions_confirmed: true` in the modeling config. |
| `sign_convention_likely_flipped_axis_<x/y/z>` | Distal predicted sign disagrees with measured on that axis | Re-check `tendon_displacement_sign_convention` and `model_frame_convention` in modeling config. |
| `intermediate_sign_convention_likely_flipped_axis_<x/y/z>` | Distal signs OK but intermediate (bottom-segment endpoint) is flipped | Bottom-segment frame is likely flipped relative to top. |
| `magnitude_off_but_signs_ok_inspect_geometry` | All signs match but residuals are above threshold | Re-check segment lengths and tendon positions in config; could also be a hardware constant-curvature gap. |
| `no_samples_provided` / `no_predictions_generated` | The probe couldn't operate on the supplied run | Use `--allow-lower-trust` if the run is servo-only; otherwise run a proper trusted dataset first. |

The probe NEVER edits config. Your call.

## Stage 4: Repeatability scaffold (~10 min)

Run `two_segment_repeatability`. Recommended first-run config:

```yaml
target_set: "default_demo"
bottom_inner_radius_cm: 0.05
bottom_outer_radius_cm: 0.10
top_inner_radius_cm: 0.05
top_outer_radius_cm: 0.10
points_per_ring: 4
repeat_visits: 3
samples_per_visit: 1
settle_time_s: 1.5
allow_servo_only_test_run: false
run_trust_mode: "thesis_trusted"
```

Outputs:
- `two_segment_repeatability_summary.txt` — aggregate distal/intermediate RMS.
- `two_segment_repeatability_per_target.csv` — per-target scatter.
- `two_segment_repeatability_distal_scatter.png` — visual scatter map.
- `two_segment_repeatability_per_target_rms.png` — RMS bar chart per target.

Interpretation:
- This is **open-loop repeatability**. The number includes tendon hysteresis,
  tracker noise, and command repeat error. Treat it as a floor on what a
  closed-loop controller could achieve later, not as final accuracy.
- Configure `target_distal_rms_mm` in the config to record your operator
  acceptance criterion; the run reports whether measurements meet it but does
  not auto-fail.

## Stage 5: Modeling (~5-15 min, offline)

```bash
.venv/bin/python -m continuum_robot.modeling.two_segment.cli \
    --latest \
    --config config/modeling_two_segment.example.yaml \
    --output-root data/experiments/two_segment_modeling
```

Open the resulting `two_segment_modeling_summary.txt`. You should see lines
like:

```
models_completed: ['linear_baseline', 'ann']
models_unavailable: ['camarillo', 'mike_constant_curvature']
best_model: ann (xyz_rmse_mm=2.1)
```

Each model's full status is in `model_status.json`. Physics-model gating is in
`physics_model_parameter_report.txt`.

## Stage 6: Evidence index + handoff (~2 min)

After marking advisor-quality runs in the Data tab (`thesis_candidate` /
`advisor_share`):

```bash
.venv/bin/python -m continuum_robot.data.build_thesis_evidence_index \
    --project-root . \
    --output-dir data/exports
```

Then bundle for handoff:

```bash
.venv/bin/python -m continuum_robot.data.export_run_bundle \
    --latest two_segment_collect_pose_command_dataset \
    --zip --include-samples
```

The bundle now includes `long_run_health.json`,
`transport_recovery_report.json`, `sample_failure_events.jsonl`, and the
repeatability outputs.

## Watch chips on the Data tab

Open the run in the Data tab; the detail panel surfaces:

| Row | Means | When to act |
|---|---|---|
| `Long-Run Health` | Stop reason + cycle/sample counts | If `stop_reason` is anything other than `target_valid_sample_count_reached` or `scheduled_repeat_count_reached`, investigate the budget that tripped. |
| `Current/Load Summary` | Sustained servos + peak load | If `sustained_servos=[<id>]` shows up, inspect that tendon physically. |
| `Two-Segment Foundation` | bottom/top labels + IDs | Verify against your physical assembly. |
| `Two-Segment Pose Roles` | distal/intermediate availability | `distal_only=true` is fine but limits modeling to distal-only labels. |
| `Two-Segment Repeatability Scatter` | Aggregate RMS | Compare against your operator target if you set one. |

## Quick sanity tests (pre-bench)

Before connecting hardware, run:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_tests.sh quick           # ~1 sec
PYTHON_BIN=.venv/bin/python scripts/run_tests.sh hardware-safe   # ~25 sec
PYTHON_BIN=.venv/bin/python scripts/run_tests.sh two-segment     # ~17 sec
PYTHON_BIN=.venv/bin/python scripts/run_tests.sh single-segment  # ~30 sec (optional, for parity rehearsal)
```

The `two-segment` mode runs the full two-segment regression net (~150 tests):
math invariants, structural invariants, all three experiments, modeling
pipeline, Mike CC probe, end-to-end smoke, validators, exports, and evidence
index. If any of these don't pass on the bench machine, do NOT power on the
rig.

Re-run after each change to `config/system.local.yaml` or any code change.

## Confidence dial — what's hardware-day risk vs. tested

| Item | Status |
|---|---|
| Bottom/top role abstraction | Tested invariants — safe to flip in YAML. |
| Servo-ID assignment | 10 invariants pinned — physical IDs never move. |
| Live telemetry rejection | Tested: missing measured positions rejected on trusted runs. |
| Long-run continue-until-valid | Tested. |
| Current/load sustained-jam detection | Tested. |
| Mike CC math invariants | Tested vs textbook closed-form. |
| Mike CC sign/frame conventions | Tested via probe; YOU validate on hardware first. |
| ANN training quality | Tested when torch installed. Untested on real two-segment data. |
| 1 Mbps actual bus latency | Configurable + diagnostic available; reflash + measure on day-1. |
| Two-segment kinematics control | NOT IMPLEMENTED. Manual pretension + open-loop motion only. |
| Automatic two-segment pretension | NOT IMPLEMENTED. Manual staged capture only. |
| Camarillo predictions | UNAVAILABLE until you measure stiffness/cable/routing parameters. |
