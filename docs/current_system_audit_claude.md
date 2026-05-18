# Skeptical System Audit (Claude)

Date: 2026-05-14. Worktree: `agitated-vaughan-8e590d`.

This is a written audit of the continuum-robot operator stack as it stands.
I deliberately did not touch product code; this pass is read-only. Every
finding below has a file:line reference or an empirical run-output reference
so it can be verified independently.

## 1. Empirical snapshot (most recent real runs)

**`data/experiments/collect_pose_command_dataset/20260514_223215_collect_pose_command_dataset/`**

| key | value | comment |
|---|---|---|
| `status` | `success` | run completed |
| `sample_counts.total` | 4111 (1 + 4109 + 1) | counts match |
| `experiment_metrics.unrecovered_packet_error_count` | 18 | 18 drops |
| `experiment_metrics.recovered_packet_error_count` | **0** | **see Finding F-2** |
| `experiment_metrics.dropped_post_motion_telemetry_samples` | 18 | matches |
| `experiment_metrics.accepted_sample_count` | 4093 | target met |
| `experiment_metrics.valid_for_model_training` | **False** | drops > 0, allow_partial=false |
| `experiment_metrics.not_model_training_ready` | **False** | **contradicts the line above (F-3)** |

`sample_failure_events.jsonl` summary: 18 × `post_motion_telemetry_packet_error` +
18 × `post_motion_telemetry_resync_success`. All 18 are servo 8 (and two
adjacent retry hits on servo 7). No `pre_motion_*` events. So the bus is
recovering at the transport level, but the sample is still being dropped and
no "recovered" counter is incremented.

**`data/experiments/penprobe_chasing_demo/20260514_223132_penprobe_chasing_demo/`**

| key | value | comment |
|---|---|---|
| `mapping_mode_used` | `aggressive_tick_demo` | ok |
| `stop_reason` | `cap_limited_target_unreachable` | "stop" wording for normal cap-bound termination is alarming |
| `max_tick_delta_used` | 250 | binding cap |
| `valid_for_thesis_repeatability` | **True** | **contradicts** `penprobe_demo_valid_for_thesis=False` (F-4) |
| `penprobe_demo_valid_for_thesis` | False | demo correctly marked |
| `include_in_evidence_index` | False | ok |
| `recovered_write_packet_error_total` | 0 | no retries hit in this run |

**`config/neutral_setpoints.json`** (live calibration on disk)

- `robot_mode: single_segment`, `active_segment_key: segment_b`
- `commanded_servo_ids: [5,6,7,8]`, `expected_servo_ids: [5,6,7,8]` ✓
- `selected_servo_id: 1` while active segment is B (cosmetic mismatch)
- Servos 1–4 calibrated 2026-04-19 with custom pretension thresholds
  (78/42/60/60 mA).
- Servos 5–8 calibrated 2026-05-13 with the **default** 220 mA threshold.
  Mixed-age, mixed-tightness calibration on the same physical robot.

## 2. Component scoring (thesis/hardware readiness)

| Subsystem | Score | One-line reason |
|---|---:|---|
| Bootstrap / config | 70 | Coherent, but `baudrate: 57600` is still the default and 1 Mbps is not first-class. |
| Servo transport + DXL control | 75 | parallel-single command expansion is *displacement-based*, not raw-mirror — that part is correct. Baud is wired through but not pre-set to 1 Mbps anywhere. |
| Servo safety & bounds | 78 | Wrap guard, per-servo magnitude current limit 850 mA, voltage/temp gates all consistent. The 851 mA servo-2 event would have been blocked by `servo_reported_current_hard_limit_ma`. |
| Manual startup/neutral/pretension | 60 | Two calibration generations coexist on disk; servos 1–4 vs 5–8 have very different thresholds and capture dates. |
| Automatic pretension | 70 | Correctly single-segment-only. Not exercised in latest runs. |
| Tracker / runtime tip policy | 72 | `coil_as_tip` is the trusted demo path; `latest_accepted` is `lower_trust`. Penprobe demo doesn't precheck calibrated tip presence when `use_calibrated_runtime_tip=true`. |
| Registration / pivot / grid | 72 | Math protected; API surface coherent. Not audited deeply this pass. |
| Collect-pose / motor babble | 62 | Pre-motion AND post-motion recovery paths both exist. **But** the "recovered" packet-error counter is dead (always 0), and a "drop + resync" is misnamed as "recovered" in events. |
| Long-run recovery | 60 | Drops samples then resyncs the bus — never re-attempts the failed sample. `long_run_health.recommendation` correctly transitions to `completed_with_dropped_samples` on finalize. |
| Penprobe chasing | 50 | Off-by-one in retry counter; "aggressive" defaults are 250-tick total / 25-tick step (looks stuck); `hard_max_tick_delta_from_startup` is only used in precheck. |
| Single-segment repeatability | 70 | Not directly audited; existing path looks structurally clean. |
| Parallel-single demo | 50 | Command path is correct, but GUI cannot tell the operator that all 8 are active; status flags scattered between `parallel_single_demo`, `mirrored_parallel`, `true_two_segment_control=false`. |
| Two-segment scaffold | 72 | `two_segment_collect_pose_dataset.py` is correctly gated to `dual_segment` only and explicitly does not implement two-segment kinematics. |
| ANN / modeling | unknown | Not audited this pass. |
| Data tab / run mgmt | 65 | Protected statuses, sidecars, trash/archive paths exist. |
| Export / evidence index | 55 | `_include_run` (build_thesis_evidence_index.py:101–106) trusts `review_status` without re-checking `valid_for_model_training` or `valid_for_thesis_repeatability`. |
| GUI edit-state stability | 65 | `editable_update_blocked` is implemented in view_utils, but the live combo boxes never set `popup_open`, so the guard cannot fire for combo popups in the running app. |
| Tests + diagnostics | 70 | Wide surface coverage. Several gaps: no test for penprobe retry off-by-one, no test forcing `recovered_packet_error_count > 0`, no test that parallel_single GUI shows all-8 readiness. |
| Docs / runbooks | 75 | README and plans.md are thoughtful and consistent. |

## 3. Confirmed bugs (file:line)

### F-1. Penprobe retry counter is off-by-one for >2 attempts
`continuum_robot/experiments/penprobe_chasing_demo.py:1162-1168`
```python
recovered = 0
for idx in range(attempts):
    try:
        writer(...)
        if idx > 0:
            recovered += int(idx)   # should be: recovered += 1
        return recovered
```
With `goal_write_retry_attempts=3` and success on the third try (idx=2), this
returns `recovered=2`, not 1. Reported "recovered write retries" overstates
reality. With the example yaml default of 2 attempts this happens to behave,
which is why it has not been caught.

### F-2. `recovered_packet_error_count` is a dead metric (always 0)
`continuum_robot/experiments/builtins.py:6001, 5463-5471, 6456-6464`

- Line 6001 initializes the metric to 0.
- The only synthetic `command_result` returned after recovery
  (`_recover_collect_pose_post_motion_packet_error`, lines 5452-5481)
  sets `"unrecovered_packet_error_count": 1` in `command_metadata`, and
  never sets `"recovered_packet_error_count"`. The pre-motion success path
  (lines 5822-5852) returns the real command_result directly, so its
  metadata also does not carry a `recovered_packet_error_count`.
- Line 6463 reads `metadata.get("recovered_packet_error_count", 0)`, so the
  session metric is `+= 0` forever.
- Empirically: latest run had 18 `post_motion_telemetry_resync_success`
  events and `recovered_packet_error_count = 0`. The counter does not track
  anything that ever happens.

### F-3. `not_model_training_ready` does not mean what its name says
`continuum_robot/experiments/builtins.py:5977, 6880, 7977`
```python
"not_model_training_ready": bool(servo_only_mode or parallel_single_demo or not robot_positions)
```
This is True only on mode/coverage gates. The other gate that flips
`valid_for_model_training=False` is "drops > 0 and not allow_partial"
(line 6878). In the latest real run, `valid_for_model_training=False` AND
`not_model_training_ready=False` were both written. An advisor or downstream
script reading either label gets contradictory signal.

### F-4. Penprobe demo emits `valid_for_thesis_repeatability=True`
`data/experiments/penprobe_chasing_demo/20260514_223132_.../summary.json`

The most recent demo run has:
- `physical_tip_chasing: False`
- `penprobe_demo_valid_for_thesis: False`
- `valid_for_thesis_repeatability: True`  ← misleading

`build_thesis_evidence_index._include_run` reads `valid_for_thesis_repeatability`
indirectly via `review_status`. A penprobe demo marked `thesis_candidate` by
mistake would enter the thesis evidence index because the index does not
cross-check the run's own demo-specific flags.

### F-5. Parallel-single all-8 readiness is invisible in the GUI
`continuum_robot/gui/controllers/servos_controller.py:971-982`
```python
if self.state.robot_mode == "dual_segment":
    ...                                # builds "All-8 readiness: ..."
else:
    self.state.all_8_readiness_summary = ""
```
In `parallel_single` mode the entire all-8 readiness line is blanked, and
the single-segment readiness branch at line 996 is also skipped (it gates
on `robot_mode == "single_segment"`). The result is that the GUI shows
*nothing* describing all-8 readiness in the one mode where the operator
most needs it.

### F-6. Evidence index trusts review_status without cross-checking trust flags
`continuum_robot/data/build_thesis_evidence_index.py:101-107`
```python
if run.review.review_status in {"debug", "garbage", "archived"}:
    return bool(include_debug)
if run.review.include_in_evidence_index:
    return True
if run.review.review_status in {"thesis_candidate", "advisor_share"}:
    return True
return False
```
No verification that `valid_for_model_training` or
`valid_for_thesis_repeatability` is actually `True` in the run's summary
before the run lands in the evidence index. Combined with F-4 this is a
real provenance hole.

### F-7. `hard_max_tick_delta_from_startup` is precheck-only, not a runtime fallback
`continuum_robot/experiments/penprobe_chasing_demo.py:184-188, 1237`

`max_tick_delta_from_startup > hard_max_...` triggers a precheck error
(line 184). At runtime only `max_tick_delta_from_startup` clamps motion
(line 1237). The "hard cap" exists only to validate config; it never acts
as a safety net.

### F-8. Aggressive demo defaults look-stuck at 250 / 25 ticks
`continuum_robot/experiments/penprobe_chasing_demo.py:45-46`, example yaml
default. `max_tick_delta_from_startup=250`, `max_step_ticks=25` is the
binding pair. `hard_max=800` is unused (see F-7). Operators expecting
"visibly aggressive" motion will perceive this as the system being stuck
at a tiny envelope unless they explicitly raise both values.

### F-9. Combo-box popup guard is never triggered in live tabs
`continuum_robot/gui/widgets/view_utils.py` exposes `editable_update_blocked`,
which inspects `widget.property("popup_open")`. Searching the live tabs
for `setProperty("popup_open"` yields zero matches (the test does set it
manually). The guard exists in code and is tested, but
no live `QComboBox.showPopup()` ever flips it, so a refresh that lands
while a dropdown is open can still overwrite the operator's choice.

### F-10. Duplicated assignment in collect-pose execute path
`continuum_robot/experiments/builtins.py:5966-5967`
```python
parallel_single_demo = _collect_pose_parallel_single_mode(session)
parallel_single_demo = _collect_pose_parallel_single_mode(session)
```
Idempotent, not dangerous, but a clear rushed-codex artifact.

### F-11. Pre-motion recovery skips the burst cooldown
Pre-motion failure handler `builtins.py:5766-5922` uses
`telemetry_retry_delay_s` + resync; the post-motion path
`_recover_collect_pose_post_motion_packet_error` (line 5605+) writes the
event log and then resyncs but also does not apply a cooldown other than
the resync delay. Burst recovery is symmetric in that *both* are
single-retry-then-resync, but neither inserts a true bus cooldown of the
shape the user described.

### F-12. `baudrate: 57600` is still the default
`config/system.yaml:15`. The hardware path (`continuum_robot/hardware/dxl_bus.py:302+`)
accepts any baud, and the GUI baudrate spinbox allows up to 4 Mbps. There
is no preset for 1 Mbps, no warning when bringing all 8 up at 57600, and
no documented Wizard procedure for migrating the bus.

### F-13. Penprobe stop_reason wording for a normal cap-bound end
`stop_reason = "cap_limited_target_unreachable"` is logged for what is
often a normal end-of-run state (the demo target was outside the configured
tick cap). The word "unreachable" implies failure; the operator UX needs
a clear distinction between "cap binds" and "real failure".

### F-14. selected_servo_id=1 in neutral_setpoints.json while active_segment=segment_b
`config/neutral_setpoints.json:106`. Cosmetic but the kind of mismatch
that confuses an operator scanning the file for current state.

### F-15. Mock data hygiene check is soft, not enforced
`scripts/check_data_for_git.py:122-128` flags mock data under
`data/experiments/` but only as a warning. No directory-level barrier.

## 4. What is actually fine (don't fix)

- **Parallel-single command expansion** is displacement-based per segment,
  not raw-tick mirror. Each segment uses its own startup neutral.
  (`builtins.py:7455-7472`, verified by `tests/test_servo_service.py:796`.)
- **`valid_for_model_training` and `valid_for_thesis_repeatability`** are
  forced `False` for both `parallel_single` and `servo_only` modes in the
  collect-pose path. (`builtins.py:5975-5977, 6878-6881, 7979-7981`.)
- **Wrap guard, current hard limit (850 mA per servo, absolute magnitude),
  telemetry stale gate (0.25 s), voltage / temperature gates** are all
  consistently applied. The reported servo-2 851 mA event would have been
  blocked by `safety_guard.py:100-112` — that is policy working, not a
  bug to relax.
- **`torque_off_on_disconnect: false`** is honored by `dxl_bus.disconnect()`;
  no torque-disable on close.
- **Pre-motion and post-motion recovery paths both exist.** Bad rows are
  excluded from `_build_export_rows` via `modeling_export_exclude=True`
  (`builtins.py:6663`, `modeling_dataset_outputs.py:328`).
  Pre-motion deferred commands never produce a sample at all
  (`builtins.py:6126-6139`). The export filter is sound.
- **`two_segment_collect_pose_dataset`** is gated to `dual_segment` only
  and refuses to run otherwise (`two_segment_collect_pose_dataset.py:124`).

## 5. Prioritized fix list

### P0 — must fix before next hardware run
| # | Fix | Why P0 |
|---|---|---|
| F-5 | Show all-8 readiness in parallel_single in `servos_controller`. | The operator cannot tell from the GUI whether all 8 are alive in the only demo that drives all 8. |
| F-4 | Force `valid_for_thesis_repeatability=False` for penprobe demo. | Direct mislabel; lands in evidence index if marked thesis_candidate. |
| F-6 | Evidence index must cross-check `valid_for_model_training` or `valid_for_thesis_repeatability` before including `thesis_candidate` / `advisor_share` runs. | Provenance hole. |

### P1 — should fix before final data collection
| # | Fix | Why P1 |
|---|---|---|
| F-2 | Either delete `recovered_packet_error_count` from outputs or actually wire a recovery path that increments it (e.g. a within-command retry that succeeds without dropping the sample). Decide the semantics first. | The metric is empirically meaningless; advisor-facing summaries will read 0 forever. |
| F-3 | Rename or recompute `not_model_training_ready` so it tracks `not valid_for_model_training`. | Two adjacent flags contradict each other in the same `summary.json`. |
| F-1 | Penprobe retry counter `recovered += 1`. | Misreports retries when `goal_write_retry_attempts > 2`. |
| F-7 / F-8 | Decide whether `hard_max_tick_delta_from_startup` is a runtime safety cap or just a config validator. If it is a cap, enforce it at runtime. Either way, bump example-yaml defaults so "aggressive demo" actually looks aggressive (e.g. `max_tick_delta_from_startup=600`, `max_step_ticks=60`, with `hard_max=800` enforced). | Operator perception today: demo is stuck. |
| F-9 | Hook `QComboBox.showPopup()` / `hidePopup()` to set/unset `popup_open` in the live tabs. | Today the guard is tested but cannot fire during real interaction. |
| Pretension mix | Recapture pretension for servos 1–4 at 1 Mbps with current threshold consistent with 5–8 (or vice versa) before the next thesis-grade run. | Mixed-age, mixed-threshold calibration on the same robot. |

### P2 — useful polish
- F-12: First-class 1 Mbps baud — config preset, GUI preset menu, "active baud" readout, warning when bringing all 8 up at 57600.
- F-11: Make pre-motion vs post-motion recovery symmetric (both apply a real bus cooldown before resync). Probably a single `_recover_collect_pose_burst` helper.
- F-13: Rename `stop_reason` for cap-bound demo ends from `cap_limited_target_unreachable` → `cap_bounded_end` (or similar) and make the GUI say "cap reached", not "unreachable".
- F-10: Delete the duplicated `parallel_single_demo = ...` line.
- F-15: Make `check_data_for_git.py` fail (not warn) on mock data under `data/experiments/`.
- F-14: Clear / hide `selected_servo_id` when it does not belong to the active segment.

### P3 — later
- Two-segment provenance combo check in `check_data_for_git.py`.
- `validate_run_bundle.py` should enforce `sum(phase_counts) == total`.
- `long_run_health.recommendation` set explicitly to a terminal value on `StopRequested`, not only on success.

## 6. Clarifying questions (please answer before I make code changes)

1. **F-2 — `recovered_packet_error_count` semantics.** Do you want this to
   count "within-command retries that produced a usable sample"
   (i.e. real recoveries), or "bus resyncs after a drop" (i.e. the
   current `post_motion_telemetry_resync_success` events)? Today both are
   accidentally zero. I lean toward "real recoveries only" and renaming
   the existing event-log success to `bus_resynced_after_drop`.

2. **F-3 — `not_model_training_ready`.** Should this flag track
   *all* reasons a run is not training-valid (mode + drops + missing
   robot positions + lower-trust runtime tip), or should we delete it and
   keep only `valid_for_model_training`? I lean toward delete-and-keep-one.

3. **F-4 — Penprobe `valid_for_thesis_repeatability`.** Confirm:
   penprobe demo runs should always emit
   `valid_for_thesis_repeatability=False`, regardless of how cleanly the
   demo executed. Correct?

4. **F-6 — Evidence index gating.** When a `thesis_candidate` run has
   `valid_for_thesis_repeatability=False`, do you want the index build
   to (a) refuse to include it and surface a warning, (b) auto-downgrade
   the review status to `debug`, or (c) include it with a `lower_trust`
   tag? I lean toward (a).

5. **F-7 / F-8 — "aggressive" defaults.** Are you happy with bumping the
   example yaml defaults to `max_tick_delta_from_startup=600`,
   `max_step_ticks=60`, `hard_max_tick_delta_from_startup=800`, and
   making `hard_max` an enforced ceiling? Or do you want to keep the
   conservative example and only raise via GUI?

6. **F-11 — pre/post motion recovery symmetry.** Should the cooldown
   be applied as `resync_delay_s * resync_read_attempts`, or do you want
   a separate `bus_cooldown_after_burst_s` parameter (default 0.35 s)?

7. **1 Mbps migration.** Are you ready for me to ship 1 Mbps as the
   default `config/system.yaml` value, or do you want to keep 57600 as
   committed default and only flip it locally via `system.local.yaml`?
   (I will leave `system.local.yaml` alone in any case.)

8. **Pretension consistency.** Do you want me to (a) only audit/report
   the 1–4 vs 5–8 threshold drift, or (b) include a one-shot diagnostic
   that recaptures the four-servo pretension and rewrites
   `neutral_setpoints.json` consistently? (b) touches a calibration
   artifact, so I won't do it without explicit approval.

## 7. Proposed bounded first fix (pending your answers)

A single small commit covering F-4 + F-5 + F-6, all three of which are
small, surgical, and directly improve thesis-grade trust on the next
hardware run without touching servo control, kinematics, or transform
math.

Exact changes I would make:

1. **F-4.** In `continuum_robot/experiments/penprobe_chasing_demo.py`
   where the run summary metrics are emitted (the block that already
   sets `penprobe_demo_valid_for_thesis=False`), also force-set
   `valid_for_thesis_repeatability=False` and
   `valid_for_model_training=False` for any penprobe demo run. Add a
   regression test in `tests/test_penprobe_chasing_demo.py` asserting
   both flags are `False` on a successful demo summary.

2. **F-5.** In `continuum_robot/gui/controllers/servos_controller.py:971`
   extend the predicate to
   `if self.state.robot_mode in {"dual_segment", "parallel_single"}:`
   and reuse the same readiness summary for parallel_single. Add a
   GUI controller test asserting that, with `parallel_single` mode and
   8 healthy mock telemetry rows, `state.all_8_readiness_summary`
   contains "all expected servos are readable".

3. **F-6.** In
   `continuum_robot/data/build_thesis_evidence_index.py:101-107`
   add an additional check: if `run.review.review_status` is
   `thesis_candidate` or `advisor_share` but the run's summary
   `valid_for_thesis_repeatability` (or `valid_for_model_training`,
   when the index is for model evidence) is `False`, exclude the run
   and surface a warning in the index output. Add a test in
   `tests/test_thesis_evidence_index.py` covering both
   (a) `thesis_candidate` + `valid_for_thesis_repeatability=False` →
   excluded, (b) `thesis_candidate` + `valid_for_thesis_repeatability=True`
   → included.

All three changes touch fewer than ~40 lines of product code and ~60
lines of test code. None of them touch:

- transform math
- registration math
- kinematics
- servo write path
- servo safety
- ANN / modeling code
- pretension capture
- references/ or tools/

## 8. Tests I will run for the first fix

```bash
scripts/run_tests.sh quick
.venv/bin/pytest tests/test_penprobe_chasing_demo.py -q
.venv/bin/pytest tests/test_gui_controllers.py -q
.venv/bin/pytest tests/test_thesis_evidence_index.py -q
.venv/bin/pytest tests/test_servo_service.py::test_parallel_single_displacement_mirror -q  # already exists, smoke
```

And if any of those fail, no claim of "fixed".

## 9. Hardware verification steps (after the bounded fix is merged)

```bash
# Bus health (start here every session)
.venv/bin/python -m continuum_robot.diagnostics.servo_transport_diagnostic \
  --port /dev/ttyACM0 \
  --baud 57600 \
  --servo-ids 1,2,3,4,5,6,7,8 \
  --duration 30 \
  --read-rate-hz 10 \
  --fields minimal

# GUI smoke for the three fixes
scripts/run_gui.sh
# 1) Switch robot_mode to parallel_single. Confirm Servos tab shows
#    "All-8 readiness: all expected servos are readable." (F-5)
# 2) Open Data tab. Confirm the most recent penprobe demo run does NOT
#    appear with valid_for_thesis_repeatability=True. Re-run a short
#    penprobe demo and inspect summary.json → both training+thesis
#    flags must be False. (F-4)
# 3) Use the manage_runs CLI to set the penprobe run to thesis_candidate
#    and rebuild the evidence index. Confirm the run is EXCLUDED with
#    a warning, not silently included. (F-6)
```

## 10. Known limitations of this audit

- I did not audit ANN/modeling training in depth. The code under
  `continuum_robot/modeling/` and `modeling_two_segment.example.yaml`
  was only skimmed.
- I did not deeply audit registration/runtime-tip code; only the
  control-flow seams that touch the demos.
- I did not run the real test suite in this audit pass (no code
  changes yet). I will run `scripts/run_tests.sh quick` + targeted
  tests in the fix pass.
- I did not bench-validate any of these findings. Empirical evidence
  is from committed run artifacts only.
