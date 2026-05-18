# Timing & Synchronization Audit

Companion audit for the two validation experiments that characterize the data pipeline:

- `tracker_timing_validation` — how fast and how consistently the Aurora tracker delivers frames through the Python wrapper.
- `servo_tracker_sync_validation` — how tightly servo telemetry and tracker frames are aligned on host monotonic time, which directly underwrites `collect_pose_command_dataset` data quality.

This doc walks the data path, explains the realized rate vs theoretical, and points to the thesis figures each experiment now emits.

## The tracker data path

Every Aurora frame the Python wrapper publishes goes through three measured stages, defined in [ndi_backend.py:1184-1195](../continuum_robot/tracking/ndi_backend.py:1184):

| Stage | What it measures |
|---|---|
| `backend_call_ms` | time the NDI library spends fetching a frame from Aurora (USB I/O, hardware wait) |
| `parse_ms` | time decoding the raw response into per-tool poses + frame number |
| `state_commit_ms` | time writing the parsed record into shared state and notifying listeners |
| `total_cycle_ms` | sum of the above — wall-clock per cycle the wrapper is "busy" for |

These per-cycle fields appear in `samples.jsonl` for any experiment that records tracker timing, and `extract_tracker_timing_records` in [timing_benchmark.py:69](../continuum_robot/tracking/timing_benchmark.py:69) is the canonical parser.

### Aurora's theoretical ceiling

Aurora nominally delivers a new frame every 25 ms (40 Hz). When `backend_call_ms` is short (no new frame ready), the wrapper records that as a **duplicate frame** — `is_duplicate_frame=True` and the same `frame_number` as the previous cycle. These cycles do not deliver fresh data; they only deliver "we asked, got nothing new." Duplicate frames inflate the effective polling rate without contributing fresh information.

### The rate gap

In practice we observe `total_cycle_ms` around 40 ms, i.e. **~24 Hz realized vs 40 Hz theoretical**. Two thesis figures characterize this:

- **[thesis_01_cycle_time_distribution.png](#)** (in any `data/experiments/tracker_timing_validation/<run>/`) shows the per-cycle distribution with the 25 ms (40 Hz) reference and the observed mean / p95 / p99 marked. The shaded region to the left of 25 ms is the "made the budget" zone; samples to the right of it are cycles that didn't hit 40 Hz.
- **[thesis_02_stage_time_budget.png](#)** decomposes the per-cycle time at four percentiles (median / mean / p95 / p99) into backend_call / parse / state_commit contributions, and labels each bar with total ms ≈ equivalent Hz. Reading this figure: the dominant stage is where the 40 → 24 Hz gap actually lives, and that is the answer to "if we wanted to optimize, where would we start?"

### Where to look further

`debug.json` in the run directory holds:
- `duplicate_frames`: count + percentage. If this is high, the realized "loop Hz" overstates the rate of fresh data.
- `per_tool_valid_rate`: fraction of cycles each requested tool was `tracked`.
- `stage_stats`: full mean/std/percentile breakdown per stage.
- `errors`: backend errors + invalid/missing requested-tool counts.

The text summary (`aurora_timing_summary.txt`) is no longer written — debug.json + thesis figures cover everything that was in it without duplication.

## The servo–tracker sync architecture

`collect_pose_command_dataset` and other dynamic experiments record three parallel streams during motion:

1. **Tracker frames** — published by the Aurora backend on its own cadence (~24 Hz).
2. **Servo telemetry** — polled at a configurable interval (typically every 10 ms) into the same `samples.jsonl` with `record_kind="servo_timing"`.
3. **Servo commands** — appended when each new goal position is written, with `record_kind="servo_command"`.

All three streams record `monotonic_time_s` from the same host clock. To produce paired `(servo_state, tracker_pose)` records — what downstream modeling actually consumes — the analysis matches each tracker frame to its **nearest-neighbor servo telemetry sample** by monotonic time. The bigger the residual offset of that match, the bigger the temporal uncertainty in the pair.

The matching is performed in the experiment analysis side and surfaced as `tracker_to_servo_telemetry_offsets_ms` in the experiment metrics. `servo_tracker_sync_validation` runs this same matching against a scripted motion, characterizing the offset distribution under controlled conditions.

### Why this is a forward-link to collect_pose

The pairing mechanism that `servo_tracker_sync_validation` characterizes is the **same** mechanism that produces the modeling dataset. If the validation experiment shows pairs within, say, 8 ms p95, then the modeling dataset has the same per-pair temporal uncertainty. The validation figures are therefore prerequisites for trusting the modeling chapter's data.

### Two thesis figures

- **[thesis_01_pair_time_alignment.png](#)** is a histogram + CDF of the per-pair offsets with reference lines at 5 / 10 / 25 ms and the cross rate annotated for each. The 5 ms line is the tight bar; the 25 ms line is the very loose bar. Title explicitly names the cross-experiment connection.
- **[thesis_02_motion_correspondence.png](#)** is a stacked-panel time series. Top: servo measured position (travel from start, ticks). Bottom: tracker tip displacement (mm). Same X axis (monotonic time). If the streams are co-temporal, the two curves rise and fall together; visible lag would indicate either a sync problem or a kinematic delay between servo motion and tip motion. This is a *sanity-check figure*: the alignment metric in thesis_01 is the headline; thesis_02 demonstrates that what we're aligning is actually the same motion observed two ways.

### Where to look further

`debug.json` in the run directory holds:
- `alignment.threshold_cross_rates`: % of pairs within 5/10/20/25 ms.
- `sample_counts`: tracker / telemetry / command sample counts (sanity check that all streams were captured).
- `per_tool_valid_rate`: tracker health during the motion.
- `motion`: max displacement, requested tool IDs, robot-frame-tip preference.
- `note`: pointer back to the collect_pose linkage.

## Reading the two experiments together

Together they answer two questions every dynamic-motion claim in the thesis depends on:

1. **"Is the tracker fresh enough?"** Answered by tracker_timing thesis_01 (distribution narrowness + tail) and thesis_02 (which stage dominates).
2. **"Are the paired records actually paired?"** Answered by servo_tracker_sync thesis_01 (offset distribution + threshold rates) and thesis_02 (motion shows up in both streams at the same time).

When `collect_pose_command_dataset` later produces a modeling dataset of `(commanded_position, observed_tip_pose)` pairs, these two experiments are the upstream evidence that those pairs are tight in both time and freshness.

## File contract per run

Both experiments now emit the same minimal layout:

```
data/experiments/<experiment_name>/<timestamp>_<experiment_name>/
  samples.jsonl                  # canonical timeseries records
  metadata.json                  # canonical run identity
  summary.json                   # canonical results + per-stage metrics
  config_snapshot.yaml           # canonical config snapshot
  run_review.json                # operator review sidecar (when reviewed)
  transform_chain_summary.json   # transforms/config (from transform_chain_outputs)
  debug.json                     # everything that didn't make it onto thesis figures
  thesis_01_*.png                # one thesis-quality figure
  thesis_02_*.png                # one thesis-quality figure
```

No metrics.csv, no `*_summary.txt`, no per-stage report PNGs — those duplicated information already in the JSON + thesis figures and had no real downstream consumers.
