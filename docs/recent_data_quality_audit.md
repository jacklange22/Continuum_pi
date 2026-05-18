# Recent Data Quality Audit

Date: 2026-05-14. Worktree: `agitated-vaughan-8e590d`. Read-only audit.

This is a snapshot of what real run data on this rig actually shows. Each
finding has a path or empirical number behind it. The point is to choose the
next feature build from real evidence, not intuition.

## 1. Inventory

| Experiment family | Run count | Latest |
|---|---:|---|
| `collect_pose_command_dataset` | 34 | `20260514_223215_…` |
| `single_segment_repeatability` | 25 | `20260514_144931_…` |
| `penprobe_chasing_demo` | 30 | `20260514_223132_…` |
| `pretension_validation` | 7 | `20260512_223202_…` |
| `servo_tracker_sync_validation` | 5 | `20260513_163241_…` |
| `pivot_validation` | 3 | `20260424_014411_…` |
| `aurora_grid_accuracy` | 2 | `20260412_214736_…` |
| `repeatability_dataset` | 2 | `20260328_230219_…` (legacy) |
| `tracker_validation` | 1 | `20260420_150651_…` (no summary) |
| `tracker_timing_validation` | 1 | `20260412_214943_…` |
| `command_schedule_validation` | 1 | `20260417_230147_…` |

Missing on this disk: `data/experiments_archived/`, `data/mock_experiments/`,
`data/logs/`, `data/experiments/registration_validation/`,
`data/experiments/two_segment_*` (only test fixtures exist under
`tests/fixtures/two_segment_modeling_trainable/`).

## 2. Registration — the central empirical finding

**Every saved registration on this rig uses L1–L4 with ~5 samples per label.**
Across seven saved registrations the FRE varies by **3×**:

| Artifact | Labels | Samples/label | FRE (mm) |
|---|---|---|---:|
| `latest_registration.json` (2026-05-14) | L1, L2, L3, L4 | 5, 6, 8 | **0.523** |
| `20260423_211137_registration.json` | L1–L4 | 5 | 0.762 |
| `20260423_210928_registration.json` | L1–L4 | 5 | 1.339 |
| `20260423_203241_registration.json` | L1–L4 | 5, 6 | 0.632 |
| `registration_20260420T151038_270547Z.json` | L1–L4 | 5 | 0.549 |
| `registration_20260419T225301_998672Z.json` | L1–L4 | 5 | **1.487** |
| `registration_20260401T193748Z.json` | L1–L4 | 5, 6, 7 | 0.718 |

Per-landmark residual summary across the last 6 registrations
(`data/registrations/validation/latest_registration_validation.json`):

- L1: mean 0.79 ± 0.30 mm  (consistent worst)
- L2: mean 0.60 ± 0.47 mm
- L3: mean 0.77 ± 0.49 mm
- L4: mean 0.60 ± 0.36 mm
- Worst landmark across recent runs: **L1** (mean residual 0.79 mm).

`config/registration.yaml` declares **12 candidate landmarks** (L1–L12) in
body frame, but only the first four are ever captured. The infrastructure
(label-driven capture session, repository, rigid solver) supports an
arbitrary subset. The 12-point dataset has never been collected on this rig.

## 3. Repeatability — thesis goal missed by 60%

`data/experiments/single_segment_repeatability/` last 8 runs:

| Run | RMS (mm) | Path-dep RMS | Targets | Status |
|---|---:|---:|---:|---|
| `20260514_144931_…` | **1.614** | 1.614 | 17 | success |
| `20260514_005948_…` | 1.934 | 1.934 | 17 | success |
| `20260513_230513_…` | 1.713 | 1.713 | 17 | invalid (operator stop) |
| `20260513_135134_…` | — | — | 17 | invalid (stale telemetry s7/s8) |
| `20260513_011613_…` | 1.532 | 1.532 | 17 | invalid (operator stop) |
| 3 earlier 2026-04 runs | — | — | 17 | invalid (insufficient samples) |

`thesis_goal_rms_mm = 1.0`. The best recent run is **1.53 mm**, the latest
"success" is **1.61 mm**. The system is **~60% over the thesis goal**.

The repeatability summary does not carry `registration_fre_mm` or
`accepted_sample_count` through to `experiment_metrics` (provenance gap).

## 4. Modeling / motor babble — the latest 5 collect-pose runs

`data/experiments/collect_pose_command_dataset/`:

| Run | Status | Accepted | Unrec drops | Train-valid | Notes |
|---|---|---:|---:|---|---|
| `20260514_223215_…` | success | 4093 | 18 (all servo 8) | False (drops > 0) | clean otherwise |
| `20260514_202345_…` | failed | 1759 | 21 | False | exceeded max_consecutive_packet_failures=3 |
| `20260514_202300_…` | failed | 8 | 0 | True | operator stop |
| `20260514_165618_…` | failed | 1494 | 3 | False | servo 8 stale telemetry stopped run |
| `20260514_162835_…` | success | 182 | 0 | True | short clean smoke |

Two observations:

- Servo 8 dominates failure events (very consistent — every recent failure
  cites s7 or s8).
- The "real recovery" counter (`recovered_packet_error_count`) is 0
  everywhere (consistent with F-2 from the previous audit — the metric was
  wired in this session but the disk artifacts predate the fix).

## 5. Tracker grid + per-point spread

`data/experiments/aurora_grid_accuracy/20260412_214736_…`:

- 9 points × 3 samples = 27 captures.
- Overall RMS residual after alignment: **0.256 mm**.
- Mean within-point spread: **0.289 mm**. Max: **0.485 mm**.
- Per-point max residual: P03 = 0.41 mm.

This is a **strong baseline** for tracker frame consistency, but only **3
samples/point** — not enough to characterize the per-point distribution's
mean, std, or tails for choosing a registration averaging method.

## 6. Penprobe demo

Latest demo run (`20260514_223132_penprobe_chasing_demo`):
- `mapping_mode_used = aggressive_tick_demo`
- `max_tick_delta_used = 250` (default config)
- `stop_reason = "cap_limited_target_unreachable"` (cap-bound normal end, but
  the wording sounds like failure)
- `physical_tip_chasing = False`, `penprobe_demo_valid_for_thesis = False`
- **`valid_for_thesis_repeatability = True`** — contradiction with the
  demo-only flags. The previous-fix evidence-index `lower_trust` tag now
  catches this if the run is mismarked `thesis_candidate`.

## 7. Review hygiene

**Every single recent run has `review_status = "debug"`.** No run is marked
`thesis_candidate`, `advisor_share`, `keep`, or `archived`. The evidence
index has zero curated content. This is a workflow gap, not a code gap.

## 8. What this audit settles

| Question | Answer from data |
|---|---|
| Is registration good enough? | **Unknown.** FRE 0.5–1.5 mm range with only 4 points. |
| Does >4 points improve registration? | **Never tested on this rig.** |
| How many samples per point are needed? | **Never tested.** All recent runs use 5. |
| Which runs are advisor-ready? | **None** — nothing curated above `debug`. |
| Is collect-pose data clean? | Mostly. ~0.4% drops, correctly excluded. |
| Is two-segment data gathering scoped? | Code is there; no real two-segment runs exist on this rig yet. |
| Are failure modes obvious from summaries? | Largely yes — `stop_reason` is consistently set. |

## 9. Prioritized gaps

### P0 (must address before final thesis evidence)

1. **No multi-point registration evidence.** Repeatability is 60% over goal;
   registration FRE varies 3× across runs; only 4 of 12 candidate landmarks
   are ever used. There is no data to decide whether tighter registration
   would close the gap. → `registration_sampling_study`.

2. **Per-point tracker repeatability not characterized.** Aurora grid runs
   used only 3 samples/point. To choose mean vs median vs trimmed-mean
   averaging for registration captures, we need per-point spread at
   20+ samples. → Folded into `registration_sampling_study` (task B).

### P1 (needed before final data)

3. **Servo 7/8 are the dominant failure cause** across single-segment
   repeatability and collect-pose runs. Likely physical or
   calibration-pretension drift (see audit-1 F-7/F-8 P1).

4. **Repeatability summary does not carry registration FRE / sample counts
   through provenance.** Hard to compare across runs without re-opening
   each run's metadata.

5. **No advisor/thesis-curated runs exist.** Evidence index is empty in
   practice.

### P2 (useful polish)

6. Penprobe `stop_reason = "cap_limited_target_unreachable"` wording.
7. Penprobe `valid_for_thesis_repeatability=True` mislabel (now mitigated by
   the new `lower_trust` tag; still worth fixing at source).
8. `data/experiments_archived/` and `data/logs/` don't exist on disk — the
   data-tab archive flow has never been used here.

### P3

9. Pretension threshold drift between servos 1–4 (78/42/60/60 mA) and 5–8
   (220 mA default). Likely a contributor to s7/s8 failures.

## 10. Recommended next feature

**Build `registration_sampling_study`.** Evidence:

- The thesis-blocking question is whether registration error is a major
  contributor to the 1.6 mm repeatability RMS (vs the 1.0 mm goal). We
  cannot answer it without a 12-point N-samples-per-point dataset.
- All required pieces already exist: 12 candidate landmarks in
  `config/registration.yaml`, the registration solver
  (`continuum_robot/registration/rigid_solver.py`), the label-driven
  capture session, and a `per_point_*` metrics pattern in
  `aurora_grid_accuracy`. The study is a thin analysis layer, not a
  rewrite.
- It does **not** require new hardware — same pen-probe, same tracker, same
  jig.
- It directly tells the operator the recommended protocol (how many points,
  how many samples per point, mean vs median vs trimmed), which is also a
  durable thesis methodology contribution.

Two-segment data readiness is also a real gap, but lower-value right now
because there is no real two-segment hardware data on disk yet and the
single-segment validation ladder is still bleeding (1.6 mm vs 1.0 mm goal).
