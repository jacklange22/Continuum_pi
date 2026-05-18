# Registration Trial Report

- Run ID: `e157e98f034d`
- Generated: 2026-05-18T19:46:47.233267+00:00
- Landmarks captured: ['L1', 'L10', 'L11', 'L12', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9']
- Captures per landmark: target=50 actual_min=50 actual_max=50

## Method comparison

| Method | FRE (mm) | Max residual (mm) | Worst label | LOO max drop (mm) |
|---|---:|---:|---|---:|
| mean | 0.4082 | 0.6456 | L3 | 0.0337 |
| median | 0.4090 | 0.6405 | L2 | 0.0352 |
| mad_filtered_mean | 0.4112 | 0.6470 | L3 | 0.0342 |
| trimmed_mean | 0.4125 | 0.6464 | L3 | 0.0336 |

**Best method: `mean` at 0.4082 mm**

## Subset search

_Subset search used the averaged points from method `mean`. Averaging knob is not stacked on top of subset knob to keep results comparable._

| Size | # subsets | Best FRE (mm) | Best subset | Max residual (mm) | Rank | Cond. # |
|---:|---:|---:|---|---:|---:|---:|
| 4 | 495 | 0.0797 | ['L1', 'L5', 'L6', 'L9'] | 0.1096 | 2 | 3.5 |
| 5 | 792 | 0.1032 | ['L1', 'L10', 'L5', 'L6', 'L9'] | 0.1296 | 2 | 3.6 |
| 6 | 924 | 0.1518 | ['L1', 'L10', 'L2', 'L5', 'L6', 'L9'] | 0.2572 | 2 | 4.4 |
| 7 | 792 | 0.2221 | ['L1', 'L10', 'L2', 'L5', 'L6', 'L8', 'L9'] | 0.4182 | 2 | 1.7 |
| 8 | 495 | 0.2764 | ['L1', 'L10', 'L12', 'L4', 'L5', 'L6', 'L8', 'L9'] | 0.4248 | 2 | 1.6 |

**Global best subset: size=4 labels=['L1', 'L5', 'L6', 'L9'] FRE=0.0797 mm**

## Samples per landmark

_Mean averaging with random k-subsets of the captured pool. Bootstrap iterations smooth the choice of which k samples were drawn._

| k | iterations | FRE mean (mm) | FRE std (mm) | FRE p95 (mm) | FRE min (mm) | FRE max (mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 40 | 0.4178 | 0.0197 | 0.4432 | 0.3686 | 0.4626 |
| 3 | 40 | 0.4101 | 0.0083 | 0.4253 | 0.3886 | 0.4260 |
| 5 | 40 | 0.4119 | 0.0064 | 0.4196 | 0.4027 | 0.4306 |
| 10 | 40 | 0.4101 | 0.0036 | 0.4153 | 0.4017 | 0.4173 |
| 20 | 40 | 0.4078 | 0.0030 | 0.4114 | 0.4008 | 0.4159 |
| 30 | 40 | 0.4079 | 0.0019 | 0.4107 | 0.4034 | 0.4109 |
| 50 | 40 | 0.4082 | 0.0000 | 0.4082 | 0.4082 | 0.4082 |

**Recommended k = 1 (mean FRE 0.4178 mm, within 0.020 mm of the captured pool's best 0.4078 mm).**

## Recommendations

- All averaging methods agree to within 0.01 mm on this dataset. With the current capture count, the averaging choice is not the bottleneck. Capture more samples per landmark (50+) to give MAD / trimmed mean something to act on.
- Truth landmarks are rank-deficient (rank=2): they sit on a plane. The third axis is recovered only from measurement noise. Add a landmark at a different height (non-coplanar z) before expecting sub-0.5 mm FRE consistently.
- Best landmark subset: ['L1', 'L5', 'L6', 'L9'] (size=4) at FRE=0.0797 mm.
- Samples per landmark: k=1 reaches mean FRE 0.4178 mm, within 0.020 mm of the captured pool's best (0.4078 mm). Capturing more samples per landmark beyond that point does not measurably improve FRE on this data.

## Report figures

- `registration_trial_point_spread_report.png`
- `registration_trial_subset_rms_report.png`
- `registration_trial_samples_per_point_report.png`
- `registration_trial_method_comparison_report.png`

