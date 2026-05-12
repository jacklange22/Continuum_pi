# Data Curation and GUI Performance Audit

Generated: 2026-05-12

## Data Inventory

- Report JSON: `data_curation_report.json`
- Report Markdown: `data_curation_report.md`
- Timestamped report directory: `data/diagnostics/data_curation/20260512_223156_data_curation/`
- Cleanup preview manifest: `data/diagnostics/data_curation/20260512_223159_cleanup_preview/cleanup_manifest.md`

Current inventory summary:

- `data/experiments`: 44 run items, 17.6 MB, 44 missing `run_review.json`, 3 mock/lower-trust candidates.
- `data/mock_experiments`: missing/empty.
- `data/experiments_archived`: missing/empty.
- `data/trash`: missing/empty.
- `data/exports`: present/empty.
- `data/calibration`: 2 items, 458.2 KB.
- `data/pivot_calibration`: 21 items, 3.4 MB, 18 missing `run_review.json`, 2 mock/lower-trust candidates.
- `data/runtime_tip_calibration`: 2 items, 76.6 KB.
- `data/diagnostics`: 42 generated diagnostic items, 758.8 KB.
- `data/modeling_results`: 1 generated result item, 114.4 KB.
- `data/models`: 1 keep candidate, 80.7 KB.

Classification counts:

- `generated_ignore`: 43
- `keep_candidate`: 6
- `needs_human_review`: 57
- `protected_active_alias`: 2
- `trash_candidate`: 5

No files over the 25 MB large-file threshold were found. No duplicate run groups were detected by run id or summary digest.

## Cleanup Preview

The generated preview proposes moving 5 lower-trust/mock candidates to `data/trash` and does not permanently delete anything:

- `data/pivot_calibration/20260329_013002_pivot_calibration`
- `data/pivot_calibration/20260328_155844_pivot_calibration`
- `data/experiments/repeatability_dataset/20260328_230219_repeatability_dataset`
- `data/experiments/repeatability_dataset/20260328_155844_repeatability_dataset`
- `data/experiments/aurora_grid_accuracy/20260328_155844_aurora_grid_accuracy`

## Ranked GUI Performance Findings

1. `continuum_robot/gui/controllers/data_management_controller.py::refresh` repeatedly called `summarize_run()` during filter/sort, filter-option refresh, detail rendering, and table rendering. `summarize_run()` validates the run, finds report/CSV files, and recursively computes run size, so this amplified filesystem walks on every filter change.
2. `continuum_robot/gui/tabs/data_management_tab.py::_sync_table` called `summarize_run()` per visible row while populating the table. This duplicated controller work and made table rendering scale poorly with run count.
3. `continuum_robot/gui/app_window.py::refresh` refreshed the System controller before checking the active tab, then discarded that work on Data, Modeling, Experiment, Tracking, Registration, Servos, and Pretension tabs.
4. `continuum_robot/data/run_management.py::summarize_run` still performs recursive size and figure discovery. It is now cached in the Data controller, but future work could split it into lightweight table summaries and selected-row details if the dataset grows much larger.
5. `continuum_robot/gui/controllers/modeling_controller.py::refresh` discovers modeling datasets and artifacts when the catalog is dirty. This is already gated, but trainability checks can still be expensive for selected two-segment datasets.
6. `continuum_robot/gui/controllers/experiment_controller.py::refresh` intentionally polls preflight at 0.25 s for selected manual experiments. It has cache keys and background history loading; no low-risk change was made.
7. `continuum_robot/gui/controllers/servos_controller.py::refresh` can perform multi-servo telemetry. `AppWindow._refresh_servo_state` already uses selected-servo refresh cycles; no algorithmic or hardware behavior was changed.

## Fixes Made

- Added cached run summaries in `DataManagementController`, invalidated on catalog changes and keyed by metadata/summary/review file signatures.
- Added refresh timing rows for total refresh, catalog discovery, filter/sort, filter options, details, visible summaries, table population, and summary-cache size.
- Reused cached summaries when building Data tab table rows.
- Added Data tab presets for Today, Real hardware, Single segment, Segment B, Needs review, Thesis/advisor, Large files, Generated diagnostics, and Trash candidates.
- Avoided unconditional System controller refresh when non-System tabs are active.
- Added safe curation report and cleanup manifest commands with dry-run/default preview behavior.

## Deferred Work

- Do not move expensive Data tab catalog discovery off the UI thread until a larger dataset demonstrates that summary caching is insufficient.
- Do not change experiment, transform, modeling, or robot control math.
- Do not delete generated diagnostics or trash candidates automatically; use the manifest-driven commands below.
