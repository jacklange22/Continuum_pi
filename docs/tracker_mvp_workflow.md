# Tracker MVP Workflow

This is the operator path for tomorrow's Pi session. Ignore servos and full experiments for this pass.

The canonical GUI homes are now:

- `Tracking`: tracker connect, validation, live tool status, staged `0B` pivot calibration, and accepted tip-file review
- `Registration`: registration prerequisites, 4-point capture, solve/review/save, and live robot-frame pose summary

`Tracker Legacy` remains in the GUI only as a compatibility/diagnostic surface.

## Exact Order

1. Connect tracker
2. Validate tracker health
3. Confirm tool visibility and IDs
4. Confirm tool `0A` and `0B` transforms are valid
5. Start `0B` pivot collection
6. Stop collection and solve pivot calibration
7. Review RMSE, used/rejected samples, and the staged tip file
8. Accept and save the tip file
9. Select 4 registration landmarks
10. Capture registration samples
11. Solve registration
12. Review FRE and per-landmark residuals
13. Save the accepted registration
14. Confirm live robot-frame pose availability from `0A`

The focused GUI launcher is:

```bash
.venv/bin/python scripts/run_tracker_mvp.py
```

That launcher now opens the consolidated `Tracking` tab directly.

Or through the workflow wrapper:

```bash
.venv/bin/python scripts/run_lab_workflow.py tracker-mvp
```

## What Gets Saved

Tracker validation:

- `data/diagnostics/tracker_validation/<run>/tracker_validation_report.json`

Pivot calibration:

- raw live capture CSV under `data/pivot_calibration/captures/*_pivot_0B_samples.csv`
- review dataset bundle under `data/pivot_calibration/*_pivot_calibration_review/`
- older repo-root `runs/` bundles are legacy only and should be migrated into the canonical `data/` layout
- `metadata.json`
- `samples.jsonl`
- `summary.json`
- staged tip file under `data/pivot_calibration/staged/*_generated_penprobe_tip.csv`
- accepted tip file at `data/pivot_calibration/generated_penprobe_tip.csv`

Registration:

- accepted registration artifact under `data/registrations/registration_*.json`
- `data/registrations/latest_registration.json`

The accepted registration artifact already includes:

- captured landmark samples
- averaged centroids
- residuals
- FRE / RMSE
- per-landmark residual norms
- transforms and tool IDs
- `capture_tip_provenance`, which records the accepted `0B` tip file path, hash, loaded vector, and the fact that the offset was applied before solving
- `live_pose_tip_transform`, which records the saved `T_coil_tip` source and whether identity is being used by design

## Pivot -> Registration Linkage

The accepted pivot calibration feeds registration through the `0B` pen-probe capture path, not by post-editing the rigid-registration solve.

For the current simple tracker MVP:

- `data/pivot_calibration/generated_penprobe_tip.csv` defines `T_tool0B_tip`
- registration capture applies that offset during each live `0B` sample capture
- the captured tip points are then averaged and used to solve `T_robot_aurora`
- the saved registration artifact stores `capture_tip_provenance` so you can prove which accepted tip file was used

`T_coil_tip` is now owned by the separate advanced runtime tip calibration workflow launched from `Registration`.

- The runtime tip workflow uses the hat truth geometry to solve `T_tip_aurora` from `0B` captures.
- It averages stationary live `0A` coil poses and computes `T_coil_tip`.
- Tracking then uses the full chain `T_robot_tip = T_robot_aurora @ T_aurora_coil @ T_coil_tip`.
- If no accepted runtime tip artifact exists, tracking reports explicit identity fallback instead of pretending the chain is fully calibrated.

## Pass / Fail Interpretation

Tracker validation passes when:

- backend is connected
- frames are arriving
- tracker data is fresh
- tool `0B` is visible and tracked
- tool `0A` and `0B` transforms are rigid-valid
- no blocking tracker error is active

Pivot calibration passes when:

- tool `0B` is visible before capture
- the collection is explicitly stopped
- minimum sample count is met
- least-squares solve succeeds
- RMSE is reported
- used / rejected sample counts are reported
- the staged tip file is reviewed and accepted
- tip file is written to `data/pivot_calibration/generated_penprobe_tip.csv`

Registration passes when:

- a tip file is loaded and registration geometry is ready
- 4 unique landmarks are selected
- tracker data is fresh when each `0B` capture is taken
- all captures are complete
- solve succeeds with reported FRE / RMSE and residuals
- accepted registration is saved

Live robot-frame pose is available when:

- accepted registration is loaded
- tool `0A` is visible and tracked
- tracker data is fresh
- `tip_pose_status` becomes `ok`

Tracker operational-with-warning means:

- frames are arriving
- `0A` and `0B` are visible with rigid-valid transforms
- the tracker can still support the MVP workflow
- but one strict validation target, usually FPS, is below the configured threshold

Hard tracker failure means:

- no backend connection, no frames, missing required tools, invalid rigid transforms, or stale data
- do not proceed until the hard failure is resolved

## If A Step Fails

Tracker connect fails:

- confirm `aurora_port`
- confirm the correct serial device on the Pi
- re-run validation after reconnecting

Tracker validation fails:

- inspect the saved JSON report in `data/diagnostics/tracker_validation/`
- check backend selected, FPS, freshness, and visible tool IDs
- fix tool visibility before attempting pivot or registration

Pivot calibration fails:

- confirm `0B` is tracked continuously
- use the live collection panel to confirm sample count is rising
- increase sample quality by moving through a wider range of orientations
- do not proceed until the staged tip file has been explicitly accepted

Registration begin is blocked:

- confirm tracker validation passed
- confirm the accepted tip file exists and is loaded
- confirm 4 landmarks are selected

Registration solve or save fails:

- inspect the on-screen FRE / RMSE, residuals, and capture completeness
- if FRE is too high, retry with a wider spread of landmarks or recapture the worst point
- reload the latest registration only after a successful save

Live robot-frame pose is missing after save:

- confirm accepted registration path
- confirm `0A` is visible
- confirm `tip_pose_status` in the `Registration` tab dependency summary
