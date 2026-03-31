# Tracker MVP Workflow

This is the operator path for tomorrow's Pi session. Ignore servos and full experiments for this pass.

## Exact Order

1. Connect tracker
2. Validate tracker health
3. Confirm tool visibility and IDs
4. Confirm tool `0A` and `0B` transforms are valid
5. Run pivot calibration on tool `0B`
6. Save and inspect the tip file
7. Select 4 registration landmarks
8. Capture registration samples
9. Solve registration
10. Save the accepted registration
11. Confirm live robot-frame pose availability from `0A`

The focused GUI launcher is:

```bash
python scripts/run_tracker_mvp.py
```

Or through the workflow wrapper:

```bash
python scripts/run_lab_workflow.py tracker-mvp
```

## What Gets Saved

Tracker validation:

- `data/tracker_validations/*_tracker_validation.json`

Pivot calibration:

- raw dataset bundle under `data/experiments/*_pivot_calibration_*`
- `metadata.json`
- `samples.jsonl`
- `summary.json`
- tip file at `data/tip_cals/generated_penprobe_tip.csv`

Registration:

- accepted registration artifact under `data/registrations/registration_*.json`
- `data/registrations/latest_registration.json`

The accepted registration artifact already includes:

- captured landmark samples
- averaged centroids
- residuals
- FRE / RMSE
- transforms and tool IDs

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
- minimum sample count is met
- least-squares solve succeeds
- RMSE is reported
- tip file is written to `data/tip_cals/generated_penprobe_tip.csv`

Registration passes when:

- a tip file is loaded and registration geometry is ready
- 4 unique landmarks are selected
- all captures are complete
- solve succeeds with reported FRE / RMSE
- accepted registration is saved

Live robot-frame pose is available when:

- accepted registration is loaded
- tool `0A` is visible and tracked
- `tip_pose_status` becomes `ok`

## If A Step Fails

Tracker connect fails:

- confirm `aurora_port`
- confirm the correct serial device on the Pi
- re-run validation after reconnecting

Tracker validation fails:

- inspect the saved JSON report in `data/tracker_validations/`
- check backend selected, FPS, freshness, and visible tool IDs
- fix tool visibility before attempting pivot or registration

Pivot calibration fails:

- confirm `0B` is tracked continuously
- increase sample quality by moving through a wider range of orientations
- rerun until the tip file is rewritten cleanly

Registration begin is blocked:

- confirm tracker validation passed
- confirm the tip file exists and is loaded
- confirm 4 landmarks are selected

Registration solve or save fails:

- inspect the on-screen FRE / RMSE and capture completeness
- reload the latest registration only after a successful save

Live robot-frame pose is missing after save:

- confirm accepted registration path
- confirm `0A` is visible
- confirm `tip_pose_status` in the tracker-first workspace
