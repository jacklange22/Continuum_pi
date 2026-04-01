# Tracker MVP Workflow

This is the operator path for tomorrow's Pi session. Ignore servos and full experiments for this pass.

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

- raw live capture CSV under `data/pivot_captures/*_pivot_0B_samples.csv`
- review dataset bundle under `runs/*_pivot_calibration_review/`
- `metadata.json`
- `samples.jsonl`
- `summary.json`
- staged tip file under `data/tip_cals/staged/*_generated_penprobe_tip.csv`
- accepted tip file at `data/tip_cals/generated_penprobe_tip.csv`

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
- tip file is written to `data/tip_cals/generated_penprobe_tip.csv`

Registration passes when:

- a tip file is loaded and registration geometry is ready
- 4 unique landmarks are selected
- all captures are complete
- solve succeeds with reported FRE / RMSE and residuals
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
- confirm `tip_pose_status` in the tracker-first workspace
