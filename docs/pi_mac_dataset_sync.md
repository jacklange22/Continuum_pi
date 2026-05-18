# Pi/Mac Dataset Sync

Use one data layout everywhere:

```text
data/experiments/<experiment>/<run_id>/
```

The Pi writes runs there. The Mac sync tool copies runs into the same folder
structure in this repo, so the GUI Modeling/ANN tabs can discover them normally.
Raw JSONL files stay local and are ignored by Git.

## Pull Latest Dataset

From the Mac:

```bash
scripts/pull_pi_data.sh
```

This pulls the latest `collect_pose_command_dataset` from:

```text
continuum-pi@10.28.63.49:/home/continuum-pi/Continuum_pi
```

into:

```text
data/experiments/collect_pose_command_dataset/<run_id>/
```

## Pull A Different Experiment

```bash
scripts/pull_pi_data.sh workspace_repeatability_map
```

## Pull A Specific Run

```bash
scripts/pull_pi_data.sh collect_pose_command_dataset 20260518_164344_collect_pose_command_dataset
```

## Preview Without Copying

```bash
scripts/pull_pi_data.sh collect_pose_command_dataset 20260518_164344_collect_pose_command_dataset --print-only
```

## After Sync

The run should exist under:

```text
data/experiments/collect_pose_command_dataset/<run_id>/
```

Then refresh/restart the GUI. The Modeling/ANN tabs scan `data/experiments/...`.

CLI training uses the same path:

```bash
python3 scripts/ann_model_sweep.py \
  --dataset-path data/experiments/collect_pose_command_dataset/<run_id>
```

## What Not To Commit

Do not commit raw datapoint files:

```text
samples.jsonl
modeling_dataset_export.jsonl
modeling_dataset_legacy_compat.dat
workspace_map_visits.jsonl
raw_point_samples.jsonl
rejected_samples.jsonl
sample_failure_events.jsonl
```

They are ignored in `.gitignore`. Keep summaries/configs/plots in the run folder
if you want, but check staged files before pushing.

## If Git Push Fails Because Of JSONL Files

On the Pi:

```bash
git status --short
git ls-files 'data/**/*.jsonl'
python3 scripts/check_data_for_git.py
```

If raw files are tracked, untrack them without deleting local files:

```bash
python3 scripts/sync_pi_dataset.py git-clean
python3 scripts/sync_pi_dataset.py git-clean --apply
```

If GitHub still rejects the push, the large files are already inside an unpushed
commit. Reset/rewrite that local commit before pushing.
