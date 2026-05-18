# Pi/Mac Dataset Sync

Use one data layout everywhere:

```text
data/experiments/<experiment>/<run_id>/
```

The Pi writes runs there. The Mac sync tool copies runs into the same folder
structure in this repo, so the GUI Modeling/ANN tabs can discover them normally.
Raw JSONL files stay local and are ignored by Git.

## Pull Everything From The Pi

From the Mac:

```bash
scripts/pull_pi_data.sh all
```

This pulls the full Pi experiment tree:

```text
continuum-pi@10.28.63.49:/home/continuum-pi/Continuum_pi/data/experiments/
```

into the Mac repo at:

```text
data/experiments/
```

This is the normal "make my Mac have the Pi datasets" command. It is resumable:
if the transfer stops, run it again and `rsync` continues from what is already
present. It does not delete local Mac files by default.

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

Do not commit generated experiment runs. They are ignored in `.gitignore` and
move through `rsync`, not GitHub. If you intentionally want one small summary or
plot in Git for thesis documentation, copy it into `docs/` or force-add that
specific file after checking its size.

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
