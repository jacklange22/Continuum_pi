# Pi/Mac Dataset Sync

Use GitHub for code, configs, summaries, plots, and lightweight run indexes.
Use `rsync` for raw experiment folders.

This avoids pushing 10-100GB `samples.jsonl` / `modeling_dataset_export.jsonl`
files through GitHub while still keeping enough metadata in the repo to know
what datasets exist and where they came from.

## Pull The Latest Run

Run this on the Mac:

```bash
python3 scripts/sync_pi_dataset.py pull \
  --pi pi@continuum-pi.local \
  --experiment collect_pose_command_dataset
```

What it does:

- finds the latest Pi run under
  `/home/continuum-pi/Continuum_pi/data/experiments/collect_pose_command_dataset/`
- copies the full run folder to `~/ContinuumData/pi_runs/data/experiments/...`
- uses resumable `rsync -avP --partial --inplace`
- writes `dataset_sync_manifest.json` in the local mirrored run
- writes a small GitHub-safe index under `data/synced_run_index/`
- prints the local training path

## Pull A Specific Run

```bash
python3 scripts/sync_pi_dataset.py pull \
  --pi pi@continuum-pi.local \
  --experiment collect_pose_command_dataset \
  --run 20260518_220000_collect_pose_command_dataset
```

You can also pass a repo-relative path:

```bash
python3 scripts/sync_pi_dataset.py pull \
  --pi pi@continuum-pi.local \
  --run data/experiments/collect_pose_command_dataset/20260518_220000_collect_pose_command_dataset
```

## Preview The Transfer

```bash
python3 scripts/sync_pi_dataset.py pull \
  --pi pi@continuum-pi.local \
  --experiment collect_pose_command_dataset \
  --dry-run
```

Use `--print-only` if you only want the generated `rsync` command.

## Train On The Mac

After pull, use the printed local run path:

```bash
python3 scripts/ann_model_sweep.py \
  --dataset-path ~/ContinuumData/pi_runs/data/experiments/collect_pose_command_dataset/<run_id>
```

## Publish A Lightweight Index Only

If a run is already copied locally:

```bash
python3 scripts/sync_pi_dataset.py index \
  --run-dir ~/ContinuumData/pi_runs/data/experiments/collect_pose_command_dataset/<run_id>
```

The index intentionally excludes raw JSONL files. It copies small files like:

- `summary.json`
- `metadata.json`
- `dataset_quality_summary.json`
- `modeling_dataset_summary.txt`
- `workspace_map_summary.json`
- report PNGs
- `dataset_sync_manifest.json`

Commit `data/synced_run_index/...` if you want dataset visibility from GitHub.

## Large-Run Rules

- Do not commit raw `samples.jsonl`.
- Do not commit raw `modeling_dataset_export.jsonl` when it is large.
- Do not zip 100GB runs just to move them.
- Prefer repeated `rsync` pulls; it resumes interrupted transfers.
- Use `--sha256` only when you need full byte-level verification. It will read
  the entire 100GB run after transfer.

## If The Pi Hostname Is Different

Use the actual SSH target:

```bash
python3 scripts/sync_pi_dataset.py pull \
  --pi jack@192.168.1.44 \
  --experiment workspace_repeatability_map
```

If the Pi repo is not at `/home/continuum-pi/Continuum_pi`, pass:

```bash
--remote-project-root /path/to/Continuum_pi
```
