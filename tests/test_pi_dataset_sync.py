from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from continuum_robot.data.pi_dataset_sync import (
    build_experiments_tree_pull_plan,
    build_rsync_pull_plan,
    list_tracked_raw_data_files,
    local_run_dir_for_relative_path,
    resolve_latest_remote_run_relative_path,
    resolve_run_relative_path,
    untrack_raw_data_files,
    write_dataset_sync_manifest,
)


def test_resolve_run_relative_path_accepts_run_id_and_relative_path() -> None:
    run_id = "20260518_220000_collect_pose_command_dataset"

    resolved = resolve_run_relative_path(experiment="collect_pose_command_dataset", run=run_id)
    assert resolved == PurePosixPath("data/experiments/collect_pose_command_dataset") / run_id

    explicit = resolve_run_relative_path(
        experiment="ignored",
        run=f"data/experiments/collect_pose_command_dataset/{run_id}",
    )
    assert explicit == resolved


def test_build_rsync_pull_plan_uses_resume_flags(tmp_path: Path) -> None:
    destination = tmp_path / "mirror" / "run"
    plan = build_rsync_pull_plan(
        ssh_target="pi@continuum-pi.local",
        remote_run_path="/home/continuum-pi/Continuum_pi/data/experiments/demo/run",
        local_run_dir=destination,
        dry_run=True,
    )

    assert plan.args[0] == "rsync"
    assert "-avP" in plan.args
    assert "--partial" in plan.args
    assert "--inplace" in plan.args
    assert "--dry-run" in plan.args
    assert plan.source == "pi@continuum-pi.local:/home/continuum-pi/Continuum_pi/data/experiments/demo/run/"
    assert plan.destination == destination.resolve()
    assert not destination.exists()


def test_build_experiments_tree_pull_plan_targets_data_experiments(tmp_path: Path) -> None:
    plan = build_experiments_tree_pull_plan(
        ssh_target="pi@continuum-pi.local",
        remote_project_root=PurePosixPath("/home/continuum-pi/Continuum_pi"),
        local_mirror_root=tmp_path / "mirror",
        dry_run=True,
    )

    assert plan.source == "pi@continuum-pi.local:/home/continuum-pi/Continuum_pi/data/experiments/"
    assert plan.destination == tmp_path / "mirror" / "data" / "experiments"
    assert "--dry-run" in plan.args


def test_local_run_dir_preserves_data_experiments_layout(tmp_path: Path) -> None:
    relative = PurePosixPath("data/experiments/collect_pose_command_dataset/run-a")

    local = local_run_dir_for_relative_path(local_mirror_root=tmp_path / "mirror", run_relative_path=relative)

    assert local == tmp_path / "mirror" / "data" / "experiments" / "collect_pose_command_dataset" / "run-a"


def test_resolve_latest_remote_run_uses_remote_glob(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_check_output(args, *, text):
        calls.append(list(args))
        assert text is True
        return "/home/continuum-pi/Continuum_pi/data/experiments/collect_pose_command_dataset/run-b/\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    relative = resolve_latest_remote_run_relative_path(
        ssh_target="pi@continuum-pi.local",
        remote_project_root=PurePosixPath("/home/continuum-pi/Continuum_pi"),
        experiment="collect_pose_command_dataset",
    )

    assert relative == PurePosixPath("data/experiments/collect_pose_command_dataset/run-b")
    assert calls
    assert "/*/" in calls[0][2]


def test_manifest_records_file_sizes_without_hashing_by_default(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text("{}", encoding="utf-8")
    (run / "samples.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    manifest = write_dataset_sync_manifest(run_dir=run)
    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "dataset_sync_manifest_v1"
    assert payload["file_count"] == 2
    assert {item["path"] for item in payload["files"]} == {"summary.json", "samples.jsonl"}
    assert all(item["sha256"] is None for item in payload["files"])


def test_git_clean_lists_and_untracks_raw_data_without_deleting(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
    run = repo / "data" / "experiments" / "collect_pose_command_dataset" / "run-a"
    run.mkdir(parents=True)
    raw = run / "samples.jsonl"
    summary = run / "summary.json"
    raw.write_text('{"raw": true}\n', encoding="utf-8")
    summary.write_text("{}", encoding="utf-8")
    subprocess.run(["git", "add", "data"], cwd=repo, check=True)

    tracked = list_tracked_raw_data_files(project_root=repo)
    assert tracked == ["data/experiments/collect_pose_command_dataset/run-a/samples.jsonl"]

    untrack_raw_data_files(project_root=repo, files=tracked)

    assert raw.exists()
    remaining = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    assert "data/experiments/collect_pose_command_dataset/run-a/samples.jsonl" not in remaining
    assert "data/experiments/collect_pose_command_dataset/run-a/summary.json" in remaining
