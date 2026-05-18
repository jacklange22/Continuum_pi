from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from continuum_robot.data.pi_dataset_sync import (
    build_rsync_pull_plan,
    local_run_dir_for_relative_path,
    resolve_latest_remote_run_relative_path,
    publish_lightweight_index,
    resolve_run_relative_path,
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


def test_publish_lightweight_index_excludes_raw_jsonl(tmp_path: Path) -> None:
    project = tmp_path / "project"
    run = tmp_path / "mirror" / "data" / "experiments" / "collect_pose_command_dataset" / "run-a"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"experiment_name": "collect_pose_command_dataset"}),
        encoding="utf-8",
    )
    (run / "metadata.json").write_text("{}", encoding="utf-8")
    (run / "samples.jsonl").write_text('{"too": "large"}\n', encoding="utf-8")
    (run / "modeling_dataset_export.jsonl").write_text('{"raw": true}\n', encoding="utf-8")
    (run / "workspace_map_3d_report.png").write_bytes(b"png")

    result = publish_lightweight_index(run_dir=run, project_root=project)

    copied_names = {path.name for path in result.copied_files}
    assert "summary.json" in copied_names
    assert "metadata.json" in copied_names
    assert "workspace_map_3d_report.png" in copied_names
    assert "README.md" in copied_names
    assert not (result.index_dir / "samples.jsonl").exists()
    assert not (result.index_dir / "modeling_dataset_export.jsonl").exists()
