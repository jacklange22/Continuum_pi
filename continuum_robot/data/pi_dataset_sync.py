"""Simple Pi-to-Mac dataset sync helpers.

This module intentionally stays small: raw experiment folders move directly with
``rsync`` while Git only carries lightweight summaries/manifests.  It is meant
for 10-100GB run folders where GitHub/Git LFS would add more friction than value.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
from typing import Any, Sequence

from continuum_robot.experiments.dataset_io import sanitize_output_name


DEFAULT_REMOTE_PROJECT_ROOT = PurePosixPath("/home/continuum-pi/Continuum_pi")
DEFAULT_LOCAL_MIRROR_ROOT = Path("~/ContinuumData/pi_runs")
DEFAULT_INDEX_ROOT = Path("data/synced_run_index")

RAW_DATA_FILENAMES = {
    "samples.jsonl",
    "modeling_dataset_export.jsonl",
    "modeling_dataset_legacy_compat.dat",
    "workspace_map_visits.jsonl",
    "raw_point_samples.jsonl",
    "rejected_samples.jsonl",
    "sample_failure_events.jsonl",
}

LIGHTWEIGHT_EXACT_FILENAMES = {
    "summary.json",
    "metadata.json",
    "run_review.json",
    "config_snapshot.yaml",
    "dataset_quality_summary.json",
    "dataset_quality_summary.txt",
    "modeling_dataset_summary.txt",
    "workspace_map_summary.json",
    "workspace_map_per_target.csv",
    "registration_validation_summary.txt",
    "pivot_validation_summary.txt",
    "repeatability_summary.txt",
    "two_segment_dataset_summary.txt",
    "two_segment_repeatability_summary.txt",
    "two_segment_modeling_summary.txt",
    "training_summary.txt",
    "training_metadata.json",
    "training_config.json",
    "evaluation_metadata.json",
    "dataset_sync_manifest.json",
}

LIGHTWEIGHT_SUFFIXES = (
    "_report.png",
    "_summary.txt",
    "_summary.json",
    "_per_target.csv",
)


@dataclass(frozen=True)
class RsyncPlan:
    """A concrete rsync command plus source/destination metadata."""

    args: list[str]
    source: str
    destination: Path
    local_run_dir: Path
    remote_run_path: str

    @property
    def shell_command(self) -> str:
        return " ".join(shlex.quote(part) for part in self.args)


@dataclass(frozen=True)
class FileManifestEntry:
    """One file entry in ``dataset_sync_manifest.json``."""

    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str | None = None


@dataclass(frozen=True)
class DatasetSyncManifest:
    """Small local record describing a synced run folder."""

    run_dir: Path
    manifest_path: Path
    file_count: int
    total_size_bytes: int
    files: list[FileManifestEntry] = field(default_factory=list)


@dataclass(frozen=True)
class LightweightIndexResult:
    """Result of copying GitHub-safe metadata into ``data/synced_run_index``."""

    index_dir: Path
    copied_files: list[Path] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)


def build_rsync_pull_plan(
    *,
    ssh_target: str,
    remote_run_path: str | PurePosixPath,
    local_run_dir: Path,
    dry_run: bool = False,
    compress: bool = False,
    delete: bool = False,
) -> RsyncPlan:
    """Return the resumable rsync command used to pull one run folder."""

    ssh_target = str(ssh_target or "").strip()
    if not ssh_target:
        raise ValueError("ssh_target is required, e.g. pi@continuum-pi.local")
    remote_path = _with_trailing_slash(str(PurePosixPath(str(remote_run_path))))
    local_run_dir = Path(local_run_dir).expanduser().resolve()
    source = f"{ssh_target}:{remote_path}"
    args = [
        "rsync",
        "-avP",
        "--partial",
        "--inplace",
        "--human-readable",
        "--exclude",
        ".DS_Store",
        "--exclude",
        "__pycache__/",
    ]
    if compress:
        args.append("-z")
    if delete:
        args.append("--delete")
    if dry_run:
        args.append("--dry-run")
    args.extend([source, _with_trailing_slash(str(local_run_dir))])
    return RsyncPlan(
        args=args,
        source=source,
        destination=local_run_dir,
        local_run_dir=local_run_dir,
        remote_run_path=str(PurePosixPath(str(remote_run_path))),
    )


def run_rsync_plan(plan: RsyncPlan) -> None:
    """Execute a prepared rsync plan."""

    plan.local_run_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(plan.args, check=True)


def resolve_run_relative_path(*, experiment: str, run: str) -> PurePosixPath:
    """Resolve a run name or relative run path under ``data/experiments``."""

    run = str(run or "").strip()
    if not run or run == "latest":
        raise ValueError("latest must be resolved on the remote host first")
    normalized = run.lstrip("/")
    if normalized.startswith("data/"):
        return PurePosixPath(normalized)
    experiment_name = sanitize_output_name(experiment, default="experiment")
    return PurePosixPath("data") / "experiments" / experiment_name / normalized


def resolve_latest_remote_run_relative_path(
    *,
    ssh_target: str,
    remote_project_root: PurePosixPath,
    experiment: str,
) -> PurePosixPath:
    """Ask the Pi for the newest run directory for ``experiment``."""

    experiment_name = sanitize_output_name(experiment, default="experiment")
    remote_experiment_root = PurePosixPath(remote_project_root) / "data" / "experiments" / experiment_name
    remote_glob = shlex.quote(str(remote_experiment_root)) + "/*/"
    command = (
        "latest=$(ls -1dt "
        + remote_glob
        + " 2>/dev/null | head -n 1); "
        + "test -n \"$latest\" && printf '%s\\n' \"$latest\""
    )
    output = subprocess.check_output(["ssh", str(ssh_target), command], text=True).strip()
    if not output:
        raise FileNotFoundError(f"No remote runs found under {remote_experiment_root}")
    latest_path = PurePosixPath(output.rstrip("/"))
    try:
        return latest_path.relative_to(PurePosixPath(remote_project_root))
    except ValueError:
        # Keep a useful error instead of silently mirroring an unexpected path.
        raise RuntimeError(
            f"Latest remote run {latest_path} is not under remote project root {remote_project_root}"
        )


def local_run_dir_for_relative_path(*, local_mirror_root: Path, run_relative_path: PurePosixPath) -> Path:
    """Map ``data/experiments/...`` to the local mirror root."""

    return Path(local_mirror_root).expanduser().resolve().joinpath(*run_relative_path.parts)


def write_dataset_sync_manifest(
    *,
    run_dir: Path,
    include_sha256: bool = False,
    manifest_name: str = "dataset_sync_manifest.json",
) -> DatasetSyncManifest:
    """Write a compact file manifest for a synced run folder.

    SHA256 is optional because hashing a 100GB JSONL can take a long time.  The
    default manifest records path, byte size, and mtime for quick inventory.
    """

    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    entries: list[FileManifestEntry] = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == manifest_name:
            continue
        stat = path.stat()
        sha = _sha256_file(path) if include_sha256 else None
        entries.append(
            FileManifestEntry(
                path=path.relative_to(run_dir).as_posix(),
                size_bytes=int(stat.st_size),
                mtime_ns=int(stat.st_mtime_ns),
                sha256=sha,
            )
        )
    total_size = sum(entry.size_bytes for entry in entries)
    manifest_path = run_dir / manifest_name
    payload = {
        "schema_version": "dataset_sync_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "include_sha256": bool(include_sha256),
        "file_count": int(len(entries)),
        "total_size_bytes": int(total_size),
        "files": [entry.__dict__ for entry in entries],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return DatasetSyncManifest(
        run_dir=run_dir,
        manifest_path=manifest_path,
        file_count=len(entries),
        total_size_bytes=total_size,
        files=entries,
    )


def publish_lightweight_index(
    *,
    run_dir: Path,
    project_root: Path,
    index_root: Path = DEFAULT_INDEX_ROOT,
    max_file_bytes: int = 10 * 1024 * 1024,
) -> LightweightIndexResult:
    """Copy only GitHub-safe run metadata/plots into a small index folder."""

    run_dir = Path(run_dir).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    experiment_name = _experiment_name_for_run(run_dir)
    run_name = run_dir.name
    index_dir = project_root / index_root / experiment_name / run_name
    index_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    skipped: list[str] = []
    for source in sorted(p for p in run_dir.iterdir() if p.is_file()):
        if source.name in RAW_DATA_FILENAMES or source.suffix == ".jsonl":
            skipped.append(f"{source.name}: raw dataset file")
            continue
        if source.stat().st_size > int(max_file_bytes):
            skipped.append(f"{source.name}: exceeds {int(max_file_bytes)} bytes")
            continue
        if not _is_lightweight_index_file(source.name):
            skipped.append(f"{source.name}: not a lightweight index artifact")
            continue
        destination = index_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)

    readme = index_dir / "README.md"
    readme.write_text(
        _render_index_readme(
            run_dir=run_dir,
            experiment_name=experiment_name,
            copied_files=[path.name for path in copied],
            skipped_files=skipped,
        ),
        encoding="utf-8",
    )
    copied.append(readme)
    return LightweightIndexResult(index_dir=index_dir, copied_files=copied, skipped_files=skipped)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "pull":
            return _cmd_pull(args)
        if args.command == "manifest":
            return _cmd_manifest(args)
        if args.command == "index":
            return _cmd_index(args)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return int(exc.returncode or 1)
    except Exception as exc:
        print(f"Dataset sync failed: {exc}", file=sys.stderr)
        return 1
    parser.error("Choose a command.")
    return 2


def _cmd_pull(args: argparse.Namespace) -> int:
    ssh_target = str(args.pi or "").strip()
    if not ssh_target:
        raise ValueError("--pi is required, e.g. --pi pi@continuum-pi.local")
    remote_root = PurePosixPath(str(args.remote_project_root or DEFAULT_REMOTE_PROJECT_ROOT))
    run_value = str(args.run or "latest")
    if run_value == "latest":
        run_relative = resolve_latest_remote_run_relative_path(
            ssh_target=ssh_target,
            remote_project_root=remote_root,
            experiment=str(args.experiment),
        )
    else:
        run_relative = resolve_run_relative_path(experiment=str(args.experiment), run=run_value)
    remote_run_path = remote_root / run_relative
    local_run_dir = (
        Path(args.local_run_dir).expanduser().resolve()
        if args.local_run_dir
        else local_run_dir_for_relative_path(
            local_mirror_root=Path(args.local_mirror_root),
            run_relative_path=run_relative,
        )
    )
    plan = build_rsync_pull_plan(
        ssh_target=ssh_target,
        remote_run_path=remote_run_path,
        local_run_dir=local_run_dir,
        dry_run=bool(args.dry_run),
        compress=bool(args.compress),
        delete=bool(args.delete),
    )
    print("Rsync command:")
    print(plan.shell_command)
    if bool(args.print_only):
        return 0
    run_rsync_plan(plan)
    if bool(args.dry_run):
        return 0
    manifest = write_dataset_sync_manifest(
        run_dir=local_run_dir,
        include_sha256=bool(args.sha256),
    )
    print(f"Synced run: {local_run_dir}")
    print(f"Manifest: {manifest.manifest_path}")
    print(f"Files: {manifest.file_count} | Size: {_format_bytes(manifest.total_size_bytes)}")
    if not bool(args.no_index):
        index = publish_lightweight_index(
            run_dir=local_run_dir,
            project_root=Path(args.project_root),
            index_root=Path(args.index_root),
            max_file_bytes=int(args.index_max_mb * 1024 * 1024),
        )
        print(f"Lightweight index: {index.index_dir}")
        print(f"Indexed files: {len(index.copied_files)}")
    print("Training path:")
    print(local_run_dir)
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    manifest = write_dataset_sync_manifest(
        run_dir=Path(args.run_dir),
        include_sha256=bool(args.sha256),
    )
    print(f"Manifest: {manifest.manifest_path}")
    print(f"Files: {manifest.file_count} | Size: {_format_bytes(manifest.total_size_bytes)}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    index = publish_lightweight_index(
        run_dir=Path(args.run_dir),
        project_root=Path(args.project_root),
        index_root=Path(args.index_root),
        max_file_bytes=int(args.index_max_mb * 1024 * 1024),
    )
    print(f"Lightweight index: {index.index_dir}")
    print(f"Copied files: {len(index.copied_files)}")
    if index.skipped_files:
        print(f"Skipped files: {len(index.skipped_files)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync huge experiment run folders directly between the Pi and this machine with rsync."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pull = subparsers.add_parser("pull", help="Pull one run folder from the Pi to a local mirror.")
    pull.add_argument("--pi", help="SSH target, e.g. pi@continuum-pi.local. Required for pull.")
    pull.add_argument("--experiment", default="collect_pose_command_dataset", help="Experiment name for --run latest or run IDs.")
    pull.add_argument("--run", default="latest", help="'latest', a run folder name, or a data/experiments/... relative path.")
    pull.add_argument("--remote-project-root", default=str(DEFAULT_REMOTE_PROJECT_ROOT), help="Project root path on the Pi.")
    pull.add_argument("--local-mirror-root", default=str(DEFAULT_LOCAL_MIRROR_ROOT), help="Local raw-data mirror root.")
    pull.add_argument("--local-run-dir", help="Override local destination directory for the run.")
    pull.add_argument("--project-root", default=".", help="This repo root for lightweight index output.")
    pull.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT), help="Repo-relative lightweight index root.")
    pull.add_argument("--index-max-mb", type=float, default=10.0, help="Max single file size copied into the index.")
    pull.add_argument("--no-index", action="store_true", help="Do not publish the lightweight GitHub-safe index.")
    pull.add_argument("--sha256", action="store_true", help="Hash every file after sync. Slow for 100GB runs.")
    pull.add_argument("--compress", action="store_true", help="Pass -z to rsync. Useful on slow networks; slower on weak CPUs.")
    pull.add_argument("--delete", action="store_true", help="Mirror deletes from Pi into local copy. Off by default for safety.")
    pull.add_argument("--dry-run", action="store_true", help="Show what rsync would transfer without copying.")
    pull.add_argument("--print-only", action="store_true", help="Print the rsync command without executing it.")

    manifest = subparsers.add_parser("manifest", help="Write dataset_sync_manifest.json for a local run folder.")
    manifest.add_argument("--run-dir", required=True, help="Local run directory.")
    manifest.add_argument("--sha256", action="store_true", help="Hash every file. Slow for 100GB runs.")

    index = subparsers.add_parser("index", help="Publish a GitHub-safe summary/index for a local run folder.")
    index.add_argument("--run-dir", required=True, help="Local run directory.")
    index.add_argument("--project-root", default=".", help="This repo root.")
    index.add_argument("--index-root", default=str(DEFAULT_INDEX_ROOT), help="Repo-relative lightweight index root.")
    index.add_argument("--index-max-mb", type=float, default=10.0, help="Max single file size copied into the index.")
    return parser


def _experiment_name_for_run(run_dir: Path) -> str:
    summary_path = Path(run_dir) / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            experiment = str(payload.get("experiment_name") or "").strip()
            if experiment:
                return sanitize_output_name(experiment, default="experiment")
        except Exception:
            pass
    parent = Path(run_dir).parent.name
    if parent:
        return sanitize_output_name(parent, default="experiment")
    return "experiment"


def _is_lightweight_index_file(name: str) -> bool:
    if name in LIGHTWEIGHT_EXACT_FILENAMES:
        return True
    return any(name.endswith(suffix) for suffix in LIGHTWEIGHT_SUFFIXES)


def _render_index_readme(
    *,
    run_dir: Path,
    experiment_name: str,
    copied_files: Sequence[str],
    skipped_files: Sequence[str],
) -> str:
    lines = [
        f"# Synced Run Index: {Path(run_dir).name}",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Raw data location on this machine: `{Path(run_dir)}`",
        "- Raw JSONL files are intentionally not copied here.",
        "- Use `scripts/sync_pi_dataset.py pull` to recreate/update the raw local mirror.",
        "",
        "## Included",
    ]
    lines.extend(f"- `{name}`" for name in sorted(copied_files))
    if skipped_files:
        lines.extend(["", "## Skipped"])
        lines.extend(f"- {item}" for item in skipped_files[:50])
        if len(skipped_files) > 50:
            lines.append(f"- ... {len(skipped_files) - 50} more")
    return "\n".join(lines).strip() + "\n"


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else value + "/"


def _format_bytes(value: int) -> str:
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{int(value)} B"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
