#!/usr/bin/env python3
"""Check runtime data folders for GitHub-safe commit hygiene."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

from continuum_robot.data.run_management import discover_experiment_run_dirs, discover_run_dirs, load_run_review, summarize_run


DEFAULT_WARN_BYTES = 50 * 1024 * 1024
DEFAULT_FAIL_BYTES = 100 * 1024 * 1024
FORBIDDEN_GENERATED_DIRS = (
    Path("data/exports"),
    Path("data/trash"),
    Path("data/diagnostics"),
    Path("data/mock_experiments"),
    Path("data/mock_calibration"),
    Path("data/calibration/servo_calibration"),
    Path("data/runtime_tip_calibration"),
    Path("data/pivot_calibration"),
)
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z"}
IMPORTANT_REVIEW_STATUSES = {"thesis_candidate", "advisor_share", "keep"}


@dataclass(frozen=True)
class HygieneFinding:
    level: str
    path: str
    message: str


@dataclass
class HygieneReport:
    status: str
    findings: list[HygieneFinding] = field(default_factory=list)
    run_status_counts: dict[str, int] = field(default_factory=dict)
    experiment_run_count: int = 0
    archived_run_count: int = 0
    tracked_data_file_count: int = 0
    untracked_data_file_count: int = 0


def run_data_hygiene_check(
    *,
    project_root: Path,
    warn_bytes: int = DEFAULT_WARN_BYTES,
    fail_bytes: int = DEFAULT_FAIL_BYTES,
) -> HygieneReport:
    """Return a non-mutating report for data folders that are risky to commit."""
    project_root = Path(project_root).resolve()
    findings: list[HygieneFinding] = []
    git_info = _git_data_status(project_root)

    if _git_check_ignored(project_root, Path("data/experiments/__hygiene_check__")):
        findings.append(
            HygieneFinding(
                "FAIL",
                "data/experiments/",
                "data/experiments is ignored; important experiment runs would be hidden from Git.",
            )
        )
    if _git_check_ignored(project_root, Path("data/experiments_archived/__hygiene_check__")):
        findings.append(
            HygieneFinding(
                "WARN",
                "data/experiments_archived/",
                "data/experiments_archived is ignored; archived but important runs cannot be committed.",
            )
        )
    if (project_root / ".git").exists():
        for ignored_root in (
            Path("data/mock_experiments/__hygiene_check__"),
            Path("data/exports/__hygiene_check__"),
            Path("data/trash/__hygiene_check__"),
            Path("data/diagnostics/__hygiene_check__"),
            Path("data/mock_calibration/__hygiene_check__"),
        ):
            if not _git_check_ignored(project_root, ignored_root):
                findings.append(
                    HygieneFinding(
                        "WARN",
                        str(ignored_root.parent) + "/",
                        "Generated/mock/export/trash/diagnostic root is not ignored; add it to .gitignore before syncing.",
                    )
                )

    run_status_counts: dict[str, int] = {}
    experiment_runs = discover_experiment_run_dirs(project_root)
    archived_runs = _discover_archived_run_dirs(project_root)
    mock_runs = discover_run_dirs(project_root, root_name="mock_experiments")
    for run_dir in [*experiment_runs, *archived_runs]:
        review = load_run_review(run_dir)
        run_status_counts[review.review_status] = run_status_counts.get(review.review_status, 0) + 1
        if not (run_dir / "run_review.json").exists():
            findings.append(
                HygieneFinding("WARN", _rel(project_root, run_dir), "Run is missing run_review.json classification.")
            )
        try:
            summary = summarize_run(run_dir)
        except Exception as exc:
            findings.append(HygieneFinding("WARN", _rel(project_root, run_dir), f"Could not summarize run trust metadata: {exc}"))
            summary = None
        if summary is not None:
            if run_dir in experiment_runs and (summary.mock_mode is True or summary.run_trust_mode == "mock"):
                findings.append(
                    HygieneFinding(
                        "FAIL",
                        _rel(project_root, run_dir),
                        "Mock/debug run is in data/experiments; move it under data/mock_experiments or regenerate with mock root.",
                    )
                )
            if run_dir in experiment_runs and review.review_status == "debug":
                findings.append(
                    HygieneFinding(
                        "WARN",
                        _rel(project_root, run_dir),
                        "Real experiment run is still unreviewed/debug; mark keep/thesis_candidate/advisor_share only after inspection.",
                    )
                )
        if review.review_status in IMPORTANT_REVIEW_STATUSES:
            _check_candidate_run(project_root, run_dir, findings)
    for run_dir in mock_runs:
        review = load_run_review(run_dir)
        run_status_counts[review.review_status] = run_status_counts.get(review.review_status, 0) + 1
        if review.include_in_evidence_index or review.review_status in IMPORTANT_REVIEW_STATUSES:
            findings.append(
                HygieneFinding(
                    "FAIL",
                    _rel(project_root, run_dir),
                    "Mock run is marked for evidence/review inclusion; set include_in_evidence_index=false and review_status=debug/garbage.",
                )
            )

    for path in _iter_data_files(project_root):
        relative = _relative_path(project_root, path)
        size = path.stat().st_size
        if size >= fail_bytes:
            findings.append(
                HygieneFinding("FAIL", str(relative), f"File is {_format_bytes(size)}, at or above the fail threshold.")
            )
        elif size >= warn_bytes:
            findings.append(
                HygieneFinding("WARN", str(relative), f"File is {_format_bytes(size)}, above the warning threshold.")
            )
        if _is_generated_or_forbidden(relative):
            level = "FAIL" if str(relative) in git_info.tracked_files else "WARN"
            findings.append(
                HygieneFinding(
                    level,
                    str(relative),
                    "Generated/mock/export/trash/diagnostic/calibration artifact should not be committed; keep it ignored or move reviewed real data to data/experiments.",
                )
            )
        _check_calibration_artifact(project_root, path, relative, findings)

    status = "PASS"
    if any(item.level == "FAIL" for item in findings):
        status = "FAIL"
    elif findings:
        status = "WARN"
    return HygieneReport(
        status=status,
        findings=findings,
        run_status_counts=run_status_counts,
        experiment_run_count=len(experiment_runs),
        archived_run_count=len(archived_runs),
        tracked_data_file_count=len(git_info.tracked_files),
        untracked_data_file_count=len(git_info.untracked_files),
    )


def render_report(report: HygieneReport, *, max_findings: int = 80) -> str:
    lines = [
        f"{report.status}: data Git hygiene check",
        f"Experiment runs: {report.experiment_run_count}",
        f"Archived runs: {report.archived_run_count}",
        f"Tracked data files: {report.tracked_data_file_count}",
        f"Untracked data files: {report.untracked_data_file_count}",
        "Review statuses: "
        + (", ".join(f"{key}={value}" for key, value in sorted(report.run_status_counts.items())) or "none"),
    ]
    if report.findings:
        lines.append("Findings:")
        for finding in report.findings[:max_findings]:
            lines.append(f"- {finding.level}: {finding.path}: {finding.message}")
        if len(report.findings) > max_findings:
            lines.append(f"... {len(report.findings) - max_findings} more finding(s) omitted")
    else:
        lines.append("No data hygiene issues found.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report data files that are risky to commit to GitHub.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument("--warn-mb", type=float, default=50.0, help="Warn at or above this file size in MB.")
    parser.add_argument("--fail-mb", type=float, default=100.0, help="Fail at or above this file size in MB.")
    parser.add_argument("--warn-bytes", type=int, help="Test/helper override for warning byte threshold.")
    parser.add_argument("--fail-bytes", type=int, help="Test/helper override for failure byte threshold.")
    parser.add_argument("--max-findings", type=int, default=80, help="Maximum findings to print.")
    parser.add_argument("--fail-on-warn", action="store_true", help="Return nonzero when warnings are present.")
    args = parser.parse_args(argv)
    if args.fail_bytes is not None:
        fail_bytes = int(args.fail_bytes)
    else:
        fail_bytes = int(float(args.fail_mb) * 1024 * 1024)
    if args.warn_bytes is not None:
        warn_bytes = int(args.warn_bytes)
    else:
        warn_bytes = int(float(args.warn_mb) * 1024 * 1024)
    report = run_data_hygiene_check(project_root=Path(args.project_root), warn_bytes=warn_bytes, fail_bytes=fail_bytes)
    print(render_report(report, max_findings=int(args.max_findings)))
    if report.status == "FAIL" or (args.fail_on_warn and report.status == "WARN"):
        return 1
    return 0


@dataclass(frozen=True)
class _GitDataStatus:
    tracked_files: set[str] = field(default_factory=set)
    untracked_files: set[str] = field(default_factory=set)


def _git_data_status(project_root: Path) -> _GitDataStatus:
    tracked = _run_git(project_root, ["ls-files", "data"])
    status = _run_git(project_root, ["status", "--porcelain", "--", "data"])
    tracked_files = set(tracked.splitlines()) if tracked else set()
    untracked_files: set[str] = set()
    for line in status.splitlines():
        if line.startswith("?? "):
            untracked_files.add(line[3:].strip())
    return _GitDataStatus(tracked_files=tracked_files, untracked_files=untracked_files)


def _git_check_ignored(project_root: Path, relative_path: Path) -> bool:
    if not (project_root / ".git").exists():
        return False
    command = ["git", "check-ignore", "-q", "--", str(relative_path)]
    completed = subprocess.run(command, cwd=project_root, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return completed.returncode == 0


def _run_git(project_root: Path, args: list[str]) -> str:
    if not (project_root / ".git").exists():
        return ""
    completed = subprocess.run(["git", *args], cwd=project_root, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _discover_archived_run_dirs(project_root: Path) -> list[Path]:
    archived_root = Path(project_root) / "data" / "experiments_archived"
    if not archived_root.exists():
        return []
    runs: list[Path] = []
    for experiment_root in archived_root.iterdir():
        if not experiment_root.is_dir():
            continue
        for candidate in experiment_root.iterdir():
            if candidate.is_dir() and ((candidate / "summary.json").exists() or (candidate / "metadata.json").exists()):
                runs.append(candidate)
    return sorted(runs, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def _check_candidate_run(project_root: Path, run_dir: Path, findings: list[HygieneFinding]) -> None:
    try:
        summary = summarize_run(run_dir)
    except Exception as exc:
        findings.append(HygieneFinding("FAIL", _rel(project_root, run_dir), f"Could not summarize candidate run: {exc}"))
        return
    if summary.validation_status != "PASS":
        findings.append(
            HygieneFinding(
                "WARN",
                _rel(project_root, run_dir),
                f"Candidate run validation is {summary.validation_status}; inspect trust/provenance before commit.",
            )
        )
    if not summary.report_figures:
        findings.append(HygieneFinding("WARN", _rel(project_root, run_dir), "Candidate run has no *_report.png figures."))
    if summary.run_trust_mode in {"unknown", ""}:
        findings.append(HygieneFinding("WARN", _rel(project_root, run_dir), "Candidate run is missing run_trust_mode."))
    if summary.run_trust_mode in {"mock", "servo_only", "current_only", "lower_trust", "debug", "debug_only"}:
        findings.append(
            HygieneFinding(
                "FAIL",
                _rel(project_root, run_dir),
                f"Candidate run has non-thesis trust mode {summary.run_trust_mode}; do not include in evidence/GitHub sync.",
            )
        )
    if summary.mock_mode is True:
        findings.append(HygieneFinding("FAIL", _rel(project_root, run_dir), "Candidate run was produced in mock_mode."))


def _check_calibration_artifact(project_root: Path, path: Path, relative: Path, findings: list[HygieneFinding]) -> None:
    relative_text = str(relative)
    if not (
        relative_text.startswith("data/mock_calibration/")
        or relative_text.startswith("data/calibration/servo_calibration/")
        or relative_text.startswith("data/runtime_tip_calibration/")
        or relative_text.startswith("data/pivot_calibration/")
    ):
        return
    payload = _read_json(path)
    robot = payload.get("robot") if isinstance(payload.get("robot"), dict) else {}
    trust = str(robot.get("calibration_trust") or payload.get("calibration_trust") or "").strip().lower()
    mock_mode = bool(robot.get("mock_mode", payload.get("mock_mode", False)))
    if relative_text.startswith("data/mock_calibration/") or mock_mode or trust == "mock":
        findings.append(
            HygieneFinding(
                "FAIL",
                relative_text,
                "Mock calibration artifact detected; keep it under data/mock_calibration and do not sync as hardware startup truth.",
            )
        )
        return
    if relative_text.startswith("data/calibration/servo_calibration/") and trust not in {"hardware", "reviewed_hardware"}:
        findings.append(
            HygieneFinding(
                "WARN",
                relative_text,
                "Servo calibration archive lacks calibration_trust=hardware/reviewed_hardware; review before committing.",
            )
        )


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_data_files(project_root: Path):
    data_root = Path(project_root) / "data"
    if not data_root.exists():
        return
    for path in data_root.rglob("*"):
        if path.is_file():
            yield path


def _is_generated_or_forbidden(relative_path: Path) -> bool:
    for forbidden in FORBIDDEN_GENERATED_DIRS:
        try:
            relative_path.relative_to(forbidden)
            return True
        except ValueError:
            pass
    return relative_path.suffix.lower() in ARCHIVE_SUFFIXES


def _relative_path(project_root: Path, path: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path


def _rel(project_root: Path, path: Path) -> str:
    return str(_relative_path(project_root, path))


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
