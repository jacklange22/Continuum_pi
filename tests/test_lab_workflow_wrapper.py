from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_lab_workflow.py"


def test_lab_workflow_wrapper_lists_available_commands() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "list"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "tracker-mvp" in completed.stdout
    assert "tracker-smoke" in completed.stdout
    assert "registration-runtime-sanity" in completed.stdout
    assert "experiment" in completed.stdout


def test_lab_workflow_wrapper_dry_run_preserves_forwarded_args() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "tracker-smoke",
            "--",
            "--tracker-port",
            "/dev/ttyUSB0",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scripts/run_tracker_smoke.py" in completed.stdout
    assert "--tracker-port" in completed.stdout
    assert "/dev/ttyUSB0" in completed.stdout
