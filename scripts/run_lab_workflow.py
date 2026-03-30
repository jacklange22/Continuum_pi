#!/usr/bin/env python3
"""Unified wrapper around the common operator and validation workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class WorkflowCommand:
    """One top-level lab workflow command."""

    name: str
    target: str
    description: str


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMMANDS: dict[str, WorkflowCommand] = {
    "gui": WorkflowCommand("gui", "scripts/run_gui.py", "Launch the operator GUI."),
    "tracker-mvp": WorkflowCommand(
        "tracker-mvp",
        "scripts/run_tracker_mvp.py",
        "Launch the focused tracker-first MVP GUI workflow.",
    ),
    "tracker-doctor": WorkflowCommand(
        "tracker-doctor",
        "scripts/run_tracker_doctor.py",
        "Run the canonical tracker diagnostics.",
    ),
    "tracker-smoke": WorkflowCommand(
        "tracker-smoke",
        "scripts/run_tracker_smoke.py",
        "Run the short tracker smoke test.",
    ),
    "tracker-benchmark": WorkflowCommand(
        "tracker-benchmark",
        "scripts/run_tracker_benchmark.py",
        "Benchmark tracking performance and freshness.",
    ),
    "registration-runtime-sanity": WorkflowCommand(
        "registration-runtime-sanity",
        "scripts/run_registration_runtime_sanity.py",
        "Validate runtime tip-pose computation from saved registration plus tracker data.",
    ),
    "registration-validation": WorkflowCommand(
        "registration-validation",
        "scripts/run_registration_validation.py",
        "Validate registration outputs from a CSV or saved session artifact.",
    ),
    "registration-from-csv": WorkflowCommand(
        "registration-from-csv",
        "scripts/run_registration_from_csv.py",
        "Solve and save registration directly from a saved Aurora CSV.",
    ),
    "compare-registration": WorkflowCommand(
        "compare-registration",
        "scripts/compare_registration_outputs.py",
        "Compare two registration outputs numerically.",
    ),
    "experiment": WorkflowCommand(
        "experiment",
        "scripts/run_experiment.py",
        "Run the canonical experiment CLI.",
    ),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a common continuum-robot workflow from one entry point.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="list",
        choices=["list", *sorted(COMMANDS)],
        help="Workflow name, or `list` to print the available commands.",
    )
    parser.add_argument(
        "command_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying script. Prefix with `--` when needed.",
    )
    return parser.parse_args(argv)


def build_command(command_name: str, command_args: list[str] | None = None) -> list[str]:
    """Return the concrete command for one workflow."""
    if command_name not in COMMANDS:
        available = ", ".join(sorted(COMMANDS))
        raise KeyError(f"Unknown workflow {command_name!r}. Available workflows: {available}")
    command = COMMANDS[command_name]
    forwarded = list(command_args or [])
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return [sys.executable, str(PROJECT_ROOT / command.target), *forwarded]


def _render_command_list() -> str:
    lines = ["Available workflows:"]
    for command_name in sorted(COMMANDS):
        command = COMMANDS[command_name]
        lines.append(f"  {command.name:<28} {command.description}")
    lines.append("")
    lines.append("Example:")
    lines.append(
        "  python scripts/run_lab_workflow.py tracker-smoke -- --tracker-port /dev/ttyUSB0"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "list":
        print(_render_command_list())
        return 0

    command = build_command(args.command, args.command_args)
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0

    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
