from __future__ import annotations

from pathlib import Path

from continuum_robot.diagnostics.hardware_readiness_check import (
    render_hardware_readiness_report,
    run_hardware_readiness_check,
)
from continuum_robot.diagnostics.prehardware_dry_run import render_dry_run_report, run_prehardware_dry_run


def test_prehardware_dry_run_command_checks_core_paths(tmp_path: Path) -> None:
    report = run_prehardware_dry_run(project_root=tmp_path, output_root=tmp_path / "dry_run")
    text = render_dry_run_report(report)

    assert report.passed
    assert "Config / operating modes" in text
    assert "Export bundle" in text
    assert (report.output_dir / "prehardware_dry_run_summary.json").exists()
    assert list((report.output_dir / "exports_no_samples").glob("*.zip"))


def test_hardware_readiness_check_reports_operator_subsystems(tmp_path: Path) -> None:
    report = run_hardware_readiness_check(
        project_root=Path.cwd(),
        output_root=tmp_path / "hardware_readiness",
        include_prehardware_dry_run=False,
    )
    text = render_hardware_readiness_report(report)

    assert report.passed
    assert "Hardware Readiness Check" in text
    assert "config" in text
    assert "serial_ports" in text
    assert "gui_pages" in text
    assert (report.output_dir / "hardware_readiness_summary.json").exists()
