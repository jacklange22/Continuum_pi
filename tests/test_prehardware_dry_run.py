from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from continuum_robot.diagnostics.hardware_readiness_check import (
    _dual_segment_baud_readiness_check,
    _tracker_backend_fallback_check,
    render_hardware_readiness_report,
    run_hardware_readiness_check,
)
from continuum_robot.config.schemas import RobotConfig, RobotSegmentConfig, SerialConfig
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

    assert "Hardware Readiness Check" in text
    assert "config" in text
    assert "serial_ports" in text
    assert "selected_segment_calibration" in text
    assert "gui_pages" in text
    assert (report.output_dir / "hardware_readiness_summary.json").exists()


def test_tracker_backend_fallback_is_fail_when_not_mock() -> None:
    settings = SimpleNamespace(
        runtime=SimpleNamespace(mock_mode=False),
        serial=SimpleNamespace(
            tracker_backend="ndi",
            tracker_fallback_enabled=True,
            tracker_fallback_backend="bridge",
        ),
    )
    check = _tracker_backend_fallback_check(settings=settings)
    assert check.status == "FAIL"
    assert "trusted" in check.message.lower()
    assert check.details.get("trusted_backend_ready") is False


def test_tracker_backend_fallback_warn_in_mock_mode() -> None:
    settings = SimpleNamespace(
        runtime=SimpleNamespace(mock_mode=True),
        serial=SimpleNamespace(
            tracker_backend="ndi",
            tracker_fallback_enabled=True,
            tracker_fallback_backend="bridge",
        ),
    )
    check = _tracker_backend_fallback_check(settings=settings)
    assert check.status == "WARN"


def test_dual_segment_baud_readiness_recommends_1mbps_for_all8() -> None:
    robot = RobotConfig(
        mode="dual_segment",
        servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
        segments={
            "segment_a": RobotSegmentConfig(
                key="segment_a",
                label="Segment A",
                servo_ids=[1, 2, 3, 4],
                pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
            ),
            "segment_b": RobotSegmentConfig(
                key="segment_b",
                label="Segment B",
                servo_ids=[5, 6, 7, 8],
                pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
            ),
        },
    )
    slow = SimpleNamespace(robot=robot, serial=SerialConfig(baudrate=57600))
    fast = SimpleNamespace(robot=robot, serial=SerialConfig(baudrate=1_000_000))

    slow_check = _dual_segment_baud_readiness_check(settings=slow)
    fast_check = _dual_segment_baud_readiness_check(settings=fast)

    assert slow_check.status == "WARN"
    assert "1 000 000" in slow_check.message
    assert fast_check.status == "PASS"
