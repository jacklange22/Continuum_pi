"""Operator-facing hardware readiness rehearsal without requiring hardware."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

from continuum_robot.config.config_loader import ConfigLoader
from continuum_robot.diagnostics.prehardware_dry_run import (
    DryRunCheck,
    render_dry_run_report,
    run_prehardware_dry_run,
)
from continuum_robot.experiments.dataset_io import canonical_timestamp_token
from continuum_robot.hardware.serial_ports import discover_serial_ports


@dataclass(frozen=True)
class HardwareReadinessCheck:
    """One hardware-day readiness finding."""

    subsystem: str
    status: str
    message: str
    next_action: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareReadinessReport:
    """Complete hardware readiness rehearsal output."""

    output_dir: Path
    checks: list[HardwareReadinessCheck]

    @property
    def passed(self) -> bool:
        return not any(check.status == "FAIL" for check in self.checks)


def run_hardware_readiness_check(
    *,
    project_root: Path,
    output_root: Path | None = None,
    include_prehardware_dry_run: bool = True,
) -> HardwareReadinessReport:
    """Run no-hardware checks that mirror hardware-day operator decisions."""

    project_root = Path(project_root)
    output_dir = Path(output_root or project_root / "data" / "diagnostics" / "hardware_readiness" / canonical_timestamp_token())
    output_dir.mkdir(parents=True, exist_ok=False)
    checks: list[HardwareReadinessCheck] = []
    settings = None
    loader = ConfigLoader(base_dir=project_root / "config")
    try:
        settings = loader.load_settings()
        context = settings.robot.operating_context()
        checks.append(
            HardwareReadinessCheck(
                subsystem="config",
                status="PASS",
                message=(
                    f"Config loaded. robot_config={settings.runtime.robot_config}, "
                    f"mode={context.operating_mode}, expected_servo_ids={context.expected_servo_ids}."
                ),
                next_action="Confirm the GUI System page shows the same operating mode before connecting hardware.",
                details={"warnings": list(loader.last_warnings), "mode_metadata": context.metadata()},
            )
        )
    except Exception as exc:
        checks.append(
            HardwareReadinessCheck(
                subsystem="config",
                status="FAIL",
                message=f"Config could not be loaded: {exc}",
                next_action="Fix config/system.yaml, config/system.local.yaml, and the selected robot profile.",
            )
        )
    if settings is not None:
        checks.append(_serial_readiness_check(settings=settings))
        checks.append(_tracking_role_check(settings=settings))
        checks.append(_physics_config_check(project_root=project_root))
    checks.append(_gui_construction_check())
    if include_prehardware_dry_run:
        checks.extend(_prehardware_dry_run_checks(project_root=project_root, output_dir=output_dir))
    report = HardwareReadinessReport(output_dir=output_dir, checks=checks)
    (output_dir / "hardware_readiness_summary.txt").write_text(render_hardware_readiness_report(report), encoding="utf-8")
    (output_dir / "hardware_readiness_summary.json").write_text(
        json.dumps(
            {
                "passed": report.passed,
                "output_dir": str(output_dir),
                "checks": [check.__dict__ for check in report.checks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def render_hardware_readiness_report(report: HardwareReadinessReport) -> str:
    lines = [
        "Hardware Readiness Check",
        f"Overall: {'PASS' if report.passed else 'FAIL'}",
        f"Output: {report.output_dir}",
        "",
    ]
    for check in report.checks:
        lines.append(f"{check.status}: {check.subsystem} - {check.message}")
        if check.next_action:
            lines.append(f"  next: {check.next_action}")
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run hardware-day software readiness checks without connecting hardware.")
    parser.add_argument("--project-root", default=".", help="Repository root.")
    parser.add_argument("--output-root", help="Optional output directory for readiness artifacts.")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip the nested prehardware dry-run fixture/export check.")
    args = parser.parse_args(argv)
    report = run_hardware_readiness_check(
        project_root=Path(args.project_root).resolve(),
        output_root=Path(args.output_root).resolve() if args.output_root else None,
        include_prehardware_dry_run=not bool(args.skip_dry_run),
    )
    print(render_hardware_readiness_report(report))
    return 0 if report.passed else 1


def _serial_readiness_check(*, settings) -> HardwareReadinessCheck:
    ports = []
    try:
        ports = discover_serial_ports()
    except Exception as exc:
        return HardwareReadinessCheck(
            subsystem="serial_ports",
            status="WARN",
            message=f"Serial port discovery was unavailable in this environment: {exc}",
            next_action="On the Pi, confirm Aurora and OpenRB ports on the System page before connecting.",
        )
    port_names = [port.device for port in ports]
    aurora_port = str(settings.serial.aurora_port or "")
    openrb_port = str(settings.serial.openrb_port or "")
    if aurora_port and openrb_port and aurora_port == openrb_port:
        return HardwareReadinessCheck(
            subsystem="serial_ports",
            status="FAIL",
            message=f"Aurora and OpenRB are configured to the same port: {aurora_port}.",
            next_action="Choose distinct tracker and OpenRB ports; OpenRB candidate logic should not reuse the tracker port.",
            details={"available_ports": port_names},
        )
    if settings.runtime.mock_mode:
        return HardwareReadinessCheck(
            subsystem="serial_ports",
            status="WARN",
            message="Config is in mock_mode; hardware connect buttons should remain safe rehearsals until mock_mode is disabled.",
            next_action="For hardware day, set mock_mode=false in system.local.yaml and verify OpenRB/Aurora ports.",
            details={"available_ports": port_names, "aurora_port": aurora_port, "openrb_port": openrb_port},
        )
    missing = [label for label, value in (("aurora_port", aurora_port), ("openrb_port", openrb_port)) if not value]
    if missing:
        return HardwareReadinessCheck(
            subsystem="serial_ports",
            status="WARN",
            message="Hardware mode is active but configured port(s) are empty: " + ", ".join(missing),
            next_action="Select ports on the System page or update config/system.local.yaml.",
            details={"available_ports": port_names},
        )
    return HardwareReadinessCheck(
        subsystem="serial_ports",
        status="PASS",
        message="Configured tracker/OpenRB ports are distinct and non-empty.",
        next_action="Connect tracker first, then OpenRB, and confirm expected servo IDs.",
        details={"available_ports": port_names, "aurora_port": aurora_port, "openrb_port": openrb_port},
    )


def _tracking_role_check(*, settings) -> HardwareReadinessCheck:
    roles = dict(getattr(settings.registration, "two_segment_tracking_roles", {}) or {})
    required = ["registration_probe", "distal_tip", "intermediate_segment", "debug_tool"]
    present = sorted(str(key) for key in roles)
    missing = [role for role in required if role not in roles]
    status = "PASS" if "distal_tip" in roles else "WARN"
    message = (
        f"Two-segment tracking role config present: {present}."
        if roles
        else "Two-segment tracking roles are not configured."
    )
    next_action = (
        "Before trusted two-segment datasets, confirm distal_tip live status and intermediate_segment if using two-coil labels."
        if roles
        else "Configure tracking roles before trusted two-segment model-training data collection."
    )
    return HardwareReadinessCheck(
        subsystem="tracking_roles",
        status=status,
        message=message,
        next_action=next_action,
        details={"present_roles": present, "missing_optional_or_required_roles": missing},
    )


def _physics_config_check(*, project_root: Path) -> HardwareReadinessCheck:
    path = project_root / "config" / "modeling_two_segment.example.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return HardwareReadinessCheck(
            subsystem="two_segment_physics_config",
            status="FAIL",
            message=f"Two-segment modeling config could not be loaded: {exc}",
            next_action="Fix config/modeling_two_segment.example.yaml.",
        )
    physics = dict(dict(payload or {}).get("physics_models", {}) or {})
    mike = dict(physics.get("mike_constant_curvature", {}) or {})
    segments = dict(physics.get("segments", {}) or {})
    geometry_known = bool(mike.get("geometry_complete")) and all(
        _wrapped_value(dict(segments.get(segment, {}) or {}).get("segment_length_mm")) is not None
        for segment in ("segment_a", "segment_b")
    )
    status = "PASS" if geometry_known else "WARN"
    return HardwareReadinessCheck(
        subsystem="two_segment_physics_config",
        status=status,
        message=(
            "Design geometry is present; Mike remains gated until required_conventions_confirmed and hardware validation."
            if geometry_known
            else "Two-segment design geometry is incomplete."
        ),
        next_action="Use tracked single-axis motions to confirm sign/frame conventions before enabling Mike predictions.",
        details={
            "geometry_complete": mike.get("geometry_complete"),
            "required_conventions_confirmed": mike.get("required_conventions_confirmed"),
            "frame_convention_hardware_validated": mike.get("frame_convention_hardware_validated"),
        },
    )


def _gui_construction_check() -> HardwareReadinessCheck:
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from continuum_robot.app.bootstrap import build_app_context
        from continuum_robot.gui.app_window import AppWindow

        app = QApplication.instance() or QApplication([])
        _ = app
        window = AppWindow(build_app_context())
        labels = [window.tab_widget.tabText(index) for index in range(window.tab_widget.count())]
        window.close()
    except Exception as exc:
        return HardwareReadinessCheck(
            subsystem="gui_pages",
            status="WARN",
            message=f"Full GUI construction skipped/failed in this environment: {exc}",
            next_action="Run scripts/run_tests.sh gui or launch scripts/run_gui.sh on the operator machine.",
        )
    expected = ["System", "Tracking", "Registration", "Servos", "Pretension", "Experiment", "Modeling", "Data"]
    missing = [label for label in expected if label not in labels]
    if missing:
        return HardwareReadinessCheck(
            subsystem="gui_pages",
            status="FAIL",
            message=f"GUI is missing expected page(s): {missing}",
            next_action="Fix AppWindow tab construction before hardware day.",
            details={"labels": labels},
        )
    return HardwareReadinessCheck(
        subsystem="gui_pages",
        status="PASS",
        message="System, Tracking, Registration, Servos, Pretension, Experiment, Modeling, and Data pages construct offscreen.",
        next_action="On hardware day, use the System page first to select mode and connect tracker/OpenRB.",
        details={"labels": labels},
    )


def _prehardware_dry_run_checks(*, project_root: Path, output_dir: Path) -> list[HardwareReadinessCheck]:
    try:
        dry_report = run_prehardware_dry_run(project_root=project_root, output_root=output_dir / "prehardware_dry_run")
    except Exception as exc:
        return [
            HardwareReadinessCheck(
                subsystem="prehardware_dry_run",
                status="FAIL",
                message=f"Prehardware dry run failed unexpectedly: {exc}",
                next_action="Run .venv/bin/python -m continuum_robot.diagnostics.prehardware_dry_run and fix the first failing check.",
            )
        ]
    rendered = render_dry_run_report(dry_report)
    status = "PASS" if dry_report.passed else "FAIL"
    checks = [
        HardwareReadinessCheck(
            subsystem="prehardware_dry_run",
            status=status,
            message=f"Nested prehardware dry run completed with {len(dry_report.checks)} check(s).",
            next_action="Inspect prehardware_dry_run_summary.txt for export/validator details.",
            details={"output_dir": str(dry_report.output_dir), "summary": rendered},
        )
    ]
    for check in dry_report.checks:
        checks.append(_from_dry_run_check(check))
    return checks


def _from_dry_run_check(check: DryRunCheck) -> HardwareReadinessCheck:
    return HardwareReadinessCheck(
        subsystem=f"prehardware:{check.name}",
        status=check.status,
        message=check.message,
        next_action="Resolve before hardware day." if check.status == "FAIL" else "",
        details={"paths": [str(path) for path in check.paths]},
    )


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
