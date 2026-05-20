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
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService, ServoCalibrationContext
from continuum_robot.servos.segment_readiness import evaluate_selected_segment_readiness
from continuum_robot.servos.sign_mapping_check import ServoMappingCheckRepository


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
        checks.append(_dual_segment_baud_readiness_check(settings=settings))
        checks.append(_tracker_backend_fallback_check(settings=settings))
        checks.append(_selected_segment_calibration_check(project_root=project_root, settings=settings))
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


def _tracker_backend_fallback_check(*, settings) -> HardwareReadinessCheck:
    fallback_enabled = bool(getattr(settings.serial, "tracker_fallback_enabled", False))
    fallback_backend = str(getattr(settings.serial, "tracker_fallback_backend", "") or "")
    configured_backend = str(getattr(settings.serial, "tracker_backend", "") or "")
    trusted_run_intent = not bool(getattr(settings.runtime, "mock_mode", True))
    if fallback_enabled:
        status = "FAIL" if trusted_run_intent else "WARN"
        return HardwareReadinessCheck(
            subsystem="tracker_backend",
            status=status,
            message=(
                f"Tracker fallback is enabled (configured={configured_backend}, fallback={fallback_backend}). "
                "Trusted hardware runs should use a single explicit backend path to avoid silent backend swaps."
            ),
            next_action=(
                "Set tracker_backend=ndi, tracker_fallback_enabled=false, and tracker_fallback_backend empty "
                "before thesis-trusted runs."
            ),
            details={
                "tracker_backend": configured_backend,
                "tracker_fallback_backend": fallback_backend,
                "trusted_run_intent": trusted_run_intent,
                "trusted_backend_ready": False,
            },
        )
    return HardwareReadinessCheck(
        subsystem="tracker_backend",
        status="PASS",
        message=f"Tracker fallback is disabled; configured backend is {configured_backend or 'default'}.",
        next_action="Confirm live tracking uses the Python NDI backend before thesis-intended runs.",
        details={
            "tracker_backend": configured_backend,
            "tracker_fallback_backend": fallback_backend,
            "trusted_run_intent": trusted_run_intent,
            "trusted_backend_ready": configured_backend in {"", "ndi"},
        },
    )


def _dual_segment_baud_readiness_check(*, settings) -> HardwareReadinessCheck:
    context = settings.robot.operating_context()
    mode = str(context.operating_mode)
    expected_ids = [int(value) for value in list(context.expected_servo_ids or [])]
    baud = int(getattr(settings.serial, "baudrate", 57600) or 57600)
    all8_mode = mode in {"dual_segment", "parallel_single"} or len(expected_ids) >= 8
    if all8_mode and baud < 1_000_000:
        return HardwareReadinessCheck(
            subsystem="dynamixel_baud",
            status="WARN",
            message=(
                f"Configured DYNAMIXEL baud is {baud} bps for {mode}. "
                "Dual-segment/all-8 trusted operation is designed around 1 000 000 bps; "
                "57 600 is debug/legacy/single-segment acceptable but can bottleneck all-8 telemetry."
            ),
            next_action=(
                "Do not change baud blindly. Reflash every servo with DYNAMIXEL Wizard, update config, "
                "then run servo_transport_diagnostic for IDs 1-8 at --baud 1000000."
            ),
            details={"operating_mode": mode, "expected_servo_ids": expected_ids, "baudrate": baud},
        )
    return HardwareReadinessCheck(
        subsystem="dynamixel_baud",
        status="PASS",
        message=f"Configured DYNAMIXEL baud is {baud} bps for {mode}.",
        next_action="For trusted dual-segment work, confirm all eight servos respond at the configured baud.",
        details={"operating_mode": mode, "expected_servo_ids": expected_ids, "baudrate": baud},
    )


def _selected_segment_calibration_check(*, project_root: Path, settings) -> HardwareReadinessCheck:
    context = settings.robot.operating_context()
    neutral_path = project_root / str(settings.calibration.neutral_setpoints_path or "config/neutral_setpoints.json")
    service = NeutralCalibrationService(
        path=neutral_path,
        context=_calibration_context_from_settings(settings),
    )
    summary = service.get_calibration_summary()
    sign_mapping = None
    if context.operating_mode == "single_segment":
        sign_mapping = ServoMappingCheckRepository(
            project_root / "data" / "calibration" / "servo_mapping_checks"
        ).latest_for_segment(
            active_segment_key=str(context.active_segment_key or ""),
            expected_servo_ids=[int(value) for value in context.expected_servo_ids],
        )
    readiness = evaluate_selected_segment_readiness(
        operating_mode=context.operating_mode,
        active_segment_key=context.active_segment_key,
        active_segment_label=context.active_segment_label,
        expected_servo_ids=[int(value) for value in context.expected_servo_ids],
        calibration_summary=summary,
        mock_mode=bool(settings.runtime.mock_mode),
        servo_connected=False,
        sign_mapping=sign_mapping,
    )
    neutral = readiness.neutral_safe_calibration
    startup = readiness.startup_pretension
    artifact_robot = dict(summary.robot_metadata or {})
    artifact_mode = str(artifact_robot.get("operating_mode") or artifact_robot.get("robot_mode") or "unknown")
    artifact_active_segment = str(artifact_robot.get("active_segment_key") or "none")
    artifact_expected_ids = [
        int(value)
        for value in list(artifact_robot.get("expected_servo_ids") or [])
        if value is not None
    ]
    runtime_mode = str(context.operating_mode)
    runtime_segment = str(context.active_segment_key or "none")
    runtime_expected_ids = [int(value) for value in context.expected_servo_ids]
    segment_mismatch = bool(neutral.mismatch_reasons)
    safe_to_use = bool(neutral.ready and startup.ready and not segment_mismatch)

    if context.operating_mode == "dual_segment":
        if not safe_to_use:
            return HardwareReadinessCheck(
                subsystem="selected_segment_calibration",
                status="FAIL",
                message=(
                    "Calibration/startup artifact is not dual-segment-safe for current runtime. "
                    f"runtime(mode={runtime_mode}, segment={runtime_segment}, expected_ids={runtime_expected_ids}) "
                    f"artifact(mode={artifact_mode}, segment={artifact_active_segment}, expected_ids={artifact_expected_ids})."
                ),
                next_action=(
                    "Run two_segment_startup_validation (baseline -> segment_a_pretensioned -> "
                    "segment_b_pretensioned -> segment_a_recheck -> final_accept) to capture a compatible all-8 startup."
                ),
                details={
                    "runtime_mode": runtime_mode,
                    "runtime_segment": runtime_segment,
                    "runtime_expected_servo_ids": runtime_expected_ids,
                    "artifact_mode": artifact_mode,
                    "artifact_segment": artifact_active_segment,
                    "artifact_expected_servo_ids": artifact_expected_ids,
                    "safe_to_use": safe_to_use,
                    "readiness": readiness.to_dict(),
                },
            )
        return HardwareReadinessCheck(
            subsystem="selected_segment_calibration",
            status="PASS",
            message=(
                "Calibration/startup artifact matches dual-segment runtime and is safe to use "
                f"(runtime expected IDs={runtime_expected_ids})."
            ),
            next_action="Proceed with two_segment_startup_validation checkpointing, then dataset collection.",
            details={
                "runtime_mode": runtime_mode,
                "runtime_segment": runtime_segment,
                "runtime_expected_servo_ids": runtime_expected_ids,
                "artifact_mode": artifact_mode,
                "artifact_segment": artifact_active_segment,
                "artifact_expected_servo_ids": artifact_expected_ids,
                "safe_to_use": safe_to_use,
                "readiness": readiness.to_dict(),
            },
        )
    if context.operating_mode != "single_segment":
        return HardwareReadinessCheck(
            subsystem="selected_segment_calibration",
            status="WARN",
            message=(
                f"Current mode is {context.operating_mode}; Wednesday target should be single_segment "
                "with Segment B [5,6,7,8] or Segment A [1,2,3,4]."
            ),
            next_action="Select single_segment / Segment B for [5,6,7,8].",
            details=readiness.to_dict(),
        )
    if not neutral.ready:
        status = "WARN" if settings.runtime.mock_mode and neutral.is_mock else "FAIL"
        if segment_mismatch:
            next_action = (
                "Current runtime segment does not match artifact segment. Recapture startup/calibration for the "
                f"current single_segment target ({context.active_segment_label} / {context.active_segment_key}, "
                f"IDs {runtime_expected_ids})."
            )
        else:
            next_action = (
                "Set mock_mode=false in system.local.yaml or System tab before hardware data."
                if settings.runtime.mock_mode
                else "Capture valid neutral/safe calibration before repeatability."
            )
        startup_note = " Startup reference exists but is not safe to use for the current runtime context." if startup.accepted else ""
        return HardwareReadinessCheck(
            subsystem="selected_segment_calibration",
            status=status,
            message=(
                f"{context.active_segment_label} ({context.active_segment_key}) expected IDs {context.expected_servo_ids}: "
                f"{neutral.message}{startup_note} "
                f"runtime(mode={runtime_mode}, segment={runtime_segment}, expected_ids={runtime_expected_ids}) "
                f"artifact(mode={artifact_mode}, segment={artifact_active_segment}, expected_ids={artifact_expected_ids}) "
                f"safe_to_use={safe_to_use}"
            ),
            next_action=next_action,
            details={
                "runtime_mode": runtime_mode,
                "runtime_segment": runtime_segment,
                "runtime_expected_servo_ids": runtime_expected_ids,
                "artifact_mode": artifact_mode,
                "artifact_segment": artifact_active_segment,
                "artifact_expected_servo_ids": artifact_expected_ids,
                "safe_to_use": safe_to_use,
                "readiness": readiness.to_dict(),
            },
        )
    if not startup.ready:
        return HardwareReadinessCheck(
            subsystem="selected_segment_calibration",
            status="WARN",
            message=(
                f"{context.active_segment_label} ({context.active_segment_key}) has neutral/safe calibration, "
                f"but startup/pretension reference is not ready: {startup.message}"
            ),
            next_action="Capture manual startup or run pretension_validation before repeatability.",
            details=readiness.to_dict(),
        )
    mapping_warning = sign_mapping is not None and not sign_mapping.confirmed
    return HardwareReadinessCheck(
        subsystem="selected_segment_calibration",
        status="WARN" if mapping_warning else "PASS",
        message=readiness.next_action,
        next_action=(
            "Run tiny jog sign checklist before pretension."
            if mapping_warning
            else "Run pretension_validation, then repeatability only after tracker/registration preflight passes."
        ),
        details=readiness.to_dict(),
    )


def _calibration_context_from_settings(settings) -> ServoCalibrationContext:
    operating_context = settings.robot.operating_context()
    return ServoCalibrationContext(
        robot_mode=settings.robot.operating_mode(),
        robot_config_name=settings.runtime.robot_config,
        servo_ids=list(settings.robot.servo_ids),
        tendon_to_servo=list(settings.robot.tendon_to_servo),
        active_segment_key=settings.robot.active_segment_key(),
        active_segment_label=settings.robot.active_segment_label(),
        active_segment_servo_ids=settings.robot.active_segment_servo_ids(),
        active_segment_pairs=settings.robot.active_segment_pairs(),
        segments=settings.robot.segment_metadata(),
        segment_order=settings.robot.segment_order(),
        selected_servo_id=settings.robot.selected_servo_id,
        expected_servo_ids=settings.robot.expected_servo_ids(),
        commanded_servo_ids=settings.robot.commanded_servo_ids(),
        mirror_pairs=settings.robot.parallel_mirror_pairs(),
        mode_profile=operating_context.mode_profile,
        mode_capabilities=operating_context.mode_capabilities,
        mode_notes=operating_context.mode_notes,
        ticks_per_revolution=settings.robot.ticks_per_revolution,
        spool_diameter_cm=settings.robot.spool_diameter_cm,
        position_min_offset_ticks=settings.safety.position_min_offset_ticks,
        position_max_offset_ticks=settings.safety.position_max_offset_ticks,
        default_pretension_current_threshold_ma=settings.safety.default_pretension_current_threshold_ma,
        tightening_rotation_by_servo=dict(settings.robot.tightening_rotation_by_servo),
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
    expected = ["System", "Tracking", "Registration", "Servos", "Experiment", "Modeling", "Data"]
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
        message=(
            "System, Tracking, Registration, Servos, Experiment, Modeling, and Data pages construct offscreen. "
            "Pretension workflow is integrated in Servos."
        ),
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
