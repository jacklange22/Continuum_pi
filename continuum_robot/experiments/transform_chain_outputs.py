"""Transform-chain summary artifacts shared by experiments and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib
from typing import Any

from continuum_robot.tracking.runtime_tip_policy import (
    WORKFLOW_GENERIC,
    evaluate_runtime_tip_trust,
)
from continuum_robot.utils.time_utils import utc_now_iso


TRANSFORM_CHAIN_EXPRESSION = "T_robot_tip = T_robot_aurora @ T_aurora_coil_0A @ T_coil_tip"


def build_transform_chain_summary(
    *,
    snapshot,
    workflow: str = WORKFLOW_GENERIC,
    allow_lower_trust: bool = False,
    provenance_note: str = "",
) -> dict[str, Any]:
    """Build a compact, thesis-facing summary of the active transform chain."""

    policy = evaluate_runtime_tip_trust(
        snapshot=snapshot,
        workflow=workflow,
        allow_lower_trust=allow_lower_trust,
    )
    registration_path = str(getattr(snapshot, "registration_path", "") or "")
    runtime_tip_path = str(getattr(snapshot, "runtime_tip_selected_artifact_path", None) or getattr(snapshot, "runtime_tip_calibration_path", "") or "")
    warnings = list(policy.warnings)
    warnings.extend(str(message) for message in getattr(snapshot, "warning_messages", []) or [])
    if getattr(snapshot, "last_error", None):
        warnings.append(str(getattr(snapshot, "last_error")))
    return {
        "schema_version": "1.0",
        "generated_at_utc": utc_now_iso(),
        "provenance_note": str(provenance_note or ""),
        "tracker": {
            "backend_identity": str(getattr(snapshot, "backend_identity", "") or ""),
            "configured_backend_name": str(getattr(snapshot, "configured_backend_name", "") or ""),
            "selected_backend_name": str(getattr(snapshot, "selected_backend_name", "") or ""),
            "canonical_state": str(getattr(snapshot, "canonical_state", "") or ""),
            "last_frame_number": getattr(snapshot, "last_frame_number", None),
            "tracker_data_age_s": getattr(snapshot, "tracker_data_age_s", None),
            "runtime_coil_tool_id": str(getattr(snapshot, "runtime_coil_tool_id", "0A") or "0A"),
            "registration_tool_id": str(getattr(snapshot, "registration_tool_id", "0B") or "0B"),
        },
        "registration": {
            "state": str(getattr(snapshot, "registration_state", "") or ""),
            "artifact_path": registration_path,
            "artifact_exists": bool(Path(registration_path).exists()) if registration_path else False,
            "stored_timestamp_utc": getattr(snapshot, "stored_registration_timestamp_utc", None),
            "fre_mm": getattr(snapshot, "stored_registration_fre_mm", None),
            "measurement_tool_id": getattr(snapshot, "stored_registration_measurement_tool_id", None),
            "coil_tool_id": getattr(snapshot, "stored_registration_coil_tool_id", None),
        },
        "runtime_tip": {
            "mode": policy.mode,
            "trust_label": policy.trust_label,
            "thesis_trusted": bool(policy.thesis_trusted),
            "allowed_for_workflow": bool(policy.allowed_for_workflow),
            "workflow": policy.workflow,
            "uses_coil_as_tip": bool(policy.uses_coil_as_tip),
            "uses_calibrated_tip_transform": bool(policy.uses_calibrated_tip_transform),
            "uses_identity_transform": bool(policy.uses_identity_transform),
            "calibration_state": policy.calibration_state,
            "tip_pose_status": policy.tip_pose_status,
            "mode_message": str(getattr(snapshot, "runtime_tip_mode_message", "") or policy.status_message),
            "selected_artifact_kind": getattr(snapshot, "runtime_tip_selected_artifact_kind", None),
            "selected_artifact_path": runtime_tip_path or None,
            "selected_artifact_exists": bool(Path(runtime_tip_path).exists()) if runtime_tip_path else False,
            "stored_timestamp_utc": getattr(snapshot, "stored_runtime_tip_timestamp_utc", None),
            "measurement_tool_id": getattr(snapshot, "stored_runtime_tip_measurement_tool_id", None),
            "coil_tool_id": getattr(snapshot, "stored_runtime_tip_coil_tool_id", None),
            "identity_fallback": bool(getattr(snapshot, "runtime_tip_identity_fallback", False)),
            "policy": policy.to_dict(),
        },
        "transform_chain": {
            "expression": TRANSFORM_CHAIN_EXPRESSION,
            "frame_conventions": {
                "T_A_B": "maps coordinates from frame B into frame A",
                "robot": "registered robot/body frame",
                "aurora": "NDI Aurora tracker frame",
                "coil_0A": "runtime 0A coil frame",
                "tip": "declared runtime tip frame; coil origin when runtime_tip.mode is coil_as_tip",
            },
            "active_transforms": {
                "T_robot_aurora": getattr(snapshot, "T_robot_aurora", None) is not None,
                "T_aurora_coil_0A": "0A" in (getattr(snapshot, "tools", {}) or {}),
                "T_coil_tip": bool(policy.uses_calibrated_tip_transform or policy.uses_identity_transform),
                "T_robot_tip": getattr(snapshot, "T_robot_tip", None) is not None,
            },
        },
        "warnings": warnings,
        "reasons": list(policy.reasons),
    }


def write_transform_chain_outputs(
    *,
    output_dir: Path,
    snapshot,
    workflow: str = WORKFLOW_GENERIC,
    allow_lower_trust: bool = False,
    provenance_note: str = "",
) -> dict[str, Path]:
    """Write JSON, text, and PNG transform-chain summary artifacts."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = build_transform_chain_summary(
        snapshot=snapshot,
        workflow=workflow,
        allow_lower_trust=allow_lower_trust,
        provenance_note=provenance_note,
    )
    json_path = output_root / "transform_chain_summary.json"
    txt_path = output_root / "transform_chain_summary.txt"
    png_path = output_root / "transform_chain_overview.png"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    txt_path.write_text(format_transform_chain_summary_text(summary), encoding="utf-8")
    write_transform_chain_overview_png(summary=summary, path=png_path)
    return {"json": json_path, "txt": txt_path, "png": png_path}


def format_transform_chain_summary_text(summary: dict[str, Any]) -> str:
    tracker = dict(summary.get("tracker", {}) or {})
    registration = dict(summary.get("registration", {}) or {})
    runtime_tip = dict(summary.get("runtime_tip", {}) or {})
    chain = dict(summary.get("transform_chain", {}) or {})
    warnings = [str(item) for item in summary.get("warnings", []) or []]
    reasons = [str(item) for item in summary.get("reasons", []) or []]
    lines = [
        "Transform Chain Summary",
        f"Generated UTC: {summary.get('generated_at_utc', 'unknown')}",
        f"Expression: {chain.get('expression', TRANSFORM_CHAIN_EXPRESSION)}",
        "",
        "Tracker",
        f"- backend: {tracker.get('selected_backend_name') or tracker.get('backend_identity') or 'unknown'}",
        f"- state: {tracker.get('canonical_state', 'unknown')}",
        f"- runtime coil tool: {tracker.get('runtime_coil_tool_id', '0A')}",
        f"- registration tool: {tracker.get('registration_tool_id', '0B')}",
        "",
        "Registration",
        f"- state: {registration.get('state', 'unknown')}",
        f"- artifact: {registration.get('artifact_path') or 'none'}",
        f"- FRE mm: {_fmt_optional_float(registration.get('fre_mm'))}",
        f"- timestamp UTC: {registration.get('stored_timestamp_utc') or 'unknown'}",
        "",
        "Runtime Tip Policy",
        f"- mode: {runtime_tip.get('mode', 'unknown')}",
        f"- trust: {runtime_tip.get('trust_label', 'unknown')}",
        f"- thesis trusted: {'yes' if runtime_tip.get('thesis_trusted') else 'no'}",
        f"- workflow allowed: {'yes' if runtime_tip.get('allowed_for_workflow') else 'no'}",
        f"- coil-as-tip: {'yes' if runtime_tip.get('uses_coil_as_tip') else 'no'}",
        f"- calibrated T_coil_tip: {'yes' if runtime_tip.get('uses_calibrated_tip_transform') else 'no'}",
        f"- artifact: {runtime_tip.get('selected_artifact_path') or 'none'}",
        f"- message: {runtime_tip.get('mode_message') or 'none'}",
    ]
    if warnings:
        lines.extend(["", "Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    if reasons:
        lines.extend(["", "Blocking / Missing Reasons"])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.append("")
    return "\n".join(lines)


def write_transform_chain_overview_png(*, summary: dict[str, Any], path: Path) -> None:
    """Write a compact overview figure. Uses Qt when available, with a PNG fallback."""

    try:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    except Exception:
        _write_fallback_png(Path(path), trusted=bool((summary.get("runtime_tip", {}) or {}).get("thesis_trusted")))
        return
    if QGuiApplication.instance() is None:
        _write_fallback_png(Path(path), trusted=bool((summary.get("runtime_tip", {}) or {}).get("thesis_trusted")))
        return

    width, height = 1200, 680
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(QColor("#ffffff"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)

    runtime_tip = dict(summary.get("runtime_tip", {}) or {})
    registration = dict(summary.get("registration", {}) or {})
    tracker = dict(summary.get("tracker", {}) or {})
    trusted = bool(runtime_tip.get("thesis_trusted"))
    accent = QColor("#1f7a4d" if trusted else "#b45309")
    muted = QColor("#5f6b7a")
    border = QColor("#c8d0d9")

    title_font = QFont("Arial", 22)
    title_font.setBold(True)
    label_font = QFont("Arial", 13)
    label_font.setBold(True)
    body_font = QFont("Arial", 11)

    painter.setFont(title_font)
    painter.setPen(QColor("#14213d"))
    painter.drawText(QRectF(40, 26, 1120, 40), "Transform Chain Overview")
    painter.setFont(body_font)
    painter.setPen(muted)
    painter.drawText(QRectF(40, 62, 1120, 28), str(summary.get("generated_at_utc", "")))

    boxes = [
        (70, 170, 190, 100, "Robot frame", "body/base frame"),
        (330, 170, 190, 100, "Aurora frame", str(tracker.get("selected_backend_name") or tracker.get("backend_identity") or "tracker")),
        (590, 170, 190, 100, "0A coil", "runtime coil pose"),
        (850, 170, 240, 100, "Runtime tip", str(runtime_tip.get("mode", "unknown")).replace("_", " ")),
    ]
    painter.setPen(QPen(border, 2))
    for x, y, w, h, label, body in boxes:
        painter.setBrush(QColor("#f8fafc"))
        painter.drawRoundedRect(x, y, w, h, 8, 8)
        painter.setFont(label_font)
        painter.setPen(QColor("#111827"))
        painter.drawText(QRectF(x + 14, y + 18, w - 28, 24), label)
        painter.setFont(body_font)
        painter.setPen(muted)
        painter.drawText(QRectF(x + 14, y + 50, w - 28, 36), body)
        painter.setPen(QPen(border, 2))

    painter.setPen(QPen(QColor("#334155"), 3))
    for x0, x1 in ((260, 330), (520, 590), (780, 850)):
        painter.drawLine(x0, 220, x1, 220)
        painter.drawLine(x1 - 12, 212, x1, 220)
        painter.drawLine(x1 - 12, 228, x1, 220)
    painter.setFont(body_font)
    painter.setPen(QColor("#334155"))
    painter.drawText(QRectF(270, 190, 70, 20), "T_robot_aurora")
    painter.drawText(QRectF(525, 190, 80, 20), "T_aurora_coil")
    painter.drawText(QRectF(790, 190, 70, 20), "T_coil_tip")

    status_rect = QRectF(70, 340, 1020, 92)
    painter.setBrush(QColor("#ecfdf5" if trusted else "#fff7ed"))
    painter.setPen(QPen(accent, 2))
    painter.drawRoundedRect(status_rect, 8, 8)
    painter.setFont(label_font)
    painter.setPen(accent)
    painter.drawText(QRectF(92, 358, 260, 28), f"Trust: {runtime_tip.get('trust_label', 'unknown')}")
    painter.setFont(body_font)
    painter.setPen(QColor("#111827"))
    painter.drawText(
        QRectF(92, 392, 960, 26),
        "Thesis trusted: yes" if trusted else "Thesis trusted: no - use as lower-trust/debug evidence",
    )

    metrics = [
        ("Registration FRE (mm)", _fmt_optional_float(registration.get("fre_mm"))),
        ("Registration state", str(registration.get("state", "unknown"))),
        ("Tip pose status", str(runtime_tip.get("tip_pose_status", "unknown"))),
        ("Coil-as-tip", "yes" if runtime_tip.get("uses_coil_as_tip") else "no"),
        ("Calibrated T_coil_tip", "yes" if runtime_tip.get("uses_calibrated_tip_transform") else "no"),
    ]
    x = 70
    y = 470
    for label, value in metrics:
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(x, y, 195, 82, 6, 6)
        painter.setFont(body_font)
        painter.setPen(muted)
        painter.drawText(QRectF(x + 12, y + 14, 171, 20), label)
        painter.setFont(label_font)
        painter.setPen(QColor("#111827"))
        painter.drawText(QRectF(x + 12, y + 42, 171, 28), value)
        x += 205

    warning_text = "; ".join(str(w) for w in summary.get("warnings", [])[:2]) or "No current warnings."
    painter.setFont(body_font)
    painter.setPen(muted)
    painter.drawText(QRectF(70, 590, 1020, 48), warning_text)
    painter.end()
    image.save(str(path), "PNG")


def _fmt_optional_float(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _write_fallback_png(path: Path, *, trusted: bool) -> None:
    width, height = 640, 240
    bg = (255, 255, 255)
    accent = (31, 122, 77) if trusted else (180, 83, 9)
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            pixel = bg
            if 24 <= x <= width - 24 and 24 <= y <= 76:
                pixel = accent
            elif 24 <= x <= width - 24 and 96 <= y <= 168:
                pixel = (248, 250, 252)
            row.extend(pixel)
        rows.append(bytes(row))
    raw = b"".join(rows)
    payload = b"\x89PNG\r\n\x1a\n"
    for chunk_type, data in (
        (b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        (b"IDAT", zlib.compress(raw, 9)),
        (b"IEND", b""),
    ):
        payload += struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    path.write_bytes(payload)
