"""Saved artifacts for the Motor Babble modeling dataset collector."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dat_writer import DatRunWriter
from continuum_robot.experiments.plotting import add_metric_box, color, create_figure, legend, save_figure, set_equal_xy, style_axes
from continuum_robot.experiments.tracker_timing_outputs import (
    _draw_bar_chart,
    _draw_panel,
    _draw_summary_pairs,
    _draw_title,
    _ensure_plot_qt_app,
    _fmt,
    _new_image,
)
from continuum_robot.data.model_training_validity import NON_TRAINING_PHASES, sample_has_complete_command_servo_tip
try:
    from continuum_robot.gui.theme import COLORS
except Exception:
    class _FallbackColors:
        scene_measurement = "#0f766e"
        scene_residual = "#dc2626"
        text_muted = "#64748b"
        surface_border = "#dbe4ee"
        chart_grid = "#e2e8f0"

    COLORS = _FallbackColors()
try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPainter, QPen
    from PySide6.QtWidgets import QApplication

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


LOG = logging.getLogger(__name__)


def build_modeling_dataset_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    command_range = dict(metrics.get("command_pair_range_cm", {}) or {})
    workspace_span = dict(metrics.get("workspace_span_mm", {}) or {})
    return [
        ("Dataset Mode", str(metrics.get("dataset_mode", "unknown") or "unknown").replace("_", " ")),
        ("Run Label", str(metrics.get("run_label", "") or "n/a")),
        ("Dataset Tag", str(metrics.get("dataset_tag", "") or "n/a")),
        ("Accepted Samples", str(int(metrics.get("accepted_sample_count", 0) or 0))),
        ("Rejected Samples", str(int(metrics.get("rejected_sample_count", 0) or 0))),
        (
            "Acceptance Rate",
            (
                f"{float(metrics.get('accepted_capture_rate', 0.0) or 0.0) * 100.0:.1f}%"
                if metrics.get("accepted_capture_rate") is not None
                else "n/a"
            ),
        ),
        ("Command Steps", str(int(metrics.get("command_step_count", 0) or 0))),
        ("Samples / Command", str(int(metrics.get("samples_per_command", 0) or 0))),
        ("Runtime Tip", f"{runtime_tip.get('mode', 'unknown')} ({runtime_tip.get('trust_level', 'unknown')})"),
        ("Pretension", f"{pretension.get('active_source_type', 'unknown')} ({pretension.get('status', 'unknown')})"),
        (
            "Pair 1/3 Range",
            (
                f"{_fmt(command_range.get('pair_13_min_cm'))} .. {_fmt(command_range.get('pair_13_max_cm'))} cm"
                if command_range
                else "n/a"
            ),
        ),
        (
            "Pair 2/4 Range",
            (
                f"{_fmt(command_range.get('pair_24_min_cm'))} .. {_fmt(command_range.get('pair_24_max_cm'))} cm"
                if command_range
                else "n/a"
            ),
        ),
        (
            "Workspace Span",
            (
                f"x={_fmt(workspace_span.get('x_span_mm'))} mm, "
                f"y={_fmt(workspace_span.get('y_span_mm'))} mm, "
                f"z={_fmt(workspace_span.get('z_span_mm'))} mm"
                if workspace_span
                else "n/a"
            ),
        ),
        ("Lower-Trust Override", "yes" if metrics.get("lower_trust_active") else "no"),
    ]


def build_modeling_dataset_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    registration = dict(provenance.get("base_registration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    lines = [
        "Motor Babble Modeling Dataset Summary",
        "This run collected ordered single-segment command/pose samples for offline forward-model training and later state-aware model comparison.",
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_modeling_dataset_summary_pairs(metrics=metrics):
        lines.append(f"{label}: {value}")
    validity_status = str(metrics.get("model_training_validity_status", "not_applicable") or "not_applicable")
    validity_reason = str(metrics.get("model_training_validity_reason", "") or "")
    hard_invalid_reasons = [str(value) for value in list(metrics.get("model_training_hard_invalidation_reasons", []) or []) if str(value)]
    validity_warnings = [str(value) for value in list(metrics.get("model_training_warnings", []) or []) if str(value)]
    lines.extend(
        [
            "",
            "Trainability:",
            f"- Status: {validity_status}",
            f"- Reason: {validity_reason or 'n/a'}",
            (
                "- Row counts: "
                f"export={int(metrics.get('modeling_export_row_count', 0) or 0)}, "
                f"legacy={int(metrics.get('modeling_legacy_row_count', 0) or 0)}, "
                f"accepted={int(metrics.get('accepted_training_row_count', 0) or 0)}"
            ),
            f"- Accepted rows complete: {bool(metrics.get('accepted_rows_complete'))}",
            f"- Dropped rows excluded: {bool(metrics.get('dropped_samples_excluded_from_training'))}",
        ]
    )
    if hard_invalid_reasons:
        lines.append("- Hard invalidation reasons:")
        for reason in hard_invalid_reasons:
            lines.append(f"  - {reason}")
    if validity_warnings:
        lines.append("- Warnings:")
        for warning in validity_warnings:
            lines.append(f"  - {warning}")
    lines.extend(
        [
            "",
            "Provenance:",
            f"- Registration: {registration.get('path', 'n/a')} @ {registration.get('stored_timestamp_utc', registration.get('modified_at_utc', 'n/a'))}",
            (
                f"- Runtime tip: {runtime_tip.get('path', 'n/a')} "
                f"(mode={runtime_tip.get('mode', 'unknown')}, trust={runtime_tip.get('trust_level', 'unknown')}, "
                f"timestamp={runtime_tip.get('stored_timestamp_utc', runtime_tip.get('modified_at_utc', 'n/a'))})"
            ),
            (
                f"- Pretension: {pretension.get('path', 'n/a')} "
                f"(source={pretension.get('active_source_type', 'unknown')}, "
                f"updated={pretension.get('active_source_updated_at_utc', pretension.get('modified_at_utc', 'n/a'))})"
            ),
        ]
    )
    if runtime_tip.get("mode_message"):
        lines.append(f"- Runtime tip detail: {runtime_tip.get('mode_message')}")
    if pretension.get("active_source_message"):
        lines.append(f"- Pretension detail: {pretension.get('active_source_message')}")
    rejection_reasons = dict(metrics.get("rejection_reasons", {}) or {})
    if rejection_reasons:
        lines.extend(["", "Rejected captures:"])
        for reason, count in sorted(rejection_reasons.items()):
            lines.append(f"- {reason}: {int(count)}")
    lines.extend(
        [
            "",
            "Offline handoff:",
            "- `samples.jsonl` preserves canonical full-fidelity ordered samples.",
            "- `modeling_dataset_export.jsonl` preserves ordered export rows with an explicit accepted flag for offline filtering or sequential multi-input modeling.",
            "- `modeling_dataset_legacy_compat.dat` contains accepted rows only for quick legacy ANN/comparison-stack conversion.",
        ]
    )
    return lines


def write_modeling_dataset_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    summary_text_path = output_dir / "modeling_dataset_summary.txt"
    export_jsonl_path = output_dir / "modeling_dataset_export.jsonl"
    legacy_dat_path = output_dir / "modeling_dataset_legacy_compat.dat"
    workspace_plot_path = output_dir / "modeling_workspace_coverage.png"
    command_plot_path = output_dir / "modeling_command_distribution.png"
    workspace_report_path = output_dir / "modeling_workspace_coverage_report.png"
    commanded_tendon_report_path = output_dir / "commanded_tendon_space_report.png"

    summary_text_path.write_text(
        "\n".join(
            build_modeling_dataset_summary_lines(
                metadata=metadata,
                summary=summary,
                metrics=metrics,
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    export_rows = _build_export_rows(samples=samples)
    with export_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    legacy_written_path: Path | None = None
    if bool(metrics.get("legacy_export_enabled", True)):
        rows = _build_legacy_dat_rows(export_rows=export_rows)
        if rows:
            writer = DatRunWriter(output_dir=output_dir)
            legacy_written_path = writer.write_run(
                num_cables=4,
                rows=rows,
                filename_stem="modeling_dataset_legacy_compat",
            )
    try:
        _write_workspace_plot(workspace_plot_path=workspace_plot_path, export_rows=export_rows, metrics=metrics)
        _write_workspace_report_plot(
            workspace_plot_path=workspace_report_path,
            export_rows=export_rows,
            metrics=metrics,
        )
        _write_commanded_tendon_space_report(
            plot_path=commanded_tendon_report_path,
            export_rows=export_rows,
            metrics=metrics,
        )
    except Exception:
        LOG.exception("Matplotlib workspace plot failed; writing placeholder %s", workspace_plot_path)
        _write_plot_placeholder(workspace_plot_path)
        _write_plot_placeholder(workspace_report_path)
        _write_plot_placeholder(commanded_tendon_report_path)
    if _qt_plotting_is_safe():
        _ensure_plot_qt_app()
        _write_command_plot(command_plot_path=command_plot_path, export_rows=export_rows, metrics=metrics)
    else:
        _write_plot_placeholder(command_plot_path)
        LOG.warning("Qt plotting backend unavailable; wrote placeholder modeling plots under %s", output_dir)
    outputs: dict[str, Path] = {
        "summary_text_path": summary_text_path,
        "export_jsonl_path": export_jsonl_path,
        "workspace_plot_path": workspace_plot_path,
        "command_plot_path": command_plot_path,
        "workspace_report_path": workspace_report_path,
        "commanded_tendon_space_report_path": commanded_tendon_report_path,
    }
    if legacy_written_path is not None:
        outputs["legacy_dat_path"] = legacy_written_path
    elif legacy_dat_path.exists():
        outputs["legacy_dat_path"] = legacy_dat_path
    return outputs


def _write_plot_placeholder(path: Path) -> None:
    path.write_bytes(
        bytes(
            (
                137,
                80,
                78,
                71,
                13,
                10,
                26,
                10,
                0,
                0,
                0,
                13,
                73,
                72,
                68,
                82,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                8,
                6,
                0,
                0,
                0,
                31,
                21,
                196,
                137,
                0,
                0,
                0,
                13,
                73,
                68,
                65,
                84,
                120,
                156,
                99,
                96,
                0,
                0,
                0,
                2,
                0,
                1,
                226,
                33,
                188,
                51,
                0,
                0,
                0,
                0,
                73,
                69,
                78,
                68,
                174,
                66,
                96,
                130,
            )
        )
    )


def _qt_plotting_is_safe() -> bool:
    """Return whether creating a Qt plotting application is safe in this process."""
    if not _QT_AVAILABLE:
        return False
    if QApplication.instance() is not None:
        return True
    if os.environ.get("QT_QPA_PLATFORM"):
        return True
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return False
    if sys.platform == "darwin" and not os.environ.get("DISPLAY"):
        # Headless macOS pytest sessions can abort at QApplication construction.
        return False
    return True


def _build_export_rows(*, samples) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_sequence_index, sample in enumerate(samples):
        phase = str(getattr(sample, "phase", "") or "")
        if phase in NON_TRAINING_PHASES:
            continue
        extra = dict(sample.extra or {})
        if bool(extra.get("modeling_export_exclude")):
            continue
        if not bool(extra.get("capture_accepted")):
            continue
        if not sample_has_complete_command_servo_tip(sample):
            continue
        tip_payload = dict(sample.pose_in_robot_frame.get("tip", {}) or {})
        tool_payload = dict(sample.pose_in_tracker_frame.get("0A", {}) or {})
        sequence_index = len(rows)
        rows.append(
            {
                "sequence_index": int(sequence_index),
                "source_sequence_index": int(source_sequence_index),
                "phase": str(sample.phase or ""),
                "step_index": int(sample.step_index),
                "sample_index": int(sample.sample_index),
                "accepted": bool(extra.get("capture_accepted")),
                "capture_rejection_reason": extra.get("capture_rejection_reason"),
                "dataset_mode": extra.get("dataset_mode"),
                "requested_pair_command_cm": list(extra.get("requested_pair_command_cm", []) or []),
                "resolved_pair_command_cm": list(extra.get("resolved_pair_command_cm", []) or []),
                "previous_pair_command_cm": list(extra.get("previous_pair_command_cm", []) or []),
                "requested_cable_command_cm": list(extra.get("requested_cable_command_cm", []) or []),
                "resolved_cable_command_cm": list(extra.get("resolved_cable_command_cm", []) or []),
                "raw_goal_ticks_by_servo": dict(extra.get("raw_goal_ticks_by_servo", {}) or {}),
                "final_goal_ticks_by_servo": dict(extra.get("final_goal_ticks_by_servo", {}) or {}),
                "servo_feedback_at_capture": dict(extra.get("servo_feedback_at_capture", {}) or {}),
                "tracker_frame_id": sample.tracker_frame_id,
                "tracker_freshness_s": sample.freshness_s,
                "tracker_status_flags": list(sample.status_flags or []),
                "tip_position_xyz_mm": list(tip_payload.get("translation_mm", []) or []),
                "tip_quaternion_wxyz": list(tip_payload.get("quaternion_wxyz", []) or []),
                "tip_tangent_xyz": list(tip_payload.get("tangent_xyz", []) or []),
                "tool_0A_translation_mm": list(tool_payload.get("translation_mm", []) or []),
                "tool_0A_quaternion_wxyz": list(tool_payload.get("quaternion_wxyz", []) or []),
                "tool_0A_tangent_xyz": list(tool_payload.get("tangent_xyz", []) or []),
            }
        )
    return rows


def _build_legacy_dat_rows(*, export_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in export_rows:
        if not bool(row.get("accepted")):
            continue
        servo_feedback = dict(row.get("servo_feedback_at_capture", {}) or {})
        ordered_servo_ids = sorted(servo_feedback, key=lambda value: int(value))
        rows.append(
            {
                "timestamp_utc": "",
                "index": int(row.get("sequence_index", 0) or 0),
                "repeat_index": int(row.get("sample_index", 0) or 0),
                "commanded_displacement_cm": list(row.get("resolved_cable_command_cm", []) or []),
                "commanded_goal_ticks": [
                    (dict(row.get("final_goal_ticks_by_servo", {}) or {}).get(servo_id))
                    for servo_id in ordered_servo_ids
                ],
                "servo_position_ticks": [
                    (dict(servo_feedback.get(servo_id, {}) or {}).get("present_position_ticks"))
                    for servo_id in ordered_servo_ids
                ],
                "servo_current_ma": [
                    (dict(servo_feedback.get(servo_id, {}) or {}).get("present_current_ma"))
                    for servo_id in ordered_servo_ids
                ],
                "servo_voltage_mv": [
                    (dict(servo_feedback.get(servo_id, {}) or {}).get("present_voltage_mv"))
                    for servo_id in ordered_servo_ids
                ],
                "tool_0A_translation_mm": list(row.get("tool_0A_translation_mm", []) or []),
                "tool_0B_translation_mm": [],
                "tip_position_xyz": list(row.get("tip_position_xyz_mm", []) or []),
                "tip_tangent_xyz": list(row.get("tip_tangent_xyz", []) or []),
            }
        )
    return rows


def _write_workspace_plot(*, workspace_plot_path: Path, export_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    accepted_rows = [row for row in export_rows if row.get("accepted") and len(row.get("tip_position_xyz_mm", [])) == 3]
    rejected_rows = [row for row in export_rows if not row.get("accepted") and len(row.get("tip_position_xyz_mm", [])) == 3]
    accepted_points = [(float(row["tip_position_xyz_mm"][0]), float(row["tip_position_xyz_mm"][1])) for row in accepted_rows]
    rejected_points = [(float(row["tip_position_xyz_mm"][0]), float(row["tip_position_xyz_mm"][1])) for row in rejected_rows]
    fig, ax = create_figure(size="square")
    if rejected_points:
        ax.scatter(
            [point[0] for point in rejected_points],
            [point[1] for point in rejected_points],
            s=18,
            c=color("rejected"),
            alpha=0.35,
            linewidths=0,
            label="Rejected",
        )
    if accepted_points:
        alpha = 0.65 if len(accepted_points) < 500 else 0.35
        ax.scatter(
            [point[0] for point in accepted_points],
            [point[1] for point in accepted_points],
            s=16,
            c=color("measured"),
            alpha=alpha,
            linewidths=0,
            label="Accepted",
        )
    if accepted_points or rejected_points:
        all_points = accepted_points + rejected_points
        set_equal_xy(ax, x_values=[point[0] for point in all_points], y_values=[point[1] for point in all_points], minimum_span=5.0)
    else:
        ax.text(0.5, 0.5, "No robot-frame tip samples available", transform=ax.transAxes, ha="center", va="center")
    style_axes(
        ax,
        title="Motor Babble Workspace Coverage",
        xlabel="Robot-frame X position (mm)",
        ylabel="Robot-frame Y position (mm)",
    )
    add_metric_box(
        ax,
        [
            f"Accepted: {int(metrics.get('accepted_sample_count', len(accepted_points)) or 0)}",
            f"Rejected: {int(metrics.get('rejected_sample_count', len(rejected_points)) or 0)}",
            f"Mode: {str(metrics.get('dataset_mode', 'unknown')).replace('_', ' ')}",
        ],
        loc="upper right",
    )
    legend(ax, loc="lower right")
    save_figure(fig, workspace_plot_path)


def _write_workspace_report_plot(*, workspace_plot_path: Path, export_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    accepted_rows = [row for row in export_rows if row.get("accepted") and len(row.get("tip_position_xyz_mm", [])) == 3]
    rejected_rows = [row for row in export_rows if not row.get("accepted") and len(row.get("tip_position_xyz_mm", [])) == 3]
    accepted_points = [(float(row["tip_position_xyz_mm"][0]), float(row["tip_position_xyz_mm"][1])) for row in accepted_rows]
    rejected_points = [(float(row["tip_position_xyz_mm"][0]), float(row["tip_position_xyz_mm"][1])) for row in rejected_rows]
    trust_info = (
        dict(metrics.get("run_trust"))
        if isinstance(metrics.get("run_trust"), dict)
        else (dict(metrics.get("run_trust_info")) if isinstance(metrics.get("run_trust_info"), dict) else {})
    )
    trust_mode = str(
        metrics.get("run_trust_mode")
        or trust_info.get("run_trust_mode")
        or metrics.get("dataset_mode")
        or "unknown"
    )
    valid_for_training = bool(
        metrics.get("valid_for_model_training")
        if metrics.get("valid_for_model_training") is not None
        else trust_info.get("valid_for_model_training", bool(accepted_points))
    )
    fig, ax = create_figure(size="square")
    if rejected_points:
        ax.scatter(
            [point[0] for point in rejected_points],
            [point[1] for point in rejected_points],
            s=18,
            c=color("rejected"),
            alpha=0.30,
            linewidths=0,
            label="Rejected",
        )
    if accepted_points:
        alpha = 0.70 if len(accepted_points) < 500 else 0.40
        ax.scatter(
            [point[0] for point in accepted_points],
            [point[1] for point in accepted_points],
            s=16,
            c=color("measured"),
            alpha=alpha,
            linewidths=0,
            label="Accepted",
        )
    if accepted_points or rejected_points:
        all_points = accepted_points + rejected_points
        set_equal_xy(ax, x_values=[point[0] for point in all_points], y_values=[point[1] for point in all_points], minimum_span=5.0)
    else:
        ax.text(
            0.5,
            0.5,
            "Servo-only run: no robot-frame tip labels\nnot valid for model training",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        set_equal_xy(ax, x_values=[-1.0, 1.0], y_values=[-1.0, 1.0], minimum_span=5.0)
    metric_lines = [
        f"Accepted: {int(metrics.get('accepted_sample_count', len(accepted_points)) or 0)}",
        f"Rejected: {int(metrics.get('rejected_sample_count', len(rejected_points)) or 0)}",
        f"Trust: {trust_mode.replace('_', ' ')}",
        f"Training valid: {'yes' if valid_for_training else 'no'}",
    ]
    add_metric_box(ax, metric_lines, loc="upper right")
    style_axes(
        ax,
        title="Modeling Workspace Coverage",
        xlabel="Robot-frame X position (mm)",
        ylabel="Robot-frame Y position (mm)",
    )
    legend(ax, loc="lower right")
    save_figure(fig, workspace_plot_path)


def _write_commanded_tendon_space_report(*, plot_path: Path, export_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    accepted_points = [
        (float(row["resolved_pair_command_cm"][0]), float(row["resolved_pair_command_cm"][1]))
        for row in export_rows
        if row.get("accepted") and len(row.get("resolved_pair_command_cm", [])) == 2
    ]
    rejected_points = [
        (float(row["resolved_pair_command_cm"][0]), float(row["resolved_pair_command_cm"][1]))
        for row in export_rows
        if not row.get("accepted") and len(row.get("resolved_pair_command_cm", [])) == 2
    ]
    fig, ax = create_figure(size="square")
    if rejected_points:
        ax.scatter(
            [point[0] for point in rejected_points],
            [point[1] for point in rejected_points],
            s=18,
            color=color("rejected"),
            alpha=0.30,
            linewidths=0,
            label="Rejected",
        )
    if accepted_points:
        ax.scatter(
            [point[0] for point in accepted_points],
            [point[1] for point in accepted_points],
            s=18,
            color=color("measured"),
            alpha=0.60,
            linewidths=0,
            label="Accepted",
        )
    if accepted_points or rejected_points:
        all_points = accepted_points + rejected_points
        set_equal_xy(ax, x_values=[point[0] for point in all_points], y_values=[point[1] for point in all_points], minimum_span=0.2)
    else:
        ax.text(0.5, 0.5, "No pair-command rows available", transform=ax.transAxes, ha="center", va="center")
    add_metric_box(
        ax,
        [
            f"Commands: {int(metrics.get('command_step_count', 0) or 0)}",
            f"Mode: {str(metrics.get('dataset_mode', 'unknown')).replace('_', ' ')}",
        ],
        loc="upper right",
    )
    style_axes(
        ax,
        title="Commanded Tendon Space",
        xlabel="Axis A pair command (cm)",
        ylabel="Axis B pair command (cm)",
    )
    legend(ax, loc="best")
    save_figure(fig, plot_path)


def _write_command_plot(*, command_plot_path: Path, export_rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 820)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "MODELING DATASET COMMAND DISTRIBUTION",
        "Resolved antagonistic pair commands and capture acceptance summary.",
    )
    scatter_rect = QRectF(36.0, 108.0, 820.0, 660.0)
    summary_rect = QRectF(888.0, 108.0, 356.0, 660.0)
    _draw_panel(painter, scatter_rect, "Pair Command Space")
    _draw_panel(painter, summary_rect, "Acceptance Summary")
    accepted_rows = [row for row in export_rows if row.get("accepted") and len(row.get("resolved_pair_command_cm", [])) == 2]
    rejected_rows = [row for row in export_rows if not row.get("accepted") and len(row.get("resolved_pair_command_cm", [])) == 2]
    _draw_xy_scatter(
        painter,
        scatter_rect.adjusted(22.0, 42.0, -22.0, -28.0),
        accepted_points=[(float(row["resolved_pair_command_cm"][0]), float(row["resolved_pair_command_cm"][1])) for row in accepted_rows],
        rejected_points=[(float(row["resolved_pair_command_cm"][0]), float(row["resolved_pair_command_cm"][1])) for row in rejected_rows],
        x_label="Pair 1/3 command (cm)",
        y_label="Pair 2/4 command (cm)",
        empty_text="No pair-command rows were available.",
    )
    _draw_bar_chart(
        painter,
        summary_rect.adjusted(16.0, 38.0, -16.0, -320.0),
        labels=["Accepted", "Rejected"],
        values=[
            float(metrics.get("accepted_sample_count", 0) or 0),
            float(metrics.get("rejected_sample_count", 0) or 0),
        ],
        color=QColor(COLORS.scene_measurement),
        y_label="Samples",
        empty_text="No capture rows were available.",
    )
    phase_counts = dict(metrics.get("phase_counts", {}) or {})
    _draw_bar_chart(
        painter,
        summary_rect.adjusted(16.0, 370.0, -16.0, -16.0),
        labels=[str(key) for key in phase_counts],
        values=[float(value) for value in phase_counts.values()],
        color=QColor(COLORS.scene_residual),
        y_label="Samples",
        empty_text="No phase counts were recorded.",
    )
    painter.end()
    image.save(str(command_plot_path))


def _draw_xy_scatter(
    painter: QPainter,
    rect: QRectF,
    *,
    accepted_points: list[tuple[float, float]],
    rejected_points: list[tuple[float, float]],
    x_label: str,
    y_label: str,
    empty_text: str,
) -> None:
    painter.save()
    if not accepted_points and not rejected_points:
        painter.setPen(QColor(COLORS.text_muted))
        painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, empty_text)
        painter.restore()
        return
    all_points = accepted_points + rejected_points
    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if abs(max_x - min_x) < 1e-6:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-6:
        min_y -= 1.0
        max_y += 1.0
    plot_rect = rect.adjusted(56.0, 18.0, -22.0, -48.0)
    painter.setPen(QPen(QColor(COLORS.surface_border), 1.0))
    painter.drawRect(plot_rect)
    painter.setPen(QColor(COLORS.text_muted))
    painter.drawText(QRectF(plot_rect.left(), rect.bottom() - 26.0, plot_rect.width(), 20.0), Qt.AlignCenter, x_label)
    painter.save()
    painter.translate(rect.left() + 18.0, plot_rect.center().y())
    painter.rotate(-90.0)
    painter.drawText(QRectF(-plot_rect.height() / 2.0, -10.0, plot_rect.height(), 20.0), Qt.AlignCenter, y_label)
    painter.restore()
    for value in np.linspace(0.0, 1.0, 5):
        x = plot_rect.left() + (plot_rect.width() * float(value))
        y = plot_rect.bottom() - (plot_rect.height() * float(value))
        grid_pen = QPen(QColor(COLORS.chart_grid), 1.0)
        painter.setPen(grid_pen)
        painter.drawLine(QPointF(x, plot_rect.top()), QPointF(x, plot_rect.bottom()))
        painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))
    accepted_pen = QColor(COLORS.scene_measurement)
    rejected_pen = QColor(COLORS.scene_residual)
    for point in accepted_points:
        painter.setPen(Qt.NoPen)
        painter.setBrush(accepted_pen)
        painter.drawEllipse(_map_point_xy(point, plot_rect, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y), 4.0, 4.0)
    for point in rejected_points:
        painter.setPen(Qt.NoPen)
        painter.setBrush(rejected_pen)
        painter.drawEllipse(_map_point_xy(point, plot_rect, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y), 3.5, 3.5)
    painter.restore()


def _map_point_xy(
    point: tuple[float, float],
    rect: QRectF,
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> QPointF:
    x_ratio = (float(point[0]) - float(min_x)) / float(max_x - min_x)
    y_ratio = (float(point[1]) - float(min_y)) / float(max_y - min_y)
    x = rect.left() + (rect.width() * x_ratio)
    y = rect.bottom() - (rect.height() * y_ratio)
    return QPointF(float(x), float(y))
