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
from continuum_robot.experiments.plotting import (
    color,
    create_3d_figure,
    create_figure,
    legend,
    report_style,
    save_figure,
    set_equal_xyz,
    style_3d_axes,
    style_axes,
)
from continuum_robot.experiments.tracker_timing_outputs import _fmt
from continuum_robot.data.model_training_validity import NON_TRAINING_PHASES, sample_has_complete_command_servo_tip


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


def write_modeling_dataset_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write canonical artifacts for one Motor Babble collect_pose run.

    Figure contract: 2 thesis-quality PNGs.
      - thesis_01_workspace_coverage_3d.png: 3D scatter of accepted tip
        positions in robot frame, colored by sample index (time order).
        Shows workspace coverage and the order the babble walked through it.
      - thesis_02_command_and_workspace_2d.png: stacked-panel 2D scatters.
        Top = pair_13 vs pair_24 (command space). Bottom = tip XY (workspace
        projection). Both colored by sample index with a shared colorbar.

    Other artifacts:
      - debug.json: acceptance / rejection breakdown, workspace + command
        extents, trainability flags, provenance (registration / runtime_tip /
        pretension), run_review fold-in.
      - modeling_dataset_export.jsonl: ordered export rows with explicit
        accepted flag (consumed by ANN training pipeline). UNCHANGED.
      - modeling_dataset_legacy_compat.dat: accepted rows only in legacy DAT
        format (consumed by legacy ANN). UNCHANGED.

    Dropped (no real consumers / duplicated by debug.json):
      - modeling_dataset_summary.txt (only path-captured by ann_training as
        an optional reference, never parsed)
      - modeling_workspace_coverage.png + modeling_workspace_coverage_report.png
      - modeling_command_distribution.png + commanded_tendon_space_report.png
    """
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    export_jsonl_path = output_dir / "modeling_dataset_export.jsonl"
    legacy_dat_path = output_dir / "modeling_dataset_legacy_compat.dat"
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_workspace_coverage_3d.png"
    thesis_02_path = output_dir / "thesis_02_command_and_workspace_2d.png"

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

    _write_collect_pose_debug_json(
        path=debug_json_path,
        output_dir=output_dir,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
        export_rows=export_rows,
        samples=samples,
    )
    for path, writer in [
        (thesis_01_path, lambda: _write_collect_pose_thesis_01_workspace_coverage_3d(
            path=thesis_01_path, export_rows=export_rows, metrics=metrics,
        )),
        (thesis_02_path, lambda: _write_collect_pose_thesis_02_command_and_workspace_2d(
            path=thesis_02_path, export_rows=export_rows, metrics=metrics,
        )),
    ]:
        try:
            writer()
        except Exception:
            LOG.exception("Failed to render %s; writing placeholder", path)
            _write_plot_placeholder(path)

    outputs: dict[str, Path] = {
        "export_jsonl_path": export_jsonl_path,
        "debug_json_path": debug_json_path,
        "thesis_01_path": thesis_01_path,
        "thesis_02_path": thesis_02_path,
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


def _write_collect_pose_thesis_01_workspace_coverage_3d(
    *,
    path: Path,
    export_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 1: 3D scatter of accepted tip positions colored by time order.

    Each point is one accepted training sample at its tip position in the
    robot frame; color encodes its index in the collection sequence
    (viridis from first to last). Reader sees the workspace extent AND the
    order the babble walked through it. Mode-agnostic — works for
    workspace_coverage, angular_test_mesh, hysteresis, or repeatability.
    """
    points: list[tuple[float, float, float]] = []
    for row in export_rows:
        tip = row.get("tip_position_xyz_mm")
        if not (isinstance(tip, list) and len(tip) >= 3):
            continue
        try:
            points.append((float(tip[0]), float(tip[1]), float(tip[2])))
        except (TypeError, ValueError):
            continue

    fig, ax = create_3d_figure(size="thesis_3d")
    if not points:
        ax.text2D(0.5, 0.5, "No accepted tip positions available",
                  transform=ax.transAxes, ha="center", va="center")
        style_3d_axes(ax, title="")
        fig.suptitle("Workspace Coverage During Babble Collection",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    indices = np.arange(len(points), dtype=float)

    scatter = ax.scatter(
        xs, ys, zs,
        c=indices, cmap="viridis",
        s=14, depthshade=True, alpha=0.85, linewidths=0,
    )

    set_equal_xyz(ax, x_values=xs, y_values=ys, z_values=zs, minimum_span=5.0, pad_fraction=0.08)
    style_3d_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)", zlabel="Robot Z (mm)")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.55, pad=0.12)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    mode = str(metrics.get("dataset_mode", "unknown") or "unknown").replace("_", " ")
    fig.suptitle(f"Workspace Coverage During Babble Collection  ({mode})",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    workspace_span = dict(metrics.get("workspace_span_mm", {}) or {})
    x_span = workspace_span.get("x_span_mm") or (max(xs) - min(xs))
    y_span = workspace_span.get("y_span_mm") or (max(ys) - min(ys))
    z_span = workspace_span.get("z_span_mm") or (max(zs) - min(zs))
    accepted = int(metrics.get("accepted_sample_count", len(points)) or len(points))
    rejected = int(metrics.get("rejected_sample_count", 0) or 0)
    total = accepted + rejected
    accept_rate = (accepted / total * 100.0) if total > 0 else 100.0
    fig.text(
        0.015, 0.02,
        "  •  ".join(_compact_strip([
            f"Mode: {mode}",
            f"Accepted: {accepted}/{total}  ({accept_rate:.1f}%)",
            f"Workspace span: X={x_span:.1f} mm  Y={y_span:.1f} mm  Z={z_span:.1f} mm",
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_collect_pose_thesis_02_command_and_workspace_2d(
    *,
    path: Path,
    export_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    """Thesis figure 2: stacked 2D scatters of command space (top) and tip XY (bottom).

    Both colored by sample index so the reader can visually correlate a
    command sample with its corresponding tip outcome — the same dot color
    in the top and bottom panels is the same sample in time. Single shared
    colorbar on the right.
    """
    commands: list[tuple[float, float]] = []
    tip_xy: list[tuple[float, float]] = []
    indices: list[int] = []
    for index, row in enumerate(export_rows):
        pair = row.get("requested_pair_command_cm")
        tip = row.get("tip_position_xyz_mm")
        if not (isinstance(pair, list) and len(pair) >= 2):
            continue
        if not (isinstance(tip, list) and len(tip) >= 2):
            continue
        try:
            commands.append((float(pair[0]), float(pair[1])))
            tip_xy.append((float(tip[0]), float(tip[1])))
        except (TypeError, ValueError):
            continue
        indices.append(index)

    with report_style() as plt:
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(7.6, 7.6), constrained_layout=False,
        )
    fig.subplots_adjust(left=0.10, right=0.86, top=0.92, bottom=0.13, hspace=0.30)

    if not commands or not tip_xy:
        for ax in (ax_top, ax_bottom):
            ax.text(0.5, 0.5, "No accepted samples available",
                    transform=ax.transAxes, ha="center", va="center")
        fig.suptitle("Command Space (top) and Tip Workspace XY (bottom)",
                     fontsize=13, fontweight="bold", x=0.04, ha="left")
        save_figure(fig, path)
        return

    color_values = np.asarray(indices, dtype=float)

    cmd_x = [point[0] for point in commands]
    cmd_y = [point[1] for point in commands]
    sc_top = ax_top.scatter(cmd_x, cmd_y, c=color_values, cmap="viridis",
                            s=14, alpha=0.85, linewidths=0)
    style_axes(ax_top, xlabel="Pair 1/3 command (cm)", ylabel="Pair 2/4 command (cm)")
    ax_top.set_title("Command space", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_top.set_aspect("equal", adjustable="datalim")

    tip_x = [point[0] for point in tip_xy]
    tip_y = [point[1] for point in tip_xy]
    ax_bottom.scatter(tip_x, tip_y, c=color_values, cmap="viridis",
                      s=14, alpha=0.85, linewidths=0)
    style_axes(ax_bottom, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
    ax_bottom.set_title("Tip workspace (top-down XY)", loc="left", pad=8, fontsize=11, fontweight="bold")
    ax_bottom.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(sc_top, ax=(ax_top, ax_bottom), shrink=0.85, pad=0.025)
    cbar.set_label("Sample index (time order)")
    cbar.outline.set_edgecolor(color("grid"))

    mode = str(metrics.get("dataset_mode", "unknown") or "unknown").replace("_", " ")
    fig.suptitle(f"Command Space (top) and Tip Workspace XY (bottom)  —  {mode}",
                 fontsize=13, fontweight="bold", x=0.04, ha="left")

    cmd_range_x = max(cmd_x) - min(cmd_x)
    cmd_range_y = max(cmd_y) - min(cmd_y)
    tip_range_x = max(tip_x) - min(tip_x)
    tip_range_y = max(tip_y) - min(tip_y)
    fig.text(
        0.015, 0.02,
        "  •  ".join(_compact_strip([
            f"Samples: {len(commands)}",
            f"Command extent: pair_13 {cmd_range_x:.2f} cm, pair_24 {cmd_range_y:.2f} cm",
            f"Tip XY extent: X {tip_range_x:.1f} mm, Y {tip_range_y:.1f} mm",
        ])),
        fontsize=9, color=color("text"), ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": color("grid"), "alpha": 0.94},
    )
    save_figure(fig, path)


def _write_collect_pose_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
    export_rows: list[dict[str, Any]],
    samples,
) -> None:
    """Consolidate every diagnostic that doesn't make it onto the thesis figures."""
    accepted = int(metrics.get("accepted_sample_count", 0) or 0)
    rejected = int(metrics.get("rejected_sample_count", 0) or 0)
    rejection_reasons = dict(metrics.get("rejection_reasons", {}) or {})
    phase_counts: dict[str, int] = {}
    for sample in samples:
        phase = str(getattr(sample, "phase", "") or "")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    # Compute workspace + command extents from export rows so this file is
    # self-contained even when summary.json doesn't carry the precomputed stats.
    tip_xyz = np.asarray(
        [row.get("tip_position_xyz_mm") for row in export_rows
         if isinstance(row.get("tip_position_xyz_mm"), list) and len(row.get("tip_position_xyz_mm")) >= 3],
        dtype=float,
    )
    cmd_pair = np.asarray(
        [row.get("requested_pair_command_cm") for row in export_rows
         if isinstance(row.get("requested_pair_command_cm"), list) and len(row.get("requested_pair_command_cm")) >= 2],
        dtype=float,
    )
    workspace_extent_mm = None
    if tip_xyz.size:
        workspace_extent_mm = {
            "x_min": float(tip_xyz[:, 0].min()), "x_max": float(tip_xyz[:, 0].max()), "x_span": float(np.ptp(tip_xyz[:, 0])),
            "y_min": float(tip_xyz[:, 1].min()), "y_max": float(tip_xyz[:, 1].max()), "y_span": float(np.ptp(tip_xyz[:, 1])),
            "z_min": float(tip_xyz[:, 2].min()), "z_max": float(tip_xyz[:, 2].max()), "z_span": float(np.ptp(tip_xyz[:, 2])),
        }
    command_extent_cm = None
    if cmd_pair.size:
        command_extent_cm = {
            "pair_13_min": float(cmd_pair[:, 0].min()), "pair_13_max": float(cmd_pair[:, 0].max()),
            "pair_24_min": float(cmd_pair[:, 1].min()), "pair_24_max": float(cmd_pair[:, 1].max()),
        }

    provenance = dict(metrics.get("run_provenance", {}) or {})
    trainability = {
        "model_training_validity_status": metrics.get("model_training_validity_status"),
        "model_training_validity_reason": metrics.get("model_training_validity_reason"),
        "model_training_hard_invalidation_reasons": list(metrics.get("model_training_hard_invalidation_reasons", []) or []),
        "model_training_warnings": list(metrics.get("model_training_warnings", []) or []),
        "modeling_export_row_count": int(metrics.get("modeling_export_row_count", 0) or 0),
        "modeling_legacy_row_count": int(metrics.get("modeling_legacy_row_count", 0) or 0),
        "accepted_training_row_count": int(metrics.get("accepted_training_row_count", 0) or 0),
        "accepted_rows_complete": bool(metrics.get("accepted_rows_complete")),
        "dropped_samples_excluded_from_training": bool(metrics.get("dropped_samples_excluded_from_training")),
        "valid_for_model_training": bool(metrics.get("valid_for_model_training")),
        "valid_for_thesis_repeatability": bool(metrics.get("valid_for_thesis_repeatability")),
        "lower_trust_active": bool(metrics.get("lower_trust_active")),
    }
    long_run_health = {
        "recovered_packet_error_count": int(metrics.get("recovered_packet_error_count", 0) or 0),
        "unrecovered_packet_error_count": int(metrics.get("unrecovered_packet_error_count", 0) or 0),
        "servo_telemetry_retry_count": int(metrics.get("servo_telemetry_retry_count", 0) or 0),
        "dropped_post_motion_telemetry_samples": int(metrics.get("dropped_post_motion_telemetry_samples", 0) or 0),
        "dropped_pre_motion_telemetry_samples": int(metrics.get("dropped_pre_motion_telemetry_samples", 0) or 0),
        "consecutive_post_motion_packet_failures": int(metrics.get("consecutive_post_motion_packet_failures", 0) or 0),
        "total_post_motion_packet_failure_events": int(metrics.get("total_post_motion_packet_failure_events", 0) or 0),
        "write_goal_packet_error_count": int(metrics.get("write_goal_packet_error_count", 0) or 0),
    }
    review_payload: dict[str, Any] | None = None
    review_path = output_dir / "run_review.json"
    if review_path.exists():
        try:
            review_payload = json.loads(review_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            review_payload = {"parse_error": True}

    payload = {
        "schema_version": "1.0",
        "run_id": getattr(metadata, "run_id", None),
        "experiment_name": getattr(metadata, "experiment_name", "collect_pose_command_dataset"),
        "status": getattr(summary, "status", None),
        "dataset_mode": metrics.get("dataset_mode"),
        "run_label": metrics.get("run_label"),
        "dataset_tag": metrics.get("dataset_tag"),
        "acceptance": {
            "accepted_sample_count": accepted,
            "rejected_sample_count": rejected,
            "acceptance_rate": (accepted / (accepted + rejected)) if (accepted + rejected) > 0 else None,
            "rejection_reasons": rejection_reasons,
        },
        "phase_counts": phase_counts,
        "workspace_extent_mm": workspace_extent_mm,
        "command_extent_cm": command_extent_cm,
        "trainability": trainability,
        "provenance": provenance,
        "long_run_health": long_run_health,
        "run_review": review_payload,
        "note": (
            "Per-pair time alignment between tracker frames and servo telemetry "
            "is characterized by servo_tracker_sync_validation. The same pairing "
            "mechanism is used during this experiment; see docs/timing_audit.md."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_collect_pose_json_default), encoding="utf-8")


def _compact_strip(items: list[str | None]) -> list[str]:
    return [str(item) for item in items if item]


def _collect_pose_json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unserialisable type: {type(value).__name__}")
