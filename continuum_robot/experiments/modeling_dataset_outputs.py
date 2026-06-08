"""Saved artifacts for the Motor Babble modeling dataset collector."""

from __future__ import annotations

import json
import logging
import os
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
    # `accepted_sample_count` includes start/end neutral-bracket captures
    # that are filtered out of modeling_dataset_export.jsonl. The training
    # row count is `modeling_export_row_count` (or `accepted_training_row_count`)
    # which excludes those bracket captures. Surface both so a reader doesn't
    # cite the wrong "training rows collected" number.
    accepted = int(metrics.get("accepted_sample_count", 0) or 0)
    training_rows = int(
        metrics.get("modeling_export_row_count", metrics.get("accepted_training_row_count", 0)) or 0
    )
    bracket_rows = max(0, accepted - training_rows)
    return [
        ("Dataset Mode", str(metrics.get("dataset_mode", "unknown") or "unknown").replace("_", " ")),
        ("Run Label", str(metrics.get("run_label", "") or "n/a")),
        ("Dataset Tag", str(metrics.get("dataset_tag", "") or "n/a")),
        (
            "Accepted Samples (incl brackets)",
            (
                f"{accepted}  ({training_rows} training + {bracket_rows} neutral brackets)"
                if bracket_rows > 0
                else str(accepted)
            ),
        ),
        ("Training Rows Exported", str(training_rows)),
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


def write_modeling_dataset_outputs(
    *,
    output_dir: Path,
    metadata,
    summary,
    samples,
    averaged_samples: list | None = None,
    raw_tracker_frame_rows: list | None = None,
    tracker_samples_per_command: int = 1,
    averaged_label_enabled: bool = False,
    export_first_sample_label: bool = True,
    export_averaged_sample_label: bool = False,
) -> dict[str, Path]:
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

    Multi-frame post-settle averaging extras (only when
    tracker_samples_per_command > 1):
      - samples_first.jsonl, samples_averaged.jsonl: training-ready label
        variants. Schema matches samples.jsonl row-for-row.
      - raw_tracker_samples.jsonl: every per-frame snapshot captured at each
        command, so the operator can investigate whether per-command spread
        is random noise vs settling drift / outliers.

    Skeptical framing: averaging reduces *random* frame-to-frame label noise.
    It does NOT remove systematic registration/transform bias, mechanical
    hysteresis, settling drift, or outliers. Raw samples are preserved so
    these can be inspected separately.
    """
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    export_jsonl_path = output_dir / "modeling_dataset_export.jsonl"
    legacy_dat_path = output_dir / "modeling_dataset_legacy_compat.dat"
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_workspace_coverage_3d.png"
    thesis_02_path = output_dir / "thesis_02_command_and_workspace_2d.png"

    averaging_active = bool(tracker_samples_per_command > 1)

    # Canonical training-export file: always reflects the first-frame label
    # set (matches today's behaviour exactly so existing ANN training and
    # validate_legacy_ann_rows keep working unchanged).
    export_rows = _build_export_rows(samples=samples)
    with export_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Two label-variant export files appear ONLY when averaging is active.
    # Their schema is byte-identical to modeling_dataset_export.jsonl so the
    # downstream ANN loader can swap them transparently — only the row
    # contents (first frame vs averaged frame) differ.
    export_first_path: Path | None = None
    export_averaged_path: Path | None = None
    if averaging_active and export_first_sample_label:
        export_first_path = output_dir / "modeling_dataset_export_first.jsonl"
        # Same rows as canonical export; written as a separate file so the
        # operator and the ANN popout's variant selector can identify "first"
        # explicitly without parsing extra.label_kind.
        with export_first_path.open("w", encoding="utf-8") as handle:
            for row in export_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    if averaging_active and averaged_label_enabled and export_averaged_sample_label and averaged_samples:
        averaged_export_rows = _build_export_rows(samples=averaged_samples)
        export_averaged_path = output_dir / "modeling_dataset_export_averaged.jsonl"
        with export_averaged_path.open("w", encoding="utf-8") as handle:
            for row in averaged_export_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Multi-frame averaging extras: only emit the new files when actually
    # averaging. When tracker_samples_per_command == 1, samples.jsonl is the
    # sole canonical samples file (preserves today's layout for existing runs).
    samples_first_path: Path | None = None
    samples_averaged_path: Path | None = None
    raw_tracker_samples_path: Path | None = None
    if averaging_active:
        if export_first_sample_label:
            samples_first_path = output_dir / "samples_first.jsonl"
            # samples_first is exactly the canonical samples list when averaging
            # is on, because session.samples accumulates only first-frame rows
            # in that mode. We re-serialize from the live sample objects so the
            # file is independent of samples.jsonl's writer.
            with samples_first_path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
        if averaged_label_enabled and export_averaged_sample_label and averaged_samples:
            samples_averaged_path = output_dir / "samples_averaged.jsonl"
            with samples_averaged_path.open("w", encoding="utf-8") as handle:
                for sample in averaged_samples:
                    handle.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
        if raw_tracker_frame_rows:
            raw_tracker_samples_path = output_dir / "raw_tracker_samples.jsonl"
            with raw_tracker_samples_path.open("w", encoding="utf-8") as handle:
                for row in raw_tracker_frame_rows:
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
    plot_jobs: list[tuple[Path, Any]] = [
        (thesis_01_path, lambda: _write_collect_pose_thesis_01_workspace_coverage_3d(
            path=thesis_01_path, export_rows=export_rows, metrics=metrics,
        )),
        (thesis_02_path, lambda: _write_collect_pose_thesis_02_command_and_workspace_2d(
            path=thesis_02_path, export_rows=export_rows, metrics=metrics,
        )),
    ]

    # Multi-frame averaging variability figures appear only when averaging
    # actually ran and we have at least one averaged sample to plot.
    variability_paths: dict[str, Path] = {}
    if averaging_active and averaged_label_enabled and averaged_samples:
        records = _tracker_variability_records(averaged_samples)
        raw_rows = list(raw_tracker_frame_rows or [])
        if records:
            variability_paths = {
                "tracker_variability_workspace_xy_path": output_dir / "tracker_variability_workspace_xy.png",
                "tracker_variability_std_histogram_path": output_dir / "tracker_variability_std_histogram.png",
                "tracker_variability_first_vs_mean_path": output_dir / "tracker_variability_first_vs_mean.png",
                "tracker_variability_sample_spread_path": output_dir / "tracker_variability_sample_spread.png",
                "tracker_variability_std_vs_command_index_path": output_dir / "tracker_variability_std_vs_command_index.png",
            }
            plot_jobs.extend([
                (variability_paths["tracker_variability_workspace_xy_path"], lambda: _write_tracker_variability_workspace_xy(
                    path=variability_paths["tracker_variability_workspace_xy_path"], records=records,
                )),
                (variability_paths["tracker_variability_std_histogram_path"], lambda: _write_tracker_variability_std_histogram(
                    path=variability_paths["tracker_variability_std_histogram_path"], records=records,
                )),
                (variability_paths["tracker_variability_first_vs_mean_path"], lambda: _write_tracker_variability_first_vs_mean(
                    path=variability_paths["tracker_variability_first_vs_mean_path"], records=records,
                )),
                (variability_paths["tracker_variability_sample_spread_path"], lambda: _write_tracker_variability_sample_spread(
                    path=variability_paths["tracker_variability_sample_spread_path"],
                    records=records,
                    raw_rows=raw_rows,
                )),
                (variability_paths["tracker_variability_std_vs_command_index_path"], lambda: _write_tracker_variability_std_vs_command_index(
                    path=variability_paths["tracker_variability_std_vs_command_index_path"], records=records,
                )),
            ])

    for path, writer in plot_jobs:
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
    if samples_first_path is not None:
        outputs["samples_first_path"] = samples_first_path
    if samples_averaged_path is not None:
        outputs["samples_averaged_path"] = samples_averaged_path
    if raw_tracker_samples_path is not None:
        outputs["raw_tracker_samples_path"] = raw_tracker_samples_path
    if export_first_path is not None:
        outputs["export_first_path"] = export_first_path
    if export_averaged_path is not None:
        outputs["export_averaged_path"] = export_averaged_path
    for key, path in variability_paths.items():
        outputs[key] = path
    return outputs


def _summarize_tracker_variability(averaged_samples: list) -> dict[str, Any]:
    """Reduce per-command averaged-sample stats into run-level summary numbers.

    Returns mean / median / max of position_std_rms_mm and
    first_vs_mean_position_diff_mm across all averaged-row commands, plus
    counts. Pure summary, no thresholds or pass/fail logic.
    """
    if not averaged_samples:
        return {
            "command_point_count": 0,
            "orientation_average_available": False,
        }
    std_rms_values: list[float] = []
    first_vs_mean_values: list[float] = []
    orientation_spread_values: list[float] = []
    underflow_count = 0
    orientation_available_any = False
    for sample in averaged_samples:
        stats = dict(sample.extra.get("tracker_averaging", {}) or {})
        std_rms = stats.get("position_std_rms_mm")
        if isinstance(std_rms, (int, float)):
            std_rms_values.append(float(std_rms))
        first_vs_mean = stats.get("first_vs_mean_position_diff_mm")
        if isinstance(first_vs_mean, (int, float)):
            first_vs_mean_values.append(float(first_vs_mean))
        if stats.get("orientation_average_available"):
            orientation_available_any = True
            spread = stats.get("orientation_max_spread_deg")
            if isinstance(spread, (int, float)):
                orientation_spread_values.append(float(spread))
        if bool(sample.extra.get("tracker_averaging_underflow")):
            underflow_count += 1

    def _agg(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "max": None}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
        }

    return {
        "command_point_count": int(len(averaged_samples)),
        "underflow_command_count": int(underflow_count),
        "position_std_rms_mm": _agg(std_rms_values),
        "first_vs_mean_position_diff_mm": _agg(first_vs_mean_values),
        "orientation_average_available": bool(orientation_available_any),
        "orientation_max_spread_deg": _agg(orientation_spread_values),
        "skeptical_framing": (
            "averaging reduces random frame-to-frame label noise; "
            "it does not remove systematic registration/transform bias, "
            "mechanical hysteresis, settling drift, or outliers"
        ),
    }


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
            # `accepted_sample_count` includes the start/end neutral-bracket
            # captures even though they are filtered from the training export.
            # `modeling_export_row_count` (under trainability) is the number
            # actually written to modeling_dataset_export.jsonl. Do not cite
            # `accepted_sample_count` as the training-row count.
            "accepted_sample_count": accepted,
            "accepted_sample_count_includes_brackets": True,
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


# =========================================================================== #
# Tracker-variability figures (multi-frame post-settle averaging diagnostics)  #
# =========================================================================== #


def _tracker_variability_records(averaged_samples: list) -> list[dict[str, Any]]:
    """Pull per-command (averaged-row) stats into plot-friendly dicts.

    Reads `extra.tracker_averaging` produced by the multi-frame averaging
    path and pulls the averaged tip XY position from the averaged-sample's
    robot-frame pose. Returns one dict per averaged command. Empty rows
    (no T_robot_tip) are skipped so plots stay clean.
    """
    rows: list[dict[str, Any]] = []
    for sample in averaged_samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        stats = dict(extra.get("tracker_averaging", {}) or {})
        tip_payload = dict((getattr(sample, "pose_in_robot_frame", {}) or {}).get("tip", {}) or {})
        translation = list(tip_payload.get("translation_mm", []) or [])
        if len(translation) < 2:
            continue
        rows.append(
            {
                "command_index": int(getattr(sample, "step_index", 0) or 0),
                "averaged_x_mm": float(translation[0]),
                "averaged_y_mm": float(translation[1]),
                "averaged_z_mm": float(translation[2]) if len(translation) >= 3 else 0.0,
                "position_std_rms_mm": _safe_float(stats.get("position_std_rms_mm")),
                "position_max_deviation_mm": _safe_float(stats.get("position_max_deviation_mm")),
                "first_vs_mean_position_diff_mm": _safe_float(stats.get("first_vs_mean_position_diff_mm")),
                "orientation_max_spread_deg": _safe_float(stats.get("orientation_max_spread_deg")),
                "valid_sample_count": int(stats.get("valid_sample_count", 0) or 0),
                "sample_window_s": _safe_float(stats.get("sample_window_s")),
            }
        )
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Clean visual language for the tracker_variability_* diagnostic figure set.
# Keep explanatory prose in debug.json / docs; the PNGs should carry only the
# data, units, and the smallest labels needed to read them in a thesis.
# ---------------------------------------------------------------------------

_THESIS_FIG_TITLE_SIZE = 12.5
_THESIS_FIG_AXES_LABEL_SIZE = 10.5
_THESIS_FIG_TICK_SIZE = 9.5
_THESIS_FIG_LEGEND_SIZE = 9.0

_METRIC_LABEL = "3D RMS spread (mm)"
"""Canonical name for the per-command variability metric.

This is the RMS combination of per-axis sample standard deviations across the
post-settle tracker frames at one commanded pose. Reported as a sample estimate
(ddof=1, see the 2026-05-20 audit pass) and expressed in millimetres.
"""


def _set_variability_title(fig, title: str) -> None:
    fig.suptitle(
        title,
        fontsize=_THESIS_FIG_TITLE_SIZE,
        fontweight="bold",
        x=0.04,
        ha="left",
        y=0.975,
    )


def _set_variability_tick_sizes(ax) -> None:
    ax.tick_params(axis="both", labelsize=_THESIS_FIG_TICK_SIZE)
    ax.xaxis.label.set_size(_THESIS_FIG_AXES_LABEL_SIZE)
    ax.yaxis.label.set_size(_THESIS_FIG_AXES_LABEL_SIZE)


def _write_tracker_variability_workspace_xy(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Workspace map of within-command pose spread.

    Each marker is one commanded pose plotted at its averaged tip XY in the
    robot frame; colour encodes the per-command 3D RMS spread across the
    N post-settle tracker frames at that command. Reader sees whether
    spread is spatially uniform or clusters in specific workspace regions —
    a clustered pattern usually indicates a real mechanical or measurement
    effect at that location rather than uniform tracker noise.
    """
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.88, top=0.90, bottom=0.12)

    valid = [
        (r["averaged_x_mm"], r["averaged_y_mm"], r["position_std_rms_mm"])
        for r in records
        if r["position_std_rms_mm"] is not None
    ]
    if not valid:
        ax.text(0.5, 0.5, "No averaged samples with position spread data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
        _set_variability_title(fig, "Workspace Variability")
        save_figure(fig, path)
        return

    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    stds = [v[2] for v in valid]
    scatter = ax.scatter(
        xs,
        ys,
        c=stds,
        cmap="plasma",
        s=44,
        edgecolors="white",
        linewidths=0.45,
        alpha=0.95,
    )
    ax.set_aspect("equal", adjustable="datalim")
    style_axes(ax, xlabel="Robot X (mm)", ylabel="Robot Y (mm)")
    _set_variability_tick_sizes(ax)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.025, shrink=0.88)
    cbar.set_label(_METRIC_LABEL, fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=_THESIS_FIG_TICK_SIZE)
    cbar.outline.set_edgecolor(color("grid"))

    _set_variability_title(fig, "Workspace Variability")
    save_figure(fig, path)


def _write_tracker_variability_std_histogram(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Distribution of per-command 3D RMS spread across the run.

    Answers: across the run as a whole, how big is the within-command
    pose spread typically, and how long is the tail? A heavy tail says
    a few commands had much wider spread than the median — often the
    same commands that show settling-drift directional clusters in
    :func:`_write_tracker_variability_sample_spread`.
    """
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.16)

    stds = [r["position_std_rms_mm"] for r in records if r["position_std_rms_mm"] is not None]
    if not stds:
        ax.text(0.5, 0.5, "No averaged samples with position spread data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel=_METRIC_LABEL, ylabel="Command count")
        _set_variability_title(fig, "Spread Distribution")
        save_figure(fig, path)
        return
    arr = np.asarray(stds, dtype=float)
    bins = min(30, max(10, len(arr) // 4))
    ax.hist(arr, bins=bins, color=color("measured"), edgecolor="white", alpha=0.92)
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    ax.axvline(median, color=color("reference"), linestyle="--", linewidth=1.4,
               label=f"median {median:.3f}")
    ax.axvline(p95, color=color("rejected"), linestyle="-.", linewidth=1.3,
               label=f"p95 {p95:.3f}")
    style_axes(ax, xlabel=_METRIC_LABEL, ylabel="Command count")
    _set_variability_tick_sizes(ax)
    legend_obj = ax.legend(loc="upper right", frameon=True, facecolor="white",
                           edgecolor=color("grid"), framealpha=0.95,
                           fontsize=_THESIS_FIG_LEGEND_SIZE)
    if legend_obj is not None:
        legend_obj.get_frame().set_linewidth(0.6)
    _set_variability_title(fig, "Spread Distribution")
    save_figure(fig, path)


def _write_tracker_variability_first_vs_mean(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Distance between the first-frame label and the averaged label per command.

    For each commanded pose this is ``‖first_position − mean_position‖``
    across the N post-settle frames. Indicates how much the first-frame
    label set (the legacy capture path) shifts when replaced by the
    frame-averaged label set — useful for arguing whether the two label
    variants train materially different ANN models.
    """
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.16)

    diffs = [
        r["first_vs_mean_position_diff_mm"]
        for r in records
        if r["first_vs_mean_position_diff_mm"] is not None
    ]
    if not diffs:
        ax.text(0.5, 0.5, "No first-vs-mean difference data available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel="First-to-mean shift (mm)", ylabel="Command count")
        _set_variability_title(fig, "First Frame vs Mean")
        save_figure(fig, path)
        return
    arr = np.asarray(diffs, dtype=float)
    bins = min(30, max(10, len(arr) // 4))
    ax.hist(arr, bins=bins, color=color("fit"), edgecolor="white", alpha=0.92)
    median = float(np.median(arr))
    p95 = float(np.percentile(arr, 95))
    ax.axvline(median, color=color("reference"), linestyle="--", linewidth=1.4,
               label=f"median {median:.3f}")
    ax.axvline(p95, color=color("rejected"), linestyle="-.", linewidth=1.3,
               label=f"p95 {p95:.3f}")
    style_axes(ax, xlabel="First-to-mean shift (mm)", ylabel="Command count")
    _set_variability_tick_sizes(ax)
    legend_obj = ax.legend(loc="upper right", frameon=True, facecolor="white",
                           edgecolor=color("grid"), framealpha=0.95,
                           fontsize=_THESIS_FIG_LEGEND_SIZE)
    if legend_obj is not None:
        legend_obj.get_frame().set_linewidth(0.6)
    _set_variability_title(fig, "First Frame vs Mean")
    save_figure(fig, path)


def _write_tracker_variability_sample_spread(
    *,
    path: Path,
    records: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> None:
    """Raw per-frame XY clusters at three representative commands.

    Shows the actual per-frame XY positions (centred on each command's
    mean) for: the command with the tightest cluster (min 3D RMS spread),
    the command nearest the median, and the command with the loosest
    cluster (max 3D RMS spread). If the loose cluster shows a directional
    drift instead of an isotropic blob, averaging is hiding settling/drift
    rather than just smoothing tracker noise.
    """
    with report_style() as plt:
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.15, wspace=0.28)

    valid = [(r["command_index"], r["position_std_rms_mm"])
             for r in records if r["position_std_rms_mm"] is not None]
    if not valid or not raw_rows:
        for ax in np.atleast_1d(axes):
            ax.text(0.5, 0.5, "No raw frames", transform=ax.transAxes,
                    ha="center", va="center", fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
            style_axes(ax, xlabel="dX (mm)", ylabel="dY (mm)")
        _set_variability_title(fig, "Raw Frame Spread")
        save_figure(fig, path)
        return

    valid_sorted = sorted(valid, key=lambda item: item[1])
    pick_min = valid_sorted[0]
    pick_med = valid_sorted[len(valid_sorted) // 2]
    pick_max = valid_sorted[-1]
    picks = [
        (int(pick_min[0]), float(pick_min[1]), "tightest"),
        (int(pick_med[0]), float(pick_med[1]), "median"),
        (int(pick_max[0]), float(pick_max[1]), "loosest"),
    ]
    raw_by_cmd: dict[int, list[dict[str, Any]]] = {}
    for row in raw_rows:
        try:
            cmd_idx = int(row.get("command_index", -1))
        except (TypeError, ValueError):
            cmd_idx = -1
        raw_by_cmd.setdefault(cmd_idx, []).append(row)

    picked_frame_sets: list[tuple[int, float, str, list[float], list[float]]] = []
    common_limit = 0.05
    for cmd_idx, std_rms, name in picks:
        frames = raw_by_cmd.get(cmd_idx, [])
        xs: list[float] = []
        ys: list[float] = []
        for frame in frames:
            robot_pose = dict(frame.get("pose_in_robot_frame", {}) or {})
            translation = list(robot_pose.get("translation_mm", []) or [])
            if len(translation) < 2:
                continue
            xs.append(float(translation[0]))
            ys.append(float(translation[1]))
        rel_xs: list[float] = []
        rel_ys: list[float] = []
        if xs:
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            rel_xs = [x - cx for x in xs]
            rel_ys = [y - cy for y in ys]
            common_limit = max(common_limit, max(abs(value) for value in rel_xs + rel_ys))
        picked_frame_sets.append((cmd_idx, std_rms, name, rel_xs, rel_ys))

    common_limit *= 1.15
    for ax_index, (ax, (cmd_idx, std_rms, name, rel_xs, rel_ys)) in enumerate(
        zip(np.atleast_1d(axes), picked_frame_sets)
    ):
        if len(rel_xs) < 1:
            ax.text(0.5, 0.5, f"No raw frames\nfor cmd {cmd_idx}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
            style_axes(ax, xlabel="dX (mm)", ylabel="dY (mm)" if ax_index == 0 else "")
            ax.set_xlim(-common_limit, common_limit)
            ax.set_ylim(-common_limit, common_limit)
            ax.set_title(f"{name}\ncmd {cmd_idx} | {std_rms:.3f} mm",
                         fontsize=_THESIS_FIG_AXES_LABEL_SIZE, color=color("text"),
                         loc="left", pad=8)
            continue
        ax.scatter(rel_xs, rel_ys, color=color("measured"), s=34,
                   edgecolors="white", linewidths=0.4)
        ax.axhline(0, color=color("grid"), linewidth=0.6)
        ax.axvline(0, color=color("grid"), linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-common_limit, common_limit)
        ax.set_ylim(-common_limit, common_limit)
        style_axes(ax, xlabel="dX (mm)", ylabel="dY (mm)" if ax_index == 0 else "")
        _set_variability_tick_sizes(ax)
        if ax_index != 0:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(
            f"{name}\ncmd {cmd_idx} | {std_rms:.3f} mm | n={len(rel_xs)}",
            fontsize=_THESIS_FIG_AXES_LABEL_SIZE,
            color=color("text"),
            loc="left",
            pad=8,
        )

    _set_variability_title(fig, "Raw Frame Spread")
    save_figure(fig, path)


def _write_tracker_variability_std_vs_command_index(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Per-command 3D RMS spread over collection order.

    Does the within-command pose spread grow over the session? A flat trend
    says it stays stationary; a rising trend points at time-correlated
    drift — the tracker warming up, a coil shifting, mechanical loosening
    — that the averaging path cannot remove on its own.
    """
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.98, top=0.86, bottom=0.16)

    rows_sorted = sorted(
        (r for r in records if r["position_std_rms_mm"] is not None),
        key=lambda r: int(r["command_index"]),
    )
    if not rows_sorted:
        ax.text(0.5, 0.5, "No averaged samples with position spread data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=_THESIS_FIG_AXES_LABEL_SIZE)
        style_axes(ax, xlabel="Command index (collection order)",
                   ylabel=_METRIC_LABEL)
        _set_variability_title(fig, "Variability Over Collection")
        save_figure(fig, path)
        return
    xs = [int(r["command_index"]) for r in rows_sorted]
    ys = [float(r["position_std_rms_mm"]) for r in rows_sorted]
    ax.plot(xs, ys, color=color("measured"), linewidth=0.8,
            marker="o", markersize=3.0, markerfacecolor=color("measured"),
            markeredgecolor="white", markeredgewidth=0.25, alpha=0.80,
            label="command")
    # Rolling mean (window ≈ N/20, min 5) so the underlying trend reads at a
    # glance without the operator scrolling through every dot.
    window = max(5, len(ys) // 20)
    if len(ys) >= window:
        kernel = np.ones(window) / float(window)
        smoothed = np.convolve(ys, kernel, mode="valid")
        smooth_x = xs[(window - 1) // 2 : (window - 1) // 2 + len(smoothed)]
        if len(smooth_x) == len(smoothed):
            ax.plot(smooth_x, smoothed, color=color("rejected"),
                    linewidth=1.6, label=f"rolling mean ({window})")
    style_axes(ax, xlabel="Command index (collection order)",
               ylabel=_METRIC_LABEL)
    _set_variability_tick_sizes(ax)
    legend_obj = ax.legend(loc="upper right", frameon=True, facecolor="white",
                           edgecolor=color("grid"), framealpha=0.95,
                           fontsize=_THESIS_FIG_LEGEND_SIZE)
    if legend_obj is not None:
        legend_obj.get_frame().set_linewidth(0.6)
    _set_variability_title(fig, "Variability Over Collection")
    save_figure(fig, path)
