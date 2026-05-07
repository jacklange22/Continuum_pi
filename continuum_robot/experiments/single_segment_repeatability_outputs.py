"""Saved artifacts for the single-segment repeatability experiment."""

from __future__ import annotations

import logging
import csv
from pathlib import Path
from typing import Any

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tip_or_tool_position_mm
from continuum_robot.experiments.plotting import (
    add_metric_box,
    color,
    create_figure,
    legend,
    save_figure,
    set_equal_xy,
    style_axes,
)
from continuum_robot.experiments.tracker_timing_outputs import (
    _body_font,
    _draw_bar_chart,
    _draw_panel,
    _draw_summary_pairs,
    _draw_title,
    _ensure_plot_qt_app,
    _fmt,
    _new_image,
    _qt_plotting_is_safe,
)
try:
    from continuum_robot.gui.theme import COLORS
except Exception:
    class _FallbackColors:
        scene_truth = "#2563eb"
        scene_residual = "#dc2626"
        scene_measurement = "#7c3aed"
        scene_tip = "#f59e0b"
        selection_bg = "#0891b2"

    COLORS = _FallbackColors()
try:
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QPen, QPainter

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


LOG = logging.getLogger(__name__)


def build_single_segment_repeatability_summary_pairs(*, metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Return compact rows for GUI/report summaries."""
    run_validity = dict(metrics.get("run_validity", {}) or {})
    criteria = dict(run_validity.get("criteria", {}) or {})
    observed = dict(run_validity.get("observed", {}) or {})
    geometry = dict(metrics.get("target_geometry", {}) or {})
    repeat_valid = int(metrics.get("valid_repeat_sample_count", 0) or 0)
    repeat_planned = int(
        criteria.get("planned_repeat_capture_count", metrics.get("planned_visit_count", 0) or 0) or 0
    )
    rejected_count = int(metrics.get("rejected_capture_count", 0) or 0)
    rejected_fraction = observed.get("rejected_capture_fraction")
    pairs = [
        ("Protocol", "Legacy 17 target / all-other approaches"),
        ("Runtime Tip Mode", str(metrics.get("runtime_tip_mode_used", "unknown") or "unknown")),
        ("Runtime Tip Trust", str(metrics.get("runtime_tip_trust_level", "unknown") or "unknown")),
        ("Thesis Trusted", "yes" if metrics.get("thesis_trusted_run") else "no"),
        ("Targets", str(int(metrics.get("target_count", 0) or 0))),
        (
            "Ring Radii",
            (
                f"inner {_fmt(geometry.get('inner_ring_radius_mm'), suffix=' mm')} "
                f"({_fmt(geometry.get('inner_ring_radius_ticks'), suffix=' ticks')}), "
                f"outer {_fmt(geometry.get('outer_ring_radius_mm'), suffix=' mm')} "
                f"({_fmt(geometry.get('outer_ring_radius_ticks'), suffix=' ticks')})"
                if geometry
                else "n/a"
            ),
        ),
        (
            "Amplitude Cap",
            (
                f"{_fmt(geometry.get('max_target_tick_delta_from_startup_cap'), suffix=' ticks')} "
                f"(max target {_fmt(geometry.get('max_target_abs_tick_delta_from_startup'), suffix=' ticks')})"
                if geometry
                else "n/a"
            ),
        ),
        (
            "Spool Diameter",
            _fmt(geometry.get("spool_diameter_mm"), suffix=" mm") if geometry else "n/a",
        ),
        ("Planned Captures", str(int(metrics.get("planned_capture_count", 0) or 0))),
        ("Run Validity", "PASS" if run_validity.get("thesis_valid_run") else "FAIL"),
        (
            "Repeat Coverage",
            f"{repeat_valid}/{repeat_planned}" if repeat_planned > 0 else str(repeat_valid),
        ),
        (
            "Per-Target Minimum",
            (
                f">= {int(criteria.get('min_repeat_captures_per_target', 0) or 0)} repeats"
                if criteria
                else "n/a"
            ),
        ),
        ("Valid Repeat Captures", str(repeat_valid)),
        ("Valid Approach Captures", str(int(metrics.get("valid_approach_sample_count", 0) or 0))),
        (
            "Rejected Captures",
            f"{rejected_count} ({float(rejected_fraction) * 100.0:.1f}%)"
            if rejected_fraction is not None
            else str(rejected_count),
        ),
        ("Pose Frame", str(metrics.get("position_frame", "unknown") or "unknown")),
        ("Overall RMS", _fmt(metrics.get("overall_repeatability_rms_mm"), suffix=" mm")),
        ("Overall Max", _fmt(metrics.get("overall_max_deviation_mm"), suffix=" mm")),
        ("Path Dependence RMS", _fmt(metrics.get("path_dependence_rms_mm"), suffix=" mm")),
        ("Thesis Goal", f"<= {_fmt(metrics.get('thesis_goal_rms_mm'), suffix=' mm')}"),
        ("Goal Result", "PASS" if metrics.get("thesis_goal_pass") else "not met"),
    ]
    comparison = dict(metrics.get("baseline_comparison", {}) or {})
    if comparison.get("available"):
        overall = dict(comparison.get("overall_rms", {}) or {})
        max_dev = dict(comparison.get("overall_max_deviation", {}) or {})
        path = dict(comparison.get("path_dependence_rms", {}) or {})
        rejected = dict(comparison.get("rejected_capture_count", {}) or {})
        pairs.extend(
            [
                ("Baseline Comparison", "available"),
                ("Delta Overall RMS", _signed_fmt(overall.get("delta"), suffix=" mm")),
                ("Delta Overall Max", _signed_fmt(max_dev.get("delta"), suffix=" mm")),
                ("Delta Path RMS", _signed_fmt(path.get("delta"), suffix=" mm")),
                ("Delta Rejected Captures", _signed_int(rejected.get("delta"))),
            ]
        )
    elif comparison:
        pairs.append(("Baseline Comparison", f"unavailable: {comparison.get('error', 'unknown error')}"))
    return pairs


def build_single_segment_repeatability_summary_lines(*, metadata, summary, metrics: dict[str, Any]) -> list[str]:
    """Return human-readable summary text for a repeatability run."""
    lines = [
        "Single-Segment Repeatability Summary",
        (
            "This run recreates the legacy 17-point single-segment protocol using the current "
            "tracker, servo, registration, and runtime-tip chain. Repeatability metrics are computed "
            "from repeat captures after returning to the desired target from every other target."
        ),
        "",
        f"Run ID: {metadata.run_id}",
        f"Timestamp: {metadata.timestamp_utc}",
        f"Status: {summary.status}",
    ]
    for label, value in build_single_segment_repeatability_summary_pairs(metrics=metrics):
        lines.append(f"{label}: {value}")
    run_validity = dict(metrics.get("run_validity", {}) or {})
    criteria = dict(run_validity.get("criteria", {}) or {})
    observed = dict(run_validity.get("observed", {}) or {})
    lines.extend(["", "Run validity:"])
    lines.append(
        f"- Accepted repeat coverage: {int(observed.get('valid_repeat_sample_count', 0) or 0)} / "
        f"{int(criteria.get('planned_repeat_capture_count', 0) or 0)} "
        f"({_fmt((observed.get('accepted_repeat_fraction') or 0.0) * 100.0, suffix=' %')})"
        if observed.get("accepted_repeat_fraction") is not None
        else "- Accepted repeat coverage: n/a"
    )
    lines.append(
        f"- Per-target minimum: {int(criteria.get('min_repeat_captures_per_target', 0) or 0)} repeats "
        f"for all {int(metrics.get('target_count', 0) or 0)} targets"
    )
    lines.append(
        f"- Rejected capture rate: {_fmt((observed.get('rejected_capture_fraction') or 0.0) * 100.0, suffix=' %')}"
        if observed.get("rejected_capture_fraction") is not None
        else "- Rejected capture rate: n/a"
    )
    for reason in list(run_validity.get("failure_reasons", []) or []):
        lines.append(f"- Invalid reason: {reason}")
    for reason in list(run_validity.get("warning_reasons", []) or []):
        lines.append(f"- Coverage note: {reason}")
    legacy_style = dict(metrics.get("legacy_style_comparison", {}) or {})
    if legacy_style:
        selection = dict(legacy_style.get("sample_selection", {}) or {})
        lines.extend(
            [
                "",
                "Legacy-style comparison:",
                "- Definition: desired-target repeat captures only; mean XYZ centroid; "
                "XY_RMSE uses XY-plane distances; XYZ_RMSE/max use 3D Euclidean distances. "
                "The legacy plot labels used XYZ_RMSE while displaying XY scatter.",
                f"- Pose frame used: {legacy_style.get('pose_frame_used', 'unknown')}",
                (
                    f"- Selected repeat samples: {selection.get('selected_repeat_sample_count', 'n/a')} "
                    f"(rejected/excluded {selection.get('rejected_repeat_sample_count', 'n/a')}; "
                    f"filter {selection.get('outlier_filter', 'n/a')})"
                ),
                f"- Overall legacy-style XY RMSE: {_fmt(legacy_style.get('overall_XY_RMSE_mm'), suffix=' mm')}",
                f"- Overall legacy-style XYZ RMSE: {_fmt(legacy_style.get('overall_XYZ_RMSE_mm'), suffix=' mm')}",
                f"- Overall legacy-style max deviation: {_fmt(legacy_style.get('overall_max_deviation_mm'), suffix=' mm')}",
            ]
        )
    transform_audit = dict(metrics.get("transform_chain_audit", {}) or {})
    if transform_audit:
        lines.extend(
            [
                "",
                "Transform-chain audit:",
                f"- Registration artifact: {transform_audit.get('registration_artifact_path', 'n/a')}",
                f"- Runtime tip artifact: {transform_audit.get('runtime_tip_artifact_path', 'n/a')}",
                f"- Runtime tip mode/trust: {transform_audit.get('runtime_tip_mode', 'unknown')} / {transform_audit.get('runtime_tip_trust_level', 'unknown')}",
                f"- Pose frame used for analysis: {transform_audit.get('pose_frame_used_for_analysis', 'unknown')}",
                f"- Identity/fallback behavior: {'yes' if transform_audit.get('identity_or_fallback') else 'no'}",
            ]
        )
    per_target = sorted(
        (metrics.get("per_target_metrics", {}) or {}).values(),
        key=lambda item: int(item.get("target_index", 10**9)),
    )
    lines.extend(["", "Per-target repeat captures:"])
    for row in per_target:
        lines.append(
            f"- {row.get('label', 'target')} ({row.get('ring', 'unknown')}): "
            f"n={int(row.get('repeat_sample_count', 0) or 0)}, "
            f"RMSE={_fmt(row.get('spread_rms_mm'), suffix=' mm')}, "
            f"max={_fmt(row.get('max_deviation_mm'), suffix=' mm')}"
        )
    comparison = dict(metrics.get("baseline_comparison", {}) or {})
    if comparison.get("available"):
        lines.extend(["", "Baseline comparison:"])
        lines.append(f"Baseline path: {comparison.get('baseline_path', '')}")
        for label, key in [
            ("Overall RMS", "overall_rms"),
            ("Overall max deviation", "overall_max_deviation"),
            ("Path-dependence RMS", "path_dependence_rms"),
        ]:
            row = dict(comparison.get(key, {}) or {})
            lines.append(
                f"- {label}: current={_fmt(row.get('current'), suffix=' mm')}, "
                f"baseline={_fmt(row.get('baseline'), suffix=' mm')}, "
                f"delta={_signed_fmt(row.get('delta'), suffix=' mm')}"
            )
        rejected = dict(comparison.get("rejected_capture_count", {}) or {})
        lines.append(
            f"- Rejected captures: current={rejected.get('current', 'n/a')}, "
            f"baseline={rejected.get('baseline', 'n/a')}, delta={_signed_int(rejected.get('delta'))}"
        )
        ring_groups = ((comparison.get("group_metrics", {}) or {}).get("ring", {}) or {})
        for ring in ["inner", "outer"]:
            row = dict(ring_groups.get(ring, {}) or {})
            if row:
                lines.append(
                    f"- {ring.title()} ring mean RMSE: current={_fmt(row.get('current'), suffix=' mm')}, "
                    f"baseline={_fmt(row.get('baseline'), suffix=' mm')}, "
                    f"delta={_signed_fmt(row.get('delta'), suffix=' mm')}"
                )
    elif comparison:
        lines.extend(["", f"Baseline comparison unavailable: {comparison.get('error', 'unknown error')}"])
    provenance = dict(metrics.get("run_provenance", {}) or {})
    if provenance:
        base_registration = dict(provenance.get("base_registration", {}) or {})
        runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
        pivot_tip = dict(provenance.get("pivot_tip", {}) or {})
        pretension = dict(provenance.get("pretension_artifact", {}) or {})
        transform_chain = dict(provenance.get("transform_chain_summary", {}) or {})
        lines.extend(
            [
                "",
                "Run provenance:",
                f"- Tracker backend: {provenance.get('backend_identity', 'unknown')} ({provenance.get('selected_backend_name', 'unknown')})",
                f"- 0B pivot tip: {pivot_tip.get('path', 'n/a')} @ {pivot_tip.get('modified_at_utc', 'n/a')}",
                (
                    f"- Base registration: {base_registration.get('path', 'n/a')} "
                    f"(state={base_registration.get('state', 'unknown')}, "
                    f"timestamp={base_registration.get('stored_timestamp_utc', 'n/a')}, "
                    f"FRE={_fmt(base_registration.get('stored_fre_mm'), suffix=' mm')}, "
                    f"sha256={_short_hash(base_registration.get('sha256'))})"
                ),
                (
                    f"- Runtime tip calibration: {runtime_tip.get('path', 'n/a')} "
                    f"(mode={runtime_tip.get('mode', 'unknown')}, "
                    f"trust={runtime_tip.get('trust_level', 'unknown')}, "
                    f"state={runtime_tip.get('state', 'unknown')}, "
                    f"artifact_used={runtime_tip.get('artifact_used', 'unknown')}, "
                    f"timestamp={runtime_tip.get('stored_timestamp_utc', 'n/a')}, "
                    f"sha256={_short_hash(runtime_tip.get('sha256'))})"
                ),
                (
                    f"- Pretension artifact: {pretension.get('path', 'n/a')} "
                    f"(status={pretension.get('status', 'unknown')}, "
                    f"source={pretension.get('active_source_type', 'unknown')}, "
                    f"updated={pretension.get('updated_at_utc', 'n/a')}, "
                    f"sha256={_short_hash(pretension.get('sha256'))})"
                ),
                f"- Transform chain: {transform_chain.get('expression', 'n/a')}",
                (
                    "- Thesis trust: "
                    + (
                        f"thesis_trusted runtime tip policy ({runtime_tip.get('mode', 'unknown')})"
                        if provenance.get("thesis_trusted_runtime_tip")
                        else "lower-trust/debug; do not treat as thesis-trusted"
                    )
                ),
            ]
        )
        if runtime_tip.get("identity_marker"):
            lines.append(f"- Runtime tip identity marker: {runtime_tip.get('identity_marker')}")
        if runtime_tip.get("mode_message"):
            lines.append(f"- Runtime tip detail: {runtime_tip.get('mode_message')}")
        if pretension.get("active_source_message"):
            lines.append(f"- Pretension detail: {pretension.get('active_source_message')}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "- This validates repeatability after startup calibration and pretension, not spatial accuracy by itself.",
            "- Use robot-frame metrics only when registration and 0A runtime tip calibration are accepted and active.",
            "- Path-dependence values summarize approach-conditioned deviations from each target centroid.",
        ]
    )
    return lines


def write_single_segment_repeatability_outputs(*, output_dir: Path, metadata, summary, samples) -> dict[str, Path]:
    """Write thesis-oriented figures and summary text for one run."""
    output_dir = Path(output_dir)
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    summary_text_path = output_dir / "repeatability_summary.txt"
    diagnostics_csv_path = output_dir / "repeatability_debug_samples.csv"
    legacy_comparison_csv_path = output_dir / "repeatability_legacy_style_comparison.csv"
    clusters_report_path = output_dir / "repeatability_clusters_report.png"
    error_by_target_report_path = output_dir / "repeatability_error_by_target_report.png"
    group_summary_report_path = output_dir / "repeatability_group_summary_report.png"
    clusters_path = output_dir / "repeatability_clusters.png"
    legacy_clusters_path = output_dir / "tip_pos_clusters.png"
    rmse_path = output_dir / "repeatability_rmse_summary.png"
    path_dependence_path = output_dir / "repeatability_path_dependence.png"
    commanded_vs_measured_path = output_dir / "repeatability_commanded_vs_measured_tip.png"
    acceptance_path = output_dir / "repeatability_acceptance_timeline.png"
    servo_goal_tip_path = output_dir / "repeatability_servo_goal_ticks_vs_tip.png"
    provenance_path = output_dir / "repeatability_provenance_summary.png"
    summary_text_path.write_text(
        "\n".join(
            build_single_segment_repeatability_summary_lines(
                metadata=metadata,
                summary=summary,
                metrics=metrics,
            )
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    _write_debug_samples_csv(diagnostics_csv_path, samples=samples)
    _write_legacy_style_comparison_csv(legacy_comparison_csv_path, metrics=metrics)
    for path, writer in [
        (
            clusters_report_path,
            lambda: _write_cluster_report_figure(clusters_path=clusters_report_path, samples=samples, metrics=metrics),
        ),
        (
            error_by_target_report_path,
            lambda: _write_error_by_target_report_figure(path=error_by_target_report_path, metrics=metrics),
        ),
        (
            group_summary_report_path,
            lambda: _write_group_summary_report_figure(path=group_summary_report_path, metrics=metrics),
        ),
    ]:
        try:
            writer()
        except Exception:
            LOG.exception("Repeatability report figure failed: %s", path.name)
            _write_plot_placeholder(path)
    try:
        if _qt_plotting_is_safe():
            _ensure_plot_qt_app()
            _write_cluster_figure(clusters_path=clusters_path, samples=samples, metrics=metrics)
        else:
            _write_plot_placeholder(clusters_path)
    except Exception:
        LOG.exception("Repeatability compatibility cluster figure was not written.")
        _write_plot_placeholder(clusters_path)
    if _qt_plotting_is_safe():
        _ensure_plot_qt_app()
        _write_legacy_tip_clusters_figure(path=legacy_clusters_path, metrics=metrics)
        _write_rmse_figure(rmse_path=rmse_path, metrics=metrics)
        _write_path_dependence_figure(path_dependence_path=path_dependence_path, metrics=metrics)
        _write_commanded_vs_measured_figure(path=commanded_vs_measured_path, samples=samples, metrics=metrics)
        _write_acceptance_timeline_figure(path=acceptance_path, samples=samples, metrics=metrics)
        _write_servo_goal_tip_figure(path=servo_goal_tip_path, samples=samples, metrics=metrics)
        _write_provenance_figure(path=provenance_path, metrics=metrics)
    else:
        for path in [
            legacy_clusters_path,
            rmse_path,
            path_dependence_path,
            commanded_vs_measured_path,
            acceptance_path,
            servo_goal_tip_path,
            provenance_path,
        ]:
            _write_plot_placeholder(path)
        LOG.warning("Qt plotting backend unavailable; wrote placeholder repeatability plots under %s", output_dir)
    return {
        "summary_text_path": summary_text_path,
        "diagnostics_csv_path": diagnostics_csv_path,
        "legacy_comparison_csv_path": legacy_comparison_csv_path,
        "clusters_report_path": clusters_report_path,
        "error_by_target_report_path": error_by_target_report_path,
        "group_summary_report_path": group_summary_report_path,
        "clusters_path": clusters_path,
        "legacy_clusters_path": legacy_clusters_path,
        "rmse_path": rmse_path,
        "path_dependence_path": path_dependence_path,
        "commanded_vs_measured_path": commanded_vs_measured_path,
        "acceptance_path": acceptance_path,
        "servo_goal_tip_path": servo_goal_tip_path,
        "provenance_path": provenance_path,
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


def _write_debug_samples_csv(path: Path, *, samples) -> None:
    rows = _sample_diagnostic_rows(samples)
    fieldnames = [
        "sample_index",
        "phase",
        "capture_accepted",
        "accept_reject_reason",
        "target_label",
        "command_target_id",
        "command_target_index",
        "commanded_cable_displacement_mm",
        "target_cable_deltas_ticks",
        "resolved_servo_goal_ticks",
        "servo_positions_before_command",
        "servo_positions_after_command",
        "raw_tracker_translation_mm",
        "robot_tip_translation_mm",
        "tracker_stale_age_s",
        "tracker_frame_number",
        "runtime_tip_mode",
        "runtime_tip_trust_level",
        "runtime_telemetry_ok",
        "runtime_telemetry_failure_code",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _sample_diagnostic_rows(samples) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        raw_pose = dict(extra.get("raw_tracker_tool_pose", {}) or {})
        tip_pose = dict(extra.get("robot_frame_tip_pose_used", {}) or {})
        telemetry_status = dict(extra.get("runtime_telemetry_status", {}) or {})
        rows.append(
            {
                "sample_index": int(getattr(sample, "sample_index", -1)),
                "phase": str(getattr(sample, "phase", "") or ""),
                "capture_accepted": bool(extra.get("capture_accepted", False)),
                "accept_reject_reason": str(extra.get("accept_reject_reason") or extra.get("capture_reject_reason") or "ok"),
                "target_label": str(extra.get("target_label", "") or ""),
                "command_target_id": str(extra.get("command_target_id", "") or ""),
                "command_target_index": extra.get("command_target_index"),
                "commanded_cable_displacement_mm": _jsonish(extra.get("commanded_cable_displacement_mm")),
                "target_cable_deltas_ticks": _jsonish(extra.get("target_cable_deltas_ticks")),
                "resolved_servo_goal_ticks": _jsonish(extra.get("resolved_servo_goal_ticks")),
                "servo_positions_before_command": _jsonish(extra.get("current_servo_positions_before_command")),
                "servo_positions_after_command": _jsonish(extra.get("current_servo_positions_after_command")),
                "raw_tracker_translation_mm": _jsonish(raw_pose.get("translation_mm")),
                "robot_tip_translation_mm": _jsonish(tip_pose.get("translation_mm")),
                "tracker_stale_age_s": extra.get("tracker_stale_age_s", getattr(sample, "freshness_s", None)),
                "tracker_frame_number": extra.get("tracker_frame_number", getattr(sample, "tracker_frame_id", None)),
                "runtime_tip_mode": str(extra.get("runtime_tip_mode", "") or ""),
                "runtime_tip_trust_level": str(extra.get("runtime_tip_trust_level", "") or ""),
                "runtime_telemetry_ok": telemetry_status.get("ok"),
                "runtime_telemetry_failure_code": telemetry_status.get("failure_code"),
            }
        )
    return rows


def _write_legacy_style_comparison_csv(path: Path, *, metrics: dict[str, Any]) -> None:
    fieldnames = [
        "target_id",
        "n_samples",
        "XY_RMSE_mm",
        "XYZ_RMSE_mm",
        "max_deviation_mm",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
        "analysis_mode",
    ]
    rows: list[dict[str, Any]] = []
    legacy = dict(metrics.get("legacy_style_comparison", {}) or {})
    legacy_per_target = dict(legacy.get("per_target", {}) or {})
    for key in sorted(legacy_per_target, key=lambda value: int(value) if str(value).isdigit() else 10**9):
        row = dict(legacy_per_target.get(key, {}) or {})
        rows.append(
            {
                "target_id": row.get("target_id", row.get("target_index", key)),
                "n_samples": row.get("n_samples"),
                "XY_RMSE_mm": row.get("XY_RMSE_mm"),
                "XYZ_RMSE_mm": row.get("XYZ_RMSE_mm"),
                "max_deviation_mm": row.get("max_deviation_mm"),
                "centroid_x_mm": row.get("centroid_x_mm"),
                "centroid_y_mm": row.get("centroid_y_mm"),
                "centroid_z_mm": row.get("centroid_z_mm"),
                "analysis_mode": "legacy_style",
            }
        )
    strict_per_target = dict(metrics.get("per_target_metrics", {}) or {})
    for key in sorted(strict_per_target, key=lambda value: int(value) if str(value).isdigit() else 10**9):
        row = dict(strict_per_target.get(key, {}) or {})
        centroid = row.get("centroid_mm") if isinstance(row.get("centroid_mm"), list) else []
        rows.append(
            {
                "target_id": row.get("target_index", key),
                "n_samples": row.get("repeat_sample_count"),
                "XY_RMSE_mm": row.get("XY_RMSE_mm"),
                "XYZ_RMSE_mm": row.get("XYZ_RMSE_mm", row.get("spread_rms_mm")),
                "max_deviation_mm": row.get("max_deviation_mm"),
                "centroid_x_mm": centroid[0] if len(centroid) >= 1 else None,
                "centroid_y_mm": centroid[1] if len(centroid) >= 2 else None,
                "centroid_z_mm": centroid[2] if len(centroid) >= 3 else None,
                "analysis_mode": "thesis_strict",
            }
        )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cluster_report_figure(*, clusters_path: Path, samples, metrics: dict[str, Any]) -> None:
    points = _repeat_points_by_target(samples=samples)
    centroids = {
        int(row.get("target_index", -1)): row.get("centroid_mm")
        for row in _ordered_per_target(metrics)
        if isinstance(row.get("centroid_mm"), list) and len(row.get("centroid_mm")) >= 2
    }
    all_xy: list[tuple[float, float]] = []
    for target_points in points.values():
        all_xy.extend((float(point[0]), float(point[1])) for point in target_points)
    all_xy.extend((float(point[0]), float(point[1])) for point in centroids.values())
    fig, ax = create_figure(size="square")
    if not all_xy:
        ax.text(0.5, 0.5, "No accepted robot-frame repeat captures", transform=ax.transAxes, ha="center", va="center")
    else:
        target_ids = sorted(set(points) | set(centroids))
        palette = [
            "#2563eb",
            "#0f766e",
            "#7c3aed",
            "#d97706",
            "#0891b2",
            "#be123c",
            "#4f46e5",
            "#15803d",
        ]
        for index, target_index in enumerate(target_ids):
            target_points = points.get(target_index, [])
            target_color = palette[index % len(palette)]
            if target_points:
                ax.scatter(
                    [float(point[0]) for point in target_points],
                    [float(point[1]) for point in target_points],
                    s=14,
                    alpha=0.48,
                    color=target_color,
                    linewidths=0,
                    label="Repeat captures" if index == 0 else None,
                )
            centroid = centroids.get(target_index)
            if centroid:
                ax.scatter(
                    [float(centroid[0])],
                    [float(centroid[1])],
                    s=38,
                    marker="s",
                    facecolors="white",
                    edgecolors=color("reference"),
                    linewidths=1.1,
                    zorder=4,
                    label="Target centroid" if index == 0 else None,
                )
        set_equal_xy(
            ax,
            x_values=[point[0] for point in all_xy],
            y_values=[point[1] for point in all_xy],
            minimum_span=5.0,
        )
    style_axes(
        ax,
        title="Single-Segment Repeatability",
        xlabel="Robot-frame X position (mm)",
        ylabel="Robot-frame Y position (mm)",
    )
    legend(ax, loc="lower right")
    max_deviation = metrics.get("overall_max_deviation_mm")
    thesis_goal = metrics.get("thesis_goal_rms_mm")
    add_metric_box(
        ax,
        _compact_lines(
            [
                f"Overall RMS: {_fmt(metrics.get('overall_repeatability_rms_mm'), suffix=' mm')}",
                f"Max deviation: {_fmt(max_deviation, suffix=' mm')}" if max_deviation is not None else None,
                f"Repeat samples: {int(metrics.get('valid_repeat_sample_count', sum(len(v) for v in points.values())) or 0)}",
                f"Goal: <= {_fmt(thesis_goal, suffix=' mm')}" if thesis_goal is not None else None,
            ]
        ),
        loc="upper right",
    )
    save_figure(fig, clusters_path)


def _write_error_by_target_report_figure(*, path: Path, metrics: dict[str, Any]) -> None:
    rows = _ordered_per_target(metrics)
    labels = [str(row.get("label", f"T{index:02d}")) for index, row in enumerate(rows)]
    values = [_repeatability_error_value(row) for row in rows]
    fig, ax = create_figure(size="wide")
    numeric_values = [float(value) for value in values if value is not None]
    if not rows or not numeric_values:
        ax.text(0.5, 0.5, "No per-target repeatability errors available", transform=ax.transAxes, ha="center", va="center")
    else:
        x_positions = list(range(len(rows)))
        bars = ax.bar(
            x_positions,
            [float(value) if value is not None else 0.0 for value in values],
            color=color("measured"),
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        thesis_goal = _optional_float(metrics.get("thesis_goal_rms_mm"))
        if thesis_goal is not None:
            ax.axhline(
                thesis_goal,
                color=color("threshold"),
                linestyle="--",
                linewidth=1.2,
                label=f"Thesis goal ({thesis_goal:.2f} mm)",
            )
        overall = _optional_float(metrics.get("overall_repeatability_rms_mm"))
        if overall is not None:
            ax.axhline(
                overall,
                color=color("reference"),
                linestyle="-",
                linewidth=1.0,
                alpha=0.75,
                label=f"Overall RMS ({overall:.2f} mm)",
            )
        if len(bars) <= 17:
            labels_above = [f"{float(value):.2f}" if value is not None else "" for value in values]
            ax.bar_label(bars, labels=labels_above, padding=2, fontsize=8)
        legend(ax, loc="upper right")
        ax.set_ylim(0.0, max(numeric_values + [thesis_goal or 0.0, overall or 0.0]) * 1.18 + 0.02)
    style_axes(
        ax,
        title="Repeatability Error by Target",
        xlabel="Target",
        ylabel="Repeatability error (mm)",
    )
    save_figure(fig, path)


def _write_group_summary_report_figure(*, path: Path, metrics: dict[str, Any]) -> None:
    rows = _ring_group_rows(metrics)
    fig, ax = create_figure(size="wide")
    if not rows:
        ax.text(0.5, 0.5, "No target-group repeatability data available", transform=ax.transAxes, ha="center", va="center")
    else:
        labels = [label.title() for label, _value in rows]
        values = [float(value) for _label, value in rows]
        bar_colors = [
            color("reference") if label == "center" else color("measured") if label == "inner" else color("fit")
            for label, _value in rows
        ]
        bars = ax.bar(labels, values, color=bar_colors, edgecolor="white", linewidth=0.7)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9)
        thesis_goal = _optional_float(metrics.get("thesis_goal_rms_mm"))
        if thesis_goal is not None:
            ax.axhline(
                thesis_goal,
                color=color("threshold"),
                linestyle="--",
                linewidth=1.2,
                label=f"Thesis goal ({thesis_goal:.2f} mm)",
            )
            legend(ax, loc="upper right")
        ax.set_ylim(0.0, max(values + [thesis_goal or 0.0]) * 1.2 + 0.02)
    style_axes(
        ax,
        title="Repeatability by Target Group",
        xlabel="Target group",
        ylabel="Mean repeatability error (mm)",
    )
    save_figure(fig, path)


def _repeatability_error_value(row: dict[str, Any]) -> float | None:
    for key in ("spread_rms_mm", "rmse_mm", "XYZ_RMSE_mm", "XY_RMSE_mm"):
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _ring_group_rows(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    ring_groups = dict((metrics.get("group_metrics", {}) or {}).get("ring", {}) or {})
    rows: list[tuple[str, float]] = []
    for label in ["center", "inner", "outer"]:
        group = dict(ring_groups.get(label, {}) or {})
        value = _optional_float(group.get("mean_target_rms_mm"))
        if value is not None:
            rows.append((label, value))
    if rows:
        return rows
    values_by_ring: dict[str, list[float]] = {}
    for row in _ordered_per_target(metrics):
        value = _repeatability_error_value(row)
        if value is None:
            continue
        ring = str(row.get("ring", "") or "")
        if not ring:
            target_index = int(row.get("target_index", -1) or -1)
            ring = "center" if target_index == 0 else "inner" if target_index <= 8 else "outer"
        values_by_ring.setdefault(ring, []).append(float(value))
    for label in ["center", "inner", "outer"]:
        values = values_by_ring.get(label, [])
        if values:
            rows.append((label, float(np.mean(values))))
    return rows


def _compact_lines(lines: list[str | None]) -> list[str]:
    return [str(line) for line in lines if line]


def _write_cluster_figure(*, clusters_path: Path, samples, metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 920)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "SINGLE-SEGMENT REPEATABILITY CLUSTERS",
        "Accepted repeat captures by target. Metrics use repeat captures after approach from every other target.",
    )
    chart_rect = QRectF(36.0, 108.0, 820.0, 760.0)
    summary_rect = QRectF(888.0, 108.0, 356.0, 760.0)
    _draw_panel(painter, chart_rect, "Robot-Frame Tip Clusters (XY)")
    _draw_panel(painter, summary_rect, "Run Summary")
    _draw_cluster_scatter(painter, chart_rect.adjusted(22.0, 42.0, -22.0, -28.0), samples=samples, metrics=metrics)
    _draw_summary_pairs(
        painter,
        summary_rect.adjusted(16.0, 38.0, -16.0, -16.0),
        build_single_segment_repeatability_summary_pairs(metrics=metrics),
    )
    painter.end()
    image.save(str(clusters_path))


def _write_legacy_tip_clusters_figure(*, path: Path, metrics: dict[str, Any]) -> None:
    legacy = dict(metrics.get("legacy_style_comparison", {}) or {})
    per_target = [
        dict(row)
        for _key, row in sorted(
            dict(legacy.get("per_target", {}) or {}).items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else 10**9,
        )
    ]
    image = _new_image(1360, 920)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1292.0, 58.0),
        "TIP POSITION CLUSTERS AND RMSE",
        "Legacy-style comparison: XY scatter shown, labels/reporting use 3D RMSE to each target centroid.",
    )
    chart_rect = QRectF(48.0, 108.0, 920.0, 760.0)
    legend_rect = QRectF(996.0, 108.0, 328.0, 760.0)
    _draw_panel(painter, chart_rect, "Tip Position Clusters (XY, mm)")
    _draw_panel(painter, legend_rect, "Per-Target XYZ RMSE")
    plot_rect = chart_rect.adjusted(48.0, 48.0, -30.0, -58.0)
    points_by_target: dict[int, list[list[float]]] = {}
    centroids: dict[int, list[float]] = {}
    rmse_by_target: dict[int, float | None] = {}
    combined_xy: list[tuple[float, float]] = []
    for row in per_target:
        target_id = int(row.get("target_id", row.get("target_index", -1)) or -1)
        samples = [
            [float(value) for value in sample]
            for sample in list(row.get("sample_positions_mm", []) or [])
            if isinstance(sample, list) and len(sample) >= 2
        ]
        points_by_target[target_id] = samples
        centroid = row.get("centroid_mm")
        if isinstance(centroid, list) and len(centroid) >= 2:
            centroids[target_id] = [float(value) for value in centroid]
            combined_xy.append((float(centroid[0]), float(centroid[1])))
        rmse_by_target[target_id] = _optional_float(row.get("XYZ_RMSE_mm"))
        combined_xy.extend((float(sample[0]), float(sample[1])) for sample in samples)
    if not combined_xy:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(plot_rect, Qt.AlignCenter, "No legacy-style repeat captures were available.")
        painter.end()
        image.save(str(path))
        return
    xs = [point[0] for point in combined_xy]
    ys = [point[1] for point in combined_xy]
    min_x, max_x = _expand_range(min(xs), max(xs), minimum_span=5.0, pad_fraction=0.12)
    min_y, max_y = _expand_range(min(ys), max(ys), minimum_span=5.0, pad_fraction=0.12)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(plot_rect.left(), plot_rect.bottom()), QPointF(plot_rect.right(), plot_rect.bottom()))
    painter.drawLine(QPointF(plot_rect.left(), plot_rect.top()), QPointF(plot_rect.left(), plot_rect.bottom()))
    painter.setFont(_body_font())
    painter.setPen(QColor("#475569"))
    painter.drawText(QRectF(plot_rect.left(), plot_rect.bottom() + 12.0, plot_rect.width(), 18.0), Qt.AlignCenter, "x (mm)")
    painter.drawText(QRectF(plot_rect.left(), plot_rect.top() - 26.0, plot_rect.width(), 18.0), Qt.AlignRight, "y (mm)")
    legend_pairs: list[tuple[str, str]] = []
    for index, target_id in enumerate(sorted(points_by_target)):
        color = QColor.fromHsv(int((index * 37) % 360), 185, 205)
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        for point in points_by_target[target_id]:
            painter.drawEllipse(
                QPointF(
                    _scale_x(plot_rect, float(point[0]), min_x, max_x),
                    _scale_y(plot_rect, float(point[1]), min_y, max_y),
                ),
                4.0,
                4.0,
            )
        centroid = centroids.get(target_id)
        if centroid:
            center = QPointF(
                _scale_x(plot_rect, float(centroid[0]), min_x, max_x),
                _scale_y(plot_rect, float(centroid[1]), min_y, max_y),
            )
            painter.setPen(QPen(QColor("#0f172a"), 1.3))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(QRectF(center.x() - 5.0, center.y() - 5.0, 10.0, 10.0))
            painter.setPen(QColor("#334155"))
            painter.drawText(QPointF(center.x() + 7.0, center.y() - 4.0), str(target_id + 1))
        rmse = rmse_by_target.get(target_id)
        legend_pairs.append((f"{target_id + 1}", _fmt(rmse, suffix=" mm")))
    _draw_summary_pairs(
        painter,
        legend_rect.adjusted(16.0, 38.0, -16.0, -16.0),
        legend_pairs,
    )
    painter.end()
    image.save(str(path))


def _write_commanded_vs_measured_figure(*, path: Path, samples, metrics: dict[str, Any]) -> None:
    _ = metrics
    points: list[tuple[float, float]] = []
    for sample in samples:
        extra = dict(getattr(sample, "extra", {}) or {})
        command_mm = extra.get("commanded_cable_displacement_mm") or extra.get("target_cable_deltas_mm")
        position, frame = extract_tip_or_tool_position_mm(sample, tool_id="0A", prefer_robot_frame=True)
        if isinstance(command_mm, list) and position is not None and frame == "robot":
            points.append((_norm(command_mm), _norm(position[:2])))
    _write_xy_diagnostic_figure(
        path=path,
        title="COMMANDED TARGET VS MEASURED TIP",
        subtitle="Cable displacement command norm vs measured robot-frame tip XY radius.",
        points=points,
        x_label="Commanded cable displacement norm (mm)",
        y_label="Measured tip XY radius (mm)",
    )


def _write_acceptance_timeline_figure(*, path: Path, samples, metrics: dict[str, Any]) -> None:
    _ = metrics
    points = [
        (
            float(getattr(sample, "sample_index", index)),
            1.0 if bool((getattr(sample, "extra", {}) or {}).get("capture_accepted", False)) else 0.0,
        )
        for index, sample in enumerate(samples)
    ]
    _write_xy_diagnostic_figure(
        path=path,
        title="CAPTURE ACCEPTANCE TIMELINE",
        subtitle="Accepted samples plot at 1; rejected/stale samples plot at 0.",
        points=points,
        x_label="Sample index",
        y_label="Accepted (1) / rejected (0)",
    )


def _write_servo_goal_tip_figure(*, path: Path, samples, metrics: dict[str, Any]) -> None:
    _ = metrics
    points: list[tuple[float, float]] = []
    for sample in samples:
        goals = [
            float(value)
            for value in dict(getattr(sample, "commanded_motor_values", {}) or {}).values()
            if value is not None
        ]
        position, frame = extract_tip_or_tool_position_mm(sample, tool_id="0A", prefer_robot_frame=True)
        if goals and position is not None and frame == "robot":
            points.append((max(goals) - min(goals), _norm(position[:2])))
    _write_xy_diagnostic_figure(
        path=path,
        title="SERVO GOAL TICKS VS TIP POSITION",
        subtitle="Servo goal spread vs measured robot-frame tip XY radius.",
        points=points,
        x_label="Servo goal spread (ticks)",
        y_label="Measured tip XY radius (mm)",
    )


def _write_provenance_figure(*, path: Path, metrics: dict[str, Any]) -> None:
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "REPEATABILITY RUNTIME PROVENANCE",
        "Transform/runtime-tip/registration/pretension state captured at run start.",
    )
    rect = QRectF(36.0, 112.0, 1208.0, 540.0)
    _draw_panel(painter, rect, "Run Trust And Transform Chain")
    provenance = dict(metrics.get("run_provenance", {}) or {})
    runtime_tip = dict(provenance.get("runtime_tip_calibration", {}) or {})
    registration = dict(provenance.get("base_registration", {}) or {})
    pretension = dict(provenance.get("pretension_artifact", {}) or {})
    transform_chain = dict(provenance.get("transform_chain_summary", {}) or {})
    _draw_summary_pairs(
        painter,
        rect.adjusted(18.0, 42.0, -18.0, -18.0),
        [
            ("Runtime tip mode", str(runtime_tip.get("mode", metrics.get("runtime_tip_mode_used", "unknown")))),
            ("Runtime tip trust", str(runtime_tip.get("trust_level", metrics.get("runtime_tip_trust_level", "unknown")))),
            ("Thesis trusted", "yes" if provenance.get("thesis_trusted_runtime_tip") else "no"),
            ("Runtime tip artifact", str(runtime_tip.get("path", "n/a"))),
            ("Registration artifact", str(registration.get("path", "n/a"))),
            ("Registration FRE", _fmt(registration.get("stored_fre_mm"), suffix=" mm")),
            ("Pretension artifact", str(pretension.get("path", "n/a"))),
            ("Pretension source", str(pretension.get("active_source_type", "unknown"))),
            ("Transform chain", str(transform_chain.get("expression", "n/a"))),
        ],
    )
    painter.end()
    image.save(str(path))


def _write_xy_diagnostic_figure(
    *,
    path: Path,
    title: str,
    subtitle: str,
    points: list[tuple[float, float]],
    x_label: str,
    y_label: str,
) -> None:
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(painter, QRectF(34.0, 24.0, 1212.0, 58.0), title, subtitle)
    chart_rect = QRectF(70.0, 120.0, 1110.0, 500.0)
    _draw_panel(painter, chart_rect, "")
    plot_rect = chart_rect.adjusted(44.0, 34.0, -28.0, -54.0)
    if not points:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(plot_rect, Qt.AlignCenter, "No diagnostic points were available.")
        painter.end()
        image.save(str(path))
        return
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = _expand_range(min(xs), max(xs), minimum_span=1.0, pad_fraction=0.08)
    min_y, max_y = _expand_range(min(ys), max(ys), minimum_span=1.0, pad_fraction=0.12)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(plot_rect.left(), plot_rect.bottom()), QPointF(plot_rect.right(), plot_rect.bottom()))
    painter.drawLine(QPointF(plot_rect.left(), plot_rect.top()), QPointF(plot_rect.left(), plot_rect.bottom()))
    painter.setBrush(QColor(COLORS.scene_measurement))
    painter.setPen(Qt.NoPen)
    for x_value, y_value in points:
        painter.drawEllipse(
            QPointF(
                _scale_x(plot_rect, float(x_value), min_x, max_x),
                _scale_y(plot_rect, float(y_value), min_y, max_y),
            ),
            3.0,
            3.0,
        )
    painter.setFont(_body_font())
    painter.setPen(QColor("#475569"))
    painter.drawText(QRectF(plot_rect.left(), plot_rect.bottom() + 12.0, plot_rect.width(), 18.0), Qt.AlignCenter, x_label)
    painter.drawText(QRectF(plot_rect.left(), plot_rect.top() - 26.0, plot_rect.width(), 18.0), Qt.AlignRight, y_label)
    painter.end()
    image.save(str(path))


def _write_rmse_figure(*, rmse_path: Path, metrics: dict[str, Any]) -> None:
    per_target = _ordered_per_target(metrics)
    labels = [str(row.get("label", f"T{index:02d}")) for index, row in enumerate(per_target)]
    rmse_values = [row.get("spread_rms_mm") for row in per_target]
    max_values = [row.get("max_deviation_mm") for row in per_target]
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "PER-TARGET REPEATABILITY SUMMARY",
        "RMSE and maximum deviation from each target centroid for accepted repeat captures.",
    )
    rmse_rect = QRectF(36.0, 112.0, 586.0, 540.0)
    max_rect = QRectF(658.0, 112.0, 586.0, 540.0)
    _draw_panel(painter, rmse_rect, "Repeatability RMSE")
    _draw_panel(painter, max_rect, "Maximum Deviation")
    _draw_bar_chart(
        painter,
        rmse_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=rmse_values,
        color=QColor(COLORS.scene_truth),
        y_label="RMSE (mm)",
        empty_text="No accepted repeat captures were available.",
    )
    _draw_bar_chart(
        painter,
        max_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=max_values,
        color=QColor(COLORS.scene_residual),
        y_label="Max deviation (mm)",
        empty_text="No accepted repeat captures were available.",
    )
    painter.end()
    image.save(str(rmse_path))


def _write_path_dependence_figure(*, path_dependence_path: Path, metrics: dict[str, Any]) -> None:
    per_target = _ordered_per_target(metrics)
    labels = [str(row.get("label", f"T{index:02d}")) for index, row in enumerate(per_target)]
    path_entries = list(metrics.get("path_dependence_by_approach", []) or [])
    max_by_target: dict[int, float] = {}
    mean_by_target: dict[int, float] = {}
    for row in per_target:
        target_index = int(row.get("target_index", -1))
        values = [
            float(entry.get("deviation_mm", 0.0) or 0.0)
            for entry in path_entries
            if int(entry.get("target_index", -2)) == target_index
        ]
        if values:
            max_by_target[target_index] = float(np.max(values))
            mean_by_target[target_index] = float(np.mean(values))
    max_values = [max_by_target.get(int(row.get("target_index", -1))) for row in per_target]
    mean_values = [mean_by_target.get(int(row.get("target_index", -1))) for row in per_target]
    image = _new_image(1280, 720)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)
    _draw_title(
        painter,
        QRectF(34.0, 24.0, 1212.0, 58.0),
        "PATH-DEPENDENCE SUMMARY",
        "Approach-conditioned deviation from each target centroid. One repeat capture is recorded per approach target.",
    )
    mean_rect = QRectF(36.0, 112.0, 586.0, 540.0)
    max_rect = QRectF(658.0, 112.0, 586.0, 540.0)
    _draw_panel(painter, mean_rect, "Mean Approach-Conditioned Deviation")
    _draw_panel(painter, max_rect, "Max Approach-Conditioned Deviation")
    _draw_bar_chart(
        painter,
        mean_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=mean_values,
        color=QColor(COLORS.scene_measurement),
        y_label="Mean deviation (mm)",
        empty_text="No path-dependence deviations were available.",
    )
    _draw_bar_chart(
        painter,
        max_rect.adjusted(14.0, 38.0, -14.0, -20.0),
        labels=labels,
        values=max_values,
        color=QColor(COLORS.scene_residual),
        y_label="Max deviation (mm)",
        empty_text="No path-dependence deviations were available.",
    )
    painter.end()
    image.save(str(path_dependence_path))


def _draw_cluster_scatter(painter: QPainter, rect: QRectF, *, samples, metrics: dict[str, Any]) -> None:
    points = _repeat_points_by_target(samples=samples)
    centroids = {
        int(row.get("target_index", -1)): row.get("centroid_mm")
        for row in _ordered_per_target(metrics)
        if isinstance(row.get("centroid_mm"), list) and len(row.get("centroid_mm")) >= 2
    }
    combined_xy: list[tuple[float, float]] = []
    for target_points in points.values():
        combined_xy.extend((float(point[0]), float(point[1])) for point in target_points)
    combined_xy.extend((float(point[0]), float(point[1])) for point in centroids.values())
    if not combined_xy:
        painter.setFont(_body_font())
        painter.setPen(QColor("#64748b"))
        painter.drawText(rect, Qt.AlignCenter, "No accepted robot-frame repeat captures were available.")
        return
    xs = [point[0] for point in combined_xy]
    ys = [point[1] for point in combined_xy]
    min_x, max_x = _expand_range(min(xs), max(xs), minimum_span=5.0, pad_fraction=0.12)
    min_y, max_y = _expand_range(min(ys), max(ys), minimum_span=5.0, pad_fraction=0.12)
    painter.setPen(QPen(QColor("#e2e8f0"), 1.0))
    painter.drawLine(QPointF(rect.left(), rect.bottom()), QPointF(rect.right(), rect.bottom()))
    painter.drawLine(QPointF(rect.left(), rect.top()), QPointF(rect.left(), rect.bottom()))
    palette = [
        QColor(COLORS.scene_truth),
        QColor(COLORS.scene_measurement),
        QColor(COLORS.scene_tip),
        QColor(COLORS.scene_residual),
        QColor(COLORS.selection_bg),
    ]
    per_target = {int(row.get("target_index", -1)): row for row in _ordered_per_target(metrics)}
    for index, target_index in enumerate(sorted(points)):
        color = palette[index % len(palette)]
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        for point in points[target_index]:
            painter.drawEllipse(
                QPointF(
                    _scale_x(rect, float(point[0]), min_x, max_x),
                    _scale_y(rect, float(point[1]), min_y, max_y),
                ),
                3.0,
                3.0,
            )
        centroid = centroids.get(target_index)
        if centroid:
            center = QPointF(
                _scale_x(rect, float(centroid[0]), min_x, max_x),
                _scale_y(rect, float(centroid[1]), min_y, max_y),
            )
            painter.setPen(QPen(QColor("#0f172a"), 1.4))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(QRectF(center.x() - 4.5, center.y() - 4.5, 9.0, 9.0))
            painter.setPen(QColor("#334155"))
            painter.setFont(_body_font())
            label = str(per_target.get(target_index, {}).get("label", f"T{target_index:02d}"))
            rmse = per_target.get(target_index, {}).get("spread_rms_mm")
            painter.drawText(QPointF(center.x() + 6.0, center.y() - 4.0), f"{label} {_fmt(rmse, suffix=' mm')}")
    painter.setPen(QColor("#475569"))
    painter.setFont(_body_font())
    painter.drawText(QRectF(rect.left(), rect.bottom() + 4.0, rect.width(), 18.0), Qt.AlignCenter, "X (mm)")
    painter.drawText(QRectF(rect.left(), rect.top() - 2.0, rect.width(), 18.0), Qt.AlignRight, "Y (mm)")


def _repeat_points_by_target(*, samples) -> dict[int, list[list[float]]]:
    points: dict[int, list[list[float]]] = {}
    for sample in samples:
        if sample.phase != "repeat":
            continue
        if not bool(sample.extra.get("capture_accepted", True)):
            continue
        position, frame = extract_tip_or_tool_position_mm(sample, tool_id="0A", prefer_robot_frame=True)
        if position is None or frame != "robot":
            continue
        target_index = int(sample.target_index if sample.target_index is not None else -1)
        points.setdefault(target_index, []).append([float(value) for value in position])
    return points


def _ordered_per_target(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [dict(value) for value in (metrics.get("per_target_metrics", {}) or {}).values()],
        key=lambda item: int(item.get("target_index", 10**9)),
    )


def _signed_fmt(value: Any, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    return f"{numeric:+.3f}{suffix}"


def _signed_int(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):+d}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return text[:12] if text else "n/a"


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        import json

        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _norm(values: Any) -> float:
    if not isinstance(values, (list, tuple)):
        return 0.0
    numeric = [float(value) for value in values if value is not None]
    return float(np.linalg.norm(np.asarray(numeric, dtype=float))) if numeric else 0.0


def _expand_range(low: float, high: float, *, minimum_span: float, pad_fraction: float) -> tuple[float, float]:
    span = max(float(high - low), float(minimum_span))
    pad = span * float(pad_fraction)
    center = (float(low) + float(high)) / 2.0
    half = (span / 2.0) + pad
    return center - half, center + half


def _scale_x(rect: QRectF, value: float, low: float, high: float) -> float:
    if high <= low:
        return rect.center().x()
    return rect.left() + ((float(value) - low) / (high - low)) * rect.width()


def _scale_y(rect: QRectF, value: float, low: float, high: float) -> float:
    if high <= low:
        return rect.center().y()
    return rect.bottom() - ((float(value) - low) / (high - low)) * rect.height()
