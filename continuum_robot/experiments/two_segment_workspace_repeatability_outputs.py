"""Output writers + repeatability metrics for two-segment workspace repeatability.

Mirrors the single-segment ``workspace_repeatability_map_outputs`` structure
in shape and figure set, adapted for the 4D two-segment command space.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# Canonical filenames (the validator + export both look up these literal names).
TARGETS_JSON = "repeatability_targets.json"
VISIT_PLAN_CSV = "repeatability_visit_plan.csv"
TARGET_CAPTURES_CSV = "target_captures.csv"
REPEATABILITY_METRICS_JSON = "repeatability_metrics.json"
REPEATABILITY_METRICS_CSV = "repeatability_metrics.csv"
PER_TARGET_REPEATABILITY_CSV = "per_target_repeatability.csv"
FAILURE_EVENTS_JSONL = "failure_events.jsonl"
SUMMARY_TXT = "two_segment_workspace_repeatability_summary.txt"

# Thesis figures (same naming as single-segment workspace_repeatability_map).
TS_THESIS_01_PNG = "two_segment_thesis_01_workspace_rms_3d.png"
TS_THESIS_02_PNG = "two_segment_thesis_02_workspace_rms_map.png"
TS_THESIS_03_PNG = "two_segment_thesis_03_rms_vs_amplitude.png"
TS_THESIS_04_PNG = "two_segment_thesis_04_2d_repeatability_map.png"

THESIS_FIGURE_FILENAMES = (
    TS_THESIS_01_PNG,
    TS_THESIS_02_PNG,
    TS_THESIS_03_PNG,
    TS_THESIS_04_PNG,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_workspace_repeatability_metrics(
    *,
    visit_results: Sequence[Any],
    targets: Sequence[Any],
) -> list[dict[str, Any]]:
    """Per-target distal-XYZ scatter stats for the workspace repeatability run.

    Accepts both the experiment's ``_VisitResult`` dataclass instances and
    plain dicts (so tests can feed synthetic rows). Drops rejected captures
    before computing per-target metrics.
    """
    visit_dicts = [_as_dict(visit) for visit in visit_results]
    accepted_by_target: dict[int, list[dict[str, Any]]] = {}
    for visit in visit_dicts:
        if not bool(visit.get("accepted") or visit.get("capture_accepted")):
            continue
        xyz = visit.get("distal_xyz_robot_mm")
        if not isinstance(xyz, list) or len(xyz) < 3:
            continue
        target_idx = int(visit.get("target_index"))
        accepted_by_target.setdefault(target_idx, []).append(visit)

    targets_by_index = {int(_as_dict(t).get("target_index", index)): _as_dict(t) for index, t in enumerate(targets)}

    rows: list[dict[str, Any]] = []
    for target_index, target_dict in sorted(targets_by_index.items()):
        visits = accepted_by_target.get(target_index, [])
        n = len(visits)
        if n == 0:
            rows.append(
                {
                    "target_index": int(target_index),
                    "target_id": str(target_dict.get("target_id", f"WS_{target_index:04d}")),
                    "group_tag": str(target_dict.get("group_tag", "")),
                    "amplitude_cm": float(target_dict.get("amplitude_cm") or 0.0),
                    "bottom_x_cm": float(target_dict.get("bottom_x_cm") or 0.0),
                    "bottom_y_cm": float(target_dict.get("bottom_y_cm") or 0.0),
                    "top_x_cm": float(target_dict.get("top_x_cm") or 0.0),
                    "top_y_cm": float(target_dict.get("top_y_cm") or 0.0),
                    "accepted_repeats": 0,
                    "centroid_xyz_mm": None,
                    "rms_spread_mm": None,
                    "mean_radial_mm": None,
                    "median_radial_mm": None,
                    "max_radial_mm": None,
                    "std_x_mm": None,
                    "std_y_mm": None,
                    "std_z_mm": None,
                    "target_amplitude_mm": float(target_dict.get("amplitude_cm") or 0.0) * 10.0,
                    "x_mm": float(target_dict.get("bottom_x_cm") or 0.0) * 10.0,
                    "y_mm": float(target_dict.get("bottom_y_cm") or 0.0) * 10.0,
                }
            )
            continue
        positions = np.asarray(
            [[float(v) for v in visit["distal_xyz_robot_mm"][:3]] for visit in visits], dtype=float
        )
        centroid = positions.mean(axis=0)
        deltas = positions - centroid
        radial = np.linalg.norm(deltas, axis=1)
        rms_spread = float(np.sqrt(np.mean(radial ** 2)))
        std = positions.std(axis=0)
        rows.append(
            {
                "target_index": int(target_index),
                "target_id": str(target_dict.get("target_id", f"WS_{target_index:04d}")),
                "group_tag": str(target_dict.get("group_tag", "")),
                "amplitude_cm": float(target_dict.get("amplitude_cm") or 0.0),
                "bottom_x_cm": float(target_dict.get("bottom_x_cm") or 0.0),
                "bottom_y_cm": float(target_dict.get("bottom_y_cm") or 0.0),
                "top_x_cm": float(target_dict.get("top_x_cm") or 0.0),
                "top_y_cm": float(target_dict.get("top_y_cm") or 0.0),
                "accepted_repeats": int(n),
                "centroid_xyz_mm": [float(c) for c in centroid.tolist()],
                "rms_spread_mm": rms_spread,
                "mean_radial_mm": float(np.mean(radial)),
                "median_radial_mm": float(np.median(radial)),
                "max_radial_mm": float(np.max(radial)),
                "std_x_mm": float(std[0]),
                "std_y_mm": float(std[1]),
                "std_z_mm": float(std[2]),
                "target_amplitude_mm": float(target_dict.get("amplitude_cm") or 0.0) * 10.0,
                # 2D map projection (commanded bottom XY in mm). For the
                # workspace-RMS-map figure we plot the per-target points in
                # commanded bottom XY since that's the most operator-readable
                # workspace projection; the top dimensions are encoded via
                # the colour scale through `target_amplitude_mm`.
                "x_mm": float(target_dict.get("bottom_x_cm") or 0.0) * 10.0,
                "y_mm": float(target_dict.get("bottom_y_cm") or 0.0) * 10.0,
            }
        )
    return rows


def summarize_workspace_repeatability(
    per_target_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_repeats_per_target: int,
    target_distal_rms_threshold_mm: float | None = None,
) -> dict[str, Any]:
    """Roll up per-target rows into the overall repeatability summary."""
    rows_with_data = [row for row in per_target_rows if row.get("rms_spread_mm") is not None]
    if not rows_with_data:
        return {
            "planned_target_count": int(len(per_target_rows)),
            "targets_with_repeats": 0,
            "targets_below_minimum_repeats": int(len(per_target_rows)),
            "minimum_repeats_per_target": int(minimum_repeats_per_target),
            "overall_distal_rms_mm": None,
            "median_per_target_rms_mm": None,
            "p95_per_target_rms_mm": None,
            "worst_target_rms_mm": None,
            "worst_target_id": None,
            "best_target_rms_mm": None,
            "best_target_id": None,
            "mean_repeats_per_target": 0.0,
            "target_distal_rms_threshold_mm": target_distal_rms_threshold_mm,
            "targets_above_threshold": None,
        }
    rms_values = np.asarray([float(row["rms_spread_mm"]) for row in rows_with_data])
    repeats = np.asarray([int(row.get("accepted_repeats", 0)) for row in rows_with_data])
    weighted_rms = float(np.sqrt(np.sum((rms_values ** 2) * repeats) / max(1, np.sum(repeats))))
    median_rms = float(np.median(rms_values))
    p95_rms = float(np.percentile(rms_values, 95.0))
    worst_idx = int(np.argmax(rms_values))
    best_idx = int(np.argmin(rms_values))
    below_min = sum(
        1 for row in per_target_rows if int(row.get("accepted_repeats", 0)) < int(minimum_repeats_per_target)
    )
    targets_above_threshold: int | None = None
    if target_distal_rms_threshold_mm is not None:
        targets_above_threshold = int(np.sum(rms_values > float(target_distal_rms_threshold_mm)))
    return {
        "planned_target_count": int(len(per_target_rows)),
        "targets_with_repeats": int(len(rows_with_data)),
        "targets_below_minimum_repeats": int(below_min),
        "minimum_repeats_per_target": int(minimum_repeats_per_target),
        "overall_distal_rms_mm": float(weighted_rms),
        "median_per_target_rms_mm": float(median_rms),
        "p95_per_target_rms_mm": float(p95_rms),
        "worst_target_rms_mm": float(rms_values[worst_idx]),
        "worst_target_id": str(rows_with_data[worst_idx].get("target_id", "")),
        "best_target_rms_mm": float(rms_values[best_idx]),
        "best_target_id": str(rows_with_data[best_idx].get("target_id", "")),
        "mean_repeats_per_target": float(np.mean(repeats)),
        "target_distal_rms_threshold_mm": target_distal_rms_threshold_mm,
        "targets_above_threshold": targets_above_threshold,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_two_segment_workspace_repeatability_outputs(
    *,
    output_dir: Path,
    targets: Sequence[Any],
    visit_results: Sequence[Any],
    failure_events: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the full experiment run folder (targets, captures, metrics, figures)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dicts = [_as_dict(t) for t in targets]
    visit_dicts = [_as_dict(v) for v in visit_results]
    per_target_rows = compute_workspace_repeatability_metrics(
        visit_results=visit_results, targets=targets
    )
    minimum_repeats = int(metrics.get("repeats_per_target", 20))
    summary = summarize_workspace_repeatability(
        per_target_rows,
        minimum_repeats_per_target=minimum_repeats,
    )

    paths: dict[str, Path] = {}

    # targets json
    paths["targets_json"] = output_dir / TARGETS_JSON
    paths["targets_json"].write_text(
        json.dumps({"schema_version": "two_segment_workspace_repeatability_targets_v1", "targets": target_dicts}, indent=2),
        encoding="utf-8",
    )

    # visit plan csv (synthesized from visit results)
    paths["visit_plan_csv"] = _write_visit_plan_csv(output_dir / VISIT_PLAN_CSV, visit_dicts)

    # target captures csv
    paths["target_captures_csv"] = _write_target_captures_csv(output_dir / TARGET_CAPTURES_CSV, visit_dicts)

    # per-target csv
    paths["per_target_csv"] = _write_per_target_csv(
        output_dir / PER_TARGET_REPEATABILITY_CSV, per_target_rows
    )

    # metrics json + csv
    paths["metrics_json"] = output_dir / REPEATABILITY_METRICS_JSON
    paths["metrics_json"].write_text(
        json.dumps(
            {
                "schema_version": "two_segment_workspace_repeatability_metrics_v1",
                "summary": summary,
                "per_target": per_target_rows,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    paths["metrics_csv"] = _write_metrics_summary_csv(output_dir / REPEATABILITY_METRICS_CSV, summary)

    # failure events jsonl
    paths["failure_events_jsonl"] = output_dir / FAILURE_EVENTS_JSONL
    with paths["failure_events_jsonl"].open("w", encoding="utf-8") as handle:
        for event in failure_events:
            handle.write(json.dumps(_as_dict(event), default=_json_default) + "\n")

    # summary text
    paths["summary_text"] = _write_summary_text(output_dir / SUMMARY_TXT, metrics=dict(metrics), summary=summary)

    # figures (best-effort)
    try:
        from continuum_robot.experiments.plotting import (
            create_figure,
            create_3d_figure,
            save_figure,
            style_axes,
        )

        max_amplitude_mm = float(metrics.get("max_segment_displacement_cm") or 0.0) * 10.0
        rows_with_data = [row for row in per_target_rows if row.get("rms_spread_mm") is not None]
        if rows_with_data:
            paths["thesis_01"] = _write_thesis_01(
                output_dir / TS_THESIS_01_PNG,
                rows_with_data=rows_with_data,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_3d_figure=create_3d_figure,
                save_figure=save_figure,
                style_axes=style_axes,
            )
            paths["thesis_02"] = _write_thesis_02(
                output_dir / TS_THESIS_02_PNG,
                rows_with_data=rows_with_data,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_figure=create_figure,
                save_figure=save_figure,
                style_axes=style_axes,
            )
            paths["thesis_03"] = _write_thesis_03(
                output_dir / TS_THESIS_03_PNG,
                rows_with_data=rows_with_data,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_figure=create_figure,
                save_figure=save_figure,
                style_axes=style_axes,
            )
            paths["thesis_04"] = _write_thesis_04(
                output_dir / TS_THESIS_04_PNG,
                rows_with_data=rows_with_data,
                summary=summary,
                max_amplitude_mm=max_amplitude_mm,
                create_figure=create_figure,
                save_figure=save_figure,
                style_axes=style_axes,
            )
    except Exception:
        # Figures are nice-to-have. Never block on missing matplotlib.
        pass

    return paths


# ---------------------------------------------------------------------------
# CSV / JSON writers
# ---------------------------------------------------------------------------


def _write_visit_plan_csv(path: Path, visit_dicts: Sequence[Mapping[str, Any]]) -> Path:
    fields = ["visit_position", "cycle_index", "visit_in_cycle", "target_index", "target_id", "group_tag", "amplitude_cm"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for visit in visit_dicts:
            writer.writerow({k: visit.get(k) for k in fields})
    return path


def _write_target_captures_csv(path: Path, visit_dicts: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "timestamp_utc",
        "monotonic_time_s",
        "target_index",
        "target_id",
        "cycle_index",
        "visit_in_cycle",
        "visit_position",
        "bottom_x_cm",
        "bottom_y_cm",
        "top_x_cm",
        "top_y_cm",
        "distal_x_mm",
        "distal_y_mm",
        "distal_z_mm",
        "intermediate_x_mm",
        "intermediate_y_mm",
        "intermediate_z_mm",
        "command_success",
        "capture_accepted",
        "reject_reason",
        "tracker_age_s",
        "servo_telemetry_age_s",
        "group_tag",
        "amplitude_cm",
    ] + [f"goal_tick_servo_{i}" for i in range(1, 9)] + [
        f"present_tick_servo_{i}" for i in range(1, 9)
    ] + [f"load_proxy_servo_{i}_ma" for i in range(1, 9)]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for visit in visit_dicts:
            distal = visit.get("distal_xyz_robot_mm") or [None, None, None]
            intermediate = visit.get("intermediate_xyz_robot_mm") or [None, None, None]
            goal_ticks = visit.get("all_8_goal_ticks") or {}
            present_ticks = visit.get("all_8_present_position_ticks") or {}
            load_proxies = visit.get("all_8_current_load_proxy_ma") or {}
            row = {
                "timestamp_utc": visit.get("timestamp_utc"),
                "monotonic_time_s": visit.get("monotonic_time_s"),
                "target_index": visit.get("target_index"),
                "target_id": visit.get("target_id"),
                "cycle_index": visit.get("cycle_index"),
                "visit_in_cycle": visit.get("visit_in_cycle"),
                "visit_position": visit.get("visit_position"),
                "bottom_x_cm": visit.get("bottom_x_cm"),
                "bottom_y_cm": visit.get("bottom_y_cm"),
                "top_x_cm": visit.get("top_x_cm"),
                "top_y_cm": visit.get("top_y_cm"),
                "distal_x_mm": distal[0] if len(distal) >= 1 else None,
                "distal_y_mm": distal[1] if len(distal) >= 2 else None,
                "distal_z_mm": distal[2] if len(distal) >= 3 else None,
                "intermediate_x_mm": intermediate[0] if intermediate and len(intermediate) >= 1 else None,
                "intermediate_y_mm": intermediate[1] if intermediate and len(intermediate) >= 2 else None,
                "intermediate_z_mm": intermediate[2] if intermediate and len(intermediate) >= 3 else None,
                "command_success": int(bool(visit.get("command_success"))) if visit.get("command_success") is not None else None,
                "capture_accepted": int(bool(visit.get("accepted") or visit.get("capture_accepted"))),
                "reject_reason": visit.get("reject_reason"),
                "tracker_age_s": visit.get("tracker_age_s"),
                "servo_telemetry_age_s": visit.get("servo_telemetry_age_s"),
                "group_tag": visit.get("group_tag"),
                "amplitude_cm": visit.get("amplitude_cm"),
            }
            for sid in range(1, 9):
                row[f"goal_tick_servo_{sid}"] = goal_ticks.get(str(sid))
                row[f"present_tick_servo_{sid}"] = present_ticks.get(str(sid))
                row[f"load_proxy_servo_{sid}_ma"] = load_proxies.get(str(sid))
            writer.writerow(row)
    return path


def _write_per_target_csv(path: Path, per_target_rows: Sequence[Mapping[str, Any]]) -> Path:
    fields = [
        "target_index",
        "target_id",
        "group_tag",
        "amplitude_cm",
        "bottom_x_cm",
        "bottom_y_cm",
        "top_x_cm",
        "top_y_cm",
        "accepted_repeats",
        "rms_spread_mm",
        "mean_radial_mm",
        "median_radial_mm",
        "max_radial_mm",
        "std_x_mm",
        "std_y_mm",
        "std_z_mm",
        "centroid_x_mm",
        "centroid_y_mm",
        "centroid_z_mm",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_target_rows:
            centroid = row.get("centroid_xyz_mm") or [None, None, None]
            writer.writerow(
                {
                    "target_index": row.get("target_index"),
                    "target_id": row.get("target_id"),
                    "group_tag": row.get("group_tag"),
                    "amplitude_cm": row.get("amplitude_cm"),
                    "bottom_x_cm": row.get("bottom_x_cm"),
                    "bottom_y_cm": row.get("bottom_y_cm"),
                    "top_x_cm": row.get("top_x_cm"),
                    "top_y_cm": row.get("top_y_cm"),
                    "accepted_repeats": row.get("accepted_repeats"),
                    "rms_spread_mm": row.get("rms_spread_mm"),
                    "mean_radial_mm": row.get("mean_radial_mm"),
                    "median_radial_mm": row.get("median_radial_mm"),
                    "max_radial_mm": row.get("max_radial_mm"),
                    "std_x_mm": row.get("std_x_mm"),
                    "std_y_mm": row.get("std_y_mm"),
                    "std_z_mm": row.get("std_z_mm"),
                    "centroid_x_mm": centroid[0] if centroid else None,
                    "centroid_y_mm": centroid[1] if centroid else None,
                    "centroid_z_mm": centroid[2] if centroid else None,
                }
            )
    return path


def _write_metrics_summary_csv(path: Path, summary: Mapping[str, Any]) -> Path:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([str(key), _csv_value(value)])
    return path


def _write_summary_text(path: Path, *, metrics: Mapping[str, Any], summary: Mapping[str, Any]) -> Path:
    lines = [
        "Two-Segment Workspace Repeatability",
        f"schema_version: {metrics.get('schema_version', '')}",
        f"experiment_name: {metrics.get('experiment_name', 'two_segment_workspace_repeatability')}",
        f"demo_only: {metrics.get('demo_only', False)}",
        f"valid_for_repeatability_analysis: {metrics.get('valid_for_repeatability_analysis', True)}",
        f"valid_for_thesis_repeatability: {metrics.get('valid_for_thesis_repeatability', False)}",
        f"valid_for_model_training: {metrics.get('valid_for_model_training', False)}",
        f"primary_metric: {metrics.get('primary_metric', 'distal_xyz_repeatability')}",
        f"controlled_point: {metrics.get('controlled_point', 'distal_tip coil origin in robot base frame')}",
        f"expected_distal_tool_id: {metrics.get('expected_distal_tool_id', '0A')}",
        "",
        "Protocol:",
        f"  target_count:                  {metrics.get('target_count', 0)}",
        f"  repeats_per_target:            {metrics.get('repeats_per_target', 0)}",
        f"  planned_visits:                {metrics.get('planned_visits', 0)}",
        f"  accepted_captures:             {metrics.get('accepted_captures', 0)}",
        f"  rejected_captures:             {metrics.get('rejected_captures', 0)}",
        f"  stop_reason:                   {metrics.get('stop_reason', '')}",
        f"  target_generator_mode:         {metrics.get('target_generator_mode', '')}",
        f"  random_seed:                   {metrics.get('random_seed', 0)}",
        f"  max_segment_displacement_cm:   {metrics.get('max_segment_displacement_cm', 0)}",
        f"  return_to_neutral_between:     {metrics.get('return_to_neutral_between_visits', True)}",
        f"  neutral_settle_s:              {metrics.get('neutral_settle_s', 0)}",
        f"  target_settle_s:               {metrics.get('target_settle_s', 0)}",
        f"  bottom_segment_key:            {metrics.get('bottom_segment_key', '')}",
        f"  top_segment_key:               {metrics.get('top_segment_key', '')}",
        "",
        "Repeatability summary (distal XYZ):",
        f"  overall_distal_rms_mm:         {summary.get('overall_distal_rms_mm')}",
        f"  median_per_target_rms_mm:      {summary.get('median_per_target_rms_mm')}",
        f"  p95_per_target_rms_mm:         {summary.get('p95_per_target_rms_mm')}",
        f"  worst_target_rms_mm:           {summary.get('worst_target_rms_mm')} (id={summary.get('worst_target_id')})",
        f"  best_target_rms_mm:            {summary.get('best_target_rms_mm')} (id={summary.get('best_target_id')})",
        f"  targets_with_repeats:          {summary.get('targets_with_repeats')}",
        f"  targets_below_minimum_repeats: {summary.get('targets_below_minimum_repeats')}",
        f"  minimum_repeats_per_target:    {summary.get('minimum_repeats_per_target')}",
        f"  mean_repeats_per_target:       {summary.get('mean_repeats_per_target')}",
        "",
        "This is a data-collection workflow. Repeatability is reported honestly;",
        "the run can be valid even if RMS exceeds any operator-configured target.",
    ]
    Path(path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Figures — mirror single-segment workspace_repeatability_map figure set
# ---------------------------------------------------------------------------


def _color_vmax(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    arr = np.asarray(values, dtype=float)
    p95 = float(np.percentile(arr, 95.0))
    return max(p95, float(arr.max()) * 0.5, 1e-6)


def _write_thesis_01(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_3d_figure,
    save_figure,
    style_axes,
) -> Path:
    fig, ax = create_3d_figure()
    xs = [float(r["x_mm"]) for r in rows_with_data]
    ys = [float(r["y_mm"]) for r in rows_with_data]
    zs = [float(r["rms_spread_mm"]) for r in rows_with_data]
    vmax = _color_vmax(zs)
    sc = ax.scatter(xs, ys, zs, c=zs, cmap="viridis", vmin=0.0, vmax=vmax, s=20)
    ax.set_xlabel("Bottom X (mm, commanded)")
    ax.set_ylabel("Bottom Y (mm, commanded)")
    ax.set_zlabel("RMS spread (mm)")
    ax.set_title(
        "Two-Segment Workspace Repeatability: per-target distal RMS (3D)\n"
        f"overall RMS = {_fmt(summary.get('overall_distal_rms_mm'))} mm"
    )
    fig.colorbar(sc, ax=ax, label="RMS spread (mm)", shrink=0.7)
    save_figure(fig, path)
    return path


def _write_thesis_02(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_figure,
    save_figure,
    style_axes,
) -> Path:
    fig, ax = create_figure(size="square")
    xs = [float(r["x_mm"]) for r in rows_with_data]
    ys = [float(r["y_mm"]) for r in rows_with_data]
    zs = [float(r["rms_spread_mm"]) for r in rows_with_data]
    vmax = _color_vmax(zs)
    sc = ax.scatter(xs, ys, c=zs, cmap="viridis", vmin=0.0, vmax=vmax, s=30, edgecolors="white", linewidths=0.4)
    if max_amplitude_mm > 0:
        circle = np.linspace(0.0, 2.0 * math.pi, 64)
        ax.plot(
            float(max_amplitude_mm) * np.cos(circle),
            float(max_amplitude_mm) * np.sin(circle),
            color="#94a3b8",
            linewidth=0.7,
            linestyle="--",
            label=f"workspace ±{max_amplitude_mm:.1f} mm",
        )
    fig.colorbar(sc, ax=ax, label="RMS spread (mm)")
    style_axes(
        ax,
        title="Two-Segment Workspace RMS Map (commanded bottom XY)",
        xlabel="Bottom X (mm)",
        ylabel="Bottom Y (mm)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    if max_amplitude_mm > 0:
        ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, path)
    return path


def _write_thesis_03(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_figure,
    save_figure,
    style_axes,
) -> Path:
    fig, ax = create_figure(size="wide")
    amplitudes = [float(r.get("target_amplitude_mm") or 0.0) for r in rows_with_data]
    rms = [float(r["rms_spread_mm"]) for r in rows_with_data]
    ax.scatter(amplitudes, rms, s=20, alpha=0.7)
    if summary.get("overall_distal_rms_mm") is not None:
        ax.axhline(
            float(summary["overall_distal_rms_mm"]),
            color="#dc2626",
            linewidth=0.8,
            linestyle="--",
            label=f"overall RMS = {float(summary['overall_distal_rms_mm']):.3f} mm",
        )
    if summary.get("target_distal_rms_threshold_mm") is not None:
        ax.axhline(
            float(summary["target_distal_rms_threshold_mm"]),
            color="#0ea5e9",
            linewidth=0.8,
            linestyle=":",
            label=f"operator threshold = {float(summary['target_distal_rms_threshold_mm']):.3f} mm",
        )
    style_axes(
        ax,
        title="Two-Segment Per-Target RMS vs Commanded Amplitude",
        xlabel="Target commanded amplitude (mm, 4D L2-norm)",
        ylabel="RMS spread (mm)",
    )
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, path)
    return path


def _write_thesis_04(
    path: Path,
    *,
    rows_with_data: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    max_amplitude_mm: float,
    create_figure,
    save_figure,
    style_axes,
) -> Path:
    fig, ax = create_figure(size="square")
    xs = [float(r["x_mm"]) for r in rows_with_data]
    ys = [float(r["y_mm"]) for r in rows_with_data]
    zs = [float(r["rms_spread_mm"]) for r in rows_with_data]
    vmax = _color_vmax(zs)
    sizes = [40.0 + 200.0 * float(z) / max(vmax, 1e-6) for z in zs]
    sc = ax.scatter(xs, ys, s=sizes, c=zs, cmap="viridis", vmin=0.0, vmax=vmax, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="RMS spread (mm)")
    style_axes(
        ax,
        title="Two-Segment 2D Repeatability Map (bigger = worse)",
        xlabel="Bottom X (mm)",
        ylabel="Bottom Y (mm)",
    )
    ax.set_aspect("equal", adjustable="datalim")
    save_figure(fig, path)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if hasattr(value, "__dict__"):
        return {k: getattr(value, k) for k in vars(value)}
    return {}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, default=_json_default)
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)
