"""Thesis-quality output writer for the neutral-setpoint drift validation experiment.

Mirrors the contract used by ``write_registration_validation_outputs`` and
``write_pivot_validation_outputs``:

  - 4 thesis-quality PNGs covering drift over time, per-servo distribution,
    drift magnitude, and event/source breakdown.
  - debug.json with the full event list + per-servo timelines + summary stats.
  - per-event rows live in summary.json under
    ``experiment_metrics.per_servo_rows`` and ``experiment_metrics.events``.
  - samples.jsonl is the typed timeseries contract; intentionally empty here.
  - metrics.csv and a summary text are intentionally NOT written; debug.json
    is the structured fallback for downstream consumers.

A figure-write failure replaces the PNG with a placeholder so the bundle
exporter never sees a missing artifact.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from continuum_robot.experiments.plotting import (
    add_metric_box,
    color,
    create_figure,
    legend,
    save_figure,
    style_axes,
)
from continuum_robot.experiments.tracker_timing_outputs import (
    _write_plot_placeholder,
)


# Palette: one stable colour per servo so the same servo is the same colour
# across every figure (matches the report convention used in pretension
# validation plots).
_SERVO_PALETTE = [
    color("measured"),  # blue
    color("fit"),       # orange / gold
    color("accepted"),  # teal
    color("prediction"),# pink/magenta
    color("target"),    # red
    color("reference"), # dark slate
    color("neutral"),   # slate grey
    color("rejected"),  # red-orange
]


def _servo_color(index: int) -> str:
    return _SERVO_PALETTE[int(index) % len(_SERVO_PALETTE)]


def _safe_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def write_neutral_setpoint_drift_validation_outputs(
    *,
    output_dir: Path,
    metadata,
    summary,
) -> dict[str, Path]:
    """Write the canonical 4-figure thesis bundle + debug.json."""
    metrics = summary.experiment_metrics if isinstance(summary.experiment_metrics, dict) else {}
    output_dir = Path(output_dir)
    debug_json_path = output_dir / "debug.json"
    thesis_01_path = output_dir / "thesis_01_neutral_setpoint_drift_timeline.png"
    thesis_02_path = output_dir / "thesis_02_neutral_setpoint_distribution_per_servo.png"
    thesis_03_path = output_dir / "thesis_03_drift_magnitude_per_servo.png"
    thesis_04_path = output_dir / "thesis_04_event_type_and_source_breakdown.png"

    _write_debug_json(
        path=debug_json_path,
        output_dir=output_dir,
        metadata=metadata,
        summary=summary,
        metrics=metrics,
    )
    figure_writers = [
        (thesis_01_path, lambda: _write_thesis_01_drift_timeline(path=thesis_01_path, metrics=metrics)),
        (thesis_02_path, lambda: _write_thesis_02_distribution(path=thesis_02_path, metrics=metrics)),
        (thesis_03_path, lambda: _write_thesis_03_drift_magnitude(path=thesis_03_path, metrics=metrics)),
        (thesis_04_path, lambda: _write_thesis_04_event_breakdown(path=thesis_04_path, metrics=metrics)),
    ]
    for path, writer in figure_writers:
        try:
            writer()
        except Exception:  # pragma: no cover - placeholder safety net
            _write_plot_placeholder(path)

    return {
        "debug_json_path": debug_json_path,
        "thesis_01_path": thesis_01_path,
        "thesis_02_path": thesis_02_path,
        "thesis_03_path": thesis_03_path,
        "thesis_04_path": thesis_04_path,
    }


def _write_debug_json(
    *,
    path: Path,
    output_dir: Path,
    metadata,
    summary,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "neutral_setpoint_drift_validation_debug_v1",
        "experiment_name": metrics.get("experiment_name"),
        "output_dir": str(output_dir),
        "run_id": getattr(metadata, "run_id", None),
        "experiment_timestamp_utc": getattr(metadata, "timestamp_utc", None),
        "summary_status": getattr(summary, "status", None),
        "event_count": int(metrics.get("event_count") or 0),
        "servo_count": int(metrics.get("servo_count") or 0),
        "first_event_timestamp_utc": metrics.get("first_event_timestamp_utc"),
        "last_event_timestamp_utc": metrics.get("last_event_timestamp_utc"),
        "filter_event_types": list(metrics.get("filter_event_types") or []),
        "filter_servo_ids": list(metrics.get("filter_servo_ids") or []),
        "event_type_counts": dict(metrics.get("event_type_counts") or {}),
        "capture_source_counts": dict(metrics.get("capture_source_counts") or {}),
        "per_servo_rows": list(metrics.get("per_servo_rows") or []),
        "timelines_by_servo": dict(metrics.get("timelines_by_servo") or {}),
        "events": list(metrics.get("events") or []),
        "definitions": dict(metrics.get("definitions") or {}),
        "units": dict(metrics.get("units") or {}),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figure 1 — drift timeline per servo
# --------------------------------------------------------------------------- #


def _write_thesis_01_drift_timeline(*, path: Path, metrics: dict[str, Any]) -> None:
    """Per-servo neutral_setpoint vs event index (a left-to-right timeline).

    Operators read this figure to spot:
      - A servo whose neutral drifted monotonically (loose mount, slipping spool)
      - A discrete jump (someone re-zeroed)
      - Convergence after repeated captures
    """
    timelines = dict(metrics.get("timelines_by_servo") or {})
    fig, ax = create_figure(size="wide")
    if not timelines:
        ax.text(0.5, 0.5, "No events to plot", transform=ax.transAxes, ha="center", va="center")
        style_axes(
            ax,
            title="Neutral setpoint drift over time",
            xlabel="Event index",
            ylabel="Neutral setpoint (ticks)",
        )
        save_figure(fig, path)
        return

    drift_for_box: list[str] = []
    for index, (sid_str, entries) in enumerate(sorted(timelines.items(), key=lambda item: int(item[0]))):
        sid = int(sid_str)
        xs: list[int] = []
        ys: list[int] = []
        for row in entries:
            value = row.get("neutral_setpoint")
            if value is None:
                continue
            xs.append(int(row.get("event_index", 0)))
            ys.append(int(value))
        if not ys:
            continue
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=_servo_color(index),
            alpha=0.92,
            label=f"Servo {sid}",
        )
        # First-value reference (dashed horizontal segment over the first 5%
        # of the timeline) so the eye can compare drift visually.
        first_value = ys[0]
        ax.axhline(
            y=first_value,
            xmin=0.0,
            xmax=0.05,
            color=_servo_color(index),
            linewidth=2.0,
            alpha=0.45,
            linestyle="--",
        )
        # Per-servo annotation: drift magnitude.
        drift_for_box.append(
            f"Servo {sid}: drift={ys[-1] - ys[0]:+d} ticks  "
            f"max_excursion={max(abs(v - first_value) for v in ys)} ticks"
        )

    style_axes(
        ax,
        title="Neutral setpoint drift over time",
        xlabel="Event index (chronological)",
        ylabel="Neutral setpoint (ticks)",
    )
    if drift_for_box:
        add_metric_box(ax, drift_for_box, loc="upper right")
    legend(ax, loc="lower right", ncol=2)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Figure 2 — distribution per servo (boxplot)
# --------------------------------------------------------------------------- #


def _write_thesis_02_distribution(*, path: Path, metrics: dict[str, Any]) -> None:
    """Box / strip plot of every recorded neutral_setpoint per servo.

    Reads at a glance the variance for each servo: short box = tight neutral
    behaviour, tall box = drift over the recorded period."""
    import matplotlib.pyplot as plt  # local; matplotlib lives behind plotting helpers

    timelines = dict(metrics.get("timelines_by_servo") or {})
    sid_keys = sorted(timelines, key=lambda s: int(s))
    data: list[list[int]] = []
    labels: list[str] = []
    for sid_str in sid_keys:
        values = [
            int(row["neutral_setpoint"])
            for row in timelines.get(sid_str, [])
            if row.get("neutral_setpoint") is not None
        ]
        if not values:
            continue
        data.append(values)
        labels.append(f"S{int(sid_str)}")

    fig, ax = create_figure(size="wide")
    if not data:
        ax.text(0.5, 0.5, "No neutral data to plot", transform=ax.transAxes, ha="center", va="center")
        style_axes(
            ax,
            title="Neutral setpoint distribution per servo",
            xlabel="Servo",
            ylabel="Neutral setpoint (ticks)",
        )
        save_figure(fig, path)
        return

    box = ax.boxplot(
        data,
        tick_labels=labels,
        widths=0.55,
        patch_artist=True,
        medianprops={"color": color("reference"), "linewidth": 1.5},
        whiskerprops={"color": color("axis"), "linewidth": 1.2},
        capprops={"color": color("axis"), "linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": color("rejected"),
            "markeredgecolor": color("rejected"),
            "markersize": 5,
            "alpha": 0.7,
        },
    )
    # Per-servo colours on the box body.
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(_servo_color(index))
        patch.set_alpha(0.40)
        patch.set_edgecolor(_servo_color(index))

    # Overlay strip plot showing every event as a small marker so an
    # operator can see "I captured this 6 times" at a glance.
    for index, values in enumerate(data):
        ax.scatter(
            [index + 1 + 0.15] * len(values),
            values,
            s=22,
            color=_servo_color(index),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
        )

    style_axes(
        ax,
        title="Neutral setpoint distribution per servo",
        xlabel="Servo",
        ylabel="Neutral setpoint (ticks)",
    )
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Figure 3 — drift magnitude per servo (bar)
# --------------------------------------------------------------------------- #


def _write_thesis_03_drift_magnitude(*, path: Path, metrics: dict[str, Any]) -> None:
    """Bar chart: drift_ticks (last - first) and max excursion per servo.

    Two bars per servo so the operator sees both:
      - signed end-to-end drift (positive = neutral moved looser direction)
      - peak absolute excursion (largest single deviation seen at any point)
    A long max-excursion bar with a small drift_ticks bar tells the operator
    the servo wandered then came back; both small = stable."""
    per_servo_rows = list(metrics.get("per_servo_rows") or [])
    labels: list[str] = []
    drift_values: list[int] = []
    excursion_values: list[int] = []
    for row in per_servo_rows:
        sid = int(row.get("servo_id"))
        labels.append(f"S{sid}")
        drift_values.append(int(row.get("neutral_drift_ticks") or 0))
        excursion_values.append(int(row.get("max_excursion_from_first_ticks") or 0))

    fig, ax = create_figure(size="wide")
    if not labels:
        ax.text(0.5, 0.5, "No drift data", transform=ax.transAxes, ha="center", va="center")
        style_axes(
            ax,
            title="Per-servo neutral drift magnitude",
            xlabel="Servo",
            ylabel="Drift (ticks)",
        )
        save_figure(fig, path)
        return

    import numpy as np

    x = np.arange(len(labels))
    width = 0.38
    ax.bar(
        x - width / 2,
        drift_values,
        width,
        color=color("measured"),
        alpha=0.85,
        edgecolor=color("axis"),
        linewidth=0.6,
        label="End-to-end drift (last − first)",
    )
    ax.bar(
        x + width / 2,
        excursion_values,
        width,
        color=color("rejected"),
        alpha=0.85,
        edgecolor=color("axis"),
        linewidth=0.6,
        label="Max |excursion| from first",
    )
    # Zero reference line so signed drift is unambiguous.
    ax.axhline(y=0, color=color("axis"), linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(
        ax,
        title="Per-servo neutral drift magnitude",
        xlabel="Servo",
        ylabel="Drift (ticks)",
    )
    legend(ax, loc="best", ncol=2)
    save_figure(fig, path)


# --------------------------------------------------------------------------- #
# Figure 4 — event-type + capture-source breakdown
# --------------------------------------------------------------------------- #


def _write_thesis_04_event_breakdown(*, path: Path, metrics: dict[str, Any]) -> None:
    """Two-panel summary: which kinds of events landed in this analysis, and
    which capture sources they came from. Useful for the operator to spot
    "I only have manual_pretension_captured events, never accepted" or
    "every event came from soft_release_to_zero_current"."""
    import matplotlib.pyplot as plt

    event_counts = dict(metrics.get("event_type_counts") or {})
    source_counts = dict(metrics.get("capture_source_counts") or {})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    def _draw_bar(ax, counts: dict[str, int], title: str, color_key: str) -> None:
        if not counts:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title, loc="left", pad=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            return
        # Order by count desc, ties broken by name for stability.
        ordered = OrderedDict(
            sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        )
        labels = list(ordered.keys())
        values = [int(v) for v in ordered.values()]
        ax.barh(
            list(reversed(labels)),
            list(reversed(values)),
            color=color(color_key),
            alpha=0.85,
            edgecolor="#1f2937",
            linewidth=0.6,
        )
        for spine_side in ("top", "right"):
            ax.spines[spine_side].set_visible(False)
        ax.set_title(title, loc="left", pad=10, fontweight="bold")
        ax.set_xlabel("Event count")
        ax.grid(True, alpha=0.3, axis="x")
        # Numeric label at end of each bar.
        for y, count in enumerate(reversed(values)):
            ax.text(count, y, f" {count}", va="center", fontsize=9, color="#1f2937")

    _draw_bar(axes[0], event_counts, "Events by type", "measured")
    _draw_bar(axes[1], source_counts, "Events by capture source", "accepted")
    save_figure(fig, path)
