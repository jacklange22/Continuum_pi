"""Sci-fi waypoint relay output writers.

Emitted alongside the standard slow motion demo bundle when the pattern
is ``sci_fi_waypoint_relay``. Includes:

- ``waypoints.json``               — resolved 4D waypoint list + resolved seed
- ``waypoint_schedule.csv``        — issue_time_s + waypoint_index rows
- ``sci_fi_waypoint_preview.png``  — bottom and top XY workspace preview
- ``sci_fi_servo_goal_trace.png``  — per-servo goal tick timeline (when trace has data)
- ``sci_fi_waypoint_relay_summary.txt`` — operator-facing summary

All writers are pure (no hardware). Figure writers degrade gracefully
when matplotlib is unavailable.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence


LOG = logging.getLogger(__name__)


def write_sci_fi_relay_outputs(
    *,
    output_dir: Path,
    config: Any,
    waypoints: Sequence[Any],
    schedule: Sequence[Any],
    resolved_seed: int,
    advisory: Any,
    profile_restore_success: bool | None,
    profile_restore_failure_reason: str | None,
    trace: Sequence[Any],
) -> dict[str, Path]:
    """Write every sci-fi relay output and return the path map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["waypoints_json"] = output_dir / "waypoints.json"
    _write_waypoints_json(
        path=paths["waypoints_json"],
        config=config,
        waypoints=waypoints,
        resolved_seed=resolved_seed,
        advisory=advisory,
    )

    paths["schedule_csv"] = output_dir / "waypoint_schedule.csv"
    _write_schedule_csv(path=paths["schedule_csv"], schedule=schedule)

    paths["sci_fi_summary_txt"] = output_dir / "sci_fi_waypoint_relay_summary.txt"
    _write_sci_fi_summary_text(
        path=paths["sci_fi_summary_txt"],
        config=config,
        waypoints=waypoints,
        schedule=schedule,
        resolved_seed=resolved_seed,
        advisory=advisory,
        profile_restore_success=profile_restore_success,
        profile_restore_failure_reason=profile_restore_failure_reason,
        trace=trace,
    )

    try:
        paths["preview_png"] = output_dir / "sci_fi_waypoint_preview.png"
        _write_waypoint_preview_png(
            path=paths["preview_png"],
            config=config,
            waypoints=waypoints,
            schedule=schedule,
        )
    except Exception as exc:  # noqa: BLE001 — never block bundle on figure failure
        LOG.warning("sci_fi_waypoint_preview render failed: %s", exc)
        paths.pop("preview_png", None)

    try:
        paths["goal_trace_png"] = output_dir / "sci_fi_servo_goal_trace.png"
        _write_servo_goal_trace_png(path=paths["goal_trace_png"], trace=trace)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("sci_fi_servo_goal_trace render failed: %s", exc)
        paths.pop("goal_trace_png", None)

    return paths


# ---------------------------------------------------------------------------
# Text + JSON
# ---------------------------------------------------------------------------


def _write_waypoints_json(
    *,
    path: Path,
    config: Any,
    waypoints: Sequence[Any],
    resolved_seed: int,
    advisory: Any,
) -> None:
    payload = {
        "schema_version": "sci_fi_waypoint_relay_v1",
        "pattern": "sci_fi_waypoint_relay",
        "waypoint_source": str(getattr(config, "waypoint_source", "preset_weird")),
        "configured_seed": int(getattr(config, "relay_seed", 0)),
        "resolved_seed": int(resolved_seed),
        "auto_select_seed": bool(getattr(config, "auto_select_seed", True)),
        "amplitude_cm": float(getattr(config, "amplitude_cm", 0.0)),
        "video_duration_s": float(getattr(config, "video_duration_s", 0.0)),
        "early_switch_fraction": float(getattr(config, "early_switch_fraction", 0.0)),
        "profile_velocity": getattr(config, "profile_velocity", None),
        "profile_acceleration": getattr(config, "profile_acceleration", None),
        "speed_class": str(getattr(advisory, "speed_class", "unknown")) if advisory else "unknown",
        "advisory_warnings": (
            list(getattr(advisory, "warnings", ())) if advisory else []
        ),
        "waypoints": [
            {
                "index": int(w.index),
                "bottom_x_cm": float(w.bottom_x_cm),
                "bottom_y_cm": float(w.bottom_y_cm),
                "top_x_cm": float(w.top_x_cm),
                "top_y_cm": float(w.top_y_cm),
            }
            for w in waypoints
        ],
        "demo_only": True,
        "closed_loop_control": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_schedule_csv(*, path: Path, schedule: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["waypoint_index", "issue_time_s", "bottom_x_cm", "bottom_y_cm", "top_x_cm", "top_y_cm"])
        for entry in schedule:
            wp = entry.waypoint
            writer.writerow(
                [
                    int(entry.waypoint_index),
                    f"{float(entry.issue_time_s):.4f}",
                    f"{float(wp.bottom_x_cm):.4f}",
                    f"{float(wp.bottom_y_cm):.4f}",
                    f"{float(wp.top_x_cm):.4f}",
                    f"{float(wp.top_y_cm):.4f}",
                ]
            )


def _write_sci_fi_summary_text(
    *,
    path: Path,
    config: Any,
    waypoints: Sequence[Any],
    schedule: Sequence[Any],
    resolved_seed: int,
    advisory: Any,
    profile_restore_success: bool | None,
    profile_restore_failure_reason: str | None,
    trace: Sequence[Any],
) -> None:
    sent_count = sum(1 for row in trace if bool(getattr(row, "command_sent", False)))
    skipped_count = sum(1 for row in trace if getattr(row, "skip_reason", None) is not None)
    lines = [
        "Two-Segment Sci-Fi Waypoint Relay Demo Summary",
        "=" * 46,
        f"Pattern:                 sci_fi_waypoint_relay",
        f"Video duration:          {float(getattr(config, 'video_duration_s', 0.0)):.1f} s",
        f"Waypoint count:          {int(getattr(config, 'waypoint_count', 0))}",
        f"Amplitude:               {float(getattr(config, 'amplitude_cm', 0.0)):.2f} cm",
        f"Early switch fraction:   {float(getattr(config, 'early_switch_fraction', 0.0)):.2f}",
        f"Waypoint source:         {str(getattr(config, 'waypoint_source', ''))}",
        f"Configured seed:         {int(getattr(config, 'relay_seed', 0))}",
        f"Resolved seed:           {int(resolved_seed)}",
        f"Auto-select seed:        {bool(getattr(config, 'auto_select_seed', False))}",
        f"Profile velocity:        {getattr(config, 'profile_velocity', None)}",
        f"Profile acceleration:    {getattr(config, 'profile_acceleration', None)}",
        f"Command rate (bookkeep): {float(getattr(config, 'command_rate_hz', 0.0)):.1f} Hz",
        f"Dry-run:                 {bool(getattr(config, 'dry_run', True))}",
        "",
        f"Speed class:             {str(getattr(advisory, 'speed_class', 'unknown')) if advisory else 'unknown'}",
        f"Seconds per waypoint:    {float(getattr(advisory, 'seconds_per_waypoint', 0.0)) if advisory else 0.0:.2f}",
        f"Estimated bus writes:    {int(getattr(advisory, 'estimated_bus_writes', 0)) if advisory else 0}",
        f"Schedule entries:        {len(schedule)}",
        f"Trace rows (all):        {len(trace)}",
        f"Trace rows sent:         {sent_count}",
        f"Trace rows skipped:      {skipped_count}",
        "",
        "Profile restore:",
        f"  success:                 {profile_restore_success}",
        f"  failure_reason:          {profile_restore_failure_reason or 'n/a'}",
    ]
    if advisory and getattr(advisory, "warnings", None):
        lines.append("")
        lines.append("Advisory warnings:")
        for w in advisory.warnings:
            lines.append(f"  - {w}")
    lines.extend(
        [
            "",
            "Validity:",
            "  demo_only: True",
            "  closed_loop_control: False",
            "  motion_pattern_demo: True",
            "  valid_for_model_training: False",
            "  valid_for_thesis_repeatability: False",
            "",
            "This is a presentation-only motion demo. It is not data, not closed-loop control,",
            "and not suitable for model training or thesis claims. The relay deliberately sends",
            "a new waypoint command before the spine fully settles at the previous one to",
            "produce a smooth 'almost reaching / changing its mind' motion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures (matplotlib)
# ---------------------------------------------------------------------------


def _write_waypoint_preview_png(
    *,
    path: Path,
    config: Any,
    waypoints: Sequence[Any],
    schedule: Sequence[Any],
) -> None:
    """Bottom + top XY workspace preview with numbered waypoints and arrows."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bottom_xy = [(w.bottom_x_cm, w.bottom_y_cm) for w in waypoints]
    top_xy = [(w.top_x_cm, w.top_y_cm) for w in waypoints]
    amp = float(getattr(config, "amplitude_cm", 0.5))

    fig, (ax_b, ax_t) = plt.subplots(1, 2, figsize=(10.0, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    for ax, xys, title in (
        (ax_b, bottom_xy, "Bottom segment command"),
        (ax_t, top_xy, "Top segment command"),
    ):
        xs = [pt[0] for pt in xys]
        ys = [pt[1] for pt in xys]
        # Arrows connecting consecutive waypoints (visualises the relay order).
        for i in range(len(xs) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color="#9ca3af", lw=1.2, alpha=0.65),
            )
        ax.scatter(xs, ys, s=130, color="#2563eb", edgecolors="white", linewidths=0.8, zorder=3)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x, y, str(i), color="white", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)
        # Workspace square at ±amplitude
        ax.plot([-amp, amp, amp, -amp, -amp], [-amp, -amp, amp, amp, -amp],
                color="#cbd5e1", linewidth=0.8, linestyle="--")
        ax.set_xlim(-amp * 1.15, amp * 1.15)
        ax.set_ylim(-amp * 1.15, amp * 1.15)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        ax.grid(True, color="#e2e8f0", linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"Sci-Fi Waypoint Relay  —  {len(waypoints)} waypoints, "
        f"video {float(getattr(config, 'video_duration_s', 0.0)):.1f} s, amp {amp:.2f} cm",
        fontsize=12, fontweight="bold", x=0.04, ha="left",
    )
    # Schedule strip footer
    if schedule:
        issue_times = ", ".join(f"#{int(e.waypoint_index)}@{float(e.issue_time_s):.1f}s"
                                for e in list(schedule)[: min(6, len(schedule))])
        if len(schedule) > 6:
            issue_times += f", …(+{len(schedule) - 6} more)"
        fig.text(
            0.015, 0.015,
            f"Relay schedule: {issue_times}",
            fontsize=8.5, color="#374151", ha="left",
        )
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _write_servo_goal_trace_png(*, path: Path, trace: Sequence[Any]) -> None:
    """Per-servo goal-tick timeline from the demo trace.

    One line per servo (1..8), x-axis = elapsed seconds, y-axis = goal tick.
    Useful for spotting whether the relay actually produced visibly
    different commands per servo over the run.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if not trace:
        return
    times = [float(row.elapsed_s) for row in trace]
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14)
    palette = ("#2563eb", "#1d4ed8", "#1e40af", "#1e3a8a",
               "#d97706", "#b45309", "#92400e", "#78350f")
    plotted = False
    for sid in range(1, 9):
        ys = [
            int(row.goal_ticks_by_servo.get(sid, row.goal_ticks_by_servo.get(str(sid), float("nan"))))
            if (sid in row.goal_ticks_by_servo or str(sid) in row.goal_ticks_by_servo)
            else None
            for row in trace
        ]
        if any(y is not None for y in ys):
            xs_plot = [t for t, y in zip(times, ys) if y is not None]
            ys_plot = [y for y in ys if y is not None]
            ax.plot(xs_plot, ys_plot, color=palette[sid - 1], linewidth=1.2,
                    marker="o", markersize=3, label=f"servo {sid}")
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "No goal ticks recorded in trace",
                transform=ax.transAxes, ha="center", va="center", color="#6b7280")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Goal tick")
    ax.set_title("Per-servo goal-tick relay trace", loc="left", fontweight="bold", pad=6)
    ax.grid(True, color="#e2e8f0", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if plotted:
        ax.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.92)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
