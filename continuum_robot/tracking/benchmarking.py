"""Tracker diagnostics and benchmark helpers built on top of TrackingService snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import time

from continuum_robot.services.models import TrackingSnapshot
from continuum_robot.utils.time_utils import utc_now_iso


@dataclass
class NumericStats:
    """Basic descriptive stats for one numeric series."""

    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    stdev: float | None = None
    discarded_non_finite: int = 0


@dataclass
class ToolBenchmarkMetrics:
    """Per-tool tracking quality summary over the benchmark window."""

    tool_id: str
    present_frames: int = 0
    tracked_frames: int = 0
    missing_frames: int = 0
    invalid_frames: int = 0
    warmup_invalid_frames: int = 0
    warmup_nonfinite_invalid_frames: int = 0
    post_warmup_invalid_frames: int = 0
    unknown_frames: int = 0
    max_consecutive_missing_frames: int = 0
    time_to_first_tracked_frame_s: float | None = None
    quality: NumericStats | None = None
    latest_tracking_state: str = "unknown"


@dataclass
class TrackerBenchmarkThresholds:
    """Acceptance thresholds for tracker validation."""

    min_effective_fps: float = 20.0
    max_stale_interval_s: float = 0.25
    max_consecutive_missing_frames: int = 20
    require_valid_transforms: bool = True


@dataclass
class TrackerBenchmarkReport:
    """Serializable tracker benchmark summary."""

    generated_at_utc: str
    configured_backend_name: str
    selected_backend_name: str
    backend_identity: str
    port: str
    duration_s: float
    samples_observed: int
    unique_frames_observed: int
    backend_frame_counter_final: int
    frame_interval_s: NumericStats
    effective_frame_rate_hz: float | None
    stale_sample_count: int
    stale_sample_ratio: float
    max_data_age_s: float | None
    first_frame_latency_s: float | None
    first_valid_frame_latency_s: float | None
    startup_state: str
    warmup_invalid_frame_count: int
    warmup_nonfinite_invalid_frame_count: int
    warmup_invalid_frame_count_by_tool: dict[str, int]
    warmup_nonfinite_invalid_frame_count_by_tool: dict[str, int]
    registration_loaded: bool
    tip_pose_computable: bool
    canonical_state_final: str
    final_connection_state: str
    final_last_error: str | None
    warning_messages_final: list[str]
    error_messages_final: list[str]
    registration_state_final: str
    tip_pose_status_final: str
    raw_live_tool_ids_final: list[str]
    normalized_live_tool_ids_final: list[str]
    runtime_role_mappings_final: dict[str, str]
    unmapped_live_tool_ids_final: list[str]
    tool_metrics: dict[str, ToolBenchmarkMetrics]
    thresholds: TrackerBenchmarkThresholds
    passed: bool
    failures: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def collect_tracking_snapshots(
    tracking_service,
    *,
    duration_s: float,
    sample_period_s: float = 0.05,
    wait_for_first_frame_s: float = 0.0,
) -> list[tuple[float, TrackingSnapshot]]:
    """Collect timestamped tracking snapshots for a fixed window."""
    poll_period = max(0.01, float(sample_period_s))
    if wait_for_first_frame_s > 0.0:
        deadline = time.monotonic() + max(0.0, float(wait_for_first_frame_s))
        while time.monotonic() < deadline:
            snapshot = tracking_service.get_snapshot()
            frame_key = snapshot.backend_frame_counter if snapshot.backend_frame_counter > 0 else snapshot.last_frame_number
            if frame_key is not None:
                break
            time.sleep(poll_period)

    start = time.monotonic()
    deadline = start + max(0.1, float(duration_s))
    samples: list[tuple[float, TrackingSnapshot]] = []

    while True:
        now = time.monotonic()
        samples.append((max(0.0, now - start), tracking_service.get_snapshot()))
        if now >= deadline:
            break
        time.sleep(poll_period)
    return samples


def compute_tracker_benchmark_report(
    samples: list[tuple[float, TrackingSnapshot]],
    *,
    thresholds: TrackerBenchmarkThresholds,
    required_tool_ids: tuple[str, ...] = ("0A", "0B"),
) -> TrackerBenchmarkReport:
    """Compute tracker performance and correctness metrics from sampled snapshots."""
    if not samples:
        raise ValueError("At least one tracking snapshot is required")

    duration_s = float(samples[-1][0] - samples[0][0]) if len(samples) > 1 else 0.0
    final_snapshot = samples[-1][1]
    tool_ids = set(required_tool_ids)
    for _, snapshot in samples:
        tool_ids.update(snapshot.tools.keys())
    metrics = {tool_id: ToolBenchmarkMetrics(tool_id=tool_id) for tool_id in sorted(tool_ids)}

    unique_frame_samples: list[tuple[float, TrackingSnapshot]] = []
    previous_frame_key = object()
    for offset_s, snapshot in samples:
        frame_key = snapshot.backend_frame_counter if snapshot.backend_frame_counter > 0 else snapshot.last_frame_number
        if frame_key is None:
            continue
        if frame_key != previous_frame_key:
            unique_frame_samples.append((offset_s, snapshot))
            previous_frame_key = frame_key

    stale_values = [float(snapshot.tracker_data_age_s) for _, snapshot in samples if snapshot.tracker_data_age_s is not None]
    stale_sample_count = sum(1 for _, snapshot in samples if snapshot.tracker_data_stale)

    frame_interval_values: list[float] = []
    previous_offset_s: float | None = None
    for offset_s, _snapshot in unique_frame_samples:
        if previous_offset_s is not None:
            frame_interval_values.append(max(0.0, offset_s - previous_offset_s))
        previous_offset_s = offset_s

    consecutive_missing = {tool_id: 0 for tool_id in metrics}
    quality_values = {tool_id: [] for tool_id in metrics}
    first_valid_frame_latency_s: float | None = None
    for offset_s, snapshot in unique_frame_samples:
        frame_has_tracked_tool = False
        for tool_id, tool_metrics in metrics.items():
            tool = snapshot.tools.get(tool_id)
            if tool is None:
                tool_metrics.unknown_frames += 1
                tool_metrics.latest_tracking_state = "unknown"
                consecutive_missing[tool_id] = 0
                continue

            tool_metrics.latest_tracking_state = tool.tracking_state
            if tool.present:
                tool_metrics.present_frames += 1
            if tool.quality is not None:
                quality_values[tool_id].append(float(tool.quality))

            if tool.tracking_state == "tracked":
                tool_metrics.tracked_frames += 1
                consecutive_missing[tool_id] = 0
                frame_has_tracked_tool = True
                if tool_metrics.time_to_first_tracked_frame_s is None:
                    tool_metrics.time_to_first_tracked_frame_s = offset_s
            elif tool.tracking_state == "missing":
                tool_metrics.missing_frames += 1
                consecutive_missing[tool_id] += 1
                tool_metrics.max_consecutive_missing_frames = max(
                    tool_metrics.max_consecutive_missing_frames, consecutive_missing[tool_id]
                )
            elif tool.tracking_state == "invalid":
                tool_metrics.invalid_frames += 1
                if tool_metrics.time_to_first_tracked_frame_s is None:
                    tool_metrics.warmup_invalid_frames += 1
                    if _status_has_nonfinite_values(tool.status):
                        tool_metrics.warmup_nonfinite_invalid_frames += 1
                else:
                    tool_metrics.post_warmup_invalid_frames += 1
                consecutive_missing[tool_id] = 0
            else:
                tool_metrics.unknown_frames += 1
                consecutive_missing[tool_id] = 0
        if frame_has_tracked_tool and first_valid_frame_latency_s is None:
            first_valid_frame_latency_s = offset_s

    for tool_id, tool_metrics in metrics.items():
        tool_metrics.quality = _build_numeric_stats(quality_values[tool_id])

    effective_frame_rate_hz: float | None = None
    if len(unique_frame_samples) >= 2:
        elapsed = unique_frame_samples[-1][0] - unique_frame_samples[0][0]
        if elapsed > 0:
            effective_frame_rate_hz = (len(unique_frame_samples) - 1) / elapsed

    registration_loaded = any(snapshot.registration_state == "loaded" for _, snapshot in samples)
    tip_pose_computable = any(
        snapshot.tip_pose_status == "ok" and snapshot.T_robot_tip is not None for _, snapshot in samples
    )
    first_frame_latency_s = final_snapshot.first_frame_latency_s
    if first_frame_latency_s is None and unique_frame_samples:
        first_frame_latency_s = unique_frame_samples[0][0]

    warmup_invalid_frame_count_by_tool = {
        tool_id: metrics[tool_id].warmup_invalid_frames for tool_id in metrics
    }
    warmup_nonfinite_invalid_frame_count_by_tool = {
        tool_id: metrics[tool_id].warmup_nonfinite_invalid_frames for tool_id in metrics
    }
    warmup_invalid_frame_count = sum(warmup_invalid_frame_count_by_tool.values())
    warmup_nonfinite_invalid_frame_count = sum(warmup_nonfinite_invalid_frame_count_by_tool.values())
    startup_state = _classify_startup_state(
        final_snapshot=final_snapshot,
        unique_frames_observed=len(unique_frame_samples),
        tool_metrics=metrics,
        required_tool_ids=required_tool_ids,
    )

    failures: list[str] = []
    if not unique_frame_samples:
        failures.append("No tracker frames were observed during the benchmark window")
        if final_snapshot.connection_state not in {"tracking", "reconnecting"}:
            failures.append(f"Final connection state remained {final_snapshot.connection_state}")
        if final_snapshot.raw_live_tool_ids:
            failures.append(
                "Live tools were detected but no advancing backend frame counter was observed in benchmark samples"
            )
    elif effective_frame_rate_hz is None or effective_frame_rate_hz < thresholds.min_effective_fps:
        failures.append(
            f"Effective frame rate {effective_frame_rate_hz or 0.0:.2f} Hz is below "
            f"minimum {thresholds.min_effective_fps:.2f} Hz"
        )

    max_data_age_s = max(stale_values) if stale_values else None
    if max_data_age_s is not None and max_data_age_s > thresholds.max_stale_interval_s:
        failures.append(
            f"Tracker data age {max_data_age_s:.3f}s exceeds "
            f"maximum {thresholds.max_stale_interval_s:.3f}s"
        )

    for tool_id in required_tool_ids:
        tool_metrics = metrics[tool_id]
        if tool_metrics.tracked_frames <= 0:
            failures.append(f"Required tool {tool_id} never reached tracked state")
            if final_snapshot.raw_live_tool_ids and not final_snapshot.runtime_role_mappings.get(tool_id):
                failures.append(
                    f"Required tool {tool_id} is not mapped from live ids {final_snapshot.raw_live_tool_ids}"
                )
        if tool_metrics.max_consecutive_missing_frames > thresholds.max_consecutive_missing_frames:
            failures.append(
                f"Tool {tool_id} missing streak {tool_metrics.max_consecutive_missing_frames} exceeds "
                f"limit {thresholds.max_consecutive_missing_frames}"
            )
        if thresholds.require_valid_transforms and tool_metrics.post_warmup_invalid_frames > 0:
            failures.append(
                f"Tool {tool_id} reported {tool_metrics.post_warmup_invalid_frames} invalid transform frame(s) after startup"
            )

    return TrackerBenchmarkReport(
        generated_at_utc=utc_now_iso(),
        configured_backend_name=final_snapshot.configured_backend_name,
        selected_backend_name=final_snapshot.selected_backend_name,
        backend_identity=final_snapshot.backend_identity,
        port=final_snapshot.port,
        duration_s=duration_s,
        samples_observed=len(samples),
        unique_frames_observed=len(unique_frame_samples),
        backend_frame_counter_final=final_snapshot.backend_frame_counter,
        frame_interval_s=_build_numeric_stats(frame_interval_values),
        effective_frame_rate_hz=effective_frame_rate_hz,
        stale_sample_count=stale_sample_count,
        stale_sample_ratio=(stale_sample_count / len(samples)) if samples else 0.0,
        max_data_age_s=max_data_age_s,
        first_frame_latency_s=first_frame_latency_s,
        first_valid_frame_latency_s=first_valid_frame_latency_s,
        startup_state=startup_state,
        warmup_invalid_frame_count=warmup_invalid_frame_count,
        warmup_nonfinite_invalid_frame_count=warmup_nonfinite_invalid_frame_count,
        warmup_invalid_frame_count_by_tool=warmup_invalid_frame_count_by_tool,
        warmup_nonfinite_invalid_frame_count_by_tool=warmup_nonfinite_invalid_frame_count_by_tool,
        registration_loaded=registration_loaded,
        tip_pose_computable=tip_pose_computable,
        canonical_state_final=final_snapshot.canonical_state,
        final_connection_state=final_snapshot.connection_state,
        final_last_error=final_snapshot.last_error,
        warning_messages_final=list(final_snapshot.warning_messages),
        error_messages_final=list(final_snapshot.error_messages),
        registration_state_final=final_snapshot.registration_state,
        tip_pose_status_final=final_snapshot.tip_pose_status,
        raw_live_tool_ids_final=list(final_snapshot.raw_live_tool_ids),
        normalized_live_tool_ids_final=list(final_snapshot.normalized_live_tool_ids),
        runtime_role_mappings_final=dict(final_snapshot.runtime_role_mappings),
        unmapped_live_tool_ids_final=list(final_snapshot.unmapped_live_tool_ids),
        tool_metrics=metrics,
        thresholds=thresholds,
        passed=not failures,
        failures=failures,
    )


def _build_numeric_stats(values: list[float]) -> NumericStats:
    finite_values: list[float] = []
    discarded_non_finite = 0
    for value in values:
        numeric = float(value)
        if math.isfinite(numeric):
            finite_values.append(numeric)
        else:
            discarded_non_finite += 1
    if not finite_values:
        return NumericStats(discarded_non_finite=discarded_non_finite)
    if len(finite_values) == 1:
        return NumericStats(
            count=1,
            minimum=float(finite_values[0]),
            maximum=float(finite_values[0]),
            mean=float(finite_values[0]),
            stdev=0.0,
            discarded_non_finite=discarded_non_finite,
        )
    return NumericStats(
        count=len(finite_values),
        minimum=float(min(finite_values)),
        maximum=float(max(finite_values)),
        mean=float(statistics.fmean(finite_values)),
        stdev=float(statistics.pstdev(finite_values)),
        discarded_non_finite=discarded_non_finite,
    )


def _status_has_nonfinite_values(status: str | None) -> bool:
    normalized = str(status or "").strip().lower()
    return "non-finite" in normalized or "nonfinite" in normalized


def _classify_startup_state(
    *,
    final_snapshot: TrackingSnapshot,
    unique_frames_observed: int,
    tool_metrics: dict[str, ToolBenchmarkMetrics],
    required_tool_ids: tuple[str, ...],
) -> str:
    backend_connected = bool(final_snapshot.backend_connected) or final_snapshot.connection_state in {"tracking", "connected"}
    if not backend_connected and final_snapshot.canonical_state not in {"mock", "streaming_healthy", "streaming_degraded"}:
        return "no_serial_connection"
    if unique_frames_observed <= 0:
        return "serial_connected_no_frames"
    if any((tool_metrics.get(tool_id) and tool_metrics[tool_id].tracked_frames > 0) for tool_id in required_tool_ids):
        return "valid_tracked_frames"
    if any(
        (tool_metrics.get(tool_id) and tool_metrics[tool_id].warmup_nonfinite_invalid_frames > 0)
        for tool_id in required_tool_ids
    ):
        return "frames_arriving_warmup_invalid"
    return "frames_arriving_unclassified"
