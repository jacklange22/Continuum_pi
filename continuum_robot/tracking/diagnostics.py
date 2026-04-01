"""Tracking diagnostics, staged validation, and failure classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from continuum_robot.services.models import TrackingSnapshot
from continuum_robot.tracking.benchmarking import (
    NumericStats,
    TrackerBenchmarkReport,
    TrackerBenchmarkThresholds,
    collect_tracking_snapshots,
    compute_tracker_benchmark_report,
)
from continuum_robot.utils.time_utils import utc_now_iso


@dataclass
class TrackingValidationStage:
    """One staged validation result for tracking bring-up."""

    stage: str
    status: str
    message: str


@dataclass
class TrackingDiagnosticsReport:
    """Detailed tracking doctor report."""

    generated_at_utc: str
    configured_backend_name: str
    selected_backend_name: str
    backend_identity: str
    configured_port: str
    tracker_port_source: str
    canonical_state: str
    connection_state: str
    capability_report: dict[str, dict]
    backend_details: dict
    startup_messages: list[str]
    startup_state: str
    unique_frames_observed: int
    backend_frame_counter_final: int
    effective_frame_rate_hz: float | None
    frame_interval_s: NumericStats
    max_data_age_s: float | None
    first_frame_latency_s: float | None
    first_valid_frame_latency_s: float | None
    warmup_invalid_frame_count: int
    warmup_nonfinite_invalid_frame_count: int
    warmup_invalid_frame_count_by_tool: dict[str, int]
    warmup_nonfinite_invalid_frame_count_by_tool: dict[str, int]
    raw_live_tool_ids: list[str]
    normalized_live_tool_ids: list[str]
    runtime_role_mappings: dict[str, str]
    tracker_faults: list[str]
    pipeline_faults: list[str]
    warning_messages: list[str]
    error_messages: list[str]
    registration_state: str
    registration_loaded: bool
    tip_pose_status: str
    tip_pose_computable: bool
    stage_results: list[TrackingValidationStage]
    failure_codes: list[str]
    tracker_ready: bool
    tracker_operational: bool
    tracker_verdict: str
    tracker_verdict_message: str
    full_pose_pipeline_ready: bool
    benchmark_report: TrackerBenchmarkReport

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["benchmark_report"] = self.benchmark_report.to_dict()
        return payload


def build_live_stage_results(
    snapshot: TrackingSnapshot,
    *,
    thresholds: TrackerBenchmarkThresholds,
    required_tool_ids: tuple[str, ...],
) -> list[TrackingValidationStage]:
    """Build live staged status from one current tracking snapshot."""
    observed_tools = set(snapshot.normalized_live_tool_ids) | set(snapshot.tools.keys())
    tracker_connect_ok = snapshot.canonical_state in {"mock", "streaming_healthy", "streaming_degraded"} or bool(
        snapshot.backend_connected
    )
    stage_1 = TrackingValidationStage(
        stage="Stage 1: backend connect",
        status="passed" if tracker_connect_ok else "failed",
        message=(
            f"Backend state is {snapshot.canonical_state}."
            if tracker_connect_ok
            else f"Backend state is {snapshot.canonical_state}; last_error={snapshot.last_error or 'none'}."
        ),
    )
    stage_2 = TrackingValidationStage(
        stage="Stage 2: frames arriving",
        status=(
            "passed"
            if snapshot.unique_frames_observed > 0
            and (
                snapshot.effective_frame_rate_hz is None
                or snapshot.effective_frame_rate_hz >= thresholds.min_effective_fps
            )
            else (
                "warning"
                if snapshot.unique_frames_observed > 0
                else ("pending" if tracker_connect_ok and snapshot.unique_frames_observed <= 0 else "failed")
            )
        ),
        message=(
            f"startup_state={_classify_live_startup_state(snapshot, required_tool_ids)}, "
            f"unique_frames={snapshot.unique_frames_observed}, "
            f"effective_fps={snapshot.effective_frame_rate_hz}."
        ),
    )
    missing_tools = [tool_id for tool_id in required_tool_ids if tool_id not in observed_tools]
    stage_3 = TrackingValidationStage(
        stage="Stage 3: expected tool ids visible",
        status="passed" if not missing_tools else "failed",
        message=(
            f"Observed normalized tools: {sorted(observed_tools)}."
            if not missing_tools
            else f"Missing expected tools: {missing_tools}; observed normalized tools: {sorted(observed_tools)}."
        ),
    )
    transform_ready = (
        not snapshot.tracker_data_stale
        and all(snapshot.tools.get(tool_id) and snapshot.tools[tool_id].tracking_state == "tracked" for tool_id in required_tool_ids)
    )
    if thresholds.require_valid_transforms:
        transform_ready = transform_ready and all(
            snapshot.tools.get(tool_id) is not None and snapshot.tools[tool_id].tracking_state != "invalid"
            for tool_id in required_tool_ids
        )
    live_startup_state = _classify_live_startup_state(snapshot, required_tool_ids)
    stage_4_status = "passed" if transform_ready else ("pending" if live_startup_state != "valid_tracked_frames" else "failed")
    stage_4 = TrackingValidationStage(
        stage="Stage 4: transforms valid and fresh",
        status=stage_4_status,
        message=(
            f"startup_state={live_startup_state}, tracker_data_stale={snapshot.tracker_data_stale}, "
            f"tracker_faults={snapshot.tracker_faults}; "
            f"{_render_transform_debug_summary(snapshot, required_tool_ids)}"
        ),
    )
    if snapshot.registration_state != "loaded":
        stage_5_status = "pending"
        stage_5_message = "Registration not loaded yet. Stage 5 is pending by design."
    elif snapshot.tip_pose_status == "ok" and snapshot.T_robot_tip is not None:
        stage_5_status = "passed"
        stage_5_message = "T_robot_tip is computable."
    else:
        stage_5_status = "failed"
        stage_5_message = f"Registration is loaded but tip pose status is {snapshot.tip_pose_status}."
    stage_5 = TrackingValidationStage(
        stage="Stage 5: T_robot_tip computable",
        status=stage_5_status,
        message=stage_5_message,
    )
    return [stage_1, stage_2, stage_3, stage_4, stage_5]


def build_tracking_diagnostics_report(
    tracking_service,
    *,
    duration_s: float,
    sample_period_s: float,
    wait_for_first_frame_s: float,
    thresholds: TrackerBenchmarkThresholds,
    required_tool_ids: tuple[str, ...],
) -> TrackingDiagnosticsReport:
    """Collect snapshots from TrackingService and build one diagnostics report."""
    capability_report = tracking_service.probe_live_backend_capabilities()
    samples = collect_tracking_snapshots(
        tracking_service,
        duration_s=duration_s,
        sample_period_s=sample_period_s,
        wait_for_first_frame_s=wait_for_first_frame_s,
    )
    benchmark_report = compute_tracker_benchmark_report(
        samples,
        thresholds=thresholds,
        required_tool_ids=required_tool_ids,
    )
    final_snapshot = samples[-1][1]
    stage_results = _build_stage_results_from_benchmark(
        final_snapshot,
        benchmark_report=benchmark_report,
        thresholds=thresholds,
        required_tool_ids=required_tool_ids,
    )
    failure_codes = _classify_failure_codes(
        final_snapshot,
        benchmark_report=benchmark_report,
        capability_report=capability_report,
        required_tool_ids=required_tool_ids,
        thresholds=thresholds,
    )
    tracker_port, tracker_port_source = _resolve_tracker_port(final_snapshot, capability_report)
    tracker_ready = all(stage.status == "passed" for stage in stage_results[:4])
    tracker_operational = _is_tracker_operational(stage_results)
    tracker_verdict, tracker_verdict_message = _derive_tracker_verdict(
        stage_results,
        benchmark_report=benchmark_report,
        thresholds=thresholds,
        tracker_ready=tracker_ready,
        tracker_operational=tracker_operational,
    )
    full_pose_pipeline_ready = tracker_operational and stage_results[4].status == "passed"
    return TrackingDiagnosticsReport(
        generated_at_utc=utc_now_iso(),
        configured_backend_name=final_snapshot.configured_backend_name,
        selected_backend_name=final_snapshot.selected_backend_name,
        backend_identity=final_snapshot.backend_identity,
        configured_port=tracker_port,
        tracker_port_source=tracker_port_source,
        canonical_state=final_snapshot.canonical_state,
        connection_state=final_snapshot.connection_state,
        capability_report=capability_report,
        backend_details=dict(final_snapshot.backend_details),
        startup_messages=list(final_snapshot.backend_startup_messages),
        startup_state=benchmark_report.startup_state,
        unique_frames_observed=benchmark_report.unique_frames_observed,
        backend_frame_counter_final=benchmark_report.backend_frame_counter_final,
        effective_frame_rate_hz=benchmark_report.effective_frame_rate_hz,
        frame_interval_s=benchmark_report.frame_interval_s,
        max_data_age_s=benchmark_report.max_data_age_s,
        first_frame_latency_s=benchmark_report.first_frame_latency_s,
        first_valid_frame_latency_s=benchmark_report.first_valid_frame_latency_s,
        warmup_invalid_frame_count=benchmark_report.warmup_invalid_frame_count,
        warmup_nonfinite_invalid_frame_count=benchmark_report.warmup_nonfinite_invalid_frame_count,
        warmup_invalid_frame_count_by_tool=dict(benchmark_report.warmup_invalid_frame_count_by_tool),
        warmup_nonfinite_invalid_frame_count_by_tool=dict(benchmark_report.warmup_nonfinite_invalid_frame_count_by_tool),
        raw_live_tool_ids=list(benchmark_report.raw_live_tool_ids_final),
        normalized_live_tool_ids=list(benchmark_report.normalized_live_tool_ids_final),
        runtime_role_mappings=dict(benchmark_report.runtime_role_mappings_final),
        tracker_faults=list(final_snapshot.tracker_faults),
        pipeline_faults=list(final_snapshot.pipeline_faults),
        warning_messages=list(final_snapshot.warning_messages),
        error_messages=list(final_snapshot.error_messages),
        registration_state=final_snapshot.registration_state,
        registration_loaded=benchmark_report.registration_loaded,
        tip_pose_status=final_snapshot.tip_pose_status,
        tip_pose_computable=benchmark_report.tip_pose_computable,
        stage_results=stage_results,
        failure_codes=failure_codes,
        tracker_ready=tracker_ready,
        tracker_operational=tracker_operational,
        tracker_verdict=tracker_verdict,
        tracker_verdict_message=tracker_verdict_message,
        full_pose_pipeline_ready=full_pose_pipeline_ready,
        benchmark_report=benchmark_report,
    )


def render_live_tracking_lines(
    snapshot: TrackingSnapshot,
    *,
    thresholds: TrackerBenchmarkThresholds,
    required_tool_ids: tuple[str, ...],
) -> list[str]:
    """Render one compact GUI-friendly tracking diagnostic summary."""
    stages = build_live_stage_results(snapshot, thresholds=thresholds, required_tool_ids=required_tool_ids)
    lines = [
        f"backend_state={snapshot.canonical_state}",
        f"backend_selected={snapshot.selected_backend_name or snapshot.backend_identity}",
        f"backend_configured={snapshot.configured_backend_name}",
        f"fallback_used={snapshot.fallback_used}",
        f"unique_frames={snapshot.unique_frames_observed}",
        f"effective_fps={snapshot.effective_frame_rate_hz}",
        f"raw_live_tool_ids={snapshot.raw_live_tool_ids}",
        f"normalized_live_tool_ids={snapshot.normalized_live_tool_ids}",
        f"role_mappings={snapshot.runtime_role_mappings}",
        f"tracker_faults={snapshot.tracker_faults}",
        f"pipeline_faults={snapshot.pipeline_faults}",
    ]
    for stage in stages:
        lines.append(f"{stage.stage}: {stage.status} | {stage.message}")
    return lines


def render_tracking_diagnostics_report_lines(report: TrackingDiagnosticsReport) -> list[str]:
    """Render a detailed doctor report into plain-text lines."""
    lines = [
        f"configured_backend={report.configured_backend_name}",
        f"selected_backend={report.selected_backend_name}",
        f"backend_identity={report.backend_identity}",
        f"configured_port={report.configured_port or '/dev/mock-aurora'}",
        f"tracker_port_source={report.tracker_port_source}",
        f"canonical_state={report.canonical_state}",
        f"connection_state={report.connection_state}",
        f"startup_state={report.startup_state}",
        f"tracker_ready={report.tracker_ready}",
        f"tracker_operational={report.tracker_operational}",
        f"tracker_verdict={report.tracker_verdict}",
        f"tracker_verdict_message={report.tracker_verdict_message}",
        f"full_pose_pipeline_ready={report.full_pose_pipeline_ready}",
        f"registration_state={report.registration_state}",
        f"registration_loaded={report.registration_loaded}",
        f"tip_pose_status={report.tip_pose_status}",
        f"tip_pose_computable={report.tip_pose_computable}",
        f"unique_frames_observed={report.unique_frames_observed}",
        f"backend_frame_counter_final={report.backend_frame_counter_final}",
        f"effective_frame_rate_hz={report.effective_frame_rate_hz}",
        (
            "frame_interval_s="
            f"count={report.frame_interval_s.count}, "
            f"min={report.frame_interval_s.minimum}, "
            f"max={report.frame_interval_s.maximum}, "
            f"mean={report.frame_interval_s.mean}, "
            f"stdev={report.frame_interval_s.stdev}"
        ),
        f"max_data_age_s={report.max_data_age_s}",
        f"first_frame_latency_s={report.first_frame_latency_s}",
        f"first_valid_frame_latency_s={report.first_valid_frame_latency_s}",
        f"warmup_invalid_frame_count={report.warmup_invalid_frame_count}",
        f"warmup_nonfinite_invalid_frame_count={report.warmup_nonfinite_invalid_frame_count}",
        f"warmup_invalid_frame_count_by_tool={report.warmup_invalid_frame_count_by_tool}",
        f"warmup_nonfinite_invalid_frame_count_by_tool={report.warmup_nonfinite_invalid_frame_count_by_tool}",
        f"raw_live_tool_ids={report.raw_live_tool_ids}",
        f"normalized_live_tool_ids={report.normalized_live_tool_ids}",
        f"runtime_role_mappings={report.runtime_role_mappings}",
        f"tracker_faults={report.tracker_faults}",
        f"pipeline_faults={report.pipeline_faults}",
        f"failure_codes={report.failure_codes}",
    ]
    for tool_id, metrics in sorted(report.benchmark_report.tool_metrics.items()):
        lines.append(
            f"{tool_id}_timing=time_to_first_tracked_s={metrics.time_to_first_tracked_frame_s}, "
            f"warmup_invalid={metrics.warmup_invalid_frames}, "
            f"warmup_nonfinite={metrics.warmup_nonfinite_invalid_frames}, "
            f"post_warmup_invalid={metrics.post_warmup_invalid_frames}"
        )
    if report.backend_details:
        lines.append(f"backend_details={report.backend_details}")
    if report.startup_messages:
        lines.append("startup_messages:")
        lines.extend(f"  - {message}" for message in report.startup_messages)
    if report.capability_report:
        lines.append("capability_report:")
        for backend_name, capability in report.capability_report.items():
            lines.append(
                "  - "
                f"{backend_name}: available={capability.get('available')} "
                f"code={capability.get('code', 'unknown')} "
                f"reason={capability.get('reason')}"
            )
    lines.append("stage_results:")
    lines.extend(f"  - {stage.stage}: {stage.status} | {stage.message}" for stage in report.stage_results)
    return lines


def _build_stage_results_from_benchmark(
    final_snapshot: TrackingSnapshot,
    *,
    benchmark_report: TrackerBenchmarkReport,
    thresholds: TrackerBenchmarkThresholds,
    required_tool_ids: tuple[str, ...],
) -> list[TrackingValidationStage]:
    observed_tools = set(benchmark_report.normalized_live_tool_ids_final)
    stage_results: list[TrackingValidationStage] = []
    stage_results.append(
        TrackingValidationStage(
            stage="Stage 1: backend connect",
            status=(
                "passed"
                if final_snapshot.canonical_state in {"mock", "streaming_healthy", "streaming_degraded"}
                or bool(final_snapshot.backend_connected)
                else "failed"
            ),
            message=f"canonical_state={final_snapshot.canonical_state}, connection_state={final_snapshot.connection_state}",
        )
    )
    stage_results.append(
        TrackingValidationStage(
            stage="Stage 2: frames arriving",
            status=(
                "passed"
                if benchmark_report.unique_frames_observed > 0
                and (
                    benchmark_report.effective_frame_rate_hz is None
                    or benchmark_report.effective_frame_rate_hz >= thresholds.min_effective_fps
                )
                else (
                    "warning"
                    if benchmark_report.unique_frames_observed > 0
                    else ("pending" if benchmark_report.startup_state == "serial_connected_no_frames" else "failed")
                )
            ),
            message=(
                f"startup_state={benchmark_report.startup_state}, "
                f"unique_frames={benchmark_report.unique_frames_observed}, "
                f"effective_fps={benchmark_report.effective_frame_rate_hz}, "
                f"first_frame_latency_s={benchmark_report.first_frame_latency_s}, "
                f"first_valid_frame_latency_s={benchmark_report.first_valid_frame_latency_s}"
            ),
        )
    )
    missing_tools = [tool_id for tool_id in required_tool_ids if tool_id not in observed_tools]
    stage_results.append(
        TrackingValidationStage(
            stage="Stage 3: expected tool ids visible",
            status="passed" if not missing_tools else "failed",
            message=(
                f"Observed normalized tools: {sorted(observed_tools)}."
                if not missing_tools
                else f"Missing expected tools: {missing_tools}; observed normalized tools: {sorted(observed_tools)}."
            ),
        )
    )
    stage_4_failed = False
    stage_4_pending = False
    if benchmark_report.max_data_age_s is not None and benchmark_report.max_data_age_s > thresholds.max_stale_interval_s:
        stage_4_failed = True
    for tool_id in required_tool_ids:
        metrics = benchmark_report.tool_metrics.get(tool_id)
        if metrics is None or metrics.tracked_frames <= 0:
            if metrics is not None and metrics.warmup_nonfinite_invalid_frames > 0:
                stage_4_pending = True
            else:
                stage_4_failed = True
            break
        if thresholds.require_valid_transforms and metrics.post_warmup_invalid_frames > 0:
            stage_4_failed = True
            break
    stage_results.append(
        TrackingValidationStage(
            stage="Stage 4: transforms valid and fresh",
            status="failed" if stage_4_failed else ("pending" if stage_4_pending else "passed"),
            message=(
                f"startup_state={benchmark_report.startup_state}, "
                f"max_data_age_s={benchmark_report.max_data_age_s}, "
                f"warmup_invalid_frames={benchmark_report.warmup_invalid_frame_count_by_tool}, "
                f"tracker_faults={final_snapshot.tracker_faults}; "
                f"{_render_transform_debug_summary(final_snapshot, required_tool_ids)}"
            ),
        )
    )
    if not benchmark_report.registration_loaded:
        stage_5_status = "pending"
        stage_5_message = "Registration not loaded yet. Stage 5 is pending by design."
    elif benchmark_report.tip_pose_computable:
        stage_5_status = "passed"
        stage_5_message = "T_robot_tip was computed during the diagnostics window."
    else:
        stage_5_status = "failed"
        stage_5_message = f"Registration loaded but tip_pose_status={final_snapshot.tip_pose_status}."
    stage_results.append(
        TrackingValidationStage(
            stage="Stage 5: T_robot_tip computable",
            status=stage_5_status,
            message=stage_5_message,
        )
    )
    return stage_results


def _render_transform_debug_summary(snapshot: TrackingSnapshot, required_tool_ids: tuple[str, ...]) -> str:
    debug_root = dict(snapshot.backend_details or {}).get("ndi_transform_debug", {})
    debug_tools = dict(debug_root.get("tool_transform_debug", {}) or {})
    parts: list[str] = []
    for tool_id in required_tool_ids:
        tool = snapshot.tools.get(tool_id)
        if tool is None:
            parts.append(f"{tool_id}=missing(no tool snapshot)")
            continue
        if tool.tracking_state == "tracked":
            parts.append(f"{tool_id}=tracked")
            continue
        debug_entry = dict(debug_tools.get(tool_id, {}) or {})
        failure_stage = debug_entry.get("failure_stage") or "unknown"
        payload_kind = debug_entry.get("parse_mode") or debug_entry.get("raw_payload_summary") or "unknown_payload"
        det = debug_entry.get("rotation_determinant")
        det_text = f", det={det:.6f}" if isinstance(det, (int, float)) else ""
        reason = str(debug_entry.get("invalid_reason") or tool.status or "unknown_reason")
        parts.append(
            f"{tool_id}={tool.tracking_state}({failure_stage}, payload={payload_kind}{det_text}): {reason}"
        )
    return "; ".join(parts)


def _classify_live_startup_state(snapshot: TrackingSnapshot, required_tool_ids: tuple[str, ...]) -> str:
    tracker_connect_ok = snapshot.canonical_state in {"mock", "streaming_healthy", "streaming_degraded"} or bool(
        snapshot.backend_connected
    )
    if not tracker_connect_ok:
        return "no_serial_connection"
    if snapshot.unique_frames_observed <= 0:
        return "serial_connected_no_frames"
    if all(snapshot.tools.get(tool_id) and snapshot.tools[tool_id].tracking_state == "tracked" for tool_id in required_tool_ids):
        return "valid_tracked_frames"
    if any(_tool_has_warmup_nonfinite_status(snapshot.tools.get(tool_id)) for tool_id in required_tool_ids):
        return "frames_arriving_warmup_invalid"
    return "frames_arriving_unclassified"


def _tool_has_warmup_nonfinite_status(tool) -> bool:
    if tool is None or tool.tracking_state != "invalid":
        return False
    normalized = str(tool.status or "").strip().lower()
    return "non-finite" in normalized or "nonfinite" in normalized


def _classify_failure_codes(
    final_snapshot: TrackingSnapshot,
    *,
    benchmark_report: TrackerBenchmarkReport,
    capability_report: dict[str, dict],
    required_tool_ids: tuple[str, ...],
    thresholds: TrackerBenchmarkThresholds,
) -> list[str]:
    failure_codes: list[str] = []
    for backend_name, capability in capability_report.items():
        if final_snapshot.selected_backend_name in {"mock", "disabled"} and backend_name != final_snapshot.selected_backend_name:
            continue
        code = str(capability.get("code", "")).strip().lower()
        if code and code not in {"ok", "disabled"} and code not in failure_codes:
            failure_codes.append(code)
    if final_snapshot.backend_connected and benchmark_report.unique_frames_observed == 0:
        failure_codes.append("tracker_connected_but_no_frames")
    observed_tools = set(benchmark_report.normalized_live_tool_ids_final)
    if benchmark_report.unique_frames_observed > 0 and any(tool_id not in observed_tools for tool_id in required_tool_ids):
        failure_codes.append("frames_arriving_but_requested_tools_not_present")
    if benchmark_report.max_data_age_s is not None and benchmark_report.max_data_age_s > thresholds.max_stale_interval_s:
        failure_codes.append("stale_frames")
    if (
        benchmark_report.unique_frames_observed > 0
        and benchmark_report.effective_frame_rate_hz is not None
        and benchmark_report.effective_frame_rate_hz < thresholds.min_effective_fps
    ):
        failure_codes.append("fps_below_target")
    invalid_only = True
    for tool_id in required_tool_ids:
        metrics = benchmark_report.tool_metrics.get(tool_id)
        if metrics is None:
            continue
        if metrics.tracked_frames > 0:
            invalid_only = False
            break
        if metrics.invalid_frames <= 0:
            invalid_only = False
            break
    if invalid_only and required_tool_ids:
        if benchmark_report.warmup_nonfinite_invalid_frame_count > 0 and benchmark_report.first_valid_frame_latency_s is None:
            failure_codes.append("warmup_invalid_transforms_only")
        else:
            failure_codes.append("invalid_transforms_only")
    if benchmark_report.warmup_nonfinite_invalid_frame_count > 0:
        failure_codes.append("warmup_nonfinite_payloads_seen")
    if not benchmark_report.registration_loaded:
        failure_codes.append("registration_missing")
    elif not benchmark_report.tip_pose_computable:
        failure_codes.append("transform_chain_incomplete")
    return failure_codes


def _is_tracker_operational(stage_results: list[TrackingValidationStage]) -> bool:
    stage_by_name = {stage.stage: stage for stage in stage_results}
    stage_1 = stage_by_name.get("Stage 1: backend connect")
    stage_2 = stage_by_name.get("Stage 2: frames arriving")
    stage_3 = stage_by_name.get("Stage 3: expected tool ids visible")
    stage_4 = stage_by_name.get("Stage 4: transforms valid and fresh")
    return bool(
        stage_1 is not None
        and stage_1.status == "passed"
        and stage_2 is not None
        and stage_2.status in {"passed", "warning"}
        and stage_3 is not None
        and stage_3.status == "passed"
        and stage_4 is not None
        and stage_4.status == "passed"
    )


def _derive_tracker_verdict(
    stage_results: list[TrackingValidationStage],
    *,
    benchmark_report: TrackerBenchmarkReport,
    thresholds: TrackerBenchmarkThresholds,
    tracker_ready: bool,
    tracker_operational: bool,
) -> tuple[str, str]:
    if tracker_ready:
        return "passed", "All strict tracker validation thresholds passed."
    if tracker_operational:
        fps = benchmark_report.effective_frame_rate_hz
        if fps is not None and fps < thresholds.min_effective_fps:
            return (
                "operational_with_warning",
                f"Operational with warning: effective FPS {fps:.2f} is below target {thresholds.min_effective_fps:.2f}, "
                "but frames, tool visibility, and rigid transforms are usable.",
            )
        return (
            "operational_with_warning",
            "Operational with warning: tracker data is usable, but one or more strict validation targets were missed.",
        )
    first_failure = next((stage for stage in stage_results if stage.status == "failed"), None)
    if first_failure is not None:
        return "failed", f"Tracker not operational: {first_failure.stage} failed. {first_failure.message}"
    first_pending = next((stage for stage in stage_results if stage.status == "pending"), None)
    if first_pending is not None:
        return "failed", f"Tracker not operational: {first_pending.stage} is still pending. {first_pending.message}"
    return "failed", "Tracker not operational."


def _resolve_tracker_port(snapshot: TrackingSnapshot, capability_report: dict[str, dict]) -> tuple[str, str]:
    backend_details = dict(snapshot.backend_details or {})
    tracker_settings = dict(backend_details.get("tracker_settings", {}) or {})
    selected_backend = str(snapshot.selected_backend_name or snapshot.configured_backend_name or "").strip()
    capability_details = {}
    if selected_backend:
        capability = capability_report.get(selected_backend, {})
        if isinstance(capability, dict):
            capability_details = dict(capability.get("details", {}) or {})
    candidates = [
        ("backend_details.aurora_port", backend_details.get("aurora_port")),
        ("backend_details.tracker_settings.serial port", tracker_settings.get("serial port")),
        ("capability_report.details.aurora_port", capability_details.get("aurora_port")),
        ("tracking_snapshot.port", snapshot.port),
    ]
    for source, candidate in candidates:
        if candidate not in (None, ""):
            return str(candidate), source
    return "", "unknown"
