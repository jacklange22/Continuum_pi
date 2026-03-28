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
    canonical_state: str
    connection_state: str
    capability_report: dict[str, dict]
    backend_details: dict
    startup_messages: list[str]
    unique_frames_observed: int
    backend_frame_counter_final: int
    effective_frame_rate_hz: float | None
    frame_interval_s: NumericStats
    max_data_age_s: float | None
    first_frame_latency_s: float | None
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
            else "failed"
        ),
        message=(
            f"unique_frames={snapshot.unique_frames_observed}, effective_fps={snapshot.effective_frame_rate_hz}."
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
    stage_4 = TrackingValidationStage(
        stage="Stage 4: transforms valid and fresh",
        status="passed" if transform_ready else "failed",
        message=(
            f"tracker_data_stale={snapshot.tracker_data_stale}, tracker_faults={snapshot.tracker_faults}."
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
    tracker_ready = all(stage.status == "passed" for stage in stage_results[:4])
    full_pose_pipeline_ready = tracker_ready and stage_results[4].status == "passed"
    return TrackingDiagnosticsReport(
        generated_at_utc=utc_now_iso(),
        configured_backend_name=final_snapshot.configured_backend_name,
        selected_backend_name=final_snapshot.selected_backend_name,
        backend_identity=final_snapshot.backend_identity,
        configured_port=final_snapshot.port,
        canonical_state=final_snapshot.canonical_state,
        connection_state=final_snapshot.connection_state,
        capability_report=capability_report,
        backend_details=dict(final_snapshot.backend_details),
        startup_messages=list(final_snapshot.backend_startup_messages),
        unique_frames_observed=benchmark_report.unique_frames_observed,
        backend_frame_counter_final=benchmark_report.backend_frame_counter_final,
        effective_frame_rate_hz=benchmark_report.effective_frame_rate_hz,
        frame_interval_s=benchmark_report.frame_interval_s,
        max_data_age_s=benchmark_report.max_data_age_s,
        first_frame_latency_s=benchmark_report.first_frame_latency_s,
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
        f"canonical_state={report.canonical_state}",
        f"connection_state={report.connection_state}",
        f"tracker_ready={report.tracker_ready}",
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
        f"raw_live_tool_ids={report.raw_live_tool_ids}",
        f"normalized_live_tool_ids={report.normalized_live_tool_ids}",
        f"runtime_role_mappings={report.runtime_role_mappings}",
        f"tracker_faults={report.tracker_faults}",
        f"pipeline_faults={report.pipeline_faults}",
        f"failure_codes={report.failure_codes}",
    ]
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
                else "failed"
            ),
            message=(
                f"unique_frames={benchmark_report.unique_frames_observed}, "
                f"effective_fps={benchmark_report.effective_frame_rate_hz}, "
                f"first_frame_latency_s={benchmark_report.first_frame_latency_s}"
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
    if benchmark_report.max_data_age_s is not None and benchmark_report.max_data_age_s > thresholds.max_stale_interval_s:
        stage_4_failed = True
    for tool_id in required_tool_ids:
        metrics = benchmark_report.tool_metrics.get(tool_id)
        if metrics is None or metrics.tracked_frames <= 0:
            stage_4_failed = True
            break
        if thresholds.require_valid_transforms and metrics.invalid_frames > 0:
            stage_4_failed = True
            break
    stage_results.append(
        TrackingValidationStage(
            stage="Stage 4: transforms valid and fresh",
            status="failed" if stage_4_failed else "passed",
            message=(
                f"max_data_age_s={benchmark_report.max_data_age_s}, tracker_faults={final_snapshot.tracker_faults}."
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
        failure_codes.append("invalid_transforms_only")
    if not benchmark_report.registration_loaded:
        failure_codes.append("registration_missing")
    elif not benchmark_report.tip_pose_computable:
        failure_codes.append("transform_chain_incomplete")
    return failure_codes
