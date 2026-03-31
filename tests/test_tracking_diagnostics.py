from continuum_robot.services.models import (
    ServiceHealthSnapshot,
    ToolTrackingSnapshot,
    TrackingSnapshot,
)
from continuum_robot.tracking.benchmarking import TrackerBenchmarkThresholds
from continuum_robot.tracking.diagnostics import (
    build_tracking_diagnostics_report,
    render_tracking_diagnostics_report_lines,
)


def _tool(tool_id: str, *, state: str, frame_number: int | None, timestamp: str) -> ToolTrackingSnapshot:
    return ToolTrackingSnapshot(
        tool_id=tool_id,
        present=state in {"tracked", "invalid"},
        valid=(False if state == "invalid" else None),
        validity_known=(state == "invalid"),
        tracking_state=state,
        status=state,
        frame_number=frame_number,
        last_update_utc=timestamp,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0) if state == "tracked" else None,
        translation_mm=(1.0, 2.0, 3.0) if state == "tracked" else None,
        quality=0.1,
    )


def _snapshot(
    *,
    frame_number: int | None,
    timestamp: str,
    tool_0a_state: str,
    tool_0b_state: str,
    registration_state: str = "missing_registration",
    tip_pose_status: str = "missing_registration",
    tracker_data_age_s: float | None = 0.01,
    tracker_data_stale: bool = False,
    backend_connected: bool = True,
) -> TrackingSnapshot:
    normalized_ids = []
    role_mappings = {}
    if tool_0a_state != "unknown":
        normalized_ids.append("0A")
        role_mappings["0A"] = "10"
    if tool_0b_state != "unknown":
        normalized_ids.append("0B")
        role_mappings["0B"] = "11"
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(name="tracking_service", health="healthy", state="tracking", status="ok"),
        connection_state="tracking",
        canonical_state="streaming_degraded" if tracker_data_stale else "streaming_healthy",
        backend_identity="ndi_tracker_python",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        port="/dev/ttyUSB0",
        backend_running=True,
        backend_connected=backend_connected,
        backend_frame_counter=frame_number or 0,
        last_frame_number=frame_number,
        last_packet_utc=timestamp,
        tracker_data_age_s=tracker_data_age_s,
        tracker_data_stale=tracker_data_stale,
        first_frame_latency_s=0.05 if frame_number is not None else None,
        raw_live_tool_ids=["10", "11"][: len(normalized_ids)],
        normalized_live_tool_ids=normalized_ids,
        runtime_role_mappings=role_mappings,
        backend_startup_messages=["Selected backend ndi."],
        backend_capability_report={
            "ndi": {
                "backend": "ndi",
                "available": True,
                "code": "ok",
                "reason": "ready",
                "details": {"aurora_port": "/dev/ttyUSB0"},
            }
        },
        backend_details={"selected_backend_name": "ndi", "aurora_port": "/dev/ttyUSB0"},
        registration_state=registration_state,
        tip_pose_status=tip_pose_status,
        T_robot_tip=[[1.0, 0.0, 0.0, 7.0], [0.0, 1.0, 0.0, 8.0], [0.0, 0.0, 1.0, 9.0], [0.0, 0.0, 0.0, 1.0]]
        if tip_pose_status == "ok"
        else None,
        tools={
            "0A": _tool("0A", state=tool_0a_state, frame_number=frame_number, timestamp=timestamp),
            "0B": _tool("0B", state=tool_0b_state, frame_number=frame_number, timestamp=timestamp),
        },
    )


class _FakeTrackingService:
    def __init__(self, snapshots: list[TrackingSnapshot], capability_report: dict | None = None) -> None:
        self._snapshots = list(snapshots)
        self._index = 0
        self._capability_report = capability_report or {
            "ndi": {
                "backend": "ndi",
                "available": True,
                "code": "ok",
                "reason": "ready",
                "details": {"aurora_port": "/dev/ttyUSB0"},
            }
        }

    def probe_live_backend_capabilities(self) -> dict:
        return self._capability_report

    def get_snapshot(self) -> TrackingSnapshot:
        snapshot = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return snapshot


def test_tracking_diagnostics_reports_tracker_ready_before_registration() -> None:
    service = _FakeTrackingService(
        [
            _snapshot(frame_number=1, timestamp="2026-01-01T00:00:00.000Z", tool_0a_state="tracked", tool_0b_state="tracked"),
            _snapshot(frame_number=2, timestamp="2026-01-01T00:00:00.050Z", tool_0a_state="tracked", tool_0b_state="tracked"),
            _snapshot(frame_number=3, timestamp="2026-01-01T00:00:00.100Z", tool_0a_state="tracked", tool_0b_state="tracked"),
        ]
    )

    report = build_tracking_diagnostics_report(
        service,
        duration_s=0.05,
        sample_period_s=0.01,
        wait_for_first_frame_s=0.0,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=5.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=2,
            require_valid_transforms=True,
        ),
        required_tool_ids=("0A", "0B"),
    )

    lines = render_tracking_diagnostics_report_lines(report)
    assert report.tracker_ready is True
    assert report.full_pose_pipeline_ready is False
    assert report.registration_loaded is False
    assert report.stage_results[4].status == "pending"
    assert "registration_missing" in report.failure_codes
    assert any("configured_backend=ndi" == line for line in lines)


def test_tracking_diagnostics_classifies_missing_requested_tool() -> None:
    service = _FakeTrackingService(
        [
            _snapshot(frame_number=1, timestamp="2026-01-01T00:00:00.000Z", tool_0a_state="tracked", tool_0b_state="unknown"),
            _snapshot(frame_number=2, timestamp="2026-01-01T00:00:00.050Z", tool_0a_state="tracked", tool_0b_state="unknown"),
        ]
    )

    report = build_tracking_diagnostics_report(
        service,
        duration_s=0.05,
        sample_period_s=0.01,
        wait_for_first_frame_s=0.0,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=5.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=1,
            require_valid_transforms=True,
        ),
        required_tool_ids=("0A", "0B"),
    )

    assert report.tracker_ready is False
    assert "frames_arriving_but_requested_tools_not_present" in report.failure_codes


def test_tracking_diagnostics_classifies_backend_connected_but_no_frames() -> None:
    service = _FakeTrackingService(
        [
            _snapshot(
                frame_number=None,
                timestamp="2026-01-01T00:00:00.000Z",
                tool_0a_state="unknown",
                tool_0b_state="unknown",
                backend_connected=True,
            ),
            _snapshot(
                frame_number=None,
                timestamp="2026-01-01T00:00:00.050Z",
                tool_0a_state="unknown",
                tool_0b_state="unknown",
                backend_connected=True,
            ),
        ]
    )

    report = build_tracking_diagnostics_report(
        service,
        duration_s=0.05,
        sample_period_s=0.01,
        wait_for_first_frame_s=0.0,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=5.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=1,
            require_valid_transforms=True,
        ),
        required_tool_ids=("0A", "0B"),
    )

    assert "tracker_connected_but_no_frames" in report.failure_codes


def test_tracking_diagnostics_stage_4_reports_transform_failure_stage_details() -> None:
    snapshot = _snapshot(
        frame_number=3,
        timestamp="2026-01-01T00:00:00.100Z",
        tool_0a_state="invalid",
        tool_0b_state="invalid",
    )
    snapshot.backend_details = {
        "ndi_transform_debug": {
            "tool_transform_debug": {
                "0A": {
                    "failure_stage": "validation",
                    "parse_mode": "ndarray(4, 4):matrix",
                    "rotation_determinant": 0.87,
                    "invalid_reason": "T_aurora_0A[0:3,0:3] is not orthonormal",
                },
                "0B": {
                    "failure_stage": "conversion",
                    "parse_mode": "ndarray(7,):pose_vector_wxyz_xyz",
                    "invalid_reason": "Quaternion norm is zero",
                },
            }
        }
    }
    service = _FakeTrackingService([snapshot, snapshot])

    report = build_tracking_diagnostics_report(
        service,
        duration_s=0.05,
        sample_period_s=0.01,
        wait_for_first_frame_s=0.0,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=5.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=1,
            require_valid_transforms=True,
        ),
        required_tool_ids=("0A", "0B"),
    )

    stage_4 = report.stage_results[3]
    assert stage_4.status == "failed"
    assert "0A=invalid(validation" in stage_4.message
    assert "0B=invalid(conversion" in stage_4.message
