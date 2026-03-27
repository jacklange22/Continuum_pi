from continuum_robot.services.models import (
    ServiceHealthSnapshot,
    ToolTrackingSnapshot,
    TrackingSnapshot,
)
from continuum_robot.tracking.benchmarking import (
    TrackerBenchmarkThresholds,
    compute_tracker_benchmark_report,
)


def _snapshot(
    *,
    frame_number: int | None,
    timestamp: str,
    tool_0a_state: str,
    tool_0b_state: str,
    backend_frame_counter: int = 0,
    tracker_data_age_s: float | None = 0.01,
    tracker_data_stale: bool = False,
    tip_pose_status: str = "missing_registration",
    registration_state: str = "missing_registration",
) -> TrackingSnapshot:
    tools = {
        "0A": ToolTrackingSnapshot(
            tool_id="0A",
            present=tool_0a_state in {"tracked", "invalid"},
            valid=(False if tool_0a_state == "invalid" else None),
            validity_known=(tool_0a_state == "invalid"),
            tracking_state=tool_0a_state,
            status=tool_0a_state,
            frame_number=frame_number,
            last_update_utc=timestamp,
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0) if tool_0a_state == "tracked" else None,
            translation_mm=(1.0, 2.0, 3.0) if tool_0a_state == "tracked" else None,
            quality=0.15,
        ),
        "0B": ToolTrackingSnapshot(
            tool_id="0B",
            present=tool_0b_state in {"tracked", "invalid"},
            valid=(False if tool_0b_state == "invalid" else None),
            validity_known=(tool_0b_state == "invalid"),
            tracking_state=tool_0b_state,
            status=tool_0b_state,
            frame_number=frame_number,
            last_update_utc=timestamp,
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0) if tool_0b_state == "tracked" else None,
            translation_mm=(4.0, 5.0, 6.0) if tool_0b_state == "tracked" else None,
            quality=0.11,
        ),
    }
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(
            name="tracking_service",
            health="healthy",
            state="tracking",
            status="ok",
        ),
        connection_state="tracking",
        backend_identity="ndi_tracker_python",
        port="/dev/ttyUSB0",
        baudrate=115200,
        backend_running=True,
        backend_connected=True,
        backend_frame_counter=backend_frame_counter,
        last_frame_number=frame_number,
        last_packet_utc=timestamp,
        tracker_data_age_s=tracker_data_age_s,
        tracker_data_stale=tracker_data_stale,
        first_frame_latency_s=0.05 if frame_number is not None else None,
        registration_state=registration_state,
        tip_pose_status=tip_pose_status,
        T_robot_tip=[[1.0, 0.0, 0.0, 7.0], [0.0, 1.0, 0.0, 8.0], [0.0, 0.0, 1.0, 9.0], [0.0, 0.0, 0.0, 1.0]]
        if tip_pose_status == "ok"
        else None,
        tools=tools,
    )


def test_tracker_benchmark_report_passes_on_good_stream() -> None:
    samples = [
        (0.00, _snapshot(frame_number=1, backend_frame_counter=1, timestamp="2026-01-01T00:00:00.000Z", tool_0a_state="tracked", tool_0b_state="tracked")),
        (0.05, _snapshot(frame_number=2, backend_frame_counter=2, timestamp="2026-01-01T00:00:00.050Z", tool_0a_state="tracked", tool_0b_state="tracked", registration_state="loaded", tip_pose_status="ok")),
        (0.10, _snapshot(frame_number=3, backend_frame_counter=3, timestamp="2026-01-01T00:00:00.100Z", tool_0a_state="tracked", tool_0b_state="tracked", registration_state="loaded", tip_pose_status="ok")),
    ]

    report = compute_tracker_benchmark_report(
        samples,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=10.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=2,
            require_valid_transforms=True,
        ),
    )

    assert report.passed is True
    assert report.unique_frames_observed == 3
    assert report.effective_frame_rate_hz is not None
    assert report.registration_loaded is True
    assert report.tip_pose_computable is True
    assert report.tool_metrics["0A"].tracked_frames == 3
    assert report.tool_metrics["0B"].tracked_frames == 3


def test_tracker_benchmark_report_fails_on_missing_and_invalid_data() -> None:
    samples = [
        (0.00, _snapshot(frame_number=1, backend_frame_counter=1, timestamp="2026-01-01T00:00:00.000Z", tool_0a_state="missing", tool_0b_state="tracked", tracker_data_age_s=0.30, tracker_data_stale=True)),
        (0.20, _snapshot(frame_number=2, backend_frame_counter=2, timestamp="2026-01-01T00:00:00.200Z", tool_0a_state="missing", tool_0b_state="invalid", tracker_data_age_s=0.31, tracker_data_stale=True)),
        (0.40, _snapshot(frame_number=3, backend_frame_counter=3, timestamp="2026-01-01T00:00:00.400Z", tool_0a_state="missing", tool_0b_state="tracked", tracker_data_age_s=0.29, tracker_data_stale=True)),
    ]

    report = compute_tracker_benchmark_report(
        samples,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=20.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=2,
            require_valid_transforms=True,
        ),
    )

    assert report.passed is False
    assert any("below minimum" in item for item in report.failures)
    assert any("exceeds maximum" in item for item in report.failures)
    assert any("Required tool 0A never reached tracked state" in item for item in report.failures)
    assert any("Tool 0A missing streak" in item for item in report.failures)
    assert any("Tool 0B reported 1 invalid transform" in item for item in report.failures)


def test_tracker_benchmark_uses_backend_frame_counter_when_device_frame_number_missing() -> None:
    samples = [
        (0.00, _snapshot(frame_number=None, backend_frame_counter=1, timestamp="2026-01-01T00:00:00.000Z", tool_0a_state="tracked", tool_0b_state="tracked")),
        (0.05, _snapshot(frame_number=None, backend_frame_counter=2, timestamp="2026-01-01T00:00:00.050Z", tool_0a_state="tracked", tool_0b_state="tracked")),
        (0.10, _snapshot(frame_number=None, backend_frame_counter=3, timestamp="2026-01-01T00:00:00.100Z", tool_0a_state="tracked", tool_0b_state="tracked")),
    ]

    report = compute_tracker_benchmark_report(
        samples,
        thresholds=TrackerBenchmarkThresholds(
            min_effective_fps=10.0,
            max_stale_interval_s=0.25,
            max_consecutive_missing_frames=2,
            require_valid_transforms=True,
        ),
    )

    assert report.passed is True
    assert report.unique_frames_observed == 3
    assert report.backend_frame_counter_final == 3
