from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time

import numpy as np
import pytest
from unittest.mock import patch

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RobotSegmentConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.experiments.dataset_io import ExperimentDatasetLoader, ExperimentDatasetWriter
from continuum_robot.experiments import dataset_io
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.experiments.framework import BaseExperiment, ExperimentHardwareRequirements, ExperimentSession
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.schedules import CommandScheduleConfig, command_schedule_checksum, generate_command_schedule
from continuum_robot.experiments.schemas import ExperimentMetadata, ExperimentSummary, ExperimentTimeseriesSample
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.services.models import HEALTH_HEALTHY, ServiceHealthSnapshot, ToolTrackingSnapshot, TrackingSnapshot
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService, ServoTelemetryRetryError
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(mode="4-servo", spool_diameter_cm=1.2, ticks_per_revolution=4096, servo_ids=[1, 2, 3, 4]),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _parallel_single_settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=False, poll_rate_hz=20, robot_config="robot_8servo.yaml"),
        robot=RobotConfig(
            mode="parallel_single",
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
            active_segment="segment_a",
            segments={
                "segment_a": RobotSegmentConfig(
                    key="segment_a",
                    label="Spine 1",
                    segment_label="Segment A",
                    segment_role="proximal",
                    segment_order_index=0,
                    servo_ids=[1, 2, 3, 4],
                    pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
                ),
                "segment_b": RobotSegmentConfig(
                    key="segment_b",
                    label="Spine 2",
                    segment_label="Segment B",
                    segment_role="distal",
                    segment_order_index=1,
                    servo_ids=[5, 6, 7, 8],
                    pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
                ),
            },
        ),
        serial=SerialConfig(aurora_port="/dev/mock-aurora", openrb_port="/dev/mock-openrb", baudrate=57600),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=120,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _servo_service(tmp_path: Path, *, dxl_bus: MockDxlBus | None = None) -> ServoService:
    return ServoService(
        dxl_bus=dxl_bus or MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )


def _poison_last_known_positions_for_gui_stale_cache(service: ServoService, servo_ids: list[int]) -> None:
    with service._bus_state_lock:
        for servo_id in servo_ids:
            old = service._last_telemetry_by_id.get(int(servo_id))
            if old is None:
                continue
            service._last_telemetry_by_id[int(servo_id)] = replace(
                old,
                present_position=None,
                telemetry_error="mock_stale_gui_cache_missing_position",
            )


class _CollectPosePacketErrorBus(MockDxlBus):
    def __init__(self, *, failed_servo_id: int = 3, failures: int = 3) -> None:
        super().__init__([1, 2, 3, 4])
        self.failed_servo_id = int(failed_servo_id)
        self.failures_remaining = int(failures)
        self.goal_written = False

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        self.goal_written = True

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        result = super().read_telemetry(servo_ids, **kwargs)
        if self.goal_written and self.failed_servo_id in [int(value) for value in servo_ids] and self.failures_remaining > 0:
            self.failures_remaining -= 1
            telemetry = result[self.failed_servo_id]
            telemetry.present_position = None
            telemetry.telemetry_error = "[TxRxResult] Incorrect status packet!"
            telemetry.hardware_error = "[TxRxResult] Incorrect status packet!"
        return result


class _PostMotionCorruptBudgetBus(MockDxlBus):
    """Corrupt ``read_live_telemetry`` for a servo with a bounded budget.

    Arming options (mutually exclusive intent):

    - ``corrupt_after_goal_writes``: after this many completed ``write_goal_positions`` calls on the bus,
      corrupt live telemetry reads (stable for collect-pose: skips precheck-only live reads).
    - Otherwise ``arm_after_live_reads``: corrupt only after this many ``read_live_telemetry`` events.
    """

    def __init__(
        self,
        *,
        failed_servo_id: int = 3,
        corrupt_budget: int = 40,
        arm_after_live_reads: int = 48,
        corrupt_after_goal_writes: int | None = None,
    ) -> None:
        super().__init__([1, 2, 3, 4])
        self.failed_servo_id = int(failed_servo_id)
        self.corrupt_budget = int(corrupt_budget)
        self.arm_after_live_reads = int(arm_after_live_reads)
        self.corrupt_after_goal_writes = (
            int(corrupt_after_goal_writes) if corrupt_after_goal_writes is not None else None
        )
        self._live_read_events = 0
        self._goal_writes_seen = 0

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        self._goal_writes_seen += 1

    def read_live_telemetry(self, servo_ids: list[int]):
        result = super().read_live_telemetry(servo_ids)
        self._live_read_events += 1
        goal_armed = (
            self.corrupt_after_goal_writes is not None
            and self._goal_writes_seen >= int(self.corrupt_after_goal_writes)
        )
        read_armed = self.corrupt_after_goal_writes is None and self._live_read_events >= self.arm_after_live_reads
        if (not goal_armed and not read_armed) or self.corrupt_budget <= 0 or self.failed_servo_id not in result:
            return result
        self.corrupt_budget -= 1
        telemetry = result[self.failed_servo_id]
        telemetry.present_position = None
        telemetry.telemetry_error = "[TxRxResult] Incorrect status packet!"
        telemetry.hardware_error = "[TxRxResult] Incorrect status packet!"
        return result


class _PostMotionAndResyncCorruptBudgetBus(_PostMotionCorruptBudgetBus):
    """Corrupt both post-motion live reads and recovery resync reads, but keep minimal pre-motion reads clean."""

    def read_minimal_telemetry(self, servo_ids: list[int]):
        return MockDxlBus.read_minimal_telemetry(self, servo_ids)

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        result = super().read_telemetry(servo_ids, **kwargs)
        goal_armed = (
            self.corrupt_after_goal_writes is not None
            and self._goal_writes_seen >= int(self.corrupt_after_goal_writes)
        )
        read_armed = self.corrupt_after_goal_writes is None and self._live_read_events >= self.arm_after_live_reads
        if (not goal_armed and not read_armed) or self.corrupt_budget <= 0 or self.failed_servo_id not in result:
            return result
        self.corrupt_budget -= 1
        telemetry = result[self.failed_servo_id]
        telemetry.present_position = None
        telemetry.telemetry_error = "[TxRxResult] Incorrect status packet!"
        telemetry.hardware_error = "[TxRxResult] Incorrect status packet!"
        return result


class _PreMotionMinimalCorruptBudgetBus(MockDxlBus):
    """Corrupt minimal pre-motion telemetry reads with a bounded budget."""

    def __init__(self, *, failed_servo_id: int = 4, corrupt_budget: int = 1, arm_after_minimal_reads: int = 1) -> None:
        super().__init__([1, 2, 3, 4])
        self.failed_servo_id = int(failed_servo_id)
        self.corrupt_budget = int(corrupt_budget)
        self.arm_after_minimal_reads = int(arm_after_minimal_reads)
        self._minimal_reads = 0

    def read_minimal_telemetry(self, servo_ids: list[int]):
        result = super().read_minimal_telemetry(servo_ids)
        self._minimal_reads += 1
        if (
            self._minimal_reads < self.arm_after_minimal_reads
            or self.corrupt_budget <= 0
            or self.failed_servo_id not in result
        ):
            return result
        self.corrupt_budget -= 1
        telemetry = result[self.failed_servo_id]
        telemetry.present_position = None
        telemetry.telemetry_error = "[TxRxResult] Incorrect status packet!"
        telemetry.telemetry_error_code = "packet_or_status_error"
        telemetry.hardware_error = "[TxRxResult] Incorrect status packet!"
        return result


class _HighCurrentBus(MockDxlBus):
    """Inject high current values for quality-warning tests."""

    def __init__(self, current_ma: int = 900) -> None:
        super().__init__([1, 2, 3, 4])
        self.current_ma = int(current_ma)
        self._armed = False

    def arm(self) -> None:
        self._armed = True

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        result = super().read_telemetry(servo_ids, **kwargs)
        if not self._armed:
            return result
        for servo_id in servo_ids:
            result[int(servo_id)].present_current_ma = int(self.current_ma)
        return result


class _PostMoveCurrentSequenceBus(MockDxlBus):
    """Inject one post-write current sample on servo 2 per command."""

    def __init__(self, samples_ma: list[int], *, baseline_ma: int = 120) -> None:
        super().__init__([1, 2, 3, 4])
        self._samples_ma = [int(value) for value in samples_ma]
        self._pending_sample_ma: int | None = None
        self._baseline_ma = int(baseline_ma)

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        self._pending_sample_ma = self._samples_ma.pop(0) if self._samples_ma else self._baseline_ma

    def read_telemetry(self, servo_ids: list[int], **kwargs):
        result = super().read_telemetry(servo_ids, **kwargs)
        value = self._baseline_ma
        if self._pending_sample_ma is not None:
            value = int(self._pending_sample_ma)
            self._pending_sample_ma = None
        if 2 in result:
            result[2].present_current_ma = int(value)
        return result

class _FlakyWriteGoalBus(MockDxlBus):
    def __init__(self, *, fail_writes: int = 1) -> None:
        super().__init__([1, 2, 3, 4])
        self.fail_writes_remaining = int(fail_writes)
        self.experiment_writes_only = False

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        if self.experiment_writes_only and self.fail_writes_remaining > 0:
            self.fail_writes_remaining -= 1
            raise RuntimeError("Failed to write goal position for servo 3: [TxRxResult] Incorrect status packet!")
        super().write_goal_positions(positions_by_id)


class _AlwaysFailExperimentWriteBus(MockDxlBus):
    """Fail goal writes once ``_goal_writes`` exceeds ``arm_after_goal_writes`` (skips pretension/setup)."""

    def __init__(self, *, arm_after_goal_writes: int = 22) -> None:
        super().__init__([1, 2, 3, 4])
        self.arm_after_goal_writes = int(arm_after_goal_writes)
        self._goal_writes = 0

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        self._goal_writes += 1
        if self._goal_writes > self.arm_after_goal_writes:
            raise RuntimeError("Failed to write goal position for servo 3: [TxRxResult] Incorrect status packet!")
        super().write_goal_positions(positions_by_id)


class _MinimalStripPositionBus(MockDxlBus):
    """After ``arm_faulting_live_reads()``, coordinated precheck live reads omit Present Position."""

    def __init__(self, servo_ids: list[int] | None = None) -> None:
        super().__init__(servo_ids or [1, 2, 3, 4])
        self._fault_live_reads = False

    def arm_faulting_live_reads(self) -> None:
        self._fault_live_reads = True

    def read_live_telemetry(self, servo_ids: list[int]):
        result = super().read_live_telemetry(servo_ids)
        if not self._fault_live_reads:
            return result
        for servo_id in servo_ids:
            tel = result[int(servo_id)]
            tel.present_position = None
            tel.telemetry_error = "mock_live_read_missing_position"
            tel.telemetry_error_code = "missing_position"
        return result

    def read_minimal_telemetry(self, servo_ids: list[int]):
        result = super().read_minimal_telemetry(servo_ids)
        if not self._fault_live_reads:
            return result
        for servo_id in servo_ids:
            tel = result[int(servo_id)]
            tel.present_position = None
            tel.telemetry_error = "mock_minimal_read_missing_position"
            tel.telemetry_error_code = "missing_position"
        return result


class _PretensionExperimentBus(MockDxlBus):
    def __init__(self) -> None:
        super().__init__([1, 2, 3, 4])
        self._current_sequence = [180, 230, 180, 235, 180, 240]
        for telemetry in self._state.values():
            telemetry.torque_enabled = True
            telemetry.present_position = 4031
            telemetry.present_current_ma = 150

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        if 1 in positions_by_id and self._current_sequence:
            self._state[1].present_current_ma = self._current_sequence.pop(0)


class _StagedPretensionExperimentBus(MockDxlBus):
    def __init__(self) -> None:
        super().__init__([1, 2, 3, 4])
        self._current_sequences = {
            1: [180, 235],
            2: [182, 236],
            3: [179, 234],
            4: [181, 237],
        }
        for telemetry in self._state.values():
            telemetry.torque_enabled = True
            telemetry.present_position = 4031
            telemetry.present_current_ma = 150

    def write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        super().write_goal_positions(positions_by_id)
        for servo_id in positions_by_id:
            sequence = self._current_sequences.get(int(servo_id), [])
            if sequence:
                self._state[int(servo_id)].present_current_ma = int(sequence.pop(0))


class _SequencedTrackingService:
    def __init__(self, snapshots: list[TrackingSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self._index = 0
        self._thread = object()

    def peek_snapshot(self) -> TrackingSnapshot:
        if not self._snapshots:
            raise RuntimeError("No tracking snapshots configured for test.")
        if self._index >= len(self._snapshots):
            return self._snapshots[-1]
        return self._snapshots[self._index]

    def get_snapshot(self) -> TrackingSnapshot:
        if not self._snapshots:
            raise RuntimeError("No tracking snapshots configured for test.")
        if self._index >= len(self._snapshots):
            return self._snapshots[-1]
        snapshot = self._snapshots[self._index]
        self._index += 1
        return snapshot

    def start(self) -> None:
        self._thread = object()

    def stop(self) -> None:
        self._thread = None


class _DisconnectedTrackingService:
    _thread = object()

    def peek_snapshot(self):
        return SimpleNamespace(
            canonical_state="disconnected",
            selected_backend_name="mock",
            backend_identity="mock",
            registration_state="missing_registration",
            tip_pose_status="missing_runtime_tip",
            runtime_tip_mode="latest_accepted",
            runtime_tip_trust_level="missing",
            runtime_tip_mode_message="Tracker disconnected.",
            runtime_tip_calibration_state="missing_runtime_tip_calibration",
            runtime_tip_selected_artifact_kind=None,
            runtime_tip_selected_artifact_path=None,
        )

    def get_snapshot(self):
        return self.peek_snapshot()


class _TimingDiagnosticTrackingService:
    def __init__(self, records: list[dict], *, backend_identity: str = "ndi_tracker_python") -> None:
        self._records = [dict(record) for record in records]
        self._listeners: list = []
        self._thread = None
        self._backend_identity = backend_identity
        self._emitter_thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = object()

    def stop(self) -> None:
        self._thread = None
        emitter = self._emitter_thread
        if emitter is not None and emitter.is_alive():
            emitter.join(timeout=1.0)
        self._emitter_thread = None

    def register_timing_listener(self, listener) -> None:
        self._listeners.append(listener)
        if self._emitter_thread is None:
            self._emitter_thread = threading.Thread(target=self._emit_records, daemon=True)
            self._emitter_thread.start()

    def unregister_timing_listener(self, listener) -> None:
        self._listeners = [item for item in self._listeners if item is not listener]

    def _emit_records(self) -> None:
        for record in self._records:
            payload = dict(record)
            payload.setdefault("backend_identity", self._backend_identity)
            for listener in list(self._listeners):
                listener(payload)
            time.sleep(0.001)

    def peek_snapshot(self) -> TrackingSnapshot:
        return self.get_snapshot()

    def get_snapshot(self) -> TrackingSnapshot:
        return TrackingSnapshot(
            health=ServiceHealthSnapshot(
                name="tracking_service",
                health=HEALTH_HEALTHY,
                state="tracking",
                status="ok",
            ),
            connection_state="tracking",
            canonical_state="streaming_healthy",
            backend_identity=self._backend_identity,
            configured_backend_name="ndi",
            selected_backend_name="ndi",
            runtime_coil_tool_id="0A",
            registration_tool_id="0B",
            tracker_data_age_s=0.01,
            tracker_data_stale=False,
            last_frame_number=(
                int(self._records[-1]["frame_number"])
                if self._records and self._records[-1].get("frame_number") is not None
                else None
            ),
            tools={
                "0A": ToolTrackingSnapshot(tool_id="0A"),
                "0B": ToolTrackingSnapshot(tool_id="0B"),
            },
        )


def _tracking_snapshot(*, translation_mm: list[float]) -> TrackingSnapshot:
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(
            name="tracking_service",
            health=HEALTH_HEALTHY,
            state="tracking",
            status="ok",
        ),
        connection_state="tracking",
        canonical_state="streaming_healthy",
        backend_identity="mock",
        selected_backend_name="mock",
        runtime_coil_tool_id="0A",
        registration_tool_id="0B",
        tracker_data_age_s=0.01,
        tracker_data_stale=False,
        last_frame_number=1,
        normalized_live_tool_ids=["0A"],
        tools={
            "0A": ToolTrackingSnapshot(
                tool_id="0A",
                present=True,
                valid=True,
                validity_known=True,
                tracking_state="valid",
                status="ok",
                frame_number=1,
                translation_mm=tuple(float(value) for value in translation_mm),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            "0B": ToolTrackingSnapshot(tool_id="0B"),
        },
    )


def _trusted_modeling_snapshot(
    *,
    translation_mm: list[float],
    registration_path: Path,
    frame_number: int = 1,
    tracker_age_s: float = 0.01,
    tracker_stale: bool = False,
    runtime_tip_mode: str = "coil_as_tip",
    registration_state: str = "loaded",
    tip_pose_status: str = "ok",
    include_robot_tip: bool = True,
) -> TrackingSnapshot:
    T_robot_tip = np.eye(4).tolist() if include_robot_tip else None
    if T_robot_tip is not None:
        T_robot_tip[0][3] = float(translation_mm[0])
        T_robot_tip[1][3] = float(translation_mm[1])
        T_robot_tip[2][3] = float(translation_mm[2])
    return TrackingSnapshot(
        health=ServiceHealthSnapshot(
            name="tracking_service",
            health=HEALTH_HEALTHY,
            state="tracking",
            status="ok",
        ),
        connection_state="tracking",
        canonical_state="streaming_healthy",
        backend_identity="ndi_tracker_python",
        configured_backend_name="ndi",
        selected_backend_name="ndi",
        runtime_coil_tool_id="0A",
        registration_tool_id="0B",
        tracker_data_age_s=float(tracker_age_s),
        tracker_data_stale=bool(tracker_stale),
        last_frame_number=int(frame_number),
        registration_state=str(registration_state),
        registration_path=str(registration_path),
        T_robot_aurora=np.eye(4).tolist() if registration_state == "loaded" else None,
        runtime_tip_calibration_state=(
            "coil_as_tip"
            if runtime_tip_mode == "coil_as_tip" and include_robot_tip
            else ("loaded" if include_robot_tip else "missing_runtime_tip_calibration")
        ),
        runtime_tip_calibration_path=str(registration_path.parent / "latest_runtime_tip_calibration.json"),
        runtime_tip_mode=str(runtime_tip_mode),
        runtime_tip_trust_level="thesis_trusted" if runtime_tip_mode == "coil_as_tip" else "lower_trust",
        runtime_tip_mode_message="Test runtime tip mode",
        runtime_tip_selected_artifact_kind="latest_runtime_tip_calibration",
        runtime_tip_selected_artifact_path=str(registration_path.parent / "latest_runtime_tip_calibration.json"),
        runtime_tip_identity_fallback=bool(runtime_tip_mode == "coil_as_tip"),
        tip_pose_status=("coil_as_tip" if runtime_tip_mode == "coil_as_tip" and tip_pose_status == "ok" else str(tip_pose_status)),
        T_robot_tip=T_robot_tip,
        normalized_live_tool_ids=["0A", "0B"],
        tools={
            "0A": ToolTrackingSnapshot(
                tool_id="0A",
                present=True,
                valid=True,
                validity_known=True,
                tracking_state="valid",
                status="ok",
                frame_number=int(frame_number),
                translation_mm=tuple(float(value) for value in translation_mm),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            "0B": ToolTrackingSnapshot(
                tool_id="0B",
                present=True,
                valid=True,
                validity_known=True,
                tracking_state="valid",
                status="ok",
                frame_number=int(frame_number),
                translation_mm=(0.0, 0.0, 0.0),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        },
    )


def _tracking_service(settings: Settings, tmp_path: Path, registration_path: Path | None = None) -> TrackingService:
    return TrackingService(
        live_backend=MockTrackerManager(poll_hz=30),
        port=settings.serial.aurora_port,
        registration_path=registration_path or (tmp_path / "latest_registration.json"),
        config_source="test",
        runtime_coil_tool_id=settings.registration.coil_tool_id,
        registration_tool_id=settings.registration.capture_tool_id,
    )


def _ready_modeling_servo_service(tmp_path: Path, *, dxl_bus: MockDxlBus | None = None) -> ServoService:
    service = _servo_service(tmp_path, dxl_bus=dxl_bus)
    service.connect("/dev/mock-openrb", 115200)
    # Fresh temp neutral.json leaves robot_mode unset; collect-pose then skips the simple
    # experiment-motion path and reads post-command telemetry via read_telemetry instead of
    # read_live_telemetry. Match the usual 4-servo lab context so mock packet faults on
    # read_live_telemetry exercise the same code path as production.
    ctx = service.neutral_calibration.context
    if getattr(ctx, "robot_mode", None) in (None, ""):
        ctx.robot_mode = str(_settings().robot.mode or "4-servo").strip().lower().replace("-", "_")
    for servo_id in [1, 2, 3, 4]:
        service.save_startup_calibration(servo_id=servo_id)
    service.capture_manual_pretension_state(note="test modeling startup")
    service.accept_manual_pretension_state()
    return service


def _ready_parallel_modeling_servo_service(tmp_path: Path) -> ServoService:
    service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4, 5, 6, 7, 8]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral_parallel.json"),
        pretension_validation=PretensionValidationService(),
    )
    service.connect("/dev/mock-openrb", 57600)
    for servo_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        service.save_startup_calibration(servo_id=servo_id)
    service.capture_manual_pretension_state(note="parallel-single demo startup")
    service.accept_manual_pretension_state()
    return service


def _runner(
    tmp_path: Path,
    *,
    settings: Settings | None = None,
    servo_service: ServoService | None = None,
    tracking_service: TrackingService | None = None,
    registration_path: Path | None = None,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunner:
    settings = settings or _settings()
    registration_path = registration_path or (tmp_path / "latest_registration.json")
    return ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=tracking_service or _tracking_service(settings, tmp_path, registration_path),
        servo_service=servo_service or _servo_service(tmp_path),
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=registration_path,
        sleep_fn=lambda _seconds: None,
        experiment_registry=registry,
    )


class LifecycleProbeExperiment(BaseExperiment):
    name = "lifecycle_probe"
    description = "Exercise lifecycle hooks for tests."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    @classmethod
    def from_dict(cls, payload=None):
        return cls(config=dict(payload or {}))

    def setup(self, session: ExperimentSession) -> None:
        session.set_metric("setup_called", True)

    def precheck(self, session: ExperimentSession) -> None:
        session.set_metric("precheck_called", True)

    def execute(self, session: ExperimentSession) -> None:
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=0.0,
                wall_time_utc="2026-01-01T00:00:00+00:00",
                phase="execute",
                step_index=0,
                sample_index=0,
                status_flags=["ok"],
            )
        )

    def finalize(self, session: ExperimentSession) -> None:
        session.set_metric("finalize_called", True)

    def summarize(self, session: ExperimentSession) -> dict:
        return {"summary_called": True}


class IntegrityLeakProbeExperiment(BaseExperiment):
    name = "integrity_leak_probe"
    description = "Simulate an experiment that falls back to synthetic data after metadata is built."
    hardware_requirements = ExperimentHardwareRequirements(mock_compatible=True)

    @classmethod
    def from_dict(cls, payload=None):
        return cls(config=dict(payload or {}))

    def execute(self, session: ExperimentSession) -> None:
        session.set_metric("dry_run", True)
        session.add_sample(
            ExperimentTimeseriesSample(
                monotonic_time_s=0.0,
                wall_time_utc="dry_run",
                phase="capture",
                step_index=0,
                sample_index=0,
                status_flags=["dry_run", "synthetic_capture"],
                backend_health={"capture_mode": "synthetic_dry_run"},
                extra={"capture_mode": "synthetic_dry_run", "dry_run": True},
            )
        )


def test_experiment_lifecycle_records_stage_results(tmp_path: Path) -> None:
    registry = ExperimentRegistry()
    registry.register(
        name=LifecycleProbeExperiment.name,
        description=LifecycleProbeExperiment.description,
        factory=LifecycleProbeExperiment.from_dict,
    )
    runner = _runner(tmp_path, registry=registry)
    result = runner.run_experiment("lifecycle_probe")

    assert result.success is True
    assert result.summary.stage_pass_fail == {
        "setup": "passed",
        "precheck": "passed",
        "execute": "passed",
        "finalize": "passed",
    }
    assert result.summary.experiment_metrics["setup_called"] is True
    assert result.summary.experiment_metrics["precheck_called"] is True
    assert result.summary.experiment_metrics["finalize_called"] is True
    assert result.summary.experiment_metrics["summary_called"] is True


def test_runner_promotes_sample_level_dry_run_to_run_level_non_evidence(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registry = ExperimentRegistry()
    registry.register(
        name=IntegrityLeakProbeExperiment.name,
        description=IntegrityLeakProbeExperiment.description,
        factory=IntegrityLeakProbeExperiment.from_dict,
    )
    runner = _runner(tmp_path, settings=settings, registry=registry)

    result = runner.run_experiment("integrity_leak_probe", config={"dry_run": False})

    metrics = result.summary.experiment_metrics
    reasons = metrics["not_thesis_evidence_reasons"]
    assert result.metadata.trust_info["not_thesis_evidence"] is True
    assert metrics["not_thesis_evidence"] is True
    assert metrics["valid_for_model_training"] is False
    assert metrics["valid_for_thesis_repeatability"] is False
    assert "metrics.dry_run=true" in reasons
    assert any("samples.status_flags.dry_run" in reason for reason in reasons)
    assert any("samples.capture_mode.synthetic_or_dry_run" in reason for reason in reasons)


def test_dataset_writer_roundtrip_loads_canonical_bundle(tmp_path: Path) -> None:
    writer = ExperimentDatasetWriter(tmp_path / "datasets")
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="roundtrip_test",
        run_id="abc123",
        timestamp_utc="2026-01-01T00:00:00+00:00",
        git_commit=None,
        backend_info={"mock_mode": True},
        registration_info={"exists": False},
        config_used={"alpha": 1},
        operator_notes="note",
    )
    samples = [
        ExperimentTimeseriesSample(
            monotonic_time_s=0.0,
            wall_time_utc="2026-01-01T00:00:00+00:00",
            phase="sample",
            step_index=0,
            sample_index=0,
            status_flags=["ok"],
        )
    ]
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="roundtrip_test",
        run_id="abc123",
        success=True,
        sample_counts={"total": 1},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"execute": "passed"},
    )

    paths = writer.write_dataset(metadata, samples, summary)
    bundle = ExperimentDatasetLoader().load_dataset(paths.output_dir)

    assert bundle.metadata.experiment_name == "roundtrip_test"
    assert paths.output_dir.parent == tmp_path / "datasets" / "roundtrip_test"
    assert len(bundle.samples) == 1
    assert bundle.summary.success is True


def test_dataset_writer_uses_timestamp_experiment_name_and_collision_suffix(tmp_path: Path, monkeypatch) -> None:
    class _FixedDateTime:
        @staticmethod
        def now(_tz=None):
            return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(dataset_io, "datetime", _FixedDateTime)
    writer = ExperimentDatasetWriter(tmp_path / "datasets")
    metadata = ExperimentMetadata(
        schema_version="1.0",
        experiment_name="registration_validation",
        run_id="abc123",
        timestamp_utc="2026-01-02T03:04:05+00:00",
        git_commit=None,
        backend_info={"mock_mode": True},
        registration_info={"exists": False},
        config_used={},
    )
    summary = ExperimentSummary(
        schema_version="1.0",
        experiment_name="registration_validation",
        run_id="abc123",
        success=True,
        sample_counts={"total": 0},
        dropped_frames=0,
        invalid_transforms=0,
        stage_pass_fail={"execute": "passed"},
    )
    first = writer.write_dataset(metadata, [], summary)
    second = writer.write_dataset(metadata, [], summary)

    assert first.output_dir.name == "20260102_030405_registration_validation"
    assert second.output_dir.name == "20260102_030405_registration_validation_01"


def test_tracker_pipeline_mock_runs_and_logs_samples(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment(
        "tracker_pipeline_mock",
        config={"sample_count": 4, "sample_period_s": 0.0},
    )

    assert result.success is True
    assert result.sample_count == 4
    bundle = runner.load_dataset(result.paths.output_dir)
    assert all(sample.phase == "sample" for sample in bundle.samples)
    assert bundle.summary.sample_counts["total"] == 4


def test_transform_chain_validation_reports_zero_error(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    result = runner.run_experiment("transform_chain_validation")

    assert result.success is True
    assert result.summary.experiment_metrics["translation_error_mm"] < 1e-9


def test_command_schedule_generation_is_deterministic_and_bounded() -> None:
    config = CommandScheduleConfig(kind="babble", dimensions=4, amplitude_cm=0.2, babble_count=8, seed=7)
    first = generate_command_schedule(config)
    second = generate_command_schedule(config)

    assert command_schedule_checksum(first) == command_schedule_checksum(second)
    for point in first:
        assert all(-0.2 <= value <= 0.2 for value in point.tendon_displacement_cm)


def test_replay_runner_loads_existing_dataset(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    source = runner.run_experiment("dataset_schema_roundtrip", config={"sample_count": 2})
    replay = runner.run_experiment("replay_runner", config={"dataset_path": str(source.paths.output_dir)})

    assert replay.success is True
    assert replay.summary.experiment_metrics["source_sample_count"] == 2
    assert replay.sample_count == 2


def test_tracker_timing_validation_experiment_writes_canonical_outputs_and_summary(tmp_path: Path) -> None:
    timing_records = [
        {
            "sample_start_monotonic_ns": 1_000_000_000,
            "backend_call_start_ns": 1_000_000_000,
            "backend_call_end_ns": 1_012_000_000,
            "parse_complete_ns": 1_013_000_000,
            "state_commit_complete_ns": 1_014_000_000,
            "sample_commit_monotonic_ns": 1_014_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.000Z",
            "frame_number": 11,
            "frame_number_source": "device",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A", "0B"],
            "raw_tool_ids": ["10", "11"],
            "normalized_tool_ids": ["0A", "0B"],
            "runtime_role_mappings": {"0A": "10", "0B": "11"},
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "valid_transform_count": 2,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
        {
            "sample_start_monotonic_ns": 1_020_000_000,
            "backend_call_start_ns": 1_020_000_000,
            "backend_call_end_ns": 1_032_000_000,
            "parse_complete_ns": 1_033_000_000,
            "state_commit_complete_ns": 1_034_000_000,
            "sample_commit_monotonic_ns": 1_034_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.020Z",
            "frame_number": 11,
            "frame_number_source": "device",
            "is_new_frame": False,
            "is_duplicate_frame": True,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A"],
            "raw_tool_ids": ["10"],
            "normalized_tool_ids": ["0A"],
            "runtime_role_mappings": {"0A": "10"},
            "tool_validity": {"0A": "tracked", "0B": "missing"},
            "valid_transform_count": 1,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
        {
            "sample_start_monotonic_ns": 1_040_000_000,
            "backend_call_start_ns": 1_040_000_000,
            "backend_call_end_ns": 1_052_000_000,
            "parse_complete_ns": 1_053_000_000,
            "state_commit_complete_ns": 1_054_000_000,
            "sample_commit_monotonic_ns": 1_054_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.040Z",
            "frame_number": 12,
            "frame_number_source": "device",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A", "0B"],
            "raw_tool_ids": ["10", "11"],
            "normalized_tool_ids": ["0A", "0B"],
            "runtime_role_mappings": {"0A": "10", "0B": "11"},
            "tool_validity": {"0A": "tracked", "0B": "tracked"},
            "valid_transform_count": 2,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
    ]
    runner = _runner(
        tmp_path,
        tracking_service=_TimingDiagnosticTrackingService(timing_records),
    )

    result = runner.run_experiment(
        "tracker_timing_validation",
        config={
            "requested_tool_ids": ["0A", "0B"],
            "sample_count_target": 2,
            "warmup_samples": 1,
            "timeout_s": 1.0,
            "run_label": "bench_a",
            "enable_servo_logging": False,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["backend_identity"] == "ndi_tracker_python"
    assert result.summary.experiment_metrics["sample_count_analyzed"] == 2
    assert result.summary.experiment_metrics["warmup_discarded_count"] == 1
    assert result.summary.experiment_metrics["duplicate_frame_count"] == 1
    assert result.summary.experiment_metrics["unique_frame_count"] == 1
    assert result.summary.experiment_metrics["requested_tool_ids"] == ["0A", "0B"]
    assert result.summary.experiment_metrics["servo_sync"]["enabled"] is False
    assert result.paths.output_dir.joinpath("thesis_01_rate_vs_ceiling.png").exists()
    assert result.paths.output_dir.joinpath("thesis_02_inter_frame_interval.png").exists()
    assert result.paths.output_dir.joinpath("thesis_03_cycle_time_budget.png").exists()
    assert result.paths.output_dir.joinpath("debug.json").exists()
    for removed in [
        "aurora_timing_histogram.png",
        "aurora_timing_breakdown.png",
        "aurora_timing_timeseries.png",
        "aurora_timing_summary.txt",
        "aurora_timing_sync_offsets.png",
        # Pre-thesis-grade tracker artifacts: replaced by the 3 figures above.
        "thesis_01_cycle_time_distribution.png",
        "thesis_02_stage_time_budget.png",
        "tracker_inter_frame_interval_histogram.png",
        "tracker_unique_frame_rate_over_time.png",
        "tracker_duplicate_invalid_timeline.png",
        "tracker_polling_vs_unique_frame_rate.png",
        "tracker_valid_pose_rate_over_time.png",
    ]:
        assert not result.paths.output_dir.joinpath(removed).exists(), f"deprecated tracker_timing artifact should be gone: {removed}"
    debug_payload = json.loads((result.paths.output_dir / "debug.json").read_text(encoding="utf-8"))
    assert debug_payload["experiment_name"] == "tracker_timing_validation"
    assert "duplicate_frames" in debug_payload
    assert "stage_stats" in debug_payload
    assert debug_payload["rate"]["aurora_theoretical_max_hz"] == 40.0
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any(sample.extra.get("record_kind") == "tracker_timing" for sample in bundle.samples)


def test_tracker_timing_validation_experiment_outputs_do_not_crash_when_optional_fields_are_missing(tmp_path: Path) -> None:
    timing_records = [
        {
            "sample_start_monotonic_ns": 2_000_000_000,
            "backend_call_start_ns": 2_000_000_000,
            "backend_call_end_ns": 2_180_000_000,
            "parse_complete_ns": None,
            "state_commit_complete_ns": 2_180_000_000,
            "sample_commit_monotonic_ns": 2_180_000_000,
            "observed_at_utc": "2026-01-01T00:00:01.000Z",
            "frame_number": None,
            "frame_number_source": "missing",
            "is_new_frame": None,
            "is_duplicate_frame": None,
            "raw_payload_available": False,
            "parsed_payload_available": False,
            "output_committed": False,
            "error_flag": True,
            "error_stage": "get_frame",
            "error_message": "timeout",
            "tools_visible": [],
            "raw_tool_ids": [],
            "normalized_tool_ids": [],
            "runtime_role_mappings": {},
            "tool_validity": {},
            "valid_transform_count": 0,
            "total_cycle_ms": 180.0,
            "backend_call_ms": 180.0,
            "parse_ms": None,
            "state_commit_ms": 0.0,
        }
    ]
    runner = _runner(
        tmp_path,
        tracking_service=_TimingDiagnosticTrackingService(timing_records),
    )

    result = runner.run_experiment(
        "tracker_timing_validation",
        config={
            "requested_tool_ids": ["0A"],
            "sample_count_target": 1,
            "warmup_samples": 0,
            "timeout_s": 1.0,
            "enable_servo_logging": False,
        },
    )

    assert result.success is True
    assert result.paths.output_dir.joinpath("thesis_01_rate_vs_ceiling.png").exists()
    assert result.paths.output_dir.joinpath("thesis_02_inter_frame_interval.png").exists()
    assert result.paths.output_dir.joinpath("thesis_03_cycle_time_budget.png").exists()
    assert result.paths.output_dir.joinpath("debug.json").exists()
    assert result.summary.experiment_metrics["error_sample_count"] == 1


def test_servo_tracker_sync_validation_experiment_writes_canonical_outputs_and_summary(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    servo_service = _servo_service(tmp_path)
    servo_service.connect("/dev/mock-openrb", 115200)
    timing_records = [
        {
            "sample_start_monotonic_ns": 1_000_000_000,
            "backend_call_start_ns": 1_000_000_000,
            "backend_call_end_ns": 1_010_000_000,
            "parse_complete_ns": 1_011_000_000,
            "state_commit_complete_ns": 1_012_000_000,
            "sample_commit_monotonic_ns": 1_012_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.000Z",
            "frame_number": 101,
            "frame_number_source": "synthetic",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A"],
            "raw_tool_ids": ["10"],
            "normalized_tool_ids": ["0A"],
            "runtime_role_mappings": {"0A": "10"},
            "tool_validity": {"0A": "tracked"},
            "tool_pose_payload": {
                "0A": {
                    "tracking_state": "tracked",
                    "frame_number": 101,
                    "translation_mm": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            },
            "valid_transform_count": 1,
            "total_cycle_ms": 12.0,
            "backend_call_ms": 10.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
        {
            "sample_start_monotonic_ns": 1_020_000_000,
            "backend_call_start_ns": 1_020_000_000,
            "backend_call_end_ns": 1_030_000_000,
            "parse_complete_ns": 1_031_000_000,
            "state_commit_complete_ns": 1_032_000_000,
            "sample_commit_monotonic_ns": 1_032_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.020Z",
            "frame_number": 102,
            "frame_number_source": "synthetic",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A"],
            "raw_tool_ids": ["10"],
            "normalized_tool_ids": ["0A"],
            "runtime_role_mappings": {"0A": "10"},
            "tool_validity": {"0A": "tracked"},
            "tool_pose_payload": {
                "0A": {
                    "tracking_state": "tracked",
                    "frame_number": 102,
                    "translation_mm": [1.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            },
            "valid_transform_count": 1,
            "total_cycle_ms": 12.0,
            "backend_call_ms": 10.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
        {
            "sample_start_monotonic_ns": 1_040_000_000,
            "backend_call_start_ns": 1_040_000_000,
            "backend_call_end_ns": 1_050_000_000,
            "parse_complete_ns": 1_051_000_000,
            "state_commit_complete_ns": 1_052_000_000,
            "sample_commit_monotonic_ns": 1_052_000_000,
            "observed_at_utc": "2026-01-01T00:00:00.040Z",
            "frame_number": 103,
            "frame_number_source": "synthetic",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A"],
            "raw_tool_ids": ["10"],
            "normalized_tool_ids": ["0A"],
            "runtime_role_mappings": {"0A": "10"},
            "tool_validity": {"0A": "tracked"},
            "tool_pose_payload": {
                "0A": {
                    "tracking_state": "tracked",
                    "frame_number": 103,
                    "translation_mm": [2.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            },
            "valid_transform_count": 1,
            "total_cycle_ms": 12.0,
            "backend_call_ms": 10.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        },
    ]
    runner = _runner(
        tmp_path,
        settings=settings,
        servo_service=servo_service,
        tracking_service=_TimingDiagnosticTrackingService(timing_records),
    )

    result = runner.run_experiment(
        "servo_tracker_sync_validation",
        config={
            "servo_ids": [1],
            "requested_tool_ids": ["0A"],
            "run_duration_s": 0.12,
            "warmup_duration_s": 0.0,
            "command_amplitude_ticks": 8,
            "step_period_s": 0.04,
            "telemetry_poll_interval_s": 0.02,
            "timeout_s": 1.0,
            "include_robot_frame_tip_pose": True,
            "run_label": "sync_a",
        },
    )

    assert result.success is True
    assert result.paths.output_dir.parent == tmp_path / "data" / "experiments" / "servo_tracker_sync_validation"
    assert result.summary.experiment_metrics["backend_identity"] == "ndi_tracker_python"
    assert result.summary.experiment_metrics["selected_servo_ids"] == [1]
    assert result.summary.experiment_metrics["servo_telemetry_sample_count"] > 0
    assert result.summary.experiment_metrics["servo_command_sample_count"] > 0
    assert result.summary.experiment_metrics["servo_tracker_sync"]["available"] is True
    following = result.summary.experiment_metrics["servo_tracker_sync"]["servo_position_following"]
    assert following["available"] is True
    assert following["sample_count"] > 0
    assert following["max_abs_error_ticks"] is not None
    assert result.paths.output_dir.joinpath("thesis_01_pair_time_alignment.png").exists()
    assert result.paths.output_dir.joinpath("thesis_02_motion_correspondence.png").exists()
    assert result.paths.output_dir.joinpath("debug.json").exists()
    for removed in [
        "servo_tracker_offset_histogram.png",
        "servo_tracker_offset_timeseries.png",
        "servo_tracker_pose_command_timeseries.png",
        "servo_tracker_validity_summary.png",
        "servo_tracker_sync_summary.txt",
    ]:
        assert not result.paths.output_dir.joinpath(removed).exists(), f"deprecated servo_tracker_sync artifact should be gone: {removed}"
    debug_payload = json.loads((result.paths.output_dir / "debug.json").read_text(encoding="utf-8"))
    assert debug_payload["experiment_name"] == "servo_tracker_sync_validation"
    assert "alignment" in debug_payload
    assert "threshold_cross_rates" in debug_payload["alignment"]
    assert debug_payload["servo_position_following"]["available"] is True
    assert "sample_counts" in debug_payload
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any(sample.extra.get("record_kind") == "servo_command" for sample in bundle.samples)
    assert any(sample.extra.get("record_kind") == "servo_timing" for sample in bundle.samples)
    assert any(
        sample.extra.get("record_kind") == "servo_timing" and "position_error_ticks" in sample.extra
        for sample in bundle.samples
    )
    assert any(sample.extra.get("record_kind") == "tracker_timing" for sample in bundle.samples)


def test_servo_tracker_sync_validation_outputs_do_not_crash_without_tip_pose(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    servo_service = _servo_service(tmp_path)
    servo_service.connect("/dev/mock-openrb", 115200)
    timing_records = [
        {
            "sample_start_monotonic_ns": 2_000_000_000,
            "backend_call_start_ns": 2_000_000_000,
            "backend_call_end_ns": 2_012_000_000,
            "parse_complete_ns": 2_013_000_000,
            "state_commit_complete_ns": 2_014_000_000,
            "sample_commit_monotonic_ns": 2_014_000_000,
            "observed_at_utc": "2026-01-01T00:00:01.000Z",
            "frame_number": None,
            "frame_number_source": "missing",
            "is_new_frame": True,
            "is_duplicate_frame": False,
            "raw_payload_available": True,
            "parsed_payload_available": True,
            "output_committed": True,
            "error_flag": False,
            "tools_visible": ["0A"],
            "raw_tool_ids": ["10"],
            "normalized_tool_ids": ["0A"],
            "runtime_role_mappings": {"0A": "10"},
            "tool_validity": {"0A": "tracked"},
            "tool_pose_payload": {
                "0A": {
                    "tracking_state": "tracked",
                    "frame_number": None,
                    "translation_mm": [0.0, 0.0, 0.0],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            },
            "valid_transform_count": 1,
            "total_cycle_ms": 14.0,
            "backend_call_ms": 12.0,
            "parse_ms": 1.0,
            "state_commit_ms": 1.0,
        }
    ]
    runner = _runner(
        tmp_path,
        settings=settings,
        servo_service=servo_service,
        tracking_service=_TimingDiagnosticTrackingService(timing_records),
    )

    result = runner.run_experiment(
        "servo_tracker_sync_validation",
        config={
            "servo_ids": [1],
            "requested_tool_ids": ["0A"],
            "run_duration_s": 0.06,
            "warmup_duration_s": 0.0,
            "command_amplitude_ticks": 4,
            "step_period_s": 0.04,
            "telemetry_poll_interval_s": 0.02,
            "timeout_s": 1.0,
            "include_robot_frame_tip_pose": True,
        },
    )

    assert result.paths.output_dir.joinpath("thesis_01_pair_time_alignment.png").exists()
    assert result.paths.output_dir.joinpath("thesis_02_motion_correspondence.png").exists()
    assert result.paths.output_dir.joinpath("debug.json").exists()


def test_servo_tracker_sync_validation_blocks_legacy_bridge_backend(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    servo_service = _servo_service(tmp_path)
    servo_service.connect("/dev/mock-openrb", 115200)
    runner = _runner(
        tmp_path,
        settings=settings,
        servo_service=servo_service,
        tracking_service=_TimingDiagnosticTrackingService([], backend_identity="tracker_bridge_json"),
    )

    result = runner.run_experiment(
        "servo_tracker_sync_validation",
        config={"servo_ids": [1], "requested_tool_ids": ["0A"]},
    )

    assert result.success is False
    assert "tracker_bridge" in result.message


def test_pretension_validation_experiment_records_current_displacement_trace_and_outputs(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=_PretensionExperimentBus(),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=3,
            pretension_current_filter_window=1,
            pretension_current_delta_threshold_ma=60,
            pretension_absolute_trigger_current_ma=500,
            pretension_max_travel_ticks=320,
        ),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "pretension_neutral.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    tracking_service = _SequencedTrackingService(
        [
            _tracking_snapshot(translation_mm=[0.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[0.0, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.2, 0.0, 0.0]),
            _tracking_snapshot(translation_mm=[1.2, 0.0, 0.0]),
        ]
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        servo_service=servo_service,
        tracking_service=tracking_service,
    )

    result = runner.run_experiment(
        "pretension_validation",
        config={
            "servo_id": 1,
            "move_to_reference": False,
            "include_tracker_displacement": True,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["servo_id"] == 1
    assert result.summary.experiment_metrics["accepted"] is True
    assert result.summary.experiment_metrics["stop_reason"] == "baseline_delta_trigger"
    assert result.summary.experiment_metrics["baseline_current_ma"] == pytest.approx(150.0)
    assert result.summary.experiment_metrics["effective_trigger_current_ma"] == 210
    assert result.summary.experiment_metrics["max_observed_displacement_mm"] == pytest.approx(1.2)
    assert result.paths.output_dir.joinpath("pretension_response.png").exists()
    summary_note = result.paths.output_dir / "pretension_summary.txt"
    assert summary_note.exists()
    summary_text = summary_note.read_text(encoding="utf-8").lower()
    assert "engagement proxy" in summary_text
    assert "tendon tension" not in summary_text
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any(sample.phase == "pretension_baseline" for sample in bundle.samples)
    assert any(sample.phase == "pretension_step" for sample in bundle.samples)
    assert any(sample.phase == "pretension_result" for sample in bundle.samples)
    assert any(sample.extra.get("travel_from_untensioned_ticks") is not None for sample in bundle.samples)
    assert any(sample.extra.get("tracker_displacement_mm") is not None for sample in bundle.samples)


def test_pretension_validation_experiment_writes_outputs_without_tracker_data(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=_PretensionExperimentBus(),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=3,
            pretension_current_filter_window=1,
            pretension_current_delta_threshold_ma=60,
            pretension_absolute_trigger_current_ma=500,
            pretension_max_travel_ticks=320,
        ),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "pretension_neutral.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    runner = ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=None,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "pretension_validation",
        config={
            "servo_id": 1,
            "move_to_reference": False,
            "include_tracker_displacement": True,
        },
    )

    assert result.success is True
    assert result.summary.experiment_metrics["tracker_metric_sample_count"] == 0
    assert result.paths.output_dir.joinpath("pretension_response.png").exists()
    assert result.paths.output_dir.joinpath("pretension_summary.txt").exists()


def test_pretension_validation_single_segment_staged_writes_units_metrics_and_plots(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=_StagedPretensionExperimentBus(),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(
            min_offset_ticks=-600,
            max_offset_ticks=600,
            max_current_ma=850,
            default_pretension_current_threshold_ma=220,
            fine_jog_step_ticks=5,
            coarse_jog_step_ticks=25,
            software_position_margin_ticks=64,
            telemetry_stale_after_s=0.25,
            pretension_step_ticks=2,
            pretension_timeout_s=2.0,
            pretension_settle_time_s=0.0,
            pretension_baseline_sample_count=3,
            pretension_current_filter_window=1,
            pretension_current_delta_threshold_ma=60,
            pretension_absolute_trigger_current_ma=500,
            pretension_max_travel_ticks=320,
        ),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "pretension_neutral_staged.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    tracking_service = _SequencedTrackingService(
        [
            _trusted_modeling_snapshot(
                translation_mm=[0.0, 0.0, 60.0],
                registration_path=tmp_path / "latest_registration.json",
                frame_number=1,
                runtime_tip_mode="coil_as_tip",
            ),
            _trusted_modeling_snapshot(
                translation_mm=[0.5, -0.2, 60.0],
                registration_path=tmp_path / "latest_registration.json",
                frame_number=2,
                runtime_tip_mode="coil_as_tip",
            ),
            _trusted_modeling_snapshot(
                translation_mm=[0.2, 0.1, 60.0],
                registration_path=tmp_path / "latest_registration.json",
                frame_number=3,
                runtime_tip_mode="coil_as_tip",
            ),
        ]
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        servo_service=servo_service,
        tracking_service=tracking_service,
    )

    result = runner.run_experiment(
        "pretension_validation",
        config={
            "mode": "single_segment_staged",
            "servo_ids": [1, 2, 3, 4],
            "repeat_runs": 1,
            "include_tracker_displacement": True,
            "allow_current_only_when_tracker_missing": True,
            "enable_tip_centering": False,
            "equalization_max_iterations": 2,
            # Pin the start mode to the no-motion legacy default. This test is
            # about output file generation, not the new
            # ``soft_release_to_zero_current`` default which would consume
            # more sequenced tracker snapshots than this fixture provides.
            "pretension_start_mode": "current_position",
            # The mock bus doesn't drive currents to -30 mA, so relax the
            # signed-tension acceptance gate to 0 mA for this output-file
            # generation test. The tension gate itself is exercised by the
            # dedicated unit tests in test_pretension_validation_experiment.
            "takeup_target_holding_tension_ma": 0.0,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["mode"] == "single_segment_staged"
    assert metrics["run_count"] == 1
    assert metrics["units"]["baseline_current_ma"] == "mA"
    assert metrics["units"]["tip_position_mm"] == "mm"
    assert result.paths.output_dir.joinpath("metrics.csv").exists()
    assert result.paths.output_dir.joinpath("pretension_summary.txt").exists()
    assert result.paths.output_dir.joinpath("pretension_current_vs_position.png").exists()
    assert result.paths.output_dir.joinpath("pretension_tip_xy_path.png").exists()
    assert result.paths.output_dir.joinpath("pretension_final_current_distribution.png").exists()
    assert result.paths.output_dir.joinpath("pretension_repeatability_summary.png").exists()
    # Thesis report set: one synchronized timeline plus one final-state
    # consistency figure. Older overlapping plots remain diagnostics only.
    report_pngs = sorted(path.name for path in result.paths.output_dir.glob("*_report.png"))
    assert report_pngs == [
        "pretension_final_state_consistency_report.png",
        "pretension_telemetry_timeline_report.png",
    ]
    from continuum_robot.data.run_management import summarize_run

    run_summary = summarize_run(result.paths.output_dir)
    assert sorted(run_summary.report_figures) == report_pngs
    assert result.paths.output_dir.joinpath("pretension_quality_summary.json").exists()


def test_pretension_validation_single_segment_staged_requires_tracker_when_current_only_disabled(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = ServoService(
        dxl_bus=_StagedPretensionExperimentBus(),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "pretension_neutral_staged.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    runner = ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=None,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "pretension_validation",
        config={
            "mode": "single_segment_staged",
            "servo_ids": [1, 2, 3, 4],
            "repeat_runs": 1,
            "include_tracker_displacement": True,
            "allow_current_only_when_tracker_missing": False,
            "enable_tip_centering": False,
        },
    )

    assert result.success is False
    assert "tracking_service is unavailable" in result.message


def test_collect_pose_command_dataset_marks_registration_missing(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    snapshot = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0],
        registration_path=registration_path,
        frame_number=1,
        registration_state="missing_registration",
        tip_pose_status="missing_runtime_tip",
        include_robot_tip=False,
    )
    runner = _runner(
        tmp_path,
        tracking_service=_SequencedTrackingService([snapshot, snapshot, snapshot]),
        registration_path=registration_path,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": True,
            "sample_count_target": 3,
            "samples_per_command": 1,
            "require_robot_frame_tip": False,
            "allow_lower_trust_runtime_tip": True,
        },
    )

    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any("registration_missing" in sample.status_flags for sample in bundle.samples)
    assert bundle.summary.status == "partial_success"
    assert not any("full_pose_available" in sample.status_flags for sample in bundle.samples if sample.phase != "initial_neutral")


def test_collect_pose_command_dataset_records_full_pose_when_registration_exists(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    snapshots = [
        _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1),
        _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=2),
        _trusted_modeling_snapshot(translation_mm=[1.0, 0.5, 0.2], registration_path=registration_path, frame_number=3),
        _trusted_modeling_snapshot(translation_mm=[2.0, 1.0, 0.3], registration_path=registration_path, frame_number=4),
    ]
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService(snapshots),
        registration_path=registration_path,
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
        },
    )

    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    assert any("full_pose_available" in sample.status_flags for sample in bundle.samples)
    accepted_samples = [sample for sample in bundle.samples if sample.extra.get("capture_accepted")]
    assert accepted_samples
    assert accepted_samples[0].pose_in_robot_frame["tip"]["tangent_xyz"] == [0.0, 0.0, 1.0]
    assert accepted_samples[0].pose_in_robot_frame["tip"]["quaternion_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert accepted_samples[0].extra["motion_profile"]["operating_mode_label"] == "Position Control"
    assert accepted_samples[0].extra["motion_profile"]["goal_current_ma"] is None
    assert bundle.paths.output_dir.joinpath("modeling_dataset_export.jsonl").exists()
    assert bundle.paths.output_dir.joinpath("thesis_01_workspace_coverage_3d.png").exists()
    assert bundle.paths.output_dir.joinpath("thesis_02_command_and_workspace_2d.png").exists()
    assert bundle.paths.output_dir.joinpath("debug.json").exists()
    for removed in [
        "modeling_dataset_summary.txt",
        "modeling_workspace_coverage.png",
        "modeling_workspace_coverage_report.png",
        "commanded_tendon_space_report.png",
        "modeling_command_distribution.png",
    ]:
        assert not bundle.paths.output_dir.joinpath(removed).exists(), f"deprecated collect_pose artifact should be gone: {removed}"
    import json as _json
    debug_payload = _json.loads(bundle.paths.output_dir.joinpath("debug.json").read_text(encoding="utf-8"))
    assert debug_payload["experiment_name"] == "collect_pose_command_dataset"
    assert "acceptance" in debug_payload
    assert "trainability" in debug_payload
    assert "workspace_extent_mm" in debug_payload
    run_provenance = dict(result.summary.experiment_metrics.get("run_provenance", {}) or {})
    preflight = dict(run_provenance.get("run_start_preflight", {}) or {})
    assert preflight.get("operating_mode") == "single_segment"
    assert preflight.get("active_segment") == "segment_a"
    assert preflight.get("commanded_servo_ids") == [1, 2, 3, 4]
    assert bundle.paths.output_dir.joinpath("modeling_dataset_legacy_compat.dat").exists()
    metrics = bundle.summary.experiment_metrics
    assert metrics["run_provenance"]["runtime_tip_calibration"]["mode"] == "coil_as_tip"
    assert metrics["run_provenance"]["runtime_tip_calibration"]["trust_level"] == "thesis_trusted"
    assert metrics["run_provenance"]["pretension_artifact"]["active_source_type"] == "manual"
    assert metrics["run_provenance"]["startup_reference_source"] == "manual"
    export_rows = [
        json.loads(line)
        for line in bundle.paths.output_dir.joinpath("modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert export_rows
    assert [row["sequence_index"] for row in export_rows] == list(range(len(export_rows)))
    assert any(row["accepted"] for row in export_rows)
    assert any(row["tip_tangent_xyz"] == [0.0, 0.0, 1.0] for row in export_rows if row["accepted"])


def test_collect_pose_command_dataset_runs_servo_only_without_tracker_when_explicit(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _ready_modeling_servo_service(tmp_path)
    runner = ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=_DisconnectedTrackingService(),
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "allow_no_tracker_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True
    assert result.metadata.trust_info["run_trust_mode"] == "mock"
    assert result.metadata.trust_info["valid_for_model_training"] is False
    assert result.metadata.provenance_info["hardware_profile"] == settings.runtime.robot_config
    assert result.metadata.provenance_info["expected_servo_ids"] == settings.robot.expected_servo_ids()
    metrics = result.summary.experiment_metrics
    assert metrics["run_trust_mode"] == "mock"
    assert metrics["run_trust"]["valid_for_model_training"] is False
    assert metrics["valid_for_model_training"] is False
    assert metrics["valid_for_thesis_repeatability"] is False
    assert metrics["valid_for_two_segment_model_training"] is False
    assert "mock_mode" in metrics["data_quality_warnings"]
    assert metrics["legacy_export_enabled"] is False
    assert metrics["position_frame"] == "none"
    assert not result.paths.output_dir.joinpath("modeling_dataset_legacy_compat.dat").exists()
    export_rows = [
        json.loads(line)
        for line in result.paths.output_dir.joinpath("modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert export_rows == []


def test_collect_pose_parallel_single_demo_records_all_8_metadata_and_non_training_valid(tmp_path: Path) -> None:
    settings = _parallel_single_settings()
    servo_service = _ready_parallel_modeling_servo_service(tmp_path)
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    snapshot = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0],
        registration_path=registration_path,
        frame_number=1,
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snapshot]),
        registration_path=registration_path,
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "workspace_amplitude_cm": 0.25,
        },
    )

    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert metrics["parallel_single_demo"] is True
    assert metrics["true_two_segment_control"] is False
    assert metrics["valid_for_model_training"] is False
    assert metrics["valid_for_thesis_repeatability"] is False
    assert metrics["not_model_training_ready"] is True

    bundle = runner.load_dataset(result.paths.output_dir)
    command_samples = [
        sample
        for sample in bundle.samples
        if sample.phase not in {"initial_neutral", "final_neutral"}
    ]
    assert command_samples
    metadata = {}
    for sample in command_samples:
        candidate = dict(sample.extra.get("command_metadata", {}) or {})
        if candidate.get("mirrored_parallel"):
            metadata = candidate
            break
    assert metadata
    assert metadata["parallel_single_demo"] is True
    assert metadata["true_two_segment_control"] is False
    assert len(metadata["shared_4_tendon_command_cm"]) == 4
    assert len(metadata["segment_a_command_cm"]) == 4
    assert len(metadata["segment_b_command_cm"]) == 4
    assert sorted(int(value) for value in metadata["all_8_goal_ticks"].keys()) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert sorted(int(value) for value in metadata["segment_a_goal_ticks"].keys()) == [1, 2, 3, 4]
    assert sorted(int(value) for value in metadata["segment_b_goal_ticks"].keys()) == [5, 6, 7, 8]
    assert metadata["segment_a"]["segment_role"] == "proximal"
    assert metadata["segment_b"]["segment_role"] == "distal"

    run_provenance = metrics["run_provenance"]
    assert run_provenance["parallel_single_demo"] is True
    assert run_provenance["true_two_segment_control"] is False
    assert run_provenance["valid_for_model_training"] is False
    assert "parallel_single" in run_provenance
    preflight = dict(run_provenance.get("run_start_preflight", {}) or {})
    assert preflight.get("operating_mode") == "parallel_single"
    assert preflight.get("commanded_servo_ids") == [1, 2, 3, 4, 5, 6, 7, 8]
    assert preflight.get("parallel_single_demo") is True


def test_collect_pose_packet_status_failure_writes_failure_context_and_quality(tmp_path: Path) -> None:
    settings = _settings()
    bus = _CollectPosePacketErrorBus(failed_servo_id=3, failures=3)
    servo_service = ServoService(
        dxl_bus=bus,
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
        sleep_fn=lambda _seconds: None,
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    for servo_id in [1, 2, 3, 4]:
        servo_service.save_startup_calibration(servo_id=servo_id)
    servo_service.capture_manual_pretension_state(note="packet error test")
    servo_service.accept_manual_pretension_state()
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_DisconnectedTrackingService(),
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "allow_no_tracker_test_run": True,
            "run_trust_mode": "servo_only",
            "telemetry_retry_count": 2,
            "telemetry_retry_delay_s": 0.0,
        },
    )

    assert result.success is False
    failure_context_path = result.paths.output_dir / "failure_context.json"
    quality_path = result.paths.output_dir / "dataset_quality_summary.json"
    assert failure_context_path.exists()
    assert quality_path.exists()
    failure_context = json.loads(failure_context_path.read_text(encoding="utf-8"))
    assert failure_context["failed_servo_id"] == 3
    assert failure_context["sample_index_at_failure"] == 0
    assert failure_context["retry_count"] == 2
    assert "Incorrect status packet" in failure_context["telemetry_error_code"]
    assert failure_context["failure_category"] == "servo_telemetry_packet_error"
    assert result.summary.experiment_metrics["failure_context"]["failed_servo_id"] == 3
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["recommendation"] == "failed_due_to_telemetry"
    assert quality["failure_count"] == 1
    assert int(quality["unrecovered_packet_error_count"]) >= 1
    assert int(quality["servo_telemetry_retry_count"]) >= 2
    summary_disk = json.loads((result.paths.output_dir / "summary.json").read_text(encoding="utf-8"))
    em = summary_disk["experiment_metrics"]
    assert int(em.get("unrecovered_packet_error_count", 0)) >= 1
    assert int(em.get("servo_telemetry_retry_count", 0)) >= 2
    assert int(em.get("write_goal_packet_error_count", 0)) == 0


def test_collect_pose_long_run_recovery_drops_one_sample_and_continues(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMotionCorruptBudgetBus(
        failed_servo_id=3,
        corrupt_budget=3,
        corrupt_after_goal_writes=2,
    )
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 2,
            "samples_per_command": 1,
            "telemetry_retry_count": 2,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "max_consecutive_packet_failures": 5,
            "max_total_packet_failures": 20,
            "resync_read_attempts": 2,
            "resync_delay_s": 0.0,
        },
    )
    assert result.success is True
    out = result.paths.output_dir
    events = [
        json.loads(line)
        for line in (out / "sample_failure_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(events) >= 1
    packet_events = [event for event in events if event.get("event") == "post_motion_telemetry_packet_error"]
    assert packet_events
    assert any(event.get("event") == "sample_quarantined" for event in events)
    assert any(event.get("event") == "transport_burst_recovery_started" for event in events)
    assert any(event.get("event") == "transport_burst_resync_attempt" for event in events)
    assert any(event.get("event") == "transport_burst_resync_success" for event in events)
    assert int(packet_events[0].get("retry_count", 0) or 0) > 0
    metrics = result.summary.experiment_metrics
    assert int(metrics.get("dropped_post_motion_telemetry_samples", 0) or 0) >= 1
    assert int(metrics.get("accepted_sample_count", 0) or 0) >= 1
    assert int(metrics.get("consecutive_post_motion_packet_failures", -1)) == 0
    assert int(metrics.get("transport_burst_count", 0) or 0) == len(packet_events)
    dropped_reasons = [
        json.loads(line).get("extra", {}).get("capture_rejection_reason")
        for line in (out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Transport recovery may quarantine by event/log metrics without always emitting a capture-rejected sample row.
    assert (
        any(r == "post_motion_telemetry_packet_error" for r in dropped_reasons)
        or any(event.get("event") == "post_motion_telemetry_packet_error" for event in events)
    )
    export_lines = [
        ln for ln in (out / "modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert export_lines
    assert not any("post_motion_telemetry_packet_error" in ln for ln in export_lines)
    quality = json.loads((out / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    health = json.loads((out / "long_run_health.json").read_text(encoding="utf-8"))
    assert int(metrics.get("accepted_sample_count", 0) or 0) == int(quality.get("accepted_sample_count", 0) or 0)
    assert int(metrics.get("dropped_post_motion_telemetry_samples", 0) or 0) == int(
        quality.get("dropped_post_motion_telemetry_samples", 0) or 0
    )
    assert int(health.get("accepted_sample_count", 0) or 0) == int(quality.get("accepted_sample_count", 0) or 0)
    assert int(health.get("dropped_post_motion_telemetry_samples", 0) or 0) == int(
        quality.get("dropped_post_motion_telemetry_samples", 0) or 0
    )
    assert int(health.get("transport_burst_count", 0) or 0) == int(quality.get("transport_burst_count", 0) or 0)
    assert health.get("run_status") == "success"
    assert health.get("run_success") is True
    # F-2: post-motion drops are not "recoveries"; the sample is lost. The bus is resynced
    # for the next command, and the event is named bus_resynced_after_drop, not the old
    # misleading post_motion_telemetry_resync_success.
    event_names = {event.get("event") for event in events}
    assert "bus_resynced_after_drop" in event_names
    assert "post_motion_telemetry_resync_success" not in event_names
    bus_resynced = [event for event in events if event.get("event") == "bus_resynced_after_drop"]
    assert bus_resynced and bus_resynced[0].get("sample_recovered") is False
    assert int(metrics.get("recovered_packet_error_count", 0) or 0) == 0
    assert int(quality.get("recovered_packet_error_count", 0) or 0) == 0
    assert int(health.get("recovered_packet_error_count", 0) or 0) == 0


def test_collect_pose_long_run_recovery_stops_at_max_consecutive(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMotionCorruptBudgetBus(
        failed_servo_id=3,
        corrupt_budget=400,
        corrupt_after_goal_writes=2,
    )
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(
        translation_mm=[0.1, 0.0, 0.0], registration_path=registration_path, frame_number=1
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 5,
            "samples_per_command": 1,
            "telemetry_retry_count": 1,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "max_consecutive_packet_failures": 2,
            "transport_burst_resync_attempts": 1,
            "transport_burst_resync_delay_s": 0.0,
            "transport_burst_cooldown_s": 0.0,
        },
    )
    assert result.success is False
    assert (
        "max_consecutive_transport_bursts" in result.message
        or "exceeded max_consecutive_transport_bursts" in result.message
    )
    quality = json.loads((result.paths.output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    health = json.loads((result.paths.output_dir / "long_run_health.json").read_text(encoding="utf-8"))
    assert int(quality["unrecovered_packet_error_count"]) >= 1
    assert int(quality.get("consecutive_transport_burst_failures", 0) or 0) >= 1
    report_path = result.paths.output_dir / "transport_recovery_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "stop_reason" in report
    m = result.summary.experiment_metrics
    assert int(m.get("next_command_index_to_resume", -1)) >= 1
    assert health.get("run_status") != "running"
    assert health.get("run_success") is False
    assert int(health.get("next_command_index_to_resume", -1)) == int(m.get("next_command_index_to_resume", -2))
    events = [
        json.loads(line)
        for line in (result.paths.output_dir / "sample_failure_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("event") == "run_stop_budget_exceeded" for event in events)


def test_collect_pose_pre_motion_packet_error_retries_and_continues(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    original_command_displacement = servo_service.command_displacement
    injected = {"done": False}

    def _flaky_command_displacement(*args, **kwargs):
        if not injected["done"]:
            injected["done"] = True
            raise ServoTelemetryRetryError(
                "Simple single-segment experiment motion rejected: servo 4 blocked motion: Present Position is unavailable.",
                context={
                    "failure_category": "simple_experiment_motion_rejected",
                    "failure_reason": "servo 4 blocked motion: Present Position is unavailable.",
                    "failed_servo_id": 4,
                    "telemetry_error_code": "packet_or_status_error",
                    "missing_fields": {"4": ["present_position"]},
                    "last_valid_telemetry_by_servo": {
                        "4": {"telemetry_error_code": "packet_or_status_error", "present_position_ticks": None}
                    },
                    "command_metadata": {
                        "pre_motion_read_source": "experiment_owned_minimal_read_after_configuration",
                        "pre_motion_telemetry_profile": "minimal",
                    },
                },
            )
        return original_command_displacement(*args, **kwargs)

    servo_service.command_displacement = _flaky_command_displacement
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 2,
            "samples_per_command": 1,
            "telemetry_retry_count": 2,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "resync_read_attempts": 2,
            "resync_delay_s": 0.0,
        },
    )
    assert result.success is True
    out = result.paths.output_dir
    events_path = out / "sample_failure_events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event.get("event") == "pre_motion_telemetry_packet_error" for event in events)
    assert any(event.get("event") == "pre_motion_command_retry" for event in events)
    metrics = result.summary.experiment_metrics
    assert int(metrics.get("servo_telemetry_retry_count", 0) or 0) >= 1
    quality = json.loads((out / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    health = json.loads((out / "long_run_health.json").read_text(encoding="utf-8"))
    retries = int(metrics.get("servo_telemetry_retry_count", 0) or 0)
    assert retries == int(quality.get("servo_telemetry_retry_count", 0) or 0)
    assert retries == int(health.get("servo_telemetry_retry_count", 0) or 0)
    # F-2: a within-command retry that produces a usable sample is a real recovery.
    # The same counter must agree across summary / quality / health writers.
    recovered = int(metrics.get("recovered_packet_error_count", 0) or 0)
    assert recovered >= 1
    assert recovered == int(quality.get("recovered_packet_error_count", 0) or 0)
    assert recovered == int(health.get("recovered_packet_error_count", 0) or 0)


def test_collect_pose_workspace_boundary_rejection_skips_command_and_continues(tmp_path: Path) -> None:
    """A hard-bound rejection from the safety layer should skip the command, not abort the run.

    Reproduces the failure mode seen in run 20260515_010617 where one command produced a
    servo-1 target of 4142 (raw hardware range is [0, 4095]). The safety layer correctly
    rejected the motion, but the experiment treated it as fatal. With the workspace-boundary
    skip path in place, the run should continue.
    """

    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    original_command_displacement = servo_service.command_displacement
    injected = {"done": False}

    def _bound_rejecting_command_displacement(*args, **kwargs):
        if not injected["done"]:
            injected["done"] = True
            raise ServoTelemetryRetryError(
                "Simple single-segment experiment motion rejected after antagonistic-pair projection "
                "(parallel_single mirrored source servos 1->5, 2->6, 3->7, 4->8): "
                "hard bound rejection: servo 1 target 4142 exceeds the raw hardware range [0, 4095].",
                context={
                    "failure_category": "simple_experiment_motion_rejected",
                    "failure_reason": "hard bound rejection: servo 1 target 4142 exceeds the raw hardware range [0, 4095].",
                    "failed_servo_id": 1,
                },
            )
        return original_command_displacement(*args, **kwargs)

    servo_service.command_displacement = _bound_rejecting_command_displacement
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 2,
            "samples_per_command": 1,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 2,
            "telemetry_retry_count": 1,
            "telemetry_retry_delay_s": 0.0,
        },
    )
    assert result.success is True, result.message
    metrics = result.summary.experiment_metrics
    assert int(metrics.get("workspace_boundary_skip_count", 0) or 0) >= 1
    events_path = result.paths.output_dir / "sample_failure_events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(event.get("event") == "command_skipped_workspace_boundary" for event in events)


def test_collect_pose_pre_motion_packet_error_stops_after_failure_budget(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PreMotionMinimalCorruptBudgetBus(failed_servo_id=4, corrupt_budget=200, arm_after_minimal_reads=1)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    bus.corrupt_budget = 200
    bus._minimal_reads = 0
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 3,
            "samples_per_command": 1,
            "telemetry_retry_count": 1,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "resync_read_attempts": 1,
            "resync_delay_s": 0.0,
            "max_consecutive_packet_failures": 2,
        },
    )
    assert result.success is False
    assert "max_consecutive_packet_failures" in result.message


def test_collect_pose_continue_until_valid_samples_hits_target_with_drops(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMotionCorruptBudgetBus(failed_servo_id=3, corrupt_budget=1, corrupt_after_goal_writes=2)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 2,
            "samples_per_command": 1,
            "continue_until_valid_samples": True,
            "target_valid_sample_count": 6,
            "telemetry_retry_count": 1,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "resync_read_attempts": 2,
            "resync_delay_s": 0.0,
        },
    )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert int(metrics.get("complete_training_row_count", 0) or 0) >= 6
    assert int(metrics.get("target_valid_sample_count", 0) or 0) == 6
    assert int(metrics.get("remaining_complete_training_rows", 0) or 0) == 0
    assert int(metrics.get("target_valid_sample_count", 0) or 0) == 6
    export_rows = [
        json.loads(ln)
        for ln in (result.paths.output_dir / "modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert all(bool(row.get("accepted")) for row in export_rows)
    assert len(export_rows) == int(metrics.get("complete_training_row_count", 0) or 0)


def test_collect_pose_continue_until_valid_targets_complete_rows_not_accepted_rows(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    import continuum_robot.experiments.builtins as collect_pose_builtins

    original_capture = collect_pose_builtins.CollectPoseCommandDatasetExperiment._capture_dataset_sample
    injected_missing_capture_rows = {"count": 0}

    def _capture_with_incomplete_servo_feedback(self, *args, **kwargs):
        sample = original_capture(self, *args, **kwargs)
        phase = str(getattr(sample, "phase", "") or "")
        if (
            phase not in {"initial_neutral", "final_neutral"}
            and bool(sample.extra.get("capture_accepted"))
            and injected_missing_capture_rows["count"] < 2
        ):
            servo_feedback = dict(sample.extra.get("servo_feedback_at_capture", {}) or {})
            if servo_feedback:
                first_servo_id = sorted(servo_feedback, key=lambda value: int(value))[0]
                first_entry = dict(servo_feedback.get(first_servo_id, {}) or {})
                first_entry["present_position_ticks"] = None
                servo_feedback[first_servo_id] = first_entry
                sample.extra["servo_feedback_at_capture"] = servo_feedback
                injected_missing_capture_rows["count"] += 1
        return sample

    with patch.object(
        collect_pose_builtins.CollectPoseCommandDatasetExperiment,
        "_capture_dataset_sample",
        _capture_with_incomplete_servo_feedback,
    ):
        result = runner.run_experiment(
            "collect_pose_command_dataset",
            config={
                "dry_run": False,
                "sample_count_target": 2,
                "samples_per_command": 1,
                "continue_until_valid_samples": True,
                "target_valid_sample_count": 6,
                "max_total_attempts": 30,
            },
        )
    assert result.success is True
    metrics = result.summary.experiment_metrics
    assert int(metrics.get("complete_training_row_count", 0) or 0) >= 6
    assert int(metrics.get("accepted_sample_count", 0) or 0) > 6
    assert int(metrics.get("incomplete_accepted_workspace_row_count", 0) or 0) >= 2
    assert int(metrics.get("remaining_complete_training_rows", 0) or 0) == 0
    export_rows = [
        json.loads(ln)
        for ln in (result.paths.output_dir / "modeling_dataset_export.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(export_rows) == int(metrics.get("complete_training_row_count", 0) or 0)
    for row in export_rows:
        feedback = dict(row.get("servo_feedback_at_capture", {}) or {})
        assert feedback
        for servo_payload in feedback.values():
            assert dict(servo_payload or {}).get("present_position_ticks") is not None


def test_collect_pose_transport_counter_alignment_across_outputs(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMotionCorruptBudgetBus(failed_servo_id=3, corrupt_budget=2, corrupt_after_goal_writes=2)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 3,
            "samples_per_command": 1,
            "telemetry_retry_count": 0,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "transport_burst_cooldown_s": 0.0,
            "transport_burst_resync_delay_s": 0.0,
        },
    )
    assert result.success is True
    out = result.paths.output_dir
    metrics = result.summary.experiment_metrics
    quality = json.loads((out / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    health = json.loads((out / "long_run_health.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (out / "sample_failure_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    packet_events = [event for event in events if event.get("event") == "post_motion_telemetry_packet_error"]
    assert int(metrics.get("transport_burst_count", 0) or 0) == len(packet_events)
    assert int(metrics.get("transport_burst_count", 0) or 0) == int(quality.get("transport_burst_count", 0) or 0)
    assert int(metrics.get("transport_burst_count", 0) or 0) == int(health.get("transport_burst_count", 0) or 0)
    assert int(metrics.get("total_post_motion_packet_failure_events", 0) or 0) == len(packet_events)


def test_collect_pose_retry_same_command_after_resync_records_event(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMotionCorruptBudgetBus(failed_servo_id=3, corrupt_budget=1, corrupt_after_goal_writes=2)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 3,
            "samples_per_command": 1,
            "telemetry_retry_count": 0,
            "telemetry_retry_delay_s": 0.0,
            "long_run_recovery_enabled": True,
            "on_unrecovered_post_motion_telemetry": "drop_sample_and_resync",
            "retry_same_command_after_resync": True,
            "transport_burst_cooldown_s": 0.0,
            "transport_burst_resync_delay_s": 0.0,
        },
    )
    assert result.success is True
    events = [
        json.loads(line)
        for line in (result.paths.output_dir / "sample_failure_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event.get("event") == "command_retry_after_resync" for event in events)
    assert any(event.get("event") == "command_retry_after_resync_success" for event in events)


def test_collect_pose_high_current_warning_trips_threshold(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    high_current_bus = _HighCurrentBus(current_ma=700)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=high_current_bus)
    high_current_bus.arm()
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "max_current_warning_ma": 500,
        },
    )
    assert result.success is True
    quality = json.loads((result.paths.output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    assert int(quality.get("max_abs_current_ma", 0) or 0) >= 700
    assert int(quality.get("max_current_warning_ma", 0) or 0) == 500
    assert bool(quality.get("high_current_warning")) is True
    by_servo = dict(quality.get("max_abs_current_ma_by_servo", {}) or {})
    mean_by_servo = dict(quality.get("mean_abs_current_ma_by_servo", {}) or {})
    p95_by_servo = dict(quality.get("p95_abs_current_ma_by_servo", {}) or {})
    peak_by_servo = dict(quality.get("peak_current_sample_by_servo", {}) or {})
    vmin_by_servo = dict(quality.get("input_voltage_min_mv_by_servo", {}) or {})
    assert by_servo
    assert mean_by_servo
    assert p95_by_servo
    assert peak_by_servo
    assert vmin_by_servo
    assert all(int(value) >= 700 for value in by_servo.values())
    assert int(quality.get("max_abs_current_ma", 0) or 0) == max(int(value) for value in by_servo.values())
    for servo_id, peak in peak_by_servo.items():
        assert int(peak.get("abs_current_ma", 0) or 0) >= 700
        assert isinstance(peak.get("sequence_index"), int)
        assert isinstance(peak.get("step_index"), int)
        assert isinstance(peak.get("sample_index"), int)
        assert isinstance(peak.get("resolved_cable_command_cm"), list)
        assert isinstance(peak.get("command_at_peak_current_cm"), list)
        assert str(servo_id) in by_servo
    assert quality.get("transient_current_spike_count_by_servo") is not None
    assert quality.get("sustained_current_exceedance_count_by_servo") is not None


def test_collect_pose_transient_current_spike_drops_sample_and_continues(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMoveCurrentSequenceBus([851, 120, 120], baseline_ma=120)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 2,
            "samples_per_command": 1,
            "transient_current_spike_ma": 850,
            "sustained_jam_current_ma": 900,
            "sustained_jam_cycles": 3,
            "transient_spike_policy": "warn_drop_sample_continue",
            "current_spike_resync_enabled": True,
            "current_spike_cooldown_s": 0.0,
            "resync_read_attempts": 1,
            "resync_delay_s": 0.0,
        },
    )
    assert result.success is True
    quality = json.loads((result.paths.output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    transient = dict(quality.get("transient_current_spike_count_by_servo", {}) or {})
    sustained = dict(quality.get("sustained_current_exceedance_count_by_servo", {}) or {})
    assert int(transient.get("2", 0)) >= 1
    assert int(sustained.get("2", 0)) == 0
    assert int(quality.get("transient_current_spike_drop_count", 0) or 0) >= 1


def test_collect_pose_sustained_current_spike_stops_after_configured_cycles(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _PostMoveCurrentSequenceBus([851, 851, 851, 851], baseline_ma=120)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1)
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 5,
            "samples_per_command": 1,
            "transient_current_spike_ma": 850,
            "sustained_jam_current_ma": 850,
            "sustained_jam_cycles": 3,
            "transient_spike_policy": "warn_drop_sample_continue",
            "sustained_jam_policy": "stop_safely",
            "current_spike_resync_enabled": False,
            "current_spike_cooldown_s": 0.0,
        },
    )
    assert result.success is False
    assert "sustained overcurrent/jam detected" in str(result.message).lower()


def test_collect_pose_ramp_splits_large_step_and_records_ramp_metadata(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = True
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    snap = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0],
        registration_path=registration_path,
        frame_number=1,
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": True,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "workspace_amplitude_cm": 1.0,
            "envelope_utilization": 0.75,
            "command_transition_ramp_enabled": True,
            "max_delta_cm_per_ramp_step": 0.10,
            "ramp_step_settle_s": 0.0,
        },
    )
    assert result.success is True
    bundle = runner.load_dataset(result.paths.output_dir)
    accepted_samples = [sample for sample in bundle.samples if bool(sample.extra.get("capture_accepted"))]
    non_neutral = [sample for sample in accepted_samples if sample.phase not in {"initial_neutral", "final_neutral"}]
    assert non_neutral
    ramp_meta = dict(non_neutral[0].extra.get("command_metadata", {}).get("ramp", {}) or {})
    assert int(ramp_meta.get("ramp_step_count", 1) or 1) >= 1
    assert "resolved_cable_command_cm" in non_neutral[0].extra


def test_collect_pose_write_goal_persistent_failure_classifies_write_goal_error(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    bus = _AlwaysFailExperimentWriteBus(arm_after_goal_writes=1)
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    snap = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    import continuum_robot.experiments.builtins as collect_pose_builtins

    def _noop_finalize(self, session):
        return

    with patch.object(
        collect_pose_builtins.CollectPoseCommandDatasetExperiment,
        "finalize",
        _noop_finalize,
    ):
        result = runner.run_experiment(
            "collect_pose_command_dataset",
            config={
                "dry_run": False,
                "sample_count_target": 1,
                "samples_per_command": 1,
                "goal_write_retry_attempts": 2,
            },
        )
    assert result.success is False
    failure = json.loads((result.paths.output_dir / "failure_context.json").read_text(encoding="utf-8"))
    assert failure.get("failure_category") == "write_goal_packet_error"
    quality = json.loads((result.paths.output_dir / "dataset_quality_summary.json").read_text(encoding="utf-8"))
    assert quality["recommendation"] == "failed_due_to_write_goal_packet_error"
    assert int(quality.get("write_goal_packet_error_count", 0)) >= 1


def test_collect_pose_chunk_checkpoint_and_long_run_health_fields(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    snap = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0], registration_path=registration_path, frame_number=1
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snap]),
        registration_path=registration_path,
        servo_service=servo_service,
    )
    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 2,
            "samples_per_command": 1,
            "chunk_flush_every_n_commands": 1,
            "long_run_health_write_interval_samples": 1,
        },
    )
    assert result.success is True
    ck = json.loads((result.paths.output_dir / "collect_pose_checkpoint.json").read_text(encoding="utf-8"))
    assert ck.get("schema_version") == "collect_pose_checkpoint_v1"
    assert int(ck.get("next_command_index_to_resume", -1)) >= 1
    health = json.loads((result.paths.output_dir / "long_run_health.json").read_text(encoding="utf-8"))
    assert health.get("schema_version") == "collect_pose_long_run_health_v1"
    assert "max_abs_current_ma" in health
    assert "tracker_stale_count" in health
    assert "complete_training_row_count" in health
    assert "target_valid_sample_count" in health
    assert "remaining_complete_training_rows" in health


def test_collect_pose_command_dataset_blocks_without_tracker_when_override_disabled(tmp_path: Path) -> None:
    settings = _settings()
    servo_service = _ready_modeling_servo_service(tmp_path)
    runner = ExperimentRunner(
        project_root=Path(__file__).resolve().parents[1],
        settings=settings,
        tracking_service=None,
        servo_service=servo_service,
        output_dir=tmp_path / "data" / "experiments",
        default_settle_time_s=0.0,
        registration_path=tmp_path / "latest_registration.json",
        sleep_fn=lambda _seconds: None,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
        },
    )

    assert result.success is False
    assert "requires tracking_service" in result.message


def test_collect_pose_command_dataset_rejects_stale_tracker_samples(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    stale_snapshot = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0],
        registration_path=registration_path,
        frame_number=1,
        tracker_age_s=0.6,
        tracker_stale=True,
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([stale_snapshot, stale_snapshot, stale_snapshot]),
        registration_path=registration_path,
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "capture_timeout_s": 0.001,
            "capture_poll_interval_s": 0.0,
            "max_tracker_age_s": 0.05,
        },
    )

    assert result.success is False
    bundle = runner.load_dataset(result.paths.output_dir)
    assert bundle.summary.status == "invalid_due_to_insufficient_samples"
    rejected = [sample for sample in bundle.samples if sample.extra.get("capture_accepted") is False]
    assert rejected
    assert any("tracker_data_stale" in sample.extra.get("capture_rejection_reason", "") for sample in rejected)


def test_collect_pose_command_dataset_blocks_lower_trust_runtime_tip_by_default(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(json.dumps({"T_robot_aurora": np.eye(4).tolist()}), encoding="utf-8")
    servo_service = _ready_modeling_servo_service(tmp_path)
    snapshot = _trusted_modeling_snapshot(
        translation_mm=[0.0, 0.0, 0.0],
        registration_path=registration_path,
        frame_number=1,
        runtime_tip_mode="latest_accepted",
    )
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_SequencedTrackingService([snapshot]),
        registration_path=registration_path,
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
        },
    )

    assert result.success is False
    assert "runtime tip policy" in result.message.lower()
    bundle = runner.load_dataset(result.paths.output_dir)
    assert bundle.summary.status == "invalid_due_to_insufficient_samples"


def test_collect_pose_command_dataset_servo_precheck_passes_with_stale_gui_telemetry_cache(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    servo_service = _ready_modeling_servo_service(tmp_path)
    _poison_last_known_positions_for_gui_stale_cache(servo_service, [1, 2, 3, 4])
    assert servo_service.last_known_telemetry([1])[1].present_position is None
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_DisconnectedTrackingService(),
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dataset_mode": "workspace_coverage",
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "allow_no_tracker_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is True


def test_collect_pose_command_dataset_servo_precheck_failure_writes_precheck_failure_context(tmp_path: Path) -> None:
    settings = _settings()
    settings.runtime.mock_mode = False
    bus = _MinimalStripPositionBus()
    servo_service = _ready_modeling_servo_service(tmp_path, dxl_bus=bus)
    bus.arm_faulting_live_reads()
    runner = _runner(
        tmp_path,
        settings=settings,
        tracking_service=_DisconnectedTrackingService(),
        servo_service=servo_service,
    )

    result = runner.run_experiment(
        "collect_pose_command_dataset",
        config={
            "dry_run": False,
            "sample_count_target": 1,
            "samples_per_command": 1,
            "allow_no_tracker_test_run": True,
            "run_trust_mode": "servo_only",
        },
    )

    assert result.success is False
    assert "Servo precheck failed" in result.message
    assert "Present Position is unavailable" in result.message
    ctx_path = result.paths.output_dir / "precheck_failure_context.json"
    assert ctx_path.exists()
    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    assert ctx["experiment"] == "collect_pose_command_dataset"
    assert ctx["failed_servo_id"] == 1
    assert "present_position" in ctx["missing_fields"]
    assert ctx["fresh_precheck_read_attempted"] is True
    assert ctx["telemetry_read_source"] == "experiment_owned"
