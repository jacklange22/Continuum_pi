"""Experiment loop coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from continuum_robot.experiments.dat_writer import DatRunWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.services.models import ToolTrackingSnapshot
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.tip_pose_service import TipPoseService
from continuum_robot.tracking.transforms import make_transform_A_B


@dataclass
class ExperimentRunSummary:
    """Summary returned after a run completes."""

    output_path: Path
    rows_written: int
    message: str


class ExperimentRunner:
    """Coordinates point execution, settle, sampling, and logging."""

    def __init__(
        self,
        servo_service: ServoService,
        tracking_service,
        dat_writer: DatRunWriter,
        neutral_servo_ids: list[int],
        default_settle_time_s: float,
        registration_path: Path,
        sleep_fn=time.sleep,
    ) -> None:
        self.servo_service = servo_service
        self.tracking_service = tracking_service
        self.dat_writer = dat_writer
        self.neutral_servo_ids = list(neutral_servo_ids)
        self.default_settle_time_s = default_settle_time_s
        self.registration_path = registration_path
        self.sleep_fn = sleep_fn

    def run(
        self,
        points: list[ExperimentPoint],
        progress_callback=None,
        stop_requested=None,
    ) -> ExperimentRunSummary:
        """Execute one experiment and write a `.dat` output."""
        if not self.servo_service.is_connected:
            raise RuntimeError("Servo bus is not connected. Connect OpenRB/DYNAMIXEL before running experiments.")
        neutral_map = self.servo_service.load_neutral_setpoints()
        if not neutral_map:
            raise RuntimeError("Neutral calibration is missing. Capture and save neutral setpoints first.")
        neutral_ticks = [neutral_map[sid] for sid in self.neutral_servo_ids if sid in neutral_map]
        if len(neutral_ticks) != len(self.neutral_servo_ids):
            raise RuntimeError("Neutral calibration does not cover all configured servo IDs.")
        if not self.registration_path.exists():
            raise RuntimeError("Registration is missing. Complete registration before running experiments.")

        tip_service = TipPoseService.from_registration_file(self.registration_path)
        rows: list[dict] = []
        total = sum(max(1, point.repeat) for point in points)
        completed = 0

        for point in points:
            for repeat_index in range(max(1, point.repeat)):
                if stop_requested is not None and stop_requested():
                    raise RuntimeError("Experiment run stopped by operator.")

                command = self.servo_service.command_displacement(
                    tendon_displacements_cm=point.tendon_displacement_cm,
                    neutral_ticks=neutral_ticks,
                    servo_ids=self.neutral_servo_ids,
                )
                settle_time = point.settle_time_s if point.settle_time_s is not None else self.default_settle_time_s
                if settle_time > 0:
                    self.sleep_fn(settle_time)

                tool_0a = self.tracking_service.get_latest_tool("0A")
                tool_0b = self.tracking_service.get_latest_tool("0B")
                if tool_0a is None:
                    raise RuntimeError("Tracker sample for tool 0A is unavailable during experiment run.")
                if tool_0a.tracking_state != "tracked":
                    raise RuntimeError(f"Tracker sample for tool 0A is invalid: {tool_0a.status}")
                row = self._build_row(
                    point=point,
                    repeat_index=repeat_index,
                    tool_0a=tool_0a,
                    tool_0b=tool_0b,
                    tip_service=tip_service,
                    command=command.positions_by_id,
                    telemetry=command.telemetry_by_id,
                )
                rows.append(row)

                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total, row)

        output_path = self.dat_writer.write_run(
            num_cables=len(self.neutral_servo_ids),
            rows=rows,
        )
        return ExperimentRunSummary(
            output_path=output_path,
            rows_written=len(rows),
            message=f"Completed {len(rows)} measurement row(s).",
        )

    def _build_row(
        self,
        point: ExperimentPoint,
        repeat_index: int,
        tool_0a: ToolTrackingSnapshot | None,
        tool_0b: ToolTrackingSnapshot | None,
        tip_service: TipPoseService,
        command: dict[int, int],
        telemetry: dict[int, object],
    ) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        tip_position_xyz = [float("nan"), float("nan"), float("nan")]
        tip_tangent_xyz = [float("nan"), float("nan"), float("nan")]
        if tool_0a is not None and tool_0a.tracking_state == "tracked":
            T_robot_tip = tip_service.compute_T_robot_tip(
                T_robot_aurora=tip_service.inputs.T_robot_aurora,
                T_aurora_coil=make_transform_A_B(tool_0a.quaternion_wxyz, tool_0a.translation_mm),
                T_coil_tip=tip_service.inputs.T_coil_tip,
            )
            tip_position_xyz = [float(v) for v in T_robot_tip[0:3, 3]]
            tip_tangent_xyz = [float(v) for v in T_robot_tip[0:3, 2]]

        def _tool_translation(tool: ToolTrackingSnapshot | None) -> list[float]:
            if tool is None:
                return [float("nan"), float("nan"), float("nan")]
            return [float(v) for v in tool.translation_mm]

        def _sorted_values(attr: str) -> list[float]:
            values = []
            for servo_id in self.neutral_servo_ids:
                item = telemetry[servo_id]
                value = getattr(item, attr)
                values.append(float(value) if value is not None else float("nan"))
            return values

        return {
            "timestamp_utc": timestamp,
            "index": point.index,
            "repeat_index": repeat_index,
            "commanded_displacement_cm": [float(v) for v in point.tendon_displacement_cm],
            "commanded_goal_ticks": [int(command[sid]) for sid in self.neutral_servo_ids],
            "servo_position_ticks": _sorted_values("present_position"),
            "servo_current_ma": _sorted_values("present_current_ma"),
            "servo_voltage_mv": _sorted_values("present_voltage_mv"),
            "tool_0A_translation_mm": _tool_translation(tool_0a),
            "tool_0B_translation_mm": _tool_translation(tool_0b),
            "tip_position_xyz": tip_position_xyz,
            "tip_tangent_xyz": tip_tangent_xyz,
        }
