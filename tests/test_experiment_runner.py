import json
from pathlib import Path

import numpy as np

from continuum_robot.experiments.dat_writer import DatRunWriter
from continuum_robot.experiments.experiment_models import ExperimentPoint
from continuum_robot.experiments.experiment_runner import ExperimentRunner
from continuum_robot.hardware.mock_dxl_bus import MockDxlBus
from continuum_robot.services.tracking_service import TrackingService
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper
from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService
from continuum_robot.servos.pretension_validation_service import PretensionValidationService
from continuum_robot.servos.safety_guard import SafetyGuard
from continuum_robot.servos.servo_service import ServoService
from continuum_robot.tracking.mock_tracker_manager import MockTrackerManager


def test_experiment_runner_writes_one_dat_file_per_run(tmp_path: Path) -> None:
    registration_path = tmp_path / "latest_registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "T_robot_aurora": np.eye(4).tolist(),
                "T_coil_tip": np.eye(4).tolist(),
            }
        ),
        encoding="utf-8",
    )

    servo_service = ServoService(
        dxl_bus=MockDxlBus([1, 2, 3, 4]),
        mapper=TendonDisplacementMapper(spool_diameter_cm=1.2),
        safety_guard=SafetyGuard(min_offset_ticks=-600, max_offset_ticks=600, max_current_ma=850),
        neutral_calibration=NeutralCalibrationService(path=tmp_path / "neutral.json"),
        pretension_validation=PretensionValidationService(),
    )
    servo_service.connect("/dev/mock-openrb", 115200)
    servo_service.save_neutral_setpoints(servo_service.capture_neutral_setpoints([1, 2, 3, 4]))

    tracking_service = TrackingService(
        live_backend=MockTrackerManager(poll_hz=10),
        port="/dev/mock-aurora",
        registration_path=registration_path,
        config_source="test",
    )
    tracking_service.start()
    try:
        runner = ExperimentRunner(
            servo_service=servo_service,
            tracking_service=tracking_service,
            dat_writer=DatRunWriter(tmp_path / "runs"),
            neutral_servo_ids=[1, 2, 3, 4],
            default_settle_time_s=0.0,
            registration_path=registration_path,
            sleep_fn=lambda _seconds: None,
        )
        summary = runner.run([ExperimentPoint(index=0, tendon_displacement_cm=[0.0, 0.1, -0.1, 0.0])])
    finally:
        tracking_service.stop()

    text = summary.output_path.read_text(encoding="utf-8")
    assert summary.rows_written == 1
    assert summary.output_path.exists()
    assert "NUM_MEASUREMENTS: 1" in text
    assert "0,0,0.0,0.1,-0.1,0.0" in text
