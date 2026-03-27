import json
from pathlib import Path

from continuum_robot.servos.neutral_calibration_service import NeutralCalibrationService


def test_neutral_calibration_service_archives_previous_latest(tmp_path: Path) -> None:
    service = NeutralCalibrationService(path=tmp_path / "neutral_setpoints.json")

    service.save_neutral_setpoints({1: 100})
    service.save_neutral_setpoints({1: 200})

    latest = json.loads((tmp_path / "neutral_setpoints.json").read_text(encoding="utf-8"))
    archives = sorted(tmp_path.glob("neutral_setpoints_*.json"))

    assert latest == {"1": 200}
    assert len(archives) == 1
    assert json.loads(archives[0].read_text(encoding="utf-8")) == {"1": 100}
