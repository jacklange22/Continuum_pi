"""Tests for the Servos-tab pretension trial controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json

import pytest

from continuum_robot.config.schemas import (
    CalibrationConfig,
    ExperimentConfig,
    RegistrationWorkflowConfig,
    RobotConfig,
    RuntimeConfig,
    SafetyConfig,
    SerialConfig,
)
from continuum_robot.config.settings import Settings
from continuum_robot.gui.controllers.pretension_trial_controller import (
    PretensionTrialController,
)


# --- minimal stubs ---------------------------------------------------------


@dataclass
class _StubTelemetryEntry:
    present_position: int | None
    present_current_ma: float | None


class _StubServoService:
    """Just enough surface for the trial controller's manual baseline path."""

    def __init__(self, *, telemetry: dict[int, _StubTelemetryEntry], connected: bool = True) -> None:
        self._telemetry = dict(telemetry)
        self.is_connected = bool(connected)
        self.read_live_telemetry_calls: list[list[int]] = []

    def read_live_telemetry(self, servo_ids: list[int]) -> dict[int, _StubTelemetryEntry]:
        self.read_live_telemetry_calls.append(list(servo_ids))
        return {int(sid): self._telemetry[int(sid)] for sid in servo_ids if int(sid) in self._telemetry}


class _StubTrackingService:
    def __init__(self, *, tip_xyz: tuple[float, float, float] | None = (1.0, -0.5, 12.3)) -> None:
        if tip_xyz is None:
            self._snapshot = SimpleNamespace(T_robot_tip=None)
        else:
            x, y, z = tip_xyz
            self._snapshot = SimpleNamespace(
                T_robot_tip=[
                    [1.0, 0.0, 0.0, float(x)],
                    [0.0, 1.0, 0.0, float(y)],
                    [0.0, 0.0, 1.0, float(z)],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

    def peek_snapshot(self):
        return self._snapshot

    def get_snapshot(self):
        return self._snapshot


@dataclass
class _StubResultPaths:
    output_dir: Path


@dataclass
class _StubResultSummary:
    experiment_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubExperimentResult:
    success: bool
    message: str
    paths: _StubResultPaths
    summary: _StubResultSummary


class _StubExperimentRunner:
    def __init__(self, *, output_dir: Path, success: bool = True) -> None:
        self._output_dir = output_dir
        self._success = success
        self.last_kwargs: dict[str, Any] = {}

    def run_experiment(self, experiment_name: str, *, config=None, operator_notes: str = "", **kwargs) -> _StubExperimentResult:
        self.last_kwargs = {
            "experiment_name": experiment_name,
            "config": dict(config or {}),
            "operator_notes": str(operator_notes),
        }
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return _StubExperimentResult(
            success=self._success,
            message="ok" if self._success else "failed",
            paths=_StubResultPaths(output_dir=self._output_dir),
            summary=_StubResultSummary(
                experiment_metrics={
                    "accepted_run_count": 5,
                    "run_count": 5,
                    "pretension_comparison_report": {
                        "algorithm_population_summary": {
                            "tip_xy_error_to_target_mm": {"mean": 0.42},
                        }
                    },
                }
            ),
        )


def _settings() -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, poll_rate_hz=20, robot_config="robot_4servo.yaml"),
        robot=RobotConfig(
            mode="single_segment",
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4],
            tendon_to_servo=[1, 2, 3, 4],
            tightening_rotation_by_servo={1: "cw", 2: "cw", 3: "cw", 4: "cw"},
        ),
        serial=SerialConfig(aurora_port="/dev/mock", openrb_port="/dev/mock", baudrate=115200),
        safety=SafetyConfig(
            position_min_offset_ticks=-600,
            position_max_offset_ticks=600,
            max_current_ma=850,
            pretension_current_balance_tolerance_ma=15,
        ),
        registration=RegistrationWorkflowConfig(capture_tool_id="0B", coil_tool_id="0A", max_fre_mm=None),
        experiment=ExperimentConfig(default_settle_time_s=0.0, sample_count_per_point=1, output_dir="data/experiments"),
        calibration=CalibrationConfig(
            neutral_setpoints_path="config/neutral_setpoints.json",
            latest_registration_path="data/registrations/latest_registration.json",
        ),
    )


def _build_controller(tmp_path: Path, *, with_tracker: bool = True, run_success: bool = True):
    settings = _settings()
    servos = _StubServoService(
        telemetry={
            1: _StubTelemetryEntry(present_position=2500, present_current_ma=28.0),
            2: _StubTelemetryEntry(present_position=2510, present_current_ma=30.0),
            3: _StubTelemetryEntry(present_position=2495, present_current_ma=32.0),
            4: _StubTelemetryEntry(present_position=2505, present_current_ma=29.0),
        }
    )
    tracker = _StubTrackingService() if with_tracker else None
    runner = _StubExperimentRunner(output_dir=tmp_path / "data" / "experiments" / "pretension_validation" / "stub_run", success=run_success)
    return PretensionTrialController(
        servo_service=servos,
        tracking_service=tracker,
        experiment_runner=runner,
        settings=settings,
        project_root=tmp_path,
        manual_baseline_path="data/diagnostics/pretension_manual_baselines.json",
    )


# --- manual baseline path -------------------------------------------------


def test_record_manual_baseline_captures_positions_currents_and_tip(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    state = ctrl.record_manual_baseline(note="hand-tensioned slowly")
    assert state.manual_baseline_count == 1
    record = state.manual_baseline_records[0]
    assert record["positions_by_servo"][1] == 2500
    assert record["currents_ma_by_servo"][1] == pytest.approx(28.0)
    assert record["tip_xy_mm"] == pytest.approx([1.0, -0.5])
    assert record["note"] == "hand-tensioned slowly"
    # File got written.
    path = tmp_path / "data" / "diagnostics" / "pretension_manual_baselines.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1


def test_record_manual_baseline_appends_across_calls(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    ctrl.record_manual_baseline()
    ctrl.record_manual_baseline()
    ctrl.record_manual_baseline()
    assert ctrl.state.manual_baseline_count == 3
    path = tmp_path / "data" / "diagnostics" / "pretension_manual_baselines.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 3
    assert [r["index"] for r in payload["records"]] == [0, 1, 2]


def test_record_manual_baseline_errors_if_disconnected(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    ctrl.servo_service.is_connected = False
    with pytest.raises(RuntimeError, match="not connected"):
        ctrl.record_manual_baseline()


def test_record_manual_baseline_without_tracker_records_xy_none(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path, with_tracker=False)
    state = ctrl.record_manual_baseline()
    record = state.manual_baseline_records[0]
    assert record["tip_xy_mm"] is None
    assert record["tip_xyz_mm"] is None


def test_clear_manual_baselines_empties_file(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    ctrl.record_manual_baseline()
    ctrl.record_manual_baseline()
    state = ctrl.clear_manual_baselines()
    assert state.manual_baseline_count == 0
    path = tmp_path / "data" / "diagnostics" / "pretension_manual_baselines.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["records"] == []


def test_controller_reloads_existing_records_on_start(tmp_path: Path) -> None:
    # Pre-seed a baselines file.
    diag = tmp_path / "data" / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    pre = {"records": [{"index": 0, "positions_by_servo": {1: 2500}}]}
    (diag / "pretension_manual_baselines.json").write_text(json.dumps(pre), encoding="utf-8")
    ctrl = _build_controller(tmp_path)
    assert ctrl.state.manual_baseline_count == 1


# --- run trial path -------------------------------------------------------


def test_run_pretension_trial_invokes_experiment_with_active_servo_ids(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    # Write a YAML config that has empty servo_ids so the controller fills them.
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "experiment_pretension_validation.example.yaml").write_text(
        "mode: single_segment_staged\nstaged_strategy: conservative_startup\nservo_ids: []\nrepeat_runs: 5\n",
        encoding="utf-8",
    )
    state = ctrl.run_pretension_trial()
    kwargs = ctrl.experiment_runner.last_kwargs
    assert kwargs["experiment_name"] == "pretension_validation"
    cfg = kwargs["config"]
    assert sorted(cfg["servo_ids"]) == [1, 2, 3, 4]
    assert cfg["repeat_runs"] == 5
    assert state.last_run_accepted is True
    assert state.last_run_output_dir is not None


def test_run_pretension_trial_wires_manual_baseline_path_when_records_exist(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    ctrl.record_manual_baseline()
    state = ctrl.run_pretension_trial()
    cfg = ctrl.experiment_runner.last_kwargs["config"]
    assert "manual_baseline_record_path" in cfg
    # When the GUI captured the baselines we don't want the inline capture
    # phase to also fire inside the experiment.
    assert int(cfg.get("manual_baseline_capture_count", 0)) == 0


def test_run_pretension_trial_errors_when_disconnected(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path)
    ctrl.servo_service.is_connected = False
    with pytest.raises(RuntimeError, match="not connected"):
        ctrl.run_pretension_trial()


def test_run_pretension_trial_propagates_runner_failure(tmp_path: Path) -> None:
    ctrl = _build_controller(tmp_path, run_success=False)
    state = ctrl.run_pretension_trial()
    assert state.last_run_accepted is False
    assert state.last_error is not None


# --- Tightening-sign verification tests -----------------------------------


class _SignTestServoService:
    """Stub that simulates servo current responding to commanded-tick direction.

    Each servo has a `sign` parameter:
    - +1 (algorithm-consistent): decreasing tick raises current
    - -1 (inverted): increasing tick raises current
    - 0 (low response): current never changes
    """

    def __init__(self, *, sign_by_servo: dict[int, int], baseline_current_ma: float = 25.0, swing_ma: float = 6.0) -> None:
        self._sign = dict(sign_by_servo)
        self._baseline_current = float(baseline_current_ma)
        self._swing = float(swing_ma)
        self._positions: dict[int, int] = {int(sid): 2500 for sid in sign_by_servo}
        self._current_offsets: dict[int, float] = {int(sid): 0.0 for sid in sign_by_servo}
        self.is_connected = True
        self._sleep_fn = lambda _s: None
        self._goal_writes: list[dict[int, int]] = []

    def read_live_telemetry(self, servo_ids: list[int]):
        from dataclasses import dataclass

        @dataclass
        class _Entry:
            present_position: int | None
            present_current_ma: float | None

        out = {}
        for sid in servo_ids:
            sid = int(sid)
            out[sid] = _Entry(
                present_position=self._positions.get(sid, 2500),
                present_current_ma=self._baseline_current + self._current_offsets.get(sid, 0.0),
            )
        return out

    def _write_goal_positions(self, positions_by_id):
        self._goal_writes.append(dict(positions_by_id))
        for sid, target in positions_by_id.items():
            sid = int(sid)
            prev = self._positions.get(sid, 2500)
            delta = int(target) - int(prev)
            self._positions[sid] = int(target)
            # Sign convention: +1 sign means "decreasing tick raises current".
            # delta < 0 (tighten) under sign=+1 => offset = +swing.
            # delta > 0 (release) under sign=+1 => offset = -swing.
            sign = self._sign.get(sid, 0)
            if delta == 0:
                self._current_offsets[sid] = 0.0
            elif sign > 0:
                self._current_offsets[sid] = -self._swing if delta > 0 else self._swing
            elif sign < 0:
                self._current_offsets[sid] = self._swing if delta > 0 else -self._swing
            else:
                self._current_offsets[sid] = 0.0


def _build_sign_test_controller(tmp_path: Path, *, sign_by_servo: dict[int, int]):
    settings = _settings()
    servos = _SignTestServoService(sign_by_servo=sign_by_servo)
    runner = _StubExperimentRunner(output_dir=tmp_path / "out")
    return PretensionTrialController(
        servo_service=servos,
        tracking_service=None,
        experiment_runner=runner,
        settings=settings,
        project_root=tmp_path,
    )


def test_verify_tightening_signs_all_consistent(tmp_path: Path) -> None:
    ctrl = _build_sign_test_controller(tmp_path, sign_by_servo={1: 1, 2: 1, 3: 1, 4: 1})
    report = ctrl.verify_tightening_signs(probe_ticks=5, settle_s=0.0)
    assert report["all_consistent"] is True
    assert report["any_inverted"] is False
    assert "OK" in report["overall_verdict"]
    assert all(row["verdict"] == "consistent" for row in report["per_servo"])


def test_verify_tightening_signs_detects_inverted_servo(tmp_path: Path) -> None:
    # Servo 3 is wired backwards: decreasing tick should release, increasing should tighten.
    ctrl = _build_sign_test_controller(tmp_path, sign_by_servo={1: 1, 2: 1, 3: -1, 4: 1})
    report = ctrl.verify_tightening_signs(probe_ticks=5, settle_s=0.0)
    assert report["all_consistent"] is False
    assert report["any_inverted"] is True
    assert "STOP" in report["overall_verdict"]
    verdicts = {row["servo_id"]: row["verdict"] for row in report["per_servo"]}
    assert verdicts[1] == "consistent"
    assert verdicts[2] == "consistent"
    assert verdicts[3] == "inverted"
    assert verdicts[4] == "consistent"


def test_verify_tightening_signs_flags_low_response(tmp_path: Path) -> None:
    # All servos report flat current (slack spine).
    ctrl = _build_sign_test_controller(tmp_path, sign_by_servo={1: 0, 2: 0, 3: 0, 4: 0})
    report = ctrl.verify_tightening_signs(probe_ticks=5, settle_s=0.0)
    assert report["all_consistent"] is False
    assert report["any_inverted"] is False
    assert "REVIEW" in report["overall_verdict"]
    assert all(row["verdict"] == "low_response" for row in report["per_servo"])


def test_verify_tightening_signs_errors_when_disconnected(tmp_path: Path) -> None:
    ctrl = _build_sign_test_controller(tmp_path, sign_by_servo={1: 1, 2: 1, 3: 1, 4: 1})
    ctrl.servo_service.is_connected = False
    with pytest.raises(RuntimeError, match="not connected"):
        ctrl.verify_tightening_signs(probe_ticks=5)


def test_verify_tightening_signs_restores_start_tick(tmp_path: Path) -> None:
    """After the probe, every servo must be back at its starting tick."""
    ctrl = _build_sign_test_controller(tmp_path, sign_by_servo={1: 1, 2: 1, 3: 1, 4: 1})
    start_positions = {sid: ctrl.servo_service._positions[sid] for sid in (1, 2, 3, 4)}
    report = ctrl.verify_tightening_signs(probe_ticks=5, settle_s=0.0)
    for sid, start in start_positions.items():
        assert ctrl.servo_service._positions[sid] == start, f"servo {sid} not restored to {start}"
