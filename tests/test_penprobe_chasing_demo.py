from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from continuum_robot.experiments.builtins import register_builtin_experiments
from continuum_robot.experiments.framework import ExperimentContext, ExperimentSession
from continuum_robot.experiments.penprobe_chasing_demo import (
    MAPPING_LEGACY_POLYNOMIAL_WORKSPACE,
    MAPPING_PAIRED_XY_PROPORTIONAL,
    PenprobeChasingDemoConfig,
    PenprobeChasingDemoExperiment,
    _clamp_tick_deltas_to_startup_raw_bounds,
    _extract_robot_frame_tool_pose,
    _resolve_mapping_mode,
    _step_and_clamp_tick_deltas,
    paired_xy_proportional_tick_request,
)
from continuum_robot.experiments.registry import ExperimentRegistry
from continuum_robot.experiments.schemas import ExperimentMetadata
from continuum_robot.hardware.dxl_bus import ServoTelemetry
from continuum_robot.servos.displacement_mapper import TendonDisplacementMapper


def _settings(active_segment: str = "segment_a", *, mode: str = "single_segment") -> Settings:
    return Settings(
        runtime=RuntimeConfig(mock_mode=True, robot_config="robot_8servo.yaml"),
        robot=RobotConfig(
            mode=mode,
            spool_diameter_cm=1.2,
            ticks_per_revolution=4096,
            servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            tendon_to_servo=[1, 2, 3, 4, 5, 6, 7, 8],
            active_segment=active_segment,
            segments={
                "segment_a": RobotSegmentConfig(
                    key="segment_a",
                    label="Spine 1",
                    servo_ids=[1, 2, 3, 4],
                    pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
                ),
                "segment_b": RobotSegmentConfig(
                    key="segment_b",
                    label="Spine 2",
                    servo_ids=[5, 6, 7, 8],
                    pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
                ),
            },
        ),
        serial=SerialConfig(),
        safety=SafetyConfig(),
        registration=RegistrationWorkflowConfig(),
        experiment=ExperimentConfig(),
        calibration=CalibrationConfig(),
    )


def _tool(tool_id: str, xyz: tuple[float, float, float], *, state: str = "tracked"):
    return SimpleNamespace(
        tool_id=tool_id,
        present=True,
        valid=True,
        tracking_state=state,
        translation_mm=xyz,
        frame_number=7,
        T_aurora_tool=[
            [1.0, 0.0, 0.0, float(xyz[0])],
            [0.0, 1.0, 0.0, float(xyz[1])],
            [0.0, 0.0, 1.0, float(xyz[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _snapshot(*, stale: bool = False, missing_0b: bool = False):
    tools = {
        "0A": _tool("0A", (1.0, 2.0, 3.0)),
    }
    if not missing_0b:
        tools["0B"] = _tool("0B", (11.0, 7.0, 4.0))
    return SimpleNamespace(
        tracker_data_stale=stale,
        tracker_data_age_s=0.01,
        T_robot_aurora=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        tools=tools,
        last_frame_number=7,
        normalized_live_tool_ids=list(tools),
        canonical_state="mock",
        registration_state="loaded",
    )


class _TrackingService:
    def __init__(self, snapshots):
        self._snapshots = list(snapshots)

    def get_snapshot(self):
        if len(self._snapshots) > 1:
            return self._snapshots.pop(0)
        return self._snapshots[0]


class _PretensionSummary:
    accepted = True
    usable = True
    message = "accepted startup reference"

    def __init__(self, servo_ids):
        self.positions_by_servo = {int(servo_id): 2000 + int(servo_id) for servo_id in servo_ids}


class _CalibrationSummary:
    exists = True
    compatible = True
    message = "ok"

    def __init__(self, servo_ids):
        self.servo_ids = list(servo_ids)

    def pretension_source_summary(self, servo_ids):
        if any(int(servo_id) not in self.servo_ids for servo_id in servo_ids):
            return SimpleNamespace(accepted=False, usable=False, message="wrong segment", positions_by_servo={})
        return _PretensionSummary(servo_ids)


class _SafetyGuard:
    def validate_telemetry_freshness(self, value):
        assert value is not None

    def validate_currents(self, values, *, require_present):
        assert not require_present or all(value is not None for value in values)

    def validate_voltage(self, value, *, require_present):
        assert not require_present or value is not None

    def validate_temperature(self, value, *, require_present):
        assert not require_present or value is not None

    def telemetry_age_s(self, _last_read):
        return 0.0


class _ServoService:
    is_connected = True

    def __init__(self, servo_ids):
        self.servo_ids = list(servo_ids)
        self.neutral_calibration = SimpleNamespace(
            get_calibration_summary=lambda: _CalibrationSummary(self.servo_ids)
        )
        self.safety_guard = _SafetyGuard()
        self.commanded_servo_ids = []

    def read_live_telemetry(self, servo_ids):
        return {
            int(servo_id): ServoTelemetry(
                servo_id=int(servo_id),
                present_position=2000 + int(servo_id),
                present_current_ma=20,
                present_voltage_mv=7400,
                present_temperature_c=30,
                hardware_error_code=0,
                hardware_error=None,
                last_read_monotonic_s=1.0,
            )
            for servo_id in servo_ids
        }

    @contextmanager
    def exclusive_bus_operation(self, *, owner, reason):
        assert owner == "penprobe_chasing_demo"
        assert reason
        yield

    def command_displacement(self, tendon_displacements_cm, neutral_ticks, servo_ids, *, motion_workflow):
        self.commanded_servo_ids = [int(value) for value in servo_ids]
        telemetry = self.read_live_telemetry(servo_ids)
        return SimpleNamespace(
            positions_by_id={int(servo_id): int(neutral_ticks[index]) for index, servo_id in enumerate(servo_ids)},
            telemetry_by_id=telemetry,
            clamp_reasons_by_id={},
        )

    def telemetry_age_s(self, telemetry):
        return self.safety_guard.telemetry_age_s(telemetry.last_read_monotonic_s)


def _session(*, active_segment: str = "segment_a", mode: str = "single_segment", tracking=None, servo=None):
    settings = _settings(active_segment=active_segment, mode=mode)
    clock = {"value": 0.0}

    def _monotonic():
        clock["value"] += 0.01
        return clock["value"]

    return ExperimentSession(
        context=ExperimentContext(
            project_root=Path.cwd(),
            settings=settings,
            tracking_service=tracking or _TrackingService([_snapshot()]),
            servo_service=servo or _ServoService(settings.robot.operating_context().expected_servo_ids),
            registration_path=Path("registration.json"),
            output_root=Path("data"),
            monotonic_fn=_monotonic,
            sleep_fn=lambda _seconds: None,
        ),
        metadata=ExperimentMetadata(
            schema_version="1.0",
            experiment_name="penprobe_chasing_demo",
            run_id="test",
            timestamp_utc="2026-01-01T00:00:00Z",
            git_commit=None,
            backend_info={},
            registration_info={},
            config_used={},
        ),
    )


def test_penprobe_chasing_demo_registered() -> None:
    registry = ExperimentRegistry()
    register_builtin_experiments(registry)

    descriptor = registry.get("penprobe_chasing_demo")

    assert descriptor.name == "penprobe_chasing_demo"
    assert descriptor.default_config_path == "config/experiment_penprobe_chasing_demo.example.yaml"


@pytest.mark.parametrize("mode", ["one_servo", "dual_segment", "parallel_single"])
def test_penprobe_chasing_blocks_outside_single_segment(mode: str) -> None:
    experiment = PenprobeChasingDemoExperiment(PenprobeChasingDemoConfig.from_dict({}))

    with pytest.raises(RuntimeError, match="single_segment"):
        experiment.precheck(_session(mode=mode))


def test_active_segment_a_and_b_pairs_drive_selected_servo_ids_only() -> None:
    mapper = TendonDisplacementMapper(spool_diameter_cm=1.2, ticks_per_rev=4096)
    config = PenprobeChasingDemoConfig.from_dict({"displacement_gain_cm_per_mm": 0.01})

    request_a, _ = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[5.0, 3.0],
        servo_ids=[1, 2, 3, 4],
        pairs={"axis_a": [1, 3], "axis_b": [2, 4]},
        mapper=mapper,
        config=config,
    )
    request_b, _ = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[5.0, 3.0],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        mapper=mapper,
        config=config,
    )

    assert set(request_a) == {1, 2, 3, 4}
    assert request_a[1] == -request_a[3]
    assert request_a[2] == -request_a[4]
    assert set(request_b) == {5, 6, 7, 8}
    assert request_b[5] == -request_b[7]
    assert request_b[6] == -request_b[8]


def test_command_clamp_stays_within_default_500_ticks_and_raw_bounds() -> None:
    stepped = _step_and_clamp_tick_deltas(
        current_delta_by_servo={1: 490, 2: -490},
        desired_delta_by_servo={1: 900, 2: -900},
        max_step_ticks=50,
        max_abs_delta_ticks=500,
    )
    bounded, clips = _clamp_tick_deltas_to_startup_raw_bounds(
        stepped,
        startup_reference_by_servo={1: 4080, 2: 10},
    )

    assert stepped == {1: 500, 2: -500}
    assert bounded == {1: 15, 2: -10}
    assert clips


def test_requires_startup_reference_for_active_segment() -> None:
    experiment = PenprobeChasingDemoExperiment(PenprobeChasingDemoConfig.from_dict({}))
    wrong_servo_service = _ServoService([1, 2, 3, 4])

    with pytest.raises(RuntimeError, match="wrong segment"):
        experiment.precheck(_session(active_segment="segment_b", servo=wrong_servo_service))


def test_requires_0a_and_0b_tracking() -> None:
    _extract_robot_frame_tool_pose(_snapshot(), "0A", max_tracker_age_s=0.1)

    with pytest.raises(RuntimeError, match="tracker_tool_missing:0B"):
        _extract_robot_frame_tool_pose(_snapshot(missing_0b=True), "0B", max_tracker_age_s=0.1)


def test_tracker_loss_stops_and_records_failure_metrics() -> None:
    experiment = PenprobeChasingDemoExperiment(
        PenprobeChasingDemoConfig.from_dict({"max_iterations": 2, "max_duration_s": 0.0})
    )
    servo = _ServoService([5, 6, 7, 8])
    session = _session(
        active_segment="segment_b",
        tracking=_TrackingService([_snapshot(), _snapshot(), _snapshot(stale=True)]),
        servo=servo,
    )

    experiment.precheck(session)
    with pytest.raises(RuntimeError, match="tracker_stale"):
        experiment.execute(session)

    assert servo.commanded_servo_ids == [5, 6, 7, 8]
    assert session.metrics["stop_reason"] == "tracker_stale"
    assert session.metrics["active_segment_key"] == "segment_b"
    assert session.metrics["controlled_point_source"] == "0A coil origin in robot frame"
    assert "0B tool origin" in session.metrics["target_point_source"]
    assert session.metrics["operating_context"]["active_segment"]["servo_ids"] == [5, 6, 7, 8]
    assert session.samples


def test_legacy_mapping_mode_requires_configured_existing_files(tmp_path: Path) -> None:
    config = PenprobeChasingDemoConfig.from_dict({"mapping_mode": MAPPING_LEGACY_POLYNOMIAL_WORKSPACE})

    with pytest.raises(RuntimeError, match="missing required file config"):
        _resolve_mapping_mode(config, tmp_path)

    files = {}
    for key in ["x_coeffs_full", "zfits_poly3_full_circle", "known_surface"]:
        path = tmp_path / f"{key}.csv"
        path.write_text("0\n", encoding="utf-8")
        files[key] = str(path)
    config = PenprobeChasingDemoConfig.from_dict(
        {"mapping_mode": MAPPING_LEGACY_POLYNOMIAL_WORKSPACE, "legacy_polynomial_workspace": files}
    )

    mode, warnings = _resolve_mapping_mode(config, tmp_path)

    assert mode == MAPPING_LEGACY_POLYNOMIAL_WORKSPACE
    assert warnings


def test_fallback_mapping_works_without_legacy_files() -> None:
    config = PenprobeChasingDemoConfig.from_dict({"mapping_mode": MAPPING_PAIRED_XY_PROPORTIONAL})

    mode, warnings = _resolve_mapping_mode(config, Path.cwd())

    assert mode == MAPPING_PAIRED_XY_PROPORTIONAL
    assert warnings == []
