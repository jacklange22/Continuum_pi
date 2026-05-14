from __future__ import annotations

import json
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
    MAPPING_AGGRESSIVE_TICK_DEMO,
    MAPPING_LEGACY_POLYNOMIAL_WORKSPACE,
    MAPPING_PAIRED_XY_PROPORTIONAL,
    PenprobeChasingDemoConfig,
    PenprobeChasingDemoExperiment,
    _chase_motion_limiter_summary,
    _clamp_tick_deltas_to_startup_raw_bounds,
    _extract_robot_frame_tool_pose,
    _penprobe_write_goals_with_retry,
    _resolve_mapping_mode,
    _step_and_clamp_tick_deltas,
    aggressive_tick_demo_cycle,
    paired_xy_proportional_tick_request,
    _servo_write_allowed,
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
        runtime_tip_mode="coil_as_tip",
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
        self.live_read_count = 0
        self.minimal_read_count = 0
        self.command_count = 0
        self.goal_writes: list[dict[int, int]] = []

    def _write_goal_positions(self, goals: dict[int, int]) -> None:
        self.goal_writes.append({int(k): int(v) for k, v in dict(goals).items()})
    def read_live_telemetry(self, servo_ids):
        self.live_read_count += 1
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

    def read_minimal_telemetry(self, servo_ids):
        self.minimal_read_count += 1
        return self.read_live_telemetry(servo_ids)

    @contextmanager
    def exclusive_bus_operation(self, *, owner, reason):
        assert owner == "penprobe_chasing_demo"
        assert reason
        yield

    def command_displacement(
        self,
        tendon_displacements_cm,
        neutral_ticks,
        servo_ids,
        *,
        motion_workflow,
        chase_tight_loop_writes=False,
        prevalidated_telemetry_by_id=None,
        skip_post_command_telemetry=False,
        write_goal_attempts=None,
        **kwargs,
    ):
        self.command_count += 1
        self.commanded_servo_ids = [int(value) for value in servo_ids]
        telemetry = prevalidated_telemetry_by_id or self.read_live_telemetry(servo_ids)
        return SimpleNamespace(
            positions_by_id={int(servo_id): int(neutral_ticks[index]) for index, servo_id in enumerate(servo_ids)},
            telemetry_by_id=telemetry,
            clamp_reasons_by_id={},
        )

    def bus_ownership_status(self):
        return SimpleNamespace(
            active=False,
            owner=None,
            reason=None,
            servo_id=None,
            held_by_current_thread=False,
            started_at_monotonic_s=None,
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


def test_penprobe_chasing_demo_config_goal_write_retry_attempts() -> None:
    assert PenprobeChasingDemoConfig.from_dict({}).goal_write_retry_attempts == 2
    cfg = PenprobeChasingDemoConfig.from_dict({"goal_write_retry_attempts": 5})
    assert cfg.goal_write_retry_attempts == 5
    assert PenprobeChasingDemoConfig.from_dict({}).mapping_mode == MAPPING_AGGRESSIVE_TICK_DEMO


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
    assert request_b[5] < 0
    assert request_b[6] < 0


def test_large_xy_error_steps_near_configured_max_not_one_tick() -> None:
    mapper = TendonDisplacementMapper(spool_diameter_cm=1.2, ticks_per_rev=4096)
    config = PenprobeChasingDemoConfig.from_dict(
        {"proportional_gain_ticks_per_mm": 8.0, "max_tick_step_per_cycle": 50}
    )
    desired, _ = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[20.0, 0.0],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        mapper=mapper,
        config=config,
    )
    stepped, _reasons = _step_and_clamp_tick_deltas(
        current_delta_by_servo={5: 0, 6: 0, 7: 0, 8: 0},
        desired_delta_by_servo=desired,
        max_step_ticks=config.max_tick_step_per_cycle,
        max_abs_delta_ticks=500,
    )

    assert desired[int(5)] == -50
    assert stepped[int(5)] == -50
    assert stepped[int(7)] == 50
    assert abs(stepped[int(5)]) > 1


def test_max_tick_step_per_cycle_clamps_to_100_when_configured_higher() -> None:
    config = PenprobeChasingDemoConfig.from_dict({"max_tick_step_per_cycle": 250, "max_step_ticks": 250})

    assert config.max_tick_step_per_cycle == 100
    assert config.max_step_ticks == 100


def test_max_tick_delta_600_allows_cumulative_travel_with_per_cycle_step_cap() -> None:
    cur = {5: 0}
    for _ in range(10):
        stepped, _reasons = _step_and_clamp_tick_deltas(
            current_delta_by_servo=cur,
            desired_delta_by_servo={5: 700},
            max_step_ticks=100,
            max_abs_delta_ticks=600,
        )
        cur = {5: stepped[5]}
    assert cur[5] == 600


def test_max_tick_delta_100_caps_total_despite_large_per_cycle_step() -> None:
    cur = {5: 0}
    for _ in range(20):
        stepped, _reasons = _step_and_clamp_tick_deltas(
            current_delta_by_servo=cur,
            desired_delta_by_servo={5: 999},
            max_step_ticks=100,
            max_abs_delta_ticks=100,
        )
        cur = {5: stepped[5]}
    assert cur[5] == 100


def test_chase_motion_limiter_prefers_startup_raw_bounds_over_demo_delta_cap() -> None:
    assert (
        _chase_motion_limiter_summary(
            raw_bound_clips={5: "raw_goal_5000_clamped_to_4095"},
            step_clamp_reasons={5: ["max_tick_delta_from_startup"]},
            mapping_debug={},
            skipped_write_reason=None,
        )
        == "startup_raw_bounds:raw_goal_5000_clamped_to_4095"
    )


def test_chase_motion_limiter_reports_demo_delta_cap_when_raw_bounds_clear() -> None:
    assert (
        _chase_motion_limiter_summary(
            raw_bound_clips={},
            step_clamp_reasons={5: ["max_tick_delta_from_startup"]},
            mapping_debug={},
            skipped_write_reason=None,
        )
        == "max_tick_delta_from_startup"
    )


def test_penprobe_precheck_warns_when_max_tick_delta_exceeds_300() -> None:
    experiment = PenprobeChasingDemoExperiment(
        PenprobeChasingDemoConfig.from_dict({"max_tick_delta_from_startup": 600})
    )
    session = _session(active_segment="segment_b")
    experiment.precheck(session)
    assert any("Large penprobe demo travel" in msg for msg in session.warning_messages)


def test_deadband_suppresses_tiny_penprobe_jitter() -> None:
    mapper = TendonDisplacementMapper(spool_diameter_cm=1.2, ticks_per_rev=4096)
    config = PenprobeChasingDemoConfig.from_dict({"xy_deadband_mm": 1.0, "proportional_gain_ticks_per_mm": 10.0})

    desired, debug = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[0.2, 0.2],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        mapper=mapper,
        config=config,
    )

    assert desired == {5: 0, 6: 0, 7: 0, 8: 0}
    assert debug["clamped_xy_error_mm"] == [0.0, 0.0]


def test_command_clamp_stays_within_default_500_ticks_and_raw_bounds() -> None:
    stepped, _reasons = _step_and_clamp_tick_deltas(
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
        PenprobeChasingDemoConfig.from_dict(
            {
                "max_iterations": 30,
                "max_duration_s": 0.0,
                "stale_tracker_persist_cycles_before_stop": 3,
                "mapping_mode": MAPPING_PAIRED_XY_PROPORTIONAL,
            }
        )
    )
    servo = _ServoService([5, 6, 7, 8])

    class _StaleAfterCalls:
        def __init__(self, fresh_calls: int) -> None:
            self.calls = 0
            self.fresh_calls = int(fresh_calls)

        def get_snapshot(self):
            self.calls += 1
            stale = self.calls > self.fresh_calls
            return _snapshot(stale=stale)

    tracking = _StaleAfterCalls(5)
    session = _session(active_segment="segment_b", tracking=tracking, servo=servo)

    experiment.precheck(session)
    with pytest.raises(RuntimeError, match="tracker_stale_persisted"):
        experiment.execute(session)

    assert servo.commanded_servo_ids == [5, 6, 7, 8]
    assert session.metrics["stop_reason"] == "tracker_stale_persisted"
    assert session.metrics["controlled_point_source"] == "0A coil origin in robot frame"
    assert session.samples


def test_servo_rate_gate_skips_extra_writes_under_fast_control_loop() -> None:
    experiment = PenprobeChasingDemoExperiment(
        PenprobeChasingDemoConfig.from_dict(
            {
                "max_iterations": 240,
                "max_duration_s": 0.0,
                "loop_period_s": 0.02,
                "max_servo_write_hz": 2.0,
                "saturation_stop_cycles": 99999,
                "mapping_mode": MAPPING_PAIRED_XY_PROPORTIONAL,
            }
        )
    )
    servo = _ServoService([5, 6, 7, 8])
    session = _session(active_segment="segment_b", tracking=_TrackingService([_snapshot()]), servo=servo)
    experiment.precheck(session)
    experiment.execute(session)
    reasons = [str(row.extra.get("skipped_write_reason")) for row in session.samples]
    assert sum(value == "rate_limit" for value in reasons) >= 140
    assert sum(value == "servo_written" for value in reasons) >= 5


def test_penprobe_chase_decouples_write_loop_from_telemetry_health_reads() -> None:
    experiment = PenprobeChasingDemoExperiment(
        PenprobeChasingDemoConfig.from_dict(
            {
                "max_iterations": 20,
                "max_duration_s": 0.0,
                "loop_period_s": 0.02,
                "max_servo_write_hz": 25.0,
                "telemetry_health_hz": 1.0,
                "saturation_stop_cycles": 99999,
                "mapping_mode": MAPPING_PAIRED_XY_PROPORTIONAL,
            }
        )
    )
    servo = _ServoService([5, 6, 7, 8])
    session = _session(active_segment="segment_b", tracking=_TrackingService([_snapshot()]), servo=servo)

    experiment.precheck(session)
    experiment.execute(session)

    assert servo.command_count > 1
    assert servo.minimal_read_count < servo.command_count
    assert session.metrics["configured_telemetry_health_hz"] == 1.0
    assert all("last_valid_packet_age_by_servo" in row.extra for row in session.samples)


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


def test_resolve_mapping_selects_aggressive_tick_demo() -> None:
    config = PenprobeChasingDemoConfig.from_dict({})
    mode, warnings = _resolve_mapping_mode(config, Path.cwd())
    assert mode == MAPPING_AGGRESSIVE_TICK_DEMO
    assert warnings == []


def test_two_mm_positive_x_error_has_sixteen_raw_ticks_before_norm_clamp() -> None:
    mapper = TendonDisplacementMapper(spool_diameter_cm=1.2, ticks_per_rev=4096)
    config = PenprobeChasingDemoConfig.from_dict({"proportional_gain_ticks_per_mm": 8.0, "max_tick_step_per_cycle": 100})
    _, dbg = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[2.0, 0.0],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        mapper=mapper,
        config=config,
    )
    assert abs(float(dbg["raw_axis_request_ticks_xy"][0]) - 16.0) < 1e-6
    assert abs(float(dbg["raw_axis_request_ticks_xy"][1])) < 1e-6


def test_flip_x_inverts_signed_axis_ticks() -> None:
    mapper = TendonDisplacementMapper(spool_diameter_cm=1.2, ticks_per_rev=4096)
    base_cfg = PenprobeChasingDemoConfig.from_dict({"proportional_gain_ticks_per_mm": 8.0, "max_tick_step_per_cycle": 100})
    flip_cfg = PenprobeChasingDemoConfig.from_dict(
        {"proportional_gain_ticks_per_mm": 8.0, "max_tick_step_per_cycle": 100, "flip_x": True}
    )
    a_ticks, _ = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[2.0, 0.0],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        mapper=mapper,
        config=base_cfg,
    )
    b_ticks, _ = paired_xy_proportional_tick_request(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[2.0, 0.0],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        mapper=mapper,
        config=flip_cfg,
    )
    assert a_ticks[int(5)] == -b_ticks[int(5)]
    assert a_ticks[int(7)] == -b_ticks[int(7)]


def test_servo_write_allowed_respects_spacing() -> None:
    ok, reason = _servo_write_allowed(now_s=0.2, last_write_monotonic_s=0.0, max_hz=10.0)
    assert ok is True and reason is None
    ok, reason = _servo_write_allowed(now_s=0.05, last_write_monotonic_s=0.0, max_hz=10.0)
    assert ok is False and reason == "rate_limit"


def test_chase_motion_limiter_prefers_global_limiter_reason() -> None:
    assert (
        _chase_motion_limiter_summary(
            raw_bound_clips={},
            step_clamp_reasons={},
            mapping_debug={"global_limiter_reason": "aggressive_current_limit_ma"},
            skipped_write_reason=None,
        )
        == "aggressive_current_limit_ma"
    )


def test_aggressive_tick_demo_accumulates_beyond_300_ticks() -> None:
    cfg = PenprobeChasingDemoConfig.from_dict(
        {
            "mapping_mode": MAPPING_AGGRESSIVE_TICK_DEMO,
            "max_tick_step_per_cycle": 100,
            "max_tick_delta_from_startup": 800,
            "proportional_gain_ticks_per_mm": 30.0,
            "aggressive_current_limit_ma": 0,
        }
    )
    startup = {5: 2000, 6: 2000, 7: 2000, 8: 2000}
    internal = dict(startup)
    tele = {
        sid: ServoTelemetry(
            servo_id=sid,
            present_position=2000,
            present_current_ma=10,
            present_voltage_mv=12000,
            present_temperature_c=25,
            hardware_error_code=0,
            hardware_error=None,
            last_read_monotonic_s=1.0,
        )
        for sid in (5, 6, 7, 8)
    }
    pairs = {"axis_a": [5, 7], "axis_b": [6, 8]}
    max_used = 0
    for _ in range(60):
        internal, _des, cmd, _dbg, _clips, _rs, _gl = aggressive_tick_demo_cycle(
            tip_xy_mm=[0.0, 0.0],
            target_xy_mm=[50.0, 40.0],
            servo_ids=[5, 6, 7, 8],
            pairs=pairs,
            config=cfg,
            internal_goals=internal,
            startup_ref=startup,
            telemetry_by_id=tele,
            max_step_ticks=100,
        )
        max_used = max(max_used, max(abs(int(v)) for v in cmd.values()))
    assert max_used > 300


def test_aggressive_current_limit_blocks_when_peak_exceeds_ma() -> None:
    cfg = PenprobeChasingDemoConfig.from_dict(
        {
            "mapping_mode": MAPPING_AGGRESSIVE_TICK_DEMO,
            "max_tick_step_per_cycle": 50,
            "max_tick_delta_from_startup": 800,
            "proportional_gain_ticks_per_mm": 10.0,
            "aggressive_current_limit_ma": 500,
        }
    )
    startup = {5: 2000, 6: 2000, 7: 2000, 8: 2000}
    tele = {
        sid: ServoTelemetry(
            servo_id=sid,
            present_position=2000,
            present_current_ma=600,
            present_voltage_mv=12000,
            present_temperature_c=25,
            hardware_error_code=0,
            hardware_error=None,
            last_read_monotonic_s=1.0,
        )
        for sid in (5, 6, 7, 8)
    }
    work, _d, cmd, dbg, _, _, gl = aggressive_tick_demo_cycle(
        tip_xy_mm=[0.0, 0.0],
        target_xy_mm=[20.0, 0.0],
        servo_ids=[5, 6, 7, 8],
        pairs={"axis_a": [5, 7], "axis_b": [6, 8]},
        config=cfg,
        internal_goals=dict(startup),
        startup_ref=startup,
        telemetry_by_id=tele,
        max_step_ticks=50,
    )
    assert gl == "aggressive_current_limit_ma"
    assert dbg["global_limiter_reason"] == "aggressive_current_limit_ma"
    assert cmd == {5: 0, 6: 0, 7: 0, 8: 0}
    assert work == startup


def test_penprobe_goal_write_retry_recovers_status_packet_error() -> None:
    class _Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def _write_goal_positions(self, goals: dict[int, int]) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("[TxRxResult] Incorrect status packet!")

    svc = _Flaky()
    recovered = _penprobe_write_goals_with_retry(svc, {5: 2100}, attempts=3)
    assert recovered == 1
    assert svc.calls == 2


def test_aggressive_execute_writes_goal_positions() -> None:
    experiment = PenprobeChasingDemoExperiment(
        PenprobeChasingDemoConfig.from_dict(
            {
                "max_iterations": 6,
                "max_duration_s": 0.0,
                "mapping_mode": MAPPING_AGGRESSIVE_TICK_DEMO,
                "max_tick_step_per_cycle": 40,
                "max_tick_delta_from_startup": 800,
                "proportional_gain_ticks_per_mm": 12.0,
                "saturation_stop_cycles": 99999,
                "max_servo_write_hz": 50.0,
            }
        )
    )
    servo = _ServoService([5, 6, 7, 8])
    session = _session(active_segment="segment_b", tracking=_TrackingService([_snapshot()]), servo=servo)
    experiment.precheck(session)
    experiment.execute(session)
    assert servo.goal_writes
    assert session.metrics.get("mapping_mode_used") == MAPPING_AGGRESSIVE_TICK_DEMO


def test_recent_penprobe_audit_fixture_shows_paired_l2_cap() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "data/experiments/penprobe_chasing_demo/20260513_225832_penprobe_chasing_demo/summary.json"
    )
    if not path.exists():
        pytest.skip("fixture summary not present in checkout")
    data = json.loads(path.read_text(encoding="utf-8"))
    em = data.get("experiment_metrics") or {}
    clamps = em.get("chase_clamp_counts_by_reason") or {}
    assert em.get("mapping_mode_used") == "paired_xy_proportional"
    assert int(clamps.get("xy_mapping_l2_vector_cap", 0)) > 0
