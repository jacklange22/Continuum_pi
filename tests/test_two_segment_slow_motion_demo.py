"""Tests for the two-segment slow motion demo controller.

The demo writes the bus only in live mode; these tests exercise:
- the config parser (validation + caps),
- the bottom/top -> segment_a/segment_b mapping,
- the dry-run execute path (no servo writes, full trace produced),
- the output bundle (CSV, JSONL, summary.json, summary.txt),
- registration into builtin registry,
- demo-only metadata flags propagate through summary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from continuum_robot.demo.two_segment_motion_patterns import (
    COUPLING_PHASE_SHIFTED,
    PATTERN_CIRCLE,
    PATTERN_FIGURE8,
    PHASE_HOLD_END,
    PHASE_HOLD_START,
    TwoSegmentPatternPoint,
)
from continuum_robot.experiments.two_segment_slow_motion_demo import (
    DemoTraceRow,
    TwoSegmentSlowMotionDemoConfig,
    TwoSegmentSlowMotionDemoExperiment,
    build_two_segment_command_for_point,
    write_demo_trace_csv,
    write_demo_trace_jsonl,
    write_demo_summary,
    register_two_segment_slow_motion_demo,
)
from continuum_robot.two_segment import SEGMENT_A, SEGMENT_B


# ---------------------------------------------------------------------------
# Config parser
# ---------------------------------------------------------------------------


class TestConfigParser:
    def test_defaults_are_safe(self) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict(None)
        assert config.pattern == PATTERN_FIGURE8
        assert config.amplitude_cm == pytest.approx(0.25)
        assert config.cycle_duration_s == pytest.approx(45.0)
        assert config.cycles == 2
        assert config.update_rate_hz == pytest.approx(3.0)
        assert config.coupling == COUPLING_PHASE_SHIFTED
        # Dry-run defaults to True for the demo so the operator can preview.
        assert config.dry_run is True

    def test_unsupported_pattern_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported pattern"):
            TwoSegmentSlowMotionDemoConfig.from_dict({"pattern": "spiral_of_doom"})

    def test_unsupported_coupling_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported coupling"):
            TwoSegmentSlowMotionDemoConfig.from_dict({"coupling": "random_top"})

    def test_amplitude_above_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="cap"):
            TwoSegmentSlowMotionDemoConfig.from_dict({"amplitude_cm": 2.5})

    def test_soft_cap_above_hard_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds hard cap"):
            TwoSegmentSlowMotionDemoConfig.from_dict(
                {"max_tick_delta_from_startup": 1000, "hard_max_tick_delta_from_startup": 500}
            )

    def test_lower_bounds_are_clamped_not_silently_zeroed(self) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict({"cycles": 0, "cycle_duration_s": 0.0})
        # Cycles is clamped to 1 (not zeroed).
        assert config.cycles == 1
        # cycle_duration is clamped to 1.0 (positive).
        assert config.cycle_duration_s == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bottom/top -> segment_a/segment_b mapping
# ---------------------------------------------------------------------------


def _make_point(bottom_x: float = 0.1, bottom_y: float = 0.0, top_x: float = 0.05, top_y: float = 0.0) -> TwoSegmentPatternPoint:
    return TwoSegmentPatternPoint(
        sample_index=0,
        elapsed_s=0.0,
        phase_label="pattern",
        phase_fraction=0.5,
        bottom_x_cm=bottom_x,
        bottom_y_cm=bottom_y,
        top_x_cm=top_x,
        top_y_cm=top_y,
        bottom_tendon_cm=(-bottom_x, -bottom_y, bottom_x, bottom_y),
        top_tendon_cm=(-top_x, -top_y, top_x, top_y),
        all_8_tendon_cm=(-bottom_x, -bottom_y, bottom_x, bottom_y, -top_x, -top_y, top_x, top_y),
    )


class TestBottomTopMapping:
    def test_bottom_on_segment_a_puts_bottom_tendon_into_segment_a(self) -> None:
        point = _make_point(bottom_x=0.1, top_x=0.05)
        command = build_two_segment_command_for_point(point, bottom_segment_key=SEGMENT_A)
        assert command.segment_a == point.bottom_tendon_cm
        assert command.segment_b == point.top_tendon_cm

    def test_bottom_on_segment_b_swaps_assignment(self) -> None:
        point = _make_point(bottom_x=0.1, top_x=0.05)
        command = build_two_segment_command_for_point(point, bottom_segment_key=SEGMENT_B)
        assert command.segment_a == point.top_tendon_cm
        assert command.segment_b == point.bottom_tendon_cm

    def test_invalid_bottom_segment_key_raises(self) -> None:
        point = _make_point()
        with pytest.raises(ValueError, match="bottom_segment_key"):
            build_two_segment_command_for_point(point, bottom_segment_key="middle")


# ---------------------------------------------------------------------------
# Dry-run execute path
# ---------------------------------------------------------------------------


@dataclass
class _StubSession:
    """Minimal ExperimentSession-shaped stub for dry-run execution."""

    samples: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    stage_pass_fail: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Provide a context with the surface the controller reaches into.
        # We do NOT provide a robot.operating_context so that the dry-run
        # path falls back to _DryRunContext / _DryRunMapper.
        self.context = MagicMock()
        self.context.settings = None
        self.context.servo_service = None
        self.context.sleep_fn = lambda _s: None
        self.context.monotonic_fn = lambda: 0.0

    def add_sample(self, sample) -> None:
        self.samples.append(sample)

    def set_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def update_progress(self, *args, **kwargs) -> None:
        pass

    def stop_requested(self) -> bool:
        return False

    def set_stage(self, stage: str, status: str, message: str = "") -> None:
        self.stage_pass_fail[stage] = status


class TestDryRunExecute:
    def test_dry_run_produces_trace_without_servo_writes(self) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "pattern": PATTERN_FIGURE8,
                "amplitude_cm": 0.10,
                "cycle_duration_s": 2.0,
                "cycles": 1,
                "update_rate_hz": 5.0,
                "ramp_in_s": 0.5,
                "ramp_out_s": 0.5,
                "hold_at_start_s": 0.2,
                "hold_at_end_s": 0.2,
                "dry_run": True,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        session = _StubSession()

        experiment.setup(session)
        experiment.precheck(session)  # dry-run short-circuits hardware checks
        experiment.execute(session)

        assert len(session.samples) > 0
        # Every sample should be flagged demo_only / not_thesis_evidence.
        for sample in session.samples:
            flags = list(sample.status_flags or [])
            assert "demo_only" in flags
            assert "not_thesis_evidence" in flags
            assert "dry_run" in flags
        # Trace records command_sent = False because dry_run is True.
        assert len(experiment._trace) == len(session.samples)
        assert all(not row.command_sent for row in experiment._trace)
        assert all(row.skip_reason == "dry_run" for row in experiment._trace)

    def test_dry_run_trajectory_starts_and_ends_at_neutral_in_samples(self) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "pattern": PATTERN_FIGURE8,
                "amplitude_cm": 0.10,
                "cycle_duration_s": 2.0,
                "cycles": 1,
                "update_rate_hz": 5.0,
                "ramp_in_s": 0.5,
                "ramp_out_s": 0.5,
                "hold_at_start_s": 0.2,
                "hold_at_end_s": 0.2,
                "dry_run": True,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        session = _StubSession()
        experiment.setup(session)
        experiment.execute(session)
        first = experiment._trace[0]
        last = experiment._trace[-1]
        assert first.phase_label == PHASE_HOLD_START
        assert last.phase_label == PHASE_HOLD_END
        assert (first.bottom_x_cm, first.bottom_y_cm) == (0.0, 0.0)
        assert (last.bottom_x_cm, last.bottom_y_cm) == (0.0, 0.0)

    def test_dry_run_does_not_call_exclusive_bus_operation(self) -> None:
        # Regression: dry-run should not try to acquire bus ownership
        # because there is no live servo_service to lock against.
        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "amplitude_cm": 0.10,
                "cycle_duration_s": 1.0,
                "cycles": 1,
                "ramp_in_s": 0.1,
                "ramp_out_s": 0.1,
                "hold_at_start_s": 0.0,
                "hold_at_end_s": 0.0,
                "dry_run": True,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        session = _StubSession()
        experiment.setup(session)
        experiment.execute(session)
        # If the demo tried to acquire bus ownership on a stub session, the
        # MagicMock(servo_service=None) would have raised AttributeError on
        # exclusive_bus_operation. Reaching here means the demo correctly
        # short-circuits the lock for dry-run.
        assert len(session.samples) > 0

    def test_summary_marks_demo_only(self) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "amplitude_cm": 0.10,
                "cycle_duration_s": 1.0,
                "cycles": 1,
                "ramp_in_s": 0.1,
                "ramp_out_s": 0.1,
                "hold_at_start_s": 0.0,
                "hold_at_end_s": 0.0,
                "dry_run": True,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        session = _StubSession()
        experiment.setup(session)
        experiment.execute(session)
        summary = experiment.summarize(session)
        assert summary["demo_only"] is True
        assert summary["valid_for_model_training"] is False
        assert summary["valid_for_thesis_repeatability"] is False
        assert summary["closed_loop_control"] is False
        assert summary["motion_pattern_demo"] is True
        assert summary["dry_run"] is True


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _trace_rows(n: int = 3) -> list[DemoTraceRow]:
    rows = []
    for i in range(n):
        rows.append(
            DemoTraceRow(
                sample_index=i,
                elapsed_s=float(i) * 0.1,
                phase_label="pattern",
                bottom_x_cm=0.05 * i,
                bottom_y_cm=0.01 * i,
                top_x_cm=0.02 * i,
                top_y_cm=0.0,
                intended_flat_cm=[0.0] * 8,
                servo_flat_cm=[0.0] * 8,
                goal_ticks_by_servo={k: 2048 + k for k in range(1, 9)},
                present_ticks_by_servo={k: 2048 + k for k in range(1, 9)},
                max_current_ma=300 + i * 50,
                command_sent=True,
                skip_reason=None,
                safety_state="ok",
            )
        )
    return rows


class _BusOwnershipRecorder:
    """Records enter/exit of ``exclusive_bus_operation`` and orders all bus writes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.entered = False
        self.write_calls: list[dict[int, int]] = []

    def exclusive_bus_operation(self, *, owner: str, reason: str | None = None, servo_id=None):
        self.events.append(("acquire", {"owner": owner, "reason": reason}))
        recorder = self

        class _Ctx:
            def __enter__(self_inner):
                recorder.entered = True
                recorder.events.append(("enter", {}))
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                recorder.events.append(("exit", {"exc_type": exc_type.__name__ if exc_type else None}))
                recorder.entered = False
                return False

        return _Ctx()


class _LiveServoServiceStub:
    """ServoService stub that records bus writes and tracks exclusive ownership."""

    is_connected = True
    mapper = None  # filled in below

    def __init__(self) -> None:
        self.bus_recorder = _BusOwnershipRecorder()
        self.write_calls: list[dict[int, int]] = []
        self.telemetry_calls = 0

    def exclusive_bus_operation(self, **kwargs):
        return self.bus_recorder.exclusive_bus_operation(**kwargs)

    def _write_goal_positions(self, positions_by_id: dict[int, int]) -> None:
        # Asserts the lock is held when the write happens — exactly what the
        # production guard would do, but we surface it as a test failure
        # rather than a ServoBusBusyError so the test diff is readable.
        assert self.bus_recorder.entered, (
            "two_segment_slow_motion_demo wrote the bus without holding exclusive ownership; "
            "this regresses the fix for spurious 'bus is owned' errors mid-demo."
        )
        self.write_calls.append(dict(positions_by_id))

    def read_live_telemetry(self, servo_ids):
        self.telemetry_calls += 1
        # Return a minimal telemetry shape — no current, no position, no jam.
        from types import SimpleNamespace
        return {
            int(sid): SimpleNamespace(
                present_position=2048,
                present_current_ma=10,
                present_current_raw_unit=10,
            )
            for sid in servo_ids
        }


class _LiveMapperStub:
    def to_goal_positions(self, displacements_cm, neutral_ticks):
        # No-op mapping: a flat command rounds to the same ticks as neutral.
        return [int(neutral) for neutral in neutral_ticks]


class TestLiveBusOwnership:
    """Regression coverage for the missing exclusive_bus_operation wrap."""

    def _make_live_session(self) -> tuple[_StubSession, _LiveServoServiceStub]:
        session = _StubSession()
        servo_service = _LiveServoServiceStub()
        servo_service.mapper = _LiveMapperStub()
        session.context.servo_service = servo_service
        # Build an operating context shape with commanded ids 1..8 and
        # bottom_segment_key=segment_a so the demo can drive a real loop.
        from types import SimpleNamespace
        # to_flat() pulls segment definitions from context.metadata() rather
        # than from .segments directly. Provide a metadata payload that
        # carries the canonical segment_a / segment_b structure.
        segments_metadata = {
            "segment_a": {"key": "segment_a", "servo_ids": [1, 2, 3, 4]},
            "segment_b": {"key": "segment_b", "servo_ids": [5, 6, 7, 8]},
        }
        op_context = SimpleNamespace(
            operating_mode="dual_segment",
            commanded_servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            expected_servo_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            bottom_segment_key="segment_a",
            top_segment_key="segment_b",
            bottom_servo_ids=[1, 2, 3, 4],
            top_servo_ids=[5, 6, 7, 8],
            physical_assembly_issues=[],
            segment_order=["segment_a", "segment_b"],
        )
        op_context.metadata = lambda: {"segments": segments_metadata}
        session.context.settings = SimpleNamespace(
            robot=SimpleNamespace(operating_context=lambda: op_context)
        )
        return session, servo_service

    def test_live_demo_acquires_exclusive_bus_before_writing(self) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "amplitude_cm": 0.10,
                "cycle_duration_s": 1.0,
                "cycles": 1,
                "ramp_in_s": 0.1,
                "ramp_out_s": 0.1,
                "hold_at_start_s": 0.0,
                "hold_at_end_s": 0.0,
                "update_rate_hz": 5.0,
                "dry_run": False,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        # Build the trajectory directly (skip setup, which reaches into
        # the real ServoService.resolve_startup_reference_ticks path).
        from continuum_robot.experiments.two_segment_slow_motion_demo import (
            generate_pattern_trajectory,
        )
        experiment._trajectory = generate_pattern_trajectory(config.pattern_request())
        experiment._startup_ticks_by_servo = {sid: 2048 for sid in range(1, 9)}
        experiment._startup_provenance = {"accepted_all_8_startup": True, "source": "test"}
        session, servo_service = self._make_live_session()

        # Jump straight to execute to exercise the bus-ownership wrap.
        experiment.execute(session)

        events = servo_service.bus_recorder.events
        # Exactly one acquire → enter → exit cycle for the whole run.
        kinds = [event[0] for event in events]
        assert kinds.count("acquire") == 1
        assert kinds.count("enter") == 1
        assert kinds.count("exit") == 1
        # Acquire happens before any bus write.
        first_event = events[0]
        assert first_event[0] == "acquire"
        assert first_event[1]["owner"] == "two_segment_slow_motion_demo"
        # At least one write actually landed (sanity).
        assert servo_service.write_calls, "expected at least one bus write during the live demo loop"
        # And the assertion inside _write_goal_positions enforces "lock held".

    def test_live_demo_releases_lock_even_on_safety_halt(self) -> None:
        # Configure overcurrent thresholds so the FIRST telemetry triggers a halt.
        # The demo should still release the bus.
        config = TwoSegmentSlowMotionDemoConfig.from_dict(
            {
                "amplitude_cm": 0.10,
                "cycle_duration_s": 1.0,
                "cycles": 1,
                "ramp_in_s": 0.1,
                "ramp_out_s": 0.1,
                "hold_at_start_s": 0.0,
                "hold_at_end_s": 0.0,
                "update_rate_hz": 5.0,
                "sustained_overcurrent_ma": 1,
                "sustained_overcurrent_sample_count": 1,
                "dry_run": False,
            }
        )
        experiment = TwoSegmentSlowMotionDemoExperiment(config=config)
        from continuum_robot.experiments.two_segment_slow_motion_demo import (
            generate_pattern_trajectory,
        )
        experiment._trajectory = generate_pattern_trajectory(config.pattern_request())
        experiment._startup_ticks_by_servo = {sid: 2048 for sid in range(1, 9)}
        experiment._startup_provenance = {"accepted_all_8_startup": True, "source": "test"}
        session, servo_service = self._make_live_session()

        experiment.execute(session)

        events = servo_service.bus_recorder.events
        # Acquire + enter at start, exit at end — exactly one cycle even
        # when the run aborts early on overcurrent.
        kinds = [event[0] for event in events]
        assert kinds[0] == "acquire"
        assert kinds[-1] == "exit"
        assert kinds.count("exit") == 1
        # The exit happened cleanly (no exception propagated through __exit__).
        exit_event = next(event for event in events if event[0] == "exit")
        assert exit_event[1]["exc_type"] is None


class TestOutputBundle:
    def test_csv_writer_has_expected_columns(self, tmp_path: Path) -> None:
        rows = _trace_rows()
        path = tmp_path / "demo_trace.csv"
        write_demo_trace_csv(path, rows)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        header = text.splitlines()[0]
        expected = [
            "sample_index",
            "elapsed_s",
            "phase_label",
            "bottom_x_cm",
            "bottom_y_cm",
            "top_x_cm",
            "top_y_cm",
            "intended_flat_cm",
            "servo_flat_cm",
            "goal_ticks_by_servo",
            "present_ticks_by_servo",
            "max_current_ma",
            "command_sent",
            "skip_reason",
            "safety_state",
        ]
        for column in expected:
            assert column in header

    def test_jsonl_writer_emits_one_record_per_row(self, tmp_path: Path) -> None:
        rows = _trace_rows(n=4)
        path = tmp_path / "demo_trace.jsonl"
        write_demo_trace_jsonl(path, rows)
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 4
        for line in lines:
            record = json.loads(line)
            assert "sample_index" in record
            assert "phase_label" in record

    def test_summary_writer_marks_demo_only(self, tmp_path: Path) -> None:
        config = TwoSegmentSlowMotionDemoConfig.from_dict({"amplitude_cm": 0.10})
        rows = _trace_rows()
        write_demo_summary(tmp_path, summary={"sample_count": len(rows)}, config=config, trace=rows)
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["demo_only"] is True
        assert summary["valid_for_model_training"] is False
        text = (tmp_path / "two_segment_slow_motion_demo_summary.txt").read_text(encoding="utf-8")
        assert "demo_only: True" in text
        assert "valid_for_model_training: False" in text


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_function_adds_descriptor(self) -> None:
        registry = MagicMock()
        register_two_segment_slow_motion_demo(registry)
        registry.register.assert_called_once()
        call = registry.register.call_args
        assert call.kwargs["name"] == "two_segment_slow_motion_demo"
        assert call.kwargs["category"] == "demo"
        assert "Demo" in call.kwargs["tags"]
        assert "TwoSegment" in call.kwargs["tags"]
        assert "MotionPattern" in call.kwargs["tags"]

    def test_builtin_registry_includes_slow_motion_demo(self) -> None:
        from continuum_robot.experiments.builtins import register_builtin_experiments

        registry = MagicMock()
        register_builtin_experiments(registry)
        names = [call.kwargs.get("name") for call in registry.register.call_args_list]
        assert "two_segment_slow_motion_demo" in names


# ---------------------------------------------------------------------------
# GUI integration: page constructs, visibility, controller labels
# ---------------------------------------------------------------------------


class TestGuiIntegration:
    """Pins the GUI wiring: visibility map, page factory, controller labels.

    These would silently break the GUI flow even if the underlying experiment
    module is fine. They are also cheap (no hardware) so they run in the
    quick suite.
    """

    def test_visible_only_in_dual_segment_mode(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            MODE_EXPERIMENT_VISIBILITY,
        )

        assert "two_segment_slow_motion_demo" in MODE_EXPERIMENT_VISIBILITY["dual_segment"]
        assert "two_segment_slow_motion_demo" not in MODE_EXPERIMENT_VISIBILITY["single_segment"]
        assert "two_segment_slow_motion_demo" not in MODE_EXPERIMENT_VISIBILITY.get(
            "parallel_single", set()
        )

    def test_mode_label_branches_by_dry_run(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            ExperimentController,
        )

        assert (
            ExperimentController._mode_label(
                "two_segment_slow_motion_demo", {"dry_run": True}
            )
            == "dry-run two-segment demo"
        )
        assert (
            ExperimentController._mode_label(
                "two_segment_slow_motion_demo", {"dry_run": False}
            )
            == "live two-segment slow motion demo"
        )

    def test_config_summary_label_reports_pattern_amplitude_cycles(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            ExperimentController,
        )

        label = ExperimentController._config_summary_label(
            "two_segment_slow_motion_demo",
            {
                "pattern": "figure8",
                "amplitude_cm": 0.25,
                "cycles": 2,
                "cycle_duration_s": 45.0,
                "coupling": "phase_shifted",
                "dry_run": True,
            },
        )
        assert "figure8" in label
        assert "0.25 cm" in label
        assert "2 cycle" in label
        assert "phase_shifted" in label
        assert "dry-run" in label

    def test_history_metric_label_surfaces_pattern_and_max_current(self) -> None:
        from continuum_robot.gui.controllers.experiment_controller import (
            ExperimentController,
        )

        label = ExperimentController._history_metric_label(
            experiment_name="two_segment_slow_motion_demo",
            metrics={
                "pattern": "figure8",
                "amplitude_cm": 0.25,
                "sample_count": 200,
                "command_sent_count": 195,
                "max_current_observed_ma": 450,
            },
        )
        assert "pattern=figure8" in label
        assert "samples=200" in label
        assert "sent=195" in label
        assert "maxI=450mA" in label

    def test_factory_resolves_page_offscreen(self, tmp_path: Path) -> None:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.widgets.experiment_pages import (
            TwoSegmentSlowMotionDemoPage,
            build_experiment_page,
        )

        controller = _experiment_controller(tmp_path)
        page = build_experiment_page(controller, "two_segment_slow_motion_demo")
        try:
            assert isinstance(page, TwoSegmentSlowMotionDemoPage)
            assert hasattr(page, "pattern_combo")
            assert hasattr(page, "amplitude_combo")
            assert hasattr(page, "cycle_duration_spin")
            assert hasattr(page, "cycles_spin")
            assert hasattr(page, "update_rate_spin")
            assert hasattr(page, "ramp_in_spin")
            assert hasattr(page, "ramp_out_spin")
            assert hasattr(page, "coupling_combo")
            assert hasattr(page, "soft_cap_spin")
            assert hasattr(page, "hard_cap_spin")
            assert hasattr(page, "step_cap_spin")
            assert hasattr(page, "dry_run_check")
            assert hasattr(page, "return_to_neutral_check")
            assert hasattr(page, "preview_label")
        finally:
            page.deleteLater()
        _ = app

    def _preflight_kwargs(self, controller, tmp_path: Path) -> dict:
        return {
            "settings": controller.settings,
            "project_root": Path(__file__).resolve().parents[1],
            "tracking_snapshot": controller.tracking_service.get_snapshot(),
            "servo_calibration_summary": controller.servo_service.get_calibration_summary(),
            "servo_connected": False,
            "neutral_setpoints": {},
            "output_root": tmp_path,
            "planned_output_dir": tmp_path / "planned",
            "active_run_output_dir": None,
            "registration_path": tmp_path / "missing.json",
            "config_error": None,
        }

    def test_preflight_demo_only_info_is_always_surfaced(self, tmp_path: Path) -> None:
        """No matter the operating mode, the preflight must surface the
        demo_only notice so the operator sees it on every screen."""
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import evaluate_preflight

        controller = _experiment_controller(tmp_path)
        report = evaluate_preflight(
            experiment_name="two_segment_slow_motion_demo",
            config_payload={"dry_run": True, "amplitude_cm": 0.10},
            **self._preflight_kwargs(controller, tmp_path),
        )
        keys = {c.key for c in report.checks}
        assert "demo_only" in keys
        assert "tick_cap" in keys
        assert "amplitude" in keys

    def test_preflight_warns_on_high_amplitude(self, tmp_path: Path) -> None:
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import (
            evaluate_preflight,
            PREFLIGHT_WARNING,
        )

        controller = _experiment_controller(tmp_path)
        report = evaluate_preflight(
            experiment_name="two_segment_slow_motion_demo",
            config_payload={"dry_run": True, "amplitude_cm": 0.75},
            **self._preflight_kwargs(controller, tmp_path),
        )
        amp_check = next((c for c in report.checks if c.key == "amplitude"), None)
        assert amp_check is not None
        assert amp_check.status == PREFLIGHT_WARNING
        assert "0.75 cm" in amp_check.message

    def test_preflight_blocks_when_not_dual_segment(self, tmp_path: Path) -> None:
        """The default test settings put the runtime in single_segment;
        the preflight must surface that and block."""
        from tests.test_gui_controllers import _experiment_controller
        from continuum_robot.gui.experiment_preflight import (
            evaluate_preflight,
            PREFLIGHT_BLOCKED,
        )

        controller = _experiment_controller(tmp_path)
        report = evaluate_preflight(
            experiment_name="two_segment_slow_motion_demo",
            config_payload={"dry_run": False, "amplitude_cm": 0.10},
            **self._preflight_kwargs(controller, tmp_path),
        )
        mode_check = next((c for c in report.checks if c.key == "operating_mode"), None)
        assert mode_check is not None
        # The test rig defaults to single_segment, so this must block.
        assert mode_check.status == PREFLIGHT_BLOCKED
