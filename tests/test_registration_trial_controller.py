from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from continuum_robot.experiments.registration_trial import CAPTURES_SESSION_KEY
from continuum_robot.gui.controllers.registration_trial_controller import (
    RegistrationTrialController,
)


# --- stubs -----------------------------------------------------------------


@dataclass
class _StubMeasurementStatus:
    ready: bool = True
    message: str = "ready"


class _StubRegistrationService:
    """Returns a deterministic point per call, driven by an internal counter."""

    def __init__(self, *, points: list[list[float]] | None = None, fail_after: int | None = None) -> None:
        self._points = points or [[0.0, 0.0, 0.0]]
        self._index = 0
        self.fail_after = fail_after
        self.call_count = 0

    def sample_measurement_point_capture(self) -> dict[str, Any]:
        self.call_count += 1
        if self.fail_after is not None and self.call_count > self.fail_after:
            raise RuntimeError("simulated tracker failure")
        point = self._points[self._index % len(self._points)]
        self._index += 1
        return {
            "point_xyz_mm": list(point),
            "measurement_tool_snapshot": None,
            "measurement_point_status": {"ready": True, "message": "ok"},
        }


@dataclass
class _StubResultPaths:
    output_dir: Path


@dataclass
class _StubResultSummary:
    experiment_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubResult:
    success: bool
    message: str
    paths: _StubResultPaths
    summary: _StubResultSummary


class _StubExperimentRunner:
    """Records what was passed and returns a pre-fabricated success result."""

    def __init__(self, output_dir: Path, success: bool = True, error_message: str = "") -> None:
        self.output_dir = output_dir
        self.success = success
        self.error_message = error_message
        self.last_kwargs: dict[str, Any] = {}

    def run_experiment(self, experiment_name: str, **kwargs) -> _StubResult:
        self.last_kwargs = {"experiment_name": experiment_name, **kwargs}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Touch the reports so the controller's existence checks fire.
        (self.output_dir / "trial_report.md").write_text("# stub", encoding="utf-8")
        (self.output_dir / "trial_report.json").write_text("{}", encoding="utf-8")
        return _StubResult(
            success=self.success,
            message="ok" if self.success else self.error_message,
            paths=_StubResultPaths(output_dir=self.output_dir),
            summary=_StubResultSummary(experiment_metrics={"injected": True}),
        )


# --- tests -----------------------------------------------------------------


def _controller(tmp_path: Path, runner: _StubExperimentRunner, points: list[list[float]] | None = None):
    return RegistrationTrialController(
        registration_service=_StubRegistrationService(points=points),
        experiment_runner=runner,
        project_root=tmp_path,
        registration_yaml_path="config/registration.yaml",
    )


def test_start_requires_at_least_one_label(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner)
    with pytest.raises(ValueError):
        ctrl.start([], captures_per_landmark=10)


def test_start_rejects_duplicate_labels(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner)
    with pytest.raises(ValueError):
        ctrl.start(["L1", "L1"], captures_per_landmark=10)


def test_start_initializes_state(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner)
    state = ctrl.start(["L1", "L2", "L3"], captures_per_landmark=20)
    assert state.is_active is True
    assert state.current_label == "L1"
    assert state.captured_counts_by_label == {"L1": 0, "L2": 0, "L3": 0}
    assert state.captures_per_landmark == 20


def test_capture_one_records_into_current_label(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(
        tmp_path,
        runner,
        points=[[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]],
    )
    ctrl.start(["L1", "L2"], captures_per_landmark=3)
    for _ in range(3):
        ctrl.capture_one()
    assert ctrl.state.captured_counts_by_label["L1"] == 3
    assert ctrl.state.captured_counts_by_label["L2"] == 0
    assert ctrl.captures_by_label["L1"][0] == [1.0, 2.0, 3.0]


def test_advance_moves_to_next_label_then_completes(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner)
    ctrl.start(["L1", "L2"], captures_per_landmark=2)
    assert ctrl.state.current_label == "L1"
    next_label = ctrl.advance_to_next_label()
    assert next_label == "L2"
    assert ctrl.state.current_label == "L2"
    final = ctrl.advance_to_next_label()
    assert final is None
    assert ctrl.state.is_complete is True
    assert ctrl.state.current_label is None


def test_skip_drops_collected_captures_for_label(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner, points=[[1, 1, 1]])
    ctrl.start(["L1", "L2"], captures_per_landmark=5)
    ctrl.capture_one()
    ctrl.capture_one()
    assert ctrl.state.captured_counts_by_label["L1"] == 2
    ctrl.skip_current_label()
    # Skipped L1 should have its captures dropped; the controller is now on L2.
    assert ctrl.captures_by_label["L1"] == []
    assert ctrl.state.current_label == "L2"


def test_capture_failure_surfaces_to_state(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    service = _StubRegistrationService(points=[[0, 0, 0]], fail_after=1)
    ctrl = RegistrationTrialController(
        registration_service=service,
        experiment_runner=runner,
        project_root=tmp_path,
    )
    ctrl.start(["L1"], captures_per_landmark=3)
    ctrl.capture_one()  # ok
    with pytest.raises(RuntimeError):
        ctrl.capture_one()  # fails
    assert ctrl.state.last_error == "simulated tracker failure"


def test_run_analysis_passes_captures_to_runner_via_inject(tmp_path: Path) -> None:
    output_dir = tmp_path / "data" / "experiments" / "registration_trial" / "run1"
    runner = _StubExperimentRunner(output_dir)
    ctrl = _controller(tmp_path, runner, points=[[1, 2, 3]])
    ctrl.start(["L1", "L2"], captures_per_landmark=2)
    ctrl.capture_one()
    ctrl.capture_one()
    ctrl.advance_to_next_label()
    ctrl.capture_one()
    ctrl.capture_one()
    ctrl.advance_to_next_label()
    assert ctrl.state.is_complete is True
    state = ctrl.run_analysis(subset_sizes=[4, 5], averaging_methods=["mean", "median"])
    kwargs = runner.last_kwargs
    assert kwargs["experiment_name"] == "registration_trial"
    injected = kwargs["inject_session_metrics"]
    assert CAPTURES_SESSION_KEY in injected
    assert set(injected[CAPTURES_SESSION_KEY].keys()) == {"L1", "L2"}
    assert state.last_run_output_dir == output_dir
    assert state.last_trial_report_md == output_dir / "trial_report.md"


def test_run_analysis_raises_when_no_captures(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner)
    with pytest.raises(RuntimeError):
        ctrl.run_analysis()


def test_run_analysis_propagates_runner_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "out" / "failed"
    runner = _StubExperimentRunner(output_dir, success=False, error_message="solver blew up")
    ctrl = _controller(tmp_path, runner, points=[[0, 0, 0]])
    ctrl.start(["L1"], captures_per_landmark=1)
    ctrl.capture_one()
    with pytest.raises(RuntimeError):
        ctrl.run_analysis()
    assert ctrl.state.last_error == "solver blew up"


def test_remaining_for_current_label_tracks_progress(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner, points=[[0, 0, 0]])
    ctrl.start(["L1"], captures_per_landmark=5)
    assert ctrl.remaining_for_current_label() == 5
    ctrl.capture_one()
    assert ctrl.remaining_for_current_label() == 4
    for _ in range(4):
        ctrl.capture_one()
    assert ctrl.remaining_for_current_label() == 0


def test_reset_clears_state_and_captures(tmp_path: Path) -> None:
    runner = _StubExperimentRunner(tmp_path / "out")
    ctrl = _controller(tmp_path, runner, points=[[1, 1, 1]])
    ctrl.start(["L1"], captures_per_landmark=2)
    ctrl.capture_one()
    ctrl.reset()
    assert ctrl.captures_by_label == {}
    assert ctrl.state.is_active is False
    assert ctrl.state.current_label is None
