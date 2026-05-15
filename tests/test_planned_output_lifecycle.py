"""Tests for planned-output-path lifecycle classification used by the experiment preflight.

The lifecycle states distinguish between "no folder yet", a folder that belongs to the
currently active run, a folder that exists from a prior run (a real conflict), and the
unexpected-conflict bucket. The fix here resolves the user-facing bug where the GUI
would mark the active run's freshly created folder as a blocked planned-output conflict.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from types import SimpleNamespace

from continuum_robot.gui.experiment_preflight import (
    PLANNED_OUTPUT_AVAILABLE,
    PLANNED_OUTPUT_CONFLICT,
    PLANNED_OUTPUT_EXISTING_PREVIOUS_RUN,
    PLANNED_OUTPUT_NOT_CREATED_YET,
    PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN,
    classify_planned_output_state,
)


def test_classify_state_not_created_yet(tmp_path: Path) -> None:
    planned = tmp_path / "data" / "experiments" / "x" / "20260512_120000_x"
    state = classify_planned_output_state(planned_output_dir=planned, active_run_output_dir=None)
    assert state == PLANNED_OUTPUT_NOT_CREATED_YET


def test_classify_state_owned_by_current_run(tmp_path: Path) -> None:
    planned = tmp_path / "data" / "experiments" / "x" / "20260512_120000_x"
    planned.mkdir(parents=True)
    state = classify_planned_output_state(
        planned_output_dir=planned, active_run_output_dir=planned
    )
    assert state == PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN


def test_classify_state_existing_previous_run(tmp_path: Path) -> None:
    planned = tmp_path / "data" / "experiments" / "x" / "20260512_120000_x"
    planned.mkdir(parents=True)
    state = classify_planned_output_state(planned_output_dir=planned, active_run_output_dir=None)
    assert state == PLANNED_OUTPUT_EXISTING_PREVIOUS_RUN


def test_classify_state_conflict_when_not_a_directory(tmp_path: Path) -> None:
    planned = tmp_path / "data" / "experiments" / "x" / "20260512_120000_x"
    planned.parent.mkdir(parents=True)
    planned.write_text("collision file", encoding="utf-8")
    state = classify_planned_output_state(planned_output_dir=planned, active_run_output_dir=None)
    assert state == PLANNED_OUTPUT_CONFLICT


def test_classify_state_owned_by_current_run_when_active_dir_does_not_yet_exist(
    tmp_path: Path,
) -> None:
    # The runner has not yet finished mkdir-ing the active output folder. Even with the
    # path missing, ownership is implied by ``active_run_output_dir`` so preflight does
    # not falsely flag a conflict.
    planned = tmp_path / "data" / "experiments" / "x" / "20260512_120000_x"
    state = classify_planned_output_state(planned_output_dir=planned, active_run_output_dir=planned)
    assert state == PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN


def test_preflight_report_marks_owned_when_active(tmp_path: Path) -> None:
    """If preflight is called while the active run's folder already exists,
    the planned output check should be ok / owned_by_current_run rather than blocked."""
    from continuum_robot.gui.experiment_preflight import evaluate_preflight

    settings = SimpleNamespace(
        runtime=SimpleNamespace(mock_mode=False),
        serial=SimpleNamespace(tracker_backend="ndi"),
        registration=SimpleNamespace(penprobe_file=None),
        robot=SimpleNamespace(
            operating_mode=lambda: "single_segment",
            active_segment_label=lambda: "Spine 2",
            operating_context=lambda: SimpleNamespace(
                operating_mode="single_segment",
                segments={},
                segment_order=[],
                mirror_pairs={},
                active_segment_label="Spine 2",
            ),
        ),
    )
    snapshot = SimpleNamespace(
        selected_backend_name="none",
        backend_identity="none",
        canonical_state="disconnected",
        tracker_data_age_s=None,
        tracker_data_stale=True,
        registration_state="loaded",
        T_robot_aurora=None,
        runtime_tip_mode="coil_as_tip",
        runtime_tip_trust_level="thesis_trusted",
        runtime_tip_mode_message="",
        runtime_tip_calibration_state="loaded",
        runtime_tip_identity_fallback=False,
        tip_pose_status="ok",
        T_robot_tip=None,
        tools={},
        tracking_state="invalid",
    )
    planned = tmp_path / "data" / "experiments" / "pretension_validation" / "20260512_120000_run"
    planned.mkdir(parents=True)
    # Use an unsupported experiment name so the workflow-specific branches short-circuit
    # but the planned-output check still runs at the top of evaluate_preflight.
    report = evaluate_preflight(
        experiment_name="__unsupported_for_test__",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={5: 2048, 6: 2048, 7: 2048, 8: 2048},
        registration_path=tmp_path / "registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=planned,
        project_root=tmp_path,
        servo_calibration_summary=None,
        active_run_output_dir=planned,
    )
    assert report.planned_output_state == PLANNED_OUTPUT_OWNED_BY_CURRENT_RUN
    planned_check = next(check for check in report.checks if check.key == "planned_output_dir")
    assert planned_check.status == "ok"
    assert "active run" in planned_check.message


def test_preflight_report_blocks_existing_previous_run(tmp_path: Path) -> None:
    from continuum_robot.gui.experiment_preflight import evaluate_preflight

    settings = SimpleNamespace(
        runtime=SimpleNamespace(mock_mode=False),
        serial=SimpleNamespace(tracker_backend="ndi"),
        registration=SimpleNamespace(penprobe_file=None),
        robot=SimpleNamespace(
            operating_mode=lambda: "single_segment",
            active_segment_label=lambda: "Spine 2",
            operating_context=lambda: SimpleNamespace(
                operating_mode="single_segment",
                segments={},
                segment_order=[],
                mirror_pairs={},
                active_segment_label="Spine 2",
            ),
        ),
    )
    snapshot = SimpleNamespace(
        selected_backend_name="none",
        backend_identity="none",
        canonical_state="disconnected",
        tracker_data_age_s=None,
        tracker_data_stale=True,
        registration_state="loaded",
        T_robot_aurora=None,
        runtime_tip_mode="coil_as_tip",
        runtime_tip_trust_level="thesis_trusted",
        runtime_tip_mode_message="",
        runtime_tip_calibration_state="loaded",
        runtime_tip_identity_fallback=False,
        tip_pose_status="ok",
        T_robot_tip=None,
        tools={},
        tracking_state="invalid",
    )
    planned = tmp_path / "data" / "experiments" / "pretension_validation" / "20260512_120000_run"
    planned.mkdir(parents=True)
    report = evaluate_preflight(
        experiment_name="__unsupported_for_test__",
        config_payload={},
        config_error=None,
        settings=settings,
        tracking_snapshot=snapshot,
        servo_connected=True,
        neutral_setpoints={5: 2048, 6: 2048, 7: 2048, 8: 2048},
        registration_path=tmp_path / "registration.json",
        output_root=tmp_path / "data" / "experiments",
        planned_output_dir=planned,
        project_root=tmp_path,
        servo_calibration_summary=None,
        active_run_output_dir=None,
    )
    assert report.planned_output_state == PLANNED_OUTPUT_EXISTING_PREVIOUS_RUN
    planned_check = next(check for check in report.checks if check.key == "planned_output_dir")
    assert planned_check.status == "blocked"
    assert "previous run" in planned_check.message
