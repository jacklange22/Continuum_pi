"""Phase 6: GUI controls + CLI surfacing for the two timing experiments.

These tests cover:
  - Every parameter widget on both pages has a setToolTip() call. This
    is the operator-facing 'what does this knob do' contract for the
    timing experiments — see Phase 6 spec.
  - Both pages include a 'Live Run Mode' row in their parameter summary,
    populated by the shared _live_run_mode_hint helper, so the operator
    sees BEFORE clicking Run whether the run will be tagged as thesis
    evidence or as debug_or_synthetic.
  - The CLI surfaces thesis_eligibility verdict + reasons and the list
    of generated figure PNG paths from the run output dir, both in
    stdout and (when --save-result is given) in the JSON sidecar.

We use inspect.getsource on the page methods (no live QApplication) and
direct module-level introspection on the CLI helpers.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from continuum_robot.experiments import cli as cli_module


# --------------------------------------------------------------------------- #
# GUI page tooltips                                                            #
# --------------------------------------------------------------------------- #


def _tracker_timing_page_source() -> str:
    from continuum_robot.gui.widgets.experiment_pages import (
        TrackerTimingValidationPage,
    )

    return inspect.getsource(TrackerTimingValidationPage)


def _sync_page_source() -> str:
    from continuum_robot.gui.widgets.experiment_pages import (
        ServoTrackerSyncValidationPage,
    )

    return inspect.getsource(ServoTrackerSyncValidationPage)


def test_tracker_timing_page_attaches_tooltip_to_every_parameter_widget() -> None:
    src = _tracker_timing_page_source()
    # Every primary widget on this page should have a tooltip. We anchor on
    # each widget's attribute name rather than counting setToolTip calls,
    # so adding a new widget without a tooltip fails the test loudly.
    for widget_name in (
        "tool_mode_combo",
        "enable_servo_logging_check",
        "run_label_edit",
        "duration_spin",
        "sample_target_spin",
        "warmup_spin",
        "timeout_spin",
    ):
        anchor = f"self.{widget_name}.setToolTip"
        assert anchor in src, (
            f"tracker_timing page widget {widget_name!r} is missing a setToolTip() "
            "call — operators won't know what the knob does."
        )


def test_sync_page_attaches_tooltip_to_every_parameter_widget() -> None:
    src = _sync_page_source()
    for widget_name in (
        "servo_ids_edit",
        "tool_mode_combo",
        "include_tip_pose_check",
        "run_label_edit",
        "duration_spin",
        "warmup_spin",
        "amplitude_spin",
        "step_period_spin",
        "telemetry_poll_spin",
        "timeout_spin",
    ):
        anchor = f"self.{widget_name}.setToolTip"
        assert anchor in src, (
            f"sync page widget {widget_name!r} is missing a setToolTip() "
            "call — operators won't know what the knob does."
        )


def test_sync_page_amplitude_tooltip_documents_25_tick_default_rationale() -> None:
    """The 25-tick amplitude default is a thesis-specific value (small
    enough that 1500–2500 tick servos can't trip wrap safety). The
    tooltip must explain that rationale so an operator who bumps it up
    knows what they're trading away."""
    src = _sync_page_source()
    assert "25 ticks" in src, "amplitude tooltip should mention the 25-tick default"
    assert "wrap-safety" in src or "wrap safety" in src, (
        "amplitude tooltip should warn that larger values can hit the wrap-safety check"
    )


# --------------------------------------------------------------------------- #
# GUI Live Run Mode hint                                                       #
# --------------------------------------------------------------------------- #


def test_tracker_timing_page_summary_includes_live_run_mode_row() -> None:
    src = _tracker_timing_page_source()
    assert '"Live Run Mode"' in src or "'Live Run Mode'" in src, (
        "tracker_timing parameter summary must include a Live Run Mode row "
        "so operator sees thesis-eligibility hint BEFORE running."
    )
    assert "_live_run_mode_hint" in src, (
        "tracker_timing page should compute its Live Run Mode hint via the shared "
        "_live_run_mode_hint helper on the base class."
    )


def test_sync_page_summary_includes_live_run_mode_row() -> None:
    src = _sync_page_source()
    assert '"Live Run Mode"' in src or "'Live Run Mode'" in src
    assert "_live_run_mode_hint" in src
    # Sync experiment requires a servo; the call must reflect that.
    assert "servo_required=True" in src, (
        "sync page must call _live_run_mode_hint(servo_required=True) so the "
        "operator is warned when the servo isn't connected."
    )


def test_tracker_timing_page_passes_servo_required_false_to_hint() -> None:
    """tracker_timing_validation does not require servos to be connected
    for the headline metric, so the hint should not flag servo absence."""
    src = _tracker_timing_page_source()
    assert "servo_required=False" in src


def test_live_run_mode_hint_helper_is_on_base_class() -> None:
    """The helper must be on ExperimentPageBase so future pages can reuse it
    rather than duplicating runtime/servo introspection."""
    from continuum_robot.gui.widgets.experiment_pages import ExperimentPageBase

    assert hasattr(ExperimentPageBase, "_live_run_mode_hint"), (
        "_live_run_mode_hint should live on ExperimentPageBase, not on a subclass"
    )
    src = inspect.getsource(ExperimentPageBase._live_run_mode_hint)
    # Helper should at least look at mock_mode and (optionally) servo.
    assert "mock_mode" in src
    assert "servo" in src.lower()


# --------------------------------------------------------------------------- #
# CLI output extension                                                         #
# --------------------------------------------------------------------------- #


def _make_result(
    *,
    metrics: dict | None = None,
    output_dir: Path,
    success: bool = True,
) -> SimpleNamespace:
    """Build a minimal duck-typed result struct matching what
    cli.main() reads from runner.run_experiment()."""
    summary = SimpleNamespace(
        status="success",
        experiment_metrics=dict(metrics or {}),
        stage_pass_fail={"execute": "passed"},
        error_messages=[],
    )
    paths = SimpleNamespace(
        output_dir=output_dir,
        metadata_path=output_dir / "metadata.json",
        samples_path=output_dir / "samples.parquet",
        summary_path=output_dir / "summary.json",
    )
    return SimpleNamespace(
        experiment_name="tracker_timing_validation",
        run_id="run_phase6_cli_test",
        success=success,
        summary=summary,
        message="ok",
        paths=paths,
        sample_count=42,
    )


def test_cli_extract_thesis_eligibility_returns_block_when_present() -> None:
    result = _make_result(
        metrics={
            "thesis_eligibility": {
                "eligible": True,
                "label": "thesis_evidence",
                "reasons": [],
            },
        },
        output_dir=Path("/tmp/does_not_exist"),
    )
    extracted = cli_module._extract_thesis_eligibility(result)
    assert extracted is not None
    assert extracted["label"] == "thesis_evidence"


def test_cli_extract_thesis_eligibility_returns_none_when_absent() -> None:
    """Experiments that don't stamp the verdict must produce None, not an
    empty placeholder — keeps the CLI output clean."""
    result = _make_result(metrics={}, output_dir=Path("/tmp/does_not_exist"))
    assert cli_module._extract_thesis_eligibility(result) is None


def test_cli_extract_thesis_eligibility_returns_none_for_empty_dict() -> None:
    result = _make_result(metrics={"thesis_eligibility": {}}, output_dir=Path("/tmp/x"))
    assert cli_module._extract_thesis_eligibility(result) is None


def test_cli_extract_generated_figures_lists_png_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "thesis_01.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "sync_combined.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")  # ignored
    result = _make_result(output_dir=tmp_path)
    figures = cli_module._extract_generated_figures(result)
    assert [path.name for path in figures] == ["sync_combined.png", "thesis_01.png"]


def test_cli_extract_generated_figures_handles_missing_output_dir(tmp_path: Path) -> None:
    """Output dir that doesn't exist on disk yet (failed run) → empty list."""
    result = _make_result(output_dir=tmp_path / "does_not_exist")
    assert cli_module._extract_generated_figures(result) == []


def test_cli_main_prints_eligibility_and_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end check: stub the runner to return our fixture result and
    confirm cli.main() prints thesis_eligibility_* and figure= lines."""
    (tmp_path / "fig_a.png").write_bytes(b"\x89PNG")
    (tmp_path / "fig_b.png").write_bytes(b"\x89PNG")
    fake_result = _make_result(
        metrics={
            "thesis_eligibility": {
                "eligible": False,
                "label": "debug_or_synthetic",
                "reasons": ["runtime is in mock_mode (not live hardware)"],
            },
        },
        output_dir=tmp_path,
    )

    class FakeRunner:
        def run_experiment(self, name, config, operator_notes, output_dir):
            return fake_result

        def available_experiments(self):
            return []

    class FakeServices:
        def get(self, key):
            return FakeRunner()

    fake_ctx = SimpleNamespace(services=FakeServices())
    monkeypatch.setattr(cli_module, "build_app_context", lambda: fake_ctx)

    exit_code = cli_module.main(["--experiment", "tracker_timing_validation"])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "thesis_eligibility_label=debug_or_synthetic" in captured
    assert "thesis_eligibility_eligible=False" in captured
    assert "thesis_eligibility_reason=runtime is in mock_mode" in captured
    assert "figure=" in captured
    assert "fig_a.png" in captured
    assert "fig_b.png" in captured


def test_cli_save_result_json_includes_eligibility_and_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--save-result must include thesis_eligibility + generated_figures."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "fig_x.png").write_bytes(b"\x89PNG")
    fake_result = _make_result(
        metrics={
            "thesis_eligibility": {
                "eligible": True,
                "label": "thesis_evidence",
                "reasons": [],
            },
        },
        output_dir=output_dir,
    )

    class FakeRunner:
        def run_experiment(self, name, config, operator_notes, output_dir):
            return fake_result

        def available_experiments(self):
            return []

    class FakeServices:
        def get(self, key):
            return FakeRunner()

    fake_ctx = SimpleNamespace(services=FakeServices())
    monkeypatch.setattr(cli_module, "build_app_context", lambda: fake_ctx)

    save_path = tmp_path / "result.json"
    exit_code = cli_module.main(
        [
            "--experiment",
            "tracker_timing_validation",
            "--save-result",
            str(save_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(save_path.read_text(encoding="utf-8"))
    assert payload["thesis_eligibility"]["label"] == "thesis_evidence"
    assert any(
        path.endswith("fig_x.png") for path in payload["generated_figures"]
    )
