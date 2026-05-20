"""Focused tests for multi-frame post-settle tracker averaging.

These tests exercise the math (`_average_tracker_frames`) and the per-command
collection helper (`_collect_post_settle_tracker_frames`) directly so we don't
depend on a full bootstrap. End-to-end coverage is provided by the existing
collect-pose integration tests; the bookkeeping there confirms the new path
doesn't regress the N=1 legacy behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from continuum_robot.experiments.builtins import (
    CollectPoseCommandDatasetConfig,
    _average_tracker_frames,
    _collect_post_settle_tracker_frames,
    _serialize_raw_tracker_frame,
)


# --------------------------------------------------------------------------- #
# Stubs                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class _StubTool:
    tool_id: str
    tracking_state: str = "tracked"
    present: bool = True
    valid: bool = True
    frame_number: int | None = 0
    translation_mm: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    quaternion_wxyz: tuple[float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0)


@dataclass
class _StubSnapshot:
    """Minimal tracking snapshot stub for the multi-frame helpers."""

    last_frame_number: int = 0
    tracker_data_age_s: float = 0.01
    tracker_data_stale: bool = False
    canonical_state: str = "streaming_healthy"
    tip_pose_status: str = "ok"
    registration_state: str = "loaded"
    T_robot_tip: list[list[float]] | None = field(default_factory=lambda: np.eye(4).tolist())
    tools: dict[str, _StubTool] = field(default_factory=dict)
    tracker_faults: list[str] = field(default_factory=list)
    pipeline_faults: list[str] = field(default_factory=list)
    last_error: str | None = None
    normalized_live_tool_ids: list[str] = field(default_factory=list)
    runtime_tip_mode: str = "coil_as_tip"
    runtime_tip_calibration_state: str = "loaded"
    selected_backend_name: str = "stub"
    backend_identity: str = "stub"
    effective_frame_rate_hz: float | None = 60.0
    runtime_tip_trust_level: str = "thesis_trusted"
    runtime_tip_mode_message: str = "stub"


class _StubTrackingService:
    """Returns a sequence of pre-built snapshots one at a time."""

    def __init__(self, snapshots: list[_StubSnapshot]) -> None:
        self._snapshots = list(snapshots)
        self._index = 0
        self._fallback = snapshots[-1] if snapshots else _StubSnapshot()

    def get_snapshot(self):  # noqa: D401
        if self._index < len(self._snapshots):
            snap = self._snapshots[self._index]
            self._index += 1
            return snap
        return self._fallback


class _StubSession:
    """Minimal experiment session matching what the helpers need."""

    def __init__(self, snapshots: list[_StubSnapshot]) -> None:
        self.context = _StubContext(snapshots)

    def raise_if_stop_requested(self) -> None:  # noqa: D401
        return None


class _StubContext:
    def __init__(self, snapshots: list[_StubSnapshot]) -> None:
        self.tracking_service = _StubTrackingService(snapshots)
        self._mono = 0.0

    def monotonic_fn(self) -> float:
        return float(self._mono)

    def sleep_fn(self, seconds: float) -> None:
        self._mono += float(seconds)


# --------------------------------------------------------------------------- #
# Config wiring                                                                #
# --------------------------------------------------------------------------- #


def test_config_back_compat_defaults_are_unchanged() -> None:
    """N=1 path keeps the today behaviour: no new file emissions."""
    cfg = CollectPoseCommandDatasetConfig()
    assert cfg.samples_per_command == 1
    assert cfg.tracker_samples_per_command == 1
    assert cfg.tracker_per_frame_max_wait_s == 1.0
    assert cfg.tracker_sample_period_s is None
    assert cfg.averaged_label_enabled is None
    assert cfg.export_first_sample_label is True
    assert cfg.export_averaged_sample_label is None


def test_config_parser_accepts_new_keys() -> None:
    cfg = CollectPoseCommandDatasetConfig.from_dict(
        {
            "tracker_samples_per_command": 20,
            "tracker_per_frame_max_wait_s": 2.5,
            "tracker_sample_period_s": 0.01,
            "averaged_label_enabled": True,
            "export_first_sample_label": True,
            "export_averaged_sample_label": True,
        }
    )
    assert cfg.tracker_samples_per_command == 20
    assert cfg.tracker_per_frame_max_wait_s == 2.5
    assert cfg.tracker_sample_period_s == 0.01
    assert cfg.averaged_label_enabled is True
    assert cfg.export_first_sample_label is True
    assert cfg.export_averaged_sample_label is True


# --------------------------------------------------------------------------- #
# _collect_post_settle_tracker_frames                                          #
# --------------------------------------------------------------------------- #


def _snapshot_with_tip(*, frame_id: int, tip_xyz_mm: tuple[float, float, float]) -> _StubSnapshot:
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = tip_xyz_mm
    snap = _StubSnapshot(last_frame_number=frame_id, T_robot_tip=T.tolist())
    snap.tools = {"0A": _StubTool(tool_id="0A", frame_number=frame_id, translation_mm=tip_xyz_mm)}
    snap.normalized_live_tool_ids = ["0A"]
    return snap


def test_collect_post_settle_frames_dedupes_by_frame_id() -> None:
    """A cached snapshot returned twice in a row counts as one fresh frame."""
    snapshots = [
        _snapshot_with_tip(frame_id=10, tip_xyz_mm=(0.0, 0.0, 0.0)),
        _snapshot_with_tip(frame_id=10, tip_xyz_mm=(0.0, 0.0, 0.0)),  # cached duplicate
        _snapshot_with_tip(frame_id=11, tip_xyz_mm=(0.1, 0.0, 0.0)),
        _snapshot_with_tip(frame_id=12, tip_xyz_mm=(0.2, 0.0, 0.0)),
    ]
    session = _StubSession(snapshots)
    frames, reason = _collect_post_settle_tracker_frames(
        session=session,
        tool_id="0A",
        max_tracker_age_s=0.15,
        per_frame_max_wait_s=1.0,
        poll_interval_s=0.001,
        require_robot_frame_tip=False,  # gate semantics covered elsewhere; here we test dedup/timeout
        allow_mock_state=False,
        allow_lower_trust_runtime_tip=False,
        tracker_samples_per_command=3,
    )
    assert reason is None
    assert len(frames) == 3
    assert [f["snapshot"].last_frame_number for f in frames] == [10, 11, 12]
    # frame_index counts the deduped slots, not cache reads.
    assert [f["frame_index"] for f in frames] == [0, 1, 2]


def test_collect_post_settle_frames_returns_partial_on_timeout() -> None:
    """If a fresh frame doesn't arrive within the per-frame budget, return what we got."""
    snapshots = [
        _snapshot_with_tip(frame_id=10, tip_xyz_mm=(0.0, 0.0, 0.0)),
    ]
    # After the first snapshot, the stub falls back to the same snapshot
    # forever, so future polls all see frame_id=10 (already seen). Per-frame
    # budget elapses, helper bails out.
    session = _StubSession(snapshots)
    frames, reason = _collect_post_settle_tracker_frames(
        session=session,
        tool_id="0A",
        max_tracker_age_s=0.15,
        per_frame_max_wait_s=0.05,
        poll_interval_s=0.01,
        require_robot_frame_tip=False,  # gate semantics covered elsewhere; here we test dedup/timeout
        allow_mock_state=False,
        allow_lower_trust_runtime_tip=False,
        tracker_samples_per_command=5,
    )
    assert reason == "frame_wait_timeout_at_index_1"
    assert len(frames) == 1
    assert frames[0]["snapshot"].last_frame_number == 10


# --------------------------------------------------------------------------- #
# _average_tracker_frames                                                      #
# --------------------------------------------------------------------------- #


def _frame_record(*, frame_id: int, xyz_mm: tuple[float, float, float], mono_t: float = 0.0) -> dict[str, Any]:
    snap = _snapshot_with_tip(frame_id=frame_id, tip_xyz_mm=xyz_mm)
    gate = {"accepted": True, "reason": "ok"}
    return {"snapshot": snap, "gate": gate, "monotonic_t": mono_t, "frame_index": frame_id}


def test_average_tracker_frames_mean_position_matches_numpy() -> None:
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    frames = [_frame_record(frame_id=i, xyz_mm=p, mono_t=i * 0.01) for i, p in enumerate(points)]
    averaged = _average_tracker_frames(frames=frames, tool_id="0A")
    assert averaged["averaged_robot_position_mm"] == pytest.approx([1.0, 0.0, 0.0])
    matrix = averaged["averaged_T_robot_tip"]
    assert [matrix[i][3] for i in range(3)] == pytest.approx([1.0, 0.0, 0.0])


def test_average_tracker_frames_spread_stats_are_consistent() -> None:
    points = [(0.0, 0.0, 0.0), (0.0, 0.6, 0.0), (0.0, -0.6, 0.0)]
    frames = [_frame_record(frame_id=i, xyz_mm=p, mono_t=i * 0.01) for i, p in enumerate(points)]
    averaged = _average_tracker_frames(frames=frames, tool_id="0A")
    stats = averaged["stats"]
    assert stats["valid_sample_count"] == 3
    assert stats["position_std_per_axis_mm"][0] == pytest.approx(0.0, abs=1e-9)
    # Sample std (ddof=1) — was ddof=0 pre-2026-05-20. The unbiased
    # estimator is the project-wide convention for any noise statistic the
    # thesis reports.
    assert stats["position_std_per_axis_mm"][1] == pytest.approx(np.std([0.0, 0.6, -0.6], ddof=1))
    assert stats["position_std_per_axis_mm"][2] == pytest.approx(0.0, abs=1e-9)
    expected_rms = float(np.sqrt(np.mean(np.asarray(stats["position_std_per_axis_mm"]) ** 2)))
    assert stats["position_std_rms_mm"] == pytest.approx(expected_rms)
    assert stats["position_max_deviation_mm"] == pytest.approx(0.6)
    assert stats["first_vs_mean_position_diff_mm"] == pytest.approx(0.0)
    assert stats["sample_window_s"] == pytest.approx(0.02)
    assert stats["orientation_average_available"] is True
    # All frames had identity rotation, so spread should be ~0 degrees.
    assert stats["orientation_max_spread_deg"] == pytest.approx(0.0, abs=1e-6)


def test_average_tracker_frames_first_vs_mean_position_diff() -> None:
    points = [(2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    frames = [_frame_record(frame_id=i, xyz_mm=p) for i, p in enumerate(points)]
    averaged = _average_tracker_frames(frames=frames, tool_id="0A")
    # mean = (2/3, 0, 0); first = (2, 0, 0); diff = 4/3 mm.
    assert averaged["stats"]["first_vs_mean_position_diff_mm"] == pytest.approx(4.0 / 3.0)


def test_average_tracker_frames_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No frames to average"):
        _average_tracker_frames(frames=[], tool_id="0A")


def test_average_tracker_frames_rejects_missing_T_robot_tip() -> None:
    frame = _frame_record(frame_id=0, xyz_mm=(0.0, 0.0, 0.0))
    frame["snapshot"].T_robot_tip = None  # invalid for averaging
    with pytest.raises(ValueError, match="T_robot_tip"):
        _average_tracker_frames(frames=[frame], tool_id="0A")


# --------------------------------------------------------------------------- #
# _serialize_raw_tracker_frame                                                 #
# --------------------------------------------------------------------------- #


def test_serialize_raw_tracker_frame_preserves_pose_and_timestamps() -> None:
    frame = _frame_record(frame_id=42, xyz_mm=(1.0, 2.0, 3.0), mono_t=0.123)
    session = _StubSession([frame["snapshot"]])
    row = _serialize_raw_tracker_frame(
        session=session,
        command_index=7,
        frame=frame,
        tool_id="0A",
    )
    assert row["command_index"] == 7
    assert row["frame_index"] == 42  # we passed frame_id as frame_index for clarity
    assert row["tracker_frame_id"] == 42
    assert row["monotonic_time_s"] == pytest.approx(0.123)
    assert row["gate_accepted"] is True
    assert row["pose_in_tracker_frame"]["translation_mm"] == [1.0, 2.0, 3.0]
    assert row["pose_in_robot_frame"]["translation_mm"] == pytest.approx([1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# Mutual-exclusion guard                                                       #
# --------------------------------------------------------------------------- #


def test_setup_rejects_both_loop_knobs_above_one(monkeypatch) -> None:
    """samples_per_command and tracker_samples_per_command both > 1 is ambiguous."""
    from continuum_robot.experiments.builtins import CollectPoseCommandDatasetExperiment

    cfg = CollectPoseCommandDatasetConfig(samples_per_command=3, tracker_samples_per_command=5)
    exp = CollectPoseCommandDatasetExperiment(cfg)

    class _StubServoService:
        is_connected = False

    class _StubSettings:
        pass

    class _Ctx:
        tracking_service = None
        servo_service = _StubServoService()
        settings = _StubSettings()
        registration_path = type("p", (), {"exists": lambda self=None: False})()

    class _Session:
        context = _Ctx()
        metrics: dict = {}

        def set_metric(self, key, value):
            self.metrics[key] = value

    # We expect setup to fail before reaching the servo / neutral-load step.
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        exp.setup(_Session())
