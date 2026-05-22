"""Registration sampling consistency study.

This experiment captures many samples per body-frame landmark with the
pen-probe (default `0B`) and runs an offline analysis layer over the existing
rigid registration solver to answer:

- Does using more than 4 registration points improve registration?
- How many samples per point are needed for stable centers?
- Which points are unstable/outliers?
- Which averaging method (mean, median, trimmed mean) is best on real data?

The capture phase is intentionally close in shape to
`aurora_grid_accuracy`: walk labels, capture N samples per label, save raw
samples in the canonical sample schema. The analysis phase is in
`registration_sampling_study_outputs` and computes per-point spread,
subset-of-points solves, leave-one-out residuals, and a recommended
operator protocol.

This experiment never writes `data/registrations/latest_registration.json`
itself. It writes a `registration_candidate.json` inside its own run
directory; promotion is an explicit separate operator action (see
`continuum_robot.data.promote_registration_study`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from continuum_robot.experiments.framework import (
    BaseExperiment,
    ExperimentHardwareRequirements,
    ExperimentSession,
)
from continuum_robot.experiments.sample_builders import sample_from_tracking_snapshot


EXPERIMENT_NAME = "registration_sampling_study"
DEFAULT_SUBSET_SIZES: tuple[int, ...] = (4, 6, 8, 10, 12)
DEFAULT_AVERAGING_METHODS: tuple[str, ...] = ("mean", "median", "trimmed_mean")
DEFAULT_OPTIMIZE_FOR = "fre_mm"
ALLOWED_OPTIMIZE_FOR = ("fre_mm", "leave_one_out_mm", "within_point_spread_mm")


@dataclass
class RegistrationSamplingStudyConfig:
    """Configuration payload for the registration sampling study.

    Defaults match the operator policy chosen in the audit pass:
    12 landmarks, 20 samples/point, subsets [4,6,8,10,12], mean + median +
    trimmed-mean centers, FRE as the optimization criterion, no auto outlier
    deletion (only flagging), no auto-promote of active registration.
    """

    pen_probe_tool_id: str = "0B"
    samples_per_point: int = 20
    settle_time_s: float = 0.0
    capture_poll_interval_s: float = 0.05
    landmark_labels: list[str] = field(default_factory=list)
    truth_points_by_label: dict[str, list[float]] = field(default_factory=dict)
    subset_sizes: list[int] = field(default_factory=lambda: list(DEFAULT_SUBSET_SIZES))
    averaging_methods: list[str] = field(default_factory=lambda: list(DEFAULT_AVERAGING_METHODS))
    trimmed_mean_proportion: float = 0.1
    bootstrap_iterations: int = 200
    random_seed: int = 7
    dry_run: bool = False
    synthetic_noise_std_mm: float = 0.5
    synthetic_outlier_label: str | None = None
    synthetic_outlier_offset_mm: float = 0.0
    outlier_flag_z_threshold: float = 3.5
    optimize_for: str = DEFAULT_OPTIMIZE_FOR
    operator_notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "RegistrationSamplingStudyConfig":
        payload = dict(payload or {})
        labels = [str(value).strip() for value in (payload.get("landmark_labels") or []) if str(value).strip()]
        truth = {
            str(key).strip(): [float(component) for component in list(value)[:3]]
            for key, value in dict(payload.get("truth_points_by_label") or {}).items()
            if str(key).strip() and isinstance(value, (list, tuple)) and len(list(value)) >= 3
        }
        subsets = sorted({int(value) for value in (payload.get("subset_sizes") or DEFAULT_SUBSET_SIZES) if int(value) >= 3})
        averaging = [str(value).strip().lower() for value in (payload.get("averaging_methods") or DEFAULT_AVERAGING_METHODS) if str(value).strip()]
        averaging = [value for value in averaging if value in DEFAULT_AVERAGING_METHODS]
        if not averaging:
            averaging = list(DEFAULT_AVERAGING_METHODS)
        optimize = str(payload.get("optimize_for", DEFAULT_OPTIMIZE_FOR) or DEFAULT_OPTIMIZE_FOR).strip().lower()
        if optimize not in ALLOWED_OPTIMIZE_FOR:
            optimize = DEFAULT_OPTIMIZE_FOR
        return cls(
            pen_probe_tool_id=str(payload.get("pen_probe_tool_id", "0B") or "0B").strip(),
            samples_per_point=max(1, int(payload.get("samples_per_point", 20))),
            settle_time_s=max(0.0, float(payload.get("settle_time_s", 0.0))),
            capture_poll_interval_s=max(0.0, float(payload.get("capture_poll_interval_s", 0.05))),
            landmark_labels=labels,
            truth_points_by_label=truth,
            subset_sizes=subsets,
            averaging_methods=averaging,
            trimmed_mean_proportion=max(0.0, min(0.49, float(payload.get("trimmed_mean_proportion", 0.1)))),
            bootstrap_iterations=max(0, int(payload.get("bootstrap_iterations", 200))),
            random_seed=int(payload.get("random_seed", 7)),
            dry_run=bool(payload.get("dry_run", False)),
            synthetic_noise_std_mm=max(0.0, float(payload.get("synthetic_noise_std_mm", 0.5))),
            synthetic_outlier_label=(
                str(payload["synthetic_outlier_label"]).strip()
                if payload.get("synthetic_outlier_label")
                else None
            ),
            synthetic_outlier_offset_mm=float(payload.get("synthetic_outlier_offset_mm", 0.0)),
            outlier_flag_z_threshold=max(0.0, float(payload.get("outlier_flag_z_threshold", 3.5))),
            optimize_for=optimize,
            operator_notes=str(payload.get("operator_notes", "") or ""),
        )


def load_truth_points_from_registration_yaml(
    project_root: Path,
    *,
    requested_labels: list[str] | None = None,
) -> tuple[list[str], dict[str, list[float]]]:
    """Read enabled candidate landmarks from `config/registration.yaml`.

    Returns `(labels_in_declared_order, truth_xyz_by_label)`. If
    `requested_labels` is given, the result is restricted to that order and
    set.
    """
    config_path = Path(project_root) / "config" / "registration.yaml"
    if not config_path.exists():
        return [], {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return [], {}
    candidates = payload.get("candidate_landmarks") or []
    labels: list[str] = []
    truth: dict[str, list[float]] = {}
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        if not bool(entry.get("enabled", True)):
            continue
        label = str(entry.get("id", "")).strip()
        xyz = entry.get("xyz_mm")
        if not label or not isinstance(xyz, (list, tuple)) or len(list(xyz)) < 3:
            continue
        labels.append(label)
        truth[label] = [float(component) for component in list(xyz)[:3]]
    if requested_labels:
        wanted = [str(value).strip() for value in requested_labels if str(value).strip()]
        labels = [label for label in wanted if label in truth]
        truth = {label: truth[label] for label in labels}
    return labels, truth


class RegistrationSamplingStudyExperiment(BaseExperiment):
    """Capture many samples per body-frame landmark and study registration trade-offs."""

    name = EXPERIMENT_NAME
    description = (
        "Capture N samples per body-frame landmark with the pen-probe tool, then run an "
        "offline analysis layer over the existing rigid registration solver to recommend "
        "how many points and samples-per-point a thesis-grade registration protocol needs."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=False,  # dry_run is supported for tests
        servo_required=False,
        registration_required=False,
        mock_compatible=True,
    )

    def __init__(self, config: RegistrationSamplingStudyConfig) -> None:
        super().__init__(config=config)
        self.config: RegistrationSamplingStudyConfig = config
        self._tracking_started_here = False
        self._labels: list[str] = []
        self._truth: dict[str, list[float]] = {}
        self._effective_dry_run: bool = bool(config.dry_run)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None = None) -> "RegistrationSamplingStudyExperiment":
        return cls(config=RegistrationSamplingStudyConfig.from_dict(payload))

    def setup(self, session: ExperimentSession) -> None:
        # Resolve landmark set: explicit config wins; otherwise read from registration.yaml.
        labels = list(self.config.landmark_labels)
        truth = dict(self.config.truth_points_by_label)
        if not labels or not truth:
            yaml_labels, yaml_truth = load_truth_points_from_registration_yaml(
                session.context.project_root,
                requested_labels=labels or None,
            )
            if not labels:
                labels = yaml_labels
            # Backfill truth for any labels that don't yet have coordinates.
            for label in labels:
                if label not in truth and label in yaml_truth:
                    truth[label] = list(yaml_truth[label])
        # Drop any labels for which we have no truth coordinates.
        labels = [label for label in labels if label in truth]
        self._labels = labels
        self._truth = truth
        # Always treat as dry_run when there is no tracking service, so tests / CI can run.
        if session.context.tracking_service is None:
            self._effective_dry_run = True
        # Start tracking only when we are about to use it.
        if not self._effective_dry_run and getattr(session.context.tracking_service, "_thread", None) is None:
            session.context.tracking_service.start()
            self._tracking_started_here = True

    def precheck(self, session: ExperimentSession) -> None:
        if not self._labels:
            raise RuntimeError(
                "registration_sampling_study requires at least 3 landmark labels with truth coordinates "
                "(set them in the config or enable candidates in config/registration.yaml)."
            )
        if len(self._labels) < 3:
            raise RuntimeError(
                "registration_sampling_study requires at least 3 landmarks; "
                f"resolved {len(self._labels)}."
            )
        # Enforce subset sizes against the available label count.
        usable_subsets = sorted({size for size in self.config.subset_sizes if 3 <= size <= len(self._labels)})
        if not usable_subsets:
            raise RuntimeError(
                f"registration_sampling_study has no usable subset sizes; requested={self.config.subset_sizes}, "
                f"landmark count={len(self._labels)}."
            )
        session.set_metric("usable_subset_sizes", list(usable_subsets))
        session.set_metric("landmark_labels", list(self._labels))
        session.set_metric("landmark_count", int(len(self._labels)))
        session.set_metric("samples_per_point_target", int(self.config.samples_per_point))
        session.set_metric("optimize_for", str(self.config.optimize_for))
        session.set_metric("averaging_methods", list(self.config.averaging_methods))
        session.set_metric("bootstrap_iterations", int(self.config.bootstrap_iterations))
        session.set_metric("random_seed", int(self.config.random_seed))
        session.set_metric("dry_run", bool(self._effective_dry_run))
        session.set_stage("precheck", "passed", "registration_sampling_study precheck passed.")

    def execute(self, session: ExperimentSession) -> None:
        labels = list(self._labels)
        truth = dict(self._truth)
        samples_per_point = max(1, int(self.config.samples_per_point))
        total = len(labels) * samples_per_point
        progress = 0
        rng = np.random.default_rng(int(self.config.random_seed))
        synthetic_T_aurora_robot: np.ndarray | None = None
        synthetic_outlier_offset: np.ndarray | None = None
        if self._effective_dry_run:
            # Build a known T_aurora_robot for the synthetic case so tests can verify recovery.
            synthetic_T_aurora_robot = _synthetic_T_aurora_robot(rng)
            if self.config.synthetic_outlier_label and self.config.synthetic_outlier_offset_mm > 0:
                synthetic_outlier_offset = rng.standard_normal(3)
                synthetic_outlier_offset = (
                    float(self.config.synthetic_outlier_offset_mm)
                    * synthetic_outlier_offset
                    / max(float(np.linalg.norm(synthetic_outlier_offset)), 1e-9)
                )
        for label_index, label in enumerate(labels):
            truth_xyz = list(truth[label])
            for sample_index in range(samples_per_point):
                session.raise_if_stop_requested()
                if self.config.settle_time_s > 0.0:
                    session.context.sleep_fn(float(self.config.settle_time_s))
                aurora_xyz: list[float]
                status_flags: list[str] = []
                if self._effective_dry_run:
                    truth_vec = np.asarray(truth_xyz, dtype=float)
                    # Map truth -> aurora by inverting the synthetic robot_from_aurora and adding noise.
                    T = synthetic_T_aurora_robot  # type: ignore[assignment]
                    aurora_vec = (T[0:3, 0:3] @ truth_vec) + T[0:3, 3]
                    aurora_vec = aurora_vec + rng.normal(0.0, float(self.config.synthetic_noise_std_mm), size=3)
                    if (
                        synthetic_outlier_offset is not None
                        and self.config.synthetic_outlier_label
                        and label == self.config.synthetic_outlier_label
                    ):
                        aurora_vec = aurora_vec + synthetic_outlier_offset
                    aurora_xyz = [float(value) for value in aurora_vec]
                    status_flags.extend(["dry_run", "synthetic_capture"])
                else:
                    aurora_xyz = _read_pen_probe_aurora_xyz(
                        session=session,
                        tool_id=self.config.pen_probe_tool_id,
                        poll_interval_s=float(self.config.capture_poll_interval_s),
                    )
                    if aurora_xyz is None:
                        status_flags.append("capture_rejected")
                snapshot = _peek_snapshot(session) if not self._effective_dry_run else None
                sample = _build_study_sample(
                    session=session,
                    snapshot=snapshot,
                    label=label,
                    label_index=label_index,
                    sample_index=sample_index,
                    truth_body_xyz_mm=truth_xyz,
                    aurora_xyz_mm=aurora_xyz,
                    pen_probe_tool_id=self.config.pen_probe_tool_id,
                    status_flags=status_flags,
                    dry_run=self._effective_dry_run,
                )
                session.add_sample(sample)
                progress += 1
                session.update_progress(
                    progress,
                    total,
                    {
                        "phase": "capture",
                        "label": label,
                        "label_index": int(label_index),
                        "sample_index": int(sample_index),
                    },
                )
        session.set_metric("captured_sample_count", int(len(session.samples)))
        if synthetic_T_aurora_robot is not None:
            session.set_metric(
                "synthetic_truth_T_aurora_robot",
                [[float(value) for value in row] for row in np.asarray(synthetic_T_aurora_robot, dtype=float).tolist()],
            )

    def finalize(self, session: ExperimentSession) -> None:
        if self._tracking_started_here and session.context.tracking_service is not None:
            try:
                session.context.tracking_service.stop()
            except Exception:
                pass
            self._tracking_started_here = False

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        # Defer to the outputs module so this file stays focused on capture.
        from continuum_robot.experiments.registration_sampling_study_outputs import (
            write_registration_sampling_study_outputs,
        )

        write_registration_sampling_study_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
            samples=session.samples,
            labels=list(self._labels),
            truth_points_by_label=dict(self._truth),
            subset_sizes=[int(value) for value in session.metrics.get("usable_subset_sizes", []) or []],
            averaging_methods=[str(value) for value in self.config.averaging_methods],
            trimmed_mean_proportion=float(self.config.trimmed_mean_proportion),
            bootstrap_iterations=int(self.config.bootstrap_iterations),
            random_seed=int(self.config.random_seed),
            outlier_flag_z_threshold=float(self.config.outlier_flag_z_threshold),
            optimize_for=str(self.config.optimize_for),
        )


def _synthetic_T_aurora_robot(rng: np.random.Generator) -> np.ndarray:
    """Build a deterministic but non-identity synthetic T_aurora_robot for tests / dry_run."""
    # Small rotation + a translation in the typical Aurora workspace.
    axis = rng.standard_normal(3)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    theta = 0.10  # ~5.7 deg
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    t = np.asarray([25.0, 30.0, -180.0], dtype=float)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


def _peek_snapshot(session: ExperimentSession):
    reader = getattr(session.context.tracking_service, "peek_snapshot", None)
    if callable(reader):
        return reader()
    return session.context.tracking_service.get_snapshot()


def _read_pen_probe_aurora_xyz(
    *,
    session: ExperimentSession,
    tool_id: str,
    poll_interval_s: float,
) -> list[float] | None:
    """Read one calibrated pen-probe position in the tracker (aurora) frame.

    Returns the tool translation in mm or ``None`` if the tool is missing or
    its translation is unavailable.
    """
    snapshot = _peek_snapshot(session)
    tool = snapshot.tools.get(tool_id) if hasattr(snapshot, "tools") else None
    if tool is None or getattr(tool, "translation_mm", None) is None:
        return None
    translation = list(tool.translation_mm)
    if len(translation) < 3:
        return None
    return [float(value) for value in translation[:3]]


def _build_study_sample(
    *,
    session: ExperimentSession,
    snapshot,
    label: str,
    label_index: int,
    sample_index: int,
    truth_body_xyz_mm: list[float],
    aurora_xyz_mm: list[float] | None,
    pen_probe_tool_id: str,
    status_flags: list[str],
    dry_run: bool,
):
    """Build one canonical study sample.

    The aurora measurement and the body-frame truth are stored under
    `extra.registration_sample` so the offline analyzer can find them
    without scanning the full pose dicts.
    """
    accepted = bool(aurora_xyz_mm is not None and "capture_rejected" not in status_flags)
    capture_mode = "synthetic_dry_run" if dry_run else "live_tracker"
    extra = {
        "record_kind": "registration_sampling_sample",
        "label": str(label),
        "label_index": int(label_index),
        "capture_accepted": bool(accepted),
        "capture_mode": capture_mode,
        "registration_sample": {
            "label": str(label),
            "truth_body_xyz_mm": [float(value) for value in list(truth_body_xyz_mm)[:3]],
            "aurora_xyz_mm": (
                [float(value) for value in list(aurora_xyz_mm)[:3]] if aurora_xyz_mm is not None else None
            ),
            "pen_probe_tool_id": str(pen_probe_tool_id),
            "dry_run": bool(dry_run),
            "capture_mode": capture_mode,
        },
    }
    if snapshot is not None:
        return sample_from_tracking_snapshot(
            session,
            snapshot=snapshot,
            phase="capture",
            step_index=label_index,
            sample_index=sample_index,
            commanded_cable_deltas_cm=[],
            commanded_motor_values={},
            status_flags=sorted(set(status_flags)),
            extra=extra,
            target_index=label_index,
            tracker_tool_id=pen_probe_tool_id,
            override_tracker_position_mm=aurora_xyz_mm,
        )
    # Dry-run fallback path: no snapshot needed, build a minimal sample directly.
    from continuum_robot.experiments.schemas import ExperimentTimeseriesSample
    return ExperimentTimeseriesSample(
        monotonic_time_s=session.elapsed_s(),
        wall_time_utc=datetime.now(timezone.utc).isoformat(),
        phase="capture",
        step_index=int(label_index),
        sample_index=int(sample_index),
        target_index=int(label_index),
        commanded_motor_values={},
        commanded_cable_deltas_cm=[],
        tracker_frame_id=None,
        tool_ids_seen=[pen_probe_tool_id],
        transform_validity={pen_probe_tool_id: "OK"} if accepted else {pen_probe_tool_id: "MISSING"},
        pose_in_tracker_frame={
            pen_probe_tool_id: {
                "tracking_state": "OK" if accepted else "MISSING",
                "translation_mm": (
                    [float(value) for value in aurora_xyz_mm] if aurora_xyz_mm is not None else None
                ),
                "quaternion_wxyz": None,
                "tangent_xyz": None,
                "frame_number": None,
            }
        },
        pose_in_robot_frame={},
        freshness_s=0.0,
        latency_s=0.0,
        status_flags=sorted(set(status_flags)),
        backend_health={"capture_mode": capture_mode},
        extra=extra,
    )
