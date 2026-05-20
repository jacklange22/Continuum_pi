"""Registration trial experiment.

Captures or replays N landmarks * K captures, then exhaustively analyzes the
result: every averaging method, leave-one-out FRE, and a 4..8-of-N subset
sweep. The intent is to design an honest experiment that tells the operator
which knobs actually move FRE on real hardware, instead of chasing intuitions.

The experiment supports two flows:

- **Replay**. ``source_record_path`` points at either a saved registration
  record (``data/registrations/*.json`` with ``raw_captured_landmarks_aurora_xyz``)
  or a raw-captures JSON of the form
  ``{"captures_by_label": {"L1": [[x,y,z], ...], ...}}``. No hardware needed.
- **Live**. The GUI / controller records captures and writes them into
  ``session.metrics["registration_trial_captures"]`` before ``execute`` runs.
  This commit ships the experiment plumbing; the live-capture wiring lives
  in the Registration tab follow-up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from continuum_robot.experiments.framework import (
    BaseExperiment,
    ExperimentHardwareRequirements,
    ExperimentSession,
)
from continuum_robot.registration.trial_analysis import (
    AVERAGING_METHODS,
    DEFAULT_SAMPLES_PER_POINT_LADDER,
    RegistrationTrialResult,
    SubsetSolveResult,
    aggregate_samples_per_point,
    evaluate_all_subsets,
    recommend_samples_per_point,
    samples_per_point_study,
    summarize_subset_search,
    summarize_trial,
    sweep_methods,
)


CAPTURES_SESSION_KEY = "registration_trial_captures"
"""Key under which a live caller deposits raw captures in session.metrics."""


@dataclass
class RegistrationTrialConfig:
    """Trial experiment configuration."""

    source_record_path: str = ""
    """Optional path to a saved registration record or raw-captures JSON."""

    captures_per_landmark: int = 50
    """Target captures per landmark. Used in live mode to validate input."""

    landmark_labels: list[str] = field(default_factory=list)
    """Operator-selected landmarks. Empty = use all labels available in truth."""

    subset_sizes: list[int] = field(default_factory=lambda: [4, 5, 6, 7, 8])
    """Subset sizes the search will exhaustively evaluate."""

    averaging_methods: list[str] = field(default_factory=lambda: list(AVERAGING_METHODS))
    """Averaging methods evaluated by the sweep."""

    trimmed_fraction: float = 0.2
    mad_k: float = 3.5

    registration_yaml_path: str = "config/registration.yaml"
    """Source of truth landmark coordinates. Must define candidate_landmarks."""

    samples_per_point_ladder: list[int] = field(
        default_factory=lambda: list(DEFAULT_SAMPLES_PER_POINT_LADDER)
    )
    """k values evaluated by the samples-per-point study. Capped at min(K) at runtime."""

    samples_per_point_bootstrap_iterations: int = 40
    """Random k-subset draws per k. Smooths out the choice of which samples are in the subset."""

    samples_per_point_epsilon_mm: float = 0.02
    """Diminishing-returns tolerance: smallest k whose mean FRE is within epsilon of the best."""

    random_seed: int = 7

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "RegistrationTrialConfig":
        payload = dict(payload or {})
        return cls(
            source_record_path=str(payload.get("source_record_path") or ""),
            captures_per_landmark=int(payload.get("captures_per_landmark") or 50),
            landmark_labels=[str(v) for v in (payload.get("landmark_labels") or []) if str(v).strip()],
            subset_sizes=[int(v) for v in (payload.get("subset_sizes") or [4, 5, 6, 7, 8])],
            averaging_methods=[str(v) for v in (payload.get("averaging_methods") or list(AVERAGING_METHODS))],
            trimmed_fraction=float(payload.get("trimmed_fraction") or 0.2),
            mad_k=float(payload.get("mad_k") or 3.5),
            registration_yaml_path=str(payload.get("registration_yaml_path") or "config/registration.yaml"),
            samples_per_point_ladder=[
                int(v) for v in (payload.get("samples_per_point_ladder") or DEFAULT_SAMPLES_PER_POINT_LADDER)
            ],
            samples_per_point_bootstrap_iterations=int(
                payload.get("samples_per_point_bootstrap_iterations") or 40
            ),
            samples_per_point_epsilon_mm=float(
                payload.get("samples_per_point_epsilon_mm") or 0.02
            ),
            random_seed=int(payload.get("random_seed") or 7),
        )


class RegistrationTrialExperiment(BaseExperiment):
    """Replay or live-capture trial sweep over averaging methods and subsets."""

    name = "registration_trial"
    description = (
        "Captures or replays many samples across many landmarks, then sweeps the "
        "averaging method and label subset to surface what actually lowers FRE."
    )
    hardware_requirements = ExperimentHardwareRequirements(
        tracking_required=False,
        servo_required=False,
        registration_required=False,
        mock_compatible=True,
    )

    def __init__(self, config: RegistrationTrialConfig) -> None:
        super().__init__(config=config)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None = None) -> "RegistrationTrialExperiment":
        return cls(config=RegistrationTrialConfig.from_dict(payload))

    def precheck(self, session: ExperimentSession) -> None:
        truth_path = self._resolve_truth_path(session)
        if not truth_path.exists():
            raise RuntimeError(f"Truth landmark config not found at {truth_path}.")
        truth = load_truth_landmarks(truth_path)
        if not truth:
            raise RuntimeError(f"No truth landmarks defined in {truth_path}.")
        if self.config.source_record_path:
            source = self._resolve_source_path(session)
            if not source.exists():
                raise RuntimeError(f"Source record not found: {source}.")
        # In live mode the captures key is populated by the controller before execute;
        # we only verify the contract is reasonable. Empty + no source_record means the
        # caller forgot to deposit captures, which we'll surface in execute().

    def execute(self, session: ExperimentSession) -> None:
        truth = load_truth_landmarks(self._resolve_truth_path(session))
        captures = self._load_captures(session)
        if not captures:
            raise RuntimeError(
                "No captures available for the trial. Provide source_record_path or write captures "
                f"to session.metrics[{CAPTURES_SESSION_KEY!r}] before running."
            )
        selected_labels = (
            sorted(set(self.config.landmark_labels) & set(captures.keys()))
            if self.config.landmark_labels
            else sorted(captures.keys())
        )
        if len(selected_labels) < 3:
            raise RuntimeError(
                f"At least 3 landmarks are required; have {len(selected_labels)} after filtering."
            )
        captures = {label: captures[label] for label in selected_labels}
        truth_subset = {label: truth[label] for label in selected_labels if label in truth}
        missing_truth = sorted(set(selected_labels) - set(truth_subset))
        if missing_truth:
            raise RuntimeError(
                f"Truth coordinates missing for captured labels: {missing_truth}."
            )

        sweep = sweep_methods(
            captures,
            truth_subset,
            methods=self.config.averaging_methods,
            trimmed_fraction=float(self.config.trimmed_fraction),
            mad_k=float(self.config.mad_k),
        )
        sweep_summary = summarize_trial(sweep)
        sweep_summary["captures_per_label"] = {
            label: len(points) for label, points in captures.items()
        }
        sweep_summary["captures_min"] = min(sweep_summary["captures_per_label"].values())
        sweep_summary["captures_max"] = max(sweep_summary["captures_per_label"].values())

        # Subset search uses the lowest-FRE averaging method's averaged points so we are
        # not double-counting the averaging knob inside the subset knob.
        best_method = sweep_summary.get("best_method") or "mean"
        best_method_averaging = sweep[best_method].averaging
        averaged_by_label = {
            label: avg.averaged_xyz_mm for label, avg in best_method_averaging.items()
        }
        subsets = evaluate_all_subsets(
            averaged_by_label,
            truth_subset,
            sizes=self.config.subset_sizes,
        )
        subset_summary = summarize_subset_search(subsets)
        subset_summary["averaging_method_used"] = best_method

        # Samples-per-point study: does the captured K actually help? Mean averaging
        # is fixed here so the samples-per-point axis is not stacked on top of the
        # averaging-method axis (which is the sweep's job).
        spp_rows = samples_per_point_study(
            captures,
            truth_subset,
            k_values=self.config.samples_per_point_ladder,
            bootstrap_iterations=int(self.config.samples_per_point_bootstrap_iterations),
            random_seed=int(self.config.random_seed),
        )
        spp_summary = aggregate_samples_per_point(spp_rows)
        spp_recommendation = recommend_samples_per_point(
            spp_summary,
            epsilon_mm=float(self.config.samples_per_point_epsilon_mm),
        )

        metrics = {
            "method_sweep": _sweep_results_to_payload(sweep),
            "method_summary": sweep_summary,
            "subset_search_summary": subset_summary,
            "subset_search_count": len(subsets),
            "samples_per_point_summary": spp_summary,
            "samples_per_point_recommendation": spp_recommendation,
            "landmark_labels_captured": selected_labels,
            "captures_per_landmark_target": int(self.config.captures_per_landmark),
            "trial_recommendations": _build_recommendations(
                sweep_summary,
                subset_summary,
                sweep,
                spp_recommendation=spp_recommendation,
            ),
        }
        session.metrics.update(metrics)
        # NOTE: pre-2026-05-20 this set ``force_status = "success"`` regardless
        # of the actual FRE numbers. The audit caught that ``status: success``
        # could not be used as a verdict — readers must read
        # ``method_summary.best_fre_mm`` and threshold it themselves.
        # We now report whether the run *completed* (the trial captured the
        # configured points and produced a method/subset sweep) separately
        # from any FRE-based judgment.
        run_completed = bool(metrics.get("method_summary"))
        session.metrics["run_completed"] = bool(run_completed)
        session.metrics["validation_passed"] = None  # the trial is a diagnostic, not a pass/fail gate
        session.metrics["summary_requirements"] = {
            "force_status": "success" if run_completed else "failed",
            "force_status_reason": (
                "registration_trial is a diagnostic; the run "
                "completed and produced a method/subset sweep. Validation "
                "verdict belongs to the operator using best_fre_mm."
                if run_completed
                else "registration_trial did not produce a method summary; treat as failed."
            ),
        }
        # Persist the raw captures we used so the run folder is self-contained.
        session.metrics["raw_captures_by_label"] = {
            label: [list(map(float, point)) for point in points]
            for label, points in captures.items()
        }
        session.metrics["truth_by_label"] = {
            label: list(map(float, truth_subset[label])) for label in selected_labels
        }

    def write_outputs(self, session: ExperimentSession, paths, summary) -> None:
        from continuum_robot.experiments.registration_trial_outputs import (
            write_registration_trial_outputs,
        )

        write_registration_trial_outputs(
            output_dir=paths.output_dir,
            metadata=session.metadata,
            summary=summary,
        )

    # --- helpers ---------------------------------------------------------

    def _resolve_truth_path(self, session: ExperimentSession) -> Path:
        raw = Path(self.config.registration_yaml_path)
        if raw.is_absolute():
            return raw
        return Path(session.context.project_root) / raw

    def _resolve_source_path(self, session: ExperimentSession) -> Path:
        raw = Path(self.config.source_record_path)
        if raw.is_absolute():
            return raw
        return Path(session.context.project_root) / raw

    def _load_captures(self, session: ExperimentSession) -> dict[str, list[list[float]]]:
        live_payload = session.metrics.get(CAPTURES_SESSION_KEY)
        if isinstance(live_payload, Mapping) and live_payload:
            return _normalize_captures(live_payload)
        if self.config.source_record_path:
            return _load_captures_from_disk(self._resolve_source_path(session))
        return {}


def load_truth_landmarks(yaml_path: Path) -> dict[str, list[float]]:
    """Parse a registration.yaml-style file for truth landmark coordinates."""
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    truth: dict[str, list[float]] = {}
    nominal = payload.get("nominal_landmarks_robot_xyz_mm") or {}
    if isinstance(nominal, Mapping):
        for label, xyz in nominal.items():
            truth[str(label)] = [float(v) for v in xyz]
    for entry in payload.get("candidate_landmarks", []) or []:
        if not isinstance(entry, Mapping):
            continue
        landmark_id = str(entry.get("id") or entry.get("label") or "").strip()
        xyz = entry.get("xyz_mm") or entry.get("coordinates_mm") or entry.get("xyz")
        if not landmark_id or xyz is None:
            continue
        truth.setdefault(landmark_id, [float(v) for v in xyz])
    return truth


def _normalize_captures(payload: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for label, points in payload.items():
        if not isinstance(points, Sequence):
            continue
        rows = [list(map(float, p)) for p in points if isinstance(p, Sequence) and len(p) == 3]
        if rows:
            out[str(label)] = rows
    return out


def _load_captures_from_disk(path: Path) -> dict[str, list[list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Accept either the live capture dump or a saved registration record.
    raw_block = payload.get("captures_by_label")
    if not isinstance(raw_block, Mapping):
        raw_block = payload.get("raw_captured_landmarks_aurora_xyz")
    if not isinstance(raw_block, Mapping):
        raise RuntimeError(
            f"{path} does not contain 'captures_by_label' or 'raw_captured_landmarks_aurora_xyz'."
        )
    return _normalize_captures(raw_block)


def _sweep_results_to_payload(results: Mapping[str, RegistrationTrialResult]) -> dict[str, Any]:
    return {method: asdict(result) for method, result in results.items()}


def _build_recommendations(
    sweep_summary: Mapping[str, Any],
    subset_summary: Mapping[str, Any],
    sweep_results: Mapping[str, RegistrationTrialResult],
    *,
    spp_recommendation: Mapping[str, Any] | None = None,
) -> list[str]:
    """Plain-English findings the operator can act on. No new math, no fudging."""
    recs: list[str] = []
    rows = sorted(
        list(sweep_summary.get("method_rows") or []),
        key=lambda row: float(row["fre_mm"]),
    )
    if rows and len(rows) > 1:
        spread = float(rows[-1]["fre_mm"]) - float(rows[0]["fre_mm"])
        if spread < 0.01:
            recs.append(
                "All averaging methods agree to within 0.01 mm on this dataset. With the current "
                "capture count, the averaging choice is not the bottleneck. Capture more samples "
                "per landmark (50+) to give MAD / trimmed mean something to act on."
            )
        else:
            best_method = sweep_summary.get("best_method")
            recs.append(
                f"Best averaging method on this dataset: {best_method} "
                f"(FRE={float(rows[0]['fre_mm']):.4f} mm, worst method "
                f"{float(rows[-1]['fre_mm']):.4f} mm, spread {spread:.4f} mm)."
            )
    # Geometry / coplanarity check from the first method's geometry block.
    if sweep_results:
        first = next(iter(sweep_results.values()))
        geom = first.geometry if isinstance(first.geometry, dict) else {}
        rank = geom.get("geometry_rank")
        if rank is not None and int(rank) < 3:
            recs.append(
                f"Truth landmarks are rank-deficient (rank={rank}): they sit on a plane. "
                "The third axis is recovered only from measurement noise. Add a landmark at a "
                "different height (non-coplanar z) before expecting sub-0.5 mm FRE consistently."
            )
    # Subset search recommendation.
    global_best = subset_summary.get("global_best") if isinstance(subset_summary, Mapping) else None
    if isinstance(global_best, Mapping):
        recs.append(
            "Best landmark subset: {labels} (size={size}) at FRE={fre:.4f} mm.".format(
                labels=global_best.get("labels"),
                size=global_best.get("size"),
                fre=float(global_best.get("fre_mm") or 0.0),
            )
        )
    # Per-size diminishing-returns hint.
    per_size = subset_summary.get("per_size_best") if isinstance(subset_summary, Mapping) else {}
    if isinstance(per_size, Mapping) and len(per_size) >= 2:
        ordered_sizes = sorted(per_size.keys())
        for prev, nxt in zip(ordered_sizes, ordered_sizes[1:]):
            prev_fre = float(per_size[prev].get("best_fre_mm") or 0.0)
            next_fre = float(per_size[nxt].get("best_fre_mm") or 0.0)
            delta = prev_fre - next_fre
            if abs(delta) < 0.02:
                recs.append(
                    f"Adding landmarks from {prev} to {nxt} changes best-FRE by only "
                    f"{delta:+.4f} mm. Diminishing returns past size {prev} on this data."
                )
                break
    # Samples-per-point recommendation: smallest k that lands within epsilon of
    # the captured pool's best FRE. Lets the operator stop over-capturing.
    if isinstance(spp_recommendation, Mapping):
        recommended_k = spp_recommendation.get("recommended_k")
        if recommended_k is not None:
            best_fre = spp_recommendation.get("best_fre_mean_mm")
            achieved = spp_recommendation.get("recommended_fre_mean_mm")
            epsilon = spp_recommendation.get("epsilon_mm")
            if best_fre is not None and achieved is not None:
                recs.append(
                    f"Samples per landmark: k={int(recommended_k)} reaches mean FRE "
                    f"{float(achieved):.4f} mm, within "
                    f"{float(epsilon) if epsilon is not None else 0.02:.3f} mm of the captured "
                    f"pool's best ({float(best_fre):.4f} mm). Capturing more samples per landmark "
                    "beyond that point does not measurably improve FRE on this data."
                )
            else:
                recs.append(f"Samples-per-point recommendation: k={int(recommended_k)}.")
    return recs


def register_registration_trial_experiment(registry) -> None:
    """Register the experiment in the global registry."""
    registry.register(
        name=RegistrationTrialExperiment.name,
        title="Registration Trial",
        description=RegistrationTrialExperiment.description,
        category="analysis",
        tags=["Registration", "Trial", "FRE"],
        default_config_path="config/experiment_registration_trial.example.yaml",
        factory=RegistrationTrialExperiment.from_dict,
    )
