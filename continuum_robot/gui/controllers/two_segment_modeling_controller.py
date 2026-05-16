"""Controller for the dedicated Two-Segment Modeling tab.

Owns all two-segment state and the offline analysis worker. Carved out of
``ModelingController`` so the single-segment Modeling tab can focus on its own
workflow (Mike / Camarillo / forward ANN comparison) without competing for screen
real estate or controller state with the two-segment pipeline.

The ``ModelingController`` retains its two-segment fields and methods as a
back-compat layer; this controller is the canonical owner going forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading

from continuum_robot.data.export_run_bundle import ExportBundleResult, export_run_bundle
from continuum_robot.data.run_management import discover_experiment_run_dirs
from continuum_robot.modeling.two_segment import (
    TwoSegmentModelingConfig,
    TwoSegmentModelingResult,
    load_two_segment_modeling_dataset,
    run_two_segment_modeling,
)


@dataclass
class TwoSegmentModelingViewState:
    """UI-facing snapshot for the Two-Segment Modeling tab."""

    dataset_runs: list[str] = field(default_factory=list)
    selected_run_paths: list[str] = field(default_factory=list)
    trainability_pairs: list[tuple[str, str]] = field(default_factory=list)
    include_linear: bool = True
    include_ann: bool = True
    include_camarillo: bool = False
    include_mike: bool = False
    strict_mode: bool = True
    allow_lower_trust: bool = False
    label_mode: str = "auto"
    include_orientation_if_available: bool = False
    ann_sweep_enabled: bool = False
    ann_hidden_layers: str = "128,128"
    ann_epochs: int = 200
    test_fraction: float = 0.25
    active: bool = False
    can_run: bool = False
    status_message: str = "Select one or more two-segment dataset runs to analyze."
    last_output_path: str | None = None
    can_open_output: bool = False
    can_export_output: bool = False
    export_path: str | None = None


class TwoSegmentModelingController:
    """Manages two-segment dataset discovery, validation, and offline analysis."""

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._catalog_dirty = True
        self._last_result: TwoSegmentModelingResult | None = None
        self.state = TwoSegmentModelingViewState()
        # Per-tick disk-read cache for trainability validation — the 5 Hz refresh used
        # to re-parse every selected samples.jsonl on every tick, blocking the paint
        # thread. Keyed by (paths, per-path mtimes, flags) so it auto-invalidates when
        # files change.
        self._trainability_cache: tuple[tuple, list[tuple[str, str]]] | None = None

    # ---- catalog -----------------------------------------------------------------
    def discover_dataset_runs(self) -> list[Path]:
        """Return ``two_segment_collect_pose_command_dataset`` runs under data/experiments."""
        return discover_experiment_run_dirs(
            self.project_root,
            experiment_name="two_segment_collect_pose_command_dataset",
        )

    def validate_trainability(
        self,
        run_dirs: list[Path],
        *,
        allow_lower_trust: bool = False,
    ) -> dict[str, object]:
        """Cheap "is this trainable?" summary without running any model."""
        dataset = load_two_segment_modeling_dataset(run_dirs, allow_lower_trust=bool(allow_lower_trust))
        return {
            "runs_scanned": len(run_dirs),
            "samples_scanned": dataset.accepted_count + dataset.rejected_count,
            "samples_accepted": dataset.accepted_count,
            "samples_rejected": dataset.rejected_count,
            "rejection_reasons": dataset.rejection_counts(),
            "trainable": dataset.accepted_count >= 2,
            "orientation_available": dataset.orientation_available,
            "includes_intermediate_pose": dataset.includes_intermediate_pose,
            "two_coil_xyz_available": dataset.two_coil_xyz_available,
            "two_coil_orientation_available": dataset.two_coil_orientation_available,
        }

    def invalidate_catalog(self) -> None:
        with self._lock:
            self._catalog_dirty = True

    def refresh(self) -> TwoSegmentModelingViewState:
        with self._lock:
            catalog_dirty = self._catalog_dirty
            selected_paths = list(self.state.selected_run_paths)
            active = self.state.active

        dataset_runs = list(self.state.dataset_runs)
        if catalog_dirty:
            dataset_runs = [str(path) for path in self.discover_dataset_runs()]
            if not selected_paths and dataset_runs:
                selected_paths = [dataset_runs[0]]

        trainability_pairs = self._trainability_pairs(
            selected_paths,
            allow_lower_trust=bool(self.state.allow_lower_trust and not self.state.strict_mode),
        )
        can_run = bool(selected_paths) and not active and bool(self._enabled_model_keys())

        with self._lock:
            self._catalog_dirty = False
            self.state.dataset_runs = dataset_runs
            self.state.selected_run_paths = selected_paths
            self.state.trainability_pairs = trainability_pairs
            self.state.can_run = can_run
            self.state.can_open_output = bool(self.state.last_output_path)
            self.state.can_export_output = bool(self.state.last_output_path)
            return self.state

    # ---- selection + setters -----------------------------------------------------
    def select_runs(self, paths: list[str]) -> None:
        cleaned = [str(p) for p in paths if str(p).strip()]
        with self._lock:
            self.state.selected_run_paths = cleaned

    def set_model_enabled(self, model_key: str, value: bool) -> None:
        with self._lock:
            key = str(model_key)
            if key == "linear_baseline":
                self.state.include_linear = bool(value)
            elif key == "ann":
                self.state.include_ann = bool(value)
            elif key == "camarillo":
                self.state.include_camarillo = bool(value)
            elif key in {"mike", "mike_constant_curvature"}:
                self.state.include_mike = bool(value)

    def set_strict_mode(self, value: bool) -> None:
        with self._lock:
            self.state.strict_mode = bool(value)
            if bool(value):
                self.state.allow_lower_trust = False

    def set_allow_lower_trust(self, value: bool) -> None:
        with self._lock:
            self.state.allow_lower_trust = bool(value)
            if bool(value):
                self.state.strict_mode = False

    def set_label_mode(self, value: str) -> None:
        with self._lock:
            self.state.label_mode = str(value or "auto")

    def set_include_orientation_if_available(self, value: bool) -> None:
        with self._lock:
            self.state.include_orientation_if_available = bool(value)

    def set_ann_sweep_enabled(self, value: bool) -> None:
        with self._lock:
            self.state.ann_sweep_enabled = bool(value)

    def set_ann_hidden_layers(self, value: str) -> None:
        with self._lock:
            self.state.ann_hidden_layers = str(value or "128,128")

    def set_ann_epochs(self, value: int) -> None:
        with self._lock:
            self.state.ann_epochs = max(1, int(value))

    def set_test_fraction(self, value: float) -> None:
        with self._lock:
            self.state.test_fraction = min(0.9, max(0.05, float(value)))

    # ---- execution ---------------------------------------------------------------
    def run_analysis(self) -> None:
        with self._lock:
            if self.state.active:
                return
            run_dirs = [Path(path) for path in self.state.selected_run_paths]
            model_keys = self._enabled_model_keys()
            allow_lower_trust = bool(self.state.allow_lower_trust and not self.state.strict_mode)
            label_mode = str(self.state.label_mode)
            include_orientation = bool(self.state.include_orientation_if_available)
            ann_sweep = bool(self.state.ann_sweep_enabled)
            ann_hidden_layers = str(self.state.ann_hidden_layers)
            ann_epochs = int(self.state.ann_epochs)
            test_fraction = float(self.state.test_fraction)
            self.state.active = True
            self.state.status_message = "Running two-segment modeling analysis..."
        if not run_dirs:
            with self._lock:
                self.state.active = False
                self.state.status_message = "Select one or more two-segment dataset runs first."
            return
        if not model_keys:
            with self._lock:
                self.state.active = False
                self.state.status_message = "Select at least one two-segment model family."
            return
        self._worker_thread = threading.Thread(
            target=self._modeling_worker,
            kwargs={
                "run_dirs": run_dirs,
                "model_keys": model_keys,
                "allow_lower_trust": allow_lower_trust,
                "label_mode": label_mode,
                "include_orientation": include_orientation,
                "ann_sweep": ann_sweep,
                "ann_hidden_layers": ann_hidden_layers,
                "ann_epochs": ann_epochs,
                "test_fraction": test_fraction,
            },
            daemon=True,
        )
        self._worker_thread.start()

    def export_last_bundle(self) -> ExportBundleResult:
        with self._lock:
            output_path = self.state.last_output_path
        if not output_path:
            raise ValueError("No two-segment modeling output is available to export.")
        result = export_run_bundle(
            run_dir=Path(output_path),
            project_root=self.project_root,
            include_samples=False,
            include_debug=False,
            make_zip=True,
        )
        with self._lock:
            self.state.export_path = str(result.final_path)
            self.state.status_message = f"Exported two-segment modeling bundle to {result.final_path}."
        return result

    def shutdown(self) -> None:
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    # ---- internals ---------------------------------------------------------------
    def _modeling_worker(
        self,
        *,
        run_dirs: list[Path],
        model_keys: list[str],
        allow_lower_trust: bool,
        label_mode: str,
        include_orientation: bool,
        ann_sweep: bool,
        ann_hidden_layers: str,
        ann_epochs: int,
        test_fraction: float,
    ) -> None:
        try:
            hidden_layers = _parse_hidden_layers(ann_hidden_layers)
            result = run_two_segment_modeling(
                run_dirs=run_dirs,
                project_root=self.project_root,
                config=TwoSegmentModelingConfig(
                    allow_lower_trust=bool(allow_lower_trust),
                    model_keys=list(model_keys),
                    label_mode=str(label_mode),
                    include_orientation_if_available=bool(include_orientation),
                    test_fraction=float(test_fraction),
                    model_config={
                        "ann": {
                            "hidden_layers": hidden_layers,
                            "hidden_layer_options": [[32, 32], [64, 64], [128, 128]],
                            "sweep_enabled": bool(ann_sweep),
                            "epochs": int(ann_epochs),
                            "batch_size": 64,
                            "learning_rate": 0.001,
                            "patience": 20,
                            "seeds": [42],
                        }
                    },
                    output_root=str(self.project_root / "data" / "experiments"),
                    random_seed=42,
                ),
            )
        except ValueError as exc:
            requirements = (
                "Required: dual_segment dataset, accepted all-8 startup, distal_tip pose role, "
                "non-servo-only trusted run, successful commands, and "
                "valid_for_two_segment_model_training=true."
            )
            with self._lock:
                self.state.active = False
                self.state.status_message = f"Two-segment modeling could not start: {exc} {requirements}"
            return
        with self._lock:
            self._last_result = result
            self.state.active = False
            self.state.last_output_path = str(result.output_dir)
            self.state.status_message = self._result_message(result)

    def _enabled_model_keys(self) -> list[str]:
        keys: list[str] = []
        if self.state.include_linear:
            keys.append("linear_baseline")
        if self.state.include_ann:
            keys.append("ann")
        if self.state.include_camarillo:
            keys.append("camarillo")
        if self.state.include_mike:
            keys.append("mike_constant_curvature")
        return keys

    def _trainability_pairs(self, run_paths: list[str], *, allow_lower_trust: bool) -> list[tuple[str, str]]:
        if not run_paths:
            return [
                ("Selection", "Select one or more two_segment_collect_pose_command_dataset runs."),
                (
                    "Required",
                    "dual_segment, all-8 startup, distal_tip robot-frame pose, trusted non-servo-only samples.",
                ),
            ]
        # Cache keyed by (paths, per-path mtimes, flags) — see __init__ comment.
        mtimes: list[float] = []
        for path in run_paths:
            try:
                mtimes.append(Path(path).stat().st_mtime)
            except OSError:
                mtimes.append(0.0)
        cache_key = (
            tuple(sorted(str(p) for p in run_paths)),
            tuple(mtimes),
            bool(allow_lower_trust),
            str(self.state.label_mode),
            "trainability_v1",
        )
        if self._trainability_cache is not None and self._trainability_cache[0] == cache_key:
            return self._trainability_cache[1]
        try:
            summary = self.validate_trainability(
                [Path(path) for path in run_paths],
                allow_lower_trust=bool(allow_lower_trust),
            )
        except Exception as exc:
            pairs: list[tuple[str, str]] = [("Trainability", f"Could not inspect selected runs: {exc}")]
            self._trainability_cache = (cache_key, pairs)
            return pairs
        pairs = [
            ("Runs", str(summary["runs_scanned"])),
            ("Samples", f"{summary['samples_accepted']} accepted / {summary['samples_rejected']} rejected"),
            ("Trainable", str(summary["trainable"])),
            ("Rejection Reasons", str(summary["rejection_reasons"] or {})),
            ("Orientation", "available" if summary["orientation_available"] else "XYZ only"),
            ("Intermediate Pose", str(summary["includes_intermediate_pose"])),
            ("Two-Coil Labels", str(summary.get("two_coil_xyz_available", False))),
            ("Label Mode", str(self.state.label_mode)),
            ("Input Features", "8 tendon displacements (mm)"),
            (
                "Physics Models",
                "Mike/Camarillo require validated geometry, stiffness, sign, and frame config.",
            ),
        ]
        self._trainability_cache = (cache_key, pairs)
        return pairs

    @staticmethod
    def _result_message(result: TwoSegmentModelingResult) -> str:
        completed = [item for item in result.model_results if item.status == "completed"]
        unavailable = [
            f"{item.model_key}: {item.reason}"
            for item in result.model_results
            if str(item.status).startswith("unavailable")
        ]
        best = (
            min(
                completed,
                key=lambda item: float(item.metrics.get("xyz_rmse_mm", float("inf"))),
            )
            if completed
            else None
        )
        pieces = [
            f"Two-segment modeling saved to {result.output_dir.name}.",
            f"Accepted {result.dataset.accepted_count}, rejected {result.dataset.rejected_count}.",
        ]
        if best is not None:
            pieces.append(f"Best XYZ RMSE: {best.model_key} {best.metrics.get('xyz_rmse_mm'):.4g} mm.")
        if unavailable:
            pieces.append("Unavailable: " + "; ".join(unavailable[:3]))
        return " ".join(pieces)


def _parse_hidden_layers(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    except ValueError:
        return [128, 128]
    return values or [128, 128]
