#!/usr/bin/env python3
"""Run a shared-split ANN model sweep (linear ridge optional + default ANN sizes + extras)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from continuum_robot.modeling.ann_training import (
    AnnTrainingConfig,
    load_modeling_dataset_summary,
    run_model_sweep,
    validate_legacy_ann_rows,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True, help="Repository / project root.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to one collect_pose_command_dataset run folder (with modeling_dataset_export.jsonl).",
    )
    parser.add_argument("--artifact-root", type=str, default="", help="Override AnnTrainingConfig.artifact_root.")
    parser.add_argument("--artifact-name", type=str, default="cli_model_sweep", help="Base artifact name prefix.")
    parser.add_argument("--epochs", type=int, default=256, help="Epochs per ANN (same for all sweep models).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default="", help="cpu | mps | cuda (empty = auto).")
    parser.add_argument("--no-linear", action="store_true", help="Skip linear ridge baseline.")
    parser.add_argument(
        "--extra-architectures",
        type=str,
        default="",
        help='Optional extra ANN widths, groups separated by | e.g. "48,48 | 96".',
    )
    parser.add_argument(
        "--allow-exploratory-incomplete-target",
        action="store_true",
        help=(
            "Train on a source run whose only invalidity is target_valid_sample_count not being met. "
            "Row-filter policy still applies; artifacts are annotated as exploratory and are not thesis-citable."
        ),
    )
    parser.add_argument(
        "--seeds-per-architecture",
        type=int,
        default=1,
        help=(
            "Number of random seeds to train per architecture (Wolfe MS thesis §3.2.2 trains 10 "
            "per architecture and picks lowest-val-loss). Each seed gets its own artifact subdir "
            "and its own row in the sweep summary; the sweep-level 'best' selection picks the "
            "lowest validation_position_rmse_xyz_mm across all seeds and architectures. The test "
            "split is reserved as the final held-out report for the selected model only."
        ),
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    dataset_path = args.dataset_path.resolve()
    cfg_kwargs: dict = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "random_seed": int(args.random_seed),
        "artifact_name": str(args.artifact_name).strip() or "cli_model_sweep",
    }
    if str(args.artifact_root or "").strip():
        cfg_kwargs["artifact_root"] = str(args.artifact_root).strip()
    base = AnnTrainingConfig(**cfg_kwargs)
    backend = str(args.backend or "").strip() or None
    if backend is None:
        from continuum_robot.modeling.ann_training import detect_training_backends

        backend = detect_training_backends().selected_backend

    def _status(msg: str) -> None:
        print(msg, flush=True)

    row_filter = validate_legacy_ann_rows(dataset_path)
    print(
        f"Row filter: total_rows={row_filter.total_export_rows} accepted={row_filter.accepted_export_rows} "
        f"complete={row_filter.complete_row_count} excluded={row_filter.excluded_row_count} "
        f"target={row_filter.target_complete_row_count} can_train={row_filter.can_train}",
        flush=True,
    )
    if row_filter.excluded_by_reason:
        for reason, count in sorted(row_filter.excluded_by_reason.items()):
            print(f"  excluded[{reason}] = {count}", flush=True)

    # Build training provenance from the dataset summary so the sweep artifacts carry the same
    # exploratory tagging the GUI controller would emit.
    training_provenance: dict = {
        "exploratory_training_override": bool(args.allow_exploratory_incomplete_target),
    }
    try:
        summary = load_modeling_dataset_summary(dataset_path)
        training_provenance.update(
            {
                "source_run_valid_for_model_training": summary.valid_for_model_training_flag,
                "source_run_model_training_validity_status": summary.model_training_validity_status,
                "source_validity_reason": summary.model_training_validity_reason,
                "ann_training_category": summary.ann_training_category,
                "complete_training_row_count": int(summary.complete_training_row_count),
                "target_valid_sample_count": int(summary.target_valid_sample_count),
                "source_run_id": summary.run_id,
                "source_run_path": str(summary.path),
            }
        )
    except Exception:
        pass

    if (
        not bool(args.allow_exploratory_incomplete_target)
        and training_provenance.get("source_run_valid_for_model_training") is False
    ):
        print(
            "ERROR: source run is not valid_for_model_training. "
            "Pass --allow-exploratory-incomplete-target to train on its complete rows anyway.",
            flush=True,
        )
        return 2

    result = run_model_sweep(
        project_root=project_root,
        dataset_path=dataset_path,
        base_config=base,
        backend_name=str(backend),
        include_linear_baseline=not bool(args.no_linear),
        extra_hidden_layers_text=str(args.extra_architectures or ""),
        status_callback=_status,
        training_provenance=training_provenance,
        seeds_per_architecture=int(args.seeds_per_architecture),
    )
    print(f"Sweep root: {result.sweep_root}", flush=True)
    print(f"Summary JSON: {result.summary_json_path}", flush=True)
    if result.best_model:
        print(f"Best: {result.best_model.get('model_label')} ({result.best_model.get('artifact_subdir')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
