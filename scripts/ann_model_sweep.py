#!/usr/bin/env python3
"""Run a shared-split ANN model sweep (linear ridge optional + default ANN sizes + extras)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from continuum_robot.modeling.ann_training import AnnTrainingConfig, run_model_sweep


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

    result = run_model_sweep(
        project_root=project_root,
        dataset_path=dataset_path,
        base_config=base,
        backend_name=str(backend),
        include_linear_baseline=not bool(args.no_linear),
        extra_hidden_layers_text=str(args.extra_architectures or ""),
        status_callback=_status,
    )
    print(f"Sweep root: {result.sweep_root}", flush=True)
    print(f"Summary JSON: {result.summary_json_path}", flush=True)
    if result.best_model:
        print(f"Best: {result.best_model.get('model_label')} ({result.best_model.get('artifact_subdir')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
