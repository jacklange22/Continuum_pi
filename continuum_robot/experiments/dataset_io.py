"""Canonical experiment dataset writer and loader."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import yaml

from continuum_robot.experiments.schemas import (
    ExperimentDatasetBundle,
    ExperimentDatasetPaths,
    ExperimentMetadata,
    ExperimentSummary,
    ExperimentTimeseriesSample,
)


def canonical_experiment_output_root(output_root: Path, experiment_name: str) -> Path:
    """Return the canonical directory root for one experiment type."""

    root = Path(output_root)
    safe_name = str(experiment_name or "").strip().replace(" ", "_")
    if not safe_name:
        return root
    return root if root.name == safe_name else root / safe_name


class ExperimentDatasetWriter:
    """Write canonical experiment datasets to one run directory."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def write_dataset(
        self,
        metadata: ExperimentMetadata,
        samples: list[ExperimentTimeseriesSample],
        summary: ExperimentSummary,
        *,
        output_root: Path | None = None,
        output_dir_name: str | None = None,
    ) -> ExperimentDatasetPaths:
        """Write one dataset bundle and return its paths."""
        base_root = Path(output_root) if output_root is not None else self.output_root
        root = canonical_experiment_output_root(base_root, metadata.experiment_name)
        root.mkdir(parents=True, exist_ok=True)
        if output_dir_name:
            output_dir = root / str(output_dir_name)
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = root / f"{timestamp}_{metadata.run_id}"
        output_dir.mkdir(parents=True, exist_ok=False)
        metadata_path = output_dir / "metadata.json"
        samples_path = output_dir / "samples.jsonl"
        summary_path = output_dir / "summary.json"
        config_snapshot_path = output_dir / "config_snapshot.yaml"
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
        with samples_path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
        summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        config_snapshot_path.write_text(
            yaml.safe_dump(metadata.config_used or {}, sort_keys=False) or "{}\n",
            encoding="utf-8",
        )
        return ExperimentDatasetPaths(
            output_dir=output_dir,
            metadata_path=metadata_path,
            samples_path=samples_path,
            summary_path=summary_path,
            config_snapshot_path=config_snapshot_path,
        )


class ExperimentDatasetLoader:
    """Load canonical experiment datasets from disk."""

    def load_dataset(self, path: Path) -> ExperimentDatasetBundle:
        """Load a dataset bundle from a run directory or metadata/summary file path."""
        root = Path(path)
        if root.is_file():
            root = root.parent
        metadata_path = root / "metadata.json"
        samples_path = root / "samples.jsonl"
        summary_path = root / "summary.json"
        metadata = ExperimentMetadata.from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))
        summary = ExperimentSummary.from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
        samples: list[ExperimentTimeseriesSample] = []
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                samples.append(ExperimentTimeseriesSample.from_dict(json.loads(raw)))
        return ExperimentDatasetBundle(
            metadata=metadata,
            samples=samples,
            summary=summary,
            paths=ExperimentDatasetPaths(
                output_dir=root,
                metadata_path=metadata_path,
                samples_path=samples_path,
                summary_path=summary_path,
                config_snapshot_path=(root / "config_snapshot.yaml"),
            ),
        )
