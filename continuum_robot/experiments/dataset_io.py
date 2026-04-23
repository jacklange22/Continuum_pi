"""Canonical experiment dataset writer and loader."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
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


def canonical_timestamp_token(timestamp_utc: str | None = None) -> str:
    """Return one canonical UTC timestamp token for operator-facing outputs."""
    if timestamp_utc:
        raw = str(timestamp_utc).strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            match = re.search(r"(\d{8})[T_]?(\d{6})", raw)
            if match:
                return f"{match.group(1)}_{match.group(2)}"
        else:
            return parsed.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sanitize_output_name(name: str, *, default: str = "artifact") -> str:
    """Return a filesystem-safe artifact label for canonical output naming."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name or "").strip())
    return cleaned.strip("_") or default


def canonical_timestamped_name(
    root: Path,
    label: str,
    *,
    timestamp_utc: str | None = None,
    extension: str = "",
) -> str:
    """Return a collision-safe `YYYYMMDD_HHMMSS_label` name under one root."""
    root = Path(root)
    stamp = canonical_timestamp_token(timestamp_utc)
    safe_label = sanitize_output_name(label)
    safe_extension = str(extension or "")
    base_stem = f"{stamp}_{safe_label}"
    candidate = root / f"{base_stem}{safe_extension}"
    if not candidate.exists():
        return candidate.name
    suffix = 1
    while True:
        named = root / f"{base_stem}_{suffix:02d}{safe_extension}"
        if not named.exists():
            return named.name
        suffix += 1


def canonical_timestamped_path(
    root: Path,
    label: str,
    *,
    timestamp_utc: str | None = None,
    extension: str = "",
) -> Path:
    """Return a collision-safe canonical artifact path under one root."""
    root = Path(root)
    return root / canonical_timestamped_name(
        root,
        label,
        timestamp_utc=timestamp_utc,
        extension=extension,
    )


def default_experiment_output_dir_name(root: Path, experiment_name: str) -> str:
    """Return a timestamped canonical run-directory name with collision handling."""
    return canonical_timestamped_name(Path(root), experiment_name, extension="")


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
            output_dir = root / default_experiment_output_dir_name(root, metadata.experiment_name)
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
