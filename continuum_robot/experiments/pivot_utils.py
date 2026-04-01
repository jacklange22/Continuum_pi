"""Least-squares pivot calibration helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from continuum_robot.experiments.dataset_tools import extract_tool_transforms_from_dataset
from continuum_robot.registration.legacy_compat import AuroraPoseSample, parse_aurora_csv
from continuum_robot.tracking.transforms import quat_wxyz_to_rotmat


@dataclass
class PivotCalibrationResult:
    """Output of a pivot calibration solve."""

    tip_vector_local_mm: list[float]
    pivot_point_tracker_mm: list[float]
    rmse_mm: float
    sample_count_total: int
    sample_count_used: int
    sample_count_rejected: int
    residuals_mm: list[list[float]]
    inlier_mask: list[bool]
    rejected_indices: list[int]


@dataclass
class PivotCsvLoadReport:
    """Operator-facing parse report for pivot input files."""

    source_path: str
    detected_format: str
    tool_id: str
    header_columns: list[str] = field(default_factory=list)
    total_data_rows: int = 0
    usable_rows: int = 0
    filtered_other_tool_rows: int = 0
    rejected_rows: list[dict[str, object]] = field(default_factory=list)

    def to_metrics(self) -> dict[str, object]:
        """Convert the parse report into experiment-summary metrics."""
        return {
            "pivot_input_path": self.source_path,
            "pivot_input_format": self.detected_format,
            "pivot_input_tool_id": self.tool_id,
            "pivot_input_header_columns": list(self.header_columns),
            "pivot_input_total_rows": int(self.total_data_rows),
            "pivot_input_usable_rows": int(self.usable_rows),
            "pivot_input_filtered_other_tool_rows": int(self.filtered_other_tool_rows),
            "pivot_input_rejected_row_count": len(self.rejected_rows),
            "pivot_input_rejected_rows": [dict(row) for row in self.rejected_rows],
        }


@dataclass
class PivotTransformLoadResult:
    """Transforms plus a parse/load report."""

    transforms: list[np.ndarray]
    report: PivotCsvLoadReport


class PivotInputParseError(ValueError):
    """Raised when a pivot input file cannot be parsed into usable tool samples."""

    def __init__(self, message: str, report: PivotCsvLoadReport) -> None:
        super().__init__(message)
        self.report = report


_CANONICAL_PIVOT_COLUMNS = ("tool_id", "qw", "qx", "qy", "qz", "x", "y", "z")


def solve_pivot_calibration(
    rotations: list[np.ndarray],
    translations_mm: list[np.ndarray],
    *,
    std_dev_threshold: float = 3.0,
    min_samples: int = 8,
) -> PivotCalibrationResult:
    """Solve the standard least-squares pivot calibration problem."""
    if len(rotations) != len(translations_mm):
        raise ValueError("rotations and translations_mm length mismatch")
    if len(rotations) < int(min_samples):
        raise ValueError("Insufficient samples for pivot calibration")
    A, b = _assemble_system(rotations, translations_mm)
    x_initial, *_ = np.linalg.lstsq(A, b, rcond=None)
    residuals_initial = (A @ x_initial - b).reshape(-1, 3)
    keep_mask = _inlier_mask_from_residuals(residuals_initial, std_dev_threshold=std_dev_threshold)
    if int(np.count_nonzero(keep_mask)) < int(min_samples):
        raise ValueError("Pivot calibration has insufficient inlier samples after outlier rejection")
    inlier_rows = np.repeat(keep_mask, 3)
    A_inlier = A[inlier_rows, :]
    b_inlier = b[inlier_rows]
    x_final, *_ = np.linalg.lstsq(A_inlier, b_inlier, rcond=None)
    residuals_final = (A_inlier @ x_final - b_inlier).reshape(-1, 3)
    rmse_mm = float(np.sqrt(np.mean(np.sum(residuals_final**2, axis=1))))
    return PivotCalibrationResult(
        tip_vector_local_mm=[float(value) for value in x_final[0:3]],
        pivot_point_tracker_mm=[float(value) for value in x_final[3:6]],
        rmse_mm=rmse_mm,
        sample_count_total=len(rotations),
        sample_count_used=int(np.count_nonzero(keep_mask)),
        sample_count_rejected=int(len(rotations) - int(np.count_nonzero(keep_mask))),
        residuals_mm=[[float(value) for value in row] for row in residuals_final],
        inlier_mask=[bool(value) for value in keep_mask.tolist()],
        rejected_indices=[int(index) for index, keep in enumerate(keep_mask.tolist()) if not keep],
    )


def load_pivot_transforms(
    path: Path,
    *,
    tool_id: str = "0B",
) -> list[np.ndarray]:
    """Load pivot pose transforms from a canonical dataset or CSV."""
    return load_pivot_transforms_with_report(path, tool_id=tool_id).transforms


def load_pivot_transforms_with_report(
    path: Path,
    *,
    tool_id: str = "0B",
) -> PivotTransformLoadResult:
    """Load pivot transforms and return a parse report suitable for operator review."""
    source = Path(path)
    if source.suffix.lower() != ".csv":
        transforms = extract_tool_transforms_from_dataset(source, tool_id=tool_id)
        report = PivotCsvLoadReport(
            source_path=str(source),
            detected_format="canonical_dataset_bundle",
            tool_id=tool_id,
            total_data_rows=len(transforms),
            usable_rows=len(transforms),
        )
        return PivotTransformLoadResult(transforms=transforms, report=report)

    format_hint = _detect_csv_format(source)
    if format_hint == "canonical_headered_csv":
        return _load_canonical_headered_pivot_csv(source, tool_id=tool_id)
    if format_hint == "legacy_aurora_csv":
        return _load_legacy_pivot_csv(source, tool_id=tool_id)
    report = PivotCsvLoadReport(source_path=str(source), detected_format="unknown_csv", tool_id=tool_id)
    raise PivotInputParseError(
        f"Pivot input CSV format is ambiguous: {source}. Expected either the canonical header "
        f"{','.join(_CANONICAL_PIVOT_COLUMNS)} or a supported legacy Aurora CSV layout.",
        report,
    )


def write_tip_vector_file(path: Path, tip_vector_local_mm: list[float]) -> Path:
    """Write the pen-probe tip vector in the repo's expected 3-value format."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = ",".join(f"{float(value):+0.4f}" for value in tip_vector_local_mm)
    target.write_text(payload, encoding="utf-8")
    return target


def _assemble_system(rotations: list[np.ndarray], translations_mm: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    for rotation, translation in zip(rotations, translations_mm):
        R = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float).reshape(3)
        rows.append(np.hstack([R, -1.0 * np.eye(3)]))
        rhs.append(-t.reshape(3, 1))
    A = np.vstack(rows)
    b = np.vstack(rhs).reshape(-1)
    return A, b


def _inlier_mask_from_residuals(residuals_mm: np.ndarray, *, std_dev_threshold: float) -> np.ndarray:
    residuals = np.asarray(residuals_mm, dtype=float)
    means = residuals.mean(axis=0)
    stdevs = residuals.std(axis=0, ddof=0)
    deviations = np.zeros_like(residuals)
    for axis in range(3):
        if stdevs[axis] <= 1e-12:
            deviations[:, axis] = 0.0
        else:
            deviations[:, axis] = np.abs((residuals[:, axis] - means[axis]) / stdevs[axis])
    outlier_mask = np.any(deviations > float(std_dev_threshold), axis=1)
    return ~outlier_mask


def _detect_csv_format(path: Path) -> str:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            cells = [str(item).strip() for item in row if str(item).strip() != ""]
            if not cells:
                continue
            lowered = [cell.lower() for cell in cells]
            if "tool_id" in lowered or any(cell in {"qw", "qx", "qy", "qz", "x", "y", "z"} for cell in lowered):
                missing = [column for column in _CANONICAL_PIVOT_COLUMNS if column not in lowered]
                if missing:
                    report = PivotCsvLoadReport(
                        source_path=str(path),
                        detected_format="canonical_headered_csv",
                        tool_id="",
                        header_columns=lowered,
                    )
                    raise PivotInputParseError(
                        "CSV header detected but required columns missing: "
                        + ", ".join(missing)
                        + f". Expected columns: {','.join(_CANONICAL_PIVOT_COLUMNS)}",
                        report,
                    )
                return "canonical_headered_csv"
            return "legacy_aurora_csv"
    raise PivotInputParseError(
        f"Pivot input CSV contains no rows: {path}",
        PivotCsvLoadReport(source_path=str(path), detected_format="empty_csv", tool_id=""),
    )


def _load_canonical_headered_pivot_csv(path: Path, *, tool_id: str) -> PivotTransformLoadResult:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            report = PivotCsvLoadReport(source_path=str(path), detected_format="canonical_headered_csv", tool_id=tool_id)
            raise PivotInputParseError(f"Pivot input CSV has no header row: {path}", report)
        header_lookup = {str(field).strip().lower(): str(field) for field in reader.fieldnames if field is not None}
        missing = [column for column in _CANONICAL_PIVOT_COLUMNS if column not in header_lookup]
        report = PivotCsvLoadReport(
            source_path=str(path),
            detected_format="canonical_headered_csv",
            tool_id=tool_id,
            header_columns=list(header_lookup.keys()),
        )
        if missing:
            raise PivotInputParseError(
                "CSV header detected but required columns missing: "
                + ", ".join(missing)
                + f". Expected columns: {','.join(_CANONICAL_PIVOT_COLUMNS)}",
                report,
            )
        samples: list[AuroraPoseSample] = []
        for row_index, raw in enumerate(reader, start=2):
            if raw is None:
                continue
            if all((value is None or str(value).strip() == "") for value in raw.values()):
                continue
            report.total_data_rows += 1
            raw_tool_value = raw.get(header_lookup["tool_id"], "")
            raw_tool_id = "" if raw_tool_value is None else str(raw_tool_value).strip()
            if not raw_tool_id:
                report.rejected_rows.append({"row": row_index, "reason": "missing tool_id"})
                continue
            if raw_tool_id != tool_id:
                report.filtered_other_tool_rows += 1
                continue
            missing_values = [
                column
                for column in _CANONICAL_PIVOT_COLUMNS
                if _is_blank_csv_value(raw.get(header_lookup[column], ""))
            ]
            if missing_values:
                report.rejected_rows.append(
                    {"row": row_index, "reason": "missing values for " + ", ".join(missing_values)}
                )
                continue
            try:
                quat = tuple(float(raw[header_lookup[column]]) for column in ("qw", "qx", "qy", "qz"))
                trans = tuple(float(raw[header_lookup[column]]) for column in ("x", "y", "z"))
            except (TypeError, ValueError) as exc:
                report.rejected_rows.append({"row": row_index, "reason": f"non-float pose value: {exc}"})
                continue
            samples.append(
                AuroraPoseSample(
                    tool_id=raw_tool_id,
                    quaternion_wxyz=quat,  # type: ignore[arg-type]
                    translation_mm=trans,  # type: ignore[arg-type]
                    source_row=row_index,
                    source_token="canonical_headered_csv",
                )
            )
        report.usable_rows = len(samples)
    if not samples:
        rejected_summary = _render_rejected_rows(report.rejected_rows)
        message = (
            f"No usable {tool_id} samples found in canonical headered CSV {path}. "
            f"Filtered other-tool rows={report.filtered_other_tool_rows}, rejected rows={len(report.rejected_rows)}."
        )
        if rejected_summary:
            message += f" Rejected rows: {rejected_summary}"
        raise PivotInputParseError(message, report)
    return PivotTransformLoadResult(transforms=_samples_to_transforms(samples), report=report)


def _load_legacy_pivot_csv(path: Path, *, tool_id: str) -> PivotTransformLoadResult:
    report = PivotCsvLoadReport(
        source_path=str(path),
        detected_format="legacy_aurora_csv",
        tool_id=tool_id,
    )
    try:
        parsed = parse_aurora_csv(path)
    except Exception as exc:
        raise PivotInputParseError(f"Legacy Aurora CSV parse failed for {path}: {exc}", report) from exc
    all_samples = [sample for samples in parsed.values() for sample in samples]
    report.total_data_rows = len(all_samples)
    target_samples = list(parsed.get(tool_id, []))
    report.usable_rows = len(target_samples)
    report.filtered_other_tool_rows = max(0, len(all_samples) - len(target_samples))
    if not target_samples:
        raise PivotInputParseError(
            f"No usable {tool_id} samples found in legacy Aurora CSV {path}. "
            f"Rows parsed={report.total_data_rows}, filtered other-tool rows={report.filtered_other_tool_rows}.",
            report,
        )
    return PivotTransformLoadResult(transforms=_samples_to_transforms(target_samples), report=report)


def _samples_to_transforms(samples: list[AuroraPoseSample]) -> list[np.ndarray]:
    transforms: list[np.ndarray] = []
    for sample in samples:
        T = np.eye(4, dtype=float)
        T[0:3, 0:3] = quat_wxyz_to_rotmat(sample.quaternion_wxyz)
        T[0:3, 3] = np.asarray(sample.translation_mm, dtype=float)
        transforms.append(T)
    return transforms


def _render_rejected_rows(rejected_rows: list[dict[str, object]], *, limit: int = 4) -> str:
    preview = rejected_rows[:limit]
    rendered = ", ".join(
        f"row {row.get('row')}: {row.get('reason')}"
        for row in preview
    )
    if len(rejected_rows) > limit:
        rendered += f", and {len(rejected_rows) - limit} more"
    return rendered


def _is_blank_csv_value(value: object) -> bool:
    return value is None or str(value).strip() == ""
