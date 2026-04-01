from pathlib import Path

import numpy as np
import pytest

from continuum_robot.experiments.pivot_utils import (
    PivotInputParseError,
    load_pivot_transforms_with_report,
)


def _write_headered_pivot_csv(path: Path, rows: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float]]]) -> Path:
    lines = ["tool_id,qw,qx,qy,qz,x,y,z"]
    for tool_id, quat, trans in rows:
        lines.append(
            ",".join(
                [
                    tool_id,
                    *(f"{float(value):0.9f}" for value in quat),
                    *(f"{float(value):0.9f}" for value in trans),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_pivot_transforms_with_report_parses_canonical_headered_csv_and_filters_tools(tmp_path: Path) -> None:
    csv_path = _write_headered_pivot_csv(
        tmp_path / "pivot_headered.csv",
        [
            ("0A", (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            ("0B", (1.0, 0.0, 0.0, 0.0), (10.0, 20.0, 30.0)),
            ("0B", (0.70710678, 0.70710678, 0.0, 0.0), (11.0, 21.0, 31.0)),
        ],
    )

    result = load_pivot_transforms_with_report(csv_path, tool_id="0B")

    assert result.report.detected_format == "canonical_headered_csv"
    assert result.report.usable_rows == 2
    assert result.report.filtered_other_tool_rows == 1
    assert np.allclose(result.transforms[0][0:3, 3], [10.0, 20.0, 30.0])
    assert np.allclose(result.transforms[1][0:3, 3], [11.0, 21.0, 31.0])


def test_load_pivot_transforms_with_report_requires_canonical_header_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "pivot_missing_column.csv"
    csv_path.write_text(
        "\n".join(
            [
                "tool_id,qw,qx,qy,x,y,z",
                "0B,1.0,0.0,0.0,10.0,20.0,30.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PivotInputParseError, match="CSV header detected but required columns missing: qz"):
        load_pivot_transforms_with_report(csv_path, tool_id="0B")


def test_load_pivot_transforms_with_report_reports_malformed_rows_with_useful_message(tmp_path: Path) -> None:
    csv_path = tmp_path / "pivot_bad_rows.csv"
    csv_path.write_text(
        "\n".join(
            [
                "tool_id,qw,qx,qy,qz,x,y,z",
                "0B,1.0,0.0,0.0,0.0,bad,20.0,30.0",
                "0B,1.0,0.0,0.0,0.0,,,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PivotInputParseError, match="No usable 0B samples found"):
        load_pivot_transforms_with_report(csv_path, tool_id="0B")

    try:
        load_pivot_transforms_with_report(csv_path, tool_id="0B")
    except PivotInputParseError as exc:
        assert exc.report.detected_format == "canonical_headered_csv"
        assert exc.report.usable_rows == 0
        assert exc.report.rejected_rows[0]["row"] == 2
        assert "non-float pose value" in str(exc.report.rejected_rows[0]["reason"])
        assert exc.report.rejected_rows[1]["row"] == 3
        assert "missing values for x, y, z" in str(exc.report.rejected_rows[1]["reason"])


def test_load_pivot_transforms_with_report_rejects_no_usable_target_rows(tmp_path: Path) -> None:
    csv_path = _write_headered_pivot_csv(
        tmp_path / "pivot_only_0a.csv",
        [
            ("0A", (1.0, 0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
            ("0A", (1.0, 0.0, 0.0, 0.0), (4.0, 5.0, 6.0)),
        ],
    )

    with pytest.raises(PivotInputParseError, match="No usable 0B samples found"):
        load_pivot_transforms_with_report(csv_path, tool_id="0B")


def test_load_pivot_transforms_with_report_supports_legacy_aurora_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "legacy_pivot.csv"
    csv_path.write_text(
        "\n".join(
            [
                "0A,1.0,0.0,0.0,0.0,1.0,2.0,3.0",
                "0B,1.0,0.0,0.0,0.0,10.0,20.0,30.0",
                "0B,0.70710678,0.70710678,0.0,0.0,11.0,21.0,31.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = load_pivot_transforms_with_report(csv_path, tool_id="0B")

    assert result.report.detected_format == "legacy_aurora_csv"
    assert result.report.usable_rows == 2
    assert result.report.filtered_other_tool_rows == 1
