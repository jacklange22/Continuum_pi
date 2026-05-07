from __future__ import annotations

from continuum_robot.experiments.modeling_dataset_outputs import _write_workspace_report_plot
from continuum_robot.experiments.plotting import (
    FIGURE_QUALITY_DPI,
    create_figure,
    figure_dpi,
    save_figure,
    set_figure_output_quality,
    style_axes,
)


def test_figure_output_quality_mapping() -> None:
    assert set_figure_output_quality("low") == "low"
    assert figure_dpi() == FIGURE_QUALITY_DPI["low"]
    assert set_figure_output_quality("medium") == "medium"
    assert figure_dpi() == FIGURE_QUALITY_DPI["medium"]
    assert set_figure_output_quality("production") == "production"
    assert figure_dpi() == FIGURE_QUALITY_DPI["production"]
    assert set_figure_output_quality("unknown") == "production"


def test_plotting_helper_saves_png_and_labels_axes(tmp_path) -> None:
    fig, ax = create_figure(size="wide")
    ax.plot([0, 1], [0, 1])
    style_axes(ax, title="Demo Figure", xlabel="X position (mm)", ylabel="Y position (mm)")
    output = tmp_path / "figure.png"

    save_figure(fig, output, quality="low")

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")
    assert ax.get_xlabel() == "X position (mm)"
    assert ax.get_ylabel() == "Y position (mm)"


def test_modeling_workspace_report_plot_uses_expected_output_path(tmp_path) -> None:
    rows = [
        {"accepted": True, "tip_position_xyz_mm": [0.0, 0.0, 10.0]},
        {"accepted": True, "tip_position_xyz_mm": [1.0, 2.0, 11.0]},
        {"accepted": False, "tip_position_xyz_mm": [3.0, -1.0, 12.0]},
    ]
    metrics = {
        "accepted_sample_count": 2,
        "rejected_sample_count": 1,
        "dataset_mode": "workspace_coverage",
    }
    output = tmp_path / "modeling_workspace_coverage_report.png"

    _write_workspace_report_plot(workspace_plot_path=output, export_rows=rows, metrics=metrics)

    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")
