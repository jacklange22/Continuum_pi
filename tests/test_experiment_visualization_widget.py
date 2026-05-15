import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QApplication

from continuum_robot.gui.experiment_visualization import ScatterSeries3D
from continuum_robot.gui.widgets.experiment_3d_widget import (
    BACKEND_PLACEHOLDER,
    BACKEND_PROJECTION,
    Experiment3DWidget,
    resolve_visualization_backend,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_resolve_visualization_backend_uses_projection_on_macos() -> None:
    backend, message = resolve_visualization_backend(
        requested_mode="auto",
        safe_effects=True,
        platform_name="darwin",
        qpa_platform="cocoa",
        qt_3d_import_ok=True,
    )

    assert backend == BACKEND_PROJECTION
    assert "macOS" in message


def test_resolve_visualization_backend_uses_placeholder_when_headless() -> None:
    backend, message = resolve_visualization_backend(
        requested_mode="3d",
        safe_effects=True,
        platform_name="linux",
        qpa_platform="offscreen",
        qt_3d_import_ok=True,
    )

    assert backend == BACKEND_PLACEHOLDER
    assert "Headless" in message


def test_resolve_visualization_backend_honors_projection_mode() -> None:
    backend, message = resolve_visualization_backend(
        requested_mode="2d",
        safe_effects=True,
        platform_name="linux",
        qpa_platform="xcb",
        qt_3d_import_ok=True,
    )

    assert backend == BACKEND_PROJECTION
    assert "Projection" in message or "projection" in message


def test_experiment_3d_widget_headless_mode_stays_safe() -> None:
    _app()
    widget = Experiment3DWidget(requested_mode="2d", safe_effects=True)
    widget.set_series(
        [
            ScatterSeries3D(
                name="Measured",
                color_hex="#2563eb",
                points_xyz=[(0.0, 0.0, 0.0), (10.0, 5.0, 2.0)],
            )
        ]
    )
    widget.set_view_options(show_axes=True, show_labels=False)

    assert widget.backend_mode == BACKEND_PLACEHOLDER
    assert widget.mode_label.text() == "Visualization Placeholder"
    assert "Headless" in widget.mode_label.toolTip()
