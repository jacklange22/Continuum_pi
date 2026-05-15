from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.gui

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from continuum_robot.gui.widgets.no_wheel_combo_box import NoWheelComboBox


def test_no_wheel_combo_box_ignores_closed_wheel_events() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    combo = NoWheelComboBox()
    combo.addItem("A", "a")
    combo.addItem("B", "b")
    combo.setCurrentIndex(0)
    event = QWheelEvent(
        QPointF(1, 1),
        QPointF(1, 1),
        QPoint(0, 120),
        QPoint(0, 120),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )

    combo.wheelEvent(event)

    assert combo.currentIndex() == 0


def test_no_wheel_combo_box_opening_popup_does_not_commit_selection() -> None:
    app = QApplication.instance() or QApplication([])
    _ = app
    combo = NoWheelComboBox()
    combo.addItem("A", "a")
    combo.addItem("B", "b")
    combo.setCurrentIndex(0)

    combo.showPopup()
    QTest.qWait(10)
    combo.hidePopup()

    assert combo.currentData() == "a"
