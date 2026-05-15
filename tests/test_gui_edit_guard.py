from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytestmark = [pytest.mark.gui]

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox

from continuum_robot.gui.view_utils import set_combo_value, set_line_edit_text, set_spinbox_value


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_line_edit_refresh_does_not_overwrite_dirty_edit() -> None:
    _app()
    edit = QLineEdit()
    edit.setText("operator typing")
    edit.setModified(True)

    changed = set_line_edit_text(edit, "background refresh", skip_if_focused=True, block_signals=True)

    assert changed is False
    assert edit.text() == "operator typing"


def test_spinbox_refresh_does_not_overwrite_focused_value() -> None:
    app = _app()
    spin = QSpinBox()
    spin.show()
    spin.setFocus()
    app.processEvents()
    spin.setValue(7)

    changed = set_spinbox_value(spin, 42, skip_if_editing=True, block_signals=True)

    assert changed is False
    assert spin.value() == 7


def test_double_spinbox_refresh_does_not_overwrite_focused_value() -> None:
    app = _app()
    spin = QDoubleSpinBox()
    spin.show()
    spin.setFocus()
    app.processEvents()
    spin.setValue(1.25)

    changed = set_spinbox_value(spin, 9.5, skip_if_editing=True, block_signals=True)

    assert changed is False
    assert spin.value() == 1.25


def test_combo_refresh_does_not_overwrite_popup_open_value() -> None:
    _app()
    combo = QComboBox()
    combo.addItem("A", "a")
    combo.addItem("B", "b")
    combo.setProperty("popup_open", True)
    combo.setCurrentIndex(0)

    changed = set_combo_value(combo, "b", skip_if_editing=True, block_signals=True)

    assert changed is False
    assert combo.currentData() == "a"


def test_edit_guard_allows_refresh_after_commit_state_cleared() -> None:
    _app()
    edit = QLineEdit()
    edit.setText("old")
    edit.setModified(False)

    changed = set_line_edit_text(edit, "new", skip_if_focused=True, block_signals=True)

    assert changed is True
    assert edit.text() == "new"
