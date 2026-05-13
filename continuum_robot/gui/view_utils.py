"""Shared GUI update helpers that avoid user-hostile refresh behavior."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QSpinBox, QSplitter, QTextEdit


TextDocumentWidget = QPlainTextEdit | QTextEdit


class ResponsiveSplitterController:
    """Switch one splitter between wide and stacked modes based on available width."""

    def __init__(
        self,
        splitter: QSplitter,
        *,
        collapse_below_width: int,
        horizontal_sizes: list[int] | tuple[int, ...] | None = None,
        vertical_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        self.splitter = splitter
        self.collapse_below_width = int(collapse_below_width)
        self.horizontal_sizes = list(horizontal_sizes or [])
        self.vertical_sizes = list(vertical_sizes or [])

    def apply(self, available_width: int) -> bool:
        """Apply the correct orientation for the current available width."""

        target_orientation = (
            Qt.Vertical if int(available_width) < self.collapse_below_width else Qt.Horizontal
        )
        if self.splitter.orientation() == target_orientation:
            return False
        self.splitter.setOrientation(target_orientation)
        if target_orientation == Qt.Horizontal and self.horizontal_sizes:
            self.splitter.setSizes(list(self.horizontal_sizes))
        elif target_orientation == Qt.Vertical and self.vertical_sizes:
            self.splitter.setSizes(list(self.vertical_sizes))
        return True


def preserve_scroll_position(
    widget,
    update_fn: Callable[[], None],
    *,
    stick_to_bottom_if_at_bottom: bool = False,
) -> None:
    """Run one widget update while preserving the current scroll position."""

    v_scroll = getattr(widget, "verticalScrollBar", lambda: None)()
    h_scroll = getattr(widget, "horizontalScrollBar", lambda: None)()
    if v_scroll is None or h_scroll is None:
        update_fn()
        return
    old_v = v_scroll.value()
    old_h = h_scroll.value()
    was_at_bottom = old_v >= max(0, v_scroll.maximum() - 2)
    update_fn()
    if stick_to_bottom_if_at_bottom and was_at_bottom:
        v_scroll.setValue(v_scroll.maximum())
    else:
        v_scroll.setValue(min(old_v, v_scroll.maximum()))
    h_scroll.setValue(min(old_h, h_scroll.maximum()))


def set_text_document(
    widget: TextDocumentWidget,
    text: str,
    *,
    skip_if_focused: bool = False,
    block_signals: bool = False,
    stick_to_bottom_if_at_bottom: bool = True,
) -> bool:
    """Update a text document widget only when needed and preserve view state."""

    new_text = str(text)
    if widget.toPlainText() == new_text:
        return False
    if skip_if_focused and editable_update_blocked(widget):
        return False

    def _apply() -> None:
        context = QSignalBlocker(widget) if block_signals else nullcontext()
        with context:
            widget.setPlainText(new_text)

    preserve_scroll_position(
        widget,
        _apply,
        stick_to_bottom_if_at_bottom=stick_to_bottom_if_at_bottom,
    )
    return True


def set_line_edit_text(
    widget: QLineEdit,
    text: str,
    *,
    skip_if_focused: bool = False,
    block_signals: bool = False,
) -> bool:
    """Update a line edit only when its value actually changed."""

    new_text = str(text)
    if widget.text() == new_text:
        return False
    if skip_if_focused and editable_update_blocked(widget):
        return False
    context = QSignalBlocker(widget) if block_signals else nullcontext()
    with context:
        widget.setText(new_text)
    return True


def editable_update_blocked(widget) -> bool:
    """Return True when a background refresh should not overwrite an editable widget."""
    if widget.hasFocus():
        return True
    is_modified = getattr(widget, "isModified", None)
    if callable(is_modified) and bool(is_modified()):
        return True
    document = getattr(widget, "document", lambda: None)()
    if document is not None and getattr(document, "isModified", lambda: False)():
        return True
    view = getattr(widget, "view", lambda: None)()
    if view is not None and getattr(view, "isVisible", lambda: False)():
        return True
    qt_property = getattr(widget, "property", lambda _name: None)("popup_open")
    if bool(qt_property) or bool(getattr(widget, "popup_open", False)):
        return True
    return False


def set_spinbox_value(
    widget: QSpinBox | QDoubleSpinBox,
    value: int | float,
    *,
    skip_if_editing: bool = True,
    block_signals: bool = False,
) -> bool:
    """Update a spinbox without clobbering in-progress operator edits."""
    numeric_value = float(value) if isinstance(widget, QDoubleSpinBox) else int(value)
    if skip_if_editing and editable_update_blocked(widget):
        return False
    if isinstance(widget, QDoubleSpinBox):
        if abs(float(widget.value()) - float(numeric_value)) < 1e-12:
            return False
    elif int(widget.value()) == int(numeric_value):
        return False
    context = QSignalBlocker(widget) if block_signals else nullcontext()
    with context:
        widget.setValue(numeric_value)
    return True


def set_combo_value(
    widget: QComboBox,
    value: str | int | bool,
    *,
    skip_if_editing: bool = True,
    block_signals: bool = False,
) -> bool:
    """Set a combo by data/text while preserving active popup/focused edits."""
    if skip_if_editing and editable_update_blocked(widget):
        return False
    target = value
    index = widget.findData(target)
    if index < 0:
        index = widget.findData(str(target))
    if index < 0:
        index = widget.findText(str(target))
    if index < 0 or widget.currentIndex() == index:
        return False
    context = QSignalBlocker(widget) if block_signals else nullcontext()
    with context:
        widget.setCurrentIndex(index)
    return True
