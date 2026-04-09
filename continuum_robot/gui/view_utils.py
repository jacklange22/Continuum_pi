"""Shared GUI update helpers that avoid user-hostile refresh behavior."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QLineEdit, QPlainTextEdit, QTextEdit


TextDocumentWidget = QPlainTextEdit | QTextEdit


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
    if skip_if_focused and widget.hasFocus():
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
    if skip_if_focused and widget.hasFocus():
        return False
    context = QSignalBlocker(widget) if block_signals else nullcontext()
    with context:
        widget.setText(new_text)
    return True
