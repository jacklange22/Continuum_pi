"""Combo box helpers that avoid accidental scroll-wheel selection changes."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox


class NoWheelComboBox(QComboBox):
    """A QComboBox that ignores wheel events unless its popup is open."""

    def __init__(self, *args, max_visible_items: int = 12, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._popup_open = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMaxVisibleItems(int(max(3, max_visible_items)))

    @property
    def popup_open(self) -> bool:
        """Return whether the drop-down list is currently open."""
        return bool(self._popup_open or self.view().isVisible())

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        self._popup_open = True
        super().showPopup()

    def hidePopup(self) -> None:  # noqa: N802 - Qt override
        super().hidePopup()
        self._popup_open = False

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.popup_open:
            super().wheelEvent(event)
            return
        event.ignore()
