"""Generic parameter editor for experiment configs."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.experiment_parameters import ExperimentParameterField


class ExperimentParameterEditor(QWidget):
    """Render grouped generic experiment parameters without per-experiment widget branches."""

    fieldChanged = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._signature: list[tuple[str, str, str, str, bool]] = []
        self._editors: dict[str, QWidget] = {}
        self._error_labels: dict[str, QLabel] = {}
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(12)

    def set_fields(self, fields: list[ExperimentParameterField]) -> None:
        signature = [(field.key, field.group, field.label, field.value_kind, field.multiline) for field in fields]
        if signature != self._signature:
            self._rebuild(fields)
            self._signature = signature
            return
        for field in fields:
            self._update_field(field)

    def _rebuild(self, fields: list[ExperimentParameterField]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._editors.clear()
        self._error_labels.clear()
        groups: dict[str, list[ExperimentParameterField]] = {}
        for field in fields:
            groups.setdefault(field.group, []).append(field)
        for group_name, group_fields in groups.items():
            card = QFrame()
            card.setProperty("role", "card")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)
            title = QLabel(group_name)
            title.setProperty("role", "section-title")
            layout.addWidget(title)
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(10)
            for field in group_fields:
                editor = self._create_editor(field)
                row_widget = QWidget()
                row_layout = QVBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                row_layout.addWidget(editor)
                error_label = QLabel()
                error_label.setWordWrap(True)
                error_label.setVisible(False)
                error_label.setStyleSheet("color: #b91c1c; font-size: 12px;")
                row_layout.addWidget(error_label)
                form.addRow(field.label, row_widget)
                self._editors[field.key] = editor
                self._error_labels[field.key] = error_label
                self._update_field(field)
            layout.addLayout(form)
            self._layout.addWidget(card)
        self._layout.addStretch(1)

    def _create_editor(self, field: ExperimentParameterField) -> QWidget:
        if field.value_kind == "bool":
            editor = QComboBox()
            editor.addItem("True", "true")
            editor.addItem("False", "false")
            editor.currentIndexChanged.connect(
                lambda _index, key=field.key, widget=editor: self.fieldChanged.emit(key, str(widget.currentData()))
            )
            return editor
        if field.multiline:
            editor = QPlainTextEdit()
            editor.setMinimumHeight(88)
            editor.setTabStopDistance(24)
            editor.textChanged.connect(
                lambda key=field.key, widget=editor: self.fieldChanged.emit(key, widget.toPlainText())
            )
            return editor
        editor = QLineEdit()
        editor.editingFinished.connect(
            lambda key=field.key, widget=editor: self.fieldChanged.emit(key, widget.text())
        )
        return editor

    def _update_field(self, field: ExperimentParameterField) -> None:
        editor = self._editors.get(field.key)
        if editor is None:
            return
        if isinstance(editor, QComboBox):
            target = str(field.raw_value or "false").lower()
            index = editor.findData(target)
            if index < 0:
                index = 0
            if editor.currentIndex() != index:
                editor.blockSignals(True)
                editor.setCurrentIndex(index)
                editor.blockSignals(False)
        elif isinstance(editor, QPlainTextEdit):
            if editor.toPlainText() != field.raw_value and not editor.hasFocus():
                editor.blockSignals(True)
                editor.setPlainText(field.raw_value)
                editor.blockSignals(False)
        elif isinstance(editor, QLineEdit):
            if editor.text() != field.raw_value and not editor.hasFocus():
                editor.blockSignals(True)
                editor.setText(field.raw_value)
                editor.blockSignals(False)
        error_label = self._error_labels.get(field.key)
        if error_label is not None:
            error_label.setText(field.error or "")
            error_label.setVisible(bool(field.error))

