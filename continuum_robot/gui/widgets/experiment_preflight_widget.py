"""Preflight status widget for the experiment workspace."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.experiment_preflight import (
    PREFLIGHT_BLOCKED,
    PREFLIGHT_INFO,
    PREFLIGHT_OK,
    PREFLIGHT_WARNING,
    RUN_BLOCKED,
    RUN_OK,
    RUN_WARNING,
)
from continuum_robot.gui.theme import COLORS, chip_stylesheet, semantic_chip_colors
from continuum_robot.gui.view_utils import preserve_scroll_position


_ATTENTION_STATUSES = (PREFLIGHT_BLOCKED, PREFLIGHT_WARNING)


class ExperimentPreflightWidget(QWidget):
    """Render experiment preflight checks and overall run readiness."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._report_signature: str | None = None
        self._show_all = False
        self._last_report = None

        self.status_chip = QLabel("Preflight")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setMinimumWidth(108)
        self.summary_label = QLabel("Preflight not evaluated.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(f"color: {COLORS.text_secondary};")

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(10)
        summary_row.addWidget(self.status_chip, 0)
        summary_row.addWidget(self.summary_label, 1)

        self.empty_label = QLabel("All checks passed.")
        self.empty_label.setStyleSheet(f"color: {COLORS.text_muted}; padding: 4px 2px;")
        self.empty_label.setVisible(False)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setSpacing(8)
        self.list.setStyleSheet("border: none; background: transparent;")
        self.list.setMaximumHeight(260)

        self.toggle_button = QPushButton("Show all checks")
        self.toggle_button.setProperty("variant", "ghost")
        self.toggle_button.setCheckable(True)
        self.toggle_button.toggled.connect(self._on_toggle)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(summary_row)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.list)
        layout.addWidget(self.toggle_button, 0, Qt.AlignLeft)

    def set_report(self, report) -> None:
        signature = repr(asdict(report))
        if self._report_signature == signature:
            return
        self._report_signature = signature
        self._last_report = report
        chip_kind, chip_text = {
            RUN_OK: ("ready", "Ready"),
            RUN_WARNING: ("warning", "Warning"),
            RUN_BLOCKED: ("blocked", "Blocked"),
        }.get(report.overall_status, ("neutral", "Info"))
        bg, fg = semantic_chip_colors(chip_kind)
        self.status_chip.setText(chip_text)
        self.status_chip.setStyleSheet(chip_stylesheet(background=bg, foreground=fg))
        self.summary_label.setText(report.summary)
        self._render_list()

    def _on_toggle(self, checked: bool) -> None:
        self._show_all = bool(checked)
        self.toggle_button.setText("Hide details" if checked else "Show all checks")
        self._render_list()

    def _render_list(self) -> None:
        report = self._last_report
        if report is None:
            return
        if self._show_all:
            visible = list(report.checks)
        else:
            visible = [check for check in report.checks if check.status in _ATTENTION_STATUSES]
        hidden_count = len(report.checks) - len(visible)
        if not self._show_all and hidden_count > 0:
            self.toggle_button.setText(f"Show all checks ({hidden_count} passing)")
            self.toggle_button.setVisible(True)
        elif self._show_all:
            self.toggle_button.setVisible(True)
        else:
            self.toggle_button.setVisible(False)

        if not visible:
            self.list.setVisible(False)
            self.empty_label.setVisible(True)
            self.list.clear()
            return
        self.list.setVisible(True)
        self.empty_label.setVisible(False)

        def _rebuild() -> None:
            self.list.clear()
            for check in visible:
                widget = _PreflightRowWidget(check.label, _severity_label(check.status), check.message, check.status)
                item = QListWidgetItem()
                item.setSizeHint(widget.sizeHint())
                item.setData(Qt.UserRole, check.key)
                self.list.addItem(item)
                self.list.setItemWidget(item, widget)

        preserve_scroll_position(self.list, _rebuild)


class _PreflightRowWidget(QWidget):
    def __init__(self, label: str, severity: str, message: str, status: str, parent=None) -> None:
        super().__init__(parent)
        severity_label = QLabel(severity)
        bg, fg = _severity_colors(status)
        severity_label.setStyleSheet(chip_stylesheet(background=bg, foreground=fg))
        title_label = QLabel(label)
        title_label.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 700;")
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet(f"color: {COLORS.text_muted};")

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_row.addWidget(title_label, 1)
        title_row.addWidget(severity_label, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        layout.addLayout(title_row)
        layout.addWidget(message_label)
        self.setStyleSheet(
            f"background: {COLORS.surface_bg}; border: 1px solid {COLORS.surface_border}; border-radius: 10px;"
        )


def _severity_colors(status: str) -> tuple[str, str]:
    return {
        PREFLIGHT_OK: semantic_chip_colors("ready"),
        PREFLIGHT_WARNING: semantic_chip_colors("warning"),
        PREFLIGHT_BLOCKED: semantic_chip_colors("blocked"),
        PREFLIGHT_INFO: semantic_chip_colors("info"),
    }.get(status, semantic_chip_colors("neutral"))


def _severity_label(status: str) -> str:
    return {
        PREFLIGHT_OK: "Ready",
        PREFLIGHT_WARNING: "Warning",
        PREFLIGHT_BLOCKED: "Blocked",
        PREFLIGHT_INFO: "Info",
    }.get(status, status)
