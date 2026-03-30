"""Preflight status widget for the experiment workspace."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QHBoxLayout, QVBoxLayout, QWidget

from continuum_robot.gui.experiment_preflight import (
    PREFLIGHT_BLOCKED,
    PREFLIGHT_INFO,
    PREFLIGHT_OK,
    PREFLIGHT_WARNING,
    RUN_BLOCKED,
    RUN_OK,
    RUN_WARNING,
)


class ExperimentPreflightWidget(QWidget):
    """Render experiment preflight checks and overall run readiness."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.status_chip = QLabel("Preflight")
        self.status_chip.setAlignment(Qt.AlignCenter)
        self.status_chip.setMinimumWidth(108)
        self.summary_label = QLabel("Preflight not evaluated.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("color: #334155;")

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(10)
        summary_row.addWidget(self.status_chip, 0)
        summary_row.addWidget(self.summary_label, 1)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setSpacing(8)
        self.list.setStyleSheet("border: none; background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addLayout(summary_row)
        layout.addWidget(self.list)

    def set_report(self, report) -> None:
        color = {
            RUN_OK: ("#dcfce7", "#166534", "Ready"),
            RUN_WARNING: ("#fef3c7", "#92400e", "Warning"),
            RUN_BLOCKED: ("#fee2e2", "#991b1b", "Blocked"),
        }.get(report.overall_status, ("#e2e8f0", "#334155", "Info"))
        self.status_chip.setText(color[2])
        self.status_chip.setStyleSheet(
            f"padding: 6px 10px; border-radius: 999px; background: {color[0]}; color: {color[1]}; font-weight: 700;"
        )
        self.summary_label.setText(report.summary)

        self.list.clear()
        for check in report.checks:
            widget = _PreflightRowWidget(check.label, _severity_label(check.status), check.message, check.status)
            item = QListWidgetItem()
            item.setSizeHint(widget.sizeHint())
            item.setData(Qt.UserRole, check.key)
            self.list.addItem(item)
            self.list.setItemWidget(item, widget)


class _PreflightRowWidget(QWidget):
    def __init__(self, label: str, severity: str, message: str, status: str, parent=None) -> None:
        super().__init__(parent)
        severity_label = QLabel(severity)
        bg, fg = _severity_colors(status)
        severity_label.setStyleSheet(
            f"padding: 4px 8px; border-radius: 999px; background: {bg}; color: {fg}; font-weight: 700;"
        )
        title_label = QLabel(label)
        title_label.setStyleSheet("color: #0f172a; font-weight: 700;")
        message_label = QLabel(message)
        message_label.setWordWrap(True)
        message_label.setStyleSheet("color: #475569;")

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
        self.setStyleSheet("background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;")


def _severity_colors(status: str) -> tuple[str, str]:
    return {
        PREFLIGHT_OK: ("#dcfce7", "#166534"),
        PREFLIGHT_WARNING: ("#fef3c7", "#92400e"),
        PREFLIGHT_BLOCKED: ("#fee2e2", "#991b1b"),
        PREFLIGHT_INFO: ("#dbeafe", "#1d4ed8"),
    }.get(status, ("#e2e8f0", "#334155"))


def _severity_label(status: str) -> str:
    return {
        PREFLIGHT_OK: "Ready",
        PREFLIGHT_WARNING: "Warning",
        PREFLIGHT_BLOCKED: "Blocked",
        PREFLIGHT_INFO: "Info",
    }.get(status, status)
