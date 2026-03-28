"""Preflight status widget for the experiment workspace."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from continuum_robot.gui.experiment_preflight import PREFLIGHT_BLOCKED, PREFLIGHT_OK, PREFLIGHT_WARNING, RUN_BLOCKED, RUN_OK, RUN_WARNING


class ExperimentPreflightWidget(QWidget):
    """Render experiment preflight checks and overall run readiness."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.summary_label = QLabel("Preflight not evaluated.")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Check", "Status", "Detail"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.table)

    def set_report(self, report) -> None:
        status_color = {
            RUN_OK: "#0f766e",
            RUN_WARNING: "#d97706",
            RUN_BLOCKED: "#b91c1c",
        }.get(report.overall_status, "#374151")
        self.summary_label.setText(f"{report.overall_status}: {report.summary}")
        self.summary_label.setStyleSheet(
            f"padding: 8px; border-radius: 6px; background: {status_color}; color: white; font-weight: 600;"
        )
        self.table.setRowCount(len(report.checks))
        for row, check in enumerate(report.checks):
            self.table.setItem(row, 0, QTableWidgetItem(check.label))
            status_item = QTableWidgetItem(check.status)
            detail_item = QTableWidgetItem(check.message)
            for item in (status_item, detail_item):
                item.setForeground(QColor(_check_color(check.status)))
            self.table.setItem(row, 1, status_item)
            self.table.setItem(row, 2, detail_item)
        self.table.resizeColumnsToContents()


def _check_color(status: str) -> str:
    return {
        PREFLIGHT_OK: "#0f766e",
        PREFLIGHT_WARNING: "#d97706",
        PREFLIGHT_BLOCKED: "#b91c1c",
    }.get(status, "#374151")
