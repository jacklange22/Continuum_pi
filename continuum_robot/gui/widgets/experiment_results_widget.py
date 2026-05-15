"""Integrated experiment results viewer using QtCharts."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QLineSeries,
    QScatterSeries,
    QValueAxis,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.theme import COLORS, qcolor
from continuum_robot.gui.experiment_visualization import ChartModel, VisualizationModel
from continuum_robot.gui.view_utils import set_text_document


class ExperimentResultsWidget(QWidget):
    """Render summary text and result charts for experiment runs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model_signature: str | None = None
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet(
            f"background: {COLORS.input_bg}; border: 1px solid {COLORS.input_border}; "
            f"border-radius: 12px; padding: 8px; color: {COLORS.text_primary};"
        )
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet(
            f"""
            QTabBar::tab {{
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                background: {COLORS.tab_bg};
                color: {COLORS.text_secondary};
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {COLORS.tab_selected_bg};
                color: {COLORS.tab_selected_fg};
            }}
            """
        )
        self.tabs.addTab(self.summary_text, "Summary")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

    def set_model(self, model: VisualizationModel) -> None:
        signature = repr(asdict(model))
        if self._model_signature == signature:
            return
        self._model_signature = signature
        current_tab_name = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else "Summary"
        while self.tabs.count() > 1:
            widget = self.tabs.widget(1)
            self.tabs.removeTab(1)
            widget.deleteLater()
        set_text_document(self.summary_text, "\n".join(model.summary_lines or ["No results loaded."]))
        for chart in model.charts:
            self.tabs.addTab(self._chart_widget(chart), chart.title)
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) == current_tab_name:
                self.tabs.setCurrentIndex(index)
                break

    def save_current_view(self, path: str) -> bool:
        widget = self.tabs.currentWidget()
        if widget is None:
            return False
        return widget.grab().save(str(path))

    def _chart_widget(self, model: ChartModel) -> QWidget:
        if model.kind == "bar":
            chart = QChart()
            series = QBarSeries()
            bar_set = QBarSet(model.title)
            bar_set.setColor(QColor(model.color_hex))
            for value in model.values:
                bar_set.append(float(value))
            series.append(bar_set)
            chart.addSeries(series)
            axis_x = QBarCategoryAxis()
            axis_x.append([str(category) for category in model.categories])
            axis_y = QValueAxis()
            axis_y.setTitleText(model.y_title)
            chart.addAxis(axis_x, Qt.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            _style_chart(chart, model.title)
            chart.legend().setVisible(False)
            return _chart_panel(chart, model.caption)
        if model.kind == "line":
            chart = QChart()
            series_models = model.series_xy or []
            if series_models:
                for series_model in series_models:
                    series = QLineSeries()
                    series.setName(series_model.name)
                    series.setColor(QColor(series_model.color_hex))
                    for x_value, y_value in series_model.points_xy:
                        series.append(float(x_value), float(y_value))
                    chart.addSeries(series)
            else:
                series = QLineSeries()
                series.setColor(QColor(model.color_hex))
                for x_value, y_value in model.points_xy:
                    series.append(float(x_value), float(y_value))
                chart.addSeries(series)
            axis_x = QValueAxis()
            axis_y = QValueAxis()
            axis_x.setTitleText(model.x_title)
            axis_y.setTitleText(model.y_title)
            chart.addAxis(axis_x, Qt.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignLeft)
            for series in chart.series():
                series.attachAxis(axis_x)
                series.attachAxis(axis_y)
            _style_chart(chart, model.title)
            chart.legend().setVisible(bool(series_models))
            return _chart_panel(chart, model.caption)
        if model.kind == "scatter":
            chart = QChart()
            series_models = model.series_xy or []
            if not series_models:
                series_models = []
                if model.points_xy:
                    series_models.append(
                        type("_Series", (), {"name": model.title, "points_xy": model.points_xy, "color_hex": model.color_hex})()
                    )
            for series_model in series_models:
                series = QScatterSeries()
                series.setName(series_model.name)
                series.setColor(QColor(series_model.color_hex))
                series.setMarkerSize(8.0)
                for x_value, y_value in series_model.points_xy:
                    series.append(float(x_value), float(y_value))
                chart.addSeries(series)
            axis_x = QValueAxis()
            axis_y = QValueAxis()
            axis_x.setTitleText(model.x_title)
            axis_y.setTitleText(model.y_title)
            chart.addAxis(axis_x, Qt.AlignBottom)
            chart.addAxis(axis_y, Qt.AlignLeft)
            for series in chart.series():
                series.attachAxis(axis_x)
                series.attachAxis(axis_y)
            _style_chart(chart, model.title)
            chart.legend().setVisible(bool(series_models))
            return _chart_panel(chart, model.caption)
        if model.kind == "table":
            return _build_table_widget(model)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"Unsupported chart type: {model.kind}"))
        return widget


def _build_table_widget(model: ChartModel) -> QWidget:
    """Render a ChartModel(kind='table') into a styled, scrollable table panel."""
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    title = QLabel(model.title)
    title.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 600; font-size: 13px;")
    layout.addWidget(title)
    table = QTableWidget()
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setStyleSheet(
        f"QTableWidget {{ background: {COLORS.surface_alt_bg}; color: {COLORS.text_primary}; "
        f"alternate-background-color: {COLORS.surface_bg}; border: 1px solid {COLORS.surface_border}; "
        f"border-radius: 8px; gridline-color: {COLORS.surface_border}; }}"
        f"QHeaderView::section {{ background: {COLORS.surface_bg}; color: {COLORS.text_muted}; "
        f"padding: 6px 8px; border: none; font-weight: 700; }}"
    )
    headers = list(model.table_headers or [])
    rows = list(model.table_rows or [])
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for col_index in range(len(headers)):
            value = row[col_index] if col_index < len(row) else ""
            item = QTableWidgetItem(str(value))
            table.setItem(row_index, col_index, item)
    table.verticalHeader().setVisible(False)
    horizontal = table.horizontalHeader()
    if horizontal is not None:
        horizontal.setStretchLastSection(True)
        horizontal.setSectionResizeMode(QHeaderView.ResizeToContents)
    table.setMinimumHeight(180)
    layout.addWidget(table)
    if model.caption:
        caption_label = QLabel(model.caption)
        caption_label.setWordWrap(True)
        caption_label.setStyleSheet(f"color: {COLORS.text_muted};")
        layout.addWidget(caption_label)
    return container


def _chart_view(chart: QChart) -> QChartView:
    view = QChartView(chart)
    view.setRenderHint(QPainter.Antialiasing, True)
    view.setStyleSheet("background: transparent;")
    return view


def _chart_panel(chart: QChart, caption: str) -> QWidget:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(_chart_view(chart))
    if caption:
        caption_label = QLabel(caption)
        caption_label.setWordWrap(True)
        caption_label.setStyleSheet(f"color: {COLORS.text_muted};")
        layout.addWidget(caption_label)
    return widget


def _style_chart(chart: QChart, title: str) -> None:
    chart.setTitle(title)
    chart.setBackgroundBrush(qcolor(COLORS.surface_alt_bg))
    chart.setPlotAreaBackgroundVisible(True)
    chart.setPlotAreaBackgroundBrush(qcolor(COLORS.surface_bg))
    chart.setMargins(chart.margins())
    chart.setTitleBrush(qcolor(COLORS.text_primary))
    for axis in chart.axes():
        axis.setLabelsColor(qcolor(COLORS.text_secondary))
        axis.setTitleBrush(qcolor(COLORS.text_secondary))
        axis.setGridLineColor(qcolor(COLORS.chart_grid))
        axis.setLinePenColor(qcolor(COLORS.surface_border))
