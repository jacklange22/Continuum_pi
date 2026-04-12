"""Integrated experiment results viewer using QtCharts."""

from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QTabWidget, QTextEdit, QVBoxLayout, QWidget

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
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            _style_chart(chart, model.title)
            chart.legend().setVisible(False)
            return _chart_panel(chart, model.caption)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"Unsupported chart type: {model.kind}"))
        return widget


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
