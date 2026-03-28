"""Integrated experiment results viewer using QtCharts."""

from __future__ import annotations

from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QLineSeries, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QLabel, QTabWidget, QTextEdit, QVBoxLayout, QWidget

from continuum_robot.gui.experiment_visualization import ChartModel, VisualizationModel


class ExperimentResultsWidget(QWidget):
    """Render summary text and result charts for experiment runs."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.summary_text, "Summary")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

    def set_model(self, model: VisualizationModel) -> None:
        while self.tabs.count() > 1:
            widget = self.tabs.widget(1)
            self.tabs.removeTab(1)
            widget.deleteLater()
        self.summary_text.setPlainText("\n".join(model.summary_lines or ["No results loaded."]))
        for chart in model.charts:
            self.tabs.addTab(self._chart_widget(chart), chart.title)

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
            chart.setTitle(model.title)
            chart.legend().setVisible(False)
            return _chart_view(chart)
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
            chart.setTitle(model.title)
            chart.legend().setVisible(False)
            return _chart_view(chart)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"Unsupported chart type: {model.kind}"))
        return widget


def _chart_view(chart: QChart) -> QChartView:
    view = QChartView(chart)
    view.setRenderHint(QPainter.Antialiasing, True)
    return view
