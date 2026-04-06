"""Experiment shell/router tab."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.experiment_controller import ExperimentViewState
from continuum_robot.gui.widgets.experiment_pages import (
    EmptyExperimentWorkspace,
    build_experiment_page,
)


class ExperimentTab(QWidget):
    """Simple experiment selector shell that routes to custom experiment pages."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._pages: dict[str, QWidget] = {}
        self.setObjectName("experimentWorkspace")
        self.setStyleSheet(
            """
            QWidget#experimentWorkspace {
                background: #eef3f8;
                color: #0f172a;
            }
            QWidget#experimentWorkspace QLabel[role="page-title"] {
                font-size: 24px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#experimentWorkspace QLabel[role="section-title"] {
                font-size: 15px;
                font-weight: 700;
                color: #0f172a;
            }
            QWidget#experimentWorkspace QLabel[role="body"] {
                color: #475569;
            }
            QWidget#experimentWorkspace QLabel[role="muted"] {
                color: #556476;
            }
            QWidget#experimentWorkspace QLabel[role="chip"] {
                padding: 5px 10px;
                border-radius: 999px;
                background: #e2e8f0;
                color: #0f172a;
                font-weight: 700;
            }
            QWidget#experimentWorkspace QFrame[role="card"] {
                background: #ffffff;
                border: 1px solid #d9e3ec;
                border-radius: 16px;
            }
            QWidget#experimentWorkspace QComboBox {
                min-height: 48px;
                border: 1px solid #dbe4ee;
                border-radius: 14px;
                background: #fbfdff;
                padding: 8px 12px;
                font-size: 15px;
                font-weight: 600;
            }
            QWidget#experimentWorkspace QPushButton {
                min-height: 38px;
                padding: 0 14px;
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #f8fafc;
                color: #0f172a;
                font-weight: 600;
            }
            QWidget#experimentWorkspace QPushButton[variant="ghost"] {
                background: transparent;
                color: #334155;
            }
            QWidget#experimentWorkspace QLineEdit,
            QWidget#experimentWorkspace QPlainTextEdit,
            QWidget#experimentWorkspace QTextEdit,
            QWidget#experimentWorkspace QListWidget,
            QWidget#experimentWorkspace QSpinBox,
            QWidget#experimentWorkspace QDoubleSpinBox,
            QWidget#experimentWorkspace QCheckBox,
            QWidget#experimentWorkspace QProgressBar {
                border: 1px solid #dbe4ee;
                border-radius: 10px;
                background: #fbfdff;
            }
            """
        )

        self.page_title = QLabel("Experiments")
        self.page_title.setProperty("role", "page-title")
        self.page_subtitle = QLabel(
            "Choose a structured validation or data-generation run. Routine setup, calibration, and tuning workflows stay in their dedicated tabs."
        )
        self.page_subtitle.setProperty("role", "body")
        self.page_subtitle.setWordWrap(True)

        self.selected_status_chip = QLabel("No Selection")
        self.selected_status_chip.setProperty("role", "chip")
        self.selected_experiment_title = QLabel("Select An Experiment")
        self.selected_experiment_title.setProperty("role", "section-title")
        self.selected_experiment_description = QLabel(
            "Pick an experiment from the dropdown below to load its custom workspace."
        )
        self.selected_experiment_description.setProperty("role", "body")
        self.selected_experiment_description.setWordWrap(True)
        self.selected_badges_label = QLabel("")
        self.selected_badges_label.setProperty("role", "muted")
        self.selected_badges_label.setWordWrap(True)

        header_card = _ShellCard()
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(18)
        header_left = QWidget()
        header_left_layout = QVBoxLayout(header_left)
        header_left_layout.setContentsMargins(0, 0, 0, 0)
        header_left_layout.setSpacing(6)
        header_left_layout.addWidget(self.page_title)
        header_left_layout.addWidget(self.page_subtitle)
        header_right = QWidget()
        header_right_layout = QVBoxLayout(header_right)
        header_right_layout.setContentsMargins(0, 0, 0, 0)
        header_right_layout.setSpacing(8)
        header_right_layout.addWidget(self.selected_status_chip, 0, Qt.AlignRight)
        header_right_layout.addWidget(self.selected_experiment_title)
        header_right_layout.addWidget(self.selected_experiment_description)
        header_right_layout.addWidget(self.selected_badges_label)
        header_row.addWidget(header_left, 3)
        header_row.addWidget(header_right, 4)
        header_card.body_layout.addLayout(header_row)

        self.experiment_combo = QComboBox()
        self.experiment_combo.setMinimumHeight(48)
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_selected)
        self.load_defaults_button = QPushButton("Load Defaults")
        self.load_defaults_button.setProperty("variant", "ghost")
        self.load_defaults_button.clicked.connect(self.controller.load_defaults)
        selector_card = _ShellCard(
            "Experiment Selection",
            "Select the validation workflow you want to run. The matching custom page will load below.",
        )
        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(10)
        selector_row.addWidget(self.experiment_combo, 1)
        selector_row.addWidget(self.load_defaults_button)
        selector_card.body_layout.addLayout(selector_row)

        self.page_stack = QStackedWidget()
        self.empty_page = EmptyExperimentWorkspace()
        self.page_stack.addWidget(self.empty_page)
        self.page_stack.setCurrentWidget(self.empty_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(16)
        layout.addWidget(header_card)
        layout.addWidget(selector_card)
        layout.addWidget(self.page_stack, 1)

    def update(self, state: ExperimentViewState) -> None:
        self._update_selector(state)
        if not state.selected_experiment:
            self.selected_experiment_title.setText("Select An Experiment")
            self.selected_experiment_description.setText(
                "Pick an experiment from the dropdown to open its custom page."
            )
            self.selected_badges_label.setText("")
            self.selected_status_chip.setText("No Selection")
            self.selected_status_chip.setStyleSheet(
                "padding: 5px 12px; border-radius: 999px; background: #e2e8f0; color: #334155; font-weight: 700;"
            )
            self.page_stack.setCurrentWidget(self.empty_page)
            self.load_defaults_button.setEnabled(False)
            return

        self.load_defaults_button.setEnabled(True)
        self.selected_experiment_title.setText(state.experiment_title)
        self.selected_experiment_description.setText(state.experiment_description)
        self.selected_badges_label.setText("  •  ".join(state.experiment_badges))
        self._update_status_chip(state)
        page = self._page_for(state.selected_experiment)
        page.set_state(state)
        self.page_stack.setCurrentWidget(page)

    def _update_selector(self, state: ExperimentViewState) -> None:
        target_keys = ["", *[option.name for option in state.experiment_options]]
        current_keys = [self.experiment_combo.itemData(index) for index in range(self.experiment_combo.count())]
        if current_keys != target_keys:
            with QSignalBlocker(self.experiment_combo):
                self.experiment_combo.clear()
                self.experiment_combo.addItem("Select an experiment...", "")
                for option in state.experiment_options:
                    self.experiment_combo.addItem(option.title, option.name)
        target_index = 0
        for index in range(self.experiment_combo.count()):
            if self.experiment_combo.itemData(index) == state.selected_experiment:
                target_index = index
                break
        with QSignalBlocker(self.experiment_combo):
            self.experiment_combo.setCurrentIndex(target_index)

    def _on_experiment_selected(self, row: int) -> None:
        if row < 0:
            return
        raw_name = self.experiment_combo.itemData(row)
        if not raw_name:
            if self.controller.state.selected_experiment:
                self.controller.clear_selection()
            return
        if raw_name != self.controller.state.selected_experiment:
            self.controller.select_experiment(str(raw_name))

    def _page_for(self, experiment_name: str) -> QWidget:
        if experiment_name not in self._pages:
            page = build_experiment_page(self.controller, experiment_name)
            self._pages[experiment_name] = page
            self.page_stack.addWidget(page)
        return self._pages[experiment_name]

    def _update_status_chip(self, state: ExperimentViewState) -> None:
        status = state.preflight_report.overall_status
        if status == "blocked":
            bg, fg, text = "#fee2e2", "#991b1b", "Blocked"
        elif status == "ok_with_warning":
            bg, fg, text = "#fef3c7", "#92400e", "Ready With Warning"
        else:
            bg, fg, text = "#dcfce7", "#166534", "Ready"
        self.selected_status_chip.setText(text)
        self.selected_status_chip.setStyleSheet(
            f"padding: 5px 12px; border-radius: 999px; background: {bg}; color: {fg}; font-weight: 700;"
        )


class _ShellCard(QFrame):
    def __init__(self, title: str | None = None, subtitle: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        if title:
            title_label = QLabel(title)
            title_label.setProperty("role", "section-title")
            layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setProperty("role", "body")
            subtitle_label.setWordWrap(True)
            layout.addWidget(subtitle_label)
        self.body_layout = layout
