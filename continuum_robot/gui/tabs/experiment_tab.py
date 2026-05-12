"""Experiment shell/router tab."""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QCoreApplication, QSignalBlocker, QThread, Qt
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
from continuum_robot.gui.theme import COLORS, chip_stylesheet, experiment_shell_stylesheet, semantic_chip_colors
from continuum_robot.gui.widgets.experiment_pages import (
    EmptyExperimentWorkspace,
    build_experiment_page,
)


LOG = logging.getLogger(__name__)


class ExperimentTab(QWidget):
    """Simple experiment selector shell that routes to custom experiment pages."""

    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self._pages: dict[str, QWidget] = {}
        self._page_state_fingerprints: dict[str, tuple[object, ...]] = {}
        self._page_state_applied_at_s: dict[str, float] = {}
        self._last_logged_selection = ""
        self.setObjectName("experimentWorkspace")
        self.setStyleSheet(experiment_shell_stylesheet(object_name="experimentWorkspace"))

        self.page_title = QLabel("Experiments")
        self.page_title.setProperty("role", "page-title")
        self.page_subtitle = QLabel(
            "Structured validation and dataset runs. Routine setup, calibration, and tuning stay in their dedicated tabs."
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
        self.selected_badges_label.hide()

        self.experiment_combo = QComboBox()
        self.experiment_combo.setObjectName("experimentSelectorCombo")
        self.experiment_combo.setMinimumHeight(36)
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_selected)
        self.filter_summary_label = QLabel("")
        self.filter_summary_label.setProperty("role", "hint")
        self.filter_summary_label.setWordWrap(True)
        self.load_defaults_button = QPushButton("Load Defaults")
        self.load_defaults_button.setProperty("variant", "ghost")
        self.load_defaults_button.clicked.connect(self.controller.load_defaults)
        header_card = _ShellCard()
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title_column = QWidget()
        title_layout = QVBoxLayout(title_column)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_layout.addWidget(self.page_title)
        title_layout.addWidget(self.page_subtitle)
        title_row.addWidget(title_column, 1)
        title_row.addWidget(self.selected_status_chip, 0, Qt.AlignRight | Qt.AlignTop)
        header_card.body_layout.addLayout(title_row)

        selector_row = QHBoxLayout()
        selector_row.setContentsMargins(0, 0, 0, 0)
        selector_row.setSpacing(10)
        selector_row.addWidget(self.experiment_combo, 1)
        selector_row.addWidget(self.load_defaults_button)
        header_card.body_layout.addLayout(selector_row)
        header_card.body_layout.addWidget(self.filter_summary_label)
        selected_summary = QWidget()
        selected_summary_layout = QVBoxLayout(selected_summary)
        selected_summary_layout.setContentsMargins(0, 0, 0, 0)
        selected_summary_layout.setSpacing(4)
        selected_summary_layout.addWidget(self.selected_experiment_title)
        selected_summary_layout.addWidget(self.selected_experiment_description)
        selected_summary_layout.addWidget(self.selected_badges_label)
        header_card.body_layout.addWidget(selected_summary)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("experimentPageStack")
        self.empty_page = EmptyExperimentWorkspace()
        self.page_stack.addWidget(self.empty_page)
        self.page_stack.setCurrentWidget(self.empty_page)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(header_card)
        layout.addWidget(self.page_stack, 1)

    def update(self, state: ExperimentViewState) -> bool:
        started = time.monotonic()
        self._update_selector(state)
        self.filter_summary_label.setText(state.experiment_filter_summary)
        if not state.selected_experiment:
            self.selected_experiment_title.setText("Select An Experiment")
            self.selected_experiment_description.setText(
                "Pick an experiment from the dropdown to open its custom page."
            )
            self.selected_badges_label.setText("")
            self.selected_badges_label.hide()
            self.selected_status_chip.setText("No Selection")
            bg, fg = semantic_chip_colors("neutral")
            self.selected_status_chip.setStyleSheet(
                chip_stylesheet(background=bg, foreground=fg)
            )
            self.page_stack.setCurrentWidget(self.empty_page)
            self.load_defaults_button.setEnabled(False)
            return True

        page = self._page_for(state.selected_experiment)
        fingerprint = page.state_fingerprint(state)
        previous_fingerprint = self._page_state_fingerprints.get(state.selected_experiment)
        page_changed = self.page_stack.currentWidget() is not page
        min_interval_s = 0.15 if bool(state.run_active) else 0.0
        last_applied = float(self._page_state_applied_at_s.get(state.selected_experiment, 0.0))
        if (
            previous_fingerprint == fingerprint
            and not page_changed
        ):
            return False
        if (
            min_interval_s > 0.0
            and not page_changed
            and previous_fingerprint is not None
            and (time.monotonic() - last_applied) < min_interval_s
        ):
            return False
        if (
            getattr(page, "refresh_policy", "live") == "manual"
            and previous_fingerprint == fingerprint
        ):
            if page_changed:
                self.page_stack.setCurrentWidget(page)
            if hasattr(page, "_log_interaction_snapshot"):
                page._log_interaction_snapshot("tab_manual_update_skipped", selector=self.experiment_combo)
            return False

        self.load_defaults_button.setEnabled(True)
        self.selected_experiment_title.setText(state.experiment_title)
        self.selected_experiment_description.setText(state.experiment_description)
        self.selected_badges_label.setText("  •  ".join(state.experiment_badges))
        self.selected_badges_label.setVisible(bool(state.experiment_badges))
        if state.selected_experiment != self._last_logged_selection:
            LOG.debug("Activating experiment workspace page: %s", state.selected_experiment)
            self._last_logged_selection = state.selected_experiment
            self._log_event("page_activation", experiment=state.selected_experiment)
        self._update_status_chip(state)
        set_state_started = time.monotonic()
        page.set_state(state)
        self._page_state_fingerprints[state.selected_experiment] = fingerprint
        self._page_state_applied_at_s[state.selected_experiment] = time.monotonic()
        self._log_event(
            "set_state_applied",
            experiment=state.selected_experiment,
            elapsed_ms=(time.monotonic() - set_state_started) * 1000.0,
        )
        self.page_stack.setCurrentWidget(page)
        if hasattr(page, "_log_interaction_snapshot"):
            page._log_interaction_snapshot("tab_update_complete", selector=self.experiment_combo)
        self._log_event(
            "update_complete",
            experiment=state.selected_experiment,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )
        return True

    def shutdown(self) -> None:
        for page in list(self._pages.values()):
            shutdown = getattr(page, "shutdown", None)
            if callable(shutdown):
                shutdown()
            page.close()

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
        selected_before = self.controller.state.selected_experiment
        self._log_event(
            "page_selection",
            row=row,
            experiment=raw_name or "",
            selected_before=selected_before or "",
        )
        if not raw_name:
            if self.controller.state.selected_experiment:
                self.controller.clear_selection()
                self._log_event("page_selection_cleared", selected_before=selected_before or "")
            return
        if raw_name != self.controller.state.selected_experiment:
            self.controller.select_experiment(str(raw_name))
            self._log_event(
                "page_selection_requested",
                experiment=raw_name,
                selected_before=selected_before or "",
                selected_after=self.controller.state.selected_experiment or "",
            )

    def _page_for(self, experiment_name: str) -> QWidget:
        if experiment_name not in self._pages:
            started = time.monotonic()
            page = build_experiment_page(self.controller, experiment_name)
            self._pages[experiment_name] = page
            self.page_stack.addWidget(page)
            duration_ms = (time.monotonic() - started) * 1000.0
            LOG.debug("Built experiment workspace page %s in %.1f ms", experiment_name, duration_ms)
        return self._pages[experiment_name]

    def needs_state_update(self, state: ExperimentViewState) -> bool:
        if not state.selected_experiment:
            return self.page_stack.currentWidget() is not self.empty_page
        page = self._page_for(state.selected_experiment)
        fingerprint = page.state_fingerprint(state)
        return self._page_state_fingerprints.get(state.selected_experiment) != fingerprint

    @staticmethod
    def _is_gui_thread() -> bool:
        app = QCoreApplication.instance()
        if app is None:
            return False
        return bool(QThread.currentThread() == app.thread())

    def _log_event(self, stage: str, **fields) -> None:
        details = dict(fields)
        if "elapsed_ms" in details:
            details["elapsed_ms"] = f"{float(details['elapsed_ms']):.1f}"
        details["gui_thread"] = self._is_gui_thread()
        rendered = " ".join(f"{key}={value}" for key, value in details.items())
        log_fn = LOG.debug if stage in {"set_state_applied", "update_complete"} else LOG.info
        log_fn("ExperimentTab %s | %s", stage, rendered)

    def _update_status_chip(self, state: ExperimentViewState) -> None:
        status = state.preflight_report.overall_status
        if status == "blocked":
            bg, fg = semantic_chip_colors("blocked")
            text = "Blocked"
        elif status == "ok_with_warning":
            bg, fg = semantic_chip_colors("warning")
            text = "Ready With Warning"
        else:
            bg, fg = semantic_chip_colors("ready")
            text = "Ready"
        self.selected_status_chip.setText(text)
        self.selected_status_chip.setStyleSheet(chip_stylesheet(background=bg, foreground=fg))


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
