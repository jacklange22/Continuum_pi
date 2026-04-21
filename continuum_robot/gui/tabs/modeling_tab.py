"""Modeling analysis and comparison tab."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.modeling_controller import ModelingController, ModelingViewState
from continuum_robot.gui.theme import COLORS, grouped_workspace_stylesheet
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


class ModelingTab(QWidget):
    """Dataset/artifact browser plus evaluation workspace."""

    def __init__(self, controller: ModelingController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("modelingWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="modelingWorkspace",
                input_selectors=["QComboBox", "QListWidget"],
            )
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        outer_layout.addWidget(self.scroll_area)

        content = QWidget()
        self.scroll_area.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        title = QLabel("Modeling Analysis")
        title.setProperty("role", "title")
        hint = QLabel(
            "Browse canonical modeling datasets and trained ANN artifacts, then compare Mike, Camarillo, "
            "and ANN predictions on the same selected dataset without mixing this workflow into training or control."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(hint)

        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(12)
        root.addLayout(columns, 1)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)
        columns.addLayout(left, 3)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(12)
        columns.addLayout(right, 4)

        dataset_card = _Card("Modeling Datasets", "Canonical `collect_pose_command_dataset` runs available for evaluation.")
        dataset_buttons = QHBoxLayout()
        dataset_buttons.setContentsMargins(0, 0, 0, 0)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setProperty("variant", "ghost")
        self.refresh_button.clicked.connect(self._refresh_catalogs)
        self.open_dataset_button = QPushButton("Open Dataset Folder")
        self.open_dataset_button.setProperty("variant", "ghost")
        self.open_dataset_button.clicked.connect(self._open_selected_dataset)
        dataset_buttons.addWidget(self.refresh_button)
        dataset_buttons.addWidget(self.open_dataset_button)
        dataset_buttons.addStretch(1)
        dataset_card.body_layout.addLayout(dataset_buttons)
        self.dataset_list = QListWidget()
        self.dataset_list.setMinimumHeight(220)
        self.dataset_list.currentItemChanged.connect(self._on_dataset_selected)
        dataset_card.body_layout.addWidget(self.dataset_list)
        self.dataset_pairs = _PairsWidget()
        dataset_card.body_layout.addWidget(self.dataset_pairs)
        left.addWidget(dataset_card)

        artifact_card = _Card("ANN Artifacts", "Previously trained ANN bundles from the ANN Training popout.")
        artifact_buttons = QHBoxLayout()
        artifact_buttons.setContentsMargins(0, 0, 0, 0)
        self.open_artifact_button = QPushButton("Open Artifact Folder")
        self.open_artifact_button.setProperty("variant", "ghost")
        self.open_artifact_button.clicked.connect(self._open_selected_artifact)
        artifact_buttons.addWidget(self.open_artifact_button)
        artifact_buttons.addStretch(1)
        artifact_card.body_layout.addLayout(artifact_buttons)
        self.artifact_list = QListWidget()
        self.artifact_list.setMinimumHeight(200)
        self.artifact_list.currentItemChanged.connect(self._on_artifact_selected)
        artifact_card.body_layout.addWidget(self.artifact_list)
        self.artifact_pairs = _PairsWidget()
        artifact_card.body_layout.addWidget(self.artifact_pairs)
        left.addWidget(artifact_card, 1)

        controls_card = _Card("Evaluation Controls", "Choose which models to compare and which dataset slice to use.")
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)
        controls_row.setSpacing(10)
        self.mike_check = QCheckBox("Mike")
        self.mike_check.setChecked(True)
        self.mike_check.toggled.connect(lambda value: self.controller.set_include_mike(bool(value)))
        self.camarillo_check = QCheckBox("Camarillo")
        self.camarillo_check.setChecked(True)
        self.camarillo_check.toggled.connect(lambda value: self.controller.set_include_camarillo(bool(value)))
        self.ann_check = QCheckBox("ANN")
        self.ann_check.setChecked(True)
        self.ann_check.toggled.connect(lambda value: self.controller.set_include_ann(bool(value)))
        controls_row.addWidget(self.mike_check)
        controls_row.addWidget(self.camarillo_check)
        controls_row.addWidget(self.ann_check)
        controls_row.addStretch(1)
        controls_card.body_layout.addLayout(controls_row)

        scope_row = QHBoxLayout()
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_label = QLabel("Evaluation Scope")
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Artifact Held-Out Split", "artifact_test_split")
        self.scope_combo.addItem("Full Accepted Dataset", "full_dataset")
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        scope_row.addWidget(scope_label)
        scope_row.addWidget(self.scope_combo, 1)
        controls_card.body_layout.addLayout(scope_row)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        self.evaluate_button = QPushButton("Run Comparison")
        self.evaluate_button.setProperty("role", "primary")
        self.evaluate_button.clicked.connect(self.controller.evaluate)
        self.open_results_button = QPushButton("Open Results Folder")
        self.open_results_button.setProperty("variant", "ghost")
        self.open_results_button.clicked.connect(self._open_results_folder)
        action_row.addWidget(self.evaluate_button)
        action_row.addWidget(self.open_results_button)
        action_row.addStretch(1)
        controls_card.body_layout.addLayout(action_row)
        left.addWidget(controls_card)

        results_card = _Card("Comparison Summary", "Compact saved-output summary and side-by-side error views for the selected models.")
        self.status_label = QLabel("Select a dataset, choose models, then run a comparison.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS.text_primary};")
        results_card.body_layout.addWidget(self.status_label)
        self.evaluation_pairs = _PairsWidget()
        results_card.body_layout.addWidget(self.evaluation_pairs)
        self.results_widget = ExperimentResultsWidget()
        results_card.body_layout.addWidget(self.results_widget)
        right.addWidget(results_card, 1)

    def update(self, state: ModelingViewState) -> None:
        self._sync_dataset_list(state)
        self._sync_artifact_list(state)
        self.dataset_pairs.set_pairs(state.dataset_summary_pairs)
        self.artifact_pairs.set_pairs(state.artifact_summary_pairs)
        self.evaluation_pairs.set_pairs(state.evaluation_summary_pairs)
        self.results_widget.set_model(state.visualization_model)
        self.status_label.setText(state.status_message)
        self.evaluate_button.setEnabled(state.can_evaluate)
        self.open_dataset_button.setEnabled(bool(state.selected_dataset_path))
        self.open_artifact_button.setEnabled(bool(state.selected_artifact_path))
        self.open_results_button.setEnabled(bool(state.last_output_path))
        self._set_checkbox(self.mike_check, state.include_mike)
        self._set_checkbox(self.camarillo_check, state.include_camarillo)
        self._set_checkbox(self.ann_check, state.include_ann)
        self._set_combo(self.scope_combo, state.evaluation_scope)

    def _refresh_catalogs(self) -> None:
        self.controller.set_dataset_output_root(self.controller.dataset_output_root)
        self.controller.set_artifact_root(self.controller.artifact_root)
        self.update(self.controller.refresh())

    def _sync_dataset_list(self, state: ModelingViewState) -> None:
        current_paths = [self.dataset_list.item(index).data(Qt.UserRole) for index in range(self.dataset_list.count())]
        target_paths = [str(dataset.path) for dataset in state.datasets]
        if current_paths != target_paths:
            with QSignalBlocker(self.dataset_list):
                self.dataset_list.clear()
                for dataset in state.datasets:
                    item = QListWidgetItem(
                        f"{dataset.run_name} | {dataset.dataset_mode} | {dataset.accepted_count} accepted"
                    )
                    item.setData(Qt.UserRole, str(dataset.path))
                    self.dataset_list.addItem(item)
        with QSignalBlocker(self.dataset_list):
            for index in range(self.dataset_list.count()):
                if self.dataset_list.item(index).data(Qt.UserRole) == state.selected_dataset_path:
                    self.dataset_list.setCurrentRow(index)
                    break

    def _sync_artifact_list(self, state: ModelingViewState) -> None:
        current_paths = [self.artifact_list.item(index).data(Qt.UserRole) for index in range(self.artifact_list.count())]
        target_paths = [str(artifact.path) for artifact in state.artifacts]
        if current_paths != target_paths:
            with QSignalBlocker(self.artifact_list):
                self.artifact_list.clear()
                for artifact in state.artifacts:
                    item = QListWidgetItem(
                        f"{artifact.artifact_name} | {artifact.backend_name} | {artifact.status}"
                    )
                    item.setData(Qt.UserRole, str(artifact.path))
                    self.artifact_list.addItem(item)
        with QSignalBlocker(self.artifact_list):
            for index in range(self.artifact_list.count()):
                if self.artifact_list.item(index).data(Qt.UserRole) == state.selected_artifact_path:
                    self.artifact_list.setCurrentRow(index)
                    break

    def _on_dataset_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.controller.select_dataset(str(current.data(Qt.UserRole)))

    def _on_artifact_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.controller.select_artifact(str(current.data(Qt.UserRole)))

    def _on_scope_changed(self, _index: int) -> None:
        value = self.scope_combo.currentData()
        if value:
            self.controller.set_evaluation_scope(str(value))

    def _open_selected_dataset(self) -> None:
        state = self.controller.refresh()
        if state.selected_dataset_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.selected_dataset_path))))

    def _open_selected_artifact(self) -> None:
        state = self.controller.refresh()
        if state.selected_artifact_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.selected_artifact_path))))

    def _open_results_folder(self) -> None:
        state = self.controller.refresh()
        if state.last_output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.last_output_path))))

    @staticmethod
    def _set_checkbox(widget: QCheckBox, value: bool) -> None:
        if widget.isChecked() == bool(value):
            return
        with QSignalBlocker(widget):
            widget.setChecked(bool(value))

    @staticmethod
    def _set_combo(widget: QComboBox, value: str) -> None:
        for index in range(widget.count()):
            if widget.itemData(index) == value:
                with QSignalBlocker(widget):
                    widget.setCurrentIndex(index)
                return


class _Card(QFrame):
    def __init__(self, title: str, subtitle: str, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setProperty("role", "section-title")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet(f"color: {COLORS.text_secondary};")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        self.body_layout = layout


class _PairsWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._signature: tuple[tuple[str, str], ...] | None = None
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(8)

    def set_pairs(self, pairs: list[tuple[str, str]]) -> None:
        signature = tuple((str(label), str(value)) for label, value in pairs)
        if signature == self._signature:
            return
        self._signature = signature
        while self.layout_.count():
            item = self.layout_.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for label, value in signature:
            row = QFrame()
            row.setStyleSheet(
                f"background: {COLORS.surface_bg}; border: 1px solid {COLORS.surface_border}; border-radius: 10px;"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 8, 10, 8)
            row_layout.setSpacing(12)
            key = QLabel(label)
            key.setStyleSheet(f"color: {COLORS.text_muted};")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 600;")
            row_layout.addWidget(key, 1)
            row_layout.addWidget(val, 2)
            self.layout_.addWidget(row)
        self.layout_.addStretch(1)
