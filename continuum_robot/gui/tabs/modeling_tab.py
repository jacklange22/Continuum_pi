"""Modeling analysis and comparison tab."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QSpinBox,
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

        two_segment_card = _Card(
            "Two-Segment Modeling",
            "Offline analysis for `two_segment_collect_pose_command_dataset` runs. This does not enable live control.",
        )
        two_segment_buttons = QHBoxLayout()
        two_segment_buttons.setContentsMargins(0, 0, 0, 0)
        self.two_segment_refresh_button = QPushButton("Refresh Runs")
        self.two_segment_refresh_button.setProperty("variant", "ghost")
        self.two_segment_refresh_button.clicked.connect(self._refresh_catalogs)
        self.two_segment_open_output_button = QPushButton("Open Output")
        self.two_segment_open_output_button.setProperty("variant", "ghost")
        self.two_segment_open_output_button.clicked.connect(self._open_two_segment_output)
        self.two_segment_export_button = QPushButton("Export Bundle")
        self.two_segment_export_button.setProperty("variant", "ghost")
        self.two_segment_export_button.clicked.connect(self._export_two_segment_output)
        two_segment_buttons.addWidget(self.two_segment_refresh_button)
        two_segment_buttons.addWidget(self.two_segment_open_output_button)
        two_segment_buttons.addWidget(self.two_segment_export_button)
        two_segment_buttons.addStretch(1)
        two_segment_card.body_layout.addLayout(two_segment_buttons)
        self.two_segment_run_list = QListWidget()
        self.two_segment_run_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.two_segment_run_list.setMinimumHeight(140)
        self.two_segment_run_list.itemSelectionChanged.connect(self._on_two_segment_selection_changed)
        two_segment_card.body_layout.addWidget(self.two_segment_run_list)
        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        self.two_segment_linear_check = QCheckBox("Linear")
        self.two_segment_linear_check.setChecked(True)
        self.two_segment_linear_check.toggled.connect(lambda value: self.controller.set_two_segment_model_enabled("linear_baseline", bool(value)))
        self.two_segment_ann_check = QCheckBox("ANN")
        self.two_segment_ann_check.setChecked(True)
        self.two_segment_ann_check.toggled.connect(lambda value: self.controller.set_two_segment_model_enabled("ann", bool(value)))
        self.two_segment_camarillo_check = QCheckBox("Camarillo unavailable")
        self.two_segment_camarillo_check.toggled.connect(lambda value: self.controller.set_two_segment_model_enabled("camarillo", bool(value)))
        self.two_segment_mike_check = QCheckBox("Mike unavailable")
        self.two_segment_mike_check.toggled.connect(lambda value: self.controller.set_two_segment_model_enabled("mike_constant_curvature", bool(value)))
        model_row.addWidget(self.two_segment_linear_check)
        model_row.addWidget(self.two_segment_ann_check)
        model_row.addWidget(self.two_segment_camarillo_check)
        model_row.addWidget(self.two_segment_mike_check)
        model_row.addStretch(1)
        two_segment_card.body_layout.addLayout(model_row)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.addWidget(QLabel("Label Mode"))
        self.two_segment_label_mode_combo = QComboBox()
        for label, value in [
            ("Auto", "auto"),
            ("Distal XYZ", "distal_xyz"),
            ("Distal Pose 6", "distal_pose6"),
            ("Two-Coil XYZ", "two_coil_xyz"),
            ("Two-Coil Pose 12", "two_coil_pose12"),
        ]:
            self.two_segment_label_mode_combo.addItem(label, value)
        self.two_segment_label_mode_combo.currentIndexChanged.connect(self._on_two_segment_label_mode_changed)
        self.two_segment_orientation_check = QCheckBox("Orientation if available")
        self.two_segment_orientation_check.toggled.connect(
            lambda value: self.controller.set_two_segment_include_orientation_if_available(bool(value))
        )
        label_row.addWidget(self.two_segment_label_mode_combo, 1)
        label_row.addWidget(self.two_segment_orientation_check)
        two_segment_card.body_layout.addLayout(label_row)
        ann_row = QHBoxLayout()
        ann_row.setContentsMargins(0, 0, 0, 0)
        self.two_segment_ann_sweep_check = QCheckBox("ANN sweep")
        self.two_segment_ann_sweep_check.toggled.connect(lambda value: self.controller.set_two_segment_ann_sweep_enabled(bool(value)))
        ann_row.addWidget(self.two_segment_ann_sweep_check)
        ann_row.addWidget(QLabel("Hidden"))
        self.two_segment_hidden_combo = QComboBox()
        for label in ["128,128", "64,64", "32,32"]:
            self.two_segment_hidden_combo.addItem(label, label)
        self.two_segment_hidden_combo.currentIndexChanged.connect(self._on_two_segment_hidden_changed)
        ann_row.addWidget(self.two_segment_hidden_combo)
        ann_row.addWidget(QLabel("Epochs"))
        self.two_segment_epochs_spin = QSpinBox()
        self.two_segment_epochs_spin.setRange(1, 5000)
        self.two_segment_epochs_spin.setValue(200)
        self.two_segment_epochs_spin.valueChanged.connect(lambda value: self.controller.set_two_segment_ann_epochs(int(value)))
        ann_row.addWidget(self.two_segment_epochs_spin)
        ann_row.addWidget(QLabel("Test"))
        self.two_segment_test_fraction_spin = QDoubleSpinBox()
        self.two_segment_test_fraction_spin.setRange(0.05, 0.9)
        self.two_segment_test_fraction_spin.setSingleStep(0.05)
        self.two_segment_test_fraction_spin.setValue(0.25)
        self.two_segment_test_fraction_spin.valueChanged.connect(lambda value: self.controller.set_two_segment_test_fraction(float(value)))
        ann_row.addWidget(self.two_segment_test_fraction_spin)
        ann_row.addStretch(1)
        two_segment_card.body_layout.addLayout(ann_row)
        trust_row = QHBoxLayout()
        trust_row.setContentsMargins(0, 0, 0, 0)
        self.two_segment_strict_check = QCheckBox("Strict")
        self.two_segment_strict_check.setChecked(True)
        self.two_segment_strict_check.toggled.connect(lambda value: self.controller.set_two_segment_strict_mode(bool(value)))
        self.two_segment_lower_trust_check = QCheckBox("Allow lower trust")
        self.two_segment_lower_trust_check.toggled.connect(lambda value: self.controller.set_two_segment_allow_lower_trust(bool(value)))
        self.two_segment_run_button = QPushButton("Run Two-Segment Modeling")
        self.two_segment_run_button.setProperty("role", "primary")
        self.two_segment_run_button.clicked.connect(self.controller.run_two_segment_modeling_analysis)
        trust_row.addWidget(self.two_segment_strict_check)
        trust_row.addWidget(self.two_segment_lower_trust_check)
        trust_row.addWidget(self.two_segment_run_button)
        trust_row.addStretch(1)
        two_segment_card.body_layout.addLayout(trust_row)
        self.two_segment_pairs = _PairsWidget()
        two_segment_card.body_layout.addWidget(self.two_segment_pairs)
        left.addWidget(two_segment_card)

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
        self._sync_two_segment_run_list(state)
        self.two_segment_pairs.set_pairs(state.two_segment_summary_pairs)
        self.two_segment_run_button.setEnabled(state.two_segment_can_run)
        self.two_segment_open_output_button.setEnabled(state.two_segment_can_open_output)
        self.two_segment_export_button.setEnabled(state.two_segment_can_export_output)
        self._set_checkbox(self.mike_check, state.include_mike)
        self._set_checkbox(self.camarillo_check, state.include_camarillo)
        self._set_checkbox(self.ann_check, state.include_ann)
        self._set_checkbox(self.two_segment_linear_check, state.two_segment_include_linear)
        self._set_checkbox(self.two_segment_ann_check, state.two_segment_include_ann)
        self._set_checkbox(self.two_segment_camarillo_check, state.two_segment_include_camarillo)
        self._set_checkbox(self.two_segment_mike_check, state.two_segment_include_mike)
        self._set_checkbox(self.two_segment_strict_check, state.two_segment_strict_mode)
        self._set_checkbox(self.two_segment_lower_trust_check, state.two_segment_allow_lower_trust)
        self._set_checkbox(self.two_segment_orientation_check, state.two_segment_include_orientation_if_available)
        self._set_checkbox(self.two_segment_ann_sweep_check, state.two_segment_ann_sweep_enabled)
        self._set_combo(self.scope_combo, state.evaluation_scope)
        self._set_combo(self.two_segment_label_mode_combo, state.two_segment_label_mode)
        self._set_combo(self.two_segment_hidden_combo, state.two_segment_ann_hidden_layers)
        self._set_spin_value(self.two_segment_epochs_spin, state.two_segment_ann_epochs)
        self._set_spin_value(self.two_segment_test_fraction_spin, state.two_segment_test_fraction)

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

    def _sync_two_segment_run_list(self, state: ModelingViewState) -> None:
        current_paths = [self.two_segment_run_list.item(index).data(Qt.UserRole) for index in range(self.two_segment_run_list.count())]
        target_paths = list(state.two_segment_dataset_runs)
        if current_paths != target_paths:
            with QSignalBlocker(self.two_segment_run_list):
                self.two_segment_run_list.clear()
                for path in target_paths:
                    run_path = Path(path)
                    item = QListWidgetItem(run_path.name)
                    item.setData(Qt.UserRole, str(path))
                    self.two_segment_run_list.addItem(item)
        selected = set(state.selected_two_segment_run_paths)
        with QSignalBlocker(self.two_segment_run_list):
            self.two_segment_run_list.clearSelection()
            for index in range(self.two_segment_run_list.count()):
                item = self.two_segment_run_list.item(index)
                if item.data(Qt.UserRole) in selected:
                    item.setSelected(True)

    def _on_dataset_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.controller.select_dataset(str(current.data(Qt.UserRole)))

    def _on_artifact_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.controller.select_artifact(str(current.data(Qt.UserRole)))

    def _on_two_segment_selection_changed(self) -> None:
        paths = [
            str(item.data(Qt.UserRole))
            for item in self.two_segment_run_list.selectedItems()
        ]
        self.controller.select_two_segment_runs(paths)
        self.update(self.controller.refresh())

    def _on_scope_changed(self, _index: int) -> None:
        value = self.scope_combo.currentData()
        if value:
            self.controller.set_evaluation_scope(str(value))

    def _on_two_segment_label_mode_changed(self, _index: int) -> None:
        value = self.two_segment_label_mode_combo.currentData()
        if value:
            self.controller.set_two_segment_label_mode(str(value))

    def _on_two_segment_hidden_changed(self, _index: int) -> None:
        value = self.two_segment_hidden_combo.currentData()
        if value:
            self.controller.set_two_segment_ann_hidden_layers(str(value))

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

    def _open_two_segment_output(self) -> None:
        state = self.controller.refresh()
        if state.two_segment_last_output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.two_segment_last_output_path))))

    def _export_two_segment_output(self) -> None:
        try:
            self.controller.export_last_two_segment_modeling_bundle()
        except Exception as exc:
            self.status_label.setText(f"Two-segment export failed: {exc}")
            return
        self.update(self.controller.refresh())

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

    @staticmethod
    def _set_spin_value(widget, value) -> None:
        if widget.value() == value:
            return
        with QSignalBlocker(widget):
            widget.setValue(value)


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
