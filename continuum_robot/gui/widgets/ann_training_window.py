"""Popout window for canonical dataset -> legacy ANN training."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.ann_training_controller import AnnTrainingController, AnnTrainingViewState
from continuum_robot.modeling.ann_training import format_hidden_layers_for_ann_ui
from continuum_robot.gui.theme import COLORS
from continuum_robot.gui.view_utils import editable_update_blocked, set_line_edit_text, set_spinbox_value
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


class AnnTrainingWindow(QWidget):
    """Non-modal ANN training popout."""

    REFRESH_INTERVAL_MS = 250

    def __init__(self, controller: AnnTrainingController, parent=None) -> None:
        super().__init__(parent, Qt.Window)
        self.controller = controller
        self.setWindowTitle("ANN Training")
        self.resize(1440, 920)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_state)
        self._hidden_layers_user_editing = False
        self._build_ui()
        self._refresh_timer.start(self.REFRESH_INTERVAL_MS)
        self._refresh_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        self.main_scroll_area = QScrollArea()
        self.main_scroll_area.setWidgetResizable(True)
        self.main_scroll_area.setFrameShape(QFrame.NoFrame)
        root.addWidget(self.main_scroll_area)

        content_widget = QWidget()
        self.main_scroll_area.setWidget(content_widget)
        content_root = QVBoxLayout(content_widget)
        content_root.setContentsMargins(0, 0, 0, 0)
        content_root.setSpacing(12)

        hint = QLabel(
            "Train the legacy full-pose ANN directly from canonical modeling datasets. "
            "V1 keeps the scope to dataset selection, backend/device inspection, warmup runtime estimate, "
            "training, cancellation, and artifact management."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS.text_secondary};")
        content_root.addWidget(hint)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        content_root.addLayout(content, 1)

        left_column = QVBoxLayout()
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(12)
        content.addLayout(left_column, 3)

        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(12)
        content.addWidget(right_panel, 4)

        dataset_card = _Card(
            "Modeling Datasets",
            "Discovered from data/experiments (and optional mock/archived roots). Train/Benchmark require structurally valid exports "
            "and trust policy unless you enable debug overrides.",
        )
        dataset_toolbar = QHBoxLayout()
        dataset_toolbar.setContentsMargins(0, 0, 0, 0)
        self.refresh_catalog_button = QPushButton("Refresh")
        self.refresh_catalog_button.setProperty("variant", "ghost")
        self.refresh_catalog_button.clicked.connect(self._refresh_catalogs)
        self.open_dataset_button = QPushButton("Open Dataset Folder")
        self.open_dataset_button.setProperty("variant", "ghost")
        self.open_dataset_button.clicked.connect(self._open_selected_dataset)
        dataset_toolbar.addWidget(self.refresh_catalog_button)
        dataset_toolbar.addWidget(self.open_dataset_button)
        dataset_toolbar.addStretch(1)
        dataset_card.body_layout.addLayout(dataset_toolbar)
        self.dataset_catalog_hint = QLabel("")
        self.dataset_catalog_hint.setWordWrap(True)
        self.dataset_catalog_hint.setStyleSheet(f"color: {COLORS.text_secondary};")
        dataset_card.body_layout.addWidget(self.dataset_catalog_hint)
        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(14)
        self.show_non_trainable_check = QCheckBox("Show non-trainable / debug datasets")
        self.show_non_trainable_check.toggled.connect(self._on_show_non_trainable_toggled)
        self.show_mock_roots_check = QCheckBox("Show mock dataset roots")
        self.show_mock_roots_check.toggled.connect(self._on_show_mock_roots_toggled)
        self.include_archived_check = QCheckBox("Include archived runs")
        self.include_archived_check.toggled.connect(self._on_include_archived_toggled)
        filter_row.addWidget(self.show_non_trainable_check)
        filter_row.addWidget(self.show_mock_roots_check)
        filter_row.addWidget(self.include_archived_check)
        filter_row.addStretch(1)
        dataset_card.body_layout.addLayout(filter_row)
        dbg_row = QHBoxLayout()
        dbg_row.setContentsMargins(0, 0, 0, 0)
        dbg_row.setSpacing(14)
        self.allow_mock_training_check = QCheckBox("Allow training on mock datasets (debug)")
        self.allow_mock_training_check.toggled.connect(self._on_allow_mock_training_toggled)
        self.allow_lower_trust_check = QCheckBox("Include lower-trust / invalid-flag training (debug)")
        self.allow_lower_trust_check.toggled.connect(self._on_allow_lower_trust_toggled)
        dbg_row.addWidget(self.allow_mock_training_check)
        dbg_row.addWidget(self.allow_lower_trust_check)
        dbg_row.addStretch(1)
        dataset_card.body_layout.addLayout(dbg_row)
        self.dataset_list = QListWidget()
        self.dataset_list.currentItemChanged.connect(self._on_dataset_selected)
        self.dataset_list.setMinimumHeight(220)
        dataset_card.body_layout.addWidget(self.dataset_list)
        self.dataset_pairs = _PairsWidget()
        dataset_card.body_layout.addWidget(self.dataset_pairs)
        left_column.addWidget(dataset_card)

        system_card = _Card("Machine / Backend", "Inspect the current runtime and pick the backend used for benchmark and training.")
        system_form = QFormLayout()
        system_form.setContentsMargins(0, 0, 0, 0)
        system_form.setSpacing(10)
        self.backend_combo = QComboBox()
        self.backend_combo.currentIndexChanged.connect(self._on_backend_selected)
        system_form.addRow("Backend", self.backend_combo)
        system_card.body_layout.addLayout(system_form)
        self.system_pairs = _PairsWidget()
        system_card.body_layout.addWidget(self.system_pairs)
        left_column.addWidget(system_card)

        artifact_card = _Card("Saved ANN Artifacts", "Inspect previously trained artifact bundles and their saved metadata.")
        artifact_toolbar = QHBoxLayout()
        artifact_toolbar.setContentsMargins(0, 0, 0, 0)
        self.artifact_root_edit = QLineEdit()
        self.artifact_root_edit.editingFinished.connect(
            lambda: self.controller.set_artifact_root(self.artifact_root_edit.text().strip())
        )
        browse_artifact_root = QPushButton("Browse")
        browse_artifact_root.setProperty("variant", "ghost")
        browse_artifact_root.clicked.connect(self._browse_artifact_root)
        self.open_artifact_button = QPushButton("Open Artifact Folder")
        self.open_artifact_button.setProperty("variant", "ghost")
        self.open_artifact_button.clicked.connect(self._open_selected_artifact)
        artifact_toolbar.addWidget(self.artifact_root_edit, 1)
        artifact_toolbar.addWidget(browse_artifact_root)
        artifact_toolbar.addWidget(self.open_artifact_button)
        artifact_card.body_layout.addLayout(artifact_toolbar)
        self.artifact_list = QListWidget()
        self.artifact_list.currentItemChanged.connect(self._on_artifact_selected)
        self.artifact_list.setMinimumHeight(180)
        artifact_card.body_layout.addWidget(self.artifact_list)
        self.artifact_pairs = _PairsWidget()
        artifact_card.body_layout.addWidget(self.artifact_pairs)
        left_column.addWidget(artifact_card, 1)

        params_card = _Card("Training Parameters", "Legacy ANN architecture with full-pose output and grouped train/validation/test splits.")
        params_form = QFormLayout()
        params_form.setContentsMargins(0, 0, 0, 0)
        params_form.setSpacing(10)
        self.hidden_layers_preset_combo = QComboBox()
        self.hidden_layers_preset_combo.addItem("Custom (field below)", None)
        for label, layers in (
            ("[32, 32]", [32, 32]),
            ("[64, 64]", [64, 64]),
            ("[128, 128]", [128, 128]),
        ):
            self.hidden_layers_preset_combo.addItem(label, layers)
        self.hidden_layers_preset_combo.currentIndexChanged.connect(self._on_hidden_layers_preset_changed)
        self.hidden_layers_edit = QLineEdit()
        self.hidden_layers_edit.textChanged.connect(self._on_hidden_layers_text_changed)
        self.hidden_layers_edit.editingFinished.connect(self._on_hidden_layers_editing_finished)
        hidden_layers_row = QHBoxLayout()
        hidden_layers_row.setContentsMargins(0, 0, 0, 0)
        hidden_layers_row.setSpacing(8)
        hidden_layers_row.addWidget(self.hidden_layers_preset_combo, 1)
        hidden_layers_row.addWidget(self.hidden_layers_edit, 2)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.000001, 1.0)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.valueChanged.connect(lambda value: self.controller.set_learning_rate(float(value)))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 65536)
        self.batch_size_spin.valueChanged.connect(lambda value: self.controller.set_batch_size(int(value)))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.valueChanged.connect(lambda value: self.controller.set_epochs(int(value)))
        self.loss_combo = QComboBox()
        self.loss_combo.addItem("Pose", "pose")
        self.loss_combo.addItem("Position", "position")
        self.loss_combo.addItem("Orientation", "orientation")
        self.loss_combo.currentIndexChanged.connect(
            lambda _index: self.controller.set_loss_kind(str(self.loss_combo.currentData()))
        )
        self.pose_scale_spin = QDoubleSpinBox()
        self.pose_scale_spin.setDecimals(3)
        self.pose_scale_spin.setRange(0.001, 1000.0)
        self.pose_scale_spin.setSingleStep(0.5)
        self.pose_scale_spin.valueChanged.connect(
            lambda value: self.controller.set_pose_orientation_scale(float(value))
        )
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999999)
        self.seed_spin.valueChanged.connect(lambda value: self.controller.set_random_seed(int(value)))
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setDecimals(3)
        self.train_ratio_spin.setRange(0.0, 1.0)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.valueChanged.connect(lambda value: self.controller.set_train_ratio(float(value)))
        self.validation_ratio_spin = QDoubleSpinBox()
        self.validation_ratio_spin.setDecimals(3)
        self.validation_ratio_spin.setRange(0.0, 1.0)
        self.validation_ratio_spin.setSingleStep(0.05)
        self.validation_ratio_spin.valueChanged.connect(
            lambda value: self.controller.set_validation_ratio(float(value))
        )
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setDecimals(3)
        self.test_ratio_spin.setRange(0.0, 1.0)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.valueChanged.connect(lambda value: self.controller.set_test_ratio(float(value)))
        self.checkpoint_check = QCheckBox("Save per-epoch checkpoints")
        self.checkpoint_check.toggled.connect(lambda value: self.controller.set_checkpointing(bool(value)))
        self.artifact_name_edit = QLineEdit()
        self.artifact_name_edit.editingFinished.connect(
            lambda: self.controller.set_artifact_name(self.artifact_name_edit.text().strip())
        )
        params_form.addRow("Hidden Layers", hidden_layers_row)
        params_form.addRow("Learning Rate", self.learning_rate_spin)
        params_form.addRow("Batch Size", self.batch_size_spin)
        params_form.addRow("Epochs", self.epochs_spin)
        params_form.addRow("Loss", self.loss_combo)
        params_form.addRow("Pose Weight", self.pose_scale_spin)
        params_form.addRow("Random Seed", self.seed_spin)
        params_form.addRow("Train Ratio", self.train_ratio_spin)
        params_form.addRow("Validation Ratio", self.validation_ratio_spin)
        params_form.addRow("Test Ratio", self.test_ratio_spin)
        params_form.addRow("Checkpointing", self.checkpoint_check)
        params_form.addRow("Artifact Name", self.artifact_name_edit)
        params_card.body_layout.addLayout(params_form)
        self.config_error_label = QLabel("")
        self.config_error_label.setWordWrap(True)
        self.config_error_label.setStyleSheet(f"color: {COLORS.error_fg};")
        params_card.body_layout.addWidget(self.config_error_label)
        self.right_layout.addWidget(params_card)

        runtime_card = _Card("Runtime Estimate / Training", "Run a real warmup benchmark, then train asynchronously without blocking the rest of the app.")
        sweep_row = QHBoxLayout()
        sweep_row.setContentsMargins(0, 0, 0, 0)
        sweep_row.setSpacing(10)
        self.sweep_include_linear_check = QCheckBox("Include linear ridge baseline in sweep")
        self.sweep_include_linear_check.setChecked(True)
        self.sweep_include_linear_check.toggled.connect(
            lambda value: self.controller.set_model_sweep_include_linear_baseline(bool(value))
        )
        self.sweep_extra_edit = QLineEdit()
        self.sweep_extra_edit.setPlaceholderText('Extra ANN widths (optional), e.g. "48,48 | 96"')
        self.sweep_extra_edit.editingFinished.connect(
            lambda: self.controller.set_model_sweep_extra_hidden_layers_text(self.sweep_extra_edit.text().strip())
        )
        self.run_sweep_button = QPushButton("Run Model Sweep")
        self.run_sweep_button.setProperty("variant", "ghost")
        self.run_sweep_button.clicked.connect(self.controller.run_sweep)
        sweep_row.addWidget(self.sweep_include_linear_check, 2)
        sweep_row.addWidget(self.sweep_extra_edit, 3)
        sweep_row.addWidget(self.run_sweep_button, 1)
        self.sweep_best_label = QLabel("")
        self.sweep_best_label.setWordWrap(True)
        self.sweep_warning_label = QLabel("")
        self.sweep_warning_label.setWordWrap(True)
        self.sweep_warning_label.setStyleSheet(f"color: {COLORS.warning_fg};")
        runtime_toolbar = QHBoxLayout()
        runtime_card.body_layout.addLayout(sweep_row)
        runtime_card.body_layout.addWidget(self.sweep_best_label)
        runtime_card.body_layout.addWidget(self.sweep_warning_label)
        runtime_toolbar.setContentsMargins(0, 0, 0, 0)
        self.benchmark_button = QPushButton("Benchmark Estimate")
        self.benchmark_button.setProperty("variant", "ghost")
        self.benchmark_button.clicked.connect(self.controller.benchmark)
        self.train_button = QPushButton("Train Legacy ANN")
        self.train_button.setProperty("variant", "primary")
        self.train_button.clicked.connect(self.controller.train)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("variant", "danger")
        self.cancel_button.clicked.connect(self.controller.cancel)
        runtime_toolbar.addWidget(self.benchmark_button)
        runtime_toolbar.addWidget(self.train_button)
        runtime_toolbar.addWidget(self.cancel_button)
        runtime_toolbar.addStretch(1)
        runtime_card.body_layout.addLayout(runtime_toolbar)
        self.progress_bar = QProgressBar()
        runtime_card.body_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Select a modeling dataset to prepare ANN training.")
        self.status_label.setWordWrap(True)
        runtime_card.body_layout.addWidget(self.status_label)
        self.runtime_pairs = _PairsWidget()
        runtime_card.body_layout.addWidget(self.runtime_pairs)
        self.right_layout.addWidget(runtime_card)

        history_card = _Card("Loss History", "Live train / validation loss curves for the current run or the selected saved artifact.")
        self.results_widget = ExperimentResultsWidget()
        history_card.body_layout.addWidget(self.results_widget)
        self.right_layout.addWidget(history_card, 1)
        self.right_layout.addStretch(1)

    def _refresh_state(self) -> None:
        state = self.controller.refresh()
        self._sync_dataset_list(state)
        self._sync_artifact_list(state)
        self._sync_backend_combo(state)
        self._sync_parameters(state)
        self.dataset_pairs.set_pairs(state.dataset_summary_pairs)
        self.system_pairs.set_pairs(state.system_summary_pairs)
        self.artifact_pairs.set_pairs(state.artifact_summary_pairs)
        self.runtime_pairs.set_pairs(state.estimate_summary_pairs)
        self.results_widget.set_model(state.visualization_model)
        self.status_label.setText(state.status_message)
        self.config_error_label.setText(state.config_error or "")
        self.config_error_label.setVisible(bool(state.config_error))
        self.open_dataset_button.setEnabled(bool(state.selected_dataset_path))
        self.open_artifact_button.setEnabled(bool(state.selected_artifact_path))
        self.benchmark_button.setEnabled(state.can_benchmark)
        self.train_button.setEnabled(state.can_train)
        self.run_sweep_button.setEnabled(state.can_run_sweep)
        self.cancel_button.setEnabled(state.training_active or state.sweep_active)
        self.sweep_include_linear_check.setEnabled(not state.sweep_active)
        self.sweep_extra_edit.setEnabled(not state.sweep_active)
        self.sweep_best_label.setText(state.sweep_best_model_text or "")
        self.sweep_warning_label.setText(state.sweep_split_warning or "")
        self.sweep_warning_label.setVisible(bool(state.sweep_split_warning))
        self.progress_bar.setMaximum(max(1, state.total_epochs or 1))
        self.progress_bar.setValue(int(state.current_epoch))
        hint = state.dataset_catalog_message or ""
        if state.catalog_trainable_total or state.catalog_non_trainable_total:
            hint = (
                f"{hint}\nCatalog: {state.catalog_trainable_total} trainable, "
                f"{state.catalog_non_trainable_total} non-trainable (all roots)."
            ).strip()
        self.dataset_catalog_hint.setText(hint.strip())
        self._sync_dataset_filters(state)

    def _sync_dataset_filters(self, state: AnnTrainingViewState) -> None:
        with QSignalBlocker(self.show_non_trainable_check):
            self.show_non_trainable_check.setChecked(bool(state.show_non_trainable_datasets))
        with QSignalBlocker(self.show_mock_roots_check):
            self.show_mock_roots_check.setChecked(bool(state.show_mock_dataset_roots))
        with QSignalBlocker(self.include_archived_check):
            self.include_archived_check.setChecked(bool(state.include_archived_datasets))
        with QSignalBlocker(self.allow_mock_training_check):
            self.allow_mock_training_check.setChecked(bool(state.allow_mock_training))
        with QSignalBlocker(self.allow_lower_trust_check):
            self.allow_lower_trust_check.setChecked(bool(state.allow_lower_trust_training))

    def _sync_dataset_list(self, state: AnnTrainingViewState) -> None:
        current_paths = [self.dataset_list.item(index).data(Qt.UserRole) for index in range(self.dataset_list.count())]
        target_paths = [str(dataset.path) for dataset in state.datasets]
        if current_paths != target_paths:
            with QSignalBlocker(self.dataset_list):
                self.dataset_list.clear()
                for dataset in state.datasets:
                    tag = "trainable" if dataset.trainable_for_legacy_ann else "blocked"
                    label = (
                        f"[{tag}] {dataset.run_name} | root={dataset.dataset_scan_root} | "
                        f"{dataset.accepted_legacy_trainable_count} legacy rows | "
                        f"{dataset.accepted_count} accepted"
                    )
                    item = QListWidgetItem(label)
                    item.setData(Qt.UserRole, str(dataset.path))
                    self.dataset_list.addItem(item)
        with QSignalBlocker(self.dataset_list):
            for index in range(self.dataset_list.count()):
                if self.dataset_list.item(index).data(Qt.UserRole) == state.selected_dataset_path:
                    self.dataset_list.setCurrentRow(index)
                    break

    def _sync_artifact_list(self, state: AnnTrainingViewState) -> None:
        current_paths = [self.artifact_list.item(index).data(Qt.UserRole) for index in range(self.artifact_list.count())]
        target_paths = [str(artifact.path) for artifact in state.artifacts]
        if current_paths != target_paths:
            with QSignalBlocker(self.artifact_list):
                self.artifact_list.clear()
                for artifact in state.artifacts:
                    label = (
                        f"{artifact.artifact_name} | {artifact.backend_name} | "
                        f"{artifact.status}"
                    )
                    item = QListWidgetItem(label)
                    item.setData(Qt.UserRole, str(artifact.path))
                    self.artifact_list.addItem(item)
        with QSignalBlocker(self.artifact_list):
            for index in range(self.artifact_list.count()):
                if self.artifact_list.item(index).data(Qt.UserRole) == state.selected_artifact_path:
                    self.artifact_list.setCurrentRow(index)
                    break

    def _sync_backend_combo(self, state: AnnTrainingViewState) -> None:
        report = state.backend_report
        if report is None:
            return
        with QSignalBlocker(self.backend_combo):
            self.backend_combo.clear()
            for option in report.backend_options:
                suffix = " (Recommended)" if option.recommended else ""
                available = "" if option.available else " [Unavailable]"
                self.backend_combo.addItem(f"{option.label}{suffix}{available}", option.name)
            for index in range(self.backend_combo.count()):
                if self.backend_combo.itemData(index) == report.selected_backend:
                    self.backend_combo.setCurrentIndex(index)
                    break

    def _sync_parameters(self, state: AnnTrainingViewState) -> None:
        config = self.controller.config_snapshot()
        if not self.hidden_layers_edit.hasFocus() and not self._hidden_layers_user_editing:
            self._set_line_text(self.hidden_layers_edit, self.controller.hidden_layers_text())
        self._sync_hidden_layers_preset_combo(config)
        self._set_double(self.learning_rate_spin, float(config.learning_rate))
        self._set_spin(self.batch_size_spin, int(config.batch_size))
        self._set_spin(self.epochs_spin, int(config.epochs))
        self._set_combo(self.loss_combo, str(config.loss_kind))
        self._set_double(self.pose_scale_spin, float(config.pose_orientation_scale))
        self._set_spin(self.seed_spin, int(config.random_seed))
        self._set_double(self.train_ratio_spin, float(config.train_ratio))
        self._set_double(self.validation_ratio_spin, float(config.validation_ratio))
        self._set_double(self.test_ratio_spin, float(config.test_ratio))
        self._set_checkbox(self.checkpoint_check, bool(config.checkpointing))
        self._set_line_text(self.artifact_name_edit, str(config.artifact_name))
        with QSignalBlocker(self.sweep_include_linear_check):
            self.sweep_include_linear_check.setChecked(bool(config.model_sweep_include_linear_baseline))
        self._set_line_text(self.sweep_extra_edit, str(config.model_sweep_extra_hidden_layers_text or ""))
        self._set_line_text(self.artifact_root_edit, self.controller.artifact_root_text())

    def _on_dataset_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        self.controller.select_dataset(str(current.data(Qt.UserRole)))

    def _on_artifact_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        self.controller.select_artifact(str(current.data(Qt.UserRole)))

    def _on_backend_selected(self, _index: int) -> None:
        backend_name = self.backend_combo.currentData()
        if backend_name:
            self.controller.select_backend(str(backend_name))

    def _on_hidden_layers_text_changed(self, text: str) -> None:
        if self.hidden_layers_edit.signalsBlocked():
            return
        canonical = self.controller.hidden_layers_text()
        if str(text).strip() == str(canonical).strip():
            self._hidden_layers_user_editing = False
        else:
            self._hidden_layers_user_editing = True

    def _on_hidden_layers_editing_finished(self) -> None:
        self.controller.set_hidden_layers_text(self.hidden_layers_edit.text())
        self._hidden_layers_user_editing = False

    def _on_hidden_layers_preset_changed(self, _index: int) -> None:
        layers = self.hidden_layers_preset_combo.currentData()
        if layers is None:
            return
        text = format_hidden_layers_for_ann_ui(list(layers))
        self.controller.set_hidden_layers_text(text)
        self._hidden_layers_user_editing = False
        with QSignalBlocker(self.hidden_layers_edit):
            self.hidden_layers_edit.setText(text)

    def _sync_hidden_layers_preset_combo(self, config) -> None:
        layers_tuple = tuple(int(x) for x in config.hidden_layers)
        target_index = 0
        for i in range(1, self.hidden_layers_preset_combo.count()):
            data = self.hidden_layers_preset_combo.itemData(i)
            if isinstance(data, list) and tuple(int(x) for x in data) == layers_tuple:
                target_index = i
                break
        with QSignalBlocker(self.hidden_layers_preset_combo):
            self.hidden_layers_preset_combo.setCurrentIndex(target_index)

    def _refresh_catalogs(self) -> None:
        self.controller.invalidate_catalog()
        self.controller.set_artifact_root(self.artifact_root_edit.text().strip())
        self._refresh_state()

    def _browse_artifact_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select ANN Artifact Root", self.artifact_root_edit.text().strip())
        if selected:
            self.controller.set_artifact_root(selected)
            self._refresh_state()

    def _on_show_non_trainable_toggled(self, value: bool) -> None:
        self.controller.set_show_non_trainable_datasets(bool(value))
        self._refresh_state()

    def _on_show_mock_roots_toggled(self, value: bool) -> None:
        self.controller.set_show_mock_dataset_roots(bool(value))
        self._refresh_state()

    def _on_include_archived_toggled(self, value: bool) -> None:
        self.controller.set_include_archived_datasets(bool(value))
        self._refresh_state()

    def _on_allow_mock_training_toggled(self, value: bool) -> None:
        self.controller.set_allow_mock_training(bool(value))
        self._refresh_state()

    def _on_allow_lower_trust_toggled(self, value: bool) -> None:
        self.controller.set_allow_lower_trust_training(bool(value))
        self._refresh_state()

    def _open_selected_dataset(self) -> None:
        state = self.controller.refresh()
        if state.selected_dataset_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.selected_dataset_path))))

    def _open_selected_artifact(self) -> None:
        state = self.controller.refresh()
        if state.selected_artifact_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.selected_artifact_path))))

    @staticmethod
    def _set_line_text(widget: QLineEdit, value: str) -> None:
        set_line_edit_text(widget, str(value), skip_if_focused=True, block_signals=True)

    @staticmethod
    def _set_spin(widget: QSpinBox, value: int) -> None:
        set_spinbox_value(widget, int(value), skip_if_editing=True, block_signals=True)

    @staticmethod
    def _set_double(widget: QDoubleSpinBox, value: float) -> None:
        set_spinbox_value(widget, float(value), skip_if_editing=True, block_signals=True)

    @staticmethod
    def _set_checkbox(widget: QCheckBox, value: bool) -> None:
        if widget.isChecked() == value:
            return
        with QSignalBlocker(widget):
            widget.setChecked(value)

    @staticmethod
    def _set_combo(widget: QComboBox, value: str) -> None:
        if editable_update_blocked(widget):
            return
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
