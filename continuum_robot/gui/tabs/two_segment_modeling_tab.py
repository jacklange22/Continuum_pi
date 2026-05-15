"""Dedicated Two-Segment Modeling tab.

Carved out of the original combined Modeling tab so the single-segment workflow
(Mike / Camarillo / forward ANN comparison) has clear visual real estate and the
two-segment pipeline gets its own well-organized panel rather than competing for
column space.

UI design priorities:

- Single column, top-down flow: dataset selection → model selection → analysis
  config → trust gating → run → results.
- Each logical group is its own ``_Card`` so the eye can scan quickly.
- The run button stays primary and centered; everything else is supporting.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.two_segment_modeling_controller import (
    TwoSegmentModelingController,
    TwoSegmentModelingViewState,
)
from continuum_robot.gui.theme import COLORS, grouped_workspace_stylesheet


class TwoSegmentModelingTab(QWidget):
    """Two-segment offline modeling workspace."""

    def __init__(self, controller: TwoSegmentModelingController, parent=None) -> None:
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("twoSegmentModelingWorkspace")
        self.setStyleSheet(
            grouped_workspace_stylesheet(
                object_name="twoSegmentModelingWorkspace",
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

        title = QLabel("Two-Segment Modeling")
        title.setProperty("role", "title")
        hint = QLabel(
            "Offline analysis for ``two_segment_collect_pose_command_dataset`` runs. "
            "This tab does not enable live two-segment control."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(hint)

        # Dataset selection
        dataset_card = _Card(
            "Dataset Runs",
            "Pick one or more two-segment runs to combine for offline modeling.",
        )
        dataset_buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Runs")
        self.refresh_button.setProperty("variant", "ghost")
        self.refresh_button.clicked.connect(self._refresh)
        dataset_buttons.addWidget(self.refresh_button)
        dataset_buttons.addStretch(1)
        dataset_card.body_layout.addLayout(dataset_buttons)
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.run_list.setMinimumHeight(160)
        self.run_list.itemSelectionChanged.connect(self._on_selection_changed)
        dataset_card.body_layout.addWidget(self.run_list)
        self.trainability_pairs = _PairsWidget()
        dataset_card.body_layout.addWidget(self.trainability_pairs)
        root.addWidget(dataset_card)

        # Models selection
        models_card = _Card(
            "Models",
            "Pick which model families to fit and compare for this two-segment analysis.",
        )
        model_row = QHBoxLayout()
        self.linear_check = QCheckBox("Linear baseline")
        self.linear_check.setChecked(True)
        self.linear_check.toggled.connect(
            lambda value: self.controller.set_model_enabled("linear_baseline", bool(value))
        )
        self.ann_check = QCheckBox("ANN")
        self.ann_check.setChecked(True)
        self.ann_check.toggled.connect(
            lambda value: self.controller.set_model_enabled("ann", bool(value))
        )
        self.camarillo_check = QCheckBox("Camarillo (needs geometry)")
        self.camarillo_check.toggled.connect(
            lambda value: self.controller.set_model_enabled("camarillo", bool(value))
        )
        self.mike_check = QCheckBox("Mike (needs geometry)")
        self.mike_check.toggled.connect(
            lambda value: self.controller.set_model_enabled("mike_constant_curvature", bool(value))
        )
        for box in (self.linear_check, self.ann_check, self.camarillo_check, self.mike_check):
            model_row.addWidget(box)
        model_row.addStretch(1)
        models_card.body_layout.addLayout(model_row)
        root.addWidget(models_card)

        # Labels and ANN config
        ann_card = _Card(
            "ANN Configuration",
            "Pose-label scope + ANN architecture / training knobs for the two-segment fit.",
        )
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label Mode"))
        self.label_mode_combo = QComboBox()
        for label, value in [
            ("Auto", "auto"),
            ("Distal XYZ", "distal_xyz"),
            ("Distal Pose 6", "distal_pose6"),
            ("Two-Coil XYZ", "two_coil_xyz"),
            ("Two-Coil Pose 12", "two_coil_pose12"),
        ]:
            self.label_mode_combo.addItem(label, value)
        self.label_mode_combo.currentIndexChanged.connect(self._on_label_mode_changed)
        self.orientation_check = QCheckBox("Orientation if available")
        self.orientation_check.toggled.connect(
            lambda value: self.controller.set_include_orientation_if_available(bool(value))
        )
        label_row.addWidget(self.label_mode_combo, 1)
        label_row.addWidget(self.orientation_check)
        ann_card.body_layout.addLayout(label_row)

        config_row = QHBoxLayout()
        self.sweep_check = QCheckBox("ANN sweep")
        self.sweep_check.toggled.connect(
            lambda value: self.controller.set_ann_sweep_enabled(bool(value))
        )
        config_row.addWidget(self.sweep_check)
        config_row.addWidget(QLabel("Hidden"))
        self.hidden_combo = QComboBox()
        for label in ["128,128", "64,64", "32,32"]:
            self.hidden_combo.addItem(label, label)
        self.hidden_combo.currentIndexChanged.connect(self._on_hidden_changed)
        config_row.addWidget(self.hidden_combo)
        config_row.addWidget(QLabel("Epochs"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 5000)
        self.epochs_spin.setValue(200)
        self.epochs_spin.valueChanged.connect(lambda value: self.controller.set_ann_epochs(int(value)))
        config_row.addWidget(self.epochs_spin)
        config_row.addWidget(QLabel("Test fraction"))
        self.test_fraction_spin = QDoubleSpinBox()
        self.test_fraction_spin.setRange(0.05, 0.9)
        self.test_fraction_spin.setSingleStep(0.05)
        self.test_fraction_spin.setValue(0.25)
        self.test_fraction_spin.valueChanged.connect(
            lambda value: self.controller.set_test_fraction(float(value))
        )
        config_row.addWidget(self.test_fraction_spin)
        config_row.addStretch(1)
        ann_card.body_layout.addLayout(config_row)
        root.addWidget(ann_card)

        # Trust + run
        trust_card = _Card(
            "Run",
            "Strict mode rejects lower-trust runs by default. Allow lower trust only for debug.",
        )
        trust_row = QHBoxLayout()
        self.strict_check = QCheckBox("Strict")
        self.strict_check.setChecked(True)
        self.strict_check.toggled.connect(lambda value: self.controller.set_strict_mode(bool(value)))
        self.lower_trust_check = QCheckBox("Allow lower trust")
        self.lower_trust_check.toggled.connect(
            lambda value: self.controller.set_allow_lower_trust(bool(value))
        )
        trust_row.addWidget(self.strict_check)
        trust_row.addWidget(self.lower_trust_check)
        trust_row.addStretch(1)
        self.run_button = QPushButton("Run Two-Segment Modeling")
        self.run_button.setProperty("role", "primary")
        self.run_button.clicked.connect(self.controller.run_analysis)
        self.run_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        trust_row.addWidget(self.run_button, 1)
        trust_card.body_layout.addLayout(trust_row)

        actions_row = QHBoxLayout()
        self.open_output_button = QPushButton("Open Output")
        self.open_output_button.setProperty("variant", "ghost")
        self.open_output_button.clicked.connect(self._open_output)
        self.export_button = QPushButton("Export Bundle")
        self.export_button.setProperty("variant", "ghost")
        self.export_button.clicked.connect(self._export_bundle)
        actions_row.addWidget(self.open_output_button)
        actions_row.addWidget(self.export_button)
        actions_row.addStretch(1)
        trust_card.body_layout.addLayout(actions_row)
        self.status_label = QLabel(self.controller.state.status_message)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS.text_secondary};")
        trust_card.body_layout.addWidget(self.status_label)
        root.addWidget(trust_card)
        root.addStretch(1)

    # ---- updates -----------------------------------------------------------------
    def update(self, state: TwoSegmentModelingViewState) -> None:
        self._sync_run_list(state)
        self.trainability_pairs.set_pairs(state.trainability_pairs)
        self.status_label.setText(state.status_message)
        self.run_button.setEnabled(state.can_run)
        self.open_output_button.setEnabled(state.can_open_output)
        self.export_button.setEnabled(state.can_export_output)
        for box, value in (
            (self.linear_check, state.include_linear),
            (self.ann_check, state.include_ann),
            (self.camarillo_check, state.include_camarillo),
            (self.mike_check, state.include_mike),
            (self.strict_check, state.strict_mode),
            (self.lower_trust_check, state.allow_lower_trust),
            (self.orientation_check, state.include_orientation_if_available),
            (self.sweep_check, state.ann_sweep_enabled),
        ):
            if box.isChecked() != bool(value):
                with QSignalBlocker(box):
                    box.setChecked(bool(value))
        self._set_combo(self.label_mode_combo, state.label_mode)
        self._set_combo(self.hidden_combo, state.ann_hidden_layers)
        if self.epochs_spin.value() != int(state.ann_epochs):
            with QSignalBlocker(self.epochs_spin):
                self.epochs_spin.setValue(int(state.ann_epochs))
        if abs(self.test_fraction_spin.value() - float(state.test_fraction)) > 1e-6:
            with QSignalBlocker(self.test_fraction_spin):
                self.test_fraction_spin.setValue(float(state.test_fraction))

    # ---- handlers ----------------------------------------------------------------
    def _refresh(self) -> None:
        self.controller.invalidate_catalog()
        self.update(self.controller.refresh())

    def _on_selection_changed(self) -> None:
        paths = [
            str(item.data(Qt.UserRole))
            for item in self.run_list.selectedItems()
        ]
        self.controller.select_runs(paths)
        self.update(self.controller.refresh())

    def _on_label_mode_changed(self, _index: int) -> None:
        value = self.label_mode_combo.currentData()
        if value:
            self.controller.set_label_mode(str(value))

    def _on_hidden_changed(self, _index: int) -> None:
        value = self.hidden_combo.currentData()
        if value:
            self.controller.set_ann_hidden_layers(str(value))

    def _open_output(self) -> None:
        state = self.controller.refresh()
        if state.last_output_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(state.last_output_path))))

    def _export_bundle(self) -> None:
        try:
            self.controller.export_last_bundle()
        except Exception as exc:
            self.status_label.setText(f"Two-segment export failed: {exc}")

    # ---- helpers -----------------------------------------------------------------
    def _sync_run_list(self, state: TwoSegmentModelingViewState) -> None:
        current_paths = [self.run_list.item(i).data(Qt.UserRole) for i in range(self.run_list.count())]
        target_paths = list(state.dataset_runs)
        if current_paths != target_paths:
            with QSignalBlocker(self.run_list):
                self.run_list.clear()
                for path in target_paths:
                    item = QListWidgetItem(Path(path).name)
                    item.setData(Qt.UserRole, str(path))
                    self.run_list.addItem(item)
        with QSignalBlocker(self.run_list):
            selected = set(state.selected_run_paths)
            for i in range(self.run_list.count()):
                item = self.run_list.item(i)
                item.setSelected(item.data(Qt.UserRole) in selected)

    @staticmethod
    def _set_combo(widget: QComboBox, value: str) -> None:
        for index in range(widget.count()):
            if widget.itemData(index) == value:
                if widget.currentIndex() != index:
                    with QSignalBlocker(widget):
                        widget.setCurrentIndex(index)
                return


class _Card(QFrame):
    """Visual grouping card. Matches the styling of the single-segment Modeling tab."""

    def __init__(self, title: str, subtitle: str, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
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
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(6)

    def set_pairs(self, pairs: list[tuple[str, str]]) -> None:
        signature = tuple((str(label), str(value)) for label, value in pairs)
        if signature == self._signature:
            return
        self._signature = signature
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for label, value in signature:
            row = QFrame()
            row.setStyleSheet(
                f"background: {COLORS.surface_bg}; border: 1px solid {COLORS.surface_border}; border-radius: 8px;"
            )
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(10, 6, 10, 6)
            row_layout.setSpacing(10)
            key = QLabel(label)
            key.setStyleSheet(f"color: {COLORS.text_muted};")
            val = QLabel(value)
            val.setWordWrap(True)
            val.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 600;")
            row_layout.addWidget(key, 1)
            row_layout.addWidget(val, 2)
            self._layout.addWidget(row)
