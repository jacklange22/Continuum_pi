"""Modeling analysis and comparison tab."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from continuum_robot.gui.controllers.modeling_controller import (
    HeadlineMetric,
    ModelingController,
    ModelingViewState,
)
from continuum_robot.gui.theme import COLORS, grouped_workspace_stylesheet
from continuum_robot.gui.widgets.experiment_results_widget import ExperimentResultsWidget


class ModelingTab(QWidget):
    """Dataset/artifact browser plus evaluation workspace."""

    def __init__(
        self,
        controller: ModelingController,
        *,
        open_in_ann_training=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        # Callback that opens the ANN training popout, pre-selecting the highlighted dataset.
        # Provided by the AppWindow so the Modeling tab can launch training without owning the
        # ANN controller/window itself.
        self._ann_training_opener = open_in_ann_training
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

        title = QLabel("Single-Segment Modeling")
        title.setProperty("role", "title")
        hint = QLabel(
            "Compare Mike, Camarillo, and a trained ANN on a single-segment "
            "collect_pose_command_dataset run. Optionally evaluate against a separate "
            "test acquisition (Wolfe §3.2.3) for thesis-grade cross-acquisition numbers. "
            "Two-segment work lives on its own tab."
        )
        hint.setProperty("role", "hint")
        hint.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(hint)

        # Workflow header — explicit 4-step guide so the operator's eye lands on the
        # ordered sequence of actions, not on whichever card happens to be biggest.
        workflow_frame = QFrame()
        workflow_frame.setStyleSheet(
            f"background: {COLORS.surface_alt_bg}; border: 1px solid {COLORS.surface_border}; "
            f"border-radius: 10px; padding: 8px 12px;"
        )
        workflow_layout = QHBoxLayout(workflow_frame)
        workflow_layout.setContentsMargins(12, 10, 12, 10)
        workflow_layout.setSpacing(12)
        for step_number, step_text in (
            ("1", "Pick a training dataset"),
            ("2", "Train ANN  →  popout"),
            ("3", "(Optional) Pick a separate test dataset"),
            ("4", "Run Comparison  →  read RMSE per model"),
        ):
            step_widget = QWidget()
            step_layout = QHBoxLayout(step_widget)
            step_layout.setContentsMargins(0, 0, 0, 0)
            step_layout.setSpacing(8)
            number_label = QLabel(step_number)
            number_label.setStyleSheet(
                f"color: white; background: {COLORS.selection_bg}; border-radius: 11px; "
                f"min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; "
                f"qproperty-alignment: AlignCenter; font-weight: 700;"
            )
            number_label.setAlignment(Qt.AlignCenter)
            text_label = QLabel(step_text)
            text_label.setStyleSheet(f"color: {COLORS.text_primary}; font-weight: 500;")
            step_layout.addWidget(number_label)
            step_layout.addWidget(text_label)
            workflow_layout.addWidget(step_widget)
        workflow_layout.addStretch(1)
        root.addWidget(workflow_frame)

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

        dataset_card = _Card(
            "Modeling Datasets",
            "Canonical collect_pose_command_dataset runs. "
            "workspace_coverage runs are usually training sets; "
            "angular_test_mesh runs are ideal Test Datasets (Wolfe §3.2.3).",
        )
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
        # Empty-state hint — shown only when zero datasets are discovered, with a clear
        # next action so the operator isn't left staring at a blank list.
        self.dataset_empty_hint = QLabel(
            "No collect_pose_command_dataset runs found. "
            "Run one via the Experiment tab to populate this list."
        )
        self.dataset_empty_hint.setWordWrap(True)
        self.dataset_empty_hint.setStyleSheet(
            f"color: {COLORS.text_muted}; font-style: italic; padding: 6px 10px;"
        )
        self.dataset_empty_hint.setVisible(False)
        dataset_card.body_layout.addWidget(self.dataset_empty_hint)
        # Positive identifier chip — surfaces dataset_mode at a glance. Especially useful
        # for spotting the new angular_test_mesh acquisitions intended as eval targets.
        self.dataset_mode_chip = QLabel("")
        self.dataset_mode_chip.setWordWrap(True)
        self.dataset_mode_chip.setStyleSheet(
            f"color: {COLORS.text_secondary}; padding: 4px 8px; border-radius: 6px;"
        )
        self.dataset_mode_chip.setVisible(False)
        dataset_card.body_layout.addWidget(self.dataset_mode_chip)
        self.dataset_pairs = _PairsWidget()
        dataset_card.body_layout.addWidget(self.dataset_pairs)
        left.addWidget(dataset_card)

        artifact_card = _Card(
            "ANN Artifacts",
            "Forward (cable → tip pose) ANN bundles trained via the ANN Training popout. "
            "Inverse models are hidden — train one via the popout for forward comparison here.",
        )
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
        # Mismatch chip — fires when the selected artifact was trained on a different
        # dataset than the currently-selected one. Operator sees this before clicking
        # Run Comparison and getting unexpected numbers.
        self.artifact_mismatch_chip = QLabel("")
        self.artifact_mismatch_chip.setWordWrap(True)
        self.artifact_mismatch_chip.setStyleSheet(
            f"color: {COLORS.warning_fg}; background: {COLORS.warning_bg}; "
            f"border: 1px solid {COLORS.warning_fg}; border-radius: 8px; "
            f"padding: 8px 12px; font-weight: 500;"
        )
        self.artifact_mismatch_chip.setVisible(False)
        artifact_card.body_layout.addWidget(self.artifact_mismatch_chip)
        self.artifact_pairs = _PairsWidget()
        artifact_card.body_layout.addWidget(self.artifact_pairs)
        left.addWidget(artifact_card, 1)

        # Past evaluations on the selected dataset — lets you compare today's
        # numbers to yesterday's without leaving the tab.
        history_card = _Card(
            "Evaluation History",
            "Recent comparison runs on the selected dataset. Click a row to open the folder.",
        )
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(120)
        self.history_list.itemDoubleClicked.connect(self._on_history_item_double_clicked)
        history_card.body_layout.addWidget(self.history_list)
        self.history_hint = QLabel("No prior evaluations yet for this dataset.")
        self.history_hint.setStyleSheet(f"color: {COLORS.text_muted}; font-style: italic;")
        history_card.body_layout.addWidget(self.history_hint)
        left.addWidget(history_card)

        controls_card = _Card(
            "Evaluation Controls",
            "Pick models, scope, and (optionally) a separate test dataset. Defaults are "
            "Mike + Camarillo + ANN on the artifact's held-out split.",
        )
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
        # Optional separate test-dataset selector (Wolfe §3.2.3 cross-acquisition eval).
        # "(same as training)" = use the artifact's training dataset (current behavior).
        # Any other entry evaluates the ANN against that dataset's samples instead.
        self.test_dataset_combo = QComboBox()
        self.test_dataset_combo.setToolTip(
            "Optional separate test dataset (Wolfe MS thesis §3.2.3). "
            "Defaults to evaluating on the training dataset's held-out split; "
            "pick a different run (e.g. angular_test_mesh) for thesis-grade numbers."
        )
        self.test_dataset_combo.currentIndexChanged.connect(self._on_test_dataset_changed)
        scope_row.addWidget(scope_label)
        scope_row.addWidget(self.scope_combo, 1)
        controls_card.body_layout.addLayout(scope_row)

        test_dataset_row = QHBoxLayout()
        test_dataset_row.setContentsMargins(0, 0, 0, 0)
        test_dataset_label = QLabel("Test Dataset")
        test_dataset_row.addWidget(test_dataset_label)
        test_dataset_row.addWidget(self.test_dataset_combo, 1)
        controls_card.body_layout.addLayout(test_dataset_row)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        self.evaluate_button = QPushButton("Run Comparison")
        self.evaluate_button.setProperty("role", "primary")
        self.evaluate_button.clicked.connect(self.controller.evaluate)
        self.train_ann_button = QPushButton("Train ANN")
        self.train_ann_button.setProperty("role", "primary")
        self.train_ann_button.setToolTip(
            "Open the ANN training popout with the currently selected modeling dataset preloaded."
        )
        self.train_ann_button.clicked.connect(self._open_ann_training_from_modeling)
        self.validate_rows_button = QPushButton("Validate Rows")
        self.validate_rows_button.setProperty("variant", "ghost")
        self.validate_rows_button.setToolTip(
            "Apply the complete_rows_only filter to the test dataset (or training "
            "dataset if no override) and surface the count + exclusion reasons."
        )
        self.validate_rows_button.clicked.connect(self._on_validate_rows_clicked)
        self.open_results_button = QPushButton("Open Results Folder")
        self.open_results_button.setProperty("variant", "ghost")
        self.open_results_button.clicked.connect(self._open_results_folder)
        action_row.addWidget(self.evaluate_button)
        action_row.addWidget(self.train_ann_button)
        action_row.addWidget(self.validate_rows_button)
        action_row.addWidget(self.open_results_button)
        action_row.addStretch(1)
        controls_card.body_layout.addLayout(action_row)
        # Row-filter status echo. Hidden until Validate Rows is clicked at least once.
        self.row_filter_status_label = QLabel("")
        self.row_filter_status_label.setWordWrap(True)
        self.row_filter_status_label.setStyleSheet(f"color: {COLORS.text_secondary};")
        self.row_filter_status_label.setVisible(False)
        controls_card.body_layout.addWidget(self.row_filter_status_label)
        left.addWidget(controls_card)

        # Two-segment work moved to its own tab (TwoSegmentModelingTab). This tab
        # focuses exclusively on the single-segment forward-comparison workflow
        # (Mike / Camarillo / ANN cable → tip pose).

        # ── Right column: three clearly-separated cards ────────────────────────
        #   1) Headline — big color-coded per-model RMSE blocks (the answer at-a-glance)
        #   2) Details  — scope, per-axis breakdown, same-session caveat
        #   3) Plots    — workspace overlay, histograms, heatmaps, loss curve
        headline_card = _Card(
            "Per-Model RMSE",
            "Surgical target: ≤1mm (green). 1–3mm amber. >3mm red.",
        )
        # Quick-share toolbar — operator-friendly export buttons next to the headline.
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        self.copy_summary_button = QPushButton("Copy summary")
        self.copy_summary_button.setProperty("variant", "ghost")
        self.copy_summary_button.setToolTip(
            "Copy a plain-text summary of the last evaluation (per-model RMSE + scope) "
            "to the clipboard. Easy to paste into a lab notebook or chat."
        )
        self.copy_summary_button.clicked.connect(self._copy_summary_to_clipboard)
        toolbar_row.addWidget(self.copy_summary_button)
        toolbar_row.addStretch(1)
        headline_card.body_layout.addLayout(toolbar_row)
        self.status_label = QLabel("Select a dataset, choose models, then run a comparison.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS.text_secondary};")
        headline_card.body_layout.addWidget(self.status_label)
        self.headline_metrics_widget = _HeadlineMetricsWidget()
        headline_card.body_layout.addWidget(self.headline_metrics_widget)
        # Same-session caveat chip. Visible when the test split is drawn from the
        # training run (default behavior). Tells the operator at a glance that the
        # numbers aren't from a separate test-mesh acquisition.
        self.same_session_chip = QLabel(
            "⚠ Same-session evaluation — test samples come from the training dataset. "
            "Collect a separate test acquisition for thesis-grade numbers."
        )
        self.same_session_chip.setWordWrap(True)
        self.same_session_chip.setStyleSheet(
            f"color: {COLORS.warning_fg}; background: {COLORS.warning_bg}; "
            f"border: 1px solid {COLORS.warning_fg}; border-radius: 8px; "
            f"padding: 8px 12px; font-weight: 500;"
        )
        self.same_session_chip.setVisible(False)
        headline_card.body_layout.addWidget(self.same_session_chip)
        # Positive chip — replaces the caveat when the operator does the right thing.
        # Wolfe §3.2.3 calls this out: a thesis-grade comparison evaluates on a
        # separately-collected test acquisition, not the artifact's training split.
        self.thesis_grade_chip = QLabel(
            "✓ Thesis-grade evaluation — ANN was trained on the dataset above and tested "
            "against a SEPARATE acquisition, matching Wolfe §3.2.3 p85 methodology."
        )
        self.thesis_grade_chip.setWordWrap(True)
        self.thesis_grade_chip.setStyleSheet(
            f"color: #86efac; background: #064e3b; border: 1px solid #16a34a; "
            f"border-radius: 8px; padding: 8px 12px; font-weight: 500;"
        )
        self.thesis_grade_chip.setVisible(False)
        headline_card.body_layout.addWidget(self.thesis_grade_chip)
        # Sample-count warning — fires whenever the effective evaluation dataset has fewer
        # than 100 accepted rows. RMSE on small evals is too noisy to cite.
        self.eval_sample_count_chip = QLabel("")
        self.eval_sample_count_chip.setWordWrap(True)
        self.eval_sample_count_chip.setStyleSheet(
            f"color: {COLORS.warning_fg}; background: {COLORS.warning_bg}; "
            f"border: 1px solid {COLORS.warning_fg}; border-radius: 8px; "
            f"padding: 8px 12px; font-weight: 500;"
        )
        self.eval_sample_count_chip.setVisible(False)
        headline_card.body_layout.addWidget(self.eval_sample_count_chip)
        right.addWidget(headline_card)

        details_card = _Card(
            "Details",
            "Scope, per-axis breakdown, and any saved-output paths.",
        )
        self.evaluation_pairs = _PairsWidget()
        details_card.body_layout.addWidget(self.evaluation_pairs)
        right.addWidget(details_card)

        plots_card = _Card(
            "Plots",
            "Workspace overlay, error histograms, spatial heatmaps, and ANN loss curve.",
        )
        self.results_widget = ExperimentResultsWidget()
        plots_card.body_layout.addWidget(self.results_widget)
        right.addWidget(plots_card, 1)

        # ── Side-by-Side Model Comparison card ─────────────────────────────
        # Two independent model slots. Each slot has a combo box (lists every
        # discovered ANN artifact) and an "Upload .pt…" button. The operator
        # can compare two saved artifacts, one saved + one uploaded, or two
        # uploads — whatever combination they need. The combo automatically
        # picks up new uploads on the next refresh (set_*_path archives the
        # upload under data/models/ann/uploaded_*/).
        comparison_card = _Card(
            "Side-by-Side Model Comparison (3D Error)",
            "Pick TWO ANN artifacts to compare on the currently-selected "
            "dataset. Each slot can be filled from the dropdown (any "
            "discovered artifact) or by uploading a .pt file. Both panels "
            "share a viridis colorbar so the same color = the same mm of "
            "error on each.",
        )
        # Slot A: combo + upload button.
        slot_a_row = QHBoxLayout()
        slot_a_row.setContentsMargins(0, 0, 0, 0)
        slot_a_row.setSpacing(8)
        slot_a_label = QLabel("Model A")
        slot_a_label.setMinimumWidth(60)
        slot_a_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        slot_a_row.addWidget(slot_a_label)
        self.comparison_model_a_combo = QComboBox()
        self.comparison_model_a_combo.setToolTip(
            "Pick any discovered ANN artifact as Model A, or use the upload "
            "button on the right to add an external .pt."
        )
        self.comparison_model_a_combo.currentIndexChanged.connect(
            lambda _i: self._on_comparison_combo_changed("a")
        )
        slot_a_row.addWidget(self.comparison_model_a_combo, 1)
        self.comparison_upload_a_button = QPushButton("Upload .pt…")
        self.comparison_upload_a_button.setProperty("variant", "ghost")
        self.comparison_upload_a_button.setToolTip(
            "Pick a saved ANN state_dict (.pt) or a full artifact directory "
            "to fill Model A. Bare .pt files work — the loader auto-detects "
            "the architecture and input-unit scale."
        )
        self.comparison_upload_a_button.clicked.connect(
            lambda: self._on_comparison_upload_clicked("a")
        )
        slot_a_row.addWidget(self.comparison_upload_a_button)
        comparison_card.body_layout.addLayout(slot_a_row)
        # Slot B: combo + upload button.
        slot_b_row = QHBoxLayout()
        slot_b_row.setContentsMargins(0, 0, 0, 0)
        slot_b_row.setSpacing(8)
        slot_b_label = QLabel("Model B")
        slot_b_label.setMinimumWidth(60)
        slot_b_label.setStyleSheet(f"color: {COLORS.text_secondary}; font-weight: 600;")
        slot_b_row.addWidget(slot_b_label)
        self.comparison_model_b_combo = QComboBox()
        self.comparison_model_b_combo.setToolTip(
            "Pick any discovered ANN artifact as Model B, or use the upload "
            "button on the right to add an external .pt."
        )
        self.comparison_model_b_combo.currentIndexChanged.connect(
            lambda _i: self._on_comparison_combo_changed("b")
        )
        slot_b_row.addWidget(self.comparison_model_b_combo, 1)
        self.comparison_upload_b_button = QPushButton("Upload .pt…")
        self.comparison_upload_b_button.setProperty("variant", "ghost")
        self.comparison_upload_b_button.setToolTip(
            "Pick a saved ANN state_dict (.pt) or a full artifact directory "
            "to fill Model B. Bare .pt files work — the loader auto-detects "
            "the architecture and input-unit scale."
        )
        self.comparison_upload_b_button.clicked.connect(
            lambda: self._on_comparison_upload_clicked("b")
        )
        slot_b_row.addWidget(self.comparison_upload_b_button)
        comparison_card.body_layout.addLayout(slot_b_row)
        # Internal: remember the last paths we synced into the combos so we
        # can skip redundant repopulations and avoid triggering
        # currentIndexChanged in a loop.
        self._comparison_combo_signature: tuple[str, ...] = ()
        # Row: Generate + Save buttons.
        comparison_actions_row = QHBoxLayout()
        comparison_actions_row.setContentsMargins(0, 0, 0, 0)
        comparison_actions_row.setSpacing(10)
        self.comparison_generate_button = QPushButton("Generate Side-by-Side Plot")
        self.comparison_generate_button.setProperty("role", "primary")
        self.comparison_generate_button.clicked.connect(
            self.controller.run_external_comparison
        )
        self.comparison_save_button = QPushButton("Save PNG…")
        self.comparison_save_button.setProperty("variant", "ghost")
        self.comparison_save_button.setToolTip(
            "Write the side-by-side figure to disk at 300 DPI (publication quality)."
        )
        self.comparison_save_button.clicked.connect(self._on_comparison_save_clicked)
        self.comparison_save_button.setEnabled(False)
        comparison_actions_row.addWidget(self.comparison_generate_button)
        comparison_actions_row.addWidget(self.comparison_save_button)
        comparison_actions_row.addStretch(1)
        comparison_card.body_layout.addLayout(comparison_actions_row)
        # Status + error.
        self.comparison_status_label = QLabel("")
        self.comparison_status_label.setWordWrap(True)
        self.comparison_status_label.setStyleSheet(f"color: {COLORS.text_secondary};")
        comparison_card.body_layout.addWidget(self.comparison_status_label)
        self.comparison_error_label = QLabel("")
        self.comparison_error_label.setWordWrap(True)
        self.comparison_error_label.setStyleSheet(
            "color: #dc2626; font-weight: 600;"
        )
        self.comparison_error_label.setVisible(False)
        comparison_card.body_layout.addWidget(self.comparison_error_label)
        # Matplotlib canvas — local import so the class loads even on torch-less
        # machines (the canvas itself doesn't need torch, but matplotlib's
        # Qt-Agg backend is heavy enough to keep out of the import path until
        # the tab is actually constructed).
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from matplotlib.figure import Figure

        self._mpl_FigureCanvas = FigureCanvasQTAgg
        self._mpl_NavToolbar = NavigationToolbar2QT
        self._mpl_Figure = Figure
        # Start with a placeholder figure so the canvas has fixed sizing.
        placeholder = Figure(figsize=(10, 4.5))
        ax = placeholder.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Run a comparison to see two side-by-side 3D error plots.",
            ha="center",
            va="center",
            fontsize=10,
            color="#888",
        )
        ax.set_axis_off()
        self.comparison_canvas = FigureCanvasQTAgg(placeholder)
        self.comparison_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.comparison_canvas.setMinimumHeight(380)
        comparison_card.body_layout.addWidget(self.comparison_canvas)
        # Toolbar gives mouse-wheel zoom + drag-rotate on the 3D axes.
        self.comparison_nav_toolbar = NavigationToolbar2QT(
            self.comparison_canvas, self
        )
        comparison_card.body_layout.addWidget(self.comparison_nav_toolbar)
        # Internal: track the most recent comparison_result_id we've rendered so
        # we re-render the canvas only when a new result lands (cheap idempotent
        # update() that won't repaint on every refresh tick).
        self._last_comparison_rendered_id = 0
        right.addWidget(comparison_card, 1)

    def update(self, state: ModelingViewState) -> None:
        self._sync_dataset_list(state)
        self._sync_artifact_list(state)
        self._sync_history_list(state)
        self._sync_test_dataset_combo(state)
        self.dataset_pairs.set_pairs(state.dataset_summary_pairs)
        self._sync_dataset_empty_hint(state)
        self._sync_dataset_mode_chip(state)
        self.artifact_pairs.set_pairs(state.artifact_summary_pairs)
        self.artifact_mismatch_chip.setText(state.artifact_dataset_mismatch)
        self.artifact_mismatch_chip.setVisible(bool(state.artifact_dataset_mismatch))
        self.evaluation_pairs.set_pairs(state.evaluation_summary_pairs)
        self.headline_metrics_widget.set_metrics(state.headline_metrics)
        # Copy-summary button is meaningful only when there's something to copy.
        self.copy_summary_button.setEnabled(bool(state.headline_metrics))
        self.validate_rows_button.setEnabled(bool(state.selected_dataset_path))
        if state.row_filter_status_text:
            self.row_filter_status_label.setText(state.row_filter_status_text)
            self.row_filter_status_label.setVisible(True)
        else:
            self.row_filter_status_label.setVisible(False)
        # Show exactly one of the chips: positive when thesis-grade, caveat when not,
        # nothing before any evaluation has run.
        ran_an_eval = bool(state.headline_metrics)
        self.thesis_grade_chip.setVisible(bool(ran_an_eval and state.last_eval_thesis_grade))
        self.same_session_chip.setVisible(
            bool(ran_an_eval and state.last_eval_same_session and not state.last_eval_thesis_grade)
        )
        # Always show the sample-count warning when the selected eval dataset is small,
        # regardless of whether we've run anything yet (it's a property of the dataset,
        # not the eval result). Lets operators avoid clicking Run on a too-small dataset.
        self.eval_sample_count_chip.setText(state.eval_sample_count_warning)
        self.eval_sample_count_chip.setVisible(bool(state.eval_sample_count_warning))
        self.results_widget.set_model(state.visualization_model)
        # Prefer the structured headline label when available; fall back to status_message
        # only when no contextual label has been computed yet.
        self.status_label.setText(state.headline_status_label or state.status_message)
        self.evaluate_button.setEnabled(state.can_evaluate)
        self.open_dataset_button.setEnabled(bool(state.selected_dataset_path))
        self.open_artifact_button.setEnabled(bool(state.selected_artifact_path))
        self.open_results_button.setEnabled(bool(state.last_output_path))
        self._set_checkbox(self.mike_check, state.include_mike)
        self._set_checkbox(self.camarillo_check, state.include_camarillo)
        self._set_checkbox(self.ann_check, state.include_ann)
        self._set_combo(self.scope_combo, state.evaluation_scope)
        # ── Side-by-Side Model Comparison card ────────────────────────────
        # Repopulate the slot combos with every discovered artifact. We use a
        # signature compare to skip the work when nothing has changed, which
        # also keeps the 5 Hz refresh cheap.
        self._sync_comparison_combos(state)
        # Status + error messages.
        self.comparison_status_label.setText(state.comparison_status_message)
        if state.comparison_error_message:
            self.comparison_error_label.setText(state.comparison_error_message)
            self.comparison_error_label.setVisible(True)
        else:
            self.comparison_error_label.setVisible(False)
        # Generate enabled when dataset + both slots are set and no run active.
        # Slot A back-compat: if comparison_model_a_path is empty but the main
        # artifact dropdown above has a selection, that fallback path applies.
        slot_a_set = bool(state.comparison_model_a_path) or state.artifact_details is not None
        slot_b_set = bool(state.comparison_model_b_path)
        ready_to_generate = (
            bool(state.selected_dataset_path)
            and slot_a_set
            and slot_b_set
            and not state.comparison_active
        )
        self.comparison_generate_button.setEnabled(ready_to_generate)
        self.comparison_generate_button.setText(
            "Generating…" if state.comparison_active else "Generate Side-by-Side Plot"
        )
        # Re-render the canvas only when a new comparison result lands.
        if state.comparison_result_id != self._last_comparison_rendered_id:
            self._last_comparison_rendered_id = state.comparison_result_id
            self._render_comparison_canvas()

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

    def _sync_comparison_combos(self, state: ModelingViewState) -> None:
        """Repopulate the Slot A / Slot B combos with every discovered artifact.

        Each combo has an empty first entry ("— pick model —"), followed by
        every artifact's display name. The path is stored in the item's
        ``Qt.UserRole`` so the change handler can pull it out. Signature
        compare prevents redundant rebuilds on every 5 Hz refresh tick — the
        combos only rebuild when the catalog or selected paths change.
        """
        signature = (
            tuple((a.artifact_name, str(a.path)) for a in state.artifacts),
            state.comparison_model_a_path,
            state.comparison_model_b_path,
        )
        if signature == self._comparison_combo_signature:
            return
        self._comparison_combo_signature = signature
        for combo, target_path in (
            (self.comparison_model_a_combo, state.comparison_model_a_path),
            (self.comparison_model_b_combo, state.comparison_model_b_path),
        ):
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItem("— pick model —", "")
                selected_index = 0
                for i, artifact in enumerate(state.artifacts, start=1):
                    combo.addItem(artifact.artifact_name, str(artifact.path))
                    if str(artifact.path) == target_path:
                        selected_index = i
                # When the slot points at a freshly-archived .pt that hasn't
                # surfaced in the catalog yet (e.g., the operator just clicked
                # Upload but the next refresh's catalog re-scan is pending),
                # add an extra entry so the slot's current path is still
                # represented in the combo.
                if target_path and selected_index == 0:
                    display = Path(target_path).name
                    if display == "model.pt":
                        display = Path(target_path).parent.name
                    combo.addItem(display, target_path)
                    selected_index = combo.count() - 1
                combo.setCurrentIndex(selected_index)

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

    def _sync_history_list(self, state: ModelingViewState) -> None:
        """Render the past-evaluations list (newest first) for the selected dataset."""
        target_signature = tuple(
            (str(e.output_dir), e.timestamp_utc, e.best_label, e.best_rmse_mm)
            for e in state.past_evaluations
        )
        current_signature = tuple(
            (
                self.history_list.item(i).data(Qt.UserRole),
                self.history_list.item(i).data(Qt.UserRole + 1) or "",
                self.history_list.item(i).data(Qt.UserRole + 2) or "",
                self.history_list.item(i).data(Qt.UserRole + 3),
            )
            for i in range(self.history_list.count())
        )
        if current_signature != target_signature:
            with QSignalBlocker(self.history_list):
                self.history_list.clear()
                for past in state.past_evaluations:
                    rmse_text = (
                        f"{float(past.best_rmse_mm):.2f} mm" if past.best_rmse_mm is not None else "n/a"
                    )
                    label = (
                        f"{past.timestamp_utc}  •  best: {past.best_label} {rmse_text}"
                        f"  •  {past.selected_sample_count} samples"
                    )
                    item = QListWidgetItem(label)
                    item.setData(Qt.UserRole, str(past.output_dir))
                    item.setData(Qt.UserRole + 1, past.timestamp_utc)
                    item.setData(Qt.UserRole + 2, past.best_label)
                    item.setData(Qt.UserRole + 3, past.best_rmse_mm)
                    self.history_list.addItem(item)
        self.history_hint.setVisible(self.history_list.count() == 0)

    def _on_history_item_double_clicked(self, item: QListWidgetItem) -> None:
        path_str = str(item.data(Qt.UserRole) or "")
        if path_str:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))

    def _on_validate_rows_clicked(self) -> None:
        self.controller.validate_rows_for_eval()
        self.update(self.controller.refresh())

    def _copy_summary_to_clipboard(self) -> None:
        """Push a plain-text version of the last evaluation to the system clipboard."""
        state = self.controller.refresh()
        if not state.headline_metrics:
            self.status_label.setText("Nothing to copy yet — run a comparison first.")
            return
        lines: list[str] = []
        lines.append(f"Modeling comparison — {state.selected_dataset_path}")
        if state.selected_test_dataset_path:
            lines.append(f"Test dataset (override): {state.selected_test_dataset_path}")
        if state.last_eval_thesis_grade:
            lines.append("Eval: thesis-grade (separate test acquisition)")
        elif state.last_eval_same_session:
            lines.append("Eval: same-session caveat applies — not thesis-grade")
        for metric in state.headline_metrics:
            if metric.rmse_mm is not None:
                lines.append(f"  {metric.label}: {metric.rmse_mm:.3f} mm RMSE")
            else:
                lines.append(f"  {metric.label}: {metric.status} ({metric.reason or 'unavailable'})")
        if state.eval_sample_count_warning:
            lines.append(state.eval_sample_count_warning)
        text = "\n".join(lines)
        # Cheap clipboard access — works headless when no real Qt app is shown.
        try:
            from PySide6.QtGui import QGuiApplication

            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
                self.status_label.setText("Copied summary to clipboard.")
                return
        except Exception:
            pass
        # Fall back to status label echo.
        self.status_label.setText(text)

    def _sync_dataset_empty_hint(self, state: ModelingViewState) -> None:
        self.dataset_empty_hint.setVisible(len(state.datasets) == 0)

    def _sync_dataset_mode_chip(self, state: ModelingViewState) -> None:
        """Show the selected dataset's mode as a colored chip on the dataset card."""
        selected = next(
            (d for d in state.datasets if str(d.path) == state.selected_dataset_path),
            None,
        )
        if selected is None:
            self.dataset_mode_chip.setVisible(False)
            return
        mode = str(selected.dataset_mode or "").strip().lower()
        # angular_test_mesh = ideal eval target → green; everything else neutral.
        if mode == "angular_test_mesh":
            text = "Wolfe angular test mesh — recommended Test Dataset for thesis-grade eval"
            bg, fg, border = "#064e3b", "#86efac", "#16a34a"
        elif mode == "workspace_coverage":
            text = "Workspace coverage — training/eval dataset"
            bg, fg, border = "#1e293b", "#cbd5e1", "#475569"
        elif mode == "hysteresis_path_dependence":
            text = "Hysteresis / path dependence — diagnostic dataset"
            bg, fg, border = "#3b1d0c", "#fdba74", "#c2410c"
        elif mode == "repeatability_linked":
            text = "Repeatability linked — startup-block dataset"
            bg, fg, border = "#172554", "#bfdbfe", "#1d4ed8"
        else:
            text = f"Dataset mode: {mode or 'unknown'}"
            bg, fg, border = "#1e293b", "#cbd5e1", "#475569"
        self.dataset_mode_chip.setText(text)
        self.dataset_mode_chip.setStyleSheet(
            f"color: {fg}; background: {bg}; border: 1px solid {border}; "
            f"border-radius: 6px; padding: 6px 10px; font-weight: 500;"
        )
        self.dataset_mode_chip.setVisible(True)

    def _sync_test_dataset_combo(self, state: ModelingViewState) -> None:
        """Mirror the available datasets into the test-dataset picker."""
        target_paths = ["(same as training)"] + [str(d.path) for d in state.datasets]
        current_paths = [self.test_dataset_combo.itemData(i) for i in range(self.test_dataset_combo.count())]
        target_data = [""] + [str(d.path) for d in state.datasets]
        if current_paths != target_data:
            with QSignalBlocker(self.test_dataset_combo):
                self.test_dataset_combo.clear()
                self.test_dataset_combo.addItem("(same as training)", "")
                for dataset in state.datasets:
                    label = f"{dataset.run_name} [{dataset.dataset_mode}]"
                    self.test_dataset_combo.addItem(label, str(dataset.path))
        # Select the controller's current choice.
        with QSignalBlocker(self.test_dataset_combo):
            target = state.selected_test_dataset_path or ""
            for index in range(self.test_dataset_combo.count()):
                if self.test_dataset_combo.itemData(index) == target:
                    self.test_dataset_combo.setCurrentIndex(index)
                    break

    def _on_test_dataset_changed(self, _index: int) -> None:
        data = self.test_dataset_combo.currentData()
        self.controller.set_test_dataset_path(str(data or ""))

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

    def _open_ann_training_from_modeling(self) -> None:
        """Hand the currently-selected dataset path off to the ANN training popout."""
        if self._ann_training_opener is None:
            self.status_label.setText(
                "ANN training popout is not available in this context."
            )
            return
        state = self.controller.refresh()
        if not state.selected_dataset_path:
            self.status_label.setText("Select a modeling dataset before opening ANN training.")
            return
        self._ann_training_opener(state.selected_dataset_path)

    # ── External Model Comparison handlers ────────────────────────────────
    def _on_comparison_upload_clicked(self, slot: str = "b") -> None:
        """Open a file picker for the given comparison slot ('a' or 'b').

        Allows both .pt files and full artifact directories. Qt's
        QFileDialog handles one type at a time so we default to ``.pt`` —
        operators with a folder can still type the directory path manually,
        or pick from the slot's combo box if it's already discovered.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose Model {slot.upper()} (.pt file)",
            "",
            "PyTorch model (*.pt);;All files (*)",
        )
        if not path:
            return
        if slot == "a":
            self.controller.set_comparison_model_a_path(path)
        else:
            self.controller.set_comparison_model_b_path(path)
        self.update(self.controller.refresh())

    def _on_comparison_combo_changed(self, slot: str) -> None:
        """Combo selection changed — pull the path stored in the item's UserRole."""
        combo = (
            self.comparison_model_a_combo
            if slot == "a"
            else self.comparison_model_b_combo
        )
        index = combo.currentIndex()
        path = combo.itemData(index, Qt.UserRole) if index >= 0 else ""
        path = str(path or "")
        if slot == "a":
            self.controller.set_comparison_model_a_path(path)
        else:
            self.controller.set_comparison_model_b_path(path)
        self.update(self.controller.refresh())

    def _on_comparison_save_clicked(self) -> None:
        """Save the rendered comparison to a PNG via a save-as dialog."""
        result = self.controller.get_last_comparison_result()
        if result is None:
            self.comparison_error_label.setText(
                "Run a comparison first — nothing to save yet."
            )
            self.comparison_error_label.setVisible(True)
            return
        default_name = (
            f"{result.dataset_run_name}_side_by_side_comparison.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Side-by-Side Comparison",
            default_name,
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        saved = self.controller.save_last_comparison_png(Path(path))
        if saved is not None:
            self.comparison_error_label.setVisible(False)
        self.update(self.controller.refresh())

    def _render_comparison_canvas(self) -> None:
        """Rebuild the embedded matplotlib figure from the latest comparison result.

        Called only when ``state.comparison_result_id`` increments (a fresh
        result landed). Avoids redrawing on every 5 Hz refresh tick.
        """
        result = self.controller.get_last_comparison_result()
        if result is None:
            return
        # Lazy-import the figure builder so the tab loads without torch present.
        from continuum_robot.modeling.model_comparison import build_comparison_figure

        figure = build_comparison_figure(result)
        # Swap the canvas's figure in place. matplotlib's Qt-Agg canvas supports
        # ``.figure = new_figure`` then ``draw_idle()`` for live updates.
        self.comparison_canvas.figure = figure
        figure.set_canvas(self.comparison_canvas)
        # ── Link the two 3D axes so they rotate together ────────────────────
        # Pull the two 3D axes from the figure. matplotlib creates them in
        # subplot-add order, so the first two ``name == '3d'`` axes are Model A
        # (left) and Model B (right). The colorbar axis is 2D, so the filter is
        # essential.
        three_d_axes = [ax for ax in figure.axes if ax.name == "3d"]
        if len(three_d_axes) == 2:
            ax_a, ax_b = three_d_axes
            # Clear any prior sync hooks so we don't accumulate handlers across
            # renders (the old canvas would have its own cids we'd otherwise leak).
            for attr in ("_comparison_sync_cid", "_comparison_zoom_cid"):
                prior_cid = getattr(self, attr, None)
                if prior_cid is not None:
                    try:
                        self.comparison_canvas.mpl_disconnect(prior_cid)
                    except Exception:
                        pass
                setattr(self, attr, None)
            sync_state = {"in_progress": False}

            def _copy_view(src, dst):
                """Mirror src's full 3D view (elev, azim, roll, x/y/zlim) to dst."""
                roll = getattr(src, "roll", None)
                if roll is not None:
                    try:
                        dst.view_init(elev=src.elev, azim=src.azim, roll=roll)
                    except TypeError:
                        # Older matplotlib without ``roll`` kwarg.
                        dst.view_init(elev=src.elev, azim=src.azim)
                else:
                    dst.view_init(elev=src.elev, azim=src.azim)
                # Also mirror zoom (axis limits) so scroll-wheel zoom syncs too.
                try:
                    dst.set_xlim(src.get_xlim())
                    dst.set_ylim(src.get_ylim())
                    dst.set_zlim(src.get_zlim())
                except Exception:
                    pass

            def _on_motion(event, ax_a=ax_a, ax_b=ax_b, state=sync_state, canvas=self.comparison_canvas):
                """Mirror rotation while one of the 3D axes is being dragged.

                matplotlib's own Axes3D._on_move uses ``self.button_pressed`` to
                detect an active drag — ``event.button`` is None during
                motion_notify because the press happened in a separate event.
                Reading the axis attribute matches matplotlib's contract so the
                sync fires on every real drag (button 1 OR button 3 — orbit OR
                pan-rotate).
                """
                if state["in_progress"]:
                    return
                a_dragging = getattr(ax_a, "button_pressed", None) is not None
                b_dragging = getattr(ax_b, "button_pressed", None) is not None
                if a_dragging and not b_dragging:
                    src, dst = ax_a, ax_b
                elif b_dragging and not a_dragging:
                    src, dst = ax_b, ax_a
                else:
                    return
                # Cheap short-circuit when angles already match.
                same_view = (
                    dst.elev == src.elev
                    and dst.azim == src.azim
                    and getattr(dst, "roll", 0.0) == getattr(src, "roll", 0.0)
                )
                if same_view:
                    return
                state["in_progress"] = True
                try:
                    _copy_view(src, dst)
                    canvas.draw_idle()
                finally:
                    state["in_progress"] = False

            def _on_scroll(event, ax_a=ax_a, ax_b=ax_b, state=sync_state, canvas=self.comparison_canvas):
                """Mirror zoom (axis limits) when the scroll wheel changes one panel."""
                if state["in_progress"]:
                    return
                if event.inaxes is ax_a:
                    src, dst = ax_a, ax_b
                elif event.inaxes is ax_b:
                    src, dst = ax_b, ax_a
                else:
                    return
                state["in_progress"] = True
                try:
                    # Scroll-zoom updates lims; mirror them. (View angles are
                    # untouched by scroll, so this is purely an xlim/ylim/zlim copy.)
                    dst.set_xlim(src.get_xlim())
                    dst.set_ylim(src.get_ylim())
                    dst.set_zlim(src.get_zlim())
                    canvas.draw_idle()
                finally:
                    state["in_progress"] = False

            self._comparison_sync_cid = self.comparison_canvas.mpl_connect(
                "motion_notify_event", _on_motion
            )
            self._comparison_zoom_cid = self.comparison_canvas.mpl_connect(
                "scroll_event", _on_scroll
            )
            # Force an initial alignment so both panels share the same view
            # before the operator touches anything.
            _copy_view(ax_a, ax_b)
        self.comparison_canvas.draw_idle()
        self.comparison_save_button.setEnabled(True)

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


# Color thresholds — matched to surgical-accuracy convention (Wolfe §3.2.4 p86
# cites sub-millimeter as the surgical threshold). The amber tier covers Wolfe's
# 2.24mm result; red is everything worse than that.
_HEADLINE_THRESHOLD_GREEN_MM = 1.0
_HEADLINE_THRESHOLD_AMBER_MM = 3.0


def _headline_color_for_rmse(rmse_mm: float | None) -> tuple[str, str]:
    """Return ``(border_color, accent_color)`` for the headline tile."""
    if rmse_mm is None:
        return (COLORS.text_muted, COLORS.text_muted)
    if rmse_mm <= _HEADLINE_THRESHOLD_GREEN_MM:
        return ("#16a34a", "#86efac")  # green
    if rmse_mm <= _HEADLINE_THRESHOLD_AMBER_MM:
        return ("#d97706", "#fcd34d")  # amber
    return ("#dc2626", "#fca5a5")  # red


class _HeadlineMetricsWidget(QWidget):
    """Big color-coded per-model RMSE tiles.

    Each completed model gets a tile: small label up top, big mm number in the
    middle, threshold-coded border + accent. Models that ran but came back
    unavailable get a muted tile with the reason.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._signature: tuple = ()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(10)
        self._empty_label = QLabel("Run a comparison to see per-model RMSE.")
        self._empty_label.setStyleSheet(f"color: {COLORS.text_muted}; font-style: italic;")
        self._layout.addWidget(self._empty_label)

    def set_metrics(self, metrics: list[HeadlineMetric]) -> None:
        signature = tuple(
            (m.label, m.model_key, m.rmse_mm, m.status, m.reason) for m in metrics
        )
        if signature == self._signature:
            return
        self._signature = signature
        # Clear previous tiles.
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not metrics:
            self._empty_label = QLabel("Run a comparison to see per-model RMSE.")
            self._empty_label.setStyleSheet(
                f"color: {COLORS.text_muted}; font-style: italic;"
            )
            self._layout.addWidget(self._empty_label)
            return
        for metric in metrics:
            tile = self._build_tile(metric)
            self._layout.addWidget(tile, 1)

    @staticmethod
    def _build_tile(metric: HeadlineMetric) -> QFrame:
        border_color, accent_color = _headline_color_for_rmse(metric.rmse_mm)
        tile = QFrame()
        tile.setStyleSheet(
            f"background: {COLORS.surface_bg}; border: 2px solid {border_color}; "
            f"border-radius: 12px;"
        )
        # Tooltip: full per-axis + mean/max breakdown without taking screen space.
        tooltip_lines: list[str] = [f"{metric.label}"]
        if metric.rmse_mm is not None:
            tooltip_lines.append(f"Position RMSE: {metric.rmse_mm:.3f} mm")
        if metric.per_axis_rmse_mm is not None:
            x, y, z = metric.per_axis_rmse_mm
            tooltip_lines.append(f"Per-axis RMSE — X {x:.3f}  Y {y:.3f}  Z {z:.3f} mm")
        if metric.mean_position_error_mm is not None:
            tooltip_lines.append(f"Mean position error: {metric.mean_position_error_mm:.3f} mm")
        if metric.max_position_error_mm is not None:
            tooltip_lines.append(f"Max position error: {metric.max_position_error_mm:.3f} mm")
        if metric.status != "completed" or metric.rmse_mm is None:
            tooltip_lines.append(f"Status: {metric.status}")
            if metric.reason:
                tooltip_lines.append(f"Reason: {metric.reason}")
        tile.setToolTip("\n".join(tooltip_lines))
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(2)
        label = QLabel(metric.label)
        label.setStyleSheet(
            f"color: {COLORS.text_muted}; font-weight: 700; font-size: 11px; "
            f"letter-spacing: 0.6px;"
        )
        layout.addWidget(label)
        if metric.status == "completed" and metric.rmse_mm is not None:
            value = QLabel(f"{metric.rmse_mm:.2f}")
            value.setStyleSheet(
                f"color: {accent_color}; font-weight: 800; font-size: 30px;"
            )
            layout.addWidget(value)
            unit = QLabel("mm RMSE")
            unit.setStyleSheet(f"color: {COLORS.text_secondary}; font-size: 11px;")
            layout.addWidget(unit)
        else:
            value = QLabel("—")
            value.setStyleSheet(
                f"color: {COLORS.text_muted}; font-weight: 600; font-size: 30px;"
            )
            layout.addWidget(value)
            reason = QLabel(metric.reason or metric.status or "unavailable")
            reason.setStyleSheet(f"color: {COLORS.text_muted}; font-size: 10px;")
            reason.setWordWrap(True)
            layout.addWidget(reason)
        layout.addStretch(1)
        return tile
