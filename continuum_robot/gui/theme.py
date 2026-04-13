"""Shared GUI theme helpers."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeColors:
    workspace_bg: str = "#0d1217"
    surface_bg: str = "#141b22"
    surface_alt_bg: str = "#19222b"
    surface_border: str = "#384859"
    input_bg: str = "#10161d"
    input_border: str = "#506172"
    button_bg: str = "#1b242d"
    button_border: str = "#5a6b7c"
    button_primary_bg: str = "#334759"
    button_primary_fg: str = "#f4f7fa"
    button_primary_border: str = "#7f9cb5"
    button_danger_bg: str = "#5a3136"
    button_danger_fg: str = "#f9e8ea"
    button_danger_border: str = "#956569"
    text_primary: str = "#f5f7fa"
    text_secondary: str = "#dce4eb"
    text_muted: str = "#b1bcc7"
    text_subtle: str = "#8492a0"
    status_bg: str = "#202b35"
    status_fg: str = "#f5f7fa"
    tab_bg: str = "#1a232c"
    tab_selected_bg: str = "#141b22"
    tab_selected_fg: str = "#f4f7fa"
    overlay_bg: str = "#0c1218"
    overlay_border: str = "#384859"
    chart_grid: str = "#5d6c7c"
    selection_bg: str = "#58718a"
    selection_fg: str = "#f4f7fa"
    success_bg: str = "#273830"
    success_fg: str = "#edf6f0"
    warning_bg: str = "#584924"
    warning_fg: str = "#fff1c8"
    error_bg: str = "#552d32"
    error_fg: str = "#fbe8ea"
    info_bg: str = "#233847"
    info_fg: str = "#e8f0f7"
    scene_truth: str = "#88a8c4"
    scene_measurement: str = "#93ada0"
    scene_live_0a: str = "#7faaa3"
    scene_live_0b: str = "#b99b70"
    scene_tip: str = "#bb7871"
    scene_residual: str = "#c9857a"
    scene_grid: str = "#66727f"
    scene_axis_x: str = "#7da8cd"
    scene_axis_y: str = "#85aa90"
    scene_axis_z: str = "#bb8278"

    @property
    def accent(self) -> str:
        """Backward-compatible accent token for focused/highlighted UI elements."""
        return self.button_primary_border


COLORS = ThemeColors()


def qcolor(hex_color: str, *, alpha: int | None = None) -> QColor:
    color = QColor(hex_color)
    if alpha is not None:
        color.setAlpha(int(alpha))
    return color


def apply_dark_theme(app: QApplication | None) -> None:
    """Apply the shared app-wide dark palette once."""
    if app is None:
        return
    if bool(app.property("continuum_dark_theme_applied")):
        return

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, qcolor(COLORS.workspace_bg))
    palette.setColor(QPalette.WindowText, qcolor(COLORS.text_primary))
    palette.setColor(QPalette.Base, qcolor(COLORS.input_bg))
    palette.setColor(QPalette.AlternateBase, qcolor(COLORS.surface_alt_bg))
    palette.setColor(QPalette.ToolTipBase, qcolor(COLORS.surface_alt_bg))
    palette.setColor(QPalette.ToolTipText, qcolor(COLORS.text_primary))
    palette.setColor(QPalette.Text, qcolor(COLORS.text_primary))
    palette.setColor(QPalette.Button, qcolor(COLORS.button_bg))
    palette.setColor(QPalette.ButtonText, qcolor(COLORS.text_primary))
    palette.setColor(QPalette.BrightText, qcolor("#ffffff"))
    palette.setColor(QPalette.Highlight, qcolor(COLORS.selection_bg))
    palette.setColor(QPalette.HighlightedText, qcolor(COLORS.selection_fg))
    palette.setColor(QPalette.PlaceholderText, qcolor(COLORS.text_subtle))
    app.setPalette(palette)
    app.setStyleSheet(
        f"""
        QWidget {{
            color: {COLORS.text_primary};
        }}
        QFrame {{
            color: {COLORS.text_primary};
        }}
        QToolTip {{
            background: {COLORS.surface_alt_bg};
            color: {COLORS.text_primary};
            border: 1px solid {COLORS.surface_border};
        }}
        QMenu {{
            background: {COLORS.surface_bg};
            color: {COLORS.text_primary};
            border: 1px solid {COLORS.surface_border};
        }}
        QMenu::item:selected {{
            background: {COLORS.selection_bg};
            color: {COLORS.selection_fg};
        }}
        QHeaderView::section {{
            background: {COLORS.surface_bg};
            color: {COLORS.text_secondary};
            border: 1px solid {COLORS.surface_border};
            padding: 6px 8px;
            font-weight: 700;
        }}
        QGroupBox {{
            border: 1px solid {COLORS.surface_border};
            border-radius: 14px;
            margin-top: 16px;
            padding-top: 10px;
            background: {COLORS.surface_alt_bg};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {COLORS.text_primary};
            font-weight: 700;
        }}
        QTableView, QTableWidget, QListView, QTreeView {{
            background: {COLORS.input_bg};
            alternate-background-color: {COLORS.surface_alt_bg};
            gridline-color: {COLORS.surface_border};
            color: {COLORS.text_primary};
            selection-background-color: {COLORS.selection_bg};
            selection-color: {COLORS.selection_fg};
            border: 1px solid {COLORS.input_border};
            border-radius: 10px;
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {COLORS.input_bg};
            color: {COLORS.text_primary};
            border: 1px solid {COLORS.input_border};
            border-radius: 10px;
            padding: 4px 8px;
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {COLORS.button_primary_border};
        }}
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
            background: {COLORS.button_bg};
            border-left: 1px solid {COLORS.input_border};
            width: 16px;
        }}
        QComboBox QAbstractItemView {{
            background: {COLORS.surface_bg};
            color: {COLORS.text_primary};
            selection-background-color: {COLORS.selection_bg};
            selection-color: {COLORS.selection_fg};
        }}
        QTabWidget::pane {{
            border: 1px solid {COLORS.surface_border};
            background: {COLORS.surface_bg};
            border-radius: 12px;
        }}
        QTabBar::tab {{
            background: {COLORS.tab_bg};
            color: {COLORS.text_secondary};
            padding: 8px 14px;
            border: 1px solid {COLORS.surface_border};
            border-bottom: none;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            margin-right: 4px;
        }}
        QTabBar::tab:selected {{
            background: {COLORS.tab_selected_bg};
            color: {COLORS.tab_selected_fg};
        }}
        QPushButton {{
            min-height: 34px;
            padding: 0 12px;
            border: 1px solid {COLORS.button_border};
            border-radius: 10px;
            background: {COLORS.button_bg};
            color: {COLORS.text_primary};
            font-weight: 600;
        }}
        QPushButton:hover {{
            border-color: {COLORS.button_primary_border};
        }}
        QPushButton:pressed {{
            background: {COLORS.surface_alt_bg};
        }}
        QPushButton:disabled {{
            color: {COLORS.text_subtle};
            border-color: {COLORS.surface_border};
        }}
        QSplitter::handle {{
            background: {COLORS.workspace_bg};
        }}
        QScrollBar:vertical {{
            background: {COLORS.surface_bg};
            width: 12px;
            margin: 2px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS.surface_border};
            min-height: 24px;
            border-radius: 6px;
        }}
        QScrollBar:horizontal {{
            background: {COLORS.surface_bg};
            height: 12px;
            margin: 2px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLORS.surface_border};
            min-width: 24px;
            border-radius: 6px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            width: 0px;
            height: 0px;
            border: none;
            background: transparent;
        }}
        """
    )
    app.setProperty("continuum_dark_theme_applied", True)


def chip_stylesheet(*, background: str, foreground: str) -> str:
    return (
        f"padding: 5px 12px; border-radius: 999px; "
        f"background: {background}; color: {foreground}; font-weight: 700;"
    )


def semantic_chip_colors(kind: str) -> tuple[str, str]:
    return {
        "ready": (COLORS.success_bg, COLORS.success_fg),
        "ok": (COLORS.success_bg, COLORS.success_fg),
        "warning": (COLORS.warning_bg, COLORS.warning_fg),
        "blocked": (COLORS.error_bg, COLORS.error_fg),
        "error": (COLORS.error_bg, COLORS.error_fg),
        "info": (COLORS.info_bg, COLORS.info_fg),
        "accent": (COLORS.selection_bg, COLORS.selection_fg),
        "neutral": (COLORS.status_bg, COLORS.text_secondary),
    }.get(kind, (COLORS.status_bg, COLORS.text_secondary))


def chart_palette() -> list[str]:
    return [
        COLORS.scene_live_0a,
        COLORS.scene_truth,
        COLORS.scene_tip,
        COLORS.scene_live_0b,
        COLORS.scene_measurement,
        COLORS.selection_bg,
        COLORS.info_fg,
        COLORS.text_muted,
    ]


def grouped_workspace_stylesheet(*, object_name: str, input_selectors: list[str], extra_rules: str = "") -> str:
    inputs = ",\n".join(f"QWidget#{object_name} {selector}" for selector in input_selectors)
    input_block = (
        f"""
        {inputs} {{
            border: 1px solid {COLORS.input_border};
            border-radius: 10px;
            background: {COLORS.input_bg};
            color: {COLORS.text_primary};
        }}
        """
        if inputs
        else ""
    )
    return f"""
        QWidget#{object_name} {{
            background: {COLORS.workspace_bg};
            color: {COLORS.text_primary};
        }}
        QWidget#{object_name} QGroupBox {{
            border: 1px solid {COLORS.surface_border};
            border-radius: 16px;
            margin-top: 16px;
            padding-top: 10px;
            background: {COLORS.surface_alt_bg};
        }}
        QWidget#{object_name} QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {COLORS.text_primary};
            font-weight: 700;
        }}
        QWidget#{object_name} QLabel[role="title"] {{
            font-size: 24px;
            font-weight: 700;
            color: {COLORS.text_primary};
        }}
        QWidget#{object_name} QLabel[role="hint"] {{
            color: {COLORS.text_muted};
        }}
        QWidget#{object_name} QLabel[role="status"] {{
            padding: 8px 10px;
            border-radius: 8px;
            background: {COLORS.status_bg};
            color: {COLORS.status_fg};
            font-weight: 700;
        }}
        QWidget#{object_name} QPushButton {{
            min-height: 36px;
            padding: 0 14px;
            border: 1px solid {COLORS.button_border};
            border-radius: 10px;
            background: {COLORS.button_bg};
            color: {COLORS.text_primary};
            font-weight: 600;
        }}
        QWidget#{object_name} QPushButton[role="primary"] {{
            background: {COLORS.button_primary_bg};
            border-color: {COLORS.button_primary_border};
            color: {COLORS.button_primary_fg};
        }}
        QWidget#{object_name} QPushButton[role="danger"] {{
            background: {COLORS.button_danger_bg};
            border-color: {COLORS.button_danger_border};
            color: {COLORS.button_danger_fg};
        }}
        QWidget#{object_name} QPushButton[variant="ghost"] {{
            background: transparent;
            color: {COLORS.text_secondary};
        }}
        QWidget#{object_name} QPushButton:checked {{
            background: {COLORS.success_bg};
            border-color: {COLORS.button_primary_border};
            color: {COLORS.success_fg};
        }}
        {input_block}
        {extra_rules}
    """


def experiment_shell_stylesheet(*, object_name: str) -> str:
    return f"""
        QWidget#{object_name} {{
            background: {COLORS.workspace_bg};
            color: {COLORS.text_primary};
        }}
        QWidget#{object_name} QLabel[role="page-title"] {{
            font-size: 20px;
            font-weight: 700;
            color: {COLORS.text_primary};
        }}
        QWidget#{object_name} QLabel[role="section-title"] {{
            font-size: 15px;
            font-weight: 700;
            color: {COLORS.text_primary};
        }}
        QWidget#{object_name} QLabel[role="body"] {{
            color: {COLORS.text_secondary};
        }}
        QWidget#{object_name} QLabel[role="muted"] {{
            color: {COLORS.text_muted};
        }}
        QWidget#{object_name} QLabel[role="chip"] {{
            {chip_stylesheet(background=COLORS.status_bg, foreground=COLORS.status_fg)}
        }}
        QWidget#{object_name} QFrame[role="card"] {{
            background: {COLORS.surface_alt_bg};
            border: 1px solid {COLORS.surface_border};
            border-radius: 16px;
        }}
        QWidget#{object_name} QComboBox {{
            min-height: 36px;
            border: 1px solid {COLORS.input_border};
            border-radius: 12px;
            background: {COLORS.input_bg};
            color: {COLORS.text_primary};
            padding: 4px 10px;
            font-size: 14px;
            font-weight: 600;
        }}
        QWidget#{object_name} QPushButton {{
            min-height: 34px;
            padding: 0 12px;
            border: 1px solid {COLORS.button_border};
            border-radius: 10px;
            background: {COLORS.button_bg};
            color: {COLORS.text_primary};
            font-weight: 600;
        }}
        QWidget#{object_name} QPushButton[variant="ghost"] {{
            background: transparent;
            color: {COLORS.text_secondary};
        }}
        QWidget#{object_name} QLineEdit,
        QWidget#{object_name} QPlainTextEdit,
        QWidget#{object_name} QTextEdit,
        QWidget#{object_name} QListWidget,
        QWidget#{object_name} QSpinBox,
        QWidget#{object_name} QDoubleSpinBox,
        QWidget#{object_name} QCheckBox,
        QWidget#{object_name} QProgressBar {{
            border: 1px solid {COLORS.input_border};
            border-radius: 10px;
            background: {COLORS.input_bg};
            color: {COLORS.text_primary};
        }}
    """
