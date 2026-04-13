"""Shared GUI theme helpers."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeColors:
    workspace_bg: str = "#14181e"
    surface_bg: str = "#1a2028"
    surface_alt_bg: str = "#202834"
    surface_border: str = "#465262"
    input_bg: str = "#161b22"
    input_border: str = "#5b6878"
    button_bg: str = "#26303b"
    button_border: str = "#5d6979"
    button_primary_bg: str = "#495f75"
    button_primary_fg: str = "#f7fafc"
    button_primary_border: str = "#829ab1"
    button_danger_bg: str = "#6a3436"
    button_danger_fg: str = "#fde8e9"
    button_danger_border: str = "#b97979"
    text_primary: str = "#f3f6f8"
    text_secondary: str = "#d7dee6"
    text_muted: str = "#aab4c0"
    text_subtle: str = "#7c8794"
    status_bg: str = "#2a3644"
    status_fg: str = "#f3f6f8"
    tab_bg: str = "#293340"
    tab_selected_bg: str = "#1a2028"
    tab_selected_fg: str = "#f7fafc"
    overlay_bg: str = "#141a21"
    overlay_border: str = "#465262"
    chart_grid: str = "#4b5868"
    selection_bg: str = "#597289"
    selection_fg: str = "#f7fafc"
    success_bg: str = "#2f4738"
    success_fg: str = "#e9f6ed"
    warning_bg: str = "#5f4d27"
    warning_fg: str = "#fff3cf"
    error_bg: str = "#633033"
    error_fg: str = "#fde8e9"
    info_bg: str = "#304556"
    info_fg: str = "#eaf2f8"
    scene_truth: str = "#7ea4c4"
    scene_measurement: str = "#82a596"
    scene_live_0a: str = "#7aa49c"
    scene_live_0b: str = "#b89564"
    scene_tip: str = "#b56b6d"
    scene_residual: str = "#c77a72"
    scene_grid: str = "#495462"
    scene_axis_x: str = "#7e9fbe"
    scene_axis_y: str = "#7da088"
    scene_axis_z: str = "#b6786f"

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
        QToolTip {{
            background: {COLORS.surface_alt_bg};
            color: {COLORS.text_primary};
            border: 1px solid {COLORS.surface_border};
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
