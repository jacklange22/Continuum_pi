"""Shared GUI theme helpers."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeColors:
    workspace_bg: str = "#0b1220"
    surface_bg: str = "#111827"
    surface_alt_bg: str = "#172033"
    surface_border: str = "#334155"
    input_bg: str = "#0f172a"
    input_border: str = "#475569"
    button_bg: str = "#1f2937"
    button_border: str = "#475569"
    button_primary_bg: str = "#1d4ed8"
    button_primary_fg: str = "#eff6ff"
    button_primary_border: str = "#3b82f6"
    button_danger_bg: str = "#7f1d1d"
    button_danger_fg: str = "#fee2e2"
    button_danger_border: str = "#ef4444"
    text_primary: str = "#e5eef9"
    text_secondary: str = "#cbd5e1"
    text_muted: str = "#94a3b8"
    text_subtle: str = "#7f8ea3"
    status_bg: str = "#22314a"
    status_fg: str = "#e5eef9"
    tab_bg: str = "#243244"
    tab_selected_bg: str = "#111827"
    tab_selected_fg: str = "#f8fafc"
    overlay_bg: str = "#0f172a"
    overlay_border: str = "#334155"
    chart_grid: str = "#334155"
    selection_bg: str = "#1d4ed8"
    selection_fg: str = "#eff6ff"

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
            background: #14532d;
            border-color: #22c55e;
            color: #dcfce7;
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
