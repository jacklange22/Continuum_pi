"""Shared thesis/report plotting helpers for saved experiment figures."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable


FIGURE_QUALITY_DPI = {
    "low": 120,
    "medium": 200,
    "production": 300,
}

DEFAULT_FIGURE_QUALITY = "production"
_configured_figure_quality = DEFAULT_FIGURE_QUALITY


SEMANTIC_COLORS = {
    "measured": "#2563eb",
    "target": "#dc2626",
    "reference": "#111827",
    "model": "#0f766e",
    "prediction": "#7c3aed",
    "fit": "#d97706",
    "threshold": "#64748b",
    "accepted": "#0f766e",
    "rejected": "#dc2626",
    "baseline": "#111827",
    "segment_a": "#2563eb",
    "segment_b": "#d97706",
    "neutral": "#64748b",
    "grid": "#e2e8f0",
    "axis": "#334155",
    "text": "#111827",
}


FIGURE_SIZES = {
    "wide": (7.2, 4.5),
    "square": (5.4, 5.0),
    "tall": (6.0, 7.0),
    "two_panel": (7.2, 5.8),
    "three_panel": (7.2, 7.2),
    "thesis_3d": (8.0, 7.2),
}


def set_figure_output_quality(quality: str | None) -> str:
    """Set process-local default PNG quality for subsequent saved figures."""
    global _configured_figure_quality
    normalized = normalize_figure_quality(quality)
    _configured_figure_quality = normalized
    return normalized


def normalize_figure_quality(quality: str | None) -> str:
    normalized = str(quality or DEFAULT_FIGURE_QUALITY).strip().lower()
    if normalized not in FIGURE_QUALITY_DPI:
        return DEFAULT_FIGURE_QUALITY
    return normalized


def figure_dpi(quality: str | None = None) -> int:
    """Return the configured PNG DPI."""
    normalized = normalize_figure_quality(quality or _configured_figure_quality)
    return int(FIGURE_QUALITY_DPI[normalized])


def color(name: str) -> str:
    return str(SEMANTIC_COLORS.get(str(name), str(name)))


def import_matplotlib():
    """Import matplotlib with the non-interactive backend expected in tests and headless Pi runs."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


@contextmanager
def report_style():
    """Matplotlib rc-context for light thesis/report figures."""
    plt = import_matplotlib()
    with plt.rc_context(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": SEMANTIC_COLORS["axis"],
            "axes.labelcolor": SEMANTIC_COLORS["text"],
            "xtick.color": SEMANTIC_COLORS["axis"],
            "ytick.color": SEMANTIC_COLORS["axis"],
            "text.color": SEMANTIC_COLORS["text"],
            "axes.grid": True,
            "grid.color": SEMANTIC_COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.0,
            "scatter.marker": "o",
            "figure.dpi": 120,
        }
    ):
        yield plt


def create_figure(*, size: str = "wide", constrained_layout: bool = True):
    """Create a styled matplotlib figure and axes."""
    with report_style() as plt:
        fig, ax = plt.subplots(figsize=FIGURE_SIZES.get(size, FIGURE_SIZES["wide"]), constrained_layout=constrained_layout)
    return fig, ax


def create_3d_figure(*, size: str = "square", constrained_layout: bool = True):
    """Create a styled matplotlib figure with a 3D axes (mplot3d)."""
    with report_style() as plt:
        fig = plt.figure(figsize=FIGURE_SIZES.get(size, FIGURE_SIZES["square"]), constrained_layout=constrained_layout)
        ax = fig.add_subplot(111, projection="3d")
    return fig, ax


def style_3d_axes(
    ax,
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    labelpad: float = 10.0,
    view_elev: float = 22.0,
    view_azim: float = -55.0,
) -> None:
    """Apply common 3D axis title/label treatment matching style_axes.

    Default labelpad and view angle are tuned for thesis figures: enough label
    padding to avoid clipping inside a constrained-layout canvas, and a slight
    front-right camera that keeps the data cube readable.
    """
    if title:
        ax.set_title(str(title), loc="left", pad=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(str(xlabel), labelpad=labelpad)
    if ylabel:
        ax.set_ylabel(str(ylabel), labelpad=labelpad)
    if zlabel:
        ax.set_zlabel(str(zlabel), labelpad=labelpad)
    ax.tick_params(colors=SEMANTIC_COLORS["axis"])
    for axis_label in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis_label.label.set_color(SEMANTIC_COLORS["text"])
        axis_label.pane.set_edgecolor(SEMANTIC_COLORS["grid"])
        axis_label.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.grid(True)
    ax.view_init(elev=float(view_elev), azim=float(view_azim))


def set_equal_xyz(
    ax,
    *,
    x_values: Iterable[float],
    y_values: Iterable[float],
    z_values: Iterable[float],
    pad_fraction: float = 0.12,
    minimum_span: float = 1.0,
) -> None:
    """Set equal x/y/z scaling around the supplied values (cubic bounding box)."""
    xs = [float(value) for value in x_values]
    ys = [float(value) for value in y_values]
    zs = [float(value) for value in z_values]
    if not xs or not ys or not zs:
        return
    x_mid = (min(xs) + max(xs)) / 2.0
    y_mid = (min(ys) + max(ys)) / 2.0
    z_mid = (min(zs) + max(zs)) / 2.0
    span = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        float(minimum_span),
    )
    half_span = (span * (1.0 + float(pad_fraction))) / 2.0
    ax.set_xlim(x_mid - half_span, x_mid + half_span)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_zlim(z_mid - half_span, z_mid + half_span)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except (AttributeError, ValueError):
        pass


def save_figure(fig, path: Path, *, quality: str | None = None) -> Path:
    """Save a PNG using the configured quality and close the figure."""
    plt = import_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=figure_dpi(quality), format="png", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def style_axes(ax, *, title: str = "", xlabel: str = "", ylabel: str = "", grid: bool = True) -> None:
    """Apply common axis title/label/grid treatment."""
    if title:
        ax.set_title(str(title), loc="left", pad=10, fontweight="bold")
    if xlabel:
        ax.set_xlabel(str(xlabel))
    if ylabel:
        ax.set_ylabel(str(ylabel))
    ax.grid(bool(grid))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SEMANTIC_COLORS["axis"])
    ax.spines["bottom"].set_color(SEMANTIC_COLORS["axis"])


def legend(ax, *, loc: str = "best", ncol: int = 1) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        loc=loc,
        ncol=max(1, int(ncol)),
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        framealpha=0.95,
    )


def set_equal_xy(ax, *, x_values: Iterable[float], y_values: Iterable[float], pad_fraction: float = 0.12, minimum_span: float = 1.0) -> None:
    """Set equal x/y scaling around the supplied values."""
    xs = [float(value) for value in x_values]
    ys = [float(value) for value in y_values]
    if not xs or not ys:
        return
    x_mid = (min(xs) + max(xs)) / 2.0
    y_mid = (min(ys) + max(ys)) / 2.0
    span = max(max(xs) - min(xs), max(ys) - min(ys), float(minimum_span))
    half_span = (span * (1.0 + float(pad_fraction))) / 2.0
    ax.set_xlim(x_mid - half_span, x_mid + half_span)
    ax.set_ylim(y_mid - half_span, y_mid + half_span)
    ax.set_aspect("equal", adjustable="box")


def add_metric_box(ax, lines: list[str], *, loc: str = "upper right") -> None:
    """Add a small unobtrusive metric annotation box."""
    if not lines:
        return
    anchors = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }
    x, y, ha, va = anchors.get(loc, anchors["upper right"])
    ax.text(
        x,
        y,
        "\n".join(str(line) for line in lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.94,
        },
    )


def report_and_debug_names(stem: str) -> tuple[str, str]:
    """Return simple report/debug PNG names for one figure family."""
    safe = str(stem).strip().removesuffix(".png")
    return f"{safe}.png", f"{safe}_debug.png"


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None
