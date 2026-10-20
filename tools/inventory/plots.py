"""Matplotlib helpers with one consistent, quiet look (light surface)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
# Categorical slots in fixed order; charts that overlay series use at most three.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
    }
)


def save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def ref_line(ax, x: float, label: str, level: int = 0) -> None:
    """Thin solid reference line with a small label; `level` staggers labels downward so
    neighbouring lines don't collide."""
    ax.axvline(x, color=INK_2, linewidth=1)
    ax.text(
        x,
        0.98 - 0.11 * level,
        f" {label}",
        transform=ax.get_xaxis_transform(),
        color=INK_2,
        fontsize=7,
        va="top",
        ha="left",
    )


def step_hist(ax, values, bins, color, label=None, density=False) -> None:
    """Histogram as a 2px outline over a ~10% wash, so overlays stay readable."""
    counts, edges = np.histogram(values, bins=bins, density=density)
    ax.stairs(counts, edges, fill=True, color=color, alpha=0.10, linewidth=0)
    ax.stairs(counts, edges, color=color, linewidth=1.6, label=label)
    ax.set_ylim(bottom=0)


def hbar(ax, labels, values, color=SERIES[0], value_fmt="{:,.0f}") -> None:
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.7, color=color)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, labelcolor=INK_2)
    xmax = max(values) if len(values) else 1
    for yi, v in zip(y, values, strict=True):
        ax.text(v + xmax * 0.01, yi, value_fmt.format(v), va="center", fontsize=7, color=INK_2)
    ax.set_xlim(0, xmax * 1.12)


def small_multiples(n: int, ncols: int, panel_w: float = 2.6, panel_h: float = 1.8, **kw):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), squeeze=False, **kw
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    fig.subplots_adjust(hspace=0.65, wspace=0.3)
    return fig, flat[:n]
