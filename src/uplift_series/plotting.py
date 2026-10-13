"""One consistent chart style for every notebook and article.

Colours come from a validated categorical palette (checked for colour-blind
separation). Each client type always gets the same colour, in every chart.
Because two of the colours are light against a white background, charts
that use them always carry direct text labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BACKGROUND_POINTS = "#d8d7d0"

# Categorical slots, used in this fixed order.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

#: Fixed colour per client type, used across the whole series.
TYPE_COLORS = {
    "persuadable": BLUE,
    "sure_thing": ORANGE,
    "lost_cause": AQUA,
    "sleeping_dog": YELLOW,
}

TYPE_LABELS = {
    "persuadable": "Persuadables",
    "sure_thing": "Sure things",
    "lost_cause": "Lost causes",
    "sleeping_dog": "Sleeping dogs",
}

TYPE_ORDER = ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"


def set_style() -> None:
    """Apply the series chart style to matplotlib."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def save(fig, name: str) -> Path:
    """Save a figure to figures/<name>.png and return the path."""
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def percent_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format an axis holding proportions (0.25) as percentages (25%)."""
    from matplotlib.ticker import PercentFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(PercentFormatter(1.0, decimals=decimals))
