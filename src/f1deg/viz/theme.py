"""F1 project visual theme — import once to apply everywhere.

Usage (top of any notebook or script):
    from f1deg.viz.theme import apply_theme, PALETTE
    apply_theme()
"""

from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

# Core brand / chart colours (dark background friendly)
SLATE = "#1E1E2E"  # background
SURFACE = "#2A2A3C"  # card / panel background
GRID = "#3B3B50"  # grid lines
TEXT = "#CDD6F4"  # primary text
TEXT_DIM = "#7F849C"  # secondary text / axis labels

# Accent palette — 8 distinguishable colours for categorical series
ACCENT = [
    "#F38BA8",  # rose
    "#89B4FA",  # blue
    "#A6E3A1",  # green
    "#FAB387",  # peach
    "#CBA6F7",  # mauve
    "#94E2D5",  # teal
    "#F9E2AF",  # yellow
    "#74C7EC",  # sapphire
]

# F1 tire compound colours (official-ish)
COMPOUND = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC300",
    "HARD": "#DDDDDD",
    "INTERMEDIATE": "#39D353",
    "WET": "#0090FF",
}

# Semantic colours
POSITIVE = "#A6E3A1"
NEGATIVE = "#F38BA8"
NEUTRAL = "#89B4FA"

# Collect everything for easy import
PALETTE = {
    "slate": SLATE,
    "surface": SURFACE,
    "grid": GRID,
    "text": TEXT,
    "text_dim": TEXT_DIM,
    "accent": ACCENT,
    "compound": COMPOUND,
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "neutral": NEUTRAL,
}


# ---------------------------------------------------------------------------
# Matplotlib rcParams override
# ---------------------------------------------------------------------------

_RC = {
    # Figure
    "figure.facecolor": SLATE,
    "figure.edgecolor": SLATE,
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    # Axes
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.prop_cycle": mpl.cycler(color=ACCENT),
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Grid
    "grid.color": GRID,
    "grid.alpha": 0.4,
    "grid.linewidth": 0.5,
    # Ticks
    "xtick.color": TEXT_DIM,
    "ytick.color": TEXT_DIM,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    # Legend
    "legend.facecolor": SURFACE,
    "legend.edgecolor": GRID,
    "legend.fontsize": 10,
    "legend.labelcolor": TEXT,
    # Text
    "text.color": TEXT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 11,
    # Lines / scatter
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "scatter.edgecolors": "none",
    # Savefig
    "savefig.facecolor": SLATE,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}


def apply_theme() -> None:
    """Apply the F1 dark theme to all subsequent matplotlib plots."""
    mpl.rcParams.update(_RC)


def reset_theme() -> None:
    """Revert to matplotlib defaults."""
    mpl.rcdefaults()


def compound_color(name: str) -> str:
    """Return the hex colour for a tire compound name."""
    return COMPOUND.get(name.upper(), TEXT_DIM)
