"""Shared matplotlib style for model 2 figures (light theme, report-ready PNGs)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
RESULT_COLOR = {"H": SERIES[0], "D": SERIES[1], "A": SERIES[2]}
RESULT_NAME = {"H": "Home win", "D": "Draw", "A": "Away win"}
ACCENT = SERIES[0]

DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a51d1d"])


def setup():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "legend.frameon": False, "lines.linewidth": 2, "font.size": 10,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", FIG / name)
