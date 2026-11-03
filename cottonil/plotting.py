"""Matplotlib defaults for the notebooks (static PNGs render everywhere: GitHub, HF, VS Code)."""
import matplotlib.pyplot as plt

PALETTE = ["#0B3C5D", "#1D8A99", "#F4A259", "#E4572E", "#7C6A9C", "#5B8E7D", "#B8B8B8"]
NAVY, TEAL, ORANGE, RED = PALETTE[:4]


def setup():
    plt.rcParams.update({
        "figure.figsize": (9, 4), "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .25, "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.titleweight": "bold", "axes.titlesize": 12, "legend.frameon": False,
    })
