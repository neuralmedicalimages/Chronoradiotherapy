from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


PANEL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
BLUE = "#4C78A8"
LIGHT_BLUE = "#D8E4EC"
GREEN = "#2F7D45"
ORANGE = "#D47A34"
GRAY = "#BFBFBF"
DARK_GRAY = "#333333"

CANCER_COLORS = {
    "Breast Cancer": "#C28B7C",
    "Nasopharyngeal Carcinoma": "#E7C275",
    "Lung Cancer": "#8FAA83",
    "Esophageal Cancer": "#A65D70",
    "Colorectal Cancer": "#78B7BD",
    "Cervical Cancer": "#6F86A5",
    "Other": "#9AA0A6",
}


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#666666",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".pdf"))
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=300)


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<.001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def clean_term(term: str) -> str:
    replacements = {
        "timing_group_": "",
        "sex_": "",
        "year_bin_": "",
        "stage_": "",
        "kps_group_": "",
        "concurrent_chemo_": "",
        "total_dose_group_": "",
        "fraction_dose_group_": "",
    }
    out = str(term)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out
