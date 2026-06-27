from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis_utils import censor_at, ensure_dirs
from config import FIGURE_DIR, MAJOR_CANCERS, SYNTHETIC_DATA, TABLE_DIR
from plot_style import BLUE, LIGHT_BLUE, PANEL_LABELS, add_panel_label, save_figure, set_publication_style


SHORT_NAMES = {
    "Nasopharyngeal Carcinoma": "Nasopharyngeal Carcinoma",
    "Esophageal Cancer": "Esophageal Cancer",
    "Lung Cancer": "Lung Cancer",
    "Colorectal Cancer": "Colorectal Cancer",
    "Breast Cancer": "Breast Cancer",
    "Cervical Cancer": "Cervix Cancer",
}


def add_cyclic_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    theta = 2 * np.pi * out["rt_time_minutes"] / 1440.0
    out["time_sin_1"] = np.sin(theta)
    out["time_cos_1"] = np.cos(theta)
    out["time_sin_2"] = np.sin(2 * theta)
    out["time_cos_2"] = np.cos(2 * theta)
    return out


def fit_cyclic_model(df: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame]:
    model = add_cyclic_terms(censor_at(df, "time_to_os_days", "os_event", 5 * 365.25))
    cols = [
        "time_to_os_days",
        "os_event",
        "time_sin_1",
        "time_cos_1",
        "time_sin_2",
        "time_cos_2",
        "age_at_rt",
        "approval_to_rt_days",
        "sex",
        "year_bin",
        "stage",
        "kps_group",
        "concurrent_chemo",
        "total_dose_group",
        "fraction_dose_group",
    ]
    model = model[cols].dropna()
    encoded = pd.get_dummies(model, columns=["sex", "year_bin", "stage", "kps_group", "concurrent_chemo", "total_dose_group", "fraction_dose_group"], drop_first=True)
    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    encoded = encoded.dropna()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(encoded, "time_to_os_days", "os_event")
    return cph, encoded


def prediction_grid(model_frame: pd.DataFrame) -> pd.DataFrame:
    hours = np.linspace(0, 24, 193)
    minutes = (hours * 60) % 1440
    base = pd.DataFrame({"rt_time_minutes": minutes})
    base = add_cyclic_terms(base)
    for col in model_frame.columns:
        if col in {"time_to_os_days", "os_event", "time_sin_1", "time_cos_1", "time_sin_2", "time_cos_2"}:
            continue
        base[col] = model_frame[col].mean()
    base["hour"] = hours
    return base[[c for c in model_frame.columns if c not in {"time_to_os_days", "os_event"}] + ["hour"]]


def make_curve(cph: CoxPHFitter, model_frame: pd.DataFrame) -> pd.DataFrame:
    grid = prediction_grid(model_frame)
    x = grid.drop(columns=["hour"])
    reference_index = int(np.argmin(np.abs(grid["hour"] - 10.0)))
    beta = cph.params_.reindex(x.columns).fillna(0).to_numpy()
    variance = cph.variance_matrix_.reindex(index=x.columns, columns=x.columns).fillna(0).to_numpy()
    x_mat = x.to_numpy()
    diff = x_mat - x_mat[reference_index]
    log_hazard = diff @ beta
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, variance, diff), 0))
    grid["HR"] = np.exp(log_hazard)
    grid["HR_lower_95"] = np.exp(log_hazard - 1.96 * se)
    grid["HR_upper_95"] = np.exp(log_hazard + 1.96 * se)
    return grid[["hour", "HR", "HR_lower_95", "HR_upper_95"]]


def plot_panel(ax_curve, ax_hist, curve: pd.DataFrame, subset: pd.DataFrame, title: str, label: str) -> None:
    add_panel_label(ax_curve, label, x=-0.10, y=1.12)
    ax_curve.fill_between(curve["hour"], curve["HR_lower_95"], curve["HR_upper_95"], color=LIGHT_BLUE, label="95% CI")
    ax_curve.plot(curve["hour"], curve["HR"], color=BLUE, linewidth=1.3, label="Hazard Ratio (Cyclic)")
    ref_hr = float(np.interp(10.0, curve["hour"], curve["HR"]))
    ax_curve.scatter([10], [ref_hr], s=12, color="#D62728", zorder=5, label="Ref (10:00)")
    ax_curve.axhline(1, color="#999999", linewidth=0.7, linestyle="--")
    ax_curve.set_xlim(0, 24)
    y_min = max(0.4, float(curve["HR_lower_95"].quantile(0.01)) * 0.9)
    y_max = min(2.2, float(curve["HR_upper_95"].quantile(0.99)) * 1.1)
    ax_curve.set_ylim(y_min, y_max)
    ax_curve.set_title(title, loc="left", fontsize=6.8)
    ax_curve.set_ylabel("Hazard Ratio (HR)", fontsize=8)
    ax_curve.grid(alpha=0.12, linestyle=":")
    ax_curve.legend(loc="upper right", fontsize=5.8, frameon=True, edgecolor="#DDDDDD")

    ax_hist.hist(subset["rt_time_minutes"] / 60.0, bins=np.arange(0, 24.5, 0.5), color="#C8C8C8", edgecolor="white", linewidth=0.25)
    ax_hist.set_xlim(0, 24)
    ax_hist.set_ylabel("No. Patients", fontsize=8)
    ax_hist.set_xlabel("Time of Day (Hour)", fontsize=8)
    ax_hist.grid(axis="y", alpha=0.10)


def main() -> None:
    set_publication_style()
    ensure_dirs(TABLE_DIR, FIGURE_DIR)
    df = pd.read_csv(SYNTHETIC_DATA)
    cph, model_frame = fit_cyclic_model(df)
    grid = make_curve(cph, model_frame)
    out_path = TABLE_DIR / "cyclic_timing_curve.csv"
    grid.to_csv(out_path, index=False)

    fig = plt.figure(figsize=(11.6, 6.7))
    outer = GridSpec(2, 3, figure=fig, wspace=0.18, hspace=0.26)
    for index, cancer in enumerate(MAJOR_CANCERS[:6]):
        row, col = divmod(index, 3)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_curve = fig.add_subplot(inner[0])
        ax_hist = fig.add_subplot(inner[1], sharex=ax_curve)
        subset = df[df["cancer_type"] == cancer].copy()
        cancer_cph, cancer_model = fit_cyclic_model(subset)
        cancer_curve = make_curve(cancer_cph, cancer_model)
        table_path = TABLE_DIR / f"cyclic_timing_curve_{cancer.lower().replace(' ', '_')}.csv"
        cancer_curve.to_csv(table_path, index=False)
        title = f"{SHORT_NAMES.get(cancer, cancer)} (5-Year Survival) RCS Smoothing (df=4)"
        plot_panel(ax_curve, ax_hist, cancer_curve, subset, title, PANEL_LABELS[index])
        plt.setp(ax_curve.get_xticklabels(), visible=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.18, hspace=0.28)
    save_figure(fig, FIGURE_DIR / "figure3_cyclic_timing_panels")
    plt.close(fig)
    print(f"Saved cyclic timing curve: {out_path}")
    print(f"Saved cyclic timing figure: {FIGURE_DIR / 'figure3_cyclic_timing_panels.pdf'}")


if __name__ == "__main__":
    main()
