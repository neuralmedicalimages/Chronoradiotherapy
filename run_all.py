from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from analysis_utils import ensure_dirs
from config import DATA_DIR, FIGURE_DIR, OUTPUT_DIR, SYNTHETIC_DATA, TABLE_DIR
import figures
import generate_synthetic_data
import interaction
import main_cox
import pfs
import psm
import rcs_timing
import seasonality
import subgroup
import table1


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if SYNTHETIC_DATA.exists():
        SYNTHETIC_DATA.unlink()
    ensure_dirs(DATA_DIR, OUTPUT_DIR, TABLE_DIR, FIGURE_DIR)
    steps = [
        ("Synthetic analysis-ready data", generate_synthetic_data.main),
        ("Baseline characteristics", table1.main),
        ("Main multivariable Cox models", main_cox.main),
        ("Site-by-time interaction", interaction.main),
        ("Seasonality models", seasonality.main),
        ("Propensity-score matched models", psm.main),
        ("PFS models", pfs.main),
        ("Subgroup models", subgroup.main),
        ("Continuous clock-time model", rcs_timing.main),
        ("Figures", figures.main),
    ]
    for label, func in steps:
        print(f"\n=== {label} ===")
        func()
    print("\nAll synthetic analyses completed.")


if __name__ == "__main__":
    main()
