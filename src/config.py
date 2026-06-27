from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

SYNTHETIC_DATA = DATA_DIR / "synthetic_analysis_ready.csv"

TIMING_ORDER = ["Morning", "Night", "Afternoon", "Evening"]
YEAR_ORDER = ["2010-2014", "2015-2019", "2020-2025"]
SEX_ORDER = ["Female", "Male"]
STAGE_ORDER = ["Stage I", "Stage II", "Stage III", "Stage IV", "Recurrence", "Unknown"]
KPS_ORDER = [">=90", "<90", "Not documented"]
CHEMO_ORDER = ["Not documented", "Yes"]
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]

CANCER_ORDER = [
    "Breast Cancer",
    "Colorectal Cancer",
    "Lung Cancer",
    "Cervical Cancer",
    "Esophageal Cancer",
    "Nasopharyngeal Carcinoma",
    "Lymphoma",
    "Gastric Cancer",
    "Other Head and Neck Cancer",
    "Pancreatic Cancer",
    "Prostate Cancer",
    "Soft Tissue Sarcoma",
    "Liver Cancer",
    "Other",
]

MAJOR_CANCERS = [
    "Nasopharyngeal Carcinoma",
    "Esophageal Cancer",
    "Lung Cancer",
    "Colorectal Cancer",
    "Breast Cancer",
    "Cervical Cancer",
]

COVARIATE_CATEGORIES = {
    "timing_group": TIMING_ORDER,
    "sex": SEX_ORDER,
    "year_bin": YEAR_ORDER,
    "stage": STAGE_ORDER,
    "kps_group": KPS_ORDER,
    "concurrent_chemo": CHEMO_ORDER,
    "total_dose_group": ["Low", "Middle", "High"],
    "fraction_dose_group": ["Low", "Middle", "High"],
}

CONTINUOUS_COVARIATES = ["age_at_rt", "approval_to_rt_days"]
