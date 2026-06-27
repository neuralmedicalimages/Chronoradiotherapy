# Analysis Code

This repository contains a compact, runnable version of the analysis workflow used for the radiotherapy timing study. It is intended for code review and reproducibility of the statistical workflow.

The original patient-level raw data are not included because they contain protected clinical information. The scripts start from an analysis-ready patient-level table. For demonstration, `src/generate_synthetic_data.py` creates a synthetic dataset with the same general structure as the analysis table. The synthetic data are randomly generated and must not be interpreted as study results.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

The complete synthetic workflow writes outputs to:

```text
data/synthetic_analysis_ready.csv
output/tables/
output/figures/
```

Figures are exported as publication-style PDF files with PNG previews. Tables are exported as machine-readable CSV files, and selected baseline tables are also exported as formatted Excel workbooks.

## Repository Layout

```text
code_open/
  README.md
  requirements.txt
  run_all.py
  src/
    generate_synthetic_data.py
    table1.py
    main_cox.py
    interaction.py
    seasonality.py
    psm.py
    pfs.py
    subgroup.py
    rcs_timing.py
    figures.py
    analysis_utils.py
    config.py
```

## Analysis Modules

`generate_synthetic_data.py` creates an analysis-ready patient-level dataset with no real patient records.

`table1.py` generates baseline demographic and clinical characteristics by treatment timing group.

`main_cox.py` fits multivariable Cox models for overall survival in the overall population and cancer-specific populations, including 5-year and 10-year censored analyses.

`interaction.py` fits a cancer-site-by-treatment-timing interaction model and reports a likelihood-ratio test.

`seasonality.py` fits season-of-treatment models adjusted for the same core clinical variables.

`psm.py` performs propensity-score matching for night versus morning treatment in selected cancer types and fits matched Cox models.

`pfs.py` fits progression-free survival models for selected cancer types.

`subgroup.py` estimates the night-versus-morning association within clinically relevant subgroups.

`rcs_timing.py` fits a continuous clock-time model using cyclic basis terms and exports a timing curve.

`figures.py` creates example figures from the synthetic outputs.

`plot_style.py` stores shared publication-style settings, including fonts, colors, panel labels, vector output settings, and common formatting helpers.

## Analysis-Ready Data Columns

The synthetic analysis table includes:

```text
patient_id
cancer_type
age_at_rt
sex
year
year_bin
month
season
stage
approval_to_rt_days
rt_time_minutes
timing_group
kps_group
concurrent_chemo
total_dose
fraction_dose
num_fractions
total_dose_group
fraction_dose_group
technique
sbrt_like
lung_subtype
colorectal_subtype
lymphoma_subtype
time_to_os_days
os_event
time_to_pfs_days
pfs_event
```

## Model Covariates

The main adjusted models include treatment timing group, age at radiotherapy, sex, year of treatment, tumor stage, approval-to-treatment interval, total dose group, dose per fraction group, KPS group, and concurrent chemotherapy.

The public scripts intentionally exclude raw data extraction, identifier linkage, clinical text parsing, and institution-specific data cleaning. Those steps cannot be shared because they depend on protected source data and local data systems.

## Code Availability Statement

The public code provides the analysis workflow used in the study, beginning from an analysis-ready patient-level table. To protect patient confidentiality, raw clinical data and institution-specific data preparation scripts are not distributed. A synthetic dataset generator is included so that all statistical scripts can be executed end to end and reviewed for reproducibility.
