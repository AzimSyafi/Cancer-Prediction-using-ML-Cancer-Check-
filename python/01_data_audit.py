"""
Stage 2: Data audit (run this first)
FYP: Cancer Prediction Using Machine Learning

Checks each dataset on its own and saves a summary table you can paste
into Section 3.3 of the report (record counts, features, class balance,
missing values, duplicates).

Folder layout expected:
    project/
        01_data_audit.py
        data/
            lung_cancer.csv
            breast_cancer.csv
            cervical_cancer.csv
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# File name and label column for each dataset
DATASETS = {
    "Lung": ("lung_cancer.csv", "LUNG_CANCER"),
    "Breast": ("breast_cancer.csv", "diagnosis"),
    "Cervical": ("cervical_cancer.csv", "Biopsy"),
}

rows = []
for cancer, (filename, label) in DATASETS.items():
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    # Cervical uses "?" for missing values
    df = df.replace("?", np.nan)

    n_rows = len(df)
    n_dupes = int(df.duplicated().sum())
    n_features = df.shape[1] - 1  # minus the label
    missing_cells = int(df.isna().sum().sum())
    cols_with_missing = int((df.isna().sum() > 0).sum())
    balance = df[label].value_counts().to_dict()

    print("=" * 60)
    print(f"{cancer} ({filename})")
    print(f"  Rows: {n_rows}   Duplicate rows: {n_dupes}   Features: {n_features}")
    print(f"  Missing cells: {missing_cells} across {cols_with_missing} columns")
    print(f"  Label '{label}' counts: {balance}")
    if cols_with_missing:
        missing_pct = (df.isna().mean() * 100).round(1)
        print("  Missing % per column (only columns with gaps):")
        print(missing_pct[missing_pct > 0].sort_values(ascending=False).to_string())

    rows.append({
        "Dataset": cancer,
        "Rows (raw)": n_rows,
        "Duplicate rows": n_dupes,
        "Rows (after removing duplicates)": n_rows - n_dupes,
        "Features": n_features,
        "Missing cells": missing_cells,
        "Label column": label,
        "Class counts": balance,
    })

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(DATA_DIR, "audit_summary.csv"), index=False)
print("\nSaved audit summary to data/audit_summary.csv")
print(summary.to_string(index=False))
