"""
Stage 2b: Preprocessing (run after 01_data_audit.py)
FYP: Cancer Prediction Using Machine Learning

Cleans each dataset on its own and saves a cleaned copy to data/clean/.
Every cleaned file has the same layout: feature columns plus one 'target'
column (1 = cancer, 0 = no cancer).

Imputation and scaling are NOT done here on purpose. They are fitted later
inside the training pipeline on the training split only, so no information
from the test split leaks into training.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
os.makedirs(CLEAN_DIR, exist_ok=True)


def report(name, df):
    counts = df["target"].value_counts().to_dict()
    print(f"{name}: {df.shape[0]} rows, {df.shape[1] - 1} features, target counts {counts}")


# ---------------------------------------------------------------------
# LUNG
# ---------------------------------------------------------------------
lung = pd.read_csv(os.path.join(DATA_DIR, "lung_cancer.csv"))
lung.columns = [c.strip().upper().replace(" ", "_") for c in lung.columns]
lung = lung.drop_duplicates()
lung["GENDER"] = lung["GENDER"].map({"M": 1, "F": 0})
# Symptom columns are coded 1 = No, 2 = Yes. Convert to 0/1.
for c in lung.columns:
    if c not in ("GENDER", "AGE", "LUNG_CANCER") and lung[c].isin([1, 2]).all():
        lung[c] = lung[c].map({1: 0, 2: 1})
lung["target"] = lung.pop("LUNG_CANCER").map({"YES": 1, "NO": 0})
report("Lung", lung)
lung.to_csv(os.path.join(CLEAN_DIR, "lung_clean.csv"), index=False)

# ---------------------------------------------------------------------
# BREAST
# ---------------------------------------------------------------------
breast = pd.read_csv(os.path.join(DATA_DIR, "breast_cancer.csv"))
breast.columns = [c.strip().lower().replace(" ", "_") for c in breast.columns]
breast = breast.drop(columns=[c for c in breast.columns if c == "id" or c.startswith("unnamed")])
breast = breast.drop_duplicates()
breast["target"] = breast.pop("diagnosis").map({"M": 1, "B": 0})
report("Breast", breast)
breast.to_csv(os.path.join(CLEAN_DIR, "breast_clean.csv"), index=False)

# ---------------------------------------------------------------------
# CERVICAL
# ---------------------------------------------------------------------
cervical = pd.read_csv(os.path.join(DATA_DIR, "cervical_cancer.csv"))
cervical = cervical.replace("?", np.nan)
cervical = cervical.apply(lambda s: pd.to_numeric(s, errors="coerce"))
cervical = cervical.drop_duplicates()

# Drop the two columns that are over 90% empty
cervical = cervical.drop(columns=[
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
])

# Remove previous-diagnosis columns (Dx). They record earlier diagnoses,
# not risk factors, so they are left out of both cervical models.
cervical = cervical.drop(columns=["Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx"])

# The three screening tests look directly at the cervix. Keep them aside:
# the main cervical model uses risk factors only, and a second model adds
# the screening tests so the two can be compared.
SCREENING = ["Hinselmann", "Schiller", "Citology"]
screening = cervical[SCREENING].copy()
cervical = cervical.drop(columns=SCREENING)

# The STD questions are missing as one block (about 12% of patients skipped
# them). Keep a flag so the model knows the answer was not given, rather
# than treating a filled-in median as a real answer.
cervical["STDs_not_answered"] = cervical["STDs"].isna().astype(int)

# Drop columns that have only one value (e.g. no patient has STDs:AIDS).
# They carry no information and cause divide-by-zero warnings.
constant = [c for c in cervical.columns
            if c != "Biopsy" and cervical[c].nunique(dropna=True) <= 1]
cervical = cervical.drop(columns=constant)
print("Cervical: dropped constant columns:", constant)

cervical.columns = [c.strip().replace(" ", "_") for c in cervical.columns]
cervical["target"] = cervical.pop("Biopsy").astype(int)
report("Cervical", cervical)
cervical.to_csv(os.path.join(CLEAN_DIR, "cervical_clean.csv"), index=False)

# ---------------------------------------------------------------------
# CERVICAL WITH SCREENING TESTS (comparison model)
# Same patients and risk factors, plus Hinselmann, Schiller and Citology.
# ---------------------------------------------------------------------
cervical_screening = cervical.drop(columns="target").copy()
for c in SCREENING:
    cervical_screening[c] = screening.loc[cervical.index, c].astype(int)
cervical_screening["target"] = cervical["target"]
report("Cervical + screening", cervical_screening)
cervical_screening.to_csv(os.path.join(CLEAN_DIR, "cervical_screening_clean.csv"), index=False)

print(f"\nCleaned files saved to {CLEAN_DIR}")
