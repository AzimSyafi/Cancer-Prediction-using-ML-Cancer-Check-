"""
Sanity check (run after 02_preprocess.py)
FYP: Cancer Prediction Using Machine Learning

For each cleaned dataset:
  1. Baseline accuracy of always guessing the most common class.
     A real model should beat this.
  2. How strongly each feature relates to the target (correlation and
     mutual information). If every value is close to 0, the features carry
     no information about the label and no model can learn it.
"""

import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")

for cancer in ["lung", "breast", "cervical", "cervical_screening"]:
    df = pd.read_csv(os.path.join(CLEAN_DIR, f"{cancer}_clean.csv"))
    X = df.drop(columns="target").fillna(df.median(numeric_only=True))
    y = df["target"]

    baseline = y.value_counts(normalize=True).max()
    corr = X.corrwith(y).abs()
    mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)
    table = pd.DataFrame({"abs_correlation": corr, "mutual_info": mi}).round(4)
    table = table.sort_values("abs_correlation", ascending=False)

    print("=" * 60)
    print(f"{cancer.upper()}")
    print(f"  Always-guess-majority accuracy: {baseline:.3f}")
    print(f"  Strongest feature correlation with target: {corr.max():.3f}")
    print("  Top 5 features:")
    print(table.head(5).to_string())
