"""
Stage 3-7: Train, evaluate and save models (run after 02_preprocess.py)
FYP: Cancer Prediction Using Machine Learning

Same protocol for every cancer type:
  - 80/20 stratified train-test split
  - Pipeline: median imputation -> standard scaling -> random oversampling -> model
    (all fitted on training folds only, so the test split never leaks in)
  - k = 10 stratified cross-validation grid search, best model picked by F1
  - Evaluation on the held-out test split with Accuracy, Precision, Recall,
    F1 and ROC-AUC, plus CV mean and spread, confusion matrices and ROC curves

Outputs:
  results/test_results.csv        test-split scores for all 9 models
  results/cv_results.csv          10-fold CV mean and std for all 9 models
  results/feature_importance_*.csv  Random Forest importance per cancer
  figures/                        confusion matrices and ROC curves
  models/                         trained pipelines + metadata for the app

Requires: pip install scikit-learn imbalanced-learn matplotlib joblib
Expect this to take roughly 10 to 20 minutes (SVM on breast is the slowest).
"""

import os
os.environ["PYTHONWARNINGS"] = "ignore"  # hide convergence warnings in worker processes

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")
for d in (RESULTS_DIR, FIG_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
CANCERS = ["lung", "breast", "cervical", "cervical_screening"]
DISPLAY = {
    "lung": "Lung",
    "breast": "Breast",
    "cervical": "Cervical (risk factors)",
    "cervical_screening": "Cervical (+ screening tests)",
}
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def build_models():
    """Identical pipelines and grids for every cancer type."""
    def make(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("model", estimator),
        ])

    return {
        "ANN": (
            make(MLPClassifier(activation="relu", solver="adam", max_iter=500,
                               early_stopping=True, random_state=RANDOM_STATE)),
            {"model__hidden_layer_sizes": [(64, 32), (100, 50)],
             "model__alpha": [0.0001, 0.001]},
        ),
        "SVM": (
            make(SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {"model__C": [1, 10], "model__gamma": ["scale", "auto"]},
        ),
        "Random Forest": (
            make(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            {"model__max_depth": [None, 10, 20],
             "model__min_samples_split": [2, 5]},
        ),
    }


def slug(name):
    return name.lower().replace(" ", "_")


def feature_metadata(X_train):
    """Describe each feature so the app can build the right input widget."""
    meta = []
    for col in X_train.columns:
        values = X_train[col].dropna()
        is_binary = set(values.unique()).issubset({0, 1})
        meta.append({
            "name": col,
            "type": "binary" if is_binary else "numeric",
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
        })
    return meta


def main():
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    test_rows, cv_rows = [], []

    for cancer in CANCERS:
        print("\n" + "=" * 60)
        df = pd.read_csv(os.path.join(CLEAN_DIR, f"{cancer}_clean.csv"))
        X = df.drop(columns="target")
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        print(f"{cancer.upper()}: train {len(X_train)}, test {len(X_test)} "
              f"(test positives: {int(y_test.sum())})")

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        fitted = {}

        for name, (pipe, grid) in build_models().items():
            search = GridSearchCV(pipe, grid, cv=cv, scoring=SCORING,
                                  refit="f1", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            fitted[name] = best
            i = search.best_index_

            # 10-fold CV scores of the chosen settings
            cv_row = {"Cancer": cancer, "Model": name}
            for m in SCORING:
                cv_row[f"{m}_mean"] = search.cv_results_[f"mean_test_{m}"][i]
                cv_row[f"{m}_std"] = search.cv_results_[f"std_test_{m}"][i]
            cv_rows.append(cv_row)

            # Held-out test scores
            y_pred = best.predict(X_test)
            y_prob = best.predict_proba(X_test)[:, 1]
            train_f1 = f1_score(y_train, best.predict(X_train), zero_division=0)
            test_row = {
                "Cancer": cancer,
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
                "Train F1": train_f1,  # compare with test F1 to check overfitting
                "Best params": str(search.best_params_),
            }
            test_rows.append(test_row)
            print(f"  {name:<14} test F1 {test_row['F1']:.3f} | "
                  f"recall {test_row['Recall']:.3f} | AUC {test_row['ROC-AUC']:.3f} | "
                  f"CV F1 {cv_row['f1_mean']:.3f} ± {cv_row['f1_std']:.3f}")

            # Confusion matrix
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=["No cancer", "Cancer"],
                cmap="Blues", colorbar=False, ax=ax_cm,
            )
            ax_cm.set_title(f"{DISPLAY[cancer]} | {name}", fontsize=10)
            fig_cm.tight_layout()
            fig_cm.savefig(os.path.join(FIG_DIR, f"cm_{cancer}_{slug(name)}.png"), dpi=200)
            plt.close(fig_cm)

            # ROC curve on the shared plot
            RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax_roc)

            # Save the full pipeline for the app
            joblib.dump(best, os.path.join(MODEL_DIR, f"{cancer}_{slug(name)}.joblib"))

        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        ax_roc.set_title(f"ROC curves | {DISPLAY[cancer]}")
        fig_roc.tight_layout()
        fig_roc.savefig(os.path.join(FIG_DIR, f"roc_{cancer}.png"), dpi=200)
        plt.close(fig_roc)

        # Random Forest feature importance
        rf = fitted["Random Forest"].named_steps["model"]
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        imp.to_csv(os.path.join(RESULTS_DIR, f"feature_importance_{cancer}.csv"),
                   header=["importance"])
        fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
        imp.head(10).iloc[::-1].plot.barh(ax=ax_fi, color="#2a7f62")
        ax_fi.set_title(f"Top 10 features | {DISPLAY[cancer]} (Random Forest)", fontsize=10)
        ax_fi.set_xlabel("Importance")
        fig_fi.tight_layout()
        fig_fi.savefig(os.path.join(FIG_DIR, f"importance_{cancer}.png"), dpi=200)
        plt.close(fig_fi)

        # Metadata the app uses to build its input form
        with open(os.path.join(MODEL_DIR, f"{cancer}_meta.json"), "w") as f:
            json.dump({"features": feature_metadata(X_train),
                       "models": [slug(n) for n in fitted]}, f, indent=2)

    test_df = pd.DataFrame(test_rows).round(4)
    cv_df = pd.DataFrame(cv_rows).round(4)
    test_df.to_csv(os.path.join(RESULTS_DIR, "test_results.csv"), index=False)
    cv_df.to_csv(os.path.join(RESULTS_DIR, "cv_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print(test_df.drop(columns="Best params").to_string(index=False))
    print(f"\nSaved results to {RESULTS_DIR}, figures to {FIG_DIR}, models to {MODEL_DIR}")


if __name__ == "__main__":
    main()
