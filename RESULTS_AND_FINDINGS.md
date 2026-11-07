# Results and Findings: Cancer Prediction Using Machine Learning

This document summarises the results from the pipeline in this folder (`01_data_audit.py` → `02_preprocess.py` → `03_train.py`). Every number comes from `data/audit_summary.csv`, `results/test_results.csv`, `results/cv_results.csv` and `results/feature_importance_*.csv`.

---

## 1. Experimental setup

| Item | Setting |
|---|---|
| Cancer types | Lung, Breast, Cervical (risk factors only), Cervical + screening tests |
| Models | ANN (MLPClassifier), SVM (RBF kernel), Random Forest |
| Split | 80/20 stratified train/test, `random_state = 42` |
| Pipeline | Median imputation → Standard scaling → Random oversampling → Model (all fitted on training folds only) |
| Tuning | `GridSearchCV`, 10-fold stratified CV, best settings chosen by **F1** |
| Metrics | Accuracy, Precision, Recall, F1, ROC-AUC (test split), plus CV mean ± std |

The same pipeline and hyperparameter grids were used for every cancer type, so results can be compared fairly across datasets.

---

## 2. Datasets after cleaning

| Dataset | Raw rows | Duplicates removed | Clean rows | Features | Cancer / No cancer | Majority-class baseline | Test size (positives) |
|---|---|---|---|---|---|---|---|
| Lung | 309 | 33 | 276 | 15 | 238 / 38 | 0.862 | 56 (48) |
| Breast | 5015 | 0 | 5015 | 30 | 3149 / 1866 | 0.628 | 1003 (630) |
| Cervical (risk factors) | 858 | 23 | 835 | 25 | 54 / 781 | 0.935 | 167 (11) |
| Cervical + screening | 858 | 23 | 835 | 28 | 54 / 781 | 0.935 | 167 (11) |

**Majority-class baseline** is the accuracy of always guessing the most common class. A model has to beat this to be useful.

Cleaning steps for the cervical data:
- `?` was converted to missing values. The two "time since diagnosis" columns, which were over 90% empty, were dropped.
- Previous-diagnosis columns (`Dx`, `Dx:Cancer`, `Dx:CIN`, `Dx:HPV`) were removed because they record earlier diagnoses, not risk factors.
- Constant columns were dropped, such as `STDs:AIDS`, which no patient has.
- An `STDs_not_answered` flag was added, because about 12% of patients skipped the whole STD block.
- The screening tests (`Hinselmann`, `Schiller`, `Citology`) were left out of the main model and added in a separate comparison model.

---

## 3. Test-set results

| Cancer | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train F1 |
|---|---|---|---|---|---|---|---|
| Lung | ANN | 0.911 | **1.000** | 0.896 | 0.945 | **0.974** | 0.933 |
| Lung | SVM | 0.893 | 0.957 | 0.917 | 0.936 | 0.945 | 0.981 |
| Lung | Random Forest | 0.911 | 0.957 | **0.938** | **0.947** | 0.965 | 0.997 |
| Breast | ANN | **0.977** | **0.981** | **0.983** | **0.982** | 0.992 | 0.996 |
| Breast | SVM | 0.975 | 0.979 | 0.981 | 0.980 | **0.995** | 0.991 |
| Breast | Random Forest | 0.967 | 0.966 | **0.983** | 0.974 | 0.992 | 0.999 |
| Cervical | ANN | 0.796 | 0.040 | 0.091 | 0.056 | 0.418 | 0.491 |
| Cervical | SVM | 0.868 | **0.133** | **0.182** | **0.154** | **0.643** | 0.417 |
| Cervical | Random Forest | **0.898** | 0.125 | 0.091 | 0.105 | 0.595 | 0.913 |
| Cervical + screening | ANN | 0.934 | 0.500 | 0.727 | 0.593 | 0.917 | 0.762 |
| Cervical + screening | SVM | 0.934 | 0.500 | 0.636 | 0.560 | 0.914 | 0.722 |
| Cervical + screening | Random Forest | **0.982** | **0.833** | **0.909** | **0.870** | **0.933** | 0.954 |

The best value for each cancer type is in bold.

### Best hyperparameters

| Cancer | ANN | SVM | Random Forest |
|---|---|---|---|
| Lung | (100, 50), α = 0.0001 | C = 10, γ = scale | depth = None, split = 2 |
| Breast | (100, 50), α = 0.001 | C = 1, γ = auto | depth = None, split = 5 |
| Cervical | (64, 32), α = 0.0001 | C = 1, γ = scale | depth = 10, split = 2 |
| Cervical + screening | (64, 32), α = 0.0001 | C = 1, γ = scale | depth = 10, split = 2 |

---

## 4. 10-fold cross-validation results (training split)

Values are mean ± standard deviation across the 10 folds.

| Cancer | Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Lung | ANN | 0.859 ± 0.080 | 0.874 ± 0.092 | 0.912 ± 0.052 | 0.898 ± 0.101 |
| Lung | SVM | 0.891 ± 0.065 | 0.942 ± 0.080 | 0.936 ± 0.042 | 0.828 ± 0.143 |
| Lung | Random Forest | 0.900 ± 0.045 | 0.958 ± 0.052 | 0.943 ± 0.027 | 0.910 ± 0.068 |
| Breast | ANN | 0.971 ± 0.007 | 0.976 ± 0.011 | 0.977 ± 0.006 | 0.993 ± 0.004 |
| Breast | SVM | 0.974 ± 0.008 | 0.982 ± 0.007 | 0.980 ± 0.006 | 0.993 ± 0.004 |
| Breast | Random Forest | 0.965 ± 0.005 | 0.977 ± 0.008 | 0.973 ± 0.004 | 0.992 ± 0.002 |
| Cervical | ANN | 0.787 ± 0.052 | 0.335 ± 0.163 | 0.165 ± 0.082 | 0.607 ± 0.105 |
| Cervical | SVM | 0.856 ± 0.023 | 0.220 ± 0.208 | 0.142 ± 0.108 | 0.582 ± 0.172 |
| Cervical | Random Forest | 0.913 ± 0.028 | 0.100 ± 0.166 | 0.119 ± 0.198 | 0.566 ± 0.152 |
| Cervical + screening | ANN | 0.952 ± 0.021 | 0.840 ± 0.179 | 0.696 ± 0.109 | 0.908 ± 0.106 |
| Cervical + screening | SVM | 0.951 ± 0.012 | 0.770 ± 0.171 | 0.662 ± 0.076 | 0.907 ± 0.108 |
| Cervical + screening | Random Forest | 0.954 ± 0.025 | 0.680 ± 0.203 | 0.659 ± 0.148 | 0.927 ± 0.113 |

---

## 5. Most important features (Random Forest)

| Cancer | Top 5 features (importance) |
|---|---|
| Lung | AGE (0.160), ALLERGY (0.121), FATIGUE (0.080), ALCOHOL_CONSUMING (0.077), CHRONIC_DISEASE (0.076) |
| Breast | compactness_mean (0.297), symmetry_mean (0.154), compactness_worst (0.066), perimeter_se (0.060), perimeter_mean (0.051) |
| Cervical | Hormonal_Contraceptives_(years) (0.196), Age (0.156), First_sexual_intercourse (0.155), Number_of_sexual_partners (0.099), Num_of_pregnancies (0.085) |
| Cervical + screening | **Schiller (0.396)**, Hinselmann (0.140), Age (0.076), Citology (0.070), Hormonal_Contraceptives_(years) (0.063) |

Figures: `figures/importance_<cancer>.png`. Confusion matrices are in `figures/cm_<cancer>_<model>.png` and ROC curves in `figures/roc_<cancer>.png`.

---

## 6. Key findings

### 6.1 Breast cancer: all three models perform strongly
- All models reach F1 ≥ 0.974 and ROC-AUC ≥ 0.992. The **ANN** scores highest on test F1 (0.982), and the **SVM** has the highest ROC-AUC (0.995).
- The CV standard deviations are very small (≤ 0.008 for F1) and the CV scores match the test scores closely. This means the results are stable and not the product of a lucky split.
- The differences between the models (under 1 percentage point) are smaller than the CV spread, so none of the three is clearly better than the others.

### 6.2 Lung cancer: good scores, but the evidence is weaker
- **Random Forest** has the best F1 (0.947) and recall (0.938). The **ANN** has the best ROC-AUC (0.974) and made no false positives.
- Only 38 of the 276 patients do not have cancer, so the majority-class baseline is already 0.862. Test accuracy of about 0.91 is only about 5 points above it.
- The test split has only **8 non-cancer cases**, so each one misclassified changes precision and accuracy noticeably. The CV standard deviations (accuracy ± 0.045 to 0.080, ROC-AUC up to ± 0.143) confirm that these estimates are uncertain.
- Random Forest (train F1 0.997) and SVM (0.981) fit the training data almost perfectly, which suggests some overfitting on this small dataset.
- Age is the strongest feature. The remaining symptoms each contribute modestly.

### 6.3 Cervical cancer from risk factors alone: not predictable
- Every model scores **below the majority baseline (0.935) on accuracy**, and the best F1 is only 0.154 (SVM).
- Test ROC-AUC ranges from 0.42 to 0.64 and CV ROC-AUC from 0.57 to 0.61. This is close to random guessing (0.5).
- The best model detects only 2 of the 11 positive cases in the test set.
- Random Forest shows heavy overfitting: train F1 is 0.913 and test F1 is 0.105. It memorises the training data, but nothing it learns carries over to new patients.
- **Finding:** the self-reported behavioural and demographic risk factors in this dataset do not contain enough information to predict a positive biopsy. This limitation comes from the data, and changing the model or tuning would not fix it.

### 6.4 Adding screening tests changes the cervical results completely
- When `Hinselmann`, `Schiller` and `Citology` are added, ROC-AUC rises from about 0.6 to about 0.91–0.93 for all three models. **Random Forest** reaches test F1 0.870 and recall 0.909, detecting 10 of 11 positives with 2 false alarms.
- `Schiller` alone accounts for about 40% of the Random Forest importance, and `Hinselmann` for another 14%.
- **Caution 1:** the test F1 of 0.870 is much higher than the Random Forest CV F1 of 0.659 ± 0.148. With only 11 positives in the test set, the test score is probably optimistic. The CV results suggest that ANN, SVM and Random Forest perform about the same (F1 ≈ 0.66–0.70).
- **Caution 2:** the screening tests are clinical examinations that are often the reason a biopsy is ordered. Most of the improvement therefore comes from information that is already close to a diagnosis. The comparison model is useful as a decision-support tool alongside screening, but it is not an early-risk predictor.

### 6.5 Overall comparison across models
- **No single algorithm is best everywhere.** ANN is best on breast, Random Forest on lung and cervical + screening, and SVM on cervical risk factors. In most cases the gaps fall within the CV standard deviation.
- **Data quality and class balance matter more than the choice of algorithm.** The same three models and grids give F1 of about 0.98 on breast and about 0.15 on cervical risk factors.
- **Accuracy is misleading on imbalanced data.** The cervical Random Forest has 0.898 accuracy but catches only 1 of 11 cancers. This is why F1, recall and ROC-AUC were used as the main metrics.
- Random oversampling inside the pipeline increased recall on the minority class. It did not overcome the lack of signal in the cervical risk-factor features.

---

## 7. Limitations

1. **Small test sets for lung and cervical.** There are 8 negatives in the lung test set and 11 positives in the cervical test set, so single test-split scores have wide uncertainty. CV mean ± std is the more reliable estimate for these datasets.
2. **Lung class imbalance runs in the opposite direction.** About 86% of patients have cancer, which is unlike a real screening population. Scores will not transfer to a general population.
3. **Breast dataset provenance.** The breast file has 5015 rows, while the original Wisconsin dataset has 569. Malignant cases are the majority, and the feature values appear to be already standardised. This suggests the file is an augmented or synthetic version, which may inflate scores and change which features rank highest. The source should be confirmed and stated in the report.
4. **The cervical screening model relies on near-diagnostic tests,** as explained in Section 6.4.
5. **Grid search was small** (4–6 combinations per model), and no external validation dataset was used.
6. Random Forest feature importance measures how the model uses each feature. It does not show that a feature causes cancer.

---

## 8. Conclusion

The same pipeline (imputation, scaling, oversampling, 10-fold CV tuning) was applied to every cancer type. It produced reliable, high-performing classifiers for **breast cancer** (F1 ≈ 0.98, AUC ≈ 0.99) and reasonably good but less certain ones for **lung cancer** (F1 ≈ 0.94, AUC ≈ 0.95–0.97). **Cervical cancer could not be predicted from risk factors alone** (AUC ≈ 0.6). It became predictable (AUC ≈ 0.93) only after screening test results were added, and those tests are themselves part of the diagnostic process. Overall, what each dataset contains decided success more than the choice between ANN, SVM and Random Forest. The trained pipelines are saved in `models/` and served through the Streamlit app (`app.py`).
