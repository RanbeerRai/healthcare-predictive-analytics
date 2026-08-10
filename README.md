# Healthcare Predictive Analytics Platform

Cardiovascular risk classification pipeline benchmarking Logistic Regression, Random Forest, and XGBoost on the [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) dataset.

## What this does

1. Loads 303 patient records (13 clinical features: age, sex, chest pain type, resting blood pressure, cholesterol, resting ECG, max heart rate, exercise-induced angina, etc.)
2. Median-imputes a small number of missing values (angiography/thallium-scan results not recorded for every patient), then standardizes features
3. Benchmarks **Logistic Regression**, **Random Forest**, and **XGBoost** under 5-fold stratified cross-validation with grid-search hyperparameter tuning
4. Evaluates on ROC-AUC, precision-recall AUC, and calibration (Brier score) — not raw accuracy, since this is a screening context
5. Selects the final model by penalizing the train/test generalization gap, not just picking the highest raw test score

## Results (this run)

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Generalization Gap |
|---|---|---|---|---|
| Logistic Regression | 0.899 | 0.958 | 0.940 | -0.041 |
| **Random Forest (selected)** | 0.893 | **0.964** | 0.958 | -0.021 |
| XGBoost | 0.885 | 0.939 | 0.938 | -0.009 |

Full metrics: [`results/metrics_summary.json`](results/metrics_summary.json). Plots: ROC curves, precision-recall curves, calibration curves, and the winning model's confusion matrix are in [`results/`](results/).

## Key design decisions (for interview walkthroughs)

- **Why three model families, not just XGBoost?** XGBoost usually wins on tabular data, but with only 303 rows, more model complexity isn't automatically better — benchmarking a simple linear baseline alongside two ensemble methods tests that assumption instead of taking it on faith. In this run, XGBoost actually came in third.
- **Why median imputation instead of dropping rows?** The missing values (in `ca` and `thal`) affect only ~2% of rows. Dropping them would shrink an already-small dataset for no real benefit. Median (not mean) because both fields are ordinal/categorical-coded, not continuous.
- **Why penalize the generalization gap during model selection?** On a 303-row dataset, the single test-set ROC-AUC can be somewhat lucky. Selecting purely on test score risks picking a model that overfit the training folds and happened to also do well on this particular test split. Ranking by test score minus a generalization-gap penalty is a more defensible selection rule.
- **Why ROC-AUC/PR-AUC/calibration instead of accuracy?** In a risk-screening context, false negatives (missed at-risk patients) and false positives (unnecessary follow-up) have very different real costs, and accuracy hides that tradeoff entirely.

## Project structure

```
├── data/heart_disease.csv          # Raw UCI dataset
├── src/
│   ├── preprocessing.py            # Load, impute, scale, split
│   ├── models.py                   # Grid-search definitions for all 3 models
│   └── evaluate.py                 # Metrics + plotting utilities
├── notebooks/
│   └── healthcare_predictive_analytics.ipynb   # Full walkthrough with real outputs
├── run_pipeline.py                 # End-to-end script (same pipeline, no notebook)
└── results/                        # Generated plots + metrics_summary.json
```

## Running it

```bash
pip install -r requirements.txt
python run_pipeline.py
# or open notebooks/healthcare_predictive_analytics.ipynb
```
