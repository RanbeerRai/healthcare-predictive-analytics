# Healthcare Predictive Analytics Platform

Most of my hands-on ML work has been text/tabular-security-adjacent stuff through my ConnexPay internship, so I wanted a project in a completely different domain to make sure I actually understood the fundamentals and wasn't just pattern-matching one problem type. Cardiovascular risk prediction on the [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) dataset felt like a good fit — small, clean-ish, well-studied, and a real screening use case where the precision/recall tradeoff actually matters.

## Approach

I loaded the 303 patient records (13 clinical features — age, chest pain type, resting blood pressure, cholesterol, max heart rate, etc.), handled the handful of missing angiography/thallium values, and then benchmarked three models instead of just reaching for whatever's trendiest:

- **Logistic Regression** as an interpretable baseline
- **Random Forest** for nonlinear feature interactions
- **XGBoost**, mostly to see if it actually earns its reputation on a dataset this small

All three went through 5-fold stratified CV with grid-search tuning, then got evaluated on ROC-AUC, PR-AUC, and calibration — not raw accuracy, since a missed at-risk patient and an unnecessary follow-up are not the same kind of mistake.

## Results

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Generalization Gap |
|---|---|---|---|---|
| Logistic Regression | 0.899 | 0.958 | 0.940 | -0.041 |
| **Random Forest (selected)** | 0.893 | **0.964** | 0.958 | -0.021 |
| XGBoost | 0.885 | 0.939 | 0.938 | -0.009 |

Full metrics live in [`results/metrics_summary.json`](results/metrics_summary.json), and the plots (ROC curves, precision-recall curves, calibration curves, confusion matrix) are in [`results/`](results/).

Honestly, I half-expected XGBoost to just win outright — that's the reputation it has. Watching it come in third was a good reminder that "more powerful model" and "better model for this dataset" aren't the same claim, especially once you're down to 303 rows.

## Why I made the choices I made

- **Three models, not just the fancy one.** Benchmarking a plain linear baseline against two tree ensembles tests the "complexity wins" assumption instead of just trusting it.
- **Median imputation over dropping rows.** `ca` and `thal` are missing for maybe 2% of patients. Throwing those rows away shrinks an already-small dataset for basically no upside. Median specifically because both fields are ordinal-coded, not continuous — a fractional mean wouldn't mean anything.
- **Penalizing the generalization gap, not just chasing the best test score.** At this sample size, one lucky test split can flatter an overfit model. I'd rather pick the model that's consistent between train and test than the one that happened to nail this particular 61-patient holdout.
- **ROC-AUC/PR-AUC/calibration over plain accuracy.** A missed at-risk patient and an unnecessary follow-up are not equally bad mistakes, and accuracy pretends they are.

## Structure

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

## What I'd add with more time

Interpretability, mainly — SHAP values on the Random Forest to see which features are actually driving individual predictions, since "trust me, it's 96% ROC-AUC" isn't something a clinician should have to take on faith. I'd also want a second, larger dataset to check whether the model selection even holds up outside this one 303-patient sample.
