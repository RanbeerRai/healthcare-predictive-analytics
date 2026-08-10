"""Model definitions and grid-search hyperparameter tuning.

Three model families are benchmarked, chosen to span a
linear/interpretable baseline, a bagged tree ensemble, and a boosted
tree ensemble:

- Logistic Regression: interpretable baseline, coefficients map
  directly to clinical risk factors.
- Random Forest: captures nonlinear feature interactions without
  much tuning effort; robust to the dataset's modest size.
- XGBoost: usually the strongest tabular-data performer, included
  to see whether the extra model complexity is actually justified
  on only ~300 rows (spoiler: with this little data, it isn't
  guaranteed to beat the simpler models -- that's the point of
  benchmarking all three instead of assuming the "fanciest" model wins).
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

CV_FOLDS = 5


def get_search_grids():
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    logistic = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid={
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs"],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    random_forest = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid={
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 2, 4],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    xgboost = GridSearchCV(
        estimator=XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
        },
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }


def fit_all(models: dict, X_train, y_train) -> dict:
    fitted = {}
    for name, search in models.items():
        search.fit(X_train, y_train)
        fitted[name] = search
    return fitted
