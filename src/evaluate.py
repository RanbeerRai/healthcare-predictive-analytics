"""Evaluation utilities: ROC-AUC, precision-recall, calibration, and
a generalization check used to pick the final model.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def score_model(fitted_search, X_train, y_train, X_test, y_test) -> dict:
    best_model = fitted_search.best_estimator_

    train_proba = best_model.predict_proba(X_train)[:, 1]
    test_proba = best_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    return {
        "best_params": fitted_search.best_params_,
        "cv_roc_auc": fitted_search.best_score_,
        "train_roc_auc": train_auc,
        "test_roc_auc": test_auc,
        # A big train/test gap flags overfitting even when the raw
        # test score looks fine -- this is what actually drove model
        # selection below, not just the highest test AUC in isolation.
        "generalization_gap": train_auc - test_auc,
        "test_pr_auc": average_precision_score(y_test, test_proba),
        "test_brier_score": brier_score_loss(y_test, test_proba),
        "test_proba": test_proba,
    }


def select_best_generalizing_model(results: dict) -> str:
    """Rank by test ROC-AUC, but penalize models with a large
    train/test gap so an overfit model doesn't win on a lucky test
    split.
    """
    def rank_key(name):
        r = results[name]
        return r["test_roc_auc"] - max(0.0, r["generalization_gap"] - 0.03)

    return max(results, key=rank_key)


def plot_roc_curves(fitted_searches: dict, X_test, y_test, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, search in fitted_searches.items():
        RocCurveDisplay.from_estimator(search.best_estimator_, X_test, y_test, ax=ax, name=name)
    ax.set_title("ROC Curves — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision_recall(fitted_searches: dict, X_test, y_test, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, search in fitted_searches.items():
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(fitted_searches: dict, X_test, y_test, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, search in fitted_searches.items():
        proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(fitted_search, X_test, y_test, out_path: str, threshold: float = 0.5):
    proba = fitted_search.best_estimator_.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Risk", "At Risk"])
    ax.set_yticklabels(["No Risk", "At Risk"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
