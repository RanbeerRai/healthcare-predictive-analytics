"""End-to-end run: preprocess -> tune 3 models -> evaluate -> pick winner.

Run from the project root: python run_pipeline.py
"""

import json
import os

from src.evaluate import (
    plot_calibration,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curves,
    score_model,
    select_best_generalizing_model,
)
from src.models import fit_all, get_search_grids
from src.preprocessing import load_and_split

os.makedirs("results", exist_ok=True)


def main():
    X_train, X_test, y_train, y_test, imputer, scaler = load_and_split()
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}, Test positive rate: {y_test.mean():.3f}")

    grids = get_search_grids()
    fitted = fit_all(grids, X_train, y_train)

    results = {}
    for name, search in fitted.items():
        r = score_model(search, X_train, y_train, X_test, y_test)
        results[name] = r
        print(f"\n{name}")
        print(f"  Best params: {r['best_params']}")
        print(f"  CV ROC-AUC (train folds): {r['cv_roc_auc']:.4f}")
        print(f"  Train ROC-AUC: {r['train_roc_auc']:.4f}")
        print(f"  Test ROC-AUC: {r['test_roc_auc']:.4f}")
        print(f"  Generalization gap (train-test): {r['generalization_gap']:.4f}")
        print(f"  Test PR-AUC: {r['test_pr_auc']:.4f}")
        print(f"  Test Brier score: {r['test_brier_score']:.4f}")

    winner = select_best_generalizing_model(results)
    print(f"\nSelected model (best generalization, not just highest raw test AUC): {winner}")

    plot_roc_curves(fitted, X_test, y_test, "results/roc_curves.png")
    plot_precision_recall(fitted, X_test, y_test, "results/precision_recall_curves.png")
    plot_calibration(fitted, X_test, y_test, "results/calibration_curves.png")
    plot_confusion_matrix(fitted[winner], X_test, y_test, "results/confusion_matrix_winner.png")

    summary = {
        name: {k: v for k, v in r.items() if k != "test_proba"}
        for name, r in results.items()
    }
    summary["winner"] = winner
    with open("results/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSaved plots and results/metrics_summary.json")


if __name__ == "__main__":
    main()
