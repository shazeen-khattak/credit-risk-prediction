"""
threshold_tuning.py

Finds the classification threshold that minimizes total estimated
dollar loss, rather than using the default 0.5 cutoff.

Standard classifiers predict a probability and use 0.5 as the
decision boundary by default. In credit risk, this is arbitrary:
the "right" threshold depends on the relative cost of missing a
defaulter (false negative) versus wrongly rejecting a good borrower
(false positive) - which the cost_sensitive_evaluation module shows
are highly asymmetric (~5x). This module searches for the threshold
that minimizes total dollar loss under that cost structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cost_sensitive_evaluation import compute_loan_level_impact


def evaluate_thresholds(
    loan_amnt: pd.Series,
    int_rate: pd.Series,
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute total estimated dollar impact and standard classification
    metrics at each candidate threshold.

    Parameters
    ----------
    loan_amnt, int_rate, y_true : as in cost_sensitive_evaluation.
    y_pred_proba : np.ndarray
        Predicted probability of default for each test loan.
    thresholds : np.ndarray, optional
        Candidate thresholds to evaluate. Defaults to 0.05-0.95 in
        steps of 0.05.

    Returns
    -------
    pd.DataFrame
        One row per threshold, with total dollar impact, recall,
        precision, and selection rate at that cutoff.
    """
    from sklearn.metrics import precision_score, recall_score

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    rows = []
    for t in thresholds:
        y_pred_t = (y_pred_proba >= t).astype(int)

        impact_df = compute_loan_level_impact(loan_amnt, int_rate, y_true, y_pred_t)
        total_impact = impact_df["estimated_impact"].sum()

        recall = recall_score(y_true, y_pred_t, zero_division=0)
        precision = precision_score(y_true, y_pred_t, zero_division=0)
        selection_rate = y_pred_t.mean()

        rows.append({
            "threshold": round(t, 2),
            "total_impact": total_impact,
            "recall": recall,
            "precision": precision,
            "selection_rate": selection_rate,
        })

    return pd.DataFrame(rows)


def find_optimal_threshold(threshold_results: pd.DataFrame) -> dict:
    """Return the row (as a dict) with the maximum total dollar impact."""
    best_row = threshold_results.loc[threshold_results["total_impact"].idxmax()]
    return best_row.to_dict()


def plot_threshold_impact(threshold_results: pd.DataFrame, default_threshold: float = 0.5) -> None:
    """
    Plot total estimated dollar impact against threshold, marking
    both the default (0.5) and the optimal threshold.
    """
    best = find_optimal_threshold(threshold_results)

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_results["threshold"], threshold_results["total_impact"], marker="o")
    plt.axvline(default_threshold, color="gray", linestyle="--", label=f"Default threshold ({default_threshold})")
    plt.axvline(best["threshold"], color="green", linestyle="--", label=f"Optimal threshold ({best['threshold']})")
    plt.title("Total Estimated Dollar Impact by Decision Threshold", fontsize=14)
    plt.xlabel("Classification Threshold")
    plt.ylabel("Total Estimated Impact ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def run_threshold_tuning(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Run the full threshold tuning analysis: evaluate a grid of
    thresholds, report the business-optimal one, and plot the curve.

    Assumes X_test contains 'loan_amnt' and 'int_rate' in their
    original units (as produced by this project's pipeline).
    """
    threshold_results = evaluate_thresholds(
        loan_amnt=X_test["loan_amnt"],
        int_rate=X_test["int_rate"],
        y_true=y_test,
        y_pred_proba=y_pred_proba,
    )

    print("=" * 60)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 60)
    print(threshold_results.to_string(index=False))

    best = find_optimal_threshold(threshold_results)
    default_row = threshold_results.iloc[(threshold_results["threshold"] - 0.5).abs().idxmin()]

    print("\n" + "=" * 60)
    print("OPTIMAL VS. DEFAULT THRESHOLD")
    print("=" * 60)
    print(f"Default threshold (0.5): total impact = ${default_row['total_impact']:,.0f}")
    print(f"Optimal threshold ({best['threshold']}): total impact = ${best['total_impact']:,.0f}")
    print(f"Improvement from tuning: ${best['total_impact'] - default_row['total_impact']:,.0f}")

    plot_threshold_impact(threshold_results)

    return {"threshold_results": threshold_results, "best_threshold": best}
