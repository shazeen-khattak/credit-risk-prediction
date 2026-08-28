"""
fairness_audit.py

Checks whether the model's errors and predictions are distributed
unevenly across borrower groups (e.g. by home ownership status).

In lending, regulatory frameworks (e.g. the Equal Credit Opportunity
Act in the US) require that credit decisions not disproportionately
disadvantage particular groups. This module reports standard
group-fairness metrics so that any such disparity is visible and can
be discussed, rather than silently present in the model.

Note: this dataset does not include protected attributes such as
race or gender, so this audit uses home ownership status (a
legitimate, available field) as a proxy group to demonstrate the
methodology. The same functions apply to any categorical grouping.
"""

import numpy as np
import pandas as pd


def reconstruct_group_from_dummies(X: pd.DataFrame, prefix: str) -> pd.Series:
    """
    Reconstruct a single categorical column from one-hot encoded
    dummy columns produced by pd.get_dummies (e.g. 'home_ownership_RENT',
    'home_ownership_OWN', ... -> a single 'home_ownership' column).

    Rows where all dummy columns are 0 are assigned the dropped
    reference category (drop_first=True was used during encoding),
    labeled 'OTHER_OR_REFERENCE'.
    """
    dummy_cols = [c for c in X.columns if c.startswith(prefix + "_")]
    group = pd.Series("OTHER_OR_REFERENCE", index=X.index, dtype=object)

    for col in dummy_cols:
        category = col[len(prefix) + 1:]
        group[X[col] == 1] = category

    return group


def compute_group_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    group: pd.Series,
) -> pd.DataFrame:
    """
    Compute standard fairness metrics for each group:

    - selection_rate: fraction of the group predicted as default (1).
      Large differences here indicate disparate impact.
    - false_positive_rate: among truly non-defaulting borrowers in
      the group, the fraction incorrectly flagged as default.
      High FPR for a group means that group's good borrowers are
      disproportionately denied/penalized.
    - false_negative_rate: among truly defaulting borrowers, the
      fraction missed by the model.
    - avg_predicted_risk: mean predicted probability of default,
      useful for spotting systematic score shifts between groups.
    """
    df = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "group": group.values,
    })

    rows = []
    for grp_name, grp_df in df.groupby("group"):
        n = len(grp_df)
        selection_rate = grp_df["y_pred"].mean()
        avg_risk = grp_df["y_pred_proba"].mean()

        actual_negatives = grp_df[grp_df["y_true"] == 0]
        actual_positives = grp_df[grp_df["y_true"] == 1]

        fpr = (actual_negatives["y_pred"] == 1).mean() if len(actual_negatives) > 0 else np.nan
        fnr = (actual_positives["y_pred"] == 0).mean() if len(actual_positives) > 0 else np.nan

        rows.append({
            "group": grp_name,
            "n": n,
            "selection_rate": selection_rate,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "avg_predicted_risk": avg_risk,
        })

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def summarize_disparity(metrics_df: pd.DataFrame) -> None:
    """
    Print the largest gap between groups for each fairness metric,
    as a quick "is there a problem here?" signal.

    A common rule of thumb (the "80% rule" / four-fifths rule) flags
    disparate impact when the selection rate of any group falls
    below 80% of the highest-selection-rate group.
    """
    print("=" * 60)
    print("FAIRNESS METRICS BY GROUP")
    print("=" * 60)
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("DISPARITY SUMMARY")
    print("=" * 60)

    for metric in ["selection_rate", "false_positive_rate", "false_negative_rate"]:
        max_val = metrics_df[metric].max()
        min_val = metrics_df[metric].min()
        gap = max_val - min_val
        print(f"{metric}: max={max_val:.3f}, min={min_val:.3f}, gap={gap:.3f}")

    # Four-fifths rule check on selection rate
    max_selection = metrics_df["selection_rate"].max()
    threshold = 0.8 * max_selection
    flagged = metrics_df[metrics_df["selection_rate"] < threshold]

    print(f"\nFour-fifths rule threshold (80% of max selection rate): {threshold:.3f}")
    if len(flagged) > 0:
        print("Groups falling below this threshold (potential disparate impact):")
        print(flagged[["group", "selection_rate"]].to_string(index=False))
    else:
        print("No groups fall below the four-fifths threshold on selection rate.")


def run_fairness_audit(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    group_prefix: str = "home_ownership",
) -> pd.DataFrame:
    """
    Run a full fairness audit grouped by the given categorical
    feature (reconstructed from its one-hot encoded columns).

    Parameters
    ----------
    X_test : pd.DataFrame
        Test feature set (must contain the one-hot dummy columns
        for `group_prefix`, e.g. 'home_ownership_RENT').
    y_test : pd.Series
        True labels.
    y_pred : np.ndarray
        Predicted labels (0/1) from the model.
    y_pred_proba : np.ndarray
        Predicted probabilities of default from the model.
    group_prefix : str
        Prefix of the one-hot encoded group columns to audit
        (e.g. 'home_ownership').

    Returns
    -------
    pd.DataFrame
        Per-group fairness metrics.
    """
    group = reconstruct_group_from_dummies(X_test, group_prefix)
    metrics_df = compute_group_metrics(y_test, y_pred, y_pred_proba, group)
    summarize_disparity(metrics_df)
    return metrics_df
