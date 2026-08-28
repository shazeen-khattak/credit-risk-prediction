"""
cost_sensitive_evaluation.py

Translates model predictions into estimated dollar impact, using a
simplified lending profit/loss model. This reframes standard
classification metrics (precision, recall, ROC-AUC) in terms a
business stakeholder cares about: money.

Simplified cost model (per loan, based on confusion matrix outcome):

- True Negative  (predicted paid, actually paid):
    Loan is issued and repaid -> bank earns the interest income.
    Approximated as loan_amnt * int_rate.

- True Positive  (predicted default, actually defaulted):
    Loan is not issued (correctly avoided) -> no loss, no gain.
    This is the "correct catch" and has zero cost/benefit in this
    simplified model (the loan simply doesn't happen).

- False Negative (predicted paid, actually defaulted):
    Loan is issued but borrower defaults -> bank loses the
    principal (a common simplifying assumption; in reality some
    partial recovery occurs, which this model ignores for
    conservatism/simplicity). Approximated as -loan_amnt.

- False Positive (predicted default, actually paid):
    Loan is denied to a borrower who would have repaid -> bank
    loses the interest income it would have earned (opportunity
    cost), not the principal. Approximated as -(loan_amnt * int_rate).

This is a simplification (e.g. it ignores partial recovery on
defaults, fixed servicing costs, and time value of money) but is
sufficient to illustrate the relative business cost of different
error types and to compare models/thresholds on a common, dollar
denominated basis.
"""

import numpy as np
import pandas as pd


def compute_loan_level_impact(
    loan_amnt: pd.Series,
    int_rate: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the estimated dollar impact of each loan's prediction
    outcome, per the simplified cost model described in the module
    docstring.

    int_rate is expected as a percentage (e.g. 12.5 for 12.5%),
    consistent with the raw Lending Club field.
    """
    df = pd.DataFrame({
        "loan_amnt": loan_amnt.values,
        "int_rate": int_rate.values,
        "y_true": y_true.values,
        "y_pred": y_pred,
    })

    rate_fraction = df["int_rate"] / 100.0

    conditions = [
        (df["y_true"] == 0) & (df["y_pred"] == 0),  # True Negative
        (df["y_true"] == 1) & (df["y_pred"] == 1),  # True Positive
        (df["y_true"] == 1) & (df["y_pred"] == 0),  # False Negative
        (df["y_true"] == 0) & (df["y_pred"] == 1),  # False Positive
    ]
    labels = ["true_negative", "true_positive", "false_negative", "false_positive"]
    df["outcome"] = np.select(conditions, labels, default="unknown")

    impact = np.select(
        conditions,
        [
            df["loan_amnt"] * rate_fraction,      # TN: earn interest
            0.0,                                    # TP: loan avoided, no P&L
            -df["loan_amnt"],                       # FN: lose principal
            -(df["loan_amnt"] * rate_fraction),     # FP: lose interest opportunity
        ],
        default=0.0,
    )
    df["estimated_impact"] = impact

    return df


def summarize_business_impact(impact_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate loan-level impact into a summary table by outcome
    type, plus an overall total.
    """
    summary = (
        impact_df.groupby("outcome")["estimated_impact"]
        .agg(count="count", total_impact="sum", avg_impact_per_loan="mean")
        .reset_index()
    )

    total_row = pd.DataFrame([{
        "outcome": "TOTAL",
        "count": len(impact_df),
        "total_impact": impact_df["estimated_impact"].sum(),
        "avg_impact_per_loan": impact_df["estimated_impact"].mean(),
    }])

    return pd.concat([summary, total_row], ignore_index=True)


def compare_to_baseline_strategies(impact_df: pd.DataFrame, loan_amnt: pd.Series, int_rate: pd.Series, y_true: pd.Series) -> pd.DataFrame:
    """
    Compare the model's total estimated impact against two naive
    baseline strategies, to show the value the model adds:

    - "Approve All": issue every loan regardless of risk.
    - "Reject All": issue no loans at all (zero risk, zero return).
    """
    rate_fraction = int_rate / 100.0

    # Approve All: every loan issued. Paid loans earn interest,
    # defaulted loans lose principal.
    approve_all_impact = np.where(
        y_true == 0,
        loan_amnt * rate_fraction,
        -loan_amnt,
    ).sum()

    # Reject All: no loans issued, no profit and no loss.
    reject_all_impact = 0.0

    model_impact = impact_df["estimated_impact"].sum()

    comparison = pd.DataFrame([
        {"strategy": "Reject All Loans", "total_impact": reject_all_impact},
        {"strategy": "Approve All Loans", "total_impact": approve_all_impact},
        {"strategy": "Model-Based Decisions", "total_impact": model_impact},
    ])

    return comparison


def run_cost_sensitive_evaluation(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """
    Run the full cost-sensitive evaluation: per-outcome dollar
    summary, plus comparison against naive baseline strategies.

    Assumes X_test contains 'loan_amnt' and 'int_rate' columns in
    their original (pre-scaling) units, as produced by this
    project's feature_engineering pipeline.
    """
    impact_df = compute_loan_level_impact(
        loan_amnt=X_test["loan_amnt"],
        int_rate=X_test["int_rate"],
        y_true=y_test,
        y_pred=y_pred,
    )

    summary = summarize_business_impact(impact_df)
    print("=" * 60)
    print("ESTIMATED DOLLAR IMPACT BY PREDICTION OUTCOME")
    print("=" * 60)
    print(summary.to_string(index=False))

    comparison = compare_to_baseline_strategies(
        impact_df, X_test["loan_amnt"], X_test["int_rate"], y_test
    )
    print("\n" + "=" * 60)
    print("MODEL VS. NAIVE BASELINE STRATEGIES")
    print("=" * 60)
    print(comparison.to_string(index=False))

    return {"impact_df": impact_df, "summary": summary, "comparison": comparison}
