"""
explainability.py

Model interpretability using SHAP (SHapley Additive exPlanations).

SHAP values quantify each feature's contribution to individual
predictions, providing both global (which features matter most
overall) and local (why this specific loan was flagged) explanations.
This is especially valuable in credit risk modeling, where regulators
and stakeholders often require justification for automated decisions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_shap_values(model, X_sample: pd.DataFrame):
    """
    Compute SHAP values for a tree-based model (e.g. XGBoost).

    A sample (rather than the full test set) is used because SHAP
    computation is memory- and compute-intensive; a few thousand
    rows is generally sufficient for stable global explanations.

    Parameters
    ----------
    model : trained tree-based model (e.g. XGBClassifier)
    X_sample : pd.DataFrame
        A subset of the feature data to explain.

    Returns
    -------
    shap_values : shap.Explanation
        SHAP values for the provided sample.
    explainer : shap.TreeExplainer
        The fitted explainer (can be reused).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    return shap_values, explainer


def plot_summary(shap_values, X_sample: pd.DataFrame, max_display: int = 15) -> None:
    """
    Global feature importance summary plot (beeswarm).

    Shows, across all sampled predictions, which features have the
    largest impact on the model's output and in which direction
    (e.g. higher DTI pushes predictions toward "default").
    """
    import shap

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance Summary", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_bar_importance(shap_values, max_display: int = 15) -> None:
    """
    Mean absolute SHAP value per feature, as a simple bar chart.
    A simpler, more presentation-friendly alternative to the
    beeswarm summary plot.
    """
    import shap

    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Mean Absolute SHAP Value by Feature", fontsize=14)
    plt.tight_layout()
    plt.show()


def explain_single_prediction(shap_values, index: int = 0) -> None:
    """
    Local explanation (waterfall plot) for a single prediction.

    Shows how each feature pushed that specific loan's predicted
    probability of default above or below the baseline (average)
    prediction.
    """
    import shap

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.title(f"SHAP Explanation for Sample #{index}", fontsize=14)
    plt.tight_layout()
    plt.show()


def run_shap_analysis(model, X_test: pd.DataFrame, sample_size: int = 2000, random_state: int = 42):
    """
    Run a full SHAP analysis: compute values on a sample of the
    test set, then produce global (summary + bar) and one local
    (waterfall) explanation.

    Parameters
    ----------
    model : trained tree-based model
    X_test : pd.DataFrame
        Full test feature set.
    sample_size : int
        Number of test rows to sample for SHAP computation.
    random_state : int
        Seed for reproducible sampling.

    Returns
    -------
    shap_values : shap.Explanation
    X_sample : pd.DataFrame
        The sampled rows the SHAP values correspond to.
    """
    sample_n = min(sample_size, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=random_state)

    shap_values, _ = compute_shap_values(model, X_sample)

    plot_summary(shap_values, X_sample)
    plot_bar_importance(shap_values)
    explain_single_prediction(shap_values, index=0)

    return shap_values, X_sample
