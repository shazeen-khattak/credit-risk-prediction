"""
model_training.py

Train/test splitting, baseline model training, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from src.config import TEST_SIZE, RANDOM_STATE


def split_features_target(df: pd.DataFrame, target_col: str = "target"):
    """Separate features (X) from the target variable (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def sample_data(df: pd.DataFrame, sample_size: int, target_col: str = "target") -> pd.DataFrame:
    """
    Draw a stratified random sample from the full dataset.

    Used to reduce memory/compute requirements on constrained
    environments (e.g. Colab free tier), while preserving the
    original class distribution (~80% paid, ~20% default). This is
    a standard practice when working with large-scale datasets under
    hardware constraints, and does not compromise the validity of
    the analysis as long as the sample is representative.

    Parameters
    ----------
    df : pd.DataFrame
        Full cleaned/engineered dataframe (including target column).
    sample_size : int
        Number of rows to sample. If greater than or equal to the
        dataset size, the full dataframe is returned unchanged.
    target_col : str
        Name of the target column, used to preserve class balance.

    Returns
    -------
    pd.DataFrame
        A stratified sample of the requested size.
    """
    if sample_size >= len(df):
        return df

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df[target_col],
        random_state=RANDOM_STATE,
    )
    return sampled_df.reset_index(drop=True)


def train_test_split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train/test sets.

    stratify=y preserves the ~80/20 class distribution in both
    the training and test sets, which matters given the class
    imbalance in this dataset.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a baseline Logistic Regression model.

    - Features are standardized, since Logistic Regression is
      sensitive to feature scale.
    - class_weight='balanced' compensates for the ~80/20 class
      imbalance without needing synthetic oversampling (e.g. SMOTE).
    - The 'saga' solver is used for its efficiency on large datasets.

    Returns
    -------
    model : LogisticRegression
        The trained model.
    scaler : StandardScaler
        The fitted scaler (needed to transform the test set identically).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        solver="saga",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate a trained model on the test set.

    Prints the classification report, ROC-AUC score, and confusion
    matrix. ROC-AUC is used as the primary metric since accuracy
    alone is misleading on imbalanced data.

    Note: scaler may be None for tree-based models (e.g. XGBoost)
    that do not require feature scaling.
    """
    X_test_input = scaler.transform(X_test).astype(np.float32) if scaler is not None else X_test

    y_pred = model.predict(X_test_input)
    y_pred_proba = model.predict_proba(X_test_input)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_pred_proba)
    print("ROC-AUC Score:", auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {"y_pred": y_pred, "y_pred_proba": y_pred_proba, "roc_auc": auc}


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train an XGBoost classifier.

    Memory-efficiency choices for constrained environments
    (e.g. Colab free tier):
    - tree_method='hist': histogram-based split finding, far more
      memory-efficient than the default exact method on large data.
    - max_bin=128: fewer histogram bins than the default (256),
      trading a small amount of precision for lower memory use.
    - No feature scaling needed (tree-based models are scale-invariant),
      so no scaler is fit or returned here.

    scale_pos_weight compensates for the ~80/20 class imbalance,
    analogous to class_weight='balanced' in Logistic Regression.

    Returns
    -------
    model : XGBClassifier
        The trained model.
    """
    from xgboost import XGBClassifier

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        tree_method="hist",
        max_bin=128,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


def save_model_artifacts(model, feature_columns: list, output_dir: str = "model_artifacts") -> None:
    """
    Save a trained model and its expected feature column order to
    disk, so they can be loaded later (e.g. by the Streamlit
    dashboard in app.py) without retraining.

    The model is saved using XGBoost's native format (.json) rather
    than pickle. Native format is designed for cross-version
    compatibility; pickle can fail to load ("input stream corrupted")
    when the XGBoost version used to load it differs from the
    version used to train it (e.g. training on Colab, loading
    locally with a newer/older XGBoost release).
    """
    import os
    import pickle

    os.makedirs(output_dir, exist_ok=True)

    model.save_model(os.path.join(output_dir, "xgb_model.json"))

    with open(os.path.join(output_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"Model artifacts saved to '{output_dir}/'")
