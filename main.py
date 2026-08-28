"""
main.py

Runs the full Credit Risk Prediction pipeline end-to-end:
    1. Load raw data
    2. Clean data (target creation, missing values, outliers)
    3. Engineer features (derived features + encoding)
    4. Split into train/test sets
    5. Train a baseline Logistic Regression model
    6. Evaluate on the test set

Usage (in Google Colab):
    from google.colab import drive
    drive.mount('/content/drive')

    import sys
    sys.path.append('/content/drive/MyDrive/credit-risk-project')

    %run main.py
"""

from src.data_loader import load_raw_data
from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.model_training import (
    split_features_target,
    train_test_split_data,
    train_logistic_regression,
    evaluate_model,
    sample_data,
)


def run_pipeline(sample_size: int = None):
    """
    Run the full pipeline.

    Parameters
    ----------
    sample_size : int, optional
        If provided, the cleaned/engineered dataset is downsampled
        (using stratified sampling) to this many rows before
        modeling. Useful on memory-constrained environments (e.g.
        Colab free tier) where the full ~1.3M-row dataset causes
        the session to crash during model training. Class balance
        is preserved. If None, the full dataset is used.
    """
    print("Step 1: Loading raw data...")
    df = load_raw_data()
    print(f"  Loaded shape: {df.shape}")

    print("\nStep 2: Cleaning data...")
    df = clean_data(df)
    print(f"  Shape after cleaning: {df.shape}")

    print("\nStep 3: Engineering features...")
    df = engineer_features(df)
    print(f"  Final shape: {df.shape}")
    print(f"  Missing values remaining: {df.isnull().sum().sum()}")

    if sample_size is not None:
        print(f"\nStep 3b: Sampling data down to {sample_size} rows...")
        df = sample_data(df, sample_size)
        print(f"  Sampled shape: {df.shape}")
        print(f"  Target distribution (%):\n{df['target'].value_counts(normalize=True) * 100}")

    print("\nStep 4: Splitting into train/test sets...")
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")

    print("\nStep 5: Training baseline Logistic Regression model...")
    model, scaler = train_logistic_regression(X_train, y_train)

    print("\nStep 6: Evaluating model on test set...")
    results = evaluate_model(model, scaler, X_test, y_test)

    return {
        "df": df,
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
    }


if __name__ == "__main__":
    # Default to a 300,000-row sample to keep this runnable on
    # memory-constrained environments. Pass sample_size=None to
    # use the full dataset (requires more RAM, e.g. Colab Pro).
    run_pipeline(sample_size=300_000)
