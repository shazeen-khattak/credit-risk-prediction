"""
data_cleaning.py

Handles target variable creation, missing value imputation, and
outlier removal for the Lending Club dataset.
"""

import numpy as np
import pandas as pd

from src.config import VALID_LOAN_STATUSES, REDUNDANT_COLUMNS


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict the dataset to loans with a definitive outcome and
    create a binary target column.

    Loans still in progress (e.g. "Current", "Late", "In Grace
    Period") are excluded because their final outcome is unknown.

    target = 1 -> loan was charged off (defaulted)
    target = 0 -> loan was fully paid
    """
    df = df[df["loan_status"].isin(VALID_LOAN_STATUSES)].copy()
    df["target"] = (df["loan_status"] == "Charged Off").astype(np.int8)
    df = df.drop(columns=["loan_status"])
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns known to add no predictive value (e.g. constants)."""
    cols_present = [c for c in REDUNDANT_COLUMNS if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values.

    - 'emp_length' (categorical): missing values are filled with an
      explicit "Unknown" category, since a missing employment length
      may itself carry predictive signal (e.g. unemployed applicants).
    - All other numeric columns: filled with the column median, which
      is robust to the outliers common in financial data.
    """
    df["emp_length"] = df["emp_length"].fillna("Unknown")

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    cols_with_na = [c for c in numeric_cols if df[c].isnull().sum() > 0]
    for col in cols_with_na:
        df[col] = df[col].fillna(df[col].median())

    return df


def remove_dti_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with invalid debt-to-income (DTI) values.

    The raw data contains placeholder/error values such as 999
    (an impossible DTI) and negative values. These affect a very
    small fraction of rows and are dropped rather than imputed.
    """
    return df[(df["dti"] >= 0) & (df["dti"] <= 100)]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in sequence."""
    df = create_target(df)
    df = drop_redundant_columns(df)
    df = impute_missing_values(df)
    df = remove_dti_anomalies(df)
    return df
