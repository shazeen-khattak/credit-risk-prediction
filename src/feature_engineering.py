"""
feature_engineering.py

Creates derived features and encodes categorical variables for
the credit risk model.
"""

import numpy as np
import pandas as pd

from src.config import GRADE_MAP, EMP_LENGTH_MAP, BINARY_MAPS, ONE_HOT_COLUMNS


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived numeric features from raw date and financial columns.

    - credit_history_years: length of credit history at loan issuance,
      computed as (issue date - earliest credit line date).
    - loan_to_income_ratio: loan amount relative to annual income,
      capped at the 99th percentile to limit the influence of
      extreme outliers (e.g. near-zero reported income).
    - issue_year: calendar year the loan was issued, useful for
      time-based EDA.
    """
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")

    df["issue_year"] = df["issue_d"].dt.year.astype(np.int16)
    df["credit_history_years"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    ).astype(np.float32)

    df["loan_to_income_ratio"] = (
        df["loan_amnt"] / (df["annual_inc"] + 1)
    ).astype(np.float32)
    cap_value = df["loan_to_income_ratio"].quantile(0.99)
    df["loan_to_income_ratio"] = df["loan_to_income_ratio"].clip(upper=cap_value)

    df = df.drop(columns=["issue_d", "earliest_cr_line"])
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Encode simple binary categorical columns as 0/1 integers."""
    df["term"] = df["term"].apply(lambda x: 1 if "60" in x else 0).astype(np.int8)

    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping).astype(np.int8)

    return df


def encode_ordinal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns that have a natural order
    (grade, sub_grade, emp_length) using explicit integer mappings.
    """
    df["grade"] = df["grade"].map(GRADE_MAP).astype(np.int8)

    sub_grade_order = sorted(df["sub_grade"].unique())
    sub_grade_map = {val: idx for idx, val in enumerate(sub_grade_order)}
    df["sub_grade"] = df["sub_grade"].map(sub_grade_map).astype(np.int8)

    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP).astype(np.int8)

    return df


def encode_nominal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode nominal (unordered) categorical columns.

    - Low-cardinality columns (home_ownership, verification_status,
      purpose) use one-hot encoding.
    - 'addr_state' (51 categories) uses frequency encoding instead of
      one-hot encoding, to avoid creating 50+ additional columns.
    """
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True)

    state_freq = df["addr_state"].value_counts(normalize=True)
    df["addr_state"] = df["addr_state"].map(state_freq).astype(np.float32)

    return df


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to reduce memory usage."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in sequence."""
    df = add_derived_features(df)
    df = encode_binary_columns(df)
    df = encode_ordinal_columns(df)
    df = encode_nominal_columns(df)
    df = downcast_floats(df)
    return df
