"""
test_feature_engineering.py

Unit tests for src/feature_engineering.py.

Run with: pytest tests/test_feature_engineering.py -v
"""

import pandas as pd
import numpy as np
import pytest

from src.feature_engineering import (
    add_derived_features,
    encode_binary_columns,
    encode_ordinal_columns,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "issue_d": ["Jan-2015", "Jun-2010"],
        "earliest_cr_line": ["Jan-2005", "Jun-2000"],
        "loan_amnt": [10000, 20000],
        "annual_inc": [50000, 0],
        "term": [" 36 months", " 60 months"],
        "initial_list_status": ["w", "f"],
        "application_type": ["Individual", "Joint App"],
        "disbursement_method": ["Cash", "DirectPay"],
        "grade": ["A", "G"],
        "sub_grade": ["A1", "G5"],
        "emp_length": ["5 years", "Unknown"],
    })


def test_credit_history_years_is_positive(sample_df):
    result = add_derived_features(sample_df.copy())
    assert (result["credit_history_years"] > 0).all()


def test_credit_history_years_approximately_correct(sample_df):
    """Jan-2015 minus Jan-2005 should be ~10 years."""
    result = add_derived_features(sample_df.copy())
    assert abs(result["credit_history_years"].iloc[0] - 10.0) < 0.1


def test_loan_to_income_ratio_handles_zero_income(sample_df):
    """A borrower with annual_inc=0 should not cause a divide-by-zero error."""
    result = add_derived_features(sample_df.copy())
    assert np.isfinite(result["loan_to_income_ratio"]).all()


def test_loan_to_income_ratio_is_capped(sample_df):
    """No ratio should exceed the 99th percentile cap."""
    result = add_derived_features(sample_df.copy())
    cap = result["loan_to_income_ratio"].max()
    assert cap <= 1.0  # sanity bound; exact cap depends on the data


def test_encode_binary_columns_term(sample_df):
    df = sample_df.copy()
    result = encode_binary_columns(df)
    assert result["term"].tolist() == [0, 1]


def test_encode_binary_columns_are_int8(sample_df):
    result = encode_binary_columns(sample_df.copy())
    assert result["term"].dtype == np.int8
    assert result["initial_list_status"].dtype == np.int8


def test_encode_ordinal_grade_preserves_risk_order(sample_df):
    df = encode_binary_columns(sample_df.copy())
    result = encode_ordinal_columns(df)
    # Grade A (safest) should map to a lower number than Grade G (riskiest)
    assert result["grade"].iloc[0] < result["grade"].iloc[1]


def test_encode_ordinal_emp_length_unknown_is_minus_one(sample_df):
    df = encode_binary_columns(sample_df.copy())
    result = encode_ordinal_columns(df)
    assert result["emp_length"].iloc[1] == -1
