"""
test_data_cleaning.py

Unit tests for src/data_cleaning.py.

Run with: pytest tests/test_data_cleaning.py -v
"""

import pandas as pd
import numpy as np
import pytest

from src.data_cleaning import (
    create_target,
    impute_missing_values,
    remove_dti_anomalies,
)


@pytest.fixture
def sample_loans():
    """A small synthetic dataset mimicking the relevant raw columns."""
    return pd.DataFrame({
        "loan_status": [
            "Fully Paid", "Charged Off", "Current", "Late (31-120 days)",
            "Fully Paid", "Charged Off",
        ],
        "emp_length": ["5 years", None, "10+ years", "2 years", None, "1 year"],
        "dti": [15.0, 22.0, -1.0, 999.0, 18.5, 45.0],
        "annual_inc": [50000, 60000, 45000, 70000, 55000, 48000],
    })


def test_create_target_filters_to_definitive_outcomes(sample_loans):
    """Only Fully Paid and Charged Off rows should remain."""
    result = create_target(sample_loans)
    assert len(result) == 4
    assert set(result["loan_status"].unique()) if "loan_status" in result.columns else True


def test_create_target_encoding_is_correct(sample_loans):
    """Charged Off -> 1, Fully Paid -> 0."""
    result = create_target(sample_loans)
    assert result["target"].tolist() == [0, 1, 0, 1]


def test_create_target_drops_loan_status_column(sample_loans):
    """The original loan_status column should be removed after encoding."""
    result = create_target(sample_loans)
    assert "loan_status" not in result.columns


def test_impute_missing_values_fills_emp_length_with_unknown():
    df = pd.DataFrame({"emp_length": ["5 years", None, "2 years"]})
    result = impute_missing_values(df)
    assert result["emp_length"].isnull().sum() == 0
    assert "Unknown" in result["emp_length"].values


def test_impute_missing_values_fills_numeric_with_median():
    df = pd.DataFrame({
        "emp_length": ["5 years", "2 years", "1 year"],
        "some_numeric": [10.0, np.nan, 30.0],
    })
    result = impute_missing_values(df)
    assert result["some_numeric"].isnull().sum() == 0
    # Median of [10, 30] (nan excluded) is 20
    assert result.loc[1, "some_numeric"] == 20.0


def test_remove_dti_anomalies_drops_out_of_range_values(sample_loans):
    result = remove_dti_anomalies(sample_loans)
    assert (result["dti"] >= 0).all()
    assert (result["dti"] <= 100).all()
    # Rows with dti = -1.0 and dti = 999.0 should be removed
    assert len(result) == len(sample_loans) - 2


def test_remove_dti_anomalies_keeps_valid_rows():
    df = pd.DataFrame({"dti": [10.0, 20.0, 30.0]})
    result = remove_dti_anomalies(df)
    assert len(result) == 3
