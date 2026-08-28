"""
data_loader.py

Handles loading the raw Lending Club dataset from disk.
"""

import pandas as pd

from src.config import RAW_DATA_PATH, REQUIRED_COLUMNS


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw loan-level CSV, restricted to the columns defined
    in config.REQUIRED_COLUMNS.

    Loading only the required columns (rather than all 145 columns
    in the source file) significantly reduces memory usage, which
    matters given the dataset has 2.26M+ rows.

    Parameters
    ----------
    path : str
        Path to the raw loan.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe restricted to the required columns.
    """
    df = pd.read_csv(path, usecols=REQUIRED_COLUMNS, low_memory=False)
    return df
