"""
eda.py

Exploratory Data Analysis functions for the credit risk dataset.
Each function generates one visualization plus prints the
underlying summary statistics, so the insight is available in
both chart and tabular form.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_default_rate_by_grade(df: pd.DataFrame) -> None:
    """
    Bar chart of default rate by loan grade (A = safest, G = riskiest).
    Confirms whether Lending Club's internal risk grading correlates
    with actual default outcomes.
    """
    default_rate_by_grade = df.groupby("grade")["target"].mean().sort_index() * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=default_rate_by_grade.index,
        y=default_rate_by_grade.values,
        hue=default_rate_by_grade.index,
        palette="RdYlGn_r",
        legend=False,
    )
    plt.title("Default Rate by Loan Grade", fontsize=14)
    plt.xlabel("Loan Grade (0=A ... 6=G)")
    plt.ylabel("Default Rate (%)")
    plt.show()

    print("Default Rate by Grade (%):")
    print(default_rate_by_grade)


def plot_interest_rate_by_outcome(df: pd.DataFrame) -> None:
    """
    Boxplot comparing interest rate distributions between fully paid
    and charged-off loans.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="target", y="int_rate", data=df, hue="target", palette="Set2", legend=False)
    plt.title("Interest Rate Distribution: Fully Paid vs Charged Off", fontsize=14)
    plt.xlabel("Loan Outcome (0 = Fully Paid, 1 = Charged Off)")
    plt.ylabel("Interest Rate (%)")
    plt.show()

    print("Interest Rate Statistics by Loan Outcome:")
    print(df.groupby("target")["int_rate"].describe())


def plot_dti_by_outcome(df: pd.DataFrame) -> None:
    """
    Boxplot comparing debt-to-income ratio distributions between
    fully paid and charged-off loans.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="target", y="dti", data=df, hue="target", palette="Set2", legend=False)
    plt.title("Debt-to-Income Ratio: Fully Paid vs Charged Off", fontsize=14)
    plt.xlabel("Loan Outcome (0 = Fully Paid, 1 = Charged Off)")
    plt.ylabel("DTI (%)")
    plt.ylim(0, 50)
    plt.show()

    print("DTI Statistics by Loan Outcome:")
    print(df.groupby("target")["dti"].describe())


def plot_income_by_outcome(df: pd.DataFrame) -> None:
    """
    Boxplot comparing annual income distributions between fully paid
    and charged-off loans.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="target", y="annual_inc", data=df, hue="target", palette="Set2", legend=False)
    plt.title("Annual Income: Fully Paid vs Charged Off", fontsize=14)
    plt.xlabel("Loan Outcome (0 = Fully Paid, 1 = Charged Off)")
    plt.ylabel("Annual Income ($)")
    plt.ylim(0, 200000)
    plt.show()

    print("Annual Income Statistics by Loan Outcome:")
    print(df.groupby("target")["annual_inc"].describe())


def plot_default_rate_by_purpose(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of default rate by loan purpose
    (e.g. small business, debt consolidation, wedding).
    """
    default_rate_by_purpose = (
        df.groupby("purpose")["target"].mean().sort_values(ascending=False) * 100
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=default_rate_by_purpose.values,
        y=default_rate_by_purpose.index,
        hue=default_rate_by_purpose.index,
        palette="RdYlGn_r",
        legend=False,
    )
    plt.title("Default Rate by Loan Purpose", fontsize=14)
    plt.xlabel("Default Rate (%)")
    plt.ylabel("Loan Purpose")
    plt.show()

    print("Default Rate by Purpose (%):")
    print(default_rate_by_purpose)


def plot_default_rate_by_year(df: pd.DataFrame) -> None:
    """
    Line chart of default rate by loan issue year.

    Note: recent vintages (last 2-3 years in the dataset) tend to
    show inflated default rates due to right-censoring bias -
    defaults occur early in a loan's life, while fully-paid status
    requires the full loan term to elapse, so recent cohorts
    disproportionately reflect early defaulters.
    """
    default_rate_by_year = df.groupby("issue_year")["target"].mean() * 100

    plt.figure(figsize=(12, 6))
    default_rate_by_year.plot(kind="line", marker="o", color="darkred")
    plt.title("Default Rate Trend by Loan Issue Year", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Default Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.show()

    print("Default Rate by Year (%):")
    print(default_rate_by_year)


def run_full_eda(df: pd.DataFrame) -> None:
    """
    Run all EDA visualizations in sequence.

    Note: this expects a dataframe where 'grade' and 'purpose' are
    still in their original (pre-encoding) form, and 'issue_year'
    has already been derived. If called after feature_engineering's
    encoding step, grade/purpose columns will be numeric/one-hot and
    some plots may need adjustment.
    """
    plot_default_rate_by_grade(df)
    plot_interest_rate_by_outcome(df)
    plot_dti_by_outcome(df)
    plot_income_by_outcome(df)
    plot_default_rate_by_purpose(df)
    plot_default_rate_by_year(df)
