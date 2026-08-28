"""
config.py

Central configuration for the Credit Risk Prediction project.
Stores file paths, column selections, and encoding mappings so that
they are defined once and reused consistently across the pipeline.
"""

# ------------------------------------------------------------------
# File paths
# ------------------------------------------------------------------
RAW_DATA_PATH = "/content/drive/MyDrive/credit-risk-project/data/raw/loan.csv"

# ------------------------------------------------------------------
# Columns to load from the raw CSV
# ------------------------------------------------------------------
# Only columns identified as relevant (i.e., not data-leakage, not
# high-missing, not free-text/ID columns) are loaded. This keeps
# memory usage low even though the source file has 145 columns
# and 2.26M+ rows.
REQUIRED_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "issue_d", "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status", "collections_12_mths_ex_med",
    "application_type", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "total_rev_hi_lim", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens",
    "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "disbursement_method",
]

# Loan statuses that represent a definitive outcome. Loans still in
# progress ("Current", "Late", "In Grace Period", etc.) are excluded
# because their final outcome is not yet known.
VALID_LOAN_STATUSES = ["Fully Paid", "Charged Off"]

# Columns that are known to be constant / redundant and add no
# predictive value once the target has been derived.
REDUNDANT_COLUMNS = ["policy_code"]

# ------------------------------------------------------------------
# Encoding mappings
# ------------------------------------------------------------------
GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}

EMP_LENGTH_MAP = {
    "Unknown": -1, "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8,
    "9 years": 9, "10+ years": 10,
}

BINARY_MAPS = {
    "initial_list_status": {"w": 1, "f": 0},
    "application_type": {"Joint App": 1, "Individual": 0},
    "disbursement_method": {"DirectPay": 1, "Cash": 0},
}

ONE_HOT_COLUMNS = ["home_ownership", "verification_status", "purpose"]

# ------------------------------------------------------------------
# Modeling
# ------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
