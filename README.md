# Credit Risk Prediction — Lending Club Loan Default Model

An end-to-end credit risk model that predicts the probability a loan will
default, using 1.3M+ historical loans from Lending Club (2007–2015). The
project goes beyond a standard classification exercise to address the
questions a real lending business actually cares about: *how much money
does this model save, is it fair across borrower groups, and can its
decisions be explained to a regulator or a rejected applicant?*

## Problem Statement

Lenders must decide whether to approve a loan application without knowing
in advance whether the borrower will repay. Approving too liberally
increases default losses; rejecting too conservatively forfeits interest
income from creditworthy borrowers. This project builds a model to
support that decision, and quantifies the trade-off in dollar terms.

## Dataset

- **Source:** [Lending Club Loan Data, 2007–2015](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle)
- **Size:** 2.26M loans, 145 raw fields; reduced to 66 relevant, non-leaking
  fields at load time for memory efficiency
- **Target:** Binary — `Charged Off` (defaulted, 1) vs `Fully Paid` (0).
  Loans with an in-progress status (`Current`, `Late`, etc.) are excluded
  since their final outcome is not yet known.
- **Class balance:** ~80% Fully Paid / ~20% Charged Off

## Approach

| Stage | What Was Done | Module |
|---|---|---|
| 1. Data Loading | Load only pre-identified relevant columns (memory-efficient) | `src/data_loader.py` |
| 2. Data Cleaning | Target encoding, missing-value imputation, DTI outlier removal | `src/data_cleaning.py` |
| 3. Feature Engineering | Derived features (credit history length, loan-to-income ratio), categorical encoding (ordinal, one-hot, frequency) | `src/feature_engineering.py` |
| 4. EDA | Default rate by grade, interest rate, DTI, income, purpose, and issue year | `src/eda.py` |
| 5. Modeling | Logistic Regression baseline, XGBoost (tuned for memory efficiency) | `src/model_training.py` |
| 6. Explainability | SHAP global + local explanations | `src/explainability.py` |
| 7. Fairness Audit | Group-level error rates and four-fifths rule check (by home ownership, as a proxy group) | `src/fairness_audit.py` |
| 8. Business Impact | Dollar-denominated cost model comparing the model to naive strategies | `src/cost_sensitive_evaluation.py` |
| 9. Threshold Tuning | Business-optimal decision cutoff (minimizing total estimated loss, not just maximizing accuracy) | `src/threshold_tuning.py` |
| 10. Dashboard | Interactive Streamlit app for live predictions + explanations | `app.py` |

A key design constraint throughout: this pipeline was developed and run
on Google Colab's free tier (~13GB RAM). Loading only necessary columns,
downcasting dtypes (`float32`/`int8`), and stratified sampling to 300K
rows for model training were deliberate choices to keep the pipeline
reproducible without paid compute — a common real-world constraint.

## Key Results

- **Baseline (Logistic Regression):** ROC-AUC 0.719
- **XGBoost:** ROC-AUC 0.733, catching 259 more defaulters than the
  baseline at equivalent precision
- **Top predictive features (SHAP):** `sub_grade`, `term`, `issue_year`,
  and the engineered `loan_to_income_ratio` — which ranked in the top 4,
  validating that domain-informed feature engineering added signal
  beyond the raw fields
- **Business impact:** Model-based decisions reduce estimated portfolio
  loss from **-$97.8M** (approve-all baseline) to **-$50.2M** — a
  **~$47.6M** improvement across the 60,000-loan test set
- **Cost asymmetry:** an average missed default costs ~$13,206 versus
  ~$2,711 for a wrongly-rejected good borrower (~5x), which justifies
  the model's recall-oriented design
- **Fairness finding:** renters are flagged as high-risk at ~1.5x the
  rate of mortgage-holders (49.2% vs 32.3% selection rate). This likely
  reflects genuine differences in financial profiles rather than
  arbitrary bias, but warrants further review before any real deployment
  (see `notebooks/` for the full discussion)

## Project Structure

```
credit-risk-project/
├── data/raw/                    # loan.csv, data dictionary (not tracked in git)
├── src/
│   ├── config.py                 # column lists, encoding maps, constants
│   ├── data_loader.py            # raw data loading
│   ├── data_cleaning.py          # target creation, imputation, outlier removal
│   ├── feature_engineering.py    # derived features + categorical encoding
│   ├── eda.py                    # exploratory visualizations
│   ├── model_training.py         # train/test split, Logistic Regression, XGBoost
│   ├── explainability.py         # SHAP analysis
│   ├── fairness_audit.py         # group fairness metrics
│   ├── cost_sensitive_evaluation.py  # dollar-impact evaluation
│   └── threshold_tuning.py       # business-optimal threshold search
├── tests/
│   ├── test_data_cleaning.py
│   └── test_feature_engineering.py
├── app.py                        # Streamlit dashboard
├── main.py                       # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```python
from main import run_pipeline

# sample_size limits training data for memory-constrained environments;
# set to None to use the full ~1.3M-row cleaned dataset
results = run_pipeline(sample_size=300_000)
```

### 3. Run tests
```bash
pytest tests/ -v
```

### 4. Save model artifacts (for the dashboard)
```python
from src.model_training import train_xgboost, save_model_artifacts

xgb_model = train_xgboost(results["X_train"], results["y_train"])
save_model_artifacts(xgb_model, list(results["X_train"].columns))
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

## Limitations & Next Steps

- **Right-censoring bias:** default rates for 2015–2017 loan vintages
  appear inflated because defaults tend to occur early in a loan's life,
  while "fully paid" status requires the full term to elapse. Excluding
  `Current` loans disproportionately removes still-healthy recent loans
  from those cohorts.
- **No protected attributes:** the dataset lacks race/gender fields, so
  the fairness audit uses `home_ownership` as a proxy group to
  demonstrate methodology; a production deployment would need a
  dedicated fairness review with actual protected-attribute data (under
  appropriate legal/privacy safeguards).
- **Simplified cost model:** the business-impact calculation assumes
  100% loss on default and ignores partial recovery, servicing costs,
  and time value of money.
- **Future work:** model monitoring / drift detection for production
  deployment, hyperparameter tuning via cross-validation, and a
  containerized API (FastAPI + Docker) for serving predictions.
