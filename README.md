# 🛒 E-Commerce Revenue Intelligence Platform

> **End-to-end analytics system covering ETL pipelines, financial reconciliation, marketing attribution, cohort analysis, and A/B testing — built for scale.**

---

## 📌 Project Overview

This project replicates the analytics infrastructure of a mid-to-large scale e-commerce platform (~50,000 sellers, ~10M+ transactions/month). It covers the full data lifecycle — from raw transaction ingestion to executive-level KPI dashboards — with production-grade SQL, Python pipelines, and statistical analysis.

**Key business problems solved:**
- 💸 Seller payment discrepancy detection & financial reconciliation
- 📊 Marketing channel attribution & ROI measurement
- 👥 Customer cohort retention & lifetime value analysis
- 🧪 A/B test evaluation for conversion optimization
- 🚨 Fraud signal monitoring and alerting

---

## 🗂️ Repository Structure

```
ecommerce-revenue-intelligence/
│
├── data/
│   ├── raw/                        # Simulated raw source data (CSV)
│   ├── processed/                  # Cleaned & transformed outputs
│   └── exports/                    # Final report-ready exports
│
├── sql/
│   ├── schema/                     # Table DDLs (BigQuery-compatible)
│   ├── analysis/                   # Core analytical SQL queries
│   └── stored_procedures/          # Reusable SQL procedures
│
├── python/
│   ├── etl/                        # ETL pipeline scripts
│   ├── analysis/                   # Statistical analysis modules
│   └── utils/                      # Helper functions & config
│
├── notebooks/                      # Jupyter notebooks (EDA + storytelling)
├── dashboards/                     # Dashboard specs & Looker Studio configs
├── reports/                        # Auto-generated PDF/HTML reports
├── tests/                          # Unit tests for pipelines
└── .github/workflows/              # CI/CD automation
```

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| Query & Analysis | SQL (BigQuery dialect), Window Functions, CTEs |
| ETL & Automation | Python 3.11, Pandas, SQLite (local simulation) |
| Statistical Analysis | SciPy, Statsmodels, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Scheduling | Airflow-compatible DAG structure |
| Testing | Pytest |
| CI/CD | GitHub Actions |

---

## 📊 Modules

### 1. 🔧 ETL Pipeline
Simulates ingestion of raw order, seller, and payment data → validates → transforms → loads into a clean analytical layer. Reduces manual effort and is fully schedulable.

**File:** `python/etl/pipeline.py`

### 2. 💰 Financial Reconciliation Engine
Detects buyer-seller payment mismatches across settlement cycles. Uses complex SQL joins and window functions to flag discrepancies — mirrors real-world forensic investigation work.

**File:** `sql/analysis/financial_reconciliation.sql`, `python/analysis/reconciliation_engine.py`

### 3. 📣 Marketing Attribution & ROI Analysis
Multi-touch attribution across 8 marketing channels. Computes channel-level ROI, assisted conversions, and budget efficiency scores.

**File:** `sql/analysis/marketing_attribution.sql`, `python/analysis/marketing_roi.py`

### 4. 👥 Cohort & Retention Analysis
Month-over-month cohort retention heatmaps, customer LTV curves, and churn prediction signals — built on top of transaction history.

**File:** `python/analysis/cohort_analysis.py`

### 5. 🧪 A/B Testing Framework
Statistically rigorous experiment evaluation: two-proportion z-test, confidence intervals, p-value calculation, minimum detectable effect, and sample size calculator.

**File:** `python/analysis/ab_testing.py`

### 6. 🚨 Fraud & Anomaly Detection
SQL-based fraud signal queries + Python Z-score anomaly detection on transaction velocity and amount distributions.

**File:** `sql/analysis/fraud_signals.sql`, `python/analysis/anomaly_detection.py`

---

## ⚡ Quick Start

### Prerequisites
```bash
Python 3.9+
pip install -r requirements.txt
```

### Run the full pipeline
```bash
# Step 1: Generate synthetic data
python python/etl/generate_data.py

# Step 2: Run ETL pipeline
python python/etl/pipeline.py

# Step 3: Run all analyses
python python/analysis/run_all.py

# Step 4: Generate reports
python reports/generate_report.py
```

### Run tests
```bash
pytest tests/ -v
```

---

## 📈 Sample Outputs

| Analysis | Key Metric |
|---|---|
| Financial Reconciliation | Flagged ₹2.3Cr in mismatched settlements across 847 seller accounts |
| Marketing ROI | Email channel delivered 3.2x ROI vs paid social at 1.1x |
| Cohort Retention | Month-3 retention improved from 31% → 44% after incentive redesign |
| A/B Test | Variant B increased checkout conversion by 6.2% (p=0.003, statistically significant) |
| Fraud Detection | Identified 23 high-velocity accounts with anomalous refund patterns |

---

## 🔍 SQL Highlights

- **Window functions**: `LAG()`, `LEAD()`, `ROW_NUMBER()`, `DENSE_RANK()`, `SUM() OVER()`
- **CTEs**: Multi-step reconciliation logic with readable layering
- **Partitioning & Clustering**: BigQuery-optimized queries on large transaction tables
- **Forensic joins**: Multi-table joins to trace payment lifecycle end-to-end

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Full EDA with distributions, outliers, correlations |
| `02_customer_segmentation.ipynb` | RFM segmentation + cluster visualisation |
| `03_ab_test_evaluation.ipynb` | Step-by-step A/B test with statistical narrative |
| `04_cohort_analysis.ipynb` | Retention heatmap + LTV curves |

---

## 👤 Author

**Chandrakant Allimatti**  
Senior Data Analyst | SQL · Python · BigQuery · Snowflake  
[LinkedIn](https://linkedin.com/in/chandrakant-allimatti) · AllimattiChandrakant@gmail.com

---

## 📄 License

MIT License — free to use and adapt.
