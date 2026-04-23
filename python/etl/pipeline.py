"""
pipeline.py
-----------
ETL pipeline: validates, cleans, and transforms raw e-commerce CSVs
into a clean analytical layer in SQLite (mirrors BigQuery in production).

Reduces manual reporting effort — designed to be scheduled via Airflow.

Run: python python/etl/pipeline.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import logging
from datetime import datetime

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(BASE_DIR, "../../data/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "../../data/processed")
DB_PATH       = os.path.join(RAW_DIR, "ecommerce.db")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Validation helpers ────────────────────────────────────────────────────────
class DataQualityError(Exception):
    pass


def check_nulls(df: pd.DataFrame, name: str, critical_cols: list):
    for col in critical_cols:
        nulls = df[col].isnull().sum()
        if nulls > 0:
            log.warning(f"  [{name}] Column '{col}' has {nulls} nulls ({nulls/len(df):.2%})")


def check_duplicates(df: pd.DataFrame, key: str, name: str):
    dupes = df.duplicated(subset=[key]).sum()
    if dupes > 0:
        log.warning(f"  [{name}] {dupes} duplicate {key}s found — dropping")
        return df.drop_duplicates(subset=[key])
    return df


def validate_numeric_range(df, col, min_val, max_val, name):
    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
    if out_of_range > 0:
        log.warning(f"  [{name}] {out_of_range} rows with {col} outside [{min_val}, {max_val}]")


# ── Transform: Orders ─────────────────────────────────────────────────────────
def transform_orders(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Transforming orders...")
    df = df.copy()

    # Parse dates
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Derived time dimensions
    df["order_year"]    = df["order_date"].dt.year
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_week"]    = df["order_date"].dt.isocalendar().week.astype(int)
    df["order_dow"]     = df["order_date"].dt.day_name()
    df["is_weekend"]    = df["order_date"].dt.dayofweek >= 5

    # Revenue flags
    df["is_returned"]   = (df["status"] == "returned").astype(int)
    df["is_cancelled"]  = (df["status"] == "cancelled").astype(int)
    df["is_delivered"]  = (df["status"] == "delivered").astype(int)

    # Discount rate
    df["discount_rate"] = np.where(
        df["gmv"] > 0,
        np.round(df["discount_amount"] / df["gmv"], 4),
        0
    )

    # GMV band segmentation
    df["gmv_band"] = pd.cut(
        df["gmv"],
        bins=[0, 500, 2000, 10000, 50000, float("inf")],
        labels=["<500", "500-2K", "2K-10K", "10K-50K", "50K+"]
    )

    check_nulls(df, "orders", ["order_id", "customer_id", "seller_id", "gmv"])
    validate_numeric_range(df, "gmv", 0, 200_000, "orders")
    validate_numeric_range(df, "discount_rate", 0, 1, "orders")

    log.info(f"    → {len(df):,} orders transformed")
    return df


# ── Transform: Payments ───────────────────────────────────────────────────────
def transform_payments(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Transforming payments...")
    df = df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])

    df["discrepancy_pct"] = np.where(
        df["expected_payout"] > 0,
        np.round((df["discrepancy_amt"] / df["expected_payout"]) * 100, 2),
        0
    )
    df["discrepancy_severity"] = pd.cut(
        df["discrepancy_pct"],
        bins=[-1, 0, 5, 15, 100],
        labels=["none", "minor", "moderate", "critical"]
    )

    check_nulls(df, "payments", ["payment_id", "order_id", "seller_id"])
    log.info(f"    → {len(df):,} payment records transformed")
    return df


# ── Transform: Customers ──────────────────────────────────────────────────────
def transform_customers(df: pd.DataFrame) -> pd.DataFrame:
    log.info("  Transforming customers...")
    df = df.copy()
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df["days_since_reg"] = (
        pd.Timestamp("2024-06-30") - df["registration_date"]
    ).dt.days
    df["tenure_band"] = pd.cut(
        df["days_since_reg"],
        bins=[0, 30, 90, 180, 365, float("inf")],
        labels=["<1M", "1-3M", "3-6M", "6-12M", "12M+"]
    )
    log.info(f"    → {len(df):,} customers transformed")
    return df


# ── Load to processed layer ───────────────────────────────────────────────────
def load_to_processed(dfs: dict):
    log.info("  Loading to processed layer (SQLite + CSV)...")
    conn = sqlite3.connect(DB_PATH)

    for name, df in dfs.items():
        out_path = os.path.join(PROCESSED_DIR, f"{name}_clean.csv")
        # Convert categoricals to string for SQLite
        df_save = df.copy()
        for col in df_save.select_dtypes(include="category").columns:
            df_save[col] = df_save[col].astype(str)
        df_save.to_csv(out_path, index=False)
        df_save.to_sql(f"{name}_clean", conn, if_exists="replace", index=False)
        log.info(f"    ✓ {name}: {len(df):,} rows → {out_path}")

    conn.close()


# ── Build summary layer ───────────────────────────────────────────────────────
def build_summary_tables(orders_df, payments_df):
    log.info("  Building summary/aggregate tables...")
    conn = sqlite3.connect(DB_PATH)

    # Monthly revenue summary
    monthly = (
        orders_df[orders_df["is_delivered"] == 1]
        .groupby(["order_year", "order_month"])
        .agg(
            total_orders=("order_id", "count"),
            total_gmv=("gmv", "sum"),
            total_net_revenue=("net_revenue", "sum"),
            avg_order_value=("gmv", "mean"),
            total_discount=("discount_amount", "sum"),
        )
        .reset_index()
        .round(2)
    )
    monthly.to_sql("monthly_revenue_summary", conn, if_exists="replace", index=False)
    monthly.to_csv(os.path.join(PROCESSED_DIR, "monthly_revenue_summary.csv"), index=False)

    # Seller performance
    seller_perf = (
        orders_df[orders_df["is_delivered"] == 1]
        .groupby("seller_id")
        .agg(
            total_orders=("order_id", "count"),
            total_gmv=("gmv", "sum"),
            avg_order_value=("gmv", "mean"),
            return_rate=("is_returned", "mean"),
        )
        .reset_index()
        .round(3)
    )
    seller_perf.to_sql("seller_performance", conn, if_exists="replace", index=False)
    seller_perf.to_csv(os.path.join(PROCESSED_DIR, "seller_performance.csv"), index=False)

    # Discrepancy summary
    disc = payments_df[payments_df["discrepancy_flag"] == 1]
    disc_summary = (
        disc.groupby("seller_id")
        .agg(
            disputed_payments=("payment_id", "count"),
            total_discrepancy_amt=("discrepancy_amt", "sum"),
            avg_discrepancy_pct=("discrepancy_pct", "mean"),
        )
        .reset_index()
        .sort_values("total_discrepancy_amt", ascending=False)
        .round(2)
    )
    disc_summary.to_sql("discrepancy_summary", conn, if_exists="replace", index=False)
    disc_summary.to_csv(os.path.join(PROCESSED_DIR, "discrepancy_summary.csv"), index=False)

    conn.close()
    log.info(f"    ✓ monthly_revenue_summary: {len(monthly)} rows")
    log.info(f"    ✓ seller_performance: {len(seller_perf)} rows")
    log.info(f"    ✓ discrepancy_summary: {len(disc_summary)} rows")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_pipeline():
    start = datetime.now()
    log.info("=" * 60)
    log.info("🚀 ETL Pipeline starting...")
    log.info("=" * 60)

    # Extract
    log.info("\n[EXTRACT] Reading raw CSVs...")
    orders_raw    = pd.read_csv(os.path.join(RAW_DIR, "orders.csv"))
    payments_raw  = pd.read_csv(os.path.join(RAW_DIR, "payments.csv"))
    customers_raw = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    sellers_raw   = pd.read_csv(os.path.join(RAW_DIR, "sellers.csv"))
    log.info(f"  orders: {len(orders_raw):,} | payments: {len(payments_raw):,} | "
             f"customers: {len(customers_raw):,} | sellers: {len(sellers_raw):,}")

    # Dedup
    log.info("\n[VALIDATE] Checking for duplicates...")
    orders_raw    = check_duplicates(orders_raw,    "order_id",    "orders")
    payments_raw  = check_duplicates(payments_raw,  "payment_id",  "payments")
    customers_raw = check_duplicates(customers_raw, "customer_id", "customers")
    sellers_raw   = check_duplicates(sellers_raw,   "seller_id",   "sellers")

    # Transform
    log.info("\n[TRANSFORM] Applying business logic...")
    orders_clean    = transform_orders(orders_raw)
    payments_clean  = transform_payments(payments_raw)
    customers_clean = transform_customers(customers_raw)

    # Load
    log.info("\n[LOAD] Writing to processed layer...")
    load_to_processed({
        "orders"   : orders_clean,
        "payments" : payments_clean,
        "customers": customers_clean,
        "sellers"  : sellers_raw,
    })

    # Aggregates
    log.info("\n[AGGREGATE] Building summary tables...")
    build_summary_tables(orders_clean, payments_clean)

    elapsed = (datetime.now() - start).total_seconds()
    log.info("\n" + "=" * 60)
    log.info(f"✅ Pipeline completed in {elapsed:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
