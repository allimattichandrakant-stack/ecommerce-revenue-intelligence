"""
cohort_analysis.py
------------------
Customer cohort retention analysis.
Builds month-over-month retention heatmaps and LTV curves.
Mirrors the user growth analysis done at Uber.

Run: python python/analysis/cohort_analysis.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "../../data/raw/ecommerce.db")
EXPORTS_DIR = os.path.join(BASE_DIR, "../../data/exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

BG     = "#0D1117"
SURF   = "#161B22"
TEXT   = "#E6EDF3"
ACCENT = "#58A6FF"


def load_data():
    conn   = sqlite3.connect(DB_PATH)
    orders = pd.read_sql("SELECT * FROM orders_clean", conn)
    conn.close()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    return orders


def build_cohort_matrix(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a retention matrix:
      rows    = acquisition cohort month
      columns = months since first order (0, 1, 2, ...)
      values  = % of cohort still active
    """
    delivered = orders[orders["is_delivered"] == 1].copy()
    delivered["order_month"] = delivered["order_date"].dt.to_period("M")

    # First order month per customer
    first_order = (
        delivered.groupby("customer_id")["order_month"]
        .min()
        .reset_index()
        .rename(columns={"order_month": "cohort_month"})
    )
    delivered = delivered.merge(first_order, on="customer_id")

    # Months since first order
    delivered["period_number"] = (
        delivered["order_month"] - delivered["cohort_month"]
    ).apply(lambda x: x.n)

    # Keep only reasonable range
    delivered = delivered[delivered["period_number"] <= 11]

    cohort_data = (
        delivered.groupby(["cohort_month", "period_number"])["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "active_users"})
    )

    cohort_pivot = cohort_data.pivot_table(
        index="cohort_month", columns="period_number", values="active_users"
    )

    # Cohort size = period 0 count
    cohort_size = cohort_pivot[0]

    # Retention rates
    retention = cohort_pivot.divide(cohort_size, axis=0).round(4)
    return retention, cohort_size


def build_ltv_curve(orders: pd.DataFrame, cohort_size: pd.Series) -> pd.DataFrame:
    """Cumulative LTV per cohort customer over 12 months."""
    delivered = orders[orders["is_delivered"] == 1].copy()
    delivered["order_month"] = delivered["order_date"].dt.to_period("M")

    first_order = (
        delivered.groupby("customer_id")["order_month"]
        .min()
        .reset_index()
        .rename(columns={"order_month": "cohort_month"})
    )
    delivered = delivered.merge(first_order, on="customer_id")
    delivered["period_number"] = (
        delivered["order_month"] - delivered["cohort_month"]
    ).apply(lambda x: x.n)
    delivered = delivered[delivered["period_number"] <= 11]

    rev_pivot = delivered.groupby(["cohort_month", "period_number"])["net_revenue"].sum()
    rev_pivot = rev_pivot.unstack(fill_value=0)
    cumrev    = rev_pivot.cumsum(axis=1)
    ltv       = cumrev.divide(cohort_size, axis=0).round(2)
    return ltv


def plot_retention_heatmap(retention: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
    ax.set_facecolor(SURF)

    mask = retention.isnull()
    heat = sns.heatmap(
        retention * 100,
        annot=True, fmt=".0f", mask=mask,
        cmap="YlOrRd_r",
        linewidths=0.3, linecolor="#0D1117",
        ax=ax, cbar_kws={"label": "Retention %"},
        annot_kws={"size": 8, "color": "white"},
    )
    ax.set_title("Customer Cohort Retention Heatmap (%)\nRows = Acquisition Month  |  Columns = Months Since First Order",
                 fontsize=13, color=TEXT, pad=12)
    ax.set_xlabel("Month Number (0 = Acquisition)", fontsize=10, color=TEXT)
    ax.set_ylabel("Cohort (Acquisition Month)", fontsize=10, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=8)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    cbar = heat.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(TEXT)
    cbar.ax.tick_params(colors=TEXT)

    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "cohort_retention_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  📊 Heatmap saved: {out}")


def plot_ltv_curves(ltv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(SURF)

    # Plot only cohorts with enough data (at least 6 months)
    valid = ltv.dropna(thresh=6)
    cmap  = plt.cm.get_cmap("cool", len(valid))

    for i, (cohort, row) in enumerate(valid.iterrows()):
        data = row.dropna()
        ax.plot(data.index, data.values, color=cmap(i), linewidth=1.8,
                label=str(cohort), marker="o", markersize=3, alpha=0.85)

    ax.set_title("Cumulative Customer LTV by Acquisition Cohort (₹)",
                 fontsize=13, color=TEXT, pad=10)
    ax.set_xlabel("Months Since Acquisition", fontsize=10, color=TEXT)
    ax.set_ylabel("Avg Cumulative Revenue per Customer (₹)", fontsize=10, color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7,
              facecolor=SURF, labelcolor=TEXT, framealpha=0.8)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "ltv_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  📊 LTV curves saved: {out}")


def print_cohort_report(retention: pd.DataFrame, ltv: pd.DataFrame):
    avg_m1  = retention[1].mean() * 100
    avg_m3  = retention[3].mean() * 100 if 3 in retention.columns else None
    avg_m6  = retention[6].mean() * 100 if 6 in retention.columns else None
    best    = retention[1].idxmax()
    worst   = retention[1].idxmin()

    print("\n" + "═" * 55)
    print("  COHORT RETENTION REPORT")
    print("═" * 55)
    print(f"\n  Avg Month-1 Retention    : {avg_m1:.1f}%")
    if avg_m3: print(f"  Avg Month-3 Retention    : {avg_m3:.1f}%")
    if avg_m6: print(f"  Avg Month-6 Retention    : {avg_m6:.1f}%")
    print(f"\n  Best M1 Retention Cohort : {best}  ({retention.loc[best, 1]*100:.1f}%)")
    print(f"  Worst M1 Retention Cohort: {worst}  ({retention.loc[worst, 1]*100:.1f}%)")

    # LTV at month 6
    if 6 in ltv.columns:
        avg_ltv6 = ltv[6].mean()
        print(f"\n  Avg 6-Month LTV          : ₹{avg_ltv6:,.0f}")
    print("═" * 55 + "\n")


def main():
    print("\n👥 Running Cohort & Retention Analysis...")
    orders = load_data()

    retention, cohort_size = build_cohort_matrix(orders)
    ltv = build_ltv_curve(orders, cohort_size)

    retention.to_csv(os.path.join(EXPORTS_DIR, "cohort_retention_matrix.csv"))
    ltv.to_csv(os.path.join(EXPORTS_DIR, "cohort_ltv_matrix.csv"))

    plot_retention_heatmap(retention)
    plot_ltv_curves(ltv)
    print_cohort_report(retention, ltv)
    print("✅ Cohort analysis complete.\n")


if __name__ == "__main__":
    main()
