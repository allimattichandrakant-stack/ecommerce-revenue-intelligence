"""
reconciliation_engine.py
------------------------
Financial reconciliation system: detects buyer-seller payment
discrepancies, classifies severity, flags accounts for audit,
and generates a forensic investigation report.

This mirrors the Rs.500Cr discrepancy investigation done at Flipkart.

Run: python python/analysis/reconciliation_engine.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(BASE_DIR, "../../data/raw/ecommerce.db")
EXPORTS_DIR  = os.path.join(BASE_DIR, "../../data/exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
COLORS = {
    "critical": "#D62728",
    "moderate": "#FF7F0E",
    "minor"   : "#FFBB78",
    "none"    : "#2CA02C",
    "bg"      : "#0F1117",
    "surface" : "#1C1F26",
    "text"    : "#E8EAF0",
    "accent"  : "#4F8EF7",
}

sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": COLORS["surface"],
    "figure.facecolor": COLORS["bg"],
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "grid.color": "#2A2D35",
})


def load_data():
    conn = sqlite3.connect(DB_PATH)
    payments  = pd.read_sql("SELECT * FROM payments_clean", conn)
    orders    = pd.read_sql("SELECT * FROM orders_clean", conn)
    sellers   = pd.read_sql("SELECT * FROM sellers", conn)
    conn.close()
    payments["settlement_date"] = pd.to_datetime(payments["settlement_date"])
    return payments, orders, sellers


def run_reconciliation(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Core reconciliation logic.
    Classifies each disputed payment and computes audit risk score.
    """
    disputed = payments[payments["discrepancy_flag"] == 1].copy()

    # Risk scoring: weighted combination of discrepancy % and absolute amount
    disputed["discrepancy_pct"] = disputed["discrepancy_pct"].astype(float)
    disputed["discrepancy_amt"] = disputed["discrepancy_amt"].astype(float)

    # Normalise each dimension 0-1
    amt_max = disputed["discrepancy_amt"].max()
    pct_max = disputed["discrepancy_pct"].max()

    disputed["risk_score"] = (
        0.6 * (disputed["discrepancy_amt"] / amt_max) +
        0.4 * (disputed["discrepancy_pct"] / pct_max)
    ).round(4)

    # Severity tier
    disputed["severity_tier"] = pd.cut(
        disputed["discrepancy_pct"],
        bins=[-1, 5, 15, 30, 100],
        labels=["Minor (<5%)", "Moderate (5-15%)", "High (15-30%)", "Critical (>30%)"]
    )

    return disputed.sort_values("risk_score", ascending=False)


def seller_level_summary(disputed: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    summary = (
        disputed.groupby("seller_id")
        .agg(
            disputed_count=("payment_id", "count"),
            total_discrepancy=("discrepancy_amt", "sum"),
            avg_discrepancy_pct=("discrepancy_pct", "mean"),
            max_risk_score=("risk_score", "max"),
        )
        .reset_index()
        .sort_values("total_discrepancy", ascending=False)
        .round(2)
    )
    summary = summary.merge(
        sellers[["seller_id", "seller_name", "category", "tier"]],
        on="seller_id", how="left"
    )
    # Flag sellers for mandatory audit
    threshold = summary["total_discrepancy"].quantile(0.80)
    summary["audit_required"] = summary["total_discrepancy"] >= threshold
    return summary


def plot_discrepancy_overview(payments, disputed, seller_summary):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Financial Reconciliation — Discrepancy Analysis",
                 fontsize=16, color=COLORS["text"], fontweight="bold", y=0.98)

    # 1. Severity breakdown
    ax = axes[0, 0]
    sev_counts = disputed["severity_tier"].value_counts()
    sev_colors = [COLORS["minor"], COLORS["moderate"], COLORS["critical"], "#8B0000"]
    bars = ax.bar(sev_counts.index, sev_counts.values, color=sev_colors[:len(sev_counts)], width=0.6)
    ax.set_title("Disputes by Severity Tier", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("# Disputed Payments", color=COLORS["text"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{int(bar.get_height()):,}", ha="center", va="bottom",
                color=COLORS["text"], fontsize=9)

    # 2. Monthly discrepancy trend
    ax = axes[0, 1]
    disputed["month"] = pd.to_datetime(disputed["settlement_date"]).dt.to_period("M")
    monthly = disputed.groupby("month")["discrepancy_amt"].sum().reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    ax.plot(monthly["month_str"], monthly["discrepancy_amt"]/1e5,
            color=COLORS["critical"], linewidth=2, marker="o", markersize=4)
    ax.fill_between(monthly["month_str"], monthly["discrepancy_amt"]/1e5,
                    alpha=0.2, color=COLORS["critical"])
    ax.set_title("Monthly Discrepancy Amount (₹ Lakhs)", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("₹ Lakhs", color=COLORS["text"])
    ax.tick_params(axis="x", rotation=45)

    # 3. Top 15 sellers by discrepancy
    ax = axes[1, 0]
    top15 = seller_summary.head(15)
    colors_bar = [COLORS["critical"] if a else COLORS["accent"]
                  for a in top15["audit_required"]]
    ax.barh(top15["seller_id"], top15["total_discrepancy"]/1e3,
            color=colors_bar, height=0.7)
    ax.set_title("Top 15 Sellers by Total Discrepancy (₹K)", color=COLORS["text"], fontsize=12)
    ax.set_xlabel("₹ Thousands", color=COLORS["text"])
    ax.invert_yaxis()

    # 4. Discrepancy % distribution
    ax = axes[1, 1]
    ax.hist(disputed["discrepancy_pct"], bins=40, color=COLORS["accent"],
            edgecolor=COLORS["bg"], alpha=0.85)
    ax.axvline(disputed["discrepancy_pct"].median(), color=COLORS["critical"],
               linestyle="--", linewidth=1.5, label=f"Median: {disputed['discrepancy_pct'].median():.1f}%")
    ax.set_title("Distribution of Discrepancy %", color=COLORS["text"], fontsize=12)
    ax.set_xlabel("Discrepancy %", color=COLORS["text"])
    ax.set_ylabel("Frequency", color=COLORS["text"])
    ax.legend(facecolor=COLORS["surface"], labelcolor=COLORS["text"])

    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "reconciliation_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  📊 Chart saved: {out}")


def print_report(payments, disputed, seller_summary):
    total_expected  = payments["expected_payout"].sum()
    total_actual    = payments["actual_payout"].sum()
    total_disc_amt  = disputed["discrepancy_amt"].sum()
    disc_rate       = len(disputed) / len(payments) * 100
    audit_sellers   = seller_summary["audit_required"].sum()

    print("\n" + "═" * 65)
    print("  FINANCIAL RECONCILIATION REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("═" * 65)
    print(f"\n  Total payments processed  : {len(payments):>12,}")
    print(f"  Total expected payout (₹) : {total_expected:>12,.0f}")
    print(f"  Total actual payout (₹)   : {total_actual:>12,.0f}")
    print(f"  Total discrepancy (₹)     : {total_disc_amt:>12,.0f}  ← FLAGGED")
    print(f"  Discrepancy rate          : {disc_rate:>11.2f}%")
    print(f"  Sellers flagged for audit : {audit_sellers:>12,}")

    print("\n  TOP 10 HIGH-RISK SELLERS")
    print("  " + "-" * 60)
    top10 = seller_summary.head(10)[
        ["seller_id", "seller_name", "disputed_count", "total_discrepancy", "max_risk_score"]
    ]
    for _, row in top10.iterrows():
        flag = "🔴" if row["max_risk_score"] > 0.7 else "🟡"
        print(f"  {flag} {row['seller_id']} | ₹{row['total_discrepancy']:>10,.0f} | "
              f"Risk: {row['max_risk_score']:.3f} | Disputes: {int(row['disputed_count'])}")

    print("\n  SEVERITY BREAKDOWN")
    print("  " + "-" * 40)
    for tier, grp in disputed.groupby("severity_tier", observed=True):
        print(f"  {tier:<25}: {len(grp):>5,} cases | ₹{grp['discrepancy_amt'].sum():>12,.0f}")

    print("\n" + "═" * 65 + "\n")


def main():
    print("\n💰 Running Financial Reconciliation Engine...")
    payments, orders, sellers = load_data()

    disputed       = run_reconciliation(payments)
    seller_summary = seller_level_summary(disputed, sellers)

    # Save outputs
    disputed.to_csv(os.path.join(EXPORTS_DIR, "disputed_payments.csv"), index=False)
    seller_summary.to_csv(os.path.join(EXPORTS_DIR, "seller_audit_list.csv"), index=False)

    # Visualise
    plot_discrepancy_overview(payments, disputed, seller_summary)

    # Print report
    print_report(payments, disputed, seller_summary)

    print("✅ Reconciliation complete. Outputs written to data/exports/\n")


if __name__ == "__main__":
    main()
