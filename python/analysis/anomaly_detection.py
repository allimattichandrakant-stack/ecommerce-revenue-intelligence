"""
anomaly_detection.py
--------------------
Fraud signal & anomaly detection on transaction data.
Uses Z-score, IQR, and rolling-baseline methods to surface
high-velocity sellers, unusual refund patterns, and GMV spikes.

Run: python python/analysis/anomaly_detection.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "../../data/raw/ecommerce.db")
EXPORTS_DIR = os.path.join(BASE_DIR, "../../data/exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

BG, SURF, TEXT = "#0A0E1A", "#141929", "#DDE2F0"
RED, GREEN, AMBER = "#E74C3C", "#2ECC71", "#F39C12"
BLUE = "#5B9CF6"

Z_THRESHOLD = 3.0   # flag if |z| > 3


def load_data():
    conn   = sqlite3.connect(DB_PATH)
    orders = pd.read_sql("SELECT * FROM orders_clean", conn)
    conn.close()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    return orders


def z_score_anomalies(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    return (series - mu) / sigma if sigma > 0 else pd.Series(0, index=series.index)


def detect_high_velocity_sellers(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Sellers with abnormally high order velocity in a 7-day window.
    Legitimate sellers have smooth order distributions.
    """
    orders["week"] = orders["order_date"].dt.to_period("W")
    weekly = (
        orders.groupby(["seller_id", "week"])
        .agg(weekly_orders=("order_id", "count"), weekly_gmv=("gmv", "sum"))
        .reset_index()
    )
    weekly["order_z"]  = z_score_anomalies(weekly["weekly_orders"])
    weekly["gmv_z"]    = z_score_anomalies(weekly["weekly_gmv"])
    weekly["combined_z"] = (weekly["order_z"] + weekly["gmv_z"]) / 2

    flagged = weekly[weekly["combined_z"] > Z_THRESHOLD].copy()
    flagged = flagged.sort_values("combined_z", ascending=False)
    return flagged


def detect_refund_anomalies(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Sellers with statistically abnormal return/refund rates.
    High refund rate + high GMV = potential fraud signal.
    """
    seller_stats = (
        orders.groupby("seller_id")
        .agg(
            total_orders=("order_id", "count"),
            returned_orders=("is_returned", "sum"),
            total_gmv=("gmv", "sum"),
        )
        .reset_index()
    )
    seller_stats = seller_stats[seller_stats["total_orders"] >= 20]
    seller_stats["return_rate"]  = seller_stats["returned_orders"] / seller_stats["total_orders"]
    seller_stats["return_rate_z"] = z_score_anomalies(seller_stats["return_rate"])
    seller_stats["gmv_z"]         = z_score_anomalies(seller_stats["total_gmv"])
    # Composite fraud signal: high returns + high GMV = high risk
    seller_stats["fraud_score"]   = (
        0.7 * seller_stats["return_rate_z"] + 0.3 * seller_stats["gmv_z"]
    ).round(4)

    flagged = seller_stats[seller_stats["return_rate_z"] > Z_THRESHOLD - 0.5]
    return flagged.sort_values("fraud_score", ascending=False)


def detect_gmv_spikes(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Platform-level daily GMV spikes vs rolling 14-day baseline.
    """
    daily_gmv = (
        orders.groupby(orders["order_date"].dt.date)
        ["gmv"].sum()
        .reset_index()
        .rename(columns={"order_date": "date"})
    )
    daily_gmv["rolling_mean"] = daily_gmv["gmv"].rolling(14, min_periods=7).mean()
    daily_gmv["rolling_std"]  = daily_gmv["gmv"].rolling(14, min_periods=7).std()
    daily_gmv["z_score"]      = (daily_gmv["gmv"] - daily_gmv["rolling_mean"]) / daily_gmv["rolling_std"]
    daily_gmv["is_spike"]     = daily_gmv["z_score"].abs() > Z_THRESHOLD
    return daily_gmv


def plot_anomaly_dashboard(velocity_df, refund_df, gmv_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.suptitle("Fraud Signal & Anomaly Detection Dashboard",
                 fontsize=16, color=TEXT, fontweight="bold")

    # 1. High-velocity sellers
    ax = axes[0, 0]
    ax.set_facecolor(SURF)
    top20v = velocity_df.head(20)
    colors = [RED if z > 4 else AMBER for z in top20v["combined_z"]]
    ax.barh(top20v["seller_id"], top20v["combined_z"], color=colors, height=0.7)
    ax.axvline(Z_THRESHOLD, color="white", linestyle="--", linewidth=1, alpha=0.6,
               label=f"Z-threshold ({Z_THRESHOLD})")
    ax.set_title("High-Velocity Sellers\n(Combined Z-Score)", color=TEXT, fontsize=11)
    ax.set_xlabel("Z-Score", color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.legend(facecolor=SURF, labelcolor=TEXT, fontsize=8)
    ax.invert_yaxis()

    # 2. Refund anomalies scatter
    ax = axes[0, 1]
    ax.set_facecolor(SURF)
    normal  = refund_df[refund_df["return_rate_z"] <= Z_THRESHOLD - 0.5]
    flagged = refund_df[refund_df["return_rate_z"] > Z_THRESHOLD - 0.5]
    ax.scatter(normal["total_gmv"]/1e5, normal["return_rate"]*100,
               alpha=0.4, color=BLUE, s=15, label="Normal")
    ax.scatter(flagged["total_gmv"]/1e5, flagged["return_rate"]*100,
               alpha=0.9, color=RED, s=40, marker="X", label="Flagged")
    ax.set_title("Seller Return Rate vs GMV\n(Red X = Anomalous)", color=TEXT, fontsize=11)
    ax.set_xlabel("Total GMV (₹ Lakhs)", color=TEXT)
    ax.set_ylabel("Return Rate %", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.legend(facecolor=SURF, labelcolor=TEXT)

    # 3. Daily GMV with spikes
    ax = axes[1, 0]
    ax.set_facecolor(SURF)
    normal_days  = gmv_df[~gmv_df["is_spike"]]
    spike_days   = gmv_df[gmv_df["is_spike"]]
    ax.plot(range(len(gmv_df)), gmv_df["gmv"]/1e5, color=BLUE, linewidth=1.2, alpha=0.7)
    ax.plot(range(len(gmv_df)), gmv_df["rolling_mean"]/1e5, color=GREEN, linewidth=1.5,
            linestyle="--", label="14-day rolling mean")
    spike_idx = gmv_df[gmv_df["is_spike"]].index.tolist()
    ax.scatter([list(gmv_df.index).index(i) if i in list(gmv_df.index) else None
                for i in spike_idx],
               spike_days["gmv"]/1e5, color=RED, zorder=5, s=50, label="Spike detected")
    ax.set_title("Daily Platform GMV with Anomaly Flags (₹ Lakhs)", color=TEXT, fontsize=11)
    ax.set_ylabel("₹ Lakhs", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.legend(facecolor=SURF, labelcolor=TEXT, fontsize=8)

    # 4. Fraud score distribution
    ax = axes[1, 1]
    ax.set_facecolor(SURF)
    all_sellers = refund_df["fraud_score"].dropna()
    ax.hist(all_sellers, bins=40, color=BLUE, edgecolor=BG, alpha=0.85)
    threshold_val = all_sellers.quantile(0.95)
    ax.axvline(threshold_val, color=RED, linestyle="--", linewidth=1.5,
               label=f"95th pctile: {threshold_val:.2f}")
    ax.set_title("Fraud Score Distribution\n(Seller-Level)", color=TEXT, fontsize=11)
    ax.set_xlabel("Composite Fraud Score", color=TEXT)
    ax.set_ylabel("# Sellers", color=TEXT)
    ax.tick_params(colors=TEXT)
    ax.legend(facecolor=SURF, labelcolor=TEXT)

    plt.tight_layout()
    out = os.path.join(EXPORTS_DIR, "anomaly_detection_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  📊 Anomaly dashboard saved: {out}")


def main():
    print("\n🚨 Running Fraud & Anomaly Detection...")
    orders = load_data()

    velocity_df = detect_high_velocity_sellers(orders)
    refund_df   = detect_refund_anomalies(orders)
    gmv_df      = detect_gmv_spikes(orders)

    velocity_df.to_csv(os.path.join(EXPORTS_DIR, "high_velocity_sellers.csv"), index=False)
    refund_df.to_csv(os.path.join(EXPORTS_DIR, "refund_anomalies.csv"), index=False)
    gmv_df.to_csv(os.path.join(EXPORTS_DIR, "gmv_spike_log.csv"), index=False)

    plot_anomaly_dashboard(velocity_df, refund_df, gmv_df)

    print("\n" + "═" * 55)
    print("  ANOMALY DETECTION REPORT")
    print("═" * 55)
    print(f"  High-velocity seller events  : {len(velocity_df):,}")
    print(f"  Sellers w/ abnormal refunds  : {len(refund_df):,}")
    print(f"  GMV spike days detected      : {gmv_df['is_spike'].sum()}")
    print(f"\n  Top 5 high-velocity sellers:")
    for _, row in velocity_df.head(5).iterrows():
        print(f"    {row['seller_id']}  Z={row['combined_z']:.2f}  Orders={int(row['weekly_orders'])}")
    print("═" * 55 + "\n")
    print("✅ Anomaly detection complete.\n")


if __name__ == "__main__":
    main()
