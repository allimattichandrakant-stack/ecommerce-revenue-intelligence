"""
marketing_roi.py
----------------
Multi-touch marketing attribution & ROI analysis across 8 channels.
Computes channel-level ROI, assisted conversion value, budget efficiency,
and produces recommendation scorecard for budget reallocation.

Run: python python/analysis/marketing_roi.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "../../data/raw/ecommerce.db")
EXPORTS_DIR = os.path.join(BASE_DIR, "../../data/exports")
RAW_DIR     = os.path.join(BASE_DIR, "../../data/raw")
os.makedirs(EXPORTS_DIR, exist_ok=True)

COLORS = {
    "bg": "#0A0E1A", "surface": "#141929", "text": "#DDE2F0",
    "accent": "#5B9CF6", "green": "#2ECC71", "red": "#E74C3C",
    "orange": "#F39C12", "purple": "#9B59B6",
}

CHANNEL_COLORS = [
    "#5B9CF6", "#2ECC71", "#E74C3C", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6"
]


def load_data():
    conn   = sqlite3.connect(DB_PATH)
    orders = pd.read_sql("SELECT * FROM orders_clean", conn)
    conn.close()
    spend  = pd.read_csv(os.path.join(RAW_DIR, "marketing_spend.csv"))
    return orders, spend


def compute_channel_roi(orders: pd.DataFrame, spend: pd.DataFrame) -> pd.DataFrame:
    """
    Last-touch attribution: revenue credited to the channel that
    drove the final session before conversion.
    """
    delivered = orders[orders["is_delivered"] == 1].copy()

    # Revenue by channel
    channel_rev = (
        delivered.groupby("channel")
        .agg(
            orders=("order_id", "count"),
            gmv=("gmv", "sum"),
            net_revenue=("net_revenue", "sum"),
            avg_order_value=("gmv", "mean"),
        )
        .reset_index()
        .rename(columns={"channel": "channel"})
    )

    # Total spend by channel
    channel_spend = (
        spend.groupby("channel")
        .agg(
            total_spend=("spend_inr", "sum"),
            total_impressions=("impressions", "sum"),
            total_clicks=("clicks", "sum"),
        )
        .reset_index()
    )

    roi_df = channel_rev.merge(channel_spend, on="channel", how="left")
    roi_df["total_spend"] = roi_df["total_spend"].fillna(0)

    # Metrics
    roi_df["roi"]             = np.where(
        roi_df["total_spend"] > 0,
        np.round(roi_df["net_revenue"] / roi_df["total_spend"], 2),
        np.nan
    )
    roi_df["cpa"]             = np.where(
        roi_df["total_spend"] > 0,
        np.round(roi_df["total_spend"] / roi_df["orders"], 2),
        0
    )
    roi_df["ctr"]             = np.where(
        roi_df["total_impressions"] > 0,
        np.round(roi_df["total_clicks"] / roi_df["total_impressions"] * 100, 3),
        0
    )
    roi_df["revenue_share"]   = np.round(roi_df["net_revenue"] / roi_df["net_revenue"].sum() * 100, 2)
    roi_df["spend_share"]     = np.where(
        roi_df["total_spend"].sum() > 0,
        np.round(roi_df["total_spend"] / roi_df["total_spend"].sum() * 100, 2),
        0
    )
    roi_df["efficiency_score"] = np.where(
        roi_df["roi"].notna() & (roi_df["roi"] > 0),
        np.round(roi_df["revenue_share"] / roi_df["spend_share"].replace(0, np.nan), 2),
        np.nan
    )

    return roi_df.sort_values("roi", ascending=False, na_position="last")


def compute_weekly_trend(orders: pd.DataFrame) -> pd.DataFrame:
    delivered = orders[orders["is_delivered"] == 1].copy()
    delivered["order_date"] = pd.to_datetime(delivered["order_date"])
    delivered["week"] = delivered["order_date"].dt.to_period("W").dt.start_time
    trend = (
        delivered.groupby(["week", "channel"])
        .agg(net_revenue=("net_revenue", "sum"))
        .reset_index()
    )
    return trend


def plot_roi_dashboard(roi_df: pd.DataFrame, trend: pd.DataFrame):
    fig = plt.figure(figsize=(18, 11), facecolor=COLORS["bg"])
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32,
                            left=0.06, right=0.97, top=0.93, bottom=0.07)
    fig.suptitle("Marketing Channel ROI & Attribution Dashboard",
                 fontsize=17, color=COLORS["text"], fontweight="bold")

    channels = roi_df["channel"].tolist()
    c_map    = {ch: CHANNEL_COLORS[i % len(CHANNEL_COLORS)] for i, ch in enumerate(channels)}

    # ── 1. ROI by channel ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    paid = roi_df[roi_df["roi"].notna()].copy()
    bars = ax1.barh(paid["channel"], paid["roi"], color=[c_map[c] for c in paid["channel"]], height=0.65)
    ax1.set_title("ROI by Channel\n(Net Revenue / Spend)", color=COLORS["text"], fontsize=11)
    ax1.set_xlabel("ROI (x)", color=COLORS["text"])
    ax1.axvline(1, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, val in zip(bars, paid["roi"]):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}x", va="center", color=COLORS["text"], fontsize=8)
    ax1.set_facecolor(COLORS["surface"])
    ax1.tick_params(colors=COLORS["text"])

    # ── 2. Revenue vs Spend share (bubble chart) ─────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    paid2 = roi_df[roi_df["total_spend"] > 0].copy()
    scatter = ax2.scatter(
        paid2["spend_share"], paid2["revenue_share"],
        s=paid2["orders"] / 40,
        c=[c_map[c] for c in paid2["channel"]],
        alpha=0.85, edgecolors="white", linewidth=0.5
    )
    ax2.plot([0, 40], [0, 40], "--", color="white", alpha=0.4, linewidth=1)
    for _, row in paid2.iterrows():
        ax2.annotate(row["channel"].replace("_", "\n"),
                     (row["spend_share"], row["revenue_share"]),
                     fontsize=7, color=COLORS["text"],
                     xytext=(4, 4), textcoords="offset points")
    ax2.set_title("Revenue Share vs Spend Share\n(Above diagonal = efficient)", color=COLORS["text"], fontsize=11)
    ax2.set_xlabel("Spend Share %", color=COLORS["text"])
    ax2.set_ylabel("Revenue Share %", color=COLORS["text"])
    ax2.set_facecolor(COLORS["surface"])
    ax2.tick_params(colors=COLORS["text"])

    # ── 3. CPA by channel ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    paid3 = roi_df[roi_df["cpa"] > 0].sort_values("cpa")
    ax3.barh(paid3["channel"], paid3["cpa"],
             color=[c_map[c] for c in paid3["channel"]], height=0.65)
    ax3.set_title("Cost Per Acquisition (₹)\nby Channel", color=COLORS["text"], fontsize=11)
    ax3.set_xlabel("CPA (₹)", color=COLORS["text"])
    ax3.set_facecolor(COLORS["surface"])
    ax3.tick_params(colors=COLORS["text"])

    # ── 4. Weekly revenue trend top 4 channels ───────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    top4 = roi_df.head(4)["channel"].tolist()
    for ch in top4:
        ch_data = trend[trend["channel"] == ch].sort_values("week")
        ax4.plot(ch_data["week"], ch_data["net_revenue"]/1e5,
                 label=ch, color=c_map[ch], linewidth=1.8, marker="o", markersize=2)
    ax4.set_title("Weekly Net Revenue Trend — Top 4 Channels (₹ Lakhs)", color=COLORS["text"], fontsize=11)
    ax4.set_ylabel("₹ Lakhs", color=COLORS["text"])
    ax4.legend(facecolor=COLORS["surface"], labelcolor=COLORS["text"], fontsize=8)
    ax4.set_facecolor(COLORS["surface"])
    ax4.tick_params(colors=COLORS["text"])

    # ── 5. Scorecard table ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.set_facecolor(COLORS["surface"])
    scorecard = roi_df[["channel", "roi", "cpa", "revenue_share"]].copy()
    scorecard["roi"] = scorecard["roi"].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "organic")
    scorecard["cpa"] = scorecard["cpa"].apply(lambda x: f"₹{x:,.0f}" if x > 0 else "—")
    scorecard["revenue_share"] = scorecard["revenue_share"].apply(lambda x: f"{x:.1f}%")
    scorecard.columns = ["Channel", "ROI", "CPA", "Rev%"]
    tbl = ax5.table(cellText=scorecard.values, colLabels=scorecard.columns,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor(COLORS["bg"] if r == 0 else COLORS["surface"])
        cell.set_text_props(color=COLORS["text"])
        cell.set_edgecolor("#2A2D35")
    ax5.set_title("Channel Scorecard", color=COLORS["text"], fontsize=11, pad=8)

    out = os.path.join(EXPORTS_DIR, "marketing_roi_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"  📊 Dashboard saved: {out}")


def print_roi_report(roi_df: pd.DataFrame):
    print("\n" + "═" * 65)
    print("  MARKETING ROI ANALYSIS REPORT")
    print("═" * 65)
    print(f"\n  {'Channel':<22} {'ROI':>6} {'CPA':>10} {'Rev Share':>10} {'Efficiency':>12}")
    print("  " + "-" * 62)
    for _, row in roi_df.iterrows():
        roi_str  = f"{row['roi']:.1f}x" if pd.notna(row["roi"]) else "organic"
        cpa_str  = f"₹{row['cpa']:,.0f}" if row["cpa"] > 0 else "—"
        eff_str  = f"{row['efficiency_score']:.2f}" if pd.notna(row.get("efficiency_score")) else "—"
        flag = "⭐" if pd.notna(row["roi"]) and row["roi"] >= 3 else ("⚠️" if pd.notna(row["roi"]) and row["roi"] < 1.5 else "  ")
        print(f"  {flag} {row['channel']:<20} {roi_str:>6} {cpa_str:>10} "
              f"{row['revenue_share']:>9.1f}% {eff_str:>12}")

    # Recommendation
    best = roi_df[roi_df["roi"].notna()].iloc[0]
    worst = roi_df[roi_df["roi"].notna()].iloc[-1]
    print(f"\n  📌 RECOMMENDATION:")
    print(f"     Increase budget allocation to '{best['channel']}' (ROI: {best['roi']:.1f}x)")
    print(f"     Review spend on '{worst['channel']}' (ROI: {worst['roi']:.1f}x, "
          f"CPA: ₹{worst['cpa']:,.0f})")
    print("═" * 65 + "\n")


def main():
    print("\n📣 Running Marketing ROI Analysis...")
    orders, spend = load_data()
    roi_df        = compute_channel_roi(orders, spend)
    trend         = compute_weekly_trend(orders)

    roi_df.to_csv(os.path.join(EXPORTS_DIR, "channel_roi_summary.csv"), index=False)
    plot_roi_dashboard(roi_df, trend)
    print_roi_report(roi_df)
    print("✅ Marketing analysis complete.\n")


if __name__ == "__main__":
    main()
