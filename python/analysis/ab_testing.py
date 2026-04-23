"""
ab_testing.py
-------------
Statistically rigorous A/B test evaluation framework.
Covers: two-proportion z-test, confidence intervals, p-values,
sample size calculator, minimum detectable effect, and power analysis.

Evaluates the checkout redesign experiment (Variant B vs Control).

Run: python python/analysis/ab_testing.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from scipy import stats
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "../../data/raw/ecommerce.db")
EXPORTS_DIR = os.path.join(BASE_DIR, "../../data/exports")
RAW_DIR     = os.path.join(BASE_DIR, "../../data/raw")
os.makedirs(EXPORTS_DIR, exist_ok=True)

BG, SURF, TEXT = "#0A0E1A", "#141929", "#DDE2F0"
CTRL_COLOR, VAR_COLOR = "#5B9CF6", "#2ECC71"
DANGER = "#E74C3C"


# ── Statistical functions ─────────────────────────────────────────────────────

def two_proportion_z_test(
    n_control: int, conv_control: int,
    n_variant: int, conv_variant: int,
    alpha: float = 0.05
) -> dict:
    """
    Two-proportion z-test for conversion rate comparison.
    Returns full statistical summary.
    """
    p_ctrl = conv_control / n_control
    p_var  = conv_variant / n_variant

    # Pooled proportion under H0
    p_pool = (conv_control + conv_variant) / (n_control + n_variant)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_variant))
    z_stat = (p_var - p_ctrl) / se
    p_val  = 2 * (1 - norm.cdf(abs(z_stat)))  # two-tailed

    # 95% CI for the difference
    se_diff  = np.sqrt(p_ctrl*(1-p_ctrl)/n_control + p_var*(1-p_var)/n_variant)
    z_crit   = norm.ppf(1 - alpha/2)
    diff     = p_var - p_ctrl
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    # Relative lift
    rel_lift = (p_var - p_ctrl) / p_ctrl * 100

    return {
        "control_n"        : n_control,
        "variant_n"        : n_variant,
        "control_conv_rate": round(p_ctrl * 100, 4),
        "variant_conv_rate": round(p_var * 100, 4),
        "absolute_lift_pp" : round(diff * 100, 4),
        "relative_lift_pct": round(rel_lift, 2),
        "z_statistic"      : round(z_stat, 4),
        "p_value"          : round(p_val, 6),
        "ci_95_lower_pp"   : round(ci_lower * 100, 4),
        "ci_95_upper_pp"   : round(ci_upper * 100, 4),
        "is_significant"   : p_val < alpha,
        "alpha"            : alpha,
    }


def sample_size_required(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """
    Minimum sample size per variant for given effect and power.
    """
    p1  = baseline_rate
    p2  = baseline_rate + min_detectable_effect
    z_a = norm.ppf(1 - alpha/2)
    z_b = norm.ppf(power)
    p_avg = (p1 + p2) / 2
    n = ((z_a * np.sqrt(2 * p_avg * (1 - p_avg)) + z_b * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2) / (p2 - p1) ** 2
    return int(np.ceil(n))


def compute_power(
    n: int, baseline: float, effect: float, alpha: float = 0.05
) -> float:
    """Observed statistical power for given n and effect."""
    p1 = baseline
    p2 = baseline + effect
    p_pool = (p1 + p2) / 2
    se  = np.sqrt(2 * p_pool * (1 - p_pool) / n)
    z_a = norm.ppf(1 - alpha/2)
    z   = (abs(p2 - p1) / se) - z_a
    return round(norm.cdf(z), 4)


# ── Segment analysis ──────────────────────────────────────────────────────────
def segment_analysis(sessions: pd.DataFrame) -> pd.DataFrame:
    """Break down A/B results by device type."""
    rows = []
    for seg, grp in sessions.groupby("device"):
        ctrl = grp[grp["ab_variant"] == "control"]
        var  = grp[grp["ab_variant"] == "variant_b"]
        if len(ctrl) < 100 or len(var) < 100:
            continue
        res = two_proportion_z_test(
            len(ctrl), ctrl["converted"].sum(),
            len(var),  var["converted"].sum()
        )
        res["segment"] = seg
        rows.append(res)
    return pd.DataFrame(rows).sort_values("relative_lift_pct", ascending=False)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_ab_results(result: dict, sessions: pd.DataFrame, seg_df: pd.DataFrame):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35,
                            left=0.07, right=0.97, top=0.92, bottom=0.08)
    fig.suptitle("A/B Test Evaluation — Checkout Redesign Experiment",
                 fontsize=16, color=TEXT, fontweight="bold")

    # ── 1. Conversion rate comparison ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(SURF)
    rates  = [result["control_conv_rate"], result["variant_conv_rate"]]
    colors = [CTRL_COLOR, VAR_COLOR]
    bars   = ax1.bar(["Control", "Variant B"], rates, color=colors, width=0.5, alpha=0.9)
    for bar, val in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.2f}%", ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")
    ax1.set_title("Conversion Rate Comparison", color=TEXT, fontsize=11)
    ax1.set_ylabel("Conversion Rate (%)", color=TEXT)
    ax1.tick_params(colors=TEXT)
    sig_label = "✅ Significant" if result["is_significant"] else "❌ Not Significant"
    ax1.text(0.5, 0.05, f"p = {result['p_value']:.4f}  |  {sig_label}",
             transform=ax1.transAxes, ha="center", color=TEXT, fontsize=9)

    # ── 2. Confidence interval ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(SURF)
    diff   = result["absolute_lift_pp"]
    ci_lo  = result["ci_95_lower_pp"]
    ci_hi  = result["ci_95_upper_pp"]
    color  = VAR_COLOR if diff > 0 else DANGER
    ax2.errorbar(["Absolute Lift"], [diff], yerr=[[diff - ci_lo], [ci_hi - diff]],
                 fmt="o", color=color, ecolor="white", capsize=12, markersize=10, linewidth=2)
    ax2.axhline(0, color="white", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_title("Lift with 95% Confidence Interval\n(percentage points)", color=TEXT, fontsize=11)
    ax2.set_ylabel("Conversion Lift (pp)", color=TEXT)
    ax2.tick_params(colors=TEXT)
    ax2.text(0, diff + (ci_hi - diff) * 0.4,
             f"+{diff:.2f}pp\n[{ci_lo:.2f}, {ci_hi:.2f}]",
             ha="center", color=TEXT, fontsize=9)

    # ── 3. Daily conversion trend ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(SURF)
    sessions["session_date"] = pd.to_datetime(sessions["session_date"])
    sessions["date"] = sessions["session_date"].dt.date
    daily = (
        sessions.groupby(["date", "ab_variant"])["converted"]
        .agg(["sum", "count"])
        .reset_index()
    )
    daily["rate"] = daily["sum"] / daily["count"] * 100
    for variant, color in [("control", CTRL_COLOR), ("variant_b", VAR_COLOR)]:
        vd = daily[daily["ab_variant"] == variant].sort_values("date")
        # 7-day rolling
        vd = vd.copy()
        vd["rolling"] = vd["rate"].rolling(7).mean()
        ax3.plot(range(len(vd)), vd["rolling"], color=color, linewidth=2,
                 label="Control" if variant == "control" else "Variant B")
    ax3.set_title("7-Day Rolling Conversion Rate (%)", color=TEXT, fontsize=11)
    ax3.set_ylabel("Conv Rate %", color=TEXT)
    ax3.tick_params(colors=TEXT)
    ax3.legend(facecolor=SURF, labelcolor=TEXT)
    ax3.set_xlabel("Days", color=TEXT)

    # ── 4. Segment breakdown ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(SURF)
    if not seg_df.empty:
        bar_colors = [VAR_COLOR if x > 0 else DANGER for x in seg_df["relative_lift_pct"]]
        ax4.barh(seg_df["segment"], seg_df["relative_lift_pct"], color=bar_colors, height=0.5)
        ax4.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
        ax4.set_title("Relative Lift by Device Segment (%)", color=TEXT, fontsize=11)
        ax4.set_xlabel("Relative Lift %", color=TEXT)
        ax4.tick_params(colors=TEXT)

    # ── 5. Sample size power curve ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(SURF)
    baseline = result["control_conv_rate"] / 100
    ns = np.arange(1000, 200001, 5000)
    powers = [compute_power(n, baseline, 0.002) for n in ns]
    ax5.plot(ns/1000, powers, color=CTRL_COLOR, linewidth=2)
    ax5.axhline(0.8, color=VAR_COLOR, linestyle="--", linewidth=1.5, label="80% Power threshold")
    ax5.axvline(result["control_n"]/1000, color=DANGER, linestyle=":", linewidth=1.5,
                label=f"Current n = {result['control_n']:,}")
    ax5.set_title("Power Curve (MDE = 0.2pp)", color=TEXT, fontsize=11)
    ax5.set_xlabel("Sample Size per Variant (K)", color=TEXT)
    ax5.set_ylabel("Statistical Power", color=TEXT)
    ax5.legend(facecolor=SURF, labelcolor=TEXT, fontsize=8)
    ax5.tick_params(colors=TEXT)

    # ── 6. Summary stats box ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.set_facecolor(SURF)
    summary_text = [
        ("Experiment", "Checkout Redesign"),
        ("Control n", f"{result['control_n']:,}"),
        ("Variant n", f"{result['variant_n']:,}"),
        ("Control Conv", f"{result['control_conv_rate']:.2f}%"),
        ("Variant Conv", f"{result['variant_conv_rate']:.2f}%"),
        ("Absolute Lift", f"+{result['absolute_lift_pp']:.2f}pp"),
        ("Relative Lift", f"+{result['relative_lift_pct']:.1f}%"),
        ("p-value", f"{result['p_value']:.4f}"),
        ("95% CI", f"[{result['ci_95_lower_pp']:.2f}, {result['ci_95_upper_pp']:.2f}]pp"),
        ("Decision", "✅ SHIP IT" if result["is_significant"] else "⏳ CONTINUE"),
    ]
    for i, (label, val) in enumerate(summary_text):
        y = 0.95 - i * 0.092
        ax6.text(0.02, y, label, transform=ax6.transAxes, color="#8B9EC0",
                 fontsize=9, fontweight="bold")
        ax6.text(0.55, y, val, transform=ax6.transAxes, color=TEXT, fontsize=9)
    ax6.set_title("Experiment Summary", color=TEXT, fontsize=11, pad=8)

    out = os.path.join(EXPORTS_DIR, "ab_test_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  📊 A/B test chart saved: {out}")


def main():
    print("\n🧪 Running A/B Testing Analysis...")
    sessions = pd.read_csv(os.path.join(RAW_DIR, "sessions.csv"))

    ctrl = sessions[sessions["ab_variant"] == "control"]
    var  = sessions[sessions["ab_variant"] == "variant_b"]

    result = two_proportion_z_test(
        n_control=len(ctrl),   conv_control=ctrl["converted"].sum(),
        n_variant=len(var),    conv_variant=var["converted"].sum()
    )

    seg_df = segment_analysis(sessions)

    # Sample size reference
    req_n = sample_size_required(
        baseline_rate=result["control_conv_rate"]/100,
        min_detectable_effect=0.002,
    )

    plot_ab_results(result, sessions, seg_df)

    # Save
    pd.DataFrame([result]).to_csv(os.path.join(EXPORTS_DIR, "ab_test_results.csv"), index=False)
    if not seg_df.empty:
        seg_df.to_csv(os.path.join(EXPORTS_DIR, "ab_segment_results.csv"), index=False)

    # Print
    print("\n" + "═" * 55)
    print("  A/B TEST REPORT — Checkout Redesign")
    print("═" * 55)
    print(f"  Control conversion rate  : {result['control_conv_rate']:.3f}%  (n={result['control_n']:,})")
    print(f"  Variant B conv rate      : {result['variant_conv_rate']:.3f}%  (n={result['variant_n']:,})")
    print(f"  Absolute lift            : +{result['absolute_lift_pp']:.3f} pp")
    print(f"  Relative lift            : +{result['relative_lift_pct']:.1f}%")
    print(f"  p-value                  : {result['p_value']:.4f}")
    print(f"  95% CI                   : [{result['ci_95_lower_pp']:.3f}, {result['ci_95_upper_pp']:.3f}] pp")
    print(f"  Statistically significant: {'✅ YES' if result['is_significant'] else '❌ NO'}")
    print(f"  Required n (80% power)   : {req_n:,} per variant")
    print(f"  Decision                 : {'✅ Ship Variant B' if result['is_significant'] else '⏳ Extend test'}")
    print("═" * 55 + "\n")
    print("✅ A/B testing complete.\n")


if __name__ == "__main__":
    main()
