"""
generate_data.py
----------------
Generates realistic synthetic e-commerce data for analysis.
Simulates orders, sellers, payments, marketing events, and user sessions.

Run: python python/etl/generate_data.py
"""

import pandas as pd
import numpy as np
import sqlite3
import os
import random
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

N_SELLERS   = 500
N_CUSTOMERS = 20_000
N_ORDERS    = 150_000
N_SESSIONS  = 300_000
START_DATE  = datetime(2023, 1, 1)
END_DATE    = datetime(2024, 6, 30)

RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
os.makedirs(RAW_DIR, exist_ok=True)

MARKETING_CHANNELS = [
    "organic_search", "paid_search", "email", "social_paid",
    "social_organic", "direct", "affiliate", "push_notification"
]

CATEGORIES = [
    "Electronics", "Fashion", "Home & Kitchen", "Beauty",
    "Sports", "Books", "Toys", "Grocery", "Automotive", "Jewellery"
]

STATES = [
    "Karnataka", "Maharashtra", "Delhi", "Tamil Nadu", "Telangana",
    "Gujarat", "Rajasthan", "West Bengal", "Kerala", "Punjab"
]


def random_dates(start, end, n):
    delta = (end - start).days
    return [start + timedelta(days=random.randint(0, delta),
                              hours=random.randint(0, 23),
                              minutes=random.randint(0, 59))
            for _ in range(n)]


# ── 1. Sellers ───────────────────────────────────────────────────────────────
def generate_sellers():
    print("  Generating sellers...")
    seller_ids = [f"S{str(i).zfill(5)}" for i in range(1, N_SELLERS + 1)]
    df = pd.DataFrame({
        "seller_id"      : seller_ids,
        "seller_name"    : [f"Seller_{i}" for i in range(1, N_SELLERS + 1)],
        "category"       : np.random.choice(CATEGORIES, N_SELLERS),
        "state"          : np.random.choice(STATES, N_SELLERS),
        "tier"           : np.random.choice(["Gold", "Silver", "Bronze", "Platinum"],
                                             N_SELLERS, p=[0.1, 0.3, 0.45, 0.15]),
        "joined_date"    : [START_DATE - timedelta(days=random.randint(30, 900))
                            for _ in range(N_SELLERS)],
        "commission_pct" : np.round(np.random.uniform(5, 20, N_SELLERS), 2),
        "is_active"      : np.random.choice([1, 0], N_SELLERS, p=[0.92, 0.08]),
    })
    path = os.path.join(RAW_DIR, "sellers.csv")
    df.to_csv(path, index=False)
    print(f"    → {len(df):,} sellers saved to {path}")
    return df


# ── 2. Customers ─────────────────────────────────────────────────────────────
def generate_customers():
    print("  Generating customers...")
    customer_ids = [f"C{str(i).zfill(7)}" for i in range(1, N_CUSTOMERS + 1)]
    reg_dates = random_dates(START_DATE - timedelta(days=365), END_DATE, N_CUSTOMERS)
    df = pd.DataFrame({
        "customer_id"      : customer_ids,
        "state"            : np.random.choice(STATES, N_CUSTOMERS),
        "age_group"        : np.random.choice(["18-24","25-34","35-44","45-54","55+"],
                                               N_CUSTOMERS, p=[0.18,0.35,0.27,0.13,0.07]),
        "gender"           : np.random.choice(["M","F","Other"], N_CUSTOMERS, p=[0.52,0.46,0.02]),
        "registration_date": reg_dates,
        "acquisition_channel": np.random.choice(MARKETING_CHANNELS, N_CUSTOMERS,
                                                  p=[0.22,0.18,0.15,0.12,0.10,0.10,0.08,0.05]),
        "is_prime"         : np.random.choice([1, 0], N_CUSTOMERS, p=[0.35, 0.65]),
    })
    path = os.path.join(RAW_DIR, "customers.csv")
    df.to_csv(path, index=False)
    print(f"    → {len(df):,} customers saved to {path}")
    return df


# ── 3. Orders ────────────────────────────────────────────────────────────────
def generate_orders(sellers_df, customers_df):
    print("  Generating orders...")
    order_dates = random_dates(START_DATE, END_DATE, N_ORDERS)
    gmv_vals    = np.random.lognormal(mean=7.5, sigma=1.2, size=N_ORDERS)
    gmv_vals    = np.clip(gmv_vals, 99, 150_000)

    statuses = np.random.choice(
        ["delivered", "returned", "cancelled", "pending"],
        N_ORDERS, p=[0.78, 0.10, 0.08, 0.04]
    )

    df = pd.DataFrame({
        "order_id"        : [f"ORD{str(i).zfill(9)}" for i in range(1, N_ORDERS + 1)],
        "customer_id"     : np.random.choice(customers_df["customer_id"], N_ORDERS),
        "seller_id"       : np.random.choice(sellers_df["seller_id"], N_ORDERS),
        "order_date"      : order_dates,
        "category"        : np.random.choice(CATEGORIES, N_ORDERS),
        "gmv"             : np.round(gmv_vals, 2),
        "discount_amount" : np.round(gmv_vals * np.random.uniform(0, 0.35, N_ORDERS), 2),
        "shipping_fee"    : np.round(np.random.uniform(0, 150, N_ORDERS), 2),
        "status"          : statuses,
        "payment_method"  : np.random.choice(
                                ["UPI","Credit Card","Debit Card","Net Banking","COD","Wallet"],
                                N_ORDERS, p=[0.38,0.22,0.15,0.08,0.10,0.07]),
        "channel"         : np.random.choice(MARKETING_CHANNELS, N_ORDERS,
                                              p=[0.22,0.18,0.15,0.12,0.10,0.10,0.08,0.05]),
    })
    df["net_revenue"] = np.round(
        df["gmv"] - df["discount_amount"] + df["shipping_fee"], 2
    )
    path = os.path.join(RAW_DIR, "orders.csv")
    df.to_csv(path, index=False)
    print(f"    → {len(df):,} orders saved to {path}")
    return df


# ── 4. Payments / Settlements ─────────────────────────────────────────────────
def generate_payments(orders_df, sellers_df):
    print("  Generating payments & settlements (with intentional discrepancies)...")
    delivered = orders_df[orders_df["status"] == "delivered"].copy()

    # commission lookup
    comm = sellers_df.set_index("seller_id")["commission_pct"].to_dict()

    rows = []
    discrepancy_count = 0
    for _, row in delivered.iterrows():
        pct = comm.get(row["seller_id"], 10) / 100
        expected_payout = round(row["net_revenue"] * (1 - pct), 2)

        # inject ~3% discrepancies — this is what the "forensic SQL" finds
        introduce_error = random.random() < 0.03
        if introduce_error:
            actual_payout = round(expected_payout * random.uniform(0.4, 0.85), 2)
            discrepancy_count += 1
        else:
            actual_payout = expected_payout

        settlement_date = row["order_date"] + timedelta(days=random.randint(7, 21))

        rows.append({
            "payment_id"      : f"PAY{str(len(rows)+1).zfill(9)}",
            "order_id"        : row["order_id"],
            "seller_id"       : row["seller_id"],
            "expected_payout" : expected_payout,
            "actual_payout"   : actual_payout,
            "discrepancy_flag": 1 if introduce_error else 0,
            "discrepancy_amt" : round(expected_payout - actual_payout, 2),
            "settlement_date" : settlement_date,
            "settlement_cycle": f"W{settlement_date.isocalendar()[1]}",
            "status"          : "settled" if not introduce_error else "disputed",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "payments.csv")
    df.to_csv(path, index=False)
    total_disc = df[df["discrepancy_flag"] == 1]["discrepancy_amt"].sum()
    print(f"    → {len(df):,} payment records | {discrepancy_count} discrepancies "
          f"| ₹{total_disc:,.0f} at stake")
    return df


# ── 5. Marketing Spend ────────────────────────────────────────────────────────
def generate_marketing_spend():
    print("  Generating marketing spend data...")
    rows = []
    current = START_DATE
    while current <= END_DATE:
        for ch in MARKETING_CHANNELS:
            base_spend = {
                "organic_search": 0, "paid_search": 250_000, "email": 80_000,
                "social_paid": 180_000, "social_organic": 20_000,
                "direct": 0, "affiliate": 120_000, "push_notification": 30_000,
            }[ch]
            spend = max(0, base_spend * random.uniform(0.7, 1.4)) if base_spend > 0 else 0
            rows.append({
                "date"          : current.date(),
                "channel"       : ch,
                "spend_inr"     : round(spend, 2),
                "impressions"   : int(spend * random.uniform(50, 200)) if spend > 0 else int(random.uniform(5000, 50000)),
                "clicks"        : int(spend * random.uniform(2, 8)) if spend > 0 else int(random.uniform(500, 5000)),
            })
        current += timedelta(days=1)
    df = pd.DataFrame(rows)
    path = os.path.join(RAW_DIR, "marketing_spend.csv")
    df.to_csv(path, index=False)
    print(f"    → {len(df):,} daily channel spend records saved")
    return df


# ── 6. Sessions (for A/B test) ─────────────────────────────────────────────
def generate_sessions(customers_df):
    print("  Generating user sessions with A/B test variant assignment...")
    session_dates = random_dates(START_DATE, END_DATE, N_SESSIONS)
    # 50-50 split on checkout redesign experiment
    variants = np.random.choice(["control", "variant_b"], N_SESSIONS, p=[0.5, 0.5])
    # Variant B has slightly higher conversion
    base_conv = np.where(variants == "control", 0.031, 0.033)
    converted  = np.random.binomial(1, base_conv)

    df = pd.DataFrame({
        "session_id"   : [f"SES{str(i).zfill(10)}" for i in range(1, N_SESSIONS + 1)],
        "customer_id"  : np.random.choice(customers_df["customer_id"], N_SESSIONS),
        "session_date" : session_dates,
        "channel"      : np.random.choice(MARKETING_CHANNELS, N_SESSIONS),
        "device"       : np.random.choice(["mobile","desktop","tablet"], N_SESSIONS, p=[0.64,0.30,0.06]),
        "ab_variant"   : variants,
        "pages_viewed" : np.random.randint(1, 18, N_SESSIONS),
        "session_duration_sec": np.random.exponential(scale=240, size=N_SESSIONS).astype(int),
        "converted"    : converted,
        "bounce"       : np.random.choice([1, 0], N_SESSIONS, p=[0.38, 0.62]),
    })
    path = os.path.join(RAW_DIR, "sessions.csv")
    df.to_csv(path, index=False)
    print(f"    → {len(df):,} sessions | "
          f"control conv={converted[variants=='control'].mean():.3f} | "
          f"variant_b conv={converted[variants=='variant_b'].mean():.3f}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🔧 Generating synthetic e-commerce dataset...\n")
    sellers   = generate_sellers()
    customers = generate_customers()
    orders    = generate_orders(sellers, customers)
    payments  = generate_payments(orders, sellers)
    _         = generate_marketing_spend()
    _         = generate_sessions(customers)

    # ── Write to SQLite for SQL modules ──────────────────────────────────────
    db_path = os.path.join(RAW_DIR, "ecommerce.db")
    conn = sqlite3.connect(db_path)
    sellers.to_sql("sellers",   conn, if_exists="replace", index=False)
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    orders.to_sql("orders",     conn, if_exists="replace", index=False)
    payments.to_sql("payments", conn, if_exists="replace", index=False)
    conn.close()
    print(f"\n✅ SQLite database written to {db_path}")
    print("\n✅ All raw data generated successfully. Run pipeline.py next.\n")


if __name__ == "__main__":
    main()
