"""
test_pipeline.py
----------------
Unit tests for ETL pipeline and analysis modules.
Run: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.etl.pipeline import (
    transform_orders,
    transform_payments,
    transform_customers,
    check_duplicates,
)
from python.analysis.ab_testing import (
    two_proportion_z_test,
    sample_size_required,
    compute_power,
)


# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture
def sample_orders():
    return pd.DataFrame({
        "order_id"       : ["ORD001", "ORD002", "ORD003", "ORD004"],
        "customer_id"    : ["C001", "C002", "C001", "C003"],
        "seller_id"      : ["S001", "S001", "S002", "S002"],
        "order_date"     : pd.to_datetime(["2024-01-15", "2024-02-10", "2024-03-01", "2024-04-05"]),
        "category"       : ["Electronics", "Fashion", "Electronics", "Home & Kitchen"],
        "gmv"            : [12000.0, 3500.0, 8000.0, 1200.0],
        "discount_amount": [1200.0, 350.0, 0.0, 60.0],
        "shipping_fee"   : [0.0, 99.0, 49.0, 0.0],
        "net_revenue"    : [10800.0, 3249.0, 8049.0, 1140.0],
        "status"         : ["delivered", "delivered", "returned", "delivered"],
        "payment_method" : ["UPI", "Credit Card", "UPI", "Debit Card"],
        "channel"        : ["paid_search", "email", "organic_search", "direct"],
    })


@pytest.fixture
def sample_payments():
    return pd.DataFrame({
        "payment_id"      : ["PAY001", "PAY002", "PAY003"],
        "order_id"        : ["ORD001", "ORD002", "ORD004"],
        "seller_id"       : ["S001", "S001", "S002"],
        "expected_payout" : [9720.0, 2924.1, 1026.0],
        "actual_payout"   : [9720.0, 2000.0, 1026.0],   # PAY002 is disputed
        "discrepancy_flag": [0, 1, 0],
        "discrepancy_amt" : [0.0, 924.1, 0.0],
        "discrepancy_pct" : [0.0, 31.6, 0.0],
        "settlement_date" : pd.to_datetime(["2024-01-22", "2024-02-17", "2024-04-12"]),
        "settlement_cycle": ["W04", "W07", "W15"],
        "status"          : ["settled", "disputed", "settled"],
    })


@pytest.fixture
def sample_customers():
    return pd.DataFrame({
        "customer_id"        : ["C001", "C002", "C003"],
        "state"              : ["Karnataka", "Maharashtra", "Delhi"],
        "age_group"          : ["25-34", "18-24", "35-44"],
        "gender"             : ["M", "F", "M"],
        "registration_date"  : pd.to_datetime(["2023-06-01", "2023-01-15", "2022-11-10"]),
        "acquisition_channel": ["paid_search", "organic_search", "email"],
        "is_prime"           : [1, 0, 1],
    })


# ── ETL Transform Tests ───────────────────────────────────────
class TestTransformOrders:

    def test_derived_time_dimensions(self, sample_orders):
        result = transform_orders(sample_orders)
        assert "order_year" in result.columns
        assert "order_month" in result.columns
        assert "order_quarter" in result.columns
        assert result["order_year"].iloc[0] == 2024
        assert result["order_month"].iloc[0] == 1

    def test_delivered_flag(self, sample_orders):
        result = transform_orders(sample_orders)
        assert result.loc[result["order_id"] == "ORD001", "is_delivered"].iloc[0] == 1
        assert result.loc[result["order_id"] == "ORD003", "is_returned"].iloc[0] == 1

    def test_discount_rate_calculation(self, sample_orders):
        result = transform_orders(sample_orders)
        ord1 = result[result["order_id"] == "ORD001"].iloc[0]
        expected_rate = round(1200.0 / 12000.0, 4)
        assert ord1["discount_rate"] == expected_rate

    def test_discount_rate_no_negative(self, sample_orders):
        result = transform_orders(sample_orders)
        assert (result["discount_rate"] >= 0).all()
        assert (result["discount_rate"] <= 1).all()

    def test_is_weekend_flag(self, sample_orders):
        result = transform_orders(sample_orders)
        assert "is_weekend" in result.columns

    def test_output_row_count_preserved(self, sample_orders):
        result = transform_orders(sample_orders)
        assert len(result) == len(sample_orders)


class TestTransformPayments:

    def test_discrepancy_pct_preserved(self, sample_payments):
        result = transform_payments(sample_payments)
        assert "discrepancy_pct" in result.columns

    def test_severity_column_created(self, sample_payments):
        result = transform_payments(sample_payments)
        assert "discrepancy_severity" in result.columns

    def test_zero_discrepancy_is_none_severity(self, sample_payments):
        result = transform_payments(sample_payments)
        clean = result[result["discrepancy_flag"] == 0]
        assert (clean["discrepancy_severity"].astype(str) == "none").all()


class TestCheckDuplicates:

    def test_no_duplicates_unchanged(self, sample_orders):
        result = check_duplicates(sample_orders, "order_id", "orders")
        assert len(result) == len(sample_orders)

    def test_duplicates_dropped(self, sample_orders):
        duped = pd.concat([sample_orders, sample_orders.head(1)], ignore_index=True)
        result = check_duplicates(duped, "order_id", "orders")
        assert len(result) == len(sample_orders)


# ── A/B Testing Tests ─────────────────────────────────────────
class TestABTesting:

    def test_significant_result(self):
        # Large sample, clear effect — should be significant
        result = two_proportion_z_test(
            n_control=50_000, conv_control=1500,
            n_variant=50_000, conv_variant=1800
        )
        assert result["is_significant"] is True
        assert result["p_value"] < 0.05

    def test_not_significant_result(self):
        # Tiny effect, moderate sample — should NOT be significant
        result = two_proportion_z_test(
            n_control=500, conv_control=15,
            n_variant=500, conv_variant=16
        )
        assert result["is_significant"] is False

    def test_positive_lift(self):
        result = two_proportion_z_test(
            n_control=10_000, conv_control=300,
            n_variant=10_000, conv_variant=380
        )
        assert result["relative_lift_pct"] > 0
        assert result["absolute_lift_pp"] > 0

    def test_ci_contains_zero_when_not_significant(self):
        result = two_proportion_z_test(
            n_control=200, conv_control=10,
            n_variant=200, conv_variant=11
        )
        if not result["is_significant"]:
            assert result["ci_95_lower_pp"] < 0 or result["ci_95_upper_pp"] > 0

    def test_sample_size_increases_with_smaller_mde(self):
        n_large_mde = sample_size_required(baseline_rate=0.03, min_detectable_effect=0.01)
        n_small_mde = sample_size_required(baseline_rate=0.03, min_detectable_effect=0.002)
        assert n_small_mde > n_large_mde

    def test_power_increases_with_sample_size(self):
        p1 = compute_power(n=1000,  baseline=0.03, effect=0.005)
        p2 = compute_power(n=20000, baseline=0.03, effect=0.005)
        assert p2 > p1

    def test_power_between_0_and_1(self):
        p = compute_power(n=5000, baseline=0.03, effect=0.003)
        assert 0.0 <= p <= 1.0

    def test_conversion_rates_in_output(self):
        result = two_proportion_z_test(
            n_control=1000, conv_control=30,
            n_variant=1000, conv_variant=35
        )
        assert abs(result["control_conv_rate"] - 3.0) < 0.01
        assert abs(result["variant_conv_rate"] - 3.5) < 0.01


# ── Data Quality Tests ────────────────────────────────────────
class TestDataQuality:

    def test_gmv_non_negative(self, sample_orders):
        assert (sample_orders["gmv"] >= 0).all()

    def test_discount_not_exceed_gmv(self, sample_orders):
        assert (sample_orders["discount_amount"] <= sample_orders["gmv"]).all()

    def test_payment_expected_payout_positive(self, sample_payments):
        assert (sample_payments["expected_payout"] > 0).all()

    def test_discrepancy_amt_equals_difference(self, sample_payments):
        calc = (sample_payments["expected_payout"] - sample_payments["actual_payout"]).round(2)
        assert (calc == sample_payments["discrepancy_amt"]).all()
