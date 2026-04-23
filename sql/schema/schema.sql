-- =============================================================
-- schema.sql
-- E-Commerce Revenue Intelligence Platform
-- BigQuery-compatible DDL
-- Author: Chandrakant Allimatti
-- =============================================================

-- ── Orders ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `ecommerce.orders` (
    order_id          STRING      NOT NULL,
    customer_id       STRING      NOT NULL,
    seller_id         STRING      NOT NULL,
    order_date        TIMESTAMP   NOT NULL,
    category          STRING,
    gmv               NUMERIC,
    discount_amount   NUMERIC,
    shipping_fee      NUMERIC,
    net_revenue       NUMERIC,
    status            STRING,       -- delivered | returned | cancelled | pending
    payment_method    STRING,
    channel           STRING,
    -- Derived (populated by ETL)
    order_year        INT64,
    order_month       INT64,
    order_quarter     INT64,
    order_week        INT64,
    order_dow         STRING,
    is_weekend        BOOL,
    is_delivered      INT64,
    is_returned       INT64,
    is_cancelled      INT64,
    discount_rate     NUMERIC,
    gmv_band          STRING,
)
PARTITION BY DATE(order_date)
CLUSTER BY seller_id, category;

-- ── Sellers ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `ecommerce.sellers` (
    seller_id       STRING    NOT NULL,
    seller_name     STRING,
    category        STRING,
    state           STRING,
    tier            STRING,       -- Platinum | Gold | Silver | Bronze
    joined_date     DATE,
    commission_pct  NUMERIC,
    is_active       INT64,
);

-- ── Customers ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `ecommerce.customers` (
    customer_id          STRING    NOT NULL,
    state                STRING,
    age_group            STRING,
    gender               STRING,
    registration_date    TIMESTAMP,
    acquisition_channel  STRING,
    is_prime             INT64,
    days_since_reg       INT64,
    tenure_band          STRING,
);

-- ── Payments / Settlements ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS `ecommerce.payments` (
    payment_id            STRING    NOT NULL,
    order_id              STRING    NOT NULL,
    seller_id             STRING    NOT NULL,
    expected_payout       NUMERIC,
    actual_payout         NUMERIC,
    discrepancy_flag      INT64,
    discrepancy_amt       NUMERIC,
    discrepancy_pct       NUMERIC,
    settlement_date       DATE,
    settlement_cycle      STRING,
    status                STRING,    -- settled | disputed
    discrepancy_severity  STRING,    -- none | minor | moderate | critical
)
PARTITION BY settlement_date
CLUSTER BY seller_id;

-- ── Marketing Spend ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS `ecommerce.marketing_spend` (
    date          DATE      NOT NULL,
    channel       STRING    NOT NULL,
    spend_inr     NUMERIC,
    impressions   INT64,
    clicks        INT64,
)
PARTITION BY date;

-- ── Sessions (A/B Test) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS `ecommerce.sessions` (
    session_id             STRING    NOT NULL,
    customer_id            STRING,
    session_date           TIMESTAMP,
    channel                STRING,
    device                 STRING,
    ab_variant             STRING,    -- control | variant_b
    pages_viewed           INT64,
    session_duration_sec   INT64,
    converted              INT64,
    bounce                 INT64,
)
PARTITION BY DATE(session_date);
