-- =============================================================
-- financial_reconciliation.sql
-- Purpose : Forensic investigation of buyer-seller payment
--           discrepancies across settlement cycles.
-- Platform: BigQuery (also runs on SQLite with minor adaptation)
-- Author  : Chandrakant Allimatti
-- =============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. FULL PAYMENT AUDIT TRAIL
--    Joins orders → payments → sellers to build a complete
--    lifecycle view of every transaction payout.
-- ─────────────────────────────────────────────────────────────
WITH order_payment_join AS (
    SELECT
        o.order_id,
        o.customer_id,
        o.seller_id,
        s.seller_name,
        s.tier            AS seller_tier,
        s.commission_pct,
        o.order_date,
        o.gmv,
        o.net_revenue,
        o.category,
        o.channel,
        p.payment_id,
        p.expected_payout,
        p.actual_payout,
        p.discrepancy_amt,
        p.discrepancy_pct,
        p.discrepancy_flag,
        p.settlement_date,
        p.settlement_cycle,
        p.status          AS payment_status,
        p.discrepancy_severity,

        -- Running total discrepancy per seller using window function
        SUM(p.discrepancy_amt) OVER (
            PARTITION BY o.seller_id
            ORDER BY p.settlement_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_discrepancy_by_seller,

        -- Settlement delay in days
        JULIANDAY(p.settlement_date) - JULIANDAY(o.order_date) AS settlement_delay_days,

        -- Rank payments by discrepancy amount within each seller (for top-N analysis)
        ROW_NUMBER() OVER (
            PARTITION BY o.seller_id
            ORDER BY p.discrepancy_amt DESC
        ) AS rank_within_seller

    FROM orders_clean o
    INNER JOIN payments_clean p ON o.order_id = p.order_id
    INNER JOIN sellers s        ON o.seller_id = s.seller_id
    WHERE o.is_delivered = 1
),

-- ─────────────────────────────────────────────────────────────
-- 2. SELLER-LEVEL DISCREPANCY AGGREGATION
--    Flags sellers for mandatory audit based on total
--    discrepancy amount and frequency.
-- ─────────────────────────────────────────────────────────────
seller_discrepancy_summary AS (
    SELECT
        seller_id,
        seller_name,
        seller_tier,
        COUNT(*)                                    AS total_payments,
        SUM(discrepancy_flag)                       AS disputed_count,
        ROUND(SUM(discrepancy_flag) * 100.0 / COUNT(*), 2)
                                                    AS dispute_rate_pct,
        ROUND(SUM(CASE WHEN discrepancy_flag = 1 THEN discrepancy_amt ELSE 0 END), 2)
                                                    AS total_discrepancy_amt,
        ROUND(AVG(CASE WHEN discrepancy_flag = 1 THEN discrepancy_pct ELSE NULL END), 2)
                                                    AS avg_discrepancy_pct,
        MAX(discrepancy_amt)                        AS max_single_discrepancy,
        MIN(settlement_date)                        AS first_dispute_date,
        MAX(settlement_date)                        AS latest_dispute_date,

        -- Months with at least one dispute
        COUNT(DISTINCT CASE WHEN discrepancy_flag = 1
              THEN STRFTIME('%Y-%m', settlement_date) END)  AS months_with_disputes,

        -- Risk flag: top-quartile discrepancy OR > 5% dispute rate
        CASE
            WHEN SUM(CASE WHEN discrepancy_flag = 1 THEN discrepancy_amt ELSE 0 END)
                 > (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_discrepancy_amt)
                    -- Note: replace with BigQuery APPROX_QUANTILES in production
                    FROM (
                        SELECT seller_id,
                               SUM(CASE WHEN discrepancy_flag=1 THEN discrepancy_amt ELSE 0 END) AS total_discrepancy_amt
                        FROM payments_clean GROUP BY seller_id
                    ))
                THEN 'MANDATORY_AUDIT'
            WHEN ROUND(SUM(discrepancy_flag) * 100.0 / COUNT(*), 2) > 5
                THEN 'REVIEW_REQUIRED'
            ELSE 'CLEAN'
        END AS audit_status

    FROM order_payment_join
    GROUP BY seller_id, seller_name, seller_tier
),

-- ─────────────────────────────────────────────────────────────
-- 3. SETTLEMENT CYCLE ANALYSIS
--    Detects which settlement weeks had the most discrepancies.
--    Useful to identify systemic pipeline failures.
-- ─────────────────────────────────────────────────────────────
cycle_level_anomalies AS (
    SELECT
        settlement_cycle,
        COUNT(*)                                     AS total_payments,
        SUM(discrepancy_flag)                        AS disputed_in_cycle,
        ROUND(SUM(discrepancy_flag)*100.0/COUNT(*),2) AS dispute_rate_pct,
        ROUND(SUM(CASE WHEN discrepancy_flag=1 THEN discrepancy_amt ELSE 0 END), 2)
                                                     AS total_disc_amt,
        -- Week-over-week change in dispute rate
        ROUND(
            ROUND(SUM(discrepancy_flag)*100.0/COUNT(*), 2)
            - LAG(ROUND(SUM(discrepancy_flag)*100.0/COUNT(*), 2))
              OVER (ORDER BY settlement_cycle),
            2
        ) AS dispute_rate_wow_change

    FROM order_payment_join
    GROUP BY settlement_cycle
),

-- ─────────────────────────────────────────────────────────────
-- 4. PAYMENT TIMELINE: FIRST vs LAST DISPUTE PER SELLER
--    Identifies chronic offenders vs one-time errors.
-- ─────────────────────────────────────────────────────────────
dispute_timeline AS (
    SELECT
        seller_id,
        MIN(CASE WHEN discrepancy_flag = 1 THEN settlement_date END) AS first_dispute,
        MAX(CASE WHEN discrepancy_flag = 1 THEN settlement_date END) AS last_dispute,
        JULIANDAY(MAX(CASE WHEN discrepancy_flag=1 THEN settlement_date END))
        - JULIANDAY(MIN(CASE WHEN discrepancy_flag=1 THEN settlement_date END))
                                                     AS dispute_span_days,
        COUNT(DISTINCT CASE WHEN discrepancy_flag = 1
              THEN payment_id END)                   AS dispute_events
    FROM order_payment_join
    GROUP BY seller_id
    HAVING COUNT(DISTINCT CASE WHEN discrepancy_flag=1 THEN payment_id END) > 0
)

-- ─────────────────────────────────────────────────────────────
-- 5. FINAL OUTPUT: AUDIT-READY REPORT
--    Combines all CTEs for a single, actionable view.
-- ─────────────────────────────────────────────────────────────
SELECT
    sd.seller_id,
    sd.seller_name,
    sd.seller_tier,
    sd.audit_status,
    sd.total_payments,
    sd.disputed_count,
    sd.dispute_rate_pct,
    sd.total_discrepancy_amt,
    sd.avg_discrepancy_pct,
    sd.max_single_discrepancy,
    sd.months_with_disputes,
    dt.first_dispute,
    dt.last_dispute,
    dt.dispute_span_days,

    -- Categorise longevity of dispute pattern
    CASE
        WHEN dt.dispute_span_days > 180 THEN 'Chronic (>6 months)'
        WHEN dt.dispute_span_days > 60  THEN 'Recurring (2-6 months)'
        WHEN dt.dispute_span_days > 0   THEN 'Isolated (<2 months)'
        ELSE 'Single Event'
    END AS dispute_pattern,

    -- Priority score (0-100) for triage queue
    ROUND(
        LEAST(100,
            (sd.total_discrepancy_amt / 10000) * 0.5 +
            sd.dispute_rate_pct             * 0.3 +
            COALESCE(dt.dispute_span_days, 0) / 365.0 * 20
        ),
        1
    ) AS triage_priority_score

FROM seller_discrepancy_summary sd
LEFT JOIN dispute_timeline dt USING (seller_id)
WHERE sd.disputed_count > 0
ORDER BY triage_priority_score DESC;
