-- =============================================================
-- fraud_signals.sql
-- Purpose : SQL-based fraud pattern detection.
--           Identifies suspicious seller behaviour, velocity
--           anomalies, and refund manipulation patterns.
-- Platform: BigQuery / SQLite
-- Author  : Chandrakant Allimatti
-- =============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. HIGH-VELOCITY SELLERS
--    Flag sellers whose weekly order count exceeds the
--    platform mean by more than 3 standard deviations.
-- ─────────────────────────────────────────────────────────────
WITH weekly_seller_orders AS (
    SELECT
        seller_id,
        STRFTIME('%Y-W%W', order_date)      AS week,
        COUNT(order_id)                     AS weekly_orders,
        SUM(gmv)                            AS weekly_gmv,
        SUM(is_returned)                    AS weekly_returns,
        ROUND(SUM(is_returned)*100.0/COUNT(order_id), 2) AS weekly_return_rate
    FROM orders_clean
    GROUP BY seller_id, STRFTIME('%Y-W%W', order_date)
),
velocity_stats AS (
    SELECT
        AVG(weekly_orders)      AS mean_weekly,
        -- Approximate std dev (SQLite lacks STDDEV natively)
        SQRT(AVG(weekly_orders * weekly_orders) - AVG(weekly_orders)*AVG(weekly_orders))
                                AS stddev_weekly
    FROM weekly_seller_orders
),
high_velocity AS (
    SELECT
        w.seller_id,
        w.week,
        w.weekly_orders,
        w.weekly_gmv,
        w.weekly_return_rate,
        ROUND((w.weekly_orders - v.mean_weekly) / NULLIF(v.stddev_weekly, 0), 3)
                                AS z_score,
        'HIGH_VELOCITY'         AS fraud_signal_type
    FROM weekly_seller_orders w, velocity_stats v
    WHERE (w.weekly_orders - v.mean_weekly) / NULLIF(v.stddev_weekly, 0) > 3.0
),

-- ─────────────────────────────────────────────────────────────
-- 2. RETURN ABUSE DETECTION
--    Sellers with abnormally high return rates AND high GMV.
--    Pattern: list expensive items → high return volume → payout abuse.
-- ─────────────────────────────────────────────────────────────
seller_return_profile AS (
    SELECT
        seller_id,
        COUNT(order_id)                             AS total_orders,
        SUM(is_delivered)                           AS delivered,
        SUM(is_returned)                            AS returned,
        ROUND(SUM(is_returned)*100.0/COUNT(order_id), 2)
                                                    AS return_rate_pct,
        ROUND(SUM(gmv), 2)                          AS total_gmv,
        ROUND(AVG(gmv), 2)                          AS avg_order_gmv
    FROM orders_clean
    WHERE order_date >= DATE('now', '-180 days')
    GROUP BY seller_id
    HAVING total_orders >= 20
),
return_anomaly AS (
    SELECT
        seller_id,
        total_orders,
        return_rate_pct,
        total_gmv,
        avg_order_gmv,
        -- Flag if return rate is > 2x platform average
        CASE
            WHEN return_rate_pct > 2 * (SELECT AVG(return_rate_pct) FROM seller_return_profile)
             AND total_gmv > (SELECT AVG(total_gmv) FROM seller_return_profile)
            THEN 'RETURN_ABUSE_HIGH_RISK'
            WHEN return_rate_pct > 1.5 * (SELECT AVG(return_rate_pct) FROM seller_return_profile)
            THEN 'RETURN_RATE_ELEVATED'
            ELSE 'NORMAL'
        END AS return_signal
    FROM seller_return_profile
    WHERE return_rate_pct > (SELECT AVG(return_rate_pct) FROM seller_return_profile) * 1.3
),

-- ─────────────────────────────────────────────────────────────
-- 3. RAPID REPEAT ORDERS (Same customer, same seller)
--    Legitimate sellers rarely see the same customer buying
--    multiple times in a 24-hour window at high value.
-- ─────────────────────────────────────────────────────────────
repeat_order_pairs AS (
    SELECT
        o1.seller_id,
        o1.customer_id,
        o1.order_id          AS first_order_id,
        o2.order_id          AS repeat_order_id,
        o1.order_date        AS first_order_time,
        o2.order_date        AS repeat_order_time,
        ROUND((JULIANDAY(o2.order_date) - JULIANDAY(o1.order_date)) * 24, 2)
                             AS hours_between_orders,
        o1.gmv               AS first_gmv,
        o2.gmv               AS repeat_gmv
    FROM orders_clean o1
    JOIN orders_clean o2
      ON o1.seller_id   = o2.seller_id
     AND o1.customer_id = o2.customer_id
     AND o2.order_date  > o1.order_date
     AND (JULIANDAY(o2.order_date) - JULIANDAY(o1.order_date)) * 24 < 12
     AND o1.order_id   != o2.order_id
     AND o1.gmv > 5000      -- filter low-value noise
),

-- ─────────────────────────────────────────────────────────────
-- 4. PAYMENT DISCREPANCY ESCALATION PATTERN
--    Sellers whose discrepancies are growing month-over-month
--    (not random errors — systematic extraction).
-- ─────────────────────────────────────────────────────────────
monthly_discrepancy_trend AS (
    SELECT
        p.seller_id,
        STRFTIME('%Y-%m', p.settlement_date)    AS settlement_month,
        COUNT(*)                                AS payments,
        SUM(p.discrepancy_flag)                 AS disputes,
        ROUND(SUM(p.discrepancy_amt), 2)        AS disc_amt,
        ROUND(
            SUM(p.discrepancy_amt)
            - LAG(SUM(p.discrepancy_amt)) OVER (
                PARTITION BY p.seller_id
                ORDER BY STRFTIME('%Y-%m', p.settlement_date)
              ),
            2
        ) AS disc_amt_mom_change
    FROM payments_clean p
    WHERE p.discrepancy_flag = 1
    GROUP BY p.seller_id, STRFTIME('%Y-%m', p.settlement_date)
),
escalating_sellers AS (
    SELECT
        seller_id,
        COUNT(DISTINCT settlement_month)         AS months_with_disputes,
        SUM(disputes)                            AS total_disputes,
        SUM(disc_amt)                            AS total_disc_amt,
        SUM(CASE WHEN disc_amt_mom_change > 0 THEN 1 ELSE 0 END)
                                                 AS months_with_increase,
        -- Sellers where discrepancy amount is increasing in majority of months
        CASE
            WHEN SUM(CASE WHEN disc_amt_mom_change > 0 THEN 1 ELSE 0 END) * 1.0
                 / NULLIF(COUNT(DISTINCT settlement_month) - 1, 0) > 0.6
            THEN 'ESCALATING — INVESTIGATE'
            ELSE 'STABLE'
        END AS escalation_status
    FROM monthly_discrepancy_trend
    GROUP BY seller_id
    HAVING months_with_disputes >= 3
)

-- ─────────────────────────────────────────────────────────────
-- 5. CONSOLIDATED FRAUD RISK REGISTER
-- ─────────────────────────────────────────────────────────────
SELECT
    ra.seller_id,
    ra.return_rate_pct,
    ra.total_gmv,
    ra.return_signal,
    es.escalation_status,
    es.total_disc_amt,
    es.months_with_disputes,
    COUNT(DISTINCT rp.first_order_id)   AS rapid_repeat_order_events,

    -- Composite risk tier
    CASE
        WHEN ra.return_signal = 'RETURN_ABUSE_HIGH_RISK'
         AND es.escalation_status = 'ESCALATING — INVESTIGATE'
            THEN '🔴 CRITICAL — SUSPEND ACCOUNT'
        WHEN ra.return_signal IN ('RETURN_ABUSE_HIGH_RISK','RETURN_RATE_ELEVATED')
          OR es.escalation_status = 'ESCALATING — INVESTIGATE'
            THEN '🟡 HIGH — MANUAL REVIEW'
        ELSE '🟢 MONITOR'
    END AS risk_tier

FROM return_anomaly ra
LEFT JOIN escalating_sellers es  ON ra.seller_id = es.seller_id
LEFT JOIN repeat_order_pairs rp  ON ra.seller_id = rp.seller_id
GROUP BY ra.seller_id, ra.return_rate_pct, ra.total_gmv,
         ra.return_signal, es.escalation_status,
         es.total_disc_amt, es.months_with_disputes
ORDER BY
    CASE risk_tier
        WHEN '🔴 CRITICAL — SUSPEND ACCOUNT' THEN 1
        WHEN '🟡 HIGH — MANUAL REVIEW'        THEN 2
        ELSE 3
    END,
    ra.total_gmv DESC;
