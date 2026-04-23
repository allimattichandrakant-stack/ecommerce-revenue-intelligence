-- =============================================================
-- marketing_attribution.sql
-- Purpose : Multi-touch attribution, channel ROI computation,
--           and budget reallocation signals.
-- Platform: BigQuery (ANSI SQL compatible)
-- Author  : Chandrakant Allimatti
-- =============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. LAST-TOUCH CHANNEL REVENUE ATTRIBUTION
-- ─────────────────────────────────────────────────────────────
WITH channel_revenue AS (
    SELECT
        channel,
        COUNT(order_id)                        AS total_orders,
        SUM(gmv)                               AS total_gmv,
        SUM(net_revenue)                       AS net_revenue,
        AVG(gmv)                               AS avg_order_value,
        SUM(discount_amount)                   AS total_discount,
        ROUND(SUM(discount_amount)/SUM(gmv)*100, 2)
                                               AS avg_discount_rate_pct,
        -- Revenue share across all channels
        ROUND(SUM(net_revenue) * 100.0
              / SUM(SUM(net_revenue)) OVER (), 2)
                                               AS revenue_share_pct
    FROM orders_clean
    WHERE is_delivered = 1
    GROUP BY channel
),

-- ─────────────────────────────────────────────────────────────
-- 2. MONTHLY CHANNEL PERFORMANCE TREND
--    Used to detect seasonality and identify channels that
--    have improving vs declining efficiency.
-- ─────────────────────────────────────────────────────────────
monthly_channel AS (
    SELECT
        channel,
        order_year,
        order_month,
        COUNT(order_id)             AS monthly_orders,
        ROUND(SUM(net_revenue), 2)  AS monthly_net_revenue,

        -- Month-over-month revenue growth per channel
        ROUND(
            (SUM(net_revenue) - LAG(SUM(net_revenue))
                OVER (PARTITION BY channel ORDER BY order_year, order_month))
            / NULLIF(LAG(SUM(net_revenue))
                OVER (PARTITION BY channel ORDER BY order_year, order_month), 0)
            * 100,
            2
        ) AS mom_revenue_growth_pct,

        -- 3-month rolling average revenue
        ROUND(
            AVG(SUM(net_revenue)) OVER (
                PARTITION BY channel
                ORDER BY order_year, order_month
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ),
            2
        ) AS rolling_3m_avg_revenue

    FROM orders_clean
    WHERE is_delivered = 1
    GROUP BY channel, order_year, order_month
),

-- ─────────────────────────────────────────────────────────────
-- 3. CUSTOMER ACQUISITION CHANNEL QUALITY
--    Compares LTV across acquisition channels to surface
--    which channels bring higher-value customers.
-- ─────────────────────────────────────────────────────────────
customer_channel_ltv AS (
    SELECT
        c.acquisition_channel,
        COUNT(DISTINCT c.customer_id)          AS acquired_customers,
        COUNT(o.order_id)                      AS lifetime_orders,
        ROUND(SUM(o.net_revenue), 2)           AS lifetime_revenue,
        ROUND(AVG(o.net_revenue), 2)           AS avg_revenue_per_order,
        ROUND(
            SUM(o.net_revenue) / NULLIF(COUNT(DISTINCT c.customer_id), 0),
            2
        )                                      AS ltv_per_customer,
        ROUND(
            COUNT(o.order_id) * 1.0
            / NULLIF(COUNT(DISTINCT c.customer_id), 0),
            2
        )                                      AS avg_orders_per_customer,

        -- LTV rank across channels
        DENSE_RANK() OVER (
            ORDER BY SUM(o.net_revenue) / NULLIF(COUNT(DISTINCT c.customer_id), 0) DESC
        ) AS ltv_rank

    FROM customers c
    LEFT JOIN orders_clean o ON c.customer_id = o.customer_id AND o.is_delivered = 1
    GROUP BY c.acquisition_channel
),

-- ─────────────────────────────────────────────────────────────
-- 4. NEW vs RETURNING CUSTOMER MIX BY CHANNEL
--    Channels dominated by returning customers suggest
--    low acquisition power (e.g., direct/email).
-- ─────────────────────────────────────────────────────────────
customer_type_mix AS (
    SELECT
        o.channel,
        o.customer_id,
        ROW_NUMBER() OVER (
            PARTITION BY o.customer_id
            ORDER BY o.order_date
        ) AS order_seq
    FROM orders_clean o
    WHERE o.is_delivered = 1
),
channel_new_returning AS (
    SELECT
        channel,
        COUNT(DISTINCT CASE WHEN order_seq = 1 THEN customer_id END) AS new_customers,
        COUNT(DISTINCT CASE WHEN order_seq > 1 THEN customer_id END) AS returning_customers,
        COUNT(DISTINCT customer_id)                                   AS total_unique_customers,
        ROUND(
            COUNT(DISTINCT CASE WHEN order_seq = 1 THEN customer_id END) * 100.0
            / NULLIF(COUNT(DISTINCT customer_id), 0), 2
        ) AS new_customer_pct
    FROM customer_type_mix
    GROUP BY channel
)

-- ─────────────────────────────────────────────────────────────
-- 5. COMBINED CHANNEL SCORECARD
-- ─────────────────────────────────────────────────────────────
SELECT
    cr.channel,
    cr.total_orders,
    ROUND(cr.total_gmv, 0)             AS total_gmv,
    ROUND(cr.net_revenue, 0)           AS net_revenue,
    ROUND(cr.avg_order_value, 2)       AS avg_order_value,
    cr.revenue_share_pct,
    cr.avg_discount_rate_pct,
    cltv.ltv_per_customer,
    cltv.avg_orders_per_customer,
    cltv.ltv_rank,
    cnr.new_customer_pct,
    cnr.returning_customers,

    -- Strategic signal
    CASE
        WHEN cltv.ltv_rank <= 2 AND cnr.new_customer_pct > 40
            THEN 'SCALE UP — High LTV + Strong Acquisition'
        WHEN cltv.ltv_rank <= 4 AND cnr.new_customer_pct < 25
            THEN 'RETENTION CHANNEL — Good LTV, Mostly Returning'
        WHEN cr.avg_discount_rate_pct > 25
            THEN 'REVIEW — High Discount Dependency'
        ELSE 'MONITOR'
    END AS strategic_signal

FROM channel_revenue cr
LEFT JOIN customer_channel_ltv cltv
       ON cr.channel = cltv.acquisition_channel
LEFT JOIN channel_new_returning cnr
       ON cr.channel = cnr.channel
ORDER BY cltv.ltv_rank NULLS LAST, cr.net_revenue DESC;
