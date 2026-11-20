-- v_van_need: how much each Santa Clara neighborhood needs the mobile clinic van.
-- One row per census tract (population 500+). need_score 0-100, need_rank 1 = most need.
--
-- Weighting: ACCESS FIRST. The van goes where affordable care is out of reach,
-- especially for people without a car. Near a safety-net clinic, people face at most
-- one barrier (time); far from care without a car, they face two (time and distance).
--   uninsured 20%, no checkup 10%, chronic disease 10%,
--   distance to safety-net clinic 30%, no vehicle 25%, mobility difficulty 5%
--
-- Left out on purpose:
--   air quality (12 km grid: 408 tracts collapse to ~27 levels in Santa Clara)
--   disease_weekly (regional, and empty for Santa Clara)
--   shortage_areas (none designated in Santa Clara)
--   race / ethnicity (banned as an input; used only to audit the plan)
--
-- Known limit: we have no data on clinic hours or capacity.

CREATE OR REPLACE VIEW `hackathon-bigquery-515300.a4i_health.v_van_need` AS
WITH base AS (
  SELECT b.geo_id, b.total_pop, b.latitude, b.longitude,
         b.pct_uninsured, b.pct_checkup,
         b.pct_diabetes, b.pct_high_bp, b.pct_copd, b.pct_asthma,
         b.pct_no_vehicle, b.pct_mobility_difficulty,
         a.km_to_any_care, a.km_to_safety_net, a.extra_km_if_uninsured
  FROM `hackathon-bigquery-515300.a4i_health.burden_tracts` b
  LEFT JOIN `hackathon-bigquery-515300.a4i_health.tract_access` a USING (geo_id)
  WHERE b.total_pop >= 500
),
ranked AS (
  SELECT *,
    PERCENT_RANK() OVER (ORDER BY pct_uninsured)           AS r_uninsured,
    PERCENT_RANK() OVER (ORDER BY pct_checkup DESC)        AS r_no_checkup,
    ( PERCENT_RANK() OVER (ORDER BY pct_diabetes)
    + PERCENT_RANK() OVER (ORDER BY pct_high_bp)
    + PERCENT_RANK() OVER (ORDER BY pct_copd)
    + PERCENT_RANK() OVER (ORDER BY pct_asthma) ) / 4      AS r_chronic,
    PERCENT_RANK() OVER (ORDER BY km_to_safety_net)        AS r_distance,
    PERCENT_RANK() OVER (ORDER BY pct_no_vehicle)          AS r_no_vehicle,
    PERCENT_RANK() OVER (ORDER BY pct_mobility_difficulty) AS r_mobility
  FROM base
),
scored AS (
  SELECT *,
    ROUND(100 * (0.20*r_uninsured + 0.10*r_no_checkup + 0.10*r_chronic
               + 0.30*r_distance  + 0.25*r_no_vehicle + 0.05*r_mobility), 1) AS need_score
  FROM ranked
)
SELECT *,
  RANK() OVER (ORDER BY need_score DESC) AS need_rank
FROM scored;
