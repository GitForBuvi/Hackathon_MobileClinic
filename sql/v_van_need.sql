-- =====================================================================================
-- v_van_need: how much each Santa Clara neighborhood needs the mobile clinic van
-- =====================================================================================
--
-- WHAT THIS VIEW GIVES YOU
--   One row per census tract (a neighborhood of roughly 4,000 people), with:
--     need_score  0-100, higher = more need for the van
--     need_rank   1 = the neighborhood that needs the van most
--   plus the raw inputs and each factor's rank, so an agent can explain WHY a
--   neighborhood scored high ("high because it is far from care and few people own cars").
--
-- -------------------------------------------------------------------------------------
-- STEP 1: WHY WE USE PERCENTILE RANKS
-- -------------------------------------------------------------------------------------
--   The factors are in different units: uninsured is a percentage (e.g. 14.1),
--   distance is kilometers (e.g. 7.47). Adding them directly would mean nothing.
--
--   So each factor is converted to a percentile rank from 0 to 1:
--     0   = the best neighborhood in Santa Clara on that factor
--     0.5 = right in the middle
--     1   = the worst
--   (BigQuery's PERCENT_RANK() = (position in the sorted list - 1) / (number of rows - 1))
--
--   Real example from our data, neighborhood 06085512603:
--     25.4% uninsured        -> uninsured rank 1.0   (the most uninsured in the county)
--     0.92 km to safety net  -> distance rank  0.22  (closer to affordable care than most)
--   Once every factor is on the same 0-1 scale, they can be combined fairly.
--
-- -------------------------------------------------------------------------------------
-- STEP 2: WHAT THE WEIGHTS DO
-- -------------------------------------------------------------------------------------
--   Like a school grade (final exam 70% + homework 30%), the score is a weighted sum:
--
--     need_score = 100 x ( 0.20 x uninsured rank
--                        + 0.10 x no-checkup rank
--                        + 0.10 x chronic-disease rank
--                        + 0.30 x distance-to-safety-net rank
--                        + 0.25 x no-car rank
--                        + 0.05 x mobility-difficulty rank )
--
--   The weights add up to 100%, so raising one factor means lowering another.
--
--   Illustration with just two factors (made-up ranks, to show the idea):
--     Neighborhood A: uninsured rank 1.0, distance rank 0.22  (many uninsured, clinic nearby)
--     Neighborhood B: uninsured rank 0.6, distance rank 0.90  (fewer uninsured, care far away)
--
--     Weights 70% uninsured / 30% distance:
--       A = 0.70x1.0 + 0.30x0.22 = 0.77     B = 0.70x0.6 + 0.30x0.90 = 0.69   -> A wins
--     Weights 30% uninsured / 70% distance:
--       A = 0.30x1.0 + 0.70x0.22 = 0.45     B = 0.30x0.6 + 0.70x0.90 = 0.81   -> B wins
--
--   Same data, same neighborhoods: the weights alone decided where the van goes.
--
-- -------------------------------------------------------------------------------------
-- STEP 3: OUR CHOICE: "ACCESS FIRST"
-- -------------------------------------------------------------------------------------
--   Factor                          Column                    Weight
--   Uninsured (adults 18-64)        pct_uninsured               20%
--   No routine checkup              pct_checkup (reversed)      10%
--   Chronic disease                 diabetes, high BP,          10%
--                                   COPD, asthma (averaged)
--   Far from affordable care        km_to_safety_net            30%
--   No car                          pct_no_vehicle              25%
--   Mobility difficulty             pct_mobility_difficulty      5%
--
--   WHY: near a safety-net clinic, people face at most ONE barrier (finding time
--   during clinic hours). Far from care without a car, they face TWO (time AND
--   distance), on every day of the week. The van removes the distance barrier,
--   so we send it where both barriers exist.
--
--   WHAT THIS CHANGED, compared with a balanced version
--   (uninsured 30, checkup 15, chronic 15, distance 20, no car 15, mobility 5):
--
--     Neighborhood   Profile                                       Balanced  Access first
--     06085511915    11.6% uninsured, 6.48 km, 21.4% no car            1st        1st
--     06085503218    21.8% uninsured, 1.36 km, 15.3% no car            2nd        2nd
--     06085506503     8.0% uninsured, 3.67 km, 20.9% no car           23rd        3rd   (rises)
--     06085512603    25.4% uninsured, 0.92 km,  4.6% no car           14th       62nd   (falls)
--
--   The top two win under either weighting: they are safe picks. The weights mostly
--   decide the remaining Saturdays.
--
--   KNOWN LIMIT: we have no data on clinic hours or capacity. If the clinics near
--   the highly uninsured neighborhoods are overloaded or closed when people can go,
--   those neighborhoods deserve more weight than this score gives them.
--
-- -------------------------------------------------------------------------------------
-- LEFT OUT ON PURPOSE
-- -------------------------------------------------------------------------------------
--   Air quality (exposure_tracts): modeled on a 12 km grid. We tested it: rounded to
--     0.1, Santa Clara's 408 tracts collapse into ~27 air levels (about 15 tracts per
--     level), while uninsured spreads across 130 values. Useful county context,
--     too blunt to pick a neighborhood.
--   Disease trends (disease_weekly): one number for a multi-county region, and every
--     COVID value is missing for Santa Clara. Cannot rank neighborhoods.
--   Shortage areas (shortage_areas): Santa Clara has no designations. That is a
--     real fact, not a broken load.
--   Race / ethnicity: banned as an input. Used ONLY afterwards, to audit who the
--     plan lands on.
--
-- SCALE OF EVERY NUMBER
--   Health and insurance values are MODELED estimates (CDC PLACES), not counts.
--   No-vehicle share comes from a survey (Census / SVI).
--   Distances are straight lines to federally recognized clinics inside the county.
-- =====================================================================================

CREATE OR REPLACE VIEW `hackathon-bigquery-515300.a4i_health.v_van_need` AS

-- base: put each neighborhood's health data and its distances on one row.
--   LEFT JOIN keeps every neighborhood from burden_tracts, even if one had no
--   distance row (the README recommends joining LEFT from burden_tracts).
--   Neighborhoods under 500 people are skipped: too small to fill a van-day,
--   and their modeled estimates are the least reliable.
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

-- ranked: turn every factor into a 0-1 percentile rank (1 = worst in the county).
--   pct_checkup is sorted DESC because a LOW checkup rate means MORE need:
--   e.g. a neighborhood where only 70% had a checkup ranks worse than one at 79%.
--   The four chronic diseases are ranked separately, then averaged into one rank,
--   so no single disease dominates.
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

-- scored: apply the access-first weights (they add up to 1.0 = 100%).
--   Multiplying by 100 turns the 0-1 result into an easier-to-read 0-100 score.
scored AS (
  SELECT *,
    ROUND(100 * (0.20*r_uninsured + 0.10*r_no_checkup + 0.10*r_chronic
               + 0.30*r_distance  + 0.25*r_no_vehicle + 0.05*r_mobility), 1) AS need_score
  FROM ranked
)

-- final: number the neighborhoods by score, 1 = most need.
--   RANK() gives tied scores the same number (e.g. two neighborhoods both 11th).
SELECT *,
  RANK() OVER (ORDER BY need_score DESC) AS need_rank
FROM scored;
