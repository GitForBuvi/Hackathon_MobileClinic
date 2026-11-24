"""Tools for the mobile clinic planner.

Every function in this file is a tool: an ordinary Python function that an agent
can decide to call. ADK reads each function's name, parameters and docstring, and
tells Gemini what the tool does and when to use it.
"""

import math
import re

from google.cloud import bigquery

PROJECT_ID = "hackathon-bigquery-515300"
VIEW = f"{PROJECT_ID}.a4i_health.v_van_need"

# ---------------------------------------------------------------------------------
# Planning rules
# These are policy decisions, so they live here in code, not in the chat. A message
# like "skip the spacing rule" cannot change them; only editing this file can.
# ---------------------------------------------------------------------------------
NUM_SATURDAYS = 4
MIN_KM_APART = 2.0             # people without cars rarely walk 2 km, so stops 2+ km
                               # apart reach mostly different people
MIN_KM_FROM_SAFETY_NET = 1.0   # within ~1 km a clinic that sees uninsured patients
                               # is already walkable, so the van adds little there
CANDIDATES = 40                # how many top-ranked neighborhoods the planner considers

NOTES = (
    "geo_id is a census tract ID, not a neighborhood name. "
    "rank_* values run from 0 (best in Santa Clara) to 1 (worst in Santa Clara). "
    "A rank of 1.0 means among the very worst in the county, not necessarily the single worst. "
    "pct_uninsured is a MODELED estimate (CDC PLACES) for adults 18-64, not a count. "
    "pct_no_vehicle comes from a Census survey. "
    "km_to_safety_net is a straight line to the nearest federally recognized clinic "
    "that sees uninsured patients, inside the county only, so it can overstate "
    "distance near the county edge. Distances between neighborhoods are straight "
    "lines between their center points."
)


def _km_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in km between two points on Earth (haversine formula)."""
    earth_radius_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = p2 - p1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def _query_top(limit: int) -> list[dict]:
    """The top `limit` neighborhoods from v_van_need, highest need first."""
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT geo_id, need_rank, need_score, total_pop, latitude, longitude,
               pct_uninsured, pct_no_vehicle,
               ROUND(km_to_safety_net, 2) AS km_to_safety_net,
               ROUND(r_distance, 3)       AS rank_distance,
               ROUND(r_no_vehicle, 3)     AS rank_no_vehicle,
               ROUND(r_uninsured, 3)      AS rank_uninsured
        FROM `{VIEW}`
        ORDER BY need_rank
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    )
    rows = client.query(query, job_config=job_config).result()
    return [dict(row.items()) for row in rows]


def get_top_need_neighborhoods(limit: int = 10) -> dict:
    """Returns the Santa Clara County neighborhoods (census tracts) that most need the mobile clinic van.

    Use this to answer questions about the ranking itself, for example "which
    neighborhoods need the van most?". To build the actual Saturday schedule, use
    plan_saturdays instead.

    Args:
        limit: How many neighborhoods to return, highest need first. Use 5 to 20.

    Returns:
        A dict with 'neighborhoods' (highest need first) and 'notes' explaining what
        the numbers mean and how reliable they are.
    """
    limit = max(1, min(int(limit), 50))
    rows = _query_top(limit)
    for row in rows:
        row.pop("latitude", None)
        row.pop("longitude", None)
    return {"neighborhoods": rows, "notes": NOTES}


def plan_saturdays() -> dict:
    """Builds next month's plan: which 4 Santa Clara neighborhoods the mobile clinic van visits, one per Saturday.

    Works down the need ranking (highest need first) and applies two fixed rules:
      1. Skip a neighborhood within 1 km of a safety-net clinic (care is already walkable there).
      2. Skip a neighborhood within 2 km of one already picked (it would serve the same people).

    Returns:
        A dict with 'saturdays' (the picks, in order), 'skipped' (higher-ranked
        neighborhoods that were passed over, with the reason), 'rules' and 'notes'.
    """
    candidates = _query_top(CANDIDATES)

    saturdays: list[dict] = []
    skipped: list[dict] = []

    for row in candidates:
        if len(saturdays) == NUM_SATURDAYS:
            break

        # Rule 1: a safety-net clinic is already within walking distance.
        km_to_clinic = row["km_to_safety_net"]
        if km_to_clinic is not None and km_to_clinic < MIN_KM_FROM_SAFETY_NET:
            skipped.append({
                "geo_id": row["geo_id"],
                "need_rank": row["need_rank"],
                "reason": f"a safety-net clinic is only {km_to_clinic} km away",
            })
            continue

        # Rule 2: too close to a neighborhood that already has a Saturday.
        too_close = None
        for pick in saturdays:
            km = _km_between(row["latitude"], row["longitude"],
                             pick["latitude"], pick["longitude"])
            if km < MIN_KM_APART:
                too_close = (pick, km)
                break
        if too_close:
            pick, km = too_close
            skipped.append({
                "geo_id": row["geo_id"],
                "need_rank": row["need_rank"],
                "reason": (f"only {km:.1f} km from Saturday {pick['saturday']} "
                           f"({pick['geo_id']}), so it would serve the same people"),
            })
            continue

        row["saturday"] = len(saturdays) + 1
        saturdays.append(row)

    result = {
        "saturdays": saturdays,
        "skipped": skipped,
        "rules": {
            "number_of_saturdays": NUM_SATURDAYS,
            "min_km_apart": MIN_KM_APART,
            "min_km_from_safety_net": MIN_KM_FROM_SAFETY_NET,
            "candidates_considered": CANDIDATES,
        },
        "notes": NOTES,
    }
    if len(saturdays) < NUM_SATURDAYS:
        result["warning"] = (f"Only {len(saturdays)} neighborhoods met the rules among the "
                             f"top {CANDIDATES}. Say so instead of filling the gap.")
    return result


def explain_neighborhood(geo_id: str) -> dict:
    """Explains one Santa Clara neighborhood (census tract): how much it needs the van, and whether it is in this month's plan, and if not, why not.

    Use this when someone asks about a specific tract, for example "why isn't my
    neighborhood on the plan?" or "tell me about tract 06085512603".

    Args:
        geo_id: The 11-digit census tract ID, for example "06085512603".

    Returns:
        A dict with 'neighborhood' (its numbers and ranks), 'in_plan' (true or false),
        'explanation' (why it is or is not in the plan) and 'notes'.
    """
    geo_id = str(geo_id).strip()
    if not re.fullmatch(r"\d{11}", geo_id):
        return {"error": "geo_id must be an 11-digit census tract ID, for example 06085512603."}

    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT geo_id, need_rank, need_score, total_pop,
               pct_uninsured, pct_no_vehicle,
               ROUND(km_to_safety_net, 2) AS km_to_safety_net,
               ROUND(r_distance, 3)       AS rank_distance,
               ROUND(r_no_vehicle, 3)     AS rank_no_vehicle,
               ROUND(r_uninsured, 3)      AS rank_uninsured
        FROM `{VIEW}`
        WHERE geo_id = @geo_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("geo_id", "STRING", geo_id)]
    )
    rows = [dict(row.items()) for row in client.query(query, job_config=job_config).result()]
    if not rows:
        return {"error": (f"Tract {geo_id} is not among the scored Santa Clara neighborhoods. "
                          "It may be outside the county, or have fewer than 500 people.")}
    row = rows[0]

    # Run the same planner, so the explanation always matches the real plan.
    plan = plan_saturdays()
    picked = {s["geo_id"]: s["saturday"] for s in plan["saturdays"]}
    skipped = {s["geo_id"]: s["reason"] for s in plan["skipped"]}
    lowest_picked_rank = max((s["need_rank"] for s in plan["saturdays"]), default=None)

    if geo_id in picked:
        in_plan = True
        explanation = f"It is in the plan: Saturday {picked[geo_id]}."
    elif geo_id in skipped:
        in_plan = False
        explanation = f"It ranked high enough to be considered, but was skipped: {skipped[geo_id]}."
    elif row["km_to_safety_net"] is not None and row["km_to_safety_net"] < MIN_KM_FROM_SAFETY_NET:
        in_plan = False
        explanation = (f"A safety-net clinic that sees uninsured patients is only "
                       f"{row['km_to_safety_net']} km away, within walking distance, so the "
                       "planner's rules skip it regardless of rank. The van goes where "
                       "affordable care is out of reach.")
    else:
        in_plan = False
        explanation = (f"It ranks #{row['need_rank']} for need. This month's Saturdays went to "
                       f"higher-need neighborhoods (ranked #{lowest_picked_rank} or higher).")

    return {"neighborhood": row, "in_plan": in_plan, "explanation": explanation, "notes": NOTES}
