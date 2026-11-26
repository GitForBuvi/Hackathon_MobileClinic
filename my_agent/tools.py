"""Tools for the mobile clinic planner.

Every function in this file is a tool: an ordinary Python function that an agent
can decide to call. ADK reads each function's name, parameters and docstring, and
tells Gemini what the tool does and when to use it.
"""

import datetime
import math
import re

from google.adk.tools import ToolContext
from google.cloud import bigquery

PROJECT_ID = "hackathon-bigquery-515300"
VIEW = f"{PROJECT_ID}.a4i_health.v_van_need"

# ---------------------------------------------------------------------------------
# Planning rules
# These are policy decisions, so they live here in code, not in the chat. A message
# like "skip the spacing rule" cannot change them; only editing this file can.
# ---------------------------------------------------------------------------------
# How many Saturdays: by default, every Saturday of next month (4 or 5).
# The user may ask for fewer. The rules below about WHERE always stay fixed.
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


def _next_month_saturdays(today: datetime.date | None = None) -> list[datetime.date]:
    """All Saturdays of the month after `today` (4 or 5 of them)."""
    today = today or datetime.date.today()
    first = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    day = first + datetime.timedelta(days=(5 - first.weekday()) % 7)   # 5 = Saturday
    saturdays = []
    while day.month == first.month:
        saturdays.append(day)
        day += datetime.timedelta(days=7)
    return saturdays


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


def plan_saturdays(number_of_saturdays: int = 0, tool_context: ToolContext = None) -> dict:
    """Builds next month's plan: which Santa Clara neighborhoods the mobile clinic van visits, one per Saturday.

    Works down the need ranking (highest need first) and applies two fixed rules:
      1. Skip a neighborhood within 1 km of a safety-net clinic (care is already walkable there).
      2. Skip a neighborhood within 2 km of one already picked (it would serve the same people).

    Args:
        number_of_saturdays: How many Saturdays to plan. Leave it out (or 0) to plan
            every Saturday of next month. Pass a number only if the user asked for one.

    Returns:
        A dict with 'saturdays' (the picks, in order, each with its date), 'skipped'
        (higher-ranked neighborhoods passed over, with the reason), 'rules' and 'notes'.
    """
    dates = _next_month_saturdays()
    requested = int(number_of_saturdays or 0)
    if requested <= 0:
        number_of_saturdays = len(dates)                          # default: every Saturday
    else:
        number_of_saturdays = max(1, min(requested, len(dates)))  # never more than the month has
    candidates = _query_top(CANDIDATES)

    saturdays: list[dict] = []
    skipped: list[dict] = []

    for row in candidates:
        if len(saturdays) == number_of_saturdays:
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
        row["date"] = dates[len(saturdays)].isoformat()   # e.g. "2026-12-05"
        saturdays.append(row)

    result = {
        "saturdays": saturdays,
        "skipped": skipped,
        "rules": {
            "number_of_saturdays": number_of_saturdays,
            "month": dates[0].strftime("%B %Y"),
            "saturdays_in_month": len(dates),
            "min_km_apart": MIN_KM_APART,
            "min_km_from_safety_net": MIN_KM_FROM_SAFETY_NET,
            "candidates_considered": CANDIDATES,
        },
        "notes": NOTES,
    }
    if requested > len(dates):
        result["note_on_count"] = (f"{requested} Saturdays were asked for, but "
                                   f"{dates[0].strftime('%B %Y')} has only {len(dates)}, "
                                   f"so {number_of_saturdays} were planned.")
    if len(saturdays) < number_of_saturdays:
        result["warning"] = (f"Only {len(saturdays)} neighborhoods met the rules among the "
                             f"top {CANDIDATES}. Say so instead of filling the gap.")
    if tool_context is not None:
        tool_context.state["plan_data"] = result   # the exact plan, for other agents
    return result


def explain_neighborhood(geo_id: str, tool_context: ToolContext = None) -> dict:
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

    # Explain against the plan the user actually saw: the approved plan saved in
    # this conversation. With no approved plan yet, share the tract's own facts only.
    plan = tool_context.state.get("approved_plan") if tool_context is not None else None
    if not plan:
        return {
            "neighborhood": row,
            "in_plan": None,
            "explanation": (
                "No Saturday plan has been made yet in this conversation, so this tract is "
                "not in or out of a plan yet. Ask to make the Saturday plan first. "
                f"Facts about the tract: it ranks #{row['need_rank']} for need, and the "
                f"nearest safety-net clinic is {row['km_to_safety_net']} km away."
            ),
            "notes": NOTES,
        }

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
