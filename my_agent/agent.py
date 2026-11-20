from google.adk.agents import Agent
from google.cloud import bigquery

PROJECT_ID = "hackathon-bigquery-515300"
VIEW = f"{PROJECT_ID}.a4i_health.v_van_need"


def get_top_need_neighborhoods(limit: int = 10) -> dict:
    """Returns the Santa Clara County neighborhoods (census tracts) that most need the mobile clinic van.

    Neighborhoods are ranked by need_score (0-100, higher = more need). It is an
    "access first" score: it weighs distance to the nearest safety-net clinic and
    households without a car most heavily, then the uninsured rate, routine checkups,
    chronic disease and mobility difficulty.

    Args:
        limit: How many neighborhoods to return, highest need first. Use 5 to 20.

    Returns:
        A dict with 'neighborhoods' (one entry per neighborhood, highest need first)
        and 'notes' explaining what the numbers mean and how reliable they are.
    """
    limit = max(1, min(int(limit), 50))

    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT geo_id, need_rank, need_score, total_pop,
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

    return {
        "neighborhoods": [dict(row.items()) for row in rows],
        "notes": (
            "geo_id is a census tract ID, not a neighborhood name. "
            "rank_* values run from 0 (best in Santa Clara) to 1 (worst in Santa Clara). "
            "A rank of 1.0 means among the very worst in the county, not necessarily the single worst. "
            "A rank of 1.0 means among the very worst in the county, not necessarily the single worst. "
            "pct_uninsured is a MODELED estimate (CDC PLACES) for adults 18-64, not a count. "
            "pct_no_vehicle comes from a Census survey. "
            "km_to_safety_net is a straight line to the nearest federally recognized clinic "
            "that sees uninsured patients, inside the county only, so it can overstate "
            "distance near the county edge."
        ),
    }


root_agent = Agent(
    name="clinic_helper",
    model="gemini-flash-latest",
    instruction=(
        "You help the Santa Clara County health department decide where its mobile "
        "clinic van should park on Saturdays.\n"
        "When asked where the van should go, or which neighborhoods need it most, call "
        "get_top_need_neighborhoods and base your answer on its results.\n"
        "For each neighborhood you recommend, explain WHY in plain words using its "
        "rank_* values, for example 'far from a clinic that takes uninsured patients, "
        "and many households have no car'.\n"
        "Say that health and insurance figures are estimates. "
        "Refer to places by their tract ID. Do not guess neighborhood or street names, "
        "because the data does not include them. "
        "If you do not have enough information, say so instead of guessing."
    ),
    tools=[get_top_need_neighborhoods],
)
