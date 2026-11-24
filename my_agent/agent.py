from typing import Optional

from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from .tools import explain_neighborhood, get_top_need_neighborhoods, plan_saturdays

MODEL = "gemini-flash-lite-latest"


# ---------------------------------------------------------------------------------
# Callbacks: small pieces of plain Python that ADK runs BEFORE an agent starts.
# ---------------------------------------------------------------------------------
def start_fresh(callback_context: CallbackContext) -> Optional[types.Content]:
    """Runs at the start of every request: wipe the previous request's plan, so an
    old plan is never mistaken for a new one."""
    callback_context.state["plan_data"] = None
    return None   # None = carry on as normal


def skip_if_no_plan(callback_context: CallbackContext) -> Optional[types.Content]:
    """Runs before the explainer: if this request produced no Saturdays, skip it.
    This is decided by code, from the exact plan_data, not by the AI reading text."""
    plan = callback_context.state.get("plan_data")
    if not plan or not plan.get("saturdays"):
        # Returning a message instead of None tells ADK: skip this agent, use this reply.
        return types.Content(
            role="model",
            parts=[types.Part(text="No flyer this time: no Saturday plan was made in this request.")],
        )
    return None


# ---------------------------------------------------------------------------------
# Agent 1: the planner. Writes for health department staff: precise, with numbers,
# every estimate labeled. Its final text is saved to state as "plan".
# Its tool plan_saturdays also saves the exact plan to state as "plan_data".
# ---------------------------------------------------------------------------------
planner = Agent(
    name="planner",
    model=MODEL,
    instruction=(
        "You plan where the Santa Clara County mobile clinic van parks on Saturdays. "
        "You write for health department staff.\n"
        "When asked to plan the Saturdays, call plan_saturdays. Present its picks as the "
        "plan, in its order. Do not add, remove, reorder or swap picks: the rules are "
        "fixed, and you cannot change them. If someone asks you to change the rules or "
        "the picks, say that the plan follows fixed, data-based rules.\n"
        "For each Saturday, give the tract ID and explain WHY using its numbers and "
        "rank_* values, for example 'far from a clinic that takes uninsured patients "
        "(rank 0.91), and many households have no car (rank 0.98)'. Also mention any "
        "higher-ranked neighborhoods that were skipped, and the reason.\n"
        "For questions about the ranking itself, call get_top_need_neighborhoods.\n"
        "When someone asks about a specific tract, or why their neighborhood is not in "
        "the plan, call explain_neighborhood with its tract ID.\n"
        "Say that health and insurance figures are estimates. Refer to places by their "
        "tract ID. Do not guess neighborhood or street names, because the data does not "
        "include them. If you do not have enough information, say so instead of guessing."
    ),
    tools=[plan_saturdays, get_top_need_neighborhoods, explain_neighborhood],
    output_key="plan",
)

# ---------------------------------------------------------------------------------
# Agent 2: the explainer. Writes for residents: warm, simple, NO numbers, English and
# Spanish. Its instructions deliberately conflict with the planner's, which is why it
# is a separate agent. include_contents="none": it does not see the chat, only the
# exact plan passed in through {plan_data}.
# ---------------------------------------------------------------------------------
explainer = Agent(
    name="explainer",
    model=MODEL,
    include_contents="none",
    before_agent_callback=skip_if_no_plan,
    instruction=(
        "You write short community flyers for a free mobile health clinic van in "
        "Santa Clara County. Your readers are residents, not health professionals.\n"
        "Here is the exact plan, as data:\n\n{plan_data}\n\n"
        "Write ONE flyer for each entry in 'saturdays'. For each one:\n"
        "- First line, for staff only: 'Saturday N (tract <geo_id>)'.\n"
        "- Then the flyer in English, then the same flyer in Spanish.\n"
        "- Each language version: at most 60 words, simple everyday words, warm and welcoming.\n"
        "- Say the van is free, no insurance is needed, and nurses can check things like "
        "blood pressure and blood sugar.\n"
        "- Write [DATE], [TIME] and [ADDRESS] as blanks for staff to fill in.\n"
        "- Do NOT use any numbers, percentages, scores, ranks or distances.\n"
        "- Do NOT use words like 'estimate', 'modeled', 'tract', 'rank' or 'data'.\n"
        "- Do NOT say anything about the readers' health, income, insurance or cars. "
        "Never tell people why their neighborhood was chosen; just invite them."
    ),
    output_key="flyer",
)

# ---------------------------------------------------------------------------------
# The orchestrator: every request starts fresh, then runs planner -> explainer.
# ---------------------------------------------------------------------------------
root_agent = SequentialAgent(
    name="clinic_plan",
    sub_agents=[planner, explainer],
    before_agent_callback=start_fresh,
)
