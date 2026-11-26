from typing import Optional

from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.genai import types

from .tools import explain_neighborhood, get_top_need_neighborhoods, plan_saturdays

MODEL = "gemini-flash-lite-latest"
MAX_REVIEW_ROUNDS = 3


# ---------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------
def _message(text: str) -> types.Content:
    """Wraps text as a reply, for callbacks that answer instead of running an agent."""
    return types.Content(role="model", parts=[types.Part(text=text)])


def _has_plan(state) -> bool:
    """True if plan_saturdays ran in this request AND picked at least one Saturday."""
    plan = state.get("plan_data")
    return bool(plan and plan.get("saturdays"))


# ---------------------------------------------------------------------------------
# Callbacks: plain Python that ADK runs BEFORE an agent starts.
# Returning None = carry on. Returning a message = skip that agent, use the message.
# ---------------------------------------------------------------------------------
def start_fresh(callback_context: CallbackContext) -> Optional[types.Content]:
    """Start of every request: wipe the previous request's results from the whiteboard."""
    callback_context.state["plan_data"] = None
    callback_context.state["review"] = ""
    callback_context.state["plan_approved"] = False
    return None


def review_gate(callback_context: CallbackContext) -> Optional[types.Content]:
    """Before the reviewer: if this request made no Saturday plan (for example, a
    question about one tract), there is nothing to review, so leave the loop now."""
    if not _has_plan(callback_context.state):
        callback_context.actions.escalate = True   # tells the LoopAgent: stop looping
        return _message("(No Saturday plan was made in this request, so there is nothing to review.)")
    return None


def flyer_gate(callback_context: CallbackContext) -> Optional[types.Content]:
    """Before the explainer: flyers only for a plan that exists AND was approved."""
    if not _has_plan(callback_context.state):
        return _message("No flyer this time: no Saturday plan was made in this request.")
    if not callback_context.state.get("plan_approved"):
        return _message(
            f"No flyers: the reviewer did not approve the plan's write-up within "
            f"{MAX_REVIEW_ROUNDS} rounds. See the reviewer's notes above."
        )
    return None


# ---------------------------------------------------------------------------------
# The reviewer's tool: the ONLY way to approve, and the only way out of the loop.
# ---------------------------------------------------------------------------------
def approve_plan(tool_context: ToolContext) -> dict:
    """Approves the planner's written plan. Call this ONLY when every check has passed."""
    tool_context.state["plan_approved"] = True
    tool_context.actions.escalate = True   # tells the LoopAgent: stop looping
    return {"approved": True}


# ---------------------------------------------------------------------------------
# Agent 1: the planner. Writes for staff: precise, numbers, estimates labeled.
# If the reviewer sent feedback, it appears at {review?} and must be fixed.
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
        "include them. If you do not have enough information, say so instead of guessing.\n\n"
        "Reviewer feedback on your previous draft (empty on the first draft). If there "
        "is any, rewrite your plan and fix every point:\n{review?}"
    ),
    tools=[plan_saturdays, get_top_need_neighborhoods, explain_neighborhood],
    output_key="plan",
)

# ---------------------------------------------------------------------------------
# Agent 2: the reviewer. A strict fact-checker: hedge, cite, refuse if unsure.
# It does not see the chat, only the written plan and the exact data.
# ---------------------------------------------------------------------------------
reviewer = Agent(
    name="reviewer",
    model=MODEL,
    include_contents="none",
    before_agent_callback=review_gate,
    instruction=(
        "You are a strict fact-checker for a county health department. You never "
        "rewrite the plan yourself; you only check it.\n\n"
        "The planner's written plan:\n{plan}\n\n"
        "The exact data the written plan must match:\n{plan_data}\n\n"
        "Check all of these:\n"
        "1. Every tract ID and number in the written plan matches the data. Rounding to "
        "fewer decimals is fine; any other difference is an error.\n"
        "2. The Saturdays are exactly the ones in the data, in the same order.\n"
        "3. No overclaiming. Words like 'highest', 'worst' or 'only' are allowed only if "
        "the data shows it. A rank of 1.0 means among the very worst in the county, "
        "not necessarily the single worst. need_rank 1 does mean the highest need score.\n"
        "4. Health and insurance figures are described as estimates.\n"
        "5. No neighborhood or street names appear (the data has none).\n\n"
        "If every check passes, call approve_plan, then reply only 'Approved.'\n"
        "If any check fails, do NOT call approve_plan. Reply with a numbered list of "
        "fixes: quote the wrong words and give the correct value from the data."
    ),
    tools=[approve_plan],
    output_key="review",
)

# ---------------------------------------------------------------------------------
# The loop: planner writes, reviewer checks. Repeats until approve_plan is called,
# or MAX_REVIEW_ROUNDS is reached (a safety net, so it can never loop forever).
# ---------------------------------------------------------------------------------
plan_review = LoopAgent(
    name="plan_review",
    sub_agents=[planner, reviewer],
    max_iterations=MAX_REVIEW_ROUNDS,
)

# ---------------------------------------------------------------------------------
# Agent 3: the explainer. Writes for residents: warm, simple, NO numbers, EN + ES.
# ---------------------------------------------------------------------------------
explainer = Agent(
    name="explainer",
    model=MODEL,
    include_contents="none",
    before_agent_callback=flyer_gate,
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
# The orchestrator: start fresh, then (plan <-> review loop), then the explainer.
# ---------------------------------------------------------------------------------
root_agent = SequentialAgent(
    name="clinic_plan",
    sub_agents=[plan_review, explainer],
    before_agent_callback=start_fresh,
)
