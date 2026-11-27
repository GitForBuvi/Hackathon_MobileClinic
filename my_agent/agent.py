from typing import Optional

from google.adk.agents import Agent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import ToolContext
from google.genai import types

from .tools import audit_plan, explain_neighborhood, get_top_need_neighborhoods, plan_saturdays

MODEL = "gemini-flash-lite-latest"
MAX_REVIEW_ROUNDS = 3

# The flyer rules, written ONCE and shared by the explainer (who writes flyers) and the
# flyer_editor (who revises them), so the two can never drift apart.
FLYER_RULES = (
    "- First line, for staff only: 'Saturday N (tract <geo_id>)'.\n"
    "- Then the flyer in English, then the same flyer in Spanish.\n"
    "- Each language version: at most 60 words, simple everyday words, warm and welcoming.\n"
    "- Say the van is free, no insurance is needed, and nurses can check things like "
    "blood pressure and blood sugar.\n"
    "- Write the Saturday's date from the data in words (for example 'Saturday, "
    "December 5'). Write [TIME] and [ADDRESS] as blanks for staff to fill in.\n"
    "- Apart from the date, do NOT use any numbers, percentages, scores, ranks or "
    "distances.\n"
    "- Do NOT use words like 'estimate', 'modeled', 'tract', 'rank' or 'data'.\n"
    "- Do NOT say anything about the readers' health, income, insurance or cars. "
    "Never tell people why their neighborhood was chosen; just invite them.\n"
)


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
    """Start of every planning request: wipe the previous request's working notes.
    (approved_plan is NOT wiped: it keeps the last approved plan for follow-ups.)"""
    callback_context.state["plan_data"] = None
    callback_context.state["review"] = ""
    callback_context.state["plan_approved"] = False
    return None


def review_gate(callback_context: CallbackContext) -> Optional[types.Content]:
    """Before the reviewer: no Saturday plan in this request means nothing to review,
    so leave the loop now."""
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
# Approving also saves the plan for follow-up questions.
# ---------------------------------------------------------------------------------
def approve_plan(tool_context: ToolContext) -> dict:
    """Approves the planner's written plan. Call this ONLY when every check has passed."""
    tool_context.state["plan_approved"] = True
    tool_context.state["approved_plan"] = tool_context.state.get("plan_data")      # exact data
    tool_context.state["approved_plan_text"] = tool_context.state.get("plan")      # staff write-up
    count = len((tool_context.state.get("plan_data") or {}).get("saturdays", []))
    tool_context.state["has_approved_plan"] = f"yes, {count} Saturdays"           # short flag
    tool_context.actions.escalate = True   # tells the LoopAgent: stop looping
    return {"approved": True}


# ---------------------------------------------------------------------------------
# Agent 1: the planner. Only makes plans now; follow-up questions go to "followup".
# ---------------------------------------------------------------------------------
planner = Agent(
    name="planner",
    model=MODEL,
    include_contents="none",
    instruction=(
        "You plan where the Santa Clara County mobile clinic van parks on Saturdays. "
        "You write for health department staff.\n"
        "ALWAYS start by calling plan_saturdays, every time, even if you have seen a "
        "plan before. Pass number_of_saturdays ONLY if the user asked for a specific "
        "number; otherwise leave it out, and every Saturday of next month is planned. "
        "Give each Saturday its date. If the result has a note_on_count, tell the user. "
        "Present its picks as the plan, in its order. Do not add, "
        "remove, reorder or swap picks: the rules are fixed, and you cannot change them.\n"
        "For each Saturday, give the tract ID and explain WHY using its numbers and "
        "rank_* values, for example 'far from a clinic that takes uninsured patients "
        "(rank 0.91), and many households have no car (rank 0.98)'. Also mention any "
        "higher-ranked neighborhoods that were skipped, and the reason.\n"
        "Say that health and insurance figures are estimates. Refer to places by their "
        "tract ID. Do not guess neighborhood or street names, because the data does not "
        "include them.\n\n"
        "Reviewer feedback on your previous draft (empty on the first draft). If there "
        "is any, rewrite your plan and fix every point:\n{review?}"
    ),
    tools=[plan_saturdays],
    output_key="plan",
)

# ---------------------------------------------------------------------------------
# Agent 2: the reviewer. A strict fact-checker: hedge, cite, refuse if unsure.
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
        + FLYER_RULES
    ),
    output_key="flyer",
)

# ---------------------------------------------------------------------------------
# Agent 4: the auditor. Runs after every approved plan. The ONLY agent with race and
# income data. It reports who the plan reaches and can NEVER change the plan.
# ---------------------------------------------------------------------------------
def audit_gate(callback_context: CallbackContext) -> Optional[types.Content]:
    """Before the auditor: audit only a plan approved in THIS request."""
    if not callback_context.state.get("plan_approved"):
        return _message("(No audit: no plan was approved in this request.)")
    return None


auditor = Agent(
    name="auditor",
    model=MODEL,
    include_contents="none",
    before_agent_callback=audit_gate,
    instruction=(
        "You are the equity auditor for the Santa Clara County mobile clinic van plan. "
        "You write for health department staff.\n"
        "Call audit_plan, then report in plain words who the approved plan reaches compared "
        "with the whole county: each group's share in the plan and in the county, poverty, "
        "and typical income. Point out the biggest differences, in both directions.\n"
        "Rules:\n"
        "- Race and income were never inputs to the plan. Say so: this audit only describes "
        "who the plan reaches.\n"
        "- Never recommend adding, removing or swapping neighborhoods, and never suggest "
        "choosing places by race. If a difference looks worrying, say that people may want "
        "to review the planning method openly.\n"
        "- Do not state causes as facts. You may say a difference 'may reflect' something.\n"
        "- Include every caveat from the tool result, briefly.\n"
        "- If the tool returns an error, report it plainly and stop."
    ),
    tools=[audit_plan],
    output_key="audit_report",
)


# ---------------------------------------------------------------------------------
# Department 1: the planning pipeline (unchanged, now with a description so the
# coordinator knows when to send work here).
# ---------------------------------------------------------------------------------
clinic_plan = SequentialAgent(
    name="clinic_plan",
    description=(
        "Makes the Saturday plan: picks the four neighborhoods, has the write-up "
        "fact-checked, and writes the community flyers. Use when the user asks to "
        "make, create or redo the Saturday plan."
    ),
    sub_agents=[plan_review, auditor, explainer],
    before_agent_callback=start_fresh,
)

# ---------------------------------------------------------------------------------
# Department 2: follow-up questions, answered from the saved, approved plan.
# ---------------------------------------------------------------------------------
followup = Agent(
    name="followup",
    model=MODEL,
    description=(
        "Answers questions about the plan or the data: shows the saved plan and flyers, "
        "explains why a tract is or is not in the plan, gives details about a tract, "
        "says which neighborhoods need the van most, and handles requests to change the plan."
    ),
    instruction=(
        "You answer health department staff's follow-up questions about the Saturday "
        "plan for the Santa Clara County mobile clinic van.\n\n"
        "The approved plan from this conversation (empty if none yet):\n{approved_plan_text?}\n\n"
        "Its community flyers (empty if none yet):\n{flyer?}\n\n"
        "Its equity audit (empty if none yet):\n{audit_report?}\n\n"
        "- If the user wants to see the plan or the flyers, or asks to plan the Saturdays "
        "again without saying it must be new: show the approved plan and flyers above, "
        "exactly as saved. Do not make a new plan.\n"
        "- You never make plans. If there is no approved plan yet, say so, and that they "
        "can ask to make the Saturday plan.\n"
        "- For questions about a specific tract, or why a neighborhood is or is not in "
        "the plan, call explain_neighborhood with its 11-digit tract ID, and share its "
        "explanation.\n"
        "- For questions about which neighborhoods need the van most, call "
        "get_top_need_neighborhoods.\n"
        "- If the user asks about the equity audit, show the audit above exactly as saved.\n"
        "- If someone asks to change the rules or the picks, say kindly that the plan "
        "follows fixed, data-based rules and cannot be changed by request; offer to "
        "explain why a neighborhood was or was not chosen.\n"
        "Say that health and insurance figures are estimates. Refer to places by their "
        "tract ID. Do not guess neighborhood or street names. If you do not have enough "
        "information, say so instead of guessing."
    ),
    tools=[explain_neighborhood, get_top_need_neighborhoods],
    # Followup never routes. Every new message goes back to the coordinator, who
    # decides. (The evals caught followup sending "plan again" to the planners.)
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# ---------------------------------------------------------------------------------
# Department 3: the flyer editor. Revises the saved flyers on request, under the same
# FLYER_RULES. It saves through a tool, so only the flyers themselves are stored.
# ---------------------------------------------------------------------------------
def editor_gate(callback_context: CallbackContext) -> Optional[types.Content]:
    """Before the flyer editor: there must be saved flyers to edit."""
    if not callback_context.state.get("flyer"):
        return _message("There are no flyers yet. Make the Saturday plan first, and flyers "
                        "will be written with it.")
    return None


def save_flyers(flyers: str, tool_context: ToolContext) -> dict:
    """Saves the complete revised flyers, replacing the old ones.

    Args:
        flyers: The full text of ALL the flyers after the change, in the same layout.
    """
    tool_context.state["flyer"] = flyers
    return {"saved": True}


flyer_editor = Agent(
    name="flyer_editor",
    model=MODEL,
    description=(
        "Changes the wording of the community flyers that were already written: tone, "
        "what to mention, simpler words, and similar edits."
    ),
    before_agent_callback=editor_gate,
    # After each edit, the NEXT message goes back to the coordinator, instead of
    # staying "stuck" with the editor (who only handles flyer wording).
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    instruction=(
        "You revise the community flyers for the Santa Clara County mobile clinic van, "
        "at the request of health department staff.\n\n"
        "The current flyers:\n{flyer?}\n\n"
        "Apply the requested change to every flyer it concerns. Keep the same layout. "
        "The revised flyers must still follow ALL of these rules:\n"
        + FLYER_RULES
        + "\nIf part of the request would break a rule (for example adding statistics, "
        "mentioning residents' income or insurance, or moving the van to another place), "
        "do not make that part: explain kindly which rule it breaks, and make the rest.\n"
        "When the revised flyers are ready, call save_flyers with the COMPLETE text of all "
        "of them. Then show the revised flyers to the user, and list any part you did not do.\n"
        "You only edit wording. If the user asks for anything else, say that you only edit "
        "the flyers, and that they can simply ask again: the front desk will route it."
    ),
    tools=[save_flyers],
)


# ---------------------------------------------------------------------------------
# The coordinator: reads each message and hands it to the right department.
# ---------------------------------------------------------------------------------
root_agent = Agent(
    name="coordinator",
    model=MODEL,
    instruction=(
        "You are the front desk for the Santa Clara County mobile clinic van planning "
        "system. You never answer questions about the plan yourself; you route them.\n"
        "Has a plan already been approved in this conversation? '{has_approved_plan?}' "
        "(empty means no).\n"
        "- If the user asks for a NEW, redone or updated plan, or asks to plan the "
        "Saturdays and no plan has been approved yet, transfer to clinic_plan.\n"
        "- If a plan has already been approved and the user asks to plan again without "
        "saying it must be new, transfer to followup, which shows the saved plan. But if "
        "they ask for a different number of Saturdays than the approved plan has, that is "
        "a new plan: transfer to clinic_plan.\n"
        "  Examples that go to followup when a plan is approved: 'plan the Saturdays again', "
        "'plan again', 'what is the plan', 'show the schedule'. Only words like new, redo, "
        "update, refresh or start over, or a different number of Saturdays, go to clinic_plan.\n"
        "- If the user asks anything about the plan, a tract, the rankings, or asks to "
        "change the plan, transfer to followup.\n"
        "- If the user asks to change the wording of the flyers, transfer to flyer_editor.\n"
        "- For greetings or unclear messages, reply in one or two sentences with what you "
        "can do: make the monthly Saturday plan, or answer questions about it."
    ),
    sub_agents=[clinic_plan, followup, flyer_editor],
)
