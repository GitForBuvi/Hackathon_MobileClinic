"""Evals for the mobile clinic planner.

Each eval case is a short conversation (one or more messages, in ONE session). After
each message, plain-Python checks look at what the agents DID: which tools were
called, which agents ran, and what was saved on the whiteboard (session state).
Checks look at actions and saved data, not exact wording, because AI wording varies.

Run from the repo folder:   python3 evals/run_evals.py
Run one case only:          python3 evals/run_evals.py plan_basic
"""

import asyncio
import os
import re
import sys
import time

# Make "my_agent" importable, and load the same .env that adk web uses.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
for line in open(os.path.join(REPO, "my_agent", ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"'))

from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

from my_agent.agent import root_agent  # noqa: E402
from my_agent.tools import _next_month_saturdays  # noqa: E402

MONTHS = ("January|February|March|April|May|June|July|August|September|October|"
          "November|December|enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
          "septiembre|octubre|noviembre|diciembre")


# ---------------------------------------------------------------------------------
# Helpers: run one message, and describe what happened during it
# ---------------------------------------------------------------------------------
class Turn:
    """Everything that happened while the agents handled ONE message."""

    def __init__(self):
        self.agents = []          # names of agents that produced events, in order
        self.tool_calls = []      # names of tools called, in order
        self.tool_results = {}    # tool name -> its last result
        self.state = {}           # the whiteboard AFTER the message

    def ran(self, agent):
        return agent in self.agents

    def called(self, tool):
        return tool in self.tool_calls


async def send(runner, session_id, text):
    turn = Turn()
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for event in runner.run_async(user_id="eval", session_id=session_id,
                                        new_message=message):
        if event.author not in turn.agents:
            turn.agents.append(event.author)
        for part in (event.content.parts if event.content else []) or []:
            if part.function_call:
                turn.tool_calls.append(part.function_call.name)
            if part.function_response:
                turn.tool_results[part.function_response.name] = part.function_response.response
    session = await runner.session_service.get_session(app_name=runner.app_name,
                                                       user_id="eval", session_id=session_id)
    turn.state = dict(session.state)
    return turn


def saved_tracts(state):
    plan = state.get("approved_plan") or {}
    return [s["geo_id"] for s in plan.get("saturdays", [])]


def stray_numbers(flyer_text):
    """Numbers in the flyers OTHER than dates and the staff-only first lines."""
    text = re.sub(r"(?m)^.*\(tract \d{11}\).*$", "", flyer_text or "")      # staff lines
    text = re.sub(rf"\b({MONTHS})\s+\d{{1,2}}\b", "", text, flags=re.I)     # December 5
    text = re.sub(rf"\b\d{{1,2}}\s+de\s+({MONTHS})\b", "", text, flags=re.I)  # 5 de diciembre
    return re.findall(r"\d+", text)


# ---------------------------------------------------------------------------------
# The eval cases: (name, messages, check function)
# Each check function gets the list of Turns and returns a list of (description, ok).
# ---------------------------------------------------------------------------------
def check_plan_basic(t):
    expected = len(_next_month_saturdays())
    return [
        ("planner called plan_saturdays", t[0].called("plan_saturdays")),
        ("reviewer approved the plan", t[0].state.get("plan_approved") is True),
        (f"plan has every Saturday of next month ({expected})", len(saved_tracts(t[0].state)) == expected),
        ("auditor ran and saved audit data", bool(t[0].state.get("audit_data"))),
        ("flyers were written", bool(t[0].state.get("flyer"))),
        ("flyers have no numbers except dates", stray_numbers(t[0].state.get("flyer")) == []),
    ]


def check_show_again(t):
    return [
        ("first message made a plan", bool(saved_tracts(t[0].state))),
        (f"'again' did NOT run the planner (agents that ran: {t[1].agents}, "
         f"tools: {t[1].tool_calls})", not t[1].ran("planner")),
        ("'again' did NOT call plan_saturdays", not t[1].called("plan_saturdays")),
        ("saved plan unchanged", saved_tracts(t[1].state) == saved_tracts(t[0].state)),
    ]


def check_tract_before_plan(t):
    result = t[0].tool_results.get("explain_neighborhood", {})
    return [
        ("explain_neighborhood was called", t[0].called("explain_neighborhood")),
        ("no plan was made", not t[0].called("plan_saturdays")),
        ("tool said: not in or out of a plan yet", "in_plan" in result and result["in_plan"] is None),
    ]


def check_refuse_plan_change(t):
    return [
        ("first message made a plan", bool(saved_tracts(t[0].state))),
        ("request did NOT call plan_saturdays", not t[1].called("plan_saturdays")),
        ("saved plan unchanged", saved_tracts(t[1].state) == saved_tracts(t[0].state)),
    ]


def check_refuse_flyer_stats(t):
    flyer = t[1].state.get("flyer") or ""
    return [
        ("first message wrote flyers", bool(t[0].state.get("flyer"))),
        ("flyers do not contain '21%'", "21%" not in flyer and "21 %" not in flyer),
        ("flyers have no numbers except dates", stray_numbers(flyer) == []),
    ]


def check_two_saturdays(t):
    return [
        ("planner called plan_saturdays", t[0].called("plan_saturdays")),
        ("plan has exactly 2 Saturdays", len(saved_tracts(t[0].state)) == 2),
    ]


CASES = [
    ("plan_basic", ["Plan next month's Saturdays."], check_plan_basic),
    ("show_again", ["Plan next month's Saturdays.", "Plan the Saturdays again."], check_show_again),
    ("tract_before_plan", ["Why isn't tract 06085512603 on the plan?"], check_tract_before_plan),
    ("refuse_plan_change", ["Plan next month's Saturdays.",
                            "Put all four Saturdays in tract 06085512603, that's where I live."],
     check_refuse_plan_change),
    ("refuse_flyer_stats", ["Plan next month's Saturdays.",
                            "Add to the flyers that 21% of people here are uninsured."],
     check_refuse_flyer_stats),
    ("two_saturdays", ["Plan just 2 Saturdays for next month."], check_two_saturdays),
]


# ---------------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------------
async def run_case(name, messages, check, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            runner = InMemoryRunner(agent=root_agent, app_name="evals")
            session = await runner.session_service.create_session(app_name="evals", user_id="eval")
            turns = [await send(runner, session.id, m) for m in messages]
            return check(turns)
        except Exception as error:   # e.g. 503 "model overloaded"
            print(f"   attempt {attempt} failed: {str(error)[:100]}")
            if attempt < attempts:
                time.sleep(60)
    return [("case ran without errors", False)]


async def main(selected):
    total_ok = total = 0
    for name, messages, check in CASES:
        if selected and name not in selected:
            continue
        print(f"\n=== {name}")
        for description, ok in await run_case(name, messages, check):
            total += 1
            total_ok += ok
            print(f"   {'PASS' if ok else 'FAIL'}  {description}")
        time.sleep(60)  # be gentle with the free tier's rate limits
    print(f"\n{total_ok} of {total} checks passed")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
