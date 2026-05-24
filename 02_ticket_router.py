"""
Project 02 — AI Customer Support Ticket Router
================================================

LangGraph Concepts Covered:
  - Conditional edges (add_conditional_edges) — routing to different nodes
    based on the current State, instead of a fixed path.
  - Cycles / loops — the graph can loop BACK to a previous node when the
    LLM returns 'Unknown', creating a retry loop.
  - Loop prevention — a counter in the State ensures the graph doesn't
    loop forever; after N failures it breaks out to a safe exit node.
  - LLM structured output — using .with_structured_output() so the LLM
    returns a Pydantic model instead of free-form text.

Graph shape:

  START ──▶ analyze_ticket ──┐
                ▲             │
                │  (Unknown   │ route_ticket (conditional edge)
                │   & < 3)    │
                └─────────────┤
                              │ (Billing / Technical) ──▶ END
                              │
                              │ (Unknown & >= 3)
                              ▼
                       human_escalation ──▶ END
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define the State
# ---------------------------------------------------------------------------
# attempt_counter is a plain int (no reducer). Each node overwrites it
# directly — we don't need to accumulate, just track the current count.
# ---------------------------------------------------------------------------


class TicketState(TypedDict):
    ticket_text: str
    category: str
    attempt_counter: int


# ---------------------------------------------------------------------------
# 2. Define structured output schema
# ---------------------------------------------------------------------------
# with_structured_output() forces the LLM to return a Pydantic model.
# This gives us a guaranteed .category field we can branch on — no need
# to parse free-form text or hope the LLM follows formatting instructions.
# ---------------------------------------------------------------------------


class TicketCategory(BaseModel):
    category: Literal["Billing", "Technical", "Unknown"] = Field(
        description="The category of the support ticket"
    )


# ---------------------------------------------------------------------------
# 3. Create the LLM with structured output
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.0)
structured_llm = llm.with_structured_output(TicketCategory)

# ---------------------------------------------------------------------------
# 4. Define the Nodes
# ---------------------------------------------------------------------------


def analyze_ticket(state: TicketState) -> dict:
    """Ask the LLM to categorize the ticket. If 'Unknown', bump the counter."""

    ticket = state["ticket_text"]
    attempt = state["attempt_counter"]

    result: TicketCategory = structured_llm.invoke(
        f"You are a customer support classifier. "
        f"Categorize this ticket as 'Billing', 'Technical', or 'Unknown'. "
        f"Only use 'Unknown' if the ticket truly does not fit Billing or Technical.\n\n"
        f"Ticket: {ticket}"
    )

    new_attempt = attempt + 1 if result.category == "Unknown" else attempt

    return {
        "category": result.category,
        "attempt_counter": new_attempt,
    }


def human_escalation(state: TicketState) -> dict:
    """Safety net — the LLM failed too many times, escalate to a human."""

    return {
        "category": "Escalated",
    }


# ---------------------------------------------------------------------------
# 5. Define the Routing Function (conditional edge)
# ---------------------------------------------------------------------------
# This is the KEY new concept. Instead of a fixed edge like:
#     add_edge("analyze_ticket", "some_node")
#
# We use add_conditional_edges, which calls this function AFTER
# analyze_ticket finishes. The function inspects the State and returns
# a STRING that matches one of the edge mappings we define.
#
# Loop prevention lives here: if the counter hits 3, we force-route
# to human_escalation regardless of the category.
# ---------------------------------------------------------------------------


def route_ticket(state: TicketState) -> str:
    """
    Decide where to go after analyze_ticket.

    Returns a string key that maps to a node name (or END).
    """

    # Loop prevention — break out after 3 failed attempts
    if state["category"] == "Unknown" and state["attempt_counter"] >= 3:
        return "escalate"

    # Unknown but under the limit — loop back and retry
    if state["category"] == "Unknown":
        return "retry"

    # Billing or Technical — we have a valid category, finish
    return "done"


# ---------------------------------------------------------------------------
# 6. Build the Graph
# ---------------------------------------------------------------------------

graph_builder = StateGraph(TicketState)

# Register nodes
graph_builder.add_node("analyze_ticket", analyze_ticket)
graph_builder.add_node("human_escalation", human_escalation)

# Entry edge — always start at analyze_ticket
graph_builder.add_edge(START, "analyze_ticket")

# Conditional edge — after analyze_ticket, call route_ticket() to decide
# The dict maps the STRING returned by route_ticket to a node (or END).
graph_builder.add_conditional_edges(
    "analyze_ticket",               # source node
    route_ticket,                   # routing function
    {
        "retry": "analyze_ticket",  # loop back
        "escalate": "human_escalation",  # break out to human
        "done": END,                # valid category, finish
    },
)

# human_escalation always ends the graph
graph_builder.add_edge("human_escalation", END)

# Compile
router = graph_builder.compile()

# ---------------------------------------------------------------------------
# 7. Run the Graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Test 1: Clear ticket (should categorize immediately) ---
    print("=" * 60)
    print("TEST 1 — Clear billing ticket")
    print("=" * 60)

    result = router.invoke({
        "ticket_text": "I was charged twice for my subscription last month. Please refund.",
        "category": "",
        "attempt_counter": 0,
    })

    print(f"  Category : {result['category']}")
    print(f"  Attempts : {result['attempt_counter']}")

    # --- Test 2: Vague ticket (should trigger retries & escalation) ---
    print("\n" + "=" * 60)
    print("TEST 2 — Intentionally vague ticket (expect escalation)")
    print("=" * 60)

    result = router.invoke({
        "ticket_text": "hmm things feel weird lately idk",
        "category": "",
        "attempt_counter": 0,
    })

    print(f"  Category : {result['category']}")
    print(f"  Attempts : {result['attempt_counter']}")

    if result["category"] == "Escalated":
        print("  → Loop prevention caught it — routed to human escalation.")
    else:
        print(f"  → LLM managed to classify it as '{result['category']}' before hitting the limit.")
