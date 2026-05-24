"""
Project 03 — Fault Tolerance, Checkpointing & Human-in-the-Loop
=================================================================

LangGraph Concepts Covered:
  - RetryPolicy — attached to a node so LangGraph automatically retries
    on transient failures (no try/except in your code).
  - MemorySaver checkpointer — persists graph state across invocations
    using a thread_id, so you can pause and resume later.
  - interrupt_before — pauses the graph BEFORE a specific node runs,
    giving a human the chance to inspect or approve.
  - Command(resume=True) — resumes a paused graph from exactly where
    it stopped, using the same thread_id.

Graph shape:

  START ──▶ fetch_external_data ──┤ (pause here) ──▶ draft_report ──▶ END
              (RetryPolicy: 3x)       interrupt_before

Flow:
  1st invoke  → runs fetch_external_data (retries on failure) → PAUSES
  2nd invoke  → human approves → runs draft_report → END
"""

import random
from typing import TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define the State
# ---------------------------------------------------------------------------


class ReportState(TypedDict):
    topic: str
    fetched_data: str
    report: str
    status: str


# ---------------------------------------------------------------------------
# 2. Create the LLM
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

# ---------------------------------------------------------------------------
# 3. Define the Nodes
# ---------------------------------------------------------------------------


def fetch_external_data(state: ReportState) -> dict:
    """
    Simulates a flaky external API that fails ~50% of the time.

    IMPORTANT: There is NO try/except here. We let the ConnectionError
    propagate upward. LangGraph's RetryPolicy (attached when we register
    this node) catches the exception and retries automatically.

    RetryPolicy handles everything:
      - Waits between retries (exponential backoff)
      - Retries up to max_attempts times
      - If all retries fail, the error finally surfaces to the caller
    """

    print("    [fetch_external_data] Attempting to fetch data...")

    # 50% chance of failure
    if random.random() < 0.5:
        print("    [fetch_external_data] ✗ ConnectionError! (RetryPolicy will handle this)")
        raise ConnectionError("Simulated API failure — server unavailable")

    # Success path
    fake_data = (
        f"Market research data for '{state['topic']}': "
        f"The global market is projected to grow 12% YoY. "
        f"Key players include companies A, B, and C. "
        f"Consumer sentiment is trending positive."
    )

    print("    [fetch_external_data] ✓ Data fetched successfully")

    return {
        "fetched_data": fake_data,
        "status": "data_fetched",
    }


def draft_report(state: ReportState) -> dict:
    """Uses the LLM to draft a short report from the fetched data."""

    print("    [draft_report] Generating report with LLM...")

    response = llm.invoke(
        f"Write a 2-3 sentence executive summary based on this data:\n\n"
        f"{state['fetched_data']}"
    )

    return {
        "report": response.content,
        "status": "report_drafted",
    }


# ---------------------------------------------------------------------------
# 4. Build the Graph
# ---------------------------------------------------------------------------

graph_builder = StateGraph(ReportState)

# Register fetch_external_data WITH a RetryPolicy.
# This tells LangGraph: "If this node raises an exception, retry it
# automatically up to 3 times with exponential backoff."
#
# RetryPolicy params:
#   max_attempts=3    → try up to 3 times before giving up
#   initial_interval=0.5 → wait 0.5s before first retry
#   backoff_factor=2.0   → double the wait each retry (0.5s, 1s, 2s)
#   jitter=True          → add randomness to avoid thundering herd
#
# The node code stays clean — no try/except, no retry logic.
# LangGraph handles it at the framework level.
graph_builder.add_node(
    "fetch_external_data",
    fetch_external_data,
    retry_policy=RetryPolicy(max_attempts=3),
)

# draft_report has no retry — LLM calls are generally reliable
graph_builder.add_node("draft_report", draft_report)

# Wire edges
graph_builder.add_edge(START, "fetch_external_data")
graph_builder.add_edge("fetch_external_data", "draft_report")
graph_builder.add_edge("draft_report", END)

# ---------------------------------------------------------------------------
# 5. Compile with Checkpointer + Interrupt
# ---------------------------------------------------------------------------
# MemorySaver() — an in-memory checkpointer that saves the graph State
# after each node completes. When you invoke with the same thread_id,
# LangGraph loads the saved State and continues from where it left off.
#
# interrupt_before=["draft_report"] — tells LangGraph to PAUSE execution
# right before the "draft_report" node runs. The invoke() call returns
# immediately with the current State. The graph is frozen in place.
#
# This is the Human-in-the-Loop pattern:
#   1. The graph does automated work (fetch data)
#   2. It pauses for human review
#   3. A human inspects the fetched data
#   4. If approved, they resume → the graph continues to draft_report
# ---------------------------------------------------------------------------

checkpointer = MemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["draft_report"],
)

# ---------------------------------------------------------------------------
# 6. Run the Graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Config with a thread_id — this is the "session key" for the checkpointer.
    # Same thread_id = same conversation/state. Different thread_id = fresh start.
    config = {"configurable": {"thread_id": "ticket-42"}}

    # ── PHASE 1: Invoke the graph ──
    # This will:
    #   1. Run fetch_external_data (with retries if it fails)
    #   2. PAUSE before draft_report (because of interrupt_before)
    #   3. Return the State at the pause point
    print("=" * 60)
    print("PHASE 1 — Invoke (will pause before draft_report)")
    print("=" * 60)

    result = graph.invoke(
        {
            "topic": "AI in Healthcare",
            "fetched_data": "",
            "report": "",
            "status": "started",
        },
        config=config,
    )

    print(f"\n  Status      : {result['status']}")
    print(f"  Fetched data: {result['fetched_data'][:80]}...")
    print(f"  Report      : {result['report'] or '(empty — graph paused before drafting)'}")

    # ── Show the graph is paused ──
    # get_state() loads the checkpointed State for this thread_id.
    # .next tells us which node is queued to run when we resume.
    current_state = graph.get_state(config)
    print(f"\n  Next node(s): {current_state.next}")
    print("  → Graph is PAUSED. In a real app, a human reviews fetched_data here.")

    # ── PHASE 2: Human approves, resume the graph ──
    # Command(resume=True) tells LangGraph: "continue from the pause point."
    # It loads the checkpointed State (same thread_id), and picks up
    # exactly where it stopped — running draft_report next.
    #
    # No data is re-fetched. No nodes are re-run. The State is intact.
    print("\n" + "=" * 60)
    print("PHASE 2 — Human approves, resuming graph")
    print("=" * 60)

    result = graph.invoke(Command(resume=True), config=config)

    print(f"\n  Status : {result['status']}")
    print(f"  Report : {result['report']}")

    # Confirm the graph is now complete
    final_state = graph.get_state(config)
    print(f"\n  Next node(s): {final_state.next}")
    print("  → Graph is COMPLETE.")
