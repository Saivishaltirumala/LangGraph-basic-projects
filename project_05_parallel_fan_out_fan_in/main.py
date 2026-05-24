"""
Project 05 — Parallel Node Execution (Fan-Out & Fan-In)
========================================================

LangGraph Concepts Covered:
  - Fan-out — routing START to multiple nodes simultaneously so they
    run in parallel (not sequentially).
  - Fan-in — pointing multiple parallel nodes to the same downstream
    node. LangGraph automatically WAITS for ALL upstream nodes to
    finish before running the downstream node.
  - operator.add reducer — the critical piece that makes parallel
    writes safe. Without it, whichever node finishes last would
    overwrite the other's data.

Graph shape:

  START ──┬──▶ research_competitor_a ──┬──▶ compare_profiles ──▶ END
          │                            │
          └──▶ research_competitor_b ──┘

How parallel timing works:

  1. LangGraph sees two edges from START → it launches BOTH nodes
     at the same time (concurrent execution).

  2. Each node runs independently. Node A might finish in 1.2s,
     Node B in 2.5s — doesn't matter.

  3. compare_profiles has TWO incoming edges (from A and from B).
     LangGraph will NOT run it until BOTH A and B have completed
     and their results have been merged into the State.

  4. Only then does compare_profiles execute, seeing the combined
     research_results list from both nodes.

How operator.add prevents data loss:

  Without a reducer, both parallel nodes would write to "research_results"
  and the last one to finish would OVERWRITE the first:
    Node A finishes first → State: ["AWS S3 profile"]
    Node B finishes second → State: ["GCS profile"]  ← A's data is GONE

  With operator.add, LangGraph APPENDS instead of replacing:
    Node A finishes → State: ["AWS S3 profile"]
    Node B finishes → State: ["AWS S3 profile", "GCS profile"]  ← both kept

  This is why operator.add is essential for ANY key that multiple
  parallel nodes write to.
"""

import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Define the State
# ---------------------------------------------------------------------------
# research_results uses operator.add so both parallel nodes can safely
# append to it without overwriting each other.
# ---------------------------------------------------------------------------


class ComparisonState(TypedDict):
    topic: str
    research_results: Annotated[list[str], operator.add]
    final_comparison: str


# ---------------------------------------------------------------------------
# 2. Create the LLM
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

# ---------------------------------------------------------------------------
# 3. Define the Parallel Research Nodes
# ---------------------------------------------------------------------------
# Both nodes run at the SAME time. Each reads "topic" and appends ONE
# item to "research_results". Because of operator.add, both items end
# up in the list — no data loss.
# ---------------------------------------------------------------------------


def research_competitor_a(state: ComparisonState) -> dict:
    """Profile Competitor A (AWS S3) based on the topic."""

    topic = state["topic"]

    response = llm.invoke(
        f"You are a cloud technology analyst. The topic is '{topic}'.\n\n"
        f"Write a brief 3-4 sentence profile of AWS S3 covering: "
        f"key features, pricing model, and target audience. "
        f"Start your response with 'AWS S3:'"
    )

    # Returns a LIST with one item — operator.add will append it
    return {
        "research_results": [response.content],
    }


def research_competitor_b(state: ComparisonState) -> dict:
    """Profile Competitor B (Google Cloud Storage) based on the topic."""

    topic = state["topic"]

    response = llm.invoke(
        f"You are a cloud technology analyst. The topic is '{topic}'.\n\n"
        f"Write a brief 3-4 sentence profile of Google Cloud Storage covering: "
        f"key features, pricing model, and target audience. "
        f"Start your response with 'Google Cloud Storage:'"
    )

    return {
        "research_results": [response.content],
    }


# ---------------------------------------------------------------------------
# 4. Define the Fan-In Node
# ---------------------------------------------------------------------------
# This node will NOT run until both parallel nodes have finished.
# By the time it executes, research_results contains BOTH profiles
# thanks to the operator.add reducer.
# ---------------------------------------------------------------------------


def compare_profiles(state: ComparisonState) -> dict:
    """Compare both profiles using the LLM — runs only after both are ready."""

    profiles = state["research_results"]

    response = llm.invoke(
        f"You are a cloud technology analyst. Compare these two services "
        f"and write a concise pros/cons comparison.\n\n"
        f"Profile 1:\n{profiles[0]}\n\n"
        f"Profile 2:\n{profiles[1]}\n\n"
        f"Format as:\n"
        f"PROS & CONS of each, then a 1-sentence recommendation."
    )

    return {
        "final_comparison": response.content,
    }


# ---------------------------------------------------------------------------
# 5. Build the Graph
# ---------------------------------------------------------------------------

graph_builder = StateGraph(ComparisonState)

# Register all three nodes
graph_builder.add_node("research_competitor_a", research_competitor_a)
graph_builder.add_node("research_competitor_b", research_competitor_b)
graph_builder.add_node("compare_profiles", compare_profiles)

# FAN-OUT: START routes to BOTH research nodes simultaneously.
# LangGraph sees two edges from START and launches both nodes in parallel.
graph_builder.add_edge(START, "research_competitor_a")
graph_builder.add_edge(START, "research_competitor_b")

# FAN-IN: Both research nodes point to compare_profiles.
# LangGraph will WAIT for both to complete before running compare_profiles.
# This is automatic — no explicit "wait" or "barrier" needed.
graph_builder.add_edge("research_competitor_a", "compare_profiles")
graph_builder.add_edge("research_competitor_b", "compare_profiles")

# compare_profiles → END
graph_builder.add_edge("compare_profiles", END)

# Compile
comparison_graph = graph_builder.compile()

# ---------------------------------------------------------------------------
# 6. Run the Graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("PARALLEL FAN-OUT / FAN-IN PIPELINE")
    print("=" * 60)

    start_time = time.time()

    result = comparison_graph.invoke({
        "topic": "Cloud Storage",
        "research_results": [],
        "final_comparison": "",
    })

    elapsed = time.time() - start_time

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  (If sequential, this would take ~2x longer)\n")

    print("-" * 60)
    print("RESEARCH RESULT 1")
    print("-" * 60)
    print(result["research_results"][0])

    print("\n" + "-" * 60)
    print("RESEARCH RESULT 2")
    print("-" * 60)
    print(result["research_results"][1])

    print("\n" + "-" * 60)
    print("FINAL COMPARISON")
    print("-" * 60)
    print(result["final_comparison"])

    print("\n" + "-" * 60)
    print("PARALLEL PROOF")
    print("-" * 60)
    print(f"  research_results has {len(result['research_results'])} items")
    print("  → Both parallel nodes wrote to the same list without data loss.")
