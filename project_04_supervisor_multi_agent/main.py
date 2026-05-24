"""
Project 04 — Supervisor Multi-Agent Architecture with Subgraphs
================================================================

LangGraph Concepts Covered:
  - Subgraphs — compiled StateGraphs embedded as nodes inside a parent
    graph. Each subgraph has its own State schema, providing isolation.
  - State isolation — subgraph-private keys (like internal scratchpads)
    never leak into the parent State. Only keys that OVERLAP between
    the parent and child schemas are read/written across the boundary.
  - Supervisor pattern — a central LLM node inspects the parent State
    and decides which agent (subgraph) to route to next, or whether
    to finish.

Why subgraphs matter (state isolation):

  Without subgraphs, if Research and Writer were plain nodes in the
  SAME graph, they'd share the SAME State dict. The Researcher might
  store messy internal notes, raw source URLs, or a chain-of-thought
  scratchpad — and the Writer node would see ALL of that, potentially
  getting confused or polluting the final article with research noise.

  With subgraphs, each agent gets its OWN State schema:
    - ResearchState has "internal_notes" (private scratchpad)
    - WriterState has "style_notes" (private scratchpad)
    - SupervisorState (parent) has NEITHER — it only sees the clean
      shared keys: topic, research_summary, draft_article

  LangGraph handles the boundary automatically:
    - When entering a subgraph: copies MATCHING keys from parent → child
    - When exiting a subgraph: copies MATCHING keys from child → parent
    - Private keys stay inside the subgraph and are discarded on exit

Graph shape:

  START ──▶ supervisor ──┬──▶ research_agent (subgraph) ──▶ supervisor
                         │
                         ├──▶ writer_agent (subgraph) ──▶ supervisor
                         │
                         └──▶ END
"""

from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Create the LLM
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

# ===========================================================================
#  SUBGRAPH 1 — Research Agent
# ===========================================================================
# This subgraph has its OWN State schema. "internal_notes" is PRIVATE —
# the parent graph will never see it because SupervisorState doesn't
# have an "internal_notes" key. Only "topic" and "research_summary"
# overlap with the parent, so only those cross the boundary.
# ===========================================================================


class ResearchState(TypedDict):
    topic: str
    research_summary: str
    internal_notes: str  # PRIVATE — never leaks to parent


def do_research(state: ResearchState) -> dict:
    """LLM researches the topic. Stores messy notes internally."""

    topic = state["topic"]

    # The LLM generates internal notes (scratchpad / chain-of-thought)
    notes_response = llm.invoke(
        f"You are a research assistant. For the topic '{topic}', "
        f"jot down 3-4 raw bullet points of key facts, statistics, "
        f"and sources. This is your private scratchpad — be messy."
    )

    # Then produces a clean summary from those notes
    summary_response = llm.invoke(
        f"Based on these research notes:\n\n{notes_response.content}\n\n"
        f"Write a clean, concise 3-4 sentence research summary on '{topic}'."
    )

    return {
        "research_summary": summary_response.content,
        "internal_notes": notes_response.content,  # stays inside subgraph
    }


# Build and compile the Research subgraph
research_builder = StateGraph(ResearchState)
research_builder.add_node("do_research", do_research)
research_builder.add_edge(START, "do_research")
research_builder.add_edge("do_research", END)
research_agent = research_builder.compile()


# ===========================================================================
#  SUBGRAPH 2 — Writer Agent
# ===========================================================================
# Same pattern — "style_notes" is PRIVATE to this subgraph.
# The Writer only sees "research_summary" (input) and writes
# "draft_article" (output). It never sees the Researcher's
# internal_notes, keeping the writing clean and uncontaminated.
# ===========================================================================


class WriterState(TypedDict):
    research_summary: str
    draft_article: str
    style_notes: str  # PRIVATE — never leaks to parent


def write_article(state: WriterState) -> dict:
    """LLM writes an article from the research summary."""

    summary = state["research_summary"]

    # Writer's private planning step
    style_response = llm.invoke(
        f"You are an editor. Given this research summary:\n\n{summary}\n\n"
        f"Jot down a brief note about the tone, structure, and angle "
        f"you'd take for a short article. This is your private plan."
    )

    # Then writes the actual article
    article_response = llm.invoke(
        f"You are a professional writer. Using your editorial plan:\n\n"
        f"{style_response.content}\n\n"
        f"And this research summary:\n\n{summary}\n\n"
        f"Write a short 2-3 paragraph article. Make it engaging and clear."
    )

    return {
        "draft_article": article_response.content,
        "style_notes": style_response.content,  # stays inside subgraph
    }


# Build and compile the Writer subgraph
writer_builder = StateGraph(WriterState)
writer_builder.add_node("write_article", write_article)
writer_builder.add_edge(START, "write_article")
writer_builder.add_edge("write_article", END)
writer_agent = writer_builder.compile()


# ===========================================================================
#  PARENT GRAPH — Supervisor
# ===========================================================================
# The Supervisor State only has the SHARED keys: topic, research_summary,
# draft_article. It does NOT have internal_notes or style_notes.
#
# When LangGraph enters the research subgraph:
#   Parent {topic, research_summary} → copies to → ResearchState
#   ResearchState finishes with {research_summary, internal_notes}
#   Only research_summary copies back → Parent (internal_notes discarded)
#
# Same for the writer subgraph — style_notes never reaches the parent.
# ===========================================================================


class SupervisorState(TypedDict):
    topic: str
    research_summary: str
    draft_article: str
    next_step: str


# --- Structured output for the Supervisor's routing decision ---

class SupervisorDecision(BaseModel):
    next: Literal["research", "writer", "finish"] = Field(
        description=(
            "Which agent to run next: "
            "'research' if no research_summary exists yet, "
            "'writer' if research is done but no draft_article yet, "
            "'finish' if both are complete."
        )
    )


structured_supervisor = llm.with_structured_output(SupervisorDecision)


def supervisor(state: SupervisorState) -> dict:
    """
    The Supervisor inspects the current State and decides the next step.
    Uses structured output to guarantee a clean routing decision.
    """

    decision: SupervisorDecision = structured_supervisor.invoke(
        f"You are a project supervisor managing a Research Agent and a Writer Agent.\n\n"
        f"Current state:\n"
        f"  - topic: {state['topic']}\n"
        f"  - research_summary: {state['research_summary'] or '(empty)'}\n"
        f"  - draft_article: {state['draft_article'] or '(empty)'}\n\n"
        f"Decide the next step:\n"
        f"  - 'research' → if research_summary is empty, we need research first\n"
        f"  - 'writer'   → if research is done but draft_article is empty\n"
        f"  - 'finish'   → if both research_summary and draft_article are complete"
    )

    return {"next_step": decision.next}


def route_supervisor(state: SupervisorState) -> str:
    """Routing function — maps the supervisor's decision to a node."""
    return state["next_step"]


# ---------------------------------------------------------------------------
# Build the Parent Graph
# ---------------------------------------------------------------------------

parent_builder = StateGraph(SupervisorState)

# Register nodes — compiled subgraphs are added just like regular nodes
parent_builder.add_node("supervisor", supervisor)
parent_builder.add_node("research", research_agent)  # compiled subgraph as a node
parent_builder.add_node("writer", writer_agent)       # compiled subgraph as a node

# Entry: always start at the supervisor
parent_builder.add_edge(START, "supervisor")

# Supervisor decides where to go next
parent_builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "research": "research",
        "writer": "writer",
        "finish": END,
    },
)

# After each agent finishes, go back to the supervisor for the next decision
parent_builder.add_edge("research", "supervisor")
parent_builder.add_edge("writer", "supervisor")

# Compile the parent graph
supervisor_graph = parent_builder.compile()

# ---------------------------------------------------------------------------
# Run the Graph
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SUPERVISOR MULTI-AGENT PIPELINE")
    print("=" * 60)

    result = supervisor_graph.invoke({
        "topic": "The Impact of Artificial Intelligence on Modern Healthcare",
        "research_summary": "",
        "draft_article": "",
        "next_step": "",
    })

    print("\n" + "-" * 60)
    print("TOPIC")
    print("-" * 60)
    print(result["topic"])

    print("\n" + "-" * 60)
    print("RESEARCH SUMMARY")
    print("-" * 60)
    print(result["research_summary"])

    print("\n" + "-" * 60)
    print("DRAFT ARTICLE")
    print("-" * 60)
    print(result["draft_article"])

    print("\n" + "-" * 60)
    print("FINAL ROUTING DECISION")
    print("-" * 60)
    print(result["next_step"])

    # Prove that internal_notes and style_notes are NOT in the final state
    print("\n" + "-" * 60)
    print("STATE ISOLATION CHECK")
    print("-" * 60)
    print(f"  'internal_notes' in result? {'internal_notes' in result}")
    print(f"  'style_notes' in result?    {'style_notes' in result}")
    print("  → Subgraph private keys stayed private.")
