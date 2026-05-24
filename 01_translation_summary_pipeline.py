"""
Project 01 — LLM Translation & Summary Pipeline
=================================================

LangGraph Concepts Covered:
  - TypedDict State schema with typed keys
  - operator.add as a reducer for list-based state keys
  - Nodes as plain Python functions that receive the full State dict
  - Each node reads only the specific keys it needs and returns ONLY
    the keys it wants to update — LangGraph merges the returned dict
    back into the State automatically. Nodes never need to forward
    the entire State or manage chat history themselves.
  - Sequential edges: START -> translate -> summarize -> log -> END
  - Compiling and invoking the graph
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
# TypedDict gives us a fixed schema — every node sees the same shape.
# 'processing_log' uses Annotated[..., operator.add] so that when a node
# returns {"processing_log": ["some entry"]}, LangGraph APPENDS to the
# existing list instead of replacing it.  All other keys are simple
# overwrites — the last value written wins.
# ---------------------------------------------------------------------------


class PipelineState(TypedDict):
    original_text: str
    translated_text: str
    summary: str
    processing_log: Annotated[list[str], operator.add]


# ---------------------------------------------------------------------------
# 2. Create the LLM
# ---------------------------------------------------------------------------
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

# ---------------------------------------------------------------------------
# 3. Define the Nodes
# ---------------------------------------------------------------------------
# KEY INSIGHT — how nodes interact with State:
#
#   • Each node receives the FULL State dict as its only argument.
#   • It reads whichever keys it needs (e.g. state["original_text"]).
#   • It returns a dict containing ONLY the keys it wants to update.
#   • LangGraph merges that partial dict back into the State.
#
# This means:
#   - translate_text reads "original_text", writes "translated_text" + log
#   - summarize_text reads "translated_text", writes "summary" + log
#   - log_completion reads nothing from LLM, just appends to the log
#
# No node ever needs to carry forward the full State or manage a message
# history — LangGraph handles that merge for you.
# ---------------------------------------------------------------------------


def translate_text(state: PipelineState) -> dict:
    """Translate the original text to Spanish using the LLM."""

    # Read only the key this node cares about
    original = state["original_text"]

    response = llm.invoke(
        f"Translate the following English text to Spanish. "
        f"Return ONLY the translated text, nothing else.\n\n{original}"
    )

    # Return only the keys we want to update
    return {
        "translated_text": response.content,
        "processing_log": [f"translate_text: translated {len(original)} chars to Spanish"],
    }


def summarize_text(state: PipelineState) -> dict:
    """Summarize the Spanish translation using the LLM."""

    # This node reads "translated_text" — it never touches "original_text"
    spanish = state["translated_text"]

    response = llm.invoke(
        f"Summarize the following Spanish text in 1-2 concise Spanish sentences. "
        f"Return ONLY the summary.\n\n{spanish}"
    )

    return {
        "summary": response.content,
        "processing_log": [f"summarize_text: summarized {len(spanish)} chars of Spanish text"],
    }


def log_completion(state: PipelineState) -> dict:
    """Pure Python node — no LLM call, just appends a final log entry."""

    return {
        "processing_log": ["log_completion: pipeline finished successfully"],
    }


# ---------------------------------------------------------------------------
# 4. Build the Graph
# ---------------------------------------------------------------------------
# StateGraph is parameterized by our PipelineState so it knows the schema.
# We add three nodes, then wire them in a straight line:
#
#   START ──▶ translate_text ──▶ summarize_text ──▶ log_completion ──▶ END
#
# add_edge(A, B) means "after node A finishes, always run node B next."
# ---------------------------------------------------------------------------

graph_builder = StateGraph(PipelineState)

# Register nodes
graph_builder.add_node("translate_text", translate_text)
graph_builder.add_node("summarize_text", summarize_text)
graph_builder.add_node("log_completion", log_completion)

# Wire edges sequentially
graph_builder.add_edge(START, "translate_text")
graph_builder.add_edge("translate_text", "summarize_text")
graph_builder.add_edge("summarize_text", "log_completion")
graph_builder.add_edge("log_completion", END)

# Compile into a runnable graph
pipeline = graph_builder.compile()

# ---------------------------------------------------------------------------
# 5. Run the Pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_text = (
        "Artificial intelligence is transforming industries worldwide. "
        "From healthcare diagnostics to autonomous vehicles, AI systems "
        "are becoming integral to modern life. However, ethical considerations "
        "around bias, privacy, and job displacement remain critical challenges "
        "that society must address."
    )

    result = pipeline.invoke({"original_text": input_text})

    print("=" * 60)
    print("ORIGINAL TEXT")
    print("=" * 60)
    print(result["original_text"])

    print("\n" + "=" * 60)
    print("SPANISH TRANSLATION")
    print("=" * 60)
    print(result["translated_text"])

    print("\n" + "=" * 60)
    print("SPANISH SUMMARY")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("PROCESSING LOG")
    print("=" * 60)
    for entry in result["processing_log"]:
        print(f"  • {entry}")
