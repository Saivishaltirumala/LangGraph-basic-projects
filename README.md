# LangGraph Basic Projects

A hands-on learning path through LangGraph — each project introduces a unique core concept.

## Projects

| # | Project | Concepts Covered |
|---|---------|-----------------|
| 1 | Translation & Summary Pipeline | TypedDict State, `operator.add` reducer, sequential nodes & edges, compile & invoke |
| 2 | Customer Support Ticket Router | Conditional edges, cycles/loops, loop prevention, LLM structured output |
| 3 | Fault Tolerance & Human-in-the-Loop | RetryPolicy, MemorySaver checkpointing, `interrupt_before`, `Command(resume=True)` |
| 4 | Supervisor Multi-Agent Architecture | Subgraphs, state isolation, supervisor routing pattern |
| 5 | Parallel Fan-Out & Fan-In | Parallel node execution, `operator.add` for concurrent writes, automatic fan-in barrier |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and add your Anthropic API key

# 4. Run a project
python project_01_translation_summary_pipeline/main.py
```

## Prerequisites

- Python 3.10+
- An Anthropic API key
- Basic Python knowledge
