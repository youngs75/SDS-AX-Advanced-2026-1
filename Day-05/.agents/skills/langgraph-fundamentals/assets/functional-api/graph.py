"""
Functional API example: @entrypoint + @task workflow.

Demonstrates:
- @entrypoint as workflow starting point
- @task for discrete, checkpointable work units
- Injectable 'previous' parameter for short-term memory
- Parallel task execution with futures
- entrypoint.final for decoupling return from saved state
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# ChatOpenAI(model="gpt-4.1"): direct provider usage
# init_chat_model("openai:gpt-4.1"): provider-agnostic (recommended)
model = FakeListChatModel(responses=[
    "Research findings: LangGraph uses a Pregel-based execution model for durable workflows.",
    "Summary: LangGraph provides durable execution through its Pregel runtime, enabling checkpointing and replay.",
    "Final analysis: The key insight is that LangGraph's functional API simplifies workflow creation while maintaining full durability guarantees.",
])


# ==== Tasks (discrete work units) ====

@task
async def research(query: str) -> str:
    """Research a topic using the model."""
    response = await model.ainvoke([HumanMessage(content=f"Research: {query}")])
    return response.content


@task
async def summarize(text: str) -> str:
    """Summarize research findings."""
    response = await model.ainvoke([HumanMessage(content=f"Summarize: {text}")])
    return response.content


@task
async def analyze(research_result: str, summary: str) -> str:
    """Combine research and summary into final analysis."""
    response = await model.ainvoke([
        HumanMessage(content=f"Analyze based on:\nResearch: {research_result}\nSummary: {summary}")
    ])
    return response.content


# ==== Entrypoint (workflow definition) ====

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
async def research_workflow(query: str, *, previous: list = []) -> dict:
    """
    Research workflow using Functional API.

    - 'previous' injects the last saved return value (short-term memory)
    - Tasks are checkpointed individually for durability
    - Parallel tasks launch before .result() is called
    """
    # Launch tasks (can run concurrently until .result())
    research_future = research(query)

    # Get research result
    research_result = research_future.result()

    # Sequential: summarize depends on research
    summary = summarize(research_result).result()

    # Final analysis combines both
    final = analyze(research_result, summary).result()

    # Build result with history tracking
    history = previous + [{"query": query, "result": final}]

    # entrypoint.final: return summary to caller, save full history to checkpoint
    return entrypoint.final(
        value={"query": query, "analysis": final, "history_length": len(history)},
        save=history,
    )


# ==== Main ====

async def main():
    config = {"configurable": {"thread_id": "research_session_1"}}

    # First query
    result1 = await research_workflow.ainvoke("LangGraph execution model", config=config)
    print(f"Query 1: {result1['query']}")
    print(f"Analysis: {result1['analysis']}")
    print(f"History length: {result1['history_length']}")
    print()

    # Second query (same thread — 'previous' contains history)
    result2 = await research_workflow.ainvoke("Durable execution patterns", config=config)
    print(f"Query 2: {result2['query']}")
    print(f"History length: {result2['history_length']}")  # Should be 2


if __name__ == "__main__":
    asyncio.run(main())
