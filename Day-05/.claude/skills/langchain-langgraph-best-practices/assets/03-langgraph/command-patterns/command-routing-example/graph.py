"""
Command and Send patterns example.

Demonstrates:
- Command for dynamic routing with simultaneous state updates
- Send for map-reduce fan-out to parallel workers
- Annotated reducer for aggregating parallel results
"""

import asyncio
import operator
from typing import TypedDict, Annotated, Literal

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# ChatOpenAI(model="gpt-4.1"): direct provider usage
# init_chat_model("openai:gpt-4.1"): provider-agnostic (recommended)
model = FakeListChatModel(responses=[
    '{"subtasks": ["research market trends", "analyze competitors", "summarize findings"]}',
    "Market trends show growth in AI sector.",
    "Competitor analysis reveals focus on automation.",
    "Key findings: AI market growing, competitors investing in automation.",
    "Final report: AI market analysis complete with 3 key insights.",
])


# ==== State ====

class OrchestratorState(TypedDict):
    task: str
    subtasks: list[str]
    results: Annotated[list[dict], operator.add]  # reducer: append worker results
    final_report: str


# ==== Pattern 1: Command for routing + state update ====

async def planner_node(state: OrchestratorState) -> Command[Literal["fan_out"]]:
    """Plan subtasks and route to fan-out with state update."""
    response = await model.ainvoke([
        HumanMessage(content=f"Break this task into subtasks: {state['task']}")
    ])

    # Parse subtasks (simplified for demo)
    import json
    try:
        parsed = json.loads(response.content)
        subtasks = parsed["subtasks"]
    except (json.JSONDecodeError, KeyError):
        subtasks = [state["task"]]

    # Command: update state AND route in one atomic operation
    return Command(
        goto="fan_out",
        update={"subtasks": subtasks},
    )


# ==== Pattern 2: Send for map-reduce fan-out ====

async def fan_out_node(state: OrchestratorState) -> list[Send]:
    """Fan out subtasks to parallel workers using Send."""
    return [
        Send("worker", {"task": state["task"], "subtask": subtask, "results": []})
        for subtask in state["subtasks"]
    ]


async def worker_node(state: OrchestratorState) -> dict:
    """Process a single subtask. Runs in parallel for each Send."""
    response = await model.ainvoke([
        HumanMessage(content=f"Execute subtask: {state.get('subtask', 'unknown')}")
    ])
    return {
        "results": [{"subtask": state.get("subtask", "unknown"), "output": response.content}]
    }


async def aggregator_node(state: OrchestratorState) -> dict:
    """Aggregate all worker results into final report."""
    results_summary = "\n".join(
        f"- {r['subtask']}: {r['output']}" for r in state["results"]
    )
    response = await model.ainvoke([
        HumanMessage(content=f"Create final report from:\n{results_summary}")
    ])
    return {"final_report": response.content}


# ==== Graph ====

builder = StateGraph(OrchestratorState)
builder.add_node("planner", planner_node)
builder.add_node("fan_out", fan_out_node)
builder.add_node("worker", worker_node)
builder.add_node("aggregator", aggregator_node)

builder.add_edge(START, "planner")
# planner uses Command to route to fan_out
builder.add_edge("fan_out", "worker")  # Send handles actual fan-out
builder.add_edge("worker", "aggregator")
builder.add_edge("aggregator", END)

graph = builder.compile()


# ==== Main ====

async def main():
    result = await graph.ainvoke({
        "task": "Analyze the AI market",
        "subtasks": [],
        "results": [],
        "final_report": "",
    })

    print(f"Task: {result['task']}")
    print(f"Subtasks: {result['subtasks']}")
    print(f"Results collected: {len(result['results'])}")
    print(f"Final report: {result['final_report']}")


if __name__ == "__main__":
    asyncio.run(main())
