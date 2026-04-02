from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict
import operator

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
# Choose one of the following model configurations:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        '{"subtasks": [{"id": 1, "query": "Analyze introduction section"}, {"id": 2, "query": "Analyze methodology section"}, {"id": 3, "query": "Analyze conclusion section"}]}',
        "Completed analysis of introduction section with key findings.",
        "Completed analysis of methodology section with detailed observations.",
        "Completed analysis of conclusion section with summary insights.",
    ]
)

# Option 2: OpenAI ChatGPT (COMMENTED)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")

# Option 3: Generic chat model initialization (COMMENTED)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")

# ==============================================================================


class OrchestratorState(TypedDict):
    """State for orchestrator-worker pattern."""

    task: str
    subtasks: list[dict]
    results: Annotated[list[dict], operator.add]
    expected_count: int
    final_result: str


async def orchestrator_node(state: OrchestratorState) -> dict:
    """Orchestrator generates subtasks dynamically using LLM."""
    task = state["task"]

    # Use LLM to generate subtasks
    prompt = f"""Given the following task, break it down into 3-5 distinct subtasks.
Return ONLY a JSON object with this exact structure:
{{"subtasks": [{{"id": 1, "query": "description"}}, {{"id": 2, "query": "description"}}, ...]}}

Task: {task}"""

    response = await model.ainvoke(prompt)

    # Parse LLM response (FakeListChatModel returns string content)
    import json

    subtasks_data = json.loads(
        response.content if hasattr(response, "content") else str(response)
    )
    subtasks = subtasks_data.get("subtasks", [])

    return {"subtasks": subtasks, "expected_count": len(subtasks)}


async def worker_node(state: dict) -> dict:
    """Worker processes a single subtask using LLM."""
    subtask = state["subtask"]

    # Use LLM to process the subtask
    prompt = f"Process the following subtask and provide a detailed result:\n{subtask['query']}"
    response = await model.ainvoke(prompt)

    result_text = response.content if hasattr(response, "content") else str(response)
    return {"results": [{"id": subtask["id"], "result": result_text}]}


async def aggregator_node(state: OrchestratorState) -> dict:
    """Aggregator combines all worker results."""
    results = state.get("results", [])
    expected = state.get("expected_count", 0)

    # Wait for all workers to complete
    if expected and len(results) < expected:
        return {}

    # Sort by ID and combine results
    sorted_results = sorted(results, key=lambda r: r.get("id", 0))
    final_result = "\n\n".join(
        f"Subtask {r['id']}: {r['result']}" for r in sorted_results
    )
    return {"final_result": final_result}


def create_graph():
    """Create the orchestrator-worker graph."""
    graph = StateGraph(OrchestratorState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("worker", worker_node)
    graph.add_node("aggregator", aggregator_node)

    graph.add_edge(START, "orchestrator")

    def dispatch_workers(state: OrchestratorState):
        """Fan-out: Send each subtask to a worker."""
        return [Send("worker", {"subtask": subtask}) for subtask in state["subtasks"]]

    graph.add_conditional_edges("orchestrator", dispatch_workers)

    graph.add_edge("worker", "aggregator")
    graph.add_edge("aggregator", END)

    return graph.compile()


graph = create_graph()


async def main():
    """Main async entry point."""
    result = await graph.ainvoke(
        {
            "task": "Prepare a comprehensive analysis of research paper structure",
            "subtasks": [],
            "results": [],
            "expected_count": 0,
            "final_result": "",
        }
    )

    print("=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result.get("final_result", "(no result)"))
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
