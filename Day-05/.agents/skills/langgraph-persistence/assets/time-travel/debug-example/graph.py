"""
Time travel example: debugging with checkpoint history.

Demonstrates:
- Browsing checkpoint history with get_state_history()
- Resuming from a past checkpoint
- Modifying state at a checkpoint with update_state()
- Forking execution for what-if scenarios
"""

import asyncio
from typing import TypedDict, Annotated, Literal

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# ChatOpenAI(model="gpt-4.1"): direct provider usage
# init_chat_model("openai:gpt-4.1"): provider-agnostic (recommended)
model = FakeListChatModel(responses=[
    "Category: technical_support",
    "Here's how to fix the issue: restart the service and check logs.",
    "Category: billing",
    "I'll connect you with our billing department for assistance.",
])


# ==== State ====

class TicketState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    resolution: str


# ==== Nodes ====

async def classify(state: TicketState) -> dict:
    """Classify the support ticket."""
    response = await model.ainvoke(state["messages"])
    category = "technical" if "technical" in response.content.lower() else "billing"
    return {"category": category, "messages": [response]}


async def resolve_technical(state: TicketState) -> dict:
    """Handle technical support tickets."""
    response = await model.ainvoke(state["messages"])
    return {"resolution": response.content, "messages": [response]}


async def resolve_billing(state: TicketState) -> dict:
    """Handle billing tickets."""
    response = await model.ainvoke(state["messages"])
    return {"resolution": response.content, "messages": [response]}


def route_by_category(state: TicketState) -> Literal["resolve_technical", "resolve_billing"]:
    """Route based on classification result."""
    if state["category"] == "technical":
        return "resolve_technical"
    return "resolve_billing"


# ==== Graph ====

builder = StateGraph(TicketState)
builder.add_node("classify", classify)
builder.add_node("resolve_technical", resolve_technical)
builder.add_node("resolve_billing", resolve_billing)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_category)
builder.add_edge("resolve_technical", END)
builder.add_edge("resolve_billing", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ==== Main: Demonstrate time travel ====

async def main():
    config = {"configurable": {"thread_id": "ticket_001"}}

    # Step 1: Run the graph
    print("=== Step 1: Initial Run ===")
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="My server keeps crashing")],
        "category": "",
        "resolution": "",
    }, config=config)
    print(f"Category: {result['category']}")
    print(f"Resolution: {result['resolution'][:60]}...")
    print()

    # Step 2: Browse checkpoint history
    print("=== Step 2: Checkpoint History ===")
    checkpoints = []
    async for snapshot in graph.aget_state_history(config):
        checkpoints.append(snapshot)
        step = snapshot.metadata.get("step", "?")
        source = snapshot.metadata.get("source", "start")
        print(f"  Step {step} | Source: {source} | Category: {snapshot.values.get('category', 'N/A')}")
    print()

    # Step 3: Modify state — change category to "billing" at classify step
    print("=== Step 3: Time Travel — Change Classification ===")
    # Find the checkpoint after classify node
    classify_checkpoint = None
    for cp in checkpoints:
        if cp.values.get("category") and cp.metadata.get("step", 0) > 0:
            classify_checkpoint = cp
            break

    if classify_checkpoint:
        past_config = classify_checkpoint.config

        # Modify: change category from "technical" to "billing"
        await graph.aupdate_state(
            past_config,
            values={"category": "billing"},
            as_node="classify",
        )

        # Resume from modified state
        forked_result = await graph.ainvoke(None, config=past_config)
        print(f"Forked category: {forked_result['category']}")
        print(f"Forked resolution: {forked_result['resolution'][:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
