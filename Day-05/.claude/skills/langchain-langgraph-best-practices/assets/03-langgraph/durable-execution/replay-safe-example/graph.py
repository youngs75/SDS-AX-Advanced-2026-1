"""
Durable execution example: replay-safe patterns.

Demonstrates:
- Idempotent side effects in nodes
- Check-before-act pattern for safe replay
- Checkpointer for durability
- Resuming from checkpoint after failure
"""

import asyncio
from typing import TypedDict, Annotated
from typing_extensions import NotRequired

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
    "Processing order #12345: 2x Widget A, 1x Widget B",
    "Order #12345 confirmed and ready for shipment.",
])


# ==== State ====

class OrderState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: str
    processed: bool
    confirmation: NotRequired[str]


# ==== Simulated external system ====

# In-memory "database" tracking processed orders
_processed_orders: set[str] = set()


async def process_order_in_external_system(order_id: str) -> str:
    """Simulate external API call with idempotency."""
    # CHECK-BEFORE-ACT: Idempotent — safe to call multiple times
    if order_id in _processed_orders:
        return f"Order {order_id} already processed (idempotent skip)"

    # Simulate processing
    _processed_orders.add(order_id)
    return f"Order {order_id} processed successfully"


# ==== Nodes ====

async def validate_order(state: OrderState) -> dict:
    """Validate and describe the order. Pure computation — replay-safe."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


async def process_order(state: OrderState) -> dict:
    """
    Process order with idempotent side effect.

    REPLAY-SAFE because:
    1. check-before-act pattern prevents double processing
    2. External system call is idempotent
    3. State update is deterministic given the same input
    """
    result = await process_order_in_external_system(state["order_id"])
    return {
        "processed": True,
        "confirmation": result,
    }


async def confirm_order(state: OrderState) -> dict:
    """Generate confirmation message. Pure computation — replay-safe."""
    response = await model.ainvoke(
        state["messages"] + [HumanMessage(content=f"Confirmation: {state.get('confirmation', '')}")]
    )
    return {"messages": [response]}


# ==== Graph ====

builder = StateGraph(OrderState)
builder.add_node("validate", validate_order)
builder.add_node("process", process_order)
builder.add_node("confirm", confirm_order)

builder.add_edge(START, "validate")
builder.add_edge("validate", "process")
builder.add_edge("process", "confirm")
builder.add_edge("confirm", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ==== Main ====

async def main():
    config = {"configurable": {"thread_id": "order_session_1"}}
    input_data = {
        "messages": [HumanMessage(content="Process order #12345")],
        "order_id": "12345",
        "processed": False,
    }

    # First run — processes normally
    result = await graph.ainvoke(input_data, config=config)
    print(f"First run:")
    print(f"  Processed: {result['processed']}")
    print(f"  Confirmation: {result.get('confirmation', 'N/A')}")
    print()

    # Simulate replay (same session) — idempotent, won't double-process
    result2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="Process order #12345 again")],
         "order_id": "12345", "processed": False},
        config=config,
    )
    print(f"Replay (idempotent):")
    print(f"  Confirmation: {result2.get('confirmation', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(main())
