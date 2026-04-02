"""
Streaming example: multi-mode streaming with custom events.

Demonstrates:
- stream_mode="updates" for state deltas
- stream_mode="custom" with get_stream_writer()
- Multi-mode streaming (simultaneous modes)
- Async streaming with astream()
"""

import asyncio
import operator
from typing import TypedDict, Annotated

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.config import get_stream_writer

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# ChatOpenAI(model="gpt-4.1"): direct provider usage
# init_chat_model("openai:gpt-4.1"): provider-agnostic (recommended)
model = FakeListChatModel(responses=[
    "I found 3 relevant sources about LangGraph streaming.",
    "Based on my research, LangGraph supports 5 streaming modes: values, updates, custom, messages, and debug.",
])


# ==== State ====

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    sources_found: int


# ==== Nodes ====

async def search_node(state: ResearchState) -> dict:
    """Search node with custom progress events."""
    writer = get_stream_writer()

    # Emit custom progress events
    writer({"event": "search_start", "query": state["messages"][-1].content})

    # Simulate search phases
    writer({"event": "progress", "phase": "indexing", "percent": 30})
    response = await model.ainvoke(state["messages"])
    writer({"event": "progress", "phase": "ranking", "percent": 70})

    writer({"event": "search_complete", "percent": 100})

    return {
        "messages": [response],
        "sources_found": 3,
    }


async def summarize_node(state: ResearchState) -> dict:
    """Summarize findings."""
    writer = get_stream_writer()
    writer({"event": "summarize_start"})

    response = await model.ainvoke(state["messages"])
    writer({"event": "summarize_complete"})

    return {"messages": [response]}


# ==== Graph ====

builder = StateGraph(ResearchState)
builder.add_node("search", search_node)
builder.add_node("summarize", summarize_node)
builder.add_edge(START, "search")
builder.add_edge("search", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()


# ==== Main: Demonstrate streaming modes ====

async def main():
    input_data = {"messages": [HumanMessage(content="How does LangGraph streaming work?")]}

    # --- Mode 1: updates (state deltas per step) ---
    print("=== stream_mode='updates' ===")
    async for chunk in graph.astream(input_data, stream_mode="updates"):
        node_name = list(chunk.keys())[0]
        print(f"  [{node_name}] keys updated: {list(chunk[node_name].keys())}")

    print()

    # --- Mode 2: custom (user-defined events) ---
    print("=== stream_mode='custom' ===")
    async for event in graph.astream(input_data, stream_mode="custom"):
        print(f"  {event}")

    print()

    # --- Mode 3: multi-mode (updates + custom simultaneously) ---
    print("=== stream_mode=['updates', 'custom'] ===")
    async for mode, chunk in graph.astream(input_data, stream_mode=["updates", "custom"]):
        if mode == "custom":
            print(f"  [custom] {chunk}")
        elif mode == "updates":
            node_name = list(chunk.keys())[0]
            print(f"  [update] {node_name}: {list(chunk[node_name].keys())}")


if __name__ == "__main__":
    asyncio.run(main())
