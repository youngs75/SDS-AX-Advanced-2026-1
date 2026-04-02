"""
Multi-mode streaming example demonstrating token-level and state update streaming.

Shows how to use astream with multiple stream modes and custom events via get_stream_writer.

Asset: assets/02-langchain/streaming-example/graph.py
Reference: references/02-langchain/streaming.md
"""
import asyncio
from langchain_core.language_models import FakeListChatModel
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# --- Model Configuration ---
# FakeListChatModel: testing/prototyping
# ChatOpenAI(model="gpt-4o"): direct provider
# init_chat_model("openai:gpt-4o"): universal init (recommended)
model = FakeListChatModel(responses=[
    "I'll analyze the sales data for Q4.",
    "Analysis complete. Q4 sales show a 15% increase over Q3."
])

from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


@tool
async def analyze_data(query: str) -> str:
    """Analyze data with progress updates."""
    writer = get_stream_writer()

    # Simulate multi-stage analysis with custom events
    await asyncio.sleep(0.1)
    writer({"status": "loading", "progress": 0, "message": "Loading data..."})

    await asyncio.sleep(0.1)
    writer({"status": "analyzing", "progress": 50, "message": "Analyzing patterns..."})

    await asyncio.sleep(0.1)
    writer({"status": "complete", "progress": 100, "message": "Analysis complete"})

    return f"Analysis complete for: {query}"


# Create agent with tool
agent = create_agent(
    model=model,
    tools=[analyze_data],
)


async def main():
    """Demonstrate different streaming modes."""
    input_msg = {"messages": [{"role": "user", "content": "Analyze sales data for Q4"}]}

    # Multi-mode streaming: messages + updates
    print("=== Multi-mode Streaming (messages + updates) ===")
    async for stream_mode, chunk in agent.astream(
        input_msg,
        stream_mode=["messages", "updates"]
    ):
        if stream_mode == "messages":
            # Token-level streaming
            msg_chunk = chunk[0] if isinstance(chunk, tuple) else chunk
            if hasattr(msg_chunk, 'content') and msg_chunk.content:
                print(f"[Token] {msg_chunk.content}", end="", flush=True)
        elif stream_mode == "updates":
            # State updates per node
            for node, update in chunk.items():
                print(f"\n[Update] Node: {node}, Keys: {list(update.keys())}")

    # Custom event streaming (from get_stream_writer)
    print("\n\n=== Custom Events ===")
    async for chunk in agent.astream(input_msg, stream_mode="custom"):
        if isinstance(chunk, dict):
            status = chunk.get("status", "unknown")
            progress = chunk.get("progress", 0)
            message = chunk.get("message", "")
            print(f"[Custom] {status.upper()}: {progress}% - {message}")

    # Values mode: full state after each node
    print("\n\n=== Values Mode (full state snapshots) ===")
    async for chunk in agent.astream(input_msg, stream_mode="values"):
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            print(f"[Values] {last_msg.type}: {last_msg.content[:50]}...")

    print("\n\n=== Streaming Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
