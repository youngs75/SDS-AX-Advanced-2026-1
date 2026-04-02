"""
ToolRuntime Injection Demonstration

Demonstrates how to access graph state, store, context, and stream_writer from within tools
using the ToolRuntime injection pattern.

Reference: references/02-langchain/35-tools.md
"""

import asyncio
from typing import Literal

from langchain_core.language_models import FakeListChatModel
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command


# ============================================================================
# TOOLS WITH RUNTIME INJECTION
# ============================================================================


@tool
async def get_conversation_summary(runtime: ToolRuntime) -> str:
    """
    Get a summary of the current conversation by accessing the graph state.

    Demonstrates:
    - runtime.state: read current conversation state
    - Accessing messages list length
    """
    state = runtime.state
    messages = state.get("messages", [])
    message_count = len(messages)

    # Access context for user identification
    context = runtime.context
    user_id = context.get("configurable", {}).get("user_id", "unknown")

    return f"Conversation has {message_count} messages. User: {user_id}"


@tool
async def save_user_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime,
) -> str:
    """
    Save a user preference to long-term store (persists across sessions).

    Demonstrates:
    - runtime.store: write to persistent cross-session storage
    - runtime.context: extract user_id from config

    Args:
        preference_key: The preference identifier (e.g., "theme", "language")
        preference_value: The preference value
    """
    # Extract user_id from context
    context = runtime.context
    user_id = context.get("configurable", {}).get("user_id", "default_user")

    # Create namespace for user preferences
    namespace = ("preferences", user_id)

    # Save to store
    await runtime.store.aput(
        namespace=namespace,
        key=preference_key,
        value={"preference": preference_value}
    )

    return f"Saved preference '{preference_key}' = '{preference_value}' for user {user_id}"


@tool
async def get_user_preference(
    preference_key: str,
    runtime: ToolRuntime,
) -> str:
    """
    Retrieve a user preference from long-term store.

    Demonstrates:
    - runtime.store: read from persistent storage
    - runtime.context: extract user_id

    Args:
        preference_key: The preference identifier to retrieve
    """
    context = runtime.context
    user_id = context.get("configurable", {}).get("user_id", "default_user")

    namespace = ("preferences", user_id)

    # Retrieve from store
    item = await runtime.store.aget(namespace=namespace, key=preference_key)

    if item is None:
        return f"No preference found for '{preference_key}'"

    preference_value = item.value.get("preference", "N/A")
    return f"User {user_id} preference '{preference_key}' = '{preference_value}'"


@tool
async def update_conversation_mode(
    mode: Literal["casual", "formal", "technical"],
    runtime: ToolRuntime,
) -> Command:
    """
    Update the conversation mode by mutating graph state.

    Demonstrates:
    - runtime.state: read current state
    - Command(update={...}): return state mutation
    - runtime.stream_writer: emit progress events

    Args:
        mode: The desired conversation mode
    """
    # Emit progress event
    await runtime.stream_writer(
        {"type": "mode_change", "from": runtime.state.get("mode"), "to": mode}
    )

    # Return Command to update state
    return Command(
        update={"mode": mode},
        goto=None,  # Continue normal flow
    )


@tool
async def long_running_analysis(
    query: str,
    runtime: ToolRuntime,
) -> str:
    """
    Simulate a long-running analysis with progress streaming.

    Demonstrates:
    - runtime.stream_writer: emit multiple progress events
    - runtime.config: access RunnableConfig

    Args:
        query: The analysis query
    """
    # Access config for additional metadata
    config = runtime.config
    run_id = config.get("run_id", "unknown")

    # Emit progress events during execution
    await runtime.stream_writer({"type": "progress", "step": "starting", "query": query})
    await asyncio.sleep(0.1)  # Simulate work

    await runtime.stream_writer({"type": "progress", "step": "processing", "query": query})
    await asyncio.sleep(0.1)  # Simulate work

    await runtime.stream_writer({"type": "progress", "step": "finalizing", "query": query})
    await asyncio.sleep(0.1)  # Simulate work

    result = f"Analysis complete for: '{query}' (run_id: {run_id})"

    await runtime.stream_writer({"type": "progress", "step": "done", "result": result})

    return result


# ============================================================================
# AGENT SETUP
# ============================================================================


def create_runtime_demo_agent():
    """
    Create an agent with ToolRuntime-enabled tools.

    Uses:
    - FakeListChatModel for predictable tool calling
    - InMemorySaver for checkpointing
    - InMemoryStore for cross-session storage
    """
    # Use FakeListChatModel with tool call responses
    model = FakeListChatModel(
        responses=[
            # First call: get conversation summary
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "get_conversation_summary",
                        "args": {},
                    }
                ],
            },
            # Second call: save preference
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "name": "save_user_preference",
                        "args": {
                            "preference_key": "theme",
                            "preference_value": "dark",
                        },
                    }
                ],
            },
            # Third call: get preference
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_3",
                        "name": "get_user_preference",
                        "args": {"preference_key": "theme"},
                    }
                ],
            },
            # Final response
            {"role": "assistant", "content": "All runtime features demonstrated!"},
        ]
    )

    # Real model example (commented out):
    # model = ChatOpenAI(model="gpt-4", temperature=0)
    # Or:
    # model = init_chat_model(model="gpt-4", temperature=0)

    tools = [
        get_conversation_summary,
        save_user_preference,
        get_user_preference,
        update_conversation_mode,
        long_running_analysis,
    ]

    checkpointer = InMemorySaver()
    store = InMemoryStore()

    # Create agent with modern API
    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        store=store,
        state_schema=AgentState,
    )

    return agent, store


# ============================================================================
# DEMONSTRATION
# ============================================================================


async def main():
    """
    Comprehensive demonstration of ToolRuntime injection patterns.
    """
    print("=" * 80)
    print("ToolRuntime Injection Demonstration")
    print("=" * 80)

    agent, store = create_runtime_demo_agent()

    # -------------------------------------------------------------------
    # Demo 1: State Access
    # -------------------------------------------------------------------
    print("\n[Demo 1] Accessing graph state from tool")
    print("-" * 80)

    config = {
        "configurable": {
            "thread_id": "demo-thread-1",
            "user_id": "alice",
        }
    }

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Show me the conversation summary"}]},
        config=config,
    )

    print(f"Tool accessed state: {result['messages'][-1].content}")

    # -------------------------------------------------------------------
    # Demo 2: Store Write (Long-term Memory)
    # -------------------------------------------------------------------
    print("\n[Demo 2] Writing to persistent store")
    print("-" * 80)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Save my theme preference as dark"}]},
        config=config,
    )

    print(f"Tool saved to store: {result['messages'][-2].content}")

    # -------------------------------------------------------------------
    # Demo 3: Store Read (Cross-session Persistence)
    # -------------------------------------------------------------------
    print("\n[Demo 3] Reading from persistent store")
    print("-" * 80)

    # New thread, same user - preference should persist
    config_new_session = {
        "configurable": {
            "thread_id": "demo-thread-2",
            "user_id": "alice",
        }
    }

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's my theme preference?"}]},
        config=config_new_session,
    )

    print(f"Tool retrieved from store: {result['messages'][-2].content}")

    # -------------------------------------------------------------------
    # Demo 4: Stream Writer (Progress Events)
    # -------------------------------------------------------------------
    print("\n[Demo 4] Streaming progress events from tool")
    print("-" * 80)

    # Use stream to capture intermediate events
    events = []
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": "Analyze user behavior patterns"}]},
        config=config,
        stream_mode="updates",
    ):
        events.append(chunk)
        print(f"Event: {chunk}")

    print(f"\nTotal events captured: {len(events)}")

    # -------------------------------------------------------------------
    # Demo 5: State Mutation via Command
    # -------------------------------------------------------------------
    print("\n[Demo 5] Mutating state from tool via Command")
    print("-" * 80)

    # Create new agent instance for mode change demo
    model_command = FakeListChatModel(
        responses=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_mode",
                        "name": "update_conversation_mode",
                        "args": {"mode": "technical"},
                    }
                ],
            },
            {"role": "assistant", "content": "Mode updated successfully!"},
        ]
    )

    agent_command = create_agent(
        model=model_command,
        tools=[update_conversation_mode],
        checkpointer=InMemorySaver(),
        state_schema=AgentState,
    )

    result = await agent_command.ainvoke(
        {"messages": [{"role": "user", "content": "Switch to technical mode"}]},
        config={"configurable": {"thread_id": "demo-thread-3"}},
    )

    # Check if state was updated
    state = await agent_command.aget_state(
        {"configurable": {"thread_id": "demo-thread-3"}}
    )
    print(f"State after Command: mode = {state.values.get('mode', 'not set')}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ToolRuntime Components Demonstrated:")
    print("=" * 80)
    print("✓ runtime.state      - Read current conversation state")
    print("✓ runtime.context    - Access immutable config (user_id, session metadata)")
    print("✓ runtime.store      - Read/write long-term memory (cross-session)")
    print("✓ runtime.stream_writer - Emit progress events from tools")
    print("✓ runtime.config     - Access RunnableConfig")
    print("✓ Command(update={}) - Mutate state from tool return")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
