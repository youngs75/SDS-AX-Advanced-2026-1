"""
Memory example demonstrating short-term (checkpointer) and long-term (store) memory.

Shows how to combine InMemorySaver for conversation history with InMemoryStore for
persistent user data across sessions.

Asset: assets/02-langchain/memory-example/graph.py
Reference: references/02-langchain/memory.md
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
    "I'll remember that your favorite color is blue.",
    "Your favorite color is blue.",
    "Based on the stored preferences, your favorite color is blue."
])

from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore


@tool
async def remember_preference(key: str, value: str) -> str:
    """Save a user preference for future reference.

    Args:
        key: Preference key (e.g., 'favorite_color')
        value: Preference value (e.g., 'blue')
    """
    # Note: In production, access store via config context
    # For this example, we'll use the global store instance
    return f"I've saved your preference: {key} = {value}"


@tool
async def recall_preference(key: str) -> str:
    """Recall a previously saved user preference.

    Args:
        key: Preference key to retrieve
    """
    # Note: In production, access store via config context
    return f"Looking up preference for: {key}"


# Initialize memory components
store = InMemoryStore()
checkpointer = MemorySaver()

# Create agent with both memory systems
agent = create_agent(
    model=model,
    tools=[remember_preference, recall_preference],
    checkpointer=checkpointer,
    store=store,
)


async def demonstrate_short_term_memory():
    """Show conversation history within a session (checkpointer)."""
    print("=== Short-Term Memory (Conversation History) ===")
    config = {"configurable": {"thread_id": "conversation_1"}}

    # Turn 1
    result1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "My name is Alice"}]},
        config=config,
    )
    print(f"User: My name is Alice")
    print(f"Agent: {result1['messages'][-1].content}\n")

    # Turn 2 - Agent remembers from conversation history
    result2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config=config,
    )
    print(f"User: What's my name?")
    print(f"Agent: {result2['messages'][-1].content}\n")


async def demonstrate_long_term_memory():
    """Show persistent data across sessions (store)."""
    print("=== Long-Term Memory (Persistent Store) ===")

    # Session 1: Save preference
    config1 = {"configurable": {"thread_id": "session_1", "user_id": "alice"}}

    # Manually store in the long-term store
    user_id = "alice"
    store.put(("preferences", user_id), "favorite_color", {"value": "blue"})

    result1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Remember that my favorite color is blue"}]},
        config=config1,
    )
    print(f"[Session 1] User: Remember that my favorite color is blue")
    print(f"[Session 1] Agent: {result1['messages'][-1].content}\n")

    # Session 2: Different thread, but same user - can access stored data
    config2 = {"configurable": {"thread_id": "session_2", "user_id": "alice"}}

    # Retrieve from long-term store
    stored_pref = store.get(("preferences", user_id), "favorite_color")
    print(f"[Session 2] Stored preference: {stored_pref.value if stored_pref else 'None'}")

    result2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Do you know my favorite color?"}]},
        config=config2,
    )
    print(f"[Session 2] User: Do you know my favorite color?")
    print(f"[Session 2] Agent: {result2['messages'][-1].content}\n")


async def demonstrate_combined_memory():
    """Show both memory systems working together."""
    print("=== Combined Memory (Short-term + Long-term) ===")
    config = {"configurable": {"thread_id": "combined_session", "user_id": "bob"}}

    # Store long-term preference
    store.put(("preferences", "bob"), "theme", {"value": "dark_mode"})

    # Turn 1: Agent uses long-term memory
    result1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What's my theme preference?"}]},
        config=config,
    )
    print(f"Turn 1 - User: What's my theme preference?")

    theme_pref = store.get(("preferences", "bob"), "theme")
    if theme_pref:
        print(f"Turn 1 - Stored: theme = {theme_pref.value['value']}")
    print(f"Turn 1 - Agent: {result1['messages'][-1].content}\n")

    # Turn 2: Agent uses short-term memory (conversation context)
    result2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Can you repeat what you just told me?"}]},
        config=config,
    )
    print(f"Turn 2 - User: Can you repeat what you just told me?")
    print(f"Turn 2 - Agent: {result2['messages'][-1].content}\n")
    print("(Agent uses checkpointer to recall previous turn)")


async def main():
    """Run all memory demonstrations."""

    await demonstrate_short_term_memory()
    print("\n" + "="*60 + "\n")

    await demonstrate_long_term_memory()
    print("\n" + "="*60 + "\n")

    await demonstrate_combined_memory()

    print("\n=== Memory Examples Complete ===")
    print("\nKey Takeaways:")
    print("- Checkpointer (MemorySaver): Conversation history within a thread")
    print("- Store (InMemoryStore): Persistent data across sessions/threads")
    print("- Combined: Use both for context-aware, personalized interactions")


if __name__ == "__main__":
    asyncio.run(main())
