"""
Persistence example: Store with namespaces for cross-thread memory.

Demonstrates:
- InMemoryStore for cross-session key-value storage
- Namespace organization: ("users", user_id)
- Store access from graph nodes
- Combining checkpointer (short-term) with store (long-term)
"""

import asyncio
from typing import TypedDict, Annotated
from typing_extensions import NotRequired

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# ChatOpenAI(model="gpt-4.1"): direct provider usage
# init_chat_model("openai:gpt-4.1"): provider-agnostic (recommended)
model = FakeListChatModel(
    responses=[
        "I've noted your preference for Python. I'll keep that in mind!",
        "Based on your stored preference for Python, I recommend using LangGraph with Python.",
    ]
)


# ==== State ====


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    user_preferences: NotRequired[dict]


# ==== Nodes ====


async def load_user_context(state: ChatState, *, store) -> dict:
    """Load user preferences from long-term store."""
    user_id = state["user_id"]

    # Read from store namespace
    prefs_item = await store.aget(("users", user_id), "preferences")
    if prefs_item:
        return {"user_preferences": prefs_item.value}
    return {"user_preferences": {}}


async def chat_node(state: ChatState) -> dict:
    """Generate response using model with user context."""
    context = ""
    if state.get("user_preferences"):
        context = f"User preferences: {state['user_preferences']}\n"

    messages = state["messages"]
    if context:
        from langchain_core.messages import SystemMessage

        messages = [SystemMessage(content=context)] + list(messages)

    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def save_user_context(state: ChatState, *, store) -> dict:
    """Extract and save user preferences to long-term store."""
    user_id = state["user_id"]
    last_message = state["messages"][-2].content if len(state["messages"]) >= 2 else ""

    # Simple preference extraction (production would use LLM)
    if "prefer" in last_message.lower() or "favorite" in last_message.lower():
        current_prefs = state.get("user_preferences", {})
        current_prefs["mentioned_preference"] = last_message

        # Save to store — persists across threads/sessions
        await store.aput(("users", user_id), "preferences", current_prefs)

    return {}


# ==== Graph ====

builder = StateGraph(ChatState)
builder.add_node("load_context", load_user_context)
builder.add_node("chat", chat_node)
builder.add_node("save_context", save_user_context)

builder.add_edge(START, "load_context")
builder.add_edge("load_context", "chat")
builder.add_edge("chat", "save_context")
builder.add_edge("save_context", END)

# Combine checkpointer (short-term) + store (long-term)
checkpointer = InMemorySaver()
store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)


# ==== Main ====


async def main():
    user_id = "user_alice"

    # Session 1: User mentions a preference
    config1 = {"configurable": {"thread_id": "session_1"}}
    result1 = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="I prefer Python for AI development")],
            "user_id": user_id,
        },
        config=config1,
    )
    print(f"Session 1: {result1['messages'][-1].content}")
    print()

    # Session 2: Different thread, but store remembers preferences
    config2 = {"configurable": {"thread_id": "session_2"}}
    result2 = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="What language should I use for LangGraph?")
            ],
            "user_id": user_id,
        },
        config=config2,
    )
    print(f"Session 2: {result2['messages'][-1].content}")
    print(f"Loaded preferences: {result2.get('user_preferences', {})}")


if __name__ == "__main__":
    asyncio.run(main())
