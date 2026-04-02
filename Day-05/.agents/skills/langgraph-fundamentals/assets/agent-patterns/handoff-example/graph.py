from __future__ import annotations

import asyncio
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# =============================================================================
# MODEL CONFIGURATION - Choose one option below
# =============================================================================

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models.fake_chat_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        "As a researcher, I've gathered comprehensive information on the topic. Key findings include: industry trends, best practices, and actionable insights that will form the foundation of our content.",
        "As a writer, I've crafted a compelling draft incorporating the research findings. The content flows naturally, maintains reader engagement, and clearly communicates the key points identified during research.",
        "As an editor, I've refined the draft for clarity, consistency, and impact. The final content is polished, error-free, and ready for publication with enhanced readability and professional tone.",
    ]
)

# Option 2: OpenAI ChatGPT (COMMENTED)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")

# Option 3: Generic init_chat_model (COMMENTED)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")

# =============================================================================


def merge_dict(left: dict, right: dict) -> dict:
    return {**left, **right}


class HandoffState(TypedDict):
    """State for handoff pattern."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: Literal["researcher", "writer", "editor", "FINISH"]
    context: Annotated[dict, merge_dict]
    current_agent: str


async def researcher_node(state: HandoffState) -> dict:
    """Researcher agent gathers information and key points."""
    # Invoke LLM to generate research insights
    response = await model.ainvoke(state["messages"])

    return {
        "messages": [HumanMessage(content=f"Researcher: {response.content}")],
        "context": {"research": response.content},
        "next_agent": "writer",
        "current_agent": "researcher",
    }


async def writer_node(state: HandoffState) -> dict:
    """Writer agent creates draft content based on research."""
    research = state["context"].get("research", "")

    # Create context-aware prompt
    messages = state["messages"] + [
        HumanMessage(content=f"Using the research: {research}, create a draft.")
    ]

    # Invoke LLM to generate draft
    response = await model.ainvoke(messages)

    return {
        "messages": [HumanMessage(content=f"Writer: {response.content}")],
        "context": {"draft": response.content},
        "next_agent": "editor",
        "current_agent": "writer",
    }


async def editor_node(state: HandoffState) -> dict:
    """Editor agent polishes and finalizes content."""
    draft = state["context"].get("draft", "")

    # Create context-aware prompt
    messages = state["messages"] + [HumanMessage(content=f"Polish this draft: {draft}")]

    # Invoke LLM to polish content
    response = await model.ainvoke(messages)

    return {
        "messages": [HumanMessage(content=f"Editor: {response.content}")],
        "context": {"final": response.content},
        "next_agent": "FINISH",
        "current_agent": "editor",
    }


def create_graph():
    graph = StateGraph(HandoffState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "researcher")

    graph.add_conditional_edges(
        "researcher",
        lambda s: s["next_agent"],
        {"writer": "writer", "FINISH": END},
    )
    graph.add_conditional_edges(
        "writer",
        lambda s: s["next_agent"],
        {"editor": "editor", "FINISH": END},
    )
    graph.add_conditional_edges(
        "editor",
        lambda s: s["next_agent"],
        {"FINISH": END},
    )

    return graph.compile()


graph = create_graph()


async def main():
    """Main async entry point."""
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Create a short summary on AI trends.")],
            "next_agent": "researcher",
            "context": {},
            "current_agent": "",
        }
    )

    print("\n" + "=" * 80)
    print("FINAL CONTEXT:")
    print("=" * 80)
    for key, value in result.get("context", {}).items():
        print(f"\n[{key.upper()}]")
        print(value)

    print("\n" + "=" * 80)
    print("MESSAGE FLOW:")
    print("=" * 80)
    for message in result["messages"]:
        print(f"- {message.content}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
