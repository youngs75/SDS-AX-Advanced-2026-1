"""
Subgraph example: nested graph composition with state transformation.

Demonstrates:
- Building a child graph with its own state schema
- Invoking child graph from a parent node (Pattern A: different schemas)
- Adding compiled child as a node (Pattern B: shared keys)
- State transformation between parent and child
"""

import asyncio
from typing import TypedDict, Annotated

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

# ChatOpenAI(model="gpt-4.1"): direct provider usage
# init_chat_model("openai:gpt-4.1"): provider-agnostic (recommended)
model = FakeListChatModel(responses=[
    "Research summary: LangGraph enables durable agent workflows.",
    "Edited draft: LangGraph provides durable, resumable agent workflows with built-in persistence.",
    "Final review: The document accurately describes LangGraph's capabilities.",
])


# ==== Child Graph: Research Agent (own schema) ====

class ResearchState(TypedDict):
    query: str
    findings: str

async def research_node(state: ResearchState) -> dict:
    """Child: Research a query."""
    response = await model.ainvoke([HumanMessage(content=f"Research: {state['query']}")])
    return {"findings": response.content}

# Build child graph
research_builder = StateGraph(ResearchState)
research_builder.add_node("research", research_node)
research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)
research_graph = research_builder.compile()


# ==== Child Graph: Editor Agent (shared messages key) ====

class EditorState(TypedDict):
    messages: Annotated[list, add_messages]

async def edit_node(state: EditorState) -> dict:
    """Child: Edit/improve the draft."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

# Build editor child graph
editor_builder = StateGraph(EditorState)
editor_builder.add_node("edit", edit_node)
editor_builder.add_edge(START, "edit")
editor_builder.add_edge("edit", END)
editor_graph = editor_builder.compile()


# ==== Parent Graph ====

class ParentState(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    research_findings: str


# Pattern A: Invoke child from node (different schemas — manual transform)
async def research_wrapper(state: ParentState) -> dict:
    """Parent node that invokes child graph with state transformation."""
    # Transform: parent state → child input
    child_input = {"query": state["topic"], "findings": ""}

    # Invoke child graph
    child_result = await research_graph.ainvoke(child_input)

    # Transform: child output → parent state update
    return {
        "research_findings": child_result["findings"],
        "messages": [HumanMessage(content=f"Research complete: {child_result['findings']}")],
    }


# Pattern B: Add compiled child as node (shared 'messages' key)
# editor_graph shares 'messages' key with parent — auto-pass-through


async def review_node(state: ParentState) -> dict:
    """Final review of the edited content."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


# ==== Assemble Parent Graph ====

parent_builder = StateGraph(ParentState)

# Pattern A: wrapper node invokes child with transform
parent_builder.add_node("research", research_wrapper)

# Pattern B: compiled child added directly (shared keys)
parent_builder.add_node("editor", editor_graph)

# Regular parent node
parent_builder.add_node("review", review_node)

parent_builder.add_edge(START, "research")
parent_builder.add_edge("research", "editor")
parent_builder.add_edge("editor", "review")
parent_builder.add_edge("review", END)

graph = parent_builder.compile()


# ==== Main ====

async def main():
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Write about LangGraph capabilities")],
        "topic": "LangGraph durable workflows",
        "research_findings": "",
    })

    print("=== Final State ===")
    print(f"Topic: {result['topic']}")
    print(f"Research: {result['research_findings']}")
    print(f"Messages ({len(result['messages'])}):")
    for msg in result["messages"]:
        role = type(msg).__name__.replace("Message", "")
        print(f"  [{role}] {msg.content[:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
