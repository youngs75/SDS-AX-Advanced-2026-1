"""
Testing example: pytest patterns for LangGraph graphs.

Demonstrates:
- Node-level unit testing with FakeListChatModel
- Graph-level integration testing
- Partial execution testing with interrupt_after
- Checkpoint state assertions
- pytest-asyncio setup
"""

import operator
import pytest
from typing import TypedDict, Annotated, Literal

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver


# ==== Graph Under Test ====

class TicketState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str
    resolved: bool


async def classify_node(state: TicketState, model) -> dict:
    """Classify a support ticket."""
    response = await model.ainvoke(state["messages"])
    category = "technical" if "technical" in response.content.lower() else "general"
    return {"category": category}


async def resolve_node(state: TicketState, model) -> dict:
    """Resolve the ticket."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [response], "resolved": True}


def route_ticket(state: TicketState) -> Literal["resolve", "__end__"]:
    if state["category"]:
        return "resolve"
    return "__end__"


def build_graph(model):
    """Build graph with injectable model for testing."""

    async def _classify(state):
        return await classify_node(state, model)

    async def _resolve(state):
        return await resolve_node(state, model)

    builder = StateGraph(TicketState)
    builder.add_node("classify", _classify)
    builder.add_node("resolve", _resolve)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_ticket)
    builder.add_edge("resolve", END)
    return builder.compile(checkpointer=InMemorySaver())


# ==== Test 1: Node-Level Unit Test ====

@pytest.mark.asyncio
async def test_classify_node_technical():
    """Test classify node identifies technical tickets."""
    model = FakeListChatModel(responses=["This is a technical issue"])
    state = {
        "messages": [HumanMessage(content="Server is down")],
        "category": "",
        "resolved": False,
    }
    result = await classify_node(state, model)
    assert result["category"] == "technical"


@pytest.mark.asyncio
async def test_classify_node_general():
    """Test classify node identifies general tickets."""
    model = FakeListChatModel(responses=["This is a general inquiry"])
    state = {
        "messages": [HumanMessage(content="What are your hours?")],
        "category": "",
        "resolved": False,
    }
    result = await classify_node(state, model)
    assert result["category"] == "general"


# ==== Test 2: Graph-Level Integration Test ====

@pytest.mark.asyncio
async def test_full_flow_technical():
    """Test complete flow for technical ticket."""
    model = FakeListChatModel(responses=[
        "This is a technical issue",
        "Please restart the server to fix the issue.",
    ])
    graph = build_graph(model)
    config = {"configurable": {"thread_id": "test_tech_1"}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Server crashed")], "category": "", "resolved": False},
        config=config,
    )

    assert result["category"] == "technical"
    assert result["resolved"] is True
    assert len(result["messages"]) >= 2


# ==== Test 3: Partial Execution with Interrupt ====

@pytest.mark.asyncio
async def test_partial_execution():
    """Test running graph up to a specific point."""
    model = FakeListChatModel(responses=["This is a technical issue"])

    builder = StateGraph(TicketState)

    async def _classify(state):
        return await classify_node(state, model)

    async def _resolve(state):
        return await resolve_node(state, model)

    builder.add_node("classify", _classify)
    builder.add_node("resolve", _resolve)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_ticket)
    builder.add_edge("resolve", END)

    graph = builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_after=["classify"],  # pause after classification
    )
    config = {"configurable": {"thread_id": "test_partial_1"}}

    # Run until interrupt
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Server down")], "category": "", "resolved": False},
        config=config,
    )

    # Verify partial state
    assert result["category"] == "technical"
    assert result["resolved"] is False  # not yet resolved


# ==== Test 4: Checkpoint State Assertion ====

@pytest.mark.asyncio
async def test_state_progression():
    """Verify state changes across supersteps."""
    model = FakeListChatModel(responses=[
        "This is a technical issue",
        "Issue resolved: restart completed.",
    ])
    graph = build_graph(model)
    config = {"configurable": {"thread_id": "test_history_1"}}

    await graph.ainvoke(
        {"messages": [HumanMessage(content="Fix server")], "category": "", "resolved": False},
        config=config,
    )

    # Browse checkpoint history
    states = []
    async for snapshot in graph.aget_state_history(config):
        states.append(snapshot.values)

    # Newest first: final state should be resolved
    assert states[0]["resolved"] is True
    assert states[0]["category"] == "technical"
