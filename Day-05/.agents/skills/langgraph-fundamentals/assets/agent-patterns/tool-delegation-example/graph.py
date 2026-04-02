from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose one of the following options:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

supervisor_model = FakeListChatModel(
    responses=[
        # First call: supervisor delegates research with tool calls
        # (Simulated - in production, model generates actual tool_calls)
        "I'll research both frontend frameworks and backend technologies in parallel.",
        # Second call: supervisor marks research complete
        "Research is complete. I have gathered comprehensive findings on both topics.",
    ]
)

researcher_model = FakeListChatModel(
    responses=[
        "## Research Findings\n\nBased on my investigation:\n\n"
        "1. **Key Finding 1**: Modern frameworks prioritize developer experience\n"
        "2. **Key Finding 2**: Performance benchmarks show significant improvements\n"
        "3. **Key Finding 3**: Community adoption drives ecosystem growth\n\n"
        "### Sources\n[1] Official Documentation: https://example.com/docs",
    ]
)

# Option 2: Hot-swappable model via init_chat_model (COMMENTED)
# from langchain.chat_models import init_chat_model
# configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))
# NOTE: Use model.bind_tools([ConductResearch, ResearchComplete, think_tool]) for supervisor

# Option 3: OpenAI via ChatOpenAI (COMMENTED)
# from langchain_openai import ChatOpenAI
# supervisor_model = ChatOpenAI(model="gpt-5")
# researcher_model = ChatOpenAI(model="gpt-4.1-mini")


# ============================================================================
# DELEGATION TOOLS (Pydantic BaseModel as tools)
# ============================================================================


class ConductResearch(BaseModel):
    """Call this tool to delegate a research task to a sub-researcher.

    The supervisor calls this tool to spawn a researcher subgraph for a specific topic.
    Multiple ConductResearch calls execute in parallel via asyncio.gather().
    """

    research_topic: str = Field(
        description="The topic to research. Should be described in high detail (at least a paragraph)."
    )


class ResearchComplete(BaseModel):
    """Call this tool to signal that all research is done.

    The supervisor calls this when satisfied with gathered findings.
    Triggers exit from the supervisor loop.
    """


# ============================================================================
# STATE DEFINITIONS
# ============================================================================


def override_reducer(current_value, new_value):
    """Reducer that supports both append and full replacement.

    Usage:
    - Normal: state["field"] = ["new_item"]  → appends via operator.add
    - Override: state["field"] = {"type": "override", "value": [...]}  → replaces entirely

    This allows nodes to either accumulate results (default) or reset state.
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class SupervisorState(TypedDict):
    """State for the supervisor managing research delegation."""

    supervisor_messages: Annotated[list[BaseMessage], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer]
    research_iterations: int


class ResearcherState(TypedDict):
    """Internal state for individual researchers."""

    researcher_messages: Annotated[list[BaseMessage], operator.add]
    research_topic: str
    compressed_research: str


class ResearcherOutputState(BaseModel):
    """Output projection — only expose compressed_research from subgraph.

    Passed as `output=ResearcherOutputState` to StateGraph() to restrict
    what the parent graph receives from the subgraph.
    """

    compressed_research: str


# ============================================================================
# RESEARCHER SUBGRAPH
# ============================================================================


async def researcher_node(state: ResearcherState) -> dict:
    """Individual researcher that conducts focused research on a specific topic."""
    messages = state.get("researcher_messages", [])
    topic = state.get("research_topic", "")

    system_prompt = SystemMessage(
        content=(
            "You are a research assistant. Conduct thorough research on the given topic. "
            "Provide comprehensive findings with specific facts, data, and source references."
        )
    )

    response = await researcher_model.ainvoke([system_prompt] + messages)
    return {"researcher_messages": [response], "compressed_research": response.content}


def create_researcher_subgraph():
    """Create researcher subgraph with output state projection.

    The output=ResearcherOutputState ensures only compressed_research
    is returned to the parent supervisor graph.
    """
    builder = StateGraph(ResearcherState, output=ResearcherOutputState)
    builder.add_node("researcher", researcher_node)
    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", END)
    return builder.compile()


researcher_subgraph = create_researcher_subgraph()


# ============================================================================
# SUPERVISOR NODES
# ============================================================================

MAX_CONCURRENT_RESEARCH = 3
MAX_SUPERVISOR_ITERATIONS = 4


async def supervisor_node(
    state: SupervisorState,
) -> Command[Literal["supervisor_tools"]]:
    """Supervisor that plans research strategy and delegates to sub-researchers.

    In production, use:
        model.bind_tools([ConductResearch, ResearchComplete, think_tool])
    to let the LLM decide when and how to delegate research.
    """
    messages = state.get("supervisor_messages", [])
    iterations = state.get("research_iterations", 0)

    # Simulate supervisor decision-making
    # In production: response = await model.bind_tools([...]).ainvoke(messages)
    response = await supervisor_model.ainvoke(messages)

    # Simulate tool calls for demonstration
    # In production, the model generates these automatically via bind_tools
    if iterations == 0:
        # First iteration: delegate parallel research tasks
        response = AIMessage(
            content="Delegating research on two parallel topics.",
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "ConductResearch",
                    "args": {
                        "research_topic": "Modern frontend frameworks: React, Vue, Svelte comparison"
                    },
                },
                {
                    "id": "call_2",
                    "name": "ConductResearch",
                    "args": {
                        "research_topic": "Backend technologies: Node.js, Python, Go for web services"
                    },
                },
            ],
        )
    else:
        # Subsequent iterations: signal completion
        response = AIMessage(
            content="Research complete.",
            tool_calls=[
                {"id": "call_done", "name": "ResearchComplete", "args": {}},
            ],
        )

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": iterations + 1,
        },
    )


async def supervisor_tools_node(
    state: SupervisorState,
) -> Command[Literal["supervisor_node", "__end__"]]:
    """Execute tools called by the supervisor.

    Handles three tool types:
    1. ConductResearch → spawns researcher subgraph (parallel via asyncio.gather)
    2. ResearchComplete → exits supervisor loop
    3. think_tool → records reflection inline (no external call)
    """
    messages = state.get("supervisor_messages", [])
    iterations = state.get("research_iterations", 0)
    last_message = messages[-1]

    # --- Exit conditions ---
    no_tool_calls = not getattr(last_message, "tool_calls", None)
    exceeded_iterations = iterations > MAX_SUPERVISOR_ITERATIONS
    research_complete = any(
        tc["name"] == "ResearchComplete"
        for tc in getattr(last_message, "tool_calls", [])
    )

    if no_tool_calls or exceeded_iterations or research_complete:
        # Collect all notes from tool messages and exit
        collected_notes = [
            msg.content for msg in messages if isinstance(msg, ToolMessage)
        ]
        return Command(
            goto=END,
            update={"notes": collected_notes},
        )

    # --- Process ConductResearch calls ---
    conduct_calls = [
        tc for tc in last_message.tool_calls if tc["name"] == "ConductResearch"
    ]

    # Enforce concurrency limit
    allowed_calls = conduct_calls[:MAX_CONCURRENT_RESEARCH]
    overflow_calls = conduct_calls[MAX_CONCURRENT_RESEARCH:]

    # Execute research tasks in parallel via asyncio.gather
    research_tasks = [
        researcher_subgraph.ainvoke(
            {
                "researcher_messages": [
                    HumanMessage(content=tc["args"]["research_topic"])
                ],
                "research_topic": tc["args"]["research_topic"],
            }
        )
        for tc in allowed_calls
    ]
    results = await asyncio.gather(*research_tasks)

    # Convert results to ToolMessages for supervisor context
    tool_messages = []
    for result, tc in zip(results, allowed_calls):
        tool_messages.append(
            ToolMessage(
                content=result.get("compressed_research", "No results"),
                name="ConductResearch",
                tool_call_id=tc["id"],
            )
        )

    # Add error messages for overflow calls
    for tc in overflow_calls:
        tool_messages.append(
            ToolMessage(
                content=f"Error: Exceeded max concurrent research units ({MAX_CONCURRENT_RESEARCH}). Try fewer parallel tasks.",
                name="ConductResearch",
                tool_call_id=tc["id"],
            )
        )

    return Command(
        goto="supervisor_node",
        update={"supervisor_messages": tool_messages},
    )


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_graph():
    """Create the supervisor + researcher delegation graph.

    Architecture:
    - Supervisor subgraph: supervisor_node ↔ supervisor_tools_node (loop)
    - Researcher subgraph: researcher_node (single pass, compiled separately)
    - ConductResearch tool calls trigger parallel researcher_subgraph.ainvoke()
    """
    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("supervisor_tools", supervisor_tools_node)

    builder.add_edge(START, "supervisor_node")
    # supervisor_node → supervisor_tools via Command
    # supervisor_tools → supervisor_node or END via Command

    return builder.compile()


graph = create_graph()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Run the tool-delegation pattern demonstration."""
    result = await graph.ainvoke(
        {
            "supervisor_messages": [
                SystemMessage(
                    content="You are a research supervisor. Delegate research tasks."
                ),
                HumanMessage(
                    content="Compare modern web development frameworks and backend technologies."
                ),
            ],
            "research_brief": "Compare frontend and backend web technologies",
            "notes": [],
            "research_iterations": 0,
        }
    )

    print("\n" + "=" * 80)
    print("TOOL-DELEGATION PATTERN COMPLETE")
    print("=" * 80)
    print(f"\nIterations: {result['research_iterations']}")
    print(f"\nCollected Notes ({len(result['notes'])} entries):")
    for i, note in enumerate(result["notes"], 1):
        print(f"\n--- Note {i} ---")
        print(note[:200] + "..." if len(note) > 200 else note)
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
