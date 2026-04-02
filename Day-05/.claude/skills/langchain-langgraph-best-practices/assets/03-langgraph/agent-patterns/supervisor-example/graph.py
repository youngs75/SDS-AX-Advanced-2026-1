from __future__ import annotations

import asyncio
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose one of the following options:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        # Supervisor routing decisions
        '{"next": "researcher"}',
        '{"next": "writer"}',
        '{"next": "reviewer"}',
        '{"next": "FINISH"}',
        # Agent outputs
        "Researcher: I've gathered comprehensive information on the topic including key facts, statistics, and relevant sources.",
        "Writer: I've drafted a well-structured response incorporating the research findings with clear explanations and examples.",
        "Reviewer: I've reviewed the content for accuracy, clarity, and completeness. The response meets quality standards.",
    ]
)

# Option 2: OpenAI via ChatOpenAI (COMMENTED)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")
# NOTE: Real models like ChatOpenAI support .with_structured_output() for type-safe routing.
#       Uncomment the supervisor_llm line in supervisor_node() when using real models.

# Option 3: OpenAI via init_chat_model (COMMENTED)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")
# NOTE: Real models like ChatOpenAI support .with_structured_output() for type-safe routing.
#       Uncomment the supervisor_llm line in supervisor_node() when using real models.

# ============================================================================
# STATE DEFINITION
# ============================================================================


class SupervisorState(TypedDict):
    """State for supervisor pattern."""

    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["researcher", "writer", "reviewer", "FINISH"]
    current_agent: str


class SupervisorDecision(TypedDict):
    """Structured output for supervisor routing decisions."""

    next: Literal["researcher", "writer", "reviewer", "FINISH"]


# ============================================================================
# NODE FUNCTIONS (ALL ASYNC)
# ============================================================================


async def supervisor_node(state: SupervisorState) -> dict:
    """Route to the next agent using LLM-driven decision making."""
    import json

    messages = state["messages"]

    # Build system message for supervisor
    system_prompt = """You are a supervisor managing a team of agents: researcher, writer, and reviewer.

Your job is to route the conversation to the appropriate agent based on the current state:
- Route to 'researcher' if information needs to be gathered
- Route to 'writer' if content needs to be drafted
- Route to 'reviewer' if content needs to be reviewed
- Route to 'FINISH' when all work is complete and the task is done

Analyze the conversation history and decide which agent should work next.
Return ONLY a JSON object with a 'next' field containing one of: researcher, writer, reviewer, or FINISH.
Example: {"next": "researcher"}"""

    # Prepare messages for LLM
    llm_messages = [HumanMessage(content=system_prompt)] + messages

    # Get routing decision from LLM
    # NOTE: For real models (ChatOpenAI, etc.), use with_structured_output for type safety:
    # supervisor_llm = model.with_structured_output(SupervisorDecision)
    # decision = await supervisor_llm.ainvoke(llm_messages)
    # return {"next": decision["next"], "current_agent": "supervisor"}

    response = await model.ainvoke(llm_messages)

    # Parse the JSON response (fallback for FakeListChatModel)
    content = response.content if hasattr(response, "content") else str(response)
    try:
        decision = json.loads(content)
        next_agent = decision.get("next", "researcher")
    except (json.JSONDecodeError, AttributeError):
        # Fallback to researcher if parsing fails
        next_agent = "researcher"

    return {
        "next": next_agent,
        "current_agent": "supervisor",
    }


async def researcher_node(state: SupervisorState) -> dict:
    """Researcher subagent - gathers information."""
    messages = state["messages"]

    # Add role context for the researcher
    researcher_prompt = HumanMessage(
        content="You are a researcher. Analyze the request and gather relevant information, facts, and sources."
    )
    llm_messages = messages + [researcher_prompt]

    # Get LLM response
    response = await model.ainvoke(llm_messages)

    return {
        "messages": [response],
        "current_agent": "researcher",
    }


async def writer_node(state: SupervisorState) -> dict:
    """Writer subagent - drafts content."""
    messages = state["messages"]

    # Add role context for the writer
    writer_prompt = HumanMessage(
        content="You are a writer. Based on the research, draft a clear and well-structured response."
    )
    llm_messages = messages + [writer_prompt]

    # Get LLM response
    response = await model.ainvoke(llm_messages)

    return {
        "messages": [response],
        "current_agent": "writer",
    }


async def reviewer_node(state: SupervisorState) -> dict:
    """Reviewer subagent - reviews and validates content."""
    messages = state["messages"]

    # Add role context for the reviewer
    reviewer_prompt = HumanMessage(
        content="You are a reviewer. Check the drafted content for accuracy, clarity, and completeness. Provide feedback or approval."
    )
    llm_messages = messages + [reviewer_prompt]

    # Get LLM response
    response = await model.ainvoke(llm_messages)

    return {
        "messages": [response],
        "current_agent": "reviewer",
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_graph():
    """Create the supervisor pattern graph."""
    graph = StateGraph(SupervisorState)

    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Start with supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor routes to agents or END
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            "FINISH": END,
        },
    )

    # All agents loop back to supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


graph = create_graph()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Main async execution function."""
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Please research the benefits of async programming in Python, write a summary, and review it."
                )
            ],
            "next": "researcher",
            "current_agent": "",
        }
    )

    print("\n" + "=" * 80)
    print("SUPERVISOR PATTERN EXECUTION COMPLETE")
    print("=" * 80)
    print("\nFinal conversation history:")
    for i, message in enumerate(result["messages"], 1):
        print(f"\n[{i}] {message.content}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
