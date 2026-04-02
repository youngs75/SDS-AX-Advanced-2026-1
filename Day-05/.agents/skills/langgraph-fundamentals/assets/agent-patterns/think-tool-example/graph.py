from __future__ import annotations

import asyncio
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose one of the following options:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        # Turn 1: Agent uses think_tool for reflection (simulated tool_calls)
        "Let me think about this systematically before searching.",
        # Turn 2: Agent provides final answer after reflection
        "Based on my analysis, the key benefits of async programming in Python are:\n\n"
        "1. **Improved I/O performance** — async handles thousands of concurrent connections\n"
        "2. **Resource efficiency** — single-threaded, no GIL contention for I/O\n"
        "3. **Scalability** — event loop scales better than thread-per-connection\n\n"
        "The main trade-off is increased code complexity with async/await syntax.",
    ]
)

# Option 2: Any provider via init_chat_model (COMMENTED)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")
# NOTE: Use model.bind_tools([think_tool, ...other_tools]) to include think_tool

# Option 3: OpenAI via ChatOpenAI (COMMENTED)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")


# ============================================================================
# THINK TOOL DEFINITION
# ============================================================================

# NOTE: Unlike native "extended thinking" (e.g., Anthropic's thinking blocks or
# OpenAI's reasoning tokens), think_tool is a PROVIDER-AGNOSTIC pattern.
# It works with any LLM that supports tool calling, giving you explicit control
# over when and how the model reflects on its progress.


@tool(description="Strategic reflection tool for research planning and decision-making")
def think_tool(reflection: str) -> str:
    """Tool for structured deliberation — no external execution, just records reasoning.

    This creates a deliberate pause in the agent workflow for quality decision-making.
    The reflection is recorded as a ToolMessage in the conversation history, making
    the agent's reasoning process transparent and traceable.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing gaps: What specific information am I still missing?
    - Before concluding: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings — what concrete information was gathered?
    2. Gap assessment — what crucial information is still missing?
    3. Quality evaluation — sufficient evidence/examples for a good answer?
    4. Strategic decision — continue searching or provide the answer?

    Args:
        reflection: Detailed reflection on progress, findings, gaps, and next steps.

    Returns:
        Confirmation that the reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"


# ============================================================================
# PROMPT WITH THINK TOOL INSTRUCTIONS
# ============================================================================

SYSTEM_PROMPT = """You are a research assistant that thinks carefully before acting.

<Available Tools>
1. **think_tool**: For strategic reflection and planning during research
2. (Other search/action tools would be listed here)

**CRITICAL: Use think_tool after each search to reflect on results and plan next steps.
Do NOT call think_tool in parallel with other tools — it should be a deliberate pause.**
</Available Tools>

<Show Your Thinking>
After each action, use think_tool to analyze:
- What key information did I find?
- What's missing?
- Do I have enough to answer the question comprehensively?
- Should I search more or provide my answer?
</Show Your Thinking>
"""


# ============================================================================
# STATE DEFINITION
# ============================================================================


class ThinkToolState(TypedDict):
    """State for the think-tool agent loop."""

    messages: Annotated[list[BaseMessage], add_messages]
    tool_call_iterations: int


# ============================================================================
# NODE FUNCTIONS (ALL ASYNC)
# ============================================================================

MAX_TOOL_ITERATIONS = 5


async def agent_node(
    state: ThinkToolState,
) -> Command[Literal["tool_execution", "__end__"]]:
    """Agent node that decides whether to use tools or provide a final answer.

    In production with real models:
        model_with_tools = model.bind_tools([think_tool, search_tool, ...])
        response = await model_with_tools.ainvoke(messages)
    """
    messages = state["messages"]
    iterations = state.get("tool_call_iterations", 0)

    # Simulate model response
    response = await model.ainvoke(messages)

    # Simulate think_tool call on first iteration
    if iterations == 0:
        response = AIMessage(
            content="Let me reflect on the question first.",
            tool_calls=[
                {
                    "id": "think_1",
                    "name": "think_tool",
                    "args": {
                        "reflection": (
                            "The user asks about async programming benefits in Python. "
                            "I should consider: 1) I/O performance gains, 2) resource efficiency vs threads, "
                            "3) scalability characteristics, 4) trade-offs and limitations. "
                            "I have enough domain knowledge to answer comprehensively without searching."
                        )
                    },
                }
            ],
        )
    else:
        # No tool calls → final answer
        response = AIMessage(content=response.content)

    # Route based on whether tools were called
    has_tool_calls = bool(getattr(response, "tool_calls", None))
    next_node = "tool_execution" if has_tool_calls else END

    return Command(
        goto=next_node,
        update={
            "messages": [response],
            "tool_call_iterations": iterations + 1,
        },
    )


async def tool_execution_node(state: ThinkToolState) -> Command[Literal["agent_node"]]:
    """Execute tools, handling think_tool specially (no external call needed).

    For think_tool:
    - Extract reflection from args
    - Create ToolMessage with reflection content directly
    - No external API call or side effect

    For other tools:
    - Execute normally via tool.ainvoke(args)
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []

    for tc in getattr(last_message, "tool_calls", []):
        if tc["name"] == "think_tool":
            # think_tool: record reflection inline — NO external execution
            reflection = tc["args"]["reflection"]
            tool_messages.append(
                ToolMessage(
                    content=f"Reflection recorded: {reflection}",
                    name="think_tool",
                    tool_call_id=tc["id"],
                )
            )
        else:
            # Other tools: execute normally
            # result = await tools_by_name[tc["name"]].ainvoke(tc["args"])
            tool_messages.append(
                ToolMessage(
                    content="Tool result placeholder",
                    name=tc["name"],
                    tool_call_id=tc["id"],
                )
            )

    return Command(
        goto="agent_node",
        update={"messages": tool_messages},
    )


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_graph():
    """Create the think-tool agent loop.

    Architecture: agent_node ↔ tool_execution (loop until no tool calls)
    The think_tool creates structured deliberation pauses without external calls.
    """
    graph = StateGraph(ThinkToolState)

    graph.add_node("agent_node", agent_node)
    graph.add_node("tool_execution", tool_execution_node)

    graph.add_edge(START, "agent_node")
    # agent_node → tool_execution or END via Command
    # tool_execution → agent_node via Command

    return graph.compile()


graph = create_graph()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Run the think-tool agent loop demonstration."""
    result = await graph.ainvoke(
        {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content="What are the key benefits of async programming in Python?"
                ),
            ],
            "tool_call_iterations": 0,
        }
    )

    print("\n" + "=" * 80)
    print("THINK-TOOL PATTERN COMPLETE")
    print("=" * 80)
    print(f"\nTotal iterations: {result['tool_call_iterations']}")
    print("\nConversation:")
    for i, msg in enumerate(result["messages"], 1):
        role = type(msg).__name__
        content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
        print(f"  [{i}] {role}: {content}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
