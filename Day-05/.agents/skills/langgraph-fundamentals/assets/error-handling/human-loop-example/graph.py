"""LangGraph human-in-the-loop approval example.

Demonstrates:
- interrupt() for human approval before sensitive operations
- Command(resume=...) for approve/reject flow
- Checkpointed pause/resume with thread_id
"""

from __future__ import annotations

import asyncio
from typing import Literal
from typing_extensions import NotRequired

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


# --- Model Configuration ---
# Choose one of these options:

# Option 1 (active): Fake model for testing
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        "I will delete_records from the users table where inactive > 90 days",
        "I will generate_report about account-cleanup",
    ]
)

# Option 2: OpenAI via ChatOpenAI
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")

# Option 3: OpenAI via init_chat_model
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")


# --- State ---
class State(MessagesState):
    action: NotRequired[str]
    action_args: NotRequired[dict]
    result: NotRequired[str]


# --- Nodes ---
async def plan_action(state: State):
    """Agent plans an action based on user request using LLM."""
    user_msg = str(state["messages"][-1].content) if state["messages"] else ""

    # Use LLM to plan the action
    planning_prompt = f"""Based on the user request: "{user_msg}"

Determine what action to take. Respond with the action name and parameters.
Common actions:
- delete_records: Remove data from database
- generate_report: Create a report
- update_settings: Modify configuration
"""

    response = await model.ainvoke([HumanMessage(content=planning_prompt)])
    response_text = response.content.lower()

    # Parse LLM response to extract action
    if "delete_records" in response_text or "delete" in response_text:
        return {
            "action": "delete_records",
            "action_args": {"table": "users", "filter": "inactive > 90 days"},
        }
    elif "generate_report" in response_text or "report" in response_text:
        return {
            "action": "generate_report",
            "action_args": {"topic": "account-cleanup"},
        }
    else:
        return {
            "action": "general_task",
            "action_args": {"description": user_msg},
        }


async def human_review(state: State) -> Command[Literal["execute", "cancel"]]:
    """Pause for human approval before executing sensitive action."""
    is_approved = interrupt(
        {
            "question": "Do you want to proceed with this action?",
            "action": state["action"],
            "args": state["action_args"],
        }
    )

    if is_approved:
        return Command(goto="execute")
    else:
        return Command(goto="cancel")


async def execute(state: State):
    """Execute the approved action."""
    action = state["action"]
    args = state["action_args"]
    # Simulate execution
    return {
        "result": f"Executed {action} with {args}",
        "messages": [AIMessage(content=f"Action completed: {action}")],
    }


async def cancel(state: State):
    """Handle rejected action."""
    return {
        "result": "cancelled",
        "messages": [AIMessage(content="Action was rejected by reviewer.")],
    }


# --- Graph ---
builder = StateGraph(State)

builder.add_node("plan_action", plan_action)
builder.add_node("human_review", human_review)
builder.add_node("execute", execute)
builder.add_node("cancel", cancel)

builder.add_edge(START, "plan_action")
builder.add_edge("plan_action", "human_review")
# human_review uses Command for routing
builder.add_edge("execute", END)
builder.add_edge("cancel", END)

# A checkpointer is required for interrupt() pause/resume flows.
graph = builder.compile(checkpointer=InMemorySaver())


# --- Usage ---
async def main():
    """Async main function demonstrating the human-in-the-loop workflow."""
    config = {"configurable": {"thread_id": "review-1"}}

    # Step 1: Run until interrupt
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="Clean up inactive users")]},
        config,
    )
    print("Interrupted:", result.get("__interrupt__"))

    # Step 2: Resume with approval
    result = await graph.ainvoke(Command(resume=True), config)
    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
