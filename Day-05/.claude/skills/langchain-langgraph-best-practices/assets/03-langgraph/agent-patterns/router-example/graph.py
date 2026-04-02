from __future__ import annotations

import asyncio
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Choose one of the following model configurations:

# Option 1 (ACTIVE): FakeListChatModel for testing
model = FakeListChatModel(
    responses=[
        "sales",  # Router responses
        "Here's our pricing information.",  # Sales agent
        "Let me help you troubleshoot that.",  # Support agent
        "Here's your invoice details.",  # Billing agent
    ]
)

# Option 2 (commented): OpenAI GPT-4
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")

# Option 3 (commented): Generic chat model initialization
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")


# ============================================================
# STRUCTURED OUTPUT FOR ROUTING
# ============================================================
class RouteDecision(BaseModel):
    """Structured output for routing decisions."""

    route: Literal["sales", "support", "billing"] = Field(
        description="The department to route the customer query to"
    )


# ============================================================
# STATE DEFINITION
# ============================================================
class RouterState(TypedDict):
    """State for router pattern."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["sales", "support", "billing"]


# ============================================================
# NODE FUNCTIONS (ALL ASYNC)
# ============================================================
async def router_node(state: RouterState) -> dict:
    """Route customer queries using LLM with structured output."""
    query = state["messages"][-1].content

    # Prepare routing prompt
    routing_prompt = f"""You are a customer service router. Analyze the customer query and route it to the appropriate department.

Customer query: {query}

Departments:
- sales: For pricing, purchasing, product information
- support: For technical issues, troubleshooting, help with using products
- billing: For invoices, payments, account billing questions

Return the appropriate department."""

    # Get routing decision from LLM
    # For FakeListChatModel: uses first response directly
    # For real models: use with_structured_output(RouteDecision)
    if isinstance(model, FakeListChatModel):
        # FakeListChatModel doesn't support structured output, so simulate it
        response = await model.ainvoke([HumanMessage(content=routing_prompt)])
        route = response.content.lower()  # Returns "sales", "support", or "billing"
    else:
        # For real models with structured output support
        router_model = model.with_structured_output(RouteDecision)
        decision = await router_model.ainvoke([HumanMessage(content=routing_prompt)])
        route = (
            decision.route if isinstance(decision, RouteDecision) else decision["route"]
        )

    return {"route": route}


async def sales_agent(state: RouterState) -> dict:
    """Handle sales-related queries."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [AIMessage(content=f"Sales: {response.content}")]}


async def support_agent(state: RouterState) -> dict:
    """Handle support-related queries."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [AIMessage(content=f"Support: {response.content}")]}


async def billing_agent(state: RouterState) -> dict:
    """Handle billing-related queries."""
    response = await model.ainvoke(state["messages"])
    return {"messages": [AIMessage(content=f"Billing: {response.content}")]}


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================
def create_graph():
    """Create and compile the router graph."""
    graph = StateGraph(RouterState)

    graph.add_node("router", router_node)
    graph.add_node("sales", sales_agent)
    graph.add_node("support", support_agent)
    graph.add_node("billing", billing_agent)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "sales": "sales",
            "support": "support",
            "billing": "billing",
        },
    )

    graph.add_edge("sales", END)
    graph.add_edge("support", END)
    graph.add_edge("billing", END)

    return graph.compile()


graph = create_graph()


# ============================================================
# MAIN EXECUTION
# ============================================================
async def main():
    """Main async entry point."""
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Can I see pricing?")],
            "route": "support",  # Initial route (will be overwritten by router)
        }
    )

    print("Final messages:")
    for message in result["messages"]:
        print(f"- {message.content}")

    print(f"\nRouted to: {result['route']}")


if __name__ == "__main__":
    asyncio.run(main())
