"""
DeepAgents quickstart: create_deep_agent() basic setup.

Demonstrates:
- Minimal agent creation with model and custom tools
- @tool decorator with Pydantic schema
- Async invocation with thread_id
- Streaming output
- Debug mode toggle
"""

import asyncio
from pydantic import BaseModel, Field
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
# FakeListChatModel: for testing/prototyping (no API key needed)
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model
# create_deep_agent(model="anthropic:claude-sonnet-4-20250514", ...) — string-based (recommended)
# create_deep_agent(model=ChatOpenAI(model="gpt-4.1"), ...) — instance-based
model = FakeListChatModel(
    responses=[
        "I'll check the weather for San Francisco.",
        "The weather in San Francisco is 18°C and sunny.",
    ]
)


# ==== Custom Tools ====


class WeatherInput(BaseModel):
    city: str = Field(description="City name (e.g., 'San Francisco')")
    units: str = Field(
        default="celsius", description="Temperature units: celsius or fahrenheit"
    )


@tool(args_schema=WeatherInput)
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city."""
    # Production: call real weather API
    return f"Weather in {city}: 18°{'C' if units == 'celsius' else 'F'}, sunny"


# ==== Agent Setup ====

# Minimal setup: model + tools
# agent = create_deep_agent(
#     model=model,
#     tools=[get_weather],
#     checkpointer=InMemorySaver(),
#     # debug=True,  # Enable to see middleware stack details
# )

# ==== Main ====


async def main():
    # Note: Uncomment create_deep_agent import and agent setup above for real execution.
    # This example shows the pattern; actual deepagents package required.

    # Basic invocation
    # config = {"configurable": {"thread_id": "quickstart-001"}}
    # result = await agent.ainvoke(
    #     {"messages": [HumanMessage(content="What's the weather in San Francisco?")]},
    #     config=config,
    # )
    # print(f"Response: {result['messages'][-1].content}")

    # Streaming invocation
    # async for event in agent.astream(
    #     {"messages": [HumanMessage(content="Check weather in Tokyo")]},
    #     config={"configurable": {"thread_id": "quickstart-002"}},
    #     stream_mode="messages",
    # ):
    #     if hasattr(event[0], 'content'):
    #         print(event[0].content, end="", flush=True)
    # print()

    # --- Demo with FakeListChatModel (no deepagents dependency) ---
    print("=== DeepAgents Quickstart Pattern ===")
    print()

    # Demonstrate tool schema
    print(f"Tool: {get_weather.name}")
    print(f"Schema: {get_weather.args_schema.model_json_schema()}")
    print()

    # Demonstrate tool execution
    result = await get_weather.ainvoke({"city": "San Francisco", "units": "celsius"})
    print(f"Tool result: {result}")
    print()

    # Demonstrate model invocation
    response = await model.ainvoke([HumanMessage(content="What's the weather?")])
    print(f"Model response: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
