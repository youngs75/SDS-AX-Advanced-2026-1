"""
Message Patterns and Content Blocks - LangChain Core Asset

Demonstrates all 5 message types and content block patterns:
1. HumanMessage - basic text and multimodal (text + image)
2. AIMessage - with .content, .tool_calls, .usage_metadata attributes
3. SystemMessage - role-based instructions
4. ToolMessage - with tool_call_id matching to AIMessage.tool_calls[].id
5. RemoveMessage - explicit history trimming with id parameter

Content block patterns:
- Text content blocks
- Image URL content blocks
- Tool use content blocks
- Tool result content blocks

Key patterns:
- Multimodal: HumanMessage(content=[{"type":"text",...}, {"type":"image_url",...}])
- Full tool-calling sequence: System → Human → AI(tool_calls) → Tool(tool_call_id) → AI(response)
- Dict coercion: {"role":"user", "content":"..."} → typed messages
- Message inspection and metadata access

CRITICAL: Only imports from langchain_core and pydantic. NO langchain, langgraph, or deepagents.

Reference: references/01-langchain-core/20-message-tool-schema.md
"""

import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
    BaseMessage,
)
from langchain_core.language_models import FakeListChatModel

# Real model imports (commented out for demo):
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic


# ==============================================================================
# Tool Schema Definitions
# ==============================================================================


class WeatherQuery(BaseModel):
    """Query for weather information."""

    location: str = Field(description="City name or location")
    units: str = Field(default="celsius", description="Temperature units")


class SearchQuery(BaseModel):
    """Query for web search."""

    query: str = Field(description="Search query text")
    max_results: int = Field(default=5, description="Maximum number of results")


# ==============================================================================
# Helper Functions
# ==============================================================================


def print_message(msg: BaseMessage, label: str = "Message") -> None:
    """Print message details in structured format."""
    print(f"\n{'='*60}")
    print(f"{label}: {msg.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Type: {msg.type}")
    print(f"Content: {msg.content}")

    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"Tool Calls: {msg.tool_calls}")

    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        print(f"Usage Metadata: {msg.usage_metadata}")

    if hasattr(msg, "tool_call_id"):
        print(f"Tool Call ID: {msg.tool_call_id}")

    if hasattr(msg, "id") and msg.id:
        print(f"Message ID: {msg.id}")

    print(f"Additional kwargs: {msg.additional_kwargs}")


def simulate_tool_execution(tool_name: str, args: Dict[str, Any]) -> str:
    """Simulate tool execution and return mock result."""
    if tool_name == "get_weather":
        location = args.get("location", "unknown")
        return f"Weather in {location}: 22°C, Partly cloudy"
    elif tool_name == "search":
        query = args.get("query", "")
        return f"Search results for '{query}': Found 3 relevant articles"
    return "Tool execution completed"


# ==============================================================================
# Pattern 1: Basic Message Types
# ==============================================================================


async def demo_basic_messages() -> None:
    """Demonstrate all 5 basic message types."""
    print("\n" + "=" * 60)
    print("PATTERN 1: Basic Message Types")
    print("=" * 60)

    # 1. SystemMessage - role-based instructions
    system_msg = SystemMessage(content="You are a helpful weather assistant.")
    print_message(system_msg, "1. SystemMessage")

    # 2. HumanMessage - user input (text only)
    human_msg = HumanMessage(content="What's the weather in London?")
    print_message(human_msg, "2. HumanMessage (text)")

    # 3. AIMessage - assistant response with tool calls
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "get_weather",
                "args": {"location": "London", "units": "celsius"},
            }
        ],
    )
    print_message(ai_msg, "3. AIMessage (with tool_calls)")

    # 4. ToolMessage - tool execution result
    tool_msg = ToolMessage(
        content="Weather in London: 15°C, Rainy",
        tool_call_id="call_1",
    )
    print_message(tool_msg, "4. ToolMessage")

    # 5. RemoveMessage - explicit history trimming
    remove_msg = RemoveMessage(id=human_msg.id)
    print_message(remove_msg, "5. RemoveMessage")


# ==============================================================================
# Pattern 2: Multimodal Content Blocks
# ==============================================================================


async def demo_multimodal_content() -> None:
    """Demonstrate multimodal content blocks (text + image)."""
    print("\n" + "=" * 60)
    print("PATTERN 2: Multimodal Content Blocks")
    print("=" * 60)

    # Multimodal HumanMessage with text + image
    multimodal_msg = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/weather-map.jpg"},
            },
        ]
    )
    print_message(multimodal_msg, "Multimodal HumanMessage")

    # Access individual content blocks
    print("\nContent Block Access:")
    for i, block in enumerate(multimodal_msg.content):
        print(f"  Block {i}: type={block['type']}")
        if block["type"] == "text":
            print(f"    text={block['text']}")
        elif block["type"] == "image_url":
            print(f"    url={block['image_url']['url']}")


# ==============================================================================
# Pattern 3: Tool-Calling Conversation Sequence
# ==============================================================================


async def demo_tool_calling_sequence() -> None:
    """Demonstrate complete tool-calling conversation flow."""
    print("\n" + "=" * 60)
    print("PATTERN 3: Tool-Calling Conversation Sequence")
    print("=" * 60)

    conversation: List[BaseMessage] = []

    # Step 1: System instruction
    system = SystemMessage(
        content="You are a helpful assistant with access to weather and search tools."
    )
    conversation.append(system)
    print_message(system, "Step 1: System")

    # Step 2: Human request
    human = HumanMessage(content="What's the weather in Paris and find news about it?")
    conversation.append(human)
    print_message(human, "Step 2: Human")

    # Step 3: AI responds with multiple tool calls
    ai_tool_calls = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_weather_1",
                "name": "get_weather",
                "args": {"location": "Paris", "units": "celsius"},
            },
            {
                "id": "call_search_1",
                "name": "search",
                "args": {"query": "Paris weather news", "max_results": 3},
            },
        ],
    )
    conversation.append(ai_tool_calls)
    print_message(ai_tool_calls, "Step 3: AI (tool_calls)")

    # Step 4: Tool results (one ToolMessage per tool_call)
    tool_result_1 = ToolMessage(
        content=simulate_tool_execution("get_weather", {"location": "Paris"}),
        tool_call_id="call_weather_1",
    )
    conversation.append(tool_result_1)
    print_message(tool_result_1, "Step 4a: Tool Result (weather)")

    tool_result_2 = ToolMessage(
        content=simulate_tool_execution("search", {"query": "Paris weather news"}),
        tool_call_id="call_search_1",
    )
    conversation.append(tool_result_2)
    print_message(tool_result_2, "Step 4b: Tool Result (search)")

    # Step 5: AI final response
    ai_response = AIMessage(
        content="The weather in Paris is 22°C and partly cloudy. I found recent news articles discussing the pleasant spring weather."
    )
    conversation.append(ai_response)
    print_message(ai_response, "Step 5: AI (final response)")

    print(f"\nTotal messages in conversation: {len(conversation)}")


# ==============================================================================
# Pattern 4: Dict Coercion and Message Inspection
# ==============================================================================


async def demo_dict_coercion() -> None:
    """Demonstrate dict → typed message coercion."""
    print("\n" + "=" * 60)
    print("PATTERN 4: Dict Coercion and Message Inspection")
    print("=" * 60)

    # Dict-style message definitions
    dict_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    # Convert to typed messages
    typed_messages = [
        SystemMessage(content=dict_messages[0]["content"]),
        HumanMessage(content=dict_messages[1]["content"]),
        AIMessage(content=dict_messages[2]["content"]),
    ]

    print("\nDict → Typed Message Coercion:")
    for i, (dict_msg, typed_msg) in enumerate(zip(dict_messages, typed_messages)):
        print(f"\n  Dict {i}: {dict_msg}")
        print(f"  Typed {i}: {typed_msg.__class__.__name__}(content={typed_msg.content!r})")

    # Message inspection
    print("\nMessage Inspection:")
    for msg in typed_messages:
        print(f"  - {msg.__class__.__name__}: type={msg.type}, id={msg.id}")


# ==============================================================================
# Pattern 5: AIMessage Attributes (content, tool_calls, usage_metadata)
# ==============================================================================


async def demo_ai_message_attributes() -> None:
    """Demonstrate AIMessage special attributes."""
    print("\n" + "=" * 60)
    print("PATTERN 5: AIMessage Attributes")
    print("=" * 60)

    # AIMessage with all attributes
    ai_msg = AIMessage(
        content="Based on the weather data, I recommend bringing an umbrella.",
        tool_calls=[
            {
                "id": "call_123",
                "name": "get_weather",
                "args": {"location": "Seattle"},
            }
        ],
        usage_metadata={
            "input_tokens": 150,
            "output_tokens": 75,
            "total_tokens": 225,
        },
    )

    print("\nAIMessage Attributes:")
    print(f"  content: {ai_msg.content}")
    print(f"  tool_calls: {ai_msg.tool_calls}")
    print(f"  usage_metadata: {ai_msg.usage_metadata}")

    # Access individual tool call
    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[0]
        print(f"\n  First tool call:")
        print(f"    id: {tool_call['id']}")
        print(f"    name: {tool_call['name']}")
        print(f"    args: {tool_call['args']}")


# ==============================================================================
# Main Demo
# ==============================================================================


async def main() -> None:
    """Run all message pattern demonstrations."""
    print("\n" + "=" * 80)
    print("LangChain Core: Message Patterns and Content Blocks")
    print("=" * 80)

    await demo_basic_messages()
    await demo_multimodal_content()
    await demo_tool_calling_sequence()
    await demo_dict_coercion()
    await demo_ai_message_attributes()

    print("\n" + "=" * 80)
    print("All demonstrations complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
