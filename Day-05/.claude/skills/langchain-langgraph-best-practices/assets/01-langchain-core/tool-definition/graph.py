"""
Tool Definition Patterns Demo (Core Level)

Demonstrates comprehensive tool definition patterns using only langchain_core:
1. @tool decorator with async functions and docstrings
2. Pydantic args_schema with Field descriptions
3. StructuredTool.from_function() programmatic creation
4. BaseTool subclass with custom _arun implementation
5. InjectedToolArg for hidden parameters
6. Tool schema inspection and introspection
7. Error handling patterns (return vs raise)
8. Multiple return type patterns (str, dict, structured)

CRITICAL: Only imports from langchain_core and pydantic allowed.
"""

import asyncio
from typing import Annotated, Optional

from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, tool
from pydantic import BaseModel, Field

# Uncomment for real model usage:
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

# ==============================================================================
# PATTERN 1: @tool Decorator with Docstring
# ==============================================================================


@tool
async def search_database(query: str, limit: int = 10) -> str:
    """Search the product database for matching items.

    Args:
        query: The search query string
        limit: Maximum number of results to return
    """
    # Simulate async database search
    await asyncio.sleep(0.1)
    return f"Found {limit} results for '{query}'"


# ==============================================================================
# PATTERN 2: Pydantic args_schema with Field Descriptions
# ==============================================================================


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""

    operation: str = Field(
        description="The operation to perform: add, subtract, multiply, divide"
    )
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


@tool(args_schema=CalculatorInput)
async def calculator(operation: str, a: float, b: float) -> dict:
    """Perform basic arithmetic operations.

    Returns a dictionary with the result and operation details.
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else None,
    }

    result = operations.get(operation)
    if result is None:
        return {"error": f"Invalid operation '{operation}' or division by zero"}

    return {
        "operation": operation,
        "operands": [a, b],
        "result": result,
    }


# ==============================================================================
# PATTERN 3: StructuredTool.from_function() Programmatic Creation
# ==============================================================================


async def fetch_weather(city: str, units: str = "celsius") -> str:
    """Fetch current weather for a city.

    Args:
        city: Name of the city
        units: Temperature units (celsius or fahrenheit)
    """
    await asyncio.sleep(0.1)
    temp = 22 if units == "celsius" else 72
    return f"Weather in {city}: {temp}°{units[0].upper()}"


# Create tool programmatically
weather_tool = StructuredTool.from_function(
    func=fetch_weather,
    name="get_weather",
    description="Get current weather information for any city",
)


# ==============================================================================
# PATTERN 4: BaseTool Subclass with Custom _arun
# ==============================================================================


class UserLookupInput(BaseModel):
    """Input for user lookup."""

    user_id: str = Field(description="The unique user identifier")
    include_email: bool = Field(
        default=False,
        description="Whether to include email in response",
    )


class UserLookupTool(BaseTool):
    """Custom tool for looking up user information."""

    name: str = "lookup_user"
    description: str = "Look up user information by user ID"
    args_schema: type[BaseModel] = UserLookupInput

    async def _arun(
        self,
        user_id: str,
        include_email: bool = False,
    ) -> dict:
        """Async implementation of the tool."""
        # Simulate database lookup
        await asyncio.sleep(0.1)

        user_data = {
            "user_id": user_id,
            "name": f"User_{user_id}",
            "status": "active",
        }

        if include_email:
            user_data["email"] = f"user_{user_id}@example.com"

        return user_data


# ==============================================================================
# PATTERN 5: InjectedToolArg for Hidden Parameters
# ==============================================================================


@tool
async def send_notification(
    message: str,
    user_id: str,
    # Hidden parameter - not exposed to model
    api_key: Annotated[str, InjectedToolArg],
) -> str:
    """Send a notification to a user.

    Args:
        message: The notification message to send
        user_id: Target user identifier
    """
    # api_key is injected at runtime, not by the model
    await asyncio.sleep(0.1)
    return f"Notification sent to {user_id} using key {api_key[:8]}..."


# ==============================================================================
# PATTERN 6: Error Handling - Return vs Raise
# ==============================================================================


@tool
async def safe_divide(a: float, b: float) -> str:
    """Divide two numbers with error handling (returns error as string)."""
    if b == 0:
        return "Error: Cannot divide by zero"
    return f"Result: {a / b}"


@tool
async def strict_divide(a: float, b: float) -> float:
    """Divide two numbers with strict error handling (raises exception)."""
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


# ==============================================================================
# PATTERN 7: Multiple Return Types
# ==============================================================================


class SearchResult(BaseModel):
    """Structured search result."""

    query: str
    total_results: int
    top_match: Optional[str] = None


@tool
async def structured_search(query: str) -> SearchResult:
    """Search with structured response using Pydantic model."""
    await asyncio.sleep(0.1)
    return SearchResult(
        query=query,
        total_results=42,
        top_match=f"Best match for '{query}'",
    )


# ==============================================================================
# Main Demo
# ==============================================================================


async def main():
    """Demonstrate all tool definition patterns."""

    print("=" * 80)
    print("Tool Definition Patterns Demo")
    print("=" * 80)

    # Pattern 1: Decorator with docstring
    print("\n1. @tool Decorator Pattern:")
    print(f"   Name: {search_database.name}")
    print(f"   Description: {search_database.description}")
    result = await search_database.ainvoke({"query": "laptop", "limit": 5})
    print(f"   Result: {result}")

    # Pattern 2: Pydantic args_schema
    print("\n2. Pydantic args_schema Pattern:")
    print(f"   Name: {calculator.name}")
    schema = calculator.args_schema.model_json_schema()
    print(f"   Required fields: {schema.get('required', [])}")
    result = await calculator.ainvoke({
        "operation": "multiply",
        "a": 7,
        "b": 6,
    })
    print(f"   Result: {result}")

    # Pattern 3: StructuredTool.from_function()
    print("\n3. StructuredTool.from_function() Pattern:")
    print(f"   Name: {weather_tool.name}")
    print(f"   Description: {weather_tool.description}")
    result = await weather_tool.ainvoke({"city": "Tokyo", "units": "celsius"})
    print(f"   Result: {result}")

    # Pattern 4: BaseTool subclass
    print("\n4. BaseTool Subclass Pattern:")
    user_tool = UserLookupTool()
    print(f"   Name: {user_tool.name}")
    result = await user_tool.ainvoke({
        "user_id": "12345",
        "include_email": True,
    })
    print(f"   Result: {result}")

    # Pattern 5: InjectedToolArg
    print("\n5. InjectedToolArg Pattern:")
    print(f"   Tool: {send_notification.name}")
    print(f"   Args schema: {send_notification.args_schema.model_json_schema()['properties'].keys()}")
    # Note: api_key is not in the schema - it's injected
    # In real usage, bind it with: send_notification.bind(api_key="secret_key_123")

    # Pattern 6: Error handling
    print("\n6. Error Handling Patterns:")
    safe_result = await safe_divide.ainvoke({"a": 10, "b": 0})
    print(f"   Safe divide (returns error): {safe_result}")

    try:
        await strict_divide.ainvoke({"a": 10, "b": 0})
    except ValueError as e:
        print(f"   Strict divide (raises): Caught {type(e).__name__}: {e}")

    # Pattern 7: Structured returns
    print("\n7. Structured Return Pattern:")
    result = await structured_search.ainvoke({"query": "ai tools"})
    print(f"   Result type: {type(result).__name__}")
    print(f"   Result: {result}")

    # Schema inspection
    print("\n8. Schema Inspection:")
    print(f"   calculator.name: {calculator.name}")
    print(f"   calculator.description: {calculator.description}")
    schema = calculator.args_schema.model_json_schema()
    print(f"   calculator.args_schema properties:")
    for prop, details in schema.get("properties", {}).items():
        print(f"     - {prop}: {details.get('description', 'N/A')}")

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
