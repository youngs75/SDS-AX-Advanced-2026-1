# Tools

## Read This When
- Defining tools for agent use
- Accessing graph state, context, or store from within a tool
- Understanding ToolNode and tool execution model

## Skip This When
- Using only MCP-provided external tools (see `30-mcp.md`)

## Official References
1. https://docs.langchain.com/oss/python/langchain/tools
   - Why: tool definition, ToolRuntime injection, and ToolNode integration

## Core Guidance

### 1. @tool decorator basics
Docstring becomes description, type hints define schema, always use `async def`:

```python
from langchain.tools import tool

@tool
async def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query."""
    return f"Found {limit} results for '{query}'"
```

### 2. Pydantic schema for complex inputs
Use `args_schema` for structured validation:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(default="celsius")

@tool(args_schema=WeatherInput)
async def get_weather(location: str, units: str = "celsius") -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: 22°{units[0].upper()}"
```

### 3. ToolRuntime injection
Access state, context, store, stream_writer. The `runtime` parameter is hidden from the model:

```python
from langchain.tools import tool, ToolRuntime

@tool
async def get_user_preference(pref_name: str, runtime: ToolRuntime) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

### 4. ToolRuntime components

| Component | Access | Purpose |
|-----------|--------|---------|
| `runtime.state` | Short-term memory | Current conversation state |
| `runtime.context` | Immutable config | User IDs, session info |
| `runtime.store` | Long-term memory | Cross-session persistence |
| `runtime.stream_writer` | Real-time updates | Progress signals |
| `runtime.config` | RunnableConfig | Execution metadata |

### 5. State update from tool
Use Command to update conversation state:

```python
from langgraph.types import Command

@tool
async def set_user_name(new_name: str) -> Command:
    """Set the user's name in conversation state."""
    return Command(update={"user_name": new_name})
```

### 6. Long-term store access
Persist data across sessions:

```python
@tool
async def save_note(user_id: str, note: str, runtime: ToolRuntime) -> str:
    """Save a note for the user."""
    runtime.store.put(("users", user_id), "notes", {"content": note})
    return "Note saved."
```

### 7. ToolNode automatic execution
Integrate tools into LangGraph for automatic execution:

```python
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END

tool_node = ToolNode([search_database, get_weather])

builder = StateGraph(MessagesState)
builder.add_node("llm", call_llm)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", tools_condition)
builder.add_edge("tools", "llm")
graph = builder.compile()
```

### 8. Error handling on ToolNode
Configure how tool errors are handled:

```python
tool_node = ToolNode(tools, handle_tool_errors=True)           # catch all
tool_node = ToolNode(tools, handle_tool_errors="Try again.")   # custom message
tool_node = ToolNode(tools, handle_tool_errors=(ValueError,))  # specific exceptions
```

### 9. Installation baseline
Basic setup for LangChain with tools:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv add langchain                # core + langgraph
uv add langchain-openai         # OpenAI provider
```

## Asset Examples

| Topic | Asset Path |
|-------|-----------|
| ToolRuntime (state, context, store, stream_writer, config) | `assets/02-langchain/tool-runtime-example/` |
| Core-level tool definitions (@tool, BaseTool, StructuredTool, InjectedToolArg) | `assets/01-langchain-core/tool-definition/` |

## Quick Checklist
- [ ] Are all @tool functions async?
- [ ] Are tool docstrings clear and descriptive (model sees them)?
- [ ] Is ToolRuntime used for state/store access (not global variables)?
- [ ] Are tool errors handled via ToolNode config?

## Next File
`40-messages.md`
