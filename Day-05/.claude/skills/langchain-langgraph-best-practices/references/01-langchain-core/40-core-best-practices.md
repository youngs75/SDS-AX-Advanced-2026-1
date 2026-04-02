# LangChain Core Best Practices

## Read This When

- You want a go/no-go check before moving to framework/runtime implementation.

## Skip This When

- You only need quick conceptual boundaries.

## Official References

1. https://docs.langchain.com/oss/python/langchain/messages
2. https://docs.langchain.com/oss/python/langchain/tools
3. https://reference.langchain.com/python/langchain_core/runnables/
   - Why: baseline contract quality for message/tool/runnable layers.

## Core Guidance

### 1. Message Contracts — Be Explicit

**Do**: Use typed message classes with explicit roles
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What's the weather?"),
    AIMessage(content="I'll check that for you"),
]
```

**Don't**: Pass raw strings or untyped dicts in critical paths
```python
# Fragile — role inference varies by provider
messages = ["You are helpful", "What's the weather?"]
```

**Why**: Explicit typing prevents role confusion and ensures consistent serialization across providers.

### 2. Tool Schemas — Keep Deterministic

**Do**: Define `args_schema` with Pydantic, use `Field(description=...)` for every param
```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name (e.g., 'San Francisco')")
    units: str = Field(default="celsius", description="Temperature units")

@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 20°{units[0].upper()}"
```

**Don't**: Rely on function signatures alone for complex inputs
```python
@tool
def get_weather(city, units="celsius"):  # Missing descriptions, type hints
    return f"Weather in {city}"
```

**Why**: Clear schemas improve tool selection accuracy and reduce hallucinated parameters.

### 3. Runnable Modes — Choose Intentionally

**Do**: Use `ainvoke`/`astream` in production (async-first)
```python
result = await chain.ainvoke(input, config=config)

async for chunk in chain.astream(input, config=config):
    print(chunk, end="", flush=True)
```

**Don't**: Mix sync and async in the same pipeline
```python
# Anti-pattern — blocks event loop
def my_node(state):
    result = chain.invoke(state)  # Sync call in async context
    return result
```

**Why**: Async methods prevent blocking and enable concurrent execution.

### 4. State Keys — Keep Stable

**Do**: Treat state keys as API contract; version when changing
```python
class MyStateV1(TypedDict):
    messages: Annotated[list, add_messages]
    step_count: int

# If you need to rename, create V2 and migrate
class MyStateV2(TypedDict):
    messages: Annotated[list, add_messages]
    iteration_count: int  # Renamed from step_count
```

**Don't**: Rename state keys without updating all consumers
```python
# Breaks existing checkpoints and downstream code
class MyState(TypedDict):
    msgs: Annotated[list, add_messages]  # Was "messages"
```

**Why**: State keys are serialized in checkpoints; changes break resume functionality.

### 5. Tool Outputs — Keep Concise

**Do**: Return only essential data from tools (summary, key fields)
```python
@tool
def search_docs(query: str) -> str:
    results = expensive_search(query)
    # Return summary, not full results
    return f"Found {len(results)} results. Top: {results[0]['title']}"
```

**Don't**: Return entire API responses or large documents
```python
@tool
def search_docs(query: str) -> dict:
    # Anti-pattern — bloats context window
    return expensive_search(query)  # Returns 50KB JSON
```

**Why**: Large outputs consume context window and slow processing.

### 6. Error Recovery — Use Structured Messages

**Do**: Return `ToolMessage` with error details for tool errors
```python
from langchain_core.messages import ToolMessage

@tool
def get_weather(city: str) -> str:
    if not city:
        return "Error: city name is required"
    try:
        return fetch_weather(city)
    except APIError as e:
        return f"Error: {str(e)}"
```

**Don't**: Raise unhandled exceptions or return raw error strings
```python
@tool
def get_weather(city: str) -> str:
    # Anti-pattern — crashes the graph
    return fetch_weather(city)  # May raise APIError
```

**Why**: Structured errors allow the agent to recover and retry.

### 7. Async-First Development

**Do**: Use `async def` for all tool functions and node functions
```python
@tool
async def get_weather(city: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/weather/{city}")
        return response.json()

async def my_node(state: MyState) -> MyState:
    result = await get_weather(state["city"])
    return {"messages": [AIMessage(content=result)]}
```

**Don't**: Use sync functions that block the event loop in production
```python
@tool
def get_weather(city: str) -> str:
    # Anti-pattern — blocks event loop
    response = requests.get(f"/weather/{city}")
    return response.json()
```

**Why**: Blocking calls prevent concurrent execution and reduce throughput.

### 8. Red Flags — Wrong-Layer Usage Patterns

These patterns indicate architectural problems:

| Red Flag | Problem | Solution |
|----------|---------|----------|
| Importing `StateGraph` in a tool function | Tool shouldn't know about graph topology | Move graph logic to orchestration layer |
| Using `RunnableConfig` to pass business data | Config is for execution metadata, not domain data | Use state schema fields |
| Calling `model.invoke()` inside a core primitive | Core shouldn't depend on specific models | Accept model as parameter or use dependency injection |
| Using `langchain_core` types to encode provider-specific behavior | Core should be provider-agnostic | Use provider-specific modules (`langchain_openai`, etc.) |

## Quick Checklist

- [ ] Message roles and payload shapes defined?
- [ ] Tool input/output/error surfaces defined?
- [ ] Async-first across all tool and node functions?
- [ ] No red flag patterns present?
- [ ] State keys stable and versioned?
- [ ] Tool outputs concise (< 2KB per call)?
- [ ] Error handling returns structured messages?

## Next File

- Move to framework layer: `../02-langchain/10-create-agent-standard.md`
