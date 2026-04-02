# Streaming

## Read This When
- Need real-time token streaming or intermediate state updates
- Building a streaming UI or API
- Streaming across sub-agent boundaries

## Skip This When
- Only need final results (use `ainvoke` instead)

## Official References
1. https://docs.langchain.com/oss/python/langchain/streaming/overview
   - Why: stream mode options and patterns

## Core Guidance

1. **stream_mode options**:
| Mode | Purpose | Emits |
|------|---------|-------|
| `updates` | State changes after each agent step | `{node_name: state_update}` |
| `messages` | LLM tokens as generated | `(token, metadata)` tuples |
| `custom` | User-defined signals from tools/nodes | arbitrary data |
| `values` | Full state after each step | complete state dict |
| `debug` | Detailed execution trace | debug events |

2. **Agent progress streaming** (`updates` mode):
```python
async for chunk in agent.astream(
    {"messages": [{"role": "user", "content": "What's the weather in SF?"}]},
    stream_mode="updates",
):
    for node_name, update in chunk.items():
        print(f"Step: {node_name}, Messages: {len(update.get('messages', []))}")
```

3. **Token-level streaming** (`messages` mode):
```python
async for token, metadata in agent.astream(
    {"messages": [{"role": "user", "content": "Tell me a story"}]},
    stream_mode="messages",
):
    if token.content:
        print(token.content, end="", flush=True)
```

4. **Multi-mode streaming** (receive multiple streams):
```python
async for stream_mode, chunk in agent.astream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = chunk
        print(f"Token: {token.content}")
    elif stream_mode == "updates":
        print(f"State update from: {list(chunk.keys())}")
```

5. **Custom stream events** from tools (via `get_stream_writer`):
```python
from langgraph.config import get_stream_writer

@tool
async def analyze_data(query: str) -> str:
    """Analyze data with progress updates."""
    writer = get_stream_writer()
    writer({"status": "loading", "progress": 0})
    # ... processing ...
    writer({"status": "analyzing", "progress": 50})
    # ... more processing ...
    writer({"status": "complete", "progress": 100})
    return "Analysis complete: ..."
```
Note: `get_stream_writer()` makes the tool dependent on LangGraph execution context.

6. **Sub-agent streaming** (disambiguate sources):
```python
agent = create_agent(model="openai:gpt-4.1", tools=[...], name="main_agent")

async for _, stream_mode, data in agent.astream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
):
    if stream_mode == "messages":
        token, metadata = data
        agent_name = metadata.get("lc_agent_name", "unknown")
        print(f"[{agent_name}] {token.content}")
```

7. **Disabling streaming** (for specific models):
```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4.1", streaming=False)
```

8. **Sync vs Async**: Always prefer `astream()` over `stream()` in production.

## Quick Checklist
- [ ] Is `astream()` used (not sync `stream()`)?
- [ ] Is `stream_mode` explicitly chosen for the use case?
- [ ] Are custom stream events used for long-running tools?
- [ ] Is sub-agent streaming handled with `subgraphs=True` when needed?

## Next File
`55-short-term-memory.md`
