# LangGraph: Streaming

## Read This When
- Need real-time output, progress updates, or token streaming
- Building streaming UI or API endpoints
- Implementing custom progress events from tools or nodes

## Skip This When
- Only need final results (use `ainvoke()`)
- No real-time display requirements

## Official References
1. https://docs.langchain.com/oss/python/langgraph/streaming
   - Why: five stream modes, custom emission, subgraph streaming.

## Core Guidance

### 1. Five Stream Modes

| Mode | Output | Use When |
|------|--------|----------|
| `values` | Full state after each step | Need complete state snapshots |
| `updates` | Only changed keys per step | Efficient progress tracking |
| `custom` | User-defined data from nodes/tools | Progress bars, status messages |
| `messages` | LLM token chunks with metadata | Real-time chat UI |
| `debug` | Full execution trace | Development debugging |

### 2. Basic Streaming
```python
# State updates (most common)
async for chunk in graph.astream(input, config=config, stream_mode="updates"):
    node_name = list(chunk.keys())[0]
    node_output = chunk[node_name]
    print(f"[{node_name}] {node_output}")
```

### 3. Token-Level Streaming (messages mode)
```python
async for msg, metadata in graph.astream(input, config=config, stream_mode="messages"):
    if msg.content:
        print(msg.content, end="", flush=True)
    # metadata includes: langgraph_node, langgraph_step, tags
```

### 4. Token Filtering
Filter by node name or tags to get tokens from specific LLM calls:
```python
async for msg, metadata in graph.astream(input, config=config, stream_mode="messages"):
    # Only show tokens from "generate" node
    if metadata["langgraph_node"] == "generate":
        print(msg.content, end="")

    # Or filter by tag
    if "user_facing" in metadata.get("tags", []):
        print(msg.content, end="")
```

### 5. Multi-Mode Streaming
Receive multiple stream types simultaneously:
```python
async for mode, chunk in graph.astream(
    input, config=config,
    stream_mode=["messages", "updates"],
):
    if mode == "messages":
        msg, metadata = chunk
        print(f"[token] {msg.content}", end="")
    elif mode == "updates":
        print(f"\n[update] {chunk}")
```

### 6. Custom Data Emission
Emit custom events from nodes or tools using `get_stream_writer()`:

```python
from langgraph.config import get_stream_writer

async def research_node(state: MyState) -> dict:
    writer = get_stream_writer()

    writer({"status": "searching", "progress": 0.0})
    results = await search(state["query"])
    writer({"status": "analyzing", "progress": 0.5})
    summary = await analyze(results)
    writer({"status": "complete", "progress": 1.0})

    return {"summary": summary}

# Consume custom events
async for chunk in graph.astream(input, config=config, stream_mode="custom"):
    print(f"Progress: {chunk['progress']:.0%} — {chunk['status']}")
```

### 7. Subgraph Streaming
Stream events from nested subgraphs:
```python
async for namespace, chunk in graph.astream(
    input, config=config,
    stream_mode="updates",
    subgraphs=True,
):
    if namespace:
        print(f"[subgraph:{namespace}] {chunk}")
    else:
        print(f"[parent] {chunk}")
```

### 8. Disabling Streaming Per Model
Some nodes shouldn't stream (e.g., classification, routing):
```python
from langchain_openai import ChatOpenAI

# This model's tokens won't appear in messages stream
classifier = ChatOpenAI(model="gpt-4.1-mini", streaming=False)
```

### 9. Async Best Practice
- Always use `astream()` (not sync `stream()`) in production
- `astream()` returns an async iterator — use `async for`
- Combine with `astream_events()` for LangChain callback-based events

## Quick Checklist
- [ ] `astream()` used (not sync `stream()`)?
- [ ] `stream_mode` explicitly chosen for use case?
- [ ] Custom events emitted for long-running tools/nodes?
- [ ] Subgraph streaming enabled when using nested graphs?
- [ ] Token filtering applied for multi-LLM graphs?

## Next File
- `60-subgraphs.md`
