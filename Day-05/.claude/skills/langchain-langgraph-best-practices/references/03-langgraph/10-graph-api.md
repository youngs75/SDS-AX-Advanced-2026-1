# LangGraph: Graph API

## Read This When

- Designing graphs with explicit node/edge topology.
- Need StateGraph fundamentals, reducer mechanics, or compilation options.

## Skip This When

- Using Functional API only.
- Question is about create_agent (framework level).

## Official References

1. https://docs.langchain.com/oss/python/langgraph/graph-api
   - Why: canonical StateGraph, nodes, edges, reducers, compilation.
2. https://docs.langchain.com/oss/python/langgraph/use-graph-api
   - Why: practical Graph API usage patterns.
3. https://docs.langchain.com/oss/python/langgraph/pregel
   - Why: underlying Pregel execution model.

## Core Guidance

### 1. StateGraph — Primary Graph Class

- Parameterized by State type: TypedDict, dataclass, or Pydantic BaseModel
- `StateGraph(MyState)` creates graph with typed state channels
- Each key in state becomes a channel with update semantics

### 2. Node Definition

- Async functions accepting state, returning partial state update dict

```python
from langgraph.graph import StateGraph, START, END

async def my_node(state: MyState) -> dict:
    return {"messages": [AIMessage(content="Hello")]}
```

- Nodes must be single-purpose: one LLM call OR one action
- Return dict with only the keys you want to update

### 3. Edge Types

- Normal: `graph.add_edge("node_a", "node_b")` — unconditional
- Conditional: `graph.add_conditional_edges("node_a", route_fn, {"path1": "node_b", "path2": "node_c"})` — function decides
- Entry: `graph.add_edge(START, "first_node")` — graph entry point
- Terminal: `graph.add_edge("last_node", END)` — graph termination

### 4. Reducer Mechanics

| Annotation | Behavior | Use When |
|-----------|----------|----------|
| (none) | Overwrite | Simple values (int, str, dict) |
| `Annotated[list, operator.add]` | Append | Accumulating results |
| `Annotated[list, add_messages]` | Smart merge (dedup by ID) | Message lists |
| Custom `Annotated[T, my_reducer]` | Custom merge logic | Complex merge needs |

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator

class MyState(TypedDict):
    messages: Annotated[list, add_messages]     # smart merge
    results: Annotated[list[str], operator.add]  # append
    step_count: int                               # overwrite
```

### 5. Input/Output Schema Separation

- `StateGraph(MyState, input=InputSchema, output=OutputSchema)` to restrict visible channels
- Private channels (not in input/output) are internal only
- Prevents leaking implementation state to callers

### 6. Compilation

```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(
    checkpointer=InMemorySaver(),     # enable persistence
    interrupt_before=["human_node"],   # pause before this node
    interrupt_after=["review_node"],   # pause after this node
)
```

- `.compile()` validates graph and returns a `CompiledGraph` (Pregel instance)
- Compiled graph has `invoke()`, `ainvoke()`, `stream()`, `astream()`

### 7. Minimal Example

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def chatbot(state: State) -> dict:
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile()
```

## Quick Checklist

- [ ] Each node single-purpose (one LLM call or one action)?
- [ ] Edge conditions testable and deterministic?
- [ ] Reducers defined for all concurrent/accumulating channels?
- [ ] Graph termination explicit (reaches END)?
- [ ] Input/output schemas restrict visible state?

## Next File

- `15-functional-api.md`
