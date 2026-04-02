# LangGraph: Subgraphs

## Read This When
- Composing reusable graphs from smaller graph modules
- Building multi-agent systems with nested graphs
- Need state transformation between parent and child graphs

## Skip This When
- Single flat graph handles all requirements
- No need for graph composition or distributed development

## Official References
1. https://docs.langchain.com/oss/python/langgraph/use-subgraphs
   - Why: subgraph patterns, state transformation, nested persistence, streaming.

## Core Guidance

### 1. Two Composition Patterns

| Pattern | State Schemas | How | Use When |
|---------|--------------|-----|----------|
| **Invoke from node** | Different | Call `subgraph.ainvoke()` inside a node function | Parent and child have different state shapes |
| **Add as node** | Shared keys | `builder.add_node("sub", compiled_subgraph)` | Parent and child share state keys (e.g., `messages`) |

### 2. Pattern A: Invoke from Node (Different Schemas)
Node function transforms parent state → subgraph input, invokes, transforms result back:

```python
# Child graph has its own state
class ChildState(TypedDict):
    query: str
    result: str

child_graph = child_builder.compile()

# Parent node wraps the child
async def research_node(state: ParentState) -> dict:
    child_input = {"query": state["messages"][-1].content}
    child_result = await child_graph.ainvoke(child_input)
    return {"research_result": child_result["result"]}
```

### 3. Pattern B: Add as Node (Shared Keys)
Compiled subgraph added directly — shared keys pass through automatically:

```python
class ParentState(TypedDict):
    messages: Annotated[list, add_messages]

class ChildState(TypedDict):
    messages: Annotated[list, add_messages]  # shared key

child_graph = child_builder.compile()

parent_builder = StateGraph(ParentState)
parent_builder.add_node("child_agent", child_graph)  # add compiled graph
parent_builder.add_edge(START, "child_agent")
parent_builder.add_edge("child_agent", END)
```

### 4. Nested Persistence
- Checkpointer from parent **propagates automatically** to subgraphs
- For **independent memory** (separate thread namespace): `child_builder.compile(checkpointer=True)`
- Subgraph checkpoints are stored under the parent's thread with nested namespace

### 5. Subgraph State Inspection
View interrupted subgraph state from the parent:

```python
# Get parent state — includes subgraph state when paused
parent_state = await parent_graph.aget_state(config)

# Access subgraph state via tasks
for task in parent_state.tasks:
    if task.state:  # subgraph state available
        print(f"Subgraph: {task.name}")
        print(f"State: {task.state.values}")
```

### 6. Streaming from Subgraphs
Enable `subgraphs=True` to receive events from nested graphs:

```python
async for namespace, chunk in parent_graph.astream(
    input, config=config,
    stream_mode="updates",
    subgraphs=True,
):
    if namespace:
        # namespace is a tuple: ("child_agent:abc123",)
        print(f"[subgraph] {chunk}")
    else:
        print(f"[parent] {chunk}")
```

### 7. Design Guidelines
- **Reusability**: Design subgraphs with minimal state requirements for maximum reuse
- **State boundary**: Keep subgraph state independent when possible — transform at boundaries
- **Testing**: Test subgraphs in isolation before composing into parent
- **Distributed development**: Different team members can own different subgraphs
- **Avoid deep nesting**: 2-3 levels maximum for debuggability

## Quick Checklist
- [ ] State transformation explicit between parent and child?
- [ ] Checkpointer propagation understood (auto vs independent)?
- [ ] Shared keys intentional (not accidental leakage)?
- [ ] Subgraph streaming enabled (`subgraphs=True`) when needed?
- [ ] Subgraphs tested independently before composition?

## Next File
- `65-workflow-patterns.md`
