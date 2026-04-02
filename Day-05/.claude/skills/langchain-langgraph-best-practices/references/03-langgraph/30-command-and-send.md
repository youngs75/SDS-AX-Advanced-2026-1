# LangGraph: Command and Send

## Read This When
- Need dynamic routing with simultaneous state updates from nodes
- Implementing map-reduce or fan-out patterns
- Choosing between Command and conditional edges

## Skip This When
- Graph uses only static edges and simple conditional routing
- Flow is linear with no dynamic branching

## Official References
1. https://docs.langchain.com/oss/python/langgraph/graph-api
   - Why: Command and Send API specification.
2. https://docs.langchain.com/oss/python/langgraph/workflows-agents
   - Why: orchestrator-worker and parallelization patterns using Send.

## Core Guidance

### 1. Command Object
Combines three capabilities in one return value:
- `goto`: Route to specific node(s)
- `update`: Mutate state simultaneously
- `resume`: Respond to an interrupt

```python
from langgraph.types import Command

async def agent_node(state: MyState) -> Command[Literal["tool_node", "end_node"]]:
    if needs_tool:
        return Command(
            goto="tool_node",
            update={"tool_request": "search"},
        )
    return Command(goto="end_node")
```

### 2. Command Type Annotation
- `Command[Literal["node_a", "node_b"]]` — required for graph validation
- LangGraph checks that goto targets match declared Literal types
- Without annotation, graph compilation may fail or skip validation

### 3. Command vs Conditional Edges

| Feature | Command | Conditional Edge |
|---------|---------|-----------------|
| Who decides routing | Node function | External routing function |
| State update with routing | Yes (atomic) | No (separate step) |
| Interrupt resume | Yes (`Command(resume=...)`) | No |
| Type safety | `Command[Literal[...]]` | Return string matching map |
| Use when | Node has context to decide | Routing logic is separate from node logic |

### 4. Send Object — Dynamic Fan-Out
- `Send("node_name", payload)` creates a parallel execution of a node with custom input
- Return a list of Send objects for map-reduce patterns
- Each Send invocation runs independently with its own payload

```python
from langgraph.types import Send

async def orchestrator(state: MyState) -> list[Send]:
    subtasks = decompose(state["task"])
    return [Send("worker", {"subtask": t}) for t in subtasks]
```

### 5. Map-Reduce Pattern
1. **Orchestrator node** returns `list[Send]` to fan-out
2. **Worker nodes** execute in parallel, each with its own payload
3. **Reducer** aggregates results using `Annotated[list, operator.add]`

```python
import operator
from typing import TypedDict, Annotated

class MapReduceState(TypedDict):
    task: str
    results: Annotated[list[dict], operator.add]  # aggregator

async def worker(state: MapReduceState) -> dict:
    result = await process(state["subtask"])
    return {"results": [result]}
```

### 6. Command for Interrupt Resume
```python
from langgraph.types import Command, interrupt

async def approval_node(state: MyState) -> Command[Literal["execute", "cancel"]]:
    decision = interrupt("Approve this action?")
    if decision == "yes":
        return Command(goto="execute")
    return Command(goto="cancel")

# Caller resumes with:
graph.invoke(Command(resume="yes"), config=config)
```

### 7. Multiple Goto Targets
```python
# Send to multiple nodes simultaneously
return Command(goto=["summarize", "translate"])
```

## Quick Checklist
- [ ] Command type annotations match actual graph nodes?
- [ ] Send payloads are JSON-serializable?
- [ ] Reducer handles fan-out merging (e.g., `operator.add`)?
- [ ] Command vs conditional edge choice intentional?
- [ ] Interrupt resume values validated before use?

## Next File
- `35-durable-execution.md`
