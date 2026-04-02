---
name: langgraph-fundamentals
description: "INVOKE THIS SKILL when writing ANY LangGraph code. Covers StateGraph, state schemas, nodes, edges, Command, Send, invoke, streaming, and error handling."
---

<overview>
LangGraph models agent workflows as **directed graphs**:

- **StateGraph**: Main class for building stateful graphs
- **Nodes**: Functions that perform work and update state
- **Edges**: Define execution order (static or conditional)
- **START/END**: Special nodes marking entry and exit points
- **State with Reducers**: Control how state updates are merged

Graphs must be `compile()`d before execution.
</overview>

<design-methodology>

### Designing a LangGraph application

Follow these 5 steps when building a new graph:

1. **Map out discrete steps** — sketch a flowchart of your workflow. Each step becomes a node.
2. **Identify what each step does** — categorize nodes: LLM step, data step, action step, or user input step. For each, determine static context (prompt), dynamic context (from state), retry strategy, and desired outcome.
3. **Design your state** — state is shared memory for all nodes. Store raw data, format prompts on-demand inside nodes.
4. **Build your nodes** — implement each step as a function that takes state and returns partial updates.
5. **Wire it together** — connect nodes with edges, add conditional routing, compile with a checkpointer if needed.

</design-methodology>

<when-to-use-langgraph>

| Use LangGraph When | Use Alternatives When |
|-------------------|----------------------|
| Need fine-grained control over agent orchestration | Quick prototyping → LangChain agents |
| Building complex workflows with branching/loops | Simple stateless workflows → LangChain direct |
| Require human-in-the-loop, persistence | Batteries-included features → Deep Agents |

</when-to-use-langgraph>

---

## State Management

<state-update-strategies>

| Need | Solution | Example |
|------|----------|---------|
| Overwrite value | No reducer (default) | Simple fields like counters |
| Append to list | Reducer (operator.add / concat) | Message history, logs |
| Custom logic | Custom reducer function | Complex merging |

</state-update-strategies>

<ex-state-with-reducer>
```python
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    name: str                              # Default: overwrites on update
    messages: Annotated[list, operator.add]  # Appends to list
    total: Annotated[int, operator.add]    # Sums integers
```
TypeScript: use `MessagesValue` or `new ReducedValue(schema, { reducer: (curr, upd) => curr.concat(upd) })`.
</ex-state-with-reducer>

<fix-forgot-reducer-for-list>
```python
# WRONG: List will be OVERWRITTEN (last write wins)
class State(TypedDict):
    messages: list  # No reducer!

# CORRECT: Annotated reducer accumulates
class State(TypedDict):
    messages: Annotated[list, operator.add]
# Node 1 returns ["A"], Node 2 returns ["B"] → final: ["A", "B"]
```
</fix-forgot-reducer-for-list>

<fix-state-must-return-dict>
```python
# WRONG: mutate and return full state
def my_node(state: State) -> State:
    state["field"] = "updated"
    return state

# CORRECT: return partial update dict only
def my_node(state: State) -> dict:
    return {"field": "updated"}
```
</fix-state-must-return-dict>

---

## Nodes

<node-function-signatures>

| Signature | When to Use |
|-----------|-------------|
| `def node(state: State)` | Simple nodes that only need state |
| `def node(state: State, config: RunnableConfig)` | Need thread_id, tags, or configurable values |
| `def node(state: State, runtime: Runtime[Context])` | Need runtime context, store, or stream_writer |

```python
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

def plain_node(state: State):
    return {"results": "done"}

def node_with_config(state: State, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    return {"results": f"Thread: {thread_id}"}

def node_with_runtime(state: State, runtime: Runtime[Context]):
    user_id = runtime.context.user_id
    return {"results": f"User: {user_id}"}
```
TypeScript: `(state, config) => {...}` — same signatures, `config?.configurable?.thread_id`.

</node-function-signatures>

---

## Edges

<edge-type-selection>

| Need | Edge Type | When to Use |
|------|-----------|-------------|
| Always go to same node | `add_edge()` | Fixed, deterministic flow |
| Route based on state | `add_conditional_edges()` | Dynamic branching |
| Update state AND route | `Command` | Combine logic in single node |
| Fan-out to multiple nodes | `Send` | Parallel processing with dynamic inputs |

</edge-type-selection>

<ex-basic-graph>
```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    output: str

def process_input(state: State) -> dict:
    return {"output": f"Processed: {state['input']}"}

def finalize(state: State) -> dict:
    return {"output": state["output"].upper()}

graph = (
    StateGraph(State)
    .add_node("process", process_input)
    .add_node("finalize", finalize)
    .add_edge(START, "process")
    .add_edge("process", "finalize")
    .add_edge("finalize", END)
    .compile()
)

result = graph.invoke({"input": "hello"})
print(result["output"])  # "PROCESSED: HELLO"
```
TypeScript: same pattern with `new StateGraph(State).addNode(...).addEdge(...).compile()`, `await graph.invoke(...)`.
</ex-basic-graph>

<ex-conditional-edges>
```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    query: str
    route: str
    result: str

def classify(state: State) -> dict:
    if "weather" in state["query"].lower():
        return {"route": "weather"}
    return {"route": "general"}

def route_query(state: State) -> Literal["weather", "general"]:
    return state["route"]

graph = (
    StateGraph(State)
    .add_node("classify", classify)
    .add_node("weather", lambda s: {"result": "Sunny, 72F"})
    .add_node("general", lambda s: {"result": "General response"})
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_query, ["weather", "general"])
    .add_edge("weather", END)
    .add_edge("general", END)
    .compile()
)
```
TypeScript: `.addConditionalEdges("classify", routeQuery, ["weather", "general"])` — identical structure.
</ex-conditional-edges>

---

## Command

Command combines state updates and routing in a single return value. Fields:
- **`update`**: State updates to apply (like returning a dict from a node)
- **`goto`**: Node name(s) to navigate to next
- **`resume`**: Value to resume after `interrupt()` — see human-in-the-loop skill

<ex-command-state-and-routing>
```python
from langgraph.types import Command
from typing import Literal

class State(TypedDict):
    count: int
    result: str

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    new_count = state["count"] + 1
    if new_count > 5:
        return Command(update={"count": new_count}, goto="node_c")
    return Command(update={"count": new_count}, goto="node_b")

graph = (
    StateGraph(State)
    .add_node("node_a", node_a)
    .add_node("node_b", lambda s: {"result": "B"})
    .add_node("node_c", lambda s: {"result": "C"})
    .add_edge(START, "node_a")
    .add_edge("node_b", END)
    .add_edge("node_c", END)
    .compile()
)
```
**Python**: annotate `Command[Literal["node_b", "node_c"]]` to declare valid destinations.
**TypeScript**: `return new Command({ update, goto })` + `.addNode("node_a", fn, { ends: ["node_b", "node_c"] })`.
</ex-command-state-and-routing>

<warning-command-static-edges>

**Warning**: `Command` only adds **dynamic** edges — static edges defined with `add_edge` / `addEdge` still execute. If `node_a` returns `Command(goto="node_c")` and you also have `graph.add_edge("node_a", "node_b")`, **both** `node_b` and `node_c` will run.

</warning-command-static-edges>

---

## Send API

Fan-out with `Send`: return `[Send("worker", {...})]` from a conditional edge to spawn parallel workers. Requires a reducer on the results field.

<ex-orchestrator-worker>
```python
from langgraph.types import Send
from typing import Annotated
import operator

class OrchestratorState(TypedDict):
    tasks: list[str]
    results: Annotated[list, operator.add]
    summary: str

def orchestrator(state: OrchestratorState):
    return [Send("worker", {"task": task}) for task in state["tasks"]]

def worker(state: dict) -> dict:
    return {"results": [f"Completed: {state['task']}"]}

def synthesize(state: OrchestratorState) -> dict:
    return {"summary": f"Processed {len(state['results'])} tasks"}

graph = (
    StateGraph(OrchestratorState)
    .add_node("worker", worker)
    .add_node("synthesize", synthesize)
    .add_conditional_edges(START, orchestrator, ["worker"])
    .add_edge("worker", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```
TypeScript: `state.tasks.map((task) => new Send("worker", { task }))` — identical pattern.
</ex-orchestrator-worker>

<fix-send-accumulator>
```python
# WRONG: No reducer - last worker overwrites
class State(TypedDict):
    results: list

# CORRECT: reducer accumulates all worker results
class State(TypedDict):
    results: Annotated[list, operator.add]
```
TypeScript: use `new ReducedValue(schema, { reducer: (curr, upd) => curr.concat(upd) })`.
</fix-send-accumulator>

For the full 6-pattern workflow classification and Graph/Functional API comparison, see `references/workflow-patterns.md`.

---

## Running Graphs: Invoke and Stream

Call `graph.invoke(input, config)` for synchronous completion. Use `graph.stream()` for incremental output.

**5 stream modes**: `values` (full state per step) · `updates` (state deltas) · `messages` (LLM tokens + metadata) · `custom` (user-defined events) · `debug` (all execution details)

```python
# Invoke to completion
result = graph.invoke({"input": "hello"}, {"configurable": {"thread_id": "1"}})

# Stream LLM tokens (messages mode)
for chunk in graph.stream({"messages": [HumanMessage("Hello")]}, stream_mode="messages"):
    token, metadata = chunk
    if hasattr(token, "content"):
        print(token.content, end="", flush=True)

# Emit custom events from a node (custom mode)
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()
    writer({"progress": "step 1"})
    return {"result": "done"}

for chunk in graph.stream({"data": "test"}, stream_mode="custom"):
    print(chunk)
```
TypeScript: `await graph.invoke(...)`, `{ streamMode: "messages" }`, `getWriter()` from `@langchain/langgraph`.

For token filtering, multi-mode composition, and subgraph streaming, see `references/streaming-advanced.md`.

---

## Error Handling

Match the error type to the right handler:

<error-handling-table>

| Error Type | Who Fixes | Strategy | Example |
|---|---|---|---|
| Transient (network, rate limits) | System | `RetryPolicy(max_attempts=3)` | `add_node(..., retry_policy=...)` |
| LLM-recoverable (tool failures) | LLM | `ToolNode(tools, handle_tool_errors=True)` | Error returned as ToolMessage |
| User-fixable (missing info) | Human | `interrupt({"message": ...})` | Collect missing data (see HITL skill) |
| Unexpected | Developer | Let bubble up | `raise` |

</error-handling-table>

<ex-retry-policy>
```python
from langgraph.types import RetryPolicy

workflow.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=1.0)
)
```
TypeScript: `.addNode("node", fn, { retryPolicy: { maxAttempts: 3, initialInterval: 1.0 } })`.
</ex-retry-policy>

<ex-tool-node-error-handling>
```python
from langgraph.prebuilt import ToolNode

# handle_tool_errors=True returns errors as ToolMessages so the LLM can recover
tool_node = ToolNode(tools, handle_tool_errors=True)
workflow.add_node("tools", tool_node)
```
TypeScript: `new ToolNode(tools, { handleToolErrors: true })` from `@langchain/langgraph/prebuilt`.
</ex-tool-node-error-handling>

---

## Common Fixes

<fix-compile-before-execution>
```python
# WRONG
builder.invoke({"input": "test"})  # AttributeError!

# CORRECT
graph = builder.compile()
graph.invoke({"input": "test"})
```
TypeScript: always `await graph.invoke(...)` — it returns a Promise.
</fix-compile-before-execution>

<fix-infinite-loop-needs-exit>
```python
# WRONG: Loops forever
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", "node_a")

# CORRECT: conditional exit
def should_continue(state):
    return END if state["count"] > 10 else "node_b"
builder.add_conditional_edges("node_a", should_continue)
```
</fix-infinite-loop-needs-exit>

<fix-common-mistakes>
```python
# Add node BEFORE referencing it in edges
builder.add_node("my_node", func)
builder.add_conditional_edges("node_a", router, ["my_node"])

# START is entry-only — cannot route back to it
builder.add_edge("node_a", START)   # WRONG!
builder.add_edge("node_a", "entry") # Use a named entry node instead

# Reducer expects matching types (list, not string)
return {"items": ["item"]}
```
</fix-common-mistakes>

<boundaries>
### What You Should NOT Do

- Mutate state directly — always return partial update dicts from nodes
- Route back to START — it's entry-only; use a named node instead
- Forget reducers on list fields — without one, last write wins
- Mix static edges with Command goto without understanding both will execute
</boundaries>

---

## Deep-Dive References

| Topic | Reference |
|-------|-----------|
| Token filtering, multi-mode, subgraph streaming | `references/streaming-advanced.md` |
| Subgraph composition (invoke-from-node, add-as-node) | `references/subgraph-composition.md` |
| 6 workflow patterns, decision matrix, API comparison | `references/workflow-patterns.md` |
| langgraph.json, Studio, deployment options | `references/local-dev-deployment.md` |
| 3 test levels, FakeListChatModel, pytest-asyncio | `references/testing.md` |

---

## Code Templates

Start from these working examples in `assets/`:

| Template | Use when | Path |
|----------|----------|------|
| State management | Define/refactor graph state schemas | `assets/state-management/` |
| Functional API | @entrypoint + @task patterns | `assets/functional-api/` |
| Command patterns | Dynamic routing, Send fan-out | `assets/command-patterns/` |
| Agent patterns | 6 multi-agent topologies | `assets/agent-patterns/` |
| Streaming | Multi-mode streaming, custom events | `assets/streaming/` |
| Project setup | Bootstrap LangGraph project scaffold | `assets/project-setup/` |
| Dynamic config | Runtime configuration, model swapping | `assets/dynamic-config/` |
| Multi-stage pipeline | Sequential processing stages | `assets/multi-stage-pipeline/` |
| Error handling | Retry, approval, resilience patterns | `assets/error-handling/` |
| Testing | pytest-asyncio test patterns | `assets/testing/` |

---

## Quick Checklist

Before finalizing your implementation:

- [ ] State schema uses TypedDict with proper type annotations?
- [ ] Reducers defined for list/append fields (Annotated[list, operator.add])?
- [ ] Graph compiles without errors (graph = builder.compile(...))?
- [ ] Checkpointer attached if persistence/interrupts needed?
- [ ] Stream mode selected appropriate for use case?
- [ ] Error handling covers tool failures and node exceptions?
- [ ] No circular edges without conditional routing?
- [ ] Thread ID provided for all stateful invocations?
