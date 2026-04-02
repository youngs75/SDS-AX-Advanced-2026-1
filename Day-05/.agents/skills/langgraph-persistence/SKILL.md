---
name: langgraph-persistence
description: "INVOKE THIS SKILL when your LangGraph needs to persist state, remember conversations, travel through history, or configure subgraph checkpointer scoping. Covers checkpointers, thread_id, time travel, Store, and subgraph persistence modes."
---

<overview>
LangGraph's persistence layer enables durable execution by checkpointing graph state:

- **Checkpointer**: Saves/loads graph state at every super-step
- **Thread ID**: Identifies separate checkpoint sequences (conversations)
- **Store**: Cross-thread memory for user preferences, facts

**Two memory types:**
- **Short-term** (checkpointer): Thread-scoped conversation history
- **Long-term** (store): Cross-thread user preferences, facts
</overview>

<checkpointer-selection>

| Checkpointer | Use Case | Production Ready |
|--------------|----------|------------------|
| `InMemorySaver` | Testing, development | No |
| `SqliteSaver` | Local development | Partial |
| `PostgresSaver` | Production | Yes |

</checkpointer-selection>

---

## Checkpointer Setup

<ex-basic-persistence>
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]

def add_message(state: State) -> dict:
    return {"messages": ["Bot response"]}

checkpointer = InMemorySaver()

graph = (
    StateGraph(State)
    .add_node("respond", add_message)
    .add_edge(START, "respond")
    .add_edge("respond", END)
    .compile(checkpointer=checkpointer)  # Pass at compile time
)

# ALWAYS provide thread_id
config = {"configurable": {"thread_id": "conversation-1"}}

result1 = graph.invoke({"messages": ["Hello"]}, config)
print(len(result1["messages"]))  # 2

result2 = graph.invoke({"messages": ["How are you?"]}, config)
print(len(result2["messages"]))  # 4 (previous + new)
```
TypeScript: `new MemorySaver()`, `.compile({ checkpointer })`, `{ configurable: { thread_id: "..." } }`.
</ex-basic-persistence>

<ex-production-postgres>
<python>
Configure PostgreSQL-backed checkpointing for production deployments.
```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
) as checkpointer:
    checkpointer.setup()  # only needed on first use to create tables
    graph = builder.compile(checkpointer=checkpointer)
```
</python>
<typescript>
Configure PostgreSQL-backed checkpointing for production deployments.
```typescript
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";

const checkpointer = PostgresSaver.fromConnString(
  "postgresql://user:pass@localhost/db"
);
await checkpointer.setup(); // only needed on first use to create tables

const graph = builder.compile({ checkpointer });
```
</typescript>
</ex-production-postgres>

---

## Thread Management

<ex-separate-threads>
<python>
Demonstrate isolated state between different thread IDs.
```python
# Different threads maintain separate state
alice_config = {"configurable": {"thread_id": "user-alice"}}
bob_config = {"configurable": {"thread_id": "user-bob"}}

graph.invoke({"messages": ["Hi from Alice"]}, alice_config)
graph.invoke({"messages": ["Hi from Bob"]}, bob_config)

# Alice's state is isolated from Bob's
```
</python>
<typescript>
Demonstrate isolated state between different thread IDs.
```typescript
// Different threads maintain separate state
const aliceConfig = { configurable: { thread_id: "user-alice" } };
const bobConfig = { configurable: { thread_id: "user-bob" } };

await graph.invoke({ messages: [new HumanMessage("Hi from Alice")] }, aliceConfig);
await graph.invoke({ messages: [new HumanMessage("Hi from Bob")] }, bobConfig);

// Alice's state is isolated from Bob's
```
</typescript>
</ex-separate-threads>

---

## State History & Time Travel

<ex-resume-from-checkpoint>
```python
config = {"configurable": {"thread_id": "session-1"}}

result = graph.invoke({"messages": ["start"]}, config)

# Browse checkpoint history
states = list(graph.get_state_history(config))

# Replay from a past checkpoint
past = states[-2]
result = graph.invoke(None, past.config)  # None = resume from checkpoint

# Or fork: update state at a past checkpoint, then resume
fork_config = graph.update_state(past.config, {"messages": ["edited"]})
result = graph.invoke(None, fork_config)
```
TypeScript: `graph.getStateHistory(config)` (async iterable), `graph.invoke(null, past.config)`, `graph.updateState(past.config, {...})`.
</ex-resume-from-checkpoint>

<ex-update-state>
<python>
Manually update graph state before resuming execution.
```python
config = {"configurable": {"thread_id": "session-1"}}

# Modify state before resuming
graph.update_state(config, {"data": "manually_updated"})

# Resume with updated state
result = graph.invoke(None, config)
```
</python>
<typescript>
Manually update graph state before resuming execution.
```typescript
const config = { configurable: { thread_id: "session-1" } };

// Modify state before resuming
await graph.updateState(config, { data: "manually_updated" });

// Resume with updated state
const result = await graph.invoke(null, config);
```
</typescript>
</ex-update-state>

---

## Subgraph Checkpointer Scoping

When compiling a subgraph, the `checkpointer` parameter controls persistence behavior. This is critical for subgraphs that use interrupts, need multi-turn memory, or run in parallel.

<subgraph-checkpointer-scoping-table>

| Feature | `checkpointer=False` | `None` (default) | `True` |
|---|---|---|---|
| Interrupts (HITL) | No | Yes | Yes |
| Multi-turn memory | No | No | Yes |
| Multiple calls (different subgraphs) | Yes | Yes | Warning (namespace conflicts possible) |
| Multiple calls (same subgraph) | Yes | Yes | No |
| State inspection | No | Warning (current invocation only) | Yes |

</subgraph-checkpointer-scoping-table>

<subgraph-checkpointer-when-to-use>

### When to use each mode

- **`checkpointer=False`** — Subgraph doesn't need interrupts or persistence. Simplest option, no checkpoint overhead.
- **`None` (default / omit `checkpointer`)** — Subgraph needs `interrupt()` but not multi-turn memory. Each invocation starts fresh but can pause/resume. Parallel execution works because each invocation gets a unique namespace.
- **`checkpointer=True`** — Subgraph needs to remember state across invocations (multi-turn conversations). Each call picks up where the last left off.

</subgraph-checkpointer-when-to-use>

<warning-stateful-subgraphs-parallel>

**Warning**: Stateful subgraphs (`checkpointer=True`) do NOT support calling the same subgraph instance multiple times within a single node — the calls write to the same checkpoint namespace and conflict.

</warning-stateful-subgraphs-parallel>

<ex-subgraph-checkpointer-modes>
<python>
Choose the right checkpointer mode for your subgraph.
```python
# No interrupts needed — opt out of checkpointing
subgraph = subgraph_builder.compile(checkpointer=False)

# Need interrupts but not cross-invocation persistence (default)
subgraph = subgraph_builder.compile()

# Need cross-invocation persistence (stateful)
subgraph = subgraph_builder.compile(checkpointer=True)
```
</python>
<typescript>
Choose the right checkpointer mode for your subgraph.
```typescript
// No interrupts needed — opt out of checkpointing
const subgraph = subgraphBuilder.compile({ checkpointer: false });

// Need interrupts but not cross-invocation persistence (default)
const subgraph = subgraphBuilder.compile();

// Need cross-invocation persistence (stateful)
const subgraph = subgraphBuilder.compile({ checkpointer: true });
```
</typescript>
</ex-subgraph-checkpointer-modes>

<parallel-subgraph-namespacing>

### Parallel subgraph namespacing

When multiple **different** stateful subgraphs run in parallel, wrap each in its own `StateGraph` with a unique node name for stable namespace isolation:

```python
from langgraph.graph import MessagesState, StateGraph

def create_sub_agent(model, *, name, **kwargs):
    """Wrap an agent with a unique node name for namespace isolation."""
    agent = create_agent(model=model, name=name, **kwargs)
    return (
        StateGraph(MessagesState)
        .add_node(name, agent)  # unique name -> stable namespace
        .add_edge("__start__", name)
        .compile()
    )

fruit_agent = create_sub_agent(
    "gpt-4.1-mini", name="fruit_agent",
    tools=[fruit_info], prompt="...", checkpointer=True,
)
```

Note: Subgraphs added as nodes (via `add_node`) already get name-based namespaces automatically and don't need this wrapper.

</parallel-subgraph-namespacing>

---

## Long-Term Memory (Store)

<ex-long-term-memory-store>
```python
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

store = InMemoryStore()

# Save user preference (available across ALL threads)
store.put(("alice", "preferences"), "language", {"preference": "short responses"})

# Node with store — access via runtime
def respond(state, runtime: Runtime):
    prefs = runtime.store.get((state["user_id"], "preferences"), "language")
    return {"response": f"Using preference: {prefs.value}"}

# Compile with BOTH checkpointer and store
graph = builder.compile(checkpointer=checkpointer, store=store)

# Both threads access same long-term memory
graph.invoke({"user_id": "alice"}, {"configurable": {"thread_id": "thread-1"}})
graph.invoke({"user_id": "alice"}, {"configurable": {"thread_id": "thread-2"}})  # Same preferences!
```
TypeScript: `new MemoryStore()`, `runtime.store?.get(...)`, `.compile({ checkpointer, store })`.
</ex-long-term-memory-store>

<ex-store-operations>
<python>
Basic store operations: put, get, search, and delete.
```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

store.put(("user-123", "facts"), "location", {"city": "San Francisco"})  # Put
item = store.get(("user-123", "facts"), "location")  # Get
results = store.search(("user-123", "facts"), filter={"city": "San Francisco"})  # Search
store.delete(("user-123", "facts"), "location")  # Delete
```
</python>
</ex-store-operations>

---

## Fixes

<fix-thread-id-required>
```python
# WRONG: No thread_id - state NOT persisted!
graph.invoke({"messages": ["Hello"]})

# CORRECT: Always provide thread_id
config = {"configurable": {"thread_id": "session-1"}}
graph.invoke({"messages": ["Hello"]}, config)
graph.invoke({"messages": ["What did I say?"]}, config)  # Remembers!
```
</fix-thread-id-required>

<fix-inmemory-not-for-production>
```python
# WRONG: Data lost on process restart
checkpointer = InMemorySaver()

# CORRECT: Use persistent storage for production
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
```
</fix-inmemory-not-for-production>

<fix-update-state-with-reducers>
```python
from langgraph.types import Overwrite

# update_state PASSES THROUGH reducers
graph.update_state(config, {"items": ["C"]})  # ["A","B","C"] - Appended!

# To REPLACE instead, use Overwrite
graph.update_state(config, {"items": Overwrite(["C"])})  # ["C"] - Replaced
```
TypeScript: `new Overwrite(["C"])` — same semantics.
</fix-update-state-with-reducers>

<fix-store-injection>
```python
# WRONG: Store not available in node
def my_node(state):
    store.put(...)  # NameError!

# CORRECT: Access store via runtime
from langgraph.runtime import Runtime
def my_node(state, runtime: Runtime):
    runtime.store.put(...)  # Correct store instance
```
</fix-store-injection>

<boundaries>
### What You Should NOT Do

- Use `InMemorySaver` in production — data lost on restart; use `PostgresSaver`
- Forget `thread_id` — state won't persist without it
- Expect `update_state` to bypass reducers — it passes through them; use `Overwrite` to replace
- Run the same stateful subgraph (`checkpointer=True`) in parallel within one node — namespace conflict
- Access store directly in a node — use `runtime.store` via the `Runtime` param
</boundaries>

---

## Deep-Dive References

| Topic | Reference |
|-------|-----------|
| 3 durability modes, replay mechanism, idempotency | `references/durable-execution.md` |
| PostgreSQL encryption, Store semantic search, message summarization | `references/production-patterns.md` |
| LangSmith tracing, PII anonymization, debug mode, metrics | `references/observability.md` |

## Code Templates

Start from these working examples in `assets/`:

| Template | Use when | Path |
|----------|----------|------|
| Persistence | Cross-thread Store with namespace key-value | `assets/persistence/` |
| Durable execution | Replay-safe graphs with idempotent side effects | `assets/durable-execution/` |
| Time travel | Checkpoint history, state modification, forked re-execution | `assets/time-travel/` |
| Subgraphs | Nested graph composition with state transformation | `assets/subgraphs/` |

---

## Quick Checklist

- Checkpointer type matches environment (InMemorySaver dev, PostgresSaver prod)?
- Thread IDs are unique and meaningful?
- Store namespaces follow consistent hierarchy?
- Durable execution nodes are idempotent (check-before-act)?
- Time travel used for debugging only (not production flow)?
- Memory cleanup strategy defined (TTL or manual pruning)?
