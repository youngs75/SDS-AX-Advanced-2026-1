# LangGraph: Durable Execution

## Read This When
- Building long-running or failure-prone workflows
- Need replay safety and deterministic execution guarantees
- Choosing between durability modes

## Skip This When
- Flow is stateless, short-lived, and failure is acceptable
- Only need basic checkpointing (see `40-persistence.md`)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/durable-execution
   - Why: durability modes, determinism, replay mechanics.
2. https://docs.langchain.com/oss/python/langgraph/pregel
   - Why: underlying execution model (Plan-Execute-Update cycle).

## Core Guidance

### 1. Three Durability Modes

| Mode | Persistence Timing | Trade-off | Best For |
|------|-------------------|-----------|----------|
| `exit` | On graph exit only | Fastest, least durable | Short tasks, development |
| `async` | Async during next step | Balanced speed/safety | Most production workloads |
| `sync` | Sync before next step | Slowest, safest | Critical financial/legal workflows |

### 2. Determinism Requirement
- Replayed workflows MUST produce identical results
- LangGraph replays from the last checkpoint to rebuild state
- Non-deterministic code (API calls, time, random) outside tasks/nodes will produce different results on replay

### 3. Replay Mechanics
1. System identifies the **starting point** (last successful checkpoint)
2. All steps from that point are **replayed** in order
3. Tasks/nodes that were already checkpointed return cached results
4. Only un-checkpointed work is actually re-executed

**Starting point differs by API**:
- Graph API: Replay starts at the beginning of the last incomplete **node**
- Functional API: Replay starts at the beginning of the **entrypoint**

### 4. Replay-Safe Code Rules

**Do**: Wrap side effects in tasks or separate nodes
```python
# Graph API — side effect isolated in its own node
async def send_email_node(state: MyState) -> dict:
    await send_email(state["recipient"], state["body"])
    return {"email_sent": True}

# Functional API — side effect in @task
@task
async def send_email_task(recipient: str, body: str) -> bool:
    await send_email(recipient, body)
    return True
```

**Don't**: Put side effects in routing logic or reducers
```python
# WRONG — side effect in conditional edge function
def route(state):
    send_notification()  # Replayed on every recovery!
    return "next_node"
```

### 5. Idempotency Strategies

| Strategy | How | When |
|----------|-----|------|
| Upsert | Use `INSERT ... ON CONFLICT UPDATE` | Database writes |
| Idempotency key | Pass unique key with API request | External API calls |
| Check-before-act | Verify state before performing action | Any side effect |
| Deduplication | Track completed operation IDs in state | Message/event processing |

### 6. Configuration
```python
from langgraph.checkpoint.memory import InMemorySaver

graph = builder.compile(checkpointer=InMemorySaver())

# Invoke with thread for durability
result = await graph.ainvoke(
    {"messages": [HumanMessage(content="Start")]},
    config={"configurable": {"thread_id": "session_1"}},
)
```
Note: Durability mode is determined by the checkpointer implementation. `InMemorySaver` uses `exit` mode. Production checkpointers (PostgresSaver) support `async` and `sync` modes.

## Quick Checklist
- [ ] Side effects idempotent or wrapped in tasks/nodes?
- [ ] Non-deterministic operations isolated from routing logic?
- [ ] Durability mode chosen for use case (exit/async/sync)?
- [ ] Can the flow safely replay from any checkpoint?
- [ ] External API calls use idempotency keys?

## Next File
- `40-persistence.md`
