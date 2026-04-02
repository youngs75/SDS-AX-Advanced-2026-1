# Runnable Composition and State Types

## Read This When

- You need consistent execution interfaces (`invoke`, `stream`, `batch`).
- You need to define state typing boundaries before orchestration.

## Skip This When

- You are selecting workflow patterns at graph level.

## Official References

1. https://reference.langchain.com/python/langchain_core/runnables/
   - Why: canonical Runnable APIs and config model.
2. https://docs.langchain.com/oss/python/langgraph/graph-api
   - Why: state schema + reducer semantics in orchestration.

## Core Guidance

### 1. Runnable Protocol — Universal Execution Interface

Every LangChain component implements the Runnable protocol, providing standardized execution methods:

| Method | Behavior | Use When |
|--------|----------|----------|
| `invoke(input)` | Sync, single input → single output | Simple synchronous calls, scripts |
| `ainvoke(input)` | Async version of invoke | Production (recommended) |
| `stream(input)` | Sync, yields output chunks | Real-time display, CLI tools |
| `astream(input)` | Async streaming | Production streaming |
| `batch(inputs)` | Sync, multiple inputs in parallel | Batch processing |
| `abatch(inputs)` | Async batch | Production batch processing |

**Key principle**: Choose async methods (`ainvoke`, `astream`, `abatch`) for production systems to avoid blocking the event loop.

### 2. RunnableConfig — Runtime Configuration

`RunnableConfig` carries metadata and execution parameters across the chain:

```python
config = {
    "configurable": {"thread_id": "abc", "model": "gpt-4"},  # runtime params
    "tags": ["production", "user_query"],                    # tracing tags
    "metadata": {"user_id": "123"},                          # custom metadata
    "callbacks": [my_handler],                               # lifecycle hooks
    "run_name": "weather_query",                            # trace name
    "max_concurrency": 5,                                   # parallel limit
}
result = await runnable.ainvoke(input, config=config)
```

**Key fields**:
- `configurable`: Runtime parameters that change behavior (thread IDs, model selection)
- `tags`: Categorization for tracing and filtering
- `metadata`: Custom context for observability
- `callbacks`: Lifecycle event handlers for logging, metrics
- `run_name`: Human-readable trace identifier
- `max_concurrency`: Limit parallel execution in batch operations

### 3. LCEL Pipe Operator — Composing Runnables

The `|` operator chains runnables sequentially, passing output to next input:

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel

# Sequential pipe
chain = prompt | model | output_parser

# Lambda wrapper for custom logic
cleanup = RunnableLambda(lambda x: x.strip())
chain = prompt | model | cleanup

# Passthrough (forward input unchanged)
chain = RunnablePassthrough() | model

# Parallel execution (fan-out)
chain = RunnableParallel(
    summary=summarize_chain,
    translation=translate_chain,
)
# Returns: {"summary": ..., "translation": ...}
```

**When to use LCEL**:
- Simple linear pipelines (prompt → model → parser)
- Stateless transformations
- No conditional branching or loops

**When NOT to use LCEL**:
- Complex control flow (use LangGraph)
- State persistence required (use LangGraph with checkpointing)
- Multi-step reasoning with backtracking (use LangGraph)

### 4. State Typing for Graphs

LangGraph requires explicit state schemas using `TypedDict` or Pydantic `BaseModel`:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# TypedDict with reducer annotation
class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only message list
    step_count: int                          # overwrite on each update
    context: dict                            # overwrite on each update

# Pydantic BaseModel alternative (recommended for validation)
from pydantic import BaseModel, Field

class MyState(BaseModel):
    messages: Annotated[list, add_messages]
    step_count: int = 0
    context: dict = Field(default_factory=dict)
```

**Key differences**:
- `TypedDict`: Lightweight, no runtime validation
- `BaseModel`: Runtime validation, default values, serialization

### 5. Reducer Annotations — Controlling Update Behavior

Annotations control how state updates are merged:

```python
# APPEND behavior (add_messages)
messages: Annotated[list, add_messages]
# Update: state["messages"] = existing_messages + new_messages

# OVERWRITE behavior (no annotation)
step_count: int
# Update: state["step_count"] = new_value
```

**Common reducers**:
- `add_messages`: Append messages to list (deduplicates by message ID)
- Custom reducer: `Annotated[list, lambda old, new: old + new]`

**Critical rule**: Without annotation, updates OVERWRITE the entire value. Annotate lists/dicts that need merge semantics.

### 6. Durability and Runtime Guarantees

**Important**: Runnable composition alone does NOT provide:
- State persistence across failures
- Checkpointing for recovery
- Multi-turn conversation memory

For these features, use LangGraph with a checkpointer (MemorySaver, SqliteSaver, PostgresSaver).

## Asset Examples

| Topic | Asset Path |
|-------|-----------|
| Runnable protocol (invoke/stream/batch), LCEL pipe, RunnableLambda, RunnableParallel | `assets/01-langchain-core/runnable-patterns/` |

## Quick Checklist

- [ ] Are execution modes chosen intentionally (ainvoke preferred)?
- [ ] Is RunnableConfig usage consistent across the pipeline?
- [ ] Are state keys and reducers explicitly defined?
- [ ] Are LCEL chains used only for simple pipelines (complex → LangGraph)?
- [ ] Is async/await used consistently (no mixing sync and async)?

## Next File

- Core readiness checklist: `40-core-best-practices.md`
