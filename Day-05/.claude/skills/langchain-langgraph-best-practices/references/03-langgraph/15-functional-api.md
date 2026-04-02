# LangGraph: Functional API

## Read This When
- Adapting procedural Python code to LangGraph
- Prefer Python control flow (if/for/while) over DAG topology
- Building simple linear or sequential workflows

## Skip This When
- Need explicit graph topology with visualization
- Complex branching with concurrent state coordination

## Official References
1. https://docs.langchain.com/oss/python/langgraph/functional-api
   - Why: @entrypoint and @task decorator semantics.
2. https://docs.langchain.com/oss/python/langgraph/use-functional-api
   - Why: practical patterns including retry, caching, chatbots.

## Core Guidance

### 1. `@entrypoint` Decorator
- Marks the workflow starting point — produces a Pregel instance (same runtime as Graph API)
- Single positional argument (the input), must be JSON-serializable
- Return value must be JSON-serializable

```python
from langgraph.func import entrypoint, task

@entrypoint(checkpointer=MemorySaver())
async def my_workflow(input: dict) -> str:
    result = await step_one(input["query"]).result()
    return result
```

### 2. `@task` Decorator
- Discrete unit of work — like a node in Graph API
- Returns a future-like object; call `.result()` to get the value
- Output is checkpointed for durability and replay

```python
@task
async def analyze(query: str) -> str:
    response = await model.ainvoke([HumanMessage(content=query)])
    return response.content
```

### 3. Injectable Parameters
| Parameter | Purpose | Annotation |
|-----------|---------|------------|
| `previous` | Short-term memory (last entrypoint return) | `previous: Any` |
| `store` | Long-term cross-thread storage | `store: BaseStore` |
| `writer` | Custom streaming events | `writer: StreamWriter` |
| `config` | Runtime configuration | `config: RunnableConfig` |

```python
@entrypoint(checkpointer=saver)
async def chatbot(messages: list, *, previous: list = []) -> list:
    all_messages = previous + messages
    response = await model.ainvoke(all_messages)
    return all_messages + [response]
```

### 4. `entrypoint.final`
- Decouple the return value (sent to caller) from the saved checkpoint value
- `entrypoint.final(value=return_val, save=checkpoint_val)`
- Useful when you want to return a summary but save full state

### 5. Determinism Rules
- Wrap ALL side effects inside `@task` — they are the durable boundary
- Keep consistent interrupt ordering (no conditional skipping)
- Non-deterministic operations (API calls, time, random) must be in tasks so they are replayed from checkpoint, not re-executed

### 6. Parallel Tasks
```python
@entrypoint(checkpointer=saver)
async def parallel_workflow(queries: list[str]) -> list[str]:
    futures = [search(q) for q in queries]
    results = [f.result() for f in futures]
    return results
```
- Tasks launched before `.result()` is called can run concurrently
- Each task is independently checkpointed

### 7. Retry and Caching
```python
@task(retry=RetryPolicy(max_attempts=3))
async def fetch_data(url: str) -> dict:
    ...

@task(cache=CachePolicy(ttl=300))  # 5 min cache
async def expensive_compute(data: str) -> str:
    ...
```

## Quick Checklist
- [ ] All side effects wrapped in `@task`?
- [ ] Returns are JSON-serializable?
- [ ] `previous` used for short-term memory?
- [ ] Interrupt order deterministic (no conditional skipping)?
- [ ] `.result()` called to materialize task outputs?

## Next File
- `20-choosing-apis.md`
