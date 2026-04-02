# LangGraph: Testing

## Read This When
- Writing tests for LangGraph nodes, graphs, or workflows
- Defining release criteria and test strategies
- Need patterns for deterministic testing with FakeListChatModel

## Skip This When
- Still in conceptual architecture phase
- Question is about production monitoring (see `75-observability.md`)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/test
   - Why: node testing, graph testing, partial execution, checkpointer testing.

## Core Guidance

### 1. Three Test Levels

| Level | What | How | Catches |
|-------|------|-----|---------|
| **Node (unit)** | Individual node logic | Invoke node function directly | Logic errors, schema issues |
| **Graph (integration)** | Routing and flow | Invoke compiled graph end-to-end | Wrong routing, state corruption |
| **Recovery (resilience)** | Retry, interrupt, resume | Use checkpointer + interrupt patterns | Broken recovery, lost state |

### 2. Node-Level Testing
Test individual nodes in isolation:

```python
import pytest
from langchain_core.language_models import FakeListChatModel

@pytest.mark.asyncio
async def test_classify_node():
    model = FakeListChatModel(responses=["support"])
    state = {"messages": [HumanMessage(content="Help me!")], "category": ""}

    result = await classify_node(state)
    assert result["category"] == "support"
```

### 3. Graph-Level Testing
Test the complete flow with deterministic model:

```python
@pytest.mark.asyncio
async def test_full_flow():
    model = FakeListChatModel(responses=["classify:support", "Here's help!"])
    graph = build_graph(model)

    result = await graph.ainvoke({
        "messages": [HumanMessage(content="I need help")]
    })

    assert len(result["messages"]) >= 2
    assert "help" in result["messages"][-1].content.lower()
```

### 4. Partial Execution Testing
Test specific graph sections using `update_state` and `interrupt_after`:

```python
@pytest.mark.asyncio
async def test_from_middle():
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer, interrupt_after=["classify"])

    config = {"configurable": {"thread_id": "test_1"}}

    # Run until interrupt
    result = await graph.ainvoke(input, config=config)

    # Inject custom state and continue
    await graph.aupdate_state(config, {"category": "billing"}, as_node="classify")
    result = await graph.ainvoke(None, config=config)

    assert result["category"] == "billing"
```

### 5. FakeListChatModel for Determinism
- Returns pre-defined responses in sequence
- No API calls — fast, free, deterministic
- Supports tool_calls via structured response strings

```python
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(responses=[
    "I'll search for that.",          # First call
    "Here are the results: ...",       # Second call
])
```

### 6. Checkpoint State Assertions
Verify state at specific supersteps:

```python
@pytest.mark.asyncio
async def test_state_progression():
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test_2"}}

    await graph.ainvoke(input, config=config)

    # Check state history
    states = []
    async for snapshot in graph.aget_state_history(config):
        states.append(snapshot)

    # Verify progression (newest first)
    assert states[0].values["status"] == "complete"
    assert states[-1].values["status"] == "pending"
```

### 7. pytest-asyncio Setup
```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-asyncio>=0.21"]
```

### 8. Release Gate Checklist
Before deploying:
1. All node-level unit tests pass
2. All flow routing tests pass
3. Retry/recovery paths tested
4. Interrupt/resume cycles verified
5. State never corrupted across supersteps
6. FakeListChatModel used (no live API in CI)

## Quick Checklist
- [ ] Unit tests per node function?
- [ ] Flow tests for routing decisions?
- [ ] Recovery tests for retry/resume?
- [ ] FakeListChatModel for deterministic tests?
- [ ] pytest-asyncio configured?

## Next File
- `75-observability.md`
