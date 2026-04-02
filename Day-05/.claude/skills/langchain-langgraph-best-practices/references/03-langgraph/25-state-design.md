# LangGraph: State Design and Terminology

## Read This When
- Designing state schema before building nodes
- Need node classification methodology
- Need LangGraph terminology reference

## Skip This When
- State is trivial (chat-only with MessagesState)
- Question is about runtime behavior, not graph design

## Official References
1. https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
   - Why: workflow mapping, node classification, state design methodology.
2. https://docs.langchain.com/oss/python/langgraph/graph-api
   - Why: state schema mechanics and reducer semantics.

## Core Guidance

### 1. Five-Step Design Methodology
1. **Map the process**: Identify all operations from input to output
2. **Classify operations**: Categorize each as node type (see below)
3. **Design state**: Define what persists between steps
4. **Implement nodes**: Write async functions for each operation
5. **Assemble graph**: Wire nodes with edges and conditions

### 2. Node Classification Types

| Node Type | Description | Example |
|-----------|-------------|---------|
| LLM Step | Model invocation for reasoning/generation | Summarize, classify, generate |
| Data Step | Retrieval, transformation, formatting | Fetch documents, parse JSON |
| Action Step | External side effect | Send email, write DB, call API |
| User Input Step | Requires human interaction | Approval, feedback, correction |

**Design rule**: Each node = exactly ONE type. Mixed nodes are harder to test, retry, and debug.

### 3. State Design Rules
- **Store raw data**, format on-demand in nodes — raw data is more flexible
- **Keep fields that persist across steps** — ephemeral data stays in node local vars
- **Use descriptive key names** — state keys are your API contract
- **Version state schemas** — changing keys breaks existing checkpoints

```python
from typing import TypedDict, Annotated
from typing_extensions import NotRequired
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]  # conversation history
    query: str                                # original user query
    sources: Annotated[list[dict], operator.add]  # accumulated sources
    summary: NotRequired[str]                 # generated only at end
```

### 4. Key Terminology

| Term | Definition |
|------|-----------|
| **State** | Shared data passed between nodes; defined as TypedDict/BaseModel |
| **Node** | Async function that reads state and returns partial update |
| **Edge** | Connection between nodes; normal or conditional |
| **START / END** | Special nodes marking graph entry and termination |
| **Reducer** | Annotation controlling how updates merge (append vs overwrite) |
| **Superstep** | One full round of node execution + state update |
| **Thread ID** | Unique identifier isolating one conversation/session |
| **Checkpoint** | Snapshot of state saved after each superstep |
| **Interrupt** | Pause point for human input or external event |
| **Command** | Object combining routing + state update + interrupt resume |
| **Send** | Object for dynamic fan-out to parallel node executions |

### 5. MessagesState as Starting Point
```python
from langgraph.graph import MessagesState
# Equivalent to: class State(TypedDict): messages: Annotated[list, add_messages]

# Extend when you need more fields:
class MyState(MessagesState):
    user_id: NotRequired[str]
    tool_results: NotRequired[dict]
```

### 6. Reducer Selection Guide

| Data Pattern | Recommended Reducer | Why |
|-------------|-------------------|-----|
| Conversation messages | `add_messages` | Deduplicates by ID, handles RemoveMessage |
| Accumulating results | `operator.add` | Simple append |
| Latest value only | (no annotation) | Default overwrite |
| Merge dicts | Custom `lambda a, b: {**a, **b}` | Shallow merge |
| Counter | Custom `lambda a, b: a + b` | Numeric accumulation |

### 7. Node Granularity Trade-offs
- **Fine-grained** (many small nodes): Easier testing, better retry granularity, clearer traces
- **Coarse** (few large nodes): Fewer state transitions, less checkpoint overhead, simpler graph
- **Rule of thumb**: Split when a node does two things that could fail independently

## Quick Checklist
- [ ] State stores raw data (not formatted text)?
- [ ] Node classification documented for each node?
- [ ] Reducers explicit for all shared/accumulating channels?
- [ ] State keys stable and descriptively named?
- [ ] Node granularity balanced (testable but not excessive)?

## Next File
- `30-command-and-send.md`
