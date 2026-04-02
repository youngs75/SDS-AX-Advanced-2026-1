# Advanced Interrupt Patterns

## Read This When

- You need parallel interrupts across multiple branches
- You need to resume multiple interrupts at once with a resume map
- You need to understand subgraph interrupt re-execution behavior

## Skip This When

- You only need a single interrupt/resume (covered in SKILL.md)
- You are building a simple approval workflow

---

## Multiple Interrupts (Parallel Branches)

When parallel branches each call `interrupt()`, all interrupts are collected. Resume them all in a single invocation by mapping each interrupt ID to its resume value.

### Python

```python
from typing import Annotated, TypedDict
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt

class State(TypedDict):
    vals: Annotated[list[str], operator.add]

def node_a(state):
    answer = interrupt("question_a")
    return {"vals": [f"a:{answer}"]}

def node_b(state):
    answer = interrupt("question_b")
    return {"vals": [f"b:{answer}"]}

graph = (
    StateGraph(State)
    .add_node("a", node_a)
    .add_node("b", node_b)
    .add_edge(START, "a")
    .add_edge(START, "b")
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Both parallel nodes hit interrupt() and pause
result = graph.invoke({"vals": []}, config)
# result["__interrupt__"] contains both Interrupt objects with IDs

# Resume all pending interrupts at once using a map of id -> value
resume_map = {
    i.id: f"answer for {i.value}"
    for i in result["__interrupt__"]
}
result = graph.invoke(Command(resume=resume_map), config)
# result["vals"] = ["a:answer for question_a", "b:answer for question_b"]
```

### TypeScript

```typescript
import { Command, END, MemorySaver, START, StateGraph, interrupt, isInterrupted, INTERRUPT, Annotation } from "@langchain/langgraph";

const State = Annotation.Root({
  vals: Annotation<string[]>({
    reducer: (left, right) => left.concat(Array.isArray(right) ? right : [right]),
    default: () => [],
  }),
});

function nodeA(_state: typeof State.State) {
  const answer = interrupt("question_a") as string;
  return { vals: [`a:${answer}`] };
}

function nodeB(_state: typeof State.State) {
  const answer = interrupt("question_b") as string;
  return { vals: [`b:${answer}`] };
}

const graph = new StateGraph(State)
  .addNode("a", nodeA)
  .addNode("b", nodeB)
  .addEdge(START, "a")
  .addEdge(START, "b")
  .addEdge("a", END)
  .addEdge("b", END)
  .compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: "1" } };

const interruptedResult = await graph.invoke({ vals: [] }, config);

// Resume all pending interrupts at once
const resumeMap: Record<string, string> = {};
if (isInterrupted(interruptedResult)) {
  for (const i of interruptedResult[INTERRUPT]) {
    if (i.id != null) {
      resumeMap[i.id] = `answer for ${i.value}`;
    }
  }
}
const result = await graph.invoke(new Command({ resume: resumeMap }), config);
// result.vals = ["a:answer for question_a", "b:answer for question_b"]
```

---

## Resume Map Pattern

Key mechanics:

1. Each `Interrupt` object has a unique `.id` property
2. Use `Command(resume={id: value, ...})` to resume multiple interrupts at once
3. All pending interrupts must be resumed in a single `invoke` call
4. If you only resume some interrupts, the others remain pending

### Building the Resume Map

```python
# Generic resume map builder
def build_resume_map(interrupt_result, resolver_fn):
    """Build a resume map from interrupt results.

    resolver_fn: (interrupt_value) -> resume_value
    """
    return {
        i.id: resolver_fn(i.value)
        for i in interrupt_result["__interrupt__"]
    }

# Example: auto-approve all
resume_map = build_resume_map(result, lambda v: {"approved": True})
```

---

## Subgraph Interrupt Re-execution

When a subgraph contains an `interrupt()`, resuming re-executes **BOTH**:
1. The **parent node** (that invoked the subgraph)
2. The **subgraph node** (that called `interrupt()`)

### Python

```python
def node_in_parent_graph(state: State):
    some_code()  # <-- Re-executes on resume
    subgraph_result = subgraph.invoke(some_input)
    # ...

def node_in_subgraph(state: State):
    some_other_code()  # <-- Also re-executes on resume
    result = interrupt("What's your name?")
    # ...
```

### TypeScript

```typescript
async function nodeInParentGraph(state: State) {
  someCode();  // <-- Re-executes on resume
  const subgraphResult = await subgraph.invoke(someInput);
  // ...
}

async function nodeInSubgraph(state: State) {
  someOtherCode();  // <-- Also re-executes on resume
  const result = interrupt("What's your name?");
  // ...
}
```

### Implications

- ALL code paths leading to the interrupt re-run on resume
- Both parent and child nodes must be **idempotent** before `interrupt()`
- Use upsert patterns, not insert patterns, before any interrupt in a subgraph chain
- Consider placing side effects in separate nodes **after** the interrupt node

---

## Nested Subgraph Interrupts

For deeply nested subgraphs (grandchild graphs), the entire call chain re-executes:

```
parent_node() → child_graph.invoke() → grandchild_node() → interrupt()
```

On resume: `parent_node` re-runs → `child_graph.invoke()` re-runs → `grandchild_node` re-runs → `interrupt()` returns resume value.

**Best practice**: Keep interrupt-containing subgraphs shallow (max 2 levels) to limit re-execution scope.
