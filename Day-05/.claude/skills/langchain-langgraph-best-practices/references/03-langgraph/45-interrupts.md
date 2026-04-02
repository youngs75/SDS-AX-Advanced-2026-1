# LangGraph: Interrupts and Human-in-the-Loop

## Read This When
- Need human approval gates or review checkpoints
- Implementing tool call review before execution
- Designing input validation loops with user feedback

## Skip This When
- Flow is fully automated with no human checkpoints
- Only need basic checkpointing without pausing

## Official References
1. https://docs.langchain.com/oss/python/langgraph/interrupts
   - Why: interrupt patterns, rules, and Command(resume=) semantics.

## Core Guidance

### 1. `interrupt()` Function
- Pauses graph execution and returns a value to the caller
- Caller resumes with `Command(resume=value)` which becomes the return value of `interrupt()`
- Requires a checkpointer (state must be saved to resume later)

```python
from langgraph.types import interrupt, Command

async def approval_node(state: MyState) -> Command[Literal["execute", "reject"]]:
    action = state["proposed_action"]
    decision = interrupt(f"Approve action: {action}?")

    if decision == "approve":
        return Command(goto="execute")
    return Command(goto="reject", update={"reason": decision})
```

### 2. Five Interrupt Patterns

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Approval** | Gate before sensitive action | "Approve sending email?" |
| **Review & Edit** | Human reviews/modifies state | "Edit this draft before sending" |
| **Tool Call Review** | Review before tool execution | "About to call delete_user(id=5)" |
| **Input Validation** | Retry loop with human correction | "Invalid date format, try again" |
| **Static Interrupt** | Debug: pause at specific node | `interrupt_before=["node_name"]` |

### 3. Critical Rules

| Rule | Why |
|------|-----|
| **NEVER wrap `interrupt()` in try/except** | Interrupt uses exception mechanism internally |
| **Maintain consistent interrupt order** | Replay must hit interrupts in same sequence |
| **JSON-serializable values only** | Values are stored in checkpoints |
| **Pre-interrupt code must be idempotent** | Node re-executes fully on resume |
| **No conditional skipping of interrupts** | Would break deterministic replay |
| **Entire node re-executes on resume** | Code before interrupt runs again |

### 4. Approval Workflow Example
```python
from langgraph.types import interrupt, Command

async def human_review(state: MyState) -> Command[Literal["proceed", "cancel"]]:
    # This code re-executes on resume — must be idempotent
    summary = format_summary(state["results"])

    decision = interrupt({
        "question": "Do you approve this action?",
        "summary": summary,
    })

    if decision["approved"]:
        return Command(goto="proceed")
    return Command(goto="cancel", update={"feedback": decision.get("reason")})

# Caller resumes:
result = await graph.ainvoke(
    Command(resume={"approved": True}),
    config=config,
)
```

### 5. Static Interrupts (Debugging Only)
```python
# Pause BEFORE a node executes
graph = builder.compile(
    checkpointer=saver,
    interrupt_before=["risky_node"],
)

# Pause AFTER a node executes
graph = builder.compile(
    checkpointer=saver,
    interrupt_after=["review_node"],
)
```
- Use for debugging and step-by-step inspection
- For production HITL, prefer `interrupt()` function inside nodes

### 6. Resuming After Interrupt
```python
# First invocation — hits interrupt, pauses
result = await graph.ainvoke(input, config=config)
# result contains the interrupt value

# Resume with user's decision
result = await graph.ainvoke(
    Command(resume="approve"),
    config=config,  # same thread_id
)
```

### 7. LangChain vs LangGraph HITL Comparison

| Feature | `HumanInTheLoopMiddleware` (langchain) | `interrupt()` (langgraph) |
|---------|---------------------------------------|--------------------------|
| Level | Framework (create_agent) | Runtime (graph nodes) |
| Granularity | Tool-call level | Any point in any node |
| Configuration | Middleware parameter | Code in node function |
| Flexibility | Preset patterns | Full custom logic |
| Use when | Standard tool approval | Custom approval flows |

## Quick Checklist
- [ ] `interrupt()` never wrapped in try/except?
- [ ] Pre-interrupt code is idempotent (safe to re-execute)?
- [ ] Interrupt order is deterministic (no conditional skipping)?
- [ ] Checkpointer configured (required for interrupt)?
- [ ] Resume values are validated before use?

## Next File
- `50-time-travel.md`
