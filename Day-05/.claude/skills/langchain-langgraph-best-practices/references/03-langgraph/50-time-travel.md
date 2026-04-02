# LangGraph: Time Travel

## Read This When
- Debugging state mutations across supersteps
- Need to resume from or modify past checkpoints
- Exploring alternative execution paths (forking)

## Skip This When
- No persistence configured (checkpointer required)
- Flow is simple enough to debug with logging

## Official References
1. https://docs.langchain.com/oss/python/langgraph/use-time-travel
   - Why: checkpoint history, resume, state modification, debugging workflows.

## Core Guidance

### 1. Checkpoint History
Every superstep creates a checkpoint. Browse them in reverse chronological order:

```python
config = {"configurable": {"thread_id": "session_1"}}

async for snapshot in graph.aget_state_history(config):
    print(f"Step: {snapshot.metadata['step']}")
    print(f"Node: {snapshot.metadata.get('source', 'start')}")
    print(f"Checkpoint ID: {snapshot.config['configurable']['checkpoint_id']}")
    print(f"State: {snapshot.values}")
    print("---")
```

### 2. Resume from Past Checkpoint
Replay execution from any historical checkpoint:

```python
# Find the checkpoint you want
target_checkpoint_id = "checkpoint_abc123"

# Resume from that point
result = await graph.ainvoke(
    None,  # no new input — resume existing state
    config={
        "configurable": {
            "thread_id": "session_1",
            "checkpoint_id": target_checkpoint_id,
        }
    },
)
```

### 3. Modify State Before Resume
Change state at a checkpoint, then continue execution with modified state:

```python
# 1. Update state at specific checkpoint
config = {"configurable": {"thread_id": "session_1", "checkpoint_id": "cp_123"}}

await graph.aupdate_state(
    config,
    values={"messages": [HumanMessage(content="Corrected input")]},
    as_node="input_node",  # pretend this update came from this node
)

# 2. Resume from the modified state
result = await graph.ainvoke(None, config=config)
```

### 4. Fork Execution
Create alternative paths by modifying state and resuming:

1. Run graph → observe result
2. Browse history → find decision point
3. Modify state at that point (different input, corrected data)
4. Resume → explore what-if scenario

This creates a **new branch** — the original history is preserved.

### 5. Debugging Workflow
```
Step 1: Run graph normally
  └─ result = await graph.ainvoke(input, config)

Step 2: Inspect — something went wrong
  └─ async for s in graph.aget_state_history(config): ...

Step 3: Identify — find the bad state transition
  └─ "Step 3 node returned wrong classification"

Step 4: Modify — fix state at that checkpoint
  └─ await graph.aupdate_state(config_at_step_2, corrected_values, as_node="classifier")

Step 5: Resume — re-run from corrected state
  └─ result = await graph.ainvoke(None, config_at_step_2)
```

### 6. `as_node` Parameter
- `update_state(config, values, as_node="node_name")` — pretend the update came from this node
- Graph will continue execution from the **next node** after `as_node`
- If `as_node` has conditional edges, the routing function runs on the updated state

### 7. Practical Tips
- Use `stream_mode="debug"` during development for maximum execution detail
- Combine with LangGraph Studio for visual checkpoint browsing
- Time travel works with both Graph API and Functional API
- Each `update_state` creates a new checkpoint (doesn't modify history)

## Quick Checklist
- [ ] Checkpointer configured (required for time travel)?
- [ ] Can reproduce issue from specific checkpoint?
- [ ] `as_node` matches actual node in graph?
- [ ] State modification tested before full resume?
- [ ] Original history preserved (forking, not overwriting)?

## Next File
- `55-streaming.md`
