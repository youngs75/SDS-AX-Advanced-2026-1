# Human-in-the-Loop

## Read This When

- Need to add approval gates to sensitive operations
- Want to understand interrupt flow and decision handling
- Implementing batch tool approval workflows
- Configuring pause points in agent execution
- Building agents that require human oversight

## Skip This When

- Running fully autonomous agents without human oversight
- All operations are low-risk and don't require approval
- Working in environments where pausing is not supported

## Official References

1. https://docs.langchain.com/oss/python/deepagents/human-in-the-loop - Why: Core documentation for interrupt configuration and decision handling patterns
2. https://docs.langchain.com/oss/python/langgraph/interrupts - Why: Underlying LangGraph interrupt mechanism and state management

## Core Guidance

### 1. interrupt_on Configuration

Map tool names to approval requirements:

```python
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_deep_agent(
    model=model,
    interrupt_on={
        "write_file": True,                  # All write_file calls need approval
        "execute": True,                      # All execute calls need approval
        "task": False,                        # Sub-agent calls auto-approved
        "edit_file": {                        # Detailed config
            "type": "tool",
            "action_description": "File modification",
        },
    },
    checkpointer=InMemorySaver(),  # REQUIRED for interrupts
)
```

**Important**: Checkpointer is REQUIRED for interrupt support. Without it, the agent cannot pause and resume.

### 2. Three Decision Types

| Decision | Effect | Resume Payload |
|----------|--------|---------------|
| `approve` | Execute the tool call as-is | `{"decisions": [{"tool_call_id": "...", "action": "approve"}]}` |
| `edit` | Modify tool arguments before executing | `{"decisions": [{"tool_call_id": "...", "action": "edit", "args": {...}}]}` |
| `reject` | Skip the tool call entirely | `{"decisions": [{"tool_call_id": "...", "action": "reject", "message": "Reason"}]}` |

### 3. Resume After Interrupt

Use `Command(resume=...)`:

```python
from langgraph.types import Command

# Agent pauses at interrupt
result = await agent.ainvoke(input, config=config)

# Check for interrupt
if "__interrupt__" in result:
    interrupt_info = result["__interrupt__"]

    # Resume with approval
    result = await agent.ainvoke(
        Command(resume={"decisions": [
            {"tool_call_id": interrupt_info[0]["tool_call_id"], "action": "approve"}
        ]}),
        config=config,
    )
```

### 4. Batch Interrupt Handling

When multiple tool calls trigger interrupts simultaneously:

```python
# Agent generates 3 write_file calls at once
# All 3 are presented for review together
decisions = [
    {"tool_call_id": "call_1", "action": "approve"},
    {"tool_call_id": "call_2", "action": "edit", "args": {"content": "modified content"}},
    {"tool_call_id": "call_3", "action": "reject", "message": "Not needed"},
]
result = await agent.ainvoke(Command(resume={"decisions": decisions}), config=config)
```

### 5. Sub-Agent Interrupt Propagation

When a sub-agent hits an interrupt, it propagates to the parent:

- Sub-agent's `interrupt_on` settings are respected
- Parent agent receives the interrupt via the `task` tool result
- User approves/rejects at the parent level
- Approval is forwarded back to the sub-agent

### 6. Streaming with Interrupts

```python
async for event in agent.astream(input, config=config, stream_mode="updates"):
    if "__interrupt__" in event:
        # Show approval UI to user
        approval = await get_user_decision(event["__interrupt__"])
        # Resume will happen on next ainvoke with Command(resume=...)
        break
    else:
        # Process normal streaming events
        print(event)
```

### 7. Accessing Interrupt State

The `__interrupt__` field in state contains:

- `tool_call_id`: ID of the interrupted tool call
- `tool_name`: Name of the tool
- `tool_args`: Arguments the agent wanted to pass
- `action_description`: Human-readable description (if configured)

## Quick Checklist

- [ ] Is checkpointer set (required for interrupts)?
- [ ] Are high-risk tools (write_file, execute) gated with interrupt_on?
- [ ] Is batch interrupt handling implemented for parallel tool calls?
- [ ] Is Command(resume=) used with proper decision format?
- [ ] Are sub-agent interrupts propagated and handled?

## Next File

- Continue to CLI: `65-cli-terminal-assistant.md`
