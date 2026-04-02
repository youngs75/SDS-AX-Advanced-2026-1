# Sub-Agents

## Read This When

- Need to configure sub-agents for delegation
- Understanding sub-agent state isolation and inheritance
- Debugging sub-agent selection or execution issues
- Implementing parallel sub-agent workflows
- Streaming sub-agent events separately from main agent

## Skip This When

- Using a single agent without delegation
- All work can be handled by one agent with tools
- No need for specialized agent roles or contexts

## Official References

1. https://docs.langchain.com/oss/python/deepagents/subagents - Why: Sub-agent configuration, inheritance rules, and streaming patterns.
2. https://docs.langchain.com/oss/python/deepagents/harness - Why: SubAgentMiddleware internals and state isolation mechanics.

## Core Guidance

### 1. Two Sub-Agent Types

| Type | Definition | Use When |
|------|-----------|----------|
| `SubAgent` (dict) | `{name, description, system_prompt, tools, model?, middleware?, interrupt_on?}` | Simple config, auto-compiled by middleware |
| `CompiledSubAgent` | `{name, description, runnable}` | Pre-built LangGraph graph as sub-agent |

**Key Difference**: SubAgent (dict) is compiled automatically by SubAgentMiddleware. CompiledSubAgent gives you full control over the graph structure.

### 2. SubAgent Dictionary Fields

```python
researcher = {
    "name": "researcher",
    "description": "Use for deep research requiring web search and source evaluation",
    "system_prompt": "You are a research specialist. Find authoritative sources and synthesize findings.",
    "tools": [web_search, think_tool],
    "model": "openai:gpt-4.1",              # Optional: override parent model
    "middleware": [custom_middleware],       # Optional: additional middleware
    "interrupt_on": {"execute": True},      # Optional: HITL for this sub-agent
}

# Use in agent creation
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    subagents=[researcher],
)
```

**Field Descriptions**:
- `name`: Unique identifier for sub-agent selection
- `description`: Used by LLM to decide when to delegate (CRITICAL for selection)
- `system_prompt`: Sub-agent's role and behavior instructions
- `tools`: Tools available to this sub-agent (merged with default_tools)
- `model`: Override parent model (optional)
- `middleware`: Additional middleware stack (optional)
- `interrupt_on`: HITL configuration for this sub-agent (optional)

### 3. CompiledSubAgent

For pre-built graphs with custom logic:

```python
from deepagents import create_deep_agent

# Build a sub-agent graph separately
sub_graph = create_deep_agent(
    model="openai:gpt-4.1",
    tools=[data_analysis_tool, visualization_tool],
    system_prompt="You are a data analysis specialist.",
)

compiled_sub = {
    "name": "analyst",
    "description": "Use for complex data analysis requiring statistical methods",
    "runnable": sub_graph,  # Pre-compiled LangGraph
}

# Use in parent agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    subagents=[compiled_sub],
)
```

**When to Use**:
- Custom graph structure (loops, conditionals)
- Pre-built agent libraries
- Complex state management needs
- Fine-grained control over execution flow

### 4. Inheritance Rules

SubAgentMiddleware propagates defaults from parent to sub-agents:

| Parent Config | Inheritance Behavior |
|--------------|---------------------|
| `default_model` | Used by sub-agents without explicit `model` |
| `default_tools` | Merged with sub-agent's own `tools` list |
| `default_middleware` | Prepended to sub-agent's `middleware` list |
| `default_interrupt_on` | Merged with sub-agent's `interrupt_on` |

**Example**:
```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    default_model="anthropic:claude-haiku-4",  # Sub-agents default to Haiku
    default_tools=[filesystem_tools],           # All sub-agents get filesystem
    subagents=[
        {
            "name": "researcher",
            "description": "Research specialist",
            "system_prompt": "...",
            "tools": [web_search],              # Gets filesystem + web_search
            "model": "openai:gpt-4.1",          # Overrides default Haiku
        }
    ],
)
```

### 5. State Isolation

`_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response"}`:

| State Key | Isolation Behavior |
|-----------|-------------------|
| `messages` | Sub-agents get fresh conversation history |
| `todos` | Sub-agents get independent task list |
| `structured_response` | Sub-agents don't inherit parent response schema |
| **All others** | Shared with parent (e.g., `files`, custom keys) |

**Why This Matters**:
- Sub-agents don't see parent's conversation (prevents context bloat)
- Sub-agents maintain independent task tracking
- Shared state enables data passing (files, artifacts)

**Example**:
```python
# Parent state
state = {
    "messages": [...parent conversation...],
    "todos": [...parent tasks...],
    "files": {"/data/input.csv": "..."},  # Shared
    "user_context": {...},                 # Shared
}

# Sub-agent receives
sub_state = {
    "messages": [],  # Fresh
    "todos": [],     # Fresh
    "files": {"/data/input.csv": "..."},  # Same reference
    "user_context": {...},                 # Same reference
}
```

### 6. general_purpose_agent

When `general_purpose_agent=True` (default), a built-in general-purpose sub-agent is automatically available:

```python
# Implicit general-purpose sub-agent
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    subagents=[researcher, analyst],
    general_purpose_agent=True,  # Default
)
# Agent has: researcher, analyst, AND a general-purpose sub-agent
```

**Behavior**:
- Has all default tools and middleware
- Used when no specific sub-agent matches the task
- Provides fallback delegation path

**Disable when**:
- You want strict sub-agent selection only
- All tasks MUST match a defined sub-agent

### 7. Parallel Execution

Main agent can call multiple `task` tools simultaneously:

```python
# LLM generates parallel tool calls in a single AIMessage
AIMessage(tool_calls=[
    {
        "name": "task",
        "args": {
            "description": "Research Python trends in 2025",
            "subagent_type": "researcher"
        }
    },
    {
        "name": "task",
        "args": {
            "description": "Research Rust trends in 2025",
            "subagent_type": "researcher"
        }
    },
])
# Both sub-agents execute in parallel with independent state copies
```

**Coordination**:
- Each sub-agent gets isolated `messages` and `todos`
- Shared state (files, custom keys) is copied (not shared reference)
- Results returned as separate tool call responses
- Parent agent synthesizes results in next step

### 8. Sub-Agent Streaming

Filter by `name` parameter for metadata:

```python
agent = create_deep_agent(
    model=model,
    subagents=[{"name": "researcher", ...}],
    name="coordinator",
)

async for event in agent.astream_events(input, version="v2"):
    metadata = event.get("metadata", {})
    node = metadata.get("langgraph_node")

    if node == "researcher":
        print("Researcher:", event)
    elif node == "coordinator":
        print("Coordinator:", event)
```

**Use Cases**:
- Monitor sub-agent progress separately
- Debug sub-agent tool calls
- Display parallel sub-agent work in UI
- Measure sub-agent performance

### 9. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Sub-agent not called | Description unclear to LLM | Improve `description` field with specific use cases |
| Context bloat in sub-agent | Too much state inherited | Check _EXCLUDED_STATE_KEYS, verify message isolation |
| Tool permission denied | Missing tool in sub-agent config | Add tool to sub-agent's `tools` list or `default_tools` |
| Sub-agent hangs | No exit condition | Set max iterations or timeout in recursion_limit |
| Wrong sub-agent selected | Overlapping descriptions | Make descriptions more distinct and specific |

**Debug Pattern**:
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check sub-agent selection in messages
for msg in state["messages"]:
    if hasattr(msg, "tool_calls"):
        for tc in msg.tool_calls:
            if tc["name"] == "task":
                print(f"Delegating to: {tc['args']['subagent_type']}")
```

## Quick Checklist

- [ ] Are sub-agent descriptions clear enough for LLM selection?
- [ ] Is state isolation understood (messages/todos excluded, files shared)?
- [ ] Are parallel task calls used for independent work?
- [ ] Is general_purpose_agent appropriate (True by default)?
- [ ] Is streaming filtering configured for sub-agent monitoring?
- [ ] Are default_tools and default_model set to avoid repetition?
- [ ] Are sub-agent names unique and descriptive?
- [ ] Is recursion_limit set to prevent infinite delegation?

## Next File

`45-skills.md` - Skill structure, progressive disclosure, and domain knowledge injection.
