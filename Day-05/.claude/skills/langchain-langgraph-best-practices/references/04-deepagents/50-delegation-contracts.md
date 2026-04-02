# Delegation Contracts

## Read This When

- Need to design delegation contracts for sub-agents
- Optimizing context usage and token efficiency
- Avoiding delegation anti-patterns
- Structuring task handoff between agents
- Determining when to delegate vs. execute directly

## Skip This When

- Using a single agent without sub-agent delegation
- Working with flat agent architectures
- Task complexity doesn't justify delegation overhead

## Official References

1. https://docs.langchain.com/oss/python/deepagents/subagents - Why: delegation patterns, contracts, and sub-agent lifecycle management.
2. https://docs.langchain.com/oss/python/deepagents/harness - Why: context management internals and isolation mechanisms.

## Core Guidance

### 1. Delegation Contract Fields

Every sub-agent task should specify clear contract fields for successful execution:

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `goal` | Yes | Clear objective | "Research 2024 AI agent frameworks" |
| `constraints` | Yes | Boundaries and limits | "Max 5 web searches, English sources only" |
| `output_format` | Recommended | Expected return shape | "Markdown report with source URLs" |
| `acceptance_criteria` | Recommended | How to verify completion | "At least 3 frameworks compared" |

### 2. When to Delegate

Delegate when the task is:
- **Multi-step**: requires several tool calls
- **Well-scoped**: clear start and end
- **Summarizable**: result can be consumed concisely
- **Independent**: doesn't need ongoing main-agent context

Do NOT delegate:
- Quick single-tool lookups
- Tasks requiring main conversation context
- Work that can't be verified from the result alone
- Simple operations that take <3 tool calls

### 3. Context Isolation Benefits

| Benefit | Description |
|---------|-------------|
| Token savings | Sub-agent starts with fresh context (only task description + inherited state) |
| Specialization | Sub-agent can have domain-specific tools and prompts |
| Parallel execution | Multiple sub-agents run concurrently |
| Failure isolation | Sub-agent crash doesn't corrupt main agent state |

### 4. Least-Privilege Tool Assignment

Give each sub-agent only the tools it needs:

```python
# Good: focused tool set
researcher = {
    "name": "researcher",
    "tools": [web_search, think_tool],  # Only research tools
}

# Bad: all tools
researcher = {
    "name": "researcher",
    "tools": [web_search, think_tool, code_execute, file_deploy, db_query],  # Over-provisioned
}
```

### 5. Return Artifact Design

Sub-agent returns should be:
- **Concise**: summarized findings, not raw data
- **Self-contained**: understandable without sub-agent's full context
- **Structured**: consistent format for main agent consumption

Example:
```python
# Good return artifact
{
    "summary": "Found 3 frameworks: LangGraph, AutoGPT, CrewAI",
    "comparison": {...},
    "sources": ["url1", "url2", "url3"]
}

# Bad return artifact
{
    "raw_search_results": [...],  # Too verbose
    "notes": "I found some stuff"  # Too vague
}
```

### 6. Delegation Overhead Analysis

| Factor | Cost | Benefit |
|--------|------|---------|
| Sub-agent initialization | ~1-2s startup | Independent context window |
| State copying | Proportional to shared state | Isolation from parent |
| Result serialization | Minimal | Clean ToolMessage return |
| Token usage | New context window | No context bloat in parent |

**Rule of thumb**: delegate if the task would take >3 tool calls in the main agent.

### 7. Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Over-delegation | Single-tool tasks delegated | Keep simple lookups in main agent |
| Vague contracts | "Do some research" | Specify goal, constraints, output_format |
| Tool over-provisioning | All tools given to every sub-agent | Least-privilege per sub-agent |
| Context forwarding | Passing full conversation to sub-agent | Let sub-agent work with task description only |
| No acceptance criteria | Can't verify sub-agent output | Define measurable completion criteria |

### 8. Orchestration Pattern

Main agent as coordinator:

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    subagents=[researcher, coder, reviewer],
    system_prompt="""You are a project coordinator.

Delegate complex work to specialized sub-agents:
- Research tasks → researcher
- Implementation → coder
- Quality review → reviewer

Always define clear goals and acceptance criteria when delegating.
Verify results before reporting to the user.
""",
)
```

## Quick Checklist

- [ ] Does each delegation have a clear goal and constraints?
- [ ] Is the output_format specified for consistent results?
- [ ] Are tools assigned with least-privilege principle?
- [ ] Is delegation reserved for multi-step tasks (not single lookups)?
- [ ] Are acceptance criteria defined for verification?
- [ ] Does the delegation overhead justify the context isolation benefit?
- [ ] Can the main agent verify the sub-agent's output?

## Next File

[55-long-term-memory.md](55-long-term-memory.md)
