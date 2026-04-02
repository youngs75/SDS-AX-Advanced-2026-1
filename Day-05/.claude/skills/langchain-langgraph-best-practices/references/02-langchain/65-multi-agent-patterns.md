# Multi-Agent Patterns

## Read This When
- Designing a system with multiple specialized agents
- Choosing between router, handoff, subagent, or skills patterns
- Need to understand the LangChain-level multi-agent API

## Skip This When
- A single agent with the right tools can handle the task

## Official References
1. https://docs.langchain.com/oss/python/langchain/multi-agent/index
   - Why: pattern catalog, decision matrix, and performance comparison

## Core Guidance

1. **Why multi-agent**: Three motivations — context management (specialized knowledge), distributed development (team boundaries), parallelization (concurrent execution).

2. **Official 5 patterns**:

| Pattern | Description | Best For |
|---------|-------------|----------|
| **Router** | Classify input → dispatch to specialized agent → synthesize | Parallel execution, large-context domains |
| **Handoffs** | Agents dynamically transfer control based on state | Direct user interaction, sequential multi-hop |
| **Subagents** | Main agent orchestrates subagents as tools | Parallelization, distributed development |
| **Skills** | Single agent loads specialized context on-demand | Simple tasks, token efficiency, repeat requests |
| **Custom Workflow** | Bespoke LangGraph flow mixing deterministic + agentic | Complex domain-specific logic |

3. **Pattern selection matrix**:

| Requirement | Subagents | Handoffs | Skills | Router |
|-------------|:---------:|:--------:|:------:|:------:|
| Distributed dev | ★★★★★ | — | ★★★★★ | ★★★ |
| Parallelization | ★★★★★ | — | ★★★ | ★★★★★ |
| Multi-hop conversations | ★★★★★ | ★★★★★ | ★★★★★ | — |
| Direct user interaction | ★ | ★★★★★ | ★★★★★ | ★★★ |

4. **Router pattern** — classify and dispatch:
```python
# Uses create_agent with routing middleware or conditional edges
# See asset: assets/03-langgraph/agent-patterns/router-example/graph.py
```

5. **Handoff pattern** — agents transfer control:
```python
# Each agent has handoff tools that transfer to the next specialist
# Context flows with the handoff for continuity
# See asset: assets/03-langgraph/agent-patterns/handoff-example/graph.py
```

6. **Subagent pattern** (via SubAgentMiddleware):
```python
from deepagents.middleware.subagents import SubAgentMiddleware

agent = create_agent(
    model="openai:gpt-4.1",
    middleware=[
        SubAgentMiddleware(
            default_model="openai:gpt-4.1",
            subagents=[
                {
                    "name": "researcher",
                    "description": "Research topics in depth.",
                    "tools": [search_tool],
                    "model": "openai:gpt-4.1",
                },
                {
                    "name": "writer",
                    "description": "Write polished content.",
                    "tools": [write_tool],
                },
            ],
        ),
    ],
)
```

7. **Skills pattern** (on-demand context loading):
```python
# Single agent loads domain-specific prompts and knowledge on demand
# Token-efficient: skills are loaded only when needed
# See deepagents docs: references/04-deepagents/45-skills.md
```

8. **Context Engineering** — the core principle: each agent should see ONLY the information relevant to its task. Quality depends on selective context, not maximum context.

9. **When NOT to use multi-agent**: "Not every complex task requires this approach — a single agent with the right (sometimes dynamic) tools and prompt can often achieve similar results."

10. **Existing asset templates** (LangGraph-level implementations):
- Router: `assets/03-langgraph/agent-patterns/router-example/`
- Supervisor: `assets/03-langgraph/agent-patterns/supervisor-example/`
- Orchestrator: `assets/03-langgraph/agent-patterns/orchestrator-example/`
- Handoff: `assets/03-langgraph/agent-patterns/handoff-example/`

## Quick Checklist
- [ ] Is multi-agent truly needed (vs single agent with good tools)?
- [ ] Is each agent's context scope clearly bounded?
- [ ] Is the right pattern selected for the requirements?
- [ ] Are handoffs preserving necessary context between agents?

## Next File
- Deprecations and migration: `90-deprecations-and-migration.md`
