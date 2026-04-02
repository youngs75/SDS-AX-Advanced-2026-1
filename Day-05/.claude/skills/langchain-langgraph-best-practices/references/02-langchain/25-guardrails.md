# Guardrails

## Read This When
- Enforcing safety, compliance, or content policy on agent behavior
- Adding PII detection or content filtering
- Designing a layered trust boundary

## Skip This When
- Agent operates in a fully trusted, internal-only environment

## Official References
1. https://docs.langchain.com/oss/python/langchain/guardrails
   - Why: guardrail types, strategies, and custom implementation

## Core Guidance

### Two Types of Guardrails

**Deterministic** (regex, keyword): Fast, predictable, rule-based checks
**Model-based** (LLM judge): Catches nuance, slower and costlier

### PII Detection (Built-in via PIIMiddleware)

Built-in types: `email`, `credit_card`, `ip`, `mac_address`, `url`

Strategies:
- `redact`: Replace with `[REDACTED]`
- `mask`: Partial obscure (e.g., `****@email.com`)
- `hash`: Deterministic hash
- `block`: Raise exception

Custom detectors: regex string, compiled pattern, or callable

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
    ],
)
```

### Human-in-the-Loop Guardrail

Gate high-risk tools with approval workflow.

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[search_tool, send_email_tool, delete_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={
            "send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
            "delete_tool": True,
            "search_tool": False,   # no approval needed
        }),
    ],
)
```

### Custom Input Guardrail (Deterministic)

Block requests before agent processes them.

```python
from langchain.agents.middleware import before_agent, AgentState, hook_config

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime) -> dict | None:
    if not state["messages"]:
        return None
    content = state["messages"][0].content.lower()
    banned = ["hack", "exploit", "malware"]
    if any(kw in content for kw in banned):
        return {
            "messages": [{"role": "assistant", "content": "I cannot process this request."}],
            "jump_to": "end",
        }
    return None
```

### Custom Output Guardrail (Model-based)

Check agent output for safety violations.

```python
from langchain.agents.middleware import after_agent, hook_config
from langchain.chat_models import init_chat_model

safety_model = init_chat_model("openai:gpt-4.1-mini")

@after_agent(can_jump_to=["end"])
def safety_check(state: AgentState, runtime) -> dict | None:
    last = state["messages"][-1]
    result = safety_model.invoke([{
        "role": "user",
        "content": f"Is this safe? Respond SAFE or UNSAFE.\n\n{last.content}"
    }])
    if "UNSAFE" in result.content:
        last.content = "I cannot provide that response."
    return None
```

### Layered Strategy (Defense in Depth)

Stack multiple guardrails for robust protection.

```python
agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    middleware=[
        content_filter,                                          # Layer 1: input filter
        PIIMiddleware("email", strategy="redact", apply_to_input=True),   # Layer 2: PII input
        PIIMiddleware("email", strategy="redact", apply_to_output=True),  # Layer 2: PII output
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),      # Layer 3: HITL
        safety_check,                                            # Layer 4: output safety
    ],
    checkpointer=InMemorySaver(),
)
```

## Quick Checklist
- [ ] Are guardrails applied at BOTH input and output boundaries?
- [ ] Is PII detection enabled for user-facing agents?
- [ ] Are high-risk tools gated by HITL approval?
- [ ] Is a layered strategy used (not single-point defense)?

## Next File
- `30-mcp.md`
