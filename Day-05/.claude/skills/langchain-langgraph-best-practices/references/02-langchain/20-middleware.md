# Middleware

## Read This When
- Adding cross-cutting behavior to the agent call flow
- Intercepting or modifying model/tool calls
- Need to understand middleware ordering

## Skip This When
- Agent needs no policy, logging, or call flow modification

## Official References
1. https://docs.langchain.com/oss/python/langchain/middleware/overview
   - Why: middleware hook model and registration pattern
2. https://docs.langchain.com/oss/python/langchain/middleware/built-in
   - Why: production-ready built-in middleware catalog
3. https://docs.langchain.com/oss/python/langchain/middleware/custom
   - Why: custom middleware authoring interface

## Core Guidance

### What middleware is
Intercepts the agent loop at model calls, tool calls, and agent lifecycle. Registered via `create_agent(middleware=[...])`.

### Built-in middleware

| Middleware | Purpose |
|-----------|---------|
| `SummarizationMiddleware` | Compress conversation history when approaching token limits |
| `HumanInTheLoopMiddleware` | Pause for human approval before tool execution |
| `ModelCallLimitMiddleware` | Prevent infinite loops with call count caps |
| `ToolCallLimitMiddleware` | Limit tool invocations per thread or run |
| `ModelFallbackMiddleware` | Auto-switch to backup model on failure |
| `PIIMiddleware` | Detect and handle PII (email, credit card, IP) |
| `TodoListMiddleware` | Equip agent with task planning tools |
| `LLMToolSelectorMiddleware` | LLM-based intelligent tool filtering |
| `ToolRetryMiddleware` | Retry failed tool calls with exponential backoff |
| `ModelRetryMiddleware` | Retry failed model calls with backoff |
| `LLMToolEmulator` | Emulate tool execution via LLM for testing |
| `ContextEditingMiddleware` | Clear old tool outputs to manage context window |
| `ShellToolMiddleware` | Expose persistent shell session to agent |
| `FilesystemFileSearchMiddleware` | Glob and grep over filesystem |
| `SubAgentMiddleware` | Delegate tasks to isolated subagents |
| `FilesystemMiddleware` | File read/write/edit tools for agent |

### Hook types

**Node-style hooks** (sequential):
- `before_agent` - runs before agent starts
- `before_model` - runs before model call
- `after_model` - runs after model responds
- `after_agent` - runs after agent completes

**Wrap-style hooks** (control flow):
- `wrap_model_call` - wraps entire model call (retry/fallback)
- `wrap_tool_call` - wraps individual tool execution

### Registration example

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, ModelCallLimitMiddleware

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        ),
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end"
        ),
    ],
)
```

### Custom middleware — decorator style

```python
from langchain.agents.middleware import before_model, AgentState
from langgraph.runtime import Runtime

@before_model
def log_messages(state: AgentState, runtime: Runtime) -> dict | None:
    print(f"Model call with {len(state['messages'])} messages")
    return None
```

### Custom middleware — wrap style (full control)

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

@wrap_model_call
def retry_with_fallback(request: ModelRequest, handler) -> ModelResponse:
    try:
        return handler(request)
    except Exception:
        fallback = init_chat_model("anthropic:claude-sonnet-4-5-20250929")
        return handler(request.override(model=fallback))
```

### Custom middleware — class style

```python
from langchain.agents.middleware import AgentMiddleware

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print(f"Messages: {len(state['messages'])}")
        return None

    def after_model(self, state, runtime):
        print(f"Response: {state['messages'][-1].content[:50]}")
        return None
```

### Execution order

- **before hooks**: run first→last
- **wrap hooks**: nest first-wraps-all
- **after hooks**: run last→first (reverse)

### Early exit (jump)

Return state with jump instruction:

```python
from langchain.agents.middleware import before_model, hook_config

@before_model
@hook_config(can_jump_to=["end"])
def emergency_stop(state, runtime):
    if detect_danger(state):
        return {"messages": state["messages"], "jump_to": "end"}
    return None
```

## Quick Checklist
- [ ] Is cross-cutting logic in middleware, not duplicated in every tool?
- [ ] Are built-in middleware preferred over custom when available?
- [ ] Is middleware ordering intentional (critical middleware first)?
- [ ] Are wrap-style hooks used for retry/fallback (not node-style)?

## Next File
- description: `25-guardrails.md`
