# LangChain Standard: `create_agent`

## Read This When

- You are implementing or reviewing app-level agent construction.
- You need the current standard entrypoint and state model.

## Skip This When

- You are only designing low-level reusable primitives.

## Official References

1. https://docs.langchain.com/oss/python/langchain/agents
   - Why: canonical `create_agent` behavior and extension points.
2. https://docs.langchain.com/oss/python/migrate/langgraph-v1
   - Why: migration and deprecation policy. (DO NOT use `create_react_agent`)

## Core Guidance

### 1. `create_agent` as the Single Standard Entrypoint

`create_agent` is the unified constructor for all agent types. It replaced legacy constructors like `create_react_agent`, `create_openai_functions_agent`, and `create_structured_chat_agent`. Use it for all new agent construction.

### 2. Key Parameters and Their Purpose

```python
from langchain.agents import create_agent, AgentState

agent = create_agent(
    model="openai:gpt-4.1",           # str or BaseChatModel instance
    tools=[...],                        # list of @tool functions
    system_prompt="You are helpful.",    # system message for the agent
    middleware=[...],                    # cross-cutting behavior (see 20-middleware.md)
    checkpointer=InMemorySaver(),       # short-term memory (see 55-short-term-memory.md)
    store=InMemoryStore(),              # long-term memory (see 60-long-term-memory.md)
    response_format=MySchema,           # structured output (see 45-structured-output.md)
    state_schema=CustomState,           # custom state extension
    name="my_agent",                    # agent name for sub-agent streaming
)
```

### 3. Parameter Reference Table

| Parameter | Type | Purpose | See |
|-----------|------|---------|-----|
| `model` | str or BaseChatModel | LLM to use | `15-models-and-providers.md` |
| `tools` | list[BaseTool] | Tools available to agent | `35-tools.md` |
| `system_prompt` | str | Agent behavior instructions | — |
| `middleware` | list[Middleware] | Cross-cutting hooks | `20-middleware.md` |
| `checkpointer` | BaseCheckpointSaver | Session persistence | `55-short-term-memory.md` |
| `store` | BaseStore | Cross-session storage | `60-long-term-memory.md` |
| `response_format` | type or Strategy | Structured output schema | `45-structured-output.md` |
| `state_schema` | type[AgentState] | Custom state fields | — |
| `name` | str | Agent identity for streaming | `50-streaming.md` |

### 4. State Extension Pattern

Extend `AgentState` using `TypedDict` with `NotRequired` fields for optional state:

```python
from langchain.agents import AgentState
from typing_extensions import NotRequired

class CustomState(AgentState):
    user_id: NotRequired[str]
    preferences: NotRequired[dict]

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    state_schema=CustomState
)
```

### 5. Canonical Imports

```python
# Framework layer
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Runtime layer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

# Provider packages
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
```

### 6. Quick Invocation Patterns

```python
# Sync
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

# Async (recommended)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})

# With thread for memory
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"configurable": {"thread_id": "session_1"}},
)
```

## Quick Checklist

- [ ] Model config explicit (not buried in logic)?
- [ ] Tool list and schemas explicit?
- [ ] Middleware ordering intentional?
- [ ] State extension based on AgentState?
- [ ] Async invocation used (ainvoke/astream)?

## Next File

- `15-models-and-providers.md`
