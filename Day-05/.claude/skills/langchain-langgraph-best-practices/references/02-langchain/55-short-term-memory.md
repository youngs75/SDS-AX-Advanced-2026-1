# Short-Term Memory

## Read This When

- Agent needs conversation memory within a session
- Managing thread_id for multi-turn conversations
- Context window is growing too large

## Skip This When

- Agent handles single-turn stateless requests only

## Official References

1. https://docs.langchain.com/oss/python/langchain/short-term-memory
   - Why: checkpointer setup, thread management, and trimming strategies

## Core Guidance

1. **Checkpointer concept**: Persists graph state between invocations. Required for multi-turn conversations.

2. **Basic setup**:
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    checkpointer=InMemorySaver(),
)

# Each conversation gets a unique thread_id
config = {"configurable": {"thread_id": "user_123_session_1"}}

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Alice."}]},
    config=config,
)

# Same thread_id → agent remembers context
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config=config,
)
# Agent responds: "Your name is Alice."
```

3. **Production checkpointers**:
| Checkpointer | Use Case |
|-------------|----------|
| `InMemorySaver` | Development and testing |
| `PostgresSaver` | Production (durable persistence) |
| `SqliteSaver` | Lightweight production |

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/mydb"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(model="openai:gpt-4.1", tools=[...], checkpointer=checkpointer)
```

4. **Custom state schema** (extend AgentState):
```python
from langchain.agents import create_agent, AgentState
from typing_extensions import NotRequired

class MyState(AgentState):
    user_id: NotRequired[str]
    preferences: NotRequired[dict]

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    state_schema=MyState,
    checkpointer=InMemorySaver(),
)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Hello"}], "user_id": "alice_123", "preferences": {"theme": "dark"}},
    config={"configurable": {"thread_id": "1"}},
)
```

5. **Message trimming** (keep context window manageable):
```python
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

@before_model
def trim_messages(state, runtime):
    messages = state["messages"]
    if len(messages) <= 10:
        return None
    first = messages[0]
    recent = messages[-8:]
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), first, *recent]}
```

6. **Summarization** (compress history instead of truncating):
```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4.1-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
```

7. **State access from tools** via ToolRuntime:
```python
from langchain.tools import tool, ToolRuntime

@tool
async def get_conversation_length(runtime: ToolRuntime) -> str:
    """Get the number of messages in current conversation."""
    return f"{len(runtime.state['messages'])} messages"
```

## Quick Checklist

- [ ] Is a checkpointer configured for multi-turn agents?
- [ ] Is thread_id unique per conversation session?
- [ ] Is message trimming or summarization in place for long conversations?
- [ ] Is InMemorySaver used only in development (not production)?

## Next File

`60-long-term-memory.md`
