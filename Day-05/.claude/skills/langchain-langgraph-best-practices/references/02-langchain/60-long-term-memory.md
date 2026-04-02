# Long-Term Memory

## Read This When
- Agent needs to remember information across sessions
- Building user profiles or knowledge bases persistent over time
- Need key-value store accessible from tools

## Skip This When
- Memory only needs to last within a single conversation (see `55-short-term-memory.md`)

## Official References
1. https://docs.langchain.com/oss/python/langchain/long-term-memory
   - Why: store-based persistence, namespace strategy, and tool access patterns

## Core Guidance

1. **Store vs Checkpointer**:
| Concern | Checkpointer (Short-term) | Store (Long-term) |
|---------|--------------------------|-------------------|
| Scope | Within a thread/session | Across all sessions |
| Data | Conversation state + messages | Arbitrary key-value data |
| Access | Automatic per thread_id | Manual via namespace + key |
| Use case | Multi-turn memory | User profiles, knowledge bases |

2. **Basic setup**:
```python
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver

store = InMemoryStore()

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[...],
    checkpointer=InMemorySaver(),  # short-term
    store=store,                    # long-term
)
```

3. **Namespace strategy**: Organize data hierarchically like folders:
- `("users", user_id)` — user profiles
- `("knowledge", topic)` — domain knowledge
- `("preferences", user_id)` — user preferences

4. **Store operations**:
```python
# Write
store.put(("users", "alice"), "profile", {"name": "Alice", "role": "engineer"})

# Read
item = store.get(("users", "alice"), "profile")
print(item.value)  # {"name": "Alice", "role": "engineer"}

# Search within namespace
results = store.search(("users",))
for item in results:
    print(f"Key: {item.key}, Value: {item.value}")
```

5. **Store access from tools** via ToolRuntime:
```python
from langchain.tools import tool, ToolRuntime

@tool
async def remember_user(user_id: str, info: dict, runtime: ToolRuntime) -> str:
    """Save user information for future reference."""
    runtime.store.put(("users", user_id), "profile", info)
    return f"Saved profile for {user_id}"

@tool
async def recall_user(user_id: str, runtime: ToolRuntime) -> str:
    """Recall saved user information."""
    item = runtime.store.get(("users", user_id), "profile")
    if item:
        return str(item.value)
    return "No information found for this user."
```

6. **Production stores**:
| Store | Use Case |
|-------|----------|
| `InMemoryStore` | Development and testing |
| `PostgresStore` | Production (durable, searchable) |

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string("postgresql://user:pass@localhost:5432/mydb")
store.setup()
```

7. **Combining both memory types** (typical production pattern):
```python
agent = create_agent(
    model="openai:gpt-4.1",
    tools=[remember_user, recall_user, ...],
    checkpointer=PostgresSaver.from_conn_string(DB_URI),
    store=PostgresStore.from_conn_string(DB_URI),
)
# Short-term: agent remembers conversation within thread
# Long-term: agent remembers user info across all threads
```

## Quick Checklist
- [ ] Is store used for cross-session data (not checkpointer)?
- [ ] Is namespace strategy consistent across tools?
- [ ] Is InMemoryStore used only in development?
- [ ] Do tools access store via ToolRuntime (not global variables)?

## Next File
`65-multi-agent-patterns.md`
