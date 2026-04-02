# LangGraph: Persistence

## Read This When
- Need resume capability, cross-session memory, or durable storage
- Choosing between checkpointer types (dev vs production)
- Implementing Store for cross-thread data

## Skip This When
- Flow is stateless and single-turn with no memory requirements
- Only need in-node data transformation

## Official References
1. https://docs.langchain.com/oss/python/langgraph/persistence
   - Why: checkpointer types, Store interface, semantic search, encryption.
2. https://docs.langchain.com/oss/python/langgraph/add-memory
   - Why: short-term and long-term memory patterns.

## Core Guidance

### 1. Checkpointer vs Store

| Aspect | Checkpointer | Store |
|--------|-------------|-------|
| Scope | Thread-scoped (one session) | Cross-thread (shared) |
| Data | Full state snapshots | Key-value items |
| Access | Automatic per superstep | Manual via `store.put/get/search` |
| Use case | Conversation memory, resume | User profiles, shared knowledge |

### 2. Checkpointer Types

| Type | Package | Best For |
|------|---------|----------|
| `InMemorySaver` | `langgraph.checkpoint.memory` | Development, testing |
| `SqliteSaver` | `langgraph.checkpoint.sqlite` | Lightweight production |
| `PostgresSaver` | `langgraph.checkpoint.postgres` | Production at scale |
| `CosmosDBSaver` | `langgraph.checkpoint.cosmosdb` | Azure deployments |

```python
from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 3. Thread Identity
- `thread_id` in config isolates each conversation/session
- Different thread_ids = independent state histories
- Same thread_id = continued conversation with full history

```python
config = {"configurable": {"thread_id": "user_123_session_1"}}
result = await graph.ainvoke(input, config=config)
```

### 4. State Access Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_state(config)` | Current state snapshot | `StateSnapshot` |
| `get_state_history(config)` | All past snapshots (reverse chronological) | Iterator of `StateSnapshot` |
| `update_state(config, values, as_node)` | Modify state manually | Updated config |

```python
# Inspect current state
snapshot = await graph.aget_state(config)
print(snapshot.values["messages"])

# Browse history
async for state in graph.aget_state_history(config):
    print(state.metadata["step"], state.values)
```

### 5. Store Interface
Cross-thread key-value storage with namespace organization:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# Write (from node or tool)
await store.aput(("users", "user_123"), "preferences", {"theme": "dark"})

# Read
item = await store.aget(("users", "user_123"), "preferences")
print(item.value)  # {"theme": "dark"}

# Search across namespace
results = await store.asearch(("users",), query="preferences")
```

### 6. Semantic Search in Store
- Configure embedding for natural language queries
- `index` parameter on store: `{"dims": 1536, "embed": embedding_fn, "fields": ["text"]}`
- `store.asearch(namespace, query="natural language query")` returns ranked results

```python
from langchain_openai import OpenAIEmbeddings

store = InMemoryStore(
    index={"dims": 1536, "embed": OpenAIEmbeddings(), "fields": ["$"]}
)
```

### 7. Encryption
- Protect sensitive checkpoint data at rest
- `EncryptedSerializer` wraps existing serializer with AES encryption
- Set `LANGGRAPH_AES_KEY` environment variable

```python
from langgraph.checkpoint.serde.encryption import EncryptedSerializer

serializer = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver(conn, serde=serializer)
```

### 8. Production Pattern
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

checkpointer = PostgresSaver(conn_string)
store = PostgresStore(conn_string)
graph = builder.compile(checkpointer=checkpointer, store=store)
```

## Quick Checklist
- [ ] Checkpointer type matches environment (InMemory=dev, Postgres=prod)?
- [ ] Thread ID strategy explicit and consistent?
- [ ] Store namespace scheme documented?
- [ ] InMemorySaver NOT used in production?
- [ ] Encryption enabled for sensitive data?

## Next File
- `45-interrupts.md`
