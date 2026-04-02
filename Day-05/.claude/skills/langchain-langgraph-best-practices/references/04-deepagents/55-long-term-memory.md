# Long-Term Memory

## Read This When

- Need cross-thread memory persistence
- Designing memory namespaces and routing
- Implementing user preferences or knowledge bases
- Choosing between storage backends
- Building agents with conversation continuity

## Skip This When

- Using ephemeral single-thread agents
- No need for persistent state across sessions
- Working with stateless request-response patterns

## Official References

1. https://docs.langchain.com/oss/python/deepagents/long-term-memory - Why: memory strategies, store configuration, and namespace patterns.
2. https://docs.langchain.com/oss/python/deepagents/backends - Why: StoreBackend and CompositeBackend for memory routing.

## Core Guidance

### 1. Memory Strategy Overview

DeepAgents supports three memory levels:

| Level | Mechanism | Scope | Lifetime |
|-------|-----------|-------|----------|
| Short-term | Checkpointer (InMemorySaver, etc.) | Single thread | Thread lifetime |
| Working files | StateBackend | Single thread | Thread lifetime (ephemeral) |
| Long-term | StoreBackend + LangGraph Store | Cross-thread | Permanent |

### 2. CompositeBackend Hybrid Strategy

Route different paths to different storage:

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # Dev; use PostgresStore for production

backend = CompositeBackend(
    default=StateBackend,           # / → ephemeral scratchpad
    routes={
        "/memories/": StoreBackend,  # /memories/ → permanent cross-thread
        "/workspace/": FilesystemBackend(root_dir="./work", virtual_mode=True),
    }
)

agent = create_deep_agent(
    model=model,
    backend=backend,
    store=store,
    checkpointer=InMemorySaver(),
)
```

### 3. Path Routing Behavior

| Path | Routes To | Behavior |
|------|-----------|----------|
| `/notes.txt` | StateBackend | Ephemeral, lost when thread ends |
| `/memories/preferences.txt` | StoreBackend | Permanent, accessible from any thread |
| `/workspace/code.py` | FilesystemBackend | Permanent on local disk |

### 4. StoreBackend Internals

Uses LangGraph Store with namespace tuples:
- Items stored as `(namespace_tuple, key)` pairs
- Namespace typically includes user_id, memory_type
- Example: `("user_123", "preferences")` → key: `"theme"`

### 5. Store Options

| Store | Use Case | Setup |
|-------|----------|-------|
| `InMemoryStore` | Development, testing | `store = InMemoryStore()` |
| `PostgresStore` | Production, persistent | `store = PostgresStore(connection_string=...)` |

### 6. Namespace Design Patterns

```python
# Hierarchical namespace for organized memory
# (user_id, memory_type) → key: specific_item
("user_123", "preferences")    → "theme": "dark mode"
("user_123", "preferences")    → "language": "Korean"
("user_123", "knowledge")      → "python_level": "advanced"
("user_123", "directives")     → "coding_style": "prefer async"
```

Best practices:
- Use consistent tuple structure across your application
- Include user_id or session_id in namespace for multi-user systems
- Group related memories by type (preferences, knowledge, directives)
- Keep namespace depth shallow (2-3 levels max)

### 7. Memory Lifecycle

Creation, update, and cleanup:

- **Create**: Agent writes to `/memories/` path → StoreBackend stores permanently
- **Read**: Agent reads from `/memories/` → StoreBackend retrieves from store
- **Update**: Agent edits file at `/memories/` path → StoreBackend updates item
- **Cleanup**: Implement periodic pruning for stale memories

### 8. Use Cases

| Use Case | Namespace Pattern | Example |
|----------|-------------------|---------|
| User preferences | `(user_id, "preferences")` | Theme, language, notification settings |
| Self-improvement directives | `(user_id, "directives")` | "Always use async", "Prefer Korean" |
| Knowledge base | `(user_id, "knowledge")` | Domain facts, learned patterns |
| Conversation summaries | `(user_id, "summaries")` | Past conversation key points |

### 9. Thread ID for Continuity

```python
# Same thread_id resumes conversation (short-term memory via checkpointer)
config = {"configurable": {"thread_id": "user_123_session_1"}}

# Different thread_id starts fresh conversation but shares long-term memory
config2 = {"configurable": {"thread_id": "user_123_session_2"}}
```

Key insight: `thread_id` controls short-term memory (conversation history), while namespaces in Store control long-term memory (cross-thread facts).

### 10. Production Deployment Pattern

```python
from langgraph.store.postgres import PostgresStore

# Production setup
store = PostgresStore(
    connection_string="postgresql://user:pass@host:5432/db"
)

backend = CompositeBackend(
    default=StateBackend,
    routes={
        "/memories/": StoreBackend,
    }
)

agent = create_deep_agent(
    model=model,
    backend=backend,
    store=store,
    checkpointer=PostgresSaver(connection_string="..."),
)
```

## Quick Checklist

- [ ] Is CompositeBackend configured with appropriate path routing?
- [ ] Is StoreBackend paired with a `store` instance?
- [ ] Is namespace design hierarchical and consistent?
- [ ] Is InMemoryStore used only for development (PostgresStore for production)?
- [ ] Is thread_id used for conversation continuity?
- [ ] Is memory cleanup strategy defined for stale entries?
- [ ] Are user_id or session_id included in namespaces for multi-user systems?
- [ ] Is the distinction between short-term (checkpointer) and long-term (store) memory clear?

## Next File

[60-human-in-the-loop.md](60-human-in-the-loop.md)
