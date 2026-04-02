# LangGraph: Memory Patterns

## Read This When
- Implementing conversation memory (short-term or long-term)
- Managing long conversations (trimming, summarization)
- Need cross-session user data or shared knowledge

## Skip This When
- Agent is stateless single-turn only
- Question is about persistence infrastructure (see `40-persistence.md`)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/add-memory
   - Why: short-term and long-term memory patterns, trimming, summarization.
2. https://docs.langchain.com/oss/python/langgraph/persistence
   - Why: Store interface for cross-thread memory.

## Core Guidance

### 1. Short-Term vs Long-Term Memory

| Aspect | Short-Term (Checkpointer) | Long-Term (Store) |
|--------|--------------------------|-------------------|
| Scope | Single thread/session | Cross-thread/session |
| Lifetime | Until thread expires | Until explicitly deleted |
| Data type | Full state snapshots | Key-value items |
| Access | Automatic via thread_id | Manual via store API |
| Use case | Conversation history | User profiles, preferences, knowledge |

### 2. Short-Term Memory Setup
Checkpointer automatically saves and restores state per thread:

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# First turn
config = {"configurable": {"thread_id": "session_1"}}
await graph.ainvoke({"messages": [HumanMessage(content="Hi, I'm Alice")]}, config)

# Second turn — remembers Alice
await graph.ainvoke({"messages": [HumanMessage(content="What's my name?")]}, config)
```

### 3. Long-Term Memory Setup
Store persists data across threads:

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = builder.compile(checkpointer=checkpointer, store=store)

# In a node or tool — save user preference
async def save_preference(state, *, store):
    user_id = state.get("user_id", "default")
    await store.aput(("users", user_id), "prefs", {"theme": "dark"})

# In another session — retrieve preference
async def load_preference(state, *, store):
    user_id = state.get("user_id", "default")
    item = await store.aget(("users", user_id), "prefs")
    return item.value if item else {}
```

### 4. Message Trimming
Prevent context window overflow in long conversations:

```python
from langgraph.graph.message import RemoveMessage

async def trim_messages(state: MyState) -> dict:
    messages = state["messages"]
    if len(messages) > 20:
        # Keep system message + last 10 messages
        to_remove = messages[1:-10]  # skip system, keep recent
        return {"messages": [RemoveMessage(id=m.id) for m in to_remove]}
    return {}
```

Alternative strategies:
- **By count**: Keep last N messages
- **By token**: Estimate tokens, trim oldest when over limit
- **By role**: Always keep system messages, trim user/assistant pairs

### 5. Summarization
Compress old messages into a summary:

```python
async def summarize_if_needed(state: MyState) -> dict:
    messages = state["messages"]
    if len(messages) > 30:
        old_messages = messages[1:-10]  # exclude system + recent
        summary = await model.ainvoke([
            SystemMessage(content="Summarize this conversation concisely:"),
            *old_messages,
        ])
        # Remove old, add summary
        removals = [RemoveMessage(id=m.id) for m in old_messages]
        return {"messages": removals + [SystemMessage(content=f"Previous conversation summary: {summary.content}")]}
    return {}
```

### 6. Semantic Search in Store
Find relevant memories using natural language:

```python
from langchain_openai import OpenAIEmbeddings

store = InMemoryStore(
    index={"dims": 1536, "embed": OpenAIEmbeddings(), "fields": ["$"]}
)

# Store knowledge
await store.aput(("knowledge",), "fact_1", {"text": "User prefers Python over JavaScript"})

# Search by meaning
results = await store.asearch(("knowledge",), query="programming language preference")
for item in results:
    print(item.value["text"])
```

### 7. Accessing Store from Nodes

**Graph API**: Use `store` parameter in compiled graph
```python
async def my_node(state: MyState, store: BaseStore) -> dict:
    items = await store.asearch(("users",), query=state["query"])
    return {"context": [item.value for item in items]}
```

**Functional API**: Use `store` injectable parameter
```python
@entrypoint(checkpointer=saver, store=store)
async def workflow(input: dict, *, store: BaseStore) -> str:
    prefs = await store.aget(("users", input["user_id"]), "prefs")
    ...
```

### 8. Production Pattern
```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

checkpointer = PostgresSaver(conn_string)  # short-term
store = PostgresStore(conn_string)          # long-term
graph = builder.compile(checkpointer=checkpointer, store=store)
```

## Quick Checklist
- [ ] Checkpointer configured for short-term memory?
- [ ] Store configured for cross-session data?
- [ ] Trimming or summarization for long conversations?
- [ ] InMemory variants only in development?
- [ ] Store namespace strategy consistent?

## Next File
- Runtime harness: `../04-deepagents/10-overview-and-quickstart.md`
