# Deprecations and Migration Guardrails

## Read This When

- You are migrating legacy code or examples
- You need a strict ban/replace checklist
- You encounter deprecated imports in existing code
- You are reviewing pull requests with legacy patterns
- You are auditing a codebase for modernization

## Skip This When

- You are starting from a clean modern codebase
- All dependencies are on LangChain 0.3+ and LangGraph 1.0+

## Official References

1. https://docs.langchain.com/oss/python/migrate/langgraph-v1
   - Why: official deprecation and migration guidance
2. https://python.langchain.com/docs/versions/v0_3/
   - Why: LangChain 0.3 breaking changes and migration paths
3. https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
   - Why: modern agentic patterns replacing legacy chains

## Core Guidance

### 1. Ban List (NEVER use these)

| Banned API | Package | Why Banned | Replacement |
|-----------|---------|------------|-------------|
| `create_react_agent` | langchain | Replaced by prebuilt | `create_agent` |
| `createReactAgent` | langchain (JS) | JS equivalent | `createAgent` |
| `AgentExecutor` | langchain.agents | Legacy chain-based runner | `create_agent` with graph |
| `initialize_agent` | langchain.agents | Legacy factory pattern | `create_agent` |
| `LLMChain` | langchain.chains | Rigid chain abstraction | Direct model calls or LCEL |
| `ConversationChain` | langchain.chains | Legacy memory pattern | `create_agent` + checkpointer |
| `SequentialChain` | langchain.chains | Static sequence | LangGraph workflow |
| `load_tools` | langchain.agents | Magic string loading | Explicit `@tool` definitions |
| `ConversationBufferMemory` | langchain.memory | Legacy memory class | `InMemorySaver` checkpointer |
| `ConversationSummaryMemory` | langchain.memory | Legacy memory class | Custom reducer in checkpointer |

### 2. Replacement Map

| Old Pattern | New Pattern | Notes |
|-------------|-------------|-------|
| `AgentExecutor.from_agent_and_tools(...)` | `create_agent(model=..., tools=...)` | Returns compiled graph |
| `initialize_agent(tools, llm, agent_type=...)` | `create_agent(model=..., tools=...)` | Single unified API |
| `LLMChain(llm=..., prompt=...)` | `model.invoke(messages)` or LCEL | Direct invocation or pipe |
| `ConversationChain(llm=..., memory=...)` | `create_agent(checkpointer=InMemorySaver())` | Memory via checkpointer |
| `ConversationBufferMemory()` | `InMemorySaver()` as checkpointer | Pass to `create_agent` |
| `load_tools(["tool_name"], llm)` | `@tool def my_tool(): ...` | Explicit function definitions |
| `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` | Provider-specific package |
| `from langchain.llms import OpenAI` | `from langchain_openai import ChatOpenAI` | Prefer chat models |

### 3. Import Migration

All provider classes moved to dedicated packages. The monolithic `langchain` package no longer exports model classes.

```python
# OLD (banned)
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

# NEW (required)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community import ChatOllama
```

Install provider packages explicitly:
```bash
pip install langchain-openai langchain-anthropic langchain-google-genai
```

### 4. Memory to Checkpointer Migration

| Old Memory Pattern | New Checkpointer Pattern |
|-------------------|-------------------------|
| `ConversationBufferMemory()` | `InMemorySaver()` |
| `ConversationSummaryMemory(llm=...)` | Custom reducer with summarization logic |
| `memory.save_context(inputs, outputs)` | Automatic via checkpointer |
| `memory.load_memory_variables({})` | Access via `state["messages"]` |

Example migration:
```python
# OLD
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

# NEW
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
checkpointer = InMemorySaver()
agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)
```

### 5. Verification Commands

Run these checks to find legacy patterns:

```bash
# Check for banned agent APIs
rg -n "create_react_agent|createReactAgent|AgentExecutor|initialize_agent" .

# Check for banned chain APIs
rg -n "LLMChain|ConversationChain|SequentialChain|load_tools" .

# Check for old import paths
rg -n "from langchain.chat_models import|from langchain.llms import" .

# Check for legacy memory
rg -n "ConversationBufferMemory|ConversationSummaryMemory" .

# Confirm modern usage
rg -n "create_agent|init_chat_model" .
rg -n "from langchain_openai import|from langchain_anthropic import" .
```

### 6. Testing After Migration

After replacing deprecated APIs, re-test these behaviors:

- **Tool routing**: Verify tools are called with correct arguments
- **Error handling**: Confirm tool errors are caught and reported
- **State persistence**: Test checkpointer saves and restores state correctly
- **Streaming**: Validate streaming output if used
- **Interrupts**: Test human-in-the-loop if applicable

### 7. Common Migration Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Missing provider package | `ImportError: cannot import ChatOpenAI` | `pip install langchain-openai` |
| Wrong model import | `AttributeError: module 'langchain' has no attribute 'ChatOpenAI'` | Use `langchain_openai` import |
| Memory not persisting | State lost between invocations | Pass `checkpointer` to `create_agent` |
| Tools not routing | Agent doesn't call tools | Verify tool schemas and model supports tool calling |
| Streaming broken | No intermediate output | Use `agent.stream()` instead of `invoke()` |

## Quick Checklist

Before marking migration complete:

- [ ] Zero matches for banned APIs in codebase?
- [ ] All provider imports from `langchain-*` packages?
- [ ] No legacy chain patterns remaining?
- [ ] No legacy memory classes remaining?
- [ ] Tool routing and error behavior re-tested after migration?
- [ ] State persistence verified with checkpointer?
- [ ] All tests passing with new APIs?
- [ ] Documentation updated to show new patterns?

## Next File

- Runtime workflow design: `../03-langgraph/10-graph-api.md`
