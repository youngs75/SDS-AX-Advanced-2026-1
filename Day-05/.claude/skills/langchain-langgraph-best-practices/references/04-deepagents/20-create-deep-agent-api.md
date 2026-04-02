# Create Deep Agent API Reference

## Read This When
- Need the complete `create_deep_agent()` function signature
- Looking up parameter types, defaults, and purposes
- Implementing custom tools with proper schemas
- Configuring model selection and provider options
- Setting up structured output with `response_format`
- Need customization patterns beyond defaults
- Enabling debug mode or graph visualization

## Skip This When
- Just getting started (see overview and quickstart)
- Need middleware internals (see harness architecture)
- Working with specific built-in tools (see built-in tools reference)
- Implementing sub-agents (see subagents reference)
- Need long-term memory or storage setup (see long-term memory reference)

## Official References
1. https://docs.langchain.com/oss/python/deepagents/customization - Why: Complete parameter reference, customization options, and examples
2. https://docs.langchain.com/oss/python/deepagents/overview - Why: Parameter overview and default values
3. https://docs.langchain.com/oss/python/deepagents/quickstart - Why: Basic usage patterns and common configurations

## Core Guidance

### 1. Full Signature

```python
from deepagents import create_deep_agent

def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph
```

### 2. Parameter Reference Table

| Parameter | Type | Default | Purpose | See Reference |
|-----------|------|---------|---------|---------------|
| `model` | `str \| BaseChatModel` | Claude Sonnet 4.5 | LLM to use. String format: `"provider:model"` | Section 3 |
| `tools` | `Sequence[BaseTool \| Callable \| dict]` | `None` | Custom tools (added to built-in tools) | Section 4 |
| `system_prompt` | `str` | `None` | Agent behavior instructions | Section 6 |
| `middleware` | `Sequence[AgentMiddleware]` | `()` | User middleware (inserted at position 7 in stack) | `15-harness-architecture.md` |
| `subagents` | `list[SubAgent \| CompiledSubAgent]` | `None` | Sub-agent definitions for delegation | `40-subagents.md` |
| `response_format` | `ResponseFormat` | `None` | Structured output schema (Pydantic model) | Section 5 |
| `context_schema` | `type[Any]` | `None` | Additional context schema for state | — |
| `checkpointer` | `Checkpointer` | `None` | Conversation state persistence backend | `55-long-term-memory.md` |
| `store` | `BaseStore` | `None` | Cross-thread persistent key-value storage | `55-long-term-memory.md` |
| `backend` | `BackendProtocol \| BackendFactory` | `StateBackend` | File storage backend for virtual filesystem | `30-backends.md` |
| `interrupt_on` | `dict[str, bool \| InterruptOnConfig]` | `None` | Human-in-the-Loop tool approval configuration | `60-human-in-the-loop.md` |
| `debug` | `bool` | `False` | Enable verbose logging of middleware/tools | Section 7 |
| `name` | `str` | `None` | Agent name (for streaming metadata filtering) | Section 7 |
| `cache` | `BaseCache` | `None` | LLM response cache (e.g., InMemoryCache) | — |

### 3. Model Parameter (Two Forms)

**String format** (recommended for simplicity):

```python
# Uses init_chat_model() internally to instantiate provider
agent = create_deep_agent(model="openai:gpt-4.1")
agent = create_deep_agent(model="anthropic:claude-sonnet-4-5-20250929")
agent = create_deep_agent(model="google-genai:gemini-2.0-flash")

# Default if omitted
agent = create_deep_agent()  # Uses anthropic:claude-sonnet-4-5-20250929
```

**BaseChatModel instance** (for advanced configuration):

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Full control over model parameters
model = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    max_tokens=4000,
    timeout=60,
)
agent = create_deep_agent(model=model)

# With streaming callbacks
from langchain_core.callbacks import StreamingStdOutCallbackHandler

model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.7,
    callbacks=[StreamingStdOutCallbackHandler()],
)
agent = create_deep_agent(model=model)
```

**Provider string formats:**

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `openai:model-name` | `openai:gpt-4.1` |
| Anthropic | `anthropic:model-name` | `anthropic:claude-opus-4-6` |
| Google | `google-genai:model-name` | `google-genai:gemini-2.0-flash` |
| AWS Bedrock | `bedrock:model-id` | `bedrock:anthropic.claude-v2` |
| Azure OpenAI | `azure-openai:deployment-name` | `azure-openai:gpt-4` |

### 4. Custom Tool Addition

**Using `@tool` decorator with schema:**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, description="Maximum number of results")
    domain_filter: str | None = Field(default=None, description="Filter results to specific domain")

@tool(args_schema=SearchInput)
async def web_search(query: str, max_results: int = 5, domain_filter: str | None = None) -> str:
    """Search the web for information about a topic.

    Returns a formatted list of search results with titles, URLs, and snippets.
    Use this when you need current information not in your training data.
    """
    # Implementation
    results = perform_search(query, max_results, domain_filter)
    return format_results(results)

agent = create_deep_agent(
    model="openai:gpt-4.1",
    tools=[web_search],
)
```

**Multiple tools:**

```python
@tool
async def fetch_url(url: str) -> str:
    """Fetch content from a URL."""
    return await http_client.get(url)

@tool
async def safe_calculate(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    # Use a safe math parser in production
    import ast
    return ast.literal_eval(expression)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[web_search, fetch_url, safe_calculate],
    system_prompt="You are a research assistant with web access and calculation abilities.",
)
```

**Tool definition best practices:**

| Aspect | Best Practice |
|--------|---------------|
| Schema | Always use `args_schema` with Pydantic models |
| Docstring | Clear description of purpose and return value |
| Naming | Verb-based, descriptive (`fetch_url` not `url`) |
| Defaults | Provide sensible defaults for optional parameters |
| Async | Use `async def` for I/O-bound operations |
| Return type | Return strings or JSON-serializable dicts |

### 5. Structured Output (Response Format)

**Using Pydantic models:**

```python
from pydantic import BaseModel, Field

class ResearchReport(BaseModel):
    """Structured research report output."""
    title: str = Field(description="Report title")
    summary: str = Field(description="Executive summary (2-3 paragraphs)")
    key_findings: list[str] = Field(description="Bullet points of key findings")
    sources: list[str] = Field(description="URLs of sources cited")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)

agent = create_deep_agent(
    model="openai:gpt-4.1",
    response_format=ResearchReport,
    system_prompt="You are a research analyst. Produce structured reports.",
)

result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Research AI safety trends")]},
    config={"configurable": {"thread_id": "research_1"}},
)

# result["messages"][-1].content is a ResearchReport instance
report: ResearchReport = result["messages"][-1].content
print(report.title)
print(report.key_findings)
```

**Nested structures:**

```python
class Citation(BaseModel):
    title: str
    url: str
    relevance_score: float

class DetailedReport(BaseModel):
    title: str
    summary: str
    sections: list[dict[str, str]]  # [{"heading": "...", "content": "..."}]
    citations: list[Citation]

agent = create_deep_agent(
    model="anthropic:claude-opus-4-6",
    response_format=DetailedReport,
)
```

**When to use:**
- Need structured data for downstream processing
- Extracting specific fields from agent output
- Building APIs that consume agent results
- Ensuring output consistency across runs

### 6. System Prompt Patterns

**Basic instruction:**

```python
agent = create_deep_agent(
    system_prompt="You are a helpful coding assistant specialized in Python and TypeScript."
)
```

**With constraints:**

```python
agent = create_deep_agent(
    system_prompt="""You are a research assistant with these guidelines:
    - Always cite sources with URLs
    - Provide confidence scores for claims
    - Use the todo list to track research tasks
    - Delegate specialized research to sub-agents when needed
    - Save important findings to /research/findings.md
    """
)
```

**With examples:**

```python
agent = create_deep_agent(
    system_prompt="""You are a data analyst. Follow this workflow:

    1. Read the dataset with read_file()
    2. Plan analysis tasks with write_todos()
    3. Perform calculations with safe_calculate() tool
    4. Write results to /analysis/results.md

    Example interaction:
    Human: Analyze sales data
    Assistant: I'll start by reading the dataset...
    [uses read_file("/data/sales.csv")]
    [creates todos: "Calculate monthly averages", "Identify trends", "Generate report"]
    """
)
```

### 7. Debug Mode and Graph Visualization

**Enable debug logging:**

```python
import logging

# Enable debug logs
logging.basicConfig(level=logging.DEBUG)

agent = create_deep_agent(
    model="openai:gpt-4.1",
    debug=True,  # Verbose middleware/tool logging
    name="ResearchAgent",
)

# Logs will show:
# - Middleware entry/exit
# - Tool calls and results
# - State transitions
# - Context summarization triggers
```

**Graph visualization:**

```python
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    name="MyAgent",
    debug=True,
)

# Generate Mermaid diagram
mermaid_code = agent.get_graph().draw_mermaid()
print(mermaid_code)

# Or ASCII diagram
ascii_diagram = agent.get_graph().draw_ascii()
print(ascii_diagram)
```

**Filtering streaming events by name:**

```python
agent = create_deep_agent(
    model="openai:gpt-4.1",
    name="SubAgent1",
)

async for event in agent.astream_events(
    {"messages": [HumanMessage(content="Research AI")]},
    config={"configurable": {"thread_id": "session_1"}},
    version="v2",
):
    # Filter to only this agent's events
    if event.get("metadata", {}).get("langgraph_node") == "SubAgent1":
        print(event)
```

### 8. Complete Example (All Parameters)

```python
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.stores import InMemoryStore
from pydantic import BaseModel, Field

# Custom tool
@tool
async def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

# Structured output
class Report(BaseModel):
    title: str
    summary: str

# Custom middleware
class LoggingMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request, handler):
        print(f"Model call: {len(request['messages'])} messages")
        return await handler(request)

# Full configuration
agent = create_deep_agent(
    model=ChatOpenAI(model="gpt-4.1", temperature=0),
    tools=[web_search],
    system_prompt="You are a research expert.",
    middleware=[LoggingMiddleware()],
    response_format=Report,
    checkpointer=MemorySaver(),
    store=InMemoryStore(),
    interrupt_on={"web_search": True},  # Require approval
    debug=True,
    name="ResearchAgent",
)

result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Research AI trends")]},
    config={"configurable": {"thread_id": "session_1"}},
)
```

## Quick Checklist

- [ ] Is model specified explicitly (string format or BaseChatModel)?
- [ ] Are custom tools using `@tool` with proper `args_schema`?
- [ ] Is `system_prompt` providing clear behavior instructions?
- [ ] Is `response_format` set if structured output is needed?
- [ ] Is `name` set for streaming and debugging identification?
- [ ] Are async patterns used (`async def` for tools, `ainvoke` for agent)?
- [ ] Is `thread_id` set in config for filesystem and conversation continuity?
- [ ] Is `checkpointer` configured if conversation persistence is needed?
- [ ] Is `debug=True` enabled when troubleshooting?
- [ ] Are custom middleware properly implementing `AgentMiddleware` interface?
- [ ] Is `interrupt_on` configured for tools requiring human approval?

## Next File

→ `25-built-in-tools.md` — Complete reference for filesystem, planning, and delegation tools
