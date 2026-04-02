# DeepAgents Overview and Quickstart

## Read This When
- Need to understand what DeepAgents is and its core value proposition
- Choosing between `create_agent`, `StateGraph`, or `create_deep_agent` for your use case
- Getting started with your first DeepAgent implementation
- Need a high-level overview of built-in capabilities
- Evaluating whether DeepAgents fits your agent requirements

## Skip This When
- Already building with DeepAgents and need specific subsystem details (see harness architecture or API reference)
- Need detailed middleware customization patterns (see harness architecture)
- Working with sub-agents or delegation logic (see subagents reference)
- Implementing Human-in-the-Loop workflows (see HITL reference)
- Need filesystem or backend details (see respective references)

## Official References
1. https://docs.langchain.com/oss/python/deepagents/overview - Why: Comprehensive explanation of what DeepAgents is, when to use it, and feature overview
2. https://docs.langchain.com/oss/python/deepagents/quickstart - Why: Installation instructions and first working agent example
3. https://docs.langchain.com/oss/python/deepagents/customization - Why: Customization options beyond the default configuration

## Core Guidance

### 1. What is DeepAgents?

DeepAgents is a framework for building "deep thinking" AI agents on top of LangGraph. It provides a pre-configured middleware stack that handles common agent patterns automatically.

**Built-in Capabilities:**
- **Filesystem-based context management** — Virtual filesystem for long-term memory and file operations
- **SubAgent delegation** — Spawn independent sub-agents for parallel task processing
- **Automatic context summarization** — Evict old messages when approaching token limits (default: 170K)
- **Human-in-the-Loop** — Optional tool approval gates for sensitive operations
- **Task planning** — Built-in todo list management with pending/in_progress/completed states
- **Prompt caching** — Reduce API costs for Anthropic models via cache control breakpoints
- **Error recovery** — Auto-repair dangling tool calls after interruptions

### 2. When to Choose DeepAgents

| Need | Use | Why |
|------|-----|-----|
| Simple agent with tools | `create_agent` | Lightweight, no filesystem overhead, minimal abstraction |
| Custom graph topology | `StateGraph` / `@entrypoint` | Full control over nodes/edges, state transitions, conditional routing |
| File-based context + delegation + planning | `create_deep_agent` | Built-in middleware stack handles it all automatically |
| Long-running research/coding tasks | `create_deep_agent` | Auto-summarization + sub-agent isolation prevent context explosion |
| Quick prototype without boilerplate | `create_deep_agent` | One function call gives you filesystem, planning, delegation |

**Choose DeepAgents when:**
- Tasks require maintaining state across multiple sessions (filesystem)
- Need delegation to specialized sub-agents with isolation
- Working with long contexts that need automatic summarization
- Want planning and task tracking built-in
- Prefer opinionated defaults over full customization

**Don't choose DeepAgents when:**
- Building a simple tool-calling agent (use `create_agent`)
- Need custom graph topology with complex routing (use `StateGraph`)
- Want minimal dependencies and full control (use raw LangGraph)
- Working with short-lived, stateless interactions

### 3. Installation

```bash
# Using uv (recommended)
uv add deepagents

# Using pip
pip install deepagents
```

**Dependencies included:**
- `langgraph` (core graph execution)
- `langchain-anthropic` (default model provider)
- `langchain` (tools and utilities)

### 4. First Agent (Minimal Example)

```python
from deepagents import create_deep_agent
from langchain_core.messages import HumanMessage

# Default configuration:
# - Model: Claude Sonnet 4.5 (anthropic:claude-sonnet-4-5-20250929)
# - Built-in tools: filesystem (8 tools) + planning (write_todos) + delegation (task)
# - Middleware: summarization, caching, error recovery, filesystem, todos, subagents
agent = create_deep_agent()

# Invoke with thread_id for conversation continuity
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Research AI agent trends and create a summary")]},
    config={"configurable": {"thread_id": "session_1"}},
)

print(result["messages"][-1].content)
```

**Key points:**
- `thread_id` is required for filesystem and conversation state persistence
- Agent automatically gets planning, filesystem, and delegation tools
- No checkpointer by default — add one for true conversation persistence (see long-term memory reference)

### 5. With Custom Tools

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Define tool with structured input schema
class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: int = Field(default=5, description="Maximum number of results to return")

@tool(args_schema=SearchInput)
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for information about a topic."""
    # Implementation here
    return f"Found {max_results} results for: {query}"

# Create agent with custom tools + built-in tools
agent = create_deep_agent(
    model="openai:gpt-4.1",
    tools=[web_search],
    system_prompt="You are a research expert. Use web search to find current information.",
)

result = await agent.ainvoke(
    {"messages": [HumanMessage(content="What are the latest LangChain features?")]},
    config={"configurable": {"thread_id": "research_session"}},
)
```

**Best practices:**
- Always use `@tool` decorator with `args_schema` for proper type validation
- Provide clear docstrings — LLM uses them to understand tool purpose
- Use async functions (`async def`) for I/O-bound operations
- Custom tools are added alongside built-in tools (not replaced)

### 6. Built-in Capabilities Overview

| Capability | Provided By | Description |
|------------|-------------|-------------|
| Task planning | `write_todos` tool | Track pending/in_progress/completed tasks in state |
| File management | 8 filesystem tools | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `fetch` |
| Delegation | `task` tool | Spawn independent sub-agents with isolated state |
| Context control | SummarizationMiddleware | Auto-summarize at 85% of model context window |
| Prompt caching | AnthropicPromptCachingMiddleware | Reduce API costs for repeated system prompts and tool definitions |
| Error recovery | PatchToolCallsMiddleware | Fix dangling tool calls (AIMessage with tool_calls but no ToolMessage) |
| Large result handling | FilesystemMiddleware | Save tool results >80K chars to files, provide pagination |

**Middleware stack depth:** 8 layers (see harness architecture for full details)

### 7. Testing with Fake Models

```python
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage

# Create fake model for testing
fake_responses = [
    AIMessage(content="I'll help you research that topic.", tool_calls=[
        {"name": "web_search", "args": {"query": "AI agents", "max_results": 5}, "id": "call_1"}
    ]),
    AIMessage(content="Based on the search results, here's what I found..."),
]
fake_model = FakeListChatModel(responses=fake_responses)

agent = create_deep_agent(model=fake_model, tools=[web_search])

# Test without real API calls
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Research AI agents")]},
    config={"configurable": {"thread_id": "test_session"}},
)
```

## Quick Checklist

- [ ] Is `create_deep_agent` the right choice (vs `create_agent` or `StateGraph`)?
- [ ] Are custom tools defined with `@tool` and proper `args_schema`?
- [ ] Is `thread_id` set in config for filesystem and conversation continuity?
- [ ] Are async invocation patterns used (`ainvoke`/`astream`)?
- [ ] Is the default Claude Sonnet 4.5 model acceptable, or should you specify a different model?
- [ ] Do you need conversation persistence (checkpointer) or cross-thread storage (store)?
- [ ] Are you aware of the 10+ built-in tools that come with DeepAgents?
- [ ] Is the system prompt providing clear behavior instructions?

## Next File

→ `15-harness-architecture.md` — Middleware stack internals, context management, and token eviction behavior
