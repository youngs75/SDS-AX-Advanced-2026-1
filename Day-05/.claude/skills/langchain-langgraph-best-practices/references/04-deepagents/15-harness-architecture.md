# DeepAgents Harness Architecture

## Read This When
- Need to understand the middleware stack and execution order
- Debugging context management or summarization behavior
- Investigating token eviction or large tool result handling
- Building custom middleware for cross-cutting concerns
- Need to understand how built-in tools are injected
- Troubleshooting dangling tool calls or cache behavior

## Skip This When
- Only using `create_deep_agent()` without customization (see overview)
- Need specific tool or sub-agent implementation details (see respective references)
- Working with checkpointers or stores (see long-term memory reference)
- Implementing Human-in-the-Loop logic (see HITL reference)
- Only need the API signature (see create-deep-agent-api reference)

## Official References
1. https://docs.langchain.com/oss/python/deepagents/harness - Why: Detailed middleware stack architecture and context management internals
2. https://docs.langchain.com/oss/python/deepagents/overview - Why: High-level harness feature summary and middleware list
3. https://docs.langchain.com/oss/python/deepagents/customization - Why: Custom middleware interface and extension patterns

## Core Guidance

### 1. Harness = Abstraction Layer on LangGraph

DeepAgents builds a **pre-configured middleware stack** on top of a LangGraph `StateGraph`. You get planning, filesystem, delegation, summarization, and caching without manual wiring.

**Key concept:**
- Middleware wraps model calls and tool calls to inject behavior
- Each middleware layer can add tools, modify state, or intercept requests/responses
- The harness manages middleware order and composition automatically

**What you get out of the box:**
- 10+ built-in tools (filesystem, planning, delegation)
- Automatic context summarization (85% threshold)
- Large tool result eviction (>80K chars)
- Dangling tool call recovery
- Prompt caching for Anthropic models

### 2. Middleware Stack Order (Outermost → Innermost)

| Order | Middleware | Injects | Purpose |
|-------|-----------|---------|---------|
| 8 | HumanInTheLoopMiddleware | (conditional) | Tool approval gates (only when `interrupt_on` set) |
| 7 | User Middleware | (custom) | User-defined cross-cutting logic |
| 6 | PatchToolCallsMiddleware | — | Repair dangling tool calls (AIMessage with tool_calls but no ToolMessage) |
| 5 | AnthropicPromptCachingMiddleware | — | Add cache_control breakpoints to reduce API costs |
| 4 | SummarizationMiddleware | — | Auto-summarize when context hits 85% of model window |
| 3 | SubAgentMiddleware | `task` tool | Create and manage sub-agents with isolated state |
| 2 | FilesystemMiddleware | 8 tools | Virtual filesystem: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`, `fetch` |
| 1 | TodoListMiddleware | `write_todos` | Task planning and tracking (pending/in_progress/completed) |

**Execution flow:**
1. Request enters at layer 8 (HITL)
2. Passes through each layer in descending order
3. Reaches the model at the core
4. Response bubbles back up through layers 1-8

**Why order matters:**
- HITL must be outermost to intercept all tool calls
- User middleware (layer 7) can observe/modify all built-in behavior
- Summarization (layer 4) runs before sub-agent logic to prevent context bloat
- TodoList (layer 1) is innermost so all tools can update task state

### 3. Context Management (SummarizationMiddleware)

**Default configuration:**
- **Trigger:** `("fraction", 0.85)` — activates at 85% of model context window
- **Keep:** `("fraction", 0.10)` — preserves most recent 10% of messages
- **Fallback:** `("tokens", 170000)` when model info unavailable

**How it works:**

```python
# Pseudo-code representation
async def awrap_model_call(request, handler):
    messages = request["messages"]

    # Check if context window is near limit
    if exceeds_threshold(messages, trigger=0.85):
        # Keep recent 10% of messages
        recent_messages = keep_recent(messages, keep=0.10)

        # Summarize the rest
        summary = await summarize_old_messages(messages[:-len(recent_messages)])

        # Replace old messages with summary
        request["messages"] = [summary] + recent_messages

    return await handler(request)
```

**Key behaviors:**
- Summarization runs **before every model call** (not just once)
- Preserves system prompt and most recent messages
- Summary is a single `HumanMessage` with role="assistant"
- Token counting uses `model_context_size` from model metadata

**Customization patterns:**

```python
from deepagents.middleware import SummarizationMiddleware

# Custom thresholds
custom_summarization = SummarizationMiddleware(
    trigger=("fraction", 0.75),  # Trigger earlier
    keep=("tokens", 20000),       # Keep fixed token count
)

agent = create_deep_agent(
    model="openai:gpt-4.1",
    middleware=[custom_summarization],  # Replaces default
)
```

### 4. Dangling Tool Call Recovery (PatchToolCallsMiddleware)

**Problem:** AIMessage has `tool_calls` but no matching `ToolMessage` in conversation history.

**Causes:**
- Agent crashed after requesting tool call but before receiving result
- Human interrupted during tool execution
- Tool execution timed out
- State was restored from checkpoint mid-execution

**Solution:** Auto-generates placeholder `ToolMessage` with cancellation notice.

```python
# Before PatchToolCallsMiddleware
messages = [
    HumanMessage(content="Search for AI agents"),
    AIMessage(content="", tool_calls=[{"name": "web_search", "args": {...}, "id": "call_1"}]),
    # Missing ToolMessage for call_1
    HumanMessage(content="What did you find?"),  # Would cause model error
]

# After PatchToolCallsMiddleware
messages = [
    HumanMessage(content="Search for AI agents"),
    AIMessage(content="", tool_calls=[{"name": "web_search", "args": {...}, "id": "call_1"}]),
    ToolMessage(content="Tool call was cancelled or did not complete.", tool_call_id="call_1"),  # Auto-inserted
    HumanMessage(content="What did you find?"),
]
```

**Why it matters:**
- Prevents model errors when resuming interrupted conversations
- Enables graceful recovery from crashes
- Works with Human-in-the-Loop rejections

### 5. Prompt Caching (AnthropicPromptCachingMiddleware)

**Purpose:** Reduce API costs for Anthropic models by marking cacheable content.

**How it works:**
- Adds `cache_control: {"type": "ephemeral"}` to system prompts
- Marks tool definitions as cacheable
- Anthropic caches unchanged prefixes between requests

**Example transformation:**

```python
# Before caching middleware
system = SystemMessage(content="You are a research expert...")
tools = [web_search, file_read, ...]

# After caching middleware
system = SystemMessage(
    content="You are a research expert...",
    additional_kwargs={"cache_control": {"type": "ephemeral"}},
)
tools = [
    web_search.with_config({"cache_control": {"type": "ephemeral"}}),
    file_read.with_config({"cache_control": {"type": "ephemeral"}}),
    ...
]
```

**Benefits:**
- System prompt cached across all requests in session
- Tool definitions cached (10+ tools = significant savings)
- Automatic — no configuration needed for Anthropic models

**Limitation:** Only works with Anthropic models (Claude family)

### 6. Large Tool Result Eviction

**Problem:** Tool results can exceed model context window (e.g., reading large files).

**Solution:** When a tool result exceeds `4 × tool_token_limit_before_evict` (default: 80,000 chars):

1. Save full result to `/large_tool_results/{tool_call_id}`
2. Return truncated result with file reference
3. Agent can paginate with `read_file(offset=..., limit=...)`

**Example:**

```python
# Agent calls: read_file("/src/large_file.py")
# Result is 200,000 characters (exceeds 80K threshold)

# FilesystemMiddleware saves to: /large_tool_results/call_abc123
# Returns to agent:
"""
[First 10 lines of content]
...
(Content truncated - full result saved to /large_tool_results/call_abc123)
Use read_file("/large_tool_results/call_abc123", offset=0, limit=100) to read more.
"""
```

**Agent can paginate:**
```python
# Read next chunk
read_file("/large_tool_results/call_abc123", offset=100, limit=100)
```

**Why it matters:**
- Prevents context explosion from large file reads
- Enables working with files larger than model context
- Automatic — no agent code changes needed

### 7. AgentMiddleware Interface (For Custom Middleware)

```python
from abc import ABC
from typing import Callable
from deepagents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool

class CustomMiddleware(AgentMiddleware):
    # Optional: inject custom tools
    tools: list[BaseTool] = []

    # Optional: extend state schema
    state_schema: type[AgentState] | None = None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable
    ) -> ModelResponse:
        """Intercept and modify model calls."""
        # Pre-processing
        print(f"Model call with {len(request['messages'])} messages")

        # Call next middleware/model
        response = await handler(request)

        # Post-processing
        print(f"Model returned: {response['messages'][-1].content[:50]}...")

        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable
    ) -> ToolMessage | Command:
        """Intercept and modify tool calls."""
        tool_name = request["tool_call"].get("name")
        print(f"Tool call: {tool_name}")

        # Call actual tool
        result = await handler(request)

        print(f"Tool result: {result.content[:50]}...")
        return result

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime
    ) -> dict | None:
        """Modify state before agent execution."""
        # Can return updates to state
        return {"custom_field": "value"}
```

**Usage:**

```python
agent = create_deep_agent(
    model="openai:gpt-4.1",
    middleware=[CustomMiddleware()],  # Inserted at position 7
)
```

**Key methods:**
- `awrap_model_call`: Intercept every LLM request/response
- `awrap_tool_call`: Intercept every tool execution
- `before_agent`: Run before each agent cycle
- `tools`: Add custom tools
- `state_schema`: Extend state with custom fields

## Quick Checklist

- [ ] Is the middleware stack order understood for debugging?
- [ ] Is summarization trigger appropriate for the model's context window?
- [ ] Are large tool results being handled (not bloating context)?
- [ ] Are dangling tool calls being auto-repaired after interruptions?
- [ ] Is prompt caching enabled for Anthropic models (automatic)?
- [ ] Does custom middleware use `awrap_model_call` or `awrap_tool_call`?
- [ ] Are custom middleware tools properly injected via `tools` property?
- [ ] Is the token eviction threshold (`tool_token_limit_before_evict`) appropriate?

## Next File

→ `20-create-deep-agent-api.md` — Full API signature, parameter reference, and customization patterns
