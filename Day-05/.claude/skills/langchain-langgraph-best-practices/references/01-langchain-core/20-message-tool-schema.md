# Message, Tool, and Schema Contracts

## Read This When

- You need to define message roles, tool args, and output schemas.
- Tool-calling behavior is unstable or inconsistent.
- You need to understand content block structure or tool message construction.

## Skip This When

- The issue is clearly runtime durability or workflow topology.

## Official References

1. https://docs.langchain.com/oss/python/langchain/messages
   - Why: message model and content block structure.
2. https://docs.langchain.com/oss/python/langchain/tools
   - Why: tool contract and runtime tool usage.
3. https://docs.langchain.com/oss/python/langchain/agents
   - Why: `create_agent` + `AgentState` state model.
4. https://docs.langchain.com/oss/python/langgraph/graph-api
   - Why: explicit state channel/reducer behavior.

## Core Guidance

1. **Use explicit message roles**: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`.

2. **Define deterministic tool input/output schemas.**

3. **In `create_agent` workflows**, treat `AgentState` as the default state base.

4. **In pure Graph API workflows**, use explicit state schema and reducers for message channels.

5. **`MessagesState` is a LangGraph convenience option**, but this skill prefers explicit state modeling for explanation quality.

6. **Content block structure detail**:

```python
# Content blocks in AIMessage (accessed via .content_blocks)
# Text content
{"type": "text", "text": "Hello world"}

# Image content
{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}

# Tool use (model requesting tool call)
{"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Seoul"}}

# Tool result (returned to model)
{"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 22°C"}
```

7. **ToolMessage construction** with tool_call_id matching:

```python
from langchain_core.messages import AIMessage, ToolMessage

# AIMessage contains tool_calls
ai_msg = AIMessage(content="", tool_calls=[
    {"id": "call_abc", "name": "search", "args": {"query": "weather"}},
])

# ToolMessage MUST match tool_call_id
tool_msg = ToolMessage(
    content="Sunny, 22°C",
    tool_call_id="call_abc",  # must match ai_msg.tool_calls[0]["id"]
    name="search",
)
```

8. **AIMessage.tool_calls list structure**:

```python
response = model.invoke(messages)
for tc in response.tool_calls:
    print(tc["id"])      # unique call identifier
    print(tc["name"])    # tool function name
    print(tc["args"])    # parsed arguments dict
```

9. **Tool schema at core level**:
   - `BaseTool` is the core contract.
   - `@tool` decorator is the framework convenience.
   - Both produce the same tool calling schema for models.

10. **Multi-turn tool-calling conversation** (complete manual example):

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

conversation = [
    SystemMessage("You have access to a weather tool."),
    HumanMessage("What's the weather in Seoul?"),
    AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_weather", "args": {"city": "Seoul"}}]),
    ToolMessage(content="Sunny, 22°C", tool_call_id="c1", name="get_weather"),
    AIMessage("The weather in Seoul is sunny at 22°C!"),
]
```

## Asset Examples

| Topic | Asset Path |
|-------|-----------|
| All 5 message types, content blocks, multimodal, RemoveMessage | `assets/01-langchain-core/message-patterns/` |
| Core tool definitions (@tool, BaseTool, StructuredTool, InjectedToolArg) | `assets/01-langchain-core/tool-definition/` |

## Quick Checklist

- [ ] Are message shapes explicit and typed?
- [ ] Does every ToolMessage have a matching tool_call_id?
- [ ] Are tool failures returned in structured, recoverable format?
- [ ] Are content blocks using standard types (not raw provider format)?
- [ ] Are schema changes versioned for compatibility?

## Next File

- Runnable composition and typing: `30-runnables-state-types.md`
