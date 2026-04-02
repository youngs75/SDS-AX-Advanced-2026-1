# Messages

## Read This When

- Constructing message sequences for model calls
- Handling multimodal content (images, audio)
- Understanding content block structure and tool call flow

## Skip This When

- Using `create_agent` which handles message flow internally

## Official References

1. https://docs.langchain.com/oss/python/langchain/messages
   - Why: message types, content blocks, and manipulation patterns

## Core Guidance

### 1. Four message types with examples

```python
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

messages = [
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What's the weather in Seoul?"),
    AIMessage(content="", tool_calls=[{"id": "call_1", "name": "get_weather", "args": {"city": "Seoul"}}]),
    ToolMessage(content="Sunny, 22°C", tool_call_id="call_1", name="get_weather"),
    AIMessage("The weather in Seoul is sunny at 22°C!"),
]
```

### 2. Dict shorthand (interchangeable with classes)

```python
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
]
```

### 3. Standard content blocks (accessed via `.content_blocks`)

| Block Type | Purpose |
|-----------|---------|
| `TextContentBlock` | Standard text |
| `ReasoningContentBlock` | Model reasoning steps |
| `ImageContentBlock` | Images (URL, base64, file ID) |
| `AudioContentBlock` | Audio data |
| `VideoContentBlock` | Video files |
| `FileContentBlock` | PDFs, documents |
| `ToolCall` | Function invocation |
| `ToolCallChunk` | Streaming tool call fragment |

### 4. Multimodal input (image + text)

```python
message = HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "url": "https://example.com/photo.jpg"},
])
# or with base64
message = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image", "data": base64_string, "media_type": "image/png"},
])
```

### 5. AIMessage key attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `.text` | str | Text content |
| `.content` | str or list | Raw content (provider format) |
| `.content_blocks` | list | Standardized content blocks |
| `.tool_calls` | list[ToolCall] | Tool invocations |
| `.usage_metadata` | dict | Token counts (`input_tokens`, `output_tokens`, `total_tokens`) |
| `.id` | str | Unique message identifier |

### 6. Tool call flow

```python
response = model.invoke(messages)
for tc in response.tool_calls:
    result = execute_tool(tc["name"], tc["args"])
    messages.append(response)
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
```

### 7. Streaming chunks — `AIMessageChunk` accumulates

```python
full = None
async for chunk in model.astream("Hello"):
    full = chunk if full is None else full + chunk
print(full.content)
```

## Quick Checklist

- [ ] Are message types explicit (not raw dicts in critical paths)?
- [ ] Does every ToolMessage have a matching tool_call_id?
- [ ] Is multimodal content using the standard content block format?
- [ ] Are streaming chunks properly accumulated before processing?

## Next File

`45-structured-output.md`
