# Structured Output

## Read This When
- Agent must return structured data (JSON, typed objects)
- Choosing between provider-native and tool-call-based extraction
- Binding Pydantic models to agent responses

## Skip This When
- Agent only needs free-form text responses

## Official References
1. https://docs.langchain.com/oss/python/langchain/structured-output
   - Why: response_format strategies and schema binding

## Core Guidance

1. **response_format on create_agent** (recommended):
```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent

class ContactInfo(BaseModel):
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")

agent = create_agent(model="openai:gpt-4.1", response_format=ContactInfo)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Extract: John"}]})
contact = result["structured_response"]  # ContactInfo instance
```

2. **Two strategies**:
| Strategy | How | Best When |
|----------|-----|-----------|
| `ProviderStrategy` | Native provider API (json_schema) | Provider supports it (OpenAI, Anthropic, Google, xAI) |
| `ToolStrategy` | Tool calling to extract structure | Universal fallback for all tool-capable models |

When you pass a schema directly, LangChain auto-selects the optimal strategy.

3. **Explicit strategy selection**:
```python
from langchain.agents import create_agent, ProviderStrategy, ToolStrategy

# Force provider-native (most reliable when available)
agent = create_agent(model="openai:gpt-4.1",
    response_format=ProviderStrategy(schema=ContactInfo, strict=True))

# Force tool-calling (universal)
agent = create_agent(model="openai:gpt-4.1",
    response_format=ToolStrategy(schema=ContactInfo, handle_errors=True))
```

4. **strict mode** (langchain >= 1.2): `strict=True` ensures 100% schema adherence on supporting providers.

5. **with_structured_output on model** (for direct model calls, not agent):
```python
model = init_chat_model("openai:gpt-4.1")
structured_model = model.with_structured_output(ContactInfo)
result = structured_model.invoke("Extract: Jane Smith, jane@test.com")
```

6. **Supported schema types**:
| Type | Returns |
|------|---------|
| Pydantic `BaseModel` | Validated model instance |
| Python `dataclass` | dict |
| `TypedDict` | dict |
| JSON Schema dict | dict |

7. **Error handling**:
```python
# Auto-retry on validation failure (default: True)
response_format=ToolStrategy(schema=ContactInfo, handle_errors=True)

# Custom error handler
def handle_error(error: Exception) -> str:
    return f"Invalid format. Error: {error}"

response_format=ToolStrategy(schema=ContactInfo, handle_errors=handle_error)
```

8. **When NOT to use structured output**:
- Open-ended chat conversations
- When response schema varies by context (use tools instead)

## Quick Checklist
- [ ] Is Pydantic model used for schema (not raw JSON schema)?
- [ ] Are Field descriptions provided for every field?
- [ ] Is strict mode enabled for production reliability?

## Next File
`50-streaming.md`