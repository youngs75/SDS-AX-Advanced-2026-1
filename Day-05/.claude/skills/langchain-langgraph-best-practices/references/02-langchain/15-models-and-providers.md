# Models and Providers

## Read This When
- Selecting or configuring a chat model for an agent
- Switching between providers or making model runtime-configurable

## Skip This When
- Model is already configured and working

## Official References
1. https://docs.langchain.com/oss/python/langchain/models
   - Why: model initialization options and provider integration
2. https://docs.langchain.com/oss/python/concepts/integrations
   - Why: provider package catalog

## Core Guidance

### Three initialization methods

```python
# Universal initializer (recommended)
from langchain.chat_models import init_chat_model
model = init_chat_model("openai:gpt-4.1")

# Provider-specific class
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4.1", temperature=0)

# String shorthand in create_agent
from langchain.agents import create_agent
agent = create_agent(model="openai:gpt-4.1", tools=[...])
```

### Provider string format

Use `"provider:model-name"` format:

- `"openai:gpt-4.1"`
- `"anthropic:claude-sonnet-4-5-20250929"`
- `"google_genai:gemini-2.5-flash"`
- `"bedrock_converse:anthropic.claude-3-5-sonnet-20240620-v1:0"`

### Configuration parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `model` | str | Model identifier (required) |
| `temperature` | float | Randomness (0=deterministic) |
| `max_tokens` | int | Response length limit |
| `timeout` | int | Request timeout in seconds |
| `max_retries` | int | Retry attempts |
| `api_key` | str | Provider credentials |

### Rate limiting

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=0.5)
model = init_chat_model("openai:gpt-4.1", rate_limiter=rate_limiter)
```

### Runtime-configurable model

Switch models without code changes:

```python
configurable_model = init_chat_model(temperature=0)
response = configurable_model.invoke(
    "Hello",
    config={"configurable": {"model": "anthropic:claude-sonnet-4-5-20250929"}}
)
```

### Installation pattern

```bash
uv add langchain                    # core (includes langgraph)
uv add "langchain[openai]"          # OpenAI provider
uv add "langchain[anthropic]"       # Anthropic provider
uv add "langchain[google-genai]"    # Google provider
```

## Quick Checklist
- [ ] Is model initialization explicit (not hardcoded deep in logic)?
- [ ] Is provider package installed for the chosen model?
- [ ] Are rate limits configured for production?
- [ ] Is the model runtime-configurable for multi-environment deployment?

## Next File
- Middleware: `20-middleware.md`
