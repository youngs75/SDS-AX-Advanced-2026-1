# MCP (Model Context Protocol)

## Read This When
- Connecting an agent to external MCP tool servers
- Configuring multi-server MCP client
- Need to filter or intercept MCP-provided tools

## Skip This When
- All tools are defined locally within the application

## Official References
1. https://docs.langchain.com/oss/python/langchain/mcp
   - Why: MCP adapter integration model and transport configuration

## Core Guidance

### What MCP is

Open protocol standardizing how apps provide tools and context to LLMs. `langchain-mcp-adapters` bridges MCP servers into LangChain's tool system.

### Installation

```bash
uv add langchain-mcp-adapters
```

### MultiServerMCPClient — the primary interface

Stateless by default, manages connections to multiple MCP servers:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": ["/path/to/math_server.py"],
    },
    "weather": {
        "transport": "http",
        "url": "http://localhost:8000/mcp",
    },
})

tools = await client.get_tools()
agent = create_agent("openai:gpt-4.1", tools)
```

### Transport types

| Transport | Use When | Config Key |
|-----------|----------|------------|
| `stdio` | Local tools, simple setups | `command`, `args` |
| `http` | Remote servers, production | `url`, `headers` |
| `sse` | Legacy MCP servers | `url` |

### HTTP transport with auth

```python
client = MultiServerMCPClient({
    "api": {
        "transport": "http",
        "url": "https://api.example.com/mcp",
        "headers": {"Authorization": "Bearer TOKEN"},
    },
})
```

### Tool interceptors (middleware for MCP tools)

```python
async def logging_interceptor(request, handler):
    print(f"Calling MCP tool: {request.name} with {request.args}")
    result = await handler(request)
    print(f"Result: {result}")
    return result

async def retry_interceptor(request, handler, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[logging_interceptor, retry_interceptor]
)
```

### Resources and Prompts

```python
# Fetch data exposed by MCP servers
blobs = await client.get_resources("server_name")
for blob in blobs:
    print(f"URI: {blob.metadata['uri']}, Content: {blob.as_string()}")

# Retrieve reusable prompt templates
messages = await client.get_prompt("server_name", "summarize")
messages = await client.get_prompt(
    "server_name",
    "code_review",
    arguments={"language": "python"}
)
```

### Stateful sessions (when server needs persistent context)

```python
async with client.session("server_name") as session:
    tools = await load_mcp_tools(session)
    agent = create_agent("openai:gpt-4.1", tools)
    result = await agent.ainvoke({"messages": [...]})
```

### Creating an MCP server (with FastMCP)

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run(transport="stdio")  # or "streamable-http"
```

### Full integration pattern

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

async def main():
    client = MultiServerMCPClient({
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["math_server.py"]
        },
    })

    tools = await client.get_tools()
    agent = create_agent("openai:gpt-4.1", tools)

    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "What's (3 + 5) × 12?"
        }]
    })

    print(result["messages"][-1].content)

asyncio.run(main())
```

## Quick Checklist
- [ ] Is `langchain-mcp-adapters` installed?
- [ ] Is transport type appropriate (stdio for local, http for remote)?
- [ ] Are interceptors used for logging/retry on unreliable servers?
- [ ] Is MCP used only where tool ownership is external?

## Next File
- description: `35-tools.md`
