"""
MCP Client Integration Example

Demonstrates Model Context Protocol (MCP) client integration with LangChain agents:
- MultiServerMCPClient configuration with stdio transport
- Tool interceptors for logging and monitoring
- Seamless integration with create_agent

Asset: assets/02-langchain/mcp-client-example/graph.py
Reference: references/02-langchain/mcp-integration.md
"""
import asyncio
from langchain_core.language_models import FakeListChatModel
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# --- Model Configuration ---
# FakeListChatModel: testing/prototyping
# ChatOpenAI(model="gpt-4o"): direct provider
# init_chat_model("openai:gpt-4o"): universal init (recommended)
model = FakeListChatModel(
    responses=[
        "I'll calculate that for you.",
        "The result of (3 + 5) × 12 is 96.",
    ]
)


# --- MCP Interceptors ---
async def logging_interceptor(request, handler):
    """
    Log all MCP tool calls for debugging and monitoring.

    Args:
        request: The MCP tool request object
        handler: The next handler in the chain

    Returns:
        The result from the handler
    """
    print(f"[MCP] Calling: {request.name}")
    print(f"[MCP] Arguments: {request.args}")

    try:
        result = await handler(request)
        print(f"[MCP] Result: {result}")
        return result
    except Exception as e:
        print(f"[MCP] Error: {e}")
        raise


async def performance_interceptor(request, handler):
    """
    Track execution time for MCP tool calls.

    Args:
        request: The MCP tool request object
        handler: The next handler in the chain

    Returns:
        The result from the handler
    """
    import time

    start_time = time.time()
    result = await handler(request)
    elapsed = time.time() - start_time

    print(f"[MCP Performance] {request.name} took {elapsed:.3f}s")
    return result


# --- Main Execution ---
async def main():
    """
    Run the MCP client integration demonstration.

    This example shows how to:
    1. Configure MultiServerMCPClient with stdio transport
    2. Apply tool interceptors for logging and monitoring
    3. Integrate MCP tools with a LangChain agent
    4. Invoke the agent with MCP-backed tools
    """
    print("Starting MCP client integration...\n")

    # Initialize MCP client with multiple server configurations
    # Note: This example uses a math_server.py that would need to exist
    # In production, you'd configure actual MCP servers here
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["math_server.py"],
            },
            # Example of additional server configuration
            # "filesystem": {
            #     "transport": "stdio",
            #     "command": "npx",
            #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed"],
            # },
        },
        tool_interceptors=[
            logging_interceptor,
            performance_interceptor,
        ],
    )

    print("[MCP] Client initialized with servers: math")

    # Retrieve all available tools from configured MCP servers
    try:
        tools = await client.get_tools()
        print(f"[MCP] Retrieved {len(tools)} tools from servers")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
    except Exception as e:
        print(f"[MCP] Warning: Could not retrieve tools: {e}")
        print("[MCP] Using demo mode without actual MCP tools")
        tools = []

    # Create agent with MCP tools
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="You are a helpful assistant with access to external tools via MCP. Use available tools to answer questions accurately.",
    )

    print("\n--- Agent Invocation ---")
    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "What is (3 + 5) × 12?"
            }
        ]
    })

    print("\n--- Agent Response ---")
    print(result["messages"][-1].content)

    print("\n--- Conversation History ---")
    for i, msg in enumerate(result["messages"], 1):
        role = msg.__class__.__name__
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. {role}: {content}")

    # Cleanup
    await client.close()
    print("\n[MCP] Client closed successfully")


if __name__ == "__main__":
    asyncio.run(main())
