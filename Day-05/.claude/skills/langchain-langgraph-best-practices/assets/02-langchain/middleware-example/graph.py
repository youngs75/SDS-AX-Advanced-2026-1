"""
Middleware Composition Example

Demonstrates how to compose multiple middleware layers with create_agent:
- SummarizationMiddleware: Condenses conversation history at 4000 tokens
- ModelCallLimitMiddleware: Enforces thread and run limits
- Custom @before_model: Logs model invocations

Asset: assets/02-langchain/middleware-example/graph.py
Reference: references/02-langchain/middleware.md
"""
import asyncio
from langchain_core.language_models import FakeListChatModel
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ModelCallLimitMiddleware,
    before_model,
)
from langchain.tools import tool
from langgraph.runtime import Runtime

# --- Model Configuration ---
# FakeListChatModel: testing/prototyping
# ChatOpenAI(model="gpt-4o"): direct provider
# init_chat_model("openai:gpt-4o"): universal init (recommended)
model = FakeListChatModel(
    responses=[
        "I'll search for that information.",
        "Based on the search results for LangChain middleware patterns, middleware allows you to intercept and modify agent execution at various points.",
    ]
)


# --- Tools ---
@tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    await asyncio.sleep(0.1)  # Simulate async operation
    return f"Search results for: {query}\n- Middleware pattern documentation\n- Best practices guide\n- Implementation examples"


# --- Custom Middleware ---
@before_model
def log_model_call(state: AgentState, runtime: Runtime) -> dict | None:
    """Log each model invocation with message count."""
    print(f"[Middleware] Model call with {len(state['messages'])} messages")
    return None


# --- Agent Creation ---
agent = create_agent(
    model=model,
    tools=[search_web],
    system_prompt="You are a helpful research assistant with expertise in software architecture patterns.",
    middleware=[
        log_model_call,
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)


# --- Main Execution ---
async def main():
    """Run the agent with middleware composition demonstration."""
    print("Starting middleware-enabled agent...\n")

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Search for LangChain middleware patterns and explain the key concepts"
            }
        ]
    })

    print("\n--- Agent Response ---")
    print(result["messages"][-1].content)

    print("\n--- Metadata ---")
    print(f"Total messages in conversation: {len(result['messages'])}")


if __name__ == "__main__":
    asyncio.run(main())
