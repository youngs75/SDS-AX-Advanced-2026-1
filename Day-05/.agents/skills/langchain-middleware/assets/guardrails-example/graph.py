"""
Guardrails Composition Example

Demonstrates layered guardrails with multiple middleware:
- PIIMiddleware: Redacts emails, masks credit cards
- ContentFilterMiddleware: Blocks banned keywords
- HumanInTheLoopMiddleware: Requires approval for sensitive tools

Asset: assets/02-langchain/guardrails-example/graph.py
Reference: references/02-langchain/middleware.md
"""
import asyncio
from langchain_core.language_models import FakeListChatModel
# from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    PIIMiddleware,
    HumanInTheLoopMiddleware,
    AgentMiddleware,
    hook_config,
)
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

# --- Model Configuration ---
# FakeListChatModel: testing/prototyping
# ChatOpenAI(model="gpt-4o"): direct provider
# init_chat_model("openai:gpt-4o"): universal init (recommended)
model = FakeListChatModel(
    responses=[
        "I'll search for that information.",
        "Here are the results from my search.",
        "I'll send that email now.",
        "Email has been sent successfully.",
    ]
)


# --- Tools ---
@tool
async def search(query: str) -> str:
    """Search for information."""
    await asyncio.sleep(0.1)  # Simulate async operation
    return f"Results for: {query}\n- Item 1: Relevant documentation\n- Item 2: Tutorial guide\n- Item 3: Code examples"


@tool
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified address."""
    await asyncio.sleep(0.1)  # Simulate async operation
    return f"Email sent to {to}: {subject}\nBody preview: {body[:50]}..."


# --- Custom Middleware ---
class ContentFilterMiddleware(AgentMiddleware):
    """Block requests containing banned keywords and terminate execution."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Check user messages for banned content before agent processing."""
        if not state["messages"]:
            return None

        # Check the first user message
        content = state["messages"][0].content.lower()

        for keyword in self.banned_keywords:
            if keyword in content:
                print(f"[ContentFilter] Blocked request containing: '{keyword}'")
                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "I cannot process this request. It contains prohibited content."
                        }
                    ],
                    "jump_to": "end",
                }

        return None


# --- Agent Creation ---
agent = create_agent(
    model=model,
    tools=[search, send_email],
    system_prompt="You are a helpful assistant with strict content and privacy policies.",
    checkpointer=InMemorySaver(),
    middleware=[
        ContentFilterMiddleware(banned_keywords=["hack", "exploit", "illegal"]),
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True,  # Require approval for emails
                "search": False,     # No approval needed for searches
            }
        ),
    ],
)


# --- Main Execution ---
async def main():
    """Run the agent with guardrails demonstration."""
    print("Starting guardrails-enabled agent...\n")

    # Test 1: Normal search (should work)
    print("=== Test 1: Normal Search ===")
    result1 = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Search for information about data privacy best practices"
            }
        ]
    })
    print(f"Response: {result1['messages'][-1].content}\n")

    # Test 2: Message with PII (should be redacted)
    print("=== Test 2: PII Redaction ===")
    result2 = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Search for info about john.doe@example.com and card 4532-1234-5678-9010"
            }
        ]
    })
    print(f"Processed input with PII redacted")
    print(f"Response: {result2['messages'][-1].content}\n")

    # Test 3: Blocked content (should be rejected)
    print("=== Test 3: Content Filter ===")
    result3 = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "How can I hack into a system?"
            }
        ]
    })
    print(f"Response: {result3['messages'][-1].content}\n")

    print("--- Guardrails demonstration complete ---")


if __name__ == "__main__":
    asyncio.run(main())
