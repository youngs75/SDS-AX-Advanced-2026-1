from __future__ import annotations

import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose one of the following options:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        "# Comprehensive Report\n\nBased on the research findings, here is a detailed analysis...\n\n"
        "## Key Findings\n1. Finding one with supporting evidence\n2. Finding two with data\n\n"
        "## Conclusion\nThe research indicates clear trends in the domain.",
    ]
)

# Option 2: Any provider via init_chat_model (COMMENTED)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")

# Option 3: OpenAI via ChatOpenAI (COMMENTED)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")


# ============================================================================
# TOKEN LIMIT DETECTION (Multi-Provider)
# ============================================================================

# NOTE: Update this mapping as new models are released.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1_047_576,
    "openai:gpt-4.1": 1_047_576,
    "anthropic:claude-opus-4": 200_000,
    "anthropic:claude-sonnet-4": 200_000,
}


def get_model_token_limit(model_string: str) -> int | None:
    """Look up the token limit for a specific model.

    Returns None if model not found in lookup table.
    """
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    return None


def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.

    Dispatches to provider-specific checkers based on model name prefix.
    If model_name is None, checks all providers.
    """
    error_str = str(exception).lower()

    # Determine provider from model name
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith("openai:"):
            provider = "openai"
        elif model_str.startswith("anthropic:"):
            provider = "anthropic"
        elif model_str.startswith("gemini:") or model_str.startswith("google:"):
            provider = "gemini"

    # Check provider-specific patterns
    if provider == "openai":
        return _check_openai_token_limit(exception, error_str)
    elif provider == "anthropic":
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == "gemini":
        return _check_gemini_token_limit(exception, error_str)

    # Unknown provider: check all
    return (
        _check_openai_token_limit(exception, error_str)
        or _check_anthropic_token_limit(exception, error_str)
        or _check_gemini_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded.

    Detects: BadRequestError with 'context_length_exceeded' or token-related keywords.
    """
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    is_openai = (
        "openai" in str(type(exception)).lower() or "openai" in module_name.lower()
    )
    is_bad_request = class_name in ["BadRequestError", "InvalidRequestError"]

    if is_openai and is_bad_request:
        keywords = ["token", "context", "length", "maximum context", "reduce"]
        if any(kw in error_str for kw in keywords):
            return True

    # Check for specific error codes
    if (
        hasattr(exception, "code")
        and getattr(exception, "code", "") == "context_length_exceeded"
    ):
        return True

    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded.

    Detects: BadRequestError with 'prompt is too long'.
    """
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    is_anthropic = (
        "anthropic" in str(type(exception)).lower()
        or "anthropic" in module_name.lower()
    )
    is_bad_request = class_name == "BadRequestError"

    if is_anthropic and is_bad_request and "prompt is too long" in error_str:
        return True
    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded.

    Detects: ResourceExhausted or GoogleGenerativeAIFetchError.
    """
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    is_google = (
        "google" in str(type(exception)).lower() or "google" in module_name.lower()
    )
    is_exhausted = class_name in ["ResourceExhausted", "GoogleGenerativeAIFetchError"]

    if is_google and is_exhausted:
        return True
    if "google.api_core.exceptions.resourceexhausted" in str(type(exception)).lower():
        return True
    return False


# ============================================================================
# MESSAGE TRUNCATION
# ============================================================================


def remove_up_to_last_ai_message(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Truncate message history by removing up to the last AI message.

    Useful for handling token limit errors by removing the most recent
    AI-generated context that likely pushed the conversation over the limit.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]
    return messages


# ============================================================================
# STATE DEFINITION
# ============================================================================


class TokenResilientState(TypedDict):
    """State for the token-resilient report generation."""

    messages: Annotated[list[BaseMessage], add_messages]
    findings: str
    final_report: str
    model_name: str


# ============================================================================
# NODE FUNCTIONS (ALL ASYNC)
# ============================================================================

MAX_RETRIES = 3


async def generate_report_with_retry(state: TokenResilientState) -> dict:
    """Generate report with progressive truncation retry on token limit errors.

    Retry strategy (from open_deep_research):
    1. First attempt: use full findings
    2. On token limit error: truncate to model_limit * 4 characters
    3. Each subsequent retry: reduce by 10%
    4. After MAX_RETRIES: return graceful error message (don't raise)

    This ensures the agent never crashes from token limits — it degrades
    gracefully by progressively reducing input size.
    """
    findings = state.get("findings", "")
    model_name = state.get("model_name", "openai:gpt-4.1")
    findings_limit = None
    current_retry = 0

    while current_retry <= MAX_RETRIES:
        try:
            # Apply truncation if we're retrying
            truncated_findings = (
                findings[:findings_limit] if findings_limit else findings
            )

            prompt = (
                f"Create a comprehensive report based on these findings:\n\n"
                f"{truncated_findings}\n\n"
                f"Include proper headings, citations, and a conclusion."
            )

            response = await model.ainvoke([HumanMessage(content=prompt)])

            return {
                "final_report": response.content,
                "messages": [AIMessage(content=response.content)],
            }

        except Exception as e:
            if is_token_limit_exceeded(e, model_name):
                current_retry += 1

                if current_retry == 1:
                    # First retry: estimate character limit from model's token limit
                    token_limit = get_model_token_limit(model_name)
                    if not token_limit:
                        # Unknown model: can't estimate, return error
                        return {
                            "final_report": (
                                f"Error: Token limit exceeded but model '{model_name}' "
                                f"not in MODEL_TOKEN_LIMITS. Update the mapping. ({e})"
                            ),
                            "messages": [
                                AIMessage(
                                    content="Report generation failed: unknown model limit"
                                )
                            ],
                        }
                    # ~4 characters per token as rough estimate
                    findings_limit = token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_limit = int(findings_limit * 0.9)

                continue
            else:
                # Non-token error: return immediately
                return {
                    "final_report": f"Error generating report: {e}",
                    "messages": [AIMessage(content="Report generation failed")],
                }

    # All retries exhausted: graceful fallback
    return {
        "final_report": "Error: Report generation failed after maximum retries due to token limits.",
        "messages": [AIMessage(content="Report generation failed after max retries")],
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_graph():
    """Create the token-resilient report generation graph.

    Single node with built-in retry logic. In production, this pattern
    is used for both compress_research and final_report_generation stages.
    """
    graph = StateGraph(TokenResilientState)

    graph.add_node("generate_report", generate_report_with_retry)

    graph.add_edge(START, "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


graph = create_graph()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Run the token-resilient report generation demonstration."""
    # Simulate large research findings
    sample_findings = (
        "## Research Finding 1\nModern web frameworks show 40% improvement...\n\n"
        "## Research Finding 2\nServer-side rendering reduces Time to Interactive...\n\n"
        "## Research Finding 3\nEdge computing deployment cuts latency by 60%...\n\n"
        "[1] Web Framework Benchmark 2025: https://example.com/benchmarks\n"
        "[2] SSR Performance Study: https://example.com/ssr-study\n"
    )

    result = await graph.ainvoke(
        {
            "messages": [],
            "findings": sample_findings,
            "final_report": "",
            "model_name": "openai:gpt-4.1",
        }
    )

    print("\n" + "=" * 80)
    print("TOKEN-RESILIENT REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nFinal Report:\n{result['final_report']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
