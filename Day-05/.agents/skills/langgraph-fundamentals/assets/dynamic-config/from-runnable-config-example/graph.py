from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose one of the following options:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

# In this asset, we simulate per-stage model selection using FakeListChatModel.
# Each "model" returns stage-appropriate responses.

drafting_model = FakeListChatModel(
    responses=[
        "Draft: React Server Components render on the server, reducing client-side JavaScript bundles. "
        "They enable direct backend access and improve initial page load performance significantly.",
    ]
)

review_model = FakeListChatModel(
    responses=[
        "Review: The draft accurately covers RSC fundamentals. Suggest adding a comparison table "
        "and noting that RSC require a compatible framework like Next.js 13+.",
    ]
)

final_model = FakeListChatModel(
    responses=[
        "# React Server Components Guide\n\n"
        "## Overview\nReact Server Components render on the server...\n\n"
        "## Comparison\n| Feature | RSC | Traditional |\n|---|---|---|\n"
        "| Rendering | Server | Client |\n| Bundle Size | Smaller | Larger |\n\n"
        "## Conclusion\nRSC represent a paradigm shift in React development.",
    ]
)

# Option 2: Hot-swappable model via init_chat_model (RECOMMENDED for production)
# from langchain.chat_models import init_chat_model
#
# configurable_model = init_chat_model(
#     configurable_fields=("model", "max_tokens", "api_key"),
# )
#
# # Per-stage binding example (used inside node functions):
# drafting_llm = configurable_model.with_config({
#     "model": configurable.drafting_model,         # e.g., "openai:gpt-4.1-mini"
#     "max_tokens": configurable.drafting_max_tokens,
#     "api_key": get_api_key_for_model(configurable.drafting_model, config),
# })

# Option 3: Direct ChatOpenAI (COMMENTED)
# from langchain_openai import ChatOpenAI
# drafting_model = ChatOpenAI(model="gpt-4.1-mini")
# review_model = ChatOpenAI(model="gpt-4.1")
# final_model = ChatOpenAI(model="gpt-4.1")


# ============================================================================
# DYNAMIC CONFIGURATION
# ============================================================================


class OutputFormat(Enum):
    """Enumeration of available output formats (like SearchAPI in open_deep_research)."""

    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN = "plain"


class Configuration(BaseModel):
    """Runtime configuration with per-stage model selection and env var fallback.

    This pattern is extracted from open_deep_research/configuration.py.
    Each field supports: default → config["configurable"] override → env var override.

    In production, pass config_schema=Configuration to StateGraph() so that
    LangGraph Studio renders configuration UI automatically.
    """

    # --- Per-Stage Model Selection ---
    drafting_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "description": "Model for drafting initial content (fast, cheap)."
        },
    )
    drafting_max_tokens: int = Field(
        default=4096,
        metadata={"description": "Max output tokens for drafting model."},
    )
    review_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "description": "Model for reviewing and improving content (accurate)."
        },
    )
    review_max_tokens: int = Field(
        default=8192,
        metadata={"description": "Max output tokens for review model."},
    )
    final_model: str = Field(
        default="anthropic:claude-sonnet-4",
        metadata={
            "description": "Model for generating the final polished output."
        },
    )
    final_max_tokens: int = Field(
        default=10000,
        metadata={"description": "Max output tokens for final model."},
    )

    # --- Feature Selection (Enum-based) ---
    output_format: OutputFormat = Field(
        default=OutputFormat.MARKDOWN,
        metadata={
            "description": "Output format for the final report.",
            "options": [
                {"label": "Markdown", "value": OutputFormat.MARKDOWN.value},
                {"label": "HTML", "value": OutputFormat.HTML.value},
                {"label": "Plain Text", "value": OutputFormat.PLAIN.value},
            ],
        },
    )

    # --- General Configuration ---
    max_retries: int = Field(
        default=3,
        metadata={"description": "Max retries for structured output calls."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create Configuration from RunnableConfig with env var fallback.

        Resolution order per field:
        1. os.environ[FIELD_NAME_UPPER] (highest priority)
        2. config["configurable"][field_name]
        3. Field default value (lowest priority)

        This pattern ensures configuration works across:
        - LangGraph Studio (via configurable)
        - CLI / scripts (via env vars)
        - Default development mode (via field defaults)
        """
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(
                field_name.upper(), configurable.get(field_name)
            )
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})


# ============================================================================
# API KEY ROUTING
# ============================================================================


def get_api_key_for_model(model_name: str, config: RunnableConfig) -> Optional[str]:
    """Route API key based on model provider prefix.

    Supports: openai:, anthropic:, google: prefixes.
    Falls back to environment variables.
    """
    model_name = model_name.lower()
    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):
        return os.getenv("GOOGLE_API_KEY")
    return None


# ============================================================================
# STATE DEFINITION
# ============================================================================


class ConfigDemoState(TypedDict):
    """State for the configuration demo graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    draft: str
    review_feedback: str
    final_output: str


# ============================================================================
# NODE FUNCTIONS (ALL ASYNC)
# ============================================================================


async def draft_node(state: ConfigDemoState, config: RunnableConfig) -> dict:
    """Draft initial content using the drafting model.

    In production with init_chat_model:
        configurable = Configuration.from_runnable_config(config)
        llm = configurable_model.with_config({
            "model": configurable.drafting_model,
            "max_tokens": configurable.drafting_max_tokens,
            "api_key": get_api_key_for_model(configurable.drafting_model, config),
        })
    """
    messages = state["messages"]
    response = await drafting_model.ainvoke(messages)
    return {"draft": response.content, "messages": [response]}


async def review_node(state: ConfigDemoState, config: RunnableConfig) -> dict:
    """Review and improve draft using the review model.

    Demonstrates per-stage model switching — a different (more capable)
    model is used for review vs drafting.
    """
    draft = state.get("draft", "")
    review_prompt = HumanMessage(
        content=f"Review and suggest improvements for this draft:\n\n{draft}"
    )
    response = await review_model.ainvoke([review_prompt])
    return {"review_feedback": response.content, "messages": [response]}


async def finalize_node(state: ConfigDemoState, config: RunnableConfig) -> dict:
    """Generate final polished output using the final model.

    In production, this uses a third model (potentially from a different provider)
    for the highest quality final output.
    """
    configurable = Configuration.from_runnable_config(config)
    draft = state.get("draft", "")
    feedback = state.get("review_feedback", "")

    prompt = HumanMessage(
        content=(
            f"Create a final polished version incorporating the feedback.\n\n"
            f"Draft:\n{draft}\n\nReview Feedback:\n{feedback}\n\n"
            f"Output format: {configurable.output_format.value}"
        )
    )
    response = await final_model.ainvoke([prompt])
    return {"final_output": response.content, "messages": [response]}


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_graph():
    """Create graph with Configuration as config_schema.

    Passing config_schema=Configuration enables:
    - LangGraph Studio auto-renders configuration UI
    - Runtime config override via config["configurable"]
    - Environment variable fallback via from_runnable_config()
    """
    graph = StateGraph(ConfigDemoState, config_schema=Configuration)

    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


graph = create_graph()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Run the configuration demo graph.

    Shows two invocation styles:
    1. Default configuration (uses field defaults)
    2. Runtime override via config["configurable"]
    """
    # Style 1: Default configuration
    result = await graph.ainvoke({
        "messages": [HumanMessage(content="Explain React Server Components")],
        "draft": "",
        "review_feedback": "",
        "final_output": "",
    })

    print("\n" + "=" * 80)
    print("DYNAMIC CONFIGURATION DEMO (Default Config)")
    print("=" * 80)
    print(f"\nDraft:\n{result['draft']}")
    print(f"\nReview:\n{result['review_feedback']}")
    print(f"\nFinal:\n{result['final_output']}")

    # Style 2: Runtime config override
    # result = await graph.ainvoke(
    #     {"messages": [HumanMessage(content="Explain RSC")], ...},
    #     config={
    #         "configurable": {
    #             "drafting_model": "openai:gpt-4.1-mini",
    #             "review_model": "anthropic:claude-sonnet-4",
    #             "final_model": "anthropic:claude-sonnet-4",
    #             "output_format": "html",
    #         }
    #     },
    # )

    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
