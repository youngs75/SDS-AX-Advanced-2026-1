"""
Multi-stage graph pipeline demonstrating sequential node processing with typed state.

Shows how to build a LangGraph StateGraph pipeline where each node performs a
distinct processing step (refine → research → report), passing results via
shared state with add_messages reducer.

For universal prompt engineering patterns (XML sections, template variables,
few-shot injection), see: assets/01-langchain-core/prompt-patterns/

Asset: assets/03-langgraph/multi-stage-pipeline/graph.py
Reference: references/03-langgraph/65-workflow-patterns.md (Pattern 1: Prompt Chaining)
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Choose one of the following options:

# Option 1: FakeListChatModel for testing (ACTIVE)
from langchain_core.language_models import FakeListChatModel

model = FakeListChatModel(
    responses=[
        # Stage 1: Question refinement output
        '{"refined_question": "What are the key differences between React Server Components and traditional client-side React, including performance implications, use cases, and migration strategies?"}',
        # Stage 2: Research output
        "## Research Findings\n\nReact Server Components (RSC) represent a paradigm shift...\n\n### Key Differences\n1. **Rendering Location**: RSC render on server, traditional on client\n2. **Bundle Size**: RSC reduce client bundle significantly\n3. **Data Fetching**: RSC access backend directly\n\n### Sources\n[1] React Documentation: https://react.dev/reference/rsc",
        # Stage 3: Final report output
        "# React Server Components: A Comprehensive Analysis\n\n## Overview\nReact Server Components fundamentally change how React applications handle rendering...\n\n## Key Findings\n...\n\n## Conclusion\n...\n\n### Sources\n[1] React Official Docs: https://react.dev",
    ]
)

# Option 2: OpenAI via ChatOpenAI (COMMENTED)
# from langchain_openai import ChatOpenAI
# model = ChatOpenAI(model="gpt-5")

# Option 3: Any provider via init_chat_model (COMMENTED)
# from langchain.chat_models import init_chat_model
# model = init_chat_model("openai:gpt-5")


# ============================================================================
# PROMPT TEMPLATES (XML-STRUCTURED)
# ============================================================================

# Stage 1: Refine the user's question into a detailed research question
# Uses structured output (JSON keys) for reliable parsing
question_refinement_prompt = """You will be given a user question and today's date.
Your job is to refine the question into a more detailed, specific research question.

<Context>
Today's date is {date}.
</Context>

<Instructions>
1. Maximize specificity — include all dimensions needed for a comprehensive answer.
2. Fill in unstated but necessary dimensions as open-ended rather than assuming.
3. Avoid unwarranted assumptions — if unspecified, state it as flexible.
4. Phrase the refined question from the user's perspective (first person).
</Instructions>

Respond in valid JSON format:
{{"refined_question": "<your refined, detailed research question>"}}
"""

# Stage 2: Research system prompt with XML sections for role, tools, limits, and thinking
# Demonstrates the full XML-structured prompt pattern from open_deep_research
research_system_prompt = """You are a research assistant conducting research on the user's input topic.
For context, today's date is {date}.

<Task>
Your job is to gather comprehensive information about the research question.
Use broad searches first, then narrow down to fill gaps.
</Task>

<Instructions>
Think like a human researcher with limited time. Follow these steps:

1. **Read the question carefully** — What specific information is needed?
2. **Start broad** — Use comprehensive queries first
3. **Assess after each step** — Do I have enough? What's missing?
4. **Fill gaps** — Execute targeted searches for missing information
5. **Stop when confident** — Don't keep searching for perfection
</Instructions>

<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: 2-3 search calls maximum
- **Complex queries**: Up to {max_iterations} search calls maximum
- **Always stop**: After {max_iterations} calls if sources not found

**Stop Immediately When**:
- You can answer the question comprehensively
- You have 3+ relevant sources
- Last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search, analyze:
- What key information did I find?
- What's missing?
- Do I have enough for a comprehensive answer?
- Should I search more or provide my answer?
</Show Your Thinking>
"""

# Stage 3: Final report generation prompt with citation rules
final_report_prompt = """Based on the research conducted, create a comprehensive report answering the research question.

<Research Question>
{topic}
</Research Question>

<Findings>
{findings}
</Findings>

<Instructions>
1. Organize with proper headings (## for sections, ### for subsections)
2. Include specific facts and insights from the research
3. Reference sources using [Title](URL) format
4. Be comprehensive — users expect detailed, thorough answers
5. Use simple, clear language
6. Do NOT refer to yourself as the writer
</Instructions>

<Citation Rules>
- Assign each unique URL a single citation number
- End with ### Sources listing each source with numbers
- Number sources sequentially without gaps (1, 2, 3...)
- Format: [1] Source Title: URL
</Citation Rules>

CRITICAL: Write the report in the same language as the original question: {language}
"""


# ============================================================================
# STRUCTURED OUTPUT MODELS
# ============================================================================


class RefinedQuestion(BaseModel):
    """Structured output for question refinement stage."""

    refined_question: str = Field(
        description="A detailed, specific research question refined from the user's input."
    )


# ============================================================================
# STATE DEFINITION
# ============================================================================


class PromptPipelineState(TypedDict):
    """State for the multi-stage prompt pipeline."""

    messages: Annotated[list[BaseMessage], add_messages]
    refined_question: str
    research_findings: str
    final_report: str
    language: str


# ============================================================================
# NODE FUNCTIONS (ALL ASYNC)
# ============================================================================


async def refine_question(state: PromptPipelineState) -> dict:
    """Stage 1: Refine user question into a detailed research question.

    Demonstrates:
    - XML <Instructions> section for structured guidance
    - JSON-keyed structured output prompt
    - .format() variable injection ({date})
    """
    messages = state["messages"]
    user_question = messages[-1].content if messages else ""

    # Format prompt with dynamic date
    prompt = question_refinement_prompt.format(
        date=datetime.now().strftime("%a %b %d, %Y")
    )

    response = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=user_question),
        ]
    )

    # Parse structured JSON response
    # NOTE: With real models, use model.with_structured_output(RefinedQuestion) instead
    content = response.content if hasattr(response, "content") else str(response)
    try:
        parsed = json.loads(content)
        refined = parsed.get("refined_question", user_question)
    except (json.JSONDecodeError, AttributeError):
        refined = user_question

    return {
        "refined_question": refined,
        "language": "English",  # detect from user_question in production
    }


async def conduct_research(state: PromptPipelineState) -> dict:
    """Stage 2: Research using XML-structured prompt with limits and thinking.

    Demonstrates:
    - Full XML sections: <Task>, <Instructions>, <Hard Limits>, <Show Your Thinking>
    - Dynamic variable injection ({date}, {max_iterations})
    - Role-based system prompt design
    """
    refined_question = state.get("refined_question", "")

    # Format research prompt with dynamic config values
    prompt = research_system_prompt.format(
        date=datetime.now().strftime("%a %b %d, %Y"),
        max_iterations=5,  # from Configuration in production
    )

    response = await model.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=refined_question),
        ]
    )

    return {"research_findings": response.content}


async def generate_report(state: PromptPipelineState) -> dict:
    """Stage 3: Generate final report with citation rules.

    Demonstrates:
    - Multi-variable .format() injection ({topic}, {findings}, {language})
    - <Citation Rules> XML section for strict formatting
    - Language consistency instruction
    """
    prompt = final_report_prompt.format(
        topic=state.get("refined_question", ""),
        findings=state.get("research_findings", ""),
        language=state.get("language", "English"),
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])

    return {
        "final_report": response.content,
        "messages": [AIMessage(content=response.content)],
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================


def create_graph():
    """Create the multi-stage prompt pipeline graph.

    Pipeline: refine_question → conduct_research → generate_report
    Each stage uses XML-structured prompts with .format() injection.
    """
    graph = StateGraph(PromptPipelineState)

    # Add pipeline stages as nodes
    graph.add_node("refine_question", refine_question)
    graph.add_node("conduct_research", conduct_research)
    graph.add_node("generate_report", generate_report)

    # Linear pipeline: START → refine → research → report → END
    graph.add_edge(START, "refine_question")
    graph.add_edge("refine_question", "conduct_research")
    graph.add_edge("conduct_research", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


graph = create_graph()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    """Run the multi-stage prompt pipeline."""
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="What are React Server Components and how do they compare to traditional React?"
                )
            ],
            "refined_question": "",
            "research_findings": "",
            "final_report": "",
            "language": "English",
        }
    )

    print("\n" + "=" * 80)
    print("XML-STRUCTURED PROMPT PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nRefined Question:\n{result['refined_question']}")
    print(f"\nFinal Report:\n{result['final_report']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
