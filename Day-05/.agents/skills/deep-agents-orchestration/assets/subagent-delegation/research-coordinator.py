"""
SubAgent delegation: research coordinator with 3 specialized sub-agents.

Demonstrates:
- SubAgent dictionary definitions (researcher, analyst, writer)
- Delegation contracts: goal, constraints, output_format
- Inheritance: default_model, default_tools
- CompiledSubAgent alternative
- general_purpose_agent flag
- Context isolation and least-privilege tool assignment
"""

import asyncio
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
# from deepagents import create_deep_agent
# from deepagents.types import SubAgent, CompiledSubAgent
# from langgraph.checkpoint.memory import InMemorySaver

# --- Model Configuration ---
model = FakeListChatModel(
    responses=[
        "I'll coordinate the research team to analyze market trends.",
        "Research complete. The analyst found 3 key insights and the writer produced the report.",
    ]
)

# ==== Custom Tools (domain-specific) ====


@tool
async def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


@tool
async def analyze_data(data: str) -> str:
    """Analyze data and extract patterns."""
    return f"Analysis of: {data} → 3 key patterns found"


@tool
async def write_report(content: str, format: str = "markdown") -> str:
    """Write a formatted report."""
    return f"Report written in {format}: {content[:100]}..."


# ==== SubAgent Definitions ====

# Pattern 1: SubAgent as dictionary
researcher_agent = {
    "name": "researcher",
    "description": "Searches the web and gathers raw information on a topic",
    "system_prompt": (
        "You are a research specialist. Your goal is to find accurate, "
        "up-to-date information. Always cite sources."
    ),
    "tools": [web_search],  # Least-privilege: only search tool
    # "model": "anthropic:claude-sonnet-4-20250514",  # Override parent model
}

analyst_agent = {
    "name": "analyst",
    "description": "Analyzes raw data and extracts actionable insights",
    "system_prompt": (
        "You are a data analyst. Focus on patterns, trends, and "
        "actionable insights. Be quantitative when possible."
    ),
    "tools": [analyze_data],  # Least-privilege: only analysis tool
}

writer_agent = {
    "name": "writer",
    "description": "Produces clear, well-structured reports from analyzed data",
    "system_prompt": (
        "You are a technical writer. Create concise, well-organized "
        "reports. Use headers, bullet points, and clear language."
    ),
    "tools": [write_report],  # Least-privilege: only writing tool
}

# Pattern 2: CompiledSubAgent (pre-built LangGraph graph)
# from langgraph.graph import StateGraph
# custom_graph = StateGraph(...).compile()
# compiled_agent = CompiledSubAgent(
#     name="custom_pipeline",
#     description="Pre-compiled analysis pipeline",
#     runnable=custom_graph,
# )


# ==== Delegation Contracts ====

# Each delegation should specify:
delegation_contract_example = {
    "goal": "Research market trends in AI agent frameworks for 2025",
    "constraints": [
        "Focus on open-source frameworks only",
        "Include adoption metrics if available",
        "Maximum 3 sources per finding",
    ],
    "output_format": "Structured JSON with findings, sources, confidence",
    "acceptance_criteria": [
        "At least 5 distinct findings",
        "Each finding has at least 1 source",
        "Confidence score for each finding",
    ],
}


# ==== Agent Setup ====

# agent = create_deep_agent(
#     model=model,
#     tools=[],  # Parent has no direct tools; delegates via task()
#     subagents=[researcher_agent, analyst_agent, writer_agent],
#     # default_model="anthropic:claude-sonnet-4-20250514",  # Inherited by subagents
#     # default_tools=[web_search],  # Inherited by all subagents
#     # general_purpose_agent=True,  # Auto-create fallback subagent
#     checkpointer=InMemorySaver(),
# )


# ==== Main ====


async def main():
    print("=== SubAgent Delegation Pattern ===")
    print()

    # Show SubAgent definitions
    agents = [researcher_agent, analyst_agent, writer_agent]
    for sa in agents:
        print(f"SubAgent: {sa['name']}")
        print(f"  Description: {sa['description']}")
        print(f"  Tools: {[t.name for t in sa['tools']]}")
        print(f"  System prompt: {sa['system_prompt'][:60]}...")
        print()

    # Show delegation contract
    print("--- Delegation Contract Example ---")
    for key, value in delegation_contract_example.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    print()

    # Demonstrate tool execution
    print("--- Tool Execution Demo ---")
    r1 = await web_search.ainvoke({"query": "AI agent frameworks 2025"})
    print(f"  researcher: {r1}")
    r2 = await analyze_data.ainvoke({"data": r1})
    print(f"  analyst: {r2}")
    r3 = await write_report.ainvoke({"content": r2, "format": "markdown"})
    print(f"  writer: {r3}")

    # Inheritance rules
    print()
    print("--- Inheritance Rules ---")
    print("  default_model    → All subagents without explicit model")
    print("  default_tools    → Appended to each subagent's tools")
    print("  default_middleware → Applied to all subagent graphs")
    print("  State isolation  → messages, todos are NOT shared")


if __name__ == "__main__":
    asyncio.run(main())
