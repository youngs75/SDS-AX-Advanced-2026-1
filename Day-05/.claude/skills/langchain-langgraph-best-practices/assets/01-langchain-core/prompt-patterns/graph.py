"""
Prompt Engineering Patterns - LangChain Core

Demonstrates universal prompt engineering patterns using only langchain_core:

1. XML-structured prompt sections (<Task>, <Instructions>, <Hard Limits>, <Context>)
2. Python .format() template variable injection with {{}} double-brace escaping
3. Few-shot example injection using alternating HumanMessage/AIMessage pairs
4. Prompt composition: building complete prompts from reusable sections
5. SystemMessage wrapping for final prompt
6. Role-based prompt construction (system/user/assistant)

Convention: Only imports from langchain_core and pydantic (NO langchain/langgraph/deepagents).

Reference: references/01-langchain-core/40-core-best-practices.md
"""

import asyncio
from typing import List, Dict, Any

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.language_models import FakeListChatModel

# Commented-out real model imports (production usage)
# from langchain_openai import ChatOpenAI
# from langchain_core.language_models import init_chat_model


# ============================================================================
# XML-Structured Prompt Templates
# ============================================================================

TASK_SECTION = """<Task>
You are a helpful AI assistant specialized in {domain}.
Your primary goal is to {goal}.
</Task>"""

INSTRUCTIONS_SECTION = """<Instructions>
1. {instruction_1}
2. {instruction_2}
3. {instruction_3}
4. Always cite sources when referencing external information
5. Be concise but complete in your responses
</Instructions>"""

HARD_LIMITS_SECTION = """<Hard Limits>
- NEVER fabricate information or sources
- NEVER provide medical, legal, or financial advice
- ALWAYS acknowledge uncertainty when present
- Response length: {min_words}-{max_words} words
</Hard Limits>"""

CONTEXT_SECTION = """<Context>
User background: {user_background}
Session context: {session_context}
</Context>"""

CITATION_RULES_SECTION = """<Citation Rules>
When citing sources:
- Format: [Source: <name>] at end of relevant sentence
- Provide specific page numbers or URLs when available
- Distinguish between direct quotes and paraphrasing
</Citation Rules>"""

THINKING_SECTION = """<Show Your Thinking>
Before providing your final answer:
1. Break down the question into sub-components
2. Consider multiple perspectives or approaches
3. Identify any assumptions or uncertainties
4. Show your reasoning process
</Show Your Thinking>"""


# ============================================================================
# Template Variable Injection with {{}} Escaping
# ============================================================================

TEMPLATE_WITH_EXAMPLES = """<Task>
Generate {output_format} code that {task_description}.
</Task>

<Instructions>
1. Use best practices for {language}
2. Include error handling
3. Add inline comments for complex logic
4. Follow style guide: {{pep8}} for Python, {{eslint}} for JavaScript
</Instructions>

<Examples>
{examples}
</Examples>

<Hard Limits>
- Code must be production-ready
- Maximum complexity: O({complexity})
- Test coverage: {coverage}%
</Hard Limits>"""


# ============================================================================
# Few-Shot Example Injection
# ============================================================================

def create_few_shot_examples(examples: List[Dict[str, str]]) -> List[BaseMessage]:
    """
    Convert example pairs into alternating HumanMessage/AIMessage format.

    Args:
        examples: List of dicts with 'input' and 'output' keys

    Returns:
        List of BaseMessage objects for few-shot learning
    """
    messages = []
    for example in examples:
        messages.append(HumanMessage(content=example["input"]))
        messages.append(AIMessage(content=example["output"]))
    return messages


# ============================================================================
# Prompt Composition Functions
# ============================================================================

def compose_system_prompt(
    domain: str,
    goal: str,
    instructions: List[str],
    min_words: int = 50,
    max_words: int = 200,
    user_background: str = "General user",
    session_context: str = "New session",
    include_citations: bool = True,
    include_thinking: bool = False,
) -> SystemMessage:
    """
    Compose a complete system prompt from reusable sections.

    Demonstrates:
    - Dynamic section assembly based on flags
    - Template variable injection
    - SystemMessage wrapping

    Args:
        domain: Specialization domain (e.g., "software engineering")
        goal: Primary goal statement
        instructions: List of specific instructions (up to 3)
        min_words: Minimum response length
        max_words: Maximum response length
        user_background: User expertise level/context
        session_context: Current session information
        include_citations: Whether to include citation rules
        include_thinking: Whether to include thinking process section

    Returns:
        SystemMessage with composed prompt
    """
    # Pad instructions list to exactly 3 items
    padded_instructions = instructions + [""] * (3 - len(instructions))

    sections = []

    # Task section (always included)
    sections.append(TASK_SECTION.format(domain=domain, goal=goal))

    # Instructions section (always included)
    sections.append(
        INSTRUCTIONS_SECTION.format(
            instruction_1=padded_instructions[0],
            instruction_2=padded_instructions[1],
            instruction_3=padded_instructions[2],
        )
    )

    # Hard limits section (always included)
    sections.append(
        HARD_LIMITS_SECTION.format(min_words=min_words, max_words=max_words)
    )

    # Context section (always included)
    sections.append(
        CONTEXT_SECTION.format(
            user_background=user_background, session_context=session_context
        )
    )

    # Optional citation rules
    if include_citations:
        sections.append(CITATION_RULES_SECTION)

    # Optional thinking process
    if include_thinking:
        sections.append(THINKING_SECTION)

    # Join all sections with double newlines
    complete_prompt = "\n\n".join(sections)

    return SystemMessage(content=complete_prompt)


def create_code_generation_prompt(
    language: str,
    task_description: str,
    output_format: str = "Python",
    complexity: str = "n log n",
    coverage: int = 80,
    examples: str = "",
) -> str:
    """
    Create a code generation prompt with {{}} escaping for literal braces.

    Demonstrates:
    - Double-brace escaping for literal curly braces
    - Multi-variable template injection
    - Structured code generation prompts

    Args:
        language: Programming language
        task_description: What the code should accomplish
        output_format: Expected output format
        complexity: Maximum algorithmic complexity
        coverage: Required test coverage percentage
        examples: Pre-formatted example string

    Returns:
        Formatted prompt string
    """
    return TEMPLATE_WITH_EXAMPLES.format(
        output_format=output_format,
        task_description=task_description,
        language=language,
        examples=examples if examples else "No examples provided.",
        complexity=complexity,
        coverage=coverage,
    )


# ============================================================================
# Demonstration
# ============================================================================

async def main():
    """Demonstrate all prompt engineering patterns."""

    print("=" * 80)
    print("Prompt Engineering Patterns - LangChain Core")
    print("=" * 80)

    # Initialize fake model for testing
    model = FakeListChatModel(responses=["This is a simulated response."])

    # Pattern 1: XML-structured prompt composition
    print("\n1. XML-Structured Prompt Composition")
    print("-" * 80)
    system_prompt = compose_system_prompt(
        domain="software engineering",
        goal="help users write clean, maintainable code",
        instructions=[
            "Analyze requirements carefully",
            "Suggest appropriate design patterns",
            "Provide working code examples",
        ],
        min_words=100,
        max_words=300,
        user_background="Senior developer",
        session_context="Code review session",
        include_citations=True,
        include_thinking=True,
    )
    print(system_prompt.content[:500] + "...\n")

    # Pattern 2: Template variable injection with {{}} escaping
    print("\n2. Template Variable Injection (with {{}} escaping)")
    print("-" * 80)
    code_prompt = create_code_generation_prompt(
        language="Python",
        task_description="implements a binary search algorithm",
        complexity="n log n",
        coverage=90,
        examples="See standard library bisect module",
    )
    print(code_prompt[:400] + "...\n")

    # Pattern 3: Few-shot example injection
    print("\n3. Few-Shot Example Injection")
    print("-" * 80)
    examples = [
        {
            "input": "What is 2+2?",
            "output": "The answer is 4. This is basic arithmetic addition.",
        },
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris. [Source: Geography textbook]",
        },
    ]
    few_shot_messages = create_few_shot_examples(examples)
    print(f"Created {len(few_shot_messages)} few-shot messages:")
    for msg in few_shot_messages:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"  [{role}] {msg.content[:50]}...")
    print()

    # Pattern 4: Complete conversation with system + few-shot + user query
    print("\n4. Complete Conversation Pattern")
    print("-" * 80)
    messages = [system_prompt] + few_shot_messages + [
        HumanMessage(content="What is the time complexity of quicksort?")
    ]
    print(f"Total messages in conversation: {len(messages)}")
    print(f"  - System: 1")
    print(f"  - Few-shot examples: {len(few_shot_messages)}")
    print(f"  - User query: 1")
    print()

    # Pattern 5: Invoke model with composed prompt
    print("\n5. Model Invocation with Composed Prompt")
    print("-" * 80)
    response = await model.ainvoke(messages)
    print(f"Model response: {response.content}\n")

    print("=" * 80)
    print("All prompt patterns demonstrated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
