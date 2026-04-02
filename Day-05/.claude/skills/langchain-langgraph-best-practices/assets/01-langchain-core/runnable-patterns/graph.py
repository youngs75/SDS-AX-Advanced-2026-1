"""
LCEL Runnable Patterns Demonstration

This module demonstrates the Runnable protocol and LangChain Expression Language (LCEL)
composition patterns using only langchain_core.

Key Concepts:
1. Runnable interface: invoke/ainvoke, stream/astream, batch/abatch
2. RunnableConfig: configurable, tags, metadata, callbacks, max_concurrency
3. LCEL pipe operator (|): composing chains
4. RunnableLambda: wrapping custom functions
5. RunnablePassthrough: forwarding input unchanged
6. RunnableParallel: fan-out execution
7. Async streaming patterns

Decision Guide: LCEL vs LangGraph
---------------------------------
Use LCEL when:
- Simple linear chains with no cycles
- No persistent state required
- No human-in-the-loop interactions
- Stateless transformations

Use LangGraph when:
- Complex stateful workflows with persistence
- Cyclic graphs (loops, conditional branching)
- Human-in-the-loop approvals or interventions
- State management across multiple steps
- Time travel / replay capabilities

Reference: references/01-langchain-core/30-runnables-state-types.md
"""

import asyncio
from typing import Any

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Real model imports (commented out for demonstration)
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic

# ==============================================================================
# Setup: Fake Model for Demonstration
# ==============================================================================

# Using FakeListChatModel for demonstration without API calls
fake_responses = [
    "Hello! I'm a helpful assistant.",
    "The capital of France is Paris.",
    "LCEL is a powerful composition framework.",
]
model = FakeListChatModel(responses=fake_responses)

# For real usage, uncomment:
# model = ChatOpenAI(model="gpt-4", temperature=0)
# model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)


# ==============================================================================
# Section 1: Runnable Interface - 6 Core Methods
# ==============================================================================


async def demo_runnable_interface():
    """
    Demonstrate all 6 Runnable methods:
    - invoke: synchronous single invocation
    - ainvoke: asynchronous single invocation
    - stream: synchronous streaming
    - astream: asynchronous streaming
    - batch: synchronous batch processing
    - abatch: asynchronous batch processing
    """
    print("=" * 80)
    print("Section 1: Runnable Interface - 6 Core Methods")
    print("=" * 80)

    messages = [HumanMessage(content="Say hello")]

    # 1. invoke: synchronous single invocation
    print("\n1. invoke (sync):")
    result = model.invoke(messages)
    print(f"   {result.content}")

    # 2. ainvoke: asynchronous single invocation
    print("\n2. ainvoke (async):")
    result = await model.ainvoke(messages)
    print(f"   {result.content}")

    # 3. stream: synchronous streaming (not shown - requires sync context)
    print("\n3. stream (sync): [skipped in async demo]")

    # 4. astream: asynchronous streaming
    print("\n4. astream (async):")
    print("   ", end="")
    async for chunk in model.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    # 5. batch: synchronous batch processing (not shown - requires sync context)
    print("\n5. batch (sync): [skipped in async demo]")

    # 6. abatch: asynchronous batch processing
    print("\n6. abatch (async):")
    batch_messages = [
        [HumanMessage(content="What is the capital of France?")],
        [HumanMessage(content="Explain LCEL")],
    ]
    results = await model.abatch(batch_messages)
    for i, result in enumerate(results, 1):
        print(f"   Batch {i}: {result.content}")


# ==============================================================================
# Section 2: RunnableConfig - Configuration Options
# ==============================================================================


async def demo_runnable_config():
    """
    Demonstrate RunnableConfig with:
    - configurable: runtime configuration parameters
    - tags: categorization for tracing
    - metadata: additional context
    - callbacks: event handlers
    - max_concurrency: parallel execution limits
    """
    print("\n" + "=" * 80)
    print("Section 2: RunnableConfig - Configuration Options")
    print("=" * 80)

    messages = [HumanMessage(content="Tell me about configuration")]

    # Create config with various options
    config = RunnableConfig(
        tags=["demo", "configuration"],
        metadata={"user_id": "user123", "session_id": "sess456"},
        max_concurrency=2,
        # callbacks would go here in real usage
    )

    print("\nConfig with tags, metadata, max_concurrency:")
    print(f"   Tags: {config.get('tags')}")
    print(f"   Metadata: {config.get('metadata')}")
    print(f"   Max Concurrency: {config.get('max_concurrency')}")

    result = await model.ainvoke(messages, config=config)
    print(f"   Result: {result.content}")


# ==============================================================================
# Section 3: LCEL Pipe Operator (|) - Chain Composition
# ==============================================================================


async def demo_lcel_pipe():
    """
    Demonstrate LCEL pipe operator (|) for composing chains:
    model | parser
    """
    print("\n" + "=" * 80)
    print("Section 3: LCEL Pipe Operator (|) - Chain Composition")
    print("=" * 80)

    # Create a chain: model -> parser
    parser = StrOutputParser()
    chain = model | parser

    print("\nChain: model | parser")
    messages = [HumanMessage(content="What is LCEL?")]
    result = await chain.ainvoke(messages)
    print(f"   Parsed result (str): {result}")
    print(f"   Type: {type(result)}")


# ==============================================================================
# Section 4: RunnableLambda - Custom Functions
# ==============================================================================


async def demo_runnable_lambda():
    """
    Demonstrate RunnableLambda for wrapping custom functions
    into the Runnable interface.
    """
    print("\n" + "=" * 80)
    print("Section 4: RunnableLambda - Custom Functions")
    print("=" * 80)

    # Custom function
    def to_uppercase(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()

    # Async custom function
    async def add_prefix(text: str) -> str:
        """Add prefix to text."""
        await asyncio.sleep(0.1)  # Simulate async work
        return f"[PREFIXED] {text}"

    # Wrap functions as Runnables
    uppercase_runnable = RunnableLambda(to_uppercase)
    prefix_runnable = RunnableLambda(add_prefix)

    print("\n1. Sync RunnableLambda (uppercase):")
    result = await uppercase_runnable.ainvoke("hello world")
    print(f"   {result}")

    print("\n2. Async RunnableLambda (add prefix):")
    result = await prefix_runnable.ainvoke("important message")
    print(f"   {result}")

    print("\n3. Chain with RunnableLambda:")
    chain = model | StrOutputParser() | uppercase_runnable
    messages = [HumanMessage(content="Say something")]
    result = await chain.ainvoke(messages)
    print(f"   {result}")


# ==============================================================================
# Section 5: RunnablePassthrough - Forwarding Input
# ==============================================================================


async def demo_runnable_passthrough():
    """
    Demonstrate RunnablePassthrough for forwarding input unchanged,
    useful in parallel chains or when you need to preserve original input.
    """
    print("\n" + "=" * 80)
    print("Section 5: RunnablePassthrough - Forwarding Input")
    print("=" * 80)

    print("\n1. Simple passthrough:")
    passthrough = RunnablePassthrough()
    result = await passthrough.ainvoke({"key": "value", "number": 42})
    print(f"   Input: {{'key': 'value', 'number': 42}}")
    print(f"   Output: {result}")

    print("\n2. Passthrough in chain (preserve original):")
    # This pattern is useful when you need both original input and processed result
    chain = RunnableParallel(
        original=RunnablePassthrough(),
        processed=RunnableLambda(lambda x: x.upper()),
    )
    result = await chain.ainvoke("hello")
    print(f"   Input: 'hello'")
    print(f"   Output: {result}")


# ==============================================================================
# Section 6: RunnableParallel - Fan-Out Execution
# ==============================================================================


async def demo_runnable_parallel():
    """
    Demonstrate RunnableParallel for fan-out execution:
    - Execute multiple runnables in parallel
    - Combine results into a dict
    - Useful for processing same input in multiple ways
    """
    print("\n" + "=" * 80)
    print("Section 6: RunnableParallel - Fan-Out Execution")
    print("=" * 80)

    # Define parallel transformations
    def to_uppercase(text: str) -> str:
        return text.upper()

    def to_lowercase(text: str) -> str:
        return text.lower()

    def count_chars(text: str) -> int:
        return len(text)

    # Create parallel runnable
    parallel = RunnableParallel(
        upper=RunnableLambda(to_uppercase),
        lower=RunnableLambda(to_lowercase),
        length=RunnableLambda(count_chars),
    )

    print("\nParallel execution with 3 transformations:")
    input_text = "Hello World"
    result = await parallel.ainvoke(input_text)
    print(f"   Input: '{input_text}'")
    print(f"   Results:")
    print(f"      upper: {result['upper']}")
    print(f"      lower: {result['lower']}")
    print(f"      length: {result['length']}")


# ==============================================================================
# Section 7: Async Streaming Patterns
# ==============================================================================


async def demo_async_streaming():
    """
    Demonstrate async streaming with LCEL chains.
    """
    print("\n" + "=" * 80)
    print("Section 7: Async Streaming Patterns")
    print("=" * 80)

    print("\n1. Stream from model:")
    messages = [HumanMessage(content="Stream this response")]
    print("   Streaming: ", end="", flush=True)
    async for chunk in model.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    print("\n2. Stream through chain (model | parser):")
    chain = model | StrOutputParser()
    messages = [HumanMessage(content="Stream through parser")]
    print("   Streaming: ", end="", flush=True)
    async for chunk in chain.astream(messages):
        print(chunk, end="", flush=True)
    print()


# ==============================================================================
# Section 8: Complex LCEL Chain Example
# ==============================================================================


async def demo_complex_chain():
    """
    Demonstrate a complex LCEL chain combining multiple patterns:
    - Parallel processing
    - Custom transformations
    - Chain composition
    """
    print("\n" + "=" * 80)
    print("Section 8: Complex LCEL Chain Example")
    print("=" * 80)

    # Define processing steps
    def extract_keywords(text: str) -> list[str]:
        """Simple keyword extraction (split by spaces)."""
        return text.split()

    def count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    # Build complex chain
    chain = (
        model
        | StrOutputParser()
        | RunnableParallel(
            original=RunnablePassthrough(),
            keywords=RunnableLambda(extract_keywords),
            word_count=RunnableLambda(count_words),
        )
    )

    print("\nComplex chain: model | parser | parallel(original, keywords, word_count)")
    messages = [SystemMessage(content="You are concise"), HumanMessage(content="Explain LCEL briefly")]
    result = await chain.ainvoke(messages)
    print(f"   Original: {result['original']}")
    print(f"   Keywords: {result['keywords']}")
    print(f"   Word count: {result['word_count']}")


# ==============================================================================
# Main Demo Runner
# ==============================================================================


async def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("LCEL RUNNABLE PATTERNS DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo uses FakeListChatModel for illustration.")
    print("In production, use real models like ChatOpenAI or ChatAnthropic.\n")

    await demo_runnable_interface()
    await demo_runnable_config()
    await demo_lcel_pipe()
    await demo_runnable_lambda()
    await demo_runnable_passthrough()
    await demo_runnable_parallel()
    await demo_async_streaming()
    await demo_complex_chain()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Runnable interface provides 6 methods for flexible execution")
    print("2. RunnableConfig enables runtime configuration and tracing")
    print("3. LCEL pipe (|) simplifies chain composition")
    print("4. RunnableLambda wraps custom functions")
    print("5. RunnablePassthrough preserves original input")
    print("6. RunnableParallel enables fan-out execution")
    print("7. Async streaming supports real-time output")
    print("\nDecision Guide:")
    print("- Use LCEL for simple stateless chains")
    print("- Use LangGraph for complex stateful workflows")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
