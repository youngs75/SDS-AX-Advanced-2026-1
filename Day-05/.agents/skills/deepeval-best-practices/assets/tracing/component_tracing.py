"""
DeepEval Tracing and Observability Example
=============================================
Demonstrates component-level tracing with @observe decorator, nested spans,
update_current_span(), and integration with EvaluationDataset.
"""

from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.dataset import EvaluationDataset, Golden


# =============================================================================
# 1. Basic @observe Decorator Usage
# =============================================================================
@observe(type="llm", name="Simple LLM Call")
def simple_llm_call(query: str) -> str:
    """
    Basic usage of @observe to create a span.
    Replace the body with your actual LLM call.
    """
    # Replace with your actual LLM call
    response = f"Here is a helpful response to: {query}"

    # Inject test case data into the current span
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
        )
    )
    return response


# =============================================================================
# 2. Nested Tracing (Agent -> Retriever -> LLM)
# =============================================================================

# Simulated vector database — replace with your actual retrieval logic
MOCK_KNOWLEDGE_BASE = {
    "refund": [
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Refunds are processed within 5-7 business days after approval.",
    ],
    "shipping": [
        "Standard shipping takes 3-5 business days.",
        "Express shipping is available for an additional $9.99.",
    ],
    "default": [
        "Please contact our support team for more information.",
    ],
}


@observe(
    type="retriever",
    name="Vector DB Retriever",
    metrics=[ContextualRelevancyMetric(threshold=0.6)],
)
def retriever(query: str) -> list[str]:
    """
    Retriever span that fetches relevant chunks.
    Replace with your actual vector database retrieval.
    """
    # Simulated retrieval — replace with real vector search
    chunks = MOCK_KNOWLEDGE_BASE.get("default", [])
    for key in MOCK_KNOWLEDGE_BASE:
        if key in query.lower():
            chunks = MOCK_KNOWLEDGE_BASE[key]
            break

    # Update span-level test case for retriever metrics
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output="placeholder",
            retrieval_context=chunks,
        )
    )

    # Also update the trace-level retrieval context
    update_current_trace(retrieval_context=chunks)

    return chunks


@observe(
    type="llm",
    name="GPT-4o Generator",
    metrics=[AnswerRelevancyMetric(threshold=0.7), FaithfulnessMetric(threshold=0.7)],
)
def generator(query: str, chunks: list[str]) -> str:
    """
    LLM generator span that produces the final response.
    Replace with your actual LLM generation call.
    """
    # Simulated LLM generation — replace with real LLM call
    context_str = " ".join(chunks)
    response = f"Based on our records: {context_str}"

    # Update span-level test case for generator metrics
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=chunks,
        )
    )

    # Update the trace-level test case
    update_current_trace(input=query, output=response)

    return response


@observe(type="tool", name="Format Response Tool")
def format_response(raw_response: str) -> str:
    """
    Tool span for post-processing the response.
    Demonstrates a tool-type span in the trace tree.
    """
    formatted = raw_response.strip()

    update_current_span(
        input=raw_response,
        output=formatted,
    )

    return formatted


@observe(type="agent", name="RAG Pipeline Agent")
def rag_pipeline(query: str) -> str:
    """
    Top-level agent span that orchestrates the full RAG pipeline.
    Nested @observe calls automatically form parent-child span relationships:

    RAG Pipeline Agent (agent)
      -> Vector DB Retriever (retriever)
      -> GPT-4o Generator (llm)
      -> Format Response Tool (tool)
    """
    # Step 1: Retrieve relevant context
    chunks = retriever(query)

    # Step 2: Generate response using context
    raw_response = generator(query, chunks)

    # Step 3: Post-process the response
    final_response = format_response(raw_response)

    return final_response


# =============================================================================
# 3. Run Traced Pipeline Directly
# =============================================================================
def run_traced_pipeline():
    """
    Run the traced RAG pipeline directly.
    Traces are automatically created and can be viewed on Confident AI.
    """
    result = rag_pipeline("What is the refund policy?")
    print(f"Response: {result}")

    result2 = rag_pipeline("How long does shipping take?")
    print(f"Response: {result2}")


# =============================================================================
# 4. Integration with EvaluationDataset and evals_iterator
# =============================================================================
def run_with_dataset():
    """
    Use evals_iterator() to run the traced pipeline across a dataset of goldens.
    Each golden is processed through the pipeline with full tracing.
    """
    # Create an evaluation dataset
    dataset = EvaluationDataset(
        goldens=[
            Golden(input="What is the refund policy?"),
            Golden(input="How long does shipping take?"),
            Golden(input="Can I get express shipping?"),
        ]
    )

    # Iterate through goldens — each call creates a full trace
    for golden in dataset.evals_iterator():
        rag_pipeline(golden.input)

    print("Dataset evaluation with tracing complete.")


# =============================================================================
# 5. observed_callback for CI/CD Integration
# =============================================================================

# Define a traced function with metrics attached via @observe
@observe(
    metrics=[AnswerRelevancyMetric(threshold=0.7)],
)
def my_llm_app(input: str) -> str:
    """
    A traced LLM app function suitable for use as an observed_callback
    in CI/CD test files with assert_test().

    Usage in test file (test_llm_app.py):
        import pytest
        from deepeval import assert_test
        from deepeval.dataset import Golden

        @pytest.mark.parametrize("golden", dataset.goldens)
        def test_llm_app(golden: Golden):
            assert_test(
                golden=golden,
                observed_callback=my_llm_app
                # No metrics here — they are defined in @observe above
            )

    Run with: deepeval test run test_llm_app.py
    """
    # Replace with your actual LLM application logic
    response = f"Response to: {input}"

    update_current_span(
        test_case=LLMTestCase(
            input=input,
            actual_output=response,
        )
    )
    return response


# =============================================================================
# 6. Accessing Goldens Inside Traced Components
# =============================================================================
@observe(type="llm", metrics=[AnswerRelevancyMetric(threshold=0.7)])
def generator_with_golden_access(query: str) -> str:
    """
    Access the current golden being evaluated inside a traced component.
    Useful when you need expected_output from the golden for comparison.
    """
    from deepeval.dataset import get_current_golden

    # Replace with your actual LLM call
    response = f"Response to: {query}"

    # Access the golden currently being evaluated (available during evals_iterator)
    golden = get_current_golden()
    expected = golden.expected_output if golden else None

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected,
        )
    )
    return response


if __name__ == "__main__":
    print("=== Direct Traced Pipeline ===")
    run_traced_pipeline()

    print("\n=== Dataset Evaluation with Tracing ===")
    run_with_dataset()
