"""
DeepEval RAG Evaluation Example
================================
Demonstrates end-to-end and component-level RAG pipeline evaluation
using DeepEval's five core RAG metrics.
"""

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden


# =============================================================================
# 1. Define your RAG pipeline (replace with your actual implementation)
# =============================================================================
def rag_pipeline(query: str) -> tuple[str, list[str]]:
    """
    Your RAG pipeline that returns (actual_output, retrieval_contexts).
    Replace this with your actual retriever + generator logic.
    """
    # Simulated retrieval
    retrieval_contexts = [
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Refunds are processed within 5-7 business days.",
    ]
    # Simulated generation
    actual_output = "We offer a 30-day full refund at no extra cost. Refunds are processed within 5-7 business days."
    return actual_output, retrieval_contexts


# =============================================================================
# 2. End-to-End RAG Evaluation
# =============================================================================
def run_end_to_end_evaluation():
    """Evaluate the full RAG pipeline with all five metrics."""

    # Define test inputs with expected outputs (for reference-based metrics)
    test_data = [
        {
            "input": "What if these shoes don't fit?",
            "expected_output": "You are eligible for a 30 day full refund at no extra cost.",
        },
        {
            "input": "How long do refunds take?",
            "expected_output": "Refunds are processed within 5-7 business days.",
        },
    ]

    # Generate test cases by running each input through the RAG pipeline
    test_cases = []
    for data in test_data:
        actual_output, contexts = rag_pipeline(data["input"])
        test_case = LLMTestCase(
            input=data["input"],
            actual_output=actual_output,
            expected_output=data["expected_output"],
            retrieval_context=contexts,
        )
        test_cases.append(test_case)

    # Define metrics
    # Generator metrics (evaluate output quality)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
    faithfulness = FaithfulnessMetric(threshold=0.7)

    # Retriever metrics (evaluate retrieval quality)
    contextual_relevancy = ContextualRelevancyMetric(threshold=0.7)
    contextual_precision = ContextualPrecisionMetric(threshold=0.7)
    contextual_recall = ContextualRecallMetric(threshold=0.7)

    # Run evaluation
    results = evaluate(
        test_cases=test_cases,
        metrics=[
            answer_relevancy,
            faithfulness,
            contextual_relevancy,
            contextual_precision,
            contextual_recall,
        ],
    )

    return results


# =============================================================================
# 3. Component-Level RAG Evaluation (with Tracing)
# =============================================================================
def run_component_level_evaluation():
    """Evaluate individual RAG components using tracing."""
    from deepeval.tracing import observe, update_current_span

    contextual_relevancy = ContextualRelevancyMetric(threshold=0.6)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.6)

    @observe()
    def rag_pipeline_traced(query: str):
        @observe(metrics=[contextual_relevancy])
        def retriever(query: str):
            contexts = [
                "All customers are eligible for a 30 day full refund.",
                "Refunds are processed within 5-7 business days.",
            ]
            update_current_span(
                test_case=LLMTestCase(
                    input=query,
                    actual_output="placeholder",
                    retrieval_context=contexts,
                )
            )
            return contexts

        @observe(metrics=[answer_relevancy])
        def generator(query: str, chunks: list[str]):
            output = "We offer a 30-day full refund. Refunds take 5-7 business days."
            update_current_span(
                test_case=LLMTestCase(input=query, actual_output=output)
            )
            return output

        chunks = retriever(query)
        return generator(query, chunks)

    # Run with dataset
    dataset = EvaluationDataset(
        goldens=[
            Golden(input="What if these shoes don't fit?"),
            Golden(input="How long do refunds take?"),
        ]
    )

    for golden in dataset.evals_iterator():
        rag_pipeline_traced(golden.input)


# =============================================================================
# 4. Standalone Metric Debugging
# =============================================================================
def debug_single_metric():
    """Debug a single metric on a single test case."""
    metric = AnswerRelevancyMetric(threshold=0.7, verbose_mode=True)

    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
    )

    metric.measure(test_case)
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Passed: {metric.is_successful()}")


if __name__ == "__main__":
    print("=== End-to-End RAG Evaluation ===")
    run_end_to_end_evaluation()

    print("\n=== Standalone Metric Debug ===")
    debug_single_metric()
