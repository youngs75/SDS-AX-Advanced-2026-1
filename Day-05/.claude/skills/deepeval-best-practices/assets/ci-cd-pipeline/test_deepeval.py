"""
DeepEval CI/CD Pipeline Test Example
======================================
Pytest-compatible test file for running DeepEval evaluations in CI/CD.

Run with: deepeval test run test_deepeval.py
Or:       deepeval test run test_deepeval.py -n 4 -c -v
"""

import pytest
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn, ToolCall
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams


# =============================================================================
# 1. Simulated LLM Application (replace with your actual app)
# =============================================================================
def your_llm_app(query: str) -> tuple[str, list[str]]:
    """
    Your LLM application that returns (response, retrieval_contexts).
    Replace this with your actual RAG pipeline or agent.
    """
    responses = {
        "What is the refund policy?": (
            "We offer a 30-day full refund at no extra cost.",
            ["All customers are eligible for a 30 day full refund at no extra cost."],
        ),
        "How do I contact support?": (
            "You can reach our support team at support@example.com or call 1-800-HELP.",
            ["Support email: support@example.com", "Support phone: 1-800-HELP"],
        ),
    }
    return responses.get(
        query, ("I'm not sure about that.", ["No relevant context found."])
    )


# =============================================================================
# 2. Dataset Setup
# =============================================================================
# Option A: Create dataset programmatically
dataset = EvaluationDataset(
    goldens=[
        Golden(
            input="What is the refund policy?",
            expected_output="You are eligible for a 30 day full refund at no extra cost.",
        ),
        Golden(
            input="How do I contact support?",
            expected_output="Contact support at support@example.com or 1-800-HELP.",
        ),
    ]
)

# Option B: Load from Confident AI (uncomment to use)
# dataset = EvaluationDataset()
# dataset.pull(alias="My Evals Dataset")

# Option C: Load from local JSON (uncomment to use)
# dataset = EvaluationDataset()
# dataset.add_goldens_from_json_file(file_path="test_data.json")


# =============================================================================
# 3. End-to-End Single-Turn Tests
# =============================================================================
@pytest.mark.parametrize("golden", dataset.goldens, ids=lambda g: g.input[:50])
def test_llm_app_relevancy(golden: Golden):
    """Test that LLM responses are relevant to the input query."""
    response, retrieval_contexts = your_llm_app(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response,
        expected_output=golden.expected_output,
        retrieval_context=retrieval_contexts,
    )

    assert_test(
        test_case=test_case,
        metrics=[AnswerRelevancyMetric(threshold=0.7)],
    )


@pytest.mark.parametrize("golden", dataset.goldens, ids=lambda g: g.input[:50])
def test_llm_app_faithfulness(golden: Golden):
    """Test that LLM responses are faithful to retrieved context."""
    response, retrieval_contexts = your_llm_app(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response,
        retrieval_context=retrieval_contexts,
    )

    assert_test(
        test_case=test_case,
        metrics=[FaithfulnessMetric(threshold=0.7)],
    )


# =============================================================================
# 4. Custom GEval Metric Tests
# =============================================================================
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether facts in actual output contradict expected output",
        "Penalize omission of important details",
        "Vague language is acceptable",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)


@pytest.mark.parametrize("golden", dataset.goldens, ids=lambda g: g.input[:50])
def test_llm_app_correctness(golden: Golden):
    """Test factual correctness using GEval."""
    response, _ = your_llm_app(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=response,
        expected_output=golden.expected_output,
    )

    assert_test(test_case=test_case, metrics=[correctness_metric])


# =============================================================================
# 5. Multi-Turn Conversation Tests
# =============================================================================
def test_conversation_flow():
    """Test a multi-turn conversation for answer relevancy."""
    convo_test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="I want to return my shoes."),
            Turn(
                role="assistant",
                content="I can help you with that. What's your order number?",
            ),
            Turn(role="user", content="Order #12345"),
            Turn(
                role="assistant",
                content="I've initiated a return for order #12345. You'll receive a full refund within 5-7 business days.",
            ),
        ],
    )

    # Use conversation-compatible metrics
    from deepeval.metrics import ConversationCompletenessMetric

    assert_test(
        test_case=convo_test_case,
        metrics=[ConversationCompletenessMetric(threshold=0.5)],
    )


# =============================================================================
# 6. Hyperparameter Logging (for Confident AI tracking)
# =============================================================================
@deepeval.log_hyperparameters(model="gpt-4.1", prompt_template="default-v2")
def hyperparameters():
    """Log hyperparameters for this test run on Confident AI."""
    return {
        "model": "gpt-4.1",
        "temperature": 0.0,
        "chunk_size": 512,
        "top_k": 5,
        "system_prompt": "You are a helpful customer support agent.",
    }


# =============================================================================
# 7. Test Run Hooks
# =============================================================================
@deepeval.on_test_run_end
def after_test_run():
    """Called after all tests complete."""
    print("All DeepEval tests finished!")
