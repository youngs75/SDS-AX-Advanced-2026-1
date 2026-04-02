"""
DeepEval Custom Metrics Example
=================================
Demonstrates three approaches to custom metrics:
1. GEval (LLM-as-a-judge with chain-of-thought)
2. BaseMetric subclass (non-LLM / traditional metrics)
3. Composite metric (combining multiple metrics)
"""

from deepeval import evaluate
from deepeval.metrics import GEval, BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


# =============================================================================
# 1. GEval: Custom LLM-as-a-Judge Metrics
# =============================================================================

# --- Correctness Metric ---
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
        "Heavily penalize omission of detail",
        "Vague language or contradicting OPINIONS are OK",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.7,
)

# --- Clarity Metric ---
clarity_metric = GEval(
    name="Clarity",
    criteria="Evaluate whether the response is clear and easy to understand.",
    evaluation_steps=[
        "Evaluate whether the response uses clear and direct language.",
        "Check if the explanation avoids jargon or explains it when used.",
        "Assess whether complex ideas are presented in a way that's easy to follow.",
        "Identify any vague or confusing parts that reduce understanding.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6,
)

# --- Professionalism Metric ---
professionalism_metric = GEval(
    name="Professionalism",
    criteria="Determine whether the output maintains a professional tone.",
    evaluation_steps=[
        "Determine whether the actual output maintains a professional tone throughout.",
        "Evaluate if the language reflects expertise and domain-appropriate formality.",
        "Ensure the output stays contextually appropriate and avoids casual expressions.",
        "Check if the output is clear, respectful, and avoids slang.",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# --- GEval with Rubric (structured scoring) ---
correctness_with_rubric = GEval(
    name="Correctness (Rubric)",
    criteria="Factual correctness based on expected output.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3, 6), expected_outcome="Mostly correct with some errors."),
        Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10, 10), expected_outcome="100% correct."),
    ],
)

# --- Domain-Specific: Medical Faithfulness ---
medical_faithfulness = GEval(
    name="Medical Faithfulness",
    criteria="Evaluate medical accuracy of the output against clinical guidelines.",
    evaluation_steps=[
        "Extract medical claims or diagnoses from the actual output.",
        "Verify each medical claim against the retrieved clinical guidelines.",
        "Identify contradictions or unsupported medical claims.",
        "Heavily penalize hallucinations that could lead to incorrect medical advice.",
        "Provide reasons emphasizing clinical accuracy and patient safety.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.9,
)


# =============================================================================
# 2. BaseMetric Subclass: Non-LLM Evaluation (ROUGE Score)
# =============================================================================
class RougeMetric(BaseMetric):
    """Custom metric using ROUGE score for text similarity."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        from deepeval.scorer import Scorer

        scorer = Scorer()
        self.score = scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1",
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"


# =============================================================================
# 3. Composite Metric: Combining Multiple Metrics
# =============================================================================
class FaithfulRelevancyMetric(BaseMetric):
    """Composite metric combining Faithfulness and Answer Relevancy."""

    def __init__(
        self,
        threshold: float = 0.5,
        model: str = "gpt-4.1",
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.model = model
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            relevancy = AnswerRelevancyMetric(
                threshold=self.threshold, model=self.model
            )
            faithfulness = FaithfulnessMetric(
                threshold=self.threshold, model=self.model
            )

            relevancy.measure(test_case)
            faithfulness.measure(test_case)

            # Take the minimum of both scores
            self.score = min(relevancy.score, faithfulness.score)
            if self.strict_mode and self.score < self.threshold:
                self.score = 0

            self.reason = (
                f"Relevancy: {relevancy.reason}\nFaithfulness: {faithfulness.reason}"
            )
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Faithful Relevancy Metric"


# =============================================================================
# 4. Usage Examples
# =============================================================================
def run_geval_example():
    """Run GEval custom metrics."""
    test_case = LLMTestCase(
        input="Who ran up the tree?",
        actual_output="It depends on interpretation.",
        expected_output="The cat ran up the tree.",
    )

    evaluate(
        test_cases=[test_case],
        metrics=[correctness_metric, clarity_metric, professionalism_metric],
    )


def run_rouge_example():
    """Run custom ROUGE metric."""
    test_case = LLMTestCase(
        input="Summarize the document.",
        actual_output="The document discusses climate change impacts.",
        expected_output="The document covers the effects of climate change.",
    )

    metric = RougeMetric(threshold=0.3)
    metric.measure(test_case)
    print(f"ROUGE Score: {metric.score}")
    print(f"Passed: {metric.is_successful()}")


def run_composite_example():
    """Run composite metric."""
    test_case = LLMTestCase(
        input="What is the refund policy?",
        actual_output="We offer a 30-day full refund.",
        retrieval_context=["All customers are eligible for a 30 day full refund."],
    )

    metric = FaithfulRelevancyMetric(threshold=0.5)
    metric.measure(test_case)
    print(f"Composite Score: {metric.score}")
    print(f"Reason: {metric.reason}")


if __name__ == "__main__":
    print("=== GEval Custom Metrics ===")
    run_geval_example()

    print("\n=== ROUGE Custom Metric ===")
    run_rouge_example()

    print("\n=== Composite Metric ===")
    run_composite_example()
