# Custom Metrics — Reference Guide

Combined from: guides-building-custom-metrics.md + guides-answer-correctness-metric.md
Sources: https://deepeval.com/guides/guides-building-custom-metrics | https://deepeval.com/guides/guides-answer-correctness-metric

## Read This When
- Need to build a custom metric by inheriting BaseMetric (non-LLM, LLM-based, or composite)
- Want to create a tailored Answer Correctness metric using GEval with custom evaluation steps and thresholds
- Looking for patterns to combine multiple DeepEval metrics into a single composite evaluation

## Skip This When
- Need API reference for BaseMetric, GEval, or Scorer parameters -- see `references/03-eval-metrics/80-custom-metrics.md`
- Want to use a built-in metric (Faithfulness, AnswerRelevancy, etc.) without customization -- see `references/03-eval-metrics/`
- Looking to use a non-OpenAI LLM as the evaluation judge -- see `references/09-guides/40-custom-llms-and-embeddings.md`

---

# Part 1: Building Custom LLM Metrics

## Overview

DeepEval enables developers to create custom LLM evaluation metrics that integrate seamlessly into the framework's ecosystem, including CI/CD pipelines, metric caching, multi-processing, and automatic result reporting to Confident AI.

## Key Reasons to Build Custom Metrics

- Achieve greater control over evaluation criteria beyond standard implementations
- Avoid LLM-based scoring for specific use cases
- Combine multiple DeepEval metrics into unified evaluations

## Rules for Creating Custom Metrics

### 1. Inherit the BaseMetric Class

```python
from deepeval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    ...
```

The `BaseMetric` class ensures DeepEval recognizes your custom metric during evaluation.

### 2. Implement the `__init__()` Method

Configure these properties in the initialization method:

| Property | Type | Purpose | Required |
|----------|------|---------|----------|
| `threshold` | float | Determines pass/fail for test cases | Required |
| `evaluation_model` | str | Name of the evaluation model | Optional |
| `include_reason` | bool | Include reason alongside score | Optional |
| `strict_mode` | bool | Pass only with perfect scores | Optional |
| `async_mode` | bool | Enable asynchronous execution | Optional |

```python
from deepeval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: str = None,
        include_reason: bool = True,
        strict_mode: bool = True,
        async_mode: bool = True
    ):
        self.threshold = threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
```

### 3. Implement `measure()` and `a_measure()` Methods

Both methods must:
- Accept an `LLMTestCase` as an argument
- Set `self.score`
- Set `self.success`
- Optionally set `self.reason` (for LLM-based evaluation)
- Optionally capture exceptions in `self.error`

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = generate_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = await async_generate_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = await async_generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise
```

**Note:** If asynchronous LLM inference is unavailable, reuse the `measure()` method in `a_measure()`:

```python
async def a_measure(self, test_case: LLMTestCase) -> float:
    return self.measure(test_case)
```

### 4. Implement the `is_successful()` Method

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success
```

### 5. Name Your Custom Metric

```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    @property
    def __name__(self):
        return "My Custom Metric"
```

## Building a Custom Non-LLM Evaluation

This example uses ROUGE scoring instead of LLM-based evaluation:

```python
from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"
```

**Usage Example:**

```python
test_case = LLMTestCase(
    input="...",
    actual_output="...",
    expected_output="..."
)
metric = RougeMetric()
metric.measure(test_case)
print(metric.is_successful())
```

**Installation Note:** Run `pip install rouge-score` if not already installed.

## Building a Custom Composite Metric

Combine multiple DeepEval metrics into a single evaluation:

```python
from deepeval.metrics import (
    BaseMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from typing import Optional

class FaithfulRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: Optional[str] = "gpt-4-turbo",
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, faithfulness_metric = self.initialize_metrics()
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            self.set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, faithfulness_metric = self.initialize_metrics()
            await relevancy_metric.a_measure(test_case)
            await faithfulness_metric.a_measure(test_case)
            self.set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Composite Relevancy Faithfulness Metric"

    def initialize_metrics(self):
        relevancy_metric = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        faithfulness_metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        return relevancy_metric, faithfulness_metric

    def set_score_reason_success(
        self,
        relevancy_metric: BaseMetric,
        faithfulness_metric: BaseMetric
    ):
        relevancy_score = relevancy_metric.score
        relevancy_reason = relevancy_metric.reason
        faithfulness_score = faithfulness_metric.score
        faithfulness_reason = faithfulness_metric.reason

        composite_score = min(relevancy_score, faithfulness_score)
        self.score = (
            0 if self.strict_mode and composite_score < self.threshold
            else composite_score
        )

        if self.include_reason:
            self.reason = relevancy_reason + "\n" + faithfulness_reason

        self.success = self.score >= self.threshold
```

**Usage Example:**

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

def test_llm():
    metric = FaithfulRelevancyMetric()
    test_case = LLMTestCase(...)
    assert_test(test_case, [metric])
```

Run with: `deepeval test run test_llm.py`

## Tips & Notes

- The `async_mode` parameter enables concurrent metric execution when using `assert_test(test_case, [metric1, metric2], run_async=True)`
- DeepEval's built-in `Scorer` module provides traditional NLP scoring methods beyond those documented
- Error handling with try-except blocks is recommended for production implementations

## Related Resources

- [LLM Evaluation Metrics Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [Using Custom LLMs for Evaluation](/guides/guides-using-custom-llms)
- [Using Custom Embedding Models](/guides/guides-using-custom-embedding-models)
- [Optimizing Hyperparameters](/guides/guides-optimizing-hyperparameters)

---

# Part 2: Answer Correctness Metric

## Overview

The Answer Correctness metric represents one of the fundamental evaluation approaches for LLM applications, scaled from 0 to 1 where 1 indicates accurate responses and 0 indicates incorrect ones.

**Key insight:** While general-purpose correctness metrics exist, "custom Correctness metrics tailored to specific LLM applications prove most valuable" to users, implementable through G-Eval.

## Critical Considerations

Assessing correctness requires attention to:
- Determining ground truth through appropriate evaluation parameters
- Establishing clear evaluation steps/criteria for output assessment
- Setting suitable thresholds for correctness scoring

## Creating Your Correctness Metric

### Step 1: Instantiate GEval Object

```python
from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    ...
)
```

**Recommendation:** "G-Eval is most effective when employing a model from the GPT-4 model family" for evaluation.

### Step 2: Select Evaluation Parameters

Available `LLMTestCaseParams` options:
- `INPUT`
- `ACTUAL_OUTPUT` (required in all metrics)
- `EXPECTED_OUTPUT`
- `CONTEXT`
- `RETRIEVAL_CONTEXT`

**Basic parameter selection:**

```python
from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    ...
)
```

**Alternative approach without expected output:**

```python
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    ...
)
```

### Step 3: Define Evaluation Criteria

**Simple example:**

```python
from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],
)
```

**Detailed example:**

```python
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        'Compare the actual output directly with the expected output to verify factual accuracy.',
        'Check if all elements mentioned in the expected output are present and correctly represented in the actual output.',
        'Assess if there are any discrepancies in details, values, or information between the actual and expected outputs.'
    ],
)
```

**Opinion-flexible approach:**

```python
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also lightly penalize omission of detail, and focus on the main idea",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
)
```

**Note:** Iteratively adjust evaluation steps until scores align with expectations. "G-Eval metrics remain relatively stable across multiple evaluations, despite the variability of LLM responses."

## Iterating Evaluation Steps

### Establishing Baseline Benchmarks

Create test cases representing expected performance levels:

```python
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Perfect correctness (score: 1)
first_test_case = LLMTestCase(
    input="Summarize the benefits of daily exercise.",
    actual_output="Daily exercise improves cardiovascular health, boosts mood, and enhances overall fitness.",
    expected_output="Daily exercise improves cardiovascular health, boosts mood, and enhances overall fitness."
)

# Partial correctness (score: 0.5)
second_test_case = LLMTestCase(
    input="Explain the process of photosynthesis.",
    actual_output="Photosynthesis is how plants make their food using sunlight.",
    expected_output="Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. It involves the green pigment chlorophyll and generates oxygen as a byproduct."
)

# No correctness (score: 0)
third_test_case = LLMTestCase(
    input="Describe the effects of global warming.",
    actual_output="Global warming leads to colder winters.",
    expected_output="Global warming causes more extreme weather, including hotter summers, rising sea levels, and increased frequency of extreme weather events."
)

test_cases = [first_test_case, second_test_case, third_test_case]
dataset = EvaluationDataset(test_cases=test_cases)
```

## Finding the Right Threshold

### Step 1: Perform Correctness Evaluation

```python
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4.1",
    evaluation_params=[
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
        "Lightly penalize omissions of detail, focusing on the main idea",
        "Vague language or contradicting opinions are permissible"
    ],
)

deepeval.login("your_api_key_here")
dataset = EvaluationDataset()
dataset.pull(alias="dataset_for_correctness")
evaluation_output = dataset.evaluate([correctness_metric])
```

### Step 2: Calculate Threshold

```python
# Extract scores from the evaluation output
scores = [output.metrics[0].score for output in evaluation_output]

def calculate_threshold(scores, percentile):
    # Sort scores in ascending order
    sorted_scores = sorted(scores)
    # Calculate index for the desired percentile
    index = int(len(sorted_scores) * (1 - percentile / 100))
    # Return the score at that index
    return sorted_scores[index]

# Set the desired percentile threshold
percentile = 75  # Targeting the top 25%
threshold = calculate_threshold(scores, percentile)
```

## Related Resources

- [Red-Teaming your LLM](/guides/guides-red-teaming)
- [LLM evaluation metrics blog](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM-as-a-judge](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
- [LLM testing strategies](https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies)
- [LLM chatbot evaluation](https://www.confident-ai.com/blog/llm-chatbot-evaluation-explained-top-chatbot-evaluation-metrics-and-testing-techniques)

## Document Metadata
**Last Updated:** February 16, 2026
**Author:** Jeffrey Ip
