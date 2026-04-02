# Custom Metrics

## Read This When
- Building a custom LLM-as-a-judge metric with arbitrary criteria using GEval (single-turn) or ConversationalGEval (multi-turn)
- Need deterministic decision-tree evaluation with guaranteed score mappings using DAGMetric or ConversationalDAGMetric
- Subclassing BaseMetric for fully custom scoring logic (ROUGE, BLEU, composite metrics) or integrating a custom LLM judge (DeepEvalBaseLLM)

## Skip This When
- Need a predefined RAG metric (AnswerRelevancy, Faithfulness, ContextualPrecision) → `references/03-eval-metrics/20-rag-metrics.md`
- Need predefined agent metrics (ToolCorrectness, TaskCompletion) → `references/03-eval-metrics/30-agent-metrics.md`
- Need deterministic non-LLM validation (ExactMatch, JsonCorrectness, PatternMatch) → `references/03-eval-metrics/70-utility-metrics.md`

---

DeepEval provides three pathways for building custom evaluation metrics: **GEval** (LLM-as-a-judge with arbitrary criteria), **DAGMetric** (deterministic decision trees), and **BaseMetric subclassing** (fully custom logic). All integrate seamlessly with DeepEval's ecosystem (CI/CD, caching, Confident AI).

---

## GEval (LLM-as-a-Judge with Custom Criteria)

G-Eval originates from the paper "NLG Evaluation using GPT-4 with Better Human Alignment." It uses chain-of-thought (CoT) to evaluate LLM outputs based on ANY custom criteria with human-level accuracy.

**Classification:** LLM-as-a-judge, Single-turn, Flexible (reference-based or referenceless)

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- Additional fields (`expected_output`, `context`, `retrieval_context`) if criteria depends on them

**Mandatory GEval parameters:**
- `name` — custom metric name (display only, does not affect evaluation)
- `criteria` — description of what to evaluate (OR `evaluation_steps` — provide one, not both)
- `evaluation_params` — list of `LLMTestCaseParams` that the criteria actually uses

**Algorithm:**
1. Generate `evaluation_steps` using CoT from `criteria` (skipped if `evaluation_steps` provided)
2. Evaluate the test case against those steps using the judge LLM
3. LLM outputs a score 1–10, which is normalized to 0–1 via output token probabilities

**GEval Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Metric display name |
| `criteria` | str | required* | Evaluation description (*or `evaluation_steps`) |
| `evaluation_params` | List[LLMTestCaseParams] | required | Test case fields to pass to judge |
| `evaluation_steps` | List[str] | None | Explicit evaluation steps (overrides criteria-generated steps) |
| `rubric` | List[Rubric] | None | Score range constraints with expected outcomes |
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM (GPT-4 family recommended) |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | GEvalTemplate | Default | Custom prompt template |

**Critical warning:** Only include `LLMTestCaseParams` in `evaluation_params` that are actually referenced in your `criteria` or `evaluation_steps`. Including unused params degrades accuracy.

### Basic Usage

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="It depends, some might consider the cat, while others might argue the dog.",
    expected_output="The cat."
)

correctness_metric.measure(test_case)
print(correctness_metric.score, correctness_metric.reason)
```

### Common GEval Use Cases

**Factual Correctness:**

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "Heavily penalize omission of detail",
        "Vague language or contradicting OPINIONS are OK"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)
```

**Clarity / Coherence:**

```python
clarity = GEval(
    name="Clarity",
    evaluation_steps=[
        "Evaluate whether the response uses clear and direct language.",
        "Check if the explanation avoids jargon or explains it when used.",
        "Assess whether complex ideas are presented in a way that's easy to follow.",
        "Identify any vague or confusing parts that reduce understanding."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)
```

**Professionalism / Tone:**

```python
professionalism = GEval(
    name="Professionalism",
    evaluation_steps=[
        "Determine whether the actual output maintains a professional tone throughout.",
        "Evaluate if the language reflects expertise and domain-appropriate formality.",
        "Ensure the output stays contextually appropriate and avoids casual expressions.",
        "Check if the output is clear, respectful, and avoids slang."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)
```

**Custom RAG — Medical Faithfulness:**

```python
medical_faithfulness = GEval(
    name="Medical Faithfulness",
    evaluation_steps=[
        "Extract medical claims or diagnoses from the actual output.",
        "Verify each medical claim against the retrieved contextual information.",
        "Identify any contradictions or unsupported medical claims.",
        "Heavily penalize hallucinations that could result in incorrect medical advice.",
        "Provide reasons emphasizing clinical accuracy and patient safety."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
)
```

### Rubric Configuration

Rubrics constrain scores to specific ranges with defined meanings (0–10 scale):

```python
from deepeval.metrics.g_eval import Rubric

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3, 6), expected_outcome="Mostly correct."),
        Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10, 10), expected_outcome="100% correct."),
    ]
)
```

Rules: score ranges must be 0–10 inclusive, no overlapping ranges.

### Custom Template

```python
from deepeval.metrics.g_eval import GEvalTemplate
import textwrap

class CustomGEvalTemplate(GEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return textwrap.dedent(f"""
            You are given evaluation criteria for assessing {parameters}. Based on the criteria,
            produce 3-4 clear steps that explain how to evaluate the quality of {parameters}.
            Criteria:
            {criteria}
            Return JSON only, in this format:
            {{
                "steps": [
                    "Step 1",
                    "Step 2",
                    "Step 3"
                ]
            }}
            JSON:
            """)

metric = GEval(evaluation_template=CustomGEvalTemplate, ...)
```

### Upload to Confident AI

```python
metric = GEval(...)
metric.upload()  # Upload for reuse as a platform metric
```

---

## ConversationalGEval

Multi-turn version of GEval for evaluating entire conversations. Uses `ConversationalTestCase` instead of `LLMTestCase`. See [50-conversation-turn-metrics.md](./50-conversation-turn-metrics.md) for full documentation.

```python
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import TurnParams

metric = ConversationalGEval(
    name="Professionalism",
    criteria="Determine whether the assistant has acted professionally based on the content.",
    evaluation_params=[TurnParams.CONTENT]
)
```

---

## DAGMetric (Deterministic Decision Tree)

The DAG (Deep Acyclic Graph) metric provides the most control over evaluation — you define a decision tree that the LLM traverses to arrive at a deterministic score.

**Key advantage over GEval:** Guaranteed score mapping — specific evaluation outcomes produce specific scores (e.g., missing a heading always scores 0, not sometimes 0.1 or 0.2).

**Mandatory parameters:**
- `name` — metric display name
- `dag` — `DeepAcyclicGraph` instance

**Optional parameters:** Same as GEval (`threshold`, `model`, `include_reason`, `strict_mode`, `async_mode`, `verbose_mode`)

### Node Types

**TaskNode** — Extracts/processes data from the test case or parent node output:

```python
from deepeval.metrics.dag import TaskNode
from deepeval.test_case import LLMTestCaseParams

extract_headings_node = TaskNode(
    instructions="Extract all headings in `actual_output`",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Summary headings",  # referenced by child nodes
    children=[correct_headings_node, correct_order_node],
    label="Extract Headings"  # optional display name
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `instructions` | str | Yes | How to process the test case fields or parent output |
| `output_label` | str | Yes | Name for the output; child nodes reference this |
| `children` | List[BaseNode] | Yes | Child nodes (must NOT contain VerdictNode directly) |
| `evaluation_params` | List[LLMTestCaseParams] | No | Test case fields to pass |
| `label` | str | No | Display name for verbose logs |

**BinaryJudgementNode** — Yes/No decision:

```python
from deepeval.metrics.dag import BinaryJudgementNode, VerdictNode

correct_headings_node = BinaryJudgementNode(
    criteria="Does the summary contain all three headings: 'intro', 'body', and 'conclusion'?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=correct_order_node),
    ],
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `criteria` | str | Yes | Yes/no question; do NOT include "true/false" instruction |
| `children` | List[VerdictNode] | Yes | Exactly 2: one `verdict=True`, one `verdict=False` |
| `evaluation_params` | List[LLMTestCaseParams] | No | Additional test case fields for context |
| `label` | str | No | Display name for verbose logs |

**NonBinaryJudgementNode** — Multiple-choice decision:

```python
from deepeval.metrics.dag import NonBinaryJudgementNode, VerdictNode

correct_order_node = NonBinaryJudgementNode(
    criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
    children=[
        VerdictNode(verdict="Yes", score=10),
        VerdictNode(verdict="Two are out of order", score=4),
        VerdictNode(verdict="All out of order", score=2),
    ],
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `criteria` | str | Yes | Open-ended question |
| `children` | List[VerdictNode] | Yes | Multiple verdicts; values must match possible LLM outputs |
| `evaluation_params` | List[LLMTestCaseParams] | No | Additional test case fields |
| `label` | str | No | Display name for verbose logs |

**VerdictNode** — Leaf node returning the final score:

```python
from deepeval.metrics.dag import VerdictNode
from deepeval.metrics import GEval

# With a fixed score
VerdictNode(verdict=True, score=10)
VerdictNode(verdict="Yes", score=8)

# With a child node (propagates to next decision)
VerdictNode(verdict=True, child=next_node)

# With a GEval metric (delegates scoring to GEval)
VerdictNode(
    verdict=False,
    child=GEval(name="Detailed Check", evaluation_steps=[...])
)
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `verdict` | str or bool | Yes | String for NonBinary parent, bool for Binary parent |
| `score` | int (0–10) | Either `score` or `child` | Final score for this verdict |
| `child` | BaseNode or metric | Either `score` or `child` | Propagate to another node or delegate to GEval |

### Complete DAG Example

```python
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import DAGMetric
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)

test_case = LLMTestCase(
    input="Summarize the meeting transcript.",
    actual_output="""Intro:
Alice outlined the agenda: product updates, blockers, and marketing alignment.
Body:
Bob reported performance issues being optimized, with fixes expected by Friday.
Conclusion:
The team aligned on next steps: engineering finalizing fixes by Friday."""
)

# Build from leaves up
correct_order_node = NonBinaryJudgementNode(
    criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
    children=[
        VerdictNode(verdict="Yes", score=10),
        VerdictNode(verdict="Two are out of order", score=4),
        VerdictNode(verdict="All out of order", score=2),
    ],
)

correct_headings_node = BinaryJudgementNode(
    criteria="Does the summary contain all three headings: 'intro', 'body', and 'conclusion'?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=correct_order_node),
    ],
)

extract_headings_node = TaskNode(
    instructions="Extract all headings in `actual_output`",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Summary headings",
    children=[correct_headings_node, correct_order_node],
)

dag = DeepAcyclicGraph(root_nodes=[extract_headings_node])
metric = DAGMetric(name="Format Correctness", dag=dag)

metric.measure(test_case)
print(metric.score)  # 10, 4, 2, or 0 — normalized to 0-1
```

### GEval vs DAGMetric

| Aspect | GEval | DAGMetric |
|--------|-------|-----------|
| Score guarantee | Non-deterministic (varies run-to-run) | Deterministic (specific verdicts = specific scores) |
| Complexity | Simple (criteria string) | Higher (requires tree design) |
| Control | Less (LLM interprets criteria) | More (explicit decision paths) |
| Best for | Subjective criteria, nuanced evaluation | Objective criteria, multi-condition scoring |

---

## BaseMetric Subclassing

Build fully custom metrics by inheriting `BaseMetric` (single-turn) or `BaseConversationalMetric` (multi-turn). Required when: you need non-LLM evaluation, want to combine multiple metrics, or need custom scoring logic beyond GEval/DAG.

### Five Rules

**Rule 1: Inherit the right base class**

```python
from deepeval.metrics import BaseMetric, BaseConversationalMetric

# For single-turn (LLMTestCase)
class CustomMetric(BaseMetric):
    ...

# For multi-turn (ConversationalTestCase)
class CustomConversationalMetric(BaseConversationalMetric):
    ...
```

**Rule 2: Implement `__init__()`**

```python
class CustomMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: str = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True
    ):
        self.threshold = threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
```

**Rule 3: Implement `measure()` and `a_measure()`**

Both must: accept test case, set `self.score`, set `self.success`, optionally set `self.reason` and `self.error`.

```python
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = self._compute_score(test_case)
            if self.include_reason:
                self.reason = self._generate_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # If no async support, just delegate to sync:
        return self.measure(test_case)
```

**Rule 4: Implement `is_successful()`**

```python
def is_successful(self) -> bool:
    if self.error is not None:
        self.success = False
    else:
        try:
            self.success = self.score >= self.threshold
        except TypeError:
            self.success = False
    return self.success
```

**Rule 5: Name the metric**

```python
@property
def __name__(self):
    return "My Custom Metric"
```

### Non-LLM Example: ROUGE Metric

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

# Usage
test_case = LLMTestCase(
    input="...",
    actual_output="The quick brown fox jumps over the lazy dog.",
    expected_output="A quick brown fox leapt over a lazy dog."
)
metric = RougeMetric(threshold=0.6)
metric.measure(test_case)
print(metric.score, metric.is_successful())
```

### Composite Metric Example: Combining Multiple Metrics

```python
from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric
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
            relevancy_metric, faithfulness_metric = self._init_metrics()
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            self._set_results(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, faithfulness_metric = self._init_metrics()
            await relevancy_metric.a_measure(test_case)
            await faithfulness_metric.a_measure(test_case)
            self._set_results(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        return self.success

    @property
    def __name__(self):
        return "Composite Relevancy Faithfulness Metric"

    def _init_metrics(self):
        relevancy = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        faithfulness = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        return relevancy, faithfulness

    def _set_results(self, relevancy, faithfulness):
        composite_score = min(relevancy.score, faithfulness.score)
        self.score = 0 if self.strict_mode and composite_score < self.threshold else composite_score
        if self.include_reason:
            self.reason = relevancy.reason + "\n" + faithfulness.reason
        self.success = self.score >= self.threshold

# Usage in pytest
from deepeval import assert_test

def test_llm():
    metric = FaithfulRelevancyMetric()
    test_case = LLMTestCase(
        input="...",
        actual_output="...",
        retrieval_context=["..."]
    )
    assert_test(test_case, [metric])
```

---

## Custom Judge Model (DeepEvalBaseLLM)

Use any LLM as the evaluation judge by subclassing `DeepEvalBaseLLM`:

```python
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

custom_model = AzureChatOpenAI(
    openai_api_version="2024-08-01-preview",
    azure_deployment="gpt-4",
    azure_endpoint="https://...",
    openai_api_key="..."
)
azure_openai = AzureOpenAI(model=custom_model)

# Use with any metric
metric = GEval(
    name="Correctness",
    criteria="...",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=azure_openai  # pass custom judge
)
```

**Required methods to implement:**
1. `get_model_name()` — returns string
2. `load_model()` — returns model object
3. `generate(prompt: str)` — returns string response
4. `a_generate(prompt: str)` — async version of generate

**Important:** Azure OpenAI requires API version `2024-08-01-preview` or later.

---

## Custom Metric Decision Guide

| Scenario | Use |
|----------|-----|
| Subjective criteria (tone, correctness, coherence) | `GEval` |
| Deterministic multi-condition scoring | `DAGMetric` |
| Non-LLM scoring (BLEU, ROUGE, regex) | `BaseMetric` subclass |
| Combining multiple existing metrics | `BaseMetric` composite |
| Multi-turn custom criteria | `ConversationalGEval` |
| Multi-turn deterministic scoring | `ConversationalDAGMetric` |
| Custom evaluation LLM | `DeepEvalBaseLLM` subclass + pass to any metric |
