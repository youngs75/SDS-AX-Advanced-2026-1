# RAG Metrics

## Read This When
- Evaluating a RAG pipeline's retrieval quality (ContextualPrecision, ContextualRecall, ContextualRelevancy)
- Measuring generator faithfulness or answer relevancy in a RAG system
- Need the RAG Triad metrics, HallucinationMetric, SummarizationMetric, or RagasMetric parameters and examples

## Skip This When
- Evaluating AI agent tool usage or task completion → `references/03-eval-metrics/30-agent-metrics.md`
- Need custom LLM-as-a-judge criteria (GEval, DAGMetric) → `references/03-eval-metrics/80-custom-metrics.md`
- Evaluating multi-turn conversation quality → `references/03-eval-metrics/50-conversation-turn-metrics.md`

---

RAG (Retrieval-Augmented Generation) evaluation in DeepEval separates into two layers: the **retriever** and the **generator**. Each layer has dedicated metrics, plus additional utility metrics for summarization and hallucination.

## The RAG Triad

The RAG Triad is the minimal, fully referenceless evaluation set — it requires no labeled `expected_output`:

| Metric | Evaluates | Key Hyperparameter |
|--------|-----------|-------------------|
| `AnswerRelevancyMetric` | Generator output relevance | Prompt template |
| `FaithfulnessMetric` | Generator hallucinations | LLM choice |
| `ContextualRelevancyMetric` | Retriever context quality | Chunk size, top-K, embedding model |

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="I'm on an F-1 visa, how long can I stay in the US after graduation?",
    actual_output="You can stay up to 30 days after completing your degree.",
    retrieval_context=[
        "If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing your degree."
    ]
)

evaluate(
    test_cases=[test_case],
    metrics=[AnswerRelevancyMetric(), FaithfulnessMetric(), ContextualRelevancyMetric()]
)
```

---

## Generator Metrics

### AnswerRelevancyMetric

Measures how relevant the `actual_output` is to the `input`.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, RAG, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`

**Formula:**
```
Answer Relevancy = Number of Relevant Statements / Total Number of Statements
```

**Process:** Extracts all statements from `actual_output`, classifies each as relevant or irrelevant to `input`, computes ratio.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | class | `AnswerRelevancyTemplate` | Custom prompt template |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost."
)

evaluate(test_cases=[test_case], metrics=[metric])
print(metric.score, metric.reason)
```

**Custom template:**

```python
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

class CustomTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Given the text, breakdown and generate a list of statements presented.
Example:
Our new laptop model features a high-resolution Retina display for crystal-clear visuals.
{{
    "statements": [
        "The new laptop model has a high-resolution Retina display."
    ]
}}
===== END OF EXAMPLE ======
Text:
{actual_output}
JSON:"""

metric = AnswerRelevancyMetric(evaluation_template=CustomTemplate)
```

---

### FaithfulnessMetric

Measures whether the `actual_output` is factually consistent with the `retrieval_context`. Designed specifically for RAG — use `HallucinationMetric` for non-RAG contexts.

**Classification:** LLM-as-a-judge, Single-turn, Reference-based, RAG, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `retrieval_context`

**Formula:**
```
Faithfulness = Number of Truthful Claims / Total Number of Claims
```

**Process:**
1. LLM extracts all claims from `actual_output`
2. Each claim is classified as truthful if it does not contradict the `retrieval_context` facts

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `truths_extraction_limit` | int | None | Max truths to extract from context |
| `penalize_ambiguous_claims` | bool | False | Exclude ambiguous claims from count |
| `evaluation_template` | class | `FaithfulnessTemplate` | Custom prompt template |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

metric = FaithfulnessMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

evaluate(test_cases=[test_case], metrics=[metric])
```

**Custom template:**

```python
from deepeval.metrics.faithfulness import FaithfulnessTemplate

class CustomTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_claims(actual_output: str):
        return f"""Based on the given text, please extract a comprehensive list of facts.
Example:
Example Text:
"CNN claims that the sun is 3 times smaller than earth."
Example JSON:
{{"claims": []}}
===== END OF EXAMPLE ======
Text:
{actual_output}
JSON:"""

metric = FaithfulnessMetric(evaluation_template=CustomTemplate)
```

---

## Retriever Metrics

### ContextualRelevancyMetric

Measures how relevant the `retrieval_context` is to answering the `input`. Evaluates retriever quality — whether chunk size, top-K, and embedding model yield minimal irrelevant content.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, RAG, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `retrieval_context`

**Formula:**
```
Contextual Relevancy = Number of Relevant Statements / Total Number of Statements
```

**Process:** Extracts all statements from `retrieval_context`, classifies each as relevant or irrelevant to `input`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | class | `ContextualRelevancyTemplate` | Custom prompt template |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRelevancyMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

evaluate(test_cases=[test_case], metrics=[metric])
```

---

### ContextualPrecisionMetric

Measures whether relevant nodes in `retrieval_context` are ranked higher than irrelevant ones. Evaluates the reranker component of the retrieval pipeline.

**Classification:** LLM-as-a-judge, Single-turn, Reference-based, RAG, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `expected_output`
- `retrieval_context`

**Formula (Weighted Cumulative Precision):**
```
Contextual Precision = (1 / Num Relevant Nodes) * Sum(k=1 to n) [(Relevant Nodes Up to k / k) * rk]
```

Where:
- `k` = position index in retrieval context
- `n` = total length of retrieval context
- `rk` = 1 if node at position k is relevant, 0 otherwise

Higher scores mean relevant nodes appear earlier in the ranking.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | class | `ContextualPrecisionTemplate` | Custom prompt template |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

metric = ContextualPrecisionMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

evaluate(test_cases=[test_case], metrics=[metric])
```

---

### ContextualRecallMetric

Measures how much of the `expected_output` can be attributed to nodes in the `retrieval_context`. Evaluates whether the embedding model retrieves all relevant information.

**Classification:** LLM-as-a-judge, Single-turn, Reference-based, RAG, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `expected_output`
- `retrieval_context`

**Formula:**
```
Contextual Recall = Number of Attributable Statements / Total Number of Statements
```

**Process:**
1. Extracts all statements from `expected_output`
2. Classifies whether each statement can be attributed to nodes in `retrieval_context`

Note: this metric uses `expected_output` (not `actual_output`) to measure retriever capability for ideal responses.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | class | `ContextualRecallTemplate` | Custom prompt template |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

metric = ContextualRecallMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

evaluate(test_cases=[test_case], metrics=[metric])
```

---

## Other RAG / Generation Utility Metrics

### HallucinationMetric

General-purpose hallucination detection comparing `actual_output` against `context` (not `retrieval_context`). Use `FaithfulnessMetric` for RAG pipelines; use `HallucinationMetric` for general LLM hallucination checking against any provided context.

**Classification:** LLM-as-a-judge, Single-turn, Reference-based

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `context`

**Formula:**
```
Hallucination = Number of Contradicted Contexts / Total Number of Contexts
```

**Note:** For this metric, lower is better — a score of 0 means no hallucinations. The `threshold` acts as a maximum allowed value.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Maximum passing threshold (lower = better) |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

context = ["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]
actual_output = "A blond drinking water in public."

test_case = LLMTestCase(
    input="What was the blond doing?",
    actual_output=actual_output,
    context=context
)

metric = HallucinationMetric(threshold=0.5)
evaluate(test_cases=[test_case], metrics=[metric])
```

---

### SummarizationMetric

Evaluates whether the `actual_output` is a factually correct and comprehensive summary of the `input`. The only default DeepEval metric that cannot be cached.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless

**Required LLMTestCase fields:**
- `input` (the original text to be summarized)
- `actual_output` (the generated summary)

**Formula:**
```
Summarization = min(Alignment Score, Coverage Score)
```

- **Alignment Score:** Identifies hallucinations or contradictions in the summary
- **Coverage Score:** Validates that necessary information from source text is included

Uses closed-ended yes/no questions to compute both scores.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `assessment_questions` | list[str] | None | Custom yes/no questions for coverage; auto-generated if None |
| `n` | int | 5 | Number of questions to auto-generate when `assessment_questions` is None |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `truths_extraction_limit` | int | None | Max truths to extract |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

input_text = """The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer."""

actual_output = """The coverage score quantifies how well a summary captures and
accurately represents key information from the original text."""

test_case = LLMTestCase(input=input_text, actual_output=actual_output)
metric = SummarizationMetric(
    threshold=0.5,
    model="gpt-4",
    assessment_questions=[
        "Is the coverage score based on a percentage of 'yes' answers?",
        "Does a higher score mean a more comprehensive summary?"
    ]
)

evaluate(test_cases=[test_case], metrics=[metric])
print(metric.score_breakdown)  # Shows alignment and coverage sub-scores
```

---

### RagasMetric

Wraps the RAGAS framework as an aggregate metric combining four component metrics. Requires `pip install ragas`.

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `expected_output`
- `retrieval_context`

**Component metrics:**
- `RAGASAnswerRelevancyMetric`
- `RAGASFaithfulnessMetric`
- `RAGASContextualPrecisionMetric`
- `RAGASContextualRecallMetric`

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str/BaseChatModel | 'gpt-3.5-turbo' | OpenAI model or LangChain chat model |
| `embeddings` | Embeddings | N/A | LangChain embedding model (for AnswerRelevancy) |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase

metric = RagasMetric(threshold=0.5, model="gpt-3.5-turbo")
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

evaluate([test_case], [metric])
```

**Individual RAGAS metrics:**

```python
from deepeval.metrics.ragas import (
    RAGASAnswerRelevancyMetric,
    RAGASFaithfulnessMetric,
    RAGASContextualRecallMetric,
    RAGASContextualPrecisionMetric
)
```

---

## RAG Metric Selection Guide

| Scenario | Metrics |
|----------|---------|
| Referenceless production eval | AnswerRelevancy + Faithfulness + ContextualRelevancy |
| Full retriever evaluation (needs `expected_output`) | ContextualPrecision + ContextualRecall + ContextualRelevancy |
| Generator-only evaluation | AnswerRelevancy + Faithfulness |
| Summarization task | SummarizationMetric |
| General hallucination check (non-RAG) | HallucinationMetric |
| Quick RAGAS compatibility | RagasMetric |

## Retriever Metrics: What Each Measures

| Metric | What it checks | Hyperparameter it diagnoses |
|--------|---------------|---------------------------|
| ContextualRelevancy | Are chunks relevant to the query? | Chunk size, top-K, embedding model |
| ContextualPrecision | Are relevant chunks ranked first? | Reranker quality |
| ContextualRecall | Does context cover the ideal answer? | Embedding model recall |

## Multimodal Support

All core RAG metrics (`AnswerRelevancy`, `Faithfulness`, `ContextualRelevancy`, `ContextualPrecision`, `ContextualRecall`) support multimodal inputs via `MLLMImage` objects embedded in text fields:

```python
from deepeval.test_case import LLMTestCase, MLLMImage

test_case = LLMTestCase(
    input=f"Tell me about this landmark: {MLLMImage(url='./eiffel.jpg', local=True)}",
    actual_output="This appears to be the Eiffel Tower in France.",
    retrieval_context=[
        f"The Eiffel Tower {MLLMImage(url='./eiffel_ref.jpg', local=True)} is in Paris."
    ]
)
```
