# RAG Evaluation — Reference Guide

Combined from: guides-rag-evaluation.md + guides-rag-triad.md
Sources: https://deepeval.com/guides/guides-rag-evaluation | https://deepeval.com/guides/guides-rag-triad

## Read This When
- Need a step-by-step workflow for evaluating a RAG pipeline end-to-end (retriever + generator)
- Want to understand and apply the RAG Triad (Answer Relevancy, Faithfulness, Contextual Relevancy) as referenceless metrics
- Setting up RAG unit tests in CI/CD with GitHub Actions or logging hyperparameters for optimization

## Skip This When
- Need API-level parameter details for RAG metrics (ContextualPrecisionMetric, FaithfulnessMetric, etc.) -- see `references/03-eval-metrics/20-rag-metrics.md`
- Looking for a complete end-to-end RAG tutorial with code you can run -- see `references/10-tutorials/20-rag-qa-agent.md`
- Want to generate synthetic test data for RAG evaluation -- see `references/09-guides/60-synthesizer.md`

---

# Part 1: RAG Evaluation

## Overview

Retrieval-Augmented Generation (RAG) enhances LLM outputs using external knowledge bases. The approach separates evaluation into two components: the **retriever** (responsible for fetching relevant context) and the **generator** (responsible for producing responses).

## Common Pitfalls in RAG Pipelines

### Retrieval Step

The retrieval process involves three key stages:

1. **Vectorizing input** using an embedding model (e.g., OpenAI's `text-embedding-3-large`)
2. **Performing vector search** on the vector store to retrieve top-K similar text chunks
3. **Reranking retrieved nodes** to align with use-case-specific relevance

Key evaluation questions:
- Does the embedding model capture domain-specific nuances?
- Does the reranker order nodes correctly?
- Are you retrieving the optimal amount of information?

### Generation Step

The generation process includes:

1. **Constructing a prompt** from user input and retrieval context
2. **Providing the prompt to an LLM** for augmented output generation

Key evaluation questions:
- Can smaller, faster, cheaper LLMs replace state-of-the-art models?
- How does temperature adjustment affect results?
- How do prompt template changes impact output quality?

## Evaluating Retrieval

DeepEval offers three metrics for retrieval evaluation:

### 1. ContextualPrecisionMetric
Evaluates whether the reranker ranks relevant nodes higher than irrelevant ones.

### 2. ContextualRecallMetric
Evaluates whether the embedding model accurately captures and retrieves relevant information.

### 3. ContextualRelevancyMetric
Evaluates whether text chunk size and top-K retrieve information with minimal irrelevancies.

### Code Example

```python
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()
```

### Test Case Definition

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="I'm on an F-1 visa, how long can I stay in the US after graduation?",
    actual_output="You can stay up to 30 days after completing your degree.",
    expected_output="You can stay up to 60 days after completing your degree.",
    retrieval_context=[
        """If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing
        your degree, unless you have applied for and been approved to participate in OPT."""
    ]
)
```

### Evaluation Execution

```python
contextual_precision.measure(test_case)
print("Score: ", contextual_precision.score)
print("Reason: ", contextual_precision.reason)

contextual_recall.measure(test_case)
print("Score: ", contextual_recall.score)
print("Reason: ", contextual_recall.reason)

contextual_relevancy.measure(test_case)
print("Score: ", contextual_relevancy.score)
print("Reason: ", contextual_relevancy.reason)
```

### Bulk Evaluation

```python
from deepeval import evaluate

evaluate(
    test_cases=[test_case],
    metrics=[contextual_precision, contextual_recall, contextual_relevancy]
)
```

## Evaluating Generation

DeepEval provides two metrics for generic generation evaluation:

### 1. AnswerRelevancyMetric
Evaluates whether the prompt template instructs the LLM to produce relevant, helpful outputs based on retrieval context.

### 2. FaithfulnessMetric
Evaluates whether the LLM output avoids hallucinations and contradictions with retrieval context information.

### Code Example

```python
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()
```

### Test Case and Evaluation

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="I'm on an F-1 visa, how long can I stay in the US after graduation?",
    actual_output="You can stay up to 30 days after completing your degree.",
    expected_output="You can stay up to 60 days after completing your degree.",
    retrieval_context=[
        """If you are in the U.S. on an F-1 visa, you are allowed to stay for 60 days after completing
        your degree, unless you have applied for and been approved to participate in OPT."""
    ]
)

answer_relevancy.measure(test_case)
print("Score: ", answer_relevancy.score)
print("Reason: ", answer_relevancy.reason)

faithfulness.measure(test_case)
print("Score: ", faithfulness.score)
print("Reason: ", faithfulness.reason)
```

### Bulk Evaluation

```python
from deepeval import evaluate

evaluate(
    test_cases=[test_case],
    metrics=[answer_relevancy, faithfulness]
)
```

## Beyond Generic Evaluation

The `GEval` metric enables custom evaluation criteria:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

dark_humor = GEval(
    name="Dark Humor",
    criteria="Determine how funny the dark humor in the actual output is",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

dark_humor.measure(test_case)
print("Score: ", dark_humor.score)
print("Reason: ", dark_humor.reason)
```

## E2E RAG Evaluation

Combine retrieval and generation metrics for comprehensive pipeline evaluation:

```python
evaluate(
    test_cases=test_cases,
    metrics=[
        contextual_precision,
        contextual_recall,
        contextual_relevancy,
        answer_relevancy,
        faithfulness,
        dark_humor  # Optional custom metrics
    ]
)
```

## Unit Testing RAG Systems in CI/CD

### Test File Creation (test_rag.py)

```python
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import pytest

dataset = EvaluationDataset(goldens=[...])

for goldens in dataset.goldens:
    dataset.add_test_case(...)  # convert golden to test case

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_rag(test_case: LLMTestCase):
    # metrics is the list of RAG metrics as shown in previous sections
    assert_test(test_case, metrics)
```

### CLI Execution

```bash
deepeval test run test_rag.py
```

### GitHub Actions Workflow (.github/workflows/rag-testing.yml)

```yaml
name: RAG Testing
on:
  push:
  pull:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
        # Some extra steps to setup and install dependencies,
        # and set OPENAI_API_KEY if you're using GPT models for evaluation
      - name: Run deepeval tests
        run: poetry run deepeval test run test_rag.py
```

## Optimizing On Hyperparameters

### Login to Confident AI

```bash
deepeval login
```

### Log Hyperparameters (test_rag.py)

```python
import deepeval

@deepeval.log_hyperparameters(model="gpt-4", prompt_template="...")
def custom_parameters():
    return {
        "embedding model": "text-embedding-3-large",
        "chunk size": 1000,
        "k": 5,
        "temperature": 0
    }
```

## Important Notes

**Vector Store Setup**: Users must populate vector stores before retrieval by chunking and vectorizing knowledge base documents.

**Test Case Input**: Do not include entire prompt templates as input; use only raw user input. Prompt templates are independent optimization variables.

**Metric Combination**: All three retrieval metrics should be used together for comprehensive evaluation, ensuring clean data flows to the generator.

**Flexible Evaluation**: All DeepEval metrics support custom thresholds, strict mode, reason inclusion, and any LLM for evaluation.

## Related Resources

- [ContextualPrecisionMetric Documentation](/docs/metrics-contextual-precision)
- [ContextualRecallMetric Documentation](/docs/metrics-contextual-recall)
- [ContextualRelevancyMetric Documentation](/docs/metrics-contextual-relevancy)
- [AnswerRelevancyMetric Documentation](/docs/metrics-answer-relevancy)
- [FaithfulnessMetric Documentation](/docs/metrics-faithfulness)
- [GEval Documentation](/docs/metrics-llm-evals)
- [Test Cases Section](/docs/evaluation-test-cases)
- [Evaluation Flags and Configs](/docs/evaluation-flags-and-configs#flags-for-deepeval-test-run)
- [CI/CD Testing Article](https://www.confident-ai.com/blog/how-to-evaluate-rag-applications-in-ci-cd-pipelines-with-deepeval)

## Community

- [GitHub Repository](https://github.com/confident-ai/deepeval)
- [Discord Server](https://discord.gg/a3K9c8GRGt)

---

**Last Updated**: February 16, 2026 by Jeffrey Ip

---

# Part 2: Using the RAG Triad for RAG Evaluation

## Introduction

Retrieval-Augmented Generation (RAG) enables Large Language Models to generate responses using external data beyond training data. Context is supplied as text chunks that are "parsed, vectorized, and indexed in vector databases for fast retrieval at inference time."

---

## What is the RAG Triad?

The RAG triad comprises three evaluation metrics designed to assess RAG pipeline performance:

### The Three Metrics

**1. Answer Relevancy**
- Measures how relevant the generated answers are
- Focuses on the **prompt template** hyperparameter
- Low scores indicate need for improved in-context learning examples or more fine-grained prompting instructions

**2. Faithfulness**
- Determines how much answers are hallucinations
- Concerns the **LLM** hyperparameter
- May require switching LLMs or fine-tuning if the model cannot leverage retrieval context effectively
- *Note: Also called "groundedness" in other documentation*

**3. Contextual Relevancy**
- Assesses whether retrieved text chunks are relevant to generating ideal answers
- Impacts **chunk size**, **top-K**, and **embedding model** hyperparameters
- Good embedding models retrieve semantically similar chunks; proper chunk size and top-K selection ensures only important information is selected

### Important Note on Metrics
The guide excludes contextual precision and contextual recall metrics because they "require a labelled expected answer (i.e. the ideal answer to a user input) which may not be possible for everyone," making this a "full referenceless RAG evaluation guide."

---

## Using the RAG Triad in DeepEval

### Step 1: Create a Test Case

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="...",
    actual_output="...",
    retrieval_context=["..."]
)
```

**Parameters:**
- `input`: The user query
- `actual_output`: The LLM-generated response
- `retrieval_context`: List of strings representing retrieved text chunks

### Step 2: Define RAG Triad Metrics

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric
)

answer_relevancy = AnswerRelevancyMetric()
faithfulness = FaithfulnessMetric()
contextual_relevancy = ContextualRelevancyMetric()
```

### Step 3: Evaluate Test Cases

```python
from deepeval import evaluate

evaluate(
    test_cases=[test_case],
    metrics=[
        answer_relevancy,
        faithfulness,
        contextual_relevancy
    ]
)
```

### Reference Documentation
- [`AnswerRelevancyMetric`](/docs/metrics-answer-relevancy)
- [`FaithfulnessMetric`](/docs/metrics-faithfulness)
- [`ContextualRelevancyMetric`](/docs/metrics-contextual-relevancy)

---

## Scaling RAG Evaluation

As evaluation efforts expand, you can:
1. Supply multiple test cases to the `test_cases` parameter
2. Generate synthetic datasets using DeepEval's synthesizer to test RAG applications at scale
3. Reference the [`evaluate()` function](/docs/evaluation-introduction#evaluating-without-pytest) documentation

---

## Related Links

**Navigation:**
- [Previous: RAG Evaluation](/guides/guides-rag-evaluation)
- [Next: Generating Synthetic Test Data](/guides/guides-using-synthesizer)

**Additional Resources:**
- [LLM evaluation metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM-as-a-judge](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
- [LLM testing](https://www.confident-ai.com/blog/llm-testing-in-2024-top-methods-and-strategies)
- [GitHub Repository](https://github.com/confident-ai/deepeval)

---

## Metadata
- **Last Updated:** February 16, 2026
- **Author:** Jeffrey Ip
- **Framework:** DeepEval by Confident AI
