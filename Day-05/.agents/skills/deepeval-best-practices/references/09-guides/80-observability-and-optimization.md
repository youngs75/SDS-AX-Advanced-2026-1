# LLM Observability and Hyperparameter Optimization — Reference Guide

Combined from: guides-llm-observability.md + guides-optimizing-hyperparameters.md
Sources: https://deepeval.com/guides/guides-llm-observability | https://deepeval.com/guides/guides-optimizing-hyperparameters

## Read This When
- Need to understand LLM observability concepts (response monitoring, automated evaluations, tracing, human-in-the-loop)
- Want to iterate over hyperparameter combinations (model, prompt template, temperature, top-K, chunk size) using nested evaluation loops
- Setting up `@deepeval.log_hyperparameters` to track configurations across CI/CD test runs on Confident AI

## Skip This When
- Need step-by-step CI/CD pipeline setup with YAML configs -- see `references/09-guides/70-ci-cd-regression.md`
- Looking for a complete tutorial that walks through hyperparameter optimization in a real project -- see `references/10-tutorials/20-rag-qa-agent.md` or `references/10-tutorials/30-summarization-agent.md`
- Want API reference for evaluation flags and configuration options -- see `references/02-llm-evals/50-ci-cd-and-configs.md`

---

# Part 1: What is LLM Observability and Monitoring?

## Definition

"LLM observability is the practice of tracking and analyzing model performance in real-world use." The practice helps teams maintain accuracy, alignment with objectives, and user responsiveness.

## Purpose

Observability tools enable teams to "monitor behavior in real-time, catch performance changes early, and address these issues" before users are negatively impacted, supporting rapid troubleshooting and scalable AI initiatives.

## Why LLM Observability is Necessary

### 1. Complex Systems

LLM applications integrate numerous components (retrievers, APIs, embedders, models), making debugging challenging. Observability identifies root causes of performance issues and bottlenecks.

### 2. Hallucination Risk

LLMs produce incorrect or misleading responses on complex queries. In critical applications, hallucinations create serious consequences, requiring detection tools.

### 3. Unpredictable Models

LLMs constantly evolve through engineering improvements, causing unforeseen performance shifts. Continuous monitoring maintains reliability and output consistency.

### 4. User Unpredictability

Despite thorough pre-production testing, applications fail to address specific user queries. Observability detects these gaps for prompt improvements.

### 5. Continuous Experimentation

Post-deployment optimization requires testing model configurations, prompt designs, and knowledge bases. Robust observability enables scenario replays and comparative analysis.

## 5 Key Components of LLM Observability

### 1. Response Monitoring

Real-time tracking of user queries, LLM responses, and operational metrics (cost, latency) provides immediate insights for system adjustments.

### 2. Automated Evaluations

Automatic assessment of monitored responses identifies issues rapidly, reducing manual intervention and serving as the initial screening layer before human review.

### 3. Advanced Filtering

Stakeholders efficiently review monitored responses, flagging underperforming outputs for investigation. This prioritization streamlines troubleshooting.

### 4. Application Tracing

Mapping connections between application components reveals bugs and performance bottlenecks, ensuring reliability and system integrity.

### 5. Human-in-the-Loop

Human feedback on flagged outputs bridges automated evaluations with expert judgment, addressing complex cases and enriching development datasets.

## LLM Observability with Confident AI

Confident AI provides a comprehensive platform supporting all observability needs:
- Response Monitoring
- Automated Evaluations
- Advanced Filtering
- Application Tracing
- Human-in-the-Loop Integration

Integration requires minimal code, with documentation available at [confident-ai.com/docs](https://www.confident-ai.com/docs).

## Related Resources

- [LLM evaluation metrics guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM-as-a-judge evaluation](https://www.confident-ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method)
- [GitHub repository](https://github.com/confident-ai/deepeval)

---

# Part 2: Optimizing Hyperparameters for LLM Applications

## Overview

"Apart from catching regressions and sanity checking your LLM applications, LLM evaluation and testing plays an pivotal role in picking the best hyperparameters for your LLM application."

In DeepEval, hyperparameters refer to independent variables affecting the final `actual_output` of your LLM application, including the LLM used, prompt template, temperature, and related settings.

## Which Hyperparameters Should I Iterate On?

The following hyperparameters are typically recommended for iteration:

- **model**: the LLM to use for generation
- **prompt template**: variations of prompt templates for generation
- **temperature**: the temperature value for generation
- **max tokens**: the max token limit for LLM generation
- **top-K**: the number of retrieved nodes in `retrieval_context` (RAG pipelines)
- **chunk size**: the size of retrieved nodes in `retrieval_context` (RAG pipelines)
- **reranking model**: the model used to rerank retrieved nodes (RAG pipelines)

## Finding The Best Hyperparameter Combination

To find optimal hyperparameter combinations:

1. Choose LLM evaluation metrics fitting your evaluation criteria
2. Execute evaluations in nested for-loops while generating `actual_outputs` at evaluation time based on current hyperparameter combinations

### Example Implementation

**Helper function to construct test cases:**

```python
from typing import List
from deepeval.test_case import LLMTestCase

def construct_test_cases(model: str, prompt_template: str) -> List[LLMTestCase]:
    prompt = format_prompt_template(prompt_template)
    llm = get_llm(model)
    test_cases: List[LLMTestCase] = []
    for input in list_of_inputs:
        test_case = LLMTestCase(
            input=input,
            actual_output=generate_actual_output(llm, prompt)
        )
        test_cases.append(test_case)
    return test_cases
```

**Metric definition and nested loop:**

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric()

for model in models:
    for prompt_template in prompt_templates:
        evaluate(
            test_cases=construct_test_cases(model, prompt_template),
            metrics=[metric],
            hyperparameter={
                "model": model,
                "prompt template": prompt_template
            }
        )
```

## Keeping Track of Hyperparameters in CI/CD

Track hyperparameters during CI/CD pipeline testing to pinpoint hyperparameter combinations associated with failing test runs.

**Login to Confident AI:**

```bash
deepeval login
```

**Test file example:**

```python
import pytest
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_cases = [...]

@pytest.mark.parametrize(
    "test_case",
    test_cases,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])

@deepeval.log_hyperparameters(model="gpt-4", prompt_template="...")
def hyperparameters():
    return {
        "temperature": 1,
        "chunk size": 500
    }
```

**Run tests:**

```bash
deepeval test run test_file.py
```

## Related Links

- [RAG Evaluation Guide](/guides/guides-rag-evaluation)
- [Building Custom Metrics](/guides/guides-building-custom-metrics)
- [Regression Testing in CI/CD](/guides/guides-regression-testing-in-cicd)
- [LLM Evaluation Metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
