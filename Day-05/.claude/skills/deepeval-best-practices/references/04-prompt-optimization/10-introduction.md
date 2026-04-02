# Prompt Optimization: Introduction and Core Concepts

## Read This When
- You need to set up `PromptOptimizer` for the first time, including the `Prompt` class, model callback, and `optimize()` method
- You want to configure `AsyncConfig`, `DisplayConfig`, or `MutationConfig` for prompt optimization runs
- You need to do systematic hyperparameter tuning (model, temperature, chunk size) using `evaluate()` loops

## Skip This When
- You need detailed algorithm internals for GEPA, MIPROv2, or COPRO -- see [20-techniques.md](./20-techniques.md)
- You want to generate synthetic test data (goldens) rather than optimize prompts -- see [../05-synthetic-data/10-synthesizer-overview.md](../05-synthetic-data/10-synthesizer-overview.md)

---

## Overview

DeepEval's `PromptOptimizer` enables automatic prompt improvement through evaluation results from 50+ metrics. Rather than manually iterating through testing and tweaking, the system generates optimized prompts algorithmically using research-backed algorithms.

Two optimization algorithms are available:

- **GEPA**: Multi-objective genetic-Pareto search maintaining a Pareto frontier using metric-driven feedback on split golden sets (default)
- **MIPROv2**: Zero-shot surrogate-based search across unbounded prompt pools using epsilon-greedy selection with bootstrapped few-shot demos
- **COPRO**: Bounded-population, zero-shot coordinate-ascent algorithm with cooperative proposals

These algorithms are adapted from DSPy implementations within DeepEval's ecosystem.

See [20-techniques.md](./20-techniques.md) for detailed algorithm documentation.

---

## Quick Start

```python
from deepeval.dataset import Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.prompt import Prompt
from deepeval.optimizer import PromptOptimizer

# Define prompt to optimize
prompt = Prompt(text_template="Respond to the query.")

# Define model callback
async def model_callback(prompt_text: str):
    return await YourApp(prompt_text)

# Create optimizer and run
optimizer = PromptOptimizer(
    metrics=[AnswerRelevancyMetric()],
    model_callback=model_callback
)

optimized_prompt = optimizer.optimize(
    prompt=prompt,
    goldens=[Golden(
        input="What is Saturn?",
        expected_output="Saturn is a car brand."
    )]
)

print(optimized_prompt.text_template)
```

---

## The Prompt Class

The `Prompt` class holds the template text that will be optimized.

```python
from deepeval.prompt import Prompt

# TEXT-style prompt (simple string template)
prompt = Prompt(text_template="You are a helpful assistant. Answer this: {input}")

# The {input} placeholder is filled via interpolate()
interpolated = prompt.interpolate(input="What is the capital of France?")
```

Prompts come in two styles:
- **TEXT**: A single string template with `{variable}` placeholders
- **LIST**: A list of message dicts (chat-style), useful for system/user/assistant role separation

---

## PromptOptimizer

### Constructor

```python
from deepeval.optimizer import PromptOptimizer

optimizer = PromptOptimizer(
    metrics=[AnswerRelevancyMetric()],
    model_callback=model_callback
)
```

### PromptOptimizer Parameters

| Parameter | Status | Description |
|-----------|--------|-------------|
| `metrics` | **Required** | List of DeepEval metrics for scoring and feedback |
| `model_callback` | **Required** | Async callback wrapping your LLM application |
| `algorithm` | Optional | Optimization algorithm instance; defaults to `GEPA()` |
| `async_config` | Optional | `AsyncConfig` instance for concurrency customization |
| `display_config` | Optional | `DisplayConfig` instance for console display control |
| `mutation_config` | Optional | `MutationConfig` controlling message rewrites in LIST-style prompts |

**Tip:** For algorithm-specific settings (GEPA iterations, minibatch sizing, tie-breaking), instantiate the algorithm with custom parameters and pass via the `algorithm` argument.

---

## Model Callback

The model callback wraps your LLM application so the optimizer can invoke it. It receives the current candidate prompt and a golden, then must return the LLM's response as a string.

```python
from deepeval.prompt import Prompt
from deepeval.datasets import Golden, ConversationalGolden
from typing import Union

async def model_callback(
    prompt: Prompt,
    golden: Union[Golden, ConversationalGolden]
) -> str:
    # Inject golden input into prompt template
    interpolated_prompt = prompt.interpolate(input=golden.input)

    # Execute your LLM application
    res = await your_llm_app(interpolated_prompt)
    return res
```

### Callback Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `Prompt` | Current candidate prompt; use `prompt.interpolate()` to inject golden input |
| `golden` | `Golden` or `ConversationalGolden` | Current golden being scored; contains `input` for interpolation |

**Return type:** Must return `str`

---

## Optimization Workflow

### Running Optimization

```python
from deepeval.dataset import Golden
from deepeval.prompt import Prompt

optimized_prompt = optimizer.optimize(
    prompt=Prompt(text_template="Respond to the query."),
    goldens=[
        Golden(
            input="What is Saturn?",
            expected_output="Saturn is a car brand."
        ),
        Golden(
            input="What is Mercury?",
            expected_output="Mercury is a planet."
        ),
    ],
)

print("Optimized prompt:", optimized_prompt.text_template)
print("Optimization report:", optimizer.optimization_report)
```

### optimize() Parameters

| Parameter | Status | Description |
|-----------|--------|-------------|
| `prompt` | **Required** | `Prompt` instance to optimize |
| `goldens` | **Required** | List of `Golden` or `ConversationalGolden` instances for evaluation |

### Async Optimization

```python
import asyncio

async def main():
    optimized_prompt = await optimizer.a_optimize(
        prompt=prompt,
        goldens=goldens
    )

asyncio.run(main())
```

Enables concurrent prompt optimization without blocking the main thread.

---

## Optimization Report

After optimization, access the report via `optimizer.optimization_report`:

```python
report = optimizer.optimization_report
print(report)
```

### Report Fields

| Field | Type | Description |
|-------|------|-------------|
| `optimization_id` | `str` | Unique identifier for the optimization run |
| `best_id` | `str` | Internal id of the final best-performing configuration |
| `accepted_iterations` | `List[AcceptedIteration]` | Records parent/child ids, module id, and before/after scores |
| `pareto_scores` | `Dict[str, List[float]]` | Configuration id to Pareto subset scores mapping |
| `parents` | `Dict[str, Optional[str]]` | Configuration id to parent id mapping; forms ancestry tree |
| `prompt_configurations` | `Dict[str, PromptConfigSnapshot]` | Lightweight snapshots of prompts; records parent id and TEXT/LIST prompts |

---

## Configuration Objects

### AsyncConfig

Controls concurrency during optimization.

```python
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.configs import AsyncConfig

optimizer = PromptOptimizer(
    metrics=[...],
    model_callback=model_callback,
    async_config=AsyncConfig(
        run_async=True,
        throttle_value=0,
        max_concurrent=20
    )
)
```

| Parameter | Status | Default | Description |
|-----------|--------|---------|-------------|
| `run_async` | Optional | `True` | Enable concurrent evaluation of test cases and metrics |
| `throttle_value` | Optional | `0` | Throttle duration (seconds) per test case; increase for rate limit handling |
| `max_concurrent` | Optional | `20` | Maximum parallel test case evaluations; decrease for rate limit handling |

**Note:** `throttle_value` and `max_concurrent` apply only when `run_async=True`. Combining both parameters effectively manages rate limiting.

### DisplayConfig

Controls console output during optimization.

```python
from deepeval.optimizer.configs import DisplayConfig

optimizer = PromptOptimizer(
    metrics=[...],
    model_callback=model_callback,
    display_config=DisplayConfig(
        show_indicator=True,
        announce_ties=False
    )
)
```

| Parameter | Status | Default | Description |
|-----------|--------|---------|-------------|
| `show_indicator` | Optional | `True` | Display CLI progress indicator during optimization |
| `announce_ties` | Optional | `False` | Print one-line message when GEPA detects configuration ties |

### MutationConfig

Controls which messages in LIST-style prompts are eligible for mutation.

```python
from deepeval.optimizer.configs import MutationConfig

optimizer = PromptOptimizer(
    metrics=[...],
    model_callback=model_callback,
    mutation_config=MutationConfig(
        target_type="random",
        target_role=None,
        target_index=0
    )
)
```

| Parameter | Status | Default | Description |
|-----------|--------|---------|-------------|
| `target_type` | Optional | `"random"` | `MutationTargetType` for LIST-style prompt mutation eligibility; options: `"random"` or `"fixed_index"` |
| `target_role` | Optional | `None` | String role filter; when set, only messages with this role (case insensitive) eligible for mutation |
| `target_index` | Optional | `0` | Zero-based index for `"fixed_index"` target type |

---

## Hyperparameter Tuning with evaluate()

Beyond automated prompt optimization, DeepEval supports systematic hyperparameter tuning through nested evaluation loops. This approach is complementary to PromptOptimizer and useful when comparing discrete configurations.

### What Are Hyperparameters?

In DeepEval, hyperparameters are independent variables affecting the final `actual_output` of your LLM application:

- **model**: the LLM to use for generation
- **prompt template**: variations of prompt templates
- **temperature**: generation temperature value
- **max tokens**: max token limit for LLM generation
- **top-K**: number of retrieved nodes in `retrieval_context` (RAG pipelines)
- **chunk size**: size of retrieved nodes in `retrieval_context` (RAG pipelines)
- **reranking model**: model used to rerank retrieved nodes (RAG pipelines)

### Nested Loop Evaluation

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from typing import List

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

### Tracking Hyperparameters in CI/CD

Log hyperparameters during CI/CD testing to trace which combinations caused failures:

```python
import pytest
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_cases = [...]

@pytest.mark.parametrize("test_case", test_cases)
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

Run with:
```bash
deepeval login
deepeval test run test_file.py
```

---

## Related Documentation

- [20-techniques.md](./20-techniques.md) - GEPA, MIPROv2, and COPRO algorithm details
- [05-synthetic-data/10-synthesizer-overview.md](../05-synthetic-data/10-synthesizer-overview.md) - Generating goldens for optimization
