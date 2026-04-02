# DeepEval Evaluation Fundamentals

## Read This When
- Need to understand or configure `evaluate()`, `assert_test()`, or `evals_iterator()` function parameters
- Setting up the `Prompt` class for prompt versioning, model settings, or hyperparameter logging
- Choosing between end-to-end and component-level evaluation modes for your application

## Skip This When
- Need test case field definitions (`LLMTestCase`, `ConversationalTestCase`) → `references/02-llm-evals/20-test-cases.md`
- Need CI/CD pipeline setup, CLI flags, or GitHub Actions YAML → `references/02-llm-evals/50-ci-cd-and-configs.md`

---

## Overview

DeepEval evaluates LLM application outputs using three core building blocks:
- **Test cases** - represent individual LLM interactions
- **Metrics** - define what "good" means for a given interaction
- **Evaluation datasets** - collections of goldens/test cases for bulk evaluation

Two evaluation types exist:
1. **End-to-end evaluation** - treats the entire LLM system as a black box, assessing observable inputs and outputs
2. **Component-level evaluation** - assesses individual internal components (retrievers, LLM calls, tool calls) using the `@observe` decorator

Both approaches integrate with `deepeval test run` (for CI/CD) and the `evaluate()` function (for Python scripts).

---

## evaluate() Function

The primary way to run evaluations outside of pytest. Returns a test run result and optionally logs to Confident AI.

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset

evaluate(
    test_cases=dataset,                  # or list of test cases
    metrics=[AnswerRelevancyMetric()],
    hyperparameters={"model": "gpt-4.1", "system_prompt": "..."},
    identifier="My Test Run v1.2",
    async_config=AsyncConfig(max_concurrent=10),
    display_config=DisplayConfig(display="failing"),
    error_config=ErrorConfig(ignore_errors=True),
    cache_config=CacheConfig(use_cache=True),
)
```

### evaluate() Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `test_cases` | `List[LLMTestCase]`, `List[ConversationalTestCase]`, or `EvaluationDataset` | Yes | Test cases to evaluate. Cannot mix `LLMTestCase` and `ConversationalTestCase` in the same call. |
| `metrics` | `List[BaseMetric]` | Yes | Evaluation metrics to apply to test cases |
| `hyperparameters` | `dict[str, Union[str, int, float, Prompt]]` | Optional | Arbitrary key-value pairs logged for optimization tracking on Confident AI; `Prompt` objects are also accepted |
| `identifier` | `str` | Optional | Human-readable label identifying this test run on Confident AI |
| `async_config` | `AsyncConfig` | Optional | Controls concurrency; uses defaults (`run_async=True`, `max_concurrent=20`) if omitted |
| `display_config` | `DisplayConfig` | Optional | Controls console output format; uses defaults if omitted |
| `error_config` | `ErrorConfig` | Optional | Controls error handling behavior; uses defaults (`skip_on_missing_params=False`, `ignore_errors=False`) if omitted |
| `cache_config` | `CacheConfig` | Optional | Controls caching; uses defaults (`use_cache=False`, `write_cache=True`) if omitted |

### Minimal Example

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

evaluate(
    test_cases=[LLMTestCase(input="What is 2+2?", actual_output="4")],
    metrics=[AnswerRelevancyMetric()]
)
```

### With Hyperparameters

```python
from deepeval import evaluate
from deepeval.prompt import Prompt

evaluate(
    test_cases=test_cases,
    metrics=[AnswerRelevancyMetric()],
    hyperparameters={
        "model": "gpt-4.1",
        "temperature": 0.7,
        "prompt": Prompt(alias="My System Prompt", text_template="You are helpful.")
    }
)
```

---

## assert_test() Function

Used with pytest for CI/CD integration. Raises an assertion error if any metric fails.

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_my_llm():
    test_case = LLMTestCase(input="...", actual_output="...")
    assert_test(test_case, [AnswerRelevancyMetric()])
```

### assert_test() Parameters — End-to-End Mode

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `test_case` | `LLMTestCase` or `ConversationalTestCase` | Yes | The test case to evaluate |
| `metrics` | `List[BaseMetric]` | Yes | List of evaluation metrics to apply |
| `run_async` | `bool` | Optional | Enables concurrent metric evaluation; defaults to `True` |

### assert_test() Parameters — Component-Level Mode

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `golden` | `Golden` | Yes | Golden providing input for the observed callback |
| `observed_callback` | callable decorated with `@observe` | Yes | The `@observe`-decorated LLM function containing span-level metrics |
| `run_async` | `bool` | Optional | Enables concurrent metric evaluation; defaults to `True` |

In component-level mode, `metrics` are NOT passed directly to `assert_test()` — they are defined at the span level via the `metrics` parameter of `@observe`. `LLMTestCase`s are created at runtime via `update_current_span()`.

---

## Evaluation Workflow: End-to-End

```
Goldens → LLM App → LLMTestCases → evaluate()/assert_test() → Metrics → Test Run
```

### Step-by-Step

```python
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval import evaluate

# 1. Define or load goldens
goldens = [
    Golden(input="What is the capital of France?"),
    Golden(input="Who wrote Hamlet?"),
]

# 2. Convert goldens to test cases by running your LLM app
dataset = EvaluationDataset(goldens=goldens)
for golden in dataset.goldens:
    result, chunks = your_llm_app(golden.input)
    dataset.add_test_case(LLMTestCase(
        input=golden.input,
        actual_output=result,
        retrieval_context=chunks
    ))

# 3. Run evaluation
evaluate(
    test_cases=dataset.test_cases,
    metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()]
)
```

### Metric Selection Guidelines

- Choose no more than **5 metrics** total
- **2-3 generic metrics** covering the application type (RAG, agent, chatbot)
- **1-2 custom metrics** for domain-specific requirements

---

## Evaluation Workflow: End-to-End with Tracing

For applications using `@observe` tracing, use `evals_iterator()` instead of creating test cases manually:

```python
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

for golden in dataset.evals_iterator(
    metrics=[AnswerRelevancyMetric()],
    identifier="Trace Run v1"
):
    your_llm_app(golden.input)  # @observe decorated function
```

The `evals_iterator()` runs metrics on the trace as a whole (end-to-end), while span-level `metrics` defined in `@observe` handle component-level evaluation.

### evals_iterator() Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `metrics` | `List[BaseMetric]` | Optional | End-to-end metrics to evaluate on the complete trace |
| `identifier` | `str` | Optional | Test run identifier on Confident AI |
| `async_config` | `AsyncConfig` | Optional | Concurrency customization |
| `display_config` | `DisplayConfig` | Optional | Console display customization |
| `error_config` | `ErrorConfig` | Optional | Error handling customization |
| `cache_config` | `CacheConfig` | Optional | Caching behavior customization |

---

## Evaluation Modes Comparison

| Feature | End-to-End | Component-Level |
|---------|------------|-----------------|
| Treats app as black box | Yes | No |
| Requires `@observe` | No | Yes |
| Metrics defined | at evaluate() | at @observe decorator |
| Test cases created | manually | at runtime via update_current_span() |
| Suitable for | RAG, PDF extraction, summarization | Agents, complex pipelines |
| Multi-turn support | Yes | No (single-turn only) |

---

## Prompt Class

The `Prompt` class links prompt templates to test runs for metrics-driven prompt optimization.

### Creating Prompts

**Messages-Based (chat format):**

```python
from deepeval.prompt import Prompt, PromptMessage

prompt = Prompt(
    alias="My System Prompt",
    messages_template=[
        PromptMessage(role="system", content="You are a helpful assistant.")
    ]
)
```

**Text-Based (single string):**

```python
from deepeval.prompt import Prompt

prompt = Prompt(
    alias="My System Prompt",
    text_template="You are a helpful assistant."
)
```

### Loading Prompts

**From Confident AI (versioned):**

```python
prompt = Prompt(alias="My System Prompt")
prompt.pull(version="00.00.01")
```

**From JSON file:**

```python
prompt = Prompt()
prompt.load(file_path="prompt.json")
# JSON format: {"messages": [{"role": "system", "content": "..."}]}
```

**From TXT file:**

```python
prompt = Prompt()
prompt.load(file_path="prompt.txt")
# TXT format: plain text string
```

Note: The filename automatically becomes the alias if no alias is specified.

### ModelSettings

Associate model configuration with a prompt:

```python
from deepeval.prompt import Prompt, ModelSettings, ModelProvider

model_settings = ModelSettings(
    provider=ModelProvider.OPEN_AI,
    name="gpt-4.1",
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

prompt = Prompt(
    alias="My Prompt",
    text_template="You are helpful.",
    model_settings=model_settings
)
```

**ModelSettings Parameters:**

| Setting | Type | Description |
|---------|------|-------------|
| `provider` | `ModelProvider` enum | Model provider (e.g., `ModelProvider.OPEN_AI`) |
| `name` | `str` | Model name (e.g., `"gpt-4.1"`) |
| `temperature` | `float` (0.0–2.0) | Randomness of output |
| `top_p` | `float` (0.0–1.0) | Nucleus sampling parameter |
| `frequency_penalty` | `float` (-2.0–2.0) | Penalizes repeated tokens |
| `presence_penalty` | `float` (-2.0–2.0) | Penalizes tokens already in context |
| `max_tokens` | `int` | Maximum tokens to generate |
| `verbosity` | `Verbosity` enum | Response detail level |
| `reasoning_effort` | `ReasoningEffort` enum | Thinking depth for reasoning models |
| `stop_sequences` | `List[str]` | Custom stop tokens |

### OutputSettings

```python
from deepeval.prompt import Prompt, OutputType
from pydantic import BaseModel

class MySchema(BaseModel):
    name: str
    age: int

prompt = Prompt(
    alias="Structured Output Prompt",
    text_template="Extract the person's info.",
    output_type=OutputType.SCHEMA,
    output_schema=MySchema
)
```

### Tools

```python
from deepeval.prompt import Prompt, Tool
from deepeval.prompt.api import ToolMode
from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str
    limit: int

prompt = Prompt(alias="Agent Prompt")
tool = Tool(
    name="WebSearch",
    description="Search the internet for information",
    mode=ToolMode.STRICT,
    structured_schema=SearchInput,
)
prompt.push(text="You are an agent.", tools=[tool])
```

### Using Prompts in Evaluations

**End-to-end (Python script):**

```python
evaluate(
    test_cases=test_cases,
    metrics=[AnswerRelevancyMetric()],
    hyperparameters={"prompt": prompt}
)
```

**End-to-end (CI/CD with pytest):**

```python
@deepeval.log_hyperparameters()
def hyperparameters():
    return {"prompt": prompt}
```

**Component-level (LLM span):**

```python
from deepeval.tracing import observe, update_llm_span

@observe(type="llm", metrics=[AnswerRelevancyMetric()])
def my_generator(input: str):
    # use prompt.messages_template in LLM call
    result = call_llm(prompt.messages_template, input)
    update_llm_span(prompt=prompt)
    return result
```

Note: `update_llm_span` can only be called inside an LLM span (where `type="llm"` in `@observe`).

---

## Running Tests via CLI

```bash
# Basic test run
deepeval test run test_file.py

# With flags
deepeval test run test_file.py -n 4 -v -c

# With identifier
deepeval test run test_file.py -id "My Test Run"
```

See `50-ci-cd-and-configs.md` for all CLI flags.

---

## Standalone Metric Measurement

Run a single metric directly without `evaluate()` or `assert_test()`:

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris"
)

metric.measure(test_case)
print(metric.score)      # float between 0 and 1
print(metric.reason)     # human-readable explanation
```

---

## Test Run Hooks

Execute custom code before/after a test run:

```python
import deepeval

@deepeval.on_test_run_end
def after_test_run():
    print("Test run complete!")
    send_notification()
```

---

## Log Hyperparameters in CI/CD

```python
import deepeval

@deepeval.log_hyperparameters()
def hyperparameters():
    return {
        "model": "gpt-4.1",
        "temperature": 0.7,
        "system_prompt": "You are a helpful assistant."
    }
```

---

## Related Reference Files

- `20-test-cases.md` - LLMTestCase, ConversationalTestCase, ArenaTestCase details
- `30-datasets-and-goldens.md` - EvaluationDataset, Golden data models
- `40-tracing-and-observability.md` - @observe decorator, update_current_span()
- `50-ci-cd-and-configs.md` - CLI flags, AsyncConfig, DisplayConfig, etc.
- `60-mcp-and-component-evals.md` - Component-level evaluation workflow
