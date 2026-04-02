# DeepEval Benchmarks: Overview

## Read This When
- You need to wrap your LLM in `DeepEvalBaseLLM` before running any benchmark (required `generate`, `a_generate`, `load_model`, `get_model_name` methods)
- You want to understand how `benchmark.evaluate()` works, including batch evaluation, result properties (`overall_score`, `task_scores`, `predictions`), and configuration options (`n_shots`, CoT)
- You are troubleshooting output format issues with benchmarks (models not generating single-letter MCQ answers)

## Skip This When
- You need the full list of 16 available benchmarks with parameters and code examples -- see [20-available-benchmarks.md](./20-available-benchmarks.md)
- You want evaluation metrics for your own LLM application (not standardized benchmarks) -- see [../03-eval-metrics/](../03-eval-metrics/)

---

## What are LLM Benchmarks?

LLM benchmarks are standardized tests that evaluate model performance on specific skills like reasoning, comprehension, math, and code generation. Each benchmark comprises:

- **Tasks**: Evaluation datasets with target labels (expected outputs)
- **Scorer**: Determines whether predictions are correct using target labels as reference
- **Prompting techniques**: Few-shot learning and/or Chain of Thought (CoT) approaches

DeepEval provides implementations of all major research-backed benchmarks. Anyone can benchmark **any** LLM of their choice in just a few lines of code. All benchmarks follow original research paper implementations.

---

## Requirement: DeepEvalBaseLLM Wrapper

Before running any benchmark, you must wrap your LLM in the `DeepEvalBaseLLM` class. This is mandatory — benchmarks will not run against bare model objects.

### Required Methods

| Method | Required | Description |
|--------|----------|-------------|
| `load_model()` | Yes | Returns the underlying model object |
| `generate(prompt: str) -> str` | Yes | Synchronous text generation |
| `a_generate(prompt: str) -> str` | Yes | Async text generation (can delegate to `generate`) |
| `get_model_name() -> str` | Yes | Returns a human-readable model name string |
| `batch_generate(prompts: List[str]) -> List[str]` | No | Batch generation for improved throughput |

### Complete Implementation Example (Mistral 7B)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import List

class Mistral7B(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        device = "cuda"
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=100, do_sample=True
        )
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()
        device = "cuda"
        model_inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).to(device)
        model.to(device)
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=100, do_sample=True
        )
        return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return "Mistral 7B"


# Initialize the wrapper
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)

# Test it works before running benchmarks
print(mistral_7b("Write me a joke"))
```

> **Note:** The `batch_generate()` method is optional but significantly improves performance when evaluating large numbers of tasks. It is used when `batch_size` is specified in the `evaluate()` call.

---

## Running a Benchmark: benchmark() / evaluate()

The primary entry point for running any benchmark is the `.evaluate()` method on the benchmark object.

### Basic Usage

```python
from deepeval.benchmarks import MMLU

benchmark = MMLU()
results = benchmark.evaluate(model=mistral_7b)
print("Overall Score:", results)
```

### Batch Evaluation

Setting `batch_size` generates outputs in batches using `batch_generate()` if it is implemented on your custom LLM:

```python
from deepeval.benchmarks import MMLU

benchmark = MMLU()
results = benchmark.evaluate(model=mistral_7b, batch_size=5)
```

> **Caution:** `batch_size` is available for all benchmarks **except** HumanEval and GSM8K.

---

## Accessing Results

After calling `evaluate()`, three result properties are available on the benchmark object:

### Overall Score

```python
print("Overall Score:", benchmark.overall_score)
```

A single float from 0 to 1 representing aggregate model performance across all specified tasks.

### Task Scores

```python
print("Task-specific Scores:", benchmark.task_scores)
```

Returns a pandas DataFrame with per-task breakdown:

| Task | Score |
|------|-------|
| high_school_computer_science | 0.75 |
| astronomy | 0.93 |

### Prediction Details

```python
print("Detailed Predictions:", benchmark.predictions)
```

Returns a pandas DataFrame with individual prediction records:

| Task | Input | Prediction | Correct |
|------|-------|-----------|---------|
| high_school_computer_science | In Python 3, which function converts a string to int? | A | 0 |
| high_school_computer_science | Let x = 1. What is `x << 3` in Python 3? | B | 1 |

---

## Configuring Benchmarks

### Task Selection

Each benchmark has a unique Task enum. Specify a subset of tasks to evaluate only specific domains:

```python
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask

tasks = [MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY]
benchmark = MMLU(tasks=tasks)
```

By default, all available tasks are evaluated. Task enums are unique per benchmark (e.g., `MMLUTask`, `BigBenchHardTask`, `HumanEvalTask`).

### Few-Shot Learning (n_shots)

Control the number of in-context learning examples provided to the model. Each benchmark defines an allowed range based on the original research specification:

```python
from deepeval.benchmarks import HellaSwag

# HellaSwag supports up to 15 shots (default: 10)
benchmark = HellaSwag(n_shots=5)
```

Increasing `n_shots` generally improves answer format consistency and overall scores, especially for multiple-choice benchmarks that require single-letter responses.

### Chain of Thought (CoT) Prompting

Some benchmarks support Chain of Thought prompting, which instructs the model to articulate reasoning steps before answering:

```python
from deepeval.benchmarks import BigBenchHard

benchmark = BigBenchHard(enable_cot=True)
```

CoT is not universally supported — only benchmarks where the original paper demonstrated CoT benefits include this option (e.g., BigBenchHard, GSM8K).

---

## Important Warnings

**Output Format Issues**: LLMs frequently fail to generate properly structured outputs for public benchmarks. Many benchmarks require single-letter MCQ answers (e.g., "A", "B", "C", "D"). Improperly formatted outputs cause faulty results — a model that answers correctly but writes "The answer is A." instead of just "A" will be scored as incorrect.

**Resolution**: Follow the JSON confinement guide for custom LLMs to constrain outputs to the expected format. Use higher `n_shots` values to improve format adherence.

---

## Summary: benchmark.evaluate() Signature

```python
benchmark.evaluate(
    model: DeepEvalBaseLLM,   # Required: your wrapped model
    batch_size: int = None,   # Optional: enables batch_generate(); not available for HumanEval/GSM8K
)
```

### HumanEval Special Case

HumanEval uses a different signature — it passes `k` to the evaluate method for the pass@k metric:

```python
benchmark.evaluate(model=gpt_4, k=10)
```
