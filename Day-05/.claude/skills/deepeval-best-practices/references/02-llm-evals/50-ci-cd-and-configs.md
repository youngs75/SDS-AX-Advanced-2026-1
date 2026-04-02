# DeepEval CI/CD and Configurations

## Read This When
- Setting up `deepeval test run` CLI flags, GitHub Actions YAML, or CI/CD regression testing pipelines
- Configuring `AsyncConfig`, `DisplayConfig`, `ErrorConfig`, or `CacheConfig` for `evaluate()`
- Need pytest integration patterns including `@pytest.mark.parametrize`, `log_hyperparameters`, or test run hooks

## Skip This When
- Need the `evaluate()` or `assert_test()` function signatures and parameters → `references/02-llm-evals/10-evaluation-fundamentals.md`
- Need to set up `@observe` tracing for component-level CI/CD testing → `references/02-llm-evals/40-tracing-and-observability.md`

---

## Overview

DeepEval integrates with CI/CD pipelines via pytest and the `deepeval test run` CLI command. Configuration objects (`AsyncConfig`, `DisplayConfig`, `ErrorConfig`, `CacheConfig`) customize evaluation behavior for both `evaluate()` and `assert_test()`.

---

## deepeval test run CLI

The primary command for running evaluations in CI/CD. Wraps pytest with DeepEval-specific features including trace collection, Confident AI reporting, and parallel execution.

```bash
deepeval test run test_file.py
```

**Important:** Use `deepeval test run` instead of bare `pytest`. While pytest works, `deepeval test run` adds reporting, tracing, caching, and Confident AI integration.

### All CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-n <int>` | Run test cases in parallel across N workers | `deepeval test run test.py -n 4` |
| `-c` | Use cached results (skip re-evaluation of unchanged test cases) | `deepeval test run test.py -c` |
| `-i` | Ignore errors during metric execution (don't fail the test run) | `deepeval test run test.py -i` |
| `-v` | Enable verbose mode for all metrics (overrides individual metric settings) | `deepeval test run test.py -v` |
| `-s` | Skip test cases with missing required metric parameters | `deepeval test run test.py -s` |
| `-id "<string>"` | Label this test run on Confident AI | `deepeval test run test.py -id "v1.2 release"` |
| `-d <mode>` | Filter displayed results: `"all"`, `"passing"`, or `"failing"` | `deepeval test run test.py -d "failing"` |
| `-r <int>` | Repeat each test case N times | `deepeval test run test.py -r 3` |

### Combined Flags Example

```bash
# Run 4 parallel workers, use cache, ignore errors, show only failing tests
deepeval test run test_example.py -n 4 -c -i -d "failing"
```

---

## pytest Integration Pattern

### Basic Test File Structure

```python
# test_llm_app.py
import pytest
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric

# Load dataset
dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")   # or load from CSV/JSON

# Convert goldens to test cases
for golden in dataset.goldens:
    result, chunks = your_llm_app(golden.input)
    dataset.add_test_case(LLMTestCase(
        input=golden.input,
        actual_output=result,
        retrieval_context=chunks
    ))

# Parametrize and test
@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_llm_app(test_case: LLMTestCase):
    assert_test(
        test_case=test_case,
        metrics=[AnswerRelevancyMetric(threshold=0.7), HallucinationMetric()]
    )

# Log hyperparameters for this test run
@deepeval.log_hyperparameters()
def hyperparameters():
    return {
        "model": "gpt-4.1",
        "temperature": 0.7,
        "system_prompt": "You are a helpful assistant."
    }
```

```bash
deepeval test run test_llm_app.py
```

### With Inline Dataset (for demonstration/small tests)

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="...", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_example(test_case: LLMTestCase):
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])
```

---

## assert_test() Reference

### End-to-End Mode (LLMTestCase)

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

assert_test(
    test_case=LLMTestCase(input="...", actual_output="..."),
    metrics=[AnswerRelevancyMetric()],
    run_async=True   # default True
)
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `test_case` | `LLMTestCase` or `ConversationalTestCase` | Yes | — | Test case to evaluate |
| `metrics` | `List[BaseMetric]` | Yes | — | Metrics to apply |
| `run_async` | `bool` | Optional | `True` | Enable concurrent metric execution |

### End-to-End Mode (ConversationalTestCase)

```python
assert_test(
    test_case=ConversationalTestCase(turns=[...]),
    metrics=[ConversationalRelevancyMetric()],
    run_async=True
)
```

### Component-Level Mode

```python
assert_test(
    golden=Golden(input="..."),
    observed_callback=your_llm_app,  # @observe decorated function
    run_async=True
)
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `golden` | `Golden` | Yes | — | Golden providing input to the callback |
| `observed_callback` | callable | Yes | — | `@observe`-decorated function; metrics defined at span level |
| `run_async` | `bool` | Optional | `True` | Enable concurrent metric execution |

In component-level mode: no `metrics` parameter needed (defined in `@observe`), no manual `LLMTestCase` creation needed (done via `update_current_span()`).

---

## Configuration Objects

All config objects are passed to `evaluate()` or configured via CLI flags for `deepeval test run`.

### AsyncConfig

Controls concurrency during evaluation.

```python
from deepeval.evaluate import AsyncConfig
from deepeval import evaluate

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=True,
        throttle_value=0,
        max_concurrent=10
    )
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `run_async` | `bool` | `True` | Enable concurrent evaluation of metrics, callbacks, and test cases |
| `throttle_value` | `int` | `0` | Seconds to wait between processing test cases (rate limiting) |
| `max_concurrent` | `int` | `20` | Maximum number of test cases evaluated in parallel |

CLI equivalent: `-n <int>` (parallelization)

---

### DisplayConfig

Controls what is printed to the console during evaluation.

```python
from deepeval.evaluate import DisplayConfig
from deepeval import evaluate

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    display_config=DisplayConfig(
        verbose_mode=True,
        display="failing",
        show_indicator=True,
        print_results=True,
        file_output_dir="./eval-results"
    )
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose_mode` | `bool` or `None` | `None` | Overrides `verbose_mode` for all individual metrics. `None` respects each metric's own setting. |
| `display` | `"all"`, `"failing"`, or `"passing"` | `"all"` | Filters which results are printed to console |
| `show_indicator` | `bool` | `True` | Shows a progress indicator during evaluation |
| `print_results` | `bool` | `True` | Prints evaluation results after completion |
| `file_output_dir` | `str` or `None` | `None` | Directory to write evaluation results as files |

CLI equivalents: `-v` (verbose), `-d <mode>` (display filter)

---

### ErrorConfig

Controls how errors are handled during evaluation.

```python
from deepeval.evaluate import ErrorConfig
from deepeval import evaluate

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    error_config=ErrorConfig(
        skip_on_missing_params=False,
        ignore_errors=False
    )
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_on_missing_params` | `bool` | `False` | Skip (rather than fail) metrics that require test case parameters not present in the test case |
| `ignore_errors` | `bool` | `False` | Ignore exceptions raised during metric execution; the metric is marked as errored but the test run continues |

Note: When both are `True`, `skip_on_missing_params` takes precedence.

CLI equivalents: `-s` (skip missing params), `-i` (ignore errors)

---

### CacheConfig

Controls caching of evaluation results to disk.

```python
from deepeval.evaluate import CacheConfig
from deepeval import evaluate

evaluate(
    test_cases=test_cases,
    metrics=metrics,
    cache_config=CacheConfig(
        use_cache=False,
        write_cache=True
    )
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_cache` | `bool` | `False` | Read from local cache instead of re-evaluating test cases that were already evaluated |
| `write_cache` | `bool` | `True` | Write evaluation results to disk cache for future reuse |

CLI equivalent: `-c` (use cache)

---

## Test Run Hooks

Execute custom code at the end of a test run:

```python
import deepeval

@deepeval.on_test_run_end
def after_test_run():
    print("Evaluation complete!")
    send_slack_notification()
    export_results_to_dashboard()
```

---

## log_hyperparameters Decorator

Log arbitrary metadata with a test run for tracking model configs and prompt versions:

```python
import deepeval
from deepeval.prompt import Prompt

@deepeval.log_hyperparameters()
def hyperparameters():
    return {
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2048,
        "prompt": Prompt(alias="System Prompt v3"),
    }
```

`Prompt` objects in hyperparameters are automatically linked on Confident AI for prompt-level metric attribution.

---

## GitHub Actions YAML Template

Full CI/CD integration for LLM regression testing on every push and PR:

```yaml
name: LLM App DeepEval Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH

      - name: Install Dependencies
        run: poetry install --no-root

      - name: Run DeepEval Unit Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
        run: poetry run deepeval test run test_llm_app.py
```

**Required secrets to configure in GitHub:**
- `OPENAI_API_KEY` — for LLM-based metric evaluation
- `CONFIDENT_API_KEY` — for Confident AI test run reporting (optional but recommended)

This pattern also works with Travis CI and CircleCI with equivalent configuration syntax.

---

## Regression Testing Patterns

### Pattern 1: Dataset-Driven Regression (Recommended)

```python
# test_llm_app.py
import pytest
import deepeval
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric

# Load from Confident AI (versioned, shareable)
dataset = EvaluationDataset()
dataset.pull(alias="Production Eval Set")

for golden in dataset.goldens:
    result = your_llm_app(golden.input)
    dataset.add_test_case(LLMTestCase(
        input=golden.input,
        actual_output=result,
        expected_output=golden.expected_output
    ))

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_llm_regression(test_case: LLMTestCase):
    assert_test(test_case, [
        AnswerRelevancyMetric(threshold=0.7),
        HallucinationMetric(threshold=0.3)
    ])

@deepeval.log_hyperparameters()
def hyperparameters():
    return {"model": "gpt-4.1", "version": "2026-02-20"}
```

### Pattern 2: CSV/JSON File Dataset

```python
dataset = EvaluationDataset()
dataset.add_goldens_from_csv_file(
    file_path="test_data/eval_set.csv",
    input_col_name="query"
)
```

### Pattern 3: Multi-Turn Regression

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.metrics import ConversationalRelevancyMetric

simulator = ConversationSimulator(model_callback=chatbot_callback)
conversational_test_cases = simulator.simulate(
    goldens=dataset.goldens,
    max_turns=10
)

@pytest.mark.parametrize("test_case", conversational_test_cases)
def test_chatbot_regression(test_case: ConversationalTestCase):
    assert_test(test_case=test_case, metrics=[ConversationalRelevancyMetric()])
```

### Pattern 4: Component-Level Regression

```python
import pytest
from deepeval import assert_test
from deepeval.dataset import Golden
from your_agent import your_llm_app  # @observe decorated

@pytest.mark.parametrize("golden", dataset.goldens)
def test_component_regression(golden: Golden):
    assert_test(
        golden=golden,
        observed_callback=your_llm_app
        # metrics are defined inside @observe on your_llm_app
    )
```

---

## Complete CI/CD Workflow

```
1. Curate dataset (goldens) -> push to Confident AI or commit to repo
2. Write test file with @pytest.mark.parametrize
3. Configure GitHub Actions YAML with secrets
4. On push/PR: deepeval test run executes
5. Results appear on Confident AI dashboard
6. Metric trends tracked across releases
7. Alerts configured for metric degradation
```

---

## Tips and Best Practices

- **Never hardcode test cases** in production test files — use dataset loading for scalability
- **Use `-n 4`** in CI to parallelize and reduce evaluation time on large datasets
- **Use caching (`-c`)** for development iterations where only some test cases change
- **Set thresholds** on metrics (`AnswerRelevancyMetric(threshold=0.7)`) to control pass/fail
- **At least 20 goldens** recommended for multi-turn simulation to get statistical significance
- **Separate CI secrets** for `OPENAI_API_KEY` and `CONFIDENT_API_KEY` — never hardcode in YAML

---

## Related Reference Files

- `10-evaluation-fundamentals.md` - evaluate() and assert_test() signatures
- `40-tracing-and-observability.md` - @observe for component-level CI/CD
- `60-mcp-and-component-evals.md` - Component-level evaluation patterns
- `30-datasets-and-goldens.md` - Dataset loading from CSV/JSON/Confident AI
