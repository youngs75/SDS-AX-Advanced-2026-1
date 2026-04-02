# Installation and Setup

## Read This When
- Starting a new DeepEval project and need installation steps, API key setup, or Confident AI login
- Need to understand key concepts like `LLMTestCase`, `ConversationalTestCase`, metrics, and `EvaluationDataset`
- Deciding between `assert_test()` (pytest) and `evaluate()` (script) for your evaluation workflow

## Skip This When
- Already have DeepEval installed and need metric-specific guidance → `references/03-eval-metrics/10-metrics-overview.md`
- Need use-case-specific quickstart code (RAG, chatbot, agents, MCP, arena) → `references/01-getting-started/20-quickstart-by-usecase.md`

---

## Installation

```bash
pip install -U deepeval
```

DeepEval automatically loads environment variables from dotenv files in the following precedence order (lowest to highest):
`.env` → `.env.<APP_ENV>` → `.env.local`

Process environment variables always take highest precedence. To disable dotenv autoloading (useful in CI/pytest):
```bash
export DEEPEVAL_DISABLE_DOTENV=1
```

## API Key Setup

Most metrics require `OPENAI_API_KEY` as an environment variable since DeepEval uses `gpt-4.1` by default for LLM-as-a-judge evaluation.

```bash
# In shell or .env.local
export OPENAI_API_KEY="your_api_key"
```

For Jupyter/Colab notebooks:
```python
%env OPENAI_API_KEY=your_api_key
```

## Connecting to Confident AI (Cloud)

Confident AI is the cloud platform for collaborative evaluation, dashboards, regression testing, and production monitoring. It is free to start.

```bash
deepeval login
```

Or with explicit key:
```bash
deepeval login --confident-api-key "ck_..."
```

Login persists the key to `.env.local` by default. To use a custom path:
```bash
deepeval login --confident-api-key "ck_..." --save dotenv:.env.custom
```

To log out (clear credentials):
```bash
deepeval logout
# Or for custom path:
deepeval logout --save dotenv:.myconf.env
```

Python-based login:
```python
import deepeval
deepeval.login("your-confident-api-key")
```

## Key Concepts

### LLMTestCase

Represents a single unit of LLM application interaction. Mandatory fields: `input`, `actual_output`. Optional fields include `expected_output`, `retrieval_context`, `context`, and more.

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    actual_output="A persistent cough and fever could signal various illnesses...",
    expected_output="A persistent cough and fever could indicate..."  # optional
)
```

### ConversationalTestCase

Represents multi-turn interactions, containing turns formatted as role/content pairs.

```python
from deepeval.test_case import ConversationalTestCase, Turn

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What is DeepEval?"),
        Turn(role="assistant", content="DeepEval is an open-source LLM eval package.")
    ]
)
```

### Metrics

- All metric scores range from 0 to 1.
- The `threshold` parameter determines test pass/fail: a test passes if `score >= threshold`.
- Default threshold is typically `0.5`.
- All metrics accept a `model` parameter to specify the LLM evaluator.

### EvaluationDataset and Golden

An `EvaluationDataset` is a collection of `Golden` objects (inputs with optional expected outputs) used for batch evaluation. Use `dataset.evals_iterator()` to loop through goldens while tracking test runs.

```python
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(goldens=[
    Golden(input="What is the capital of France?"),
    Golden(input="What is 2 + 2?")
])
```

## Writing Your First Test (Pytest-style)

Test files must use the `test_` prefix (e.g., `test_app.py`) for DeepEval recognition by pytest.

```python
# test_app.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen."
    )
    assert_test(test_case, [correctness_metric])
```

Run with:
```bash
deepeval test run test_app.py
```

## `assert_test()` vs `evaluate()`

| Function | Use Case | Integration |
|----------|----------|-------------|
| `assert_test(test_case, metrics)` | Pytest-style unit tests; raises exception on failure | Works with `deepeval test run` and pytest |
| `evaluate(test_cases, metrics)` | Script-based batch evaluation; returns results object | Works with `python main.py` |

### Using `evaluate()`

```python
# main.py
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)

test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    actual_output="A persistent cough and fever could signal various illnesses...",
    expected_output="A persistent cough and fever could indicate a range of illnesses..."
)

evaluate([test_case], [correctness_metric])
```

Run with:
```bash
python main.py
```

## CLI Basics

```bash
# Run evaluations through pytest integration
deepeval test run test_app.py

# Open latest test run on Confident AI
deepeval view

# Install/upgrade
pip install -U deepeval

# View all CLI commands
deepeval --help

# View help for a specific command
deepeval test run --help
```

## Saving Results Locally

To save results as JSON instead of (or in addition to) uploading to Confident AI:

```bash
# Linux/Mac
export DEEPEVAL_RESULTS_FOLDER="./data"

# Windows
set DEEPEVAL_RESULTS_FOLDER=.\data
```

## Metric Selection Guidance

| Use Case | Recommended Metrics |
|----------|---------------------|
| RAG pipelines | `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric` |
| Chatbots / multi-turn | `TurnRelevancyMetric`, `KnowledgeRetentionMetric`, `ConversationalGEval` |
| AI Agents | `TaskCompletionMetric`, `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric`, `PlanQualityMetric`, `PlanAdherenceMetric`, `StepEfficiencyMetric` |
| General LLM output | `GEval` (custom criteria), `AnswerRelevancyMetric`, `BiasMetric`, `ToxicityMetric` |
| MCP applications | `MCPUseMetric`, `MultiTurnMCPUseMetric`, `MCPTaskCompletionMetric` |
| Arena / comparison | `ArenaGEval` |

## Two Evaluation Modes

| Mode | Best For | Characteristics |
|------|----------|-----------------|
| **End-to-End** | Raw APIs, simple apps, chatbots | Black-box, minimal setup, CI/CD compatible |
| **Component-Level** | Agents, complex workflows, RAG pipelines | White-box, uses `@observe` tracing decorator, CI/CD compatible |

### Component-Level Evaluation with LLM Tracing

```python
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

@observe()
def llm_app(input: str):
    @observe(metrics=[AnswerRelevancyMetric()])
    def inner_component():
        update_current_span(
            test_case=LLMTestCase(
                input="Why is the sky blue?",
                actual_output="You mean why is the sky blue?"
            )
        )
    return inner_component()

dataset = EvaluationDataset(goldens=[Golden(input="Test input")])
for golden in dataset.evals_iterator():
    llm_app(golden.input)
```

Run with:
```bash
python main.py
```

## Error Handling and Retries

By default, DeepEval retries transient LLM errors once (2 attempts total):

- **Retried**: network/timeout errors and 5xx server errors
- **Rate limits (429)**: retried unless marked non-retryable
- **Backoff**: exponential with jitter (initial 1s, base 2, jitter 2s, cap 5s)

Configure via environment variables:
```bash
export DEEPEVAL_RETRY_MAX_ATTEMPTS=2         # Total attempts
export DEEPEVAL_RETRY_INITIAL_SECONDS=1.0    # Initial backoff
export DEEPEVAL_RETRY_EXP_BASE=2.0           # Exponential base
export DEEPEVAL_RETRY_JITTER=2.0             # Random jitter
export DEEPEVAL_RETRY_CAP_SECONDS=5.0        # Max sleep between retries
```

## Related Reference Files

- `20-quickstart-by-usecase.md` — Use-case-specific quickstarts (RAG, chatbot, agents, MCP, arena)
- `30-custom-models-and-embeddings.md` — Using non-OpenAI models for evaluation
- `40-integrations.md` — Model provider and framework integrations
