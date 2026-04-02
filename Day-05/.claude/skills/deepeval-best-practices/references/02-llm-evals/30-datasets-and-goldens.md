# DeepEval Datasets and Goldens

## Read This When
- Creating, loading, or saving evaluation datasets from JSON, CSV, or Confident AI
- Need to understand the difference between `Golden`, `ConversationalGolden`, and `EvaluationDataset`
- Generating synthetic goldens using `Synthesizer` or simulating conversations with `ConversationSimulator`

## Skip This When
- Need test case field definitions rather than dataset management → `references/02-llm-evals/20-test-cases.md`
- Need to run evaluations or understand `evaluate()`/`assert_test()` → `references/02-llm-evals/10-evaluation-fundamentals.md`

---

## Overview

An **evaluation dataset** is a collection of **goldens**. A golden is a precursor to a test case — it stores the input and expected results but does NOT require the LLM's actual output. During evaluation, goldens are converted to test cases by running the LLM app.

This separation enables:
- Reusing the same inputs across different LLM app versions
- Comparing performance between model iterations
- Storing expected results without running the LLM at dataset creation time

---

## Golden Data Models

### Golden (Single-Turn)

Used for single-turn (non-conversational) evaluation datasets.

```python
from deepeval.dataset import Golden
from deepeval.test_case import ToolCall

golden = Golden(
    input="What is the return policy for shoes?",
    expected_output="Customers can return shoes within 30 days for a full refund.",
    context=["All customers are eligible for a 30 day full refund at no extra cost."],
    expected_tools=[ToolCall(name="PolicySearch")],
    additional_metadata={"category": "returns", "priority": "high"},
    comments="Edge case: customer asking about policy generally",
    custom_column_key_values={"department": "support"}
)
```

### Golden Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `str` | Yes | The user query/input that will be passed to the LLM app |
| `expected_output` | `str` | Optional | The ideal/golden output; used by output quality metrics |
| `context` | `List[str]` | Optional | Ground-truth knowledge available to the LLM; static (not retrieved) |
| `expected_tools` | `List[ToolCall]` | Optional | Tools that should ideally be called for this input |
| `additional_metadata` | `Dict` | Optional | Arbitrary key-value metadata for dataset management or test case generation logic |
| `comments` | `str` | Optional | Free-text notes about this golden |
| `custom_column_key_values` | `Dict[str, str]` | Optional | Custom columns visible on Confident AI |
| `actual_output` | `str` | Avoid | The LLM's actual output; normally populated when converting to test case, not at golden creation |
| `retrieval_context` | `List[str]` | Avoid | Retrieved chunks; normally populated at test case creation time |
| `tools_called` | `List[ToolCall]` | Avoid | Tools actually called; normally populated at runtime |

---

### ConversationalGolden (Multi-Turn)

Used for multi-turn (conversational) evaluation datasets.

```python
from deepeval.dataset import ConversationalGolden

golden = ConversationalGolden(
    scenario="A frustrated customer wants to return shoes that don't fit.",
    expected_outcome="The assistant successfully processes the refund request.",
    user_description="Sarah Chen, a regular customer who bought shoes online.",
    context=["Refund policy: 30 days, full refund, no questions asked."],
    additional_metadata={"difficulty": "medium"},
    comments="Tests refund flow for apparel category",
    custom_column_key_values={"flow": "refund"}
)
```

### ConversationalGolden Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `scenario` | `str` | Yes | Description of the conversation situation/context |
| `expected_outcome` | `str` | Optional | What the conversation should ultimately achieve |
| `user_description` | `str` | Optional | Persona/background information about the user |
| `context` | `List[str]` | Optional | Static ground-truth data available to the chatbot |
| `additional_metadata` | `Dict` | Optional | Arbitrary metadata for management or simulation logic |
| `comments` | `str` | Optional | Free-text notes |
| `custom_column_key_values` | `Dict[str, str]` | Optional | Custom columns on Confident AI |
| `turns` | `Optional[Turn]` | Avoid | Conversation turns; normally populated by ConversationSimulator, not at golden creation |

---

## EvaluationDataset Class

The central container for managing goldens and test cases.

### Creating a Dataset

**Empty dataset:**

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
```

**Single-turn dataset:**

```python
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(goldens=[
    Golden(input="What is your name?"),
    Golden(input="What is the capital of France?"),
])
print(dataset._multi_turn)  # False
```

**Multi-turn dataset:**

```python
from deepeval.dataset import EvaluationDataset, ConversationalGolden

dataset = EvaluationDataset(goldens=[
    ConversationalGolden(
        scenario="User wants to open a bank account.",
        expected_outcome="Account successfully opened."
    )
])
print(dataset._multi_turn)  # True
```

**From existing test cases directly:**

```python
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

dataset = EvaluationDataset(test_cases=[
    LLMTestCase(input="...", actual_output="..."),
    LLMTestCase(input="...", actual_output="..."),
])
```

### Adding Goldens and Test Cases

```python
# Add a single golden
dataset.add_golden(Golden(input="New question?"))

# Add a conversational golden
dataset.add_golden(
    ConversationalGolden(
        scenario="User checking account balance.",
        expected_outcome="Balance displayed correctly."
    )
)

# Add a test case (after generating output from LLM)
dataset.add_test_case(
    LLMTestCase(input="...", actual_output="...")
)
```

---

## Loading Datasets

### From Confident AI

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")
print(dataset.goldens)  # inspect loaded goldens
```

### From JSON File (Goldens)

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    file_path="goldens.json",
)
```

### From JSON File (Test Cases)

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_test_cases_from_json_file(
    file_path="test_cases.json",
    input_key_name="query",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",   # optional
    context_key_name="context",                   # optional
    retrieval_context_key_name="retrieval_context"  # optional
)
```

### From CSV File (Goldens)

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_goldens_from_csv_file(
    file_path="goldens.csv",
)
```

### From CSV File (Test Cases)

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.add_test_cases_from_csv_file(
    file_path="test_cases.csv",
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",     # optional
    context_col_name="context",                      # optional
    context_col_delimiter=";",                       # delimiter for list fields
    retrieval_context_col_name="retrieval_context",  # optional
    retrieval_context_col_delimiter=";"              # delimiter for list fields
)
```

---

## Saving Datasets

### To Confident AI

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="My Dataset")

# Save as unfinalized (still being curated)
dataset.push(alias="My Dataset", finalized=False)
```

### To JSON File

```python
dataset.save_as(
    file_type="json",
    directory="./my-datasets",
    file_name="eval_set_v2",         # optional; defaults to YYYYMMDD_HHMMSS
    include_test_cases=False         # optional; defaults to False
)
```

### To CSV File

```python
dataset.save_as(
    file_type="csv",
    directory="./my-datasets",
    file_name="eval_set_v2",         # optional
    include_test_cases=False         # optional
)
```

### save_as() Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_type` | `"json"` or `"csv"` | Yes | Output format |
| `directory` | `str` | Yes | Directory path where the file will be saved |
| `file_name` | `str` | Optional | Custom filename without extension; defaults to `"YYYYMMDD_HHMMSS"` |
| `include_test_cases` | `bool` | Optional | Whether to include test case data; defaults to `False` |

---

## Converting Goldens to Test Cases

### Single-Turn Workflow

```python
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase

dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

# Convert goldens to test cases by running your LLM app
for golden in dataset.goldens:
    result, retrieved_chunks = your_llm_app(golden.input)
    dataset.add_test_case(LLMTestCase(
        input=golden.input,
        actual_output=result,
        retrieval_context=retrieved_chunks,
        expected_output=golden.expected_output,  # from the golden
        context=golden.context                   # from the golden
    ))

print(dataset.test_cases)
```

### Multi-Turn Workflow

```python
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import ConversationalTestCase

dataset = EvaluationDataset()
dataset.pull(alias="My Multi-Turn Dataset")

for golden in dataset.goldens:
    turns = generate_conversation(golden.scenario)   # your simulation logic
    dataset.add_test_case(ConversationalTestCase(
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        user_description=golden.user_description,
        turns=turns
    ))

print(dataset.test_cases)
```

---

## Running Evaluations with Datasets

### Python Script (evaluate())

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate

evaluate(test_cases=dataset.test_cases, metrics=[AnswerRelevancyMetric()])
```

Or pass the dataset directly:

```python
evaluate(test_cases=dataset, metrics=[AnswerRelevancyMetric()])
```

### CI/CD (assert_test with pytest)

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

@pytest.mark.parametrize("test_case", dataset.test_cases)
def test_llm_app(test_case: LLMTestCase):
    assert_test(test_case=test_case, metrics=[AnswerRelevancyMetric()])
```

```bash
deepeval test run test_llm_app.py
```

### Component-Level via evals_iterator()

```python
for golden in dataset.evals_iterator(
    metrics=[AnswerRelevancyMetric()],   # end-to-end metrics on the full trace
    identifier="My Component Run"
):
    your_llm_app(golden.input)   # @observe decorated function
```

---

## Generating Datasets with Synthesizer

When you don't have production data or existing goldens, use `Synthesizer` to generate them.

```python
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

synthesizer = Synthesizer()

# From documents (knowledge base)
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['knowledge_base.txt', 'faq.docx', 'manual.pdf']
)

# From prepared contexts
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=[
        ["Context chunk 1", "Context chunk 2"],
        ["Another context"]
    ]
)

# From scratch (no knowledge base)
goldens = synthesizer.generate_goldens_from_scratch(
    subject="customer service for an e-commerce shoe store",
    num_goldens=20
)

# By augmenting existing goldens
goldens = synthesizer.generate_goldens_from_goldens(
    goldens=existing_goldens,
    num_goldens_per_golden=3
)

dataset = EvaluationDataset(goldens=goldens)
```

---

## Conversation Simulator

Generates `ConversationalTestCase`s from `ConversationalGolden`s by simulating real conversations.

```python
from deepeval.simulator import ConversationSimulator
from typing import List, Dict

# Define the simulator with user intents and profile items
simulator = ConversationSimulator(
    user_intentions={"Opening a bank account": 1, "Checking account balance": 0.5},
    user_profile_items=[
        "full name",
        "current address",
        "bank account number",
        "date of birth",
        "phone number",
    ],
)

# Define the chatbot callback
async def model_callback(input: str, conversation_history: List[Dict[str, str]]) -> str:
    return your_chatbot.respond(input, conversation_history)

# Run simulation
conversational_test_cases = simulator.simulate(
    model_callback=model_callback,
    stopping_criteria="Stop when the user's banking request has been fully resolved.",
)
print(conversational_test_cases)
```

**Alternative callback signature (for E2E evals):**

```python
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.test_case import Turn

simulator = ConversationSimulator(model_callback=chatbot_callback)

# Simulate from goldens in a dataset
conversational_test_cases = simulator.simulate(
    goldens=dataset.goldens,
    max_turns=10
)
```

---

## Best Practices for Dataset Curation

1. **Ensure telling test coverage** — Include diverse real-world inputs, varying complexity levels, and edge cases
2. **Focused, quantitative test cases** — Design with clear scope enabling meaningful performance metrics
3. **Define clear objectives** — Align datasets with specific evaluation goals
4. **Minimum size** — Use at least 20 goldens for multi-turn simulation to get statistically meaningful results
5. **Use Confident AI** — Push datasets to Confident AI for versioning and team collaboration

---

## get_current_golden()

During tracing-based evaluation, access the current golden being evaluated:

```python
from deepeval.dataset import get_current_golden
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase

@observe()
def my_component(input: str):
    golden = get_current_golden()
    expected = golden.expected_output if golden else None

    result = process(input)
    update_current_span(
        test_case=LLMTestCase(
            input=input,
            actual_output=result,
            expected_output=expected
        )
    )
    return result
```

---

## Related Reference Files

- `10-evaluation-fundamentals.md` - evaluate() function, assert_test() function
- `20-test-cases.md` - LLMTestCase, ConversationalTestCase, ToolCall
- `40-tracing-and-observability.md` - @observe, update_current_span, evals_iterator
- `60-mcp-and-component-evals.md` - Component-level evaluation workflow
