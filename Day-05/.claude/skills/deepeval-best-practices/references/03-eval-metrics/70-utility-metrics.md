# Utility Metrics (Deterministic / Non-LLM)

## Read This When
- Validating LLM outputs with exact string matching (ExactMatchMetric), JSON schema compliance (JsonCorrectnessMetric), or regex pattern matching (PatternMatchMetric)
- Need zero-cost, perfectly reproducible evaluation without LLM judge API calls
- Building pre-filters for structured outputs (JSON APIs, formatted strings, classification labels) before applying LLM-based metrics

## Skip This When
- Need LLM-as-a-judge evaluation for subjective criteria (tone, correctness, coherence) → `references/03-eval-metrics/80-custom-metrics.md`
- Evaluating RAG pipeline retrieval or generation quality → `references/03-eval-metrics/20-rag-metrics.md`
- Evaluating multimodal image-text outputs → `references/03-eval-metrics/60-multimodal-metrics.md`

---

DeepEval provides 3 deterministic, non-LLM metrics for cases where exact string matching, schema validation, or regex pattern compliance is required. These metrics do not use an LLM judge for scoring — they apply programmatic rules.

## Why Use Deterministic Metrics?

- Zero LLM cost for evaluation
- Perfectly reproducible results
- Instant execution (no API calls for scoring)
- Ideal for structured outputs (JSON APIs, formatted strings, classification labels)

---

## ExactMatchMetric

Evaluates whether the `actual_output` exactly matches the `expected_output` via strict string equality.

**Classification:** Non-LLM (deterministic), Single-turn, Reference-based

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `expected_output`

**Formula:**
```
Score = 1  if actual_output == expected_output
Score = 0  otherwise
```

Performs strict string equality comparison with no LLM involvement.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 1.0 | Minimum passing threshold (default 1.0 = requires exact match) |
| `verbose_mode` | bool | False | Print intermediate calculation steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase

metric = ExactMatchMetric(threshold=1.0, verbose_mode=True)

test_case = LLMTestCase(
    input="Translate 'Hello, how are you?' in french",
    actual_output="Bonjour, comment ça va ?",
    expected_output="Bonjour, comment allez-vous ?"
)

# Standalone
metric.measure(test_case)
print(metric.score, metric.reason)
# score = 0 (strings differ)

# Via evaluate
evaluate(test_cases=[test_case], metrics=[metric])
```

**Use cases:**
- Classification label validation
- Template output verification
- Canonical answer checking
- Code generation exact match

---

## JsonCorrectnessMetric

Evaluates whether the `actual_output` conforms to a specified Pydantic JSON schema. Uses schema validation — no LLM involved for scoring. An LLM is only used to generate a reason when validation fails AND `include_reason=True`.

**Classification:** Non-LLM (deterministic schema validation), Single-turn, Reference-based

**Required LLMTestCase fields:**
- `input`
- `actual_output`

**Required metric parameter:**
- `expected_schema` — a Pydantic `BaseModel` class defining the expected JSON structure

**Formula:**
```
Score = 1  if actual_output can be successfully loaded into expected_schema
Score = 0  otherwise
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `expected_schema` | Pydantic BaseModel | Yes | — | Expected JSON schema |
| `threshold` | float | No | 0.5 | Minimum passing threshold |
| `model` | str/DeepEvalBaseLLM | No | 'gpt-4.1' | LLM for generating failure reasons only |
| `include_reason` | bool | No | True | Generate reason when schema fails |
| `strict_mode` | bool | No | False | Binary 0/1 scoring |
| `async_mode` | bool | No | True | Concurrent execution |
| `verbose_mode` | bool | No | False | Print intermediate steps |

**Code example — simple schema:**

```python
from pydantic import BaseModel
from deepeval import evaluate
from deepeval.metrics import JsonCorrectnessMetric
from deepeval.test_case import LLMTestCase

class PersonSchema(BaseModel):
    name: str
    age: int
    email: str

metric = JsonCorrectnessMetric(
    expected_schema=PersonSchema,
    model="gpt-4",
    include_reason=True
)

test_case = LLMTestCase(
    input="Give me a JSON with name, age, and email fields",
    actual_output='{"name": "John Smith", "age": 30, "email": "john@example.com"}'
)

evaluate(test_cases=[test_case], metrics=[metric])

metric.measure(test_case)
print(metric.score, metric.reason)
# score = 1 if JSON parses successfully into PersonSchema
```

**Code example — list schema:**

```python
from pydantic import BaseModel, RootModel
from typing import List

class PersonSchema(BaseModel):
    name: str
    age: int

class PeopleList(RootModel[List[PersonSchema]]):
    pass

metric = JsonCorrectnessMetric(expected_schema=PeopleList)

test_case = LLMTestCase(
    input="Give me a list of two people as JSON",
    actual_output='[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
)

metric.measure(test_case)
print(metric.score)  # 1 if valid list of PersonSchema
```

**Use cases:**
- API response validation
- Structured data extraction verification
- Tool output schema compliance
- LLM-generated JSON format checking

---

## PatternMatchMetric

Evaluates whether the `actual_output` fully matches a specified regular expression pattern. Uses Python's `re.fullmatch` — the entire string must match, not just a substring.

**Classification:** Non-LLM (deterministic regex matching), Single-turn, Referenceless

**Required LLMTestCase fields:**
- `input`
- `actual_output`

**Required metric parameter:**
- `pattern` — regular expression string

**Formula:**
```
Score = 1  if re.fullmatch(pattern, actual_output) is not None
Score = 0  otherwise
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pattern` | str | Yes | — | Regular expression for full-match validation |
| `ignore_case` | bool | No | False | Case-insensitive matching when True |
| `threshold` | float | No | 1.0 | Minimum passing threshold |
| `verbose_mode` | bool | No | False | Print calculation steps |

**Code example — email validation:**

```python
from deepeval import evaluate
from deepeval.metrics import PatternMatchMetric
from deepeval.test_case import LLMTestCase

metric = PatternMatchMetric(
    pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
    ignore_case=False,
    threshold=1.0,
    verbose_mode=True
)

test_case = LLMTestCase(
    input="Generate a valid email address.",
    actual_output="example.user@domain.com"
)

evaluate(test_cases=[test_case], metrics=[metric])

metric.measure(test_case)
print(metric.score, metric.reason)
# score = 1 — valid email format
```

**Code example — date format validation:**

```python
import re

metric = PatternMatchMetric(
    pattern=r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
    threshold=1.0
)

test_case = LLMTestCase(
    input="What is today's date in ISO format?",
    actual_output="2026-02-20"
)

metric.measure(test_case)
print(metric.score)  # 1
```

**Code example — phone number validation:**

```python
metric = PatternMatchMetric(
    pattern=r"\(\d{3}\)\s\d{3}-\d{4}",  # (XXX) XXX-XXXX
    threshold=1.0
)
```

**Common patterns:**

| Use Case | Pattern |
|----------|---------|
| Email | `r"^[\w\.-]+@[\w\.-]+\.\w+$"` |
| ISO date | `r"\d{4}-\d{2}-\d{2}"` |
| UUID | `r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"` |
| Phone (US) | `r"\(\d{3}\)\s\d{3}-\d{4}"` |
| URL | `r"https?://[\w\.-]+\.\w+(/[\w\.-]*)*"` |
| Yes/No | `r"(yes\|no)"` with `ignore_case=True` |

---

## Utility Metrics Comparison

| Metric | LLM Used? | Requires `expected_output`? | Use When |
|--------|-----------|---------------------------|---------|
| `ExactMatchMetric` | No | Yes | Output must be character-perfect match |
| `JsonCorrectnessMetric` | Only for failure reasons | No (uses `expected_schema`) | Output must conform to JSON schema |
| `PatternMatchMetric` | No | No (uses `pattern`) | Output must match regex format |

## Combining with LLM Metrics

Deterministic metrics work well as pre-filters or alongside LLM metrics:

```python
from deepeval import evaluate
from deepeval.metrics import JsonCorrectnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from pydantic import BaseModel

class ExtractedData(BaseModel):
    entity: str
    sentiment: str
    confidence: float

# Validate structure deterministically, then evaluate quality with LLM
json_check = JsonCorrectnessMetric(expected_schema=ExtractedData)
quality_check = GEval(
    name="Extraction Quality",
    criteria="Evaluate whether the entity and sentiment were correctly extracted.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

test_case = LLMTestCase(
    input="Apple reported record earnings this quarter!",
    actual_output='{"entity": "Apple", "sentiment": "positive", "confidence": 0.95}'
)

evaluate(test_cases=[test_case], metrics=[json_check, quality_check])
```
