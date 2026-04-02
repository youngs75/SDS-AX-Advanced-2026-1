# Tutorial: Summarization Agent Evaluation

## Read This When
- Want to learn by building a meeting summarization agent with separate summary and action-item extraction components
- Need a complete example of GEval custom metrics (Summary Concision, Action Item Accuracy) for evaluating non-RAG LLM outputs
- Looking for a dual-model optimization pattern (different models for different tasks) with component-level `@observe` tracing and CI/CD integration

## Skip This When
- Building a RAG-based system rather than a summarization pipeline -- see `references/10-tutorials/30-rag-qa-agent.md`
- Need the guide-level custom metrics procedure without a full tutorial -- see `references/09-guides/30-custom-metrics.md`
- Want API reference for GEval parameters and configuration -- see `references/03-eval-metrics/80-custom-metrics.md`

---

## Overview

This tutorial covers building and evaluating an LLM-powered meeting summarization agent using OpenAI and DeepEval. The agent generates concise meeting summaries and structured action item lists from transcripts — similar to tools like Otter.ai and Circleback. All concepts apply broadly to any summarization-focused LLM application.

**Technologies**: OpenAI, DeepEval

**What is evaluated**:
- Summary concision (GEval custom metric)
- Action item accuracy (GEval custom metric)
- Component-level tracing via `@observe`

---

## Stage 1: Development

### Architecture

The `MeetingSummarizer` class makes two separate LLM calls:
- `get_summary()`: generates a concise plain-text summary
- `get_action_items()`: extracts structured action items as JSON

Separating these concerns allows independent evaluation and tailored system prompts for each task.

### System Prompt for Initial Summarizer

```
You are an AI assistant tasked with summarizing meeting transcripts clearly and
accurately. Given the following conversation, generate a concise summary that captures the
key points discussed, along with a set of action items reflecting the concrete next steps
mentioned. Keep the tone neutral and factual, avoid unnecessary detail, and do not add
interpretation beyond the content of the conversation.
```

### Basic MeetingSummarizer

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class MeetingSummarizer:
    def __init__(
        self,
        model: str = "gpt-4",
        system_prompt: str = "",
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = system_prompt or (
            "You are an AI assistant tasked with summarizing meeting transcripts clearly and accurately. Given the following conversation, generate a concise summary that captures the key points discussed, along with a set of action items reflecting the concrete next steps mentioned. Keep the tone neutral and factual, avoid unnecessary detail, and do not add interpretation beyond the content of the conversation."
        )

    def summarize(self, transcript: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": transcript}
            ]
        )
        content = response.choices[0].message.content.strip()
        return content
```

> **Note:** Set the `OPENAI_API_KEY` environment variable in your `.env` file.

Generating summaries (initial):

```python
with open("meeting_transcript.txt", "r") as file:
    transcript = file.read().strip()

summarizer = MeetingSummarizer()
summary = summarizer.summarize(transcript)
print(summary)
```

### Updated Modular Architecture

Split into two specialized helper functions for better control, predictability, and component-level evaluation.

**System prompt for summary generation:**

```
You are an AI assistant summarizing meeting transcripts. Provide a clear and concise
summary of the following conversation, avoiding interpretation and unnecessary details.
Focus on the main discussion points only. Do not include any action items. Respond with
only the summary as plain text -- no headings, formatting, or explanations.
```

**Summary generation helper function:**

```python
class MeetingSummarizer:
    ...
    def get_summary(self, transcript: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.summary_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"
```

**System prompt for action items extraction:**

```
Extract all action items from the following meeting transcript. Identify individual and team-wide action items in the following format:
{
  "individual_actions": {
    "Alice": ["Task 1", "Task 2"],
    "Bob": ["Task 1"]
  },
  "team_actions": ["Task 1", "Task 2"],
  "entities": ["Alice", "Bob"]
}
Only include what is explicitly mentioned. Do not infer. You must respond strictly in valid JSON format -- no extra text or commentary.
```

**Action items generation helper function:**

```python
class MeetingSummarizer:
    ...
    def get_action_items(self, transcript: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.action_item_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            action_items = response.choices[0].message.content.strip()
            try:
                return json.loads(action_items)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from model", "raw_output": action_items}
        except Exception as e:
            print(f"Error generating action items: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}
```

**Updated summarize function:**

```python
class MeetingSummarizer:
    ...
    def summarize(self, transcript: str) -> tuple[str, dict]:
        summary = self.get_summary(transcript)
        action_items = self.get_action_items(transcript)
        return summary, action_items
```

### Complete Updated MeetingSummarizer

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MeetingSummarizer:
    def __init__(
        self,
        model: str = "gpt-4",
        summary_system_prompt: str = "",
        action_item_system_prompt: str = "",
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.summary_system_prompt = summary_system_prompt or (
            "You are an AI assistant summarizing meeting transcripts. Provide a clear "
            "and concise summary of the following conversation, avoiding interpretation "
            "and unnecessary details. Focus on the main discussion points only. Do not "
            "include any action items. Respond with only the summary as plain text -- "
            "no headings, formatting, or explanations."
        )
        self.action_item_system_prompt = action_item_system_prompt or (
            'Extract all action items from the following meeting transcript. Identify '
            'individual and team-wide action items in the following JSON format:\n'
            '{\n'
            '  "individual_actions": {"Alice": ["Task 1"], "Bob": ["Task 1"]},\n'
            '  "team_actions": ["Task 1"],\n'
            '  "entities": ["Alice", "Bob"]\n'
            '}\n'
            'Only include what is explicitly mentioned. Do not infer. Respond strictly '
            'in valid JSON format -- no extra text or commentary.'
        )

    def get_summary(self, transcript: str, model: str = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.summary_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"

    def get_action_items(self, transcript: str, model: str = None) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.action_item_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            raw = response.choices[0].message.content.strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from model", "raw_output": raw}
        except Exception as e:
            print(f"Error generating action items: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}

    def summarize(self, transcript: str) -> tuple[str, dict]:
        summary = self.get_summary(transcript)
        action_items = self.get_action_items(transcript)
        return summary, action_items
```

### Running the Updated Summarizer

```python
summarizer = MeetingSummarizer()
with open("meeting_transcript.txt", "r") as file:
    transcript = file.read().strip()

summary, action_items = summarizer.summarize(transcript)
print(summary)
print("JSON:")
print(json.dumps(action_items, indent=2))
```

Example action items JSON output:

```json
{
  "individual_actions": {
    "Ethan": ["Sync with design on the fallback UX messaging"],
    "Maya": [
      "Build the similarity metric",
      "Set up a test run for the hybrid model approach using GPT-4o and Claude"
    ]
  },
  "team_actions": [],
  "entities": ["Ethan", "Maya"]
}
```

**Key Points & Best Practices:**

1. **Modular Architecture**: Separating summary and action item generation enables independent testing and tailored prompts for each task.
2. **Output Structure**: JSON formatting for action items enables programmatic parsing and flexible UI rendering.
3. **Error Handling**: Both helper functions include try-catch blocks for API failures and JSON parsing errors.
4. **LLM Limitations**: LLMs are probabilistic and prone to inconsistency — eyeballing results will not catch subtle regressions, logical errors, or hallucinated action items.

---

## Stage 2: Evaluation

### Key Concept: Two Test Cases Per Transcript

Because `summarize()` makes two separate LLM calls, create two `LLMTestCase` objects per transcript:

```python
from deepeval.test_case import LLMTestCase

summary_test_case = LLMTestCase(
    input=transcript,
    actual_output=summary       # Plain text summary
)

action_item_test_case = LLMTestCase(
    input=transcript,
    actual_output=str(action_items)  # JSON string of action items
)
```

### Create and Store a Dataset of Goldens

```python
import os
from deepeval.dataset import Golden, EvaluationDataset

# Load transcripts from a folder
documents_path = "path/to/documents/folder"
transcripts = []
for document in os.listdir(documents_path):
    if document.endswith(".txt"):
        file_path = os.path.join(documents_path, document)
        with open(file_path, "r") as file:
            transcript = file.read().strip()
        transcripts.append(transcript)

goldens = []
for transcript in transcripts:
    golden = Golden(input=transcript)
    goldens.append(golden)

# Sanity check
for i, golden in enumerate(goldens):
    print(f"Golden {i}: ", golden.input[:20])

# Push to Confident AI cloud
dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="MeetingSummarizer Dataset")
```

### Pull Dataset and Create Test Cases

```python
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from meeting_summarizer import MeetingSummarizer

dataset = EvaluationDataset()
dataset.pull(alias="MeetingSummarizer Dataset")
summarizer = MeetingSummarizer()

summary_test_cases = []
action_item_test_cases = []

for golden in dataset.goldens:
    summary, action_items = summarizer.summarize(golden.input)
    summary_test_case = LLMTestCase(
        input=golden.input,
        actual_output=summary
    )
    action_item_test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(action_items)
    )
    summary_test_cases.append(summary_test_case)
    action_item_test_cases.append(action_item_test_case)
```

### GEval Metrics

GEval uses LLM-as-a-judge with chain-of-thought reasoning to evaluate outputs against custom criteria. This makes it ideal for summarization tasks where exact match is not applicable.

#### Summary Concision Metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

summary_concision = GEval(
    name="Summary Concision",
    criteria="Assess whether the summary is concise and focused only on the essential points of the meeting? It should avoid repetition, irrelevant details, and unnecessary elaboration.",
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)
```

#### Action Item Accuracy Metric

```python
action_item_check = GEval(
    name="Action Item Accuracy",
    criteria="Are the action items accurate, complete, and clearly reflect the key tasks or follow-ups mentioned in the meeting?",
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)
```

### Running Evaluations

```python
from deepeval import evaluate

# Evaluate summaries
evaluate(
    test_cases=summary_test_cases,
    metrics=[summary_concision]
)

# Evaluate action items
evaluate(
    test_cases=action_item_test_cases,
    metrics=[action_item_check]
)
```

Run `deepeval view` to view results on the Confident AI dashboard.

### Sample GEval Feedback

**Summary feedback**:
> "It omits extraneous details and is significantly shorter than the Input transcript. There's minimal repetition. However...some phrasing feels unnecessarily verbose"

**Action items feedback**:
> "The Actual Output captures some key action items...However, it misses several follow-ups...completeness is lacking"

> **Recommendation**: Use `gpt-4`, `gpt-4o`, or `claude-3-opus` as the evaluation model for more reliable scores and detailed reasoning.

---

## Stage 3: Improvement

### Tunable Hyperparameters

- **Prompt template**: System prompts for summary generation and action item extraction
- **Generation model**: The OpenAI model used for each task

### Pulling Datasets

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="MeetingSummarizer Dataset")
```

### Updated Prompts for Better Performance

**Updated summary prompt** (emphasizes executive-level brevity):

```
You are an AI assistant summarizing meeting transcripts for busy professionals.
Extract only high-value information: key technical insights, decisions made, and
problems discussed. Write in executive style -- concise, direct, under 60 seconds to
read. Focus on what matters. Omit pleasantries, repetition, and process details.
Respond with only the summary as plain text -- no headings or formatting.
```

**Updated action items prompt** (tighter JSON structure):

```
Extract all action items from the following meeting transcript. Identify individual and
team-wide action items strictly as they were stated. Use this exact JSON format:

{
  "individual_actions": {
    "<name>": ["<task 1>", "<task 2>"]
  },
  "team_actions": ["<task>"],
  "entities": ["<name>"]
}

Rules:
- Only include tasks explicitly mentioned in the transcript.
- Do not infer, paraphrase, or add tasks.
- Respond strictly in valid JSON -- no explanations, no commentary.
```

### Iterating Over Models

```python
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval import evaluate
from meeting_summarizer import MeetingSummarizer

dataset = EvaluationDataset()
dataset.pull(alias="MeetingSummarizer Dataset")

summary_system_prompt = "..."    # Use updated summary prompt above
action_item_system_prompt = "..." # Use updated action items prompt above
models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]

summary_concision = GEval(
    name="Summary Concision",
    criteria="Assess whether the summary is concise and focused only on the essential points of the meeting? It should avoid repetition, irrelevant details, and unnecessary elaboration.",
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)
action_item_check = GEval(
    name="Action Item Accuracy",
    criteria="Are the action items accurate, complete, and clearly reflect the key tasks or follow-ups mentioned in the meeting?",
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

for model in models:
    summarizer = MeetingSummarizer(
        model=model,
        summary_system_prompt=summary_system_prompt,
        action_item_system_prompt=action_item_system_prompt,
    )
    summary_test_cases = []
    action_item_test_cases = []

    for golden in dataset.goldens:
        summary, action_items = summarizer.summarize(golden.input)
        summary_test_case = LLMTestCase(input=golden.input, actual_output=summary)
        action_item_test_case = LLMTestCase(
            input=golden.input, actual_output=str(action_items)
        )
        summary_test_cases.append(summary_test_case)
        action_item_test_cases.append(action_item_test_case)

    evaluate(
        test_cases=summary_test_cases,
        metrics=[summary_concision],
        hyperparameters={"model": model},
    )
    evaluate(
        test_cases=action_item_test_cases,
        metrics=[action_item_check],
        hyperparameters={"model": model},
    )
```

### Results Comparison

| Model | Summary Concision | Action Item Accuracy |
|-------|-------------------|----------------------|
| gpt-3.5-turbo | 0.7 | 0.6 |
| gpt-4o | 0.9 | 0.7 |
| gpt-4-turbo | 0.8 | 0.9 |

**Key finding**: No single model excels at both tasks.
- gpt-4o excels at summary generation (0.9 score)
- gpt-4-turbo excels at action item generation (0.9 score)

### Dual-Model Solution

Use different models for different tasks based on evaluation results:

```python
from deepeval.tracing import observe

class MeetingSummarizer:
    @observe()
    def summarize(
        self,
        transcript: str,
        summary_model: str = "gpt-4o",
        action_item_model: str = "gpt-4-turbo",
    ) -> tuple[str, dict]:
        summary = self.get_summary(transcript, summary_model)
        action_items = self.get_action_items(transcript, action_item_model)
        return summary, action_items

    @observe()
    def get_summary(self, transcript: str, model: str = None) -> str:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": self.summary_system_prompt},
                {"role": "user", "content": transcript}
            ]
        )
        return response.choices[0].message.content.strip()

    @observe()
    def get_action_items(self, transcript: str, model: str = None) -> dict:
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": self.action_item_system_prompt},
                {"role": "user", "content": transcript}
            ]
        )
        raw = response.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON returned from model", "raw_output": raw}
```

> **Tip**: By logging hyperparameters in the `evaluate()` function, you can easily compare performance across runs in Confident AI and trace score changes to specific adjustments.

---

## Stage 4: Production (Evals in Prod)

### Setup Tracing with Component Metrics

Apply `@observe` with metrics on each component function and use `update_current_span()` to supply evaluation data:

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from deepeval.metrics import GEval
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv()

# Define metrics (reuse from evaluation stage)
summary_concision = GEval(
    name="Summary Concision",
    criteria=(
        "Assess whether the summary is concise and focused only on the essential "
        "points of the meeting. It should avoid repetition, irrelevant details, "
        "and unnecessary elaboration."
    ),
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

action_item_check = GEval(
    name="Action Item Accuracy",
    criteria=(
        "Are the action items accurate, complete, and clearly reflect the key tasks "
        "or follow-ups mentioned in the meeting?"
    ),
    threshold=0.9,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
)

class MeetingSummarizer:
    def __init__(
        self,
        model: str = "gpt-4",
        summary_system_prompt: str = "",
        action_item_system_prompt: str = "",
    ):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.summary_system_prompt = summary_system_prompt or "..."
        self.action_item_system_prompt = action_item_system_prompt or "..."

    @observe(type="agent")
    def summarize(
        self,
        transcript: str,
        summary_model: str = "gpt-4o",
        action_item_model: str = "gpt-4-turbo"
    ) -> tuple[str, dict]:
        summary = self.get_summary(transcript, summary_model)
        action_items = self.get_action_items(transcript, action_item_model)
        return summary, action_items

    @observe(metrics=[summary_concision], name="Summary")
    def get_summary(self, transcript: str, model: str = None) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.summary_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            summary = response.choices[0].message.content.strip()
            # Provide input/output data to the span for metric evaluation
            update_current_span(
                input=transcript, output=summary
            )
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"

    @observe(metrics=[action_item_check], name="Action Items")
    def get_action_items(self, transcript: str, model: str = None) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=[
                    {"role": "system", "content": self.action_item_system_prompt},
                    {"role": "user", "content": transcript}
                ]
            )
            action_items = response.choices[0].message.content.strip()
            try:
                action_items = json.loads(action_items)
                update_current_span(
                    input=transcript, actual_output=str(action_items)
                )
                return action_items
            except json.JSONDecodeError:
                return {"error": "Invalid JSON returned from model", "raw_output": action_items}
        except Exception as e:
            print(f"Error generating action items: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}
```

### Why Continuous Evaluation

Summarization agents process growing volumes of documents. As new document types, meeting formats, and speakers emerge, continuous testing is required to maintain reliability. DeepEval datasets allow on-the-fly test case generation from new transcripts.

### Using Datasets in CI

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="MeetingSummarizer Dataset")
```

### CI/CD Integration

**Test file** (`test_meeting_summarizer_quality.py`):

```python
import pytest
from deepeval.dataset import EvaluationDataset
from meeting_summarizer import MeetingSummarizer  # import your summarizer here
from deepeval import assert_test

dataset = EvaluationDataset()
dataset.pull(alias="MeetingSummarizer Dataset")

summarizer = MeetingSummarizer()

@pytest.mark.parametrize("golden", dataset.goldens)
def test_meeting_summarizer_components(golden):
    assert_test(golden=golden, observed_callback=summarizer.summarize)
```

**Run locally**:

```bash
poetry run deepeval test run test_meeting_summarizer_quality.py
```

**GitHub Actions workflow** (`.github/workflows/deepeval.yml`):

```yaml
name: Meeting Summarizer DeepEval Tests
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
        run: poetry run deepeval test run test_meeting_summarizer_quality.py
```

### Production Principles

- Decorate `summarize()` with `@observe(type="agent")` as the top-level span
- Decorate `get_summary()` and `get_action_items()` with `@observe(metrics=[...], name="...")`
- Call `update_current_span(input=..., output=...)` inside each component to provide evaluation data
- Separate metrics per component: `summary_concision` on Summary span, `action_item_check` on Action Items span
- Use dual-model approach: `gpt-4o` for summaries (0.9 concision), `gpt-4-turbo` for action items (0.9 accuracy)
- Run `deepeval login` to connect to Confident AI for cross-build performance tracking
