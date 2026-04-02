# DeepEval Metrics Overview

## Read This When
- Starting with DeepEval metrics and need to understand the full metric catalog, categories, and BaseMetric interface
- Choosing which metrics to use for your use case (RAG, agent, chatbot, safety, custom) via the decision tree
- Configuring common metric parameters (threshold, model, strict_mode, async_mode) or setting up a custom LLM judge

## Skip This When
- Already know which metric you need and want detailed parameters/examples → go to the specific metric file (e.g., `references/03-eval-metrics/20-rag-metrics.md`)
- Building a custom metric with GEval, DAGMetric, or BaseMetric → `references/03-eval-metrics/80-custom-metrics.md`

---

## What is a Metric?

In DeepEval, "a test case represents the thing you're trying to measure, the metric acts as the ruler based on a specific criteria." Every metric accepts an `LLMTestCase` (or `ConversationalTestCase`) and produces a score between 0 and 1 with an explanation.

DeepEval provides 50+ state-of-the-art, ready-to-use metrics. Nearly all predefined metrics use **LLM-as-a-judge** with techniques including:
- QAG (question-answer-generation)
- DAG (deep acyclic graphs)
- G-Eval

## Metric Categories

### Custom Metrics
- `GEval` - LLM-as-a-judge with custom criteria/evaluation steps
- `DAGMetric` - Deterministic decision tree evaluation
- `ConversationalGEval` - G-Eval adapted for multi-turn conversations
- `ConversationalDAGMetric` - DAG adapted for multi-turn conversations

### RAG Metrics (Retriever)
- `ContextualRelevancyMetric` - Relevance of retrieved context to input
- `ContextualPrecisionMetric` - Ranking quality of retrieved nodes
- `ContextualRecallMetric` - Coverage of retrieval context vs expected output

### RAG Metrics (Generator)
- `AnswerRelevancyMetric` - Relevance of output to input
- `FaithfulnessMetric` - Factual alignment with retrieval context

### Agentic Metrics
- `TaskCompletionMetric` - Did the agent accomplish the task?
- `ArgumentCorrectnessMetric` - Correct arguments generated for tools?
- `ToolCorrectnessMetric` - Correct tools called?
- `StepEfficiencyMetric` - Were steps efficient (no redundancy)?
- `PlanAdherenceMetric` - Did the agent follow its plan?
- `PlanQualityMetric` - Was the plan logical and complete?
- `MCPUseMetric` - MCP primitives used effectively?
- `MultiTurnMCPUseMetric` - Multi-turn MCP usage quality
- `MCPTaskCompletionMetric` - MCP task completion rate

### Multi-Turn / Chatbot Metrics
- `KnowledgeRetentionMetric` - Does the chatbot remember info from conversation?
- `RoleAdherenceMetric` - Does the chatbot maintain its assigned role?
- `ConversationCompletenessMetric` - Are user intentions satisfied?
- `TurnRelevancyMetric` - Are responses relevant throughout the conversation?
- `TurnFaithfulnessMetric` - Are responses faithful to retrieval context?
- `TurnContextualRelevancyMetric` - Is retrieved context relevant per turn?
- `TurnContextualRecallMetric` - Does retrieval context support expected outcome?
- `TurnContextualPrecisionMetric` - Are relevant context nodes ranked higher?
- `GoalAccuracyMetric` - Agent accuracy in achieving goals in multi-turn

### Safety Metrics
- `BiasMetric` - Gender, racial, or political bias in output
- `ToxicityMetric` - Toxic content in output
- `PIILeakageMetric` - Personal information leakage
- `MisuseMetric` - Domain-specific misuse
- `RoleViolationMetric` - Persona/character violations
- `NonAdviceMetric` - Inappropriate professional advice
- `PromptAlignmentMetric` - Adherence to prompt instructions
- `TopicAdherenceMetric` - Staying on relevant topics

### Multimodal Metrics
- `ImageCoherenceMetric` - Image-text alignment
- `ImageHelpfulnessMetric` - Image contributes to comprehension
- `ImageReferenceMetric` - Image accurately described by text
- `TextToImageMetric` - Text-to-image generation quality
- `ImageEditingMetric` - Image editing quality

### Utility / Deterministic Metrics
- `HallucinationMetric` - Factual contradictions with context (non-RAG)
- `SummarizationMetric` - Summary quality (alignment + coverage)
- `RagasMetric` - Aggregate RAG evaluation (RAGAS framework wrapper)
- `ExactMatchMetric` - Exact string match
- `JsonCorrectnessMetric` - JSON schema validation
- `PatternMatchMetric` - Regex pattern matching

## BaseMetric Interface

All metrics expose a consistent interface:

```python
metric.measure(test_case)        # Execute metric (blocks)
await metric.a_measure(test_case) # Async execution
metric.score                     # float 0-1
metric.reason                    # str explanation
metric.is_successful()           # bool: score >= threshold
metric.threshold                 # float (default 0.5)
metric.strict_mode               # bool: forces binary 0 or 1
metric.verbose_mode              # bool: prints debug logs
metric.async_mode                # bool: concurrent steps (default True)
```

## Common Parameters (All Metrics)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum score to pass (`is_successful()` returns True) |
| `model` | str or DeepEvalBaseLLM | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Whether to populate `metric.reason` |
| `strict_mode` | bool | False | Forces score to 1 or 0; overrides threshold to 1 |
| `async_mode` | bool | True | Run internal steps concurrently |
| `verbose_mode` | bool | False | Print intermediate computation logs |

## Threshold Concept

- All metrics score between **0 and 1**
- Default threshold is **0.5** (for most metrics)
- `is_successful()` returns `True` when `score >= threshold`
- `strict_mode=True` enforces binary 0/1 scoring (threshold becomes 1.0)
- For **safety metrics** (Bias, Toxicity), lower scores are better — these measure the proportion of problematic content, so a score of 0 is ideal. The threshold acts as a maximum: pass if `score <= threshold`.

## Running Metrics

### Standalone (for debugging)

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(input="...", actual_output="...")
metric = AnswerRelevancyMetric(threshold=0.7)
metric.measure(test_case)
print(metric.score, metric.reason)
```

Caution: standalone skips caching, reporting, and Confident AI integration.

### Via evaluate() (recommended)

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

evaluate(
    test_cases=[LLMTestCase(input="...", actual_output="...")],
    metrics=[AnswerRelevancyMetric(threshold=0.7)]
)
```

### Async batch

```python
import asyncio

async def run_all():
    await asyncio.gather(
        metric1.a_measure(test_case),
        metric2.a_measure(test_case),
    )

asyncio.run(run_all())
```

### Component-level (tracing)

```python
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden

@observe(metrics=[metric])
def inner_component():
    update_current_span(test_case=LLMTestCase(input="...", actual_output="..."))

@observe
def llm_app(input: str):
    inner_component()

evaluate(observed_callback=llm_app, goldens=[Golden(input="...")])
```

## Metric Selection Guidelines

Limit to **no more than 5 metrics** with this breakdown:
- **2–3** generic, system-specific metrics
- **1–2** custom, use-case-specific metrics

### Decision Tree by Use Case

```
What are you building?
├── RAG system
│   ├── Evaluate retriever → ContextualRelevancy + ContextualPrecision + ContextualRecall
│   ├── Evaluate generator → AnswerRelevancy + Faithfulness
│   └── Full referenceless → AnswerRelevancy + Faithfulness + ContextualRelevancy (RAG Triad)
├── AI Agent
│   ├── Explicit planning → PlanQualityMetric + PlanAdherenceMetric
│   ├── Tool calling → ToolCorrectnessMetric + ArgumentCorrectnessMetric
│   └── Overall success → TaskCompletionMetric + StepEfficiencyMetric
├── Chatbot / multi-turn
│   └── ConversationCompletenessMetric + KnowledgeRetentionMetric + RoleAdherenceMetric
├── Custom criteria
│   ├── Subjective → GEval (criteria + evaluation_steps)
│   └── Deterministic → DAGMetric (decision tree)
└── Safety
    └── BiasMetric + ToxicityMetric + PIILeakageMetric
```

### Reference-based vs Referenceless

| Type | Requires | Examples |
|------|----------|---------|
| **Reference-based** | Ground truth (`expected_output`, `expected_tools`) | ContextualRecall, ToolCorrectness |
| **Referenceless** | No labeled data needed; ideal for production | AnswerRelevancy, Faithfulness, TaskCompletion |

## Using a Custom LLM Judge

```python
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

azure_openai = AzureOpenAI(model=AzureChatOpenAI(...))
metric = AnswerRelevancyMetric(model=azure_openai)
```

**Requirements for custom LLMs:**
1. Inherit from `DeepEvalBaseLLM`
2. Implement `get_model_name()` — return string
3. Implement `load_model()` — return model object
4. Implement `generate(prompt: str)` — return string
5. Implement `a_generate(prompt: str)` — async version

## Verbose Mode Debugging

```python
metric = AnswerRelevancyMetric(verbose_mode=True)
metric.measure(test_case)
# Prints all intermediate LLM calls and reasoning steps
```

## LLM Configuration (CLI)

```bash
# OpenAI (default)
export OPENAI_API_KEY=<key>

# Azure OpenAI
deepeval set-azure-openai \
    --base-url=<endpoint> \
    --model=<model_name> \
    --deployment-name=<deployment_name> \
    --api-version=<api_version>

# Ollama (local)
deepeval set-ollama --model=deepseek-r1:1.5b

# Gemini
deepeval set-gemini --model=<model_name>
```
