# Conversation and Turn Metrics

## Read This When
- Evaluating multi-turn chatbot conversations for relevancy, faithfulness, or completeness (TurnRelevancyMetric, TurnFaithfulnessMetric, ConversationCompletenessMetric)
- Measuring per-turn RAG quality in conversations (TurnContextualRelevancyMetric, TurnContextualPrecisionMetric, TurnContextualRecallMetric)
- Assessing knowledge retention across turns (KnowledgeRetentionMetric) or building custom multi-turn evaluations (ConversationalGEval, ConversationalDAGMetric)

## Skip This When
- Evaluating single-turn RAG pipelines → `references/03-eval-metrics/20-rag-metrics.md`
- Evaluating agent tool calling or task completion → `references/03-eval-metrics/30-agent-metrics.md`
- Need single-turn custom criteria evaluation (GEval, DAGMetric) → `references/03-eval-metrics/80-custom-metrics.md`

---

DeepEval provides metrics for multi-turn conversations across two levels: **turn-level** metrics evaluate individual turns within a conversation (for RAG chatbots), and **conversation-level** metrics evaluate the entire conversation as a whole.

All conversation metrics use `ConversationalTestCase` (not `LLMTestCase`).

## ConversationalTestCase Structure

```python
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund.", retrieval_context=["..."])
    ],
    chatbot_role="...",         # for RoleAdherenceMetric
    expected_outcome="...",     # for TurnContextualRecall/Precision
    mcp_servers=[...],          # for MCP metrics
)
```

**Turn fields:**

| Field | Type | Description |
|-------|------|-------------|
| `role` | str | `"user"` or `"assistant"` |
| `content` | str | Turn content |
| `retrieval_context` | List[str] | Retrieved context for this turn (turn-level RAG metrics) |
| `tools_called` | List[ToolCall] | Tools called in this turn |
| `mcp_tools_called` | ... | MCP tools called |
| `mcp_resources_called` | ... | MCP resources called |
| `mcp_prompts_called` | ... | MCP prompts called |

**Sliding window:** Turn-level RAG metrics use a `window_size` parameter (default 10) to construct sliding windows of turns for evaluation.

---

## Turn-Level RAG Metrics

These metrics are multi-turn analogs of the single-turn RAG metrics. They evaluate per-turn and average across all assistant turns.

### TurnRelevancyMetric

Evaluates whether the assistant's responses are relevant to the preceding conversational context across all turns.

**Classification:** LLM-as-a-judge, Multi-turn, Referenceless, RAG/Chatbot, Multimodal

**Required ConversationalTestCase fields:**
- `turns` (each turn must have `role` and `content`)

**Formula:**
```
Conversation Relevancy = Number of Turns with Relevant Assistant Content / Total Number of Assistant Turns
```

Uses sliding windows — evaluates whether the assistant's final turn in each window is relevant to preceding context.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `window_size` | int | 10 | Sliding window size |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import TurnRelevancyMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What are your store hours?"),
        Turn(role="assistant", content="We're open Monday through Saturday, 9 AM to 8 PM."),
        Turn(role="user", content="Do you have blue running shoes?"),
        Turn(role="assistant", content="Yes, we carry several models of blue running shoes!")
    ]
)

metric = TurnRelevancyMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### TurnFaithfulnessMetric

Evaluates whether the assistant's responses are factually grounded in the `retrieval_context` provided in each turn.

**Classification:** LLM-as-a-judge, Multi-turn, RAG, Chatbot, Multimodal

**Required ConversationalTestCase fields:**
- `turns` (each turn must have `role`, `content`, and `retrieval_context`)

**Formula:**
```
Turn Faithfulness = Sum of Turn Faithfulness Scores / Total Number of Assistant Turns

Per turn: Faithfulness = Number of Truthful Claims / Total Number of Claims
```

Uses sliding window; extracts truths from retrieval context, generates claims from assistant responses, checks for contradictions.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `truths_extraction_limit` | int | None | Limit truths per document |
| `penalize_ambiguous_claims` | bool | False | Penalize unverifiable claims |
| `window_size` | int | 10 | Sliding window size |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import TurnFaithfulnessMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(
            role="user",
            content="What if these shoes don't fit?"
        ),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
        )
    ]
)

metric = TurnFaithfulnessMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### TurnContextualRelevancyMetric

Evaluates whether the `retrieval_context` provided in each turn contains information relevant to the user's input in that turn.

**Classification:** LLM-as-a-judge, Multi-turn, RAG, Chatbot, Multimodal

**Required ConversationalTestCase fields:**
- `turns` (each turn: `role`, `content`, `retrieval_context`)

**Formula:**
```
Turn Contextual Relevancy = Sum of Turn Contextual Relevancy Scores / Total Number of Assistant Turns

Per window: Contextual Relevancy = Number of Relevant Statements / Total Number of Statements
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `window_size` | int | 10 | Sliding window size |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import TurnContextualRelevancyMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
        )
    ],
    expected_outcome="The chatbot must explain the store policies like refunds, discounts, etc."
)

metric = TurnContextualRelevancyMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### TurnContextualPrecisionMetric

Evaluates whether relevant nodes in the `retrieval_context` are ranked higher than irrelevant nodes throughout the conversation.

**Classification:** LLM-as-a-judge, Multi-turn, RAG, Chatbot, Multimodal

**Required ConversationalTestCase fields:**
- `turns` (each turn: `role`, `content`, `retrieval_context`)
- `expected_outcome`

**Formula:**
```
Turn Contextual Precision = Sum of Turn Contextual Precision Scores / Total Number of Assistant Turns

Per window (Weighted Cumulative Precision):
Contextual Precision = (1 / Num Relevant Nodes) * Sum(k=1 to n) [(Relevant Nodes Up to k / k) * rk]
```

Where `rk` = 1 if node k is relevant, 0 otherwise. Nodes ranked higher (lower index) contribute more weight.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `window_size` | int | 10 | Sliding window size |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import TurnContextualPrecisionMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
        )
    ],
    expected_outcome="The chatbot must explain the store policies like refunds, discounts, etc."
)

metric = TurnContextualPrecisionMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### TurnContextualRecallMetric

Evaluates whether the `retrieval_context` in each turn contains sufficient information to support the `expected_outcome`.

**Classification:** LLM-as-a-judge, Multi-turn, RAG, Chatbot, Multimodal

**Required ConversationalTestCase fields:**
- `turns` (each turn: `role`, `content`, `retrieval_context`)
- `expected_outcome`

**Formula:**
```
Turn Contextual Recall = Sum of Turn Contextual Recall Scores / Total Number of Assistant Turns

Per window: Contextual Recall = Number of Attributable Statements / Total Number of Statements
```

Breaks `expected_outcome` into statements, checks each against retrieval context.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `window_size` | int | 10 | Sliding window size |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import TurnContextualRecallMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
        )
    ],
    expected_outcome="The chatbot must explain the store policies like refunds, discounts, etc."
)

metric = TurnContextualRecallMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

## Conversation-Level Metrics

These metrics evaluate the entire conversation as a whole.

### ConversationCompletenessMetric

Measures whether the chatbot satisfies all user intentions throughout the conversation. Proxy for user satisfaction.

**Classification:** LLM-as-a-judge, Multi-turn, Referenceless

**Required ConversationalTestCase fields:**
- `turns` (`role` and `content` for each)

**Formula:**
```
Conversation Completeness = Number of Satisfied User Intentions / Total Number of User Intentions
```

Extracts all user intentions from `"user"` role turns, then determines if each was satisfied by the assistant.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ConversationCompletenessMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I need to return these shoes and also check if you have my size in blue."),
        Turn(role="assistant", content="For returns, we offer a 30-day refund. Let me check stock for blue shoes in your size.")
    ]
)

metric = ConversationCompletenessMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### KnowledgeRetentionMetric

Evaluates whether the chatbot remembers factual information shared earlier in the conversation. Useful for questionnaire use cases.

**Classification:** LLM-as-a-judge, Multi-turn, Referenceless

**Required ConversationalTestCase fields:**
- `turns` (`role` and `content` for each)

**Formula:**
```
Knowledge Retention = Number of Assistant Turns without Knowledge Attritions / Total Number of Assistant Turns
```

**Process:**
1. LLM extracts knowledge from user-provided content across conversation turns
2. Determines whether each assistant response indicates inability to recall that knowledge

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Maximum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.test_case import Turn, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="My name is John and I wear size 10."),
        Turn(role="assistant", content="Nice to meet you, John! I'll keep that in mind."),
        Turn(role="user", content="Do you have running shoes in my size?"),
        Turn(role="assistant", content="What size are you looking for?")  # FAILS: forgot size 10
    ]
)

metric = KnowledgeRetentionMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### ConversationalGEval

Multi-turn adaptation of `GEval` for evaluating custom criteria across entire conversations. The optimal method for defining custom standards for chatbot conversations.

**Classification:** LLM-as-a-judge, Multi-turn

**Required ConversationalTestCase fields:**
- `turns`
- Additional fields (`retrieval_context`, `tools_called`) if criteria depend on them

**Mandatory metric parameters:**
- `name` — metric identifier (does not affect evaluation)
- `criteria` — evaluation aspect description

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Metric identifier |
| `criteria` | str | required* | Evaluation description (*either criteria or evaluation_steps) |
| `evaluation_params` | List[TurnParams] | [TurnParams.CONTENT] | Turn fields to include in evaluation |
| `evaluation_steps` | List[str] | None | Explicit steps; auto-generated from criteria if omitted |
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `evaluation_template` | class | ConversationalGEvalTemplate | Custom template |

**Algorithm:** Uses chain-of-thought (CoT) to generate `evaluation_steps` from `criteria`, then evaluates the full conversation history using those steps. Score normalized via output token probabilities.

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ConversationalGEval
from deepeval.test_case import Turn, TurnParams, ConversationalTestCase

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Hello, I need help"),
        Turn(role="assistant", content="Of course, how can I assist you today?")
    ]
)

metric = ConversationalGEval(
    name="Professionalism",
    criteria="Determine whether the assistant has acted professionally based on the content.",
    evaluation_params=[TurnParams.CONTENT]
)

evaluate(test_cases=[convo_test_case], metrics=[metric])
print(metric.score, metric.reason)
```

**Warning:** Only include `TurnParams` that are actually mentioned in `criteria`/`evaluation_steps` in `evaluation_params`.

**Available TurnParams:** `CONTENT`, `ROLE`, `RETRIEVAL_CONTEXT`, `TOOLS_CALLED`

**Custom template:**

```python
from deepeval.metrics.conversational_g_eval import ConversationalGEvalTemplate

class CustomConvoGEvalTemplate(ConversationalGEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return f"""Write 3-4 evaluation steps for judging conversations based on {parameters}.
Criteria: {criteria}
Return JSON: {{"steps": ["Step 1", "Step 2", "Step 3"]}}"""

metric = ConversationalGEval(evaluation_template=CustomConvoGEvalTemplate)
```

---

### ConversationalDAGMetric

Multi-turn version of `DAGMetric`. Evaluates conversations via deterministic decision trees. Provides greater control and transparency than `ConversationalGEval`.

**Classification:** LLM-as-a-judge (via DAG), Multi-turn

**Required ConversationalTestCase fields:**
- `turns`

**Mandatory metric parameters:**
- `name` — metric identifier
- `dag` — `DeepAcyclicGraph` with conversational node types

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Metric identifier |
| `dag` | DeepAcyclicGraph | required | Decision tree |
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4.1' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Conversational Node Types:**

| Node | Import | Purpose |
|------|--------|---------|
| `ConversationalTaskNode` | `deepeval.metrics.conversational_dag` | Extract/process data from turns |
| `ConversationalBinaryJudgementNode` | `deepeval.metrics.conversational_dag` | Yes/No decision |
| `ConversationalNonBinaryJudgementNode` | `deepeval.metrics.conversational_dag` | Multiple-choice decision |
| `ConversationalVerdictNode` | `deepeval.metrics.conversational_dag` | Leaf node returning final score |

All conversational nodes support a `turn_window` parameter (tuple of indices) to focus on a specific range of turns.

**Complete example:**

```python
from deepeval import evaluate
from deepeval.metrics import ConversationalDAGMetric
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
    ConversationalVerdictNode,
)
from deepeval.test_case import ConversationalTestCase, Turn, TurnParams

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="what's the weather like today?"),
        Turn(role="assistant", content="Where do you live bro? T~T"),
        Turn(role="user", content="Just tell me the weather in Paris"),
        Turn(role="assistant", content="The weather in Paris today is sunny and 24C."),
        Turn(role="user", content="Should I take an umbrella?"),
        Turn(role="assistant", content="You trying to be stylish? I don't recommend it."),
    ]
)

# Define DAG
non_binary_node = ConversationalNonBinaryJudgementNode(
    criteria="How was the assistant's behaviour towards user?",
    children=[
        ConversationalVerdictNode(verdict="Rude", score=0),
        ConversationalVerdictNode(verdict="Neutral", score=5),
        ConversationalVerdictNode(verdict="Playful", score=10),
    ],
)

binary_node = ConversationalBinaryJudgementNode(
    criteria="Do the assistant's replies satisfy user's questions?",
    children=[
        ConversationalVerdictNode(verdict=False, score=0),
        ConversationalVerdictNode(verdict=True, child=non_binary_node),
    ],
)

task_node = ConversationalTaskNode(
    instructions="Summarize the conversation and explain the assistant's behaviour overall.",
    output_label="Summary",
    evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
    children=[binary_node],
)

dag = DeepAcyclicGraph(root_nodes=[task_node])
metric = ConversationalDAGMetric(name="Conversation Quality", dag=dag)

evaluate([test_case], [metric])
```

---

## Conversation Metrics Quick Reference

| Metric | Level | Requires `expected_outcome` | Requires `retrieval_context` in turns |
|--------|-------|---------------------------|--------------------------------------|
| `TurnRelevancyMetric` | Turn | No | No |
| `TurnFaithfulnessMetric` | Turn | No | Yes |
| `TurnContextualRelevancyMetric` | Turn | No | Yes |
| `TurnContextualPrecisionMetric` | Turn | Yes | Yes |
| `TurnContextualRecallMetric` | Turn | Yes | Yes |
| `ConversationCompletenessMetric` | Conversation | No | No |
| `KnowledgeRetentionMetric` | Conversation | No | No |
| `ConversationalGEval` | Conversation | No | Optional |
| `ConversationalDAGMetric` | Conversation | No | Optional |
