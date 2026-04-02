# Agent Metrics

## Read This When
- Evaluating AI agent tool selection and argument generation (ToolCorrectnessMetric, ArgumentCorrectnessMetric)
- Measuring agent task completion, step efficiency, or plan quality (TaskCompletionMetric, StepEfficiencyMetric, PlanQualityMetric, PlanAdherenceMetric)
- Need MCP-specific metrics (MCPUseMetric, MultiTurnMCPUseMetric, MCPTaskCompletionMetric) or multi-turn agent metrics (GoalAccuracyMetric, ToolUseMetric)

## Skip This When
- Evaluating RAG retrieval or generation quality → `references/03-eval-metrics/20-rag-metrics.md`
- Evaluating multi-turn chatbot conversations without tool usage → `references/03-eval-metrics/50-conversation-turn-metrics.md`
- Need safety checks like bias, toxicity, or PII leakage → `references/03-eval-metrics/40-safety-metrics.md`

---

DeepEval provides comprehensive metrics for evaluating AI agents across three layers: **Reasoning** (planning), **Action** (tool calling), and **Execution** (overall task completion). Most agent metrics require tracing to analyze the full execution trace.

## Three Layers of Agent Evaluation

| Layer | What It Assesses | Key Metrics |
|-------|-----------------|-------------|
| **Reasoning** | Plans tasks, creates strategies | `PlanQualityMetric`, `PlanAdherenceMetric` |
| **Action** | Selects tools, generates arguments | `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric` |
| **Execution** | Orchestrates full loop, completes objectives | `TaskCompletionMetric`, `StepEfficiencyMetric` |

## Tracing Setup (Required for Trace-Only Metrics)

Most agent metrics analyze the agent's full execution trace via the `@observe` decorator:

```python
from deepeval.tracing import observe, update_current_trace, update_current_span
from deepeval.dataset import Golden, EvaluationDataset

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="agent")
def travel_agent(user_input):
    flights = search_flights("NYC", "Paris", "2025-03-15")
    cheapest = min(flights, key=lambda x: x["price"])
    return f"Cheapest flight: {cheapest['id']} at ${cheapest['price']}"
```

Trace-only metrics are evaluated via `evals_iterator`:

```python
from deepeval.metrics import TaskCompletionMetric

metric = TaskCompletionMetric(threshold=0.7)
dataset = EvaluationDataset(goldens=[Golden(input="Find cheapest flight to Paris")])

for golden in dataset.evals_iterator(metrics=[metric]):
    travel_agent(golden.input)
```

---

## Execution Layer Metrics

### TaskCompletionMetric

Evaluates whether the agent successfully accomplishes the intended task. This is the top-level success indicator — analyzes the full agent trace.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, Agent (trace-only), Multimodal

**Requires:** Agent tracing (`@observe` decorator)

**Formula:**
```
Task Completion Score = AlignmentScore(Task, Outcome)
```

Task and Outcome are both extracted from the trace by an LLM. Score of 1 = complete fulfillment; lower = partial or failed completion.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `task` | str | None | Task description; auto-inferred from trace if not supplied |
| `model` | str | 'gpt-4o' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

@observe()
def trip_planner_agent(input):
    @observe()
    def itinerary_generator(destination, days):
        return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]
    return itinerary_generator("Paris", 2)

task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")
dataset = EvaluationDataset(goldens=[Golden(input="Plan a 2-day trip to Paris")])

for golden in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)
```

---

### StepEfficiencyMetric

Evaluates whether the agent completes tasks without unnecessary or redundant steps. Complements `TaskCompletionMetric` — a high task completion score with low step efficiency means the agent works but wastes resources.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, Agent (trace-only), Multimodal

**Requires:** Agent tracing

**Formula:**
```
Step Efficiency Score = AlignmentScore(Task, Execution Steps)
```

Penalizes redundant tool calls, unnecessary reasoning loops, and any actions not strictly required.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4o' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval.metrics import StepEfficiencyMetric

@observe(type="agent")
def inefficient_agent(user_input):
    flights1 = search_flights("NYC", "LA", "2025-03-15")
    flights2 = search_flights("NYC", "LA", "2025-03-15")  # Redundant!
    cheapest = min(flights1, key=lambda x: x["price"])
    return book_flight(cheapest["id"])

step_efficiency = StepEfficiencyMetric(threshold=0.7, model="gpt-4o")
dataset = EvaluationDataset(goldens=[Golden(input="Book cheapest flight NYC to LA")])

for golden in dataset.evals_iterator(metrics=[step_efficiency]):
    inefficient_agent(golden.input)
# This will score low due to the redundant search_flights call
```

---

## Reasoning Layer Metrics

### PlanQualityMetric

Evaluates whether the plan the agent generates is logical, complete, and efficient. Extracts task and plan from the agent's trace, then judges plan quality with an LLM.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, Agent (trace-only), Multimodal

**Requires:** Agent tracing with visible reasoning/planning steps

**Formula:**
```
Plan Quality Score = AlignmentScore(Task, Plan)
```

**Important:** If no plan is detectable in the trace, the metric automatically passes with a score of 1 (the agent simply acts without explicit planning).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4o' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval.tracing import observe, update_current_trace
from deepeval.metrics import PlanQualityMetric
from deepeval.test_case import ToolCall

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}]

@observe(type="agent")
def agent(input):
    tools = search_flights("NYC", "Paris", "2025-03-15")
    update_current_trace(input=input, output="Done", tools_called=tools)
    return "Result"

metric = PlanQualityMetric(threshold=0.7, model="gpt-4o")
dataset = EvaluationDataset(goldens=[Golden(input="Find cheapest flight to Paris")])

for golden in dataset.evals_iterator(metrics=[metric]):
    agent(golden.input)
```

---

### PlanAdherenceMetric

Evaluates whether the agent follows its own plan during execution. Extracts the task, plan, and execution steps from the trace, then measures alignment.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, Agent (trace-only), Multimodal

**Requires:** Agent tracing

**Formula:**
```
Plan Adherence Score = AlignmentScore((Task, Plan), Execution Steps)
```

**Process:**
1. Extract task from trace (user goal/intent)
2. Extract plan from agent's thinking/reasoning (passes with score 1 if no clear plan)
3. Analyze execution steps against the plan
4. LLM generates final alignment score

**Parameters:** Same as `PlanQualityMetric` above.

**Pro tip:** Combine `PlanQualityMetric` and `PlanAdherenceMetric` — a high-quality plan that is ignored is as problematic as a poor plan that is followed perfectly.

---

## Action Layer Metrics

### ToolCorrectnessMetric

Evaluates whether the agent selects the right tools and calls them correctly. Compares `tools_called` against `expected_tools`. Uses both deterministic and LLM-based evaluation.

**Classification:** Deterministic + LLM-as-a-judge, Single-turn, Reference-based, Agent, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `tools_called`
- `expected_tools`

**Formula:**
```
Step 1 (Deterministic):
  Tool Correctness = Number of Correctly Used Tools / Total Number of Tools Called

Step 2 (LLM-based, if available_tools provided):
  Checks whether selected tools were optimal

Final Score = min(deterministic_score, llm_score)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `available_tools` | List[ToolCall] | None | All tools available to the agent (enables LLM optimization check) |
| `threshold` | float | 0.5 | Minimum passing threshold |
| `evaluation_params` | List[ToolCallParams] | [] | Strictness: add `INPUT_PARAMETERS`, `OUTPUT` for deeper checking |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `verbose_mode` | bool | False | Print intermediate steps |
| `should_consider_ordering` | bool | False | Enforce call sequence order |
| `should_exact_match` | bool | False | Require exact tool set match (overrides ordering) |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    tools_called=[ToolCall(name="WebSearch"), ToolCall(name="ToolQuery")],
    expected_tools=[ToolCall(name="WebSearch")]
)

metric = ToolCorrectnessMetric()
evaluate(test_cases=[test_case], metrics=[metric])
```

**Component-level usage (attaching to LLM span):**

```python
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import get_current_golden

tool_correctness = ToolCorrectnessMetric(threshold=0.7)

@observe(type="llm", metrics=[tool_correctness])
def call_llm(messages):
    result = get_weather("Paris")
    update_current_span(
        input=messages[-1]["content"],
        output=f"Weather: {result}",
        expected_tools=get_current_golden().expected_tools
    )
    return result
```

---

### ArgumentCorrectnessMetric

Evaluates whether the agent generates correct arguments for each tool call. Fully LLM-based and referenceless — judges argument correctness based on input context, not against expected values.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, Agent, Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `tools_called`

**Formula:**
```
Argument Correctness = Number of Correctly Generated Input Parameters / Total Number of Tool Calls
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

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

metric = ArgumentCorrectnessMetric(threshold=0.7, model="gpt-4", include_reason=True)

test_case = LLMTestCase(
    input="When did Trump first raise tariffs?",
    actual_output="Trump first raised tariffs in 2018.",
    tools_called=[
        ToolCall(
            name="WebSearch Tool",
            description="Tool to search for information on the web.",
            input={"search_query": "Trump first raised tariffs year"}
        ),
        ToolCall(
            name="History FunFact Tool",
            description="Tool to provide a fun fact about the topic.",
            input={"topic": "Trump tariffs"}
        )
    ]
)

evaluate(test_cases=[test_case], metrics=[metric])
```

---

## Multi-Turn Agent Metrics

### ToolUseMetric

Evaluates an LLM agent's tool selection and argument generation within multi-turn conversations. Uses `ConversationalTestCase`.

**Classification:** LLM-as-a-judge, Multi-turn, Referenceless, Agent, Multimodal

**Required:** `turns` in a `ConversationalTestCase`, with `available_tools` parameter on the metric

**Formula:**
```
Tool Use Score = min(Tool Selection Score, Argument Correctness Score)
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `available_tools` | List[ToolCall] | Yes | N/A | Tools available to the agent |
| `threshold` | float | No | 0.5 | Minimum passing threshold |
| `model` | str | No | 'gpt-4o' | Judge LLM |
| `include_reason` | bool | No | True | Include reasoning |
| `strict_mode` | bool | No | False | Binary 0/1 scoring |
| `async_mode` | bool | No | True | Concurrent execution |
| `verbose_mode` | bool | No | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ToolUseMetric
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the weather in Paris?"),
        Turn(role="assistant", content="Let me check.", tools_called=[ToolCall(name="get_weather")])
    ]
)

metric = ToolUseMetric(
    available_tools=[ToolCall(name="get_weather"), ToolCall(name="search_web")],
    threshold=0.5
)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### GoalAccuracyMetric

Evaluates planning and execution accuracy for achieving goals across multi-turn agent interactions.

**Classification:** LLM-as-a-judge, Multi-turn, Referenceless, Agent, Multimodal

**Required:** `turns` in a `ConversationalTestCase`

**Formula:**
```
Goal Accuracy Score = (Goal Accuracy Score + Plan Evaluation Score) / 2
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4o' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import GoalAccuracyMetric
from deepeval.test_case import Turn, ConversationalTestCase, ToolCall

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Book me a flight to Paris"),
        Turn(role="assistant", content="I'll search for flights.", tools_called=[ToolCall(name="search_flights")])
    ]
)

metric = GoalAccuracyMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

## MCP (Model Context Protocol) Metrics

### MCPUseMetric

Evaluates how effectively an MCP-based agent utilizes available MCP servers in a single-turn interaction.

**Classification:** LLM-as-a-judge, Single-turn, Referenceless, Agent (MCP), Multimodal

**Required LLMTestCase fields:**
- `input`
- `actual_output`
- `mcp_servers`

**Optional fields:** `mcp_tools_called`, `mcp_resources_called`, `mcp_prompts_called`

**Formula:**
```
MCP Use Score = AlignmentScore(Primitives Used, Primitives Available)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `model` | str | 'gpt-4o' | Judge LLM |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import MCPUseMetric
from deepeval.test_case import LLMTestCase, MCPServer

test_case = LLMTestCase(
    input="Search for recent news",
    actual_output="Here are the latest news items...",
    mcp_servers=[MCPServer(...)]
)

metric = MCPUseMetric()
evaluate([test_case], [metric])
```

---

### MultiTurnMCPUseMetric

Evaluates MCP server utilization across a multi-turn conversation.

**Classification:** LLM-as-a-judge, Multi-turn, Referenceless, Agent (MCP), Multimodal

**Required:** `turns` and `mcp_servers` in a `ConversationalTestCase`

**Formula:**
```
MCP Use Score = AlignmentScore(Primitives Used, Primitives Available) / Total Number of MCP Interactions
```

**Parameters:** Same as `MCPUseMetric` above.

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import MultiTurnMCPUseMetric
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Find me news about AI"),
        Turn(role="assistant", content="Here is what I found...")
    ],
    mcp_servers=[MCPServer(...)]
)

metric = MultiTurnMCPUseMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

### MCPTaskCompletionMetric

Evaluates how effectively an MCP-based agent accomplishes tasks in a multi-turn conversation.

**Classification:** LLM-as-a-judge, Multi-turn (Conversational), Referenceless, Agent (MCP), Multimodal

**Required:** `turns` and `mcp_servers` in a `ConversationalTestCase`

**Formula:**
```
MCP Task Completeness = Number of Tasks Satisfied in Each Interaction / Total Number of Interactions
```

**Parameters:** Same as `MCPUseMetric` above.

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import MCPTaskCompletionMetric
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer

convo_test_case = ConversationalTestCase(
    turns=[Turn(role="user", content="..."), Turn(role="assistant", content="...")],
    mcp_servers=[MCPServer(...)]
)

metric = MCPTaskCompletionMetric(threshold=0.5)
evaluate(test_cases=[convo_test_case], metrics=[metric])
```

---

## Complete End-to-End Agent Evaluation

```python
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset, get_current_golden
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import (
    TaskCompletionMetric,
    StepEfficiencyMetric,
    PlanQualityMetric,
    PlanAdherenceMetric,
    ToolCorrectnessMetric,
    ArgumentCorrectnessMetric
)

# End-to-end metrics (need full trace)
task_completion = TaskCompletionMetric()
step_efficiency = StepEfficiencyMetric()
plan_quality = PlanQualityMetric()
plan_adherence = PlanAdherenceMetric()

# Component-level metrics (attached to specific LLM span)
tool_correctness = ToolCorrectnessMetric()
argument_correctness = ArgumentCorrectnessMetric()

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_llm(user_input):
    origin, destination, date = "NYC", "Paris", "2025-03-18"
    flights = search_flights(origin, destination, date)
    cheapest = min(flights, key=lambda x: x["price"])
    booking = book_flight(cheapest["id"])
    update_current_span(
        input=user_input,
        output=f"Booked {cheapest['id']}",
        expected_tools=get_current_golden().expected_tools
    )
    return booking

@observe(type="agent")
def travel_agent(user_input):
    booking = call_llm(user_input)
    return f"Confirmation: {booking['confirmation']}"

dataset = EvaluationDataset(goldens=[
    Golden(
        input="Book a flight from NYC to Paris for next Tuesday",
        expected_tools=[ToolCall(name="search_flights"), ToolCall(name="book_flight")]
    )
])

# End-to-end metrics go in evals_iterator
for golden in dataset.evals_iterator(
    metrics=[task_completion, step_efficiency, plan_quality, plan_adherence]
):
    travel_agent(golden.input)
```

## Choosing Agent Metrics

| If Your Agent... | Prioritize |
|-----------------|------------|
| Uses explicit planning/reasoning | `PlanQualityMetric`, `PlanAdherenceMetric` |
| Calls multiple tools | `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric` |
| Has complex multi-step workflows | `StepEfficiencyMetric`, `TaskCompletionMetric` |
| Runs in production (cost-sensitive) | `StepEfficiencyMetric` |
| Is task-critical (must succeed) | `TaskCompletionMetric` |
| Uses MCP servers | `MCPUseMetric`, `MCPTaskCompletionMetric` |
| Is a multi-turn agent | `GoalAccuracyMetric`, `ToolUseMetric` |
