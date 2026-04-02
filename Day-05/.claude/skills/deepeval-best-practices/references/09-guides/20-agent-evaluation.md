# AI Agent Evaluation — Reference Guide

Combined from: guides-ai-agent-evaluation.md + guides-ai-agent-evaluation-metrics.md
Sources: https://deepeval.com/guides/guides-ai-agent-evaluation | https://deepeval.com/guides/guides-ai-agent-evaluation-metrics

## Read This When
- Need to set up evaluation for an AI agent across reasoning, action, and execution layers
- Want to understand which agent metrics to use (PlanQuality, ToolCorrectness, TaskCompletion, etc.) and how to apply them with `@observe` tracing
- Comparing development benchmarking vs. production monitoring workflows for agents

## Skip This When
- Need API-level parameter details for agent metrics -- see `references/03-eval-metrics/30-agent-metrics.md`
- Looking for a complete worked example building and evaluating a medical chatbot agent -- see `references/10-tutorials/10-medical-chatbot.md`
- Want to evaluate a RAG pipeline rather than an autonomous agent -- see `references/09-guides/10-rag-evaluation.md`

---

# Part 1: AI Agent Evaluation

## Overview

AI agent evaluation measures how well an agent reasons, selects tools, and completes tasks across distinct layers to identify failures at the component level. An AI agent is defined as "an LLM-powered system that autonomously reasons about tasks, creates plans, and executes actions using external tools to accomplish user goals."

## Core Architecture

Agents operate through two interconnected layers:

- **Reasoning Layer**: Powered by LLMs, handles planning and decision-making
- **Action Layer**: Powered by tools and function calling, executes actions in the real world

These layers work iteratively until task completion.

## Common Pitfalls in AI Agent Pipelines

### Reasoning Layer Issues

The reasoning layer is responsible for:
1. Understanding user intent by analyzing input
2. Decomposing complex tasks into manageable sub-tasks
3. Creating coherent strategies
4. Deciding which tools to use and in what order

Quality factors include:
- **LLM choice**: Larger models like `gpt-4o` or `claude-3.5-sonnet` reason better
- **Prompt template**: System prompts heavily influence task approach
- **Temperature**: Lower values produce deterministic reasoning; higher values enable creativity

Key evaluation questions:
- Is the agent creating effective, logical, and complete plans?
- Are plans appropriately scoped?
- Does the plan account for task dependencies?
- Does the agent follow its own plan during execution?

### Action Layer Issues

The action layer involves:
1. Selecting the right tool from available options
2. Generating correct arguments for tool calls
3. Calling tools in correct sequence
4. Processing tool outputs back to reasoning layer

Quality factors include:
- **Available tools**: Too many confuse the LLM; too few leave gaps
- **Tool descriptions**: Clear descriptions aid correct selection
- **Tool schemas**: Well-defined input/output schemas help argument generation
- **Tool naming**: Intuitive names facilitate proper selection

Critical evaluation questions:
- Is the agent selecting correct tools for each sub-task?
- Are the right number of tools being called?
- Are tools called in the correct order?
- Are arguments supplied correctly?
- Are values extracted accurately from context?
- Are tool descriptions sufficiently clear?

### Overall Execution

The agentic loop orchestrates reasoning and action layers iteratively:

1. Orchestrating the reasoning-action loop
2. Handling errors and edge cases gracefully
3. Iterating until task completion or determining impossibility

Evaluation focuses on:
- Did the agent complete the task?
- Is execution efficient without unnecessary steps?
- Does the agent handle failures appropriately?
- Does the agent stay focused on the original request?

## Agent Evals In Development

Development evaluation benchmarks agents using datasets and metrics, comparing different iterations on the same golden dataset. Helps answer:
- Which agent version performs best?
- How will prompt changes affect success?
- Do new tools help or hurt performance?
- Where specifically is the agent failing?

### LLM Tracing Setup

The `@observe` decorator traces execution without adding latency:

```python
import json
from openai import OpenAI
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset

client = OpenAI()
tools = [...]

@observe(type="tool")
def search_flights(origin, destination, date):
    # Simulated flight search
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    # Simulated booking
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="llm")
def call_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    return response

@observe(type="agent")
def travel_agent(user_input):
    messages = [{"role": "user", "content": user_input}]
    # LLM reasons about which tool to call
    response = call_openai(messages)
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    # Execute the tool
    flights = search_flights(args["origin"], args["destination"], args["date"])
    # LLM decides to book the cheapest
    cheapest = min(flights, key=lambda x: x["price"])
    messages.append({"role": "assistant", "content": f"Found flights. Booking cheapest: {cheapest['id']}"})
    booking = book_flight(cheapest["id"])
    return f"Booked flight {cheapest['id']} for ${cheapest['price']}. Confirmation: {booking['confirmation']}"
```

OpenAI Tools Schema:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"}
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a specific flight by ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"}
                },
                "required": ["flight_id"]
            }
        }
    }
]
```

### Evaluating the Reasoning Layer

Two metrics evaluate reasoning and planning:

- **PlanQualityMetric**: Assesses whether plans are logical, complete, and efficient
- **PlanAdherenceMetric**: Evaluates whether agents follow their own plans

Both metrics are required for comprehensive reasoning evaluation.

```python
from deepeval.metrics import PlanQualityMetric, PlanAdherenceMetric

plan_quality = PlanQualityMetric()
plan_adherence = PlanAdherenceMetric()

from deepeval.dataset import EvaluationDataset, Golden

# Create dataset
dataset = EvaluationDataset(goldens=[
    Golden(input="Book a flight from NYC to London for next Monday")])

# Loop through dataset with metrics
for golden in dataset.evals_iterator(metrics=[plan_quality, plan_adherence]):
    travel_agent(golden.input)
```

All metrics allow setting passing thresholds, enabling strict mode, including reasoning explanations, and using any LLM for evaluation.

### Evaluating the Action Layer

Two metrics evaluate tool calling ability:

- **ToolCorrectnessMetric**: Assesses correct tool selection and expected calling patterns
- **ArgumentCorrectnessMetric**: Evaluates correct argument generation for tool calls

These are component-level metrics attached to the LLM component where tool decisions occur:

```python
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric

tool_correctness = ToolCorrectnessMetric()
argument_correctness = ArgumentCorrectnessMetric()

# Add metrics to @observe on LLM component
@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_openai(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    return response

from deepeval.dataset import EvaluationDataset, Golden

# Create dataset
dataset = EvaluationDataset(goldens=[
    Golden(input="What's the weather like in San Francisco and should I bring an umbrella?")])

# Evaluate with action layer metrics
for golden in dataset.evals_iterator():
    weather_agent(golden.input)
```

ToolCorrectnessMetric supports configurable strictness via `evaluation_params`, defaulting to name-only comparison but optionally requiring input parameters and outputs to match.

### Evaluating Overall Execution

Two metrics evaluate overall execution quality:

- **TaskCompletionMetric**: Determines if agents successfully accomplish intended tasks
- **StepEfficiencyMetric**: Evaluates whether agents complete tasks efficiently without redundant steps

```python
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric

task_completion = TaskCompletionMetric()
step_efficiency = StepEfficiencyMetric()

from deepeval.dataset import EvaluationDataset, Golden

# Create dataset
dataset = EvaluationDataset(goldens=[
    Golden(input="Book the cheapest flight from NYC to LA for tomorrow")])

# Evaluate with execution metrics
for golden in dataset.evals_iterator(metrics=[task_completion, step_efficiency]):
    travel_agent(golden.input)
```

Both are trace-only metrics requiring use with `evals_iterator` or `@observe` decorator.

## Agent Evals In Production

Production evaluation shifts from benchmarking to continuous performance monitoring with asynchronous execution, minimal resource overhead, and trend tracking over time.

### Create a Metric Collection

Log into Confident AI and create a metric collection containing metrics for production evaluation.

### Reference the Collection

```python
# Reference Confident AI metric collection by name
@observe(metric_collection="my-agent-metrics")
def call_openai(messages):
    ...
```

Traces automatically export to Confident AI in OpenTelemetry fashion. Confident AI evaluates traces asynchronously using the metric collection and stores results for analysis.

Run `deepeval login` in the terminal and follow the Confident AI LLM tracing setup guide to get started.

## End-to-End vs Component-Level Evals

**End-to-end evals** analyze the entire agent trace from start to finish (reasoning layer, execution metrics passed to `evals_iterator(metrics=[...])`).

**Component-level evals** evaluate specific components in isolation (action layer metrics attached to `@observe` via `@observe(metrics=[...])`).

| Metric Type | Scope | Why |
|---|---|---|
| Reasoning & Execution | End-to-end | Need full trace to assess planning and task completion |
| Action Layer | Component-level | Tool decisions happen at LLM component |

## Using Custom Evals

For use-case-specific evaluation, `GEval` uses LLM-as-a-judge based on custom criteria defined in plain English.

### In Development

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# Define custom metric
reasoning_clarity = GEval(
    name="Reasoning Clarity",
    criteria="Evaluate how clearly the agent explains its reasoning and decision-making process before taking actions.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

# Use end-to-end
for golden in dataset.evals_iterator(metrics=[reasoning_clarity]):
    travel_agent(golden.input)

# Or component-level
@observe(type="llm", metrics=[reasoning_clarity])
def call_openai(messages):
    ...
```

### In Production

Define custom G-Eval metrics on Confident AI and reference via `metric_collection`:

```python
# Custom metrics defined on Confident AI
@observe(metric_collection="my-custom-agent-metrics")
def call_openai(messages):
    ...
```

## Metric Summary Table

| Scope | Use Case | Example Metrics |
|---|---|---|
| End-to-end | Evaluate full agent trace | PlanQualityMetric, TaskCompletionMetric |
| Component-level | Evaluate specific components | ToolCorrectnessMetric, ArgumentCorrectnessMetric |

## Development vs Production

**Development**: Benchmark and compare agent iterations using datasets with locally defined metrics

**Production**: Export traces to Confident AI and evaluate asynchronously to monitor performance over time

## Conclusion

Agents can fail at multiple layers:
- Reasoning layer: poor planning, ignored dependencies, plan deviation
- Action layer: wrong tool selection, incorrect arguments, bad call ordering
- Overall execution: incomplete tasks, inefficient steps, going off-task

DeepEval provides metrics applicable at different scopes to catch these issues systematically.

## Next Steps and Additional Resources

1. Login to Confident AI via `deepeval login`
2. Explore metrics in the [AI Agent Evaluation Metrics guide](/guides/guides-ai-agent-evaluation-metrics)
3. Read [AI Agent Evaluation: The Definitive Guide](https://www.confident-ai.com/blog/definitive-ai-agent-evaluation-guide)
4. Join the [DeepEval Discord](https://discord.com/invite/a3K9c8GRGt)

## Related Documentation Links

- [AI Agent Evaluation Metrics](/guides/guides-ai-agent-evaluation-metrics)
- [PlanQualityMetric](/docs/metrics-plan-quality)
- [PlanAdherenceMetric](/docs/metrics-plan-adherence)
- [ToolCorrectnessMetric](/docs/metrics-tool-correctness)
- [ArgumentCorrectnessMetric](/docs/metrics-argument-correctness)
- [TaskCompletionMetric](/docs/metrics-task-completion)
- [StepEfficiencyMetric](/docs/metrics-step-efficiency)
- [GEval](/docs/metrics-llm-evals)
- [DAGMetric](/docs/metrics-dag)
- [End-to-End Evals](/docs/evaluation-end-to-end-llm-evals)
- [Component-Level Evals](/docs/evaluation-component-level-llm-evals)
- [Evaluation Datasets](/docs/evaluation-datasets)
- [LLM Tracing](/docs/evaluation-llm-tracing)
- [Confident AI Docs](https://www.confident-ai.com/docs)

---

# Part 2: AI Agent Evaluation Metrics

## Overview

"AI agent evaluation metrics are purpose-built measurements that assess how well autonomous LLM systems reason, plan, execute tools, and complete tasks." Unlike traditional LLM metrics evaluating single input-output pairs, these metrics analyze the entire execution trace, capturing every reasoning step, tool call, and intermediate decision.

These metrics address fundamental differences in how AI agents fail compared to simple LLM applications -- an agent might select the correct tool but pass wrong arguments, create a brilliant plan but fail following it, or complete tasks while wasting resources on redundant steps.

**Key Requirement:** "AI agent evaluation metrics in deepeval operate on execution traces -- the full record of your agent's reasoning and actions. This requires setting up tracing to capture your agent's behavior."

---

## The Three Layers of AI Agent Evaluation

| Layer | What It Does | Key Metrics |
|-------|-------------|------------|
| **Reasoning Layer** | Plans tasks, creates strategies, decides what to do | `PlanQualityMetric`, `PlanAdherenceMetric` |
| **Action Layer** | Selects tools, generates arguments, executes calls | `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric` |
| **Execution Layer** | Orchestrates the full loop, completes objectives | `TaskCompletionMetric`, `StepEfficiencyMetric` |

---

## Reasoning Layer Metrics

### Plan Quality Metric

**Purpose:** The `PlanQualityMetric` evaluates whether "the plan your agent generates is logical, complete, and efficient for accomplishing the given task." It extracts the task and plan from your agent's trace and uses an LLM judge to assess plan quality.

**Code Example:**
```python
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanQualityMetric

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="agent")
def travel_agent(user_input):
    # Agent reasons: "I need to search for flights first, then book the cheapest"
    flights = search_flights("NYC", "Paris", "2025-03-15")
    cheapest = min(flights, key=lambda x: x["price"])
    return f"Found cheapest flight: {cheapest['id']} for ${cheapest['price']}"

# Initialize metric
plan_quality = PlanQualityMetric(threshold=0.7, model="gpt-4o")

# Evaluate agent with plan quality metric
dataset = EvaluationDataset(goldens=[Golden(input="Find me the cheapest flight to Paris")])
for golden in dataset.evals_iterator(metrics=[plan_quality]):
    travel_agent(golden.input)
```

**When to Use:** "Use PlanQualityMetric when your agent explicitly reasons about how to approach a task before taking action. This is common in agents that use chain-of-thought prompting or expose their planning process."

**Calculation Method:**
```
Plan Quality Score = AlignmentScore(Task, Plan)
```

**Important Note:** "If no plan is detectable in the trace -- meaning the agent doesn't explicitly reason about its approach -- the metric passes with a score of 1 by default."

**Reference:** [Full Plan Quality documentation](/docs/metrics-plan-quality)

---

### Plan Adherence Metric

**Purpose:** The `PlanAdherenceMetric` evaluates whether "your agent follows its own plan during execution. Creating a good plan is only half the battle -- an agent that deviates from its strategy mid-execution undermines its own reasoning."

**Code Example:**
```python
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanAdherenceMetric

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="agent")
def travel_agent(user_input):
    # Plan: 1) Search flights, 2) Book the cheapest one
    flights = search_flights("NYC", "Paris", "2025-03-15")
    cheapest = min(flights, key=lambda x: x["price"])
    booking = book_flight(cheapest["id"])
    return f"Booked flight {cheapest['id']}. Confirmation: {booking['confirmation']}"

# Initialize metric
plan_adherence = PlanAdherenceMetric(threshold=0.7, model="gpt-4o")

# Evaluate whether agent followed its plan
dataset = EvaluationDataset(goldens=[Golden(input="Book the cheapest flight to Paris")])
for golden in dataset.evals_iterator(metrics=[plan_adherence]):
    travel_agent(golden.input)
```

**When to Use:** "Use PlanAdherenceMetric alongside PlanQualityMetric when evaluating agents with explicit planning phases. If your agent creates multi-step plans, this metric ensures it actually follows through."

**Calculation Method:**
```
Plan Adherence Score = AlignmentScore((Task, Plan), Execution Steps)
```

The metric extracts the task, plan, and actual execution steps from the trace, then uses an LLM to evaluate how faithfully the agent adhered to its stated plan.

**Pro Tip:** "Combine PlanQualityMetric and PlanAdherenceMetric together -- a high-quality plan that's ignored is as problematic as a poor plan that's followed perfectly."

**Reference:** [Full Plan Adherence documentation](/docs/metrics-plan-adherence)

---

## Action Layer Metrics

### Tool Correctness Metric

**Purpose:** The `ToolCorrectnessMetric` evaluates whether "your agent selects the right tools and calls them correctly. It compares the tools your agent actually called against a list of expected tools."

**Code Example:**
```python
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset, get_current_golden
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

# Initialize metric
tool_correctness = ToolCorrectnessMetric(threshold=0.7)

@observe(type="tool")
def get_weather(city):
    return {"temp": "22C", "condition": "sunny"}

# Attach metric to the LLM component where tool decisions are made
@observe(type="llm", metrics=[tool_correctness])
def call_llm(messages):
    # LLM decides to call get_weather tool
    result = get_weather("Paris")
    # Update span with tool calling information for evaluation
    update_current_span(
        input=messages[-1]["content"],
        output=f"The weather is {result['condition']}, {result['temp']}",
        expected_tools=get_current_golden().expected_tools
    )
    return result

@observe(type="agent")
def weather_agent(user_input):
    return call_llm([{"role": "user", "content": user_input}])

# Evaluate
dataset = EvaluationDataset(goldens=[Golden(input="What's the weather in Paris?", expected_tools=[ToolCall(name="get_weather")])])
for golden in dataset.evals_iterator():
    weather_agent(golden.input)
```

**When to Use:** "Use ToolCorrectnessMetric when you have deterministic expectations about which tools should be called for a given task. It's particularly valuable for testing tool selection logic and identifying unnecessary tool calls."

**Calculation Method:**
```
Tool Correctness = Number of Correctly Used Tools / Total Number of Tools Called
```

**Strictness Levels:**
- **Tool name matching** (default) -- considers a call correct if the tool name matches
- **Input parameter matching** -- also requires input arguments to match
- **Output matching** -- additionally requires outputs to match
- **Ordering consideration** -- optionally enforces call sequence
- **Exact matching** -- requires tools_called and expected_tools to be identical

**Caution:** "When available_tools is provided, the metric also uses an LLM to evaluate whether your tool selection was optimal given all available options. The final score is the minimum of the deterministic and LLM-based scores."

**Reference:** [Full Tool Correctness documentation](/docs/metrics-tool-correctness)

---

### Argument Correctness Metric

**Purpose:** The `ArgumentCorrectnessMetric` evaluates whether "your agent generates correct arguments for each tool call. Selecting the right tool with wrong arguments is as problematic as selecting the wrong tool entirely."

**Code Example:**
```python
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

# Initialize metric
argument_correctness = ArgumentCorrectnessMetric(threshold=0.7, model="gpt-4o")

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

# Attach metric to the LLM component where arguments are generated
@observe(type="llm", metrics=[argument_correctness])
def call_llm(user_input):
    # LLM generates arguments for tool call
    origin, destination, date = "NYC", "London", "2025-03-15"
    flights = search_flights(origin, destination, date)
    # Update span with tool calling details for evaluation
    update_current_span(
        input=user_input,
        output=f"Found {len(flights)} flights",
    )
    return flights

@observe(type="agent")
def flight_agent(user_input):
    return call_llm(user_input)

# Evaluate - metric checks if arguments match what input requested
dataset = EvaluationDataset(goldens=[
    Golden(input="Search for flights from NYC to London on March 15th")])
for golden in dataset.evals_iterator():
    flight_agent(golden.input)
```

**When to Use:** "Use ArgumentCorrectnessMetric when correct argument values are critical for task success. This is especially important for agents that interact with APIs, databases, or external services where incorrect arguments cause failures."

**Calculation Method:**
```
Argument Correctness = Number of Correctly Generated Input Parameters / Total Number of Tool Calls
```

**Key Advantage:** "Unlike ToolCorrectnessMetric, this metric is fully LLM-based and referenceless -- it evaluates argument correctness based on the input context rather than comparing against expected values."

**Reference:** [Full Argument Correctness documentation](/docs/metrics-argument-correctness)

---

## Execution Layer Metrics

### Task Completion Metric

**Purpose:** The `TaskCompletionMetric` evaluates whether "your agent successfully accomplishes the intended task. This is the ultimate measure of agent success -- did it do what the user asked?"

**Code Example:**
```python
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="agent")
def travel_agent(user_input):
    flights = search_flights("NYC", "LA", "2025-03-15")
    cheapest = min(flights, key=lambda x: x["price"])
    booking = book_flight(cheapest["id"])
    return f"Booked flight {cheapest['id']} for ${cheapest['price']}. Confirmation: {booking['confirmation']}"

# Initialize metric - task can be auto-inferred or explicitly provided
task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

# Evaluate whether agent completed the task
dataset = EvaluationDataset(goldens=[
    Golden(input="Book the cheapest flight from NYC to LA for tomorrow")])
for golden in dataset.evals_iterator(metrics=[task_completion]):
    travel_agent(golden.input)
```

**When to Use:** "Use TaskCompletionMetric as a top-level success indicator for any agent. It answers the fundamental question: did the agent accomplish its goal?"

**Calculation Method:**
```
Task Completion Score = AlignmentScore(Task, Outcome)
```

The metric extracts the task (either user-provided or inferred from the trace) and the outcome, then uses an LLM to evaluate alignment. A score of 1 means complete task fulfillment; lower scores indicate partial or failed completion.

**Reference:** [Full Task Completion documentation](/docs/metrics-task-completion)

---

### Step Efficiency Metric

**Purpose:** The `StepEfficiencyMetric` evaluates whether "your agent completes tasks without unnecessary steps. An agent might complete a task but waste tokens, time, and resources on redundant or circuitous actions."

**Code Example:**
```python
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789"}

@observe(type="agent")
def inefficient_agent(user_input):
    # Inefficient: searches twice unnecessarily
    flights1 = search_flights("NYC", "LA", "2025-03-15")
    flights2 = search_flights("NYC", "LA", "2025-03-15")  # Redundant!
    cheapest = min(flights1, key=lambda x: x["price"])
    booking = book_flight(cheapest["id"])
    return f"Booked: {booking['confirmation']}"

# Initialize metric
step_efficiency = StepEfficiencyMetric(threshold=0.7, model="gpt-4o")

# Evaluate - metric will penalize the redundant search_flights call
dataset = EvaluationDataset(goldens=[
    Golden(input="Book the cheapest flight from NYC to LA")])
for golden in dataset.evals_iterator(metrics=[step_efficiency]):
    inefficient_agent(golden.input)
```

**When to Use:** "Use StepEfficiencyMetric alongside TaskCompletionMetric to ensure your agent isn't just successful but also efficient. This is critical for production agents where token costs and latency matter."

**Calculation Method:**
```
Step Efficiency Score = AlignmentScore(Task, Execution Steps)
```

The metric extracts the task and all execution steps from the trace, then uses an LLM to evaluate efficiency. It penalizes redundant tool calls, unnecessary reasoning loops, and any actions not strictly required to complete the task.

**Pro Tip:** "A high TaskCompletionMetric score with a low StepEfficiencyMetric score indicates your agent works but needs optimization. Focus on reducing unnecessary steps without sacrificing success rate."

**Reference:** [Full Step Efficiency documentation](/docs/metrics-step-efficiency)

---

## Complete End-to-End Example

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
    ArgumentCorrectnessMetric)

# End-to-end metrics (analyze full agent trace)
task_completion = TaskCompletionMetric()
step_efficiency = StepEfficiencyMetric()
plan_quality = PlanQualityMetric()
plan_adherence = PlanAdherenceMetric()

# Component-level metrics (analyze specific components)
tool_correctness = ToolCorrectnessMetric()
argument_correctness = ArgumentCorrectnessMetric()

# Define tools
@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789", "flight_id": flight_id}

# Attach component-level metrics to the LLM component
@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_llm(user_input):
    # LLM decides to search flights then book
    origin, destination, date = "NYC", "Paris", "2025-03-18"
    flights = search_flights(origin, destination, date)
    cheapest = min(flights, key=lambda x: x["price"])
    booking = book_flight(cheapest["id"])
    # Update span with tool info for component-level evaluation
    update_current_span(
        input=user_input,
        output=f"Booked {cheapest['id']}",
        expected_tools=get_current_golden().expected_tools
    )
    return booking

@observe(type="agent")
def travel_agent(user_input):
    booking = call_llm(user_input)
    return f"Flight booked! Confirmation: {booking['confirmation']}"

# Create evaluation dataset
dataset = EvaluationDataset(goldens=[
    Golden(input="Book a flight from NYC to Paris for next Tuesday", expected_tools=[ToolCall(name="search_flights"), ToolCall(name="book_flight")])])

# Run evaluation with end-to-end metrics
for golden in dataset.evals_iterator(
    metrics=[task_completion, step_efficiency, plan_quality, plan_adherence]):
    travel_agent(golden.input)
```

---

## Choosing the Right AI Agent Evaluation Metrics

| If Your Agent... | Prioritize These Metrics |
|-----------------|--------------------------|
| Uses explicit planning/reasoning | `PlanQualityMetric`, `PlanAdherenceMetric` |
| Calls multiple tools | `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric` |
| Has complex multi-step workflows | `StepEfficiencyMetric`, `TaskCompletionMetric` |
| Runs in production (cost-sensitive) | `StepEfficiencyMetric` |
| Is task-critical (must succeed) | `TaskCompletionMetric` |

**Important Note:** "All AI agent evaluation metrics in deepeval support custom LLM judges, configurable thresholds, strict mode for binary scoring, and detailed reasoning explanations. See each metric's documentation for full configuration options."

---

## Next Steps

- [Set up tracing](/docs/evaluation-llm-tracing) -- Required for all agent metrics to capture execution traces
- [AI Agent Evaluation Guide](/guides/guides-ai-agent-evaluation) -- Deep dive into evaluation strategies
- [End-to-end Evals](/docs/evaluation-end-to-end-llm-evals) -- Learn how to run metrics on full agent traces
- [Component-level Evals](/docs/evaluation-component-level-llm-evals) -- Learn how to attach metrics to specific components

---

## Footer Information

- **Last updated:** February 16, 2026 by Jeffrey Ip
- **Community:** [Discord](https://discord.gg/a3K9c8GRGt), [GitHub](https://github.com/confident-ai/deepeval)
