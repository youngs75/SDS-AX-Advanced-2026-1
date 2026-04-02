# DeepEval MCP and Component-Level Evaluations

## Read This When
- Evaluating MCP (Model Context Protocol) agents using `MCPServer`, `MCPToolCall`, `MCPResourceCall`, or `MCPPromptCall`
- Setting up component-level evaluation with `@observe`, `update_current_span()`, and `evals_iterator()`
- Need the full workflow for granular per-component metric attachment in complex agentic or RAG pipelines

## Skip This When
- Doing simple end-to-end evaluation without component instrumentation → `references/02-llm-evals/10-evaluation-fundamentals.md`
- Need `@observe` decorator parameter reference or trace/span concepts only → `references/02-llm-evals/40-tracing-and-observability.md`

---

## Overview

This reference covers two related topics:

1. **MCP (Model Context Protocol) Evaluation** — evaluating LLM agents that interact with MCP servers (tools, resources, prompts)
2. **Component-Level Evaluation** — evaluating individual internal components of LLM applications using `@observe` tracing

---

## Part 1: MCP Evaluation

### What is MCP?

Model Context Protocol (MCP) is an open-source framework by Anthropic that standardizes how AI systems interact with external tools and data sources.

**MCP Architecture:**
- **Host** — The AI application coordinating one or more MCP clients (e.g., Claude)
- **Client** — Maintains a one-to-one connection with a single server
- **Server** — Paired with one client; provides context (tools, resources, prompts) to the host

**MCP Primitives (what servers expose):**
- **Tools** — Executable functions the LLM can invoke
- **Resources** — Data sources providing contextual information
- **Prompts** — Reusable templates for structuring LLM interactions

---

### MCPServer Class

An abstraction representing an MCP server and its available primitives.

```python
from mcp import ClientSession
from deepeval.test_case import MCPServer

session = ClientSession(...)

# Retrieve primitives from the server
tool_list = await session.list_tools()
resource_list = await session.list_resources()
prompt_list = await session.list_prompts()

# Create MCPServer instance
mcp_server = MCPServer(
    server_name="GitHub",
    transport="stdio",
    available_tools=tool_list.tools,
    available_resources=resource_list.resources,
    available_prompts=prompt_list.prompts
)
```

### MCPServer Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `server_name` | `str` | Optional | Human-readable identifier for this MCP server |
| `transport` | `str` (literal) | Optional | Transport type (`"stdio"`, `"sse"`, etc.); does not affect evaluation logic |
| `available_tools` | `list` | Optional | Tools this server exposes; obtain via `session.list_tools().tools` |
| `available_prompts` | `list` | Optional | Prompts this server exposes; obtain via `session.list_prompts().prompts` |
| `available_resources` | `list` | Optional | Resources this server exposes; obtain via `session.list_resources().resources` |

Each element is of type `Tool`, `Resource`, or `Prompt` from `mcp.types` (standardized by the official MCP Python SDK).

---

### MCP Runtime Primitives

During runtime, format each primitive that was called into a deepeval object:

#### MCPToolCall

```python
from mcp import ClientSession
from deepeval.test_case import MCPToolCall

session = ClientSession(...)

tool_name = "search_files"
tool_args = {"query": "README.md", "repo": "myorg/myrepo"}

# Call the tool via MCP
result = await session.call_tool(tool_name, tool_args)

# Format for deepeval
mcp_tool_called = MCPToolCall(
    name=tool_name,
    args=tool_args,
    result=result,   # CallToolResult from mcp.types
)
```

#### MCPResourceCall

```python
from deepeval.test_case import MCPResourceCall

uri = "github://myorg/myrepo/README.md"
result = await session.read_resource(uri)

mcp_resource_called = MCPResourceCall(
    uri=uri,
    result=result,   # ReadResourceResult from mcp.types
)
```

#### MCPPromptCall

```python
from deepeval.test_case import MCPPromptCall

prompt_name = "code_review_template"
result = await session.get_prompt(prompt_name)

mcp_prompt_called = MCPPromptCall(
    name=prompt_name,
    result=result,   # GetPromptResult from mcp.types
)
```

---

### Single-Turn MCP Evaluation

Evaluating a single LLM interaction that uses MCP:

```python
from deepeval.test_case.mcp import (
    MCPServer,
    MCPToolCall,
    MCPResourceCall,
    MCPPromptCall
)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import MCPUseMetric
from deepeval import evaluate

test_case = LLMTestCase(
    input="List all Python files in the repository",
    actual_output="I found 12 Python files: main.py, utils.py, ...",
    mcp_servers=[mcp_server],
    mcp_tools_called=[MCPToolCall(
        name="search_files",
        args={"query": "*.py"},
        result=tool_result
    )],
    mcp_resources_called=[MCPResourceCall(
        uri="github://myorg/myrepo/",
        result=resource_result
    )],
    mcp_prompts_called=[]   # no prompts used in this interaction
)

evaluate(test_cases=[test_case], metrics=[MCPUseMetric()])
```

### MCP Parameters on LLMTestCase

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mcp_servers` | `List[MCPServer]` | Optional | MCP server instances the agent had access to |
| `mcp_tools_called` | `List[MCPToolCall]` | Optional | MCP tools actually invoked; required for `MCPUseMetric` |
| `mcp_resources_called` | `List[MCPResourceCall]` | Optional | MCP resources actually read; required for `MCPUseMetric` |
| `mcp_prompts_called` | `List[MCPPromptCall]` | Optional | MCP prompts actually retrieved; required for `MCPUseMetric` |

---

### Multi-Turn MCP Evaluation

For conversational agents using MCP:

```python
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.test_case.mcp import MCPServer, MCPToolCall, MCPResourceCall, MCPPromptCall
from deepeval.metrics import MultiTurnMCPMetric
from deepeval import evaluate

turns = [
    Turn(role="user", content="What files are in the repo?"),
    Turn(
        role="assistant",
        content="I'll search for you.",
        mcp_tools_called=[MCPToolCall(
            name="list_files",
            args={"repo": "myorg/myrepo"},
            result=file_list_result
        )],
    ),
    Turn(role="user", content="Can you show me the README?"),
    Turn(
        role="assistant",
        content="Here's the README content...",
        mcp_resources_called=[MCPResourceCall(
            uri="github://myorg/myrepo/README.md",
            result=readme_result
        )],
        mcp_prompts_called=[MCPPromptCall(
            name="summarize_template",
            result=prompt_result
        )],
    ),
]

test_case = ConversationalTestCase(
    turns=turns,
    mcp_servers=[MCPServer(
        server_name="GitHub",
        transport="stdio",
        available_tools=tool_list.tools,
        available_resources=resource_list.resources,
        available_prompts=prompt_list.prompts
    )]
)

evaluate(test_cases=[test_case], metrics=[MultiTurnMCPMetric()])
```

### MCP Parameters on Turn

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mcp_tools_called` | `List[MCPToolCall]` | Optional | MCP tools called in this assistant turn |
| `mcp_resources_called` | `List[MCPResourceCall]` | Optional | MCP resources read in this assistant turn |
| `mcp_prompts_called` | `List[MCPPromptCall]` | Optional | MCP prompts retrieved in this assistant turn |

Only supply MCP parameters on turns where `role="assistant"`.

---

### MCP Evaluation: 4-Step Process

1. **Define an `MCPServer`** using `session.list_tools()` / `list_resources()` / `list_prompts()`
2. **Pipe runtime primitives** into `MCPToolCall`, `MCPResourceCall`, `MCPPromptCall` objects
3. **Create test cases** using `LLMTestCase` (single-turn) or `ConversationalTestCase` (multi-turn)
4. **Run MCP metrics** using `evaluate()` or `assert_test()`

---

### MCP Metrics Reference

| Metric | Test Case Type | Description |
|--------|---------------|-------------|
| `MCPUseMetric` | `LLMTestCase` | Evaluates correctness of single-turn MCP tool/resource/prompt usage |
| `MultiTurnMCPMetric` | `ConversationalTestCase` | Evaluates MCP usage across a full conversation |

Import paths:
```python
from deepeval.metrics import MCPUseMetric
from deepeval.metrics import MultiTurnMCPMetric
```

---

## Part 2: Component-Level Evaluation

### What is Component-Level Evaluation?

Component-level evaluation assesses individual internal components of an LLM application rather than treating the whole system as a black box. Examples of components:
- Retriever (vector DB query)
- LLM generation call
- Tool call execution
- Sub-agent
- Embedding generation

**Key limitation:** Component-level evaluation is currently only supported for **single-turn** use cases.

**When to use:**
- Complex agentic workflows
- Multi-step RAG pipelines
- Code generation systems
- Text-to-SQL pipelines
- Any system where you need granular failure attribution

---

### How Component-Level Evaluation Works

```
@observe decorated app
        ↓
Goldens as input (not test cases)
        ↓
LLMTestCases created at runtime via update_current_span()
        ↓
Metrics in @observe evaluated per span
        ↓
Test run with span-level scores
```

The 4-step process:
1. Decorate your LLM app components with `@observe`
2. Provide it as `observed_callback` to `assert_test()` (or iterate with `evals_iterator()`)
3. Use `update_current_span()` at runtime to create test cases inside spans
4. Metrics defined at the `@observe` level are evaluated for each span

---

### Setup: Tracing with Span-Level Metrics

```python
from typing import List
from openai import OpenAI
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric

client = OpenAI()

@observe(type="agent", name="RAG Pipeline")
def your_llm_app(input: str) -> str:
    chunks = retriever(input)
    return generator(input, chunks)

# No metrics on retriever — just tracing
@observe(type="retriever", name="Vector DB")
def retriever(input: str) -> List[str]:
    chunks = ["Hardcoded", "text", "chunks", "from", "vectordb"]
    update_current_span(input=input, retrieval_context=chunks)
    return chunks

# Metrics evaluated at the LLM generation level
@observe(
    type="llm",
    name="GPT-4o Generator",
    metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()]
)
def generator(input: str, retrieved_chunks: List[str]) -> str:
    context = "\n\n".join(retrieved_chunks)
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": input}
        ]
    ).choices[0].message.content

    # Create test case at runtime for this span
    update_current_span(test_case=LLMTestCase(
        input=input,
        actual_output=res,
        retrieval_context=retrieved_chunks
    ))
    return res
```

---

### Running Component-Level Evals: Python Script

Using `evals_iterator()` to run evaluations:

```python
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric

# Goldens (not test cases!) are the inputs
goldens = [
    Golden(input="What is your name?"),
    Golden(input="What is the refund policy?"),
]
dataset = EvaluationDataset(goldens=goldens)

# Store the dataset
dataset.push(alias="Component Eval Dataset")

# Load and run component-level evals
dataset.pull(alias="Component Eval Dataset")

for golden in dataset.evals_iterator(
    metrics=[AnswerRelevancyMetric()],   # optional: end-to-end metrics on the full trace
    identifier="Component Eval v1.0"
):
    your_llm_app(golden.input)   # span metrics fire automatically
```

`evals_iterator()` Parameters:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `metrics` | `List[BaseMetric]` | Optional | End-to-end metrics for the full trace; span-level metrics are defined in `@observe` |
| `identifier` | `str` | Optional | Test run label on Confident AI |
| `async_config` | `AsyncConfig` | Optional | Concurrency settings |
| `display_config` | `DisplayConfig` | Optional | Console output settings |
| `error_config` | `ErrorConfig` | Optional | Error handling settings |
| `cache_config` | `CacheConfig` | Optional | Caching settings |

---

### Running Component-Level Evals: CI/CD

Using `assert_test()` with `observed_callback`:

```python
# test_llm_app.py
import pytest
import deepeval
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset()
dataset.pull(alias="Component Eval Dataset")

@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden: Golden):
    assert_test(
        golden=golden,
        observed_callback=your_llm_app  # @observe decorated function
        # No metrics here — defined in @observe
        # No test_case creation here — done via update_current_span()
    )

@deepeval.log_hyperparameters()
def hyperparameters():
    return {"model": "gpt-4o", "version": "2026-02-20"}
```

```bash
deepeval test run test_llm_app.py
```

---

### Granular Metric Attachment

Different metrics can be attached to different components:

```python
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ToolCorrectnessMetric
)

@observe(type="agent", name="Full Agent")
def agent(input: str) -> str:
    context = retriever(input)
    tools_used = tool_caller(input)
    return generator(input, context, tools_used)

# Retriever: context quality metrics
@observe(
    type="retriever",
    name="Document Retriever",
    metrics=[ContextualRecallMetric(), ContextualPrecisionMetric()]
)
def retriever(input: str) -> list[str]:
    chunks = fetch_chunks(input)
    update_current_span(
        input=input,
        retrieval_context=chunks,
        context=ground_truth_chunks  # if available
    )
    return chunks

# Tool caller: tool correctness metrics
@observe(
    type="tool",
    name="Tool Executor",
    metrics=[ToolCorrectnessMetric()]
)
def tool_caller(input: str) -> list:
    tools = select_and_call_tools(input)
    update_current_span(
        input=input,
        tools_called=tools,
        expected_tools=get_expected_tools(input)
    )
    return tools

# Generator: output quality metrics
@observe(
    type="llm",
    name="GPT-4o Generator",
    metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()]
)
def generator(input: str, context: list, tools: list) -> str:
    res = call_llm(input, context)
    update_current_span(test_case=LLMTestCase(
        input=input,
        actual_output=res,
        retrieval_context=context
    ))
    return res
```

---

### Dataset Storage for Component Evals

Component-level evaluation uses **Goldens** (not test cases) as inputs:

```python
from deepeval.dataset import EvaluationDataset, Golden

# Store goldens on Confident AI
goldens = [
    Golden(input="What is the refund policy?", expected_output="30-day full refund."),
    Golden(input="How do I track my order?"),
]
dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="My Component Eval Dataset")

# Or store locally
dataset.save_as(file_type="csv", directory="./data")
dataset.save_as(file_type="json", directory="./data")
```

Load options:

```python
# From Confident AI
dataset.pull(alias="My Component Eval Dataset")

# From CSV
dataset.add_goldens_from_csv_file(file_path="data.csv", input_col_name="query")

# From JSON
dataset.add_goldens_from_json_file(file_path="data.json", input_key_name="query")
```

---

### Traces vs. Spans: Key Concepts

| Concept | Description | Evaluation level |
|---------|-------------|-----------------|
| **Span** | A single component's execution (one `@observe` call) | Component-level |
| **Trace** | The complete execution tree from root span to leaf spans | End-to-end |
| **update_current_span()** | Sets test case for the current span | Affects span metrics |
| **update_current_trace()** | Sets test case for the full trace | Affects trace metrics |

A trace has one root span. All other spans are children or descendants of the root. Metrics on a span are evaluated using the `LLMTestCase` set via `update_current_span()` for that specific function.

---

### Component-Level vs. End-to-End: When to Use Which

| Scenario | Recommended Approach |
|----------|---------------------|
| Simple RAG QA or summarization | End-to-end |
| Complex agentic pipeline with many components | Component-level |
| Debugging: which component caused a failure | Component-level |
| Measuring retriever quality independently | Component-level (retriever span) |
| Overall system quality measurement | End-to-end |
| Multi-turn chatbot evaluation | End-to-end (component-level not supported) |
| Both needed | Use `update_current_trace()` + `evals_iterator(metrics=[...])` |

---

### Full Example: RAG App with Both Evaluation Levels

```python
from openai import OpenAI
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.dataset import EvaluationDataset, Golden, get_current_golden

client = OpenAI()

@observe(type="agent")
def rag_app(query: str) -> str:
    chunks = retriever(query)
    return generator(query, chunks)

@observe(type="retriever")
def retriever(query: str) -> list[str]:
    chunks = fetch_from_db(query)
    update_current_span(input=query, retrieval_context=chunks)
    update_current_trace(retrieval_context=chunks)  # accumulate for end-to-end
    return chunks

@observe(type="llm", metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()])
def generator(query: str, chunks: list[str]) -> str:
    golden = get_current_golden()
    expected = golden.expected_output if golden else None

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": query}]
    ).choices[0].message.content

    update_current_span(test_case=LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=chunks,
        expected_output=expected
    ))
    update_current_trace(input=query, output=response)  # for end-to-end evals
    return response

# Run both component-level (via @observe metrics) and end-to-end (via evals_iterator metrics)
dataset = EvaluationDataset()
dataset.pull(alias="RAG Eval Dataset")

for golden in dataset.evals_iterator(
    metrics=[AnswerRelevancyMetric()],   # end-to-end metric on trace
    identifier="RAG Full Eval"
):
    rag_app(golden.input)
```

---

## Related Reference Files

- `10-evaluation-fundamentals.md` - evaluate(), assert_test(), evals_iterator() signatures
- `20-test-cases.md` - LLMTestCase, ConversationalTestCase, ToolCall parameters
- `30-datasets-and-goldens.md` - EvaluationDataset, Golden, dataset loading/saving
- `40-tracing-and-observability.md` - @observe, update_current_span(), update_current_trace()
- `50-ci-cd-and-configs.md` - CLI flags, GitHub Actions YAML, config objects
