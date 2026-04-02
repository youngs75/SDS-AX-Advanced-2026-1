# DeepEval Tracing and Observability

## Read This When
- Instrumenting your LLM application with `@observe` to create traces and spans for component-level evaluation
- Need to use `update_current_span()`, `update_current_trace()`, or `update_llm_span()` for runtime test case creation
- Setting up production LLM monitoring and observability via Confident AI dashboard integration

## Skip This When
- Doing end-to-end evaluation without tracing (black-box approach) → `references/02-llm-evals/10-evaluation-fundamentals.md`
- Need MCP-specific evaluation setup or the full component-level evaluation workflow → `references/02-llm-evals/60-mcp-and-component-evals.md`

---

## Overview

DeepEval's tracing system lets you instrument LLM applications with minimal code changes. A **trace** is the complete execution of one LLM interaction; it comprises multiple **spans**, where each span represents a single component (retriever, LLM call, tool, agent, etc.).

Tracing enables:
- **Dynamic test case generation** — define `LLMTestCase`s at runtime as data flows through the system
- **Precision debugging** — identify exactly which component (tool, retriever, LLM) failed
- **Targeted metric evaluation** — attach different metrics to different components without restructuring your app
- **End-to-end evals with trace data** — use `evals_iterator()` with `metrics` for full-trace evaluation

Key property: tracing is **non-intrusive** — it adds no latency and requires no changes to production code logic.

---

## @observe Decorator

The `@observe` decorator wraps any Python function to create a span. Nested `@observe` calls form a tree of spans that together make up a trace.

```python
from deepeval.tracing import observe

@observe()
def my_function(input: str) -> str:
    return "result"
```

### @observe Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `type` | `str` | Optional | Span type: `"llm"`, `"retriever"`, `"tool"`, `"agent"`, or any custom string. Affects how the span is displayed on Confident AI. |
| `name` | `str` | Optional | Display name for this span on Confident AI. Defaults to the function name. |
| `metrics` | `List[BaseMetric]` | Optional | Metrics to evaluate this span's test case during component-level evaluation. Test case must be provided via `update_current_span()`. |
| `metric_collection` | `str` | Optional | Name of a metric collection stored on Confident AI to use for this span. |

### Span Types

| Type | Description |
|------|-------------|
| `"agent"` | Top-level orchestrator span |
| `"llm"` | LLM generation call span |
| `"retriever"` | Vector DB / document retrieval span |
| `"tool"` | Tool or function call span |
| `"embedding"` | Embedding generation span |
| custom string | Any user-defined label |

### Example with All Options

```python
from deepeval.tracing import observe
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

@observe(
    type="llm",
    name="My GPT-4 Generator",
    metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()]
)
def generator(query: str, context: list[str]) -> str:
    # LLM call here
    return response
```

---

## Nested Spans (Trace Structure)

Nested `@observe` calls automatically form parent-child span relationships:

```python
from deepeval.tracing import observe

@observe(type="agent", name="RAG Pipeline")
def rag_app(query: str) -> str:
    context = retriever(query)         # child span
    return generator(query, context)   # child span

@observe(type="retriever", name="Vector DB Retriever")
def retriever(query: str) -> list[str]:
    return fetch_from_vectordb(query)

@observe(type="llm", name="GPT-4 Generator")
def generator(query: str, context: list[str]) -> str:
    return call_openai(query, context)
```

The resulting trace has `rag_app` as the root span with `retriever` and `generator` as children.

---

## update_current_span()

Creates or updates the `LLMTestCase` for the currently executing span. Required for component-level metric evaluation.

```python
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase

@observe(type="llm", metrics=[AnswerRelevancyMetric()])
def generator(query: str, context: list[str]) -> str:
    response = call_openai(query, context)

    # Option A: pass a full LLMTestCase
    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            retrieval_context=context
        )
    )
    return response
```

### update_current_span() Parameters

**Option A — Full LLMTestCase:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `test_case` | `LLMTestCase` | Yes (if not using Option B) | Complete test case object for this span |

**Option B — Individual fields:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | `str` | Optional | The input to this component |
| `output` | `str` | Optional | The output from this component |
| `retrieval_context` | `List[str]` | Optional | Retrieved chunks (for retriever spans) |
| `context` | `List[str]` | Optional | Ground-truth context |
| `expected_output` | `str` | Optional | Expected output for comparison |
| `tools_called` | `List[ToolCall]` | Optional | Tools invoked by this component |
| `expected_tools` | `List[ToolCall]` | Optional | Tools expected to be invoked |

```python
@observe(type="retriever")
def retriever(query: str) -> list[str]:
    chunks = fetch_chunks(query)
    update_current_span(input=query, retrieval_context=chunks)
    return chunks
```

---

## update_current_trace()

Updates the end-to-end test case for the entire trace (not a specific span). Used when you want to run end-to-end metrics on the complete execution.

```python
from deepeval.tracing import observe, update_current_trace

@observe(type="agent")
def llm_app(query: str) -> str:
    @observe(type="retriever")
    def retriever(query: str) -> list[str]:
        chunks = fetch_chunks(query)
        update_current_trace(retrieval_context=chunks)  # accumulate onto trace
        return chunks

    @observe(type="llm")
    def generator(query: str, chunks: list[str]) -> str:
        result = call_openai(query, chunks)
        update_current_trace(input=query, output=result)  # set trace input/output
        return result

    return generator(query, retriever(query))
```

### update_current_trace() Parameters

Same as `update_current_span()`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `test_case` | `LLMTestCase` | Full test case; individual params override if both provided |
| `input` | `str` | Input to the overall LLM application |
| `output` | `str` | Output from the overall LLM application |
| `retrieval_context` | `List[str]` | All retrieved chunks across the trace |
| `context` | `List[str]` | Ground-truth context |
| `expected_output` | `str` | Expected final output |
| `tools_called` | `List[ToolCall]` | All tools invoked across the trace |
| `expected_tools` | `List[ToolCall]` | Expected tools for the full interaction |

Note: Individual parameters override `test_case` values when both are provided.

---

## update_llm_span()

Specialized update function for LLM spans. Used specifically to associate a `Prompt` object with an LLM span for prompt evaluation.

```python
from deepeval.tracing import observe, update_llm_span
from deepeval.prompt import Prompt, PromptMessage
from deepeval.metrics import AnswerRelevancyMetric

prompt = Prompt(
    alias="My System Prompt",
    messages_template=[PromptMessage(role="system", content="You are helpful.")]
)

@observe(type="llm", metrics=[AnswerRelevancyMetric()])
def generator(input: str) -> str:
    messages = [{"role": msg.role, "content": msg.content} for msg in prompt.messages_template]
    response = call_openai(messages, input)
    update_llm_span(prompt=prompt)  # associates prompt with this LLM span
    return response
```

Note: `update_llm_span()` can **only** be called inside a span where `type="llm"`.

---

## Complete Example: RAG Application Tracing

```python
from openai import OpenAI
from deepeval.tracing import observe, update_current_span, update_current_trace
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRecallMetric

client = OpenAI()

@observe(type="agent", name="RAG App")
def rag_app(query: str) -> str:
    chunks = retriever(query)
    return generator(query, chunks)

@observe(type="retriever", name="Vector DB")
def retriever(query: str) -> list[str]:
    chunks = fetch_from_vectordb(query)
    update_current_span(input=query, retrieval_context=chunks)
    update_current_trace(retrieval_context=chunks)  # also update trace-level
    return chunks

@observe(
    type="llm",
    name="GPT-4o Generator",
    metrics=[AnswerRelevancyMetric(), FaithfulnessMetric()]
)
def generator(query: str, chunks: list[str]) -> str:
    context_str = "\n\n".join(chunks)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Context:\n{context_str}"},
            {"role": "user", "content": query}
        ]
    ).choices[0].message.content

    # span-level test case for component metrics
    update_current_span(test_case=LLMTestCase(
        input=query,
        actual_output=response,
        retrieval_context=chunks
    ))
    # trace-level test case for end-to-end metrics
    update_current_trace(input=query, output=response)
    return response

# Run the app — tracing happens automatically
result = rag_app("What is the refund policy?")
```

---

## Using Goldens with Tracing

Access the golden currently being evaluated with `get_current_golden()`. This is useful when you need `expected_output` from the golden inside your component:

```python
from deepeval.dataset import get_current_golden
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase

@observe(type="llm", metrics=[AnswerRelevancyMetric()])
def generator(query: str) -> str:
    response = call_openai(query)

    golden = get_current_golden()
    expected = golden.expected_output if golden else None

    update_current_span(
        test_case=LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected
        )
    )
    return response
```

---

## observed_callback Pattern

For component-level CI/CD evaluation, pass the traced function as `observed_callback` to `assert_test()`:

```python
# In your LLM app module (your_agent.py)
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

@observe(metrics=[AnswerRelevancyMetric()])
def your_llm_app(input: str) -> str:
    response = call_openai(input)
    update_current_span(test_case=LLMTestCase(input=input, actual_output=response))
    return response
```

```python
# In your test file (test_llm_app.py)
import pytest
from your_agent import your_llm_app
from deepeval import assert_test
from deepeval.dataset import Golden

@pytest.mark.parametrize("golden", dataset.goldens)
def test_llm_app(golden: Golden):
    assert_test(
        golden=golden,
        observed_callback=your_llm_app
        # No metrics here — they are defined in @observe
    )
```

```bash
deepeval test run test_llm_app.py
```

---

## Environment Variables

Control tracing behavior when running outside of `evaluate()` or `assert_test()`:

```bash
# Disable verbose console logs from @observe
CONFIDENT_TRACE_VERBOSE=0

# Disable automatic trace flushing
CONFIDENT_TRACE_FLUSH=0
```

---

## LLM Observability: 5 Key Components

Tracing powers these observability capabilities:

| Component | Description |
|-----------|-------------|
| **Response Monitoring** | Real-time tracking of queries, responses, cost, and latency |
| **Automated Evaluations** | Automatic metric scoring on monitored responses |
| **Advanced Filtering** | Filter flagged outputs by metric scores, tags, or custom criteria |
| **Application Tracing** | Span/trace tree mapping component interactions and bottlenecks |
| **Human-in-the-Loop** | Human feedback on flagged outputs for quality enrichment |

### Why Observability is Necessary

1. **Complex systems** — LLM apps integrate retrievers, APIs, embedders; tracing isolates which component failed
2. **Hallucination risk** — Continuous monitoring catches incorrect/misleading responses before user impact
3. **Unpredictable model changes** — Model provider updates can shift performance; monitoring detects regressions
4. **User unpredictability** — Production traffic reveals gaps not covered in pre-production testing
5. **Continuous experimentation** — Compare model configs, prompts, and KB versions using trace replay

---

## Confident AI Dashboard Integration

When logged in to Confident AI (`deepeval login`), all traces are automatically sent to the dashboard where you can:
- View full trace trees with span-level scores
- Filter test runs by metric, tag, score, or date
- Compare metric trends across deployments
- Trigger human review on flagged responses
- Browse golden datasets linked to test runs

```bash
# Login to enable Confident AI integration
deepeval login
```

Traces are visible in the Confident AI UI after executing any `@observe`-decorated function within `evaluate()`, `assert_test()`, or `evals_iterator()`.

---

## LLM Monitoring (Production)

For ongoing production monitoring (not just testing), the same `@observe` decorator sends traces to Confident AI for:
- Real-time metric evaluation on live traffic
- Latency and cost tracking per span
- Alert setup for metric degradation

---

## Related Reference Files

- `10-evaluation-fundamentals.md` - evaluate(), assert_test(), evals_iterator()
- `20-test-cases.md` - LLMTestCase parameters
- `60-mcp-and-component-evals.md` - Component-level evaluation complete workflow
- `50-ci-cd-and-configs.md` - CI/CD integration patterns
