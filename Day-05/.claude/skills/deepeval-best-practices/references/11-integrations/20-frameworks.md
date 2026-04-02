# DeepEval Framework Integrations

Source: https://deepeval.com/integrations/frameworks/
Fetched: 2026-02-20

## Read This When
- Using OpenAI SDK, Anthropic SDK, LangChain, LangGraph, or Hugging Face and need DeepEval's native integration (client wrappers, CallbackHandler, DeepEvalHuggingFaceCallback)
- Want to set up end-to-end, component-level, or production online evaluations using framework-specific patterns (e.g., `deepeval.openai.OpenAI`, `deepeval.anthropic.Anthropic`, LangChain `CallbackHandler`)
- Need to evaluate LLM outputs during Hugging Face fine-tuning with `DeepEvalHuggingFaceCallback`

## Skip This When
- Need to configure model providers (API keys, endpoints) rather than framework integrations -- see `references/11-integrations/10-model-providers.md`
- Want to implement a custom LLM wrapper from scratch -- see `references/09-guides/40-custom-llms-and-embeddings.md`
- Looking for vector database integrations (Chroma, Elasticsearch, Qdrant, PGVector) -- see `references/11-integrations/30-vector-databases.md`

---

## Table of Contents

1. [OpenAI (Framework)](#1-openai-framework)
2. [Anthropic (Framework)](#2-anthropic-framework)
3. [LangChain](#3-langchain)
4. [LangGraph](#4-langgraph)
5. [Hugging Face](#5-hugging-face)

---

## 1. OpenAI (Framework)

Source: https://deepeval.com/integrations/frameworks/openai

### Overview
DeepEval streamlines evaluation and tracing of OpenAI applications through an "OpenAI client wrapper" supporting end-to-end and component-level evaluations, plus production-level online evaluations.

---

### 1. End-to-End Evals

This section covers evaluating OpenAI applications by replacing the standard OpenAI client with DeepEval's wrapper, passing in desired metrics.

**Supported Methods:**
- Chat Completions
- Responses
- Async Chat Completions
- Async Responses

**Code Example - Chat Completions:**
```python
from deepeval.openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

client = OpenAI()
goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
            expected_output=golden.expected_output,
        )
    ):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": golden.input}
            ],
        )
```

**Code Example - Responses:**
```python
from deepeval.openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

client = OpenAI()
goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
            expected_output=golden.expected_output,
        )
    ):
        client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant.",
            input=golden.input,
        )
```

**Code Example - Async Chat Completions:**
```python
import asyncio
from deepeval.openai import AsyncOpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

async_client = AsyncOpenAI()

async def openai_llm_call(input):
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
            expected_output=golden.expected_output,
        )
    ):
        return await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot. Always generate a string response."},
                {"role": "user", "content": input},
            ],
        )

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(openai_llm_call(golden.input))
    dataset.evaluate(task)
```

**Code Example - Async Responses:**
```python
import asyncio
from deepeval.openai import AsyncOpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

async_client = AsyncOpenAI()

async def openai_llm_call(input):
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
            expected_output=golden.expected_output,
        )
    ):
        return await async_client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant.",
            input=input,
        )

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]
dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    task = asyncio.create_task(openai_llm_call(golden.input))
    dataset.evaluate(task)
```

**Optional Parameters for `LlmSpanContext`:**

| Parameter | Type | Description |
|-----------|------|-------------|
| metrics | [Optional] BaseMetric list | Metrics to evaluate the generation |
| expected_output | [Optional] string | Expected output of OpenAI generation |
| retrieval_context | [Optional] list of strings | Retrieved contexts passed to generation |
| context | [Optional] list of strings | Ideal retrieved contexts for generation |
| expected_tools | [Optional] list of strings | Expected tools called during generation |

**Note:** DeepEval's OpenAI client automatically extracts input and actual_output from API responses, enabling metrics like Answer Relevancy out-of-the-box. For metrics requiring additional parameters (e.g., Faithfulness), explicitly set those parameters when invoking the client.

---

### 2. Component-Level Evals

This approach uses DeepEval's OpenAI client within component-level evaluations by adding the `@observe` decorator to application components and replacing existing clients with DeepEval's wrapper.

**Code Example - Chat Completions:**
```python
from deepeval.tracing import observe
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.openai import OpenAI
from deepeval.tracing import trace, LlmSpanContext

client = OpenAI()

@observe()
def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
def llm_app(input):
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        ),
    ):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": '\n'.join(retrieve_docs(input)) + "\n\nQuestion: " + input}
            ]
        )
    return response.choices[0].message.content

# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="...")])

# Iterate through goldens
for golden in dataset.evals_iterator():
    # run your LLM application
    llm_app(input=golden.input)
```

**Code Example - Responses:**
```python
from deepeval.tracing import observe
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.openai import OpenAI
from deepeval.tracing import trace, LlmSpanContext

@observe()
def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
def llm_app(input):
    client = OpenAI()
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        ),
    ):
        response = client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant.",
            input=input
        )
    return response.output_text

# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="...")])

# Iterate through goldens
for golden in dataset.evals_iterator():
    llm_app(input=golden.input)
```

**Code Example - Async Chat Completions:**
```python
import asyncio
from deepeval.tracing import observe
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.openai import AsyncOpenAI
from deepeval.tracing import trace, LlmSpanContext

@observe()
async def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
async def llm_app(input):
    client = AsyncOpenAI()
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        ),
    ):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": '\n'.join(await retrieve_docs(input)) + "\n\nQuestion: " + input}
            ],
        )
    return response.choices[0].message.content

# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="...")])

# Iterate through goldens
for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(input=golden.input))
    dataset.evaluate(task)
```

**Code Example - Async Responses:**
```python
import asyncio
from deepeval.tracing import observe
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.openai import AsyncOpenAI
from deepeval.tracing import trace, LlmSpanContext

@observe()
async def retrieve_docs(query):
    return [
        "Paris is the capital and most populous city of France.",
        "It has been a major European center of finance, diplomacy, commerce, and science."
    ]

@observe()
async def llm_app(input):
    client = AsyncOpenAI()
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()],
        ),
    ):
        response = await client.responses.create(
            model="gpt-4o",
            instructions="You are a helpful assistant.",
            input=input,
        )
    return response.output_text

# Create dataset
dataset = EvaluationDataset(goldens=[Golden(input="...")])

# Iterate through goldens
for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(input=golden.input))
    dataset.evaluate(task)
```

**Automatic Functionality Within @observe Components:**

When DeepEval's OpenAI client is used inside `@observe` components, it automatically:
- Generates an LLM span for every OpenAI API call, including nested Tool spans for tool invocations
- Attaches an LLMTestCase to each generated LLM span, capturing inputs, outputs, and tools called
- Records span-level LLM attributes such as input prompt, generated output, and token usage
- Logs hyperparameters including model name and system prompt for comprehensive experiment analysis

---

### 3. Online Evals in Production

For production OpenAI applications requiring evaluations on incoming traces, use online evals to run evaluations on Confident AI's server.

Set the `llm_metric_collection` name in the `trace` context when invoking your OpenAI client to evaluate LLM Spans.

```python
from deepeval.openai import OpenAI
from deepeval.tracing import trace, LlmSpanContext

client = OpenAI()

with trace(
    llm_span_context=LlmSpanContext(
        metric_collection="test_collection_1",
    ),
):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )
```

### Related Links

- [Next: Anthropic](/integrations/frameworks/anthropic)
- [LangChain](/integrations/frameworks/langchain)
- [LangGraph](/integrations/frameworks/langgraph)
- [GitHub Repository](https://github.com/confident-ai/deepeval)
- [Discord Community](https://discord.gg/a3K9c8GRGt)

---

## 2. Anthropic (Framework)

Source: https://deepeval.com/integrations/frameworks/anthropic

### Overview
DeepEval integrates with Anthropic models to evaluate and trace Claude LLM requests in development and production environments.

---

### Local Evaluations in Development

#### Evaluating Claude as a Standalone

**Synchronous Example:**
```python
from deepeval.anthropic import Anthropic
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

client = Anthropic()
dataset = EvaluationDataset()
datset.pull(alias="My Dataset")

for golden in dataset.evals_iterator():
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric()],
            expected_output=golden.expected_output,
        )
    ):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": golden.input
                }
            ],
        )
        return response.content[0].text
```

**Asynchronous Example:**
```python
import asyncio
from deepeval.anthropic import AsyncAnthropic
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

async_client = AsyncAnthropic()

async def llm_app(input):
    with trace(
        llm_span_context=LlmSpanContext(
            llm_metrics=[AnswerRelevancyMetric()],
            expected_output=golden.expected_output,
        )
    ):
        response = await async_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": golden.input
                }
            ],
        )
        return response.content[0].text

dataset = EvaluationDataset()
datset.pull(alias="My Dataset")

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(input=golden.input))
    dataset.evaluate(task)
```

**Trace Context Parameters (5 Optional):**
- `metrics`: List of evaluation metrics for model output assessment
- `expected_output`: Ideal output the model should produce for given input
- `retrieval_context`: Information or documents for ground truth comparison
- `context`: Ideal context snippets for answer generation
- `expected_tools`: Tool names/functions the model should invoke

**Note:** "Input and actual output are auto-extracted for every generation" with DeepEval's Anthropic client, enabling metrics like Answer Relevancy without additional configuration.

---

#### Evaluating Claude within Components

**Synchronous Example:**
```python
from deepeval.anthropic import Anthropic
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, observe, LlmSpanContext

@observe()
def retrieve_documents(query):
    return [
        "React is a popular Javascript library for building user interfaces.",
        "It allows developers to create large web applications that can update and render efficiently in response to data changes."
    ]

@observe()
def llm_app(input):
    client = Anthropic()
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric()],
            expected_output=golden.expected_output,
        )
    ):
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": golden.input
                }
            ]
        )
    return response.content[0].text

dataset = EvaluationDataset()
datset.pull(alias="My Dataset")

for golden in dataset.evals_iterator():
    llm_app(input=golden.input)
```

**Asynchronous Example:**
```python
import asyncio
from deepeval.anthropic import AsyncAnthropic
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, observe, LlmSpanContext

@observe()
def retrieve_documents(query):
    return [
        "React is a popular Javascript library for building user interfaces.",
        "It allows developers to create large web applications that can update and render efficiently in response to data changes."
    ]

@observe()
async def llm_app(input):
    async_client = AsyncAnthropic()
    with trace(
        llm_span_context=LlmSpanContext(
            metrics=[AnswerRelevancyMetric(), BiasMetric()]
        ),
    ):
        response = await async_client.responses.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": input
                }
            ],
        )
    return response.content[0].text

dataset = EvaluationDataset()
datset.pull(alias="My Dataset")

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(input=golden.input))
    dataset.evaluate(task)
```

**Automatic Capabilities within @observe Components:**
- Generates LLM spans for Messages API calls with nested Tool spans
- Attaches LLMTestCase to each LLM span capturing inputs, outputs, and tools
- Records span-level LLM attributes (input prompt, output, token usage)
- Logs hyperparameters (model name, system prompt) for experiment analysis

---

### Online Evaluations in Production

```python
from deepeval.anthropic import Anthropic
from deepeval.tracing import trace, LlmSpanContext

client = Anthropic()

with trace(
    llm_span_context=LlmSpanContext(
        metric_collection="test_collection_1",
    ),
):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system="You are a helpful assistant.",
        messages=[
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
    )
```

**Key Point:** Use `llm_metric_collection` parameter to evaluate traces against metric collections on Confident AI's server in production environments.

### Related Resources

- [Previous: OpenAI Integration](/integrations/frameworks/openai)
- [Next: LangChain Integration](/integrations/frameworks/langchain)
- [Complete Tracing Guide](/docs/evaluation-component-level-llm-evals)
- [GitHub Repository](https://github.com/confident-ai/deepeval)
- [Discord Community](https://discord.gg/a3K9c8GRGt)

---

## 3. LangChain

Source: https://deepeval.com/integrations/frameworks/langchain

### Overview

LangChain is described as "an open-source framework for developing applications powered by large language models, enabling chaining of LLMs with external data sources and expressive workflows."

**Recommendation:** Log in to Confident AI to view evaluation traces using:
```bash
deepeval login
```

---

### End-to-End Evals

DeepEval enables evaluation of LangChain applications in under a minute.

#### Configure LangChain

Create a `CallbackHandler` with task completion metrics and pass it to your LangChain application's `invoke` method.

**Code Example (main.py):**

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric

@tool
def multiply(a: int, b: int) -> int:
    """Returns the product of two numbers"""
    return a * b

llm = ChatOpenAI(model="gpt-4o-mini")
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can perform mathematical operations."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, [multiply], agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[multiply], verbose=True)

# result = agent_executor.invoke(
#    {"input": "What is 8 multiplied by 6?"},
#    config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
# )
# print(result)
```

**Important Note:** Only Task Completion metrics are supported for the LangChain integration. For other metrics, manually set up tracing instead.

#### Run Evaluations

**Synchronous Approach:**

```python
from deepeval.dataset import EvaluationDataset, Golden

dataset = EvaluationDataset(goldens=[
    Golden(input="What is 3 * 12?"),
    Golden(input="What is 8 * 6?")
])

for golden in dataset.evals_iterator():
    agent_executor.invoke(
        {"input": golden.input},
        config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
    )
```

**Asynchronous Approach:**

```python
import asyncio

dataset = EvaluationDataset(goldens=[
    Golden(input="What is 3 * 12?"),
    Golden(input="What is 8 * 6?")
])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent_executor.ainvoke(
            {"input": golden.input},
            config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
        )
    )
    dataset.evaluate(task)
```

**Result:** The `evals_iterator` automatically generates a test run with individual evaluation traces for each golden.

**Note:** For evaluating individual components, set up tracing instead.

---

### Component-level Evals

Evaluate individual LangChain application components.

#### LLM

Define metrics in the metadata of `BaseLanguageModel` instances:

```python
from langchain_openai import ChatOpenAI
from deepeval.metrics import AnswerRelevancyMetric

llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric": [AnswerRelevancyMetric()]}
).bind_tools([get_weather])
```

#### Tool

Use DeepEval's LangChain `tool` decorator to pass metrics:

```python
# from langchain_core.tools import tool
from deepeval.integrations.langchain import tool
from deepeval.metrics import AnswerRelevancyMetric

@tool(metric=[AnswerRelevancyMetric()])
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    return f"It's always sunny in {location}!"
```

---

### Evals in Production

Replace `metrics` in `CallbackHandler` with a metric collection string from Confident AI:

```python
result = agent_executor.invoke(
    {"input": "What is 8 multiplied by 6?"},
    config={"callbacks": [CallbackHandler(metric_collection="<metric-collection-name-with-task-completion>")]}
)
```

This automatically evaluates all incoming production traces with defined task completion metrics.

---

### Related Links

- [Previous: Anthropic Integration](/integrations/frameworks/anthropic)
- [Next: LangGraph Integration](/integrations/frameworks/langgraph)
- [Task Completion Metrics Documentation](/docs/metrics-task-completion)
- [LLM Tracing Setup](/docs/evaluation-llm-tracing)
- [Confident AI Cloud Docs](https://www.confident-ai.com/docs)
- [GitHub Repository](https://github.com/confident-ai/deepeval)

---

## 4. LangGraph

Source: https://deepeval.com/integrations/frameworks/langgraph

### Overview
"LangGraph is an open-source framework for developing applications powered by large language models, enabling chaining of LLMs with external data sources and expressive workflows to build advanced generative AI solutions."

---

### End-to-End Evaluations

#### Configuration
Users can create a `CallbackHandler` with task completion metrics and pass it to LangGraph's `invoke` method:

```python
from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric

def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)
```

**Important Limitation:** "Only Task Completion is supported for the LangGraph integration. To use other metrics, manually set up tracing instead."

#### Running Evaluations

**Synchronous approach:**
```python
from deepeval.dataset import Golden, EvaluationDataset

goldens = [
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
]

dataset = EvaluationDataset(goldens=goldens)

for golden in dataset.evals_iterator():
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
    )
```

**Asynchronous approach:**
```python
import asyncio
from deepeval.dataset import Golden, EvaluationDataset

dataset = EvaluationDataset(goldens=[
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
])

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent.ainvoke(
            input={"messages": [{"role": "user", "content": golden.input}]},
            config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
        )
    )
    dataset.evaluate(task)
```

---

### Component-level Evaluations

#### LLM Evaluation
Define metrics in the metadata of `BaseLanguageModel` instances:

```python
from langchain_openai import ChatOpenAI
from deepeval.metrics import AnswerRelevancyMetric

llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric": [AnswerRelevancyMetric()]}
).bind_tools([get_weather])
```

#### Tool Evaluation
Use DeepEval's LangChain tool decorator:

```python
from deepeval.integrations.langchain import tool
from deepeval.metrics import AnswerRelevancyMetric

@tool(metric=[AnswerRelevancyMetric()])
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    return f"It's always sunny in {location}!"
```

---

### Production Evaluations
For online production evaluations, replace metrics with a metric collection string from Confident AI:

```python
result = agent_executor.invoke(
    {"input": "What is 8 multiplied by 6?"},
    config={"callbacks": [CallbackHandler(metric_collection="<metric-collection-name-with-task-completion>")]}
)
```

### Key Notes
- Users should login to Confident AI to view evaluation traces
- The `evals_iterator()` automatically generates test runs with individual evaluation traces
- For component-level evaluation, manual tracing setup is required for non-task-completion metrics

### Related Links
- [Previous: LangChain Integration](/integrations/frameworks/langchain)
- [Next: OpenAI Integration](/integrations/frameworks/openai)

---

## 5. Hugging Face

Source: https://deepeval.com/integrations/frameworks/huggingface

### Overview
This documentation covers integrating DeepEval with Hugging Face's `transformers` library for evaluating LLM outputs during model fine-tuning.

### Quick Summary

The page provides a recap example using Mistral's `mistralai/Mistral-7B-v0.1` model through Hugging Face's transformers library to generate text predictions.

---

### Evals During Fine-Tuning

DeepEval integrates with Hugging Face's `transformers.Trainer` module via `DeepEvalHuggingFaceCallback`, enabling "real-time evaluation of LLM outputs during model fine-tuning for each epoch."

#### Setup Steps

**1. Prepare Dataset for Fine-tuning**
- Load dataset using `datasets.load_dataset()`
- Tokenize using `AutoTokenizer` from the pretrained model
- Map tokenization function with batching enabled

**2. Setup Training Arguments**
- Configure `TrainingArguments` with output directory, epochs, batch size, warmup steps, decay, and logging parameters

**3. Initialize LLM and Trainer**
- Load model using `AutoModelForCausalLM.from_pretrained()`
- Create `Trainer` instance with model, arguments, and tokenized dataset

**4. Define Evaluation Criteria**
- Create `Golden` objects with input data
- Initialize `EvaluationDataset` with goldens
- Define metrics like `GEval` with evaluation parameters

**5. Fine-tune and Evaluate**
- Create `DeepEvalHuggingFaceCallback` instance
- Add callback to trainer
- Execute `trainer.train()`

#### DeepEvalHuggingFaceCallback Parameters

| Parameter | Type | Purpose |
|-----------|------|---------|
| `metrics` | list | DeepEval evaluation metrics to use |
| `evaluation_dataset` | EvaluationDataset | Dataset for evaluation |
| `aggregation_method` | string | Score aggregation ('avg', 'min', 'max') |
| `trainer` | Trainer | Transformers trainer instance |
| `tokenizer_args` | dict | Tokenizer configuration arguments |

---

### Complete Code Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
prompt = "My favourite condiment is"
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
```

```python
from transformers import AutoTokenizer
from datasets import load_dataset

training_dataset = load_dataset("text", data_files={"train": "train.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenized_dataset = training_dataset.map(tokenize_function, batched=True)
```

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)
```

```python
from transformers import AutoModelForCausalLM, Trainer

llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

trainer = Trainer(
    model=llm,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)
```

```python
from deepeval.test_case import LLMTestCaseParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval

first_golden = Golden(input="...")
second_golden = Golden(input="...")
dataset = EvaluationDataset(goldens=[first_golden, second_golden])

coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - determine if the actual output is coherent with the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)
```

```python
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback

deepeval_hugging_face_callback = DeepEvalHuggingFaceCallback(
    evaluation_dataset=dataset,
    metrics=[coherence_metric],
    trainer=trainer
)

trainer.add_callback(deepeval_hugging_face_callback)
trainer.train()
```

### Key Notes

- The `EvaluationDataset` uses goldens rather than test cases since inference runs at evaluation time
- Evaluations execute on the entire dataset at each epoch conclusion
- The callback aggregates metric scores using specified methods

### Related Links

- [DeepEval GitHub Repository](https://github.com/confident-ai/deepeval)
- [Evaluation Datasets Documentation](/docs/evaluation-datasets)
