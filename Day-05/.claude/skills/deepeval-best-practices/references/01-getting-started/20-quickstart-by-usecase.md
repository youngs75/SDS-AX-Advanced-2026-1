# Quickstart by Use Case

## Read This When
- Need a minimal working code example for a specific use case: RAG, chatbot, AI agent, MCP, or LLM Arena
- Choosing which metrics to use for your particular application type
- Setting up multi-turn conversation simulation with `ConversationSimulator`

## Skip This When
- Need initial installation and environment setup → `references/01-getting-started/10-installation-and-setup.md`
- Need deep-dive into a specific metric's parameters and scoring logic → `references/03-eval-metrics/10-metrics-overview.md`

---

This file provides minimal working code examples and key metric selections for each major DeepEval use case: RAG, Chatbots, AI Agents, MCP, and LLM Arena.

---

## RAG Evaluation Quickstart

### Overview

RAG evaluation involves assessing the retriever and generator components separately. The 5 RAG-specific metrics cover every aspect of pipeline quality.

### 5 Key RAG Metrics

| Metric | What It Evaluates | Required Fields |
|--------|-------------------|-----------------|
| `AnswerRelevancyMetric` | Whether the LLM's response is relevant to the input | `input`, `actual_output` |
| `FaithfulnessMetric` | Whether the response is grounded in the retrieved context | `actual_output`, `retrieval_context` |
| `ContextualPrecisionMetric` | Whether retrieved nodes are ranked appropriately | `input`, `expected_output`, `retrieval_context` |
| `ContextualRecallMetric` | Whether retrieved context aligns with expected output | `expected_output`, `retrieval_context` |
| `ContextualRelevancyMetric` | Whether retrieved context is relevant to the input | `input`, `retrieval_context` |

### Minimal Working Example (End-to-End)

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

# Your RAG pipeline returns (output, retrieved_contexts)
def rag_pipeline(input: str):
    # Replace with your actual RAG implementation
    actual_output = "Tickets can be purchased at ticketmaster.com"
    retrieved_contexts = [
        "Coldplay tickets are available on Ticketmaster and StubHub.",
        "Concert tickets go on sale 60 days before the event."
    ]
    return actual_output, retrieved_contexts

input = "How do I purchase tickets to a Coldplay concert?"
actual_output, retrieved_contexts = rag_pipeline(input)

test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    retrieval_context=retrieved_contexts,
    expected_output="Tickets can be purchased at Ticketmaster or StubHub."  # optional
)

evaluate(
    [test_case],
    metrics=[
        AnswerRelevancyMetric(threshold=0.8),
        FaithfulnessMetric(threshold=0.8),
        ContextualPrecisionMetric(threshold=0.8),
        ContextualRecallMetric(threshold=0.8),
        ContextualRelevancyMetric(threshold=0.8),
    ]
)
```

### LangChain RAG Pipeline Example

```python
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model="gpt-4")
vectorstore = Chroma(persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_pipeline(input: str):
    retrieved_docs = retriever.get_relevant_documents(input)
    context_texts = [doc.page_content for doc in retrieved_docs]
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    result = qa_chain.invoke({"query": input})
    return result["result"], context_texts
```

### Component-Level: Evaluating the Retriever Separately

```python
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden
import os

os.environ["CONFIDENT_TRACE_FLUSH"] = "1"  # prevent trace loss

contextual_relevancy = ContextualRelevancyMetric(threshold=0.6)

@observe(metrics=[contextual_relevancy])
def retriever(query: str):
    retrieved_chunks = ["chunk 1 text", "chunk 2 text"]  # replace with real retrieval
    update_current_span(
        test_case=LLMTestCase(input=query, retrieval_context=retrieved_chunks)
    )
    return retrieved_chunks

dataset = EvaluationDataset(goldens=[Golden(input="How do I buy Coldplay tickets?")])
for golden in dataset.evals_iterator():
    retriever(golden.input)
```

### Component-Level: Combined Retriever and Generator

```python
from deepeval.tracing import observe, update_current_span
from deepeval.metrics import ContextualRelevancyMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden

contextual_relevancy = ContextualRelevancyMetric(threshold=0.6)
answer_relevancy = AnswerRelevancyMetric(threshold=0.6)

def rag_pipeline(query: str):
    @observe(metrics=[contextual_relevancy])
    def retriever(query: str):
        chunks = ["retrieved context 1", "retrieved context 2"]
        update_current_span(test_case=LLMTestCase(input=query, retrieval_context=chunks))
        return chunks

    @observe(metrics=[answer_relevancy])
    def generator(query: str, text_chunks: list):
        output = "generated answer"
        update_current_span(test_case=LLMTestCase(input=query, actual_output=output))
        return output

    chunks = retriever(query)
    return generator(query, chunks)

dataset = EvaluationDataset(goldens=[Golden(input="Test query")])
for golden in dataset.evals_iterator():
    rag_pipeline(golden.input)
```

### Multi-Turn RAG Evaluation

```python
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import TurnRelevancy, TurnFaithfulness
from deepeval import evaluate

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="I'd like to buy a ticket to a Coldplay concert."),
        Turn(
            role="assistant",
            content="Great! Which city would you like to attend?",
            retrieval_context=["Concert cities: New York, Los Angeles, Chicago"]
        ),
        Turn(role="user", content="New York, please."),
        Turn(
            role="assistant",
            content="I found VIP and standard tickets for the Coldplay concert in New York.",
            retrieval_context=["VIP ticket details", "Standard ticket details"]
        )
    ]
)

evaluate([test_case], metrics=[TurnFaithfulness(), TurnRelevancy()])
```

---

## Chatbot Evaluation Quickstart

### Overview

Chatbot evaluation differs from single-turn tasks because conversations span multiple turns. The chatbot must maintain context awareness throughout. In DeepEval, multi-turn interactions are grouped by **scenarios** for standardized benchmarking.

### Key Metrics for Chatbots

| Metric | What It Evaluates |
|--------|-------------------|
| `TurnRelevancyMetric` | Whether each assistant turn is relevant to the user's message |
| `KnowledgeRetentionMetric` | Whether the chatbot retains information across turns |
| `ConversationalGEval` | Custom multi-turn criteria via natural language |

### Minimal Working Example

```python
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric
from deepeval import evaluate

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="Hello, how are you?"),
        Turn(role="assistant", content="I'm doing well, thank you!"),
        Turn(role="user", content="I'd like to buy a ticket to a Coldplay concert."),
        Turn(role="assistant", content="I can help you with that. Which city?"),
    ]
)

evaluate(
    test_cases=[test_case],
    metrics=[TurnRelevancyMetric(), KnowledgeRetentionMetric()]
)
```

### Using ConversationalGEval for Custom Criteria

```python
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationalGEval
from deepeval import assert_test

def test_professionalism():
    professionalism_metric = ConversationalGEval(
        name="Professionalism",
        criteria="Determine whether the assistant has acted professionally.",
        threshold=0.5
    )
    test_case = ConversationalTestCase(
        turns=[
            Turn(role="user", content="What is DeepEval?"),
            Turn(role="assistant", content="DeepEval is an open-source LLM eval package.")
        ]
    )
    assert_test(test_case, [professionalism_metric])
```

### Simulating Conversations from Goldens

The best approach for multi-turn evals is simulating turns from `ConversationalGolden`s, which standardizes the test bench and automates manual prompting.

```python
from deepeval.test_case import Turn
from deepeval.dataset import EvaluationDataset, ConversationalGolden
from deepeval.conversation_simulator import ConversationSimulator
from deepeval.metrics import TurnRelevancyMetric
from deepeval import evaluate
from typing import List
from openai import AsyncOpenAI

# Step 1: Define golden scenarios
golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
)
dataset = EvaluationDataset(goldens=[golden])

# Step 2: Define chatbot callback
client = AsyncOpenAI()

async def model_callback(input: str, turns: List[Turn]) -> Turn:
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(model="gpt-4.1", messages=messages)
    return Turn(role="assistant", content=response.choices[0].message.content)

# Step 3: Simulate turns
simulator = ConversationSimulator(model_callback=model_callback)
conversational_test_cases = simulator.simulate(
    goldens=dataset.goldens,
    max_turns=10
)

# Step 4: Evaluate
evaluate(conversational_test_cases, metrics=[TurnRelevancyMetric()])
```

---

## Agent Evaluation Quickstart

### Overview

AI agents consist of two core layers:
- **Reasoning layer (LLMs)**: Handles planning and decision-making
- **Action layer (tools)**: Executes actions in the real world

### Agent Metrics by Layer

| Layer | Metric | What It Evaluates |
|-------|--------|-------------------|
| Reasoning | `PlanQualityMetric` | Whether the plan is logical, complete, and efficient |
| Reasoning | `PlanAdherenceMetric` | Whether the agent follows its own plan |
| Action | `ToolCorrectnessMetric` | Correct tool selection and calling patterns |
| Action | `ArgumentCorrectnessMetric` | Correct argument generation for tool calls |
| Overall | `TaskCompletionMetric` | Whether the agent successfully accomplishes the task |
| Overall | `StepEfficiencyMetric` | Whether execution avoids unnecessary steps |

### Setting Up LLM Tracing (Required for Agent Evals)

```python
import json
from openai import OpenAI
from deepeval.tracing import observe
from deepeval.dataset import Golden, EvaluationDataset

client = OpenAI()

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
                "properties": {"flight_id": {"type": "string"}},
                "required": ["flight_id"]
            }
        }
    }
]

@observe(type="tool")
def search_flights(origin, destination, date):
    return [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]

@observe(type="tool")
def book_flight(flight_id):
    return {"confirmation": "CONF-789", "flight_id": flight_id}

@observe(type="llm")
def call_openai(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

@observe(type="agent")
def travel_agent(user_input: str):
    messages = [{"role": "user", "content": user_input}]
    response = call_openai(messages)
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    flights = search_flights(args["origin"], args["destination"], args["date"])
    cheapest = min(flights, key=lambda x: x["price"])
    messages.append({"role": "assistant", "content": f"Booking cheapest: {cheapest['id']}"})
    booking = book_flight(cheapest["id"])
    return f"Booked {cheapest['id']} for ${cheapest['price']}. Confirmation: {booking['confirmation']}"
```

**Decorator types:**
- `@observe(type="tool")` — Marks action layer interactions
- `@observe(type="llm")` — Marks reasoning layer decisions
- `@observe(type="agent")` — Marks top-level orchestration
- Type parameter is optional but recommended for better visualization

### Evaluating the Reasoning Layer (End-to-End)

```python
from deepeval.metrics import PlanQualityMetric, PlanAdherenceMetric
from deepeval.dataset import EvaluationDataset, Golden

plan_quality = PlanQualityMetric()
plan_adherence = PlanAdherenceMetric()

dataset = EvaluationDataset(goldens=[
    Golden(input="Book a flight from NYC to London for next Monday")
])

for golden in dataset.evals_iterator(metrics=[plan_quality, plan_adherence]):
    travel_agent(golden.input)
```

### Evaluating the Action Layer (Component-Level)

```python
from deepeval.metrics import ToolCorrectnessMetric, ArgumentCorrectnessMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe

tool_correctness = ToolCorrectnessMetric()
argument_correctness = ArgumentCorrectnessMetric()

@observe(type="llm", metrics=[tool_correctness, argument_correctness])
def call_openai(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

dataset = EvaluationDataset(goldens=[
    Golden(input="What's the weather like in San Francisco?")
])

for golden in dataset.evals_iterator():
    travel_agent(golden.input)
```

### Evaluating Overall Execution

```python
from deepeval.metrics import TaskCompletionMetric, StepEfficiencyMetric
from deepeval.dataset import EvaluationDataset, Golden

task_completion = TaskCompletionMetric()
step_efficiency = StepEfficiencyMetric()

dataset = EvaluationDataset(goldens=[
    Golden(input="Book the cheapest flight from NYC to LA for tomorrow")
])

for golden in dataset.evals_iterator(metrics=[task_completion, step_efficiency]):
    travel_agent(golden.input)
```

**Important:** `TaskCompletionMetric` and `StepEfficiencyMetric` are trace-only and MUST be used with `evals_iterator` or the `@observe` decorator.

### Custom GEval for Agents

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

reasoning_clarity = GEval(
    name="Reasoning Clarity",
    criteria="Evaluate how clearly the agent explains its reasoning and decision-making process before taking actions.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

# End-to-end:
for golden in dataset.evals_iterator(metrics=[reasoning_clarity]):
    travel_agent(golden.input)

# Component-level:
@observe(type="llm", metrics=[reasoning_clarity])
def call_openai(messages):
    ...
```

### Production Agent Evals

Replace local `metrics=[...]` with a `metric_collection` string from Confident AI:

```python
@observe(metric_collection="my-agent-metrics")
def call_openai(messages):
    ...
```

---

## MCP Evaluation Quickstart

### Overview

MCP (Model Context Protocol) is an open-source framework by Anthropic that standardizes how AI systems interact with external tools and data sources. DeepEval evaluates the MCP host on:
- Primitive usage (tools, resources, prompts)
- Argument generation correctness
- Task completion

MCP evaluation supports both single-turn and multi-turn test cases.

### MCP Architecture

| Component | Role |
|-----------|------|
| **Host** | Coordinates and manages one or more MCP clients |
| **Client** | Maintains one-to-one connection with a server; retrieves context for the host |
| **Server** | Paired with single client; provides context the client passes to host |

### Single-Turn MCP Evaluation

```python
import mcp
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from deepeval.test_case import MCPServer, MCPToolCall, LLMTestCase
from deepeval.metrics import MCPUseMetric
from deepeval import evaluate

url = "https://example.com/mcp"
mcp_servers = []
tools_called = []

async def main():
    # Step 1: Create MCP Server reference
    read, write, _ = await AsyncExitStack().enter_async_context(
        streamablehttp_client(url)
    )
    session = await AsyncExitStack().enter_async_context(ClientSession(read, write))
    await session.initialize()
    tool_list = await session.list_tools()
    mcp_servers.append(MCPServer(
        name=url,
        transport="streamable-http",
        available_tools=tool_list.tools,
    ))

    # Step 2: Track MCP interactions
    available_tools = [
        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
        for t in tool_list.tools
    ]
    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=messages,
        tools=available_tools,
    )
    for content in response.content:
        if content.type == "tool_use":
            result = await session.call_tool(content.name, content.input)
            tools_called.append(MCPToolCall(
                name=content.name,
                args=content.input,
                result=result
            ))

    # Step 3: Create test case
    test_case = LLMTestCase(
        input=query,
        actual_output=response_text,
        mcp_servers=mcp_servers,
        mcp_tools_called=tools_called,
    )

    # Step 4: Evaluate
    mcp_use_metric = MCPUseMetric()
    evaluate([test_case], [mcp_use_metric])
```

**What `MCPUseMetric` evaluates:**
- Primitive usage (how well the app utilized MCP capabilities)
- Argument correctness (whether inputs generated for primitive usage were correct)
- Final score = minimum of both scores

### Multi-Turn MCP Evaluation

```python
from deepeval.test_case import MCPToolCall, Turn, ConversationalTestCase
from deepeval.metrics import MultiTurnMCPUseMetric, MCPTaskCompletionMetric
from deepeval import evaluate

# Track tool calls per turn
tool_called = MCPToolCall(name=tool_name, args=tool_args, result=result)
turns.append(Turn(
    role="assistant",
    content=f"Tool call: {tool_name}",
    mcp_tools_called=[tool_called],
))

# Create conversational test case
convo_test_case = ConversationalTestCase(
    turns=turns,
    mcp_servers=mcp_servers
)

# Evaluate
evaluate(
    [convo_test_case],
    [MultiTurnMCPUseMetric(), MCPTaskCompletionMetric()]
)
```

**Multi-turn MCP metrics:**
- `MultiTurnMCPUseMetric` — Evaluates primitive usage and argument generation across all turns
- `MCPTaskCompletionMetric` — Evaluates whether all interactions satisfied their tasks

### Viewing Results

```bash
deepeval view  # upload from local cache to Confident AI
```

---

## LLM Arena Evaluation Quickstart

### Overview

Instead of single-output LLM-as-a-Judge, Arena comparisons evaluate n-pairwise test cases to identify the best version of your LLM application. Arena evaluation does NOT produce numerical scores — it picks a winner among contestants.

**Key difference:** Unlike other metrics, the concept of a "passing" test case does not exist for arena evaluations.

### Arena Test Case Structure

- `ArenaTestCase` contains multiple `Contestant` objects
- Each `Contestant` has a `name`, optional `hyperparameters`, and an `LLMTestCase`
- DeepEval masks contestant names and randomizes positions to eliminate bias

### Minimal Working Example

```python
from deepeval.test_case import ArenaTestCase, LLMTestCase, Contestant, LLMTestCaseParams
from deepeval.metrics import ArenaGEval
from deepeval import compare

# Define contestants (different versions of your LLM app)
contestant_1 = Contestant(
    name="Version 1",
    hyperparameters={"model": "gpt-3.5-turbo"},
    test_case=LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
    ),
)
contestant_2 = Contestant(
    name="Version 2",
    hyperparameters={"model": "gpt-4o"},
    test_case=LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
    ),
)
contestant_3 = Contestant(
    name="Version 3",
    hyperparameters={"model": "gpt-4.1"},
    test_case=LLMTestCase(
        input="What is the capital of France?",
        actual_output="Absolutely! The capital of France is Paris",
    ),
)

arena_test_case = ArenaTestCase(contestants=[contestant_1, contestant_2, contestant_3])

# Define arena metric (ArenaGEval is the ONLY metric compatible with ArenaTestCase)
arena_geval = ArenaGEval(
    name="Friendly",
    criteria="Choose the more friendly contestant based on the input and actual output",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ]
)

# Run comparison
compare(test_cases=[arena_test_case], metric=arena_geval)
# Expected output: Counter({'Version 3': 1})
```

Run with:
```bash
python main.py
```

### Logging Prompts and Models

```python
from deepeval.prompt import Prompt, PromptMessage

prompt_1 = Prompt(
    alias="First Prompt",
    messages_template=[PromptMessage(role="system", content="You are a helpful assistant.")]
)
prompt_2 = Prompt(
    alias="Second Prompt",
    messages_template=[PromptMessage(role="system", content="You are a friendly assistant.")]
)

compare(
    test_cases=[arena_test_case],
    metric=arena_geval,
    hyperparameters={
        "Version 1": {"prompt": prompt_1},
        "Version 2": {"prompt": prompt_2},
    },
)
```

### Arena vs Standard Metrics

| Feature | Arena (ArenaGEval) | Standard Metrics |
|---------|-------------------|------------------|
| Output | Winner name | Numerical score (0-1) |
| Pass/fail | No concept of passing | Score >= threshold |
| Use case | Compare N versions | Evaluate single version |
| Bias control | Names masked, positions randomized | N/A |
| Function | `compare()` | `evaluate()` or `assert_test()` |

---

## Cross-Reference

- For model configuration for any metric, see `40-integrations.md`
- For custom LLM evaluators, see `30-custom-models-and-embeddings.md`
- For ConversationSimulator details, see `../09-others/20-conversation-simulator.md`
- For CLI and environment variable setup, see `../09-others/10-cli-and-environment.md`
