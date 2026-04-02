# Conversation Simulator

## Read This When
- You need to generate multi-turn conversational test cases from scenario descriptions using `ConversationSimulator`
- You want framework-specific callback examples (OpenAI, LangChain, LlamaIndex, OpenAI Agents SDK, Pydantic AI)
- You need to evaluate a chatbot but lack pre-recorded conversations and want automated scenario-driven simulation

## Skip This When
- You want to generate single-turn synthetic goldens (not full conversations) -- see [../05-synthetic-data/10-synthesizer-overview.md](../05-synthetic-data/10-synthesizer-overview.md)
- You need CLI commands or environment variable configuration -- see [10-cli-and-environment.md](./10-cli-and-environment.md)

---

The `ConversationSimulator` enables simulation of complete multi-turn dialogues between a simulated user and a chatbot. It is distinct from the `Synthesizer` (which generates individual LLM interactions) — the `ConversationSimulator` generates full conversational test cases from scenario descriptions.

---

## When to Use ConversationSimulator

Use `ConversationSimulator` when:
- You want to evaluate a chatbot but do not have pre-recorded conversations
- You want to standardize test scenarios across different chatbot versions
- You want to automate the manual effort of prompting a chatbot through multiple turns

Benefits over ad-hoc manual evaluation:
- Standardizes the test bench via `scenario` and `expected_outcome`
- Automates multi-turn prompting (which can take hours manually)
- Works with any async chatbot callback

---

## Core Classes

### `ConversationSimulator`

```python
from deepeval.simulator import ConversationSimulator

simulator = ConversationSimulator(
    model_callback=chatbot_callback,
    simulator_model="gpt-4.1",   # optional
    async_mode=True,              # optional
    max_concurrent=100            # optional
)
```

**Constructor parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_callback` | Callable | Yes | — | Async function that wraps your chatbot; generates the next assistant turn |
| `simulator_model` | string or `DeepEvalBaseLLM` | No | `"gpt-4.1"` | LLM used to simulate user turns |
| `async_mode` | bool | No | `True` | Enables concurrent conversation simulation |
| `max_concurrent` | int | No | `100` | Maximum number of parallel conversations when `async_mode=True` |

---

### `ConversationalGolden`

The input unit for the simulator. Defines a scenario, the expected outcome, and optionally the user persona.

```python
from deepeval.dataset import ConversationalGolden

golden = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
)
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `scenario` | string | Recommended | Description of the conversation's starting context and goal |
| `expected_outcome` | string | Recommended | The desired end state; simulation stops when achieved |
| `user_description` | string | No | Persona description for the simulated user |
| `turns` | `List[Turn]` | No | Pre-existing turns to initialize the conversation (e.g., a hardcoded assistant greeting) |

---

### `model_callback` Function

The callback wraps your chatbot and generates the next assistant `Turn` given the current user input and conversation history.

```python
from deepeval.test_case import Turn
from typing import List

async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    # input: the current user message
    # turns: all prior turns in the conversation
    # thread_id: unique identifier for this conversation (for stateful chatbots)
    response = await your_chatbot(input, turns, thread_id)
    return Turn(role="assistant", content=response)
```

**Callback arguments:**

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `input` | string | Yes | Current user input message |
| `turns` | `List[Turn]` | No | All prior dialogue turns (role + content pairs) |
| `thread_id` | string | No | Unique conversation ID for state persistence across turns |

**Important:** Your callback must be `async`. It must return a `Turn` object with `role="assistant"`.

---

## `simulate()` Method

```python
conversational_test_cases = simulator.simulate(
    conversational_goldens=[golden1, golden2],
    max_user_simulations=10,
    on_simulation_complete=my_callback  # optional
)
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `conversational_goldens` | `List[ConversationalGolden]` | Yes | — | List of scenario definitions to simulate |
| `max_user_simulations` | int | No | `10` | Maximum user-assistant cycles per conversation |
| `on_simulation_complete` | Callable | No | None | Hook called when each conversation completes |

**Simulation end conditions:**
1. The conversation achieves the `expected_outcome` defined in the `ConversationalGolden`
2. The `max_user_simulations` limit is reached

**Returns:** `List[ConversationalTestCase]` — one test case per golden, each containing all simulated turns.

---

## Complete Usage Example

```python
import asyncio
from openai import AsyncOpenAI
from deepeval.test_case import Turn
from deepeval.simulator import ConversationSimulator
from deepeval.dataset import EvaluationDataset, ConversationalGolden
from deepeval.metrics import TurnRelevancyMetric
from deepeval import evaluate
from typing import List

# Step 1: Define scenarios
golden1 = ConversationalGolden(
    scenario="Andy Byron wants to purchase a VIP ticket to a Coldplay concert.",
    expected_outcome="Successful purchase of a ticket.",
    user_description="Andy Byron is the CEO of Astronomer.",
)
golden2 = ConversationalGolden(
    scenario="Maria wants to get a refund for a cancelled concert.",
    expected_outcome="Refund successfully initiated.",
    user_description="Maria is an angry customer.",
)
dataset = EvaluationDataset(goldens=[golden1, golden2])

# Optionally push to Confident AI for team collaboration
# dataset.push(alias="Ticket Chatbot Scenarios")

# Step 2: Define the chatbot callback
client = AsyncOpenAI()

async def chatbot_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant."},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(model="gpt-4.1", messages=messages)
    return Turn(role="assistant", content=response.choices[0].message.content)

# Step 3: Simulate conversations
simulator = ConversationSimulator(model_callback=chatbot_callback)
conversational_test_cases = simulator.simulate(
    conversational_goldens=dataset.goldens,
    max_user_simulations=10
)

# Step 4: Evaluate
evaluate(
    test_cases=conversational_test_cases,
    metrics=[TurnRelevancyMetric()]
)
```

---

## Framework-Specific Callback Examples

### OpenAI (stateless)

```python
from openai import AsyncOpenAI
from deepeval.test_case import Turn
from typing import List

client = AsyncOpenAI()

async def model_callback(input: str, turns: List[Turn]) -> Turn:
    messages = [
        {"role": "system", "content": "You are a ticket purchasing assistant"},
        *[{"role": t.role, "content": t.content} for t in turns],
        {"role": "user", "content": input},
    ]
    response = await client.chat.completions.create(model="gpt-4.1", messages=messages)
    return Turn(role="assistant", content=response.choices[0].message.content)
```

### LangChain (stateful via session)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from deepeval.test_case import Turn

store = {}
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a ticket purchasing assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
chain_with_history = RunnableWithMessageHistory(
    prompt | llm,
    lambda session_id: store.setdefault(session_id, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="history"
)

async def model_callback(input: str, thread_id: str) -> Turn:
    response = chain_with_history.invoke(
        {"input": input},
        config={"configurable": {"session_id": thread_id}}
    )
    return Turn(role="assistant", content=response.content)
```

### LlamaIndex

```python
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from deepeval.test_case import Turn

chat_store = SimpleChatStore()
llm = OpenAI(model="gpt-4")

async def model_callback(input: str, thread_id: str) -> Turn:
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=thread_id
    )
    chat_engine = SimpleChatEngine.from_defaults(llm=llm, memory=memory)
    response = chat_engine.chat(input)
    return Turn(role="assistant", content=response.response)
```

### OpenAI Agents SDK

```python
from agents import Agent, Runner, SQLiteSession
from deepeval.test_case import Turn

sessions = {}
agent = Agent(
    name="Ticket Assistant",
    instructions="You are a helpful ticket purchasing assistant."
)

async def model_callback(input: str, thread_id: str) -> Turn:
    if thread_id not in sessions:
        sessions[thread_id] = SQLiteSession(thread_id)
    session = sessions[thread_id]
    result = await Runner.run(agent, input, session=session)
    return Turn(role="assistant", content=result.final_output)
```

### Pydantic AI

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from deepeval.test_case import Turn
from datetime import datetime
from typing import List

agent = Agent(
    "openai:gpt-4",
    system_prompt="You are a helpful assistant."
)

async def model_callback(input: str, turns: List[Turn]) -> Turn:
    message_history = []
    for turn in turns:
        if turn.role == "user":
            message_history.append(ModelRequest(
                parts=[UserPromptPart(content=turn.content, timestamp=datetime.now())],
                kind="request"
            ))
        elif turn.role == "assistant":
            message_history.append(ModelResponse(
                parts=[TextPart(content=turn.content)],
                model_name="gpt-4",
                timestamp=datetime.now(),
                kind="response"
            ))
    result = await agent.run(input, message_history=message_history)
    return Turn(role="assistant", content=result.output)
```

---

## Advanced Usage

### Lifecycle Hook: `on_simulation_complete`

Process individual test cases as they complete (useful when `async_mode=True` and conversations finish in non-deterministic order):

```python
from deepeval.test_case import ConversationalTestCase

def handle_simulation_complete(test_case: ConversationalTestCase, index: int):
    print(f"Conversation {index} completed with {len(test_case.turns)} turns")
    # could save to database, run quick checks, etc.

conversational_test_cases = simulator.simulate(
    conversational_goldens=[golden1, golden2, golden3],
    on_simulation_complete=handle_simulation_complete
)
```

**Hook parameters:**
- `test_case` — The completed `ConversationalTestCase` with all turns and metadata
- `index` — The index of the corresponding `ConversationalGolden` (ordering preserved)

**Note:** With `async_mode=True`, conversations complete in any order despite concurrent execution.

### Starting with Pre-existing Turns

Provide `turns` in the golden to initialize conversations with predefined content (e.g., a hardcoded opening assistant message):

```python
from deepeval.test_case import Turn

golden = ConversationalGolden(
    scenario="User wants to return a product.",
    expected_outcome="Return successfully initiated.",
    turns=[
        Turn(role="assistant", content="Hi! Welcome to support. How can I help you today?")
    ]
)
```

DeepEval continues simulation from the provided context.

---

## Evaluating Simulated Conversations

The output of `simulator.simulate()` is a `List[ConversationalTestCase]`, which can be passed directly to `evaluate()` or `assert_test()`:

```python
from deepeval import evaluate
from deepeval.metrics import TurnRelevancyMetric, KnowledgeRetentionMetric

evaluate(
    test_cases=conversational_test_cases,
    metrics=[TurnRelevancyMetric(), KnowledgeRetentionMetric()]
)
```

Or in pytest:
```python
from deepeval import assert_test

def test_chatbot():
    # ... generate conversational_test_cases
    for test_case in conversational_test_cases:
        assert_test(test_case, [TurnRelevancyMetric()])
```

---

## Integration with EvaluationDataset

Push goldens to Confident AI for team collaboration:

```python
from deepeval.dataset import EvaluationDataset, ConversationalGolden

dataset = EvaluationDataset(goldens=[
    ConversationalGolden(scenario="Angry user asking for a refund"),
    ConversationalGolden(scenario="Couple booking two VIP Coldplay tickets")
])

# Push to Confident AI
dataset.push(alias="Chatbot Evaluation Dataset v1")

# Pull later
dataset.pull(alias="Chatbot Evaluation Dataset v1")
```

---

## Related Reference Files

- `../01-getting-started/20-quickstart-by-usecase.md` — Chatbot evaluation quickstart with ConversationSimulator example
- `10-cli-and-environment.md` — Environment variable configuration
- `30-data-privacy-and-misc.md` — Data privacy and miscellaneous settings
