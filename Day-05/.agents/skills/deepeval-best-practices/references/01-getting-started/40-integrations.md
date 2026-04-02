# Integrations

## Read This When
- Setting up a native model provider (OpenAI, Anthropic, Gemini, Grok, Azure, Bedrock, Ollama, OpenRouter, LiteLLM) as the evaluation judge
- Integrating DeepEval with a framework (OpenAI Agents SDK, Anthropic SDK, LangChain, LangGraph, HuggingFace Trainer)
- Evaluating RAG retriever quality with a vector database (Chroma, Elasticsearch, PGVector, Qdrant)

## Skip This When
- Need to implement a fully custom `DeepEvalBaseLLM` subclass for an unsupported model → `references/01-getting-started/30-custom-models-and-embeddings.md`
- Looking for use-case-specific quickstart code rather than provider setup → `references/01-getting-started/20-quickstart-by-usecase.md`

---

DeepEval provides native integrations with model providers, LLM frameworks, and vector databases. This file covers all available integrations with setup code and parameter references.

---

## Model Provider Integrations

All model classes accept a `model` parameter when instantiating metrics:

```python
from deepeval.metrics import AnswerRelevancyMetric
metric = AnswerRelevancyMetric(model=<model_instance>)
```

### OpenAI

DeepEval uses `gpt-4.1` by default. The `GPTModel` class provides full control over the OpenAI configuration.

**API key setup:**
```bash
export OPENAI_API_KEY=<your-openai-api-key>
# or in .env.local:
OPENAI_API_KEY=<your-openai-api-key>
```

**CLI configuration:**
```bash
deepeval set-openai \
    --model=gpt-4.1 \
    --cost-per-input-token=0.000002 \
    --cost-per-output-token=0.000008

# Unset
deepeval unset-openai
```

**Python:**
```python
from deepeval.models import GPTModel
from deepeval.metrics import AnswerRelevancyMetric

model = GPTModel(
    model="gpt-4.1",
    temperature=0,
    cost_per_input_token=0.000002,
    cost_per_output_token=0.000008
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`GPTModel` parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `OPENAI_MODEL_NAME` or `gpt-4.1` | Model name |
| `api_key` | string | Optional | `OPENAI_API_KEY` | Authentication key |
| `base_url` | string | Optional | None | Custom OpenAI-compatible URL |
| `temperature` | float | Optional | `0.0` | Model temperature |
| `cost_per_input_token` | float | Optional | None | Input token cost (USD per token) |
| `cost_per_output_token` | float | Optional | None | Output token cost (USD per token) |
| `generation_kwargs` | dict | Optional | None | Additional API call parameters |

**Available OpenAI models:**
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- `gpt-4.1`, `gpt-4.5-preview`, `gpt-4o`, `gpt-4o-mini`
- `o1`, `o1-pro`, `o1-mini`, `o3-mini`
- `gpt-4-turbo`, `gpt-4`, `gpt-4-32k`
- `gpt-3.5-turbo` variants

---

### Anthropic

**API key setup:**
```bash
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

**Python:**
```python
from deepeval.models import AnthropicModel
from deepeval.metrics import AnswerRelevancyMetric

model = AnthropicModel(
    model="claude-3-7-sonnet-latest",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`AnthropicModel` parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `claude-3-7-sonnet-latest` | Claude model name |
| `api_key` | string | Optional | `ANTHROPIC_API_KEY` | Anthropic API key |
| `temperature` | float | Optional | `0.0` | Model temperature (must be >= 0) |
| `cost_per_input_token` | float | Optional | Registry value | Input token cost |
| `cost_per_output_token` | float | Optional | Registry value | Output token cost |
| `generation_kwargs` | dict | Optional | None | Additional generation parameters (e.g., `max_tokens`) |

**Available Anthropic models:**
- `claude-3-7-sonnet-latest`
- `claude-3-5-haiku-latest`
- `claude-3-5-sonnet-latest`
- `claude-3-opus-latest`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-instant-1.2`

---

### Gemini

**CLI configuration:**
```bash
deepeval set-gemini --model="gemini-2.5-flash"

# Unset
deepeval unset-gemini
```

**Python:**
```python
from deepeval.models import GeminiModel
from deepeval.metrics import AnswerRelevancyMetric

model = GeminiModel(
    model="gemini-2.5-pro",
    api_key="your-google-api-key",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`GeminiModel` parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `GEMINI_MODEL_NAME` | Gemini model identifier |
| `api_key` | string | Optional | `GOOGLE_API_KEY` | Google API key (mandatory at runtime unless Vertex AI) |
| `temperature` | float | Optional | `0.0` | Model temperature |
| `generation_kwargs` | dict | Optional | None | Additional Gemini API parameters |

**For Vertex AI (no API key needed):**
```python
model = GeminiModel(
    model="gemini-1.5-pro",
    project="your-project-id",
    location="us-central1",
    temperature=0
)
```

**Available Gemini models:**
- `gemini-3-pro-preview`, `gemini-3-flash-preview`
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `gemini-2.0-flash`, `gemini-2.0-flash-lite`
- `gemini-pro-latest`, `gemini-flash-latest`

---

### Grok (xAI)

**Installation:**
```bash
pip install xai-sdk
```

**CLI configuration:**
```bash
deepeval set-grok --model grok-4.1 --temperature=0

# Unset
deepeval unset-grok
```

**Python:**
```python
from deepeval.models import GrokModel
from deepeval.metrics import AnswerRelevancyMetric

model = GrokModel(
    model="grok-4.1",
    api_key="your-xai-api-key",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`GrokModel` parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `GROK_MODEL_NAME` | Grok model identifier |
| `api_key` | string | Optional | `GROK_API_KEY` | xAI API key |
| `temperature` | float | Optional | `0.0` | Sampling temperature |
| `cost_per_input_token` | float | Optional | Registry value | Input token cost |
| `cost_per_output_token` | float | Optional | Registry value | Output token cost |
| `generation_kwargs` | dict | Optional | None | Additional xAI SDK parameters |

**Available Grok models:**
- `grok-4.1`, `grok-4`, `grok-4-heavy`, `grok-4-fast`
- `grok-beta`, `grok-3`, `grok-2`, `grok-2-mini`
- `grok-code-fast-1`

---

### Azure OpenAI

**CLI configuration:**
```bash
deepeval set-azure-openai \
    --base-url="https://your-resource.azure.openai.com/" \
    --model-name="gpt-4.1" \
    --deployment-name="your-deployment" \
    --api-version="2025-01-01-preview"

# Unset
deepeval unset-azure-openai
```

**Python:**
```python
from deepeval.models import AzureOpenAIModel
from deepeval.metrics import AnswerRelevancyMetric

model = AzureOpenAIModel(
    model="gpt-4.1",
    deployment_name="your-deployment",
    api_key="your-azure-api-key",
    api_version="2025-01-01-preview",
    base_url="https://your-resource.azure.openai.com/",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`AzureOpenAIModel` parameters (all optional — values fall back to env vars):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `AZURE_MODEL_NAME` | Azure OpenAI model name |
| `api_key` | string | `AZURE_OPENAI_API_KEY` | Azure API key |
| `azure_ad_token` | string | `AZURE_OPENAI_AD_TOKEN` | Azure AD token |
| `azure_ad_token_provider` | callback | None | AsyncAzureADTokenProvider |
| `base_url` | string | `AZURE_OPENAI_ENDPOINT` | Endpoint URL |
| `temperature` | float | `0.0` | Model temperature |
| `cost_per_input_token` | float | Registry or None | Input token cost |
| `cost_per_output_token` | float | Registry or None | Output token cost |
| `deployment_name` | string | `AZURE_DEPLOYMENT_NAME` | Deployment name |
| `api_version` | string | `OPENAI_API_VERSION` | API version |
| `generation_kwargs` | dict | None | Additional generation parameters |

**Supported Azure OpenAI models:**
- `gpt-4.1`, `gpt-4.5-preview`, `gpt-4o`, `gpt-4o-mini`
- `gpt-4`, `gpt-4-32k`, `gpt-35-turbo`, `gpt-35-turbo-16k`
- `o1`, `o1-mini`, `o1-preview`, `o3-mini`

---

### Amazon Bedrock

**Installation:** DeepEval will prompt you to install `aiobotocore` and `botocore` if missing.

**API key setup:**
```bash
export AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
export AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>
```

**Python:**
```python
from deepeval.models import AmazonBedrockModel
from deepeval.metrics import AnswerRelevancyMetric

model = AmazonBedrockModel(
    model="anthropic.claude-3-opus-20240229-v1:0",
    region="us-east-1",
    generation_kwargs={"temperature": 0}
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`AmazonBedrockModel` parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `AWS_BEDROCK_MODEL_NAME` | Bedrock model identifier |
| `region` | string | Optional | `AWS_BEDROCK_REGION` | AWS region (e.g., `us-east-1`) |
| `aws_access_key_id` | string | Optional | env var / credentials chain | AWS access key |
| `aws_secret_access_key` | string | Optional | env var / credentials chain | AWS secret key |
| `cost_per_input_token` | float | Optional | None | Input token cost (USD) |
| `cost_per_output_token` | float | Optional | None | Output token cost (USD) |
| `generation_kwargs` | dict | Optional | None | Inference config (temperature, topP, maxTokens) |

**Common Bedrock models:**
- Claude 3 variants (Opus, Sonnet, Haiku)
- Amazon Titan, Amazon Nova
- Meta Llama, Mistral, OpenAI models

---

### Ollama (Local)

**Prerequisites:** Ensure Ollama is installed and the model is running:
```bash
ollama run deepseek-r1:1.5b
```

**CLI configuration:**
```bash
deepeval set-ollama --model=deepseek-r1:1.5b

# Custom port
deepeval set-ollama --model=deepseek-r1:1.5b --base-url="http://localhost:11434"

# Unset
deepeval unset-ollama
```

**Python:**
```python
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric

model = OllamaModel(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`OllamaModel` parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `OLLAMA_MODEL_NAME` | Ollama model name |
| `base_url` | string | Optional | `LOCAL_MODEL_BASE_URL` or `http://localhost:11434` | Ollama server URL |
| `temperature` | float | Optional | `TEMPERATURE` or `0.0` | Model temperature |
| `generation_kwargs` | dict | Optional | None | Additional Ollama chat parameters |

**Common Ollama models:** `deepseek-r1`, `llama3.1`, `gemma`, `qwen`, `mistral`, `codellama`, `phi3`, `tinyllama`, `starcoder2`

---

### OpenRouter

**CLI configuration:**
```bash
deepeval set-openrouter \
    --model "openai/gpt-4.1" \
    --base-url "https://openrouter.ai/api/v1" \
    --temperature=0 \
    --prompt-api-key

# Unset
deepeval unset-openrouter
```

**Python:**
```python
from deepeval.models import OpenRouterModel
from deepeval.metrics import AnswerRelevancyMetric

model = OpenRouterModel(
    model="openai/gpt-4.1",
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "My eval pipeline",
    },
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`OpenRouterModel` parameters (all optional):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `OPENROUTER_MODEL_NAME` or `openai/gpt-4.1` | Provider-prefixed model name |
| `api_key` | `OPENROUTER_API_KEY` | OpenRouter API key |
| `base_url` | `OPENROUTER_BASE_URL` or `https://openrouter.ai/api/v1` | API base URL |
| `temperature` | `TEMPERATURE` or `0.0` | Sampling temperature |
| `cost_per_input_token` | None | Input token cost |
| `cost_per_output_token` | None | Output token cost |
| `generation_kwargs` | None | Additional generation parameters |

Additional `**kwargs` are forwarded to the underlying OpenAI client constructor.

---

### LiteLLM

Provides access to 100+ LLMs through a unified interface.

**Installation:**
```bash
pip install litellm
```

**CLI configuration:**
```bash
# Specify provider in model name
deepeval set-litellm --model=openai/gpt-3.5-turbo
deepeval set-litellm --model=anthropic/claude-3-opus
deepeval set-litellm --model=google/gemini-pro

# With custom endpoint
deepeval set-litellm \
    --model=openai/gpt-3.5-turbo \
    --base-url="https://your-custom-endpoint.com"

# Unset
deepeval unset-litellm
```

**Python:**
```python
from deepeval.models import LiteLLMModel
from deepeval.metrics import AnswerRelevancyMetric

# OpenAI via LiteLLM
model = LiteLLMModel(model="openai/gpt-3.5-turbo", api_key="your-key", temperature=0)

# Anthropic via LiteLLM
model = LiteLLMModel(model="anthropic/claude-3-opus")

# LM Studio (local)
model = LiteLLMModel(
    model="lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Ollama (local)
model = LiteLLMModel(
    model="ollama/llama2",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

answer_relevancy = AnswerRelevancyMetric(model=model)
```

**`LiteLLMModel` parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Required | Provider-prefixed model (e.g., `openai/gpt-3.5-turbo`) |
| `api_key` | string | Optional | API authentication key |
| `base_url` | string | Optional | Custom endpoint URL |
| `temperature` | float | Optional | Model temperature |
| `generation_kwargs` | dict | Optional | Additional generation parameters |

**Environment variables for LiteLLM:**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export LITELLM_API_BASE="https://your-custom-endpoint.com"
```

---

### vLLM (Local High-Performance Inference)

**Setup:**
```bash
# Start vLLM server with OpenAI-compatible API
# (typical URL: http://localhost:8000/v1/)

deepeval set-local-model \
    --model=<model_name> \
    --base-url="http://localhost:8000/v1/"

# Unset
deepeval unset-local-model
```

Note: You may enter any value for the API key prompt if authentication is not required.

---

## Framework Integrations

### OpenAI Agents SDK

DeepEval provides a drop-in replacement `OpenAI` client that automatically traces and evaluates API calls.

**End-to-End Evaluation (Chat Completions):**
```python
from deepeval.openai import OpenAI
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import trace, LlmSpanContext

client = OpenAI()  # DeepEval's wrapper, not openai.OpenAI

dataset = EvaluationDataset(goldens=[
    Golden(input="What is the weather in Paris?"),
])

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

**Async version:**
```python
import asyncio
from deepeval.openai import AsyncOpenAI
from deepeval.tracing import trace, LlmSpanContext
from deepeval.metrics import AnswerRelevancyMetric

async_client = AsyncOpenAI()

async def openai_llm_call(input, golden):
    with trace(llm_span_context=LlmSpanContext(metrics=[AnswerRelevancyMetric()])):
        return await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": input}],
        )

for golden in dataset.evals_iterator():
    task = asyncio.create_task(openai_llm_call(golden.input, golden))
    dataset.evaluate(task)
```

**`LlmSpanContext` optional parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | List[BaseMetric] | Metrics to evaluate the generation |
| `expected_output` | string | Expected output |
| `retrieval_context` | List[str] | Retrieved contexts passed to generation |
| `context` | List[str] | Ideal retrieved contexts |
| `expected_tools` | List[str] | Expected tools called |
| `metric_collection` | string | Confident AI collection name (production use) |

**Component-Level with @observe:**
```python
from deepeval.tracing import observe
from deepeval.openai import OpenAI

client = OpenAI()

@observe()
def retrieve_docs(query):
    return ["Paris is the capital of France.", "Paris has 2 million inhabitants."]

@observe()
def llm_app(input):
    with trace(llm_span_context=LlmSpanContext(metrics=[AnswerRelevancyMetric()])):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": '\n'.join(retrieve_docs(input)) + "\n\nQuestion: " + input}
            ]
        )
    return response.choices[0].message.content
```

When used inside `@observe`, DeepEval's OpenAI client automatically:
- Generates LLM spans for every API call (including nested Tool spans)
- Attaches `LLMTestCase` to each span capturing inputs, outputs, and tools
- Records span-level attributes (prompt, output, token usage)
- Logs hyperparameters (model name, system prompt)

**Online (Production) Evals:**
```python
with trace(llm_span_context=LlmSpanContext(metric_collection="my-collection")):
    client.chat.completions.create(model="gpt-4o", messages=[...])
```

---

### Anthropic Framework

DeepEval provides a drop-in `Anthropic` client wrapper for evaluating Claude applications.

**Synchronous:**
```python
from deepeval.anthropic import Anthropic
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset
from deepeval.tracing import trace, LlmSpanContext

client = Anthropic()
dataset = EvaluationDataset()
dataset.pull(alias="My Dataset")

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
            messages=[{"role": "user", "content": golden.input}],
        )
```

**Asynchronous:**
```python
import asyncio
from deepeval.anthropic import AsyncAnthropic

async_client = AsyncAnthropic()

async def llm_app(input, golden):
    with trace(llm_span_context=LlmSpanContext(metrics=[AnswerRelevancyMetric()])):
        response = await async_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": input}],
        )
    return response.content[0].text

for golden in dataset.evals_iterator():
    task = asyncio.create_task(llm_app(golden.input, golden))
    dataset.evaluate(task)
```

**Production:**
```python
from deepeval.anthropic import Anthropic
from deepeval.tracing import trace, LlmSpanContext

client = Anthropic()
with trace(llm_span_context=LlmSpanContext(metric_collection="my-collection")):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "Hello!"}],
    )
```

---

### LangChain

Uses a `CallbackHandler` passed to LangChain's `invoke` method.

**Important limitation:** Only `TaskCompletionMetric` is supported for the LangChain integration. For other metrics, manually set up `@observe` tracing instead.

**End-to-End:**
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import EvaluationDataset, Golden

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

# Synchronous evaluation
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

**Asynchronous evaluation:**
```python
import asyncio

for golden in dataset.evals_iterator():
    task = asyncio.create_task(
        agent_executor.ainvoke(
            {"input": golden.input},
            config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
        )
    )
    dataset.evaluate(task)
```

**Component-level (LLM):**
```python
from langchain_openai import ChatOpenAI
from deepeval.metrics import AnswerRelevancyMetric

llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric": [AnswerRelevancyMetric()]}
).bind_tools([get_weather])
```

**Component-level (Tool):**
```python
from deepeval.integrations.langchain import tool  # DeepEval's tool, not langchain's
from deepeval.metrics import AnswerRelevancyMetric

@tool(metric=[AnswerRelevancyMetric()])
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    return f"It's always sunny in {location}!"
```

**Production:**
```python
agent_executor.invoke(
    {"input": "What is 8 multiplied by 6?"},
    config={"callbacks": [CallbackHandler(metric_collection="my-task-completion-collection")]}
)
```

---

### LangGraph

Uses the same `CallbackHandler` from `deepeval.integrations.langchain`.

**Important limitation:** Only `TaskCompletionMetric` is supported for the LangGraph integration.

**End-to-End:**
```python
from langgraph.prebuilt import create_react_agent
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric
from deepeval.dataset import EvaluationDataset, Golden

def get_weather(city: str) -> str:
    """Returns the weather in a city"""
    return f"It's always sunny in {city}!"

agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

dataset = EvaluationDataset(goldens=[
    Golden(input="What is the weather in Bogotá, Colombia?"),
    Golden(input="What is the weather in Paris, France?"),
])

for golden in dataset.evals_iterator():
    agent.invoke(
        input={"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler(metrics=[TaskCompletionMetric()])]}
    )
```

**Component-level (LLM metadata):**
```python
from langchain_openai import ChatOpenAI
from deepeval.metrics import AnswerRelevancyMetric

llm = ChatOpenAI(
    model="gpt-4o-mini",
    metadata={"metric": [AnswerRelevancyMetric()]}
).bind_tools([get_weather])
```

**Component-level (Tool):**
```python
from deepeval.integrations.langchain import tool
from deepeval.metrics import AnswerRelevancyMetric

@tool(metric=[AnswerRelevancyMetric()])
def get_weather(location: str) -> str:
    """Get the current weather in a location."""
    return f"It's always sunny in {location}!"
```

---

### HuggingFace (Fine-tuning Callback)

DeepEval integrates with `transformers.Trainer` via `DeepEvalHuggingFaceCallback`, enabling real-time evaluation during model fine-tuning at each epoch.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from deepeval.test_case import LLMTestCaseParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import GEval
from deepeval.integrations.hugging_face import DeepEvalHuggingFaceCallback

# Prepare data
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
training_dataset = load_dataset("text", data_files={"train": "train.txt"})

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = training_dataset.map(tokenize_function, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

llm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
trainer = Trainer(
    model=llm,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# DeepEval evaluation setup
dataset = EvaluationDataset(goldens=[
    Golden(input="First input..."),
    Golden(input="Second input...")
])

coherence_metric = GEval(
    name="Coherence",
    criteria="Determine if the actual output is coherent with the input.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

# Add callback and train
callback = DeepEvalHuggingFaceCallback(
    evaluation_dataset=dataset,
    metrics=[coherence_metric],
    trainer=trainer
)
trainer.add_callback(callback)
trainer.train()
```

**`DeepEvalHuggingFaceCallback` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | list | DeepEval evaluation metrics |
| `evaluation_dataset` | EvaluationDataset | Dataset for evaluation (uses goldens, not test cases) |
| `aggregation_method` | string | Score aggregation: `'avg'`, `'min'`, or `'max'` |
| `trainer` | Trainer | HuggingFace transformers trainer instance |
| `tokenizer_args` | dict | Tokenizer configuration arguments |

Evaluations run on the entire dataset at each epoch conclusion.

---

## Vector Database Integrations

Vector databases serve as retrievers in RAG pipelines. DeepEval evaluates retriever performance using contextual metrics: `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`.

The standard evaluation pattern for all vector databases:
1. Retrieve context from vector DB for a query
2. Generate LLM response using retrieved context
3. Create `LLMTestCase` with `input`, `actual_output`, `retrieval_context`, `expected_output`
4. Run `evaluate()` with contextual metrics

### Chroma

Chroma is also a required dependency for `Synthesizer.generate_goldens_from_docs()`, which uses it as the built-in backend for chunk storage.

```bash
pip install chromadb
```

```python
import chromadb
from sentence_transformers import SentenceTransformer
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval import evaluate

# Setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_documents")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Store documents
document_chunks = ["Chroma is an open-source vector database...", "It enables fast semantic search..."]
for i, chunk in enumerate(document_chunks):
    embedding = embedding_model.encode(chunk).tolist()
    collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"text": chunk}])

# Retrieve and evaluate
def search(query, n_results=3):
    query_embedding = embedding_model.encode(query).tolist()
    res = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return res["metadatas"][0][0]["text"] if res["metadatas"][0] else None

query = "How does Chroma work?"
retrieval_context = search(query)
actual_output = your_llm_generate(query, retrieval_context)  # your generation function

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=[retrieval_context],
    expected_output="Chroma is an open-source vector database that enables fast retrieval using cosine similarity."
)

evaluate(
    [test_case],
    metrics=[ContextualPrecisionMetric(), ContextualRecallMetric(), ContextualRelevancyMetric()]
)
```

**Tuning tip:** Adjust `n_results` (top-K) and document chunk size to improve Contextual Relevancy scores.

---

### Elasticsearch

```bash
pip install elasticsearch
```

```python
import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval import evaluate

# Connect
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")))
index_name = "rag_documents"

# Create index
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 384}
            }
        }
    })

# Index documents
model = SentenceTransformer("all-MiniLM-L6-v2")
document_chunks = ["Elasticsearch is a distributed search engine.", "RAG improves responses with context."]
for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()
    es.index(index=index_name, id=i, body={"text": chunk, "embedding": embedding})

# Search and evaluate
def search(query, k=3):
    query_embedding = model.encode(query).tolist()
    res = es.search(index=index_name, body={
        "knn": {"field": "embedding", "query_vector": query_embedding, "k": k, "num_candidates": 10}
    })
    return res["hits"]["hits"][0]["_source"]["text"] if res["hits"]["hits"] else None

query = "How does Elasticsearch work?"
retrieval_context = search(query)
actual_output = your_llm_generate(query, retrieval_context)

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=[retrieval_context],
    expected_output="Elasticsearch uses inverted indexes for keyword searches and dense vector similarity for semantic search."
)

evaluate([test_case], metrics=[ContextualRecallMetric(), ContextualPrecisionMetric(), ContextualRelevancyMetric()])
```

**Tuning tip:** Experiment with `k` and `num_candidates` values, and try different similarity functions.

---

### PGVector (PostgreSQL)

```bash
pip install psycopg2 pgvector
```

```python
import psycopg2
import os
from sentence_transformers import SentenceTransformer
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval import evaluate

# Connect
conn = psycopg2.connect(
    dbname="your_database", user="your_user",
    password=os.getenv("PG_PASSWORD"), host="localhost", port="5432"
)
cursor = conn.cursor()

# Setup
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding vector(384)
    );
""")
conn.commit()

# Store embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
document_chunks = ["PGVector brings vector search to PostgreSQL.", "Vector search enables semantic retrieval."]
for chunk in document_chunks:
    embedding = model.encode(chunk).tolist()
    cursor.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s);", (chunk, embedding))
conn.commit()

# Search and evaluate
def search(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    cursor.execute("""
        SELECT text FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s;
    """, (query_embedding, top_k))
    return [row[0] for row in cursor.fetchall()]

query = "How does PGVector work?"
retrieval_context = search(query)
actual_output = your_llm_generate(query, retrieval_context)

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output="PGVector is an extension that brings efficient vector search."
)

evaluate([test_case], metrics=[ContextualRecallMetric(), ContextualPrecisionMetric(), ContextualRelevancyMetric()])
```

**Performance optimization:** Use domain-specific embedding models such as `BAAI/bge-small-en`, `sentence-transformers/msmarco-distilbert-base-v4`, or `nomic-ai/nomic-embed-text-v1`.

---

### Qdrant

Qdrant is implemented in Rust and achieves 3ms response for 1M OpenAI Embeddings.

```bash
pip install qdrant-client
```

```python
import qdrant_client
from sentence_transformers import SentenceTransformer
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric, ContextualRelevancyMetric
from deepeval import evaluate

# Connect
client = qdrant_client.QdrantClient(url="http://localhost:6333")
collection_name = "documents"

# Create collection
if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_client.http.models.VectorParams(
            size=384,       # must match embedding model output dimension
            distance="cosine"  # try "Dot" or "Euclid" if cosine underperforms
        ),
    )

# Store documents
model = SentenceTransformer("all-MiniLM-L6-v2")
document_chunks = ["Qdrant is a vector database optimized for fast similarity search.", "It uses HNSW for efficient indexing."]
for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()
    client.upsert(
        collection_name=collection_name,
        points=[qdrant_client.http.models.PointStruct(id=i, vector=embedding, payload={"text": chunk})]
    )

# Search and evaluate
def search(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    results = client.search(collection_name=collection_name, query_vector=query_embedding, limit=top_k)
    return [hit.payload["text"] for hit in results] if results else []

query = "How does Qdrant work?"
retrieval_context = search(query)
actual_output = your_llm_generate(query, retrieval_context)

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output="Qdrant is a powerful vector database optimized for semantic search and retrieval."
)

evaluate([test_case], metrics=[ContextualRecallMetric(), ContextualPrecisionMetric(), ContextualRelevancyMetric()])
```

**Tuning tips for low precision:**
- Use domain-specific models: `BAAI/bge-small-en`, `sentence-transformers/msmarco-distilbert-base-v4`, `nomic-ai/nomic-embed-text-v1`
- Adjust vector dimensions to match embedding model output
- Apply metadata filters to exclude unrelated chunks
- Test different `distance` metrics (`cosine`, `Dot`, `Euclid`)

---

## Integration Summary Table

| Category | Integration | Class/Import | Key Feature |
|----------|-------------|--------------|-------------|
| Model | OpenAI | `GPTModel` | Default evaluator; gpt-4.1 |
| Model | Anthropic | `AnthropicModel` | Claude models |
| Model | Gemini | `GeminiModel` | Google Gemini + Vertex AI |
| Model | Grok | `GrokModel` | xAI Grok models |
| Model | Azure OpenAI | `AzureOpenAIModel` | Azure-hosted OpenAI |
| Model | Amazon Bedrock | `AmazonBedrockModel` | AWS-hosted models |
| Model | Ollama | `OllamaModel` | Local models via Ollama |
| Model | OpenRouter | `OpenRouterModel` | 200+ models via proxy |
| Model | LiteLLM | `LiteLLMModel` | 100+ models unified API |
| Model | vLLM | CLI `set-local-model` | High-perf local inference |
| Framework | OpenAI SDK | `deepeval.openai.OpenAI` | Drop-in client wrapper |
| Framework | Anthropic SDK | `deepeval.anthropic.Anthropic` | Drop-in client wrapper |
| Framework | LangChain | `CallbackHandler` | TaskCompletion only |
| Framework | LangGraph | `CallbackHandler` | TaskCompletion only |
| Framework | HuggingFace | `DeepEvalHuggingFaceCallback` | Fine-tuning evaluation |
| Vector DB | Chroma | Direct integration | Also used by Synthesizer |
| Vector DB | Elasticsearch | Direct integration | kNN vector search |
| Vector DB | PGVector | Direct integration | PostgreSQL extension |
| Vector DB | Qdrant | Direct integration | Rust-based, 3ms at 1M vectors |

---

## Related Reference Files

- `30-custom-models-and-embeddings.md` — Implementing `DeepEvalBaseLLM` and `DeepEvalBaseEmbeddingModel` subclasses
- `../09-others/10-cli-and-environment.md` — Full CLI commands and environment variables for all providers
