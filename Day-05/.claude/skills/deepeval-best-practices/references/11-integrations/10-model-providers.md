# DeepEval Model Provider Integrations

Source: https://deepeval.com/integrations/models/
Fetched: 2026-02-20

## Read This When
- Using a specific model provider (OpenAI, Azure OpenAI, Anthropic, Gemini, Grok, Amazon Bedrock, LiteLLM, Ollama, OpenRouter, vLLM) and need CLI or Python setup instructions
- Need to configure API keys, model parameters, or custom endpoints for a pre-built DeepEval model integration
- Want to compare available models across providers or find the correct model class (GPTModel, AnthropicModel, GeminiModel, etc.)

## Skip This When
- Need to implement a fully custom LLM by inheriting DeepEvalBaseLLM with JSON confinement -- see `references/09-guides/40-custom-llms-and-embeddings.md`
- Looking for framework-level integrations (LangChain, OpenAI Agents, Hugging Face) -- see `references/11-integrations/20-frameworks.md`
- Want to configure custom embedding models for the Synthesizer -- see `references/09-guides/40-custom-llms-and-embeddings.md`

---

## Table of Contents

1. [OpenAI](#1-openai)
2. [Azure OpenAI](#2-azure-openai)
3. [Anthropic](#3-anthropic)
4. [Gemini](#4-gemini)
5. [Grok](#5-grok)
6. [Amazon Bedrock](#6-amazon-bedrock)
7. [LiteLLM](#7-litellm)
8. [Ollama](#8-ollama)
9. [OpenRouter](#9-openrouter)
10. [vLLM](#10-vllm)

---

## 1. OpenAI

Source: https://deepeval.com/integrations/models/openai

### Overview
DeepEval uses `gpt-4.1` by default for evaluation metrics, requiring an OpenAI API key to function.

### API Key Setup

**Recommended (local development):**
```
# .env.local
OPENAI_API_KEY=<your-openai-api-key>
```

**Shell/CI environment:**
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

**Jupyter/Colab notebook:**
```python
%env OPENAI_API_KEY=<your-openai-api-key>
```

### Command Line Configuration

```bash
deepeval set-openai \
    --model=gpt-4.1 \
    --cost-per-input-token=0.000002 \
    --cost-per-output-token=0.000008
```

To unset OpenAI settings and use another provider:
```bash
deepeval unset-openai
```

### Python Implementation

```python
from deepeval.models import GPTModel
from deepeval.metrics import AnswerRelevancyMetric

model = GPTModel(
    model="gpt-4.1",
    temperature=0,
    cost_per_input_token=0.000002,
    cost_per_output_token=0.000008)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

### GPTModel Parameters

| Parameter | Type | Status | Description |
|-----------|------|--------|-------------|
| model | string | Optional | Model name; defaults to `OPENAI_MODEL_NAME` or `gpt-4.1` |
| api_key | string | Optional | Authentication key; defaults to `OPENAI_API_KEY` |
| base_url | string | Optional | Custom OpenAI URL |
| temperature | float | Optional | Model temperature; defaults to `0.0` |
| cost_per_input_token | float | Optional | Input token cost |
| cost_per_output_token | float | Optional | Output token cost |
| generation_kwargs | dictionary | Optional | Additional parameters for API calls |

### Available OpenAI Models

- gpt-5, gpt-5-mini, gpt-5-nano
- gpt-4.1, gpt-4.5-preview, gpt-4o, gpt-4o-mini
- o1, o1-pro, o1-mini, o3-mini
- gpt-4-turbo, gpt-4, gpt-4-32k
- gpt-3.5-turbo variants
- davinci-002, babbage-002

### Related Links

- [Azure OpenAI integration](/integrations/models/azure-openai)
- [OpenAI Agents framework](/integrations/frameworks/openai-agents)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [Official OpenAI API docs](https://platform.openai.com/docs/api-reference/responses/create)

---

## 2. Azure OpenAI

Source: https://deepeval.com/integrations/models/azure-openai

### Overview
DeepEval enables direct integration of Azure OpenAI models into all available LLM-based metrics through command-line or Python configuration.

### Configuration Methods

#### Command Line Setup
Execute this terminal command to set Azure OpenAI as the default metrics provider:

```bash
deepeval set-azure-openai \
    --base-url=<endpoint> \
    --model-name=<model_name> \
    --deployment-name=<deployment_name> \
    --api-version=<api_version>
```

**Example values:**
- base-url: `https://example-resource.azure.openai.com/`
- model-name: `gpt-4.1`
- deployment-name: `Test Deployment`
- api-version: `2025-01-01-preview`

**Unsetting Configuration:**
```bash
deepeval unset-azure-openai
```

**Persistence Option:** Use `--save` flag to persist settings permanently.

#### Python Implementation

```python
from deepeval.models import AzureOpenAIModel
from deepeval.metrics import AnswerRelevancyMetric

model = AzureOpenAIModel(
    model="gpt-4.1",
    deployment_name="Test Deployment",
    api_key="Your Azure OpenAI API Key",
    api_version="2025-01-01-preview",
    base_url="https://example-resource.azure.openai.com/",
    temperature=0)

answer_relevancy = AnswerRelevancyMetric(model=model)
```

### AzureOpenAIModel Parameters

All parameters are optional; mandatory values can come from environment variables or DeepEval settings:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Azure OpenAI model name | `AZURE_MODEL_NAME` env var |
| `api_key` | string | Azure OpenAI API key | `AZURE_OPENAI_API_KEY` env var |
| `azure_ad_token` | string | Azure AD token for authentication | `AZURE_OPENAI_AD_TOKEN` env var |
| `azure_ad_token_provider` | callback | AsyncAzureADTokenProvider or AzureADTokenProvider | None |
| `base_url` | string | Azure OpenAI endpoint URL | `AZURE_OPENAI_ENDPOINT` env var |
| `temperature` | float | Model temperature setting | `0.0` |
| `cost_per_input_token` | float | Input token cost | Model registry or `None` |
| `cost_per_output_token` | float | Output token cost | Model registry or `None` |
| `deployment_name` | string | Azure deployment name | `AZURE_DEPLOYMENT_NAME` env var |
| `api_version` | string | OpenAI API version | `OPENAI_API_VERSION` env var |
| `generation_kwargs` | dict | Additional generation parameters | None |

### Supported Azure OpenAI Models

- `gpt-4.1`
- `gpt-4.5-preview`
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4`
- `gpt-4-32k`
- `gpt-35-turbo`
- `gpt-35-turbo-16k`
- `gpt-35-turbo-instruct`
- `o1`
- `o1-mini`
- `o1-preview`
- `o3-mini`

### Related Resources

- [Environment Variables and Settings Documentation](/docs/evaluation-flags-and-configs#model-settings-azure-openai)
- [OpenAI Integration](/integrations/models/openai)
- [Ollama Integration](/integrations/models/ollama)
- [Azure OpenAI Official Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/reference#request-body)

---

## 3. Anthropic

Source: https://deepeval.com/integrations/models/anthropic

### Overview
DeepEval supports using any Anthropic model for evaluation metrics. Users need to set up an Anthropic API key to get started.

### Setting Up Your API Key

**CLI Setup:**
```bash
export ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

**Notebook Environment (Jupyter/Colab):**
```python
%env ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

### Python Implementation

```python
from deepeval.models import AnthropicModel
from deepeval.metrics import AnswerRelevancyMetric

model = AnthropicModel(
    model="claude-3-7-sonnet-latest",
    temperature=0)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

### AnthropicModel Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `claude-3-7-sonnet-latest` | Claude model to use |
| `api_key` | string | Optional | `ANTHROPIC_API_KEY` env var | Anthropic API key |
| `temperature` | float | Optional | `0.0` | Model temperature (must be >= 0) |
| `cost_per_input_token` | float | Optional | Registry value or None | Input token cost |
| `cost_per_output_token` | float | Optional | Registry value or None | Output token cost |
| `generation_kwargs` | dict | Optional | N/A | Additional generation parameters |

> "Pass generation parameters, such as `max_tokens`, via `generation_kwargs`"

### Available Anthropic Models

- `claude-3-7-sonnet-latest`
- `claude-3-5-haiku-latest`
- `claude-3-5-sonnet-latest`
- `claude-3-opus-latest`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-instant-1.2`

### Related Links

- [Environment Variables and Settings](/docs/evaluation-flags-and-configs#model-settings-anthropic)
- [Previous: OpenRouter](/integrations/models/openrouter)
- [Next: Amazon Bedrock](/integrations/models/amazon-bedrock)

---

## 4. Gemini

Source: https://deepeval.com/integrations/models/gemini

### Overview
DeepEval allows integration of Gemini models into LLM-based metrics through CLI or Python code.

### Configuration Methods

#### Command Line Setup
Execute this terminal command to configure Gemini as the default metrics provider:

```bash
deepeval set-gemini \
    --model=<model> # e.g. "gemini-2.5-flash"
```

To reset Gemini configuration:
```bash
deepeval unset-gemini
```

The `--save` flag enables persistent CLI settings storage.

#### Python Integration
Import and instantiate the `GeminiModel` class:

```python
from deepeval.models import GeminiModel
from deepeval.metrics import AnswerRelevancyMetric

model = GeminiModel(
    model="gemini-2.5-pro",
    api_key="Your Gemini API Key",
    temperature=0)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

### GeminiModel Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | No | `GEMINI_MODEL_NAME` | Gemini model identifier |
| `api_key` | string | No | `GOOGLE_API_KEY` | Google API authentication key |
| `temperature` | float | No | `TEMPERATURE` / `0.0` | Model temperature setting |
| `generation_kwargs` | dictionary | No | -- | Additional Gemini API parameters |

**Note:** API key is mandatory at runtime unless using Vertex AI. Constructor arguments take precedence over environment variables.

### Available Gemini Models

- `gemini-3-pro-preview`
- `gemini-3-flash-preview`
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-2.0-flash`
- `gemini-2.0-flash-lite`
- `gemini-pro-latest`
- `gemini-flash-latest`
- `gemini-flash-lite-latest`

For comprehensive model list, consult Gemini's official documentation.

### Related Resources
- [Environment variables and settings documentation](/docs/evaluation-flags-and-configs#model-settings-gemini)
- [Vertex AI integration](/docs/integrations/models/vertex-ai)

---

## 5. Grok

Source: https://deepeval.com/integrations/models/grok

### Overview
DeepEval enables evaluation operations using Grok models via xAI's SDK through CLI or Python interfaces. The framework validates model names against a supported list.

### Installation Requirements

**Prerequisite:** Install the xAI SDK before proceeding:
```
pip install xai-sdk
```

### Command Line Configuration

Configure Grok as the default llm-judge:
```bash
deepeval set-grok --model grok-4.1 --temperature=0
```

To remove Grok as default:
```bash
deepeval unset-grok
```

**Persistence:** Use the optional `--save` flag to persist settings across sessions.

### Python Implementation

Initialize Grok models directly in code:

```python
from deepeval.models import GrokModel
from deepeval.metrics import AnswerRelevancyMetric

model = GrokModel(
    model="grok-4.1",
    api_key="your-api-key",
    temperature=0
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

### GrokModel Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Optional | `GROK_MODEL_NAME` env var | Model identifier |
| `api_key` | string | Optional | `GROK_API_KEY` env var | Authentication token |
| `temperature` | float | Optional | `0.0` | Sampling temperature |
| `cost_per_input_token` | float | Optional | Registry lookup | Input token cost |
| `cost_per_output_token` | float | Optional | Registry lookup | Output token cost |
| `generation_kwargs` | dict | Optional | None | Additional xAI SDK parameters |

### Available Grok Models

- grok-4.1
- grok-4
- grok-4-heavy
- grok-4-fast
- grok-beta
- grok-3
- grok-2
- grok-2-mini
- grok-code-fast-1

### Related Documentation Links

- [Persisting CLI Settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save)
- [xAI Official Documentation](https://docs.x.ai/docs/guides/function-calling#function-calling-modes)
- [Previous: Vertex AI](/integrations/models/vertex-ai)
- [Next: Moonshot](/integrations/models/moonshot)

### Notes

Pass additional model parameters through `generation_kwargs`. Verify parameter compatibility with xAI SDK's official documentation before implementation.

---

## 6. Amazon Bedrock

Source: https://deepeval.com/integrations/models/amazon-bedrock

### Overview
DeepEval supports Amazon Bedrock models through the Bedrock Runtime Converse API for evaluation metrics. Setup requires AWS credentials configuration.

### Installation Requirements
"AmazonBedrockModel requires aiobotocore and botocore. deepeval will prompt you to install them if they are missing."

### API Key Setup

**CLI Configuration:**
```bash
export AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
export AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>
```

**Notebook Environment:**
```python
%env AWS_ACCESS_KEY_ID=<your-aws-access-key-id>
%env AWS_SECRET_ACCESS_KEY=<your-aws-secret-access-key>
```

### Python Implementation

```python
from deepeval.models import AmazonBedrockModel
from deepeval.metrics import AnswerRelevancyMetric

model = AmazonBedrockModel(
    model="anthropic.claude-3-opus-20240229-v1:0",
    region="us-east-1",
    generation_kwargs={"temperature": 0},
)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

### AmazonBedrockModel Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Optional | Bedrock model identifier (e.g., `anthropic.claude-3-opus-20240229-v1:0`). Defaults to `AWS_BEDROCK_MODEL_NAME` |
| `region` | string | Optional | AWS region hosting Bedrock endpoint (e.g., `us-east-1`). Defaults to `AWS_BEDROCK_REGION` |
| `aws_access_key_id` | string | Optional | AWS Access Key ID; defaults to environment variable, falls back to AWS credentials chain |
| `aws_secret_access_key` | string | Optional | AWS Secret Access Key; defaults to environment variable, falls back to AWS credentials chain |
| `cost_per_input_token` | float | Optional | Per-input-token cost in USD |
| `cost_per_output_token` | float | Optional | Per-output-token cost in USD |
| `generation_kwargs` | dictionary | Optional | Generation parameters (temperature, topP, maxTokens) sent as `inferenceConfig` |

### Available Models

Common Bedrock foundation models include:
- Claude 3 variants (Opus, Sonnet, Haiku)
- Amazon Titan models
- Amazon Nova models
- Meta Llama models
- Mistral models
- OpenAI models

### Related Resources
- [AWS Bedrock inference parameters documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-parameters.html)
- [Environment variables and settings](/docs/evaluation-flags-and-configs#model-settings-aws-amazon-bedrock)

---

## 7. LiteLLM

Source: https://deepeval.com/integrations/models/litellm

### Overview
DeepEval enables evaluation using any model supported by LiteLLM through CLI or Python. The framework allows seamless integration with multiple AI providers.

### Installation
Prior to setup, install LiteLLM separately:
```bash
pip install litellm
```

### Configuration Methods

#### Command Line Setup
Configure LiteLLM models via CLI by specifying the provider in the model name:

```bash
# OpenAI
deepeval set-litellm --model=openai/gpt-3.5-turbo

# Anthropic
deepeval set-litellm --model=anthropic/claude-3-opus

# Google
deepeval set-litellm --model=google/gemini-pro
```

**Advanced Options:**
```bash
# With custom API base
deepeval set-litellm --model=openai/gpt-3.5-turbo --base-url="https://your-custom-endpoint.com"

# With both API key and custom base
deepeval set-litellm \
    --model=openai/gpt-3.5-turbo \
    --base-url="https://your-custom-endpoint.com"
```

To unset LiteLLM as default:
```bash
deepeval unset-litellm
```

#### Python Implementation
```python
from deepeval.models import LiteLLMModel
from deepeval.metrics import AnswerRelevancyMetric

model = LiteLLMModel(
    model="openai/gpt-3.5-turbo",
    api_key="your-api-key",
    base_url="your-api-base",
    temperature=0)

answer_relevancy = AnswerRelevancyMetric(model=model)
```

### LiteLLMModel Parameters

| Parameter | Type | Status | Description |
|-----------|------|--------|-------------|
| model | string | Required | Provider and model name (e.g., "openai/gpt-3.5-turbo") |
| api_key | string | Optional | API authentication key |
| base_url | string | Optional | Custom endpoint URL |
| temperature | float | Optional | Model temperature setting |
| generation_kwargs | dict | Optional | Additional generation parameters |

### Environment Variables
```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# Google
export GOOGLE_API_KEY="your-api-key"

# Custom endpoint
export LITELLM_API_BASE="https://your-custom-endpoint.com"
```

### Supported Models

**OpenAI:** gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview

**Anthropic:** claude-3-opus, claude-3-sonnet, claude-3-haiku

**Google:** gemini-pro, gemini-ultra

**Mistral:** mistral-small, mistral-medium, mistral-large

**LM Studio:** Meta-Llama-3.1-8B-Instruct-GGUF, Mistral-7B-Instruct-v0.2-GGUF, Phi-2-GGUF

**Ollama:** llama2, mistral, codellama, neural-chat, starling-lm

### Code Examples

#### Multi-Provider Usage
```python
from deepeval.models import LiteLLMModel
from deepeval.metrics import AnswerRelevancyMetric

# OpenAI
model = LiteLLMModel(model="openai/gpt-3.5-turbo")
metric = AnswerRelevancyMetric(model=model)

# Anthropic
model = LiteLLMModel(model="anthropic/claude-3-opus")

# LM Studio (localhost:1234/v1 default)
model = LiteLLMModel(
    model="lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio")

# Ollama (localhost:11434/v1 default)
model = LiteLLMModel(
    model="ollama/llama2",
    base_url="http://localhost:11434/v1",
    api_key="ollama")
```

#### Custom Endpoint
```python
model = LiteLLMModel(
    model="custom/your-model-name",
    base_url="https://your-custom-endpoint.com",
    api_key="your-api-key")
```

#### Schema Validation
```python
from pydantic import BaseModel

class ResponseSchema(BaseModel):
    score: float
    reason: str

model = LiteLLMModel(model="openai/gpt-3.5-turbo")
response, cost = model.generate(
    "Rate this answer: 'The capital of France is Paris'",
    schema=ResponseSchema)
```

### Best Practices

1. **Provider Specification:** Always include provider prefix in model names
2. **Security:** Store API keys in environment variables, never hardcode
3. **Model Selection:** Match model complexity to task requirements
4. **Local Development:** Use LM Studio or Ollama for offline evaluation
5. **Error Handling:** Implement safeguards for rate limits and connection failures
6. **Cost Management:** Monitor usage with larger models
7. **Local Setup Requirements:**
   - LM Studio: Ensure running, model loaded, use default URL `http://localhost:1234/v1`
   - Ollama: Ensure running, model pulled, use default URL `http://localhost:11434/v1`

### Related Documentation
- [LiteLLM Official Docs](https://docs.litellm.ai/docs/providers)
- [LM Studio Integration](/integrations/models/lmstudio)
- [Vector Databases](/integrations/vector-databases/cognee)

---

## 8. Ollama

Source: https://deepeval.com/integrations/models/ollama

### Overview
DeepEval enables integration with Ollama-served models for running evaluations via CLI or Python code. The framework auto-detects capabilities like multimodal support from a known-model list.

### Prerequisites
"Before getting started, make sure your Ollama model is installed and running."

Execute: `ollama run deepseek-r1:1.5b`

### Environment Configuration

Set optional custom host in `.env.local`:
```
LOCAL_MODEL_BASE_URL=http://localhost:11434
```

### CLI Setup

Configure Ollama model (replace `deepseek-r1:1.5b` with your choice):
```bash
deepeval set-ollama --model=deepseek-r1:1.5b
```

Specify custom port/base URL:
```bash
deepeval set-ollama --model=deepseek-r1:1.5b \
    --base-url="http://localhost:11434"
```

Unset Ollama as default provider:
```bash
deepeval unset-ollama
```

### Python Implementation

```python
from deepeval.models import OllamaModel
from deepeval.metrics import AnswerRelevancyMetric

model = OllamaModel(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0)
answer_relevancy = AnswerRelevancyMetric(model=model)
```

### OllamaModel Parameters

| Parameter | Type | Status | Details |
|-----------|------|--------|---------|
| `model` | string | Optional | Model name; defaults to `OLLAMA_MODEL_NAME` env var |
| `base_url` | string | Optional | Server URL; defaults to `LOCAL_MODEL_BASE_URL` or `http://localhost:11434` |
| `temperature` | float | Optional | Model temperature; defaults to `TEMPERATURE` env var or `0.0` |
| `generation_kwargs` | dict | Optional | Additional parameters forwarded to Ollama's chat call |

### Available Models

- deepseek-r1
- llama3.1
- gemma
- qwen
- mistral
- codellama
- phi3
- tinyllama
- starcoder2

**Note:** Consult Ollama's official documentation for comprehensive model list.

### Related Resources
- [Previous: Azure OpenAI](/integrations/models/azure-openai)
- [Next: OpenRouter](/integrations/models/openrouter)
- [GitHub Repository](https://github.com/confident-ai/deepeval)

---

## 9. OpenRouter

Source: https://deepeval.com/integrations/models/openrouter

### Overview
DeepEval's OpenRouter integration enables users to "connect any OpenRouter supported model to power all of deepeval's metrics."

### Configuration Methods

#### Command Line Setup
Users can configure OpenRouter via CLI with this command structure:
```bash
deepeval set-openrouter \
    --model "openai/gpt-4.1" \
    --base-url "https://openrouter.ai/api/v1" \
    --temperature=0 \
    --prompt-api-key
```

To remove OpenRouter as default:
```bash
deepeval unset-openrouter
```

CLI settings can be persisted using the optional `--save` flag.

#### Python Implementation
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

### OpenRouterModel Parameters

**Zero mandatory and seven optional parameters:**

- `model` [Optional]: Defaults to `OPENROUTER_MODEL_NAME` environment variable; falls back to "openai/gpt-4.1"
- `api_key` [Optional]: Defaults to `OPENROUTER_API_KEY`; raises error if unset at runtime
- `base_url` [Optional]: Defaults to `OPENROUTER_BASE_URL`; falls back to "https://openrouter.ai/api/v1"
- `temperature` [Optional]: Defaults to `TEMPERATURE`; falls back to `0.0`
- `cost_per_input_token` [Optional]: Raises error if unset at runtime
- `cost_per_output_token` [Optional]: Raises error if unset at runtime
- `generation_kwargs` [Optional]: Dictionary of additional generation parameters

Additional `**kwargs` are forwarded to the underlying OpenAI client constructor.

### References
- [Official OpenRouter Documentation](https://openrouter.ai/docs)
- [Previous: Ollama Integration](/integrations/models/ollama)
- [Next: Anthropic Integration](/integrations/models/anthropic)

---

## 10. vLLM

Source: https://deepeval.com/integrations/models/vllm

### Overview
"vLLM is a high-performance inference engine for LLMs that supports OpenAI-compatible APIs." DeepEval enables connection to a running vLLM server for conducting local evaluations.

### Setup Instructions

#### Command Line Configuration

1. Launch your vLLM server with OpenAI-compatible API exposure (typical local URL: `http://localhost:8000/v1/`)
2. Execute this command:

```bash
deepeval set-local-model \
    --model=<model_name> \
    --base-url="http://localhost:8000/v1/"
```

**Note:** You may enter any value for the API key prompt if authentication isn't required.

#### Persisting Settings
Optional `--save` flag available for persisting CLI settings. Reference: [Flags and Configs -> Persisting CLI settings](/docs/evaluation-flags-and-configs#persisting-cli-settings-with---save)

### Reverting Configuration

To disable the local model and return to OpenAI:

```bash
deepeval unset-local-model
```

### Additional Resources
For advanced setup or deployment options (multi-GPU, HuggingFace models), consult the [vLLM documentation](https://vllm.ai/).

### Navigation Links
- Previous: [Portkey](/integrations/models/portkey)
- Next: [LM Studio](/integrations/models/lmstudio)
