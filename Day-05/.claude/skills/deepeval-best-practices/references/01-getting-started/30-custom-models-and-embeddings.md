# Custom Models and Embeddings

## Read This When
- Need to use a non-OpenAI model (Llama, Mistral, Azure OpenAI, Gemini, Bedrock) as the LLM-as-a-judge evaluator
- Encountering JSON validation errors from smaller or open-source LLMs and need JSON confinement solutions
- Configuring custom embedding models for the `Synthesizer` golden generation pipeline

## Skip This When
- Using OpenAI or a natively supported provider and just need to pass model parameters → `references/01-getting-started/40-integrations.md`
- Need CLI-only model configuration without writing Python code → `references/01-getting-started/40-integrations.md`

---

DeepEval's metrics default to OpenAI's `gpt-4.1` for evaluation, but any LLM or embedding model can be used as an evaluator. This is important for air-gapped environments, cost optimization, or when you want to use the same model family as your application.

---

## Custom LLM Evaluators

### The `DeepEvalBaseLLM` Interface

To use any LLM as an evaluator, subclass `DeepEvalBaseLLM` and implement six required methods:

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_model_name()` | `() -> str` | Returns a string identifier for the model |
| `load_model()` | `() -> Any` | Returns the model object instance |
| `generate()` | `(prompt: str) -> str` | Synchronous generation; returns string output |
| `a_generate()` | `async (prompt: str) -> str` | Async generation; used for parallel metric execution |

**Basic template:**

```python
from deepeval.models import DeepEvalBaseLLM

class MyCustomLLM(DeepEvalBaseLLM):
    def __init__(self):
        # initialize model here
        pass

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        # call model and return string
        return model.invoke(prompt)

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()
        res = await model.ainvoke(prompt)
        return res

    def get_model_name(self):
        return "My Custom Model"
```

**Using the custom model with any metric:**

```python
from deepeval.metrics import AnswerRelevancyMetric

custom_llm = MyCustomLLM()
metric = AnswerRelevancyMetric(model=custom_llm)
metric.measure(test_case)
```

**Important:** Custom LLM instances must be provided via the `model` parameter for each metric instance. There is no global configuration.

---

### Example: Llama-3 8B (HuggingFace Transformers)

```python
import transformers
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from deepeval.models import DeepEvalBaseLLM

class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return pipeline(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Llama-3 8B"

# Usage
custom_llm = CustomLlama3_8B()
from deepeval.metrics import AnswerRelevancyMetric
metric = AnswerRelevancyMetric(model=custom_llm)
```

---

### Example: Azure OpenAI (via LangChain)

```python
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

# Instantiate
custom_model = AzureChatOpenAI(
    openai_api_version="2025-01-01-preview",
    azure_deployment="your-deployment-name",
    azure_endpoint="https://your-resource.azure.openai.com/",
    openai_api_key="your-api-key",
)
azure_openai = AzureOpenAI(model=custom_model)
```

---

### Example: Mistral 7B (HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM

class Mistral7B(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        device = "cuda"
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        model.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        return self.tokenizer.batch_decode(generated_ids)[0]

    async def a_generate(self, prompt: str) -> str:
        # Option 1: reuse sync (simpler but blocks event loop)
        return self.generate(prompt)
        # Option 2: offload to thread executor (recommended for GPU models)
        # import asyncio
        # loop = asyncio.get_running_loop()
        # return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self):
        return "Mistral 7B"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
```

---

### Example: Google Vertex AI (Gemini via LangChain)

```python
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from deepeval.models.base_model import DeepEvalBaseLLM

class GoogleVertexAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Vertex AI Gemini"

safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

custom_model_gemini = ChatVertexAI(
    model_name="gemini-2.5-flash",
    safety_settings=safety_settings,
    project="your-project-id",
    location="us-central1"
)
vertexai_gemini = GoogleVertexAI(model=custom_model_gemini)
```

---

### Example: AWS Bedrock (via LangChain)

```python
from langchain_community.chat_models import BedrockChat
from deepeval.models.base_model import DeepEvalBaseLLM

class AWSBedrock(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        res = await self.load_model().ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "AWS Bedrock"

custom_model = BedrockChat(
    credentials_profile_name="your-profile",
    region_name="us-east-1",
    endpoint_url="your-bedrock-endpoint",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.4},
)
aws_bedrock = AWSBedrock(model=custom_model)
```

---

## JSON Confinement for Custom LLMs

### The Problem

DeepEval metrics require valid JSON outputs from the evaluation model. Smaller or open-source LLMs often fail to generate properly formatted JSON, causing:

```
ValueError: Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.
```

Example of invalid JSON from a smaller model:
```json
{
    "reaso: "The actual output does directly not address the input",
}
```
Issues: missing opening quote, incomplete key, trailing comma.

### The Solution: Modified Method Signatures

When the `schema` parameter is present in `generate()` and `a_generate()`, DeepEval automatically injects the required Pydantic schema for JSON enforcement. Change the method signature to accept and return `BaseModel`:

```python
from pydantic import BaseModel

class MyCustomLLM(DeepEvalBaseLLM):
    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        # use schema to constrain output
        pass

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)
```

### JSON Confinement Libraries

#### 1. `lm-format-enforcer`

Best for HuggingFace transformers, LangChain, LlamaIndex, llama.cpp, vLLM, Haystack, NVIDIA TensorRT-LLM, ExLlamaV2.

**Approach:** Combines character-level parser with tokenizer prefix tree for sequential token generation within format constraints.

```bash
pip install lm-format-enforcer
```

**Full example — Mistral-7B-Instruct-v0.3 with lm-format-enforcer:**

```python
import json
import torch
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from deepeval.models import DeepEvalBaseLLM

class CustomMistral7B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_length=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            gen_pipeline.tokenizer, parser
        )
        output_dict = gen_pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output)
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Mistral-7B v0.3"
```

#### 2. `instructor`

Best for GPT-3.5/4, Mistral/Mixtral, Anyscale, Ollama, llama-cpp-python.

**Approach:** Encapsulates the LLM client to extract structured data via response model specification.

```bash
pip install instructor
```

**Example — Gemini 2.5 Flash with instructor:**

```python
from pydantic import BaseModel
import google.generativeai as genai
import instructor
from deepeval.models import DeepEvalBaseLLM

class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        return instructor_client.messages.create(
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
        )

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 2.5 Flash"
```

**Example — Claude-3 Opus with instructor:**

```python
from pydantic import BaseModel
from anthropic import Anthropic
import instructor
from deepeval.models import DeepEvalBaseLLM

class CustomClaudeOpus(DeepEvalBaseLLM):
    def __init__(self):
        self.model = Anthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_anthropic(client)
        return instructor_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            response_model=schema,
        )

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Claude-3 Opus"
```

### JSON Confinement Library Comparison

| Library | Best For | Approach |
|---------|----------|----------|
| `lm-format-enforcer` | HuggingFace transformers, vLLM, llama.cpp | Character-level parser + tokenizer prefix tree |
| `instructor` | API-based models (OpenAI, Anthropic, Gemini, Ollama) | Pydantic response model wrapping LLM client |

---

## CLI-Based Model Configuration

The recommended way to configure evaluation models is through the CLI. It handles all environment variables automatically.

### Azure OpenAI via CLI

```bash
deepeval set-azure-openai \
    --base-url="https://your-resource.azure.openai.com/" \
    --model-name="gpt-4.1" \
    --deployment-name="your-deployment" \
    --api-version="2025-01-01-preview"

# Also configure embedding model
deepeval set-azure-openai-embedding --deployment-name="your-embedding-deployment"

# Revert
deepeval unset-azure-openai
```

### Ollama via CLI

```bash
# Start Ollama model first
ollama run deepseek-r1:1.5b

# Configure as default evaluator
deepeval set-ollama --model=deepseek-r1:1.5b

# With custom port
deepeval set-ollama --model=deepseek-r1:1.5b --base-url="http://localhost:11434"

# Configure Ollama embeddings
deepeval set-ollama-embeddings --model=nomic-embed-text

# Revert
deepeval unset-ollama
deepeval unset-ollama-embeddings
```

### Local LLM (LM Studio, vLLM) via CLI

```bash
# LM Studio default URL: http://localhost:1234/v1/
deepeval set-local-model --model=<model_name> \
    --base-url="http://localhost:1234/v1/"

# vLLM default URL: http://localhost:8000/v1/
deepeval set-local-model --model=<model_name> \
    --base-url="http://localhost:8000/v1/"

# Configure local embeddings
deepeval set-local-embeddings --model=<embedding_model_name> \
    --base-url="http://localhost:1234/v1/"

# Revert
deepeval unset-local-model
deepeval unset-local-embeddings
```

### LiteLLM via CLI

LiteLLM acts as a proxy to access 100+ LLMs through a unified interface.

```bash
pip install litellm

# OpenAI via LiteLLM
deepeval set-litellm --model=openai/gpt-3.5-turbo

# Anthropic via LiteLLM
deepeval set-litellm --model=anthropic/claude-3-opus

# Google via LiteLLM
deepeval set-litellm --model=google/gemini-pro

# Custom endpoint via LiteLLM
deepeval set-litellm \
    --model=openai/gpt-3.5-turbo \
    --base-url="https://your-custom-endpoint.com"

# Revert
deepeval unset-litellm
```

### Persisting CLI Settings

Use `--save` flag to persist settings permanently to a dotenv file:

```bash
deepeval set-azure-openai \
    --base-url=<endpoint> \
    --model-name=<model> \
    --deployment-name=<deployment> \
    --api-version=<version> \
    --save=dotenv
```

Set a default save target to avoid passing `--save` every time:
```bash
export DEEPEVAL_DEFAULT_SAVE=dotenv:.env.local
```

---

## Custom Embedding Models

### The `DeepEvalBaseEmbeddingModel` Interface

Custom embedding models are used by the `Synthesizer` for synthetic data generation (specifically the `generate_goldens_from_docs()` method, which uses embedding models to extract relevant context via cosine similarity).

**Required methods to implement:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_model_name()` | `() -> str` | Returns string identifier |
| `load_model()` | `() -> Any` | Returns model object |
| `embed_text()` | `(text: str) -> List[float]` | Embed a single text string |
| `embed_texts()` | `(texts: List[str]) -> List[List[float]]` | Embed multiple strings |
| `a_embed_text()` | `async (text: str) -> List[float]` | Async single embed |
| `a_embed_texts()` | `async (texts: List[str]) -> List[List[float]]` | Async batch embed |

### Example: Azure OpenAI Embeddings (code-based)

```python
from typing import List
from langchain_openai import AzureOpenAIEmbeddings
from deepeval.models import DeepEvalBaseEmbeddingModel

class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        pass

    def load_model(self):
        return AzureOpenAIEmbeddings(
            openai_api_version="2025-01-01-preview",
            azure_deployment="your-embedding-deployment",
            azure_endpoint="https://your-resource.azure.openai.com/",
            openai_api_key="your-api-key",
        )

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        return "Custom Azure Embedding Model"
```

### Using Custom Embeddings in Synthesis

Pass the custom embedder via `ContextConstructionConfig`:

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    context_construction_config=ContextConstructionConfig(
        embedder=CustomEmbeddingModel()
    )
)
```

---

## Key Notes

- **Async performance:** The `a_generate()` method is used by DeepEval for parallel metric execution. Implementing true async improves evaluation speed; simply reusing the synchronous `generate()` is acceptable if async support is unavailable.
- **Thread safety:** When offloading GPU model generation to threads, ensure resource sharing safety with concurrent async operations.
- **Per-metric configuration:** There is no global custom model setting — the `model` parameter must be specified on each metric instantiation.
- **JSON confinement is opt-in:** Only change the method signatures to accept `BaseModel` if you encounter JSON validation errors.
- **Notebook tip:** If using Jupyter/Colab, the async `a_generate()` may conflict with the notebook's event loop. Use `nest_asyncio` if needed.

---

## Related Reference Files

- `40-integrations.md` — Native model integrations (OpenAI, Anthropic, Gemini, etc.) with pre-built model classes
- `../09-others/10-cli-and-environment.md` — Full CLI command reference and environment variables for model configuration
