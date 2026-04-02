# Custom LLMs and Embedding Models — Reference Guide

Combined from: guides-using-custom-llms.md + guides-using-custom-llms-json.md + guides-using-custom-embedding-models.md
Sources: https://deepeval.com/guides/guides-using-custom-llms | https://deepeval.com/guides/guides-using-custom-llms#json-confinement-for-custom-llms | https://deepeval.com/guides/guides-using-custom-embedding-models

## Read This When
- Need to use a non-OpenAI LLM (Claude, Gemini, Llama, Mistral, Azure, Bedrock) as the evaluation judge in DeepEval metrics
- Getting JSON parsing errors from custom LLMs and need to apply JSON confinement (lm-format-enforcer, instructor)
- Want to integrate a custom embedding model for the Synthesizer's `generate_goldens_from_docs()`

## Skip This When
- Using OpenAI models and only need to configure API keys or model names -- see `references/11-integrations/10-model-providers.md`
- Want to use a pre-built integration (Ollama, LiteLLM, Gemini) via CLI setup -- see `references/11-integrations/10-model-providers.md`
- Need to build custom evaluation logic rather than swap the evaluation model -- see `references/09-guides/30-custom-metrics.md`

---

# Part 1: Using Custom LLMs for Evaluation

## Overview

DeepEval's metrics default to OpenAI's GPT models for evaluation, but the framework supports any custom LLM including Claude (Anthropic), Gemini (Google), Llama-3 (Meta), and Mistral.

## Creating A Custom LLM

### Six Rules for Implementation

1. Inherit `DeepEvalBaseLLM`
2. Implement `get_model_name()` returning a string identifier
3. Implement `load_model()` returning a model object
4. Implement `generate(prompt: str) -> str` with single string parameter
5. Return generated string output from `generate()`
6. Implement async `a_generate(prompt: str) -> str` method

### Basic Example: Llama-3 8B

```python
import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models import DeepEvalBaseLLM

class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self.model = model_4bit
        self.tokenizer = tokenizer

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
```

### Usage

```python
custom_llm = CustomLlama3_8B()
print(custom_llm.generate("Write me a joke"))

from deepeval.metrics import AnswerRelevancyMetric
metric = AnswerRelevancyMetric(model=custom_llm)
metric.measure(...)
```

## Additional Examples

### Azure OpenAI

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

custom_model = AzureChatOpenAI(
    openai_api_version=api_version,
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
    openai_api_key=openai_api_key,
)
azure_openai = AzureOpenAI(model=custom_model)
print(azure_openai.generate("Write me a joke"))
```

**Key Note**: Custom LLM instances must be provided via the `model` parameter for each metric instance.

### Mistral 7B

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
        return self.generate(prompt)

    def get_model_name(self):
        return "Mistral 7B"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)
print(mistral_7b.generate("Write me a joke"))
```

**Alternative async implementation using thread executor**:

```python
import asyncio

class Mistral7B(DeepEvalBaseLLM):
    # ... existing code ...
    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt)
```

### Google Vertex AI (Gemini)

```python
from langchain_google_vertexai import (
    ChatVertexAI,
    HarmBlockThreshold,
    HarmCategory
)
from deepeval.models.base_model import DeepEvalBaseLLM

class GoogleVertexAI(DeepEvalBaseLLM):
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
        return "Vertex AI Model"

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
    project="<project-id>",
    location="<region>"
)
vertexai_gemini = GoogleVertexAI(model=custom_model_gemini)
print(vertexai_gemini.generate("Write me a joke"))
```

### AWS Bedrock

```python
from langchain_community.chat_models import BedrockChat
from deepeval.models.base_model import DeepEvalBaseLLM

class AWSBedrock(DeepEvalBaseLLM):
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

custom_model = BedrockChat(
    credentials_profile_name="<profile-name>",
    region_name="<region-name>",
    endpoint_url="<bedrock-endpoint>",
    model_id="<model-id>",
    model_kwargs={"temperature": 0.4},
)
aws_bedrock = AWSBedrock(model=custom_model)
print(aws_bedrock.generate("Write me a joke"))
```

---

# Part 2: JSON Confinement for Custom LLMs

## Problem Statement

DeepEval metrics require valid JSON outputs from evaluation models. Smaller, open-source LLMs often fail to generate properly formatted JSON, causing:

```
ValueError: Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.
```

**Example invalid JSON from Mistral-7B**:

```json
{
    "reaso: "The actual output does directly not address the input",
}
```

Issues: missing opening quote, incomplete key, trailing comma.

## Solution: Modified Method Signatures

Rewrite `generate()` and `a_generate()` to accept and return Pydantic `BaseModel` objects:

```python
from pydantic import BaseModel

class CustomLlama3_8B(DeepEvalBaseLLM):
    ...
    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        pass

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)
```

When the `schema` parameter is defined, DeepEval automatically injects required Pydantic schemas for JSON enforcement.

## Implementation with lm-format-enforcer

Installation:

```bash
pip install lm-format-enforcer
```

**Full Example - Llama-3 8B with lm-format-enforcer**:

```python
import json
import transformers
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from deepeval.models import DeepEvalBaseLLM

class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
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

        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output)

        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Llama-3 8B"
```

## JSON Confinement Libraries

### 1. lm-format-enforcer

A versatile library for standardizing LLM output formats across multiple platforms:

**Supported frameworks:**
- transformers
- langchain
- llamaindex
- llama.cpp
- vLLM
- Haystack
- NVIDIA
- TensorRT-LLM
- ExLlamaV2

**Approach:** Combines character-level parser with tokenizer prefix tree, allowing sequential token generation within format constraints while enhancing output quality.

**Reference:** [LM-format-enforcer GitHub](https://github.com/noamgat/lm-format-enforcer)

### 2. instructor

User-friendly Python library built on Pydantic for LLM output confinement:

**Supported models:**
- GPT-3.5, GPT-4, GPT-4-Vision
- Mistral/Mixtral
- Anyscale
- Ollama
- llama-cpp-python

**Approach:** Encapsulates LLM client to extract structured data like JSON through response model specification.

**Reference:** [Instructor GitHub](https://github.com/jxnl/instructor)

## Complete Implementation Examples

### Mistral-7B-Instruct-v0.3 with lm-format-enforcer

```bash
pip install lm-format-enforcer
```

```python
import json
from pydantic import BaseModel
import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models import DeepEvalBaseLLM

class CustomMistral7B(DeepEvalBaseLLM):
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_4bit = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            device_map="auto",
            quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3"
        )
        self.model = model_4bit
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()
        pipeline = pipeline(
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
            pipeline.tokenizer, parser
        )

        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output)

        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Mistral-7B v0.3"
```

### Gemini 2.5 Flash with instructor

```bash
pip install instructor
```

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
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"
```

### Claude-3 Opus with instructor

```bash
pip install instructor
```

```python
from pydantic import BaseModel
from anthropic import Anthropic
from deepeval.models import DeepEvalBaseLLM

class CustomClaudeOpus(DeepEvalBaseLLM):
    def __init__(self):
        self.model = Anthropic()

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_anthropic(client)
        resp = instructor_client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Claude-3 Opus"
```

## Key Notes and Warnings

**Async Implementation**: The `a_generate()` method is used by DeepEval for asynchronous metric execution. Implementing true async improves evaluation speed compared to reusing synchronous `generate()`.

**Thread Safety**: When offloading generation to separate threads, ensure resource sharing safety with concurrent async operations.

**GPU Utilization**: Running generations in separate threads may not fully utilize GPU resources for GPU-based models.

**Pydantic Requirement**: BaseModel types come from the `pydantic` library, a standard Python typing library.

**JSON Confinement Flexibility**: DeepEval supports any JSON confinement library; the documentation suggests `lm-format-enforcer` and `instructor` as proven options.

**Per-Metric Configuration**: Custom LLMs must be specified via `model` parameter when instantiating each metric; no global configuration available.

**Additional Support**: Community support available through [DeepEval Discord](https://discord.com/invite/a3K9c8GRGt) for implementations not covered in documentation.

---

# Part 3: Using Custom Embedding Models

## Overview

The `generate_goldens_from_docs()` method in the `Synthesizer` for synthetic data generation uses embedding models to extract relevant context from documents via cosine similarity. This guide demonstrates how to integrate any embedding model for this purpose.

## Configuration Methods

### Azure OpenAI

Initial setup command:
```bash
deepeval set-azure-openai \
    --base-url=<endpoint> \
    --model=<model_name> \
    --deployment-name=<deployment_name> \
    --api-version=<api_version> \
    --model-version=<model_version>
```

Then configure the embedder:
```bash
deepeval set-azure-openai-embedding --deployment-name=<embedding_deployment_name>
```

### Ollama Models

```bash
deepeval set-ollama --model=<model_name>
```

To revert to default OpenAI:
```bash
deepeval unset-ollama
```

Set Ollama embeddings:
```bash
deepeval set-ollama-embeddings --model=<embedding_model_name>
```

Revert embeddings:
```bash
deepeval unset-ollama-embeddings
```

### Local LLM Models

Popular base URL examples:
- **LM Studio**: `http://localhost:1234/v1/`
- **vLLM**: `http://localhost:8000/v1/`

Configuration:
```bash
deepeval set-local-model --model=<model_name> \
    --base-url="http://localhost:1234/v1/"

deepeval set-local-embeddings --model=<embedding_model_name> \
    --base-url="http://localhost:1234/v1/"
```

Revert embeddings:
```bash
deepeval unset-local-embeddings
```

### Custom Embedding Model (Code-Based)

Complete implementation example:

```python
from typing import List, Optional
from langchain_openai import AzureOpenAIEmbeddings
from deepeval.models import DeepEvalBaseEmbeddingModel

class CustomEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        pass

    def load_model(self):
        return AzureOpenAIEmbeddings(
            openai_api_version="...",
            azure_deployment="...",
            azure_endpoint="...",
            openai_api_key="...",
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

## Required Implementation Methods

When inheriting `DeepEvalBaseEmbeddingModel`, you must implement:

1. **`get_model_name()`** - Returns a string identifier for your model
2. **`load_model()`** - Returns the model object instance
3. **`embed_text(text: str)`** - Embeds a single text string, returning `List[float]`
4. **`embed_texts(texts: List[str])`** - Embeds multiple strings, returning `List[List[float]]`
5. **`a_embed_text(text: str)`** - Async version of `embed_text()`
6. **`a_embed_texts(texts: List[str])`** - Async version of `embed_texts()`

## Usage in Synthesis

Pass the custom embedder via `ContextConstructionConfig`:

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    context_construction_config=ContextConstructionConfig(
        embedder=CustomEmbeddingModel()
    ))
```

## Notes & Tips

- **Persisting CLI settings**: Use the `--save` flag to persist configurations (see Flags and Configs documentation)
- **Async fallback**: If your embedding model lacks async support, reuse synchronous implementations in async methods
- **JSON errors**: Consult the custom LLMs guide for pydantic confinement solutions if encountering invalid JSON errors

## Related Resources

- [Previous: Using Custom LLMs for Evaluation](/guides/guides-using-custom-llms)
- [Next: Building Custom Metrics](/guides/guides-building-custom-metrics)
- [ContextConstructionConfig Documentation](/docs/synthesizer-generate-from-docs#customize-context-construction)
- [RAG Evaluation](/guides/guides-rag-evaluation)
- [LLM Observability & Monitoring](/guides/guides-llm-observability)
- [Confident AI Cloud Platform](https://www.confident-ai.com/docs)
