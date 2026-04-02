# Tutorial: RAG QA Agent Evaluation

## Read This When
- Want to learn by building a complete RAG QA agent with retriever (FAISS/Chroma) and generator (OpenAI) using LangChain
- Need a worked example of evaluating retriever quality (ContextualRelevancy, Recall, Precision) and generator quality (custom GEval for answer correctness and citation accuracy)
- Looking for a full lifecycle tutorial: development, evaluation, hyperparameter search (chunk size, embedding model, vector store, LLM), and CI/CD integration with `@observe` tracing

## Skip This When
- Building a multi-turn conversational agent rather than a single-turn QA system -- see `references/10-tutorials/20-medical-chatbot.md`
- Need only the guide-level RAG evaluation procedure without a full tutorial -- see `references/09-guides/10-rag-evaluation.md`
- Want API reference for RAG metrics parameters -- see `references/03-eval-metrics/20-rag-metrics.md`

---

## Overview

This tutorial covers building and evaluating a Retrieval-Augmented Generation (RAG) QA agent using LangChain, OpenAI, and DeepEval. The example agent answers questions about Theranos, but all concepts apply to any RAG-based application.

**Technologies**: OpenAI, LangChain, FAISS/Chroma (vector stores), DeepEval

**What is evaluated**:
- Retriever quality: contextual relevancy, recall, and precision
- Generator quality: answer correctness and citation accuracy
- Production tracing: component-level span evaluation

---

## Stage 1: Development

### Architecture

The RAG agent has two primary components:
- **Retriever**: similarity search over a vector store built from document chunks
- **Generator**: LLM that produces answers (and citations) from retrieved context

Both components are kept modular so they can be evaluated and improved independently.

### 1. Create Agent and Load Data

The foundation begins with a `RAGAgent` class that combines retrieval and generation capabilities. The process requires storing data in a vector store — a database containing vector embeddings for efficient similarity searching.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        documents.extend(splitter.create_documents([raw_text]))
        return self.vector_store_class.from_documents(documents, self.embedding_model)
```

Verification example:

```python
document_paths = ["theranos_legacy.txt"]
agent = RAGAgent(document_paths)
print(agent.vector_store)
```

### 2. Creating Retriever

The retriever component identifies the most relevant information from the knowledge base using similarity search functionality.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGAgent:
    # ... Same functions from above

    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k)
        context = [doc.page_content for doc in docs]
        return context
```

Testing the retriever:

```python
doc_path = ["theranos_legacy.txt"]
retriever = RAGAgent(doc_path)
retrieved_docs = retriever.retrieve("How many blood tests can you perform and how much blood do you need?")
print(retrieved_docs)
```

Expected output:

```
[
  'The NanoDrop 3000 is a compact, portable diagnostic device capable of performing over 300 blood tests using just 1-2 microliters of capillary blood. The device integrates microfluidics, spectrometry, and Theranos\'s patented NanoAnalysis Engine to provide lab-grade results in under 20 minutes.',
  'Key Features:\n- Sample volume: 1.2 microliters (average)\n- Test menu: 325+ assays including metabolic, hormonal, infectious, hematologic, and genomic panels',
]
```

### 3. Creating Generator

The generator produces natural language responses by combining user queries with retrieved documents using a language model.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

class RAGAgent:
    # ... Same methods as above

    def generate(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str = None
    ):
        context = "\n".join(retrieved_docs)
        model = llm_model or OpenAI(temperature=0)
        prompt = prompt_template or (
            "Answer the query using the context below.\n\nContext:\n{context}\n\nQuery:\n{query}"
            "Only use information from the context. If nothing relevant is found, respond with: 'No relevant information available.'"
        )
        prompt = prompt.format(context=context, query=query)
        return model(prompt)
```

Testing the generator:

```python
doc_path = ["theranos_legacy.txt"]
query = "How many blood tests can you perform and how much blood do you need?"
retriever = RAGAgent(doc_path)
retrieved_docs = retriever.retrieve(query)
generated_answer = retriever.generate(query, retrieved_docs)
print(generated_answer)
```

Expected output:

```
The NanoDrop 3000 can perform over 325 blood tests using just 1-2 microliters of capillary blood. This enables comprehensive diagnostics with minimal sample volume.
```

### 4. Creating the Answer Function

A unified `answer()` method orchestrates retrieval and generation in a single operation.

```python
class RAGAgent:
    # ... Same functions and imports

    def answer(
        self,
        query: str,
        llm_model=None,
        prompt_template: str = None
    ):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs, llm_model, prompt_template)
        return generated_answer, retrieved_docs
```

Complete example:

```python
document_paths = ["theranos_legacy.txt"]
query = "What is the NanoDrop 3000, and what certifications does Theranos hold?"
retriever = RAGAgent(document_paths)
answer, retrieved_docs = retriever.answer(query)
```

### Updating The RAG Agent: JSON-Formatted Response

Rather than markdown output, implementing JSON formatting enables structured data extraction and improved UI rendering. This approach facilitates citation parsing and flexible response handling.

**Enhanced Prompt Template:**

```
You are a helpful assistant. Use the context below to answer the user's query. Format your response strictly as a JSON object with the following structure:

{
  "answer": "<a concise, complete answer to the user's query>",
  "citations": [
    "<relevant quoted snippet or summary from source 1>",
    "<relevant quoted snippet or summary from source 2>",
    ...
  ]
}

Only include information that appears in the provided context. Do not make anything up.
Only respond in JSON -- No explanations needed. Only use information from the context. If nothing relevant is found, respond with: {
  "answer": "No relevant information available.",
  "citations": []
}

Context:
{context}

Query:
{query}
```

**Updated Answer Function with JSON Parsing:**

```python
import json

class RAGAgent:
    # ... Same functions from above

    def answer(self, query: str):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs)
        try:
            res = json.loads(generated_answer)
            return res
        except json.JSONDecodeError:
            return {"error": "Invalid JSON returned from model", "raw_output": generated_answer}
```

Example JSON response:

```json
{
  "answer": "The NanoDrop 3000 is a compact, portable diagnostic device developed by Theranos Technologies. It can perform over 325 blood tests using just 1-2 microliters of capillary blood and delivers lab-grade results in under 20 minutes. Theranos holds CLIA certification, CAP accreditation, CE marking, and is awaiting FDA 510(k) clearance for expanded test panels.",
  "citations": [
    "The NanoDrop 3000 is a compact, portable diagnostic device capable of performing over 300 blood tests using just 1-2 microliters of capillary blood.",
    "Key Features: Sample volume: 1.2 microliters (average), Test menu: 325+ assays",
    "Theranos labs are CLIA-certified and CAP-accredited. NanoDrop 3000 is CE-marked and pending full FDA 510(k) clearance for expanded panels."
  ]
}
```

---

## Stage 2: Evaluation

### LLMTestCase Structure

RAG evaluation uses single-turn `LLMTestCase` objects. The `retrieval_context` field is required for retriever metrics:

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="...",           # Your query
    actual_output="...",   # The answer from RAG
    retrieval_context="..." # Your retrieved context
)
```

### Two Approaches to Test Data

#### Option 1: Using Historical Data

```python
from deepeval.test_case import LLMTestCase

# Example: Fetch queries and responses from your database
queries = fetch_queries_from_db()  # Your database query here
test_cases = []

for query in queries:
    test_case = LLMTestCase(
        input=query["input"],
        actual_output=query["response"],
        retrieval_context=query["context"]
    )
    test_cases.append(test_case)

print(test_cases)
```

**Limitation**: This approach may not capture current RAG capabilities and does not represent edge cases.

#### Option 2: Generate QA Pairs via Synthesizer (Recommended)

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    # Provide the path to your documents
    document_paths=['theranos_legacy.txt', 'theranos_legacy.docx', 'theranos_legacy.pdf']
)
```

Golden objects differ from test cases: they do not require `actual_output` at creation time and serve as templates. Store them in a cloud dataset:

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(goldens=goldens)
dataset.push(alias="RAG QA Agent Dataset")
```

### Creating Test Cases from a Dataset

```python
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from rag_qa_agent import RAGAgent  # Import your RAG Agent here

dataset = EvaluationDataset()
dataset.pull("RAG QA Agent Dataset")
agent = RAGAgent()
test_cases = []

for golden in dataset.goldens:
    retrieved_docs = agent.retrieve(golden.input)
    response = agent.generate(golden.input, retrieved_docs)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(response),
        retrieval_context=retrieved_docs,
        expected_output=golden.expected_output
    )
    test_cases.append(test_case)

print(len(test_cases))
```

### Retriever Metrics

Three metrics evaluate the quality of retrieved context:

| Metric | Class | What it checks |
|--------|-------|---------------|
| Contextual Relevancy | `ContextualRelevancyMetric` | Retrieved context is relevant to the query |
| Contextual Recall | `ContextualRecallMetric` | Retrieved context sufficiently covers the answer |
| Contextual Precision | `ContextualPrecisionMetric` | Retrieved context is precise without unnecessary noise |

```python
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

relevancy = ContextualRelevancyMetric()
recall = ContextualRecallMetric()
precision = ContextualPrecisionMetric()
```

### Generator Metrics (Custom GEval)

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

answer_correctness = GEval(
    name="Answer Correctness",
    criteria="Evaluate if the actual output's 'answer' property is correct and complete from the input and retrieved context. If the answer is not correct or complete, reduce score.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
)

citation_accuracy = GEval(
    name="Citation Accuracy",
    criteria="Check if the citations in the actual output are correct and relevant based on input and retrieved context. If they're not correct, reduce score.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT]
)
```

### Running Evaluations

```python
from deepeval import evaluate

# Evaluate the retriever
retriever_metrics = [relevancy, recall, precision]
evaluate(test_cases, retriever_metrics)

# Evaluate the generator
generator_metrics = [answer_correctness, citation_accuracy]
evaluate(test_cases, generator_metrics)
```

Run `deepeval view` to open the Confident AI dashboard and inspect results.

---

## Stage 3: Improvement

### Tunable RAG Hyperparameters

| Hyperparameter | What it affects |
|----------------|----------------|
| `vector_store` | Database used for embedding storage and similarity search |
| `embedding_model` | How documents are converted to vector representations |
| `chunk_size` | Length of each document chunk (affects retrieval granularity) |
| `chunk_overlap` | Words shared between chunks (maintains context across boundaries) |
| `k` | Number of documents retrieved per query |
| `llm_model` (generator) | Model used to generate answers |
| `prompt_template` | Instructions given to the generator |

### Pull Dataset for Iteration

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="QA Agent Dataset")
```

### Retriever Hyperparameter Search

```python
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from qa_agent import RAGAgent
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS

dataset = EvaluationDataset()
dataset.pull("QA Agent Dataset")

metrics = [ContextualRelevancyMetric(), ContextualRecallMetric(), ContextualPrecisionMetric()]
chunking_strategies = [500, 1024, 2048]
embedding_models = [
    ("OpenAIEmbeddings", OpenAIEmbeddings()),
    ("HuggingFaceEmbeddings", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")),
]
vector_store_classes = [
    ("FAISS", FAISS),
    ("Chroma", Chroma)
]
document_paths = ["theranos_legacy.txt"]

for chunk_size in chunking_strategies:
    for embedding_name, embedding_model in embedding_models:
        for vector_store_class, vector_store_model in vector_store_classes:
            retriever = RAGAgent(
                document_paths,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                vector_store_class=vector_store_model,
            )
            retriever_test_cases = []
            for golden in dataset.goldens:
                retrieved_docs = retriever.retrieve(golden.input)
                context_list = [doc.page_content for doc in retrieved_docs]
                test_case = LLMTestCase(
                    input=golden.input,
                    actual_output=golden.expected_output,
                    expected_output=golden.expected_output,
                    retrieval_context=context_list
                )
                retriever_test_cases.append(test_case)
            evaluate(
                retriever_test_cases,
                metrics,
                hyperparameters={
                    "chunk_size": chunk_size,
                    "embedding_name": embedding_name,
                    "vector_store_class": vector_store_class
                }
            )
```

**Best configuration found**:
- Chunk Size: 1024
- Embedding Model: OpenAIEmbeddings
- Vector Store: Chroma

| Metric | Score |
|--------|-------|
| Contextual Relevancy | 0.8 |
| Contextual Recall | 0.9 |
| Contextual Precision | 0.8 |

### Prompt Templates

**Original Prompt Template:**

```
You are a helpful assistant. Use the context below to answer the user's query.

Format your response strictly as a JSON object with the following structure:

{
  "answer": "<a concise, complete answer to the user's query>",
  "citations": [
    "<relevant quoted snippet or summary from source 1>",
    "<relevant quoted snippet or summary from source 2>",
    ...
  ]
}

Only include information that appears in the provided context. Do not make anything up.

Only respond in JSON -- No explanations needed. Only use information from the context. If nothing relevant is found, respond with:

{
  "answer": "No relevant information available.",
  "citations": []
}

Context:
{context}

Query:
{query}
```

**Updated Prompt Template:**

```
You are a highly accurate and concise assistant. Your task is to extract and synthesize information strictly from the provided context to answer the user's query.

Respond **only** in the following JSON format:

{
  "answer": "<a clear, complete, and concise answer to the user's query, based strictly on the context>",
  "citations": [
    "<direct quote or summarized excerpt from source 1 that supports the answer>",
    "<direct quote or summarized excerpt from source 2 that supports the answer>",
    ...
  ]
}

Instructions:
- Use only the provided context to form your response. Do not include outside knowledge or assumptions.
- All parts of your answer must be explicitly supported by the context.
- If no relevant information is found, return this exact JSON:

{
  "answer": "No relevant information available.",
  "citations": []
}

Input format:

Context:
{context}

Query:
{query}
```

### Generator Hyperparameter Search

```python
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from langchain.llms import Ollama, OpenAI, HuggingFaceHub
from qa_agent import RAGAgent

metrics = [answer_correctness, citation_accuracy]
prompt_template = "..."  # Use your new system prompt here

models = [
    ("ollama", Ollama(model="llama3")),
    ("openai", OpenAI(model_name="gpt-4")),
    ("huggingface", HuggingFaceHub(repo_id="google/flan-t5-large")),
]

for model_name, model in models:
    retriever = RAGAgent(...)  # Initialize retriever with best config found above
    generator_test_cases = []
    for golden in dataset.goldens:
        answer, retrieved_docs = answer.(golden.input, prompt_template, model)
        context_list = [doc.page_content for doc in retrieved_docs]
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=str(answer),
            retrieval_context=context_list
        )
        generator_test_cases.append(test_case)
    evaluate(
        generator_test_cases,
        metrics,
        hyperparameters={
            "model_name": model_name,
        }
    )
```

**Best generator**: gpt-4

| Metric | Score |
|--------|-------|
| Answer Correctness | 0.8 |
| Citation Accuracy | 0.9 |

### Improved RAGAgent with @observe Tracing

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from deepeval.tracing import observe

class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()
        self.persist_directory = tempfile.mkdtemp()

    def _load_vector_store(self):
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            documents.extend(splitter.create_documents([raw_text]))
        return self.vector_store_class.from_documents(
            documents, self.embedding_model,
            persist_directory=self.persist_directory
        )

    @observe()
    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k)
        context = [doc.page_content for doc in docs]
        return context

    @observe()
    def generate(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str = None
    ):
        context = "\n".join(retrieved_docs)
        model = llm_model or OpenAI(model_name="gpt-4")
        prompt = prompt_template or (
            "You are an AI assistant designed for factual retrieval. Using the context below, "
            "extract only the information needed to answer the user's query. Respond in strictly "
            "valid JSON using the schema below.\n\n"
            "Response schema:\n"
            "{\n"
            '  "answer": "string -- a precise, factual answer found in the context",\n'
            '  "citations": [\n'
            '    "string -- exact quotes or summaries from the context that support the answer"\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Do not fabricate any information or cite anything not present in the context.\n"
            "- Do not include explanations or formatting -- only return valid JSON.\n"
            "- Use complete sentences in the answer.\n"
            "- Limit the answer to the scope of the context.\n"
            "- If no answer is found in the context, return:\n"
            "{\n"
            '  "answer": "No relevant information available.",\n'
            '  "citations": []\n'
            "}\n\n"
            "Context:\n{context}\n\nQuery:\n{query}"
        )
        prompt = prompt.format(context=context, query=query)
        return model(prompt)

    @observe()
    def answer(self):
        ...  # Remains same
```

Example output:

```json
{
  "answer": "The NanoDrop 3000 is a compact, portable diagnostic device developed by Theranos Technologies. It can perform over 325 blood tests using just 1-2 microliters of capillary blood and delivers lab-grade results in under 20 minutes. Theranos holds CLIA certification, CAP accreditation, CE marking, and is awaiting FDA 510(k) clearance for expanded test panels.",
  "citations": [
    "According to Theranos Technologies Inc., the NanoDrop 3000 is capable of running over 325 diagnostic tests using only 1-2 microliters of blood, delivering results in under 20 minutes through its proprietary microfluidic and NanoAnalysis technologies.",
    "Theranos states that the device holds CLIA certification, CAP accreditation, and CE marking, and is currently pending FDA 510(k) clearance for expanded diagnostic panels."
  ]
}
```

---

## Stage 4: Production (Evals in Prod)

### Setup Tracing with Metrics on Spans

Apply metrics directly to span decorators for continuous evaluation during production:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    GEval,
)
from deepeval.dataset import EvaluationDataset
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

class RAGAgent:
    def __init__(...):
        ...

    def _load_vector_store(self):
        ...

    @observe(metrics=[ContextualRelevancyMetric(), ContextualRecallMetric(), ContextualPrecisionMetric()], name="Retriever")
    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k)
        context = [doc.page_content for doc in docs]
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output="...",
                expected_output="...",
                retrieval_context=context
            )
        )
        return context

    @observe(metrics=[GEval(...), GEval(...)], name="Generator")
    def generate(
        self,
        query: str,
        retrieved_docs: list,
        llm_model=None,
        prompt_template: str = None
    ):
        context = "\n".join(retrieved_docs)
        model = llm_model or OpenAI(model_name="gpt-4")
        prompt = prompt_template or (
            "You are an AI assistant designed for factual retrieval. Using the context below, "
            "extract only the information needed to answer the user's query. Respond in strictly "
            "valid JSON using the schema below.\n\n"
            "Response schema:\n"
            "{\n"
            '  "answer": "string -- a precise, factual answer found in the context",\n'
            '  "citations": [\n'
            '    "string -- exact quotes or summaries from the context that support the answer"\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Do not fabricate any information or cite anything not present in the context.\n"
            "- Do not include explanations or formatting -- only return valid JSON.\n"
            "- Use complete sentences in the answer.\n"
            "- Limit the answer to the scope of the context.\n"
            "- If no answer is found in the context, return:\n"
            "{\n"
            '  "answer": "No relevant information available.",\n'
            '  "citations": []\n'
            "}\n\n"
            "Context:\n{context}\n\nQuery:\n{query}"
        )
        prompt = prompt.format(context=context, query=query)
        answer = model(prompt)
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=retrieved_docs
            )
        )
        return answer

    @observe(type="agent")
    def answer(
        self,
        query: str,
        llm_model=None,
        prompt_template: str = None
    ):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs, llm_model, prompt_template)
        return generated_answer, retrieved_docs
```

### Using Datasets in Production

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="QA Agent Dataset")
```

### CI/CD Integration

**Test file** (`test_rag_qa_agent.py`):

```python
import pytest
from deepeval.dataset import EvaluationDataset
from qa_agent import RAGAgent
from deepeval import assert_test

dataset = EvaluationDataset()
dataset.pull(alias="QA Agent Dataset")
agent = RAGAgent()

@pytest.mark.parametrize("golden", dataset.goldens)
def test_meeting_summarizer_components(golden):
    assert_test(golden=golden, observed_callback=agent.answer)
```

**Run locally**:

```bash
poetry run deepeval test run test_rag_qa_agent.py
```

**GitHub Actions workflow** (`.github/workflows/deepeval.yml`):

```yaml
name: RAG QA Agent DeepEval Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install Poetry
        run: |
          curl -sSL https://install.python-poetry.org | python3 -
          echo "$HOME/.local/bin" >> $GITHUB_PATH
      - name: Install Dependencies
        run: poetry install --no-root
      - name: Run DeepEval Unit Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CONFIDENT_API_KEY: ${{ secrets.CONFIDENT_API_KEY }}
        run: poetry run deepeval test run test_rag_qa_agent.py
```

### Production Principles

- Use `@observe(metrics=[...], name="...")` on both `retrieve()` and `generate()` for component-level evaluation
- Use `update_current_span(test_case=LLMTestCase(...))` to supply the span with data needed for metric computation
- Use `@observe(type="agent")` on the top-level `answer()` method
- Pull datasets from Confident AI in CI to keep test data synchronized
- Use `assert_test()` with `observed_callback` to run evaluation inside pytest
- Store API keys as GitHub secrets (`OPENAI_API_KEY`, `CONFIDENT_API_KEY`)
