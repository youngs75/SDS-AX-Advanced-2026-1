# DeepEval Vector Database Integrations

Source: https://deepeval.com/integrations/vector-databases/
Fetched: 2026-02-20

## Read This When
- Using Chroma, Elasticsearch, PGVector, or Qdrant as your vector store and need setup + evaluation integration with DeepEval
- Want to evaluate and optimize retrieval performance by tuning top-K, embedding models, vector dimensions, or distance metrics using contextual metrics
- Need code patterns for storing embeddings, running similarity search, and creating LLMTestCases from vector database results

## Skip This When
- Need the conceptual guide for RAG evaluation workflows rather than database-specific setup -- see `references/09-guides/10-rag-evaluation.md`
- Looking for framework integrations (LangChain, OpenAI, Anthropic) rather than vector databases -- see `references/11-integrations/20-frameworks.md`
- Want API reference for contextual metrics (ContextualPrecision, ContextualRecall, ContextualRelevancy) -- see `references/03-eval-metrics/20-rag-metrics.md`

---

## Table of Contents

1. [Chroma](#1-chroma)
2. [Elasticsearch](#2-elasticsearch)
3. [PGVector](#3-pgvector)
4. [Qdrant](#4-qdrant)

---

## 1. Chroma

Source: https://deepeval.com/integrations/vector-databases/chroma

### Quick Summary

**Chroma** is described as "one of the most popular open-source AI application databases" supporting retrieval features including embeddings storage, vector search, document storage, metadata filtering, and multi-modal retrieval.

DeepEval enables evaluation and optimization of Chroma retrievers by tuning hyperparameters such as `n_results` (top-K) and the embedding model used in retrieval pipelines.

### Important Dependencies
Chroma functions as both an optional retriever for evaluation AND a required dependency for `deepeval.synthesizer.generate_goldens_from_docs()`. This method uses Chroma as its built-in backend for chunk storage and retrieval during context construction.

---

### Installation

```bash
pip install chromadb
```

---

### Setup Instructions

#### Initialize Chroma Client

```python
import chromadb

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection
collection = client.get_or_create_collection(name="rag_documents")
```

#### Configure Embeddings & Store Documents

```python
from sentence_transformers import SentenceTransformer

# Load an embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example document chunks
document_chunks = [
    "Chroma is an open-source vector database for efficient embedding retrieval.",
    "It enables fast semantic search using vector similarity.",
    "Chroma retrieves relevant data with cosine similarity.",
    ...
]

# Store chunks with embeddings in Chroma
for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        metadatas=[{"text": chunk}]
    )
```

**Note:** By default, Chroma utilizes cosine similarity to find similar chunks.

---

### Evaluating Chroma Retrieval

#### Preparing Test Cases

Define a search function to retrieve relevant contexts:

```python
def search(query):
    query_embedding = model.encode(query).tolist()
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # Retrieve top-K matches
    )
    return res["metadatas"][0][0]["text"] if res["metadatas"][0] else None

query = "How does Chroma work?"
retrieval_context = search(query)
```

Generate LLM response using retrieved context:

```python
prompt = """Answer the user question based on the supporting context.
User Question:{input}
Supporting Context:{retrieval_context}"""

actual_output = generate(prompt)  # Replace with your LLM function
print(actual_output)
print(expected_output)
```

Example outputs:
- **actual_output:** "Chroma is a lightweight vector database designed for AI applications, enabling fast semantic retrieval."
- **expected_output:** "Chroma is an open-source vector database that enables fast retrieval using cosine similarity."

Create the test case:

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output=expected_output
)
```

#### Running Evaluations

Define relevant metrics:

```python
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()
```

Execute evaluation:

```python
from deepeval import evaluate

evaluate(
    [test_case],
    metrics=[contextual_recall, contextual_precision, contextual_relevancy]
)
```

---

### Improving Chroma Retrieval Performance

#### Performance Optimization Example

The documentation provides a scenario where Contextual Relevancy scores are below threshold:

| Input | Contextual Relevancy | Contextual Recall |
|-------|----------------------|-------------------|
| "How does Chroma work?" | 0.45 | 0.85 |
| "What is the retrieval process in Chroma?" | 0.43 | 0.92 |
| "Explain Chroma's vector database." | 0.55 | 0.67 |

#### Tuning Strategy

Adjust document length or modify `n_results` to retrieve more relevant contexts. The Contextual Relevancy metric evaluates both retrieved text chunks and top-K selection.

```python
def search(query, n_results):
    query_embedding = model.encode(query).tolist()
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return res["metadatas"][0][0]["text"] if res["metadatas"][0] else None

# Iterate over different top-K values
for top_k in [3, 5, 7]:
    retrieval_context = search(input_query, top_k)
    # Define test case and evaluate
    evaluate(
        [test_case],
        metrics=[contextual_recall, contextual_precision, contextual_relevancy]
    )
```

---

### Related Resources

- [Chroma Official Website](https://www.trychroma.com/)
- [RAG Evaluation Guide](/guides/guides-rag-evaluation)
- [Metric Details](/docs/metrics-contextual-precision)
- [Confident AI Platform](https://www.confident-ai.com/)

---

## 2. Elasticsearch

Source: https://deepeval.com/integrations/vector-databases/elasticsearch

### Quick Summary

DeepEval facilitates evaluation of your Elasticsearch retriever and optimization of retrieval hyperparameters including `top-K`, `embedding model`, and `similarity function`.

**Installation:**
```bash
pip install elasticsearch
```

Elasticsearch functions as "a fast and scalable search engine that works as a high-performance vector database for RAG applications" with efficient handling of large-scale retrieval workloads suitable for production environments.

---

### Setup Elasticsearch

#### Step 1: Connect to Elastic Cluster

```python
import os
from elasticsearch import Elasticsearch

username = 'elastic'
password = os.getenv('ELASTIC_PASSWORD')  # Value set in environment variable
client = Elasticsearch(
    "http://localhost:9200",
    basic_auth=(username, password)
)
```

#### Step 2: Create Index with Type Mappings

```python
# Create index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body={
        "mappings": {
            "properties": {
                "text": {"type": "text"},  # Stores chunk text
                "embedding": {"type": "dense_vector", "dims": 384}
            }
        }
    })
```

#### Step 3: Define Embedding Model and Index Documents

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

document_chunks = [
    "Elasticsearch is a distributed search engine.",
    "RAG improves AI-generated responses with retrieved context.",
    "Vector search enables high-precision semantic retrieval.",
]

# Store chunks with embeddings
for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()
    es.index(index=index_name, id=i, body={"text": chunk, "embedding": embedding})
```

---

### Evaluating Elasticsearch Retrieval

Evaluation consists of two steps:

1. Prepare an `input` query with expected LLM response; generate actual response from RAG pipeline; create `LLMTestCase`
2. Evaluate the test case using retrieval metrics

#### Preparing Your Test Case

**Step 1: Retrieve relevant context from Elasticsearch**

```python
def search(query):
    query_embedding = model.encode(query).tolist()
    res = es.search(index=index_name, body={
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 3,  # Retrieve top matches
            "num_candidates": 10  # Controls search speed vs accuracy
        }
    })
    return res["hits"]["hits"][0]["_source"]["text"] if res["hits"]["hits"] else None

query = "How does Elasticsearch work?"
retrieval_context = search(query)
```

**Step 2: Generate LLM response with retrieved context**

```python
prompt = """Answer the user question based on the supporting context
User Question: {input}
Supporting Context: {retrieval_context}"""

actual_output = generate(prompt)  # Replace with your LLM
print(actual_output)
```

Example output:
```
Elasticsearch indexes document chunks using an inverted index for fast
full-text search and retrieval.
```

**Step 3: Create LLMTestCase**

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output="Elasticsearch uses inverted indexes for keyword searches and dense vector similarity for semantic search.",
)
```

#### Running Evaluations

```python
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)

contextual_recall = ContextualRecallMetric()
contextual_precision = ContextualPrecisionMetric()
contextual_relevancy = ContextualRelevancyMetric()

from deepeval import evaluate

evaluate(
    [test_case],
    metrics=[contextual_recall, contextual_precision, contextual_relevancy]
)
```

---

### Improving Elasticsearch Retrieval

| Metric | Score |
|--------|-------|
| Contextual Precision | 0.85 |
| Contextual Recall | 0.92 |
| Contextual Relevancy | 0.44 |

Each contextual metric evaluates a specific hyperparameter. Experiment with various hyperparameter combinations and prepare test cases using different retriever versions. Analyzing contextual metric score improvements and regressions determines optimal hyperparameter combinations.

---

### Related Resources

- [RAG Evaluation Guide](/guides/guides-rag-evaluation)
- [Optimizing Hyperparameters Guide](/guides/guides-optimizing-hyperparameters)
- [Elasticsearch Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/getting-started.html)

---

## 3. PGVector

Source: https://deepeval.com/integrations/vector-databases/pgvector

### Overview

PGVector serves as "an open-source PostgreSQL extension that enables semantic search" within databases, functioning as a retrieval component in RAG pipelines.

### Installation

```bash
pip install psycopg2 pgvector
```

---

### Setup Instructions

#### Database Connection

```python
import psycopg2
import os

conn = psycopg2.connect(
    dbname="your_database",
    user="your_user",
    password=os.getenv("PG_PASSWORD"),
    host="localhost",
    port="5432")
cursor = conn.cursor()
```

#### Enable Extension and Create Table

```python
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding vector(384)
    );""")
conn.commit()
```

#### Store Embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
document_chunks = [
    "PGVector brings vector search to PostgreSQL.",
    "RAG improves AI-generated responses with retrieved context.",
    "Vector search enables high-precision semantic retrieval.",
]

for chunk in document_chunks:
    embedding = model.encode(chunk).tolist()
    cursor.execute(
        "INSERT INTO documents (text, embedding) VALUES (%s, %s);",
        (chunk, embedding)
    )
conn.commit()
```

---

### Evaluation Workflow

#### Similarity Search

```python
def search(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    cursor.execute("""
        SELECT text FROM documents
        ORDER BY embedding <-> %s
        LIMIT %s;
    """, (query_embedding, top_k))
    return [row[0] for row in cursor.fetchall()]
```

#### Test Case Creation

```python
from deepeval.test_case import LLMTestCase

query = "How does PGVector work?"
retrieval_context = search(query)

test_case = LLMTestCase(
    input=query,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output="PGVector is an extension that brings efficient vector search.",
)
```

#### Running Metrics

```python
from deepeval import evaluate
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)

evaluate(
    [test_case],
    metrics=[
        ContextualRecallMetric(),
        ContextualPrecisionMetric(),
        ContextualRelevancyMetric()
    ])
```

---

### Performance Optimization

**Suggested improvements for low precision scores:**

- Domain-specific models: `BAAI/bge-small-en`, `sentence-transformers/msmarco-distilbert-base-v4`, `nomic-ai/nomic-embed-text-v1`
- Adjust `LIMIT` parameter for retrieval result count
- Re-evaluate after parameter tuning

### Related Resources

- [RAG Evaluation Guide](/guides/guides-rag-evaluation)
- [GitHub Repository](https://github.com/confident-ai/deepeval)
- [Confident AI Platform](https://www.confident-ai.com/)

---

## 4. Qdrant

Source: https://deepeval.com/integrations/vector-databases/qdrant

### Quick Summary

Qdrant serves as "a vector database and vector similarity search engine that is **optimized for fast retrieval**." The system is implemented in Rust and achieves "3ms response for 1M Open AI Embeddings" with integrated memory compression capabilities.

**Installation:**
```bash
pip install qdrant-client
```

DeepEval enables optimization of Qdrant retriever performance by configuring hyperparameters including vector dimensionality, distance metrics, embedding models, and top-K limits.

---

### Setup Qdrant

#### Creating a Client Connection

```python
import qdrant_client
import os

client = qdrant_client.QdrantClient(
    url="http://localhost:6333"  # Change this if using Qdrant Cloud
)
```

#### Creating a Collection

```python
collection_name = "documents"

# Create collection if it doesn't exist
if collection_name not in [col.name for col in client.get_collections().collections]:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_client.http.models.VectorParams(
            size=384,  # Vector dimensionality
            distance="cosine"  # Similarity function
        ),
    )
```

**Configuration Note:** Iterate and test different hyperparameter values for `size` and `distance` if evaluation scores don't meet expectations.

#### Adding Documents to Collection

```python
from sentence_transformers import SentenceTransformer

# Load an embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example document chunks
document_chunks = [
    "Qdrant is a vector database optimized for fast similarity search.",
    "It uses HNSW for efficient high-dimensional vector indexing.",
    "Qdrant supports disk-based storage for handling large datasets.",
    ...
]

# Store chunks with embeddings
for i, chunk in enumerate(document_chunks):
    embedding = model.encode(chunk).tolist()  # Convert text to vector
    client.upsert(
        collection_name=collection_name,
        points=[
            qdrant_client.http.models.PointStruct(
                id=i, vector=embedding, payload={"text": chunk}
            )
        ]
    )
```

---

### Evaluating Qdrant Retrieval

#### Preparing Your Test Case

**Define Search Function:**

```python
def search(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k  # Retrieve the top K most similar results
    )
    return [hit.payload["text"] for hit in search_results] if search_results else None

query = "How does Qdrant work?"
retrieval_context = search(query)
```

**Generate Response:**

```python
prompt = """Answer the user question based on the supporting context

User Question:{input}

Supporting Context:{retrieval_context}"""

actual_output = generate(prompt) # hypothetical function, replace with your own LLM
print(actual_output)
```

**Create Test Case:**

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=input,
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output="Qdrant is a powerful vector database optimized for semantic search and retrieval.",
)
```

**Example Output:**
```
Qdrant is a scalable vector database optimized for high-performance retrieval.
```

#### Running Evaluations

```python
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)

contextual_recall = ContextualRecallMetric()
contextual_precision = ContextualPrecisionMetric()
contextual_relevancy = ContextualRelevancyMetric()

evaluate(
    [test_case],
    metrics=[contextual_recall, contextual_precision, contextual_relevancy]
)
```

**Recommendation:** Use `ContextualRecallMetric`, `ContextualPrecisionMetric`, and `ContextualRelevancyMetric` to assess retriever effectiveness unless custom evaluation criteria apply.

---

### Improving Qdrant Retrieval

#### Key Findings Example

| Query | Contextual Precision Score | Contextual Recall Score |
|-------|----------------------------|------------------------|
| "How does Qdrant store vector data?" | 0.39 | 0.92 |
| "Explain Qdrant's indexing method." | 0.35 | 0.89 |
| "What makes Qdrant efficient for retrieval?" | 0.42 | 0.83 |

#### Addressing Low Precision

When precision scores fall below expectations, consider:

**1. Using Domain-Specific Embedding Models**
- `BAAI/bge-small-en` for improved retrieval ranking
- `sentence-transformers/msmarco-distilbert-base-v4` for dense passage retrieval
- `nomic-ai/nomic-embed-text-v1` for extended document retrieval

**2. Adjusting Vector Dimensions**
Ensure vector dimensions in Qdrant align with embedding model outputs.

**3. Filtering Less Relevant Results**
Apply metadata filters to exclude unrelated chunks and improve precision.

#### Next Steps

After testing alternative embedding models or hyperparameters:
- Generate new test cases
- Re-evaluate retrieval quality to measure improvements
- Monitor Contextual Precision increases indicating focused context retrieval
- Consider tracking evaluations in Confident AI for deeper performance insights

---

### Related Resources

- **Previous:** [Weaviate Integration](/integrations/vector-databases/weaviate)
- **Next:** [PGVector Integration](/integrations/vector-databases/pgvector)
- **Official Qdrant Documentation:** [https://qdrant.tech/documentation/](https://qdrant.tech/documentation/)
- **RAG Evaluation Guide:** [/guides/guides-rag-evaluation](/guides/guides-rag-evaluation)

---

### Notes & Tips

**Installation:** `pip install qdrant-client` enables quick Python integration.

**Tip:** Iterate hyperparameter values and explore Qdrant documentation for advanced configurations.

**Info:** Use Confident AI platform for tracking evaluations, viewing results, and monitoring online performance.
