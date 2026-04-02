# Synthetic Data Generation: Generation Methods

## Read This When
- You need the exact method signature and parameters for `generate_goldens_from_docs`, `from_contexts`, `from_scratch`, or `from_goldens`
- You want to configure `ContextConstructionConfig` for document-based golden generation (chunk size, overlap, quality thresholds)
- You are choosing between generation methods and need the comparison table

## Skip This When
- You need `Synthesizer` constructor setup, `FiltrationConfig`, `EvolutionConfig`, or save/load patterns -- see [10-synthesizer-overview.md](./10-synthesizer-overview.md)
- You are looking for evaluation metrics to score your generated data -- see [../03-eval-metrics/](../03-eval-metrics/)

---

Four generation methods are available in the `Synthesizer`, each with a single-turn and multi-turn variant. This document covers all eight method signatures with complete parameters and examples.

See [10-synthesizer-overview.md](./10-synthesizer-overview.md) for `Synthesizer` constructor, `FiltrationConfig`, `EvolutionConfig`, `StylingConfig`, and save/load patterns.

---

## Method 1: Generate From Documents

The most complete method. Includes an automatic **context construction** step that parses documents, chunks them, embeds chunks, and groups similar chunks into contexts. Best for RAG systems where you want to generate test data directly from a knowledge base.

**Prerequisites:**

```bash
pip install chromadb langchain-core langchain-community langchain-text-splitters
```

### Single-Turn

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf',
                    'example.md', 'example.markdown', 'example.mdx'],
    include_expected_output=True,
    max_goldens_per_context=2,
    context_construction_config=ContextConstructionConfig()  # Optional
)
```

### Multi-Turn

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
conversational_goldens = synthesizer.generate_conversational_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf'],
    include_expected_outcome=True,
    max_goldens_per_context=2,
    context_construction_config=ContextConstructionConfig()  # Optional
)
```

### Parameters

#### generate_goldens_from_docs()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `document_paths` | `List[str]` | Yes | — | Paths to documents. Supports: `.txt`, `.docx`, `.pdf`, `.md`, `.markdown`, `.mdx` |
| `include_expected_output` | bool | No | `True` | Generate `expected_output` for each Golden |
| `max_goldens_per_context` | int | No | `2` | Maximum goldens per context |
| `context_construction_config` | `ContextConstructionConfig` | No | Default values | Customize context quality and attributes |

#### generate_conversational_goldens_from_docs()

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `document_paths` | `List[str]` | Yes | — | Paths to documents. Supports: `.txt`, `.docx`, `.pdf`, `.md`, `.markdown`, `.mdx` |
| `include_expected_outcome` | bool | No | `True` | Generate `expected_outcome` for ConversationalGolden |
| `max_goldens_per_context` | int | No | `2` | Maximum goldens per context |
| `context_construction_config` | `ContextConstructionConfig` | No | Default values | Customize context quality and attributes |

**Note:** Final maximum goldens = `max_goldens_per_context` x `max_contexts_per_document` (from `ContextConstructionConfig`)

### ContextConstructionConfig

The context construction config is unique to this method. It controls how documents are parsed, chunked, and grouped into contexts.

```python
from deepeval.synthesizer.config import ContextConstructionConfig

config = ContextConstructionConfig(
    critic_model="gpt-4.1",
    encoding=None,                     # Auto-detected
    max_contexts_per_document=3,
    min_contexts_per_document=1,
    max_context_length=3,              # In chunks
    min_context_length=1,              # In chunks
    chunk_size=1024,                   # In tokens
    chunk_overlap=0,
    context_quality_threshold=0.5,
    context_similarity_threshold=0.5,
    max_retries=3,
    embedder='text-embedding-3-small'
)

synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt'],
    context_construction_config=config
)
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `critic_model` | str or `DeepEvalBaseLLM` | No | Synthesizer's model or `gpt-4.1` | Model for determining context quality scores |
| `encoding` | str | No | Auto-detected | Encoding for text-based files (`.txt`, `.md`, `.markdown`, `.mdx`) |
| `max_contexts_per_document` | int | No | `3` | Maximum contexts per document |
| `min_contexts_per_document` | int | No | `1` | Minimum contexts per document |
| `max_context_length` | int | No | `3` | Maximum text chunks per context |
| `min_context_length` | int | No | `1` | Minimum text chunks per context |
| `chunk_size` | int | No | `1024` | Text chunk size in tokens |
| `chunk_overlap` | int | No | `0` | Overlap between consecutive chunks in tokens |
| `context_quality_threshold` | float | No | `0.5` | Minimum quality threshold (0-1) |
| `context_similarity_threshold` | float | No | `0.5` | Minimum cosine similarity for context grouping |
| `max_retries` | int | No | `3` | Retry attempts for quality/similarity filtering |
| `embedder` | str or `DeepEvalBaseEmbeddingModel` | No | `'text-embedding-3-small'` | Embedding model for parsing and grouping |

**Note:** The `critic_model` in `ContextConstructionConfig` can differ from the `FiltrationConfig` critic model — they serve different purposes.

### Context Construction Pipeline

Context construction runs before golden generation and has three sequential steps:

**Step 1: Document Parsing**
- Splits documents into chunks using `TokenTextSplitter`
- `chunk_size` and `chunk_overlap` operate at token level, not character level
- Chunks are embedded and stored in a vector database (ChromaDB)
- Raises an error if `chunk_size` is too large to generate unique contexts

**Step 2: Context Selection**
- Random nodes selected from vector database undergo quality filtering
- LLM (`critic_model`) scores chunks (0-1) on Clarity, Depth, Structure, Relevance
- Fallback: if quality < threshold after `max_retries`, uses highest-scoring context

**Step 3: Context Grouping**
- Selected nodes grouped with up to `max_context_length` other nodes
- Criterion: cosine similarity score above `context_similarity_threshold`
- Fallback: if similarity < threshold after `max_retries`, uses highest-similarity context

**Full workflow:** Documents → Chunks → Embedded nodes → Random selection → Quality filtering → Similar node grouping → Contexts → Golden generation

### Chunking Best Practices

```
Number of Chunks = ceil((Document Length - chunk_overlap) / (chunk_size - chunk_overlap))
```

1. **Align with your retriever**: Match chunk size and overlap with your retriever's expectations to prevent context mismatch
2. **Balance size and overlap**: Use small overlap (50-100 tokens) for interconnected content; larger chunks with minimal overlap for distinct sections
3. **Preserve document structure**: Respect natural breaks (chapters, sections, headings)
4. **Maximize coverage**: Increase `max_contexts_per_document` for large datasets

---

## Method 2: Generate From Contexts

Use this when you already have prepared contexts (e.g., from an embedded knowledge base or vector database). Skips document parsing entirely.

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=[
        ["The Earth revolves around the Sun.", "Planets are celestial bodies."],
        ["Water freezes at 0 degrees Celsius.", "The chemical formula for water is H2O."],
    ]
)
```

### Single-Turn Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contexts` | list | Yes | List of contexts; each context is a list of strings sharing common themes |
| `include_expected_output` | boolean | No | Generate `expected_output` for each Golden (default: `True`) |
| `max_goldens_per_context` | integer | No | Maximum goldens per context (default: `2`) |
| `source_files` | list | No | Source file names; length must match `contexts` |

### Multi-Turn

```python
conversational_goldens = synthesizer.generate_conversational_goldens_from_contexts(
    contexts=[
        ["The Earth revolves around the Sun.", "Planets are celestial bodies."],
        ["Water freezes at 0 degrees Celsius.", "The chemical formula for water is H2O."],
    ]
)
```

### Multi-Turn Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contexts` | list | Yes | List of contexts; each context is a list of strings sharing common themes |
| `include_expected_outcome` | boolean | No | Generate `expected_outcome` for each ConversationalGolden (default: `True`) |
| `max_goldens_per_context` | integer | No | Maximum goldens per context (default: `2`) |
| `source_files` | list | No | Source file names; length must match `contexts` |

**Key distinction from docs method:** Skips context construction. No ChromaDB or langchain dependencies required.

---

## Method 3: Generate From Scratch

Generates synthetic test data without any documents or existing contexts. Useful when your LLM application does not rely on RAG, or when you want to test on queries beyond the existing knowledge base.

`StylingConfig` (or `ConversationalStylingConfig`) is **required** to guide the generation since there is no context to ground it.

### Single-Turn

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
    input_format="Questions in English that asks for data in database.",
    expected_output_format="SQL query based on the given input",
    task="Answering text-to-SQL-related queries by querying a database and returning the results to users",
    scenario="Non-technical users trying to query a database using plain English.",
)
synthesizer = Synthesizer(styling_config=styling_config)

goldens = synthesizer.generate_goldens_from_scratch(num_goldens=25)
print(goldens)
```

### Multi-Turn

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ConversationalStylingConfig

conversational_styling_config = ConversationalStylingConfig(
    conversational_task="Answering text-to-SQL-related queries by querying a database and returning the results to users",
    scenario_context="Non-technical users trying to query a database using plain English.",
    participant_roles="Non-technical users trying to query a database using plain English."
)
synthesizer = Synthesizer(conversational_styling_config=conversational_styling_config)

conversational_goldens = synthesizer.generate_conversational_goldens_from_scratch(num_goldens=25)
print(conversational_goldens)
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `num_goldens` | integer | **Required** | The number of synthetic test cases to generate |

**Note:** `StylingConfig` is set at `Synthesizer` instantiation, not passed to the method. See [10-synthesizer-overview.md](./10-synthesizer-overview.md) for `StylingConfig` parameters.

---

## Method 4: Generate From Goldens

Augments an existing set of goldens to expand your evaluation dataset and add complexity to existing test cases. Does not require documents or context.

**Important:** By default, the method extracts `StylingConfig` from existing goldens. Providing an explicit `StylingConfig` at `Synthesizer` instantiation is recommended for better accuracy and consistency.

### Single-Turn

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
new_goldens = synthesizer.generate_goldens_from_goldens(
    goldens=existing_goldens,
    max_goldens_per_golden=2,
    include_expected_output=True,
)
```

### Single-Turn Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `goldens` | List | Yes | — | Existing Goldens from which new ones are generated |
| `max_goldens_per_golden` | integer | No | `2` | Maximum new goldens per input golden |
| `include_expected_output` | boolean | No | `True` | Generate `expected_output` for each synthetic Golden |

**Critical warning:** The generated goldens will contain `expected_output` **only** if your existing goldens contain `context`. This ensures outputs are grounded in truth rather than hallucinated.

### Multi-Turn

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
conversational_goldens = synthesizer.generate_conversational_goldens_from_goldens(
    goldens=existing_goldens,
    max_goldens_per_golden=2,
    include_expected_outcome=True,
)
```

### Multi-Turn Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `goldens` | List | Yes | — | Existing Goldens for generation |
| `max_goldens_per_golden` | integer | No | `2` | Maximum new goldens per input |
| `include_expected_outcome` | boolean | No | `True` | Generate `expected_outcome` for ConversationalGolden |

### Context Handling

- **With context**: When existing goldens include `context`, the synthesizer uses these contexts for grounding synthetic goldens in truth
- **Without context**: Falls back to the `generate_from_scratch` approach to create additional inputs from provided inputs

---

## Method Comparison

| Method | When to Use | Context Required | Docs/ChromaDB Required |
|--------|-------------|-----------------|----------------------|
| `from_docs` | Starting from raw files; RAG systems | No (auto-constructed) | Yes |
| `from_contexts` | Already have parsed contexts/vector DB | Yes (you provide) | No |
| `from_scratch` | No RAG; testing general capabilities | No | No |
| `from_goldens` | Augmenting an existing dataset | Optional (improves quality) | No |

---

## Complete Example: Full Workflow

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    FiltrationConfig, EvolutionConfig, StylingConfig, ContextConstructionConfig
)
from deepeval.synthesizer import Evolution
from deepeval.dataset import EvaluationDataset

# 1. Configure synthesizer
synthesizer = Synthesizer(
    model="gpt-4.1",
    filtration_config=FiltrationConfig(
        critic_model="gpt-4.1",
        synthetic_input_quality_threshold=0.5,
        max_quality_retries=3
    ),
    evolution_config=EvolutionConfig(
        evolutions={
            Evolution.REASONING: 0.2,
            Evolution.MULTICONTEXT: 0.2,
            Evolution.CONCRETIZING: 0.2,
            Evolution.CONSTRAINED: 0.2,
            Evolution.IN_BREADTH: 0.2,
        },
        num_evolutions=2
    ),
    styling_config=StylingConfig(
        task="Answering customer support queries",
        scenario="Customers seeking help with product issues"
    )
)

# 2. Generate goldens
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['knowledge_base.pdf', 'faq.md'],
    include_expected_output=True,
    max_goldens_per_context=3,
    context_construction_config=ContextConstructionConfig(
        chunk_size=512,
        chunk_overlap=50,
        max_contexts_per_document=5,
        context_quality_threshold=0.6
    )
)

# 3. Inspect quality
df = synthesizer.to_pandas()
print(df[["input", "context_quality", "synthetic_input_quality"]].head())

# 4. Save locally
synthesizer.save_as(
    file_type='json',
    directory="./synthetic_data",
    file_name="customer_support_goldens"
)

# 5. Push to Confident AI
dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
dataset.push(alias="Customer Support Dataset v1")
```

---

## Related Documentation

- [10-synthesizer-overview.md](./10-synthesizer-overview.md) - Synthesizer class, FiltrationConfig, EvolutionConfig, StylingConfig, save/load
- [../06-red-teaming/10-introduction.md](../06-red-teaming/10-introduction.md) - Red teaming for safety evaluation
