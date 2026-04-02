# Synthesizer — Reference Guide

Source: guides-using-synthesizer.md
URL: https://deepeval.com/guides/guides-using-synthesizer

## Read This When
- Need to generate synthetic test data (goldens) from documents for LLM evaluation at scale
- Want to understand chunking strategies, evolution types (Reasoning, Multicontext, In-breadth, etc.), and quality filtering
- Setting up `generate_goldens_from_docs()` or `generate_goldens_from_contexts()` with custom configurations

## Skip This When
- Need API reference for Synthesizer class parameters and ContextConstructionConfig -- see `references/05-synthetic-data/`
- Looking for a complete tutorial that uses synthesized data within a RAG evaluation pipeline -- see `references/10-tutorials/20-rag-qa-agent.md`
- Want to use custom embedding models with the Synthesizer -- see `references/09-guides/40-custom-llms-and-embeddings.md`

---

# Generate Synthetic Test Data for LLM Applications

## Overview

DeepEval's Synthesizer enables rapid creation of thousands of "high-quality synthetic goldens" in minutes, eliminating manual curation and capturing overlooked edge cases. A Golden differs from an LLMTestCase by not requiring `actual_output` and `retrieval_context` at initialization.

## Key Steps in Data Synthetic Generation

The `generate_goldens_from_docs()` function transforms documents through these stages:

1. **Document Loading**: Process knowledge base documents for chunking
2. **Document Chunking**: Split documents into manageable segments
3. **Context Generation**: Group similar chunks using cosine similarity
4. **Golden Generation**: Create synthetic test cases from contexts
5. **Evolution**: Increase complexity and capture edge cases

### Basic Usage

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf', 'example.md', 'example.markdown', 'example.mdx'],
)
```

### Alternative: Pre-prepared Contexts

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_contexts(
    contexts=[
        ["The Earth revolves around the Sun.", "Planets are celestial bodies."],
        ["Water freezes at 0 degrees Celsius.", "The chemical formula for water is H2O."],
    ])
```

## Document Chunking

Documents divide into fixed-size chunks used for context generation. Control this process with:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Size of each chunk in tokens | 1024 |
| `chunk_overlap` | Overlapping tokens between consecutive chunks | 0 |
| `max_contexts_per_document` | Maximum contexts generated per document | 3 |

**Note**: Measurements use tokens, not characters.

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf', 'example.md', 'example.markdown', 'example.mdx'],
    chunk_size=1024,
    chunk_overlap=0)
```

### Best Practices for Chunking

1. **Impact on Retrieval**: Align chunk size and overlap with your retriever's expectations to prevent context mismatch during golden generation

2. **Balance Between Chunk Size and Overlap**: Use small overlap (50-100 tokens) for interconnected content; larger chunks with minimal overlap work for distinct sections

3. **Consider Document Structure**: Preserve natural breaks (chapters, sections, headings) to improve synthetic golden quality

**Caution**: Setting `chunk_size` too large or `chunk_overlap` too small for shorter documents may cause errors when the document cannot generate sufficient chunks for `max_contexts_per_document`.

### Chunk Calculation Formula

```
Number of Chunks = ⌈(Document Length - chunk_overlap) / (chunk_size - chunk_overlap)⌉
```

### Maximizing Coverage

Maximum goldens = `max_contexts_per_document` x `max_goldens_per_context`

**Tip**: Increase `max_contexts_per_document` to enhance coverage across document sections, especially beneficial for large datasets when computational resources are limited.

## Evolutions

The synthesizer increases complexity through evolution methods applied randomly to inputs. Control this with:

| Parameter | Description |
|-----------|-------------|
| `evolutions` | Dictionary specifying distribution of evolution methods |
| `num_evolutions` | Number of evolution steps applied to each generated input |

**Note**: Data evolution derives from [Evol-Instruct and WizardML](https://arxiv.org/abs/2304.12244).

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt', 'example.docx', 'example.pdf', 'example.md', 'example.markdown', 'example.mdx'],
    num_evolutions=3,
    evolutions={
        Evolution.REASONING: 0.1,
        Evolution.MULTICONTEXT: 0.1,
        Evolution.CONCRETIZING: 0.1,
        Evolution.CONSTRAINED: 0.1,
        Evolution.COMPARATIVE: 0.1,
        Evolution.HYPOTHETICAL: 0.1,
        Evolution.IN_BREADTH: 0.4,
    })
```

### Seven Evolution Types

- **Reasoning**: Requires multi-step logical thinking
- **Multicontext**: Utilizes all relevant context information
- **Concretizing**: Makes abstract ideas more concrete and detailed
- **Constrained**: Introduces conditions testing ability to operate within specific limits
- **Comparative**: Requires comparison between options or contexts
- **Hypothetical**: Forces consideration of hypothetical scenarios
- **In-breadth**: Touches on related or adjacent topics

**Tip**: In-breadth enables horizontal expansion; other evolutions provide vertical complexity. In-breadth focuses on breadth, while others increase difficulty.

### Best Practices for Using Evolutions

1. **Align Evolutions with Testing Goals**: Choose based on evaluation objectives—Reasoning and Comparative for logic tests; In-breadth for broader domain testing

2. **Balance Complexity and Coverage**: Mix vertical complexity (Reasoning, Constrained) with horizontal expansion (In-breadth) for comprehensive evaluation

3. **Start Small, Then Scale**: Begin with smaller `num_evolutions` then gradually increase to control challenge levels

4. **Target Edge Cases for Stress Testing**: Increase Constrained and Hypothetical evolutions for testing under restrictive or unusual conditions

5. **Monitor Evolution Distribution**: Check distributions regularly to avoid overloading test data with single types unless focused on specific evaluation areas

### Accessing Evolutions

```python
from deepeval.synthesizer import Synthesizer

# Generate goldens from documents
goldens = synthesizer.generate_goldens_from_docs(
  document_paths=['example.txt', 'example.docx', 'example.pdf', 'example.md', 'example.markdown', 'example.mdx'],
)

# Access evolutions through the DataFrame
goldens_dataframe = synthesizer.to_pandas()
goldens_dataframe.head()

# Access evolutions directly from a specific golden
goldens[0].additional_metadata["evolutions"]
```

## Qualifying Synthetic Goldens

Synthetic generation can introduce noise, requiring qualification and filtering at three stages:

### Context Filtering

During context generation, each randomly sampled chunk scores on:

- **Clarity**: Information understandability
- **Depth**: Detail and insight level
- **Structure**: Organization and logic quality
- **Relevance**: Content relation to main topic

**Note**: Scores range 0-1; minimum passing average is 0.5 with maximum 3 retries per chunk. Additional chunks use cosine similarity threshold of 0.5.

### Synthetic Input Filtering

Generated inputs evaluate on:

- **Self-containment**: Query understandability without external context
- **Clarity**: Clear intent specification without ambiguity

**Info**: Scores range 0-1 with minimum passing threshold; maximum 3 retries if criteria unmet.

### Accessing Quality Scores

```python
from deepeval.synthesizer import Synthesizer

# Generate goldens from documents
goldens = synthesizer.generate_goldens_from_docs(
  document_paths=['example.txt', 'example.docx', 'example.pdf', 'example.md', 'example.markdown', 'example.mdx'],
)

# Access quality scores through the DataFrame
goldens_dataframe = synthesizer.to_pandas()
goldens_dataframe.head()

# Access quality scores directly from a specific golden
goldens[0].additional_metadata["synthetic_input_quality"]
goldens[0].additional_metadata["context_quality"]
```

## Navigation

- **Previous**: RAG Triad
- **Next**: Using Custom LLMs for Evaluation

## Footer Links

- [GitHub](https://github.com/confident-ai/deepeval)
- [Discord Community](https://discord.gg/a3K9c8GRGt)
- [Blog & Newsletter](https://confident-ai.com/blog)
- [Documentation](https://www.confident-ai.com/docs)

---

**Last updated**: February 16, 2026 by Jeffrey Ip
