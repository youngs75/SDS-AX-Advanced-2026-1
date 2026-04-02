# Synthetic Data Generation: Synthesizer Overview

## Read This When
- You need to create a `Synthesizer` and understand the four-step pipeline (input generation, filtration, evolution, styling)
- You want to configure `FiltrationConfig`, `EvolutionConfig`, or `StylingConfig` for synthetic data quality control
- You need to save, load, or inspect generated goldens (DataFrame, quality scores, Confident AI push)

## Skip This When
- You need detailed parameters for specific generation methods (`from_docs`, `from_contexts`, `from_scratch`, `from_goldens`) -- see [20-generation-methods.md](./20-generation-methods.md)
- You want to optimize prompts using goldens, not generate them -- see [../04-prompt-optimization/10-introduction.md](../04-prompt-optimization/10-introduction.md)

---

## Overview

DeepEval's `Synthesizer` generates high-quality single and multi-turn evaluation datasets efficiently. It proves valuable when you lack initial evaluation data, need dataset augmentation, or want to create datasets from a knowledge base.

**Key limitation:** Single-turn generation does not produce `actual_output`s. These must come from your LLM application, not the synthesizer.

The `Synthesizer` uses LLMs to generate input scenarios, then evolves them into more complex, realistic versions. These create synthetic goldens forming your `EvaluationDataset`.

---

## Quick Start

### Single-Turn

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt'],
    include_expected_output=True
)
print(goldens)
```

### Multi-Turn

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
conversational_goldens = synthesizer.generate_conversational_goldens_from_docs(
    document_paths=['example.txt'],
    include_expected_outcome=True
)
print(conversational_goldens)
```

---

## Creating a Synthesizer

```python
from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()
```

### Synthesizer Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `async_mode` | boolean | Enables concurrent golden generation | `True` |
| `model` | string or `DeepEvalBaseLLM` | OpenAI GPT model or custom model | `gpt-4.1` |
| `max_concurrent` | integer | Maximum parallel goldens at any time | `100` |
| `filtration_config` | `FiltrationConfig` | Customize filtration degree during generation | Default `FiltrationConfig` |
| `evolution_config` | `EvolutionConfig` | Customize evolution complexity | Default `EvolutionConfig` |
| `styling_config` | `StylingConfig` | Customize output styles and formats | Default `StylingConfig` |
| `cost_tracking` | boolean | Print LLM cost during synthesis | `False` |

**Note:** The `model` set on your `Synthesizer` automatically becomes the `critic_model` for `FiltrationConfig` and `ContextConstructionConfig` if custom instances are not provided.

---

## Generation Methods

Eight methods are available — four single-turn, four multi-turn:

**Single-Turn:**
- `generate_goldens_from_docs()` - From document files (includes context construction)
- `generate_goldens_from_contexts()` - From pre-prepared context lists
- `generate_goldens_from_scratch()` - Without any contexts
- `generate_goldens_from_goldens()` - Augment existing goldens

**Multi-Turn:**
- `generate_conversational_goldens_from_docs()`
- `generate_conversational_goldens_from_contexts()`
- `generate_conversational_goldens_from_scratch()`
- `generate_conversational_goldens_from_goldens()`

**Hierarchy:** `generate_goldens_from_docs()` > `generate_goldens_from_contexts()` > `generate_goldens_from_scratch()` in terms of scope and prerequisites.

See [20-generation-methods.md](./20-generation-methods.md) for complete documentation of each method.

---

## Four-Step Pipeline

The `Synthesizer` pipeline consists of four main steps:

### 1. Input Generation
Generate synthetic golden `input`s with or without provided contexts using an LLM.

### 2. Filtration
Filter initial synthetic goldens to meet generation standards. Inputs are scored (0-1) based on:
- **Self-containment**: Input is complete without external context
- **Clarity**: Intent is clearly conveyed without ambiguity

Goldens below `synthetic_input_quality_threshold` are regenerated. After `max_quality_retries`, the highest-scoring attempt is retained.

### 3. Evolution
Rewrite filtered `input`s to increase complexity and realism. Each input evolves `num_evolutions` times, with each evolution sampled from the `evolution` distribution.

Example evolution route with `num_evolutions=2`:
```
Initial input → Evolution 1 (sampled) → Evolution 2 (sampled) → Evolved output
```

### 4. Styling
Rewrite `input`s and `expected_output`s into desired formats/styles based on `scenario`, `task`, `input_format`, and `expected_output_format`.

**Additional steps:**
- **Context Construction** (before Input Generation, only for `generate_goldens_from_docs()`)
- **Expected Output Generation** (before Styling, when `include_expected_output=True`)

---

## Configuration Objects

### FiltrationConfig

Controls quality filtering strictness.

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import FiltrationConfig

filtration_config = FiltrationConfig(
    critic_model="gpt-4.1",
    synthetic_input_quality_threshold=0.5,
    max_quality_retries=3
)
synthesizer = Synthesizer(filtration_config=filtration_config)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `critic_model` | string or `DeepEvalBaseLLM` | Model for quality scoring | `gpt-4.1` or Synthesizer's model |
| `synthetic_input_quality_threshold` | float | Minimum quality threshold (0-1) | `0.5` |
| `max_quality_retries` | integer | Retries if quality is insufficient | `3` |

If quality remains below threshold after retries, the highest-scoring generation is used.

### EvolutionConfig

Controls which evolution types are applied and how many times.

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Evolution

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.REASONING: 1/4,
        Evolution.MULTICONTEXT: 1/4,
        Evolution.CONCRETIZING: 1/4,
        Evolution.CONSTRAINED: 1/4
    },
    num_evolutions=4
)
synthesizer = Synthesizer(evolution_config=evolution_config)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `evolutions` | dict | `Evolution` enum keys with probability values (must sum to 1.0) | All `Evolution`s equally weighted |
| `num_evolutions` | integer | Evolution steps per input | `1` |

### Available Evolution Types

```python
from deepeval.synthesizer import Evolution

available_evolutions = {
    Evolution.REASONING: 1/7,       # Requires multi-step logical thinking
    Evolution.MULTICONTEXT: 1/7,    # Utilizes all relevant context (sticks to context)
    Evolution.CONCRETIZING: 1/7,    # Makes abstract ideas concrete (sticks to context)
    Evolution.CONSTRAINED: 1/7,     # Introduces limiting conditions (sticks to context)
    Evolution.COMPARATIVE: 1/7,     # Requires comparison (sticks to context)
    Evolution.HYPOTHETICAL: 1/7,    # Forces hypothetical scenario consideration
    Evolution.IN_BREADTH: 1/7,      # Touches related/adjacent topics (horizontal expansion)
}
```

**Seven evolution types explained:**

| Type | Description | RAG-Safe? |
|------|-------------|-----------|
| `REASONING` | Requires multi-step logical thinking | No (may go beyond context) |
| `MULTICONTEXT` | Utilizes all relevant context information | Yes |
| `CONCRETIZING` | Makes abstract ideas more concrete and detailed | Yes |
| `CONSTRAINED` | Introduces conditions testing ability to operate within limits | Yes |
| `COMPARATIVE` | Requires comparison between options or contexts | Yes |
| `HYPOTHETICAL` | Forces consideration of hypothetical scenarios | No |
| `IN_BREADTH` | Touches on related or adjacent topics (horizontal expansion) | No |

**Note for RAG:** Only `MULTICONTEXT`, `CONCRETIZING`, `CONSTRAINED`, and `COMPARATIVE` strictly ensure answers are answerable from context.

**Best practices for evolutions:**
1. Align with testing goals: Reasoning/Comparative for logic tests; In-breadth for broader domain testing
2. Mix vertical complexity (Reasoning, Constrained) with horizontal expansion (In-breadth)
3. Start with smaller `num_evolutions`, then increase gradually
4. Increase Constrained and Hypothetical for stress/edge-case testing
5. Monitor evolution distribution to avoid overloading a single type

### StylingConfig

Customizes the format and style of generated inputs and outputs.

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
    input_format="Questions in English that asks for data in database.",
    expected_output_format="SQL query based on the given input",
    task="Answering text-to-SQL-related queries by querying a database and returning the results to users",
    scenario="Non-technical users trying to query a database using plain English."
)
synthesizer = Synthesizer(styling_config=styling_config)
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_format` | string | Desired format of generated inputs | `None` |
| `expected_output_format` | string | Desired format of generated outputs | `None` |
| `task` | string | Purpose of the LLM application | `None` |
| `scenario` | string | Setting/context of the LLM application | `None` |

These parameters enforce styles/formats on all generated goldens.

For multi-turn generation from scratch, use `ConversationalStylingConfig` instead:

```python
from deepeval.synthesizer.config import ConversationalStylingConfig

conversational_styling_config = ConversationalStylingConfig(
    conversational_task="Answering text-to-SQL-related queries...",
    scenario_context="Non-technical users trying to query a database using plain English.",
    participant_roles="Non-technical users trying to query a database using plain English."
)
synthesizer = Synthesizer(conversational_styling_config=conversational_styling_config)
```

---

## Accessing and Viewing Results

### Convert to DataFrame

```python
dataframe = synthesizer.to_pandas()
print(dataframe)
```

Example DataFrame columns:

| Column | Sample Value |
|--------|--------------|
| `input` | "Who wrote the novel '1984'?" |
| `actual_output` | `None` |
| `expected_output` | "George Orwell" |
| `context` | `["1984 is a dystopian novel published in 1949..."]` |
| `retrieval_context` | `None` |
| `n_chunks_per_context` | `1` |
| `context_length` | `60` |
| `context_quality` | `0.5` |
| `synthetic_input_quality` | `0.6` |
| `evolutions` | `None` |
| `source_file` | `file1.txt` |

### Accessing Quality Scores

```python
goldens = synthesizer.generate_goldens_from_docs(
    document_paths=['example.txt']
)

# Via DataFrame
df = synthesizer.to_pandas()
df.head()

# Directly from a specific golden
goldens[0].additional_metadata["synthetic_input_quality"]
goldens[0].additional_metadata["context_quality"]
goldens[0].additional_metadata["evolutions"]
```

---

## Saving and Loading Datasets

### Save to Confident AI

```python
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
dataset.push(alias="My Generated Dataset")
```

Retrieve and evaluate later:

```python
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset()
dataset.pull(alias="My Generated Dataset")
evaluate(dataset, metrics=[AnswerRelevancyMetric()])
```

### Save Locally

```python
synthesizer.save_as(
    file_type='json',        # 'json' or 'csv'
    directory="./synthetic_data",
    file_name="my_dataset"   # Optional; without extension
)
```

#### save_as() Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_type` | string | `'json'` or `'csv'` format |
| `directory` | string | Folder path for saved file |
| `file_name` | string (optional) | Custom filename without extension |
| `quiet` | boolean (optional) | Suppress output messages |

**Default behavior:** Generates timestamp-based filename (e.g., `"20240523_152045.json"`).

**Caution:** `file_name` should not contain periods or extensions — these are added automatically.

---

## Quality Filtering

Synthetic generation can introduce noise. Filtering occurs at three stages:

### Context Filtering (for `generate_goldens_from_docs` only)

During context generation, each randomly sampled chunk is scored on:
- **Clarity**: Information understandability
- **Depth**: Detail and insight level
- **Structure**: Organization and logic quality
- **Relevance**: Content relation to main topic

Scores range 0-1; minimum passing average is 0.5 with maximum 3 retries per chunk.

### Synthetic Input Filtering

Generated inputs are evaluated on:
- **Self-containment**: Query understandable without external context
- **Clarity**: Clear intent specification without ambiguity

Minimum threshold configurable via `FiltrationConfig.synthetic_input_quality_threshold`.

### Chunk Calculation Formula

```
Number of Chunks = ceil((Document Length - chunk_overlap) / (chunk_size - chunk_overlap))
```

Maximum goldens per document: `max_contexts_per_document` x `max_goldens_per_context`

---

## Important Notes

- **Synthetic data should be manually inspected and edited** where possible before use in production evaluation
- The `Synthesizer` uses the **data evolution method** originally from [Evol-Instruct and WizardML](https://arxiv.org/abs/2304.12244)
- Single-turn generation produces `Golden` objects; multi-turn generation produces `ConversationalGolden` objects

---

## Related Documentation

- [20-generation-methods.md](./20-generation-methods.md) - Detailed parameters and examples for all 4 generation methods
- [../04-prompt-optimization/10-introduction.md](../04-prompt-optimization/10-introduction.md) - Using goldens with PromptOptimizer
