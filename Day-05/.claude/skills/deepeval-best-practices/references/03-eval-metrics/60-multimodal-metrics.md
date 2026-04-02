# Multimodal Metrics

## Read This When
- Evaluating image-text coherence, helpfulness, or reference accuracy in multimodal LLM responses (ImageCoherenceMetric, ImageHelpfulnessMetric, ImageReferenceMetric)
- Measuring text-to-image generation quality (TextToImageMetric) or image editing task performance (ImageEditingMetric)
- Need to understand how MLLMImage objects work for embedding images in LLMTestCase fields

## Skip This When
- Evaluating text-only RAG or generation quality → `references/03-eval-metrics/20-rag-metrics.md`
- Need safety checks (bias, toxicity, PII) → `references/03-eval-metrics/40-safety-metrics.md`
- Need deterministic validation (JSON schema, regex, exact match) → `references/03-eval-metrics/70-utility-metrics.md`

---

DeepEval provides 5 metrics for evaluating multimodal LLM outputs — responses that contain or reference images. All multimodal metrics use MLLM-Eval (Multimodal LLM as a judge) and work with `LLMTestCase` using `MLLMImage` objects embedded in text fields.

## MLLMImage Usage

Images are embedded directly in `input` or `actual_output` strings using `MLLMImage`:

```python
from deepeval.test_case import LLMTestCase, MLLMImage

# Local file
MLLMImage(url="./path/to/image.png", local=True)

# Remote URL
MLLMImage(url="https://example.com/image.png", local=False)
```

Embed in text fields using f-strings:

```python
test_case = LLMTestCase(
    input=f"Tell me about this landmark: {MLLMImage(url='./eiffel.jpg', local=True)}",
    actual_output=f"This appears to be the Eiffel Tower, located in Paris, France."
)
```

---

## ImageCoherenceMetric

Evaluates the coherent alignment of images with their accompanying text in multimodal LLM responses. Assesses how well visual content complements the textual narrative.

**Classification:** MLLM-Eval (self-explaining), Single-turn

**Required LLMTestCase fields:**
- `input`
- `actual_output` (must contain image(s) embedded with `MLLMImage`)

**Formula:**
```
Per-image score: Ci = f(Context_above, Context_below, Image_i)

Overall score: O = sum(Ci) / n
```

Each image's score derives from surrounding text (above and below), constrained by `max_context_size`. Final score is the average across all images.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `max_context_size` | int | None | Maximum characters per context window |

Note: `include_reason` is supported but not listed in optional params — the metric is self-explaining and always provides reasoning.

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ImageCoherenceMetric
from deepeval.test_case import LLMTestCase, MLLMImage

metric = ImageCoherenceMetric(threshold=0.7, include_reason=True)

m_test_case = LLMTestCase(
    input="Provide step-by-step instructions on how to fold a paper airplane.",
    actual_output=f"""
        1. Take the sheet of paper and fold it lengthwise:
        {MLLMImage(url="./paper_plane_1.png", local=True)}
        2. Unfold the paper. Fold the top left and right corners towards the center.
        {MLLMImage(url="./paper_plane_2.png", local=True)}
    """
)

evaluate(test_cases=[m_test_case], metrics=[metric])

# Standalone
metric.measure(m_test_case)
print(metric.score, metric.reason)
```

---

## ImageHelpfulnessMetric

Evaluates how effectively images contribute to a user's comprehension of the accompanying text. Assesses whether images provide additional insights, clarify complex ideas, or support textual details.

**Classification:** MLLM-Eval (self-explaining), Single-turn

**Required LLMTestCase fields:**
- `input`
- `actual_output` (must contain image(s))

**Formula:**
```
Per-image score: Hi = f(Context_above, Context_below, Image_i)

Overall score: O = (sum of Hi) / n
```

When multiple images are provided, final score is the average of each image's helpfulness score.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `max_context_size` | int | None | Maximum characters in context |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ImageHelpfulnessMetric
from deepeval.test_case import LLMTestCase, MLLMImage

metric = ImageHelpfulnessMetric(threshold=0.7, include_reason=True)

m_test_case = LLMTestCase(
    input="Provide step-by-step instructions on how to fold a paper airplane.",
    actual_output=f"""
        1. Take the sheet of paper and fold it lengthwise:
        {MLLMImage(url="./paper_plane_1.png", local=True)}
        2. Unfold the paper. Fold the top corners towards the center.
        {MLLMImage(url="./paper_plane_2.png", local=True)}
    """
)

evaluate(test_cases=[m_test_case], metrics=[metric])
```

---

## ImageReferenceMetric

Evaluates how accurately images are referred to or explained by the accompanying text. Assesses the quality of textual references and descriptions for each image in the output.

**Classification:** MLLM-Eval (self-explaining), Single-turn

**Required LLMTestCase fields:**
- `input`
- `actual_output` (must contain image(s))

**Formula:**
```
Per-image score: Ri = f(Context_above, Context_below, Image_i)

Overall score: O = (sum of Ri) / n
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |
| `max_context_size` | int | None | Maximum characters per context window |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ImageReferenceMetric
from deepeval.test_case import LLMTestCase, MLLMImage

metric = ImageReferenceMetric(threshold=0.7, include_reason=True)

m_test_case = LLMTestCase(
    input="Provide step-by-step instructions on how to fold a paper airplane.",
    actual_output=f"""
        1. Take the sheet of paper and fold it lengthwise:
        {MLLMImage(url="./paper_plane_1.png", local=True)}
        2. Unfold the paper. Fold the top left and right corners towards the center.
        {MLLMImage(url="./paper_plane_2.png", local=True)}
    """
)

evaluate(test_cases=[m_test_case], metrics=[metric])

metric.measure(m_test_case)
print(metric.score, metric.reason)
```

---

## TextToImageMetric

Evaluates the quality of text-to-image generation by assessing semantic consistency (does the image match the prompt?) and perceptual quality (does the image look natural without artifacts?). Produces scores comparable to human evaluations when using GPT-4v.

**Classification:** MLLM-Eval (self-explaining), Single-turn

**Required LLMTestCase fields:**
- `input` (text prompt — should contain exactly 0 images)
- `actual_output` (generated image — should contain exactly 1 image)

**Formula:**
```
O = sqrt[min(alpha_1, ..., alpha_i) * min(beta_1, ..., beta_i)]
```

Where:
- **alpha values** = Semantic Consistency (SC) sub-scores — alignment with prompt and concept resemblance, using both input conditions and the synthesized image
- **beta values** = Perceptual Quality (PQ) sub-scores — naturalness and artifact absence, using only the synthesized image

Final score = geometric mean of minimum SC and PQ sub-scores.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import TextToImageMetric
from deepeval.test_case import LLMTestCase, MLLMImage

metric = TextToImageMetric(threshold=0.7, include_reason=True)

m_test_case = LLMTestCase(
    input="Generate an image of a blue pair of running shoes on a white background.",
    actual_output=f"{MLLMImage(url='https://shoe-images.com/generated-blue-shoes.png', local=False)}"
)

evaluate(test_cases=[m_test_case], metrics=[metric])

metric.measure(m_test_case)
print(metric.score, metric.reason)
```

---

## ImageEditingMetric

Evaluates image editing task performance by assessing semantic consistency (does the edit match the instruction?) and perceptual quality (does the edited image look natural?).

**Classification:** MLLM-Eval (self-explaining), Single-turn

**Required LLMTestCase fields:**
- `input` (editing instruction + source image — must contain exactly 1 image)
- `actual_output` (edited image — must contain exactly 1 image)

**Formula:**
```
O = sqrt[min(alpha_1, ..., alpha_i) * min(beta_1, ..., beta_i)]
```

Where:
- **alpha values (SC)** = Semantic Consistency — alignment with editing prompt and concept resemblance; uses both input conditions and synthesized image
- **beta values (PQ)** = Perceptual Quality — naturalness and artifact absence; uses only the synthesized image (to avoid input condition confusion)

Same formula as `TextToImageMetric`, but the `input` contains the source image to edit.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | float | 0.5 | Minimum passing threshold |
| `include_reason` | bool | True | Include reasoning |
| `strict_mode` | bool | False | Binary 0/1 scoring |
| `async_mode` | bool | True | Concurrent execution |
| `verbose_mode` | bool | False | Print intermediate steps |

**Code example:**

```python
from deepeval import evaluate
from deepeval.metrics import ImageEditingMetric
from deepeval.test_case import LLMTestCase, MLLMImage

metric = ImageEditingMetric(threshold=0.7, include_reason=True)

m_test_case = LLMTestCase(
    input=f"Change the color of the shoes to blue. {MLLMImage(url='./shoes.png', local=True)}",
    actual_output=f"{MLLMImage(url='https://shoe-images.com/edited-blue-shoes.png', local=False)}"
)

evaluate(test_cases=[m_test_case], metrics=[metric])

metric.measure(m_test_case)
print(metric.score, metric.reason)
```

---

## Multimodal Metrics Summary

| Metric | Input Contains | Output Contains | What It Measures |
|--------|---------------|-----------------|-----------------|
| `ImageCoherenceMetric` | Text | Text + Images | Image-text alignment/coherence |
| `ImageHelpfulnessMetric` | Text | Text + Images | Images' contribution to understanding |
| `ImageReferenceMetric` | Text | Text + Images | Accuracy of textual image references |
| `TextToImageMetric` | Text (no images) | 1 image | Text prompt to generated image quality |
| `ImageEditingMetric` | Text + 1 source image | 1 edited image | Editing instruction adherence + quality |

## Scoring Formula Comparison

| Metric | Formula Type | What Drives the Score |
|--------|-------------|----------------------|
| ImageCoherence | Average of per-image scores | Surrounding text context |
| ImageHelpfulness | Average of per-image scores | Surrounding text context |
| ImageReference | Average of per-image scores | Surrounding text context |
| TextToImage | Geometric mean of min(SC, PQ) | Semantic consistency + perceptual quality |
| ImageEditing | Geometric mean of min(SC, PQ) | Semantic consistency + perceptual quality |

## Shared Multimodal RAG Support

Beyond dedicated multimodal metrics, many standard RAG metrics also support `MLLMImage` in their test case fields:

```python
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, MLLMImage

# These RAG metrics work with multimodal inputs
test_case = LLMTestCase(
    input=f"Tell me about this landmark: {MLLMImage(url='./eiffel.jpg', local=True)}",
    actual_output="This appears to be the Eiffel Tower, located in Paris, France.",
    retrieval_context=[
        f"The Eiffel Tower {MLLMImage(url='./eiffel_ref.jpg', local=True)} is a wrought-iron lattice tower in Paris."
    ]
)
```

Multimodal-capable standard metrics: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ToolCorrectnessMetric`, `HallucinationMetric`, and others.
