# DeepEval Available Benchmarks

## Read This When
- You need to select a specific benchmark (MMLU, GSM8K, HumanEval, ARC, HellaSwag, BigBenchHard, etc.) and see its parameters, task enums, and code examples
- You want the quick reference table comparing all 16 benchmarks by category, dataset size, n_shots range, and scoring method
- You need the HumanEval special case (`generate_samples()` method, pass@k metric) or CoT-enabled benchmarks

## Skip This When
- You need to implement `DeepEvalBaseLLM` or understand `benchmark.evaluate()` basics -- see [10-overview.md](./10-overview.md)
- You want to evaluate your application's outputs with custom metrics, not run standardized benchmarks -- see [../03-eval-metrics/](../03-eval-metrics/)

---

## Quick Reference Table

| Benchmark | Class | Category | Tasks / Problems | n_shots | CoT | Scoring |
|-----------|-------|----------|-----------------|---------|-----|---------|
| ARC | `ARC` | Reasoning | ~8,000 MCQ | 0–5 (default 5) | No | Exact match |
| HellaSwag | `HellaSwag` | Reasoning | 10,000 sentence completions | 0–15 (default 10) | No | Exact match |
| WinoGrande | `Winogrande` | Reasoning | 1,267 binary-choice | 0–5 (default 5) | No | Exact match |
| LogiQA | `LogiQA` | Reasoning | 8,678 MCQ | 0–5 (default 5) | No | Exact match |
| GSM8K | `GSM8K` | Math | 1,319 word problems | 0–3 (default 3) | Yes (default True) | Exact match |
| MathQA | `MathQA` | Math | 37,000 MCQ | 0–5 (default 5) | No | Exact match |
| DROP | `DROP` | Math | 9,500+ paragraphs | 0–5 (default 5) | No | Exact match |
| MMLU | `MMLU` | Knowledge | 57 subjects, ~15K questions | 0–5 (default 5) | No | Exact match |
| TruthfulQA | `TruthfulQA` | Knowledge | 817 questions, 38 topics | N/A | No | Exact match (MC1) / truth score (MC2) |
| BoolQ | `BoolQ` | Knowledge | 3,270 yes/no questions | 0–5 (default 5) | No | Exact match |
| LAMBADA | `LAMBADA` | Language | 5,153 passages | 0–5 (default 5) | No | Exact match |
| SQuAD | `SQuAD` | Language | 100K QA pairs (10K val) | 0–5 (default 5) | No | LLM-as-judge |
| HumanEval | `HumanEval` | Code | 164 programming tasks | N/A | No | pass@k |
| IFEval | `IFEval` | Instruction Following | 500+ test cases | N/A | No | Exact match |
| BBQ | `BBQ` | Fairness | 58,000 trinary-choice | 0–5 (default 5) | No | Exact match |
| BIG-Bench Hard | `BigBenchHard` | Complex Reasoning | 23 tasks (~6,500 examples) | 0–3 (default 3) | Yes (default True) | Exact match |

---

## Grouped by Category

### Reasoning

#### ARC (AI2 Reasoning Challenge)

Science exam questions for grades 3–9. Two difficulty modes: EASY and CHALLENGE.

- **Class**: `ARC`
- **Import**: `from deepeval.benchmarks import ARC`
- **Dataset size**: ~8,000 multiple-choice questions
- **Paper**: https://arxiv.org/pdf/1803.05457v1

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `n_problems` | int | All | Optional |
| `n_shots` | int | 5 | Max 5 |
| `mode` | `ARCMode` | `ARCMode.EASY` | EASY or CHALLENGE |

```python
from deepeval.benchmarks import ARC
from deepeval.benchmarks.modes import ARCMode

benchmark = ARC(
    n_problems=100,
    n_shots=3,
    mode=ARCMode.CHALLENGE
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

---

#### HellaSwag

Commonsense reasoning via sentence completion across 160+ real-world activity categories.

- **Class**: `HellaSwag`
- **Import**: `from deepeval.benchmarks import HellaSwag`
- **Dataset size**: 10,000 multiple-choice completions
- **Paper**: https://github.com/rowanz/hellaswag

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[HellaSwagTask]` | All | Optional |
| `n_shots` | int | 10 | Max 15 |

```python
from deepeval.benchmarks import HellaSwag
from deepeval.benchmarks.tasks import HellaSwagTask

benchmark = HellaSwag(
    tasks=[HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING],
    n_shots=5
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Sample tasks: `APPLYING_SUNSCREEN`, `DISC_DOG`, `WAKEBOARDING`, `SKATEBOARDING`, `SAILING`, `PLAYING_CONGAS`, `BALLET`, and 150+ more covering sports, personal care, and household activities.

---

#### WinoGrande

Commonsense reasoning via binary-choice problems. Inspired by the original WinoGrad Schema Challenge, enhanced for scale and difficulty.

- **Class**: `Winogrande`
- **Import**: `from deepeval.benchmarks import Winogrande`
- **Dataset size**: 44,000 problems total; 1,267 in validation set
- **Paper**: https://arxiv.org/pdf/1907.10641

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `n_problems` | int | 1267 | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import Winogrande

benchmark = Winogrande(
    n_problems=100,
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

---

#### LogiQA

Logical reasoning derived from China's National Civil Servants Examination. Tests deductive reasoning including categorical, conditional, and disjunctive reasoning types.

- **Class**: `LogiQA`
- **Import**: `from deepeval.benchmarks import LogiQA`
- **Dataset size**: 8,678 multiple-choice questions with reading passages

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[LogiQATask]` | All | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import LogiQA
from deepeval.benchmarks.tasks import LogiQATask

benchmark = LogiQA(
    tasks=[
        LogiQATask.CATEGORICAL_REASONING,
        LogiQATask.SUFFICIENT_CONDITIONAL_REASONING
    ],
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Available tasks: `CATEGORICAL_REASONING`, `SUFFICIENT_CONDITIONAL_REASONING`, `NECESSARY_CONDITIONAL_REASONING`, `DISJUNCTIVE_REASONING`, `CONJUNCTIVE_REASONING`

---

### Math

#### GSM8K (Grade School Math 8K)

1,319 grade-school math word problems requiring 2–8 elementary arithmetic steps. Supports Chain of Thought prompting.

- **Class**: `GSM8K`
- **Import**: `from deepeval.benchmarks import GSM8K`
- **Dataset size**: 1,319 problems
- **Paper**: https://arxiv.org/abs/2110.14168
- **Note**: Does NOT support `batch_size` in `evaluate()`

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `n_problems` | int | 1319 | 1–1319 |
| `n_shots` | int | 3 | 0–3 |
| `enable_cot` | bool | True | Optional |

```python
from deepeval.benchmarks import GSM8K

benchmark = GSM8K(
    n_problems=100,
    n_shots=3,
    enable_cot=True
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

---

#### MathQA

37,000 multiple-choice math word problems from the AQuA dataset (GRE/GMAT level). Spans probability, geometry, physics, and more.

- **Class**: `MathQA`
- **Import**: `from deepeval.benchmarks import MathQA`
- **Dataset size**: 37,000 problems

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[MathQATask]` | All | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import MathQA
from deepeval.benchmarks.tasks import MathQATask

benchmark = MathQA(
    tasks=[MathQATask.PROBABILITY, MathQATask.GEOMETRY],
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Available tasks: `PROBABILITY`, `GEOMETRY`, `PHYSICS`, `GAIN`, `GENERAL`, `OTHER`

---

#### DROP (Discrete Reasoning Over Paragraphs)

Complex question-answering requiring numerical reasoning (addition, subtraction, counting) over paragraphs about NFL and history topics.

- **Class**: `DROP`
- **Import**: `from deepeval.benchmarks import DROP`
- **Dataset size**: 9,500+ challenges
- **Paper**: https://arxiv.org/pdf/1903.00161v2.pdf

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[DROPTask]` | All | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import DROP
from deepeval.benchmarks.tasks import DROPTask

benchmark = DROP(
    tasks=[DROPTask.HISTORY_1002, DROPTask.NFL_649],
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

760+ tasks available, categorized as NFL and HISTORY variants (e.g., `NFL_649`, `HISTORY_1002`, `HISTORY_1418`, `NFL_227`).

---

### Knowledge

#### MMLU (Massive Multitask Language Understanding)

The most comprehensive knowledge benchmark: 57 subjects spanning math, history, law, ethics, science, and more. The standard benchmark for assessing knowledge breadth.

- **Class**: `MMLU`
- **Import**: `from deepeval.benchmarks import MMLU`
- **Dataset size**: ~15,000 questions across 57 subjects
- **Paper**: https://github.com/hendrycks/test

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[MMLUTask]` | All 57 | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask

benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
    n_shots=3
)
benchmark.evaluate(model=mistral_7b, batch_size=5)
print(benchmark.overall_score)
print(benchmark.task_scores)
```

Subject categories include: high school (European History, US History, Physics, Chemistry, Biology, Computer Science, Mathematics, Geography, Psychology), college (Computer Science, Medicine, Chemistry, Mathematics, Physics, Biology), professional (Accounting, Medicine, Psychology, Law, Nursing), and specialized (Machine Learning, Clinical Knowledge, Virology, Philosophy, Economics, 40+ more).

---

#### TruthfulQA

817 questions across 38 topics targeting common misconceptions. Evaluates whether models answer truthfully rather than confidently reproducing human falsehoods.

- **Class**: `TruthfulQA`
- **Import**: `from deepeval.benchmarks import TruthfulQA`
- **Dataset size**: 817 questions
- **Paper**: https://github.com/sylinrl/TruthfulQA

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[TruthfulQATask]` | All 38 | Optional |
| `mode` | `TruthfulQAMode` | `MC1` | MC1 or MC2 |

**Modes:**
- `MC1`: Select one correct answer from 4–5 options (exact match scoring)
- `MC2`: Identify multiple correct answers from a set (truth identification scoring)

```python
from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode

benchmark = TruthfulQA(
    tasks=[TruthfulQATask.HEALTH, TruthfulQATask.LAW, TruthfulQATask.FINANCE],
    mode=TruthfulQAMode.MC2
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Available task categories: `LANGUAGE`, `NUTRITION`, `FICTION`, `SCIENCE`, `PROVERBS`, `MANDELA_EFFECT`, `ECONOMICS`, `PSYCHOLOGY`, `CONSPIRACIES`, `MISCONCEPTIONS`, `POLITICS`, `FINANCE`, `LAW`, `HISTORY`, `STATISTICS`, `MISINFORMATION`, `HEALTH`, `STEREOTYPES`, `RELIGION`, `ADVERTISING`, and 18 more.

---

#### BoolQ

Reading comprehension with naturally occurring yes/no questions paired with Wikipedia passages. Questions are generated in unprompted, real-world settings.

- **Class**: `BoolQ`
- **Import**: `from deepeval.benchmarks import BoolQ`
- **Dataset size**: 16,000 total; 3,270 validation
- **Paper**: https://arxiv.org/pdf/1905.10044

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `n_problems` | int | 3270 | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import BoolQ

benchmark = BoolQ(
    n_problems=100,
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

---

### Language

#### LAMBADA

Context-dependent word prediction. Designed so that humans cannot predict the final word without reading the full preceding passage. Tests broad discourse comprehension.

- **Class**: `LAMBADA`
- **Import**: `from deepeval.benchmarks import LAMBADA`
- **Dataset size**: 5,153 passages from BooksCorpus
- **Paper**: https://arxiv.org/abs/1606.06031

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `n_problems` | int | 5153 | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import LAMBADA

benchmark = LAMBADA(
    n_problems=200,
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

---

#### SQuAD (Stanford Question Answering Dataset)

Reading comprehension with 100K question-answer pairs drawn from 536 Wikipedia articles. Answers are text spans extracted directly from the passage.

- **Class**: `SQuAD`
- **Import**: `from deepeval.benchmarks import SQuAD`
- **Dataset size**: 100K total; 10K validation (23,215 paragraphs)
- **Paper**: https://arxiv.org/pdf/1606.05250
- **Special**: Uses LLM-as-a-judge for scoring (not exact match)

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[SQuADTask]` | All | Optional |
| `n_shots` | int | 5 | Max 5 |
| `evaluation_model` | str or `DeepEvalBaseLLM` | `gpt-4.1` | Optional |

```python
from deepeval.benchmarks import SQuAD
from deepeval.benchmarks.tasks import SQuADTask

benchmark = SQuAD(
    tasks=[SQuADTask.PHARMACY, SQuADTask.NORMANS],
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Available topics include: `PHARMACY`, `NORMANS`, `DOCTOR_WHO`, `OIL_CRISIS_1973`, `AMAZON_RAINFOREST`, `BLACK_DEATH`, `HARVARD_UNIVERSITY`, `NIKOLA_TESLA`, `PRIME_NUMBER`, `GENGHIS_KHAN`, `OXYGEN`, and 40+ more.

---

### Code

#### HumanEval

164 hand-crafted Python programming challenges. Evaluates functional correctness of generated code by running unit tests — not textual similarity to a reference solution.

- **Class**: `HumanEval`
- **Import**: `from deepeval.benchmarks import HumanEval`
- **Dataset size**: 164 tasks, averaging 7.7 unit tests per task
- **Paper**: https://github.com/openai/human-eval
- **Note**: Does NOT support `batch_size` in `evaluate()`; requires `generate_samples()` method

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[HumanEvalTask]` | All 164 | Optional |
| `n` | int | 200 | Samples per task (caution: high cost) |

**Special Requirement**: Your custom LLM must implement `generate_samples()`:

```python
from deepeval.models import DeepEvalBaseLLM
from langchain.schema import HumanMessage
from typing import Tuple

class GPT4Model(DeepEvalBaseLLM):
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[list[str], float]:
        chat_model = self.load_model()
        chat_model.n = n
        chat_model.temperature = temperature
        generations = chat_model._generate([HumanMessage(prompt)]).generations
        completions = [r.text for r in generations]
        return completions

gpt_4 = GPT4Model()
```

```python
from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.tasks import HumanEvalTask

benchmark = HumanEval(
    tasks=[HumanEvalTask.HAS_CLOSE_ELEMENTS, HumanEvalTask.SORT_NUMBERS],
    n=100
)
# k = number of top samples for pass@k metric
benchmark.evaluate(model=gpt_4, k=10)
print(benchmark.overall_score)
```

**pass@k formula**: `pass@k = 1 - C(n-c, k) / C(n, k)` where n = total samples, c = correct samples, k = top samples chosen.

Available tasks include: `HAS_CLOSE_ELEMENTS`, `SEPARATE_PAREN_GROUPS`, `TRUNCATE_NUMBER`, `MEAN_ABSOLUTE_DEVIATION`, `IS_PRIME`, `FIND_ZERO`, `FIZZ_BUZZ`, `DECODE_CYCLIC`, `PRIME_FIB`, `FIBONACCI`, `IS_PALINDROME`, `ENCRYPT`, `HISTOGRAM`, and 150+ more.

---

### Instruction Following

#### IFEval (Instruction-Following Evaluation)

Evaluates how well language models follow explicit instructions across dimensions including format compliance, constraint adherence, output structure, and specific instruction types. Based on Google's original research paper.

- **Class**: `IFEval`
- **Import**: `from deepeval.benchmarks import IFEval`
- **Paper**: https://arxiv.org/abs/2311.07911

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `n_problems` | int | None (all) | Optional |

```python
from deepeval.benchmarks import IFEval

benchmark = IFEval(n_problems=100)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

---

### Fairness

#### BBQ (Bias Benchmark of QA)

58,000 unique trinary-choice questions evaluating social biases across age, race, gender, nationality, religion, and more. Assesses both ambiguous-context bias and whether bias overrides correct answers with sufficient context.

- **Class**: `BBQ`
- **Import**: `from deepeval.benchmarks import BBQ`
- **Dataset size**: 58,000 questions
- **Paper**: https://arxiv.org/pdf/2110.08193

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[BBQTask]` | All | Optional |
| `n_shots` | int | 5 | Max 5 |

```python
from deepeval.benchmarks import BBQ
from deepeval.benchmarks.tasks import BBQTask

benchmark = BBQ(
    tasks=[BBQTask.AGE, BBQTask.GENDER_IDENTITY, BBQTask.RACE_ETHNICITY],
    n_shots=3
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Available tasks: `AGE`, `DISABILITY_STATUS`, `GENDER_IDENTITY`, `NATIONALITY`, `PHYSICAL_APPEARANCE`, `RACE_ETHNICITY`, `RACE_X_SES`, `RACE_X_GENDER`, `RELIGION`, `SES`, `SEXUAL_ORIENTATION`

---

### Complex Reasoning

#### BIG-Bench Hard (BBH)

23 challenging tasks from the BIG-Bench suite where prior LLMs have not outperformed average human raters. Supports both few-shot and Chain of Thought prompting.

- **Class**: `BigBenchHard`
- **Import**: `from deepeval.benchmarks import BigBenchHard`
- **Dataset size**: 23 tasks (~6,500 examples)
- **Paper**: https://github.com/suzgunmirac/BIG-Bench-Hard

| Parameter | Type | Default | Constraint |
|-----------|------|---------|-----------|
| `tasks` | `List[BigBenchHardTask]` | All 23 | Optional |
| `n_shots` | int | 3 | 0–3 |
| `enable_cot` | bool | True | Optional |

```python
from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks.tasks import BigBenchHardTask

benchmark = BigBenchHard(
    tasks=[
        BigBenchHardTask.BOOLEAN_EXPRESSIONS,
        BigBenchHardTask.CAUSAL_JUDGEMENT,
        BigBenchHardTask.DATE_UNDERSTANDING
    ],
    n_shots=3,
    enable_cot=True
)
benchmark.evaluate(model=mistral_7b)
print(benchmark.overall_score)
```

Available tasks include: `BOOLEAN_EXPRESSIONS`, `CAUSAL_JUDGEMENT`, `DATE_UNDERSTANDING`, `DISAMBIGUATION_QA`, `DYCK_LANGUAGES`, `FORMAL_FALLACIES`, `GEOMETRIC_SHAPES`, `HYPERBATON`, `LOGICAL_DEDUCTION_FIVE_OBJECTS`, `LOGICAL_DEDUCTION_SEVEN_OBJECTS`, `LOGICAL_DEDUCTION_THREE_OBJECTS`, `MOVIE_RECOMMENDATION`, `MULTISTEP_ARITHMETIC_TWO`, `NAVIGATE`, `OBJECT_COUNTING`, `PENGUINS_IN_A_TABLE`, `REASONING_ABOUT_COLORED_OBJECTS`, `RUIN_NAMES`, `SALIENT_TRANSLATION_ERROR_DETECTION`, `SNARKS`, `SPORTS_UNDERSTANDING`, `TEMPORAL_SEQUENCES`, `TRACKING_SHUFFLED_OBJECTS_FIVE_OBJECTS`, `TRACKING_SHUFFLED_OBJECTS_SEVEN_OBJECTS`, `TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS`, `WEB_OF_LIES`, `WORD_SORTING`

---

## Complete Example: Running Any Benchmark

This example demonstrates the full workflow — wrapping a model, selecting tasks, running evaluation, and inspecting all result types:

```python
from deepeval.benchmarks import BigBenchHard
from deepeval.benchmarks.tasks import BigBenchHardTask
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

class GPT4oMini(DeepEvalBaseLLM):
    def __init__(self):
        self.client = OpenAI()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "GPT-4o-mini"

# Instantiate model wrapper
model = GPT4oMini()

# Configure benchmark
benchmark = BigBenchHard(
    tasks=[
        BigBenchHardTask.BOOLEAN_EXPRESSIONS,
        BigBenchHardTask.CAUSAL_JUDGEMENT,
    ],
    n_shots=3,
    enable_cot=True
)

# Run evaluation
benchmark.evaluate(model=model, batch_size=4)

# Inspect results
print("Overall Score:", benchmark.overall_score)
print("\nTask Scores:")
print(benchmark.task_scores)
print("\nDetailed Predictions:")
print(benchmark.predictions)
```
