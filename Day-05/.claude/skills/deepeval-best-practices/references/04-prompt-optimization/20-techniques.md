# Prompt Optimization: Algorithms and Techniques

## Read This When
- You need to choose between GEPA, MIPROv2, and COPRO algorithms and understand their trade-offs
- You want to tune algorithm-specific parameters (iterations, pareto_size, num_candidates, population_size)
- You need few-shot demonstration optimization (MIPROv2) or bounded-population search (COPRO)

## Skip This When
- You need basic `PromptOptimizer` setup, `Prompt` class, or configuration objects -- see [10-introduction.md](./10-introduction.md)
- You want to generate evaluation datasets rather than optimize prompts -- see [../05-synthetic-data/10-synthesizer-overview.md](../05-synthetic-data/10-synthesizer-overview.md)

---

Three optimization algorithms are available in DeepEval. All are adapted from DSPy research implementations.

- **GEPA** (default): Genetic-Pareto multi-objective evolutionary search
- **MIPROv2**: Bayesian optimization with bootstrapped few-shot demonstrations
- **COPRO**: Coordinate-ascent with cooperative child proposals

See [10-introduction.md](./10-introduction.md) for `PromptOptimizer` setup and configuration objects.

---

## GEPA (Genetic-Pareto Prompt Optimization)

### Overview

GEPA is the **default algorithm**. It maintains a Pareto frontier of prompt candidates rather than converging on a single optimum. The core philosophy: "different prompts may excel at different types of problems."

**Key concept - Pareto optimality:** A prompt achieves Pareto optimality when there is no way to improve its score on one golden without making it worse on another. This prevents settling at local maxima.

**Source paper:** [GEPA: Genetic Pareto Optimization of LLM Prompts](https://arxiv.org/pdf/2507.19457)

### Basic Usage

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.prompt import Prompt
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import GEPA

prompt = Prompt(text_template="You are a helpful assistant - now answer this. {input}")

async def model_callback(prompt: Prompt, golden) -> str:
    prompt_to_llm = prompt.interpolate(input=golden.input)
    return await your_llm(prompt_to_llm)

optimizer = PromptOptimizer(
    algorithm=GEPA(),
    metrics=[AnswerRelevancyMetric()],
    model_callback=model_callback
)

optimized_prompt = optimizer.optimize(
    prompt=prompt,
    goldens=goldens
)
```

**Note:** GEPA is the default. If you do not pass an `algorithm` argument, `GEPA()` is used automatically.

### GEPA Configuration Parameters

```python
from deepeval.optimizer.algorithms import GEPA

gepa = GEPA(
    iterations=10,
    pareto_size=5,
    minibatch_size=4,
    random_seed=42,
    tie_breaker="PREFER_CHILD"
)

optimizer = PromptOptimizer(algorithm=gepa, ...)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iterations` | Optional | `5` | Total number of mutation attempts |
| `pareto_size` | Optional | `3` | Number of goldens in Pareto validation set (`D_pareto`) |
| `minibatch_size` | Optional | `8` | Number of goldens drawn per iteration (auto-clamped to available data) |
| `random_seed` | Optional | `time.time_ns()` | Controls randomness; set a fixed value (e.g., `42`) for reproducibility |
| `tie_breaker` | Optional | `PREFER_CHILD` | Policy for breaking ties: `PREFER_ROOT`, `PREFER_CHILD`, or `RANDOM` |

### Five-Step Algorithm Process

#### Step 1: Golden Splitting

GEPA divides test cases into two disjoint subsets:
- **`D_pareto`**: Fixed validation set of `pareto_size` goldens used to score all candidates fairly
- **`D_feedback`**: Remaining goldens for minibatch sampling during mutation

This train/validation split prevents overfitting. Prompts are mutated based on feedback goldens but selected based on held-out validation performance.

#### Step 2: Pareto Selection

Selection occurs in two phases:

**Finding non-dominated prompts:** A prompt "dominates" another when it scores better or equal on all goldens with at least one strictly higher score. The Pareto frontier consists of all non-dominated prompts.

Dominance example:

| Prompt | Golden 1 | Golden 2 | Golden 3 | Mean | On Frontier? |
|--------|----------|----------|----------|------|--------------|
| P0 | 0.60 | 0.55 | 0.50 | 0.55 | No (dominated by P1) |
| P1 | 0.75 | 0.70 | 0.65 | 0.70 | Yes |

No-dominance example (both on frontier despite different strengths):

| Prompt | Golden 1 | Golden 2 | Golden 3 | Mean | On Frontier? |
|--------|----------|----------|----------|------|--------------|
| P0 | 0.9 | 0.6 | 0.7 | 0.73 | Yes |
| P1 | 0.7 | 0.8 | 0.7 | 0.73 | Yes |

**Sampling from the frontier:** Candidates are selected with probability proportional to their "wins" (highest scores) across `D_pareto` goldens. This balances exploration (all non-dominated prompts eligible) with exploitation (frequent winners more likely selected).

Pareto table after 4 iterations:

| Prompt | Golden 1 | Golden 2 | Golden 3 | Mean | Wins | On Frontier? |
|--------|----------|----------|----------|------|------|--------------|
| P0 (root) | 0.60 | 0.55 | 0.50 | 0.55 | 0 | No |
| P1 | 0.75 | 0.70 | 0.60 | 0.68 | 0 | No |
| P2 | 0.65 | **0.85** | 0.55 | 0.68 | 1 | Yes |
| P3 | 0.60 | 0.60 | **0.80** | 0.67 | 1 | Yes |
| P4 | **0.80** | 0.75 | 0.70 | 0.75 | 1 | Yes |

Despite P4's highest mean, GEPA might still select P2 or P3 to explore their specialized strategies.

#### Step 3: Feedback and Mutation

1. Sample `minibatch_size` goldens from `D_feedback`
2. Execute `model_callback` with parent prompt on each minibatch item
3. Score responses using provided evaluation metrics
4. Extract `reason` field from metric evaluations (explanations of issues/successes)
5. Use an LLM to rewrite the prompt, addressing identified feedback

Mutation is targeted and metric-driven, based on actual failure cases rather than random variation.

#### Step 4: Acceptance

Child prompts are evaluated on the same minibatch as parents. Acceptance criteria:
- Child score exceeds parent score by minimum threshold (`GEPA_MIN_DELTA`)
- Accepted children are added to the candidate pool and scored on all `D_pareto` goldens
- Rejected children are discarded; next iteration begins

#### Step 5: Final Selection

1. Aggregate scores across all `D_pareto` goldens (mean by default)
2. Rank candidates by aggregate scores
3. Break ties using `tie_breaker` policy (`PREFER_CHILD` by default, favoring recently evolved prompts)

### When to Use GEPA

- Tasks with diverse problem types where different prompts excel at different sub-problems
- When maintaining diversity across problem types is important
- Tasks that do not benefit from few-shot examples
- Default choice when uncertain which algorithm to use

---

## MIPROv2

### Overview

MIPROv2 (Multiprompt Instruction PRoposal Optimizer Version 2) jointly optimizes both the **instruction** and the **few-shot demonstrations**. The core insight is that finding the best combination requires systematic Bayesian search rather than manual tuning.

**Source paper:** [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/pdf/2406.11695)

### Installation Requirement

MIPROv2 requires `optuna`:

```bash
pip install optuna
```

### Basic Usage

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.prompt import Prompt
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import MIPROV2

prompt = Prompt(text_template="You are a helpful assistant - now answer this. {input}")

async def model_callback(prompt: Prompt, golden) -> str:
    prompt_to_llm = prompt.interpolate(input=golden.input)
    return await your_llm(prompt_to_llm)

optimizer = PromptOptimizer(
    algorithm=MIPROV2(),
    metrics=[AnswerRelevancyMetric()],
    model_callback=model_callback
)

optimized_prompt = optimizer.optimize(
    prompt=prompt,
    goldens=goldens
)
```

### MIPROv2 Configuration Parameters

```python
from deepeval.optimizer.algorithms import MIPROV2

miprov2 = MIPROV2(
    num_candidates=10,
    num_trials=20,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_demo_sets=5,
    random_seed=42
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_candidates` | int | `10` | Number of diverse instruction candidates to generate in the proposal phase |
| `num_trials` | int | `20` | Number of Bayesian Optimization trials to run |
| `minibatch_size` | int | `25` | Number of goldens sampled per trial for evaluation |
| `minibatch_full_eval_steps` | int | `10` | Run full evaluation on all goldens every N trials |
| `max_bootstrapped_demos` | int | `4` | Maximum bootstrapped demonstrations per demo set |
| `max_labeled_demos` | int | `4` | Maximum labeled demonstrations per demo set |
| `num_demo_sets` | int | `5` | Number of different demo set configurations |
| `random_seed` | int | `time.time_ns()` | Seed for reproducibility |

### How MIPROv2 Works

MIPROv2 operates in two phases:

#### Phase 1: Proposal (runs once at startup)

**1a. Instruction Proposal**

Generates `num_candidates` diverse instruction variations using diverse "tips":

| Tip | Effect |
|-----|--------|
| "Be concise and direct" | Shorter, focused instructions |
| "Use step-by-step reasoning" | Emphasizes chain-of-thought |
| "Focus on clarity and precision" | Explicit, unambiguous instructions |
| "Consider edge cases and exceptions" | Robust, defensive instructions |

The original prompt is always included as candidate #0 (baseline).

**1b. Demo Bootstrapping**

Creates `num_demo_sets` different few-shot demonstration sets containing:
- **Bootstrapped demos**: Generated outputs that pass validation (up to `max_bootstrapped_demos`)
- **Labeled demos**: Taken from `expected_output` in goldens (up to `max_labeled_demos`)

A 0-shot option (empty demo set) is always included for testing.

#### Phase 2: Bayesian Optimization

Uses TPE (Tree-structured Parzen Estimator) sampler to efficiently search for the best (instruction, demo_set) combination:

1. Build a surrogate model of the objective function
2. Use the surrogate to predict promising combinations
3. Evaluate the most promising combination
4. Update the surrogate and repeat

Each trial:
1. Samples an instruction index and demo set index
2. Renders the prompt with selected demos
3. Evaluates on a minibatch of goldens
4. Reports the score back to Optuna

Every `minibatch_full_eval_steps` trials, the current best combination is evaluated on the full dataset.

#### Final Selection

After all trials:
1. Identify the (instruction, demo_set) combination with highest score
2. Run full evaluation if not cached
3. Return the optimized prompt with demos rendered inline

### When to Use MIPROv2

| Scenario | Why MIPROv2 Helps |
|----------|-------------------|
| Few-shot examples matter | Jointly optimizes instructions AND demos |
| Large search space | Bayesian optimization efficiently navigates combinations |
| Expensive evaluations | Minibatch sampling reduces costs while maintaining signal |
| Need reproducibility | Fixed random seed gives identical results |

---

## COPRO (Cooperative Prompt Optimization)

### Overview

COPRO is a bounded-population, zero-shot algorithm adapted from the MIPROv2 family. It explores candidate prompts while maintaining a fixed maximum population size. Key distinction: it proposes multiple children per iteration from shared feedback and keeps the population bounded through pruning.

### Basic Usage

```python
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.copro.configs import COPROConfig
from deepeval.optimizer.copro.loop import COPRORunner

optimizer = PromptOptimizer(
    metrics=[AnswerRelevancyMetric()],
    model_callback=model_callback
)
optimizer.set_runner(COPRORunner(config=COPROConfig()))

optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)
```

### COPROConfig Parameters

`COPROConfig` extends `MIPROConfig` with two additional fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | `4` | Maximum number of prompt candidates in active population. Exceeding triggers pruning of lower-scoring candidates while preserving current best |
| `proposals_per_step` | int | `4` | Number of child prompts proposed cooperatively from same parent per iteration. Higher values increase diversity at higher cost |

**Inherited from MIPROConfig** (behave identically to MIPROv2):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | — | Total optimization iterations |
| `minibatch_size` | `8` | Goldens per minibatch; auto-clamped to available data |
| `exploration_probability` | — | Epsilon in epsilon-greedy parent selection |
| `full_eval_every` | — | Full evaluation frequency (every N trial steps) |

### How COPRO Works

**Initialization:** Seeds root candidate (original prompt) and scores it on a minibatch.

**Each iteration:**
1. Select parent candidate using epsilon-greedy rule on mean minibatch score
   - With probability `exploration_probability`: pick random candidate
   - Otherwise: pick candidate with highest mean minibatch score
2. Draw a fresh minibatch from the full golden set
3. Compute shared `feedback_text` for parent and minibatch via metrics
4. Propose `proposals_per_step` child prompts cooperatively from the same parent using shared feedback (diversity comes from stochastic LLM sampling)
5. Score each child on the minibatch; accept any that improve on the parent
6. If population exceeds `population_size`, prune worst-scoring candidates while preserving the best
7. Optionally, if `full_eval_every` divides current trial index, run full evaluation of current best

**No-change handling:** If the rewriter returns a prompt equivalent to the parent, or if the type changes from TEXT to LIST (or reverse), the proposal is treated as no-change and ignored.

**Full evaluation strategy:**
- Minibatch scores drive local decisions
- Full evaluations at checkpoints provide reliable selection
- Final prompt chosen by aggregating full evaluation score vectors (default: `mean_of_all`)
- Fallback: if no full evaluation scores are available, selects best candidate by mean minibatch score

### What COPRO Returns

```python
optimized_prompt.text_template          # Optimized prompt string
optimized_prompt.optimization_report    # OptimizationReport with run progression
```

Report fields specific to COPRO:
- `pareto_scores`: full evaluation scores for each fully evaluated candidate
- `accepted_iterations`: when children were accepted
- `parents`: parent relationships
- `prompt_configurations`: underlying configurations

---

## Algorithm Comparison

| Aspect | GEPA | MIPROv2 | COPRO |
|--------|------|---------|-------|
| Search strategy | Pareto-based evolutionary | Bayesian Optimization (TPE) | Epsilon-greedy coordinate ascent |
| Candidate generation | Iterative mutations | All upfront (proposal phase) | Iterative with cooperative proposals |
| Few-shot demos | Not included | Jointly optimized | Not included |
| Diversity mechanism | Pareto frontier sampling | Diverse tips + multiple demo sets | Epsilon-greedy + bounded population |
| Population management | Unbounded candidate pool | N/A (trial-based) | Bounded by `population_size` |
| Extra dependency | None | `optuna` | None |
| Best for | Diverse problem types; no demos needed | Tasks where examples help; large search space | Bounded exploration with cooperative children |

**Choose GEPA** when:
- Maintaining diversity across problem types
- Task does not benefit from few-shot examples
- Default/general use case

**Choose MIPROv2** when:
- Few-shot demonstrations are important
- You have a large candidate space to search
- You need Bayesian efficiency with expensive evaluations

**Choose COPRO** when:
- You want bounded population exploration
- You want cooperative (multiple) proposals per step
- You have resource constraints and want pruning behavior

---

## Related Documentation

- [10-introduction.md](./10-introduction.md) - PromptOptimizer setup, Prompt class, AsyncConfig, DisplayConfig, MutationConfig
- [../05-synthetic-data/10-synthesizer-overview.md](../05-synthetic-data/10-synthesizer-overview.md) - Generating goldens for optimization
