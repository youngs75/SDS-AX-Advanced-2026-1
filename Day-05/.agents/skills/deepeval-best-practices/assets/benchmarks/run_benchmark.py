"""
DeepEval Benchmarks Example
==============================
Demonstrates running standardized LLM benchmarks with DeepEvalBaseLLM,
including MMLU, GSM8K, BigBenchHard, and batch evaluation patterns.
"""

from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU, GSM8K, BigBenchHard
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.tasks import BigBenchHardTask
from typing import List


# =============================================================================
# 1. DeepEvalBaseLLM Subclass (Required for All Benchmarks)
# =============================================================================
class MyCustomLLM(DeepEvalBaseLLM):
    """
    Wrap your LLM in DeepEvalBaseLLM before running any benchmark.
    Replace the OpenAI calls with your actual model provider.

    Required methods: load_model(), generate(), a_generate(), get_model_name()
    Optional method: batch_generate() for improved throughput
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI()  # Replace with your model client

    def load_model(self):
        """Returns the underlying model object."""
        return self.client

    def generate(self, prompt: str) -> str:
        """Synchronous text generation. Replace with your model's API."""
        client = self.load_model()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """Async text generation. Can delegate to generate() if async is not available."""
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        Optional batch generation for improved throughput.
        Used when batch_size is specified in evaluate().
        """
        return [self.generate(prompt) for prompt in prompts]

    def get_model_name(self) -> str:
        """Returns a human-readable model name string."""
        return self.model_name


# =============================================================================
# 2. Run MMLU Benchmark (Knowledge - 57 Subjects)
# =============================================================================
def run_mmlu_benchmark():
    """
    MMLU (Massive Multitask Language Understanding) evaluates knowledge
    breadth across 57 subjects. The standard benchmark for knowledge assessment.
    """
    model = MyCustomLLM(model_name="gpt-4o-mini")

    # Select specific tasks (subjects) to evaluate
    benchmark = MMLU(
        tasks=[
            MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE,
            MMLUTask.ASTRONOMY,
            MMLUTask.MACHINE_LEARNING,
        ],
        n_shots=5,  # Number of in-context learning examples (max 5)
    )

    # Run evaluation with batch processing
    benchmark.evaluate(model=model, batch_size=5)

    # Access results
    print(f"Overall Score: {benchmark.overall_score}")
    print(f"\nTask Scores:\n{benchmark.task_scores}")
    print(f"\nSample Predictions:\n{benchmark.predictions.head()}")

    return benchmark


# =============================================================================
# 3. Run GSM8K Benchmark (Math with Chain of Thought)
# =============================================================================
def run_gsm8k_benchmark():
    """
    GSM8K (Grade School Math 8K) evaluates mathematical reasoning with
    1,319 word problems requiring 2-8 arithmetic steps.
    Supports Chain of Thought prompting.

    Note: GSM8K does NOT support batch_size in evaluate().
    """
    model = MyCustomLLM(model_name="gpt-4o-mini")

    benchmark = GSM8K(
        n_problems=50,  # Evaluate on a subset (max 1319)
        n_shots=3,  # Few-shot examples (max 3)
        enable_cot=True,  # Enable Chain of Thought reasoning
    )

    # Note: no batch_size parameter for GSM8K
    benchmark.evaluate(model=model)

    print(f"Overall Score: {benchmark.overall_score}")
    print(f"\nSample Predictions:\n{benchmark.predictions.head()}")

    return benchmark


# =============================================================================
# 4. Run BigBenchHard Benchmark (Complex Reasoning)
# =============================================================================
def run_bigbenchhhard_benchmark():
    """
    BIG-Bench Hard contains 23 challenging tasks where prior LLMs have
    not outperformed average human raters. Supports few-shot and CoT prompting.
    """
    model = MyCustomLLM(model_name="gpt-4o-mini")

    benchmark = BigBenchHard(
        tasks=[
            BigBenchHardTask.BOOLEAN_EXPRESSIONS,
            BigBenchHardTask.CAUSAL_JUDGEMENT,
            BigBenchHardTask.DATE_UNDERSTANDING,
        ],
        n_shots=3,  # Few-shot examples (max 3)
        enable_cot=True,  # Enable Chain of Thought
    )

    benchmark.evaluate(model=model, batch_size=4)

    print(f"Overall Score: {benchmark.overall_score}")
    print(f"\nTask Scores:\n{benchmark.task_scores}")

    return benchmark


# =============================================================================
# 5. Batch Evaluation Across Multiple Benchmarks
# =============================================================================
def run_multiple_benchmarks():
    """
    Run multiple benchmarks in sequence to get a comprehensive model assessment.
    Collects results from knowledge, math, and reasoning categories.
    """
    model = MyCustomLLM(model_name="gpt-4o-mini")

    # Define benchmarks to run
    benchmarks = {
        "MMLU (Knowledge)": MMLU(
            tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY],
            n_shots=5,
        ),
        "GSM8K (Math)": GSM8K(
            n_problems=50,
            n_shots=3,
            enable_cot=True,
        ),
        "BigBenchHard (Reasoning)": BigBenchHard(
            tasks=[
                BigBenchHardTask.BOOLEAN_EXPRESSIONS,
                BigBenchHardTask.CAUSAL_JUDGEMENT,
            ],
            n_shots=3,
            enable_cot=True,
        ),
    }

    results = {}
    for name, benchmark in benchmarks.items():
        print(f"\nRunning {name}...")

        # GSM8K does not support batch_size
        if isinstance(benchmark, GSM8K):
            benchmark.evaluate(model=model)
        else:
            benchmark.evaluate(model=model, batch_size=4)

        results[name] = benchmark.overall_score
        print(f"  Score: {benchmark.overall_score:.4f}")

    # Print summary
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    for name, score in results.items():
        print(f"  {name}: {score:.4f}")

    return results


# =============================================================================
# 6. Task Selection Within a Benchmark
# =============================================================================
def explore_task_selection():
    """
    Demonstrates task selection patterns for different benchmarks.
    Each benchmark has its own Task enum for specifying subsets.
    """
    from deepeval.benchmarks.tasks import (
        BigBenchHardTask,
        HellaSwagTask,
        MathQATask,
    )
    from deepeval.benchmarks import HellaSwag, MathQA

    model = MyCustomLLM(model_name="gpt-4o-mini")

    # MMLU: Select from 57 academic subjects
    mmlu = MMLU(
        tasks=[
            MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE,
            MMLUTask.COLLEGE_MATHEMATICS,
            MMLUTask.CLINICAL_KNOWLEDGE,
        ],
        n_shots=3,
    )

    # BigBenchHard: Select from 23 reasoning tasks
    bbh = BigBenchHard(
        tasks=[
            BigBenchHardTask.BOOLEAN_EXPRESSIONS,
            BigBenchHardTask.DATE_UNDERSTANDING,
            BigBenchHardTask.WORD_SORTING,
        ],
        n_shots=3,
        enable_cot=True,
    )

    # HellaSwag: Select from 160+ activity categories
    hellaswag = HellaSwag(
        tasks=[
            HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES,
            HellaSwagTask.BATON_TWIRLING,
        ],
        n_shots=5,
    )

    # MathQA: Select from math categories
    mathqa = MathQA(
        tasks=[
            MathQATask.PROBABILITY,
            MathQATask.GEOMETRY,
        ],
        n_shots=3,
    )

    # Run one as a demonstration
    print("Running MMLU with selected tasks...")
    mmlu.evaluate(model=model, batch_size=4)
    print(f"MMLU Score: {mmlu.overall_score}")
    print(f"Task Scores:\n{mmlu.task_scores}")


if __name__ == "__main__":
    print("=== MMLU Benchmark ===")
    run_mmlu_benchmark()

    print("\n=== GSM8K Benchmark (with CoT) ===")
    run_gsm8k_benchmark()

    print("\n=== BigBenchHard Benchmark ===")
    run_bigbenchhhard_benchmark()

    print("\n=== Multiple Benchmarks ===")
    run_multiple_benchmarks()
