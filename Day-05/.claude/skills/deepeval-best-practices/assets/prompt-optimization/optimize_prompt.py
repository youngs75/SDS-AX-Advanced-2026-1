"""
DeepEval Prompt Optimization Example
=======================================
Demonstrates automatic prompt improvement using DeepEval's PromptOptimizer
with GEPA (default) and MIPROv2 algorithms.
"""

import asyncio

from deepeval.dataset import Golden
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.prompt import Prompt
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import GEPA, MIPROV2
from deepeval.optimizer.configs import AsyncConfig, DisplayConfig


# =============================================================================
# 1. Define the Prompt and Model Callback
# =============================================================================

# The Prompt class holds the template text to be optimized.
# Use {variable} placeholders that get filled via interpolate().
prompt = Prompt(
    text_template="You are a knowledgeable assistant. Answer the following question "
    "accurately and concisely. Question: {input}"
)

# Sample goldens for optimization — replace with your actual evaluation data
goldens = [
    Golden(
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
    ),
    Golden(
        input="Who wrote Romeo and Juliet?",
        expected_output="William Shakespeare wrote Romeo and Juliet.",
    ),
    Golden(
        input="What is the boiling point of water?",
        expected_output="The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
    ),
    Golden(
        input="What is photosynthesis?",
        expected_output="Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    ),
    Golden(
        input="How many planets are in the solar system?",
        expected_output="There are 8 planets in the solar system.",
    ),
    Golden(
        input="What year did World War II end?",
        expected_output="World War II ended in 1945.",
    ),
]


async def model_callback(prompt: Prompt, golden) -> str:
    """
    Model callback wrapping your LLM application.
    Receives the current candidate prompt and a golden, returns the LLM response.

    Replace the body with your actual LLM call.
    """
    # Inject golden input into prompt template
    interpolated_prompt = prompt.interpolate(input=golden.input)

    # Replace with your actual LLM application call
    # Example: response = await your_llm_client.generate(interpolated_prompt)
    response = f"Based on my knowledge: {golden.expected_output}"

    return response


# =============================================================================
# 2. Optimize with GEPA (Default Algorithm)
# =============================================================================
def optimize_with_gepa():
    """
    GEPA (Genetic-Pareto) is the default optimization algorithm.
    It maintains a Pareto frontier of prompt candidates, ensuring diversity
    across different problem types.
    """
    # Configure GEPA parameters
    gepa = GEPA(
        iterations=10,
        pareto_size=5,
        minibatch_size=4,
        random_seed=42,
        tie_breaker="PREFER_CHILD",
    )

    # Configure async behavior for rate limit management
    async_config = AsyncConfig(
        run_async=True,
        throttle_value=0,
        max_concurrent=20,
    )

    # Configure display options
    display_config = DisplayConfig(
        show_indicator=True,
        announce_ties=False,
    )

    # Create optimizer with at least 2 objective metrics
    optimizer = PromptOptimizer(
        algorithm=gepa,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
            FaithfulnessMetric(threshold=0.7),
        ],
        model_callback=model_callback,
        async_config=async_config,
        display_config=display_config,
    )

    # Run optimization
    optimized_prompt = optimizer.optimize(
        prompt=prompt,
        goldens=goldens,
    )

    print(f"Original prompt: {prompt.text_template}")
    print(f"Optimized prompt: {optimized_prompt.text_template}")

    # Access the optimization report for detailed analysis
    report = optimizer.optimization_report
    print(f"\nOptimization ID: {report.optimization_id}")
    print(f"Best config ID: {report.best_id}")
    print(f"Accepted iterations: {len(report.accepted_iterations)}")

    return optimized_prompt


# =============================================================================
# 3. Optimize with MIPROv2 (Few-Shot + Bayesian Search)
# =============================================================================
def optimize_with_miprov2():
    """
    MIPROv2 jointly optimizes instructions AND few-shot demonstrations
    using Bayesian optimization (TPE sampler).

    Prerequisites:
        pip install optuna
    """
    # Configure MIPROv2 parameters
    miprov2 = MIPROV2(
        num_candidates=10,
        num_trials=20,
        minibatch_size=4,
        minibatch_full_eval_steps=10,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
        num_demo_sets=5,
        random_seed=42,
    )

    optimizer = PromptOptimizer(
        algorithm=miprov2,
        metrics=[
            AnswerRelevancyMetric(threshold=0.7),
        ],
        model_callback=model_callback,
    )

    # Run optimization
    optimized_prompt = optimizer.optimize(
        prompt=prompt,
        goldens=goldens,
    )

    print(f"MIPROv2 optimized prompt: {optimized_prompt.text_template}")
    return optimized_prompt


# =============================================================================
# 4. Async Optimization
# =============================================================================
async def optimize_async():
    """
    Run optimization asynchronously for non-blocking execution.
    Useful in web servers or when running multiple optimizations.
    """
    optimizer = PromptOptimizer(
        metrics=[AnswerRelevancyMetric(threshold=0.7)],
        model_callback=model_callback,
        async_config=AsyncConfig(run_async=True, max_concurrent=10),
    )

    optimized_prompt = await optimizer.a_optimize(
        prompt=prompt,
        goldens=goldens,
    )

    print(f"Async optimized prompt: {optimized_prompt.text_template}")
    return optimized_prompt


# =============================================================================
# 5. Hyperparameter Tuning with evaluate()
# =============================================================================
def hyperparameter_tuning():
    """
    Systematic hyperparameter tuning using nested evaluate() loops.
    Complementary to PromptOptimizer — useful for comparing discrete configs.
    """
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase

    # Define hyperparameter combinations to test
    models = ["gpt-4o-mini", "gpt-4o"]
    prompt_templates = [
        "Answer the question: {input}",
        "You are an expert. Provide a detailed answer to: {input}",
        "Think step by step and answer: {input}",
    ]

    test_inputs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ]

    metric = AnswerRelevancyMetric(threshold=0.7)

    for model in models:
        for template in prompt_templates:
            # Build test cases for this configuration
            test_cases = []
            for input_text, expected in test_inputs:
                # Replace with actual LLM call using the model and template
                actual_output = f"Response for: {input_text}"
                test_cases.append(
                    LLMTestCase(
                        input=input_text,
                        actual_output=actual_output,
                        expected_output=expected,
                    )
                )

            # Evaluate with hyperparameter tracking
            evaluate(
                test_cases=test_cases,
                metrics=[metric],
                hyperparameter={
                    "model": model,
                    "prompt template": template,
                },
            )


if __name__ == "__main__":
    print("=== GEPA Prompt Optimization ===")
    optimize_with_gepa()

    print("\n=== MIPROv2 Prompt Optimization ===")
    optimize_with_miprov2()

    print("\n=== Hyperparameter Tuning ===")
    hyperparameter_tuning()
