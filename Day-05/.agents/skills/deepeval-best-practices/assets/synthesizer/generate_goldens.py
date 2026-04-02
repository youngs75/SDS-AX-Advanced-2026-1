"""
DeepEval Synthetic Data Generation Example
=============================================
Demonstrates generating evaluation datasets using DeepEval's Synthesizer,
including document-based, context-based, and scratch-based golden generation.
"""

from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
    ContextConstructionConfig,
)
from deepeval.dataset import EvaluationDataset, Golden
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric


# =============================================================================
# 1. Basic Synthesizer Setup
# =============================================================================
def create_synthesizer():
    """Create a Synthesizer with custom filtration and evolution configs."""

    # Control quality filtering strictness
    filtration_config = FiltrationConfig(
        critic_model="gpt-4.1",
        synthetic_input_quality_threshold=0.5,
        max_quality_retries=3,
    )

    # Control evolution complexity and distribution
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.REASONING: 1 / 4,
            Evolution.MULTICONTEXT: 1 / 4,
            Evolution.CONCRETIZING: 1 / 4,
            Evolution.CONSTRAINED: 1 / 4,
        },
        num_evolutions=2,
    )

    synthesizer = Synthesizer(
        model="gpt-4.1",
        filtration_config=filtration_config,
        evolution_config=evolution_config,
        async_mode=True,
        max_concurrent=100,
    )

    return synthesizer


# =============================================================================
# 2. Generate Goldens from Documents
# =============================================================================
def generate_from_docs():
    """
    Generate goldens directly from document files.
    Replace document_paths with your actual file paths.

    Prerequisites:
        pip install chromadb langchain-core langchain-community langchain-text-splitters
    """
    synthesizer = create_synthesizer()

    # Configure how documents are parsed, chunked, and grouped into contexts
    context_config = ContextConstructionConfig(
        chunk_size=1024,
        chunk_overlap=50,
        max_contexts_per_document=3,
        context_quality_threshold=0.5,
        context_similarity_threshold=0.5,
        embedder="text-embedding-3-small",
    )

    # Replace with your actual document paths
    # Supports: .txt, .docx, .pdf, .md, .markdown, .mdx
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=["knowledge_base.txt"],  # Replace with your files
        include_expected_output=True,
        max_goldens_per_context=2,
        context_construction_config=context_config,
    )

    print(f"Generated {len(goldens)} goldens from documents")

    # Inspect quality scores via DataFrame
    df = synthesizer.to_pandas()
    print(df[["input", "context_quality", "synthetic_input_quality"]].head())

    return synthesizer, goldens


# =============================================================================
# 3. Generate Goldens from Contexts
# =============================================================================
def generate_from_contexts():
    """
    Generate goldens from pre-prepared context lists.
    Use this when you already have parsed contexts from a vector database.
    No ChromaDB or langchain dependencies required.
    """
    synthesizer = create_synthesizer()

    # Replace with your actual context lists
    # Each context is a list of strings sharing common themes
    contexts = [
        [
            "All customers are eligible for a 30 day full refund at no extra cost.",
            "Refunds are processed within 5-7 business days after approval.",
        ],
        [
            "Premium members receive free shipping on all orders.",
            "Standard shipping takes 3-5 business days.",
        ],
        [
            "The product warranty covers manufacturing defects for 1 year.",
            "Extended warranty can be purchased for an additional 2 years.",
        ],
    ]

    goldens = synthesizer.generate_goldens_from_contexts(
        contexts=contexts,
        include_expected_output=True,
        max_goldens_per_context=2,
        source_files=["refund_policy.md", "shipping_info.md", "warranty.md"],
    )

    print(f"Generated {len(goldens)} goldens from contexts")
    return synthesizer, goldens


# =============================================================================
# 4. Generate Goldens from Scratch (No Context Needed)
# =============================================================================
def generate_from_scratch():
    """
    Generate goldens without any documents or contexts.
    Useful for non-RAG applications or testing general capabilities.
    StylingConfig is required to guide the generation.
    """
    # StylingConfig is set at Synthesizer instantiation for from_scratch
    styling_config = StylingConfig(
        input_format="Customer support questions in natural language.",
        expected_output_format="Helpful, concise customer support response.",
        task="Answering customer support queries about an e-commerce platform.",
        scenario="Customers seeking help with orders, shipping, and returns.",
    )

    synthesizer = Synthesizer(
        model="gpt-4.1",
        styling_config=styling_config,
    )

    goldens = synthesizer.generate_goldens_from_scratch(num_goldens=10)

    print(f"Generated {len(goldens)} goldens from scratch")
    return synthesizer, goldens


# =============================================================================
# 5. Save and Load Goldens
# =============================================================================
def save_and_load_goldens(synthesizer):
    """Save generated goldens locally as JSON and load them into an EvaluationDataset."""

    # Save locally as JSON
    synthesizer.save_as(
        file_type="json",
        directory="./synthetic_data",
        file_name="my_goldens",
    )
    print("Saved goldens to ./synthetic_data/my_goldens.json")

    # Create an EvaluationDataset from the generated goldens
    dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)

    # Optionally push to Confident AI (requires deepeval login)
    # dataset.push(alias="My Generated Dataset")

    return dataset


# =============================================================================
# 6. Evaluate with Generated Goldens
# =============================================================================
def evaluate_with_goldens(dataset):
    """
    Use generated goldens in an evaluation run.
    Replace the simulated LLM call with your actual application.
    """

    # Simulated LLM application — replace with your actual pipeline
    def my_llm_app(query: str) -> str:
        return f"Thank you for your question about '{query}'. Here is a helpful response."

    # Build test cases from goldens by running each through your LLM
    from deepeval.test_case import LLMTestCase

    test_cases = []
    for golden in dataset.goldens:
        actual_output = my_llm_app(golden.input)
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=actual_output,
            expected_output=golden.expected_output,
            context=golden.context,
        )
        test_cases.append(test_case)

    # Run evaluation
    metric = AnswerRelevancyMetric(threshold=0.5)
    results = evaluate(test_cases=test_cases, metrics=[metric])

    return results


if __name__ == "__main__":
    print("=== Generate Goldens from Contexts ===")
    synth, goldens = generate_from_contexts()

    print("\n=== Generate Goldens from Scratch ===")
    synth_scratch, goldens_scratch = generate_from_scratch()

    print("\n=== Save and Load Goldens ===")
    dataset = save_and_load_goldens(synth)

    print("\n=== Evaluate with Generated Goldens ===")
    evaluate_with_goldens(dataset)
