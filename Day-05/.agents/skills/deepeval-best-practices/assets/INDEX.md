# Assets Index

Code templates for common DeepEval use cases. Select one asset after choosing a reference file.

## Rules

1. Select one reference file first, then use exactly one matching asset.
2. Treat assets as templates: replace application logic, test data, and metric thresholds before finalizing.
3. In every answer, state which asset path you used and what you changed.

## Asset Map

| Asset path | Use when | How to apply | Expected output |
| --- | --- | --- | --- |
| `rag-evaluation/evaluate_rag.py` | RAG pipeline evaluation with retriever and generator metrics | Replace `rag_pipeline()` with actual retriever+generator, adjust thresholds | RAG evaluation with AnswerRelevancy, Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall |
| `agent-evaluation/evaluate_agent.py` | AI agent tool usage, task completion, or argument correctness | Replace agent functions, update `ToolCall` definitions, adjust `@observe` decorators | Agent evaluation with TaskCompletion, ToolCorrectness, ArgumentCorrectness |
| `custom-metric/custom_metric.py` | Custom metrics using GEval (LLM-as-judge) or BaseMetric (non-LLM) | Customize GEval criteria/steps, or subclass BaseMetric for traditional scoring | Custom metrics for `evaluate()` or `assert_test()` in CI/CD |
| `ci-cd-pipeline/test_deepeval.py` | pytest-compatible DeepEval tests for CI/CD | Replace `your_llm_app()`, update dataset goldens, adjust thresholds | pytest suite runnable via `deepeval test run test_deepeval.py` |
| `synthesizer/generate_goldens.py` | Synthetic test data from documents, contexts, or scratch | Configure document paths, adjust FiltrationConfig/EvolutionConfig/ScenarioConfig | Golden objects for EvaluationDataset |
| `red-teaming/red_team_scan.py` | Red team LLM app for vulnerabilities using DeepTeam | Replace model callback, select vulnerabilities, configure attack enhancements | Vulnerability assessment (Bias, Toxicity, PII Leakage, etc.) |
| `prompt-optimization/optimize_prompt.py` | Automatic prompt optimization (GEPA, MIPROv2, COPRO) | Define Prompt class, set objective metrics, choose technique | Optimized prompt templates |
| `tracing/component_tracing.py` | Component-level evaluation with @observe tracing | Wrap components with @observe, attach metrics to spans | Per-span metrics with nested tracing |
| `conversation/multi_turn_eval.py` | Multi-turn conversation evaluation | Construct ConversationalTestCase from chat history, select metrics | ConversationCompleteness, KnowledgeRetention, turn-level scoring |
| `benchmarks/run_benchmark.py` | Benchmark LLM against standard suites (MMLU, GSM8K, etc.) | Subclass DeepEvalBaseLLM, select benchmark suites | Benchmark results across evaluation suites |
