# References Index

Depth 1 file listings for each topic category. Open one file at a time.

## 01 Getting Started

base: `references/01-getting-started/`

- `10-installation-and-setup.md` -- pip install, deepeval login, first test case, evaluate() vs assert_test(), CLI basics, key concepts (LLMTestCase, metrics, evaluate).
- `20-quickstart-by-usecase.md` -- RAG evaluation (5 metrics), chatbot evaluation (conversational metrics), agent evaluation (tracing+tools), MCP evaluation, LLM Arena comparison.
- `30-custom-models-and-embeddings.md` -- DeepEvalBaseLLM subclass, DeepEvalBaseEmbeddingModel, JSON confinement, Azure/Ollama/Local/LiteLLM/vLLM CLI setup.
- `40-integrations.md` -- Model providers (OpenAI, Anthropic, Gemini, Grok, Bedrock, Azure, Ollama, OpenRouter, LiteLLM, vLLM), Framework integrations (LangChain, LangGraph, OpenAI Agents, Anthropic, HuggingFace), Vector DB integrations (Chroma, Elasticsearch, PGVector, Qdrant).

## 02 LLM Evals

base: `references/02-llm-evals/`

- `10-evaluation-fundamentals.md` -- evaluate() function (all parameters), assert_test(), end-to-end vs component-level modes, Prompt class.
- `20-test-cases.md` -- LLMTestCase (all fields), ConversationalTestCase + Turn, ArenaTestCase, MLLMTestCase, ToolCall model.
- `30-datasets-and-goldens.md` -- EvaluationDataset, Golden/ConversationalGolden, JSON/CSV load, Confident AI push/pull, Synthesizer generation.
- `40-tracing-and-observability.md` -- @observe decorator (type, name, metrics, model), update_current_span(), trace types, observed_callback, Confident AI dashboard.
- `50-ci-cd-and-configs.md` -- CLI flags, pytest integration, AsyncConfig/DisplayConfig/ErrorConfig/CacheConfig, GitHub Actions YAML, regression testing, test hooks.
- `60-mcp-and-component-evals.md` -- MCP evaluation setup, MCPTestCase, MCP metrics, component-level @observe, span-level scoring.

## 03 Eval Metrics

base: `references/03-eval-metrics/`

- `10-metrics-overview.md` -- Metric types, BaseMetric interface, threshold concept, scoring, metric selection decision tree.
- `20-rag-metrics.md` -- AnswerRelevancy, Faithfulness, ContextualRelevancy, ContextualPrecision, ContextualRecall, Hallucination, Summarization, RAGAS. Formulas, required params, RAG Triad.
- `30-agent-metrics.md` -- ToolCorrectness, ToolUse, MCPUse, MultiTurnMCPUse, MCPTaskCompletion, TaskCompletion, DAGMetric, PlanAdherence, PlanQuality, StepEfficiency, GoalAccuracy, ArgumentCorrectness.
- `40-safety-metrics.md` -- Bias, Toxicity, PIILeakage, Misuse, RoleAdherence, RoleViolation, NonAdvice, PromptAlignment, TopicAdherence.
- `50-conversation-turn-metrics.md` -- Turn-level (TurnContextualRelevancy/Recall/Precision, TurnRelevancy, TurnFaithfulness), Conversation-level (ConversationCompleteness, ConversationalGEval, ConversationalDAG, KnowledgeRetention).
- `60-multimodal-metrics.md` -- MLLMImage, TextToImage, ImageCoherence, ImageHelpfulness, ImageReference, ImageEditing. MLLMTestCase.
- `70-utility-metrics.md` -- ExactMatch, JsonCorrectness, PatternMatch. Deterministic non-LLM metrics.
- `80-custom-metrics.md` -- GEval (criteria, evaluation_steps, rubric), ConversationalGEval, BaseMetric subclass (ROUGE example), composite metrics, DeepEvalBaseLLM custom judge.

## 04 Prompt Optimization

base: `references/04-prompt-optimization/`

- `10-introduction.md` -- PromptOptimizer, Prompt class, optimization workflow, model callback, objective metrics, AsyncConfig/DisplayConfig, hyperparameter tuning.
- `20-techniques.md` -- GEPA (genetic-Pareto multi-objective), MIPROv2 (Bayesian bootstrapped demos), COPRO (coordinate-ascent). Parameters, when to use each.

## 05 Synthetic Data Generation

base: `references/05-synthetic-data/`

- `10-synthesizer-overview.md` -- Synthesizer class, FiltrationConfig, EvolutionConfig (7 types), StylingConfig, model/embedder setup, save/load.
- `20-generation-methods.md` -- generate_goldens_from_docs (ContextConstructionConfig), generate_goldens_from_contexts, generate_goldens_from_scratch (ScenarioConfig), generate_goldens (from existing goldens). Multi-turn variants.

## 06 Red-Teaming

base: `references/06-red-teaming/`

- `10-introduction.md` -- DeepTeam class (`pip install deepteam`), red_team() function, RedTeamer, model callback, single/multi-turn attacks, risk assessment, YAML CLI config.
- `20-vulnerabilities.md` -- Complete catalog: Bias, Competition, Excessive Agency, Graphic Content, Illegal Activities, Intellectual Property, Misinformation, Personal Safety, PII Leakage, Prompt Leakage, Robustness, Toxicity, Unauthorized Access.
- `30-attack-enhancements.md` -- Attack enhancement types (prompt injection, jailbreak, role-playing, etc.), configuration, custom attacks, enhancement stacking.

## 07 Benchmarks

base: `references/07-benchmarks/`

- `10-overview.md` -- DeepEvalBaseLLM wrapper, benchmark() function, batch evaluation, task selection, few-shot, CoT.
- `20-available-benchmarks.md` -- 16 benchmarks: Reasoning (ARC, HellaSwag, WinoGrande, LogiQA), Math (GSM8K, MathQA, DROP), Knowledge (MMLU, TruthfulQA, BoolQ), Language (LAMBADA, SQuAD), Code (HumanEval), Instruction (IFEval), Fairness (BBQ), Complex (BIG-Bench Hard).

## 08 Others

base: `references/08-others/`

- `10-cli-and-environment.md` -- CLI commands (deepeval test run, set-*, login, benchmark), environment variables (OPENAI_API_KEY, DEEPEVAL_*, CONFIDENT_API_KEY), flags/configs.
- `20-conversation-simulator.md` -- ConversationSimulator class, configuration, usage patterns.
- `30-data-privacy-and-misc.md` -- Data privacy policies, troubleshooting tips, miscellaneous features.

## 09 Guides (How-to)

base: `references/09-guides/`

- `10-rag-evaluation.md` -- RAG evaluation guide + RAG Triad explanation
- `20-agent-evaluation.md` -- AI agent evaluation guide + agent evaluation metrics deep dive
- `30-custom-metrics.md` -- Building custom metrics + answer correctness metric guide
- `40-custom-llms-and-embeddings.md` -- Using custom LLMs + JSON confinement + custom embedding models
- `50-red-teaming.md` -- Red teaming guide
- `60-synthesizer.md` -- Using Synthesizer for test data generation
- `70-ci-cd-regression.md` -- Regression testing in CI/CD
- `80-observability-and-optimization.md` -- LLM observability + hyperparameter optimization

## 10 Tutorials (End-to-End)

base: `references/10-tutorials/`

- `10-introduction-and-setup.md` -- Tutorial introduction and setup instructions
- `20-medical-chatbot.md` -- Full workflow: RAG chatbot with medical knowledge base
- `30-rag-qa-agent.md` -- Full workflow: QA agent with retrieval and tools
- `40-summarization-agent.md` -- Full workflow: Meeting summarizer

## 11 Integrations (Providers & Frameworks)

base: `references/11-integrations/`

- `10-model-providers.md` -- OpenAI, Azure OpenAI, Anthropic, Gemini, Grok, Amazon Bedrock, LiteLLM, Ollama, OpenRouter, vLLM
- `20-frameworks.md` -- LangChain, LangGraph, OpenAI Agents, Anthropic SDK, HuggingFace
- `30-vector-databases.md` -- Chroma, Elasticsearch, PGVector, Qdrant
