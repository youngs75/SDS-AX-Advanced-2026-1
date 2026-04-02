---
name: deepeval-best-practices
description: "`deepeval` and `deepteam` LLM evaluation guide. Covers RAG evaluation, AI agent testing, custom metrics (GEval/DAGMetric/BaseMetric), synthetic data generation, red teaming, benchmarks, prompt optimization, and CI/CD integration with pytest."
---

# DeepEval LLM Evaluation Reference Guide

Use for LLM evaluation strategy, metric selection, test case design, synthetic data generation, red teaming, and CI/CD integration with DeepEval.

## Rules

1. Apply one-page-at-a-time: open one reference page, extract decision-relevant facts, then return to the router before opening the next page.
2. Prefer **Docs** (01-08) for API reference; **Guides** (09) for how-to workflows; **Tutorials** (10) for end-to-end examples; **Integrations** (11) for provider setup.
3. Python-only guidance. Use `uv` as package manager.
4. Use official links from `deepeval.com` for deeper context.
5. Keep answers explanation-first and metric-correct: always specify required `LLMTestCase` parameters for each metric.
6. Limit metric selection to 2-3 generic + 1-2 custom metrics per evaluation (DeepEval best practice).
7. When referencing red teaming, note that it has been separated into the `deepteam` package (`uv add --group dev deepteam`).

## Minimal Load Order

1. Use the **Depth 0** table below to identify the topic category.
2. Open `references/INDEX.md` to find the exact file within that category.
3. Open only one reference file unless a concrete gap remains.
4. If a code template is needed, open `assets/INDEX.md` to select the matching asset.
5. Use the "Official External References" section only when deeper official context is required.

## Reference Documents Depth Router

### Depth 0: Topic Overview

| # | Category | Topic | Description | When to open |
| --- | --- | --- | --- | --- |
| 01 | **Docs** | Getting Started | Installation, setup, quickstarts, custom models, integrations | New project setup, first test case, model/framework integration |
| 02 | **Docs** | LLM Evals | Evaluation fundamentals, test cases, datasets, tracing, CI/CD, MCP | Running evaluations, configuring test infrastructure |
| 03 | **Docs** | Eval Metrics | RAG, agent, safety, conversation, multimodal, utility, custom metrics | Choosing and configuring metrics |
| 04 | **Docs** | Prompt Optimization | GEPA, MIPROv2, COPRO algorithms, hyperparameter tuning | Optimizing prompts automatically |
| 05 | **Docs** | Synthetic Data | Synthesizer class, 4 generation methods, evolution and filtration | Generating test data |
| 06 | **Docs** | Red-Teaming | DeepTeam vulnerabilities, attack enhancements, adversarial testing | Security testing LLM applications |
| 07 | **Docs** | Benchmarks | 16 benchmark suites, DeepEvalBaseLLM wrapper, batch evaluation | Benchmarking model performance |
| 08 | **Docs** | Others | CLI commands, environment variables, conversation simulator | Configuration, debugging, utilities |
| 09 | **Guides** | How-to Guides | RAG, agent, custom metrics, CI/CD regression, observability | Practical step-by-step workflows |
| 10 | **Tutorials** | End-to-End | Medical chatbot, RAG QA agent, summarization agent | Learning by complete worked examples |
| 11 | **Integrations** | Providers & Frameworks | Model providers, LLM frameworks, vector databases | Setting up specific providers or frameworks |

> Depth 1 file listings: `references/INDEX.md`
> Asset templates: `assets/INDEX.md`

---

## Metric Quick-Selection Guide

| Use Case | Recommended Metrics | Required Test Case Fields |
| --- | --- | --- |
| **RAG Pipeline** | AnswerRelevancy, Faithfulness, ContextualPrecision | input, actual_output, retrieval_context, expected_output (for precision) |
| **RAG + Recall** | + ContextualRecall, ContextualRelevancy | + expected_output, context |
| **AI Agent (tools)** | ToolCorrectness, TaskCompletion | input, actual_output, tools_called, expected_tools |
| **AI Agent (planning)** | PlanAdherence, PlanQuality, DAGMetric | input, actual_output + @observe tracing |
| **Chatbot (safety)** | Bias, Toxicity, RoleAdherence | input, actual_output |
| **Chatbot (multi-turn)** | ConversationCompleteness, KnowledgeRetention | ConversationalTestCase with turns |
| **Summarization** | SummarizationMetric, Faithfulness | input, actual_output, context |
| **Custom criteria** | GEval with evaluation_steps | input, actual_output (+ any custom fields) |
| **MCP Agent** | MCPUse, MCPTaskCompletion | input, actual_output, tools_called |

### Metric Comparison Guide

| Metric A | Metric B | Key Difference | When to Choose A | When to Choose B |
| --- | --- | --- | --- | --- |
| Faithfulness | Hallucination | Faithfulness uses `retrieval_context`; Hallucination uses `context` | RAG with explicit retrieval | General LLM with provided context |
| AnswerRelevancy | GEval (Relevancy) | AnswerRelevancy is pre-built; GEval is customizable | Standard relevancy checks | Domain-specific relevancy |
| ContextualPrecision | ContextualRecall | Precision = ranking quality; Recall = coverage | Optimizing reranker | Optimizing embedding model |
| TaskCompletion | ToolCorrectness | TaskCompletion uses tracing; ToolCorrectness is end-to-end | Complex multi-step agents | Simple tool-calling agents |
| ConversationalGEval | ConversationCompleteness | GEval = custom criteria; Completeness = goal achievement | Custom conversation quality | Checking if user goals were met |

### Metric Combination Patterns

| Pattern | Metrics | Use Case |
| --- | --- | --- |
| RAG Triad (Referenceless) | AnswerRelevancy + Faithfulness + ContextualRelevancy | Production RAG without labeled data |
| RAG Full Suite | RAG Triad + ContextualPrecision + ContextualRecall | Development RAG with expected outputs |
| Agent Complete | TaskCompletion + ToolCorrectness + ArgumentCorrectness | Agent tool usage evaluation |
| Safety Baseline | Bias + Toxicity + PIILeakage | Content safety checks |
| Conversation Full | ConversationCompleteness + KnowledgeRetention + TurnRelevancy | Multi-turn chatbot evaluation |

## Official External References

> Local references로 답이 안 될 때만 사용. 상세 URL 목록: `references/external-refs.md`

- **DeepEval docs**: https://deepeval.com/docs
- **GitHub**: https://github.com/confident-ai/deepeval
- **DeepTeam (Red Teaming)**: https://github.com/confident-ai/deepteam
