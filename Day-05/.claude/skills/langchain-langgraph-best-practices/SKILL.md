---
name: langchain-langgraph-best-practices
description: Python code guide for `langchain-core`, `langchain`, `langgraph`, and `deepagents`. Use when requests involve choosing the correct layer, building or migrating agents with create_agent/AgentState/middleware/guardrails/MCP/tools/messages, designing LangGraph runtime behavior (durable execution, persistence, interrupts, streaming, HITL), structured-output/streaming/memory/multi-agent patterns, or planning deepagents delegation/skills/context-engineering workflows.
---

# LangChain + LangGraph + DeepAgents Reference Guide

Use for layer selection, `create_agent` + `AgentState` standards, provider package(integrations) strategy (`langchain-*`), and DeepAgents harness of langchain's agent, sub-agent/skills (context-engineering) delegation patterns.

## Rules

1. Apply one-page-at-a-time: open one reference page, extract decision-relevant facts, then return to this router before opening the next page.
2. Python-only guidance. And use `uv` as default package manager.
3. Use official links from `docs.langchain.com`; start discovery from `https://docs.langchain.com/llms.txt`.
4. Under LangChain & LangGraph v1.0, always use the modern agent entrypoints **(see `references/02-langchain/90-deprecations-and-migration.md`)**.
5. Keep answers explanation-first and layer-correct.

## Minimal Load Order

1. Open exactly one Depth1 branch based on the user request.
2. Open only one Depth2 file unless a concrete gap remains.
3. Use the "Official External References" section below only when deeper official context is required.

## Asset Usage Map

| Asset path | Use when | How to apply | Expected output |
| --- | --- | --- | --- |
| `assets/01-langchain-core/prompt-patterns/` | Design structured prompts with XML sections, template variables, and few-shot examples using only core message types. | Start from `graph.py`, replace XML section content, adjust `.format()` variables, and customize few-shot pairs for the target domain. | Reusable prompt templates with XML structure, variable injection, and few-shot examples composable by any layer above. |
| `assets/01-langchain-core/message-patterns/` | Construct all 5 message types, content blocks, multimodal messages, tool_call_id matching, and RemoveMessage trimming. | Start from `graph.py`, adapt message types and content block structure for the target conversation pattern. | A reference catalog of every message type with correct construction, matching, and history management. |
| `assets/01-langchain-core/tool-definition/` | Define tools with @tool decorator, Pydantic args_schema, StructuredTool, BaseTool subclass, and InjectedToolArg for hidden parameters. | Start from `graph.py`, replace tool functions, schemas, and descriptions for the target domain. | Tool definitions using only core primitives, reusable by any layer above (langchain, langgraph, deepagents). |
| `assets/01-langchain-core/runnable-patterns/` | Demonstrate Runnable protocol (invoke/stream/batch), RunnableConfig, LCEL pipe composition, RunnableLambda, and RunnableParallel. | Start from `graph.py`, replace runnable compositions for the target pipeline and adjust config fields. | Runnable protocol examples with LCEL composition patterns, usable without any graph or framework dependency. |
| `assets/03-langgraph/project-setup/templates/` | Bootstrap a fresh LangGraph project and align local development baseline quickly. | Copy `pyproject.toml` and `.env.example`, then replace project metadata, dependency pins, and environment variable values. | A runnable Python project scaffold with `uv`-friendly dependencies and environment templates. |
| `assets/03-langgraph/state-management/` | Define or refactor graph state models before wiring nodes and edges. | Select one state template, then rename keys, update typing annotations, and align fields with node/tool contracts. | A typed state schema ready for graph compilation and stable handoff between nodes. |
| `assets/03-langgraph/agent-patterns/` | Choose one multi-agent topology (router, supervisor, orchestrator, handoff, tool-delegation, think-tool) for workflow control. | Start from one `graph.py`, rename nodes/edges, and replace routing rules, model setup, and tools for the target domain. | A single clear multi-agent graph implementation matching one selected pattern. |
| `assets/03-langgraph/error-handling/` | Add reliability controls such as retry policies, human approval checkpoints, and multi-provider token limit resilience. | Reuse one error-handling `graph.py`, then redefine exception handling, interrupt points, and resume/approval logic. | A graph with explicit failure paths and resumable human-in-the-loop control points. |
| `assets/03-langgraph/functional-api/` | Implement workflows using the LangGraph Functional API with `@entrypoint` and `@task` decorators. | Start from `graph.py`, replace task logic, adjust injectable params (`previous`, `store`, `writer`), and customize `entrypoint.final` return/save split. | A runnable Functional API workflow with durable tasks, short-term memory via `previous`, and separated return/save values. |
| `assets/03-langgraph/streaming/` | Add real-time streaming with multiple stream modes or custom event emission. | Start from `graph.py`, select stream modes (`values`, `updates`, `messages`, `custom`, `debug`), replace node logic and `get_stream_writer()` calls. | A graph demonstrating multi-mode streaming with token-level output and custom event emission. |
| `assets/03-langgraph/persistence/` | Add cross-thread memory using LangGraph Store with namespace-based key-value storage. | Start from `graph.py`, replace namespace scheme, item keys, and integrate `store.search()` for semantic retrieval. | A graph with Store-based long-term memory, namespace organization, and cross-thread data sharing. |
| `assets/03-langgraph/subgraphs/` | Compose complex systems by nesting one graph inside another as a subgraph node. | Start from `graph.py`, choose pattern (invoke-from-node or add-as-node), adjust state transformation and key mapping. | A parent-child graph composition with explicit state transformation and nested persistence. |
| `assets/03-langgraph/command-patterns/` | Use Command for dynamic routing with state updates, or Send for map-reduce fan-out. | Start from `graph.py`, replace Command routing targets and update payloads, or customize Send fan-out logic and reducer. | A graph demonstrating Command-based routing and/or Send-based map-reduce parallelization. |
| `assets/03-langgraph/time-travel/` | Debug graph execution using checkpoint history, state modification, and forked re-execution. | Start from `graph.py`, adapt state inspection logic, checkpoint navigation, and `update_state()` modification patterns. | A debug harness demonstrating checkpoint history traversal, state editing, and forked execution. |
| `assets/03-langgraph/testing/` | Write pytest-asyncio tests for LangGraph nodes, graphs, and partial execution flows. | Start from `graph.py`, replace graph/node definitions, adapt assertions for domain-specific state, and add partial execution tests. | A pytest-asyncio test suite covering node isolation, full graph flow, checkpoint state, and partial execution testing. |
| `assets/03-langgraph/durable-execution/` | Build replay-safe graphs with idempotent side effects and deterministic execution. | Start from `graph.py`, replace side-effect logic with check-before-act patterns, separate pure computation from I/O in distinct nodes. | A replay-safe graph with idempotent side effects, operation tracking, and deterministic node separation. |
| `assets/03-langgraph/multi-stage-pipeline/` | Build multi-stage graph pipelines where each node performs a distinct processing step with structured prompts. For universal prompt patterns, see `assets/01-langchain-core/prompt-patterns/`. | Start from `graph.py`, replace node functions and prompt templates for the target multi-stage workflow. | A linear graph pipeline demonstrating staged processing (refine → research → report) with typed state and structured output parsing. |
| `assets/03-langgraph/dynamic-config/` | Add runtime configuration with per-stage model selection, env var fallback, and `init_chat_model(configurable_fields=...)`. | Start from `graph.py`, replace `Configuration` fields with domain-specific settings, adjust per-stage model bindings, and customize enum-based feature selection. | A graph with Pydantic configuration, hot-swappable models per stage, and env var + RunnableConfig fallback chain. |
| `assets/04-deepagents/quickstart-example/` | Bootstrap a DeepAgents project with `create_deep_agent()`, custom tools, and async invocation. | Start from `graph.py`, replace model config, tools, and system prompt for the target domain. | A runnable DeepAgents agent with custom tools, async execution, and streaming output. |
| `assets/04-deepagents/backend-patterns/` | Configure CompositeBackend with path-based routing across multiple backend types. | Start from `composite-routing.py`, adjust route prefixes and backend instances for the target storage strategy. | A CompositeBackend configuration with StateBackend, StoreBackend, and FilesystemBackend routing. |
| `assets/04-deepagents/subagent-delegation/` | Set up specialized sub-agents with delegation contracts and context isolation. | Start from `research-coordinator.py`, replace sub-agent definitions, tools, and delegation contracts for the target workflow. | A multi-agent coordinator with specialized sub-agents, least-privilege tools, and delegation contracts. |
| `assets/04-deepagents/memory-strategy/` | Implement cross-thread memory with CompositeBackend and StoreBackend namespaces. | Start from `cross-thread-memory.py`, adjust namespace hierarchy, path routing, and memory lifecycle for the target use case. | A hybrid memory configuration with ephemeral workspace and persistent cross-thread storage. |
| `assets/04-deepagents/hitl-approval/` | Add human-in-the-loop approval for risky tool operations with interrupt_on configuration. | Start from `risky-operations.py`, customize interrupt_on mapping, decision handling, and batch approval logic. | An agent with interrupt-based approval for write/execute/delete operations and 3 decision types. |
| `assets/04-deepagents/skill-loading/` | Create and load SKILL.md files with progressive disclosure and multiple backends. | Start from `progressive-disclosure.py`, adapt SKILL.md structure and loading backend for the target domain knowledge. | A skill system with on-demand loading, backend selection, and sub-agent inheritance. |
| `assets/04-deepagents/data-analysis/` | Build CSV analysis workflows with sandboxed Python execution and external sharing. | Start from `csv-analysis.py`, replace sandbox provider, analysis logic, and integration tools for the target pipeline. | A data analysis pipeline with sandbox execution, file transfer, visualization, and Slack/email sharing. |
| `assets/04-deepagents/middleware-pipeline/` | Create custom middleware for DeepAgents harness with state extension and system prompt injection. | Start from `custom-middleware.py`, replace middleware logic, state schema fields, and prompt injection for the target domain. | A custom middleware class with state schema extension, process_model_request hook, and middleware stack composition. |
| `assets/04-deepagents/built-in-tools/` | Understand and customize DeepAgents built-in filesystem tools, todo tools, and large result handling. | Start from `filesystem-tools.py`, review tool signatures and descriptions, adapt path validation and result size policies. | A reference guide for all built-in tools with signatures, validation rules, and pagination patterns. |
| `assets/04-deepagents/sandbox-execution/` | Implement a custom sandbox backend by extending BaseSandbox with execute(), download, and upload. | Start from `custom-sandbox.py`, replace execute() implementation, add provider-specific download/upload, and configure timeout. | A custom sandbox backend with SandboxBackendProtocol compliance, error filtering, and provider integration. |
| `assets/04-deepagents/context-engineering/` | Assemble the full system prompt pipeline: base instructions, custom prompt, memory, skills, and local context. | Start from `prompt-assembly.py`, customize each injection layer, adjust AGENTS.md learning rules, and configure progressive disclosure. | A complete system prompt assembly with 5-layer injection, AGENTS.md learning, and context optimization. |
| `assets/04-deepagents/non-interactive-execution/` | Run DeepAgents in scripting mode with one-shot execution, streaming output, and exit code propagation. | Start from `scripting-mode.py`, replace agent config, adjust shell-allow-list, and customize HITL handling for CI/CD. | A non-interactive agent runner with auto-approve, streaming, session persistence, and exit code propagation. |
| `assets/04-deepagents/summarization-strategy/` | Add conversation offloading with auto-summarization triggers and history backup to prevent context overflow. | Start from `conversation-offloading.py`, adjust trigger/keep ratios, history storage path, and tool argument truncation. | A conversation management system with fraction-based triggers, history backup, and model-specific defaults. |
| `assets/02-langchain/tool-runtime-example/` | Access graph state, long-term store, context, and stream_writer from within tools using ToolRuntime injection. | Start from `graph.py`, replace tool functions and ToolRuntime access patterns for the target domain. | Tools demonstrating all 5 ToolRuntime components (state, context, store, stream_writer, config) with create_agent integration. |
| `assets/02-langchain/` | Apply framework-level patterns (middleware, guardrails, MCP, streaming, structured output, memory, ToolRuntime) to `create_agent` based agents. All assets use `create_agent` from `langchain.agents`. | Start from one example `graph.py`, replace model config, tool definitions, and middleware/guardrails to match the target domain. | A runnable agent demonstrating one framework-level capability with proper async patterns. |

### Asset Use Rules

1. Select one reference file first, then use exactly one matching asset example.
2. Treat assets as templates; redefine state keys, node names, and domain logic before finalizing.
3. In every answer, state which asset path you used and what you changed.

## Reference Documents Depth Router

### Depth 0: Start files

1. `langchain-core`
   - Low-level reusable contracts (messages, runnables, tool abstractions).
2. `langchain`
   - Framework layer for `create_agent`, middleware, tools, guardrails, MCP.
3. `langgraph`
   - Runtime layer for durable execution, persistence, interrupts, streaming.
4. `deepagents`
   - Harness layer for planning, context management, and delegation.

### Depth 1: `langchain-core` package (Agent Core)

base: `references/01-langchain-core/`

- `10-primitives.md`
- `20-message-tool-schema.md`
- `30-runnables-state-types.md`
- `40-core-best-practices.md`
- `45-context-engineering.md`

### Depth 1: `langchain` package (Agent Builder)

base: `references/02-langchain/`

- `10-create-agent-standard.md`
- `15-models-and-providers.md`
- `20-middleware.md`
- `25-guardrails.md`
- `30-mcp.md`
- `35-tools.md`
- `40-messages.md`
- `45-structured-output.md`
- `50-streaming.md`
- `55-short-term-memory.md`
- `60-long-term-memory.md`
- `65-multi-agent-patterns.md`
- `90-deprecations-and-migration.md`

### Depth 1: `langgraph` package (Agent Runtime)

base: `references/03-langgraph/`

- `10-graph-api.md`
- `15-functional-api.md`
- `20-choosing-apis.md`
- `25-state-design.md`
- `30-command-and-send.md`
- `35-durable-execution.md`
- `40-persistence.md`
- `45-interrupts.md`
- `50-time-travel.md`
- `55-streaming.md`
- `60-subgraphs.md`
- `65-workflow-patterns.md`
- `70-testing.md`
- `75-observability.md`
- `80-local-dev-and-deployment.md`
- `85-memory-patterns.md`

### Depth 1: `deepagents` package (Agent Harness)

base: `references/04-deepagents/`

- `10-overview-and-quickstart.md`
- `15-harness-architecture.md`
- `20-create-deep-agent-api.md`
- `25-built-in-tools.md`
- `30-backends.md`
- `35-sandboxes.md`
- `40-subagents.md`
- `45-skills.md`
- `50-delegation-contracts.md`
- `55-long-term-memory.md`
- `60-human-in-the-loop.md`
- `65-cli-terminal-assistant.md`
- `70-deep-agent-ui.md`
- `75-data-analysis-workflow.md`
- `80-security.md`

## Official External References

Use when local reference files cannot answer the question and you need direct citation from official docs.

- **Discovery root**: https://docs.langchain.com/llms.txt
- **Allowed domains**: `docs.langchain.com`, `reference.langchain.com`

| Layer | Official URLs |
| --- | --- |
| Layer boundaries | https://docs.langchain.com/oss/python/concepts/products |
| LangChain framework | https://docs.langchain.com/oss/python/langchain/agents, /models, /tools, /messages, /middleware/overview, /guardrails, /mcp, /structured-output, /streaming/overview, /short-term-memory, /long-term-memory, /multi-agent/index |
| Migration | https://docs.langchain.com/oss/python/migrate/langgraph-v1 |
| LangGraph runtime | https://docs.langchain.com/oss/python/langgraph/graph-api, /use-graph-api, /functional-api, /use-functional-api, /choosing-apis, /thinking-in-langgraph, /workflows-agents, /pregel, /durable-execution, /persistence, /add-memory, /interrupts, /streaming, /use-time-travel, /use-subgraphs, /test, /observability, /application-structure, /local-server, /studio, /deploy |
| Deep Agents harness | https://docs.langchain.com/oss/python/deepagents/overview, /quickstart, /harness, /customization, /subagents, /skills, /long-term-memory, /human-in-the-loop, /backends, /sandboxes, /cli, /data-analysis |
| Integrations | https://docs.langchain.com/oss/python/concepts/integrations, https://github.com/langchain-ai/deep-agents-ui |
| API reference | https://reference.langchain.com/python/langchain_core/, /runnables/ |
