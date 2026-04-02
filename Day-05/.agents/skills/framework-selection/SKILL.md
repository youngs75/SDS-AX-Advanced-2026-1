---
name: framework-selection
description: "INVOKE THIS SKILL at the START of any LangChain/LangGraph/Deep Agents project, before writing any agent code. Determines which framework layer is right for the task: LangChain, LangGraph, Deep Agents, or a combination. Must be consulted before other agent skills."
---

<overview>
LangChain, LangGraph, and Deep Agents are **layered**, not competing choices. Each builds on the one below it:

```
┌─────────────────────────────────────────┐
│              Deep Agents                │  ← highest level: batteries included
│   (planning, memory, skills, files)     │
├─────────────────────────────────────────┤
│               LangGraph                 │  ← orchestration: graphs, loops, state
│    (nodes, edges, state, persistence)   │
├─────────────────────────────────────────┤
│               LangChain                 │  ← foundation: models, tools, chains
│      (models, tools, prompts, RAG)      │
└─────────────────────────────────────────┘
```

Picking a higher layer does not cut you off from lower layers — you can use LangGraph graphs inside Deep Agents, and LangChain primitives inside both.

> **This skill should be loaded at the top of any project before selecting other skills or writing agent code.** The framework you choose dictates which other skills to invoke next.
</overview>

---

## Decision Guide

<decision-table>

Answer these questions in order:

| Question | Yes → | No → |
|----------|-------|-------|
| Does the task require breaking work into sub-tasks, managing files across a long session, persistent memory, or loading on-demand skills? | **Deep Agents** | ↓ |
| Does the task require complex control flow — loops, dynamic branching, parallel workers, human-in-the-loop, or custom state? | **LangGraph** | ↓ |
| Is this a single-purpose agent that takes input, runs tools, and returns a result? | **LangChain** (`create_agent`) | ↓ |
| Is this a pure model call, chain, or retrieval pipeline with no agent loop? | **LangChain** (LCEL / chain) | — |

</decision-table>

---

## Framework Profiles

<langchain-profile>

### LangChain — Use when the task is focused and self-contained

**Best for:**
- Single-purpose agents that use a fixed set of tools
- RAG pipelines and document Q&A
- Model calls, prompt templates, output parsing
- Quick prototypes where agent logic is simple

**Not ideal when:**
- The agent needs to plan across many steps
- State needs to persist across multiple sessions
- Control flow is conditional or iterative

**Skills to invoke next:** `langchain-models`, `langchain-rag`, `langchain-middleware`

</langchain-profile>

<langgraph-profile>

### LangGraph — Use when you need to own the control flow

**Best for:**
- Agents with branching logic or loops (e.g. retry-until-correct, reflection)
- Multi-step workflows where different paths depend on intermediate results
- Human-in-the-loop approval at specific steps
- Parallel fan-out / fan-in (map-reduce patterns)
- Persistent state across invocations within a session

**Not ideal when:**
- You want planning, file management, and subagent delegation handled for you (use Deep Agents instead)
- The workflow is straightforward enough for a simple agent

**Skills to invoke next:** `langgraph-fundamentals`, `langgraph-human-in-the-loop`, `langgraph-persistence`

</langgraph-profile>

<deep-agents-profile>

### Deep Agents — Use when the task is open-ended and multi-dimensional

**Best for:**
- Long-running tasks that require breaking work into a todo list
- Agents that need to read, write, and manage files across a session
- Delegating subtasks to specialized subagents
- Loading domain-specific skills on demand
- Persistent memory that survives across multiple sessions

**Not ideal when:**
- The task is simple enough for a single-purpose agent
- You need precise, hand-crafted control over every graph edge (use LangGraph directly)

**Middleware — built-in and extensible:**

Deep Agents ships with a built-in middleware layer out of the box — you configure it, you don't implement it. The following come pre-wired; you can also add your own on top:

| Middleware | What it provides | Always on? |
|------------|-----------------|------------|
| `TodoListMiddleware` | `write_todos` tool — agent plans and tracks multi-step tasks | ✓ |
| `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` tools | ✓ |
| `SubAgentMiddleware` | `task` tool — delegate work to named subagents | ✓ |
| `SkillsMiddleware` | Load SKILL.md files on demand from a skills directory | Opt-in |
| `MemoryMiddleware` | Long-term memory across sessions via a `Store` instance | Opt-in |
| `HumanInTheLoopMiddleware` | Interrupt and request human approval before sensitive tool calls | Opt-in |

**Skills to invoke next:** `deep-agents-core`, `deep-agents-memory`, `deep-agents-orchestration`

</deep-agents-profile>

---

## Mixing Layers

<mixing-layers>
Because the frameworks are layered, they can be combined in the same project. The most common pattern is using Deep Agents as the top-level orchestrator while dropping down to LangGraph for specialized subagents.

### When to mix

| Scenario | Recommended pattern |
|----------|---------------------|
| Main agent needs planning + memory, but one subtask requires precise graph control | Deep Agents orchestrator → LangGraph subagent |
| Specialized pipeline (e.g. RAG, reflection loop) is called by a broader agent | LangGraph graph wrapped as a tool or subagent |
| High-level coordination but low-level graph for a specific domain | Deep Agents + LangGraph compiled graph as a subagent |

### How it works in practice

A LangGraph compiled graph can be registered as a subagent inside Deep Agents. This means you can build a tightly-controlled LangGraph workflow (e.g. a retrieval-and-verify loop) and hand it off to the Deep Agents `task` tool as a named subagent — the Deep Agents orchestrator delegates to it without caring about its internal graph structure.

LangChain tools, chains, and retrievers can be used freely inside both LangGraph nodes and Deep Agents tools — they are the shared building blocks at every level.

</mixing-layers>

---

## Quick Reference

<quick-reference>

| | LangChain | LangGraph | Deep Agents |
|---|-----------|-----------|-------------|
| **Control flow** | Fixed (tool loop) | Custom (graph) | Managed (middleware) |
| **Middleware layer** | Callbacks only | ✗ None | ✓ Explicit, configurable |
| **Planning** | ✗ | Manual | ✓ TodoListMiddleware |
| **File management** | ✗ | Manual | ✓ FilesystemMiddleware |
| **Persistent memory** | ✗ | With checkpointer | ✓ MemoryMiddleware |
| **Subagent delegation** | ✗ | Manual | ✓ SubAgentMiddleware |
| **On-demand skills** | ✗ | ✗ | ✓ SkillsMiddleware |
| **Human-in-the-loop** | ✗ | Manual interrupt | ✓ HumanInTheLoopMiddleware |
| **Custom graph edges** | ✗ | ✓ Full control | Limited |
| **Setup complexity** | Low | Medium | Low |
| **Flexibility** | Medium | High | Medium |

> **Middleware is a concept specific to LangChain (callbacks) and Deep Agents (explicit middleware layer). LangGraph has no middleware — you wire behavior directly into nodes and edges.**

</quick-reference>

---

## Official External References

Use when local skill documents cannot answer the question and you need direct citation from official docs.

- **Discovery root**: https://docs.langchain.com/llms.txt
- **Allowed domains**: `docs.langchain.com`, `reference.langchain.com`

| Layer | Official URLs |
| --- | --- |
| Layer boundaries | https://docs.langchain.com/oss/python/concepts/products |
| LangChain framework | https://docs.langchain.com/oss/python/langchain/agents, /models, /tools, /messages, /middleware/overview, /guardrails, /mcp, /structured-output, /streaming/overview, /short-term-memory, /long-term-memory, /multi-agent/index |
| Migration | https://docs.langchain.com/oss/python/migrate/langgraph-v1 |
| LangGraph runtime | https://docs.langchain.com/oss/python/langgraph/graph-api, /use-graph-api, /functional-api, /use-functional-api, /choosing-apis, /thinking-in-langgraph, /workflows-agents, /pregel, /durable-execution, /persistence, /add-memory, /interrupts, /streaming, /use-time-travel, /use-subgraphs, /test, /observability, /application-structure, /local-server, /studio, /deploy |
| Deep Agents harness | https://docs.langchain.com/oss/python/deepagents/overview, /quickstart, /harness, /customization, /subagents, /skills, /long-term-memory, /human-in-the-loop, /backends, /sandboxes, /cli, /data-analysis |
| Integrations | https://docs.langchain.com/oss/python/concepts/integrations |
| API reference | https://reference.langchain.com/python/langchain_core/, /runnables/ |

---

## Quick Checklist

Before finalizing your implementation:

- [ ] Layer selection validated (LangChain vs LangGraph vs Deep Agents)?
- [ ] Dependency strategy matches chosen layer?
- [ ] No deprecated APIs in chosen approach?
- [ ] Official docs consulted for edge cases?
