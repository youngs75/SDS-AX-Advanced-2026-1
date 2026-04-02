# Context Engineering

## Read This When

- Designing how an agent manages, reduces, or retrieves context across multi-turn workflows.
- Diagnosing performance degradation as conversation/tool history grows.
- Choosing between offloading, summarization, retrieval, isolation, or caching strategies.
- Encountering context failure modes (poisoning, distraction, confusion, clash).

## Skip This When

- Writing single-turn, stateless chains with small context.
- Only need code-level best practices (see `40-core-best-practices.md`).

## Official References

1. https://docs.langchain.com/oss/python/deepagents/harness - Why: SummarizationMiddleware, large-result eviction, and prompt caching are production implementations of these principles.
2. https://docs.langchain.com/oss/python/langgraph/persistence - Why: persistence and memory patterns are the runtime mechanism for context offloading.

## Core Guidance

### 1. Why Context Engineering Matters

Context grows inherently with agents — multi-step tool calls, hundreds of turns, large tool outputs. More context does NOT mean better performance; **context rot** causes degradation as length increases.

**Four failure modes to watch for:**

| Failure Mode | Symptom | Mitigation |
|--------------|---------|------------|
| **Context Poisoning** | Wrong information contaminates later decisions | Validate tool outputs, remove stale data |
| **Context Distraction** | Agent repeats easy patterns instead of advancing | Prune repetitive history, reinforce goal |
| **Context Confusion** | Too many tools → wrong tool calls, hallucinated tools | Limit active tool set, hierarchical tool loading |
| **Context Clash** | Contradictory observations degrade reasoning | Detect conflicts, prefer recent data |

### 2. The Five Levers

#### 2.1 Offload — Move Heavy Data Out

Store token-heavy content in files/external storage; keep only pointers + summaries in conversation.

- Large tool outputs → save to file, return path + summary
- Plans/todos → maintain in files, not message history
- Pattern: `path + summary + reload method` in messages

#### 2.2 Reduce — Prune and Compress

Two approaches with different tradeoffs:

| Strategy | Method | Risk |
|----------|--------|------|
| **Compaction** | Remove redundant tokens, restructure format | Low information loss |
| **Summarization** | LLM-generated summary replacing originals | Potential information loss |

- Trigger at thresholds (e.g., 85% of context window, N turns)
- Structure summaries as: `decisions / rationale / unresolved questions / next actions`
- Prefer compaction before summarization

#### 2.3 Retrieve — Fetch Only What's Needed

Fill cleared space with **current-step-relevant** data only.

- Deterministic search first (grep/glob/keyword) — debuggable and predictable
- Include retrieval rationale so model understands why data was injected
- Apply to tools too: load only relevant tools per step to reduce confusion

#### 2.4 Isolate — Separate Contexts per Role

Split work into sub-agents with independent context windows.

- Each sub-agent starts fresh with task description + inherited state only
- Communicate via explicit artifacts (summaries, reports), not shared memory
- Principle: "Share memory by communicating, don't communicate by sharing memory"

#### 2.5 Cache — Reuse Stable Prefixes

Agent prompts contain stable tokens (system prompt, tool definitions, policies) that repeat every call.

- Split prompt into **prefix (stable)** + **suffix (mutable)**
- Stable prefix gets cached across requests (up to 10x cheaper on some providers)
- Don't mix volatile content into the prefix

### 3. Tool Context Pollution

Tools themselves clutter context. Three abstraction levels to manage this:

| Level | Mechanism | Context Impact |
|-------|-----------|----------------|
| **Function Calling** | Schema-safe, standard | High — all tool schemas in every call |
| **Sandbox Utilities** | CLI/shell commands | Low — doesn't modify model context |
| **Packages & APIs** | Pre-approved script calls | Minimal — results go to files |

Prefer higher abstraction levels for complex/data-heavy operations.

### 4. Operational Checklist

- [ ] Context budget defined (when to trigger reduction)?
- [ ] Large tool outputs offloaded to files, not left in messages?
- [ ] Reduction strategy layered: compaction → summarization?
- [ ] Retrieval assembles "what, why, in what order" — not just raw results?
- [ ] Pollutable tasks (exploration, long output, multi-step analysis) isolated to sub-agents?
- [ ] Prompt structure is cache-friendly (stable prefix / mutable suffix)?
- [ ] Failure modes monitored (poisoning, distraction, confusion, clash)?
- [ ] "Less context = more intelligence" tested before adding more?

## Quick Checklist

- [ ] Is the current context budget appropriate for the model's window?
- [ ] Are the five levers (offload, reduce, retrieve, isolate, cache) considered?
- [ ] Are failure modes being monitored or mitigated?
- [ ] Is tool context pollution minimized?

## Next File

- Move to framework layer: `../02-langchain/10-create-agent-standard.md`
