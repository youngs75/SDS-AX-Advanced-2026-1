# LangGraph: Workflow Patterns

## Read This When
- Choosing a workflow architecture pattern for a new project
- Need problem-shape to pattern mapping
- Evaluating which agent pattern fits your requirements

## Skip This When
- Pattern already chosen and working
- Need implementation details (see specific pattern assets)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/workflows-agents
   - Why: six official patterns with decision criteria.

## Core Guidance

### 1. Six Workflow Patterns

| Pattern | Description | Key Mechanism | Example |
|---------|-------------|---------------|---------|
| **Prompt Chaining** | Sequential LLM calls, each refining previous output | Linear edges | Draft → Review → Polish |
| **Routing** | Classify input, dispatch to specialized handler | Conditional edges | Support ticket → Sales/Tech/Billing |
| **Parallelization** | Independent tasks run concurrently | Send + reducer | Search 3 databases simultaneously |
| **Orchestrator-Worker** | LLM decomposes task, workers execute subtasks | Send + dynamic fan-out | "Research X" → [subtask1, subtask2, ...] |
| **Evaluator-Optimizer** | Generate then evaluate in loop until quality met | Conditional loop edge | Generate → Evaluate → (retry or accept) |
| **Agentic Loop** | LLM decides tool calls, iterates until done | tools_condition + ToolNode | ReAct-style tool-calling agent |

### 2. Decision Matrix

| Question | If Yes → Pattern |
|----------|-----------------|
| Steps known in advance, each transforms previous output? | **Prompt Chaining** |
| Input type determines which handler runs? | **Routing** |
| Independent subtasks, results aggregated? | **Parallelization** |
| LLM must dynamically decide subtasks? | **Orchestrator-Worker** |
| Output needs iterative quality improvement? | **Evaluator-Optimizer** |
| Open-ended tool use until task complete? | **Agentic Loop** |

### 3. Pattern Details

**Prompt Chaining**
- Simplest pattern: `node_a → node_b → node_c`
- Each node receives full state, adds/transforms its piece
- Use when: step count is fixed, each step has clear input/output

**Routing**
- Classifier node + `add_conditional_edges` for branching
- Use structured output or keyword matching for classification
- Asset: `assets/03-langgraph/agent-patterns/router-example/graph.py`

**Parallelization**
- Return `list[Send]` from orchestrator node
- Workers execute in parallel, reducer aggregates results
- Asset: `assets/03-langgraph/agent-patterns/orchestrator-example/graph.py`

**Orchestrator-Worker**
- LLM generates subtask list dynamically
- `Send("worker", subtask)` for each subtask
- Results collected via `Annotated[list, operator.add]` reducer
- Asset: `assets/03-langgraph/agent-patterns/orchestrator-example/graph.py`

**Evaluator-Optimizer**
- Generate → Evaluate → conditional edge (retry or accept)
- Set max iterations to prevent infinite loops
- Use structured output for evaluation scores

**Agentic Loop**
- Model calls tools iteratively: `agent → tools_condition → tool_node → agent → ...`
- Terminates when model returns no tool calls
- `tools_condition` from `langgraph.prebuilt` handles routing
- Asset: `assets/03-langgraph/agent-patterns/` (tool-delegation, think-tool examples)

### 4. Combining Patterns
Patterns can be nested:
- Orchestrator-Worker with Evaluator sub-loops per worker
- Router that dispatches to different Agentic Loop variants
- Prompt Chain where one step is a Parallelization fan-out

### 5. Graph API vs Functional API per Pattern

| Pattern | Graph API | Functional API |
|---------|-----------|---------------|
| Prompt Chaining | ★★★ | ★★★★★ (simpler) |
| Routing | ★★★★★ | ★★★ |
| Parallelization | ★★★★★ (Send) | ★★★★ (task futures) |
| Orchestrator-Worker | ★★★★★ (Send) | ★★★ |
| Evaluator-Optimizer | ★★★★ | ★★★★ (while loop) |
| Agentic Loop | ★★★★★ | ★★★★ |

### 6. Asset Mapping

| Pattern | Asset Path |
|---------|-----------|
| Router | `assets/03-langgraph/agent-patterns/router-example/` |
| Supervisor | `assets/03-langgraph/agent-patterns/supervisor-example/` |
| Orchestrator-Worker | `assets/03-langgraph/agent-patterns/orchestrator-example/` |
| Handoff | `assets/03-langgraph/agent-patterns/handoff-example/` |
| HITL / Retry | `assets/03-langgraph/error-handling/` |
| Command Routing | `assets/03-langgraph/command-patterns/` |
| Multi-Stage Pipeline | `assets/03-langgraph/multi-stage-pipeline/` |

## Quick Checklist
- [ ] Decomposition known in advance (chaining) or dynamic (orchestrator)?
- [ ] Branches need shared mutable state (graph) or independent (functional)?
- [ ] Quality gate explicit for evaluator patterns?
- [ ] Pattern matches problem shape (not over-engineered)?
- [ ] Max iterations set for loop patterns?

## Next File
- `70-testing.md`
