# LangGraph: Choosing APIs

## Read This When
- Deciding between Graph API and Functional API for a new project
- Need a decision framework for API selection
- Evaluating whether to migrate between APIs

## Skip This When
- API choice already made and working
- Question is about a specific API's details (see 10 or 15)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/choosing-apis
   - Why: official decision guide for Graph vs Functional API.

## Core Guidance

### 1. Quick Decision Rule
- **Need explicit topology** (visualization, concurrent branches, team routing)? → **Graph API**
- **Procedural / sequential** (standard Python flow, rapid prototyping)? → **Functional API**
- **Unsure**? Start with Functional API — less boilerplate, easier migration later

### 2. Comparison Table

| Aspect | Graph API | Functional API |
|--------|-----------|---------------|
| Paradigm | Declarative DAG | Imperative Python |
| State management | Explicit TypedDict + reducers | Function-local variables |
| Visualization | Built-in (Mermaid, PNG) | Not available |
| Control flow | Edges + conditional routing | Python if/for/while |
| Concurrency | Parallel nodes via fan-out | Parallel tasks via futures |
| Code volume | More boilerplate | Minimal |
| Team development | Clear node boundaries | Standard Python modules |
| Debugging | Step-by-step state inspection | Standard Python debugging |
| Learning curve | Higher (graph concepts) | Lower (familiar Python) |

### 3. Graph API Strengths
- **Visualization**: Auto-generated Mermaid/PNG diagrams for documentation
- **State coordination**: Explicit reducers for merging concurrent updates
- **Fan-out/fan-in**: Send API for dynamic parallel execution
- **Team development**: Clear node boundaries make parallel development easy
- **Studio integration**: Visual debugging in LangGraph Studio

### 4. Functional API Strengths
- **Minimal boilerplate**: Standard async Python, no graph wiring
- **Familiar patterns**: if/for/while instead of conditional edges
- **Rapid prototyping**: Fastest path from idea to working code
- **Legacy integration**: Wrap existing async functions with @task
- **Short-term memory**: `previous` parameter without explicit state schema

### 5. Combining Both APIs
- Use Functional API for simple subflows within a Graph API system
- Call a compiled graph from within a `@task` function
- Both produce Pregel instances — same underlying runtime
- Same checkpointer, store, and streaming work across both

### 6. Migration Between APIs
- **Functional → Graph**: Extract tasks into node functions, define State TypedDict, wire edges
- **Graph → Functional**: Convert nodes to tasks, replace edges with Python control flow
- Both directions supported because both compile to Pregel
- Data persistence is compatible — can switch APIs without losing checkpoints

## Quick Checklist
- [ ] Decision documented with rationale?
- [ ] Team aligned on chosen API?
- [ ] Does complexity warrant Graph API (or is Functional sufficient)?
- [ ] If combining: clear boundary between Graph and Functional portions?

## Next File
- `25-state-design.md`
