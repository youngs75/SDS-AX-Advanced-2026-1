# LangGraph: Observability

## Read This When
- Need production tracing, debugging, or monitoring
- Setting up LangSmith for LangGraph applications
- Implementing PII anonymization in traces

## Skip This When
- Development-only environment with no observability needs
- Question is about testing (see `70-testing.md`)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/observability
   - Why: LangSmith tracing, metadata, tags, anonymization, debug mode.

## Core Guidance

### 1. LangSmith Setup
Enable tracing via environment variables:

```bash
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT="my-agent-project"
```

All graph invocations automatically traced when these are set.

### 2. Tracing Control

**Global**: Set `LANGSMITH_TRACING=true` in environment (traces everything)

**Selective**: Use context manager for specific code blocks:
```python
import langsmith as ls

# Trace only this block
with ls.tracing_context(enabled=True):
    result = await graph.ainvoke(input, config)

# Disable tracing for this block (even if globally enabled)
with ls.tracing_context(enabled=False):
    result = await graph.ainvoke(input, config)
```

### 3. Metadata and Tags
Attach custom context to traces for filtering and debugging:

```python
config = {
    "configurable": {"thread_id": "session_1"},
    "metadata": {
        "user_id": "user_123",
        "environment": "production",
        "version": "1.2.0",
    },
    "tags": ["production", "customer_query", "priority_high"],
}
result = await graph.ainvoke(input, config=config)
```

- **Metadata**: Key-value pairs for structured context (user ID, session info)
- **Tags**: String labels for categorization and filtering in LangSmith UI
- Both propagate through all nodes and tool calls within the trace

### 4. Dynamic Project Configuration
Route traces to different projects based on runtime context:

```python
config = {
    "configurable": {"thread_id": "t1"},
    "run_name": "customer_support_query",
    "metadata": {"ls_project": "support-traces"},  # override project
}
```

### 5. Anonymization
Redact sensitive data (PII) before sending to LangSmith:

```python
import re
import langsmith as ls

def anonymize(data):
    if isinstance(data, str):
        # Redact SSN patterns
        data = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", data)
        # Redact email patterns
        data = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL_REDACTED]", data)
    return data

ls.configure(anonymizer=anonymize)
```

### 6. Debug Stream Mode
Maximum execution detail during development:

```python
async for event in graph.astream(input, config=config, stream_mode="debug"):
    print(f"Type: {event['type']}")
    print(f"Step: {event.get('step')}")
    print(f"Payload: {event.get('payload')}")
```

Events include: `task_start`, `task_result`, `checkpoint`, and detailed node execution traces.

### 7. Monitoring Patterns
Key metrics to track in production:

| Metric | How to Track | Alert On |
|--------|-------------|----------|
| Node latency | LangSmith trace duration per node | P99 > threshold |
| Tool error rate | Count ToolMessage with error content | Rate > 5% |
| Token usage | `AIMessage.usage_metadata` | Cost > budget |
| Retry frequency | Count RetryPolicy activations | Rate spike |
| Interrupt wait time | Time between interrupt and resume | > SLA |

### 8. Run Tree Structure
LangSmith displays LangGraph execution as a tree:
- **Root**: Graph invocation
  - **Child**: Each node execution
    - **Grandchild**: LLM calls, tool calls within the node
- Navigate to any level for detailed inputs/outputs/latency

## Quick Checklist
- [ ] LangSmith configured with API key and project?
- [ ] Tracing enabled for production workloads?
- [ ] PII anonymized before sending to LangSmith?
- [ ] Metadata includes user_id and environment?
- [ ] Key metrics identified and alerting configured?

## Next File
- `80-local-dev-and-deployment.md`
