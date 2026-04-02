# LangGraph: Local Development and Deployment

## Read This When
- Setting up local development environment for LangGraph
- Running langgraph dev server and Studio
- Planning deployment strategy (Cloud, hybrid, self-hosted)

## Skip This When
- Only need conceptual graph explanation
- Question is about graph design (see earlier references)

## Official References
1. https://docs.langchain.com/oss/python/langgraph/application-structure
   - Why: langgraph.json configuration and project structure.
2. https://docs.langchain.com/oss/python/langgraph/local-server
   - Why: langgraph CLI, dev server, SDK testing.
3. https://docs.langchain.com/oss/python/langgraph/deploy
   - Why: deployment options (Cloud, hybrid, self-hosted, standalone).
4. https://docs.langchain.com/oss/python/langgraph/studio
   - Why: visual debugging and interactive testing.

## Core Guidance

### 1. Environment Setup
```bash
# Python 3.11+ required (3.13 recommended)
python --version  # >= 3.11

# Install dependencies with uv
uv add langchain
uv add "langgraph-cli[inmem]"  # dev tools with in-memory mode
```

### 2. Project Structure
```
my-agent/
├── pyproject.toml          # dependencies
├── langgraph.json          # graph configuration
├── .env                    # API keys (gitignored)
└── src/
    └── agent/
        ├── __init__.py
        └── graph.py        # graph definition
```

### 3. `langgraph.json` Configuration
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent/graph.py:graph"
  },
  "env": ".env",
  "python_version": "3.13"
}
```

- `dependencies`: Python packages to install (`.` = current project)
- `graphs`: Map of graph name → import path (`module:variable`)
- `env`: Path to environment variables file
- `python_version`: Python version for the runtime

### 4. `langgraph dev` — Local Development Server
```bash
langgraph dev
```

- Starts local server at `http://localhost:2024`
- Opens LangGraph Studio (visual debugging UI)
- Serves API documentation at `/docs`
- Hot-reloads on code changes
- Uses in-memory checkpointer by default

### 5. LangGraph Studio
- **Free** visual interface for agent development
- Features: step-by-step execution, state inspection, interrupt handling
- Auto-opens when running `langgraph dev`
- Can also connect to remote deployments

Key capabilities:
- Visualize graph topology (nodes, edges, conditional routing)
- Step through execution one superstep at a time
- Inspect state at any checkpoint
- Resume interrupted graphs with custom input
- Hot-reload on code changes

### 6. SDK Testing
Test running server programmatically:

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")

# Create a thread
thread = await client.threads.create()

# Run the agent
result = await client.runs.create(
    thread_id=thread["thread_id"],
    graph_id="agent",
    input={"messages": [{"role": "user", "content": "Hello"}]},
)
```

### 7. Deployment Options

| Option | Description | Best For |
|--------|-------------|----------|
| **Cloud** (LangSmith) | Managed deployment from GitHub repo | Teams wanting zero-ops |
| **Hybrid** | Control plane in cloud, compute in your infra | Data residency requirements |
| **Self-Hosted** | Full deployment in your infrastructure | Maximum control |
| **Standalone** | Single container, no external deps | Simple deployments, edge |

**Cloud deployment** (simplest):
1. Push code to GitHub
2. Connect repo in LangSmith dashboard
3. Deploy — automatic CI/CD on push

### 8. Production Checklist

| Item | Dev | Production |
|------|-----|-----------|
| Checkpointer | `InMemorySaver` | `PostgresSaver` |
| Store | `InMemoryStore` | `PostgresStore` |
| Env vars | `.env` file | Secrets manager |
| Tracing | Optional | `LANGSMITH_TRACING=true` |
| Encryption | None | `EncryptedSerializer` |

## Quick Checklist
- [ ] Graph import path in `langgraph.json` resolves correctly?
- [ ] Environment variables loaded (API keys, config)?
- [ ] `langgraph dev` runs and Studio opens?
- [ ] Base flow runs end-to-end locally?
- [ ] Production checkpointer configured (not InMemorySaver)?

## Next File
- `85-memory-patterns.md`
