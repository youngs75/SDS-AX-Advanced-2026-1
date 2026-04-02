# Deep Agent UI

## Read This When

- Need to set up the web UI for DeepAgents
- Connecting the UI to a backend LangGraph server
- Debugging agent execution with visual interface
- Building a UI for tool call approval and visualization
- Troubleshooting connection and authentication issues

## Skip This When

- Only using CLI or library API without a web interface
- Building a completely custom UI from scratch
- Agent doesn't need visual debugging tools

## Official References

1. https://github.com/langchain-ai/deep-agents-ui - Why: UI setup, configuration, and architecture
2. https://docs.langchain.com/oss/python/langgraph/application-structure - Why: Backend structure for UI connection and deployment

## Core Guidance

### 1. Repository

`github.com/langchain-ai/deep-agents-ui` — A Next.js web application for interacting with DeepAgents.

### 2. Setup

```bash
git clone https://github.com/langchain-ai/deep-agents-ui.git
cd deep-agents-ui
npm install

# Configure environment
cp .env.example .env.local
# Edit .env.local:
# NEXT_PUBLIC_API_URL=http://localhost:2024
```

### 3. Backend Connection

The UI connects to a LangGraph server:

```bash
# Start backend (in your DeepAgent project directory)
langgraph dev

# This starts a server at localhost:2024
# UI connects to this endpoint
```

### 4. langgraph.json Configuration

Maps graph keys to assistant IDs:

```json
{
  "graphs": {
    "agent": "./src/agent.py:graph"
  },
  "env": ".env"
}
```

The UI uses the graph key (e.g., "agent") as the assistant ID.

### 5. UI Features

| Feature | Description |
|---------|-------------|
| Chat interface | Conversational interaction with the agent |
| Tool call visualization | See each tool call, arguments, and results |
| Interrupt approval | Approve/edit/reject interrupted tool calls |
| File browser | View virtual filesystem contents |
| Todo tracker | See agent's task list and progress |
| Debug mode | Step-by-step execution with state inspection |

### 6. Debug Mode

Step through agent execution:

- View each LLM call and response
- Inspect state at each node
- See tool call arguments and results
- Monitor token usage per step

### 7. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection refused | Backend not running | Start with `langgraph dev` |
| CORS error | URL mismatch | Check `NEXT_PUBLIC_API_URL` matches backend |
| Auth error | Missing API key | Set `LANGSMITH_API_KEY` in backend `.env` |
| No assistant found | Wrong graph key | Check `langgraph.json` graph mapping |
| Stale state | Cached thread | Clear thread or use new thread_id |

## Quick Checklist

- [ ] Is the UI repository cloned and dependencies installed?
- [ ] Is `NEXT_PUBLIC_API_URL` pointing to the correct backend?
- [ ] Is `langgraph dev` running before starting the UI?
- [ ] Is `langgraph.json` properly configured with graph mapping?
- [ ] Is debug mode used for development and testing?

## Next File

- Continue to data analysis: `75-data-analysis-workflow.md`
