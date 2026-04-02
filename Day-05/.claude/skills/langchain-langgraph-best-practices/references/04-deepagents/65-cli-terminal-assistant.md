# CLI Terminal Assistant

## Read This When

- Need to install the DeepAgents CLI tool
- Configuring agent profiles for different use cases
- Using terminal assistant features and slash commands
- Setting up AGENTS.md context files
- Connecting to remote sandboxes (Runloop, Daytona, Modal)

## Skip This When

- Only using DeepAgents as a library (not as CLI tool)
- Building custom UI instead of using terminal interface
- Agent is embedded in another application

## Official References

1. https://docs.langchain.com/oss/python/deepagents/cli - Why: CLI installation, configuration, and usage patterns
2. https://docs.langchain.com/oss/python/deepagents/overview - Why: CLI feature overview and capabilities

## Core Guidance

### 1. Installation

```bash
# Recommended
uv tool install deepagents-cli

# Alternative
pip install deepagents-cli

# Verify
deepagents --version
```

### 2. Terminal Assistant Workflow

The CLI provides a conversational terminal assistant with all DeepAgents capabilities (filesystem, planning, delegation, memory).

### 3. Built-in CLI Tools

12 tools available in the terminal assistant:

| Tool | Description |
|------|-------------|
| `ls` | List directory contents |
| `read_file` | Read file with pagination |
| `write_file` | Create new file |
| `edit_file` | Edit existing file |
| `glob` | Search files by pattern |
| `grep` | Search file contents |
| `execute` | Run shell commands |
| `fetch_url` | Fetch web page content |
| `web_search` | Search the web |
| `task` | Delegate to sub-agent |
| `write_todos` | Track tasks |
| `think` | Strategic reasoning (no side effects) |

### 4. Agent Profiles

Stored at `~/.deepagents/<agent_name>/`:

```
~/.deepagents/
├── default/
│   ├── config.json          # Model, tools, settings
│   └── system_prompt.txt    # Agent personality
├── researcher/
│   ├── config.json
│   └── system_prompt.txt
└── coder/
    ├── config.json
    └── system_prompt.txt
```

### 5. AGENTS.md

Context configuration files:

- Global: `~/.deepagents/AGENTS.md` — applies to all sessions
- Project: `./AGENTS.md` — applies to current directory
- Both are loaded into agent's context automatically

### 6. Slash Commands

| Command | Description |
|---------|-------------|
| `/remember <text>` | Save to long-term memory |
| `/tokens` | Show current token usage |
| `/clear` | Clear conversation history |
| `/threads` | List/switch conversation threads |

### 7. Remote Sandbox Connection

```bash
# Connect to Runloop sandbox
deepagents --sandbox runloop --sandbox-id "devbox_123"

# Connect to Daytona workspace
deepagents --sandbox daytona --workspace "my-workspace"

# Connect to Modal
deepagents --sandbox modal --setup-script "./setup.sh"
```

### 8. LangSmith Tracing

Enable observability:

```bash
export LANGSMITH_API_KEY="lsv2_pt_..."
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="my-deepagent"
deepagents  # Traces automatically captured
```

## Quick Checklist

- [ ] Is deepagents-cli installed and accessible?
- [ ] Are agent profiles configured for different use cases?
- [ ] Is AGENTS.md set up for project-specific context?
- [ ] Is LangSmith tracing configured for production?
- [ ] Is the correct sandbox provider connected for remote work?

## Next File

- Continue to UI: `70-deep-agent-ui.md`
