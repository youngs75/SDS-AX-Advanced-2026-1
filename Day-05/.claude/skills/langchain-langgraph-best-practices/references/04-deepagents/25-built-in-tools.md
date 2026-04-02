# Built-in Tools

## Read This When

- You need to understand what tools DeepAgents provides automatically
- You want to know the parameters and behavior of default tools
- You need to customize or disable built-in tool descriptions
- You're debugging tool call failures and need to verify tool contracts

## Skip This When

- You're only using custom tools without built-in filesystem operations
- You've already mastered the built-in tool set and are writing custom middleware
- You're focused purely on agent architecture without tool implementation details

## Official References

1. https://docs.langchain.com/oss/python/deepagents/harness - Why: Built-in tool descriptions, parameters, and behavior specifications.
2. https://docs.langchain.com/oss/python/deepagents/backends - Why: Tool-backend interaction layer and how tools resolve to storage.

## Core Guidance

### 1. Built-in Tools Table

All automatically injected by middleware:

| Tool | Source Middleware | Parameters | Description |
|------|-----------------|------------|-------------|
| `write_todos` | TodoListMiddleware | `todos: list[{content, status}]` | Task planning — manage pending/in_progress/completed items |
| `ls` | FilesystemMiddleware | `path: str` | List directory contents |
| `read_file` | FilesystemMiddleware | `file_path: str, offset: int=0, limit: int=2000` | Read file with pagination |
| `write_file` | FilesystemMiddleware | `file_path: str, content: str` | Create new file (fails if exists) |
| `edit_file` | FilesystemMiddleware | `file_path: str, old_string: str, new_string: str, replace_all: bool=False` | Edit existing file via string replacement |
| `glob` | FilesystemMiddleware | `pattern: str, path: str="/"` | Search files by glob pattern |
| `grep` | FilesystemMiddleware | `pattern: str, path: str=None, glob: str=None` | Search file contents by regex |
| `execute` | FilesystemMiddleware | `command: str` | Execute shell command (requires SandboxBackend) |
| `task` | SubAgentMiddleware | `description: str, subagent_type: str` | Delegate work to a sub-agent |

### 2. write_todos — Task Planning Tool

Agent calls `write_todos` to track work progression:

```python
# Status flow: pending → in_progress → completed
write_todos(todos=[
    {"content": "Research AI trends", "status": "in_progress"},
    {"content": "Write summary report", "status": "pending"},
])
```

**Common patterns:**
- Start session: Create all tasks with `status: "pending"`
- Begin task: Update to `status: "in_progress"`
- Complete task: Update to `status: "completed"`
- Never remove tasks — always preserve the full list with updated statuses

### 3. Filesystem Tool Contracts

All filesystem tools go through the backend layer. The actual storage depends on the configured backend (StateBackend, FilesystemBackend, StoreBackend, or CompositeBackend).

**Key constraints:**
- `write_file` creates new files only — attempting to write to an existing path returns an error
- `edit_file` with `replace_all=False` fails if the `old_string` appears more than once
- `read_file` supports pagination with `offset` and `limit` for large files

```python
# Example: Reading a large file in chunks
first_chunk = read_file("/large_log.txt", offset=0, limit=1000)
next_chunk = read_file("/large_log.txt", offset=1000, limit=1000)
```

### 4. Large Result Auto-Eviction

When any tool result exceeds `4 × tool_token_limit_before_evict` (default 80,000 chars):

1. Full result saved to `/large_tool_results/{tool_call_id}`
2. Agent receives first 10 lines + reference path
3. Agent can paginate with `read_file`

**Example workflow:**
```
Agent: grep("ERROR", path="/logs/")
Result: [truncated] See /large_tool_results/call_abc123
Agent: read_file("/large_tool_results/call_abc123", offset=0, limit=500)
```

### 5. execute Tool

Only available when backend implements `SandboxBackendProtocol` (e.g., Modal, Runloop, Daytona, HarborSandbox). All payloads are Base64-encoded to prevent shell injection.

```python
# Example: Running tests in a sandbox
execute("pytest tests/ -v")
execute("npm run build")
```

**Security notes:**
- Commands run in isolated environments
- No direct host filesystem access
- Timeout limits enforced by sandbox provider

### 6. Customizing Tool Descriptions

Override default tool descriptions for domain-specific clarity:

```python
from deepagents.middleware import FilesystemMiddleware

middleware = FilesystemMiddleware(
    custom_tool_descriptions={
        "write_file": "Create a new document in the workspace",
        "grep": "Search for patterns in research documents",
    },
    tool_token_limit_before_evict=10000,  # Lower threshold for eviction
)

agent = create_deep_agent(middleware=[middleware])
```

**When to customize:**
- Agent frequently misuses a tool
- Domain-specific terminology improves clarity
- You want to emphasize specific constraints

## Quick Checklist

- [ ] Are built-in tool behaviors understood (write_file creates only, edit_file replaces)?
- [ ] Is large result eviction threshold appropriate for your use case?
- [ ] Is `execute` only used with a sandbox backend configured?
- [ ] Are custom tool descriptions set if default descriptions are unclear for your domain?
- [ ] Have you tested the pagination behavior of `read_file` for large files?

## Next File

→ [30-backends.md](./30-backends.md) — Backend types, storage strategies, and CompositeBackend routing
