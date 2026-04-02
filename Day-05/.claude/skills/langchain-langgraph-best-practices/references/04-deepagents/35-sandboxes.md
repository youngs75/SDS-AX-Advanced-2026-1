# Sandboxes

## Read This When

- Need isolated code execution for agent workflows
- Choosing a sandbox provider (Modal, Daytona, Runloop, Harbor)
- Understanding sandbox security boundaries and file transfer
- Configuring working directories or GPU requirements
- Debugging sandbox-related errors or command failures

## Skip This When

- Not using the `execute` tool in your agent
- Running code locally without isolation requirements
- Working with agents that only use tool calls (no shell commands)

## Official References

1. https://docs.langchain.com/oss/python/deepagents/sandboxes - Why: Complete guide to sandbox providers, configuration, and the `execute()` method interface.
2. https://docs.langchain.com/oss/python/deepagents/backends - Why: SandboxBackendProtocol interface definition and inheritance model.

## Core Guidance

### 1. What is a Sandbox?

An isolated execution environment that provides the `execute()` method. Without a sandbox, the `execute` tool is not available. All filesystem operations (`read`, `write`, etc.) are automatically built on top of `execute()` via `BaseSandbox`.

**Key Concept**: The sandbox is both a backend (provides storage/execution) AND an execution environment. It extends `BackendProtocol` with execution capabilities.

### 2. SandboxBackendProtocol

Extends BackendProtocol with execution methods:

```python
class SandboxBackendProtocol(BackendProtocol):
    def execute(self, command: str) -> ExecuteResponse: ...
    async def aexecute(self, command: str) -> ExecuteResponse: ...

    @property
    def id(self) -> str: ...  # Unique sandbox identifier
```

**ExecuteResponse** contains:
- `stdout`: Standard output string
- `stderr`: Standard error string
- `exit_code`: Integer exit code (0 = success)

### 3. Provider Comparison

| Provider | Working Dir | Cold Start | Persistence | GPU | Best For |
|----------|------------|------------|-------------|-----|----------|
| **Modal** | `/workspace` | ~2s | Ephemeral | Yes | GPU/ML workloads, data processing |
| **Daytona** | `/home/daytona` | ~1s | Workspace-based | No | Fast iteration, web development |
| **Runloop** | `/home/user` | ~3s | Devbox ID reuse | No | Disposable development environments |
| **Harbor** | Configurable | Docker-dependent | Container-based | Configurable | Production, maximum isolation |

**Selection Criteria**:
- GPU needed → Modal
- Fastest cold start → Daytona
- Container-based workflows → Harbor
- Disposable environments → Runloop

### 4. BaseSandbox

All sandbox providers extend `BaseSandbox`. It converts filesystem operations to shell commands:

- `read_file` → `python3 -c 'read script'`
- `write_file` → `python3 -c 'write script'` (Base64-encoded payload)
- `grep` → `grep -F pattern`
- All payloads Base64-encoded to prevent shell injection

**Why this matters**: You don't need to implement file operations manually. The sandbox handles them via command execution automatically.

### 5. File Transfer

Upload/download files between local filesystem and sandbox:

```python
# Upload files to sandbox
sandbox.upload_files([
    ("/data/input.csv", csv_bytes),
    ("/scripts/analyze.py", script_bytes),
])

# Download results from sandbox
results = sandbox.download_files(["/output/report.pdf", "/output/chart.png"])
# Returns: list of (path, bytes) tuples
```

**Use Cases**:
- Upload datasets before analysis
- Download generated reports/artifacts
- Transfer configuration files
- Retrieve logs/outputs

### 6. Security Boundaries

| Aspect | Protection | Risk |
|--------|-----------|------|
| Process isolation | Yes | Command injection via agent-generated commands |
| Network isolation | No (default) | Outbound network exfiltration possible |
| Filesystem isolation | Yes | Context leakage via file contents |
| Credential storage | NO | NEVER store credentials in sandbox files |

**Best Practices**:
- Set `interrupt_on={"execute": True}` for untrusted environments
- Pass credentials via environment variables, NOT files
- Review generated commands before execution
- Use network policies to restrict outbound access (provider-dependent)

### 7. Sandbox Usage Pattern

```python
from deepagents import create_deep_agent
from deepagents.sandbox import ModalSandbox

# Modal sandbox for data analysis
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    backend=ModalSandbox(
        setup_script="pip install pandas matplotlib seaborn",
        gpu="any",  # Request GPU access
    ),
    interrupt_on={"execute": True},  # Approve commands before execution
)

# Agent can now use execute tool to run Python scripts
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze data.csv and create a plot"}]
})
```

**Configuration Options**:
- `setup_script`: Commands to run on sandbox initialization
- `gpu`: GPU requirement (Modal only)
- `working_dir`: Override default working directory
- `environment`: Environment variables for sandbox

### 8. Common Patterns

**Pattern 1: Data Processing Pipeline**
```python
# 1. Upload data
sandbox.upload_files([("/data/input.csv", data_bytes)])

# 2. Agent processes
result = agent.invoke({"messages": [{"role": "user", "content": "Process /data/input.csv"}]})

# 3. Download results
output = sandbox.download_files(["/data/output.csv"])
```

**Pattern 2: Iterative Development**
```python
# Daytona persists workspace across invocations
agent = create_deep_agent(
    backend=DaytonaSandbox(workspace_id="my-dev-env"),
    interrupt_on={"execute": True},
)

# Changes persist between agent runs
agent.invoke({"messages": [{"role": "user", "content": "Install dependencies"}]})
agent.invoke({"messages": [{"role": "user", "content": "Run tests"}]})
```

### 9. BaseSandbox Command Conversion

`BaseSandbox` converts all filesystem operations into shell commands:

| Operation | Shell Command |
|-----------|--------------|
| `read_file` | `python3 -c 'read script'` |
| `write_file` | `python3 -c 'write script'` (Base64-encoded payload) |
| `edit_file` | `python3 -c 'edit script'` (Base64-encoded payload) |
| `grep` | `grep -F pattern` |
| `glob` | `python3 -c 'glob script'` |

**Security pattern — Base64 encoding prevents shell injection:**
```python
# All payloads are Base64-encoded before passing to shell
_WRITE_COMMAND_TEMPLATE = """python3 -c "
import base64
content = base64.b64decode('{content_b64}').decode('utf-8')
with open('{file_path}', 'w') as f:
    f.write(content)
" 2>&1"""
```

**Why this matters**: You don't need to implement file operations for custom sandbox providers. `BaseSandbox` handles them via the `execute()` method automatically.

### 10. HarborSandbox (Docker-Based)

Docker container-based sandbox with maximum isolation:

```python
class HarborSandbox(SandboxBackendProtocol):
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment  # Harbor Docker environment

    async def aexecute(self, command: str) -> ExecuteResponse:
        result = await self.environment.run(command)
        return ExecuteResponse(
            output=self._filter_tty_noise(result.stdout + result.stderr),
            exit_code=result.exit_code,
        )
```

**TTY noise filtering** — Harbor auto-removes these common Docker artifacts:
- `"cannot set terminal process group (-1)"`
- `"no job control in this shell"`
- `"initialize_job_control: Bad file descriptor"`

**When to use**: Production environments requiring maximum process and filesystem isolation.

### 11. Sandbox Factory Pattern

CLI integration provides a unified factory for sandbox creation:

```python
from deepagents_cli.integrations.sandbox_factory import create_sandbox

with create_sandbox(
    provider="modal",                 # "modal" | "runloop" | "daytona"
    setup_script_path="./setup.sh",   # Initialization script
) as sandbox:
    result = sandbox.aexecute("python3 --version")
    print(result.output)
```

### 12. ExecuteResponse Type

```python
@dataclass
class ExecuteResponse:
    output: str                # stdout + stderr combined
    exit_code: int | None = None
    truncated: bool = False    # Whether output was truncated
```

## Quick Checklist

- [ ] Is a sandbox provider chosen appropriate for the workload (GPU, cold start, persistence)?
- [ ] Is `interrupt_on={"execute": True}` set for untrusted environments?
- [ ] Are credentials passed via environment variables (not stored in sandbox)?
- [ ] Is Base64 encoding used for all payloads (handled automatically by BaseSandbox)?
- [ ] Are file transfer patterns used for input/output data?
- [ ] Is the working directory understood for the chosen provider?
- [ ] Are network restrictions configured if needed (provider-dependent)?
- [ ] Is the setup script tested for dependency installation?
- [ ] Is TTY noise filtering understood when using HarborSandbox?

## Next File

`40-subagents.md` - Configuring delegation, inheritance, and parallel sub-agent execution.
