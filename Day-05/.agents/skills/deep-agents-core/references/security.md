# Security Best Practices

## Read This When

- Deploying DeepAgents to production or multi-user environments
- Need to understand the layered defense model for filesystem and execution
- Configuring restricted file access or sensitive path protection
- Reviewing security posture of an existing DeepAgent setup

## Skip This When

- Running in local development with trusted-only inputs
- Using only StateBackend with no filesystem or execution access
- Focused on agent behavior, not infrastructure security

## Official References

1. https://docs.langchain.com/oss/python/deepagents/backends - Why: Backend security features (virtual_mode, path validation, symlink protection).
2. https://docs.langchain.com/oss/python/deepagents/sandboxes - Why: Sandbox isolation, credential handling, and network boundaries.
3. https://docs.langchain.com/oss/python/deepagents/human-in-the-loop - Why: Tool approval gates as a security layer.

## Core Guidance

### 1. Four-Layer Defense Model

DeepAgents enforces security through four concentric layers:

| Layer | Mechanism | What It Protects |
|-------|-----------|-----------------|
| **Layer 1: Input Validation** | `_validate_path()`, `sanitize_tool_call_id()` | Path traversal, ID injection |
| **Layer 2: Filesystem Protection** | `virtual_mode`, `O_NOFOLLOW` | Root escape, symlink attacks |
| **Layer 3: Execution Isolation** | Base64 encoding, Sandbox process isolation | Shell injection, host access |
| **Layer 4: Result Limitation** | Token limits, large result eviction | Context overflow, memory exhaustion |

**Defense in depth**: Each layer operates independently. A bypass at one layer is caught by the next.

### 2. Production Security Configuration

```python
agent = create_deep_agent(
    model=model,
    backend=CompositeBackend(
        default=StateBackend,  # Ephemeral files in state
        routes={
            "/workspace/": FilesystemBackend(
                root_dir="./sandbox",
                virtual_mode=True,       # REQUIRED: path isolation
                max_file_size_mb=5,      # Limit grep target size
            ),
        }
    ),
    middleware=[
        FilesystemMiddleware(
            tool_token_limit_before_evict=10000,  # Lower threshold
        )
    ],
    interrupt_on={
        "execute": True,     # REQUIRED: approve shell commands
        "write_file": True,  # RECOMMENDED: approve file writes
    },
)
```

**Mandatory settings for production:**
- `virtual_mode=True` on all FilesystemBackend instances
- `interrupt_on={"execute": True}` for any sandbox-enabled agent
- Explicit `root_dir` pointing to a safe, isolated directory

### 3. Restricted Path Protection

Extend `FilesystemMiddleware` to block access to sensitive paths:

```python
class RestrictedFilesystemMiddleware(FilesystemMiddleware):
    BLOCKED_PATTERNS = ["/secrets/", "/.env", "/credentials", "/private/"]

    async def awrap_tool_call(self, request, handler):
        tool_call = request.get("tool_call", {})
        args = tool_call.get("args", {})

        # Check file_path and path arguments
        for key in ("file_path", "path"):
            path = args.get(key, "")
            if any(pattern in path for pattern in self.BLOCKED_PATTERNS):
                return ToolMessage(
                    content=f"Error: Access to '{path}' is forbidden by security policy.",
                    tool_call_id=tool_call.get("id"),
                )

        return await handler(request)
```

### 4. Credential Management

| Method | Security | Recommendation |
|--------|---------|----------------|
| Environment variables | Good | Preferred for API keys, tokens |
| Sandbox files | Bad | NEVER store credentials in sandbox filesystem |
| State/Store backend | Risky | Avoid — persisted and potentially logged |
| Injected via `context_schema` | Good | For runtime secrets the agent needs |

### 5. Network Security

Default sandbox configurations do **NOT** restrict outbound network access.

**Risk**: Agent-generated commands can exfiltrate data via HTTP, DNS, etc.

**Mitigations** (provider-dependent):
- Modal: Use `NetworkFileSystem` with restricted access
- Harbor/Docker: Apply network policies to containers
- Daytona/Runloop: Check provider documentation for network controls

### 6. Security Checklist by Deployment Stage

**Development:**
- [ ] `virtual_mode=True` on FilesystemBackend
- [ ] `interrupt_on` configured for `execute` tool

**Staging:**
- [ ] Restricted path patterns blocking sensitive directories
- [ ] Credentials passed via environment variables only
- [ ] Token eviction threshold set appropriately

**Production:**
- [ ] All four defense layers verified
- [ ] Network isolation configured for sandbox
- [ ] HITL approval for write and execute operations
- [ ] Logging and monitoring for file access patterns
- [ ] `root_dir` pointing to dedicated, permission-controlled directory

## Quick Checklist

- [ ] Is `virtual_mode=True` set on all FilesystemBackend instances?
- [ ] Is `interrupt_on={"execute": True}` configured for sandboxed agents?
- [ ] Are credentials managed via environment variables (never files)?
- [ ] Are sensitive paths blocked via custom middleware?
- [ ] Is network isolation configured for the sandbox provider?
- [ ] Are all four defense layers (input, filesystem, execution, result) active?
- [ ] Is the `root_dir` isolated and permission-controlled?

## Next File

This is the final reference in the DeepAgents section. Return to the skill router (`SKILL.md`) for other layers.
