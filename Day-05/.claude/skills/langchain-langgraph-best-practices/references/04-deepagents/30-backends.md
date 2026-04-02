# Backends

## Read This When

- You need to choose a storage backend for filesystem operations
- You want to configure CompositeBackend routing for hybrid storage
- You need to implement a custom backend for specialized storage
- You're deciding between ephemeral and persistent file storage

## Skip This When

- You're using the default StateBackend without customization
- You're focused on agent behavior, not storage implementation
- You're working with read-only data sources

## Official References

1. https://docs.langchain.com/oss/python/deepagents/backends - Why: Backend types, protocol interface, and configuration options.
2. https://docs.langchain.com/oss/python/deepagents/harness - Why: How backends integrate with middleware and tool execution flow.

## Core Guidance

### 1. BackendProtocol Interface

All backends implement this protocol:

```python
class BackendProtocol(ABC):
    def ls_info(self, path: str) -> list[FileInfo]: ...
    async def als_info(self, path: str) -> list[FileInfo]: ...

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...
    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str: ...

    def write(self, file_path: str, content: str) -> WriteResult: ...
    async def awrite(self, file_path: str, content: str) -> WriteResult: ...

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult: ...
    async def aedit(self, ...) -> EditResult: ...

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str: ...

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]: ...

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]: ...

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]: ...
```

**Key characteristics:**
- All methods have sync and async variants (async prefixed with `a`)
- Results use structured types (FileInfo, WriteResult, EditResult, GrepMatch)
- Optional methods for upload/download in sandbox backends

### 2. Backend Types Comparison

| Backend | Storage | Lifetime | Cross-Thread | Use Case |
|---------|---------|----------|--------------|----------|
| **StateBackend** | LangGraph State | Thread scope (ephemeral) | No | Temporary work files, scratchpad |
| **FilesystemBackend** | Local disk | Permanent | No (local) | Real file manipulation, local dev |
| **StoreBackend** | LangGraph Store | Permanent | Yes (namespace-based) | Long-term memory, user preferences |
| **CompositeBackend** | Routes to others | Backend-dependent | Backend-dependent | Hybrid storage strategies |

### 3. StateBackend — Default Ephemeral Storage

Files stored in LangGraph state as `dict[str, FileData]`. Ephemeral — data lost when thread ends.

**FileData structure:**
```python
{
    "content": list[str],        # Lines of text
    "created_at": str,           # ISO timestamp
    "modified_at": str,          # ISO timestamp
}
```

**When to use:**
- Agent scratchpad for intermediate work
- Temporary analysis files
- Session-scoped notes
- No persistence required

**Limitations:**
- Files vanish when thread terminates
- Not accessible across threads
- Stored in graph state (consumes state storage)

### 4. FilesystemBackend — Direct Disk Access

Real local disk operations with security constraints:

```python
from deepagents.backends import FilesystemBackend

backend = FilesystemBackend(
    root_dir="./workspace",      # Root directory
    virtual_mode=True,           # REQUIRED for security: prevents path escape
    max_file_size_mb=10,         # Skip large files in grep
)
```

**Security features:**
- `virtual_mode=True` constrains all paths under `root_dir`
- `O_NOFOLLOW` prevents symlink following (Linux/macOS)
- Path traversal (`..`, `~`) blocked

**When to use:**
- Real code file manipulation
- Local development workflows
- Integration with existing file-based tools
- Permanent storage needed

**Production requirements:**
- Always set `virtual_mode=True`
- Validate `root_dir` points to safe location
- Consider file permissions and ownership

### 5. StoreBackend — Cross-Thread Persistent Storage

Uses LangGraph Store for namespace-based persistent storage:

```python
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

backend = StoreBackend(
    store=store,
    namespace=("user", "user_123"),  # Namespace tuple
)
```

**Key characteristics:**
- Items stored under `(namespace_tuple, key)`
- Requires `store` parameter in `create_deep_agent()`
- Supports `InMemoryStore` (dev) and `PostgresStore` (production)
- Files accessible across all threads sharing the namespace

**When to use:**
- User preferences that persist across sessions
- Shared knowledge base
- Long-term memory storage
- Cross-conversation context

**Store types:**
| Store | Persistence | Use Case |
|-------|-------------|----------|
| `InMemoryStore` | Process lifetime | Development, testing |
| `PostgresStore` | Database | Production, multi-instance |

### 6. CompositeBackend — Hybrid Routing

Longest-prefix routing across multiple backends:

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend

backend = CompositeBackend(
    default=StateBackend,  # Factory (instantiated at runtime)
    routes={
        "/memories/": StoreBackend,          # Permanent cross-thread
        "/workspace/": FilesystemBackend(    # Local disk
            root_dir="./workspace",
            virtual_mode=True,
        ),
    }
)

# Routing examples:
# /notes.txt → StateBackend (default)
# /memories/facts.txt → StoreBackend
# /workspace/code.py → FilesystemBackend
```

**Routing rules:**
- Longest matching prefix wins
- Path prefix is stripped before forwarding to target backend
- Prefix restored in results
- Default backend catches all unmatched paths

**Common patterns:**
```python
# Pattern 1: Temp + Permanent
routes={
    "/memory/": StoreBackend,       # Long-term
    "/": StateBackend,              # Everything else ephemeral
}

# Pattern 2: Local + Remote
routes={
    "/local/": FilesystemBackend,   # Local files
    "/shared/": StoreBackend,       # Shared across threads
}

# Pattern 3: Sandbox + State
routes={
    "/sandbox/": SandboxBackend,    # Code execution
    "/": StateBackend,              # Scratchpad
}
```

### 7. Backend Selection Decision Tree

| Need | Backend | Why |
|------|---------|-----|
| Temporary scratchpad files | StateBackend | No persistence needed, zero config |
| Local file manipulation | FilesystemBackend | Direct disk I/O, survives restarts |
| Cross-thread shared memory | StoreBackend | Namespace-based, permanent |
| Mixed temporary + permanent | CompositeBackend | Route by path prefix |
| Sandboxed code execution | SandboxBackend (Modal/Runloop/Daytona) | Isolated execution environment |

### 8. Custom Backend Implementation

Implement `BackendProtocol` with all required methods:

```python
from deepagents.backends import BackendProtocol
from deepagents.backends.types import FileInfo, WriteResult, EditResult, GrepMatch

class CustomBackend(BackendProtocol):
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        # Your storage logic here
        pass

    def write(self, file_path: str, content: str) -> WriteResult:
        # Your storage logic here
        pass

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        # Your edit logic here
        pass

    def ls_info(self, path: str) -> list[FileInfo]:
        # Your listing logic here
        pass

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        # Your search logic here
        pass

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        # Your glob logic here
        pass
```

**Focus on:**
- `read`, `write`, `edit` — Core file operations
- `ls_info` — Directory listing
- `grep_raw`, `glob_info` — Search operations
- Async variants for I/O-bound backends

### 9. Edge Cases and Gotchas

Common pitfalls when working with backends:

| Issue | Severity | Impact | Mitigation |
|-------|----------|--------|------------|
| Windows symlink following | High | Security: path escape possible | Use `virtual_mode=True` |
| Non-UTF-8 encoding | Medium | File read/write failures | Pre-convert to UTF-8 |
| `max_file_size_mb` applies to grep only | Medium | Memory issues on large file reads | Use pagination (`offset`/`limit`) |
| Permission errors silently ignored | High | Incomplete directory listings | Monitor logs, validate results |
| TOCTOU race condition | High | Concurrent write failures | Single-thread file ops or retry logic |
| Empty file returns warning string | Medium | API inconsistency | Parse for `Error:` prefix |

#### Path Validation Rules

`_validate_path()` enforces:
1. Path traversal blocked: `..` and `~` patterns rejected
2. Windows absolute paths rejected: `C:\`, `D:\` etc.
3. Normalization: `//` removed, `.` resolved, leading `/` ensured
4. Optional prefix restriction via `allowed_prefixes` parameter

```python
# Valid paths
"/foo/bar"          # ✓
"/./foo//bar"       # ✓ → normalized to "/foo/bar"

# Rejected paths
"../etc/passwd"     # ✗ Path traversal
"C:\\Users\\file"   # ✗ Windows absolute
"/etc/f.txt"        # ✗ (when allowed_prefixes=["/data/"])
```

#### Symlink Security (Platform-Dependent)

- **Linux/macOS**: `O_NOFOLLOW` flag prevents symlink following
- **Windows**: `O_NOFOLLOW` not supported — symlinks ARE followed

**Recommendation**: Always use `virtual_mode=True` on Windows environments.

#### Edit Operation Behavior

`edit_file` with `replace_all=False` (default):
- 0 occurrences → error: `"String not found"`
- 1 occurrence → replace succeeds
- 2+ occurrences → error: `"appears N times, use replace_all=True"`

#### Error Return Patterns

Two error patterns coexist — handle both:

| Method | Return Type | Error Representation |
|--------|------------|---------------------|
| `read()` | `str` | Error string directly returned |
| `grep_raw()` | `list[GrepMatch] \| str` | Error as string, success as list |
| `write()` | `WriteResult` | `.error` field |
| `edit()` | `EditResult` | `.error` field |

```python
# Safe error handling for read()
result = backend.read("/file.txt")
if result.startswith("Error:"):
    handle_error(result)

# Safe error handling for grep_raw()
result = backend.grep_raw("pattern", "/")
if isinstance(result, str):  # Error string
    handle_error(result)
else:  # list[GrepMatch]
    process_matches(result)
```

#### Large File Handling Limits

| Setting | Value | Applies To |
|---------|-------|-----------|
| `max_file_size_mb` | 10 MB | Grep search (skips larger files) |
| `tool_token_limit_before_evict` | 20,000 tokens | Auto-save threshold for tool results |
| `read()` default limit | 2,000 lines | Pagination default |
| Line length limit | 10,000 chars | Formatting — lines split beyond this |

### 10. Key Type Definitions

```python
class FileInfo(TypedDict):
    path: str                          # Required
    is_dir: NotRequired[bool]
    size: NotRequired[int]             # Bytes
    modified_at: NotRequired[str]      # ISO 8601

@dataclass
class WriteResult:
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None  # For StateBackend

@dataclass
class EditResult:
    error: str | None = None
    path: str | None = None
    files_update: dict[str, Any] | None = None
    occurrences: int | None = None

class GrepMatch(TypedDict):
    path: str    # File path
    line: int    # Line number (1-based)
    text: str    # Matched line content
```

## Quick Checklist

- [ ] Is the correct backend chosen for the use case (ephemeral vs persistent)?
- [ ] Is `virtual_mode=True` set on FilesystemBackend in production?
- [ ] Is CompositeBackend path routing tested with expected prefixes?
- [ ] Is StoreBackend paired with `store` parameter in create_deep_agent?
- [ ] Are security boundaries (path traversal, symlink) understood and enforced?
- [ ] Have you validated namespace isolation in StoreBackend for multi-user scenarios?
- [ ] Are edge cases understood (path validation, symlink, encoding, TOCTOU)?
- [ ] Is error handling accounting for mixed error return patterns?

## Next File

→ [35-sandboxes.md](./35-sandboxes.md) — Sandbox backends, code execution, and isolation strategies
