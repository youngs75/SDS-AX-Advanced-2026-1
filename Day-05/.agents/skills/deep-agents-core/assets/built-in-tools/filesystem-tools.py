"""
DeepAgents Built-in Filesystem Tools and TodoList Tools

Demonstrates:
- 7 filesystem tools: ls, read_file, write_file, edit_file, glob, grep, execute
- TodoList tools: write_todos, read_todos
- Path normalization and validation rules
- Large result handling (>20K tokens eviction to /large_tool_results/)
- Tool availability based on backend type
- read_file line numbering and pagination
- edit_file uniqueness requirements
- grep literal text matching (NOT regex)
"""

import asyncio
from typing import Literal

# from langchain_openai import ChatOpenAI
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# Commented imports for reference:
# from deepagents.agent import create_deep_agent
# from deepagents.middleware.filesystem import FilesystemMiddleware
# from deepagents.middleware.todo_list import TodoListMiddleware
# from deepagents.backend.store import StateBackend
# from deepagents.backend.sandbox import LocalShellBackend

# --- Model Configuration ---
# model = ChatOpenAI(model="gpt-4o", temperature=0)
model = FakeListChatModel(
    responses=[
        "I'll list the directory first using ls.",
        "Now I'll read the file using read_file.",
        "I'll create a plan with write_todos.",
    ]
)

# ==== Simulated Filesystem Tools ====


async def ls_tool(path: str = "/") -> str:
    """
    List directory contents with metadata.

    Args:
        path: Directory path (default: "/")

    Returns:
        JSON array of entries with name, type, size, permissions

    Rules:
        - All paths normalized to start with /
        - Returns: [{"name": "file.txt", "type": "file", "size": 1024, "permissions": "rw-r--r--"}]
        - Directories have type: "directory"
    """
    return '[{"name": "project", "type": "directory", "size": 0, "permissions": "rwxr-xr-x"}]'


async def read_file_tool(
    file_path: str,
    offset: int = 0,
    limit: int = 2000
) -> str:
    """
    Read file with line numbers and pagination.

    Args:
        file_path: Path to file
        offset: Starting line number (0-indexed)
        limit: Max lines to read (default: 2000)

    Returns:
        Line-numbered content (cat -n style)

    Rules:
        - Line numbers start at 1
        - Lines >5000 chars split with continuation markers (5.1, 5.2)
        - Use offset/limit for large files
        - Returns: "     1  content\n     2  more content"
    """
    return "     1  def example():\n     2      return 'hello'"


async def write_file_tool(file_path: str, content: str) -> str:
    """
    Create NEW file (fails if exists).

    Args:
        file_path: Path to new file
        content: File content

    Returns:
        Success message or error if file exists

    Rules:
        - Creates NEW files only
        - Fails if file already exists
        - Use edit_file for existing files
        - All parent directories must exist
    """
    return f"File {file_path} created successfully"


async def edit_file_tool(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False
) -> str:
    """
    Replace text in existing file.

    Args:
        file_path: Path to file
        old_string: Text to replace (must be unique unless replace_all=True)
        new_string: Replacement text
        replace_all: Replace all occurrences (default: False)

    Returns:
        Success message with replacement count

    Rules:
        - old_string must be UNIQUE in file (unless replace_all=True)
        - Exact string matching (whitespace matters)
        - Fails if old_string not found
        - Fails if old_string appears multiple times and replace_all=False
    """
    if replace_all:
        return f"Replaced all occurrences in {file_path}"
    return f"Replaced 1 occurrence in {file_path}"


async def glob_tool(pattern: str, path: str = "/") -> str:
    """
    Find files by glob pattern.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "src/**/*.ts")
        path: Base directory (default: "/")

    Returns:
        Newline-separated list of matching file paths

    Rules:
        - Supports ** for recursive matching
        - Returns sorted paths
        - Large results (>20K tokens) saved to /large_tool_results/
    """
    return "/project/src/main.py\n/project/tests/test_main.py"


async def grep_tool(
    pattern: str,
    path: str = "/",
    glob: str | None = None
) -> str:
    """
    Search files by LITERAL text (NOT regex).

    Args:
        pattern: Literal text to search for
        path: Base directory (default: "/")
        glob: Optional file pattern filter (e.g., "*.py")

    Returns:
        Matches with file:line:content format

    Rules:
        - LITERAL text matching (NOT regex)
        - Case-sensitive
        - Use glob parameter to filter file types
        - Large results (>20K tokens) saved to /large_tool_results/
        - Format: "/path/file.py:42:    matching line content"
    """
    return "/project/src/main.py:10:def example():\n/project/src/main.py:15:    return example()"


async def execute_tool(command: str) -> str:
    """
    Run shell command (only if SandboxBackendProtocol).

    Args:
        command: Shell command to execute

    Returns:
        Command output (stdout + stderr)

    Availability:
        - LocalShellBackend: YES
        - BaseSandbox subclasses: YES
        - FilesystemBackend (virtual_mode=False): YES
        - StateBackend: NO
        - StoreBackend: NO
        - FilesystemBackend (virtual_mode=True): NO

    Rules:
        - Executes in sandbox environment
        - Returns combined stdout/stderr
        - Exit code available in metadata
    """
    return "Command executed successfully\nHello from shell"


# ==== TodoList Tools ====


async def write_todos_tool(todos: list[dict]) -> str:
    """
    Plan tasks with status tracking.

    Args:
        todos: List of todo items
            Each item: {"id": int, "task": str, "status": "todo"|"in_progress"|"done"}

    Returns:
        Confirmation message

    Rules:
        - Overwrites existing todo list
        - IDs must be unique
        - Status must be: "todo", "in_progress", or "done"
        - Use for planning and tracking work

    Example:
        [
            {"id": 1, "task": "Read config file", "status": "todo"},
            {"id": 2, "task": "Parse JSON", "status": "in_progress"},
            {"id": 3, "task": "Write output", "status": "done"}
        ]
    """
    return f"Updated todo list with {len(todos)} items"


async def read_todos_tool() -> str:
    """
    Get current task list.

    Returns:
        JSON array of todo items with id, task, status

    Rules:
        - Returns empty array if no todos
        - Items ordered by ID
    """
    return '[{"id": 1, "task": "Example task", "status": "todo"}]'


# ==== Path Validation Rules ====


def validate_path(path: str) -> bool:
    """
    Path validation rules enforced by filesystem middleware.

    Blocked patterns:
        - ".." anywhere in path (directory traversal)
        - Windows absolute paths (C:\\, D:\\)
        - Paths not starting with / (after normalization)

    Normalization:
        - All paths converted to start with /
        - Relative paths prefixed with /
        - Example: "src/main.py" → "/src/main.py"
    """
    if ".." in path:
        return False
    if len(path) > 1 and path[1] == ":":  # Windows absolute path
        return False
    return True


# ==== Large Result Handling ====


async def handle_large_result_example():
    """
    Demonstrates large result eviction pattern.

    When tool output exceeds 20,000 tokens:
    1. Result saved to /large_tool_results/{tool_name}_{timestamp}.txt
    2. Tool output replaced with reference message
    3. Agent must use read_file to access saved result

    Example flow:
        1. Agent calls: grep("import", path="/", glob="**/*.py")
        2. Result is 50K tokens (too large)
        3. Middleware saves to: /large_tool_results/grep_20240110_123456.txt
        4. Agent receives: "Result too large (50000 tokens). Saved to /large_tool_results/grep_20240110_123456.txt"
        5. Agent calls: read_file("/large_tool_results/grep_20240110_123456.txt", offset=0, limit=100)

    Benefits:
        - Prevents context window overflow
        - Allows agent to paginate through large results
        - Preserves all data for inspection
    """
    print("Large Result Handling Pattern:")
    print("  1. Tool returns >20K tokens")
    print("  2. Result saved to /large_tool_results/")
    print("  3. Agent receives reference message")
    print("  4. Agent uses read_file with offset/limit for pagination")


# ==== Backend Tool Availability ====


def show_backend_tool_availability():
    """
    Tool availability matrix by backend type.

    StateBackend (in-memory):
        - ls: YES
        - read_file: YES
        - write_file: YES
        - edit_file: YES
        - glob: YES
        - grep: YES
        - execute: NO (no SandboxBackendProtocol)

    StoreBackend (persistent):
        - ls: YES
        - read_file: YES
        - write_file: YES
        - edit_file: YES
        - glob: YES
        - grep: YES
        - execute: NO (no SandboxBackendProtocol)

    LocalShellBackend:
        - ls: YES
        - read_file: YES
        - write_file: YES
        - edit_file: YES
        - glob: YES
        - grep: YES
        - execute: YES (implements SandboxBackendProtocol)

    FilesystemBackend (virtual_mode=True):
        - ls: YES
        - read_file: YES
        - write_file: YES
        - edit_file: YES
        - glob: YES
        - grep: YES
        - execute: NO

    FilesystemBackend (virtual_mode=False):
        - ls: YES
        - read_file: YES
        - write_file: YES
        - edit_file: YES
        - glob: YES
        - grep: YES
        - execute: YES (delegates to LocalShellBackend)
    """
    pass


# ==== Main Demonstration ====


async def main():
    print("=" * 70)
    print("DeepAgents Built-in Filesystem and TodoList Tools")
    print("=" * 70)

    print("\n# Filesystem Tools (7 total)\n")

    print("1. ls(path: str = '/')")
    print("   - List directory contents with metadata")
    print("   - Returns JSON with name, type, size, permissions")
    result = await ls_tool("/")
    print(f"   Example: {result}\n")

    print("2. read_file(file_path: str, offset: int = 0, limit: int = 2000)")
    print("   - Read file with line numbers (cat -n style)")
    print("   - Lines >5000 chars split with continuation markers")
    print("   - Use offset/limit for pagination")
    result = await read_file_tool("/example.py")
    print(f"   Example:\n{result}\n")

    print("3. write_file(file_path: str, content: str)")
    print("   - Create NEW file only (fails if exists)")
    print("   - Use edit_file for existing files")
    result = await write_file_tool("/new.txt", "content")
    print(f"   Example: {result}\n")

    print("4. edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False)")
    print("   - Replace text in existing file")
    print("   - old_string must be UNIQUE (unless replace_all=True)")
    print("   - Exact string matching (whitespace matters)")
    result = await edit_file_tool("/file.txt", "old", "new")
    print(f"   Example: {result}\n")

    print("5. glob(pattern: str, path: str = '/')")
    print("   - Find files by glob pattern")
    print("   - Supports ** for recursive matching")
    print("   - Large results saved to /large_tool_results/")
    result = await glob_tool("**/*.py")
    print(f"   Example: {result}\n")

    print("6. grep(pattern: str, path: str = '/', glob: str | None = None)")
    print("   - Search by LITERAL text (NOT regex)")
    print("   - Case-sensitive")
    print("   - Use glob parameter to filter files")
    print("   - Large results saved to /large_tool_results/")
    result = await grep_tool("def example", glob="*.py")
    print(f"   Example: {result}\n")

    print("7. execute(command: str)")
    print("   - Run shell command (only if SandboxBackendProtocol)")
    print("   - Available with: LocalShellBackend, BaseSandbox, FilesystemBackend(virtual_mode=False)")
    print("   - NOT available with: StateBackend, StoreBackend")
    result = await execute_tool("echo hello")
    print(f"   Example: {result}\n")

    print("\n# TodoList Tools (2 total)\n")

    print("8. write_todos(todos: list[dict])")
    print("   - Plan tasks with status tracking")
    print("   - Item format: {id: int, task: str, status: 'todo'|'in_progress'|'done'}")
    print("   - Overwrites existing list")
    todos = [
        {"id": 1, "task": "Read config", "status": "todo"},
        {"id": 2, "task": "Parse data", "status": "in_progress"}
    ]
    result = await write_todos_tool(todos)
    print(f"   Example: {result}\n")

    print("9. read_todos()")
    print("   - Get current task list")
    print("   - Returns JSON array ordered by ID")
    result = await read_todos_tool()
    print(f"   Example: {result}\n")

    print("\n# Path Validation Rules\n")
    print("Blocked patterns:")
    print("  - '..' anywhere (directory traversal)")
    print("  - Windows absolute paths (C:\\, D:\\)")
    print("\nNormalization:")
    print("  - All paths start with /")
    print("  - 'src/main.py' → '/src/main.py'")
    print(f"\nValid path '/src/file.txt': {validate_path('/src/file.txt')}")
    print(f"Invalid path '../etc/passwd': {validate_path('../etc/passwd')}")

    print("\n# Large Result Handling (>20K tokens)\n")
    await handle_large_result_example()

    print("\n# Backend Tool Availability\n")
    print("StateBackend / StoreBackend:")
    print("  - All filesystem tools: YES")
    print("  - execute: NO\n")

    print("LocalShellBackend:")
    print("  - All filesystem tools: YES")
    print("  - execute: YES\n")

    print("FilesystemBackend (virtual_mode=True):")
    print("  - All filesystem tools: YES")
    print("  - execute: NO\n")

    print("FilesystemBackend (virtual_mode=False):")
    print("  - All filesystem tools: YES")
    print("  - execute: YES (delegates to LocalShellBackend)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
