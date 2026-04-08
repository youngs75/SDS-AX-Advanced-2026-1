"""에이전트에게 파일시스템 도구를 제공하는 미들웨어 모듈.

이 모듈은 Deep Agents 프레임워크에서 가장 크고 핵심적인 미들웨어로,
에이전트가 파일시스템과 상호작용할 수 있는 7개의 도구를 제공합니다.

## 제공 도구 (7개)

| 도구 | 기능 | 백엔드 요구사항 |
|------|------|----------------|
| `ls` | 디렉토리 파일 목록 조회 | BackendProtocol |
| `read_file` | 파일 읽기 (페이지네이션, 이미지 지원) | BackendProtocol |
| `write_file` | 새 파일 생성/작성 | BackendProtocol |
| `edit_file` | 기존 파일의 문자열 교체 편집 | BackendProtocol |
| `glob` | 패턴 기반 파일 검색 | BackendProtocol |
| `grep` | 파일 내 텍스트 검색 | BackendProtocol |
| `execute` | 샌드박스 셸 명령 실행 | SandboxBackendProtocol |

## 핵심 아키텍처

### 백엔드 추상화
- `BackendProtocol`: 파일 읽기/쓰기/편집/검색의 기본 인터페이스
- `SandboxBackendProtocol`: 셸 명령 실행을 추가로 지원하는 확장 인터페이스
- `CompositeBackend`: 경로별로 다른 백엔드를 라우팅 (예: /memories/는 StoreBackend, 나머지는 StateBackend)
- 백엔드가 실행을 지원하지 않으면, `execute` 도구는 동적으로 필터링됩니다.

### 대용량 결과 퇴거(Eviction) 메커니즘
컨텍스트 윈도우 포화를 방지하기 위한 두 가지 자동 메커니즘:

1. **도구 결과 퇴거** (`wrap_tool_call`):
   도구 실행 결과가 토큰 제한(`tool_token_limit_before_evict`)을 초과하면,
   전체 내용을 백엔드의 `/large_tool_results/`에 저장하고
   미리보기(head+tail)로 대체합니다.

2. **HumanMessage 퇴거** (`wrap_model_call`):
   사용자 메시지가 토큰 제한(`human_message_token_limit_before_evict`)을 초과하면,
   백엔드의 `/conversation_history/`에 저장하고 미리보기로 대체합니다.

### 동기/비동기 이중 구현
모든 도구와 미들웨어 메서드가 동기(`sync_*`) + 비동기(`async_*`) 쌍으로 구현되어
LangGraph의 양쪽 실행 컨텍스트를 모두 지원합니다.

## 사용 예시

```python
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.backends import StateBackend, CompositeBackend, StoreBackend

# 기본: 임시(ephemeral) 스토리지만 사용 (실행 미지원)
agent = create_agent(middleware=[FilesystemMiddleware()])

# 하이브리드: 임시 + 영구 스토리지
backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

# 샌드박스: 셸 명령 실행 지원
from my_sandbox import DockerSandboxBackend
sandbox = DockerSandboxBackend(container_id="my-container")
agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
```
"""
# ruff: noqa: E501

import asyncio
import concurrent.futures
import contextvars
import mimetypes
import uuid
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, cast

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.content import ContentBlock
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from deepagents.backends import StateBackend
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    BACKEND_TYPES as BACKEND_TYPES,  # Re-export type here for backwards compatibility
    BackendProtocol,
    EditResult,
    FileData as FileData,  # Re-export for backwards compatibility
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    execute_accepts_timeout,
)
from deepagents.backends.utils import (
    _get_file_type,
    check_empty_content,
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
    validate_path,
)
from deepagents.middleware._utils import append_to_system_message

# === 상수 정의 ===
EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"  # 빈 파일 경고 메시지
GLOB_TIMEOUT = 20.0  # glob 검색 타임아웃 (초) — 너무 넓은 패턴의 무한 실행 방지
LINE_NUMBER_WIDTH = 6  # 줄 번호 표시 너비
DEFAULT_READ_OFFSET = 0  # read_file 기본 시작 줄 (0-indexed)
DEFAULT_READ_LIMIT = 100  # read_file 기본 최대 줄 수
# Template for truncation message in read_file
# {file_path} will be filled in at runtime
READ_FILE_TRUNCATION_MSG = (
    "\n\n[Output was truncated due to size limits. "
    "The file content is very large. "
    "Consider reformatting the file to make it easier to navigate. "
    "For example, if this is JSON, use execute(command='jq . {file_path}') to pretty-print it with line breaks. "
    "For other formats, you can use appropriate formatting tools to split long lines.]"
)

# 토큰당 대략적인 문자 수 — 절삭(truncation) 계산에 사용
# 실제 비율은 콘텐츠에 따라 다르지만, 보수적으로 4자/토큰을 사용합니다.
# 높은 쪽으로 오차를 두어 맞을 수 있는 콘텐츠가 조기에 퇴거되는 것을 방지합니다.
NUM_CHARS_PER_TOKEN = 4


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """삭제를 지원하는 파일 데이터 병합 리듀서.

    LangGraph의 상태 관리에서 Annotated 리듀서로 사용됩니다.
    right 딕셔너리에서 None 값을 삭제 마커로 처리하여 파일 삭제를 지원합니다.

    동작 원리:
    - right에 있는 키가 left에도 있으면: right 값으로 덮어쓰기
    - right의 값이 None이면: left에서 해당 키를 삭제
    - right에만 있는 키: 새로 추가

    Args:
        left: 기존 파일 딕셔너리. 초기화 시 None일 수 있음.
        right: 병합할 새 파일 딕셔너리. None 값은 삭제 마커로 처리.

    Returns:
        병합된 딕셔너리. right가 left를 덮어쓰고, None 값은 삭제를 트리거.

    사용 예시:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # 결과: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        # /file2.txt는 None으로 삭제됨
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


class FilesystemState(AgentState):
    """파일시스템 미들웨어의 상태 스키마.

    에이전트 상태를 확장하여 파일 데이터를 저장하는 필드를 추가합니다.
    StateBackend 사용 시, 파일 내용이 이 상태 딕셔너리에 직접 저장됩니다.
    """

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """파일시스템에 저장된 파일 데이터. _file_data_reducer를 통해 병합/삭제 처리."""


class LsSchema(BaseModel):
    """Input schema for the `ls` tool."""

    path: str = Field(description="Absolute path to the directory to list. Must be absolute, not relative.")


class ReadFileSchema(BaseModel):
    """Input schema for the `read_file` tool."""

    file_path: str = Field(description="Absolute path to the file to read. Must be absolute, not relative.")
    offset: int = Field(
        default=DEFAULT_READ_OFFSET,
        description="Line number to start reading from (0-indexed). Use for pagination of large files.",
    )
    limit: int = Field(
        default=DEFAULT_READ_LIMIT,
        description="Maximum number of lines to read. Use for pagination of large files.",
    )


class WriteFileSchema(BaseModel):
    """Input schema for the `write_file` tool."""

    file_path: str = Field(description="Absolute path where the file should be created. Must be absolute, not relative.")
    content: str = Field(description="The text content to write to the file. This parameter is required.")


class EditFileSchema(BaseModel):
    """Input schema for the `edit_file` tool."""

    file_path: str = Field(description="Absolute path to the file to edit. Must be absolute, not relative.")
    old_string: str = Field(description="The exact text to find and replace. Must be unique in the file unless replace_all is True.")
    new_string: str = Field(description="The text to replace old_string with. Must be different from old_string.")
    replace_all: bool = Field(
        default=False,
        description="If True, replace all occurrences of old_string. If False (default), old_string must be unique.",
    )


class GlobSchema(BaseModel):
    """Input schema for the `glob` tool."""

    pattern: str = Field(description="Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md').")
    path: str = Field(default="/", description="Base directory to search from. Defaults to root '/'.")


class GrepSchema(BaseModel):
    """Input schema for the `grep` tool."""

    pattern: str = Field(description="Text pattern to search for (literal string, not regex).")
    path: str | None = Field(default=None, description="Directory to search in. Defaults to current working directory.")
    glob: str | None = Field(default=None, description="Glob pattern to filter which files to search (e.g., '*.py').")
    output_mode: Literal["files_with_matches", "content", "count"] = Field(
        default="files_with_matches",
        description="Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
    )


class ExecuteSchema(BaseModel):
    """Input schema for the `execute` tool."""

    command: str = Field(description="Shell command to execute in the sandbox environment.")
    timeout: int | None = Field(
        default=None,
        description="Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
    )


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in a directory.

This is useful for exploring the filesystem and finding the right file to read or edit.
You should almost ALWAYS use this tool before using the read_file or edit_file tools."""

READ_FILE_TOOL_DESCRIPTION = """Reads a file from the filesystem.

Assume this tool is able to read all files. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- By default, it reads up to 100 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Results are returned using cat -n format, with line numbers starting at 1
- Lines longer than 5,000 characters will be split into multiple lines with continuation markers (e.g., 5.1, 5.2, etc.). When you specify a limit, these continuation lines count towards the limit.
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- Image files (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`) are returned as multimodal image content blocks (see https://docs.langchain.com/oss/python/langchain/messages#multimodal).

For image tasks:
- Use `read_file(file_path=...)` for `.png/.jpg/.jpeg/.gif/.webp`
- Do NOT use `offset`/`limit` for images (pagination is text-only)
- If image details were compacted from history, call `read_file` again on the same path

- You should ALWAYS make sure a file has been read before editing it."""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must read the file before editing. This tool will error if you attempt an edit without reading the file first.
- When editing, preserve the exact indentation (tabs/spaces) from the read output. Never include line number prefixes in old_string or new_string.
- ALWAYS prefer editing existing files over creating new ones.
- Only use emojis if the user explicitly requests it."""


WRITE_FILE_TOOL_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The write_file tool will create the a new file.
- Prefer to edit existing files (with the edit_file tool) over creating new ones when possible.
"""

GLOB_TOOL_DESCRIPTION = """Find files matching a glob pattern.

Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character).
Returns a list of absolute file paths that match the pattern.

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir"""

GREP_TOOL_DESCRIPTION = """Search for a text pattern across files.

Searches for literal text (not regex) and returns matching files or content based on output_mode.
Special characters like parentheses, brackets, pipes, etc. are treated as literal characters, not regex operators.

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`
- Search for code with special chars: `grep(pattern="def __init__(self):")`"""

EXECUTE_TOOL_DESCRIPTION = """Executes a shell command in an isolated sandbox environment.

Usage:
Executes a given command in the sandbox environment with proper handling and security measures.
Before executing the command, please follow these steps:
1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists and is the intended parent directory
2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)
     - python /path/with spaces/script.py (incorrect - will fail)
   - After ensuring proper quoting, execute the command
   - Capture the output of the command
Usage notes:
  - Commands run in an isolated sandbox environment
  - Returns combined stdout/stderr output with exit code
  - If the output is very large, it may be truncated
  - For long-running commands, use the optional timeout parameter to override the default timeout (e.g., execute(command="make build", timeout=300))
  - A timeout of 0 may disable timeouts on backends that support no-timeout execution
  - VERY IMPORTANT: You MUST avoid using search commands like find and grep. Instead use the grep, glob tools to search. You MUST avoid read tools like cat, head, tail, and use read_file to read files.
  - When issuing multiple commands, use the ';' or '&&' operator to separate them. DO NOT use newlines (newlines are ok in quoted strings)
    - Use '&&' when commands depend on each other (e.g., "mkdir dir && cd dir")
    - Use ';' only when you need to run commands sequentially but don't care if earlier commands fail
  - Try to maintain your current working directory throughout the session by using absolute paths and avoiding usage of cd

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm install && npm test")
    - execute(command="make build", timeout=300)

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

FILESYSTEM_SYSTEM_PROMPT = """## Following Conventions

- Read files before editing — understand existing content before making changes
- Mimic existing style, naming conventions, and patterns

## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /. Follow the tool docs for the available tools, and use pagination (offset/limit) when reading large files.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files

## Large Tool Results

When a tool result is too large, it may be offloaded into the filesystem instead of being returned inline. In those cases, use `read_file` to inspect the saved result in chunks, or use `grep` within `/large_tool_results/` if you need to search across offloaded tool results and do not know the exact file path. Offloaded tool results are stored under `/large_tool_results/<tool_call_id>`."""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


def _supports_execution(backend: BackendProtocol) -> bool:
    """백엔드가 셸 명령 실행을 ���원하는지 확인합니다.

    CompositeBackend의 경우 기본(default) 백엔드가 실행을 지원하는지 확인합니다.
    다른 백엔드의 경�� SandboxBackendProtocol을 구현하는지 확인합니다.

    이 함수의 결과에 따라 wrap_model_call에서 execute 도구가
    동적으로 필터링(제거)됩니다.

    Args:
        backend: 확인할 백엔드.

    Returns:
        실행을 지원하면 True, 아니면 False.
    """
    # For CompositeBackend, check the default backend
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # For other backends, use isinstance check
    return isinstance(backend, SandboxBackendProtocol)


# 대용량 결과 퇴거(eviction) 로직에서 제외되는 도구 목록.
#
# 토큰 제한 초과 시 결과를 파일시스템에 퇴거하지 않는 도구들입니다.
# 도구별 제외 이유가 다릅니다:
#
# 1. 자체 절삭(truncation)이 내장된 도구 (ls, glob, grep):
#    출력이 너무 크면 스스로 절삭합니다. 많은 매치로 절삭된 출력은
#    쿼리를 더 좁혀야 한다는 신호이므로, 전체 결과 보존보다는
#    검색 기준 개선을 유도하는 것이 적절합니다.
#
# 2. 절삭이 문제가 되는 도구 (read_file):
#    단일 긴 줄(예: 매우 큰 페이로드의 JSONL 파일)이 실패 모드입니다.
#    결과를 절삭하면 에이전트가 절삭된 파일을 다시 read_file로 읽으려
#    시도하지만, 같은 결과가 반복되어 도움이 되지 않습니다.
#
# 3. 제한을 초과하지 않는 도구 (edit_file, write_file):
#    최소한의 확인 메시지만 반환하므로 토큰 제한을 초과할 일이 없어
#    검사 자체가 불필요합니다.
TOOLS_EXCLUDED_FROM_EVICTION = (
    "ls",
    "glob",
    "grep",
    "read_file",
    "edit_file",
    "write_file",
)


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""

TOO_LARGE_HUMAN_MSG = """Message content too large and was saved to the filesystem at: {file_path}

You can read the full content using the read_file tool with pagination (offset and limit parameters).

Here is a preview showing the head and tail of the content:

{content_sample}
"""


def _build_evicted_human_content(
    message: HumanMessage,
    replacement_text: str,
) -> str | list[ContentBlock]:
    """퇴거된 HumanMessage의 대체 콘텐츠를 생성���니다 (비텍스트 블록 보존).

    멀티모달 메시지(텍스트 + 이미지 등)의 경우, 모든 텍스트 블록을
    대체 텍스트로 교체하되 이미지 등 비텍스트 블록은 그대로 유지합니다.

    Args:
        message: 퇴거되는 원본 HumanMessage.
        replacement_text: 절삭 알림과 미리보기 텍스트.

    Returns:
        대체 콘텐츠: 순수 텍스트이면 문자열, 혼합 블록이면 ContentBlock 리스트.
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]


def _build_truncated_human_message(message: HumanMessage, file_path: str) -> HumanMessage:
    """Build a truncated HumanMessage for the model request.

    Computes a preview from the full content still in state and returns a
    lightweight replacement the model will see. Pure string computation — no
    backend I/O.

    Args:
        message: The original HumanMessage (full content in state).
        file_path: The backend path where the content was evicted.

    Returns:
        A new HumanMessage with truncated content and the same `id`.
    """
    content_str = _extract_text_from_message(message)
    content_sample = _create_content_preview(content_str)
    replacement_text = TOO_LARGE_HUMAN_MSG.format(
        file_path=file_path,
        content_sample=content_sample,
    )
    evicted = _build_evicted_human_content(message, replacement_text)
    return message.model_copy(update={"content": evicted})


def _create_content_preview(content_str: str, *, head_lines: int = 5, tail_lines: int = 5) -> str:
    """콘텐츠의 head(앞부분)와 tail(뒷부분)을 보여주는 미리보기를 생성합니다.

    퇴거된 대용량 콘텐츠의 구조를 LLM이 파악할 수 있도록,
    처음 N줄과 마지막 N줄을 줄 번호와 함께 보여주고
    중간에 절삭 표시를 삽입합니다.

    Args:
        content_str: 미리보기를 생성할 전체 콘텐츠 문자열.
        head_lines: 시작 부분에서 표시할 줄 수 (기본 5).
        tail_lines: 끝 부분에서 표시할 줄 수 (기본 5).

    Returns:
        줄 번호가 포함된 포맷된 미리보기 문자열.
    """
    lines = content_str.splitlines()

    if len(lines) <= head_lines + tail_lines:
        # If file is small enough, show all lines
        preview_lines = [line[:1000] for line in lines]
        return format_content_with_line_numbers(preview_lines, start_line=1)

    # Show head and tail with truncation marker
    head = [line[:1000] for line in lines[:head_lines]]
    tail = [line[:1000] for line in lines[-tail_lines:]]

    head_sample = format_content_with_line_numbers(head, start_line=1)
    truncation_notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail_sample = format_content_with_line_numbers(tail, start_line=len(lines) - tail_lines + 1)

    return head_sample + truncation_notice + tail_sample


def _extract_text_from_message(message: BaseMessage) -> str:
    """메시지에서 텍스트만 추출합니다.

    `content_blocks` 속성을 사용하여 모든 텍스트 콘텐츠 블록을 결합하고,
    비텍스트 블록(이미지, 오디오 등)은 무시합니다.
    바이너리 페이로드가 크기 측정을 부풀리는 것을 방지합니다.

    Args:
        message: 텍스트를 추출할 BaseMessage.

    Returns:
        모든 텍스트 콘텐츠 블록에서 결합된 텍스트 문자열.
    """
    texts = [block["text"] for block in message.content_blocks if block["type"] == "text"]
    return "\n".join(texts)


def _build_evicted_content(message: ToolMessage, replacement_text: str) -> str | list[ContentBlock]:
    """퇴거된 ToolMessage의 대체 콘텐츠를 생성합니다 (비텍스트 블록 보존).

    순수 문자열 콘텐츠면 대체 텍스트를 직접 반환합니다.
    혼합 블록(텍스트 + 이미지 등)이면 모든 텍스트 블록을 대체 텍스트로 교체하되
    비텍스트 블록(이미지 등)은 그대로 유지하여 멀티모달 컨텍스트를 보존합니다.

    Args:
        message: 퇴거되는 원본 ToolMessage.
        replacement_text: 절삭 알림과 미리보기 텍스트.

    Returns:
        대체 콘텐츠: 순수 텍스트이면 문자열, 혼합 블록이면 ContentBlock 리스트.
    """
    if isinstance(message.content, str):
        return replacement_text
    media_blocks = [block for block in message.content_blocks if block["type"] != "text"]
    if not media_blocks:
        # All content is text, so a plain string replacement is sufficient.
        return replacement_text
    return [cast("ContentBlock", {"type": "text", "text": replacement_text}), *media_blocks]


class FilesystemMiddleware(AgentMiddleware[FilesystemState, ContextT, ResponseT]):
    """에이전트에게 파일시스템 도구와 선택적 실행(execute) 도구를 제공하는 미들웨어.

    이 미들웨어는 에이전트에 파일시스템 도구를 추가합니다: `ls`, `read_file`,
    `write_file`, `edit_file`, `glob`, `grep`.

    `BackendProtocol`��� 구현하는 모든 백엔드를 사용하여 파일을 저장할 수 있습니다.
    백엔드가 `SandboxBackendProtocol`을 구현하면, 셸 명령 실행을 위한
    `execute` 도구도 추가됩니다.

    또한 대용량 도구 결과가 토큰 임계값을 초과하면 자동으로 파일시스���에 퇴거(evict)하여
    컨텍스트 윈도우 포화를 방지합니다.

    미들웨어 생명주기:
        1. **__init__**: 7개 도구 생성 (execute는 백엔드 미지원 시 런타임에 필터링)
        2. **wrap_model_call** (매 LLM 호출): 시스템 프롬프트 주입 + execute 도구 필터링
           + 대용량 HumanMessage 퇴거
        3. **wrap_tool_call** (매 도구 실행 후): 대용량 도구 결과 퇴거

    Args:
        backend: 파일 저장 및 선택적 실행을 위한 백엔드.
            미제공 시 `StateBackend`(에이전트 상태에 임시 저장)가 기본값입니다.
            영구 저장 또는 하이브리드 설정에는 `CompositeBackend`를 사용합니다.
            실행 지원에��� `SandboxBackendProtocol`을 구현하는 백엔드를 사용합니다.
        system_prompt: 선택적 커스텀 시스템 프롬프트 오버라이드.
        custom_tool_descriptions: 선택적 도구 설명 오버���이드.
        tool_token_limit_before_evict: 도구 결과를 파일시스템에 퇴거하기 전 토큰 제한.
            초과 시, 설정된 백엔드를 사용하여 결과를 저장하고
            절삭된 미리보기와 파일 참조로 대체합니다.

    사용 예시:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend

        # 임시(ephemeral) 스토리지만 사용 (기본값, 실행 미지원)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # 하이브리드 스토리지 (임시 + /memories/ 영구)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # 샌드박스 백엔드 (셸 실행 지원)
        from my_sandbox import DockerSandboxBackend
        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
        human_message_token_limit_before_evict: int | None = 50000,
        max_execute_timeout: int = 3600,
    ) -> None:
        """파일시스템 미들웨어를 초기화합니다.

        7개 도구(ls, read_file, write_file, edit_file, glob, grep, execute)를 생성하고,
        퇴거(eviction) 임계값과 실행 타임아웃 제한을 설정합니다.

        Args:
            backend: 파일 저장 및 선택적 실행을 위한 백엔드 또는 팩토리 callable.
                미제공 시 StateBackend(임시 저장)가 기본값.
            system_prompt: 선택적 커스텀 시스템 프롬프트 오버라이드.
            custom_tool_descriptions: 도구별 설명 오버라이드 딕셔너리 (키: 도구 이름).
            tool_token_limit_before_evict: 도구 결과 퇴거 전 토큰 제한 (기본 20000).
                None이면 퇴거 비활성화.
            human_message_token_limit_before_evict: HumanMessage 퇴거 전 토큰 제한 (기본 50000).
                None이면 퇴거 비활성화.
            max_execute_timeout: execute 도구의 명령별 타임아웃 최대 허용값 (초).
                기본 3600초(1시간). 이 값을 초과하는 타임아웃은 오류로 거부됩니다.

        Raises:
            ValueError: `max_execute_timeout`이 양수가 아닌 경우.
        """
        if max_execute_timeout <= 0:
            msg = f"max_execute_timeout must be positive, got {max_execute_timeout}"
            raise ValueError(msg)
        # Use provided backend or default to StateBackend instance
        self.backend = backend if backend is not None else StateBackend()

        # Store configuration (private - internal implementation details)
        self._custom_system_prompt = system_prompt
        self._custom_tool_descriptions = custom_tool_descriptions or {}
        self._tool_token_limit_before_evict = tool_token_limit_before_evict
        self._human_message_token_limit_before_evict = human_message_token_limit_before_evict
        self._max_execute_timeout = max_execute_timeout

        self.tools = [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
            self._create_glob_tool(),
            self._create_grep_tool(),
            self._create_execute_tool(),
        ]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            warnings.warn(
                "Passing a callable (factory) as `backend` is deprecated and "
                "will be removed in v0.7. Pass a `BackendProtocol` instance "
                "directly instead (e.g. `StateBackend()`).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.backend(runtime)  # ty: ignore[call-top-callable]
        return self.backend

    def _create_ls_tool(self) -> BaseTool:
        """Create the ls (list files) tool."""
        tool_description = self._custom_tool_descriptions.get("ls") or LIST_FILES_TOOL_DESCRIPTION

        def sync_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """Synchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ls_result = resolved_backend.ls(validated_path)
            if ls_result.error:
                return f"Error: {ls_result.error}"
            infos = ls_result.entries or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_ls(
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Absolute path to the directory to list. Must be absolute, not relative."],
        ) -> str:
            """Asynchronous wrapper for ls tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ls_result = await resolved_backend.als(validated_path)
            if ls_result.error:
                return f"Error: {ls_result.error}"
            infos = ls_result.entries or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="ls",
            description=tool_description,
            func=sync_ls,
            coroutine=async_ls,
            infer_schema=False,
            args_schema=LsSchema,
        )

    def _create_read_file_tool(self) -> BaseTool:  # noqa: C901
        """Create the read_file tool."""
        tool_description = self._custom_tool_descriptions.get("read_file") or READ_FILE_TOOL_DESCRIPTION
        token_limit = self._tool_token_limit_before_evict

        def _truncate(content: str, file_path: str, limit: int) -> str:
            lines = content.splitlines(keepends=True)
            if len(lines) > limit:
                lines = lines[:limit]
                content = "".join(lines)

            if token_limit and len(content) >= NUM_CHARS_PER_TOKEN * token_limit:
                truncation_msg = READ_FILE_TRUNCATION_MSG.format(file_path=file_path)
                max_content_length = NUM_CHARS_PER_TOKEN * token_limit - len(truncation_msg)
                content = content[:max_content_length] + truncation_msg

            return content

        def _handle_read_result(
            read_result: ReadResult | str,
            validated_path: str,
            tool_call_id: str | None,
            offset: int,
            limit: int,
        ) -> ToolMessage | str:
            if isinstance(read_result, str):
                warnings.warn(
                    "Returning a plain `str` from `backend.read()` is deprecated. "
                    "Return a `ReadResult` instead. Returning `str` will not be "
                    "supported in v0.7.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Legacy backends already format with line numbers
                return _truncate(read_result, validated_path, limit)

            if read_result.error:
                return f"Error: {read_result.error}"

            if read_result.file_data is None:
                return f"Error: no data returned for '{validated_path}'"

            file_type = _get_file_type(validated_path)
            content = read_result.file_data["content"]

            if file_type != "text":
                mime_type = mimetypes.guess_type("file" + Path(validated_path).suffix)[0] or "application/octet-stream"
                return ToolMessage(
                    content_blocks=cast("list[ContentBlock]", [{"type": file_type, "base64": content, "mime_type": mime_type}]),
                    name="read_file",
                    tool_call_id=tool_call_id,
                    additional_kwargs={"read_file_path": validated_path, "read_file_media_type": mime_type},
                )

            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg

            content = format_content_with_line_numbers(content, start_line=offset + 1)
            # We apply truncation again after formatting content as continuation lines
            # can increase line count
            return _truncate(content, validated_path, limit)

        def sync_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | str:
            """Synchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            read_result = resolved_backend.read(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        async def async_read_file(
            file_path: Annotated[str, "Absolute path to the file to read. Must be absolute, not relative."],
            runtime: ToolRuntime[None, FilesystemState],
            offset: Annotated[int, "Line number to start reading from (0-indexed). Use for pagination of large files."] = DEFAULT_READ_OFFSET,
            limit: Annotated[int, "Maximum number of lines to read. Use for pagination of large files."] = DEFAULT_READ_LIMIT,
        ) -> ToolMessage | str:
            """Asynchronous wrapper for read_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"

            read_result = await resolved_backend.aread(validated_path, offset=offset, limit=limit)
            return _handle_read_result(read_result, validated_path, runtime.tool_call_id, offset, limit)

        return StructuredTool.from_function(
            name="read_file",
            description=tool_description,
            func=sync_read_file,
            coroutine=async_read_file,
            infer_schema=False,
            args_schema=ReadFileSchema,
        )

    def _create_write_file_tool(self) -> BaseTool:
        """Create the write_file tool."""
        tool_description = self._custom_tool_descriptions.get("write_file") or WRITE_FILE_TOOL_DESCRIPTION

        def sync_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Synchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: WriteResult = resolved_backend.write(validated_path, content)
            if res.error:
                return res.error
            return f"Updated file {res.path}"

        async def async_write_file(
            file_path: Annotated[str, "Absolute path where the file should be created. Must be absolute, not relative."],
            content: Annotated[str, "The text content to write to the file. This parameter is required."],
            runtime: ToolRuntime[None, FilesystemState],
        ) -> str:
            """Asynchronous wrapper for write_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: WriteResult = await resolved_backend.awrite(validated_path, content)
            if res.error:
                return res.error
            return f"Updated file {res.path}"

        return StructuredTool.from_function(
            name="write_file",
            description=tool_description,
            func=sync_write_file,
            coroutine=async_write_file,
            infer_schema=False,
            args_schema=WriteFileSchema,
        )

    def _create_edit_file_tool(self) -> BaseTool:
        """Create the edit_file tool."""
        tool_description = self._custom_tool_descriptions.get("edit_file") or EDIT_FILE_TOOL_DESCRIPTION

        def sync_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> str:
            """Synchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: EditResult = resolved_backend.edit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        async def async_edit_file(
            file_path: Annotated[str, "Absolute path to the file to edit. Must be absolute, not relative."],
            old_string: Annotated[str, "The exact text to find and replace. Must be unique in the file unless replace_all is True."],
            new_string: Annotated[str, "The text to replace old_string with. Must be different from old_string."],
            runtime: ToolRuntime[None, FilesystemState],
            *,
            replace_all: Annotated[bool, "If True, replace all occurrences of old_string. If False (default), old_string must be unique."] = False,
        ) -> str:
            """Asynchronous wrapper for edit_file tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(file_path)
            except ValueError as e:
                return f"Error: {e}"
            res: EditResult = await resolved_backend.aedit(validated_path, old_string, new_string, replace_all=replace_all)
            if res.error:
                return res.error
            return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

        return StructuredTool.from_function(
            name="edit_file",
            description=tool_description,
            func=sync_edit_file,
            coroutine=async_edit_file,
            infer_schema=False,
            args_schema=EditFileSchema,
        )

    def _create_glob_tool(self) -> BaseTool:
        """Create the glob tool."""
        tool_description = self._custom_tool_descriptions.get("glob") or GLOB_TOOL_DESCRIPTION

        def sync_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """Synchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: ctx.run(resolved_backend.glob, pattern, path=validated_path))
                try:
                    glob_result = future.result(timeout=GLOB_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
            if glob_result.error:
                return f"Error: {glob_result.error}"
            infos = glob_result.matches or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        async def async_glob(
            pattern: Annotated[str, "Glob pattern to match files (e.g., '**/*.py', '*.txt', '/subdir/**/*.md')."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str, "Base directory to search from. Defaults to root '/'."] = "/",
        ) -> str:
            """Asynchronous wrapper for glob tool."""
            resolved_backend = self._get_backend(runtime)
            try:
                validated_path = validate_path(path)
            except ValueError as e:
                return f"Error: {e}"
            try:
                glob_result = await asyncio.wait_for(
                    resolved_backend.aglob(pattern, path=validated_path),
                    timeout=GLOB_TIMEOUT,
                )
            except TimeoutError:
                return f"Error: glob timed out after {GLOB_TIMEOUT}s. Try a more specific pattern or a narrower path."
            if glob_result.error:
                return f"Error: {glob_result.error}"
            infos = glob_result.matches or []
            paths = [fi.get("path", "") for fi in infos]
            result = truncate_if_too_long(paths)
            return str(result)

        return StructuredTool.from_function(
            name="glob",
            description=tool_description,
            func=sync_glob,
            coroutine=async_glob,
            infer_schema=False,
            args_schema=GlobSchema,
        )

    def _create_grep_tool(self) -> BaseTool:
        """Create the grep tool."""
        tool_description = self._custom_tool_descriptions.get("grep") or GREP_TOOL_DESCRIPTION

        def sync_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """Synchronous wrapper for grep tool."""
            resolved_backend = self._get_backend(runtime)
            grep_result = resolved_backend.grep(pattern, path=path, glob=glob)
            if grep_result.error:
                return grep_result.error
            matches = grep_result.matches or []
            formatted = format_grep_matches(matches, output_mode)
            return truncate_if_too_long(formatted)

        async def async_grep(
            pattern: Annotated[str, "Text pattern to search for (literal string, not regex)."],
            runtime: ToolRuntime[None, FilesystemState],
            path: Annotated[str | None, "Directory to search in. Defaults to current working directory."] = None,
            glob: Annotated[str | None, "Glob pattern to filter which files to search (e.g., '*.py')."] = None,
            output_mode: Annotated[
                Literal["files_with_matches", "content", "count"],
                "Output format: 'files_with_matches' (file paths only, default), 'content' (matching lines with context), 'count' (match counts per file).",
            ] = "files_with_matches",
        ) -> str:
            """Asynchronous wrapper for grep tool."""
            resolved_backend = self._get_backend(runtime)
            grep_result = await resolved_backend.agrep(pattern, path=path, glob=glob)
            if grep_result.error:
                return grep_result.error
            matches = grep_result.matches or []
            formatted = format_grep_matches(matches, output_mode)
            return truncate_if_too_long(formatted)

        return StructuredTool.from_function(
            name="grep",
            description=tool_description,
            func=sync_grep,
            coroutine=async_grep,
            infer_schema=False,
            args_schema=GrepSchema,
        )

    def _create_execute_tool(self) -> BaseTool:  # noqa: C901
        """Create the execute tool for sandbox command execution."""
        tool_description = self._custom_tool_descriptions.get("execute") or EXECUTE_TOOL_DESCRIPTION

        def sync_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
            timeout: Annotated[
                int | None,
                "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
            ] = None,
        ) -> str:
            """Synchronous wrapper for execute tool."""
            if timeout is not None:
                if timeout < 0:
                    return f"Error: timeout must be non-negative, got {timeout}."
                if timeout > self._max_execute_timeout:
                    return f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s)."

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            # Safe cast: _supports_execution validates that execute()/aexecute() exist
            # (either SandboxBackendProtocol or CompositeBackend with sandbox default)
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return (
                    "Error: This sandbox backend does not support per-command "
                    "timeout overrides. Update your sandbox package to the "
                    "latest version, or omit the timeout parameter."
                )
            try:
                result = executable.execute(command, timeout=timeout) if timeout is not None else executable.execute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        async def async_execute(  # noqa: PLR0911 - early returns for distinct error conditions
            command: Annotated[str, "Shell command to execute in the sandbox environment."],
            runtime: ToolRuntime[None, FilesystemState],
            # ASYNC109 - timeout is a semantic parameter forwarded to the
            # backend's implementation, not an asyncio.timeout() contract.
            timeout: Annotated[  # noqa: ASYNC109
                int | None,
                "Optional timeout in seconds for this command. Overrides the default timeout. Use 0 for no-timeout execution on backends that support it.",
            ] = None,
        ) -> str:
            """Asynchronous wrapper for execute tool."""
            if timeout is not None:
                if timeout < 0:
                    return f"Error: timeout must be non-negative, got {timeout}."
                if timeout > self._max_execute_timeout:
                    return f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_execute_timeout}s)."

            resolved_backend = self._get_backend(runtime)

            # Runtime check - fail gracefully if not supported
            if not _supports_execution(resolved_backend):
                return (
                    "Error: Execution not available. This agent's backend "
                    "does not support command execution (SandboxBackendProtocol). "
                    "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
                )

            # Safe cast: _supports_execution validates that execute()/aexecute() exist
            executable = cast("SandboxBackendProtocol", resolved_backend)
            if timeout is not None and not execute_accepts_timeout(type(executable)):
                return (
                    "Error: This sandbox backend does not support per-command "
                    "timeout overrides. Update your sandbox package to the "
                    "latest version, or omit the timeout parameter."
                )
            try:
                result = await executable.aexecute(command, timeout=timeout) if timeout is not None else await executable.aexecute(command)
            except NotImplementedError as e:
                # Handle case where execute() exists but raises NotImplementedError
                return f"Error: Execution not available. {e}"
            except ValueError as e:
                return f"Error: Invalid parameter. {e}"

            # Format output for LLM consumption
            parts = [result.output]

            if result.exit_code is not None:
                status = "succeeded" if result.exit_code == 0 else "failed"
                parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

            if result.truncated:
                parts.append("\n[Output was truncated due to size limits]")

            return "".join(parts)

        return StructuredTool.from_function(
            name="execute",
            description=tool_description,
            func=sync_execute,
            coroutine=async_execute,
            infer_schema=False,
            args_schema=ExecuteSchema,
        )

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """시스템 프롬프트 주입, 도구 필터링, 대용량 HumanMessage 퇴거를 수행합니다 (동기 버전).

        매 LLM 호출 전에 3가지 작업을 수행합니다:

        1. **execute 도구 필터링**: 백엔드가 실행을 지원하지 않으면 execute 도구를 제거
        2. **시스템 프롬프트 주입**: 파일시스템 도구 사용법 + 실행 도구 사용법(지원 시)을
           시스템 메시지에 추가
        3. **대용량 HumanMessage 퇴거**:
           - `lc_evicted_to` 태그가 있는 기존 메시지는 절삭된 미리보기로 교체
           - 가장 최근 메시지가 태그 없는 대용량 HumanMessage면 백엔드에 저장 후 태그

        Args:
            request: 처리 중인 모델 요청.
            handler: 수정된 요청으로 호출할 핸들러 함수.

        Returns:
            모델 응답, 또는 새로 퇴거된 메시지를 태그하는 상태 업데이트가
            포함된 ExtendedModelResponse.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        eviction_result = self._evict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | ExtendedModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Also evicts oversized HumanMessages to the filesystem. See
        `wrap_model_call` for full documentation.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler, or an `ExtendedModelResponse`
            with a state update tagging newly evicted messages.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)  # ty: ignore[invalid-argument-type]
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts).strip()

        if system_prompt:
            new_system_message = append_to_system_message(request.system_message, system_prompt)
            request = request.override(system_message=new_system_message)

        eviction_result = await self._aevict_and_truncate_messages(request)
        if eviction_result is not None:
            messages, state_command = eviction_result
            request = request.override(messages=messages)
            response = await handler(request)
            if state_command is not None:
                return ExtendedModelResponse(model_response=response, command=state_command)
            return response

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, bool]:
        """Process a large ToolMessage by evicting its content to filesystem.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, evicted):
            - processed_message: New ToolMessage with truncated content and file reference
            - evicted: Whether the content was evicted to the filesystem

        Note:
            Text is extracted from all text content blocks, joined, and used for both the
            size check and eviction. Non-text blocks (images, audio, etc.) are preserved in
            the replacement message so multimodal context is not lost. The model can recover
            the full text by reading the offloaded file from the backend.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        # Check if content exceeds eviction threshold
        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, False

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, True

    async def _aprocess_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, bool]:
        """Async version of _process_large_message.

        Uses async backend methods to avoid sync calls in async context.
        See _process_large_message for full documentation.
        """
        # Early exit if eviction not configured
        if not self._tool_token_limit_before_evict:
            return message, False

        content_str = _extract_text_from_message(message)

        if len(content_str) <= NUM_CHARS_PER_TOKEN * self._tool_token_limit_before_evict:
            return message, False

        # Write content to filesystem using async method
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = await resolved_backend.awrite(file_path, content_str)
        if result.error:
            return message, False

        # Create preview showing head and tail of the result
        content_sample = _create_content_preview(content_str)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        evicted = _build_evicted_content(message, replacement_text)
        processed_message = ToolMessage(
            content=cast("str | list[str | dict]", evicted),
            tool_call_id=message.tool_call_id,
            name=message.name,
            id=message.id,
            artifact=message.artifact,
            status=message.status,
            additional_kwargs=dict(message.additional_kwargs),
            response_metadata=dict(message.response_metadata),
        )
        return processed_message, True

    def _get_backend_from_runtime(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> BackendProtocol:
        """Resolve the backend from a bare `Runtime`.

        Constructs a `ToolRuntime` from the `Runtime` to satisfy the backend
        factory interface. Used by hooks like `before_agent` that receive
        `Runtime` rather than `ToolRuntime`.

        Args:
            state: The current agent state.
            runtime: The runtime context.

        Returns:
            Resolved backend instance.
        """
        if not callable(self.backend):
            return self.backend
        config = cast("RunnableConfig", getattr(runtime, "config", {}))
        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            stream_writer=runtime.stream_writer,
            store=runtime.store,
            config=config,
            tool_call_id=None,
        )
        return self.backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]

    def _check_eviction_needed(
        self,
        messages: list[AnyMessage],
    ) -> tuple[bool, bool]:
        """Check whether any message processing is needed.

        Args:
            messages: The message list to inspect.

        Returns:
            Tuple of (has_tagged, new_eviction_needed).
        """
        if not self._human_message_token_limit_before_evict:
            return False, False

        threshold = NUM_CHARS_PER_TOKEN * self._human_message_token_limit_before_evict
        has_tagged = any(isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_evicted_to") for msg in messages)
        new_eviction_needed = False
        if messages and isinstance(messages[-1], HumanMessage):
            last = messages[-1]
            if not last.additional_kwargs.get("lc_evicted_to") and len(_extract_text_from_message(last)) > threshold:
                new_eviction_needed = True
        return has_tagged, new_eviction_needed

    @staticmethod
    def _apply_eviction_and_truncate(
        messages: list[AnyMessage],
        write_result: WriteResult | None,
        file_path: str | None,
    ) -> tuple[list[AnyMessage], Command | None]:
        """Tag a newly evicted message and truncate all tagged messages.

        Args:
            messages: The message list (may be modified if write succeeded).
            write_result: Result of the backend write, or `None` if no new
                eviction was attempted.
            file_path: Path the content was written to.

        Returns:
            Tuple of (processed_messages, state_command).
        """
        state_command: Command | None = None

        if write_result is not None and file_path is not None and not write_result.error:
            last = messages[-1]
            tagged = last.model_copy(
                update={
                    "additional_kwargs": {
                        **last.additional_kwargs,
                        "lc_evicted_to": file_path,
                    }
                }
            )
            state_command = Command(update={"messages": [tagged]})
            messages = [*messages[:-1], tagged]

        processed: list[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_evicted_to"):
                processed.append(_build_truncated_human_message(msg, msg.additional_kwargs["lc_evicted_to"]))
            else:
                processed.append(msg)

        return processed, state_command

    def _evict_and_truncate_messages(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[list[AnyMessage], Command | None] | None:
        """Evict a new oversized HumanMessage and truncate all tagged messages.

        Returns `None` if no messages needed processing (fast path). Otherwise
        returns `(processed_messages, command)` where `command` is a state
        update tagging the newly evicted message, or `None` if only
        previously-tagged messages were truncated.

        Args:
            request: The model request being processed.

        Returns:
            Tuple of (messages, command) if any processing occurred, else `None`.
        """
        messages = list(request.messages)
        has_tagged, new_eviction_needed = self._check_eviction_needed(messages)
        if not has_tagged and not new_eviction_needed:
            return None

        write_result: WriteResult | None = None
        file_path: str | None = None
        if new_eviction_needed:
            backend = self._get_backend_from_runtime(request.state, request.runtime)
            file_path = f"/conversation_history/{uuid.uuid4()}.md"
            write_result = backend.write(file_path, _extract_text_from_message(messages[-1]))

        return self._apply_eviction_and_truncate(messages, write_result, file_path)

    async def _aevict_and_truncate_messages(
        self,
        request: ModelRequest[ContextT],
    ) -> tuple[list[AnyMessage], Command | None] | None:
        """Async version of `_evict_and_truncate_messages`.

        Args:
            request: The model request being processed.

        Returns:
            Tuple of (messages, command) if any processing occurred, else `None`.
        """
        messages = list(request.messages)
        has_tagged, new_eviction_needed = self._check_eviction_needed(messages)
        if not has_tagged and not new_eviction_needed:
            return None

        write_result: WriteResult | None = None
        file_path: str | None = None
        if new_eviction_needed:
            backend = self._get_backend_from_runtime(request.state, request.runtime)
            file_path = f"/conversation_history/{uuid.uuid4()}.md"
            write_result = await backend.awrite(file_path, _extract_text_from_message(messages[-1]))

        return self._apply_eviction_and_truncate(messages, write_result, file_path)

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a processed result with
            evicted content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, _evicted = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return processed_message

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, _evicted = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
            return Command(update={**update, "messages": processed_messages})
        msg = f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

    async def _aintercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Async version of _intercept_large_tool_result.

        Uses async backend methods to avoid sync calls in async context.
        See _intercept_large_tool_result for full documentation.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, _evicted = await self._aprocess_large_message(
                tool_result,
                resolved_backend,
            )
            return processed_message

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, _evicted = await self._aprocess_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
            return Command(update={**update, "messages": processed_messages})
        msg = f"Unreachable code reached in _aintercept_large_tool_result: for tool_result of type {type(tool_result)}"
        raise AssertionError(msg)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """도구 호출 결과의 크기를 확인하고, 너무 크면 파일시스템에 퇴거합니다 (동기 버전).

        TOOLS_EXCLUDED_FROM_EVICTION에 포함된 도구(ls, glob, grep, read_file 등)는
        자체 절삭 메커니즘이 있으므로 퇴거 대상에서 제외됩니다.

        퇴거 시: 전체 내용을 /large_tool_results/{tool_call_id}에 저장하고,
        원본 내용을 head/tail 미리보기와 파일 경로 참조로 교체합니다.

        Args:
            request: 처리 중인 도구 호출 요청.
            handler: 요청으로 호출할 핸들러 함수.

        Returns:
            원본 ToolMessage, 또는 결과가 상태에 저장된 의사(pseudo) 도구 메시지.
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self._tool_token_limit_before_evict is None or request.tool_call["name"] in TOOLS_EXCLUDED_FROM_EVICTION:
            return await handler(request)

        tool_result = await handler(request)
        return await self._aintercept_large_tool_result(tool_result, request.runtime)
