"""도구 호출을 위한 형식 지정 유틸리티가 CLI에 표시됩니다.

이 모듈은 TUI에 대한 렌더링 도구 호출 및 도구 메시지를 처리합니다.

`textual_adapter`에 의해 모듈 수준에서 가져옵니다(자체는 시작 경로에서 지연됨). 과도한 SDK 종속성(예: `backends`)은 함수
본문으로 연기됩니다.
"""

import json
from contextlib import suppress
from pathlib import Path
from typing import Any

from deepagents_cli.config import MAX_ARG_LENGTH, get_glyphs
from deepagents_cli.unicode_security import strip_dangerous_unicode

_HIDDEN_CHAR_MARKER = " [hidden chars removed]"
"""위험한 유니코드가 제거된 값을 표시하기 위해 추가된 마커를 통해 사용자는 안전을 위해 해당 값이 수정되었음을 알 수 있습니다."""


def _format_timeout(seconds: int) -> str:
    """사람이 읽을 수 있는 단위의 형식 시간 초과입니다(예: 300 -> '5m', 3600 -> '1h').

    Args:
        seconds: 형식화할 시간 초과 값(초)입니다.

    Returns:
        사람이 읽을 수 있는 시간 제한 문자열(예: '5m', '1h', '300s')입니다.

    """
    if seconds < 60:  # noqa: PLR2004  # Time unit boundary
        return f"{seconds}s"
    if seconds < 3600 and seconds % 60 == 0:  # noqa: PLR2004  # Time unit boundaries
        return f"{seconds // 60}m"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    # For odd values, just show seconds
    return f"{seconds}s"


def _coerce_timeout_seconds(timeout: int | str | None) -> int | None:
    """표시를 위해 시간 초과 값을 초 단위로 정규화합니다.

    정수 값과 숫자 문자열을 허용합니다. 유효하지 않은 값에 대해서는 `None`을 반환하므로 표시 형식이 절대 발생하지 않습니다.

    Args:
        timeout: 도구 인수의 원시 시간 초과 값입니다.

    Returns:
        초 단위의 정수 시간 제한 또는 사용할 수 없거나 유효하지 않은 경우 `None`입니다.

    """
    if type(timeout) is int:
        return timeout
    if isinstance(timeout, str):
        stripped = timeout.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def truncate_value(value: str, max_length: int = MAX_ARG_LENGTH) -> str:
    """max_length를 초과하는 경우 문자열 값을 자릅니다.

    Returns:
        초과하는 경우 줄임표 접미사가 있는 잘린 문자열, 그렇지 않으면 원본입니다.

    """
    if len(value) > max_length:
        return value[:max_length] + get_glyphs().ellipsis
    return value


def _sanitize_display_value(value: object, *, max_length: int = MAX_ARG_LENGTH) -> str:
    """안전하고 컴팩트한 터미널 디스플레이를 위해 값을 정리합니다.

    숨겨진/기만적인 유니코드 컨트롤은 제거됩니다. 스트리핑이 발생하면 디스플레이 안전을 위해 변경된 값을 사용자가 알 수 있도록 마커가 추가됩니다.

    Args:
        value: 표시할 값입니다.
        max_length: 잘리기 전의 최대 표시 길이입니다.

    Returns:
        표시 문자열을 정리했습니다.

    """
    raw = str(value)
    sanitized = strip_dangerous_unicode(raw)
    display = truncate_value(sanitized, max_length)
    if sanitized != raw:
        return display + _HIDDEN_CHAR_MARKER
    return display


def format_tool_display(tool_name: str, tool_args: dict) -> str:
    """도구별 스마트 서식을 사용하여 표시할 도구 호출 서식을 지정합니다.

    모든 인수가 아닌 각 도구 유형에 대해 가장 관련성이 높은 정보를 표시합니다.

    Args:
        tool_name: 호출되는 도구의 이름
        tool_args: 도구 인수 사전

    Returns:
        표시용 형식 문자열(예: ASCII 모드의 "(*) read_file(config.py)")

    Examples:
        read_file(path="/long/path/file.py") → "<prefix> read_file(file.py)"
        web_search(query="코드 작성 방법") → '<prefix> web_search("코드 작성 방법")' 실행(command="pip
        install foo") → '<prefix> 실행("pip install foo")'

    """
    prefix = get_glyphs().tool_prefix

    def abbreviate_path(path_str: str, max_length: int = 60) -> str:
        """지능적으로 파일 경로를 축약합니다. 기본 이름 또는 상대 경로를 표시합니다.

        Returns:
            표시에 적합한 단축 경로 문자열입니다.

        """
        try:
            path = Path(path_str)

            # If it's just a filename (no directory parts), return as-is
            if len(path.parts) == 1:
                return path_str

            # Try to get relative path from current working directory
            with suppress(
                ValueError,  # ValueError: path is not relative to cwd
                OSError,  # OSError: filesystem errors when resolving paths
            ):
                rel_path = path.relative_to(Path.cwd())
                rel_str = str(rel_path)
                # Use relative if it's shorter and not too long
                if len(rel_str) < len(path_str) and len(rel_str) <= max_length:
                    return rel_str

            # If absolute path is reasonable length, use it
            if len(path_str) <= max_length:
                return path_str
        except Exception:  # noqa: BLE001  # Fallback to original string on any path resolution error
            return truncate_value(path_str, max_length)
        else:
            # Otherwise, just show basename (filename only)
            return path.name

    # Tool-specific formatting - show the most important argument(s)
    if tool_name in {"read_file", "write_file", "edit_file"}:
        # File operations: show the primary file path argument (file_path or path)
        path_value = tool_args.get("file_path")
        if path_value is None:
            path_value = tool_args.get("path")
        if path_value is not None:
            path_raw = strip_dangerous_unicode(str(path_value))
            path = abbreviate_path(path_raw)
            if path_raw != str(path_value):
                path += _HIDDEN_CHAR_MARKER
            return f"{prefix} {tool_name}({path})"

    elif tool_name == "web_search":
        # Web search: show the query string
        if "query" in tool_args:
            query = _sanitize_display_value(tool_args["query"], max_length=100)
            return f'{prefix} {tool_name}("{query}")'

    elif tool_name == "grep":
        # Grep: show the search pattern
        if "pattern" in tool_args:
            pattern = _sanitize_display_value(tool_args["pattern"], max_length=70)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "execute":
        # Execute: show the command, and timeout only if non-default
        if "command" in tool_args:
            command = _sanitize_display_value(tool_args["command"], max_length=120)
            timeout = _coerce_timeout_seconds(tool_args.get("timeout"))
            from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT

            if timeout is not None and timeout != DEFAULT_EXECUTE_TIMEOUT:
                timeout_str = _format_timeout(timeout)
                return f'{prefix} {tool_name}("{command}", timeout={timeout_str})'
            return f'{prefix} {tool_name}("{command}")'

    elif tool_name == "ls":
        # ls: show directory, or empty if current directory
        if tool_args.get("path"):
            path_raw = strip_dangerous_unicode(str(tool_args["path"]))
            path = abbreviate_path(path_raw)
            if path_raw != str(tool_args["path"]):
                path += _HIDDEN_CHAR_MARKER
            return f"{prefix} {tool_name}({path})"
        return f"{prefix} {tool_name}()"

    elif tool_name == "glob":
        # Glob: show the pattern
        if "pattern" in tool_args:
            pattern = _sanitize_display_value(tool_args["pattern"], max_length=80)
            return f'{prefix} {tool_name}("{pattern}")'

    elif tool_name == "fetch_url":
        # Fetch URL: show the URL being fetched
        if "url" in tool_args:
            url = _sanitize_display_value(tool_args["url"], max_length=80)
            return f'{prefix} {tool_name}("{url}")'

    elif tool_name == "task":
        # Task: show subagent type badge
        agent_type = tool_args.get("subagent_type", "")
        if agent_type:
            agent_type = _sanitize_display_value(agent_type, max_length=40)
            return f"{prefix} {tool_name} [{agent_type}]"
        return f"{prefix} {tool_name}"

    elif tool_name == "ask_user":
        if "questions" in tool_args and isinstance(tool_args["questions"], list):
            count = len(tool_args["questions"])
            label = "question" if count == 1 else "questions"
            return f"{prefix} {tool_name}({count} {label})"

    elif tool_name == "compact_conversation":
        return f"{prefix} {tool_name}()"

    elif tool_name == "write_todos":
        if "todos" in tool_args and isinstance(tool_args["todos"], list):
            count = len(tool_args["todos"])
            return f"{prefix} {tool_name}({count} items)"

    # Fallback: generic formatting for unknown tools
    # Show all arguments in key=value format
    args_str = ", ".join(
        f"{_sanitize_display_value(k, max_length=30)}="
        f"{_sanitize_display_value(v, max_length=50)}"
        for k, v in tool_args.items()
    )
    return f"{prefix} {tool_name}({args_str})"


def _format_content_block(block: dict) -> str:
    """표시할 단일 콘텐츠 블록 사전 형식을 지정합니다.

    대규모 바이너리 페이로드(예: base64 이미지/비디오 데이터)를 사람이 읽을 수 있는 자리 표시자로 대체하여 터미널이 넘치지 않도록 합니다.

    Args:
        block: `ImageContentBlock`, `VideoContentBlock` 또는 `FileContentBlock` 사전.

    Returns:
        블록에 대한 표시 친화적인 문자열입니다.

    """
    if block.get("type") == "image" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "image")
        return f"[Image: {mime}, ~{size_kb}KB]"
    if block.get("type") == "video" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "video")
        return f"[Video: {mime}, ~{size_kb}KB]"
    if block.get("type") == "file" and isinstance(block.get("base64"), str):
        b64 = block["base64"]
        size_kb = len(b64) * 3 // 4 // 1024  # approximate decoded size
        mime = block.get("mime_type", "file")
        return f"[File: {mime}, ~{size_kb}KB]"
    try:
        # Preserve non-ASCII characters (CJK, emoji, etc.) instead of \uXXXX escapes
        return json.dumps(block, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(block)


def format_tool_message_content(content: Any) -> str:  # noqa: ANN401  # Content can be str, list, or dict
    """`ToolMessage` 콘텐츠를 인쇄 가능한 문자열로 변환합니다.

    Returns:
        도구 메시지 콘텐츠의 형식화된 문자열 표현입니다.

    """
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_format_content_block(item))
            else:
                try:
                    # Preserve non-ASCII characters (CJK, emoji, etc.)
                    parts.append(json.dumps(item, ensure_ascii=False))
                except (TypeError, ValueError):
                    parts.append(str(item))
        return "\n".join(parts)
    return str(content)
