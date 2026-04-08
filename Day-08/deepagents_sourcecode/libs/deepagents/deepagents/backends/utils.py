"""메모리 백엔드 구현체들이 공유하는 유틸리티 함수 모음.

이 모듈은 사용자에게 노출되는 문자열 포맷터와,
백엔드 및 복합 라우터에서 사용하는 구조화 헬퍼 함수를 모두 포함합니다.
구조화 헬퍼를 통해 취약한 문자열 파싱 없이 조합(composition)이 가능합니다.
"""

import os
import re
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, overload

import wcmatch.glob as wcglob

from deepagents.backends.protocol import FileData, FileInfo as _FileInfo, GrepMatch as _GrepMatch, GrepResult, ReadResult

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"

FileType = Literal["text", "image", "audio", "video", "file"]
"""파일을 확장자 기준으로 분류한 타입."""

_EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    # 이미지 (https://ai.google.dev/gemini-api/docs/image-understanding)
    ".png": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".webp": "image",
    ".heic": "image",
    ".heif": "image",
    # 동영상 (https://ai.google.dev/gemini-api/docs/video-understanding)
    ".mp4": "video",
    ".mpeg": "video",
    ".mov": "video",
    ".avi": "video",
    ".flv": "video",
    ".mpg": "video",
    ".webm": "video",
    ".wmv": "video",
    ".3gpp": "video",
    # 오디오 (https://ai.google.dev/gemini-api/docs/audio)
    ".wav": "audio",
    ".mp3": "audio",
    ".aiff": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    # 파일
    ".pdf": "file",
    ".ppt": "file",
    ".pptx": "file",
}
"""텍스트가 아닌 파일의 확장자-타입 매핑.

Google 멀티모달 API 지원 포맷 기반:

- 이미지: https://ai.google.dev/gemini-api/docs/image-understanding
- 동영상: https://ai.google.dev/gemini-api/docs/video-understanding
- 오디오: https://ai.google.dev/gemini-api/docs/audio
"""

MAX_LINE_LENGTH = 5000
LINE_NUMBER_WIDTH = 6
TOOL_RESULT_TOKEN_LIMIT = 20000  # 축출(eviction) 임계값과 동일
TRUNCATION_GUIDANCE = "... [results truncated, try being more specific with your parameters]"

# 하위 호환성을 위해 protocol 타입 재내보내기
FileInfo = _FileInfo
GrepMatch = _GrepMatch


def _normalize_content(file_data: FileData) -> str:
    """file_data의 content를 일반 문자열로 정규화합니다.

    레거시 `list[str]` 파일 포맷을 위한 단일 하위 호환 변환 지점입니다.
    신규 코드는 `content`를 일반 `str`로 저장하지만,
    기존 데이터에는 여전히 줄 목록(list of lines)이 담겨 있을 수 있습니다.

    Args:
        file_data: `content` 키를 포함하는 FileData 딕셔너리.

    Returns:
        단일 문자열로 변환된 content.
    """
    content = file_data["content"]
    if isinstance(content, list):
        warnings.warn(
            "FileData with list[str] content is deprecated. Content should be stored as a plain str.",
            DeprecationWarning,
            stacklevel=2,
        )
        return "\n".join(content)
    return content


def sanitize_tool_call_id(tool_call_id: str) -> str:
    r"""tool_call_id를 정제하여 경로 탐색 및 구분자 문제를 방지합니다.

    위험 문자(., /, \)를 언더스코어로 대체합니다.
    """
    return tool_call_id.replace(".", "_").replace("/", "_").replace("\\", "_")


def format_content_with_line_numbers(
    content: str | list[str],
    start_line: int = 1,
) -> str:
    """파일 내용에 줄 번호를 붙여 포맷합니다 (cat -n 스타일).

    MAX_LINE_LENGTH를 초과하는 줄은 연속 마커(예: 5.1, 5.2)로 청크 분할합니다.

    Args:
        content: 문자열 또는 줄 목록 형태의 파일 내용
        start_line: 시작 줄 번호 (기본값: 1)

    Returns:
        줄 번호와 연속 마커가 포함된 포맷된 내용
    """
    if isinstance(content, str):
        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
    else:
        lines = content

    result_lines = []
    for i, line in enumerate(lines):
        line_num = i + start_line

        if len(line) <= MAX_LINE_LENGTH:
            result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            # 긴 줄을 청크로 분할하고 연속 마커 부여
            num_chunks = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for chunk_idx in range(num_chunks):
                start = chunk_idx * MAX_LINE_LENGTH
                end = min(start + MAX_LINE_LENGTH, len(line))
                chunk = line[start:end]
                if chunk_idx == 0:
                    # 첫 번째 청크: 일반 줄 번호 사용
                    result_lines.append(f"{line_num:{LINE_NUMBER_WIDTH}d}\t{chunk}")
                else:
                    # 연속 청크: 소수점 표기법 사용 (예: 5.1, 5.2)
                    continuation_marker = f"{line_num}.{chunk_idx}"
                    result_lines.append(f"{continuation_marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")

    return "\n".join(result_lines)


def check_empty_content(content: str) -> str | None:
    """content가 비어 있는지 확인하고 경고 메시지를 반환합니다.

    Args:
        content: 검사할 내용

    Returns:
        비어 있으면 경고 메시지, 그렇지 않으면 None.
    """
    if not content or content.strip() == "":
        return EMPTY_CONTENT_WARNING
    return None


def _get_file_type(path: str) -> FileType:
    """파일 확장자로 파일 종류를 분류합니다.

    Args:
        path: 분류할 파일 경로.

    Returns:
        `"text"`, `"image"`, `"audio"`, `"video"`, `"file"` 중 하나.
        인식되지 않는 확장자는 `"text"`로 기본 처리합니다.
    """
    return _EXTENSION_TO_FILE_TYPE.get(PurePosixPath(path).suffix.lower(), "text")


def _to_legacy_file_data(file_data: FileData) -> dict[str, Any]:
    r"""FileData 딕셔너리를 레거시(v1) 저장 포맷으로 변환합니다.

    v1 포맷은 content를 `list[str]` (줄 단위로 `\\n` 분할)로 저장하며
    `encoding` 필드를 포함하지 않습니다.
    `list[str]` content를 기대하는 소비자와의 하위 호환성을 유지하려면
    백엔드에서 `file_format="v1"` 사용 시 이 함수를 호출하십시오.

    Args:
        file_data: `content: str`과 `encoding`이 포함된 현대(v2) FileData.

    Returns:
        `content`가 `list[str]`이고 `created_at` / `modified_at` 타임스탬프를
        포함하는 딕셔너리. `encoding` 키 없음.
    """
    content = file_data["content"]
    result: dict[str, Any] = {
        "content": content.split("\n"),
    }
    if "created_at" in file_data:
        result["created_at"] = file_data["created_at"]
    if "modified_at" in file_data:
        result["modified_at"] = file_data["modified_at"]
    return result


def file_data_to_string(file_data: FileData) -> str:
    """FileData를 일반 문자열 content로 변환합니다.

    Args:
        file_data: 'content' 키를 포함하는 FileData 딕셔너리

    Returns:
        단일 문자열로 변환된 content.
    """
    return _normalize_content(file_data)


def create_file_data(
    content: str,
    created_at: str | None = None,
    encoding: str = "utf-8",
) -> FileData:
    """타임스탬프가 포함된 FileData 객체를 생성합니다.

    Args:
        content: 문자열 형태의 파일 내용 (일반 텍스트 또는 base64 인코딩된 바이너리).
        created_at: 선택적 생성 타임스탬프 (ISO 포맷).
        encoding: 내용 인코딩 — 텍스트는 `"utf-8"`, 바이너리는 `"base64"`.

    Returns:
        content, encoding, 타임스탬프가 포함된 FileData 딕셔너리.
    """
    now = datetime.now(UTC).isoformat()

    return {
        "content": content,
        "encoding": encoding,
        "created_at": created_at or now,
        "modified_at": now,
    }


def update_file_data(file_data: FileData, content: str) -> FileData:
    """새 content로 FileData를 갱신하되 생성 타임스탬프는 유지합니다.

    Args:
        file_data: 기존 FileData 딕셔너리
        content: 문자열 형태의 새 content

    Returns:
        갱신된 FileData 딕셔너리
    """
    now = datetime.now(UTC).isoformat()

    result = FileData(
        content=content,
        encoding=file_data.get("encoding", "utf-8"),
    )
    if "created_at" in file_data:
        result["created_at"] = file_data["created_at"]
    result["modified_at"] = now
    return result


def slice_read_response(
    file_data: FileData,
    offset: int,
    limit: int,
) -> str | ReadResult:
    """요청한 줄 범위로 파일 데이터를 슬라이싱합니다 (포맷 미적용).

    요청된 창(window)에 해당하는 원시 텍스트를 반환합니다.
    줄 번호 포맷은 미들웨어 레이어에서 후처리됩니다.

    Args:
        file_data: FileData 딕셔너리.
        offset: 줄 오프셋 (0-인덱스 기반).
        limit: 최대 줄 수.

    Returns:
        성공 시 슬라이싱된 원시 content 문자열,
        오프셋이 파일 길이를 초과하면 `error`가 설정된 `ReadResult`.
    """
    content = file_data_to_string(file_data)

    if not content or content.strip() == "":
        return content

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return ReadResult(error=f"Line offset {offset} exceeds file length ({len(lines)} lines)")

    selected_lines = lines[start_idx:end_idx]
    return "\n".join(selected_lines)


def format_read_response(
    file_data: FileData,
    offset: int,
    limit: int,
) -> str:
    """줄 번호를 포함하여 읽기 응답용 파일 데이터를 포맷합니다.

    .. deprecated::
        `slice_read_response`를 사용하고
        `format_content_with_line_numbers`를 별도로 적용하십시오.

    Args:
        file_data: FileData 딕셔너리
        offset: 줄 오프셋 (0-인덱스 기반)
        limit: 최대 줄 수

    Returns:
        포맷된 content 또는 오류 메시지
    """
    content = file_data_to_string(file_data)
    empty_msg = check_empty_content(content)
    if empty_msg:
        return empty_msg

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    selected_lines = lines[start_idx:end_idx]
    return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)


def perform_string_replacement(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,  # noqa: FBT001, FBT002
) -> tuple[str, int] | str:
    """발생 횟수 검증을 포함한 문자열 교체를 수행합니다.

    Args:
        content: 원본 content
        old_string: 교체할 문자열
        new_string: 대체 문자열
        replace_all: 모든 발생을 교체할지 여부

    Returns:
        성공 시 (new_content, occurrences) 튜플, 실패 시 오류 메시지 문자열
    """
    occurrences = content.count(old_string)

    if occurrences == 0:
        return f"Error: String not found in file: '{old_string}'"

    if occurrences > 1 and not replace_all:
        return (
            f"Error: String '{old_string}' appears {occurrences} times in file. "
            f"Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        )

    new_content = content.replace(old_string, new_string)
    return new_content, occurrences


@overload
def truncate_if_too_long(result: list[str]) -> list[str]: ...


@overload
def truncate_if_too_long(result: str) -> str: ...


def truncate_if_too_long(result: list[str] | str) -> list[str] | str:
    """토큰 제한을 초과하면 리스트 또는 문자열 결과를 절삭합니다 (대략적인 추정: 4자/토큰)."""
    if isinstance(result, list):
        total_chars = sum(len(item) for item in result)
        if total_chars > TOOL_RESULT_TOKEN_LIMIT * 4:
            return result[: len(result) * TOOL_RESULT_TOKEN_LIMIT * 4 // total_chars] + [TRUNCATION_GUIDANCE]  # noqa: RUF005  # Concatenation preferred for clarity
        return result
    # 문자열 처리
    if len(result) > TOOL_RESULT_TOKEN_LIMIT * 4:
        return result[: TOOL_RESULT_TOKEN_LIMIT * 4] + "\n" + TRUNCATION_GUIDANCE
    return result


def validate_path(path: str, *, allowed_prefixes: Sequence[str] | None = None) -> str:
    r"""파일 경로를 보안상 검증하고 정규화합니다.

    디렉토리 탐색 공격을 방지하고 일관된 포맷을 강제함으로써
    경로를 안전하게 사용할 수 있도록 합니다.
    모든 경로는 슬래시로 정규화되며 앞에 슬래시가 붙습니다.

    이 함수는 가상 파일시스템 경로를 위해 설계되었으며,
    일관성 유지 및 경로 포맷 모호성 방지를 위해
    Windows 절대 경로(예: `C:/...`, `F:/...`)를 거부합니다.

    Args:
        path: 검증 및 정규화할 경로.
        allowed_prefixes: 허용된 경로 접두사 목록 (선택 사항).

            지정하면 정규화된 경로가 이 중 하나로 시작해야 합니다.

    Returns:
        `/`로 시작하고 슬래시를 사용하는 정규화된 표준 경로.

    Raises:
        ValueError: 경로에 탐색 시퀀스(`..` 또는 `~`)가 포함되거나,
            Windows 절대 경로(예: `C:/...`)이거나,
            `allowed_prefixes` 지정 시 허용된 접두사로 시작하지 않으면 발생.

    Example:
        ```python
        validate_path("foo/bar")  # Returns: "/foo/bar"
        validate_path("/./foo//bar")  # Returns: "/foo/bar"
        validate_path("../etc/passwd")  # Raises ValueError
        validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    # 부분 문자열이 아닌 경로 구성 요소로서의 탐색 문자를 확인하여
    # "foo..bar.txt" 같은 정상 파일명을 잘못 거부하는 것을 방지
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if ".." in parts or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Windows 절대 경로 거부 (예: C:\..., D:/...)
    if re.match(r"^[a-zA-Z]:", path):
        msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
        raise ValueError(msg)

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    # 심층 방어: normpath 이후에도 탐색이 발생하지 않았는지 검증
    if ".." in normalized.split("/"):
        msg = f"Path traversal detected after normalization: {path} -> {normalized}"
        raise ValueError(msg)

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


def _normalize_path(path: str | None) -> str:
    """경로를 표준 형식으로 정규화합니다.

    경로를 /로 시작하는 절대 형식으로 변환하고,
    후행 슬래시를 제거하며(루트 제외),
    경로가 비어 있지 않은지 검증합니다.

    Args:
        path: 정규화할 경로 (None이면 "/"로 기본 처리)

    Returns:
        /로 시작하는 정규화된 경로 (루트가 아니면 후행 슬래시 없음)

    Raises:
        ValueError: 경로가 유효하지 않으면 발생 (strip 후 빈 문자열)

    Example:
        _normalize_path(None) -> "/"
        _normalize_path("/dir/") -> "/dir"
        _normalize_path("dir") -> "/dir"
        _normalize_path("/") -> "/"
    """
    path = path or "/"
    if not path or path.strip() == "":
        msg = "Path cannot be empty"
        raise ValueError(msg)

    normalized = path if path.startswith("/") else "/" + path

    # 루트만 후행 슬래시 허용
    if normalized != "/" and normalized.endswith("/"):
        normalized = normalized.rstrip("/")

    return normalized


def _filter_files_by_path(files: dict[str, Any], normalized_path: str) -> dict[str, Any]:
    """정규화된 경로로 files 딕셔너리를 필터링합니다.
    정확한 파일 일치와 디렉토리 접두사 매칭을 모두 처리합니다.

    _normalize_path에서 반환된 정규화된 경로를 입력으로 기대합니다 (루트 제외 후행 슬래시 없음).

    Args:
        files: 파일 경로를 파일 데이터에 매핑하는 딕셔너리
        normalized_path: _normalize_path에서 반환된 정규화된 경로 (예: "/", "/dir", "/dir/file")

    Returns:
        경로에 매칭되는 파일들의 필터링된 딕셔너리

    Example:
        files = {"/dir/file": {...}, "/dir/other": {...}}
        _filter_files_by_path(files, "/dir/file")  # Returns {"/dir/file": {...}}
        _filter_files_by_path(files, "/dir")       # Returns both files
    """
    # 정확한 파일 일치 여부 확인
    if normalized_path in files:
        return {normalized_path: files[normalized_path]}

    # 일치하지 않으면 디렉토리 접두사로 처리
    if normalized_path == "/":
        # 루트 디렉토리 — /로 시작하는 모든 파일 매칭
        return {fp: fd for fp, fd in files.items() if fp.startswith("/")}
    # 루트가 아닌 디렉토리 — 접두사 매칭을 위해 후행 슬래시 추가
    dir_prefix = normalized_path + "/"
    return {fp: fd for fp, fd in files.items() if fp.startswith(dir_prefix)}


def _glob_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str = "/",
) -> str:
    r"""글로브 패턴에 매칭되는 경로를 files 딕셔너리에서 검색합니다.

    Args:
        files: 파일 경로를 FileData에 매핑하는 딕셔너리.
        pattern: 글로브 패턴 (예: "*.py", "**/*.ts").
        path: 검색 기준 경로.

    Returns:
        수정 시간 내림차순으로 정렬된 줄바꿈 구분 파일 경로 목록.
        매칭 없으면 "No files found" 반환.

    Example:
        ```python
        files = {"/src/main.py": FileData(...), "/test.py": FileData(...)}
        _glob_search_files(files, "*.py", "/")
        # Returns: "/test.py\n/src/main.py" (sorted by modified_at)
        ```
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No files found"

    filtered = _filter_files_by_path(files, normalized_path)

    # 표준 glob 시맨틱 준수:
    # - 경로 구분자 없는 패턴 (예: "*.py")은 `path` 기준 현재 디렉토리에서만 매칭 (비재귀).
    # - 재귀 매칭에는 명시적으로 "**" 사용.
    # 상대 경로에 대해 매칭하므로 패턴 앞의 "/"를 제거.
    effective_pattern = pattern.lstrip("/")

    matches = []
    for file_path, file_data in filtered.items():
        # 글로브 매칭을 위한 상대 경로 계산
        # normalized_path가 "/dir"이면 "/dir/file.txt" -> "file.txt"
        # normalized_path가 "/dir/file.txt"이면 (정확한 파일) -> "file.txt"
        if normalized_path == "/":
            relative = file_path[1:]  # 앞의 슬래시 제거
        elif file_path == normalized_path:
            # 정확한 파일 매칭 — 파일명만 사용
            relative = file_path.split("/")[-1]
        else:
            # 디렉토리 접두사 — 디렉토리 경로 제거
            relative = file_path[len(normalized_path) + 1 :]  # +1은 슬래시 제거 위한 것

        if wcglob.globmatch(relative, effective_pattern, flags=wcglob.BRACE | wcglob.GLOBSTAR):
            matches.append((file_path, file_data["modified_at"]))

    matches.sort(key=lambda x: x[1], reverse=True)

    if not matches:
        return "No files found"

    return "\n".join(fp for fp, _ in matches)


def _format_grep_results(
    results: dict[str, list[tuple[int, str]]],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """output_mode에 따라 grep 검색 결과를 포맷합니다.

    Args:
        results: 파일 경로를 (줄 번호, 줄 내용) 튜플 목록에 매핑하는 딕셔너리
        output_mode: 출력 형식 — "files_with_matches", "content", "count" 중 하나

    Returns:
        포맷된 문자열 출력
    """
    if output_mode == "files_with_matches":
        return "\n".join(sorted(results.keys()))
    if output_mode == "count":
        lines = []
        for file_path in sorted(results.keys()):
            count = len(results[file_path])
            lines.append(f"{file_path}: {count}")
        return "\n".join(lines)
    lines = []
    for file_path in sorted(results.keys()):
        lines.append(f"{file_path}:")
        for line_num, line in results[file_path]:
            lines.append(f"  {line_num}: {line}")
    return "\n".join(lines)


def _grep_search_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    r"""정규식 패턴으로 파일 내용을 검색합니다.

    Args:
        files: 파일 경로를 FileData에 매핑하는 딕셔너리.
        pattern: 검색할 정규식 패턴.
        path: 검색 기준 경로.
        glob: 파일 필터링용 선택적 글로브 패턴 (예: "*.py").
        output_mode: 출력 형식 — "files_with_matches", "content", "count" 중 하나.

    Returns:
        포맷된 검색 결과. 결과 없으면 "No matches found" 반환.

    Example:
        ```python
        files = {"/file.py": FileData(content="import os\nprint('hi')", ...)}
        _grep_search_files(files, "import", "/")
        # Returns: "/file.py" (with output_mode="files_with_matches")
        ```
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return "No matches found"

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    results: dict[str, list[tuple[int, str]]] = {}
    for file_path, file_data in filtered.items():
        content_str = _normalize_content(file_data)
        for line_num, line in enumerate(content_str.split("\n"), 1):
            if regex.search(line):
                if file_path not in results:
                    results[file_path] = []
                results[file_path].append((line_num, line))

    if not results:
        return "No matches found"
    return _format_grep_results(results, output_mode)


# -------- 조합을 위한 구조화 헬퍼 --------


def grep_matches_from_files(
    files: dict[str, Any],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> GrepResult:
    """인메모리 files 매핑에서 구조화된 grep 매칭 결과를 반환합니다.

    리터럴 텍스트 검색을 수행합니다 (정규식 아님).

    성공 시 매칭 결과가 담긴 GrepResult를 반환합니다.
    툴 컨텍스트에서 백엔드가 예외를 던지지 않도록
    의도적으로 예외를 발생시키지 않으며, 사용자 대면 오류 메시지를 보존합니다.
    """
    try:
        normalized_path = _normalize_path(path)
    except ValueError:
        return GrepResult(matches=[])

    filtered = _filter_files_by_path(files, normalized_path)

    if glob:
        filtered = {fp: fd for fp, fd in filtered.items() if wcglob.globmatch(Path(fp).name, glob, flags=wcglob.BRACE)}

    matches: list[GrepMatch] = []
    for file_path, file_data in filtered.items():
        content_str = _normalize_content(file_data)
        for line_num, line in enumerate(content_str.split("\n"), 1):
            if pattern in line:  # 리터럴 매칭을 위한 단순 부분 문자열 검색
                matches.append({"path": file_path, "line": int(line_num), "text": line})
    return GrepResult(matches=matches)


def build_grep_results_dict(matches: list[GrepMatch]) -> dict[str, list[tuple[int, str]]]:
    """구조화된 매칭 결과를 포매터에서 사용하는 레거시 딕셔너리 형태로 그룹화합니다."""
    grouped: dict[str, list[tuple[int, str]]] = {}
    for m in matches:
        grouped.setdefault(m["path"], []).append((m["line"], m["text"]))
    return grouped


def format_grep_matches(
    matches: list[GrepMatch],
    output_mode: Literal["files_with_matches", "content", "count"],
) -> str:
    """기존 포맷팅 로직을 사용하여 구조화된 grep 매칭 결과를 포맷합니다."""
    if not matches:
        return "No matches found"
    return _format_grep_results(build_grep_results_dict(matches), output_mode)
