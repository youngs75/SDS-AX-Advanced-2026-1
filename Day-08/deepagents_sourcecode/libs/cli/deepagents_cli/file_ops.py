"""승인 및 활동 표시를 위해 파일 변형을 요약합니다.

이 모듈의 도우미는 백엔드 파일 시스템 작업을 CLI UI에 대한 안정적인 미리 보기 개체, 차이점 및 사람이 읽을 수 있는 상태 줄로 전환합니다.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

FileOpStatus = Literal["pending", "success", "error"]


@dataclass
class ApprovalPreview:
    """HITL 미리보기를 렌더링하는 데 사용되는 데이터입니다."""

    title: str
    details: list[str]
    diff: str | None = None
    diff_title: str | None = None
    error: str | None = None


def _safe_read(path: Path) -> str | None:
    """파일 내용을 읽고, 실패하면 None을 반환합니다.

Returns:
        파일 내용을 문자열로 표시하거나, 읽지 못하는 경우 없음입니다.

    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Failed to read file %s: %s", path, e)
        return None


def _count_lines(text: str) -> int:
    """텍스트의 줄 수를 세고 빈 문자열을 0줄로 처리합니다.

Returns:
        텍스트의 줄 수입니다.

    """
    if not text:
        return 0
    return len(text.splitlines())


def compute_unified_diff(
    before: str,
    after: str,
    display_path: str,
    *,
    max_lines: int | None = 800,
    context_lines: int = 3,
) -> str | None:
    """콘텐츠 전후의 통합된 차이를 계산합니다.

Args:
        before: 원본 콘텐츠
        after: 새로운 콘텐츠
        display_path: diff 헤더에 표시할 경로
        max_lines: 최대 차이점 줄 수(무제한인 경우 없음)
        context_lines: 변경 사항 주변의 컨텍스트 줄 수(기본값 3)

Returns:
        통합 diff 문자열 또는 변경 사항이 없으면 없음

    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{display_path} (before)",
            tofile=f"{display_path} (after)",
            lineterm="",
            n=context_lines,
        )
    )
    if not diff_lines:
        return None
    if max_lines is not None and len(diff_lines) > max_lines:
        truncated = diff_lines[: max_lines - 1]
        truncated.append("...")
        return "\n".join(truncated)
    return "\n".join(diff_lines)


@dataclass
class FileOpMetrics:
    """파일 작업에 대한 라인 및 바이트 수준 메트릭입니다."""

    lines_read: int = 0
    start_line: int | None = None
    end_line: int | None = None
    lines_written: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    bytes_written: int = 0


@dataclass
class FileOperationRecord:
    """단일 파일 시스템 도구 호출을 추적합니다."""

    tool_name: str
    display_path: str
    physical_path: Path | None
    tool_call_id: str | None
    args: dict[str, Any] = field(default_factory=dict)
    status: FileOpStatus = "pending"
    error: str | None = None
    metrics: FileOpMetrics = field(default_factory=FileOpMetrics)
    diff: str | None = None
    before_content: str | None = None
    after_content: str | None = None
    read_output: str | None = None
    hitl_approved: bool = False


def resolve_physical_path(
    path_str: str | None, assistant_id: str | None
) -> Path | None:
    """가상/상대 경로를 실제 파일 시스템 경로로 변환합니다.

Returns:
        확인된 물리적 경로 또는 경로가 비어 있거나 확인에 실패한 경우 없음입니다.

    """
    if not path_str:
        return None
    try:
        if assistant_id and path_str.startswith("/memories/"):
            from deepagents_cli.config import settings

            agent_dir = settings.get_agent_dir(assistant_id)
            suffix = path_str.removeprefix("/memories/").lstrip("/")
            return (agent_dir / suffix).resolve()
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()
    except (OSError, ValueError):
        return None


def format_display_path(path_str: str | None) -> str:
    """표시할 경로의 형식을 지정합니다.

Returns:
        표시에 적합한 형식화된 경로 문자열입니다.

    """
    if not path_str:
        return "(unknown)"
    try:
        path = Path(path_str)
        if path.is_absolute():
            return path.name or str(path)
        return str(path)
    except (OSError, ValueError):
        return str(path_str)


def build_approval_preview(
    tool_name: str,
    args: dict[str, Any],
    assistant_id: str | None,
) -> ApprovalPreview | None:
    """HITL 승인을 위한 요약 정보 및 차이점을 수집합니다.

Returns:
        차이점 및 세부 정보가 포함된 ApprovalPreview 또는 도구가 지원되지 않는 경우 None입니다.

    """
    path_str = str(args.get("file_path") or args.get("path") or "")
    display_path = format_display_path(path_str)
    physical_path = resolve_physical_path(path_str, assistant_id)

    if tool_name == "write_file":
        content = str(args.get("content", ""))
        before = (
            _safe_read(physical_path)
            if physical_path and physical_path.exists()
            else ""
        )
        after = content
        diff = compute_unified_diff(before or "", after, display_path, max_lines=100)
        additions = 0
        if diff:
            additions = sum(
                1
                for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
        total_lines = _count_lines(after)
        details = [
            f"File: {path_str}",
            "Action: Create new file"
            + (" (overwrites existing content)" if before else ""),
            f"Lines to write: {additions or total_lines}",
        ]
        return ApprovalPreview(
            title=f"Write {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
        )

    if tool_name == "edit_file":
        if physical_path is None:
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
                error="Unable to resolve file path.",
            )
        before = _safe_read(physical_path)
        if before is None:
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
                error="Unable to read current file contents.",
            )
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all"))
        from deepagents.backends.utils import perform_string_replacement

        replacement = perform_string_replacement(
            before, old_string, new_string, replace_all
        )
        if isinstance(replacement, str):
            return ApprovalPreview(
                title=f"Update {display_path}",
                details=[f"File: {path_str}", "Action: Replace text"],
                error=replacement,
            )
        after, occurrences = replacement
        diff = compute_unified_diff(before, after, display_path, max_lines=None)
        additions = 0
        deletions = 0
        if diff:
            additions = sum(
                1
                for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            deletions = sum(
                1
                for line in diff.splitlines()
                if line.startswith("-") and not line.startswith("---")
            )
        action = "all occurrences" if replace_all else "single occurrence"
        details = [
            f"File: {path_str}",
            f"Action: Replace text ({action})",
            f"Occurrences matched: {occurrences}",
            f"Lines changed: +{additions} / -{deletions}",
        ]
        return ApprovalPreview(
            title=f"Update {display_path}",
            details=details,
            diff=diff,
            diff_title=f"Diff {display_path}",
        )

    return None


class FileOpTracker:
    """CLI 상호 작용 중에 파일 작업 지표를 수집합니다."""

    def __init__(
        self, *, assistant_id: str | None, backend: BackendProtocol | None = None
    ) -> None:
        """추적기를 초기화합니다."""
        self.assistant_id = assistant_id
        self.backend = backend
        self.active: dict[str | None, FileOperationRecord] = {}
        self.completed: list[FileOperationRecord] = []

    def start_operation(
        self, tool_name: str, args: dict[str, Any], tool_call_id: str | None
    ) -> None:
        """파일 작업 추적을 시작합니다.

        작업에 대한 레코드를 생성하고 쓰기/편집 작업의 경우 수정하기 전에 파일 내용을 캡처합니다.

        """
        if tool_name not in {"read_file", "write_file", "edit_file"}:
            return
        path_str = str(args.get("file_path") or args.get("path") or "")
        display_path = format_display_path(path_str)
        record = FileOperationRecord(
            tool_name=tool_name,
            display_path=display_path,
            physical_path=resolve_physical_path(path_str, self.assistant_id),
            tool_call_id=tool_call_id,
            args=args,
        )
        if tool_name in {"write_file", "edit_file"}:
            if self.backend and path_str:
                try:
                    responses = self.backend.download_files([path_str])
                    if (
                        responses
                        and responses[0].content is not None
                        and responses[0].error is None
                    ):
                        record.before_content = responses[0].content.decode("utf-8")
                    else:
                        record.before_content = ""
                except (OSError, UnicodeDecodeError, AttributeError) as e:
                    logger.debug(
                        "Failed to read before_content for %s: %s", path_str, e
                    )
                    record.before_content = ""
            elif record.physical_path:
                record.before_content = _safe_read(record.physical_path) or ""
        self.active[tool_call_id] = record

    def complete_with_message(self, tool_message: Any) -> FileOperationRecord | None:  # noqa: ANN401  # Tool message type is dynamic
        """도구 메시지 결과로 파일 작업을 완료합니다.

Returns:
            완료된 FileOperationRecord 또는 일치하는 작업이 없는 경우 None입니다.

        """
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        record = self.active.get(tool_call_id)
        if record is None:
            return None

        content = tool_message.content
        if isinstance(content, list):
            # Some tool messages may return list segments; join them for analysis.
            joined = []
            for item in content:
                if isinstance(item, str):
                    joined.append(item)
                else:
                    joined.append(str(item))
            content_text = "\n".join(joined)
        else:
            content_text = str(content) if content is not None else ""

        if getattr(
            tool_message, "status", "success"
        ) != "success" or content_text.lower().startswith("error"):
            record.status = "error"
            record.error = content_text
            self._finalize(record)
            return record

        record.status = "success"

        if record.tool_name == "read_file":
            record.read_output = content_text
            lines = _count_lines(content_text)
            record.metrics.lines_read = lines
            offset = record.args.get("offset")
            limit = record.args.get("limit")
            if isinstance(offset, int):
                if offset > lines:
                    offset = 0
                record.metrics.start_line = offset + 1
                if lines:
                    record.metrics.end_line = offset + lines
            elif lines:
                record.metrics.start_line = 1
                record.metrics.end_line = lines
            if isinstance(limit, int) and lines > limit:
                record.metrics.end_line = (record.metrics.start_line or 1) + limit - 1
        else:
            # For write/edit operations, read back from backend (or local filesystem)
            self._populate_after_content(record)
            if record.after_content is None:
                record.status = "error"
                record.error = "Could not read updated file content."
                self._finalize(record)
                return record
            record.metrics.lines_written = _count_lines(record.after_content)
            before_lines = _count_lines(record.before_content or "")
            diff = compute_unified_diff(
                record.before_content or "",
                record.after_content,
                record.display_path,
                max_lines=100,
            )
            record.diff = diff
            if diff:
                additions = sum(
                    1
                    for line in diff.splitlines()
                    if line.startswith("+") and not line.startswith("+++")
                )
                deletions = sum(
                    1
                    for line in diff.splitlines()
                    if line.startswith("-") and not line.startswith("---")
                )
                record.metrics.lines_added = additions
                record.metrics.lines_removed = deletions
            elif record.tool_name == "write_file" and not (record.before_content or ""):
                record.metrics.lines_added = record.metrics.lines_written
            record.metrics.bytes_written = len(record.after_content.encode("utf-8"))
            if (
                record.diff is None
                and (record.before_content or "") != record.after_content
            ):
                record.diff = compute_unified_diff(
                    record.before_content or "",
                    record.after_content,
                    record.display_path,
                    max_lines=100,
                )
            if record.diff is None and before_lines != record.metrics.lines_written:
                record.metrics.lines_added = max(
                    record.metrics.lines_written - before_lines, 0
                )

        self._finalize(record)
        return record

    def mark_hitl_approved(self, tool_name: str, args: dict[str, Any]) -> None:
        """tool_name 및 file_path와 일치하는 작업을 HIL 승인으로 표시합니다."""
        file_path = args.get("file_path") or args.get("path")
        if not file_path:
            return

        # Mark all active records that match
        for record in self.active.values():
            if record.tool_name == tool_name:
                record_path = record.args.get("file_path") or record.args.get("path")
                if record_path == file_path:
                    record.hitl_approved = True

    def _populate_after_content(self, record: FileOperationRecord) -> None:
        # Use backend if available (works for any BackendProtocol implementation)
        if self.backend:
            try:
                file_path = record.args.get("file_path") or record.args.get("path")
                if file_path:
                    responses = self.backend.download_files([file_path])
                    if (
                        responses
                        and responses[0].content is not None
                        and responses[0].error is None
                    ):
                        record.after_content = responses[0].content.decode("utf-8")
                    else:
                        record.after_content = None
                else:
                    record.after_content = None
            except (OSError, UnicodeDecodeError, AttributeError) as e:
                logger.debug(
                    "Failed to read after_content for %s: %s",
                    record.args.get("file_path") or record.args.get("path"),
                    e,
                )
                record.after_content = None
        else:
            # Fallback: direct filesystem read when no backend provided
            if record.physical_path is None:
                record.after_content = None
                return
            record.after_content = _safe_read(record.physical_path)

    def _finalize(self, record: FileOperationRecord) -> None:
        self.completed.append(record)
        self.active.pop(record.tool_call_id, None)
