"""HITL 도구 미리보기에 사용되는 특수 텍스트 위젯입니다.

이러한 위젯은 일반 인수 목록뿐만 아니라 잘린 콘텐츠 및 차이점 통계를 포함하는 보다 풍부한 쓰기/편집 파일 미리 보기를 렌더링합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Markdown, Static

from deepagents_cli import theme

if TYPE_CHECKING:
    from textual.app import ComposeResult

# Constants for display limits
_MAX_VALUE_LEN = 200
_MAX_LINES = 30
_MAX_DIFF_LINES = 50
_MAX_PREVIEW_LINES = 20


def _format_stats(additions: int, deletions: int) -> Content:
    """추가/삭제 통계를 스타일이 지정된 콘텐츠로 포맷합니다.

    Args:
        additions: 추가된 줄 수입니다.
        deletions: 제거된 줄 수입니다.

    Returns:
        추가 및 삭제를 표시하는 스타일 콘텐츠입니다.

    """
    colors = theme.get_theme_colors()
    parts: list[str | tuple[str, str] | Content] = []
    if additions:
        parts.append((f"+{additions}", colors.success))
    if deletions:
        if parts:
            parts.append(" ")
        parts.append((f"-{deletions}", colors.error))
    return Content.assemble(*parts) if parts else Content("")


def _file_header(
    file_path: str, additions: int = 0, deletions: int = 0
) -> ComposeResult:
    """선택적 `+N -M` 통계를 사용하여 `File:` 경로 헤더를 생성합니다.

    Args:
        file_path: 수정 중인 파일의 경로입니다.
        additions: 추가된 줄 수입니다.
        deletions: 제거된 줄 수입니다.

    Yields:
        파일 경로 헤더 및 공백 줄에 대한 정적 위젯입니다.

    """
    stats = _format_stats(additions, deletions)
    yield Static(
        Content.assemble(
            Content.from_markup("[bold cyan]File:[/bold cyan] $path  ", path=file_path),
            stats,
        )
    )
    yield Static("")


def _count_diff_stats(
    diff_lines: list[str], old_string: str, new_string: str
) -> tuple[int, int]:
    """diff 데이터의 추가 및 삭제 횟수를 계산합니다.

    Args:
        diff_lines: 통합된 diff 출력 라인.
        old_string: 원본 텍스트가 대체됩니다(차이가 없는 경우 대체).
        new_string: 대체 텍스트(차이가 없는 경우 대체)

    Returns:
        (추가 개수, 삭제 개수)의 튜플입니다.

    """
    if diff_lines:
        additions = sum(
            1
            for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1
            for line in diff_lines
            if line.startswith("-") and not line.startswith("---")
        )
    else:
        additions = new_string.count("\n") + 1 if new_string else 0
        deletions = old_string.count("\n") + 1 if old_string else 0
    return additions, deletions


class ToolApprovalWidget(Vertical):
    """도구 승인 위젯의 기본 클래스입니다."""

    def __init__(self, data: dict[str, Any]) -> None:
        """데이터로 도구 승인 위젯을 초기화합니다."""
        super().__init__(classes="tool-approval-widget")
        self.data = data

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """기본 작성 - 서브클래스에서 재정의됩니다.

        Yields:
            자리 표시자 메시지가 포함된 정적 위젯입니다.

        """
        yield Static("Tool details not available", classes="approval-description")


class GenericApprovalWidget(ToolApprovalWidget):
    """알 수 없는 도구에 대한 일반 승인 위젯입니다."""

    def compose(self) -> ComposeResult:
        """일반 도구 디스플레이를 구성합니다.

        Yields:
            도구 데이터의 각 키-값 쌍을 표시하는 정적 위젯입니다.

        """
        for key, value in self.data.items():
            if value is None:
                continue
            value_str = str(value)
            if len(value_str) > _MAX_VALUE_LEN:
                hidden = len(value_str) - _MAX_VALUE_LEN
                value_str = value_str[:_MAX_VALUE_LEN] + f"... ({hidden} more chars)"
            yield Static(
                f"{key}: {value_str}", markup=False, classes="approval-description"
            )


class WriteFileApprovalWidget(ToolApprovalWidget):
    """write_file에 대한 승인 위젯 - 구문 강조를 사용하여 파일 콘텐츠를 표시합니다."""

    def compose(self) -> ComposeResult:
        """구문 강조를 사용하여 파일 내용 표시를 구성합니다.

        Yields:
            파일 경로 헤더와 구문 강조 콘텐츠를 표시하는 위젯입니다.

        """
        file_path = self.data.get("file_path", "")
        content = self.data.get("content", "")
        file_extension = self.data.get("file_extension", "text")

        # Content with syntax highlighting via Markdown code block
        lines = content.split("\n")
        total_lines = len(lines)

        # File header with line count
        yield from _file_header(file_path, additions=total_lines if content else 0)

        if total_lines > _MAX_LINES:
            # Truncate for display
            shown_lines = lines[:_MAX_LINES]
            remaining = total_lines - _MAX_LINES
            truncated_content = (
                "\n".join(shown_lines) + f"\n... ({remaining} more lines)"
            )
            yield Markdown(f"```{file_extension}\n{truncated_content}\n```")
        else:
            yield Markdown(f"```{file_extension}\n{content}\n```")


class EditFileApprovalWidget(ToolApprovalWidget):
    """edit_file에 대한 승인 위젯 - 색상과 함께 깔끔한 차이점을 표시합니다."""

    def compose(self) -> ComposeResult:
        """색상 추가 및 삭제로 diff 디스플레이를 구성합니다.

        Yields:
            파일 경로, 통계, 색상 차이 선을 표시하는 위젯입니다.

        """
        file_path = self.data.get("file_path", "")
        diff_lines = self.data.get("diff_lines", [])
        old_string = self.data.get("old_string", "")
        new_string = self.data.get("new_string", "")

        additions, deletions = _count_diff_stats(diff_lines, old_string, new_string)
        yield from _file_header(file_path, additions, deletions)

        if not diff_lines and not old_string and not new_string:
            yield Static("No changes to display", classes="approval-description")
        elif diff_lines:
            # Render content
            yield from self._render_diff_lines_only(diff_lines)
        else:
            yield from self._render_strings_only(old_string, new_string)

    def _render_diff_lines_only(self, diff_lines: list[str]) -> ComposeResult:
        """통계를 반환하지 않고 통합 diff 라인을 렌더링합니다.

        Yields:
            적절한 스타일을 적용한 각 차이점 줄에 대한 정적 위젯.

        """
        lines_shown = 0

        for line in diff_lines:
            if lines_shown >= _MAX_DIFF_LINES:
                yield Static(
                    Content.styled(
                        f"... ({len(diff_lines) - lines_shown} more lines)", "dim"
                    )
                )
                break

            if line.startswith(("@@", "---", "+++")):
                continue

            widget = self._render_diff_line(line)
            if widget:
                yield widget
                lines_shown += 1

    def _render_strings_only(self, old_string: str, new_string: str) -> ComposeResult:
        """통계를 반환하지 않고 이전/새 문자열을 렌더링합니다.

        Yields:
            스타일을 적용하여 제거 및 추가된 콘텐츠를 보여주는 정적 위젯입니다.

        """
        colors = theme.get_theme_colors()
        if old_string:
            yield Static(Content.styled("Removing:", f"bold {colors.error}"))
            yield from self._render_string_lines(old_string, is_addition=False)
            yield Static("")

        if new_string:
            yield Static(Content.styled("Adding:", f"bold {colors.success}"))
            yield from self._render_string_lines(new_string, is_addition=True)

    @staticmethod
    def _render_diff_line(line: str) -> Static | None:
        """적절한 스타일로 단일 차이점 라인을 렌더링합니다.

        Returns:
            스타일이 지정된 diff 라인이 있는 정적 위젯 또는 비어 있거나 건너뛴 라인의 경우 None입니다.

        """
        raw = line[1:] if len(line) > 1 else ""

        if line.startswith("-"):
            return Static(
                Content.from_markup("- $text", text=raw), classes="diff-removed"
            )
        if line.startswith("+"):
            return Static(
                Content.from_markup("+ $text", text=raw), classes="diff-added"
            )
        if line.startswith(" "):
            return Static(
                Content.from_markup("  $text", text=raw), classes="diff-context"
            )
        if line.strip():
            return Static(line, markup=False)
        return None

    @staticmethod
    def _render_string_lines(text: str, *, is_addition: bool) -> ComposeResult:
        """적절한 스타일을 사용하여 문자열에서 선을 렌더링합니다.

        Yields:
            추가 또는 삭제 스타일이 포함된 각 줄의 정적 위젯.

        """
        lines = text.split("\n")
        sign = "+" if is_addition else "-"
        cls = "diff-added" if is_addition else "diff-removed"

        for line in lines[:_MAX_PREVIEW_LINES]:
            yield Static(Content.from_markup(f"{sign} $text", text=line), classes=cls)

        if len(lines) > _MAX_PREVIEW_LINES:
            remaining = len(lines) - _MAX_PREVIEW_LINES
            yield Static(Content.styled(f"... ({remaining} more lines)", "dim"))
