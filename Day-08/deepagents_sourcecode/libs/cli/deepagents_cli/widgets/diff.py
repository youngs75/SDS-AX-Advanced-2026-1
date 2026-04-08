"""통합 diff를 테마 인식 텍스트 위젯으로 렌더링합니다.

여기서 도우미는 원시 diff 텍스트를 스타일이 지정된 라인 위젯으로 분할하여 승인 및 메시지 보기가 동일한 프리젠테이션 논리를 재사용할 수 있도록 합니다.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from deepagents_cli import theme
from deepagents_cli.config import get_glyphs, is_ascii_mode

if TYPE_CHECKING:
    from textual.app import ComposeResult


def compose_diff_lines(
    diff: str,
    max_lines: int | None = 100,
) -> ComposeResult:
    """통합 diff를 위한 라인당 정적 위젯을 생성합니다.

    추가/제거된 각 줄은 CSS 클래스(`.diff-line-added`, `.diff-line-removed`)를 가져오므로 배경색은 CSS 변수에 의해
    결정되고 테마 변경 시 자동으로 업데이트됩니다.

Args:
        diff: 통합된 diff 문자열.
        max_lines: 표시할 최대 차이점 라인 수(무제한인 경우 없음).

Yields:
        적절한 CSS 클래스가 포함된 정적 위젯(diff 라인당 하나).

    """
    if not diff:
        yield Static(Content.styled("No changes detected", "dim"))
    else:
        yield from _compose_diff_content(diff, max_lines)


def _compose_diff_content(
    diff: str,
    max_lines: int | None,
) -> ComposeResult:
    """비어 있지 않은 diff 콘텐츠에 대한 스타일 diff 라인 위젯을 생성합니다.

Args:
        diff: 비어 있지 않은 통합 diff 문자열입니다.
        max_lines: 표시할 최대 차이점 라인 수(무제한인 경우 없음).

Yields:
        통계 헤더 및 개별 차이점 줄을 위한 정적 위젯입니다.

    """
    colors = theme.get_theme_colors()
    glyphs = get_glyphs()
    lines = diff.splitlines()

    # Compute stats first
    additions = sum(
        1 for ln in lines if ln.startswith("+") and not ln.startswith("+++")
    )
    deletions = sum(
        1 for ln in lines if ln.startswith("-") and not ln.startswith("---")
    )

    # Stats header
    stats_parts: list[str | tuple[str, str] | Content] = []
    if additions:
        stats_parts.append((f"+{additions}", colors.success))
    if deletions:
        if stats_parts:
            stats_parts.append(" ")
        stats_parts.append((f"-{deletions}", colors.error))
    if stats_parts:
        yield Static(Content.assemble(*stats_parts))

    # Find max line number for width calculation
    max_line = 0
    for line in lines:
        if m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            max_line = max(max_line, int(m.group(1)), int(m.group(2)))
    width = max(3, len(str(max_line + len(lines))))

    old_num = new_num = 0
    line_count = 0

    for line in lines:
        if max_lines and line_count >= max_lines:
            yield Static(
                Content.styled(f"\n... ({len(lines) - line_count} more lines)", "dim")
            )
            break

        # Skip file headers (--- and +++)
        if line.startswith(("---", "+++")):
            continue

        # Handle hunk headers - just update line numbers, don't display
        if m := re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line):
            old_num, new_num = int(m.group(1)), int(m.group(2))
            continue

        # Handle diff lines - use gutter bar instead of +/- prefix
        content = line[1:] if line else ""

        if line.startswith("-"):
            # Deletion — red gutter bar, background via CSS
            yield Static(
                Content.assemble(
                    (f"{glyphs.gutter_bar}", f"{colors.error} bold"),
                    (f"{old_num:>{width}}", "dim"),
                    f" {content}",
                ),
                classes="diff-line-removed",
            )
            old_num += 1
            line_count += 1
        elif line.startswith("+"):
            # Addition — green gutter bar, background via CSS
            yield Static(
                Content.assemble(
                    (f"{glyphs.gutter_bar}", f"{colors.success} bold"),
                    (f"{new_num:>{width}}", "dim"),
                    f" {content}",
                ),
                classes="diff-line-added",
            )
            new_num += 1
            line_count += 1
        elif line.startswith(" "):
            # Context line — dim gutter
            yield Static(
                Content.assemble(
                    (f"{glyphs.box_vertical}{old_num:>{width}}", "dim"),
                    f"  {content}",
                ),
            )
            old_num += 1
            new_num += 1
            line_count += 1
        elif line.strip() == "...":
            # Truncation marker
            yield Static(Content.styled("...", "dim"))
            line_count += 1
        else:
            # Unrecognized diff line (e.g., "\ No newline at end of file")
            yield Static(Content.styled(line, "dim"))
            line_count += 1


class EnhancedDiff(Vertical):
    """구문 강조가 포함된 통합 diff를 표시하기 위한 위젯입니다."""

    DEFAULT_CSS = """
    EnhancedDiff {
        height: auto;
        padding: 1;
        background: $surface-darken-1;
        border: round $primary;
    }

    EnhancedDiff .diff-title {
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    EnhancedDiff .diff-content {
        height: auto;
    }

    EnhancedDiff .diff-stats {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        diff: str,
        title: str = "Diff",
        max_lines: int | None = 100,
        **kwargs: Any,
    ) -> None:
        """diff 위젯을 초기화합니다.

Args:
            diff: 통합 diff 문자열
            title: 차이점 위에 표시할 제목
            max_lines: 표시할 최대 차이점 선 수
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        self._diff = diff
        self._title = title
        self._max_lines = max_lines
        self._stats = self._compute_stats()

    def _compute_stats(self) -> tuple[int, int]:
        """추가 및 삭제 횟수를 계산합니다.

Returns:
            (추가 개수, 삭제 개수)의 튜플입니다.

        """
        additions = 0
        deletions = 0
        for line in self._diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        return additions, deletions

    def on_mount(self) -> None:
        """문자셋 모드에 따라 테두리 스타일을 설정합니다."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.primary)

    def compose(self) -> ComposeResult:
        """diff 위젯 레이아웃을 구성합니다.

Yields:
            제목, 형식화된 차이점 콘텐츠, 통계용 위젯입니다.

        """
        colors = theme.get_theme_colors(self)
        glyphs = get_glyphs()
        h = glyphs.box_double_horizontal
        yield Static(
            Content.styled(
                f"{h}{h}{h} {self._title} {h}{h}{h}", f"bold {colors.primary}"
            ),
            classes="diff-title",
        )

        yield from compose_diff_lines(self._diff, self._max_lines)

        additions, deletions = self._stats
        if additions or deletions:
            content_parts: list[str | tuple[str, str]] = []
            if additions:
                content_parts.append((f"+{additions}", colors.success))
            if deletions:
                if content_parts:
                    content_parts.append(" ")
                content_parts.append((f"-{deletions}", colors.error))
            yield Static(Content.assemble(*content_parts), classes="diff-stats")
