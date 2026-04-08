"""텍스트 UI에 표시된 이기종 채팅 내용을 렌더링합니다.

이 모듈에는 사용자 프롬프트, 보조 응답, 도구 호출, 차이점, 앱 알림 및 관련 서식 지정 도우미에 대한 메시지 위젯이 포함되어 있습니다.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

from textual import on
from textual.containers import Vertical
from textual.content import Content
from textual.events import Click
from textual.reactive import var
from textual.widgets import Static

from deepagents_cli import theme
from deepagents_cli.config import (
    MODE_DISPLAY_GLYPHS,
    PREFIX_TO_MODE,
    get_glyphs,
    is_ascii_mode,
)
from deepagents_cli.formatting import format_duration
from deepagents_cli.input import EMAIL_PREFIX_PATTERN, INPUT_HIGHLIGHT_PATTERN
from deepagents_cli.tool_display import format_tool_display
from deepagents_cli.widgets._links import open_style_link
from deepagents_cli.widgets.diff import compose_diff_lines

if TYPE_CHECKING:
    from textual.app import ComposeResult
    from textual.timer import Timer
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownStream

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared formatting helpers used by multiple message widgets
# ---------------------------------------------------------------------------

def _show_timestamp_toast(widget: Static | Vertical) -> None:
    """메시지 생성 타임스탬프와 함께 토스트를 표시합니다.

    위젯이 마운트되지 않았거나 저장소에 연결된 메시지 데이터가 없는 경우 자동으로 작동하지 않습니다.

Args:
        widget: 타임스탬프를 표시할 메시지 위젯입니다.

    """
    from datetime import UTC, datetime

    try:
        app = widget.app
    except Exception:  # noqa: BLE001  # Textual raises when widget has no app
        return
    if not widget.id:
        return
    store = app._message_store  # type: ignore[attr-defined]
    data = store.get_message(widget.id)
    if not data:
        return
    dt = datetime.fromtimestamp(data.timestamp, tz=UTC).astimezone()
    label = f"{dt:%b} {dt.day}, {dt.hour % 12 or 12}:{dt:%M:%S} {dt:%p}"
    app.notify(label, timeout=3)


class _TimestampClickMixin:
    """클릭 시 타임스탬프 토스트를 표시하는 믹스인입니다.

    클릭 시 생성 타임스탬프를 표시해야 하는 메시지 위젯에 추가합니다. 추가 클릭 동작이 필요한 위젯(예: `ToolCallMessage`,
    `AppMessage`)은 `on_click`을 재정의하고 대신 `_show_timestamp_toast`을 직접 호출해야 합니다.

    """

    def on_click(self, event: Click) -> None:  # noqa: ARG002  # Textual event handler
        """클릭 시 타임스탬프 토스트를 표시합니다."""
        _show_timestamp_toast(self)  # type: ignore[arg-type]


def _mode_color(mode: str | None, widget_or_app: object | None = None) -> str:
    """모드에 대한 16진수 색상 문자열을 반환하고 기본으로 돌아갑니다.

Args:
        mode: 모드 이름(예: `'shell'`, `'command'`) 또는 `None`.
        widget_or_app: 테마 인식 조회를 위한 텍스트 위젯 또는 `App`.

Returns:
        활성 테마 `ThemeColors`의 색상 문자열입니다.

    """
    colors = theme.get_theme_colors(widget_or_app)
    if not mode:
        return colors.primary
    if mode == "shell":
        return colors.mode_bash
    if mode == "command":
        return colors.mode_command
    logger.warning("Missing color for mode '%s'; falling back to primary.", mode)
    return colors.primary


@dataclass(frozen=True, slots=True)
class FormattedOutput:
    """표시를 위한 서식 지정 도구 출력 결과입니다."""

    content: Content
    """형식화된 출력을 위해 `Content` 스타일이 지정되었습니다."""

    truncation: str | None = None
    """잘린 내용에 대한 설명(예: "10줄 더") 또는 없는 경우 없음
    잘림이 발생했습니다.
    """


# Maximum number of tool arguments to display inline
_MAX_INLINE_ARGS = 3

# Truncation limits for display
_MAX_TODO_CONTENT_LEN = 70
_MAX_WEB_CONTENT_LEN = 100

# Tools that have their key info already in the header (no need for args line)
_TOOLS_WITH_HEADER_INFO: set[str] = {
    # Filesystem tools
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",  # sandbox shell
    # Shell tools
    "shell",  # local shell
    # Web tools
    "web_search",
    "fetch_url",
    # Agent tools
    "task",
    "write_todos",
}


_SUCCESS_EXIT_RE = re.compile(r"\n?\[Command succeeded with exit code 0\]\s*$")
"""도구 출력에서 ​​SDK의 `[Command succeeded with exit code 0]` 예고편을 제거합니다."""


def _strip_success_exit_line(text: str) -> str:
    """`[Command succeeded with exit code 0]` 트레일러를 제거하세요.

    0이 아닌 종료 코드는 그대로 유지됩니다(`set_error`을 통해 제공됨).

Args:
        text: 원시 도구 출력 문자열입니다.

Returns:
        성공 종료 코드 예고편이 있는 텍스트가 제거되었습니다(있는 경우).

    """
    return _SUCCESS_EXIT_RE.sub("", text)


# ---------------------------------------------------------------------------
# Transcript widgets for user and skill-originated messages
# ---------------------------------------------------------------------------

class UserMessage(_TimestampClickMixin, Static):
    """사용자 메시지를 표시하는 위젯입니다."""

    DEFAULT_CSS = """
    UserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $primary;
    }
    """

    def __init__(self, content: str, **kwargs: Any) -> None:
        """사용자 메시지를 초기화합니다.

Args:
            content: 메시지 내용
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        self._content = content

    def on_mount(self) -> None:
        """모드별 테두리 및 ASCII 테두리 유형에 대한 CSS 클래스를 추가합니다."""
        mode = PREFIX_TO_MODE.get(self._content[:1]) if self._content else None
        if mode:
            self.add_class(f"-mode-{mode}")
        if is_ascii_mode():
            self.add_class("-ascii")

    def render(self) -> Content:
        """스타일이 지정된 사용자 메시지를 렌더링합니다.

Returns:
            모드 접두사와 강조 표시된 언급이 있는 스타일 콘텐츠입니다.

        """
        colors = theme.get_theme_colors(self)
        parts: list[str | tuple[str, str]] = []
        content = self._content

        # Use mode-specific prefix indicator when content starts with a
        # mode trigger character (e.g. "!" for shell, "/" for commands).
        # The display glyph may differ from the trigger (e.g. "$" for shell).
        mode = PREFIX_TO_MODE.get(content[:1]) if content else None
        if mode:
            glyph = MODE_DISPLAY_GLYPHS.get(mode, content[0])
            parts.append((f"{glyph} ", f"bold {_mode_color(mode, self)}"))
            content = content[1:]
        else:
            parts.append(("> ", f"bold {colors.primary}"))

        # Highlight @mentions and /commands in the content
        last_end = 0
        for match in INPUT_HIGHLIGHT_PATTERN.finditer(content):
            start, end = match.span()
            token = match.group()

            # Skip @mentions that look like email addresses
            if token.startswith("@") and start > 0:
                char_before = content[start - 1]
                if EMAIL_PREFIX_PATTERN.match(char_before):
                    continue

            # Add text before the match (unstyled)
            if start > last_end:
                parts.append(content[last_end:start])

            # The regex only matches tokens starting with / or @
            if token.startswith("/") and start == 0:
                # /command at start
                parts.append((token, f"bold {colors.warning}"))
            elif token.startswith("@"):
                # @file mention
                parts.append((token, f"bold {colors.primary}"))
            last_end = end

        # Add remaining text after last match
        if last_end < len(content):
            parts.append(content[last_end:])

        return Content.assemble(*parts)


class QueuedUserMessage(Static):
    """대기 중인(보류 중인) 사용자 메시지를 회색으로 표시하는 위젯입니다.

    이는 메시지가 대기열에서 제거되면 제거되는 임시 위젯입니다.

    """

    DEFAULT_CSS = """
    QueuedUserMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $panel;
        opacity: 0.6;
    }
    """
    """희미한 테두리 + 불투명도 감소로 대기열에 있는 메시지와 보낸 메시지를 구분할 수 있습니다."""

    def __init__(self, content: str, **kwargs: Any) -> None:
        """대기 중인 사용자 메시지를 초기화합니다.

Args:
            content: 메시지 내용
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        self._content = content

    def on_mount(self) -> None:
        """ASCII 모드에 있을 때 ASCII 테두리 클래스를 추가합니다."""
        if is_ascii_mode():
            self.add_class("-ascii")

    def render(self) -> Content:
        """대기 중인 사용자 메시지를 렌더링합니다(회색으로 표시됨).

Returns:
            흐리게 표시된 접두사와 본문이 있는 스타일이 지정된 콘텐츠입니다.

        """
        colors = theme.get_theme_colors(self)
        content = self._content
        mode = PREFIX_TO_MODE.get(content[:1]) if content else None
        if mode:
            glyph = MODE_DISPLAY_GLYPHS.get(mode, content[0])
            prefix = (f"{glyph} ", f"bold {colors.muted}")
            content = content[1:]
        else:
            prefix = ("> ", f"bold {colors.muted}")
        return Content.assemble(prefix, (content, colors.muted))


def _strip_frontmatter(text: str) -> str:
    """`---` 마커로 구분된 YAML 머리말을 제거합니다.

Args:
        text: 원시 `SKILL.md` 콘텐츠.

Returns:
        머리말이 제거되고 선행 공백이 제거된 본문 텍스트입니다.

    """
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return text
    # Find closing --- (skip the opening line)
    end = stripped.find("\n---", 3)
    if end == -1:
        return text
    # Skip past the closing --- and its trailing newline
    after = end + 4  # len("\n---")
    return stripped[after:].lstrip("\n")


class _SkillToggle(Static):
    """스킬 본체 확장을 전환하기 위한 클릭 가능한 헤더/힌트 영역입니다.

    `SkillMessage._on_toggle_click`의 `@on(Click)` CSS 선택기에서 이름으로 참조됩니다. 이름을 신중하게 바꾸세요.

    """


class SkillMessage(Vertical):
    """접을 수 있는 몸체로 스킬 호출을 표시하는 위젯입니다.

    스킬 이름, 소스 배지, 설명 및 사용자 인수를 압축 헤더로 표시합니다. 전체 SKILL.md 본문(머리말 제거됨)은 미리보기/확장 토글(클릭 또는
    Ctrl+O) 뒤에 숨겨져 있습니다.  확장된 보기는 단일 `Static` 위젯 내에서 Rich의 `Markdown`을 통해 마크다운을 렌더링합니다.

    가시성은 텍스트 반응형 `var`을 통해 전환되는 CSS 클래스(`-expanded`)에 의해 결정됩니다. 클릭 핸들러의 범위는 헤더 및 힌트
    위젯(`_SkillToggle`)으로 지정되므로 렌더링된 마크다운 본문을 클릭해도 확장 토글(예: 텍스트 선택 유지)이 트리거되지 않습니다.

    """

    DEFAULT_CSS = """
    SkillMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $skill;
    }

    SkillMessage .skill-header {
        height: auto;
    }

    SkillMessage .skill-description {
        color: $text-muted;
        margin-left: 3;
    }

    SkillMessage .skill-args {
        margin-left: 3;
        margin-top: 0;
    }

    SkillMessage #skill-md {
        margin-left: 3;
        margin-top: 0;
        padding: 0;
        display: none;
    }

    SkillMessage .skill-hint {
        margin-left: 3;
        color: $text-muted;
    }

    SkillMessage.-expanded #skill-md {
        display: block;
    }

    SkillMessage:hover {
        border-left: wide $skill-hover;
    }
    """

    _PREVIEW_LINES = 4
    _PREVIEW_CHARS = 300

    _expanded: var[bool] = var(False, toggle_class="-expanded")

    def __init__(
        self,
        skill_name: str,
        description: str = "",
        source: str = "",
        body: str = "",
        args: str = "",
        **kwargs: Any,
    ) -> None:
        """스킬 메시지를 초기화합니다.

Args:
            skill_name: 스킬 식별자.
            description: 스킬에 대한 간략한 설명입니다.
            source: 원산지 라벨(예: `'built-in'`, `'user'`)
            body: 전체 SKILL.md 콘텐츠(머리말 포함).
            args: 사용자가 제공한 인수입니다.
            **kwargs: 추가 인수가 부모에게 전달되었습니다.

        """
        super().__init__(**kwargs)
        self._skill_name = skill_name
        self._description = description
        self._source = source
        self._body = body
        self._stripped_body = _strip_frontmatter(body)
        self._args = args
        self._md_widget: Static | None = None
        self._hint_widget: _SkillToggle | None = None
        self._deferred_expanded: bool = False
        self._md_rendered: bool = False

    def compose(self) -> ComposeResult:
        """스킬 메시지 레이아웃을 구성합니다.

Yields:
            헤더, 설명, 인수 및 축소 가능한 본문에 대한 위젯입니다.

        """
        colors = theme.get_theme_colors()
        source_tag = f" [{self._source}]" if self._source else ""
        yield _SkillToggle(
            Content.styled(
                f"/ skill:{self._skill_name}{source_tag}",
                f"bold {colors.skill}",
            ),
            classes="skill-header",
        )
        if self._description:
            yield _SkillToggle(
                Content.styled(self._description, "dim"),
                classes="skill-description",
            )
        if self._args:
            yield Static(
                Content.assemble(
                    ("User request: ", "bold"),
                    self._args,
                ),
                classes="skill-args",
            )
        yield Static("", id="skill-md")
        yield _SkillToggle("", classes="skill-hint", id="skill-hint")

    def on_mount(self) -> None:
        """위젯 참조를 캐시하고 초기 상태를 렌더링합니다.

        순서 문제: 위젯 참조는 `_prepare_body` 또는 `_deferred_expanded` 할당 전에 캐시되어야 합니다. 둘 중 하나가
        동기적으로 `watch__expanded`을 실행하는 `_expanded`을 설정할 수 있기 때문입니다.

        """
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border_left = ("ascii", colors.skill)

        self._md_widget = self.query_one("#skill-md", Static)
        self._hint_widget = self.query_one("#skill-hint", _SkillToggle)

        body = self._stripped_body.strip()
        if body:
            self._prepare_body(body)

        if self._deferred_expanded:
            self._expanded = self._deferred_expanded
            self._deferred_expanded = False

    def _prepare_body(self, body: str) -> None:
        """초기 힌트 텍스트를 설정합니다. 전신 렌더링은 먼저 확장될 때까지 연기됩니다.

Args:
            body: 마크다운 본문 텍스트가 제거되었습니다.

        """
        lines = body.split("\n")
        total_lines = len(lines)
        needs_truncation = (
            total_lines > self._PREVIEW_LINES or len(body) > self._PREVIEW_CHARS
        )

        if needs_truncation:
            remaining = total_lines - self._PREVIEW_LINES
            ellipsis = get_glyphs().ellipsis
            if self._hint_widget:
                self._hint_widget.update(
                    Content.styled(
                        f"{ellipsis} {remaining} more lines"
                        " — click or Ctrl+O to expand",
                        "dim",
                    )
                )
        else:
            # Short body — show fully rendered, no preview needed.
            self._ensure_md_rendered(body)
            self._expanded = True

    def _ensure_md_rendered(self, body: str) -> None:
        """첫 번째 호출 시 정적 위젯에 마크다운을 렌더링한 다음 작동하지 않습니다.

Args:
            body: 마크다운 본문 텍스트가 제거되었습니다.

        """
        if self._md_rendered or not self._md_widget:
            return
        try:
            from rich.markdown import Markdown as RichMarkdown

            self._md_widget.update(RichMarkdown(body))
        except Exception:
            logger.warning(
                "Failed to render skill body as markdown; falling back to plain text",
                exc_info=True,
            )
            self._md_widget.update(body)
        self._md_rendered = True

    def toggle_body(self) -> None:
        """미리보기와 전신 표시 사이를 전환합니다."""
        if not self._stripped_body.strip():
            return
        self._expanded = not self._expanded

    def watch__expanded(self, expanded: bool) -> None:
        """첫 번째 확장 시 지연 렌더링 가격 인하; 힌트 텍스트를 업데이트하세요."""
        body = self._stripped_body.strip()
        if not body:
            return

        if expanded:
            self._ensure_md_rendered(body)

        if not self._hint_widget:
            return

        lines = body.split("\n")
        total_lines = len(lines)
        needs_truncation = (
            total_lines > self._PREVIEW_LINES or len(body) > self._PREVIEW_CHARS
        )

        if not needs_truncation:
            # Short body — always fully visible, no hint needed.
            self._hint_widget.display = False
            return

        if expanded:
            self._hint_widget.update(
                Content.styled("click or Ctrl+O to collapse", "dim italic")
            )
        else:
            remaining = total_lines - self._PREVIEW_LINES
            ellipsis = get_glyphs().ellipsis
            self._hint_widget.update(
                Content.styled(
                    f"{ellipsis} {remaining} more lines — click or Ctrl+O to expand",
                    "dim",
                )
            )

    @on(Click, "_SkillToggle")
    def _on_toggle_click(self, event: Click) -> None:
        """헤더나 힌트를 클릭하면 확장이 전환됩니다."""
        event.stop()
        if self._stripped_body.strip():
            self.toggle_body()
        else:
            _show_timestamp_toast(self)


# ---------------------------------------------------------------------------
# Assistant and tool-output widgets for streamed agent activity
# ---------------------------------------------------------------------------

class AssistantMessage(_TimestampClickMixin, Vertical):
    """마크다운을 지원하는 보조 메시지를 표시하는 위젯입니다.

    각 업데이트에서 전체 콘텐츠를 다시 렌더링하는 대신 보다 원활한 스트리밍을 위해 MarkdownStream을 사용합니다.

    """

    DEFAULT_CSS = """
    AssistantMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    AssistantMessage Markdown {
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, content: str = "", **kwargs: Any) -> None:
        """보조 메시지를 초기화합니다.

Args:
            content: 초기 마크다운 콘텐츠
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        self._content = content
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None

    def compose(self) -> ComposeResult:  # noqa: PLR6301  # Textual widget method convention
        """보조자 메시지 레이아웃을 구성합니다.

Yields:
            어시스턴트 콘텐츠 렌더링을 위한 마크다운 위젯.

        """
        from textual.widgets import Markdown

        yield Markdown("", id="assistant-content")

    def on_mount(self) -> None:
        """마크다운 위젯에 대한 참조를 저장합니다."""
        from textual.widgets import Markdown

        self._markdown = self.query_one("#assistant-content", Markdown)

    def _get_markdown(self) -> Markdown:
        """캐시되지 않은 경우 쿼리하여 마크다운 위젯을 가져옵니다.

Returns:
            이 메시지에 대한 마크다운 위젯입니다.

        """
        if self._markdown is None:
            from textual.widgets import Markdown

            self._markdown = self.query_one("#assistant-content", Markdown)
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        """마크다운 스트림이 초기화되었는지 확인하세요.

Returns:
            콘텐츠 스트리밍을 위한 MarkdownStream 인스턴스입니다.

        """
        if self._stream is None:
            from textual.widgets import Markdown

            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    async def append_content(self, text: str) -> None:
        """메시지에 콘텐츠를 추가합니다(스트리밍용).

        각 청크의 전체 콘텐츠를 다시 렌더링하는 대신 더 부드러운 렌더링을 위해 MarkdownStream을 사용합니다.

Args:
            text: 추가할 텍스트

        """
        if not text:
            return
        self._content += text
        stream = self._ensure_stream()
        await stream.write(text)

    async def write_initial_content(self) -> None:
        """구축 시 제공되는 경우 초기 콘텐츠를 작성합니다."""
        if self._content:
            stream = self._ensure_stream()
            await stream.write(self._content)

    async def stop_stream(self) -> None:
        """스트리밍을 중지하고 콘텐츠를 마무리하세요."""
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None

    async def set_content(self, content: str) -> None:
        """전체 메시지 내용을 설정합니다.

        이렇게 하면 활성 스트림이 중지되고 콘텐츠가 직접 설정됩니다.

Args:
            content: 표시할 마크다운 콘텐츠

        """
        await self.stop_stream()
        self._content = content
        if self._markdown:
            await self._markdown.update(content)


class ToolCallMessage(Vertical):
    """축소 가능한 출력으로 도구 호출을 표시하는 위젯입니다.

    도구 출력은 기본적으로 3줄 미리보기로 표시됩니다. 전체 출력을 확장/축소하려면 Ctrl+O를 누르세요. 도구가 실행되는 동안 애니메이션 "실행
    중..." 표시기를 표시합니다.

    """

    DEFAULT_CSS = """
    ToolCallMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border-left: wide $tool;
    }

    ToolCallMessage .tool-header {
        height: auto;
        color: $tool;
        text-style: bold;
    }

    ToolCallMessage .tool-task-desc {
        color: $text-muted;
        margin-left: 3;
        text-style: italic;
    }

    ToolCallMessage .tool-args {
        color: $text-muted;
        margin-left: 3;
    }

    ToolCallMessage .tool-status {
        margin-left: 3;
    }

    ToolCallMessage .tool-status.pending {
        color: $warning;
    }

    ToolCallMessage .tool-status.success {
        color: $success;
    }

    ToolCallMessage .tool-status.error {
        color: $error;
    }

    ToolCallMessage .tool-status.rejected {
        color: $warning;
    }

    ToolCallMessage .tool-output {
        margin-left: 0;
        margin-top: 0;
        padding: 0;
        height: auto;
    }

    ToolCallMessage .tool-output-preview {
        margin-left: 0;
        margin-top: 0;
    }

    ToolCallMessage .tool-output-hint {
        margin-left: 0;
        color: $text-muted;
    }

    ToolCallMessage:hover {
        border-left: wide $tool-hover;
    }
    """
    """왼쪽 테두리는 도구 수명주기를 추적합니다. 호버는 상호 작용을 위해 밝아집니다."""

    # Max lines/chars to show in preview mode
    _PREVIEW_LINES = 6
    _PREVIEW_CHARS = 400

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """도구 호출 메시지를 초기화합니다.

Args:
            tool_name: 호출되는 도구의 이름
            args: 도구 인수(선택 사항)
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._args = args or {}
        self._status = "pending"  # Waiting for approval or auto-approve
        self._output: str = ""
        self._expanded: bool = False
        # Widget references (set in on_mount)
        self._status_widget: Static | None = None
        self._preview_widget: Static | None = None
        self._hint_widget: Static | None = None
        self._full_widget: Static | None = None
        # Animation state
        self._spinner_position = 0
        self._start_time: float | None = None
        self._animation_timer: Timer | None = None
        # Deferred state for hydration (set by MessageData.to_widget)
        self._deferred_status: str | None = None
        self._deferred_output: str | None = None
        self._deferred_expanded: bool = False

    def compose(self) -> ComposeResult:
        """도구 호출 메시지 레이아웃을 구성합니다.

Yields:
            헤더, 인수, 상태 및 출력 표시용 위젯입니다.

        """
        tool_label = format_tool_display(self._tool_name, self._args)
        yield Static(tool_label, markup=False, classes="tool-header")
        # Task: dedicated description line (dim, truncated)
        if self._tool_name == "task":
            desc = self._args.get("description", "")
            if desc:
                max_len = 120
                suffix = "..." if len(desc) > max_len else ""
                truncated = desc[:max_len].rstrip() + suffix
                yield Static(
                    Content.styled(truncated, "dim"),
                    classes="tool-task-desc",
                )
        # Only show args for tools where header doesn't capture the key info
        elif self._tool_name not in _TOOLS_WITH_HEADER_INFO:
            args = self._filtered_args()
            if args:
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in list(args.items())[:_MAX_INLINE_ARGS]
                )
                if len(args) > _MAX_INLINE_ARGS:
                    args_str += ", ..."
                yield Static(
                    Content.from_markup("[dim]($args)[/dim]", args=args_str),
                    classes="tool-args",
                )
        # Status - shows running animation while pending, then final status
        yield Static("", classes="tool-status", id="status")
        # Output area - hidden initially, shown when output is set
        yield Static("", classes="tool-output-preview", id="output-preview")
        yield Static("", classes="tool-output", id="output-full")
        yield Static("", classes="tool-output-hint", id="output-hint")

    def on_mount(self) -> None:
        """처음에는 위젯 참조를 캐시하고 모든 상태/출력 영역을 숨깁니다."""
        if is_ascii_mode():
            self.add_class("-ascii")

        self._status_widget = self.query_one("#status", Static)
        self._preview_widget = self.query_one("#output-preview", Static)
        self._hint_widget = self.query_one("#output-hint", Static)
        self._full_widget = self.query_one("#output-full", Static)
        # Hide everything initially - status only shown when running or on error/reject
        self._status_widget.display = False
        self._preview_widget.display = False
        self._hint_widget.display = False
        self._full_widget.display = False

        # Restore deferred state if this widget was hydrated from data
        self._restore_deferred_state()

    def _restore_deferred_state(self) -> None:
        """지연된 값에서 상태를 복원합니다(데이터를 하이드레이션할 때 사용됨)."""
        if self._deferred_status is None:
            return

        status = self._deferred_status
        output = self._deferred_output or ""
        self._expanded = self._deferred_expanded

        # Clear deferred values
        self._deferred_status = None
        self._deferred_output = None
        self._deferred_expanded = False

        # Restore based on status (don't restart animations for running tools)
        colors = theme.get_theme_colors(self)
        match status:
            case "success":
                self._status = "success"
                self._output = output
                self._update_output_display()
            case "error":
                self._status = "error"
                self._output = output
                if self._status_widget:
                    self._status_widget.add_class("error")
                    error_icon = get_glyphs().error
                    self._status_widget.update(
                        Content.styled(f"{error_icon} Error", colors.error)
                    )
                    self._status_widget.display = True
                self._update_output_display()
            case "rejected":
                self._status = "rejected"
                if self._status_widget:
                    self._status_widget.add_class("rejected")
                    error_icon = get_glyphs().error
                    self._status_widget.update(
                        Content.styled(f"{error_icon} Rejected", colors.warning)
                    )
                    self._status_widget.display = True
            case "skipped":
                self._status = "skipped"
                if self._status_widget:
                    self._status_widget.add_class("rejected")
                    self._status_widget.update(Content.styled("- Skipped", "dim"))
                    self._status_widget.display = True
            case "running":
                # For running tools, show static "Running..." without animation
                # (animations shouldn't be restored for archived tools)
                self._status = "running"
                if self._status_widget:
                    self._status_widget.add_class("pending")
                    frame = get_glyphs().spinner_frames[0]
                    self._status_widget.update(
                        Content.styled(f"{frame} Running...", colors.warning)
                    )
                    self._status_widget.display = True
            case _:
                # pending or unknown - leave as default
                pass

    def set_running(self) -> None:
        """도구를 실행 중(승인 및 실행 중)으로 표시합니다.

        실행 중인 애니메이션을 시작하기 위해 승인이 부여되면 이를 호출합니다.

        """
        if self._status == "running":
            return  # Already running

        self._status = "running"
        self._start_time = time()
        if self._status_widget:
            self._status_widget.add_class("pending")
            self._status_widget.display = True
        self._update_running_animation()
        self._animation_timer = self.set_interval(0.1, self._update_running_animation)

    def _update_running_animation(self) -> None:
        """실행 중인 스피너 애니메이션을 업데이트합니다."""
        if self._status != "running" or self._status_widget is None:
            return

        spinner_frames = get_glyphs().spinner_frames
        frame = spinner_frames[self._spinner_position]
        self._spinner_position = (self._spinner_position + 1) % len(spinner_frames)

        elapsed = ""
        if self._start_time is not None:
            elapsed_secs = int(time() - self._start_time)
            elapsed = f" ({format_duration(elapsed_secs)})"

        text = f"{frame} Running...{elapsed}"
        self._status_widget.update(
            Content.styled(text, theme.get_theme_colors(self).warning)
        )

    def _stop_animation(self) -> None:
        """실행 중인 애니메이션을 중지합니다."""
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

    def set_success(self, result: str = "") -> None:
        """도구 호출을 성공으로 표시합니다.

Args:
            result: 도구 출력/표시할 결과

        """
        self._stop_animation()
        self._status = "success"
        # Strip redundant success trailer — the UI already conveys success
        self._output = _strip_success_exit_line(result)
        if self._status_widget:
            self._status_widget.remove_class("pending")
            # Hide status on success - output speaks for itself
            self._status_widget.display = False
        self._update_output_display()

    def set_error(self, error: str) -> None:
        """도구 호출을 실패로 표시합니다.

Args:
            error: 오류 메시지

        """
        self._stop_animation()
        self._status = "error"
        # For shell commands, prepend the full command so users can see what failed
        command = (
            self._args.get("command")
            if self._tool_name in {"shell", "bash", "execute"}
            else None
        )
        if command and isinstance(command, str) and command.strip():
            self._output = f"$ {command}\n\n{error}"
        else:
            self._output = error
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("error")
            error_icon = get_glyphs().error
            colors = theme.get_theme_colors(self)
            self._status_widget.update(
                Content.styled(f"{error_icon} Error", colors.error)
            )
            self._status_widget.display = True
        # Always show full error - errors should be visible
        self._expanded = True
        self._update_output_display()

    def set_rejected(self) -> None:
        """사용자가 도구 호출을 거부한 것으로 표시합니다."""
        self._stop_animation()
        self._status = "rejected"
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")
            error_icon = get_glyphs().error
            text = f"{error_icon} Rejected"
            colors = theme.get_theme_colors(self)
            self._status_widget.update(Content.styled(text, colors.warning))
            self._status_widget.display = True

    def set_skipped(self) -> None:
        """도구 호출을 건너뛴 것으로 표시합니다(다른 거부로 인해)."""
        self._stop_animation()
        self._status = "skipped"
        if self._status_widget:
            self._status_widget.remove_class("pending")
            self._status_widget.add_class("rejected")  # Use same styling as rejected
            self._status_widget.update(Content.styled("- Skipped", "dim"))
            self._status_widget.display = True

    def toggle_output(self) -> None:
        """미리보기와 전체 출력 표시 사이를 전환합니다."""
        if not self._output:
            return
        self._expanded = not self._expanded
        self._update_output_display()

    def on_click(self, event: Click) -> None:
        """출력 확장을 전환하거나 출력이 없는 경우 타임스탬프를 표시합니다."""
        event.stop()  # Prevent click from bubbling up and scrolling
        if self._output:
            self.toggle_output()
        else:
            _show_timestamp_toast(self)

    def _format_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """더 나은 표시를 위해 도구 유형에 따라 도구 출력 형식을 지정합니다.

Args:
            output: 원시 출력 문자열
            is_preview: 미리보기(잘림) 표시용인지 여부

Returns:
            콘텐츠 및 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        output = output.strip()
        if not output:
            return FormattedOutput(content=Content(""))

        # Tool-specific formatting using dispatch table
        formatters = {
            "write_todos": self._format_todos_output,
            "ls": self._format_ls_output,
            "read_file": self._format_file_output,
            "write_file": self._format_file_output,
            "edit_file": self._format_file_output,
            "grep": self._format_search_output,
            "glob": self._format_search_output,
            "shell": self._format_shell_output,
            "bash": self._format_shell_output,
            "execute": self._format_shell_output,
            "web_search": self._format_web_output,
            "fetch_url": self._format_web_output,
            "task": self._format_task_output,
        }

        formatter = formatters.get(self._tool_name)
        if formatter:
            return formatter(output, is_preview=is_preview)

        if is_preview:
            # Fallback for unknown tools: use generic truncation
            lines = output.split("\n")
            if len(lines) > self._PREVIEW_LINES:
                return self._format_lines_output(lines, is_preview=True)
            if len(output) > self._PREVIEW_CHARS:
                truncated = output[: self._PREVIEW_CHARS]
                truncation = f"{len(output) - self._PREVIEW_CHARS} more chars"
                return FormattedOutput(
                    content=Content(truncated), truncation=truncation
                )

        # Default: plain text (Content treats input as literal)
        return FormattedOutput(content=Content(output))

    def _prefix_output(self, content: Content) -> Content:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """출력 표시자와 들여쓰기 연속 줄이 있는 출력 접두사.

Args:
            content: 접두어와 들여쓰기를 적용한 스타일이 지정된 출력 콘텐츠입니다.

Returns:
            `Content` 첫 번째 줄에 출력 접두사가 있고 들여쓰기됨
                계속.

        """
        if not content.plain:
            return Content("")
        output_prefix = get_glyphs().output_prefix
        lines = content.split("\n")
        prefixed = [Content.assemble(f"{output_prefix} ", lines[0])]
        prefixed.extend(Content.assemble("  ", line) for line in lines[1:])
        return Content("\n").join(prefixed)

    def _format_todos_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """write_todos 출력 형식을 체크리스트로 지정합니다.

Returns:
            체크리스트 콘텐츠와 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        items = self._parse_todo_items(output)
        if items is None:
            return FormattedOutput(content=Content(output))

        if not items:
            return FormattedOutput(content=Content.styled("    No todos", "dim"))

        lines: list[Content] = []
        max_items = 4 if is_preview else len(items)

        # Build stats header
        stats = self._build_todo_stats(items)
        if stats:
            lines.extend([Content.assemble("    ", stats), Content("")])

        # Format each item
        lines.extend(self._format_single_todo(item) for item in items[:max_items])

        truncation = None
        if is_preview and len(items) > max_items:
            truncation = f"{len(items) - max_items} more"

        return FormattedOutput(content=Content("\n").join(lines), truncation=truncation)

    def _parse_todo_items(self, output: str) -> list | None:  # noqa: PLR6301  # Grouped as method for widget cohesion
        """출력에서 할 일 항목을 구문 분석합니다.

Returns:
            할 일 항목 목록 또는 구문 분석이 실패하면 없음입니다.

        """
        list_match = re.search(r"\[(\{.*\})\]", output.replace("\n", " "), re.DOTALL)
        if list_match:
            try:
                return ast.literal_eval("[" + list_match.group(1) + "]")
            except (ValueError, SyntaxError):
                return None
        try:
            items = ast.literal_eval(output)
            return items if isinstance(items, list) else None
        except (ValueError, SyntaxError):
            return None

    def _build_todo_stats(self, items: list) -> Content:
        """할 일 목록에 대한 통계 콘텐츠를 구축합니다.

Returns:
            활성, 대기 중, 완료된 개수를 표시하는 `Content` 스타일입니다.

        """
        colors = theme.get_theme_colors(self)
        completed = sum(
            1 for i in items if isinstance(i, dict) and i.get("status") == "completed"
        )
        active = sum(
            1 for i in items if isinstance(i, dict) and i.get("status") == "in_progress"
        )
        pending = len(items) - completed - active

        parts: list[Content] = []
        if active:
            parts.append(Content.styled(f"{active} active", colors.warning))
        if pending:
            parts.append(Content.styled(f"{pending} pending", "dim"))
        if completed:
            parts.append(Content.styled(f"{completed} done", colors.success))
        return Content.styled(" | ", "dim").join(parts) if parts else Content("")

    def _format_single_todo(self, item: dict | str) -> Content:
        """단일 할 일 항목의 형식을 지정합니다.

Returns:
            체크박스와 상태 스타일을 사용하여 `Content` 스타일을 지정했습니다.

        """
        colors = theme.get_theme_colors(self)
        if isinstance(item, dict):
            text = item.get("content", str(item))
            status = item.get("status", "pending")
        else:
            text = str(item)
            status = "pending"

        if len(text) > _MAX_TODO_CONTENT_LEN:
            text = text[: _MAX_TODO_CONTENT_LEN - 3] + "..."

        glyphs = get_glyphs()
        if status == "completed":
            return Content.assemble(
                Content.styled(f"    {glyphs.checkmark} done", colors.success),
                Content.styled(f"   {text}", "dim"),
            )
        if status == "in_progress":
            return Content.assemble(
                Content.styled(f"    {glyphs.circle_filled} active", colors.warning),
                f" {text}",
            )
        return Content.assemble(
            Content.styled(f"    {glyphs.circle_empty} todo", "dim"),
            f"   {text}",
        )

    def _format_ls_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """ls 출력을 깨끗한 디렉터리 목록으로 형식화합니다.

Returns:
            디렉터리 목록과 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        # Try to parse as a Python list (common format)
        try:
            items = ast.literal_eval(output)
            if isinstance(items, list):
                lines: list[Content] = []
                max_items = 5 if is_preview else len(items)
                for item in items[:max_items]:
                    path = Path(str(item))
                    name = path.name
                    if path.suffix in {".py", ".pyx"}:
                        lines.append(Content.styled(f"    {name}", theme.FILE_PYTHON))
                    elif path.suffix in {".json", ".yaml", ".yml", ".toml"}:
                        lines.append(Content.styled(f"    {name}", theme.FILE_CONFIG))
                    elif not path.suffix:
                        lines.append(Content.styled(f"    {name}/", theme.FILE_DIR))
                    else:
                        lines.append(Content(f"    {name}"))

                truncation = None
                if is_preview and len(items) > max_items:
                    truncation = f"{len(items) - max_items} more"

                return FormattedOutput(
                    content=Content("\n").join(lines), truncation=truncation
                )
        except (ValueError, SyntaxError):
            pass

        # Fallback: plain text
        return FormattedOutput(content=Content(output))

    def _format_file_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """파일 읽기/쓰기 출력 형식을 지정합니다.

Returns:
            파일 콘텐츠와 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        parts = [Content(line) for line in lines[:max_lines]]
        content = Content("\n").join(parts)

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

    def _format_search_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """grep/glob 검색 출력 형식을 지정합니다.

Returns:
            검색 결과 및 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        # Try to parse as a Python list (glob returns list of paths)
        try:
            items = ast.literal_eval(output.strip())
            if isinstance(items, list):
                parts: list[Content] = []
                max_items = 5 if is_preview else len(items)
                for item in items[:max_items]:
                    path = Path(str(item))
                    try:
                        rel = path.relative_to(Path.cwd())
                        display = str(rel)
                    except ValueError:
                        display = path.name
                    parts.append(Content(f"    {display}"))

                truncation = None
                if is_preview and len(items) > max_items:
                    truncation = f"{len(items) - max_items} more files"

                return FormattedOutput(
                    content=Content("\n").join(parts), truncation=truncation
                )
        except (ValueError, SyntaxError):
            pass

        # Fallback: line-based output (grep results)
        lines = output.split("\n")
        max_lines = 5 if is_preview else len(lines)

        parts = [
            Content(f"    {raw_line.strip()}")
            for raw_line in lines[:max_lines]
            if raw_line.strip()
        ]

        content = Content("\n").join(parts) if parts else Content("")
        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more"

        return FormattedOutput(content=content, truncation=truncation)

    def _format_shell_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """쉘 명령 출력 형식을 지정합니다.

Returns:
            셸 출력 및 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        parts: list[Content] = []
        for i, line in enumerate(lines[:max_lines]):
            if i == 0 and line.startswith("$ "):
                parts.append(Content.styled(line, "dim"))
            else:
                parts.append(Content(line))

        content = Content("\n").join(parts) if parts else Content("")

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

    def _format_web_output(
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """web_search/fetch_url 출력 형식을 지정합니다.

Returns:
            웹 응답 및 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        data = self._try_parse_web_data(output)
        if isinstance(data, dict):
            return self._format_web_dict(data, is_preview=is_preview)

        # Fallback: plain text
        return self._format_lines_output(output.split("\n"), is_preview=is_preview)

    @staticmethod
    def _try_parse_web_data(output: str) -> dict | None:
        """웹 출력을 JSON 또는 dict로 구문 분석해 보세요.

Returns:
            성공하면 구문 분석된 dict이고, 그렇지 않으면 None입니다.

        """
        try:
            if output.strip().startswith("{"):
                return json.loads(output)
            return ast.literal_eval(output)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            return None

    def _format_web_dict(self, data: dict, *, is_preview: bool) -> FormattedOutput:
        """구문 분석된 웹 응답 사전의 형식을 지정합니다.

Returns:
            웹 응답 콘텐츠와 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        # Handle web_search results
        if "results" in data:
            return self._format_web_search_results(
                data.get("results", []), is_preview=is_preview
            )

        # Handle fetch_url response
        if "markdown_content" in data:
            lines = data["markdown_content"].split("\n")
            return self._format_lines_output(lines, is_preview=is_preview)

        # Generic dict - show key fields
        parts: list[Content] = []
        max_keys = 3 if is_preview else len(data)
        for k, v in list(data.items())[:max_keys]:
            v_str = str(v)
            if is_preview and len(v_str) > _MAX_WEB_CONTENT_LEN:
                v_str = v_str[:_MAX_WEB_CONTENT_LEN] + "..."
            parts.append(Content(f"  {k}: {v_str}"))
        truncation = None
        if is_preview and len(data) > max_keys:
            truncation = f"{len(data) - max_keys} more"
        return FormattedOutput(
            content=Content("\n").join(parts) if parts else Content(""),
            truncation=truncation,
        )

    def _format_web_search_results(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, results: list, *, is_preview: bool
    ) -> FormattedOutput:
        """웹 검색 결과의 형식을 지정합니다.

Returns:
            검색 결과 및 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        if not results:
            return FormattedOutput(content=Content.styled("No results", "dim"))
        parts: list[Content] = []
        max_results = 3 if is_preview else len(results)
        for r in results[:max_results]:
            title = r.get("title", "")
            url = r.get("url", "")
            parts.extend(
                [
                    Content.styled(f"  {title}", "bold"),
                    Content.styled(f"  {url}", "dim"),
                ]
            )
        truncation = None
        if is_preview and len(results) > max_results:
            truncation = f"{len(results) - max_results} more results"
        return FormattedOutput(content=Content("\n").join(parts), truncation=truncation)

    def _format_lines_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, lines: list[str], *, is_preview: bool
    ) -> FormattedOutput:
        """선택적 미리보기 잘림을 사용하여 행 목록의 형식을 지정합니다.

Returns:
            줄 내용과 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        max_lines = 4 if is_preview else len(lines)
        parts = [Content(line) for line in lines[:max_lines]]
        content = Content("\n").join(parts) if parts else Content("")
        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"
        return FormattedOutput(content=content, truncation=truncation)

    def _format_task_output(  # noqa: PLR6301  # Grouped as method for widget cohesion
        self, output: str, *, is_preview: bool = False
    ) -> FormattedOutput:
        """작업(하위 에이전트) 출력 형식을 지정합니다.

Returns:
            작업 출력 및 선택적 잘림 정보가 포함된 FormattedOutput입니다.

        """
        lines = output.split("\n")
        max_lines = 4 if is_preview else len(lines)

        parts = [Content(line) for line in lines[:max_lines]]
        content = Content("\n").join(parts) if parts else Content("")

        truncation = None
        if is_preview and len(lines) > max_lines:
            truncation = f"{len(lines) - max_lines} more lines"

        return FormattedOutput(content=content, truncation=truncation)

    def _update_output_display(self) -> None:
        """확장된 상태에 따라 출력 표시를 업데이트합니다."""
        # Guard: all widgets must be initialized before updating display state
        if (
            not self._output
            or not self._preview_widget
            or not self._full_widget
            or not self._hint_widget
        ):
            return

        output_stripped = self._output.strip()
        lines = output_stripped.split("\n")
        total_lines = len(lines)
        total_chars = len(output_stripped)

        # Truncate if too many lines OR too many characters
        needs_truncation = (
            total_lines > self._PREVIEW_LINES or total_chars > self._PREVIEW_CHARS
        )

        if self._expanded:
            # Show full output with formatting
            self._preview_widget.display = False
            result = self._format_output(self._output, is_preview=False)
            prefixed = self._prefix_output(result.content)
            self._full_widget.update(prefixed)
            self._full_widget.display = True
            # Show collapse hint underneath
            self._hint_widget.update(
                Content.styled("click or Ctrl+O to collapse", "dim italic")
            )
            self._hint_widget.display = True
        else:
            # Show preview
            self._full_widget.display = False
            if needs_truncation:
                result = self._format_output(self._output, is_preview=True)
                prefixed = self._prefix_output(result.content)
                self._preview_widget.update(prefixed)
                self._preview_widget.display = True

                # Build hint with truncation info if available
                if result.truncation:
                    ellipsis = get_glyphs().ellipsis
                    hint = Content.styled(
                        f"{ellipsis} {result.truncation} — click or Ctrl+O to expand",
                        "dim",
                    )
                else:
                    hint = Content.styled("click or Ctrl+O to expand", "dim italic")
                self._hint_widget.update(hint)
                self._hint_widget.display = True
            elif output_stripped:
                # Output fits in preview, show formatted
                result = self._format_output(output_stripped, is_preview=False)
                prefixed = self._prefix_output(result.content)
                self._preview_widget.update(prefixed)
                self._preview_widget.display = True
                self._hint_widget.display = False
            else:
                self._preview_widget.display = False
                self._hint_widget.display = False

    @property
    def has_output(self) -> bool:
        """이 도구 메시지에 표시할 출력이 있는지 확인하세요.

Returns:
            출력 내용이 있으면 True, 그렇지 않으면 False입니다.

        """
        return bool(self._output)

    def _filtered_args(self) -> dict[str, Any]:
        """표시할 대형 도구 인수를 필터링합니다.

Returns:
            쓰기/편집 도구에 대한 표시 관련 키만 포함된 필터링된 인수 dict입니다.

        """
        if self._tool_name not in {"write_file", "edit_file"}:
            return self._args

        filtered: dict[str, Any] = {}
        for key in ("file_path", "path", "replace_all"):
            if key in self._args:
                filtered[key] = self._args[key]
        return filtered


# ---------------------------------------------------------------------------
# Auxiliary transcript entries for diffs, errors, and app notices
# ---------------------------------------------------------------------------

class DiffMessage(_TimestampClickMixin, Static):
    """구문 강조를 통해 차이점을 표시하는 위젯입니다."""

    DEFAULT_CSS = """
    DiffMessage {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $surface;
        border: solid $primary;
    }

    DiffMessage .diff-header {
        text-style: bold;
        margin-bottom: 1;
    }

    DiffMessage .diff-add {
        color: $text-success;
        background: $success-muted;
    }

    DiffMessage .diff-remove {
        color: $text-error;
        background: $error-muted;
    }

    DiffMessage .diff-context {
        color: $text-muted;
    }

    DiffMessage .diff-hunk {
        color: $secondary;
        text-style: bold;
    }
    """
    """테마별 Diff 구문 색상 지정: 추가, 제거, 음소거된 컨텍스트."""

    def __init__(self, diff_content: str, file_path: str = "", **kwargs: Any) -> None:
        """diff 메시지를 초기화합니다.

Args:
            diff_content: 통합된 차이점 콘텐츠
            file_path: 수정 중인 파일의 경로
            **kwargs: 부모에게 전달된 추가 인수

        """
        super().__init__(**kwargs)
        self._diff_content = diff_content
        self._file_path = file_path

    def compose(self) -> ComposeResult:
        """diff 메시지 레이아웃을 구성합니다.

Yields:
            diff 헤더와 형식이 지정된 콘텐츠를 표시하는 위젯입니다.

        """
        if self._file_path:
            yield Static(
                Content.from_markup("[bold]File: $path[/bold]", path=self._file_path),
                classes="diff-header",
            )

        # Render the diff with per-line Statics (CSS-driven backgrounds)
        yield from compose_diff_lines(self._diff_content, max_lines=100)

    def on_mount(self) -> None:
        """문자셋 모드에 따라 테두리 스타일을 설정합니다."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border = ("ascii", colors.primary)


class ErrorMessage(_TimestampClickMixin, Static):
    """오류 메시지를 표시하는 위젯입니다."""

    DEFAULT_CSS = """
    ErrorMessage {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $error-muted;
        color: white;
        border-left: wide $error;
    }
    """
    """색상이 지정된 배경 + 왼쪽 테두리로 출력에서 ​​오류를 시각적으로 구분합니다."""

    def __init__(self, error: str, **kwargs: Any) -> None:
        """오류 메시지를 초기화합니다.

Args:
            error: 오류 메시지
            **kwargs: 부모에게 전달된 추가 인수

        """
        # Store raw content for serialization
        self._content = error
        super().__init__(**kwargs)

    def render(self) -> Content:
        """테마 인식 색상으로 렌더링합니다.

Returns:
            테마에 적합한 색상으로 오류 콘텐츠 스타일을 지정했습니다.

        """
        colors = theme.get_theme_colors(self)
        return Content.assemble(
            Content.styled("Error: ", f"bold {colors.error}"),
            self._content,
        )

    def on_mount(self) -> None:
        """문자셋 모드에 따라 테두리 스타일을 설정합니다."""
        if is_ascii_mode():
            colors = theme.get_theme_colors(self)
            self.styles.border_left = ("ascii", colors.error)


class AppMessage(Static):
    """앱 메시지를 표시하는 위젯입니다."""

    # Disable Textual's auto_links to prevent a flicker cycle: Style.__add__
    # calls .copy() for linked styles, generating a fresh random _link_id on
    # each render. This means highlight_link_id never stabilizes, causing an
    # infinite hover-refresh loop.
    auto_links = False

    DEFAULT_CSS = """
    AppMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, message: str | Content, **kwargs: Any) -> None:
        """시스템 메시지를 초기화합니다.

Args:
            message: 문자열 또는 미리 스타일이 지정된 `Content` 형식의 시스템 메시지입니다.
            **kwargs: 부모에게 전달된 추가 인수

        """
        # Store raw content for serialization
        self._content = message
        rendered = (
            message
            if isinstance(message, Content)
            else Content.styled(message, "dim italic")
        )
        super().__init__(rendered, **kwargs)

    def on_click(self, event: Click) -> None:
        """한 번의 클릭으로 스타일이 포함된 하이퍼링크를 열고 타임스탬프를 표시합니다."""
        open_style_link(event)
        _show_timestamp_toast(self)


class SummarizationMessage(AppMessage):
    """요약 완료 알림을 표시하는 위젯입니다."""

    DEFAULT_CSS = """
    SummarizationMessage {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        color: $primary;
        background: $surface;
        border-left: wide $primary;
        text-style: bold;
    }
    """

    def __init__(self, message: str | Content | None = None, **kwargs: Any) -> None:
        """요약 알림 메시지를 초기화합니다.

Args:
            message: 메시지 저장소에서 복원할 때 사용되는 선택적 메시지 재정의.

                표준 요약 알림이 기본값입니다.
            **kwargs: 추가 인수가 부모에게 전달되었습니다.

        """
        self._raw_message = message
        # Pass the default text to AppMessage for _content serialization;
        # render() supplies theme-aware styling at display time.
        super().__init__(message or "✓ Conversation offloaded", **kwargs)

    def render(self) -> Content:
        """테마 인식 색상으로 렌더링합니다.

Returns:
            테마에 적합한 색상으로 스타일이 지정된 요약 콘텐츠입니다.

        """
        colors = theme.get_theme_colors(self)
        if self._raw_message is None:
            return Content.styled("✓ Conversation offloaded", f"bold {colors.primary}")
        if isinstance(self._raw_message, Content):
            return self._raw_message
        return Content.styled(self._raw_message, f"bold {colors.primary}")
